# Design: Runtime Settings Tool

## Context

Agents currently cannot alter their own runtime settings (LLM inference
parameters like temperature, or framework parameters like max_turns). All
configuration is static after startup. This design adds a tool that lets the
agent mutate per-channel settings at runtime, with an operator-configurable
blocklist of immutable keys.

## Scope

- New tool: `set_settings`
- Override storage on `Channel`
- Plumbing to thread overrides into LLM calls and config resolution
- Blocklist enforcement from `agent.yaml`

## Design

### 1. Override storage: `Channel.runtime_overrides`

Add a `runtime_overrides: dict = field(default_factory=dict)` field to the
`Channel` dataclass. This dict holds two namespaces of keys:

- **LLM inference params** — keys like `temperature`, `top_p`, `top_k`,
  `frequency_penalty`, `presence_penalty`, `max_tokens`. These are passed
  through as `extra_body` to `LLMClient.chat()`.
- **Framework params** — keys that match `ChannelConfig` fields:
  `max_turns`, `max_context_tokens`, `keep_thinking_in_history`.

Both namespaces live in the same flat dict. There's no collision risk because
the LLM API params and framework config keys don't overlap.

`runtime_overrides` is not persisted to the database. Overrides last for the
lifetime of the daemon process. This matches the expected use: the agent
adjusts its own behavior within a session, not permanently.

### 2. Config resolution changes: `ChannelConfig.resolve()`

`ChannelConfig.resolve(agent_defaults)` currently returns a merged dict of
channel config over agent defaults. Change it to accept an optional
`runtime_overrides: dict` parameter.

Resolution order (last wins):
1. Built-in defaults
2. `agent_defaults` from YAML
3. `ChannelConfig` fields (per-channel YAML)
4. `runtime_overrides` (set by tool at runtime)

Call chain: `ChannelRegistry.resolve_config(channel)` passes
`channel.runtime_overrides` to `channel.config.resolve(agent_defaults,
runtime_overrides=channel.runtime_overrides)`. `resolve()` applies the
overrides as step 4. `resolve_config()` does not apply overrides
independently — single application point.

### 3. LLM call plumbing

In `AgentPlugin._process_queue_item`, after resolving config, extract LLM
inference params from `channel.runtime_overrides` and pass them through.

Currently (line 274):
```python
result = await run_agent_turn(self.client, messages, self.tool_schemas)
```

Change `run_agent_turn` signature to accept `extra_body: dict | None = None`
and forward it to `client.chat()`:

At module level in `agent.py`:
```python
from dataclasses import fields as dc_fields
FRAMEWORK_KEYS = {f.name for f in dc_fields(ChannelConfig)}
```

Then in `_process_queue_item`, after resolving config:
```python
llm_overrides = {k: v for k, v in channel.runtime_overrides.items() if k not in FRAMEWORK_KEYS}

result = await run_agent_turn(
    self.client, messages, self.tool_schemas,
    extra_body=llm_overrides or None,
)
```

`FRAMEWORK_KEYS` is derived from `dataclasses.fields(ChannelConfig)` at
module level so it stays in sync automatically when new fields are added.

Inside `run_agent_turn`, pass `extra_body` to `client.chat()`. The existing
`LLMClient.chat()` already merges call-level `extra_body` over instance-level
`extra_body`, so this composes correctly with static `extra_body` from YAML.

### 4. The `set_settings` tool

Registered by a new `RuntimeSettingsPlugin` (or added to `CoreToolsPlugin` —
see discussion below). The tool function:

```python
async def set_settings(_ctx: ToolContext, **kwargs) -> str:
```

Wait — `tool_to_schema` builds schema from typed parameters, and `**kwargs`
won't produce a useful schema. Instead, use a single `settings: dict`
parameter:

```python
async def set_settings(settings: dict, _ctx: ToolContext) -> str:
    """Update runtime settings for this conversation. Keys include LLM
    inference parameters (temperature, top_p, top_k, frequency_penalty,
    presence_penalty, max_tokens) and framework parameters (max_turns,
    max_context_tokens, keep_thinking_in_history). Returns the current
    settings after the update."""
```

Implementation:
1. Check each key in `settings` against the blocklist. If any key is
   blocked, return an error naming the blocked keys. Don't apply partial
   updates — reject the whole call if any key is immutable.
2. Merge `settings` into `channel.runtime_overrides` (update, not replace).
3. Return a confirmation string listing the current state of all overrides.

To clear an override, the agent passes `key: null`. JSON `null` deserializes
to Python `None`. The tool must explicitly check for `None` values and
`del channel.runtime_overrides[key]` rather than storing `None` (which would
be forwarded as `extra_body` and could cause API errors). This reverts that
key to the static config value.

Order of operations: blocklist check runs first, then null-delete. If the
agent passes `null` for a blocked key, the call is rejected — blocked keys
can never be in `runtime_overrides`, so there is nothing to clear.

### 5. Blocklist configuration

In `agent.yaml`:

```yaml
agent:
  immutable_settings:
    - system_prompt
```

Default blocklist (hardcoded, merged with operator config): `["system_prompt"]`.
The operator can add more keys. The operator cannot remove `system_prompt`
from the blocklist — it's always immutable. (If we ever want to allow system
prompt mutation, that's a separate deliberate decision.)

The blocklist is read once at startup by the plugin and stored as a `set`.

### 6. Plugin placement

Two options:

**A. New `RuntimeSettingsPlugin` in `corvidae/tools/settings.py`.**
Keeps it isolated. Needs access to the blocklist from config.

**B. Add to `CoreToolsPlugin` in `corvidae/tools/__init__.py`.**
CoreToolsPlugin already reads tool config and registers tools. Simpler,
fewer files.

Recommendation: **Option A.** The tool mutates `Channel` state, which is a
different concern from shell/file/web tools. A separate plugin is cleaner
and easier to disable if an operator doesn't want agents changing their own
settings.

### 7. `run_agent_loop` (subagent path)

`run_agent_loop` is used by subagents and calls `run_agent_turn` internally.
It should also accept and forward `extra_body`. However, subagents run with
their own `LLMClient` (from `llm.background`), so channel-level overrides
don't naturally apply. No change needed for subagents in this iteration —
the tool only affects the main agent loop via the channel's serial queue.

## Files to modify

| File | Change |
|------|--------|
| `corvidae/channel.py` | Add `runtime_overrides: dict` to `Channel`. Update `ChannelConfig.resolve()` to accept and apply overrides. Update `ChannelRegistry.resolve_config()` to pass them through. |
| `corvidae/agent_loop.py` | Add `extra_body` param to `run_agent_turn()`, forward to `client.chat()`. |
| `corvidae/agent.py` | Extract LLM overrides from `channel.runtime_overrides`, pass to `run_agent_turn()`. |
| `corvidae/main.py` | Import and register `RuntimeSettingsPlugin` before `AgentPlugin` (same lazy-import pattern as other plugins). |
| `agent.yaml.example` | Document `agent.immutable_settings` option. |

## Files to create

| File | Purpose |
|------|---------|
| `corvidae/tools/settings.py` | RuntimeSettingsPlugin |
| `tests/test_runtime_settings.py` | Unit tests |

## Verification

1. Unit tests for `set_settings` tool: valid params applied, blocked params rejected, null clears override.
2. Unit tests for `ChannelConfig.resolve()` with runtime overrides.
3. Unit test that `run_agent_turn` passes `extra_body` through to `client.chat()`.
4. Integration: start daemon, send message, call `set_settings` with `temperature: 0.9`, verify the next LLM request includes `temperature: 0.9` in payload.

## Review appendix

### Review 1 (design review)

Reviewer recommended YES with six findings addressed in this revision:

1. **Important**: `Channel.runtime_overrides` must use `field(default_factory=dict)` — fixed.
2. **Important**: `None`-means-delete must be explicit in tool implementation — fixed, added implementation note.
3. **Important**: Missing `main.py` registration — added to files-to-modify.
4. **Important**: `FRAMEWORK_KEYS` should derive from `dataclasses.fields(ChannelConfig)` — fixed.
5. **Cosmetic**: Removed `corvidae/hooks.py` from files-to-modify (no changes needed).
6. **Cosmetic**: Added `agent.yaml.example` to files-to-modify for `immutable_settings` docs.
7. **Critical (acknowledged)**: `dict` param produces minimal JSON schema — accepted tradeoff. LLM relies on docstring for key guidance. Schema will be `{"type": "object"}` with no property constraints.

### Review 2 (re-review)

Verified all 7 findings from Review 1 were addressed. Four new findings:

1. **Important**: `FRAMEWORK_KEYS` snippet was inline but prose said module-level — fixed, snippet now shows module-level placement.
2. **Important**: Ambiguous call chain between `resolve_config()` and `resolve()` — fixed, explicit call chain documented. Single application point in `resolve()`.
3. **Cosmetic**: `corvidae/tools/settings.py` was in both tables — moved to "Files to create" only.
4. **Cosmetic**: `null`-for-blocked-key behavior unstated — added order-of-operations note.

Verdict: **YES — proceed to implementation.**
