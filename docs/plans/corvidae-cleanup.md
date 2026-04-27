# Corvidae Cleanup — Implementation Design

**Date:** 2026-04-27

## Overview

Five independent cleanup items targeting code quality, documentation accuracy,
and observability. Items 1 and 5 are documentation-only. Items 2, 3, and 4
touch Python source files. No behavioral changes are intended; the refactors
preserve all observable behavior.

Ordering:
- Item 3 (consolidate `max_tool_result_chars`) must complete before item 2 is
  reviewed, because item 2's `_run_turn` helper will pass
  `self._max_tool_result_chars` through to the dispatch path and reviewers
  need a stable single source to audit against.
- Items 1, 4, and 5 are independent and can be done in any order or in
  parallel.
- Item 2 can proceed after item 3 is merged.

---

## Item 1: Update `docs/design.md`

### Files affected

- `docs/design.md`

### Changes

**1a. Add MCP Client section**

Add a new top-level section "## McpClientPlugin" after the Subagent Tool
section. Content to include:

- Module: `corvidae/mcp_client.py`
- Purpose: connects to external MCP servers during `on_start`, caches their
  tool lists, and exposes them to the agent loop via `register_tools`.
- Lifecycle: `on_start` connects (stdio or SSE transports via `AsyncExitStack`);
  `register_tools` appends cached `Tool` instances synchronously; `on_stop`
  closes all sessions.
- Tool naming: each tool is prefixed with the server's `tool_prefix` (default:
  server name) joined by `__`. Example: server `files`, tool `read` → `files__read`.
- Tool name collision: if two servers expose the same prefixed name, the second
  is skipped and a warning is logged.
- Schema sanitization: `_mcp_tool_to_schema` strips top-level keys
  `$schema`, `$id`, `$comment`, `$defs`, `definitions` from
  `mcp_tool.inputSchema` before wrapping in the OpenAI function-call envelope.
  Sanitization is shallow (top-level only).
- Error handling: connection failures per server are caught, logged as
  warnings, and skipped. A failed server does not abort startup.
- Config reference (already partially present in the file header; surface in
  the section):

```yaml
mcp:
  servers:
    <name>:
      transport: stdio | sse
      command: <executable>   # stdio only
      args: [...]             # stdio only
      env: {...}              # stdio only, optional
      url: <endpoint>         # sse only
      tool_prefix: <prefix>   # optional; default: server name
      timeout_seconds: 30     # optional; default: 30
```

**1b. Add Runtime Settings Tool section**

Add a new top-level section "## RuntimeSettingsPlugin" after the McpClientPlugin
section. Content to include:

- Module: `corvidae/tools/settings.py`
- Purpose: registers the `set_settings` tool, which allows the agent to update
  per-channel LLM inference parameters and framework parameters at runtime.
- Blocklist: `"system_prompt"` is always blocked regardless of operator config.
  Additional keys are configured via `agent.immutable_settings` in `agent.yaml`.
- Tool signature:

```python
async def set_settings(settings: dict, _ctx: ToolContext) -> str
```

- Parameters in `settings`: LLM inference keys (`temperature`, `top_p`,
  `top_k`, `frequency_penalty`, `presence_penalty`, `max_tokens`) and
  framework keys (`max_turns`, `max_context_tokens`,
  `keep_thinking_in_history`). Pass `null` for a key to clear that override
  and revert to the static config value.
- Return value: confirmation string with current overrides, or an error string
  naming blocked keys.
- Config:

```yaml
agent:
  immutable_settings:   # keys the agent must not change (system_prompt always added)
    - max_turns
```

**1c. Update Plugin Registration Order**

The list at "## Plugin Registration Order" in `docs/design.md` (lines 740–751)
is missing `McpClientPlugin` and `RuntimeSettingsPlugin`. Replace it with the
correct order as implemented in `main.py` lines 73–136:

1. `ChannelRegistry` — registered as `"registry"`
2. `PersistencePlugin` — DB lifecycle and conversation initialization
3. `CoreToolsPlugin` — registers core tools
4. `CLIPlugin` — stdin/stdout transport
5. `IRCPlugin` — IRC transport
6. `TaskPlugin` — task queue
7. `SubagentPlugin` — registers the `subagent` tool
8. `McpClientPlugin` — MCP server connections and tool forwarding
9. `CompactionPlugin` — default `compact_conversation` implementation
10. `ThinkingPlugin` — `<think>` stripping and `reasoning_content` removal
11. `RuntimeSettingsPlugin` — registers the `set_settings` tool
12. `AgentPlugin` — agent loop (after all tools, transports, and support
    plugins)
13. `IdleMonitorPlugin` — idle monitor (after `AgentPlugin`)

**1d. Update dependency graph**

The dependency graph in "### Current dependency graph" does not include
`ThinkingPlugin` or `CompactionPlugin` (they declare no `depends_on`), and
does not reflect that `McpClientPlugin` and `RuntimeSettingsPlugin` also
declare no `depends_on`. Add a prose note: "Plugins with no `depends_on`
attribute declared: `CoreToolsPlugin`, `McpClientPlugin`, `CompactionPlugin`,
`ThinkingPlugin`, `RuntimeSettingsPlugin`."

**1e. Update Directory Layout**

The "## Directory Layout" section (lines 797–823) is missing `mcp_client.py`
and `tools/settings.py`. Add:

```
├── mcp_client.py         # McpClientPlugin (MCP server bridge)
```

and in the `tools/` block:

```
    └── settings.py       # RuntimeSettingsPlugin, set_settings tool
```

### Risks

- None. Documentation-only change.

---

## Item 2: Split `_process_queue_item`

### Files affected

- `corvidae/agent.py`

### Background

`_process_queue_item` (agent.py:183–382) is 198 lines implementing three
distinct responsibilities:

1. Building the conversation message dict from a `QueueItem`
2. Calling `run_agent_turn` and handling LLM errors
3. Deciding what to do with the result (tool dispatch vs. text response,
   including display text transformation)

The `transform_display_text` call block is duplicated across two branches
(lines 328–336 and 341–351). The two branches differ only in the fallback
value: the max-turns branch falls back to `MAX_TURNS_FALLBACK_MESSAGE`; the
no-tool-calls branch falls back to `result.text`.

### Changes

Extract four helpers. The top-level `_process_queue_item` becomes an
orchestrator that calls them in sequence.

**Helper 1: `_build_conversation_message`**

```python
def _build_conversation_message(
    self, item: QueueItem
) -> tuple[dict, str] | None:
    """Build the conversation message dict and request_text from a QueueItem.

    Returns (conversation_message, request_text), or None if the item role
    is unrecognized (logs an error in that case).
    """
```

Handles the `if item.role == QueueItemRole.USER / NOTIFICATION / else`
block (agent.py:214–232). Returns `None` on unknown role (caller returns
early).

**Helper 2: `_run_turn`**

```python
async def _run_turn(
    self,
    channel: Channel,
    messages: list[dict],
    tool_schemas: list[dict],
    llm_overrides: dict | None,
) -> AgentTurnResult | None:
    """Call run_agent_turn and handle LLM errors.

    Returns the AgentTurnResult on success, or None on error (error message
    already sent to the channel via send_message hook).
    """
```

Wraps the `try/except` around `run_agent_turn` (agent.py:278–299) plus the
`on_llm_error` hook call and fallback error message send. Returning `None`
signals the caller to return early.

**Helper 3: `_resolve_display_text`**

```python
async def _resolve_display_text(
    self,
    channel: Channel,
    result: AgentTurnResult,
    fallback: str | None,
) -> str:
    """Call transform_display_text hook and resolve the result.

    Returns the transformed text if the hook returns a non-None value,
    otherwise returns the hook input text. If fallback is not None,
    uses fallback when the resolved text is falsy (empty string or None).
    """
```

Deduplicates the `transform_display_text` call pattern from lines 328–351.
The `fallback` parameter is `MAX_TURNS_FALLBACK_MESSAGE` in the max-turns
branch and `None` in the normal text branch (where `result.text` is already
the input to the hook and serves as its own non-None fallback).

**Helper 4: `_handle_response`**

```python
async def _handle_response(
    self,
    result: AgentTurnResult,
    channel: Channel,
    max_turns_limit: int,
) -> None:
    """Dispatch tool calls or send text response.

    Implements the decision point at step 10 of _process_queue_item:
    - Tool calls under limit: increment counter, dispatch, return.
    - Tool calls at limit: resolve display text with MAX_TURNS_FALLBACK_MESSAGE,
      fire on_agent_response, send message.
    - No tool calls: increment counter, resolve display text, fire
      on_agent_response, send message.
    """
```

Contains the logic from agent.py:314–381.

**Revised `_process_queue_item` structure:**

```python
async def _process_queue_item(self, item: QueueItem) -> None:
    # 1. Build conversation message
    msg_result = self._build_conversation_message(item)
    if msg_result is None:
        return
    conversation_message, request_text = msg_result

    channel = item.channel

    # 2–4: conversation init, turn counter reset, config resolution,
    #       message append, compaction, before_agent_turn hook
    # (unchanged from current code, stays inline)

    # 5. Build prompt and call LLM
    messages = conv.build_prompt()
    llm_overrides = {k: v for k, v in channel.runtime_overrides.items()
                     if k not in FRAMEWORK_KEYS}
    result = await self._run_turn(channel, messages, self.tool_schemas,
                                  llm_overrides or None)
    if result is None:
        return

    # 6. Persist assistant message, after_persist_assistant hook
    # (unchanged from current code, stays inline)

    # 7. Dispatch or respond
    await self._handle_response(result, channel, max_turns_limit)
```

### Risks

- The `request_text` variable is computed in `_build_conversation_message` but
  consumed in `_handle_response` (via `on_agent_response`). It must thread
  through `_handle_response`'s parameter list or be stored on the instance.
  Prefer threading it as a parameter to `_handle_response` to avoid mutable
  state. Update the signature:

  ```python
  async def _handle_response(
      self,
      result: AgentTurnResult,
      channel: Channel,
      max_turns_limit: int,
      request_text: str,
  ) -> None:
  ```

- The `conv` local variable is referenced in `_handle_response` only
  indirectly (not referenced at all — `conv` is only used for appending
  before the helper is called). Verify after extraction that no reference
  leaks.
- All three `transform_display_text` exception handlers currently log
  `exc_info=True, extra={"channel": channel.id}`. `_resolve_display_text`
  must preserve this logging exactly.

---

## Item 3: Consolidate `max_tool_result_chars`

### Files affected

- `corvidae/tools/subagent.py`
- (No changes to `corvidae/tool.py`, `corvidae/agent.py`, or
  `corvidae/agent_loop.py`)

### Background

`MAX_TOOL_RESULT_CHARS = 100_000` is the canonical constant in `tool.py:23`.

`AgentPlugin._start_plugin` (agent.py:448) reads from config:
```python
self._max_tool_result_chars = agent_config.get("max_tool_result_chars", 100_000)
```
This is the authoritative runtime value.

`SubagentPlugin.__init__` (subagent.py:28) initializes:
```python
self._max_tool_result_chars: int = 100_000
```
`SubagentPlugin.on_start` (subagent.py:35) re-reads from config independently:
```python
self._max_tool_result_chars = agent_config.get("max_tool_result_chars", 100_000)
```
This duplicates the config read. The two reads use the same config key and
default, so they will agree — but independently. If the key name or default
ever changes in one place, they will silently diverge.

`agent_loop.py:24` re-exports `MAX_TOOL_RESULT_CHARS`:
```python
from corvidae.tool import MAX_TOOL_RESULT_CHARS, ...  # noqa: F401 — re-exported for backward compat
```
`run_agent_loop` (agent_loop.py:133) uses it as a default parameter:
```python
max_result_chars: int = MAX_TOOL_RESULT_CHARS,
```
This re-export must remain; do not remove it.

### Change

Remove `SubagentPlugin._max_tool_result_chars` and its `on_start` read.
Instead, retrieve the value from `AgentPlugin` at `_launch` time via
`get_dependency`.

In `SubagentPlugin._launch` (subagent.py:52–108), the `agent` variable is
already retrieved:
```python
agent = get_dependency(self.pm, "agent_loop", AgentPlugin)
```

Add one line after that to read the value:
```python
max_result_chars = agent._max_tool_result_chars
```

Pass `max_result_chars` to `run_agent_loop` in place of
`plugin._max_tool_result_chars`.

Remove from `SubagentPlugin.__init__`:
```python
self._max_tool_result_chars: int = 100_000
```

Remove from `SubagentPlugin.on_start`:
```python
agent_config = config.get("agent", {})
self._max_tool_result_chars = agent_config.get("max_tool_result_chars", 100_000)
```

If `on_start` has no remaining body after the removal, remove the method
entirely (verify there are no other reads in `on_start`).

### Verification

`SubagentPlugin.depends_on = {"agent_loop"}` is already declared (subagent.py:23).
`get_dependency` for `"agent_loop"` is already called inside `_launch`.
`agent._max_tool_result_chars` is set during `AgentPlugin._start_plugin`,
which runs before any tool call can be dispatched. No ordering issue.

### Risks

- Reading `agent._max_tool_result_chars` (a private attribute) creates a
  coupling between `SubagentPlugin` and `AgentPlugin`'s internal naming. This
  is acceptable given that `SubagentPlugin` already depends on `AgentPlugin`
  via `get_dependency`. Document the coupling in a comment.
- If `AgentPlugin.on_start` has not run yet when `_launch` is called, the
  attribute will not exist and an `AttributeError` will be raised. This cannot
  happen in practice (tool calls are only dispatched after `on_start`
  completes) but a comment should note the assumption.

---

## Item 4: Add audit logging to `tools/settings.py`

### Files affected

- `corvidae/tools/settings.py`

### Background

`settings.py:14` defines `logger = logging.getLogger(__name__)` but never
uses it. `set_settings` mutates `channel.runtime_overrides` with no log
trail. Other plugins (e.g., `mcp_client.py`) use structured logging with
`extra=` dicts for all mutation events.

### Changes

In `set_settings` (tools/settings.py:32–63), add two log calls:

**Blocked key case** (after line 44, before `return`):

```python
logger.warning(
    "set_settings: blocked keys rejected",
    extra={"channel": channel.id if channel else None, "blocked": sorted(blocked)},
)
```

This fires when the blocklist check rejects the call. Log before the early
return.

Note: the blocklist check at line 40 runs before the channel lookup at line
47. Move the warning log to after the channel lookup, or accept `channel =
None` in the `extra` dict for the blocked-key path (the channel lookup has
not happened yet). The simplest fix: log without channel context on the
blocked-keys path, since the channel context is unavailable at that point.

Revised blocked-keys block:

```python
blocked = [k for k in settings if k in plugin.blocklist]
if blocked:
    logger.warning(
        "set_settings: blocked keys rejected",
        extra={"blocked": sorted(blocked)},
    )
    return (
        f"Error: the following settings are immutable and cannot be changed: "
        f"{', '.join(sorted(blocked))}"
    )
```

**Successful mutation case** (after the `for key, value in settings.items()`
loop, before the return):

```python
logger.info(
    "set_settings: runtime overrides updated",
    extra={
        "channel": channel.id,
        "overrides": dict(channel.runtime_overrides),
    },
)
```

This fires once per successful `set_settings` call, after all mutations are
applied, regardless of whether the net result has active overrides.

### Pattern reference

`mcp_client.py` structured logging pattern (lines 87–90, 163–165):

```python
logger.info(
    "McpClientPlugin started",
    extra={"servers": len(self._servers), "tools": len(self._cached_tools)},
)
```

Follow the same convention: string message key, dict values in `extra`.

### Risks

- `channel.id` is accessed after the `if channel is None` guard (line 48),
  so it is safe to call in the success path. Confirm the log call is placed
  after that guard.
- The blocked-keys log has no channel context. This is acceptable; the
  relevant diagnostic information is in `blocked`.

---

## Item 5: Document unused hooks

### Files affected

- `corvidae/hooks.py`
- `docs/design.md`

### Background

Two hooks in `AgentSpec` have call sites wired in `agent.py` but no plugin
implementations in the codebase:

- `on_agent_response` (hooks.py:271, called at agent.py:359–368)
- `should_process_message` (hooks.py:312, called at agent.py:124–133)

Both are broadcast hooks. With no implementations registered, pluggy returns
an empty list; `resolve_hook_results` returns `None` or `False` depending on
strategy; the call sites handle these correctly. No behavioral issue — the
hooks fire but nothing listens.

The docstrings do not note that no implementation is currently registered.
The Hook Reference table in `docs/design.md` does not note this either.

### Changes

**5a. `hooks.py` — add Note paragraphs to docstrings**

For `on_agent_response` (hooks.py:271–280), add after the existing docstring
body:

```
Note:
    No implementation is registered in the default plugin set. The hook fires
    on every text response turn but is currently a no-op. Plugins that need
    post-response callbacks (e.g., logging, metrics) should implement this hook.
```

For `should_process_message` (hooks.py:312–319), add after the existing
docstring body:

```
Note:
    No implementation is registered in the default plugin set. With no
    implementations, the broadcast returns an empty list; REJECT_WINS
    resolves to None; the message proceeds. Plugins that filter incoming
    messages (e.g., ignore-lists, rate limiting) should implement this hook.
```

**5b. `docs/design.md` — add rows to Hook Reference table**

The Hook Reference table (design.md:103–121) already has rows for both hooks.
Extend the "Call site" column text to note the no-implementation status:

For `on_agent_response`:
> broadcast | after agent loop produces a text response; no default implementation

For `should_process_message`:
> broadcast / REJECT_WINS | `on_message`, before enqueue; no default implementation; None from empty broadcast allows all messages

### Risks

- None. Documentation-only change in `hooks.py` docstrings and `docs/design.md`.

---

## Design Report

YES — proceed to implementation.

All five items are well-scoped. The only non-trivial implementation risk is in
item 2 (`_process_queue_item` split), specifically the `request_text` threading
and the `_resolve_display_text` fallback semantics. Both are called out
explicitly above. Item 3 has one minor coupling risk (accessing a private
attribute across plugin boundaries) that is also documented. Items 1, 4, and 5
carry no implementation risk.

No design decisions remain open. No information is missing that would require
escalation before work begins.

---

## Red TDD Report — Item 4 Audit Logging Tests

**Date:** 2026-04-27

8 tests added to `tests/test_runtime_settings.py` in class `TestSetSettingsAuditLogging`:

- `test_info_log_emitted_on_successful_change` — FAIL (no INFO records)
- `test_info_log_contains_channel_id` — FAIL (no INFO records)
- `test_info_log_contains_overrides` — FAIL (no INFO records)
- `test_warning_log_emitted_on_blocked_key` — FAIL (no WARNING records)
- `test_warning_log_contains_blocked_keys` — FAIL (no WARNING records)
- `test_warning_log_blocked_is_sorted` — FAIL (no WARNING records)
- `test_no_warning_on_successful_change` — PASS (correct: no spurious WARNINGs now)
- `test_no_info_on_blocked_key` — PASS (correct: no spurious INFO now)

All 6 failures are `AssertionError: assert []` — missing log records, not code defects. The 2 passing tests are negative assertions that will remain green after implementation (they verify absence of cross-contamination between the two log levels).

**YES — proceed to implementation.**

---

## Design Review

**Reviewed by:** Chico
**Date:** 2026-04-27

### Findings

**Important: `_handle_response` Helper 4 signature omits `request_text`**

The Helper 4 block defines `_handle_response` with four parameters. The corrected five-parameter signature including `request_text: str` appears only in the Risks section. Implementers should use the Risks section signature as canonical.

**Important: `_resolve_display_text` docstring underspecifies the `fallback=None` + empty-string case**

When `fallback is None` and resolved text is empty string, it should be returned as-is (matching current behavior at line 351). Add: "When fallback is None, an empty resolved text is returned as-is."

**Important: Item 3 `on_start` removal is left as a conditional**

After removing `_max_tool_result_chars` lines, `on_start` still retains `_llm_config` assignment and debug log — it must NOT be removed. State this directly.

**Cosmetic:** Line count off by 2 (198 vs 200). Second `transform_display_text` block off by one. `set_settings` `_ctx` type annotation is new, not existing.

### Line Reference and Signature Accuracy

All verified correct (cosmetic off-by-ones noted above).

### Overall Assessment

**YES — proceed to implementation.**

Design is accurate. The three important findings are clarifications for implementers, not correctness issues.

---

## Red Tests Report

**Written by:** Zeppo
**Date:** 2026-04-27

### File

`tests/test_agent_helpers.py` (new file, 24 tests)

### Coverage

| Class | Tests |
|---|---|
| `TestBuildConversationMessage` | USER role, NOTIFICATION with tool_call_id, NOTIFICATION without tool_call_id, unknown role returns None, unknown role logs error |
| `TestRunTurn` | success returns result, LLM error returns None, LLM error sends message, LLM error fires on_llm_error hook, llm_overrides forwarded as extra_body |
| `TestResolveDisplayText` | hook value used, hook None falls back to result.text, fallback used when empty + fallback given, fallback=None + empty returned as-is, hook exception not propagated, hook exception logs warning |
| `TestHandleResponse` | text response sends message, text response increments counter, text response fires on_agent_response, tool calls under limit dispatched (no send_message), tool calls under limit increment counter, tool calls at max sends fallback, tool calls at max does not increment counter, latency_ms forwarded to send_message |

### Results

24/24 FAILED with `AttributeError: 'AgentPlugin' object has no attribute '<method>'`. No syntax or import errors.

**YES — proceed to implementation.**

---

## Red TDD Report — Item 3

**Date:** 2026-04-27

4 failing tests written in `tests/test_subagent.py`, two new classes appended:

- `TestOnStart::test_on_start_does_not_set_max_tool_result_chars` — verifies `on_start` does not overwrite a sentinel `_max_tool_result_chars` value with a config read.
- `TestMaxToolResultCharsConsolidation::test_launch_passes_agent_max_result_chars_to_run_agent_loop` — verifies `_launch` passes `AgentPlugin._max_tool_result_chars` (set to 42_000) to `run_agent_loop`; currently receives 100_000.
- `TestMaxToolResultCharsConsolidation::test_launch_uses_agent_value_not_independent_config_read` — same verification with value 75_555 (neither the default nor the __init__ sentinel).
- `TestMaxToolResultCharsConsolidation::test_subagent_does_not_have_independent_max_tool_result_chars_after_on_start` — structural check that `on_start` does not write a config-sourced value (55_555) onto the SubagentPlugin instance.

All 4 fail with `AssertionError` (behavior mismatch, not syntax). All 21 pre-existing tests pass (25 total, 4 failed).

**YES — proceed to implementation.**

---

## Red TDD Review

**Reviewed by:** Chico
**Date:** 2026-04-27

### Critical Issues

None.

### Important Concerns

1. `TestHandleResponse` does not verify `on_agent_response` is absent on the tool-dispatch path. Coverage gap, not blocking.
2. `TestHandleResponse` does not verify `on_agent_response` IS called on the max-turns path. Coverage gap, not blocking.

### Minor

- Dead positional-argument fallback in `test_passes_llm_overrides_to_run_agent_turn`
- Deferred `import logging` inside test bodies (should be top-level)

### Overall Assessment

**YES — proceed to Green TDD.**

No test accepts an incorrect implementation that the design would reject.

---

## Item 4 Implementation Report

**Written by:** Zeppo
**Date:** 2026-04-27

### Changes made

`corvidae/tools/settings.py` — two log calls added, `logger` now used:

1. **Blocked-key WARNING** (after blocklist check, before early return):
   ```python
   logger.warning(
       "set_settings: blocked keys rejected",
       extra={"blocked": sorted(blocked)},
   )
   ```

2. **Successful mutation INFO** (after the override loop, before return):
   ```python
   logger.info(
       "set_settings: runtime overrides updated",
       extra={
           "channel": channel.id,
           "overrides": dict(channel.runtime_overrides),
       },
   )
   ```

Both placements satisfy the design's constraints: the WARNING has no channel context (channel lookup hasn't happened yet); the INFO fires after the `if channel is None` guard so `channel.id` is safe.

### Test results

- `tests/test_runtime_settings.py`: **49/49 passed** (all 8 `TestSetSettingsAuditLogging` tests green)
- Full suite: **665 passed, 28 failed** — the 28 failures are pre-existing red tests for Items 2 and 3, unchanged by this implementation.

**YES — implementation complete.

---

## Item 5 Implementation Report

**Date:** 2026-04-27

### Changes made

**`corvidae/hooks.py`**

- `on_agent_response` (line 271): added `Note:` paragraph to docstring documenting it as an extension point for observability plugins (logging, metrics, analytics) with no built-in implementation.
- `should_process_message` (line 318): added `Note:` paragraph to docstring documenting it as an extension point for message filtering (rate limiting, blocklists, channel muting) with no built-in implementation, and clarifying that REJECT_WINS on an empty broadcast allows all messages.

**`docs/design.md`**

- Hook Reference table, `on_agent_response` row: Call site column updated to "after agent loop produces a text response; no default implementation".
- Hook Reference table, `should_process_message` row: Call site column updated to "`on_message`, before enqueue; no default implementation; None from empty broadcast allows all messages".

### Assumptions

None. The design specified exact text for all changes.

### Questions

None.

**YES****

---

## Item 1 Documentation Report

**Date:** 2026-04-27

### Changes made

**`docs/design.md`**

1. Added `## McpClientPlugin` section after "Subagent Tool": lifecycle
   (on_start, register_tools, on_stop), tool naming, tool name collision
   (first-wins), schema sanitization (`_mcp_tool_to_schema`, shallow,
   top-level keys only), configuration block, graceful degradation note.

2. Added `## RuntimeSettingsPlugin` section after McpClientPlugin: tool
   signature, accepted parameters (LLM inference keys and framework keys),
   return value, blocklist behavior (`system_prompt` always blocked,
   `agent.immutable_settings` for operator additions), persistence (memory
   only, lost on restart), graceful degradation note.

3. Updated "## Plugin Registration Order" list from 11 entries to 13:
   inserted `McpClientPlugin` at position 8 and `RuntimeSettingsPlugin` at
   position 11, matching `main.py` lines 105–124. Backtick-quoted all plugin
   names for consistency.

4. Added prose note to "### Current dependency graph": "Plugins with no
   `depends_on` attribute declared: `CoreToolsPlugin`, `McpClientPlugin`,
   `CompactionPlugin`, `ThinkingPlugin`, `RuntimeSettingsPlugin`."

5. Updated "## Directory Layout": added `mcp_client.py` to the top-level
   `corvidae/` block; added `settings.py` to the `tools/` block (changing
   `subagent.py` from last entry to second-to-last with `└──` on `settings.py`).

### Assumptions

None. All changes derived directly from source files (`mcp_client.py`,
`tools/settings.py`, `main.py`) and the design spec.

### Questions

None.

**YES**
