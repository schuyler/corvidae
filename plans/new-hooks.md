# New hooks for plugin developers

## Motivation

The current hook set covers the core agent lifecycle well but has gaps
that force plugin developers into workarounds — replacing entire
strategies when they only want to gate a decision, or maintaining
per-channel state with lazy initialization because there's no creation
event. This document catalogs hooks worth adding, organized by category.

---

## Gate/policy hooks

### `should_compact_conversation(channel, conversation, max_tokens) → bool | None`

**Dispatch:** Broadcast, `REJECT_WINS` (same pattern as
`should_process_message`).

**Problem:** A plugin that wants to suppress compaction during a critical
sequence (e.g., mid-tool-pipeline, during a multi-step reasoning chain)
must replace the entire compaction handler via `compact_conversation`
just to add a veto. The policy decision ("should we compact?") is
entangled with the mechanism ("how do we compact?").

**Semantics:** Return `False` to veto compaction for this turn, `True`
or `None` for no opinion (both are treated identically — the gate only
checks for `False`). `REJECT_WINS` means any `False` vetoes. Fires
before `compact_conversation`; if the gate result is `False`, the
compaction chain doesn't run.

**Call site:** Inline in `Agent._process_queue_item` step 5
(lines ~497-506), before the `pm.ahook.compact_conversation` call. There
is no wrapper method — the gate check must be added inline at that
location.

**Exception handling:** Exceptions from this hook should be treated as
`None` (no opinion) — log at WARNING and proceed to the compaction
chain. **Implementation note:** This requires its own `try/except`
block, separate from the existing `try/except` around
`compact_conversation`. If the gate check is placed inside the existing
block, an exception in the gate would be caught by that handler and
treated as "compaction skipped" — the opposite of the intended "no
opinion, proceed to compaction."

**Migration path:** CompactionPlugin's internal threshold check
(`token_estimate > compaction_threshold * max_tokens`) could move into a
default `should_compact_conversation` hookimpl, making the threshold
logic itself replaceable without touching the summarization strategy.
This eliminates the need for a `True` (force-compact) return value —
the gate is purely a veto mechanism, and custom compaction triggers
are handled by moving the threshold logic into replaceable hookimpls.

**Example use cases:**
- Suppress compaction during a multi-step tool pipeline
- Disable compaction entirely for debug/replay channels
- Custom threshold logic without replacing the compaction strategy

---

### `should_execute_tool(channel, tool_name, arguments) → bool | None`

**Dispatch:** Broadcast, `REJECT_WINS`.

**Problem:** There is no way to intercept a tool call before execution.
A plugin cannot enforce per-tool permissions, rate limits, cost budgets,
or audit requirements without patching tool dispatch or wrapping every
tool function.

**Semantics:** Return `False` to block the tool call, `True` to
explicitly allow, `None` for no opinion. `REJECT_WINS`. When a tool call
is blocked, `dispatch_tool_call` returns an error result to the LLM
(e.g., `"Tool call blocked by policy"`) rather than silently dropping it.

**Call site:** `dispatch_tool_call` in `corvidae/tool.py` — this is
where tool name lookup, argument parsing, and the unknown-tool guard
live. The gate fires after argument parsing succeeds but before
`execute_tool_call` is invoked. (`execute_tool_call` receives an
already-resolved callable and has no `tool_name` parameter.)

**Exception handling:** Exceptions from this hook should be treated as
`None` (no opinion) — log at WARNING and allow the tool call to proceed.
An exception in a policy hook should not block tool execution.

**Example use cases:**
- Blocklist specific tools per channel or per user
- Rate-limit expensive tools (web_fetch, shell)
- Enforce a cost budget by blocking tools after a spend threshold
- Audit logging of all tool invocations with arguments

---

### `filter_tools(channel, tool_names) → list[str] | None`

**Dispatch:** Broadcast with intersection at the call site.

**Problem:** `register_tools` populates a global registry once at
startup. There is no way for a plugin to vary tool availability per
channel or per message. A plugin that wants to hide certain tools from
certain channels cannot do so without patching tool dispatch. This is
distinct from `should_execute_tool` — that blocks execution after the
LLM has already chosen a tool (wasting a turn), whereas `filter_tools`
removes tools from the schema before the LLM sees them.

**Why not `firstresult=True`:** Multiple independent plugins should each
be able to narrow the tool set. A `RoleFilterPlugin` hiding admin tools
and a `CostCapPlugin` hiding expensive tools after a budget threshold
are independent concerns that must compose, not compete. With
`firstresult=True`, whichever returns first wins and the other is
silently ignored.

**Why not `REJECT_WINS`:** The semantics are "filter a set" not
"veto a decision." `REJECT_WINS` is a boolean gate; filtering requires
returning a list.

**Semantics:** Broadcast. All plugins run. Each returns a filtered list
of tool names (subset of the input) or `None` for no opinion. The call
site intersects all non-None results — only tools present in every
returned list survive. If all plugins return `None`, the full set is
used.

**Call site:** `Agent._process_queue_item` (inline, before tool schemas
are passed to `_run_turn`). The call site must:
1. Extract tool names from `self._tool_schemas` via
   `s["function"]["name"]`.
2. Call `pm.ahook.filter_tools(channel=channel, tool_names=names)`.
3. Collect all non-None results, compute intersection.
4. Build a **local** filtered copy of the schema list for this turn
   only — do not mutate `self._tool_schemas`, which is shared state
   across all turns and channels.
5. If the intersection is empty (all tools filtered out), pass an
   empty tool list to the LLM. This is a valid policy outcome — a
   plugin returning `[]` means "no tools allowed." The LLM will
   respond without tool use.

**Exception handling:** Exceptions from this hook should be treated as
`None` (no opinion) — log at WARNING and use the full tool set for this
turn.

**Subagent coverage:** This hook fires only for the main agent turn.
Subagent turns (in `corvidae/tools/subagent.py`) build their tool list
directly from `ToolCollectionPlugin.get_registry()`, bypassing
`Agent._tool_schemas`. `filter_tools` does not fire for subagent turns.
This is a known limitation — subagent tool filtering would require
changes to the subagent tool-selection path.

**Relationship to `should_execute_tool`:** `filter_tools` is advisory —
it controls what the LLM sees but does not enforce access control. An
LLM that knows of a hidden tool (from conversation history or prompt
injection) can still call it, because `dispatch_tool_call` operates on
the full `self._tools` dict. For access control enforcement, compose
`filter_tools` (hide from LLM) with `should_execute_tool` (block at
execution). Both hooks must be implemented for the access-control use
case to be secure.

**Example use cases:**
- Hide admin tools from public channels (with `should_execute_tool` for
  enforcement)
- A/B testing different tool sets per channel
- Reduce token usage by hiding irrelevant tools

---

## Observation hooks

### `on_llm_request(channel, messages, tools, parameters)`

**Dispatch:** Broadcast (side-effect only).

**Problem:** `before_agent_turn` fires once per turn, but a single turn
may involve multiple LLM API calls (the initial call plus one per
tool-result round-trip). There is no hook that fires before each
individual API call, so plugins cannot observe the actual request payload
— the messages list, tool schemas, and inference parameters — as sent to
the LLM.

**Semantics:** Fires immediately before each `LLMClient` call.
`messages` is the full message list, `tools` is the tool schema list (or
None if no tools), `parameters` is a dict of inference parameters
(model, temperature, etc.). All arguments are read-only snapshots — do
not mutate.

**Call site:** `run_agent_turn` in `corvidae/turn.py`, before the
`client.chat()` call. This requires passing `pm` into `run_agent_turn`,
which currently has no plugin system dependency. Alternatively, the hook
could fire from `Agent._run_turn` if the turn module exposes the request
parameters before calling the client — but this may not capture all
callers.

**Open question:** Should compaction LLM calls (made by
`CompactionPlugin` for summarization) also fire this hook? If yes, the
hook belongs in `LLMClient.call` so all callers are captured. If no
(only agent-turn calls), it belongs in `run_agent_turn`. This affects
cost tracking — compaction calls consume tokens too.

**Example use cases:**
- Token accounting and cost tracking per API call
- Request logging for debugging prompt construction
- Detecting prompt injection by inspecting the final message list
- Metrics on tool schema size and message count per call

---

### `on_llm_response(channel, response, latency_ms)`

**Dispatch:** Broadcast (side-effect only).

**Problem:** `on_agent_response` fires once after the entire turn
completes (possibly many LLM calls and tool dispatches). Plugins cannot
observe intermediate LLM responses during tool-use loops — e.g., the
response that selected which tools to call, or a mid-chain reasoning
step.

**Semantics:** Fires immediately after each `LLMClient` call returns.
`response` is the raw API response (before tool dispatch or display
processing). `latency_ms` is the wall-clock time for the API call.

**Call site:** Same location decision as `on_llm_request` — either
`run_agent_turn` in `corvidae/turn.py` or `LLMClient.call`. The same
open question about compaction calls applies.

**Example use cases:**
- Per-call latency tracking and alerting
- Token usage accounting (prompt + completion tokens per call)
- Observing intermediate reasoning in multi-tool turns
- Detecting and logging model refusals or safety triggers

---

### `on_channel_created(channel)`

**Dispatch:** Broadcast (side-effect only).

**Problem:** Plugins that maintain per-channel state (caches, counters,
external subscriptions, monitoring) must lazily initialize on first
`on_message` or `before_agent_turn`, with boilerplate `if channel.id not
in self._state` checks. There is no event for channel materialization.

**Semantics:** Fires when a channel is first created — either from
config at startup or on-demand when the first message arrives. Fires
after the channel's config is resolved but before any messages are
processed on it.

**Architectural prerequisite:** `ChannelRegistry` currently has no
plugin manager reference. It is not a plugin (no `@hookimpl` decorators)
and is constructed before entry points load. To fire this hook,
`ChannelRegistry` needs either a `pm` reference (set after plugin
loading) or a creation callback that `Runtime` wires to `pm.ahook`.
This is a small architectural change but it is a prerequisite — this
hook cannot be implemented without it.

**Example use cases:**
- Initialize per-channel metrics collectors
- Register external webhooks or subscriptions per channel
- Set up per-channel caches or state objects
- Log channel creation for auditing

---

### `on_metrics(name, value, tags)`

**Dispatch:** Broadcast (side-effect only).

**Problem:** The framework has no structured metrics pipeline. Plugins
that want to emit or consume metrics must DIY everything — pick a
library, decide what to measure, wire it into whichever hooks fire at
the right time. There is no standard way to produce or consume numeric
measurements.

**Semantics:** A metrics hook for consuming metric events.

```python
@hookspec
async def on_metrics(
    self,
    name: str,            # e.g. "llm.tokens.prompt", "tool.duration_ms"
    value: float,
    tags: dict[str, str], # e.g. {"channel": "irc:#general", "model": "gpt-4o"}
) -> None:
```

The shape follows the StatsD/OpenTelemetry common denominator — a named
numeric value with dimensional tags. It covers counters, gauges, and
histograms depending on how the consumer interprets the name convention.
It does not model metric types explicitly.

**Emission:** Any plugin can emit metrics by calling
`await self.pm.ahook.on_metrics(name=..., value=..., tags=...)` from
tool functions, other hook implementations, or background tasks.

**Reentrancy constraint:** Do not call `pm.ahook.on_metrics(...)` from
inside an `on_metrics` implementation — pluggy dispatches back to the
same implementation, causing infinite recursion. A plugin that both
produces and consumes metrics emits from its other hooks or tool
functions, never from `on_metrics` itself.

**Built-in emission points:** The framework emits metrics at key
points — token usage from `on_llm_response`, tool duration from
`dispatch_tool_call`, compaction events, queue depth from idle checks.
These emission points depend on the corresponding observation hooks
being implemented first (e.g., `on_llm_response` must exist before
token usage metrics can be emitted from it).

**Naming convention:** Deferred to implementation. The initial set of
built-in metric names will establish the convention (dotted hierarchy,
e.g. `llm.tokens.prompt`, `tool.duration_ms`). A formal metric type
system (counter vs gauge vs histogram) is not part of this proposal —
consumers infer type from the name convention.

**Consumer examples:**
- `PrometheusPlugin` — implements `on_metrics`, exports to Prometheus
- `CostTrackingPlugin` — filters for `llm.tokens.*`, tracks spend
- `StatsPlugin` — logs periodic summaries to the channel

---

## Transform hooks

### `transform_system_prompt(channel, prompt) → str | None`

**Dispatch:** Wrapper chain (same pattern as the planned Phase 3
migration for `transform_display_text` and `process_tool_result`).

**Problem:** System prompts are static — set in `agent.yaml` config and
fixed for the lifetime of the channel. There is no mechanism for dynamic
system prompt modification. Plugins that need to inject time-sensitive
context, user-specific instructions, or channel-state-dependent rules
into the system prompt have no hook to do so. `before_agent_turn` can
inject CONTEXT messages, but these are semantically different from system
instructions and the LLM treats them differently.

**Semantics:** Wrapper chain. Each handler receives the current prompt
string, may transform it, and yields to the next handler. All wrappers
run (no early termination). The final result is the system prompt used
for the LLM call.

**Lifecycle:** Fires on every turn, not once at conversation init. The
original system prompt (from config) is the input each time — transforms
do not accumulate across turns. This means the hook must read the
original prompt from config/channel state, not from `conv.system_prompt`
(which would contain the previously-transformed value).

**Depends on:** Phase 3 wrapper chain support in apluggy. Could ship as
`VALUE_FIRST` initially (only one transform wins) with a note that
stacking will work after Phase 3.

**Example use cases:**
- Inject current date/time into the system prompt
- Add per-user instructions based on the sender identity
- Append channel-specific rules (e.g., "in this channel, always respond
  in Spanish")
- Feature flags that modify agent behavior per channel

---

### `transform_inbound_text(channel, sender, text) → str | None`

**Dispatch:** Wrapper chain.

**Problem:** Incoming message text enters the system as-is from the
transport. There is no hook to normalize, preprocess, or rewrite it
before it enters the processing queue. Plugins that need to strip
command prefixes, normalize Unicode, detect language, or sanitize input
must do so inside `on_message` implementations, which means the
transformation is invisible to other plugins that also implement
`on_message`.

**Semantics:** Wrapper chain. Each handler receives the current text,
may transform it, and yields to the next handler. All wrappers run.
The final result is the text that enters the processing queue. Fires
after `should_process_message` passes and before the message is
enqueued.

**Depends on:** Phase 3 wrapper chain support in apluggy. Could ship as
`VALUE_FIRST` initially.

**Example use cases:**
- Strip bot mention prefixes (`@corvidae do X` → `do X`)
- Normalize Unicode (NFKC, strip zero-width characters)
- Command prefix parsing (`!help` → route to help handler)
- Content sanitization (strip embedded control characters)

---

## Lifecycle hooks

### `on_channel_config_changed(channel, old_overrides, new_overrides)`

**Dispatch:** Broadcast (side-effect only).

**Problem:** When `set_settings` modifies a channel's runtime config,
plugins that cache resolved config values (thresholds, feature flags,
parameters) have no way to know the config changed. They continue using
stale values until the next time they happen to re-read config — which
for most plugins is never, since config is read once in `on_start`.

**Semantics:** Fires after `set_settings` applies changes to a channel.
`old_overrides` is a snapshot of `channel.runtime_overrides` before the
mutation. `new_overrides` is a snapshot after. Keys cleared via `null`
in `set_settings` are present in `old_overrides` but absent from
`new_overrides` — consumers detect deletions by checking for keys
present in `old_overrides` that are missing from `new_overrides`.

**Call site:** The `set_settings` closure inside
`RuntimeSettingsPlugin.register_tools`. Snapshot `old_overrides` before
mutation, apply changes, then fire the hook with both snapshots.

**Example use cases:**
- ContextCompactPlugin updating its thresholds when
  `max_context_tokens` changes
- Observability plugins logging config changes
- Transport plugins adjusting behavior when settings change

---

## Implementation notes

### Ordering and dependencies

**Implementable now** (broadcast or `REJECT_WINS`, no new apluggy
features needed):
- `should_compact_conversation`
- `should_execute_tool`
- `filter_tools` (broadcast with intersection)
- `on_llm_request`, `on_llm_response`
- `on_metrics` (hookspec can ship immediately; built-in emission points
  that depend on other proposed hooks like `on_llm_response` ship when
  those hooks land)
- `on_channel_config_changed`

**Requires architectural change first:**
- `on_channel_created` — needs `ChannelRegistry` to gain a `pm`
  reference or creation callback

**Depends on Phase 3 wrapper chain support:**
- `transform_system_prompt`
- `transform_inbound_text`

Both could ship as `VALUE_FIRST` broadcast hooks initially (only one
transform wins) and migrate to wrapper chains in Phase 3.

### Hookspec additions

All new hooks are added to `AgentSpec` in `corvidae/hooks.py`. No
existing hooks change. This is purely additive — existing plugins are
unaffected.

### Call site integration

| Hook | Call site | Integration point |
|------|-----------|-------------------|
| `should_compact_conversation` | `Agent._process_queue_item` step 5 | Before `pm.ahook.compact_conversation` (inline, no wrapper method) |
| `should_execute_tool` | `dispatch_tool_call` in `corvidae/tool.py` | After argument parsing, before `execute_tool_call` invocation |
| `filter_tools` | `Agent._process_queue_item` step 7 (inline) | Before tool schemas are passed to `run_agent_turn` |
| `on_llm_request` | `run_agent_turn` in `corvidae/turn.py` or `LLMClient.call` | Before `client.chat()` — see open question |
| `on_llm_response` | Same as `on_llm_request` | After `client.chat()` returns |
| `on_channel_created` | `ChannelRegistry.get_or_create` (after architectural change) | After channel config resolution |
| `on_channel_config_changed` | `set_settings` closure in `RuntimeSettingsPlugin` | After overrides applied |
| `on_metrics` | Multiple sites (token usage, tool dispatch, compaction, etc.) | Various — see built-in emission points |
| `transform_system_prompt` | Prompt construction path | Per-turn, using original config prompt as input |
| `transform_inbound_text` | `Agent.on_message` or queue ingestion | After `should_process_message`, before enqueue |

### Open design questions

1. **`on_llm_request` / `on_llm_response` scope:** Should compaction
   LLM calls fire these hooks? If yes, they belong in `LLMClient`. If
   no, they belong in `run_agent_turn`. This affects cost tracking
   accuracy vs. implementation layering (adding `pm` to the turn module
   or LLM client).

2. **`run_agent_turn` and `pm`:** `corvidae/turn.py` currently has no
   plugin system dependency. Adding `on_llm_request`/`on_llm_response`
   there requires passing `pm` in, which is a layering change worth
   considering carefully.

---

## Review #1

Reviewed against: `corvidae/hooks.py`, `corvidae/agent.py`,
`corvidae/tool.py`, `corvidae/runtime.py`, `corvidae/channel.py`,
`corvidae/turn.py`, `corvidae/tools/settings.py`,
`docs/plugin-guide.md`, `plans/hook-dispatch-design.md`.

### Critical

None after revision. C1–C4 from the initial review have been addressed:
- C1: Call site corrected to `_process_queue_item` step 5 (inline)
- C2: Call site corrected to `dispatch_tool_call`
- C3: `on_channel_created` now lists architectural prerequisite
- C4: Override deletion semantics specified (old vs new snapshot
  comparison)

### Important

Findings I1–I6 from the initial review have been addressed:
- I1: Open question explicitly documented
- I2: Wrapper chain semantics corrected (yield-based, not VALUE_FIRST
  return)
- I3: Lifecycle specified (per-turn, original prompt as input)
- I4: Force-compact removed; gate is veto-only
- I5: Exception handling specified for both gate hooks
- I6: Deletion representation specified (present in old, absent in new)

### Cosmetic

- Cs1–Cs3 from the initial review addressed in revised call site table
  and implementation notes.

### New hooks added

- `filter_tools` (from reviewer suggestion G1)
- `on_metrics` (from discussion)

GATE: Pending re-review after revision.

## Review #2

Review #2 found two critical issues (C1: `filter_tools` dispatch, C2:
`on_metrics` reentrancy) and six important issues. All addressed:

- C1: `filter_tools` changed from `firstresult=True` to broadcast with
  intersection. Multiple plugins compose filters independently.
- C2: `on_metrics` reentrancy constraint documented. Plugins avoid
  calling `pm.ahook.on_metrics` from inside `on_metrics` implementations.
- I1: `filter_tools` call site specifies name→schema conversion.
- I2: `filter_tools` security gap documented — must compose with
  `should_execute_tool` for enforcement.
- I3: `filter_tools` subagent limitation documented as known.
- I4: `on_metrics` ordering dependency on `on_llm_response` documented.
- I5: `should_compact_conversation` two-try/except requirement specified.
- I6: `filter_tools` exception handling specified.

GATE: FAIL (two critical). Addressed in revision, pending re-review.

## Review #3

Review #3 confirmed all Review #2 findings resolved. Three new important
findings:

- New-I1: `filter_tools` must create a local copy, not mutate shared
  `self._tool_schemas`. Addressed — call site now specifies local copy.
- New-I2: Empty intersection edge case (all plugins return `[]`).
  Addressed — specified as valid policy outcome, empty tool list passed
  to LLM.
- New-I3: `True` return from `should_compact_conversation` ambiguous.
  Addressed — semantics now state `True` and `None` are treated
  identically (gate only checks for `False`).

GATE: FAIL (three important). Addressed in revision, pending re-review.

## Review #4

All Review #3 findings verified as resolved. No new issues. Document is
internally consistent.

GATE: PASS
