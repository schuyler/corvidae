# Corvidae Design Document

An asyncio agent daemon that connects to IRC (and later other
transports), routes messages through an LLM via the OpenAI-compatible
Chat Completions API, and supports plugin components via pluggy.

## Architecture

The daemon is a single Python asyncio process. It connects to a local
LLM served by llama-server via OpenAI-compatible API.

Three layers:

1. **Plugin system (apluggy)** — defines lifecycle hooks and extension
   points.

2. **Agent loop** — manages prompt construction, tool calling, LLM
   interaction via aiohttp. Owns conversation state. Uses single-turn
   dispatch: one LLM call per queue item, tool calls dispatched as
   background tasks, results arrive as notifications.

3. **Transport plugins** — CLI, IRC, Signal, Discord, web, ACP (stdio). Each
   transport converts platform-specific messages to/from a common Channel
   abstraction.

This document covers the kernel: the `corvidae` package. The cognition
plugins (memory, funnel, appraisal, critique, outcome log) are a second
distribution in the same repository, `corvidae-cognition`, installed with
the `cognition` extra and documented in [cognition.md](cognition.md). The
kernel never imports them; they use the plugin API listed in
[plugin-guide.md](plugin-guide.md#what-the-kernel-exports-to-plugins).

```
┌──────────────────────────────────────────────────────┐
│                   Plugin Manager                     │
│               (apluggy.PluginManager)                │
│                                                      │
│  ┌─────────────┐  ┌─────────┐  ┌──────────────────┐ │
│  │ Persistence │  │   IRC   │  │  Agent Plugin    │ │
│  │   Plugin    │  │Transport│  │  (agent loop)    │ │
│  │  (DB/conv)  │  │ Plugin  │  │                  │ │
│  └──────┬──────┘  └─────────┘  └────────┬─────────┘ │
│         │                               │           │
│  ┌──────▼──────┐              ┌──────────▼─────────┐ │
│  │   SQLite    │              │    llama-server     │ │
│  │  (sessions) │              │   (OpenAI-compat)   │ │
│  └─────────────┘              └────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Dependencies

```
apluggy          # async-aware pluggy wrapper
aiohttp          # async HTTP client (LLM API + web_fetch tool)
aiosqlite        # async SQLite for conversation persistence
pydantic         # tool schema generation from type hints
pydle            # asyncio IRC client
pyyaml           # YAML config parsing
```

## Hook Specifications

Defined in `hooks.py`. All hooks are async (called via `pm.ahook`)
except `register_tools` which is sync. This is a representative subset of
`AgentSpec` covering the turn-loop and message-lifecycle hooks; the
[Hook reference](#hook-reference) table below is the complete, authoritative
list of every hook (including lifecycle hooks like `on_init` not shown
here).

```python
class AgentSpec:
    async def on_start(self, config: dict) -> None
    async def on_stop(self) -> None
    async def on_message(self, channel: Channel, sender: str, text: str, trace_id: str) -> None
    async def send(self, channel: Channel, event: Outbound) -> None  # corvidae/outbound.py
    def register_tools(self, tool_registry: list) -> None  # sync; plugins append Tool instances or bare callables
    async def on_agent_response(self, channel: Channel, request_text: str, response_text: str, trace_id: str | None, meta: dict, logprobs: dict | None) -> None
    async def on_notify(self, channel: Channel, source: str, text: str, tool_call_id: str | None, meta: dict | None) -> None
    async def should_process_message(self, channel: Channel, sender: str, text: str, trace_id: str | None) -> bool | None
    async def should_send_response(self, channel: Channel, text: str, emission: str, trace_id: str | None, meta: dict) -> bool | None
    async def on_message_admitted(self, channel: Channel, trace_id: str, sender: str, text: str) -> None
    async def on_message_rejected(self, channel: Channel, trace_id: str, sender: str, text: str) -> None
    async def on_message_persisted(self, channel: Channel, trace_id: str, rowid: int | None, text: str, meta: dict) -> None
    async def resolve_llm_error(self, channel: Channel, error: Exception) -> str | None
    async def resolve_compaction(self, channel: Channel, conversation: ContextWindow, max_tokens: int) -> None
    async def transform_tool_result(self, tool_name: str, result: str, channel: Channel | None) -> str | None
    async def before_agent_turn(self, channel: Channel, trace_id: str | None, meta: dict) -> None
    async def after_persist_assistant(self, channel: Channel, message: dict) -> None
    async def transform_display_text(self, channel: Channel, text: str, result_message: dict) -> str | None
    async def resolve_conversation(self, channel: Channel) -> list[dict] | None
    async def resolve_channel(self, channel_id: str) -> Channel | None
    async def persist_message(self, channel: Channel, message: dict, message_type: MessageType) -> int | None
    async def before_compaction(self, channel: Channel, compacted_ids: list[int], cut: int) -> None
    async def on_compaction(self, channel: Channel, summary_msg: dict, retain_count: int, compacted_ids: list[int]) -> None
```

`create_plugin_manager()` in `hooks.py` creates the manager and adds
the `AgentSpec` hookspecs.

### Plugin-declared hookspecs

A plugin can own hooks that only it calls by setting `hookspecs` on its
class to a class of `@hookspec` methods (#229). The kernel doesn't change
per hook: `install_plugin_hookspecs(pm)` installs every declaration at
startup, after entry points load and before the arg-binding guard and
`on_init`. pluggy verifies hookimpls registered before their spec when
`add_hookspecs` runs, so registration order between owner and
implementers doesn't matter.

`install_plugin_hookspecs` makes two passes, so no outcome depends on
entry-point order. Pass one collects every declaration and fails on any
name with more than one owner, counting names that already have a spec
(`AgentSpec`). Pass two, per plugin, checks the name against the hook
grammar (`HOOK_GRAMMAR` in `hooks.py`, the same table
`tests/test_hook_grammar.py` uses for `AgentSpec`), with the owner's
registered plugin name right after the prefix (`on_web_<name>`). It also
checks the pluggy options match the prefix, installs the specs, and
requires the owner's own `trylast` seed for each `transform_` hook. Bare
imperatives are kernel-only. Every failure raises `PluginValidationError`
naming the hook, the plugin and the fix.

Implementers declare a hard (`depends_on`) or optional
(`optional_depends_on`) dependency on the owner. If an optional owner is
absent, its implementers' impls stay registered with no spec and are
never called. `report_orphan_hookimpls(pm)`, run after
`validate_dependencies`, logs each hookimpl with no spec at WARNING,
naming the plugin and the hook. The exception is a hook whose owner
segment names one of the implementer's `optional_depends_on` plugins
that isn't loaded (blocked or not installed), which logs at DEBUG.

Known limit: a plugin name containing an underscore makes the owner
segment ambiguous against a shorter name (`memory` could declare
`on_memory_tools_x`). That can't cause a silent clash, because a second
declaration of the same name still fails. Its only effect is that such an
orphan can be logged at DEBUG when the real owner is a different absent
plugin.

### Broadcast dispatch and result resolution

All hooks are called via `pm.ahook.<hook_name>(...)`. A broadcast hook
runs every registered implementation and returns a list of results; a
`firstresult=True` hook (below) stops at the first non-None result and
returns that single value. Gate hooks
that need a single resolved value pass the result list to
`resolve_reject_wins(results)` from `hooks.py`: any `False` in the results
returns `False`; otherwise any `True` returns `True`; otherwise `None`.
`should_process_message` and `should_send_response` are the hooks resolved
this way.

Hooks marked `firstresult=True` in the hookspec (`resolve_llm_error`,
`resolve_compaction`, `transform_tool_result`, `transform_display_text`,
`resolve_conversation`, `resolve_channel`) use pluggy's own first-non-None-wins dispatch — the
first handler to return a non-None value stops the chain; no separate
resolver function is involved. `transform_tool_result` and
`transform_display_text` are wrapper chains: `@hookimpl(wrapper=True)`
implementations compose transforms over an identity seed.

Pure broadcast hooks (e.g. `on_start`, `on_stop`) resolve neither way;
their return values are ignored. `persist_message` is the one broadcast
hook resolved with `resolve_single_result`: every implementation runs and
at most one (persistence) returns a value.

Hook names follow a grammar: the prefix tells you the resolution
(`on_`/`before_`/`after_` observe, `should_` vetoes, `resolve_` takes the
first non-None result, `transform_` runs a wrapper chain), and a short
allowlist of bare imperatives covers the rest. The table and the rule for
adding a hook are in [plugin-guide.md](plugin-guide.md#hook-grammar);
`tests/test_hook_grammar.py` enforces the name/option half for
`AgentSpec`, and `install_plugin_hookspecs` enforces it at startup for
[plugin-declared hooks](#plugin-declared-hookspecs), which also carry the
owner's plugin name after the prefix.

### Hook reference

| Hook | Semantics | Call site |
|------|-----------|-----------|
| `on_init` | broadcast | `Runtime.start()`, after all plugins are registered, before `on_start`; store `pm`/`config`, resolve other plugins — no runtime resources |
| `on_start` | broadcast | daemon startup, after config load |
| `on_stop` | broadcast | SIGINT/SIGTERM |
| `on_message` | broadcast | inbound message from any transport; the transport mints and passes `trace_id` (required, no default) |
| `send` | broadcast (imperative) | outbound delivery. `Agent` sends `OutboundMessage` (replies, error text), `OutboundThinking` (when the LLM response carries `reasoning_content`), `OutboundProgress` (intermediate text alongside tool calls) and `OutboundToolStatus` (tool call dispatched); `TaskPlugin` sends `OutboundToolStatus` on completion. Transports receive it through `TransportPlugin.send`, which filters by `owns(channel)` and dispatches by event type |
| `register_tools` | broadcast (sync) | startup tool collection |
| `on_agent_response` | broadcast | after agent loop produces a text response; no default implementation |
| `on_notify` | broadcast | inject a notification into a channel queue |
| `should_process_message` | broadcast / resolve_reject_wins | `on_message`, before enqueue, carrying the transport-minted `trace_id`; no default implementation; None from empty broadcast allows all messages |
| `should_send_response` | broadcast / resolve_reject_wins | the [output gate](#the-output-gate): before each outbound emission, with `emission` in `final`, `progress`, `thinking`, `error` and the item's `trace_id` and `meta`; a veto of `final` stores the row as `UNSENT`; `CritiquePlugin` vetoes a critique-origin `final` that is exactly `NO_REPLY`; a raising implementation fails open (logged, sent); no kernel implementation, so with none registered everything is sent |
| `on_message_admitted` | broadcast | `on_message`, after a USER message is admitted; fires exactly once per admitted message |
| `on_message_rejected` | broadcast | `on_message`, after a USER message is vetoed; fires exactly once per rejected message |
| `on_message_persisted` | broadcast | dequeue, after the trace-originating row is persisted; fires once per originating item regardless of outcome (`rowid` may be None) |
| `resolve_llm_error` | firstresult=True | `_run_turn`, after LLM exception; return error string or None for default |
| `resolve_compaction` | firstresult=True | `_process_queue_item`, step 6; the first non-None return stops the chain (`CompactionPlugin` is `trylast`, so a default-priority replacement runs first); the caller ignores the value |
| `transform_tool_result` | firstresult=True wrapper chain | `dispatch_tool_call` (both subagent and main agent paths), after tool execution; return replacement string or None for default |
| `before_agent_turn` | broadcast | `_process_queue_item`, before LLM call; plugins inject context into conversation log |
| `after_persist_assistant` | broadcast | `_process_queue_item`, after assistant message is persisted; plugins may mutate the in-memory dict |
| `transform_display_text` | firstresult=True wrapper chain | `_resolve_display_text`, before the reply is sent; return transformed text or None to leave unchanged |
| `resolve_conversation` | firstresult=True | `_process_queue_item`, when `channel.conversation is None`; return list of tagged message dicts or None to defer |
| `resolve_channel` | firstresult=True | `SchedulerPlugin`, for a persisted channel id the registry does not hold (e.g. a reminder due after a restart); a transport returns the live Channel for ids it owns, created with transport-specific config, or None to defer. `SignalPlugin` and `DiscordPlugin` implement it. A transport may also return None for an id it owns but cannot build yet (Signal before its first alias resolution). For an id whose prefix names a transport that implements the hook, the scheduler retries; for other ids, callers fall back to `registry.get_or_create` from the split id |
| `persist_message` | broadcast (imperative) | `_process_queue_item` after every `conv.append()`, and `FunnelPlugin` for the CONTEXT it persists; every implementation runs (persistence, JSONL logging). `PersistencePlugin` returns the `message_log` rowid and the caller resolves the list with `resolve_single_result` into `_db_id` |
| `before_compaction` | broadcast, awaited | `CompactionPlugin`, after the summary succeeded and before `replace_with_summary()`; the window still holds `messages[:cut]`, so a plugin can fork over the dropped prefix (see [Forks](#forks)); failures logged, never block compaction |
| `on_compaction` | broadcast | `CompactionPlugin`, after `replace_with_summary()`; side effects only (persistence) |
| `on_llm_request` | broadcast | `LLMClient.chat()` via LLMPlugin's injected observer, before the retry loop; side effects only (observability) |
| `on_llm_response` | broadcast | `LLMClient.chat()` via the observer, exactly once per call (success or terminal failure); side effects only (metrics, usage log) |
| `on_metrics` | broadcast | any plugin via `pm.ahook.on_metrics(...)`; built-in emission from `MetricsPlugin.on_llm_response`; never emit from inside an `on_metrics` implementation (recursion) |
| `on_config_reload` | broadcast | `ConfigWatcherPlugin`, when `agent.yaml`'s mtime changes; per-plugin error isolation |

## Channel System

`channel.py` provides multi-transport, multi-scope channel management.

**ChannelConfig** — per-channel config with agent-level fallback:

```python
@dataclass
class ChannelConfig:
    system_prompt: str | list[str] | None = None
    max_context_tokens: int | None = None
    reasoning_history: str | None = None   # drop | exchange | keep
    max_turns: int | None = None
```

`resolve(agent_defaults, runtime_overrides=None)` returns a dict with all
values resolved. Resolution order: built-in defaults → agent-level YAML →
matching transport-wide entries, least specific first → per-channel YAML →
runtime overrides (set by `set_settings` tool).
Defaults: `max_context_tokens=24000`, `reasoning_history="drop"`,
`max_turns=10`. `resolve` normalizes `reasoning_history` to the
`ReasoningHistory` enum and raises `ValueError` on any other value.
`validate_agent_config` (startup and config reload) and `load_channel_config`
reject an invalid value or the removed `keep_thinking_in_history` key up front.

**Channel** — identifies a transport + scope combination:

```python
@dataclass
class Channel:
    transport: str
    scope: str
    config: ChannelConfig
    conversation: ContextWindow | None = None
    created_at: float
    last_active: float
    turn_counter: int = 0                   # consecutive LLM turns without user message
    pending_tool_call_ids: set = set()      # tool call IDs awaiting results; cleared when all results collected
    runtime_overrides: dict = {}            # per-channel runtime overrides set by set_settings tool
    wide_entries: tuple[WideEntry, ...] = ()  # matching transport-wide entries, least specific first
```

`channel.id` returns `"{transport}:{scope}"`. `channel.disabled_tools` is
the union of `config.disabled_tools` and every `wide_entries` list.

**Transport-wide entries** (#203). A `channels:` key ending in `*` is a
`WideEntry(key, prefix, config)`: it covers every channel whose id starts
with `prefix`, the key without its `*` (`signal:*`, `discord:dm:*`,
`acp:*`). They exist for channels config can't name ahead of time: Signal
DMs keyed on ACIs, ACP sessions with minted ids, Discord channels the bot is
mentioned in. `parse_wide_key` accepts only `transport:*` and
`transport:<scope-prefix>*`; only a trailing `*` is special, so exact keys
that contain `*` (IRC allows it) load as before. Two prefixes that match the
same id are always prefixes of each other, so ordering by length is total:
more specific wins whatever order the keys appear in.

**ChannelRegistry** — lifecycle management. `get_or_create(transport,
scope, config)` returns an existing channel or creates one, assigning its
matching wide entries. `resolve_config(channel)` overlays the channel's
wide entries' scalar fields on `agent_defaults`, then calls
`channel.config.resolve(layered, runtime_overrides=channel.runtime_overrides)`,
so the existing per-field fallback gives exact > wide > agent and runtime
overrides are factored into every resolution. `set_wide_entries(entries)`
replaces the wide set and re-matches every registered channel, logging at
INFO each channel whose matches change.

`load_channel_config(config, registry)` pre-registers channels from
YAML before `on_start`. It is `parse_channel_config(config)` (pure: validates
every entry, exact and wide, and rejects keys outside `CHANNEL_ENTRY_KEYS`, touching no registry) followed by
`apply_channel_config(parsed, registry)`. Reload calls the two itself:
`ConfigWatcherPlugin` parses before it updates `agent_defaults`, so a rejected
`channels:` entry skips the whole reload. Applying sets the wide entries first, so
exact channels it then registers are matched at creation, and it never
registers a wide key as a channel. Wide entries therefore reload like
`agent:`: scalar keys apply to running channels from the next turn. A
conversation's system prompt is set once, when its `ContextWindow` is
created (compaction and forks keep it), so a prompt edit at any layer
reaches only conversations that start after the reload.

**resolve_system_prompt(value, base_dir)** — if `value` is a string,
returns it directly. If a list of paths, reads each file and
concatenates with `\n\n`. Relative paths resolve against `base_dir`
(the directory containing `agent.yaml`). Called at conversation init
time, so editing prompt files takes effect on the next new conversation.

## LLM Client

`llm.py` — thin aiohttp wrapper for the Chat Completions API.

```python
class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_body: dict | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        timeout: float | None = None,
        model_not_loaded_wait: float = 60.0,
    )
    async def start(self)
    async def stop(self)
    async def chat(self, messages: list[dict], tools: list[dict] | None = None, extra_body: dict | None = None) -> dict
    def request_model(self, extra_body: dict | None = None) -> str
```

`request_model` returns the `model` a `chat` call with that `extra_body` is
sent with: the configured model, overridden by the instance `extra_body`
and then the call's (`llm.<role>.extra_body.model` can change the model).
`chat` builds its payload from it, so the two can't drift. It is what
consolidation stamps on a record. `embed` is deliberately not routed
through it: it sends the configured `model` and ignores `extra_body`.

Retry parameters: `max_retries` (default 3, set to 0 to disable), `retry_base_delay`
(base exponential backoff delay in seconds), `retry_max_delay` (cap on backoff delay).
Retries apply to transient HTTP status codes (429, 500, 502, 503, 504) and connection
errors. Honors `Retry-After` response headers. `timeout` is the total HTTP timeout per
request in seconds; `None` uses aiohttp's session default (300s).

A llama-server router started with `--no-models-autoload` answers a request
for an unloaded model with HTTP 400 `"model is not loaded"` (#55). That
response is retried with the same backoff until a total of
`model_not_loaded_wait` seconds has been slept (the last sleep is clamped to
fit), independently of `max_retries`; `0` disables it. Any terminal non-2xx
response raises `LLMHTTPError`, a `ClientResponseError` subclass carrying the
response `body` (truncated to 2000 characters) and the client's `model`;
`is_model_not_loaded(error)` recognizes the unloaded-model case. A malformed
response envelope (missing `choices`, an empty choice list, or a choice with
no `message`) raises `ValueError` naming the provider model and endpoint, the
missing key, and the top-level keys actually received.

### Observability

Every LLM call is metered at the `LLMClient` chokepoint — not the turn
loop — so compaction, subagent, consolidation, appraisal, and critique
background calls are all covered.

**Observer seam.** `LLMClient.observer` (default `None`) is any object
with async `request(**kwargs)`/`response(**kwargs)` methods. `chat()`
mints a `request_id` per call, fires `request` before the retry loop,
and fires `response` exactly once — after success (with the envelope's
`usage` verbatim and measured latency) or when a terminal exception is
about to propagate (with `error` set, `usage=None`). Retried transient
attempts do not fire `response`. Observer failures are individually
caught and logged; they never break the call. `llm.py` stays free of
pluggy imports — `LLMPlugin` injects a `_HookObserver` (defined in
`llm_plugin.py`) that bridges to the `on_llm_request`/`on_llm_response`
hooks with the client's role ("main"/"background") and the current
attribution snapshot.

**Attribution.** `attribution.py` holds a single ContextVar dict
(`stage`, `channel_id`, `trace_id`, `meta`, plus stage-specific
extras such as `trigger` on consolidation calls). Callers set it at the
top of a logical operation and reset in a `finally`: the Agent turn loop
(`stage="turn"`), CompactionPlugin around its summary call
(`stage="compaction"`, shadowing the turn), and the subagent work loop
(`stage="subagent"`). Stages that can run as forks (compaction,
consolidation, appraisal, critique) add `mode="fork"` (set by
`Agent.fork`) or `mode="standalone"` around their own LLM call; it lands
in `usage_log.mode`. Contextvars snapshot at `asyncio.create_task`
time and TaskQueue workers are created once at startup, so `Task`
captures `contextvars.copy_context()` at creation and the worker runs
`task.work()` inside that context via
`asyncio.create_task(task.work(), context=task.ctx)`.

**Sinks.** `metrics.py` provides `MetricsPlugin` (translates
`on_llm_response` into `on_metrics` events: `llm.tokens.*`,
`llm.latency_ms`, `llm.errors`), `UsageLogPlugin` (one `usage_log` row
per call), and `MetricsJsonlPlugin` (JSONL event sink, config
`daemon.metrics_jsonl`). `outcome_log.py` owns the `exchange_log` table:
one row per exchange, written by `OutcomeLogPlugin`'s hookimpls at
message admission/rejection/persistence and updated in place as memory
retrieval, appraisal, and critique record their outcomes. All sinks are
fail-soft.

```sql
CREATE TABLE usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    stage TEXT,
    channel_id TEXT,
    trace_id TEXT,              -- NULL when the call ran outside any trace
    trigger TEXT,               -- 'compaction' or 'idle' for consolidation calls, else NULL
    mode TEXT,                  -- 'fork' or 'standalone' for stages that can fork (#51), else NULL
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER,
    latency_ms REAL,
    error TEXT
);

CREATE TABLE exchange_log (
    trace_id TEXT PRIMARY KEY,      -- one exchange = what one trace covers (see plugin-guide.md's Traces section)
    channel_id TEXT NOT NULL,
    origin TEXT,                    -- convention: 'user'|'reminder'|'critique'|'heartbeat'|'task'
    message_rowid INTEGER,          -- message_log.id of the originating message
    created_at REAL NOT NULL,
    retrieval_top_score REAL,       -- written by MemoryPlugin retrieval (decayed score)
    retrieval_top_similarity REAL,  -- written by MemoryPlugin retrieval (raw similarity, same candidate)
    retrieval_hit_count INTEGER,    -- written by MemoryPlugin retrieval
    probe_score REAL,               -- written by AppraisalPlugin stage 1
    appraisal TEXT,                 -- JSON; written by AppraisalPlugin stage 2
    provenance_snapshot TEXT,       -- JSON; written by CritiquePlugin
    outcomes TEXT                   -- JSON; merged by multiple writers
);
```

**Eval harness.** Deterministic metric functions (`recall_at_k`, `mrr`,
`tokens_of`) live in `tests/evals/metrics.py` and run in CI with no
network; the fixture format is documented there
(`tests/fixtures/memory_retrieval_basic.json` is the operator-authored
seed). The live runner `scripts/eval_memory.py` is out-of-band only;
live tests carry the `eval` marker (run with `--run-eval`).

## Agent Loop

The single-turn LLM primitive lives in `turn.py`. The multi-turn loop
that drives subagents lives in `tools/subagent.py`. Text post-processing
utilities for LLM output live in `thinking.py`.

**`turn.run_agent_turn`** — single LLM invocation. Used by Agent for
interactive message processing. Returns an `AgentTurnResult` with the
assistant message, any tool calls, text content, and latency.

```python
@dataclass
class AgentTurnResult:
    message: dict       # raw assistant message from LLM
    tool_calls: list[dict]
    text: str
    latency_ms: float

async def run_agent_turn(
    client: LLMClient,
    messages: list[dict],
    tool_schemas: list[dict],
    extra_body: dict | None = None,
) -> AgentTurnResult
```

`run_agent_turn` appends the assistant message to `messages` in place
before returning. Callers should not append it again.

**`tools.subagent.run_agent_loop`** — multi-turn blocking loop. Used
by subagent execution (its only caller is `SubagentPlugin._launch`).
Calls the LLM via `run_agent_turn`, executes tool calls inline, repeats
until a text response or max_turns. Injects `ToolContext` into tools
that declare a `_ctx` parameter. Accepts an optional `pm` keyword
argument; when provided, passes it to `dispatch_tool_call()`, which
calls the `transform_tool_result` hook after each tool execution. When
`pm` is `None`, the hook is skipped.

Also in `turn.py`:
- `_truncate(text, max_len)` — truncation utility for logging

In `thinking.py` (alongside `ThinkingPlugin`):
- `strip_thinking(text)` — removes `<think>...</think>` blocks
- `strip_reasoning_content(messages)` — removes `reasoning_content`
  from message dicts in place

## Tool System

`tool.py` provides tool registration and schema generation.

**Tool** — wraps a function with its schema:
```python
@dataclass
class Tool:
    name: str
    fn: Callable
    schema: dict
```

`Tool.from_function(fn)` generates the schema via `tool_to_schema`.

**ToolRegistry** — collection with `add()`, `as_dict()`, `schemas()`,
`exclude(*names)`.

**tool_to_schema(fn)** — generates a Chat Completions tool schema from
a function's type hints and docstring using pydantic. Parameters
starting with `_` are excluded from the schema (they are
system-injected, not LLM-supplied).

**ToolContext** — injected into tools that declare a `_ctx` parameter:

```python
@dataclass
class ToolContext:
    channel: Channel | None
    tool_call_id: str
    task_queue: TaskQueue | None
```

Tools that don't declare `_ctx` work without modification.

### Per-channel tool sets (`tool_policy.py`)

`channels.<id>.tools.disabled` (#149) narrows one channel on top of the
global `tools.disabled`. It is parsed into `ChannelConfig.disabled_tools`
by `load_channel_config`. The global list is already gone from the
registry, so the channel's effective set is the registry minus
`Channel.disabled_tools`: its own list unioned with every matching
transport-wide entry's (#203), so no layer can re-enable a tool another
disabled. `expand_disabled` adds coupled tools: `shell` and `task_pipeline`
go together (PRD §3.5.3), for the global list too.

- **Schemas are fixed per conversation.** When `Agent` initializes a
  channel's `ContextWindow` it resolves a `ToolView` (disabled set, tools
  dict, schema list) beside the system prompt and stores it as
  `conv.tool_view`. Every turn and every `Agent.fork` of the channel sends
  that same list object, so the prompt head is byte-identical across turns
  (principle 2) and forks keep prefix parity (see [Forks](#forks)).
  `Agent` also implements `on_compaction` and re-resolves the view there.
  Compaction already rebuilds the KV cache, so this is the only point where
  a running conversation's schemas can change, with one exception: a
  registry change.
- **Registry changes reach the next turn.** `ToolCollectionPlugin.generation`
  goes up whenever `rebuild_registry` yields different tool names or schemas
  (an MCP server reconnecting, #107/#109). `Agent` records the generation
  beside its `get_tools()` snapshot and, at the top of each queue item,
  re-reads the snapshot when the generation has moved, logging one INFO line
  naming the tools added and removed. Each conversation's view is then
  rebuilt lazily from the new set and the view's own `disabled` set (a
  reloaded `channels.<id>.tools.disabled` still waits for compaction). With
  no change nothing is rebuilt and the schema list stays the same object.
  The change costs one prefix-cache miss on that channel. Dispatch reads the
  refreshed dict, so a removed tool is refused as unknown.
- **Dispatch enforces.** Each tool task checks the name against
  `disabled_on(channel)` when it runs. That is the conversation view's set
  plus the live `Channel.disabled_tools`. A disabled call returns
  `refusal_message` as the tool result (naming the tool, channel and every
  config key that disables it, via `disabling_keys`, so a tool a wide
  entry disabled is reported under `channels.<wide key>.tools.disabled`)
  and is logged at WARNING with the same keys. Hiding is not access control: the
  model can call a tool it saw earlier or one named in injected text. This
  is a config list, not the `should_execute_tool` hook (#148).
- **Subagents inherit.** `SubagentPlugin` excludes `disabled_on(channel)`
  as well as itself from the registry it hands the subagent.
- **Hot reload.** `parse_channel_config` validates every entry before
  `apply_channel_config` applies any (and before `agent:` defaults change). For channels already registered it updates only
  `disabled_tools`, in place, so a Signal ACI channel that shares its
  alias's `ChannelConfig` follows. Transport-wide entries are replaced
  and re-matched on every live channel (`set_wide_entries`). Because
  dispatch reads the union, narrowing is enforced on the next call. The schemas, and any widening,
  wait for the next compaction or conversation start.

## Conversation Management

`context.py` provides in-memory context management. `persistence.py` handles all SQLite I/O.

**ContextWindow** — per-channel in-memory conversation context (`corvidae/context.py`).
All operations are synchronous; no database access:

- `append(message, message_type=MessageType.MESSAGE)` — appends to in-memory list with `_message_type` tag
- `build_prompt()` — returns `[system_msg, *messages]` with `_message_type` stripped.
  Every assistant `tool_calls` message is emitted contiguously with its results,
  at the arrival position of its *last* result (in place, with placeholders, if
  any call is still unanswered) — a corvidae convention, not compliance with a
  documented rule, since the OpenAI and Anthropic formats make tool-result
  adjacency normative but say nothing about a user turn arriving inside a tool
  cycle. The consequence this ordering secures: the emitted prompt never ends on
  an assistant message, which llama.cpp reads as a prefill and rejects under
  `enable_thinking`.
  A `tool` message whose call is not in the window (compaction dropped the
  call, #79) renders in place as a framed notification:
  `role: system`, content `[result of an earlier tool call that is no longer
  shown]` plus the result text, no `tool_call_id`. Strict chat templates
  reject a tool result with no preceding call. Because the frame is a
  `system` message it is then re-roled like any other (below). The window
  and `message_log` keep the stored `tool` message.
  `build_prompt(pending_tool_call_ids, mid_conversation_system_role="system")`
  also renders every `system` message after the leading system prompt
  (CONTEXT, untargeted notifications) under `mid_conversation_system_role`.
  `Agent` passes `agent.mid_conversation_system_role`. Strict chat templates
  (Qwen 3.8, Swift-1.5) reject a non-leading system message, and `user`
  keeps untrusted text out of the `system` role. Only the returned copy is
  re-roled: the window and `message_log` keep `role: system`. That matters
  because memory retrieval keys on a plain `user` tail, and because the
  rewrite must stay a pure function of each message for the prompt prefix
  to stay cache-stable.
- `token_estimate()` — sums the token count of every message via tiktoken
  (`cl100k_base`), with a character-based fallback when tiktoken is
  unavailable. Each message is sized by its JSON-serialized visible fields
  (role, content, `tool_calls`, `reasoning_content`, etc.), not just the
  `content` string, so a message whose size is mostly in tool-call
  arguments or reasoning content is still counted accurately.
- `replace_with_summary(summary_msg, retain_count)` — replaces older messages
  with a summary in-memory, retaining the `retain_count` most-recent entries.
  Raises `ValueError` if `retain_count` exceeds `len(messages)`.
- The window only grows at the tail (`append`) and truncates at the head
  (`replace_with_summary`). Nothing is inserted, removed or mutated
  mid-window, so there is no method for removing entries by type; stale
  `CONTEXT` entries retire by aging past the compaction boundary.

After each `conv.append()`, `Agent` fires `persist_message` (broadcast)
so persistence plugins can write to their storage. After `replace_with_summary()`,
`CompactionPlugin` fires `on_compaction` (broadcast) so persistence plugins can
update the DB boundary.

### Append-only log

Conversation history is irreplaceable and must never be deleted. The
`message_log` table in `PersistencePlugin` is append-only — no rows are
ever deleted, and their order is never rewritten. The one sanctioned
mutation is the operator-only `corvidae redact` tombstone (see
[Operator redaction](#operator-redaction-corvidae-redact)), which rewrites a
row's `message` in place but keeps the row. The DB is a complete audit log
including `reasoning_content`. This invariant applies to all persistence
paths: SQLite, JSONL, and any future storage backends.

`on_compaction` inserts a summary row whose timestamp encodes the boundary:
`oldest_retained_message.timestamp - 1e-6`. `resolve_conversation` filters with
`WHERE timestamp > summary_ts`, returning only the summary plus retained and new
messages. Old rows remain in the DB but are invisible to the working set. If the
newest summary row fails to decode, `resolve_conversation` falls back to an older
summary or the full history (see [resolve_conversation](#resolve_conversation)).

Old CONTEXT rows in the DB become invisible after the next compaction (their
timestamps fall below the boundary), exactly as they leave the in-memory window.

**Thinking token handling** — three layers:
- Display: `ThinkingPlugin.transform_display_text` calls `strip_thinking()`
  to remove `<think>` blocks before text is sent to the channel
- Persistent log: full message dict preserved (including `reasoning_content`)
- Active prompt: `ThinkingPlugin.before_agent_turn` applies the channel's
  `reasoning_history` policy to the whole in-memory window before each LLM
  call (`drop`: strip all; `exchange`: strip assistant messages before the
  last `role: user` message; `keep`: nothing). Under `drop`,
  `after_persist_assistant` also strips the just-appended message.

If `ThinkingPlugin` is not registered, `<think>` blocks pass through to the channel
and `reasoning_content` remains in the in-memory history.

### Message types

```python
class MessageType(str, enum.Enum):
    MESSAGE = "message"    # ordinary conversation turn (user or assistant)
    SUMMARY = "summary"    # compaction summary replacing older messages
    CONTEXT = "context"    # plugin-injected context (memory, notes, etc.)
    UNSENT = "unsent"      # assistant message a should_send_response gate vetoed
```

`message_type` is a persistence category, orthogonal to conversational role
(`user`, `assistant`, `tool`, `system`). `CompactionPlugin` only summarizes `MESSAGE`
entries; `SUMMARY` and `CONTEXT` rows survive compaction. `UNSENT` rows are
in the window and reload into it, but no transport sees them, including
history replay. Compaction's standalone summarizer and memory
consolidation read `MESSAGE` rows only, so they never summarize an `UNSENT`
row as said; a fork-mode compaction summary works from the window, where the
row's not-sent marker sits beside it (see [The output gate](#the-output-gate)).

In-memory message dicts carry a `_message_type` metadata key (set by `append()`).
`build_prompt()` strips it before returning messages to the LLM.

The `before_agent_turn` hook gives plugins a chance to inject entries (e.g., `CONTEXT`
rows) into the conversation before each LLM call. `Agent` fires
`persist_message` for each injected message after the hook returns.

### Database schema

```sql
CREATE TABLE message_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message TEXT NOT NULL,      -- JSON
    timestamp REAL NOT NULL,
    message_type TEXT NOT NULL DEFAULT 'message'
);
CREATE INDEX idx_log_channel ON message_log(channel_id, timestamp);
```

Plugins keep their own tables on the same connection, created lazily on
first use because the `on_start` broadcast is concurrent: `memory` and
friends ([MemoryPlugin](cognition.md#memoryplugin)), `exchange_log` (OutcomeLogPlugin),
`usage_log` (UsageLogPlugin), and `schedule`
([SchedulerPlugin](#schedulerplugin)).

**Connections to the database file.** Most plugins write on the one
persistence connection (`PersistencePlugin.db`). The others:

| Connection | Opened by | Writes |
|---|---|---|
| persistence (`PersistencePlugin.db`) | persistence `on_start` | `message_log`, and every other plugin's tables (scheduler, usage_log, retrieval_log, goals, ...) |
| memory write (`MemoryPlugin._write_db`) | memory `on_start`, after its schema exists | memory records only: consolidation (`memory`, `memory_vec`, `memory_fts` via trigger, `consolidation_watermark`) and `write_record`, each one `BEGIN IMMEDIATE` transaction. `corvidae memory reconsolidate` opens its own copy in a second process |
| appraisal probe, critique probe | each plugin's `on_start` | read only (`message_fts` probes) |

Only the memory write connection is isolated for atomicity: a commit on
the persistence connection can't land inside a memory transaction, and a
rollback there can't discard another plugin's pending row. WAL lets
readers on the persistence connection see a committed record at once.

## Agent

`agent.py` — the central plugin. Implements `on_start`, `on_message`,
`on_notify`, `on_stop`. Declares `depends_on = {"registry", "task", "llm", "tools"}`
so `validate_dependencies(pm)` verifies those dependencies are registered at startup.
Retrieves `ChannelRegistry` during `on_start` via
`get_dependency(pm, "registry", ChannelRegistry)` from `hooks.py`.

DB lifecycle (open/close) is delegated to `PersistencePlugin`. Conversation
initialization is handled directly in `Agent._process_queue_item`: a
`ContextWindow` is created, then `resolve_conversation` (firstresult=True) is
called so persistence plugins can populate history. `Agent` reads `_chars_per_token`
and `_base_dir` from config in `_start_plugin`.

The LLM client is borrowed from `LLMPlugin` (`get_dependency(pm, "llm", LLMPlugin)`)
in `_start_plugin`. The tool registry and `max_result_chars` are borrowed from
`ToolCollectionPlugin` (`get_dependency(pm, "tools", ToolCollectionPlugin)`).
Agent does not own either lifecycle.

`Agent.started` is True from the end of `on_start` until `on_stop`.
`Runtime.start()` broadcasts `on_start` to every plugin concurrently and
calls `Agent.on_start` only afterwards, so a producer whose background
task begins in its own `on_start` (`SchedulerPlugin`) waits for this flag
before calling `on_notify`; otherwise the queue consumer could run before
`Agent` has a registry and client.

Inbound messages are held, not dropped (#71). A transport can deliver from
its own `on_start`, before step 12 starts the Agent; `Agent.on_message`
checks the trace id first (an empty one raises at once), then awaits an
internal started event until the Agent has started (the gate is `_started`, not
the client: a config reload can set the client earlier). Once the Agent starts,
the held call proceeds normally: the gate, `on_message_admitted` and the
enqueue all happen before `on_message` returns, so a transport that frees
resources in a `finally` around `on_message` (the Signal and Discord typing
slots) is unaffected. Held calls from one channel resume in arrival order.
If the Agent stops before it starts (`Agent.on_stop`, or
`Runtime._cleanup_failed_start` through `Agent.release_held_messages`),
every held call returns without admission and logs one WARNING naming the
channel and trace id. There is no cap on held messages: the window is
bounded by startup.

`Agent.queues` is a public `dict[str, SerialQueue]` (keyed by channel ID).
Queues added after startup are visible to other plugins that hold a reference
to this dict because the reference is live.

### Message processing

A transport mints the trace id for each inbound user message
(`corvidae.trace.mint_trace_id()`, a stdlib-only leaf module;
`corvidae.agent` re-exports it) and passes it to `on_message`. `Agent.on_message`
uses it verbatim for the gate, the admitted/rejected hooks and the
`QueueItem`, so `on_message_persisted` carries the same id. It never mints
for a USER item: a falsy id raises `ValueError`, checked before the
readiness check so a mis-wired transport fails even while the agent starts.
Omitting the argument entirely fails at the hook call (`HookCallError`).
Notification-born items with no inherited `meta["trace_id"]` are the
exception: the Agent mints their id at dequeue.

`on_message` broadcasts `should_process_message` and passes the result
list to `resolve_reject_wins`. If the resolved result is `False` (identity
check, not falsiness), the message is dropped. Any other result (`True`
or `None`) proceeds to enqueue.

`on_message` and `on_notify` both enqueue a `QueueItem` onto a
per-channel `SerialQueue`. The queue's consumer calls
`_process_queue_item`, which delegates to four helper methods:

1. **`_build_conversation_message`** — converts a `QueueItem` into a
   conversation message dict and `request_text`. Returns `None` on
   unrecognized role (logged, item dropped).
2. When `channel.conversation is None`, creates a `ContextWindow`, calls
   `resolve_conversation` (firstresult=True) to populate history, and assigns
   `channel.conversation`
3. Resets `turn_counter` to 0 when the item originates a new trace
   (`item.originating`, set during trace resolution: every USER
   item, and a notification with no inherited `meta["trace_id"]`)
   **and** no tool cycle is in flight (`channel.pending_tool_call_ids` is
   empty). An item arriving mid-cycle is charged to the running cycle
   instead (R4), so a chatty channel cannot extend `max_turns`. Keying on
   `originating` rather than the USER role makes `max_turns` a
   per-exchange budget for clock-driven exchanges too: a channel fed only
   by scheduler notifications would otherwise exhaust it after ten
   responses and have every later tool call suppressed.
4. Resolves channel config (including `max_turns`)
5. Appends the inbound message to conversation; fires `persist_message`
   (broadcast: every implementation runs; `PersistencePlugin`'s rowid is
   resolved with `resolve_single_result` and stored as `_db_id`)
5b. **Tool result batching** — if the item is a tool-result notification
    (`role=NOTIFICATION` with `tool_call_id` set), removes the ID from
    `channel.pending_tool_call_ids`. If the set is non-empty after the
    removal, returns immediately (more results still pending). The LLM call
    is deferred until the last result clears the set. User messages can
    interleave during this window — they don't clear the pending set.
6. Calls `resolve_compaction` (`firstresult=True`): the first non-None
   return stops the chain, so a default-priority replacement runs before
   the `trylast` `CompactionPlugin`; the return value is not used
7. Calls `before_agent_turn` hook (broadcast); fires `persist_message`
   for any messages injected by the hook
8. **`_run_turn`** — calls `run_agent_turn` (single LLM invocation).
   On exception, calls `resolve_llm_error` (`firstresult=True`; if the hook
   itself raises, the failure is logged at WARNING and the default message
   is used), sends the error message to the channel, and returns `None` to abort
   processing. `LLMPlugin` implements it `trylast`: for an unloaded
   backend model it returns a message naming the model; for any non-2xx
   `LLMHTTPError` it names the status, an actionable hint (API key,
   quota, endpoint/model) and a short provider body; for connection and
   timeout failures it points at `llm.main.base_url`. It defers (`None`)
   on every other error, so the generic message still applies.
8b. **The final send gate.** When the result's tool calls will not
    dispatch (no tool calls, or `turn_counter >= max_turns`), resolves the
    display text once (the max-turns fallback on that branch) and fires
    `should_send_response(emission="final")` over it. On the max-turns
    branch, `tool_calls` is stripped from the assistant message before it
    is appended, so the row and the window copy both lose it, whatever the
    verdict.
9. Appends the assistant message to conversation (as `UNSENT` after a
   final veto, else `MESSAGE`); fires `persist_message`. Then sends
   `OutboundThinking` for `reasoning_content`, unless the final was vetoed
   (the reasoning goes with it) or `should_send_response(emission="thinking")`
   vetoes it
10. Calls `after_persist_assistant` hook (broadcast); `ThinkingPlugin`
    uses this to strip `reasoning_content` from the in-memory copy. Then
    appends a not-sent marker for each vetoed emission
11. **`_handle_response`** — decision point:
    - **Tool calls, under limit** (`turn_counter < max_turns`):
      increment counter, send the intermediate text as `OutboundProgress`
      unless `should_send_response(emission="progress")` vetoes it (then
      a marker is appended instead), dispatch tool calls as Tasks, return
      without sending a response
    - **Tool calls, at limit**: suppresses tool dispatch. Uses the display
      text step 8b resolved (`MAX_TURNS_FALLBACK_MESSAGE` as fallback),
      fires `on_agent_response`, sends text response unless the final was
      vetoed.
    - **No tool calls**: increments counter, uses the display text step
      8b resolved, fires `on_agent_response`, sends text response unless
      the final was vetoed

Step 7's `conv.build_prompt(...)` never returns a prompt ending on an
assistant message. Three facts compose to guarantee it: step 4 only ever
appends a message produced by `_build_conversation_message`, which emits
`role: "user"`, `"tool"`, or `"system"` and never `"assistant"`; compaction's
`replace_with_summary` always retains at least one message
(`retain_count >= 1`), so it cannot strand step 4's append; and
`build_prompt`'s deferral never leaves an assistant `tool_calls` message
last — its call+results block always ends on a `tool` message, real or
placeholder. This matters because llama.cpp reads a trailing assistant
message as a prefill and rejects it under `enable_thinking`.

The Agent also feeds the kernel idle tracker (see [Clock and idle
tracker](#clock-and-idle-tracker-clockpy-idlepy)). It calls
`IdleTracker.note_activity(channel_id)` when an item is enqueued, and
`note_settled(channel.id)` when an item finishes and nothing remains on the
channel: its queue is empty and it has no pending tool calls (subagent
tasks included; their results arrive as notifications that settle the
channel again). The Agent only pushes; the tracker never reads the
queues. Without an `"idle"` plugin the Agent logs a warning and idle
signals are off.

**`_resolve_display_text`** calls `transform_display_text`
(`firstresult=True` wrapper chain). If the hook returns a value,
that value is used; otherwise falls back to `result.text`. When a
`fallback` is provided and the resolved text is falsy, the fallback is
returned instead. Hook exceptions are caught, logged, and the input text
is returned.

### The output gate

`should_send_response(channel, text, emission, trace_id, meta)` is the
outbound mirror of `should_process_message`: reject-wins over every
implementation, any `False` vetoes. It fires at four sites, in two modes.

**The persistence-controlling firing** (`emission="final"`, step 8b) runs
between generation and persistence, and only for results whose tool calls
will not dispatch: a text result, or the max-turns branch. Never for a
result whose calls dispatch: hiding that assistant message would orphan its
`role: "tool"` rows on reload, which OpenAI-compatible servers reject. A
veto:

- persists the assistant message as `MessageType.UNSENT`;
- skips the `OutboundMessage` send;
- still fires `on_agent_response` (the agent thought it; critique can
  still object);
- appends a not-sent marker to the window.

The max-turns branch strips `tool_calls` from both the persisted row and
the window copy regardless of the verdict: those calls never dispatch, so a
row that kept them would dangle on reload, and stripping only the row would
leave the live and reloaded windows disagreeing.

**Per-emission firings** at the progress, thinking and error-fallback
sends suppress that emission only. The assistant row stays `MESSAGE`, keeps
its `tool_calls`, and tool dispatch proceeds. A progress or thinking veto
appends a marker; an error veto (in `_run_turn`, where no assistant message
exists) suppresses the send and nothing else. A vetoed final takes its
reasoning with it: the thinking send is skipped without consulting the gate.

**The not-sent marker** is a `CONTEXT` system row the Agent appends and
persists itself (`corvidae.context.unsent_marker_text`), right after the
assistant row, so it reloads beside it. It opens `[not sent: <emission>]`
and tells the model the channel didn't see that text and not to cite it as
said. Only the progress marker quotes its text (40 characters): `build_prompt`
moves a tool-calls block to its last result, so a progress marker renders
before the block it describes and needs the excerpt to say which text it
means. The final marker sits right below its row; the thinking marker quotes
nothing, so reasoning never outlives `reasoning_history: drop` inside one.

**Transports never see a vetoed emission, including on replay.** The web
transport's history skips `UNSENT` rows and drops the progress or thinking
frame of an assistant row followed by a progress or thinking marker. ACP's
`load_session` replay skips `UNSENT` rows and an assistant row followed by a
progress marker (it doesn't replay reasoning). `unsent_marker_emission(content)`
is the shared parser.

**Failure modes.** `_gate_send` catches any exception from the hook, logs it
at WARNING with the emission and channel, and sends: a broken gate never
silences the agent. With no implementation registered (the kernel ships
none), every emission is sent; the only behavioral change from before the
hook is the max-turns `tool_calls` strip. `meta` is the item's trace
metadata dict; origin is not a kernel concept, so a gate that needs it reads
`meta["origin"]` (stamped by producers such as the scheduler) the way
cognition's `GatePlugin` does ([cognition.md](cognition.md#gateplugin)).

### Tool call dispatch

`_dispatch_tool_calls` records all tool call IDs in
`channel.pending_tool_call_ids`, then creates a `Task` for each tool call
and enqueues it on the `TaskQueue` (accessed via
`getattr(pm.get_plugin("task"), "task_queue", None)`). Each task's `work`
closure calls `dispatch_tool_call()` from `tool.py`, which handles JSON
parsing, unknown-tool detection, invocation, error wrapping, logging, and
the `transform_tool_result` hook. The closure returns `result.content`; the
TaskPlugin delivers it via `on_notify`.

`TaskPlugin` is declared in `Agent.depends_on`. Without a task queue,
tool dispatch logs an error naming the channel and the requested tools (with
the remedy: the `task` plugin is registered, since startup requires it, but
has no queue, so check that `TaskPlugin.on_start` ran and that the plugin
registered as `task` is `TaskPlugin`) and sends an error result for each call
through `on_notify`. If that broadcast raises for a call, dispatch logs a
WARNING with the channel, tool call id and tool name, and falls back to
`Agent.deliver_tool_error` (#110). One call's failure does not stop the rest
of the batch.

Closure capture uses default argument binding to avoid the
loop-variable capture bug.

### Notification-driven continuation

When a tool task completes, `TaskPlugin` fires `on_notify` with the
result and `tool_call_id`. `Agent.on_notify` enqueues a
notification `QueueItem`. The next `_process_queue_item` call uses
`_build_conversation_message` to format it as a `role: "tool"` message
(if `tool_call_id` is set) or a system message, appends it to
conversation, and calls `_run_turn` again. The system message is stored
as such and sent as `agent.mid_conversation_system_role` (see
`build_prompt` above).

If the `on_notify` broadcast raises, `TaskPlugin` logs a WARNING with the
channel, task id and tool call id and, for a tool task, calls
`Agent.deliver_tool_error`, which queues an error `role: "tool"` result
straight onto the channel's serial queue. Without it the id would stay in
`pending_tool_call_ids` and the tool cycle would never finish (#173). The
broadcast runs every hookimpl via `asyncio.gather`, so a third-party
hookimpl can raise after `Agent.on_notify` already queued the real result.
The Agent therefore records each queued tool result as
`(channel_id, tool_call_id)` in `_queued_tool_results` until the item is
processed (dropped in the same step that discards the id from
`pending_tool_call_ids`), and `deliver_tool_error` does nothing if the
result is already queued or no longer pending. The model never sees two
results for one call. A task lost before it reaches `_on_task_complete`
is not covered. The no-task-queue path in `_dispatch_tool_calls` uses the
same fallback and guards.

Multi-turn reasoning emerges from this cycle without blocking the
channel queue.

### QueueItem

```python
class QueueItemRole(Enum):
    USER = "user"
    NOTIFICATION = "notification"

@dataclass
class QueueItem:
    role: QueueItemRole
    content: str
    channel: Channel
    sender: str | None = None
    source: str | None = None
    tool_call_id: str | None = None
    meta: dict = field(default_factory=dict)
```

## PersistencePlugin

`persistence.py` — manages the SQLite database lifecycle and conversation
persistence. Registered as the `"persistence"` entry point. Declares
`depends_on = set()`.

`PersistencePlugin.on_start` reads `daemon.session_db` from config (default
`"sessions.db"`), opens the database with `aiosqlite.connect`, and runs schema
creation. `on_stop` closes the connection.

`PersistencePlugin.db` is public for test injection: setting `persistence.db`
before `on_start` causes the `if self.db is None:` guard to skip the open.

### resolve_conversation

Implements `resolve_conversation` (firstresult=True). Called by `Agent` when
`channel.conversation is None`. Queries `message_log` for the channel, applying
the timestamp filter so only the compaction boundary and newer rows are returned.
Returns a list of tagged message dicts (with `_message_type` set). Logs an INFO
event when history is loaded.

A corrupt summary row degrades the load instead of aborting it. A summary row
that fails to decode (invalid JSON, invalid UTF-8, or JSON that is not an
object) is skipped with a WARNING naming its row id and channel, and the
next-newest summary is tried with its own timestamp boundary. Because
`message_log` is append-only, an older summary plus the rows after it still
covers the whole conversation, only less compressed. If no summary decodes, the
query widens to the full non-summary history. Either fallback can return more
history than the last compaction kept; the `resolve_compaction` pass before each
LLM call can compact it again once it crosses the compaction threshold.

A corrupt ordinary row is skipped the same way, on both the summary-boundary
branch and the no-summary branch: a non-summary row whose message is invalid
JSON or invalid UTF-8, decodes to something other than a JSON object, or carries
a `message_type` outside `MessageType` is dropped with a WARNING naming its row
id and channel, and the rows on either side of it still load in order. Every
query selects `CAST(message AS BLOB)`, so invalid UTF-8 is caught while
decoding in Python instead of raising inside the sqlite3 fetch, where the cursor
cannot step past the row.

Every other reader of `message_log.message` shares this decoder
(`decode_message_row` and `parse_message_type` in `corvidae/persistence.py`)
and the same `CAST(message AS BLOB)` select, and skips a corrupt row with a
WARNING naming its row id and channel (never its bytes) and saying to inspect
or redact the row (#232):

- **Web history replay** (`_fetch_history_rows` / `_rows_to_frames`): a corrupt
  row, or one with an unknown `message_type`, among the newest rows is skipped
  and the other rows still replay; it no longer blanks the channel's history.
- **Memory** (consolidation, `recall_raw`, the `message_fts` backfill): see
  [cognition.md](cognition.md#corrupt-message_log-rows).
- **Redact** (below): the tool that fixes a corrupt row works on it.

### persist_message

Implements `persist_message` (broadcast). Called after every `conv.append()`.
Strips `_message_type` from the message dict before inserting into `message_log`.

### on_compaction

Implements `on_compaction` (broadcast). Called after `replace_with_summary()`.
Determines the timestamp boundary from the most-recent non-retained row, inserts
the summary row with `timestamp = boundary - 1e-6`, and commits.

**Graceful degradation:** without `PersistencePlugin`, `resolve_conversation` returns
no history and conversation events are not persisted.

## Re-consolidation (`corvidae memory`)

`corvidae_cognition/memory_command.py`, registered under `corvidae.commands`.
Operator-only, never a registered tool.

Every `memory` row carries a stamp (#299): `prompt_hash` (sha256 of the
consolidation instruction text, stored once in `memory_prompt(hash, text)`),
`mode` (`fork | standalone | direct`) and `model` (what the call was sent
with, from `LLMClient.request_model` or `ForkResult.model`). NULL stamps
mean the row predates stamping ("unknown"); `direct` rows (goal retirement,
`write_record`) were written by no consolidation call. The columns are added
in place by `add_column_if_missing`; no existing row is rewritten.

```
corvidae memory --db sessions.db stamps
corvidae memory --db sessions.db reconsolidate STAMP --config agent.yaml
corvidae memory --db sessions.db reconsolidate STAMP --dry-run
```

`stamps` and `--dry-run` open the database read-only and never migrate it;
every read is built from a schema probe, so pre-change and partly migrated
databases work. A live run is a second process on a live database: it
asserts WAL, uses its own `MemoryPlugin` write connection
(`BEGIN IMMEDIATE`, busy timeout 5 s), never runs `memory.on_start` (no
retention, backfill or idle watches), drops `llm.embedding` from its config
(so it can't create `memory_vec` or write `embedding_meta`), loads sqlite-vec
only when `memory_vec` exists, to delete a superseded record's vector (it
refuses before any LLM call if it can't load), and logs to stderr only.
Re-consolidation goes through the standalone path, never a fork, and
`reconsolidate_record` re-reads the old row inside the write transaction, so
a record merged, redacted, demoted or retrieved during the LLM call is
handled. Details of selection, inheritance and failure handling are in
[cognition.md](cognition.md#operator-cli-corvidae-memory).

## Operator redaction (`corvidae redact`)

`corvidae/commands/redact.py` — operator-only CLI for permanent content
removal. Never a registered agent tool; never reachable from any hook the
model can influence.

**Forms:**

```
corvidae redact --db sessions.db message <id> [<id2>...]   # specific message_log rows
corvidae redact --db sessions.db memory <memory_id>        # a record + its raw range
corvidae redact --db sessions.db range <start_id> <end_id> # message_log id range (inclusive)
```

A corrupt row (invalid JSON or UTF-8, JSON that is not an object, an
oversized integer) is tombstoned like any other. Every read casts the column
to BLOB and decodes through `decode_message_row`, so the row becomes
`{"role": "unknown", "content": "[redacted by operator <date>]"}` (valid
UTF-8 JSON) with a WARNING naming the row and channel, and a row that did
decode keeps its `role`. The sample token for the verification step comes
back `None` for a corrupt row, so the CLI reports the FTS check as skipped
for it instead of crashing.

`--dry-run` prints affected row counts without writing. Both dry-run and
real runs count only rows that exist and were actually touched, and the
`message` and `range` forms print `<found> of <requested>` so a mistyped id
is visible. A `--db` path that is not an existing file aborts with
`Database not found: <path>` before opening a connection (opening would
silently create an empty database).

**WAL requirement.** The CLI opens its own connection (`PRAGMA busy_timeout
= 5000`, short transactions) and asserts WAL journal mode at startup. If
the mode is not WAL, it aborts with a message naming the current mode and
how to convert. The `corvidae` daemon configures WAL on startup; a
daemon-created DB is always WAL.

**sqlite-vec.** `memory_vec` is a `vec0` virtual table, so the CLI's
connection must load sqlite-vec before the cascade touches it. Each form
loads it right after connecting when `memory_vec` exists (lazily
imported, so kernel-only installs never need it). If it can't be loaded,
the form refuses before writing anything: one line naming `memory_vec`
and the remedy (install the `cognition` extra, or run redact in the
daemon's environment). Before #198 the cascade crashed on `vec0` after
`message_log` was already committed.

**Schema-presence probe (pre-memory-schema and kernel-only DBs).** Before
touching any surface, the CLI probes `sqlite_master` for the tables it
needs. The `message` and `range` forms always write the message
tombstone and skip each absent surface without printing anything: no
`message_fts` means there is nothing to clear (if `MemoryPlugin` creates
it later, its backfill indexes the tombstone, not the original); no
`memory` means no memory cascade; no `memory_vec` skips only the vec-row
delete. A scrubber whose table is absent returns None and counts 0 (the
CLI prints `0 <name> rows`; a library caller of `redact_messages` also
gets a `SKIP` line in its `notices` list). The `memory` form needs the `memory` table and exits with an error
(`memory table not present in this database`) without it. The CLI never
creates schema; schema is daemon-owned.

**Cascade (per-form behavior).**

- **`message` form:** for each supplied `message_log` id, rewrites the
  `message` JSON — keeping `role` and structural keys, replacing `content`
  with `[redacted by operator YYYY-MM-DD]`, dropping `tool_calls` and
  payload fields. Row id and timestamps are unchanged. The `message_log_au`
  trigger propagates the change to `message_fts` automatically (when
  present). Then cascades to memory: for every `memory` row whose
  `[msg_id_start, msg_id_end]` contains the redacted message's id **and**
  whose `channel_id` matches — the `message_log` id sequence is global
  across all channels, so a numeric-range match alone would over-redact
  unrelated channels — tombstones the memory summary to `[redacted by
  operator]`, sets `redacted=1`, `indexed=0`, `embedded=0`, and deletes its
  vec row. The `memory_au` trigger propagates to `memory_fts`.
- **`memory` form:** resolves the record's `[msg_id_start, msg_id_end]`
  range with `WHERE channel_id = record.channel_id AND id BETWEEN ...` —
  the per-channel predicate is load-bearing because the global-id range
  otherwise includes interleaved foreign-channel rows — then tombstones
  those messages. Tombstones the target memory record itself and cascades
  to any other same-channel memory records whose ranges intersect the
  tombstoned message ids.
- **`range` form:** collects all `message_log` ids in the range and
  delegates to the `message` form cascade.
- **Registered scrubbers (all forms, #160):** each form then runs every
  scrubber registered under the `corvidae.redact` entry-point group, once
  per channel, over the message ids it tombstoned (contract in
  [plugin-guide.md](plugin-guide.md#registering-a-redact-scrubber)). The
  kernel doesn't know the tables. The cognition extra registers
  `exchange_log` (`corvidae_cognition.outcome_log:redact_exchange_log`),
  which scrubs the conversation text that table holds. The
  exchange that owns a redacted id X (in that channel, the row with the
  largest `message_rowid <= X`) has every `appraisal.stage2.past_claims`
  claim and query replaced with the tombstone text. The number of claims
  and queries is kept for calibration, and the other appraisal fields are
  untouched. Every exchange in the channel with `message_rowid >=` the
  smallest redacted id has its `provenance_snapshot` set to NULL. That
  deliberately over-scrubs: the snapshot holds the CONTEXT in the window
  at the turn, nothing maps it back to rows, and any later window may
  have carried X's text. Exchanges with a NULL `message_rowid` aren't
  reached. The CLI prints `<N> <name> rows` per scrubber, here
  `<N> exchange_log rows` (distinct exchanges changed; a re-run that
  changes nothing counts zero), and the result dict carries the counts
  as `scrubbed: {name: N}`.

**Verification pass.** After each cascade, the CLI extracts a sample token
from the original content (before tombstoning) and runs FTS `MATCH` queries
against `message_fts` and `memory_fts`. It prints `"verified: 0 hits for
'<token>' in <tables>"` when clean or `"WARNING: FTS still contains hits for
'<token>': <table>: <N> hits"` when the cascade left searchable content — every invocation proves its own
completeness. Skipped when `--dry-run` is in effect or when no suitable
sample token can be extracted.

The memory cascade above needs the cognition extra's tables. On a kernel-only
database (`memory`, `memory_vec`, `memory_fts` and `message_fts` absent) the
`message` and `range` forms tombstone the `message_log` rows, and the
verification line reports `verified: no FTS surfaces present`; the
`memory` form exits with the error above. If the extra was installed once
and later removed, `message_fts` and its `message_log_ai`/`message_log_au`
triggers stay in `sessions.db` and keep indexing into a table nothing reads;
`redact` still clears them because it probes for them
([cognition.md](cognition.md)).

## JsonlLogPlugin

`jsonl_log.py` — writes an append-only JSONL log alongside the SQLite
conversation store. Registered as the `"jsonl_log"` entry point.

Configured via `daemon.jsonl_log_dir` in `agent.yaml`. If the key is
absent, the plugin is a complete no-op — all hookimpls return early.
When configured, `on_start` resolves the directory relative to
`_base_dir` and creates it if necessary.

### persist_message

Implements `persist_message` (broadcast). Strips `_message_type`
from the message dict and writes a JSON line with `ts`, `channel`,
`type`, and `message` fields to a per-channel `.jsonl` file.

### on_compaction

Implements `on_compaction` (broadcast). Writes the compaction summary as
a JSON line with `type: "summary"`.

### File handles

Per-channel file handles are opened lazily in `_get_handle` and held
open for the plugin lifetime. Channel IDs are sanitized (`/` and `:` →
`_`) for filenames. All handles are flushed and closed in `on_stop`.

## Clock and idle tracker (`clock.py`, `idle.py`)

Time passing and the system going quiet are kernel inputs. `Runtime.start()`
builds one `Clock` and one `IdleTracker` and registers them with the plugin
manager as `"clock"` and `"idle"` before the entry points load. They are
not entry-point plugins. Listing either in `plugins.disabled` unregisters
it, and startup then fails dependency validation. Plugins declare
`depends_on = frozenset({"clock"})` or `{"clock", "idle"}` and resolve them
with `get_dependency`.

**Clock.** `now()` is wall time, `monotonic()` is monotonic time, and
`sleep(delay, wake=None)` sleeps on the monotonic clock until `delay`
passes or the optional `asyncio.Event` is set. `call_later(delay, cb,
name=...)` and `call_at(when, cb, name=...)` run an async callback once and
return a `Timer` whose `cancel()` stops it while it is still waiting
(a no-op once the callback has started). `call_at` waits on wall time in
steps of at most an hour, so a suspend or an NTP step is noticed within
the hour. Timers are one-shot: a periodic job re-arms itself from its
callback. A callback that raises is logged with the timer's name. A callback
must let `asyncio.CancelledError` through, because `Clock.close()` stops
running callbacks that way.

**IdleTracker.** It answers "has this channel been quiet for N seconds?" and
"has the whole system?". A channel is busy from `note_activity` until
`note_settled`, and quiet from then on. The system is quiet when no
channel is busy, counted from the last settle (or from `start()`).

A plugin registers a watch:

```python
self._watch = idle.watch_channels(1800, self._on_quiet, name="my.plugin")
self._sys = idle.watch_system(60, self._on_system_quiet, name="my.plugin.sys")
```

`watch_channels(period, cb)` calls `cb(channel_id)` once a channel has been
quiet `period` seconds. `watch_system(period, cb)` calls `cb()` once the
whole system has been. Both return an `IdleWatch` with `cancel()`. A watch
repeats every `repeat` seconds (default `period`) while the quiet lasts, and
only if nothing became active while the callback ran. Activity cancels the
waiting timers. An upkeep callback (consolidation, retention) never starts
a turn or enqueues anything; it does plugin-owned work. The one agency
watcher, cognition's `HeartbeatPlugin`, starts a turn from its system
watch through `on_notify` (principle 13; [cognition.md](cognition.md#heartbeatplugin)).
A callback that awaits before acting can call `system_quiet()` to re-check
that no channel went busy in the meantime.

This is a registration API, not a hook. A hook cannot carry per-plugin
periods, and a broadcast with a global cooldown starves any plugin whose
period is longer than the cooldown.

**Restart seed.** A freshly booted daemon has seen no traffic, but its
channels have history. `PersistencePlugin.on_start` calls
`note_restored(channel_id, last_activity)` for every channel in
`message_log`, so each is quiet since its last message (wall-time age,
converted to monotonic). A channel with history that has not spoken since
before the restart therefore reaches its idle period on schedule instead
of never. `note_restored` ignores channels the tracker already knows.

**Lifecycle.** The tracker does not fire until `Runtime.start()` calls
`idle.start()`, after the Agent can take work. Shutdown runs
`agent.on_stop`, the `on_stop` broadcast, `idle.stop()` and then
`clock.close()`, which cancels any timers plugins left behind. Plugins
should still cancel their own watches and timers in `on_stop`.

**Failed start.** If `Runtime.start()` raises (`Exception`, not the
`SystemExit` a config error raises before any plugin loads),
`Runtime.run` runs `_cleanup_failed_start()` before re-raising: the
`on_stop` broadcast, `idle.stop()` and `clock.close()`, under the same 3 s
bound as `stop()` (as there, an `on_stop` that raises skips the last two;
`asyncio.run` cancels leftover timers on exit). `agent.on_stop` is skipped, since the Agent may not
have started. Without this, persistence's open aiosqlite connection (a
non-daemon worker thread) kept the interpreter from exiting, so a daemon
whose plugin failed to start stayed alive and silent (#177). A cleanup
failure or timeout is logged at WARNING and the start exception is the one
that propagates, so the process exits non-zero. A plugin's `on_stop`
therefore has to tolerate running without its `on_start` having finished.

**Config.** None. Each plugin sets its own period (for instance
`memory.idle_consolidate_after`). The former `daemon.idle_cooldown_seconds`
and `daemon.idle_poll_interval` keys are removed, and a config that
carries either fails at startup (a reload carrying one is rejected).

**Tests.** `tests/fake_clock.py` provides a fake clock and
`install_kernel_clock(pm)`, which registers a fake clock and a real
tracker on a plugin manager; tests advance time without sleeping.

## ThinkingPlugin

`thinking.py` — strips `<think>` blocks and `reasoning_content` for
display. Registered as the `"thinking"` entry point.

Implements three hooks, driven by the resolved `reasoning_history`:

- `before_agent_turn`: applies the policy to the whole in-memory window.
  `drop` strips every assistant message; `exchange` strips those before
  `completed_exchange_end()` (the last `role: user` message; tool results
  and system notifications do not end an exchange); `keep` is a no-op.
  Covering the whole window is what makes the policy hold for history
  reloaded from the DB, which always carries `reasoning_content`.
- `after_persist_assistant`: under `drop`, strips the just-appended
  message. The DB copy is already written; this only affects subsequent
  prompt builds.
- `transform_display_text`: calls `strip_thinking` on the response text.
  Returns the stripped string if it differs from the input, or `None`
  if no `<think>` tags were present.

**Graceful degradation:** without `ThinkingPlugin`, `<think>` blocks
pass through to the channel and `reasoning_content` remains in the
in-memory history regardless of `reasoning_history`.

## CompactionPlugin

`compaction.py` — implements the `resolve_compaction` hook to keep
conversation history within the configured token budget. Registered as
the `"compaction"` entry point. Declares `depends_on = {"llm"}`.
Logger name: `corvidae.compaction`.

**ContextCompactPlugin** (`context_compact.py`) has been removed: superseded
by `MemoryPlugin` (compaction-time memory consolidation plus retrieval), and
per-turn token stats now live in the `usage_log` table.

### resolve_compaction

Implements the `resolve_compaction` hook (`firstresult=True`) at
`trylast`, so it runs only if no earlier implementation (tryfirst or
default priority) returned non-None first. The caller does not use the return value.

**Algorithm:**

1. If `conversation.token_estimate()` — the same JSON-serialized
   per-message measure described above, applied consistently across the
   trigger check, the retain-walk, and `_summarize`'s truncation below —
   is less than 80% of `max_tokens`, return `None` (skip).
2. If `len(conversation.messages) <= 5`, return `None` (skip — too few messages to compact).
3. Backward walk: starting from the most recent message, accumulate messages
   until their token estimate would exceed 50% of `max_tokens`. The count
   of accumulated messages is `retain_count`. The cut
   (`len(messages) - retain_count`) is then moved so no tool cycle
   straddles it (`_clean_cut`), because a straddle drops a call while
   keeping its results, and makes the dropped prefix render differently
   from the start of the full window, which a compaction fork needs.
   First choice: back up with `snap_cut_to_tool_boundary` to keep the
   cycle, if that drops some dialog (not just the previous summary) and
   the window after compaction (system prompt, a 2000-token summary
   allowance, and the kept messages) stays at or below the midpoint
   between `compaction_retention` and `compaction_threshold`. Otherwise (a
   long-running call near the head with chat interleaved behind it) move
   forward with `snap_cut_forward` to drop the cycle whole, if at least
   half the retention budget survives. Otherwise keep the walk's cut, as
   before #51. Backing up without that bound could leave compaction
   dropping almost nothing and re-running every few messages. In-flight
   calls have no result yet and never straddle; one dropped by compaction
   leaves its later result orphaned, as does the walk's own cut when it
   stands. `build_prompt` renders such orphans as framed notifications
   (see `ContextWindow` above), so the cut logic need not prevent them.
4. If `retain_count >= len(conversation.messages)`, return `None` (all messages fit — no-op).
5. Filter the older (non-retained) messages to `MESSAGE` type only, excluding
   any `SUMMARY` entries. Strip `_message_type` metadata before passing to LLM.
6. With forks on for `compaction` (`agent.forks`, see [Forks](#forks)),
   fork the window at the cut with `agent.compaction_fork_prompt` (default
   `CompactionPlugin.DEFAULT_FORK_PROMPT`, `prompts/compaction_fork.md`)
   and use the reply as the summary. The prefix already holds any prior
   summary at its head, so no bounded copy is passed. If forks are off for
   the stage, or the fork raises or comes back empty (logged at WARNING),
   fall through to the standalone call:
   call `_summarize(new_messages_clean, prior_summaries_clean,
   max_tokens=max_tokens)` — sends the older messages to the LLM with a
   system prompt asking for a concise summary, incorporating any prior
   summaries carried forward from earlier compaction rounds. The system
   prompt is `agent.compaction_prompt` (default `DEFAULT_SUMMARY_PROMPT`)
   in every round; when prior summaries exist, the carry-forward
   instruction and the bounded prior-summary text are appended to it. Resolves the
   LLM client lazily via `get_dependency(pm, "llm", LLMPlugin)` on first
   call (not in `on_start`, because `on_start` hooks run in LIFO order and
   `CompactionPlugin.on_start` fires before `LLMPlugin.on_start`). The
   carried-forward prior-summary text is itself capped to a fraction of
   `max_tokens` (keeping the most recent portion) so it cannot grow
   unbounded across repeated compaction rounds. Derives a token budget for
   the messages being summarized as `max_tokens` minus the system prompt's
   own size minus a fixed reserve for the LLM's response and JSON overhead,
   then bounds the head and tail of the input against that budget — each
   side capped by both a message count (50) and half the token budget —
   inserting a `[...N messages omitted...]` marker between them when
   anything is dropped. This method is a separate method so tests can patch
   it via `patch.object`.
7. Build the summary message (`[Summary of earlier conversation]\n…`).
   Its role is `assistant`, unless the retained tail holds no `user`
   message (a long tool loop); then it is `user`, because strict chat
   templates (Swift-1.5 / Qwen 3.8) reject a prompt with no user query
   (#56). The role is persisted with the summary row, so a reload renders
   the same prompt.
8. Fire `before_compaction(channel, compacted_ids, cut)` (broadcast,
   awaited) while the window is still whole. `MemoryPlugin` forks the
   compaction-time memory record here. Failures are logged and ignored.
9. Call `conversation.replace_with_summary(summary_msg, retain_count)`.
10. Fire `on_compaction(channel, summary_msg, retain_count, compacted_ids)`
   (broadcast) so persistence plugins can update the DB.
11. Return `True`.

The summary LLM call is attributed `stage="compaction"` plus
`mode="fork"` or `mode="standalone"`.

**Graceful degradation:** without `CompactionPlugin`, the
`resolve_compaction` hook returns `None` on every turn and compaction
does not run. The conversation history will grow without bound until the
LLM context window is exceeded.

## Forks

`fork.py` plus `Agent.fork` — background processes run as branches of the
main window (GitHub issue #51; plan of record `plans/forks.md`). Off by
default: `agent.forks.enabled`.

A **fork** renders the channel's window exactly as the main turn does, up
to a cut, and appends one `user` instruction. The prompt is then a prefix of
what the server already holds, so it reuses the cached KV, and it carries
the agent's own system prompt, so first-person records and summaries are
written by the agent they describe.

```
fork_prompt = build_prompt(window[:cut]) + [{"role": "user", "content": instruction}]
```

**Prefix parity.** `Agent.fork(channel, instruction, *, cut=None,
extra_body=None)` builds with exactly what step 7 of the agent loop uses:
the client of the model whose window is forked (`llm.main`, unless a
stage forks another model's window, PRD §3.7 Critique 4, though no stage
does yet; its `extra_body`,
e.g. `id_slot`), the channel's
conversation tool schemas (`conv.tool_view.schemas`, per-channel since #149),
`build_prompt(pending_ids, mid_conversation_system_role)`, and the same
runtime-override filter (`_llm_overrides`). Any difference in what reaches
the template breaks the prefix and the cache with it. With a `cut`, calls before the cut whose first result falls at or after it render as pending, not abandoned, because they were pending as of the cut (#185). `reasoning_effort`
is left alone, because Qwen 3.8/Swift's template renders the effort
sentence first in the system block: a fork at a different effort would
miss the cache from its first token.
On top of that a fork sends `tool_choice: "none"` (tools stay in the
render), `reasoning_budget_tokens` from `agent.forks.reasoning_budget_tokens`
(a per-request sampler budget that never touches the template), and the
stage's `extra_body` (`response_format`; consolidation sends the memory-record schema). Those keys are applied after the
runtime overrides, so `set_settings` can't re-enable tool calls on a fork.

**Invariants.** A fork mutates nothing, persists nothing, fires no hooks,
and never dispatches tools: a reply carrying `tool_calls`, or a tool call
written out as text (`<tool_call>`, a `<function…>` tag, a bare
`{"name", "arguments"}` object), raises `ForkRefused`. No live window or
no main client raises `ForkUnavailable`.
Every caller treats any fork exception, or an empty or unparseable reply,
as a miss: it logs at WARNING and makes its standalone call instead
(except the provenance check, which skips). A
standalone JSON parse failure names the failing phase (consolidation,
appraisal, or critique) and includes up to 500 characters of the provider
text in the `ValueError`.

**Which process forks from where.** Plugins reach the Agent through
`fork_agent(pm, stage)`, which returns it only when that stage is enabled.
Stage names are opaque to core (it never names cognition plugins).

| Stage | Cut | Runs | Standalone when |
|---|---|---|---|
| `compaction` | dropped prefix | inline in `resolve_compaction` | forks off for the stage; fork miss |
| `consolidation` (compaction trigger) | dropped prefix | inline in `before_compaction`; the record is stashed and the `on_compaction` task consumes it | also: a range dialog row is not in `messages[:cut]`; the watermark moved since the fork |
| `consolidation` (idle trigger) | full window | in the idle task | also: no registered channel or live window; a range row not in the window |
| `appraisal` (stage 2) | full window at run time | silent TaskQueue task | also: the exchange's originating rowid has left the window (or was never persisted) |
| `critique` (each stylistic lens) | full window at run time | silent TaskQueue task | same as appraisal; a lens whose fork misses falls back alone |
| `critique` (provenance) | prefix ending at the exchange's final reply | silent TaskQueue task | never: a miss (forks off, no live window, reply not recorded or not in the window, a compaction since the reply) or a failed fork skips with a warning |

Seeding, imports, `consolidate_replay.py` and the `eval_*` scripts have no
live window and always run standalone. Subagents are unchanged: workers
with their own persona-neutral prompts.

The compaction and consolidation forks share one prefix: the second
fork's prefill is just its instruction. Containment for consolidation is
exact: every `message`-type row in the range must be a `_db_id` in the
fork's window (summary rows are skipped; a live window's summary has no
rowid and summaries are never consolidated). Appraisal and the stylistic lenses check
the originating rowid recorded by their own `on_message_persisted`
(critique per lens, right before each fork, since an earlier lens's fork
is an await during which compaction can drop the exchange), and read the
window at run time, so a fork after the user has said more still sees the
exchange. The instruction names it by quoting its opening message and the
response the user was sent, which can differ from the window's last
assistant message (the max-turns fallback). The provenance check instead
records the reply's rowid and the channel's compaction count in
`on_agent_response` (the count is bumped in `before_compaction`, before
the window changes), and checks both before its search and again right
before the fork.

**Instructions.** Each stage has a fork instruction beside its standalone
prompt (provenance has none), written to the agent in the second person.
The voice is a choice, not a measured result; #186's prompt-voice A/B will
say whether it matters:
`agent.compaction_fork_prompt`, `memory.consolidation_fork_prompt`,
`appraisal.stage2_fork_prompt`, `critique.fork_prompt.<lens>` (documented
copies in `prompts/*_fork.md`). They are read from static config with
`tuning.config_value`, never `resolve_tunable`, so `set_settings` can't
write them. Consolidation appends a sentence naming where the range
starts ("the whole conversation above", "what follows the summary at its
top", or a quote of the first new message when rows below the watermark
are still in view). Appraisal appends the exchange quotes and the
retrieval profile. `ForkResult.model` carries the model the fork was sent
with (`request_model` of the fork's body); consolidation stamps it on the
record (see [Re-consolidation](#re-consolidation-corvidae-memory)).
Critique appends the exchange quotes, and the provenance check also passes the CONTEXT snapshot inline, along with each
past claim and its closest earlier messages.

**Configuration** (`agent.forks`, validated at startup and on reload by
`validate_agent_config`):

```yaml
agent:
  forks:
    enabled: false                # opt-in until measured (#73)
    reasoning_budget_tokens: 2048 # per fork; null = server default
    stages: null                  # null = every stage; or a list of names
```

With forks on and `llm.background` set to a different model than
`llm.main`, `Agent.on_start` logs a warning: forked stages move to the
main model.

### Forks on the deployed backend

The PRD (§3.8) states what forks must achieve. What follows is how the
deployment meets it, and holds for llama.cpp's llama-server with one KV
slot and Swift-1.5 (a Qwen 3.8 fine-tune); another backend may answer
differently.

**Fork by default (#180, implemented by #181).** This is the target,
not the current code. Every background call that has a live window
forks it: compaction from the dropped prefix; compaction-time
consolidation from the same prefix in `before_compaction`, while the
window is whole; idle consolidation from the full window; stage-2
appraisal and each critique lens from the prefix ending at the exchange's
final reply, as provenance does; provenance
from the prefix ending at the exchange's final reply. Each forks only
while the rows it needs are still in the window; for an exchange-scoped
stage, a compaction between the reply and the fork is a miss, since the
prefix would no longer be what the agent saw. A fork that would not fit
the slot's context is a miss too. A miss skips the stage with a warning
(the rescue is a retry or a skip, never a standalone call), except
compaction, which takes its can't-fail fallback. Today only provenance
follows this: it is fork-only, and with forks off for critique (the
default config) it never runs. Every other stage still falls back to a
standalone call on a miss or failure (critique `_fork_target`,
`_fork_lens` and `_run_lens`; appraisal stage-2; `compaction.py`'s
`_fork_summary` fallback; memory consolidation), until #181 converts them
to skip-on-miss (compaction to its can't-fail fallback). Also #181's
target, not enforced today: per-stage reasoning and reply caps (today one
global `agent.forks.reasoning_budget_tokens` and no reply cap), the
headroom guarantee in the cost model below, and treating a fork that would
not fit the slot's context as a miss.

**Cost model.** A fork reuses the cached prefix and prefills only its
instruction and reply; the window is append-only, so a fork's prefix is
still cached as long as no compaction has run since. A standalone call
costs its own cold prefill of a rebuilt prompt. It doesn't evict the main
window when the server keeps `cache-ram`: on Lex (one slot,
`cache-ram = 32768`) #73's K1 measured the main turn after background
calls fully cached, restored from host RAM. Without `cache-ram`, a
standalone call on a single slot evicts the main window and the next main
turn re-prefills it. Either way only one call runs on the slot at a time,
so a background call delays a waiting main turn by its own slot time;
today only the global `agent.forks.reasoning_budget_tokens` bounds a
fork's share of that. #181's target adds the rest: exchange-scoped forks
scheduled before the slot's next main turn (PRD principle 3), per-stage
reasoning and reply caps, and headroom (the window at the compaction
threshold plus a fork's instruction and both caps fits the slot's
context, so a fork just short of compaction still runs). #182 measures
how long main turns wait today.

**Provenance is fork-only (#164).** The standalone provenance critic
judged a rebuilt context: a search of earlier messages plus a CONTEXT
snapshot. It missed support in the system prompt and the exchange's own
tool results, and objected to claims those supported (a `USER.md` fact, a
`grep` result from two turns earlier). The forked window up to the reply
is what the agent had, so provenance runs only as that fork and skips on
a miss.

**Rollout gate (#73).** Forking ships once #73's cache probes
pass: the fork row and the main turn after it reuse the cached prefix
(K1), the fork prompt begins with the main prompt (K2), and
`response_format` holds under `tool_choice: "none"` with tools present
(K3). On Lex at 913f7ab (appraisal and critique forking) all three
passed; K3 ran offline on a Lex `sessions.db` snapshot. K3 also found
the 2,048 reasoning budget binding on 5/10 appraisal and 9/10 provenance
forks; #186 tracks it. The flip of `enabled` is tracked in #73. #73's comparisons of fork
against standalone (planted-defect recall per lens, consolidation and
compaction fidelity) are benchmarks, not gates.

## Task System

`task.py` — general-purpose async work dispatch.

**Task** — an async callable with delivery context:

```python
@dataclass
class Task:
    work: Callable[[], Awaitable[str]]
    channel: Channel
    task_id: str           # auto-generated 12-char hex
    created_at: float
    tool_call_id: str | None = None
    description: str = ""
```

The queue doesn't know what `work` does — it calls `await task.work()`,
catches exceptions, and delivers the result via `on_notify`.

**TaskQueue** — FIFO async worker with configurable concurrency. Accepts
`max_workers` (default 1) to run up to that many tasks simultaneously.
Tracks active tasks and completed results (bounded deque, last 100).
`is_idle` property returns `True` when the queue has no pending items
and no tasks are currently executing.

**TaskPlugin** — hookimpl that owns the TaskQueue. Reads
`daemon.max_task_workers` from config (default 4) to set concurrency, and
`daemon.completed_task_buffer` (default 100) to set the completed-result
history bound. Starts/stops the worker on lifecycle hooks. Registers the
`task_status` tool. On task completion, fires `on_notify`.

## LLMPlugin

`llm_plugin.py` — owns `LLMClient` instance lifecycle. Registered as the
`"llm"` entry point. Declares `depends_on = set()`.

### Lifecycle

- `on_start` — reads `llm.main` (required) and `llm.background` (optional)
  from config. Creates `LLMClient` instances for each and calls `start()` on
  them.
- `on_stop` — calls `stop()` on all clients.

`_create_client` expands `${VAR}` in a role's `api_key` from the daemon's
environment (`corvidae/envref.py::expand_env_refs`, shared with MCP
`headers`). An unset, empty or multi-line variable raises a `ValueError`
naming `llm.<role>.api_key` and the variable, never the value; `on_start` and
`on_config_reload` both go through it. The resolved key lives only on the
client, so `_role_configs` keeps the raw `${VAR}` text.

### Client access

`LLMPlugin.get_client(role="main")` — returns the client for the given role.
`"background"` falls back to `main_client` if no background client is
configured. Other plugins retrieve the plugin via
`get_dependency(pm, "llm", LLMPlugin)` and call `get_client()`.

### Configuration

```yaml
llm:
  main:
    base_url: "http://host:8080"
    model: "model-name"
    api_key: "optional"
    extra_body: {}
    max_retries: 3
    retry_base_delay: 2.0
    retry_max_delay: 60.0
    model_not_loaded_wait: 60.0
    timeout: null             # optional; aiohttp default (300s) if absent
  background:                 # optional; absent means use llm.main
    base_url: "..."
    model: "..."
    # same keys as main
```

**Graceful degradation:** `llm.main` is required — `on_start` raises `KeyError`
if absent.

## ToolCollectionPlugin

`tool_collection.py` — collects tools from all plugins at startup. Registered as
the `"tools"` entry point. Declares `depends_on = set()`. Its `on_start` is
decorated with `@hookimpl(trylast=True)` so it fires after all other `on_start`
hooks have run, ensuring every tool-providing plugin has had a chance to finish
setup.

### Lifecycle

- `on_start` — reads `tools.max_result_chars` from config (default 100,000).
  Falls back to `agent.max_tool_result_chars` with a deprecation warning.
  Calls `pm.hook.register_tools(tool_registry=collected)` (sync broadcast) to
  collect tools from all plugins, then builds a `ToolRegistry`. Tools whose
  name is in `tools.disabled` are dropped from the registry here — the
  single chokepoint every registered tool passes through, so there are no
  per-plugin opt-outs. The list is validated in `on_init` (a non-list
  raises naming `tools.disabled` and the value) and widened to coupled
  tools, so `shell` and `task_pipeline` go together. Per-channel narrowing
  happens downstream in `Agent`; see "Per-channel tool sets".

### Tool access

`ToolCollectionPlugin.get_tools()` returns `(tools_dict, tool_schemas)` for the
agent loop. `ToolCollectionPlugin.get_registry()` returns the full `ToolRegistry`
for inspection or filtering (e.g., `exclude("subagent")`).

`ToolCollectionPlugin.max_result_chars` is a public attribute read by `Agent` and
`SubagentPlugin`.

### Configuration

```yaml
tools:
  max_result_chars: 100000   # truncation limit for tool result strings (default 100_000)
  disabled: [write_file]     # dropped for every channel; shell <-> task_pipeline coupled

# Legacy (deprecated):
agent:
  max_tool_result_chars: 100000   # use tools.max_result_chars instead
```

**Graceful degradation:** without `ToolCollectionPlugin`, `Agent.depends_on`
validation fails at startup.

## SchedulerPlugin

`scheduler.py` — the clock-driven producer: reminders and operator-declared
periodic jobs (#59; design rationale in
`plans/scheduler.md`). Entry point name: `"scheduler"`;
`depends_on = {"registry", "persistence", "clock"}`.

**Rows.** One `schedule` table on the persistence connection. A row is a
due time, an optional interval, a target channel id, the text, and an
origin (`reminder`; `heartbeat` is still accepted so old configs load, but
the heartbeat is now cognition's `HeartbeatPlugin`, started on system idle).
Reminders come from the `remind_me`
tool and are pinned to the calling channel. Jobs come from
`scheduler.jobs` in `agent.yaml` and are synced by `name`.

```sql
CREATE TABLE schedule (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,            -- config jobs only; NULL for tool-created rows
  channel_id TEXT NOT NULL,    -- "transport:scope", the persistence key
  origin TEXT NOT NULL,        -- 'reminder' ('heartbeat' accepted for old configs)
  text TEXT NOT NULL,          -- the objective, as the agent or operator wrote it
  next_fire_at REAL NOT NULL,  -- unix seconds, UTC
  interval_seconds REAL,       -- NULL means one-shot
  created_at REAL NOT NULL,
  created_by TEXT NOT NULL,    -- 'config' | 'tool'
  last_fired_at REAL,
  enabled INTEGER NOT NULL DEFAULT 1,
  disabled_reason TEXT,        -- 'fired' | 'cancelled' | 'missed' | 'removed_from_config'
  schedule_spec TEXT           -- config jobs: canonical every/at/timezone, to detect changes
);
CREATE INDEX schedule_due ON schedule (enabled, next_fire_at);
```

There is no firing log: a firing is a persisted conversation row, and its
`meta["schedule_id"]` travels with the trace.

**Timer.** One asyncio task sleeps until the earliest enabled row is due
(at most an hour, so a wall-clock jump is noticed) with `Clock.sleep`, and
reads time from `Clock.now`. An `asyncio.Event` wakes it early whenever a
row is added, cancelled, or re-synced. Before firing anything it waits
until the persistence DB is open and `Agent.started` is True. Tests
register a fake clock and never sleep.

**Firing.** The channel is resolved first: `registry.get`, then the
`resolve_channel` hook, then `registry.get_or_create` from the split id.
The last step is skipped for a transport that implements `resolve_channel`,
meaning a plugin whose `transport` attribute is the id's prefix. For that
transport, None means "not yet": the transport cannot apply the channel's
config, for example Signal before its first alias resolution (PRD §3.2.7,
#76). The row is then left untouched and due, its overdue note is kept, and
the pass re-arms after `CHANNEL_RETRY_SECONDS` (5 s). A WARNING is logged
once per row, not once per retry. Once the channel resolves, the row is
advanced (recurring) or disabled as `fired` (one-shot), at most once.
`channel.touch()` then runs, which only updates `last_active` and with it the order of the web and ACP channel lists. The plugin then calls:

```python
await pm.ahook.on_notify(
    channel=channel, source=row.origin, text=framed_text,
    tool_call_id=None, meta={"origin": row.origin, "schedule_id": row.id},
)
```

From there the normal pipeline runs: serial queue, persistence,
origin-keyed consumers, attribution, and `send`, whose reply on the
row's channel is the unprompted message. A job on a channel no
transport owns (such as `internal:notes`) reaches the log and persistence
only, because `TransportPlugin.send` delivers only to the transport that
owns a channel. The originating notification resets `turn_counter`
(step 3 of [Message processing](#message-processing)).

**Framing and role.** Reminder text is framed as data:

> A reminder you set on 2026-10-23 14:02 (PDT) for this conversation is
> due. Reminder text: «...». Decide whether and how to act on it.

`«` and `»` inside the text are replaced, so the text cannot close the
quote. Config jobs read `Scheduled job "<name>" (set by the operator,
every <interval>) is due. Job prompt: «...». ...`. A late catch-up firing
is prefixed `[overdue by 42m, daemon was down]`. The row is stored as
`role: system` with the `[reminder]` prefix like any
notification, and rendered under `agent.mid_conversation_system_role`
(`user` on templates that reject a non-leading system message). No role
parameter was added to `on_notify`.

**Recurrence.** A fixed interval with an optional anchor (`at`: `HH:MM`,
or an ISO datetime for a weekday anchor). The next slot is computed from
the scheduled time, not the fire time, so there is no drift; a row more
than one interval behind skips to the next future slot. Steps are taken in
local wall-clock time in `scheduler.timezone`, so `every: 1d` at 08:30
stays at 08:30 across DST.

**Startup.** `on_start` (or the timer's first pass, if the DB was not
open yet) syncs config jobs and then applies the catch-up policy. A
one-shot overdue by less than `catch_up_window` fires once with the stale
prefix; an older one is disabled as `missed`. A recurring row less than
one interval behind fires once with the prefix; one further behind skips
ahead without firing. Startup never fires directly; the timer does, once
it is ready.

**Job sync.** On `on_start` and `on_config_reload`, by `name`. A new name
is inserted. An existing name updates text, channel, and origin, and
recomputes `next_fire_at` only when `schedule_spec` changed or the row
was disabled. A name no longer configured is disabled as
`removed_from_config`, a durable stop. An invalid entry is logged at
ERROR and its existing row is left alone, so a typo does not stop a
running job. A reload wakes the timer.

**Tools.** `remind_me(when, text)`, `list_reminders()`, and
`cancel_reminder(id)` act on `_ctx.channel` only and take no channel
argument. The tool, not the model, parses `when` against
`scheduler.timezone`: a relative duration (`20m`, `1h30m`), ISO 8601 with
or without an offset, or bare `HH:MM` for the next occurrence; a bare
number is refused as ambiguous. It returns the resolved absolute time.
`scheduler.max_reminders_per_channel` caps pending tool rows per channel,
and text is capped at 2000 characters. `cancel_reminder` refuses config
jobs, and treats another channel's row as not found.

**Shutdown.** `on_stop` cancels the timer task; firings already enqueued
drain with the channel queues.

**Not in phase one:** a `send_to_channel` tool, recurring reminders by
tool, cron syntax, hook-kind jobs (see `plans/scheduler.md`).

## Subagent Tool

`tools/subagent.py` — `SubagentPlugin` registers the `subagent` tool, which
launches a background agent using the shared LLM client from `LLMPlugin` and
the full tool set minus `subagent` itself and minus whatever the calling
channel's `channels.<id>.tools.disabled` removes (#149).

### Tool signature

```python
async def subagent(instructions: str, description: str, _ctx: ToolContext) -> str
```

`_ctx` is system-injected (excluded from the LLM-visible schema). Returns a
confirmation string with the enqueued task ID on success, or an error string
if the task queue or channel context is unavailable.

### How it works

Declares `depends_on = {"llm", "tools"}`.

1. At tool-call time (not `on_start`), retrieves `ToolCollectionPlugin` via
   `get_dependency(self.pm, "tools", ToolCollectionPlugin)` and reads
   `max_result_chars` and the tool registry from it.
2. Calls `registry.exclude("subagent")` to remove itself from the subagent's tool set.
3. Retrieves the shared background `LLMClient` from `LLMPlugin` via
   `get_dependency(self.pm, "llm", LLMPlugin).get_client("background")`. The
   subagent does NOT start or stop this client — lifecycle is owned by `LLMPlugin`.
4. Builds a `messages` list (system prompt + user instructions) and a
   `work` coroutine that calls `run_agent_loop` with `channel` and `task_queue`
   from `_ctx` (enabling the subagent to enqueue nested tasks), then strips
   thinking from the result.
5. Wraps `work` in a `Task` with `tool_call_id` from `_ctx`, enqueues it on
   `ctx.task_queue`.
6. Returns immediately; result is delivered via `TaskPlugin → on_notify →
   Agent` (same path as all other tasks).

### LLM configuration

Uses the shared background `LLMClient` from `LLMPlugin` (`llm.background` if
configured, otherwise `llm.main`). The client is retrieved at tool-call time, not
captured at `on_start`.

### System prompt

The `SUBAGENT_SYSTEM_PROMPT` constant (defined in `tools/subagent.py`) is
used as the subagent's system message. It instructs the subagent to work
step-by-step and summarize results on completion.

## McpClientPlugin

`mcp_client.py` — connects to external MCP servers during `on_start`, caches
their tool lists, and exposes them to the agent loop via `register_tools`.

### Lifecycle

- `on_start` — reads the `mcp.servers` config block, iterates each server, and
  calls `_connect_server` for each. Each server's transport and session are
  owned by a long-lived task (see "Owner tasks") so they stay alive for the
  daemon's lifetime. After all connections, calls `_build_tool_list()` and caches the
  result. Connection failures per server are caught, logged as warnings, and
  skipped; a failed server does not abort startup.
- `register_tools` — extends `tool_registry` with the cached `Tool` instances
  (sync; called after `on_start` completes).
- `on_stop` — signals every server's owner task to close its session and
  transport, and awaits them all together (see "Owner tasks").

### Transports and failure handling

A server connects over stdio, SSE or streamable HTTP, chosen by its
`transport` key. Each server has its own exit stack and a bounded handshake
(see "Connect timeout"), so one server's failure never touches another.
Servers that did not connect at startup are queued in `_pending_servers` and
retried by a background `_retry_loop` task (started in `on_start` only when
something is pending) that sleeps on the kernel clock (`depends_on` "clock")
through the transports' ladder: 10 s, doubling, capped at 300 s, reset after a
round that reconnects something, exit when nothing is pending. Every MCP tool
call also runs a retry round as a fast path. One `asyncio.Lock` serialises
rounds, and a tool call that finds it held skips rather than wait out a
handshake, so the loop and a tool call never connect a server twice. A
successful retry rebuilds the cached tool list and the shared registry, and the
running Agent re-reads it on its next turn (#109). Config errors (any exception
from `_parse_server_config`, which runs before any transport opens:
`McpHeaderEnvError`, `ValueError` for a wrong type or non-finite timeout,
`KeyError` for a missing key) are reported once, naming
`mcp.servers.<name>.<key>` and the value, and never queued. A malformed
top-level block (`mcp` not a mapping, `mcp.servers` neither null nor a mapping)
is caught by `validate_mcp_config`, which `Runtime.start()` runs with the other
config checks (skipped when `mcp` is in `plugins.disabled`) and `on_init` runs
again, so it exits with a `Config error` instead of an AttributeError.
`on_stop` sets a stopping flag, cancels the loop, then
stops the owners; a connect still in flight (loop or tool call) that finishes
after the flag is set tears its owner down instead of registering it.

### Owner tasks

anyio requires a cancel scope to exit in the task that entered it, and
apluggy's `gather` runs `on_start` and `on_stop` in different tasks (a shared
`AsyncExitStack` closed from `on_stop` raised "Attempted to exit cancel scope
in a different task", #213). So `_connect_server` starts one `_ServerOwner`
task per server (`_run_owner`), which enters the transport, HTTP client and
`ClientSession` on its own `AsyncExitStack`, runs the handshake under
`asyncio.timeout`, and resolves a `ready` future with the server's state (or
the exception). It then waits on its `stop` event and closes the stack in the
same task. Owners live in `self._owners` (name to owner).

`on_stop` sets every owner's `stop` event at once and awaits all the tasks
together with `asyncio.wait`, bounded by `_STOP_BOUND_SECONDS` (2.5 s, inside
Runtime's 3 s stop budget). The SDK's `stdio_client` can take about 2 s to
terminate a process, so closing serially would blow the budget. An owner
still running at the bound is cancelled and logged at WARNING with the server
name and a "may need to be killed by hand" remedy. An exception while closing
is logged by the owner at WARNING with the server, the exception type (the
leaves of an exception group) and the traceback, and does not stop the other
servers. A `CancelledError` delivered to `on_start` stops and awaits the
owners already started before it propagates, and a cancelled or failed
mid-connect owner is awaited before `_connect_server` returns.

### Tool naming

Each tool is named `{tool_prefix}__{mcp_tool_name}`. `tool_prefix` defaults to
the server name if not set. Example: server `files`, tool `read` →
`files__read`.

### Tool name collision

`_build_tool_list` tracks seen prefixed names. If two servers expose a tool
with the same prefixed name, the second is skipped and a warning is logged
(first-wins).

### Schema sanitization

`_mcp_tool_to_schema` strips the keys `$schema`, `$id`, `$comment`, `$defs`,
and `definitions` from the top level of `mcp_tool.input_schema` before wrapping
it in the OpenAI function-call envelope. Sanitization is shallow (top-level
only); nested occurrences in `properties` are not removed.

### Configuration

```yaml
mcp:
  servers:
    <name>:
      transport: stdio | sse | streamable_http
      command: <executable>   # stdio only
      args: [...]             # stdio only
      env: {...}              # stdio only, optional
      url: <endpoint>         # sse and streamable_http
      headers: {...}          # sse and streamable_http, optional; ${VAR} from the environment
      tool_prefix: <prefix>   # optional; default: server name
      timeout_seconds: 30     # optional; per tool call; default: 30
      connect_timeout_seconds: 30  # optional; handshake bound; default: 30
```

### Connect timeout

The owner task opens each server's transport and session on its own
`AsyncExitStack`, inside `asyncio.timeout(connect_timeout_seconds)` (the
timeout wraps the handshake inside the owner, not the `ready` wait, so
expiry cancels the handshake itself and the subprocess dies with it). A
server that never answers `initialize` therefore cannot hold up the
`on_start` broadcast (#72). On success the owner keeps the stack open until
`on_stop`. On a timeout or any other failure the owner closes it
immediately, which terminates a stdio subprocess, and `_connect_server`
awaits that teardown before returning. On a timeout the owner logs it once
and `_connect_server` returns without registering the server; any other
failure propagates to `on_start`, which logs and skips it. The close runs
after the timeout has turned its cancellation into `TimeoutError`, so
teardown is not itself cancelled.

### HTTP transports and headers

`streamable_http` uses the SDK's `streamable_http_client` with an HTTP client
built by `create_mcp_http_client(headers=...)`. The SDK closes only a client
it created, so the plugin enters its client into the owner's stack, and it
closes with the server. `sse` passes the same headers to `sse_client`.
`_resolve_headers` replaces each `${VAR}` in a header value from
`os.environ` before anything opens. An unset or empty variable, or one holding CR, LF or NUL (which httpx would reject with an error quoting the value), raises
`McpHeaderEnvError` (a `ValueError`), whose message names the server, header
and variable but never a value. `on_start` logs it once and skips the server
without adding it to `_pending_servers`: the daemon's environment can't
change, so a retry would fail the same way.

The client targets the mcp 2.x API (`mcp>=2.0`): tool schemas are read from
`Tool.input_schema` and error results from `CallToolResult.is_error`.

**Graceful degradation:** without `McpClientPlugin`, no MCP tools are
registered. The plugin declares no `depends_on`.

## RuntimeSettingsPlugin

`tools/settings.py` — registers the `set_settings` tool, which allows the
agent to update per-channel LLM inference parameters and framework parameters
at runtime via tool call.

### Tool signature

```python
async def set_settings(settings: dict, _ctx: ToolContext) -> str
```

`_ctx` is system-injected (excluded from the LLM-visible schema).

### Parameters

The `settings` dict is intended for:

- LLM inference keys (`INFERENCE_KEYS`, `channel.py`): `temperature`,
  `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `max_tokens`,
  `seed`
- Framework keys (`TUNABLE_FRAMEWORK_KEYS`, `tools/settings.py`):
  `max_turns`, `max_context_tokens`, `reasoning_history`
- Dotted plugin tunables (`critique.*`, `appraisal.*`; see configuration.md)

Undotted keys are allowlisted (GitHub issue #206): any undotted key outside
the first two lists rejects the whole call, with no overrides applied, a
`null` included. The request body accepts keys that replace the
conversation (`messages`), repoint the server (`model`) or rewrite the
tool list (`tools`, `tool_choice`), and a blocklist can't name every key a
server honors; see [threat-model.md](threat-model.md), finding 1. As a
second line, `Agent._llm_overrides` forwards only `INFERENCE_KEYS` into
the request body. Values are not validated, except `reasoning_history`,
which is rejected unless it is `drop`, `exchange`, or `keep` — a stored bad
value would make every later config resolve raise.

Pass `null` for a key to clear that override and revert to the static config value.

### Return value

On success: a confirmation string listing current overrides (or noting that all
overrides have been cleared). On blocked keys: an error string naming the
blocked keys. On keys outside the allowlist: an error string naming them
and listing the accepted keys.

### Blocklist

`system_prompt` is always blocked regardless of operator config. Additional
keys are blocked via `agent.immutable_settings` in `agent.yaml`:

```yaml
agent:
  immutable_settings:   # keys the agent must not change (system_prompt always added)
    - max_turns
```

The blocklist is checked first, then the allowlist above, so it only
narrows what the allowlist accepts. Its main use is the dotted gate keys
(configuration.md, "Runtime-tunable gate parameters").

### Persistence

Changes apply to `channel.runtime_overrides` in memory only. They are lost on
daemon restart; no DB persistence.

**Graceful degradation:** without `RuntimeSettingsPlugin`, the `set_settings`
tool is not registered. The plugin declares no `depends_on`.

## Tools

Registered via `CoreToolsPlugin` in `tools/__init__.py`:

| Tool | File | Purpose |
|------|------|---------|
| `shell(command)` | `tools/shell.py` | Execute shell command, 30s timeout (configurable) |
| `read_file(path)` | `tools/files.py` | Read file (1MB limit, configurable) |
| `write_file(path, content)` | `tools/files.py` | Write file, creates parent dirs |
| `web_fetch(url)` | `tools/web.py` | Fetch URL, 15s timeout, 50KB truncation (configurable) |
| `web_search(query, max_results)` | `tools/web.py` | Search via DuckDuckGo, default 8 results (configurable) |
| `task_pipeline(definition)` | `tools/task_pipeline.py` | Execute YAML/JSON task DAG with dependency resolution |

Additionally, `SubagentPlugin` registers `subagent` and `TaskPlugin`
registers `task_status` as tool closures during `on_start`.

| Tool | File | Purpose |
|------|------|---------|
| `subagent(instructions, description)` | `tools/subagent.py` | Launch background subagent using shared LLM client |
| `task_status()` | `task.py` | Report task queue status |

`SchedulerPlugin` registers three tools scoped to the calling channel
(see [SchedulerPlugin](#schedulerplugin)):

| Tool | File | Purpose |
|------|------|---------|
| `remind_me(when, text)` | `scheduler.py` | Create a one-shot reminder on this channel; returns the resolved time |
| `list_reminders()` | `scheduler.py` | Pending rows on this channel, config jobs flagged read-only |
| `cancel_reminder(id)` | `scheduler.py` | Cancel a tool-created reminder on this channel |

## Transports

Every transport (CLI, IRC, Signal, Discord, web, ACP) subclasses
`TransportPlugin` (`hooks.py`) and sets `transport` to the name its
channel ids start with. The base class implements the outbound `send`
hook once: it drops events for channels the transport does not own
(`owns(channel)`, by default `channel.matches_transport(self.transport)`;
`WebPlugin` narrows it to also require that it is enabled), then
dispatches by event type to the plain methods `send_message`,
`send_thinking`, `send_tool_status` and `send_progress`. Those take the
channel and an event from `outbound.py` and default to no-ops, so a
transport overrides only what it can display; the per-transport sections
below name them as that transport's methods. The base class also declares
`depends_on = frozenset({"registry"})`. Inbound hooks
(`should_process_message`, `on_message_admitted`/`rejected`/`persisted`)
are not filtered by the base class; each transport's implementations open
with their own ownership check.

### CLI (`channels/cli.py`)

`CLIPlugin` — stdin/stdout transport. Routes to `cli:local` channel.
Reads stdin line-by-line in a background task. Prints responses to
stdout with latency timing.

EOF ends the session at an interactive terminal (Ctrl-D), and always
under `corvidae cli`, which is a session by definition (`corvidae cli <
script.txt` exits when the script runs out): SIGINT to the process for a
graceful shutdown. Under `corvidae serve`, EOF on a non-TTY stdin
(systemd's `/dev/null`, a closed pipe) stops just the CLI transport with a
warning and the daemon keeps running; with no stdin at all the read loop
never starts. So a service-managed config may keep a `cli:` channel. The
banner and `> ` prompts print only when stdin is a terminal.

`Runtime.run` installs its SIGINT handler before startup, so a SIGINT
that arrives while plugins are still starting (this EOF path, or a Ctrl-C
during startup) is held until startup finishes, then shuts down normally;
it logs that it is waiting, and a second interrupt force-exits. SIGTERM
keeps its default action until startup finishes, so `systemctl stop`
during a hung startup ends the process at once.

### IRC (`channels/irc.py`)

`IRCPlugin` — IRC transport via pydle. Connects with exponential
backoff retry (10s→300s cap); each retry WARNING names the host, port and
nick it failed to reach. Joins configured channels on connect.
Forwards both channel and private messages. `split_message` splits
outgoing text into 400-byte chunks preserving paragraph/sentence
boundaries.

Sender authorization (#77) is default-deny, as for Signal and Discord.
`irc.allow` is validated in `on_init` (not a list of strings raises, naming
the field) and the plugin's `should_process_message` refuses every sender
not on it, in channels and PMs, with one INFO line and no reply; a missing
`allow` logs a startup WARNING. Where pydle has account data (IRCv3
`account-notify`/`account-tag`/`extended-join` negotiated, or an account
already known for the sender), entries match the services account and an
unidentified or unknown sender is refused; with no account support they
match the nick. The allowlist is checked before the channel is resolved
(as in Signal, #103): a refused sender gets an unregistered `Channel` (or
an existing registry entry), so nothing is added to the registry and no
`message_log` rows or memory follow. With `account-tag` negotiated,
`IRCClient.on_raw` captures the PRIVMSG's own `account` tag (none means not
logged in) into a task-local `ContextVar` before any `await`; each `on_raw`
runs in its own task, so the value survives pydle's WHOIS suspensions.
`IRCPlugin._decide` reads only that value and never `users[nick]`, which is
shared across lines and which WHOIS replies rewrite for whoever holds the
nick (on a network without WHOX, an unknown nick makes pydle WHOIS it
mid-message). There is no nick fallback in that mode. `on_raw` also
overwrites the pydle record from the tag, only to keep pydle consistent. The
verdict is decided once per message, synchronously, in
`IRCPlugin._dispatch` (before any `await`) with a single account lookup, and
recorded by the message's transport-minted `trace_id`.
`should_process_message` only reports that recorded verdict (no verdict
means deny) and the entry is dropped when `on_message` returns, because
pydle runs a task per line and a later line's `on_raw` can rewrite
`users[nick]` while an earlier message is still in the hook pipeline.
Without `account-tag`, the account is whatever `users[nick]` holds when the
message arrives (pydle's tracked state; a residual risk). Entries and
senders are compared under the server's CASEMAPPING via the client's
`normalize`.

While the client is disconnected (including the reconnect backoff, up to
300s), outbound text is not dropped: it goes into a bounded in-memory queue
(`_OUTBOUND_QUEUE_MAX`, 50, a module constant, not configurable). Each queued
message logs a WARNING with the channel id, length and queue depth. When the
queue is full the oldest entry is dropped, with a WARNING naming its channel
and the number dropped so far in the outage. After a successful connect,
`on_connect` joins the configured channels and then flushes the queue in
order, splitting each message exactly as a connected send does. A send made
while the queue is non-empty queues behind it, so nothing overtakes older
output. Sends also queue from TCP connect until `on_connect` has finished its
joins: `IRCPlugin._ready` is set there and cleared in `on_disconnect`, so a
reply in that window is not sent before the bot is in the channel (#251). Limits: a connection that drops mid-flush leaves the unsent entries
queued, and a partly sent multi-line message is resent whole (duplicates
rather than losses); pydle's `join` does not await the server's JOIN, so a
channel that refuses the join drops the flushed text server-side; and the
queue is lost at shutdown (logged as a WARNING with the count).

`split_message` itself lives in `channels/split.py`, not in the IRC
plugin, so a transport can size its own chunks without importing IRC.

Its contract: `max_len` is in UTF-8 bytes and must be at least 6, or it
raises `ValueError` regardless of the text (so a bad limit is not
input-dependent; a 6-byte floor leaves room for one 4-byte codepoint
after the 2-byte paragraph separator is re-attached). For any
`max_len >= 6`, every chunk is at most `max_len` bytes and
`"".join(chunks) == text`. The word tier carries each word's leading space
with it and cuts a word that cannot fit on its own on codepoint
boundaries, so nothing is dropped and nothing overflows. Earlier versions
could lose characters or exceed the limit on long unbroken runs and
multi-byte text. That affected IRC and Signal as well, and the same fix
applies to both. `irc.message_chunk_size` and `signal.message_chunk_size`
below 6 now raise `ValueError` when a reply is first split, not at startup.

### Signal (`channels/signal.py`)

`SignalPlugin` — Signal DM transport, speaking newline-delimited JSON-RPC
2.0 to an already-running `signal-cli daemon` over a unix socket. corvidae
connects to that daemon; it never spawns or supervises it. Registering the
account and running the daemon are covered in [signal-ops.md](signal-ops.md).

With no `signal:` block in config the plugin loads, resolves its registry
dependency, and does nothing else — no socket, no channels, nothing logged
above DEBUG. A block that is present but missing `socket` or `account`, or
whose `allow` is not a list, raises in `on_init` and aborts startup.
Silence is for absence; errors are loud.

**Connection.** `on_start` launches two long-lived tasks: a connection loop
and a notification processor. The connection loop opens the socket and
reads frames until EOF, retrying on the same ladder IRC uses — 10s,
doubling, capped at 300s, reset to the initial delay after any successful
connect. A disconnect fails every outstanding request with
`ConnectionError` rather than leaving it hung, and a malformed frame is
logged and skipped rather than killing the connection.

Frames are routed by shape: one carrying a `method` is an unsolicited
notification and goes onto a queue; one carrying an `id` and no `method` is
a response and resolves the future its request is awaiting. Notifications
are never handled inline in the read loop. Handling one sends further
JSON-RPC requests (typing, receipts) and awaits their responses, and those
responses can only arrive by the read loop reading the next line —
inline handling would deadlock the transport against its own request. The
single-consumer processor also keeps rapid-succession messages in arrival
order.

**Identity.** A channel's scope is the sender's ACI — `signal:<aci>` —
never their phone number. `sourceNumber` is absent whenever a sender has
phone-number privacy on, and channel ids are the persistence key for
`sessions.db`, the jsonl logs, and memory; keying on a field that can
vanish would fork one sender's history into two.

Config may still name senders by E.164 number, both in `allow:` and as
`channels:` keys. Those are resolved forward to ACIs by a single batched
`getUserStatus` call — batched because it is a CDS lookup with a
server-side rate limit — issued after each successful connect, never at
`on_init`/`on_start`, so an unreachable daemon at boot leaves the plugin
inert instead of blocking startup. A resolved alias is cached for the
process lifetime; one that failed to resolve is retried after the next
connect. Reverse ACI→number resolution is not used anywhere: signal-cli's
contact store answers an unknown ACI with a phantom entry carrying a null
number, which is indistinguishable from "not found".

**`resolve_channel`.** For a `signal:<aci>` id the registry does not hold
(a reminder due after a restart, before the sender has written again),
`SignalPlugin.resolve_channel` goes through the same alias path as an
inbound message under phone-number privacy: ACI as scope, no
`sourceNumber`, so an E.164-keyed `channels:` override is applied through
forward resolution. It returns None for ids it does not own. It also
returns None for its own ids while the override cannot be applied yet: before
`on_start` has resolved the registry, and while any E.164 `channels:` key
that is also in `allow:` is still unresolved and no resolution attempt has
finished. Forward resolution starts only after a connect, so before the
first connect the alias's ACI is unknown. A call that lands while a
resolution is in flight waits for it, bounded at
`ALIAS_RESOLUTION_WAIT_SECONDS` (5 s). The caller retries (the scheduler
holds the row, see SchedulerPlugin) rather than caching a default-configured
channel (#76). Once an attempt has finished, an alias that failed to resolve
stops blocking. A batch failure is retried only after the next connect, so
deferring until then could hold a reminder for the life of the connection.
The channel is created without the alias's config, and a WARNING names the
channel and the unresolved aliases. An E.164 `channels:` key missing from
`allow:` is never resolved, so it never blocks.

**Inbound filtering.** An envelope is dropped unless it carries a
`dataMessage` with non-empty text from someone other than the bot's own
account, outside a group. Dropping non-`dataMessage` envelopes is what
keeps a `syncMessage` — the account's own traffic echoed from another
device signed into it — from being read as input and driving the agent in
a loop. Attachments are logged and discarded; any text accompanying them is
still processed. Reactions, quotes, edits, and remote deletes all fall out
of the empty-text rule, since none of them carry a `message` field. An
envelope whose send timestamp is more than five minutes old gets a
`[sent YYYY-MM-DD HH:MM UTC]` prefix prepended, so backlog delivered after
a reconnect reads as backlog rather than as current.

**Authorization.** `should_process_message` is a default-deny gate: the
sender's ACI must appear in `allow:` directly, or be what a configured
E.164 `allow` entry resolved to. A refusal is logged at INFO naming the
sender's ACI, and nothing is sent back. The transport fires `on_message`
for every decoded envelope regardless: the gate is the one place that
vetoes and the one place that logs the attempt. The transport checks its
allowlist before it resolves the channel, so a refused sender never
creates a registry entry (#103). Their message is gated on a transient
`Channel("signal", <aci>)` that is never registered, or on the live
entry if one already exists (a sender since dropped from `allow:`).
Otherwise anyone who knows the number could grow the registry by one
channel per distinct ACI.

**Liveness.** Two indicators, both gated on the allowlist — an
unauthorized sender gets no evidence that the account is live and
monitored.

- *Typing.* On the first outstanding message for a channel the transport
  sends `sendTyping` immediately, covering queueing delay, and starts a
  refresh loop that re-sends every 10s while the channel's outstanding
  count stays positive (Signal clients expire an indicator after roughly
  15s). `send_message` gives one outstanding slot back. So does the
  inbound handler for any message `on_message` returns without admitting:
  the transport mints the message's trace id, records its envelope
  timestamp under that id and takes the slot before calling `on_message`,
  and a `finally` releases both unless `on_message_admitted` fired for the
  id. Admission and rejection fire before `on_message` returns, so a gate
  veto, an agent that is not ready yet, or a raise or cancellation all land
  there. Only a message that took a slot (an allowlisted sender with an
  envelope timestamp) has an entry to release, so a stranger's refusal
  gives nothing back and no `on_message_rejected` implementation is needed.
  A refresh loop still asleep from an earlier
  turn is reused rather than doubled, so a channel has at most one, and
  `on_stop` cancels it. No explicit stop is ever sent — once the loop
  exits, the indicator expires on its own.
- *Read receipt.* Sent from `on_message_persisted` — processing start —
  not on arrival. A message can sit queued behind contended local
  inference, and a "read" marker followed by a long silence is worse than
  no marker. Envelope timestamps are held per channel keyed by trace
  id and popped by that id when the matching turn starts, so a vetoed or
  dropped message cannot shift a later receipt onto the wrong timestamp.
  `on_message_persisted` also fires for notification-originated turns,
  whose ids were never recorded, so those are a no-op. An admitted message
  whose turn dies before persistence (conversation init raised, the item
  was dropped) would otherwise keep its entry and its typing slot forever,
  and the slot would keep the indicator refreshing (#104). The cleanup
  relies on ordering. Entries are inserted in enqueue order, and the
  channel's serial queue processes items in that order. So when an id is
  persisted, every entry recorded before it belongs to a message whose
  turn has already ended without persisting. Those entries are dropped
  without a receipt, their slots are given back, and each is logged at
  WARNING. If every turn on a channel fails before persistence, entries
  accumulate until the fault clears, and the next message that persists
  then sweeps them all. That
  `sendReceipt`'s `targetTimestamp` takes the inbound envelope's
  `timestamp` is convention, not documented by signal-cli — it matches
  `sendReaction`, `remoteDelete`, and `quoteTimestamp`, but it has not been
  confirmed against the upstream spec.

**Outbound.** `send_message` and `send_progress` both split through
`split_message` at `message_chunk_size` bytes and deliver every chunk —
text is never truncated, and chunks go out verbatim, without the newline
stripping IRC needs. `send_progress` uses the same delivery path but skips
the outstanding-count bookkeeping: it is mid-turn liveness, not the reply.
Outbound is text only.

Signal delivers a reply to a turn of any origin on a channel it owns. A
reminder firing on a `signal:` channel injects through `on_notify`, and
the reply leaves through the same `send` path as a reply to a person; it
needs no inbound message to answer. The outstanding-reply count is guarded
at zero for such a reply (`SignalPlugin._decrement_outstanding` never goes
below 0), so a reply that answers no message neither errors nor drives the
count negative. The count can't tell which message a reply answers, though: a
reminder reply that lands while a person's message is outstanding gives back
that message's typing slot early (#101). This is
[PRD §3.2.10](prd.md): a transport delivers every emission on a channel it
owns, whatever originated the turn, and liveness and reply bookkeeping
tolerate a reply that answers no outstanding message. Sending to a channel
*other than* the turn's own is a separate capability (PRD §3.5.9: operator
allowlist that starts empty, plus a budget); see the scheduler's "Not in
phase one: a `send_to_channel` tool" note under
[SchedulerPlugin](#schedulerplugin) and the deferred entry in
`docs/threat-model.md`.

Group conversations are out of scope, as is any privacy handling specific
to disappearing messages.

### Discord (`channels/discord.py`)

`DiscordPlugin` — Discord transport, built on discord.py's gateway client.
It logs in as a bot account and talks in DMs, server channels, and threads.
Creating the application and inviting the bot are covered in
[discord-ops.md](discord-ops.md).

With no `discord:` block in config the plugin loads, resolves its registry
dependency, and does nothing else — no client, no tasks, nothing logged
above DEBUG. A block that is present but has no token (neither `token` nor
the `DISCORD_TOKEN` environment variable), an `allow` that is not a list,
or a `message_chunk_size` outside 100–2000 raises in `on_init` and aborts
startup. Silence is for absence; errors are loud.

**Connection.** discord.py owns the gateway: heartbeats, resumes, and
reconnects within a session are its job, not ours. `on_start` launches two
long-lived tasks, a connection loop and an inbound processor. The loop
builds a new client for every attempt and closes it in a `finally`, so a
failed login never leaks its HTTP session. A `LoginFailure` (bad token) is
logged once at ERROR and not retried, since retrying a rejected token does
not improve it. Any other failure, or the client stopping on its own, is
retried on the ladder IRC and Signal use: 10s, doubling, capped at 300s,
reset to 10s when the client reports ready. The plugin stays loaded after
a login failure, but is deaf and mute until the daemon restarts.

**No privileged intent.** The client uses discord.py's default intents and
does not request the privileged Message Content intent, so operators need
not enable it in the Developer Portal. Discord still delivers message
content for DMs and for messages that mention the bot, and those are the
only messages this transport processes. If a guild message mentions the
bot and arrives with empty content anyway, the transport logs a WARNING and
drops it. That should not happen, and the warning is how a change on
Discord's side would show up.

**Inbound filtering.** A message is dropped if its author is a bot (this
covers the bot's own messages and other bots', so two bots cannot talk each
other into a loop). In DMs every message from a human is processed. In
guild channels and threads the bot must itself appear in
`message.mentions`. `@everyone`, `@here`, and role pings do not count, even
though discord.py's `mentioned_in` would accept some of them. The bot's
mention token (`<@id>` or `<@!id>`) is stripped, the text is trimmed, and
a message with nothing left (a bare mention, whitespace, attachment only)
is dropped. Attachments are logged at DEBUG and discarded; any text with
them is still processed.

**Ordering.** discord.py runs every gateway event as its own task, so
handling a message inside `on_message` would let two quick messages
overtake each other. The client's `on_message` therefore only puts the
message on a queue (synchronously, which preserves arrival order), and a
single processor task consumes it. A failure handling one message is
logged and does not stop the processor.

**Identity.** A channel's scope is a numeric Discord id, never a name:
`discord:<channel_id>` for a server channel, and `discord:<thread_id>` for
a thread, which has its own id and therefore its own history, separate from
its parent channel. DMs are `discord:dm:<user_id>`, keyed on the user, not
the DM channel's id. Names (nicks, channel names, usernames) change, and
channel ids are the persistence key for `sessions.db`, the jsonl logs, and
memory, so keying on a name would fork a history on a rename. The sender
passed to `on_message` is the author's user id.

**`resolve_channel`.** For a `discord:<scope>` id the registry does not
hold (a reminder due after a restart), `DiscordPlugin.resolve_channel`
returns the registry's channel for it. It returns None for other
transports, for scopes that are not all ASCII digits (optionally prefixed
`dm:`), and before `on_start` has resolved the registry. Sending to such a
channel looks the destination up lazily (cache first, then a fetch), so it
works as long as the bot can still see that channel or user.

**Authorization.** `should_process_message` is a default-deny gate: the
sender's user id must appear in `allow:` (entries are compared as strings,
so YAML ints work). A refusal is logged at INFO naming the sender's id, and
nothing is sent back. The transport fires `on_message` for every message
that passes inbound filtering regardless: the gate is the one place that
vetoes and the one place that logs the attempt. Note what that means in a
server: anyone who can see the channel can @-mention the bot, and gets
silence. The bot has no way to tell them it is configured to ignore them.

**Liveness.** Typing only; Discord has no bot read receipts. On the first
outstanding message for a channel the transport sends the typing indicator
immediately and starts a refresh loop that re-sends every 8s (Discord's
indicator lasts about 10s) while the channel's outstanding count stays
positive. The transport mints the message's trace id, takes the
slot for an allowlisted sender, and passes the id to `on_message`; a
`finally` around that call gives the slot back unless `on_message_admitted`
fired for the id. Admission fires before `on_message` returns, so a gate
veto, an agent that is not ready yet, or a raise or cancellation all land
there. `send_message` gives back the slot of an admitted message. Only an
allowlisted sender ever took a slot, so a refused stranger @-mentioning the
bot gives nothing back and cannot stop typing owed to an allowed user in
the same channel; no `on_message_rejected` implementation is needed.
A refresh loop still asleep from an earlier turn is reused rather than
doubled, so a channel has at most one, and `on_stop` cancels it. Typing is
only ever sent for an allowlisted sender, so an unauthorized user gets no
sign the bot is live.
A failed typing call is logged and never blocks the reply.

**Outbound.** `send_message` and `send_progress` both split through
`split_discord_message` at `message_chunk_size` and send every non-blank
chunk; `send_progress` skips the outstanding-count bookkeeping, as on
Signal. Two properties apply to every send:

- *Mentions are disabled.* Each `send` passes
  `AllowedMentions.none()` explicitly, whatever the client's default is.
  The reply is built from model output, which can contain text from any
  untrusted source, and a literal `@everyone`, role mention, or user
  mention in it must render as text and ping no one.
- *Splitting is fence-aware.* `split_discord_message` builds on the shared
  `split_message` for oversized lines, but keeps ``` code fences intact:
  a fence left open at a chunk boundary is closed at the end of that chunk
  and reopened, with the same language tag, at the start of the next, so
  every chunk renders its code as code. Chunks are at most
  `message_chunk_size` UTF-8 bytes (the limit is in bytes, though
  Discord's 2000 is in characters, so multi-byte text splits a little
  early), and the chunks' bodies concatenate to the original text, so
  nothing is truncated. A chunk that is only whitespace is not sent,
  because Discord rejects empty messages.

Known limits of the splitter, none of which loses text:

- Only ``` fences are tracked. `~~~` fences and runs of four or more
  backticks are not recognized as fences, and a longer run is treated as
  ``` plus extra backticks.
- A ``` run that happens to be cut inside a single space-free word of
  about `message_chunk_size` bytes inverts the tracked fence state for the rest of the
  message. Chunks stay balanced, but later code may render as prose and
  vice versa.
- A language tag longer than 16 bytes, or one containing a backtick, is
  not carried across a boundary; the reopened fence has no tag.
- A cut can fall inside a ZWJ emoji sequence or between a base character
  and its combining marks, because splitting is by codepoint.

Outbound is text only. `send_thinking` and `send_tool_status` are not
overridden (the `TransportPlugin` no-ops apply), so reasoning and tool
traffic never reach Discord.

There is no local Discord server, so there is no harness end-to-end test
for this transport; behavior is pinned by `tests/test_discord_plugin.py`
against fake clients. The first real connection is the real test.

Replies to agent-started turns on a Discord channel (a reminder, say) are
delivered through the normal send path: `_deliver` resolves the destination
from the channel id, so no inbound message is needed. The outstanding count
is guarded at zero in `DiscordPlugin._decrement_outstanding`, as on Signal,
with the same early typing-slot release (#101).
Sending to a channel other than the turn's own is a separate capability,
not provided by the transport: PRD §3.5.9 (operator allowlist that starts
empty, plus a budget); see the scheduler's "Not in phase one: a
`send_to_channel` tool" note and `docs/threat-model.md`.

Out of scope: voice, slash commands, embeds, attachments in either
direction, and reactions.

### Web (`channels/web.py`)

`WebPlugin` serves a browser chat UI over `aiohttp.web`: a single page,
one WebSocket per tab, JSON frames both ways. Configuration and remote
access are in [configuration.md](configuration.md#web); what an attacker
can do with the listener is in [threat-model.md](threat-model.md).

With no `web:` block the plugin is inert: no listener, nothing logged above
DEBUG. A malformed block raises in `on_init`, naming the field. A bind
failure raises out of `on_start`.

**Identity.** One channel per named conversation: `web:<slug>`, where the
slug matches `[a-z0-9][a-z0-9_-]{0,63}`. Every message on a `web:*`
channel originates here, with sender `web`. Frames carry the bare slug
(`main`); internal state is keyed by channel id (`web:main`).

**Auth.** The token comes from `web.token`, then `CORVIDAE_WEB_TOKEN`, then
`secrets.token_urlsafe(32)`, which is logged once at WARNING as a login URL
(pre-formatted into the message, so it's never in `args` or an `extra`
field). Opening `/?token=…` returns a 200 page that refreshes to `/` and
sets `corvidae_web` (`HttpOnly; SameSite=Strict; Path=/; Max-Age` 400
days, plus `Secure` when the request's `Host` belongs to an `https://`
allowed origin). A 200 rather than a redirect, because a redirect after a
cross-site navigation wouldn't carry the Strict cookie. The cookie value is
`HMAC-SHA256(token, "corvidae-web-session")`, so it survives restarts with
the same token and dies when the token changes. Every route needs it.
Comparisons are constant-time and treat non-ASCII input as unequal.

Middlewares, outermost first:
1. security headers on every response, errors included: a strict CSP
   (`default-src 'self'`, scripts and styles from `'self'` only, `img-src
   'self' data:`, `connect-src 'self'` plus an explicit `ws://`/`wss://`
   source per allowed origin, IPv6 literals excepted since CSP can't
   express them), `nosniff`, `no-referrer`, `no-store`. An unexpected
   exception becomes a 500 logged without the URL (it may carry `?token=`);
2. a `Host` check against values derived from the allowed origins (a
   DNS-rebinding guard);
3. auth, as above.

The WebSocket also needs an `Origin` in the allowed set. Allowed origins
are the bind origin (for a loopback bind, all of `127.0.0.1`, `localhost`
and `[::1]` on the bound port) plus `web.allowed_origins`, all
canonicalised the same way the incoming headers are. The `AppRunner` runs
with `access_log=None`: aiohttp's access log would write `?token=` to disk.

**Protocol.** Client → server: `list`, `subscribe {conversation}`,
`send {conversation, text}`. Server → client: `conversations`,
`history {conversation, items, working}`, `user {text, id}`, `message`,
`thinking`, `progress`, `tool_status {tool, status, summary}`,
`working {value}`, `rejected {text, id}`, `divider` (replay only), and
`error {request, code, message}` with code `starting` (the agent hasn't
started; the client retries) or `bad_request`. Each socket handles its
frames one at a time, in order. Malformed input (invalid JSON, deep
nesting, a non-object, a bad slug, non-string text, lone surrogates, an
unknown type) gets `bad_request`, and the socket stays open; an exception
out of `on_message` (a third-party gate raising) is logged with its
traceback and answered the same way. Empty text is ignored.

**Unsent messages.** A user message is shown from the moment it's sent,
but its row reaches `message_log` only at dequeue. Between the two it's an
`_Unsent` entry with a server-minted display id. The entry also carries
the message's trace id, minted by the send handler when it creates
the entry and passed to `on_message`, plus an `admitted` flag. The admitted
and rejected hooks find their entry by that id, with no ambient state:
admission sets `admitted`, rejection removes the entry (and the `rejected`
frame carries its display id, so the right item is marked even when two
tabs send the same text), and `on_message_persisted` removes the entry.
A hook carrying an id no entry holds changes nothing (a rejection reports a
null id). Admission and rejection fire before `on_message` returns, so the
send handler's `finally` removes any entry still un-admitted: dropped,
raised or cancelled. Removal is idempotent.

**Working indicator.** On at admission. Off when the idle tracker reports
the channel settled (a `watch_channels(0, ...)` watch, filtered to web's
channels), unless a reply is still owed: an admitted-but-unpersisted entry
or pending tool calls; the next settle re-checks. Web never reads the
Agent's queues. A zero-period watch fires at settle, on the next loop step.
Without an idle tracker (hand-built plugin managers) web logs a warning and
clears on the final `send_message` instead.

**Replay.** `subscribe` sets the subscription, starts buffering live
frames and snapshots the unsent entries, all before its first `await`;
then it reads the newest 200 `message_log` rows (`ORDER BY timestamp
DESC, id DESC`, reversed) and queues `history` (rows mapped to frames,
then one `user` frame per snapshot entry), then the buffered frames. A
summary row becomes a `divider` at its timestamp; `context` rows,
`unsent` rows and `system` messages are skipped, and a not-sent marker for
progress or thinking drops that frame of the assistant row it follows
([The output gate](#the-output-gate)); assistant rows give `thinking` (from
`reasoning_content`), then `progress` or `message` (text through
`strip_thinking`), then a `dispatched` chip per tool call; tool rows give a
`completed` chip, named via the call id, with task.py's `[Task <id>] `
prefix removed. History favors duplicates over losses: a reply persisted
just before the read and broadcast while buffering can show twice until
the next reload. The 200-row window counts rows that render as nothing
(`context`), so visible history can be shorter. Without the persistence
plugin, history is empty (WARNING once, then DEBUG).

Replay and live differ in these ways:
- the LLM error text is sent live but never persisted, so it doesn't replay;
- a vetoed message has no row, so on reload it's gone entirely;
- an assistant message whose tool calls `max_turns` suppressed replays as
  progress plus `dispatched` chips for calls that never ran;
- with the `thinking` plugin disabled, live `message` text keeps `<think>`
  blocks and replay strips them;
- a `completed` chip is sent when the tool finishes, but its row is written
  when the result is dequeued, so a reload in between shows the call
  without its completion.

The conversation list merges `message_log`'s `web:*` channels with the
registry's live ones (the later `last_active` wins), drops names that fail
the slug rule, and sorts newest first.

**Fan-out and backpressure.** Every server frame goes through the client's
queue (1024) and a writer task. Enqueueing never awaits. A client whose
queue fills, or whose subscribe buffer passes 512 frames, is removed from
fan-out and closed with 1008 in a tracked task.

**Connection ownership and shutdown.** Each connection is owned by its
handler task. `_handlers` holds every live handler, and only the handler
removes itself; `_clients` is just the fan-out set, which the drop path
and token rotation shrink. The handler's `finally` releases the
connection: out of fan-out, writer cancelled, and the transport aborted if
it still holds unsent bytes. `on_stop`:
1. closes every fan-out socket (each close bounded at 1 s, then
   `abort()`) and awaits pending drop-path closes;
2. cancels every handler still in `_handlers` and waits up to 0.5 s. This
   matters: once a connection is lost, aiohttp drops its own reference to
   the handler task, so `runner.cleanup()` can't cancel a handler stuck in
   a gate. The cancellation unwinds through `on_message`, and the send
   handler's `finally` removes its entry;
3. runs `runner.cleanup()` (`shutdown_timeout=0.5`).

The worst case is about 2.5 s, inside Runtime's 3 s budget.
New WebSocket handshakes during shutdown get 503. Aborting rather than
closing matters with a peer that stopped reading (a phone asleep):
`transport.close()` waits to flush a full buffer forever, and aiohttp's
own close paths (a timed-out close, a write error, a heartbeat timeout)
only call `close()`.

**Config reload.** A token change swaps the session value and closes
every socket with 4001 ("token changed"); the page then shows "session
expired" and stops reconnecting. A `web.token` removed from config falls
back to `CORVIDAE_WEB_TOKEN` (as the daemon started with it), else a fresh
generated one (logged); a generated token is kept while none is provided.
Only `web.token` rotates on reload: the daemon's environment can't change
under it, so rotating `CORVIDAE_WEB_TOKEN` takes a restart. `host`, `port` and `allowed_origins` changes log "restart to
apply". A removed block logs the same and changes nothing. A malformed
block raises (the watcher logs it) and changes nothing.

**Frontend.** `web_static/` ships as package data: `index.html`,
`app.css`, `app.js` (plain JS, no build) and vendored marked 15.0.12 and
DOMPurify 3.2.6 (`vendor/README.md` records versions, hashes and
licences). Model text reaches the DOM through one sink:
`DOMPurify.sanitize(marked.parse(text))` with `style`, `class`, `id` and
`data-*` attributes and `style` tags forbidden (so a reply can't borrow the
app's classes or ids); links get `target="_blank" rel="noopener noreferrer"`.
Everything else uses `textContent`. Thinking and dividers render in closed
`<details>`. The page reconnects with backoff (1 s → 30 s) and, after
three failed connects, checks `/` for a 401 to detect an expired session.
`tests/test_web_render_smoke.py` checks the sanitising in headless Chrome;
it skips without Playwright, so the `web-smoke` CI job installs a pinned
Playwright and Chromium and fails if the test is skipped. Run it locally with
`uv run --with playwright pytest tests/test_web_render_smoke.py`.

Known limitations:
- a turn that ends in an exception without a `send_message` leaves the
  indicator on until the channel's next `send_message`;
- an exception in the agent after dequeue but before step 4c
  (`on_message_persisted`) leaves that message's entry until restart: the
  indicator stays on and the message keeps appearing at the end of
  history. That path is logged at ERROR, and such a channel gets no
  replies at all;
- a gate that's slow to decide holds up the socket's later frames, since
  frames are handled in order;
- messages in a gate, admitted or queued at shutdown are lost, as with
  every transport, and in the pane they simply disappear on reload;
- a message sent into a socket that looks open but is dead (a phone waking
  from sleep on a half-open connection) is lost: the composer clears and
  the server never sees it, so it's missing after the reconnect. Fixing
  that needs a per-send acknowledgement in the protocol.

Out of scope for v1: paging back past 200 rows, token streaming, an
inspector pane, and views of other transports' channels.

Behavior is pinned by `tests/test_web_plugin.py`, which drives a real
listener with aiohttp's client and raw sockets.
### ACP (`channels/acp.py`)

`AcpPlugin` + `corvidae acp` — Agent Client Protocol **v1** over stdio for
editors (bb `customAcpAgents`, Zed, …). Requires the optional
`agent-client-protocol` extra (`--extra acp` on the `uv sync` line).

**Process model.** ACP is a dedicated subprocess (`corvidae acp`), not
attached to long-lived `corvidae serve`. The command sets
`config["_acp_mode"]`; without that flag the plugin is inert (no stdin/
stdout ownership under `serve` / `cli`). Stdout is ACP JSON-RPC only;
logs go to stderr or a log file.

**Channels.** Sessions are `acp:<sessionId>`. `session/new` stores the
client `cwd` on `channel.runtime_overrides["cwd"]` (transport metadata,
not LLM config). Tool calls on ACP channels go through a swappable
`ToolBackend` (`LocalToolBackend` by default; see
`corvidae/tools/backends.py`).

**Session durability (Phase 2).** `initialize` advertises
`loadSession: true` plus `sessionCapabilities` for `list`, `resume`, and
`close` (`delete` is not advertised). `session/load` and `session/resume`
reattach the **same** `acp:<sessionId>` from `sessions.db` (no forked
history). `session/load` replays user/assistant text (and compaction
summaries) as `session/update` chunks before responding, skipping `UNSENT`
rows and assistant text whose progress send a gate vetoed;
`session/resume` does not. `session/close` cancels in-flight work and
drops the in-memory channel while leaving `message_log` intact.
`session/list` reads an `acp_session_index` sidecar table in the same DB
(cwd + `updated_at`).

**Hooks.** `AcpPlugin` is a `TransportPlugin` with `transport = "acp"`,
so outbound events reach it only for `acp:` channels. Tool status is
keyed on `(channel.id, tool_name)`; the event's `tool_call_id` is not used
yet. `session/prompt` mints the message's
trace id and passes it to `on_message`, and completes with `end_turn`
after tool drain, including when the final `session/update` cannot be
delivered (logged at WARNING; #99); `session/cancel` yields `cancelled`.
A second `session/prompt` on a session with one in flight is rejected
with `invalidRequest` and the first is left untouched (#93). The stdio loop
starts reading only once `Agent.started` is true: Runtime starts Agent
after the `on_start` broadcast, and a prompt read before then would be
dropped with its response never sent.

Design of record: `plans/acp-transport-design.md`. Gate B harness:
`tests/test_acp_conformance.py`. Subprocess stdio smoke (mocked LLM, no
live API): `tests/test_acp_stdio_smoke.py`.

## Configuration

```yaml
daemon:
  session_db: sessions.db

llm:
  main:
    base_url: "http://host:8080"
    model: "model-name"
    api_key: "optional"
    extra_body: {}              # optional, passed through to LLM
    max_retries: 3              # retry attempts on transient errors (default 3)
    retry_base_delay: 2.0       # base exponential backoff delay in seconds
    retry_max_delay: 60.0       # maximum retry delay cap in seconds
    model_not_loaded_wait: 60.0 # seconds to wait out an unloaded router model
    timeout: null               # HTTP timeout per request; null = aiohttp default (300s)
  background:                   # optional separate LLM for subagent tasks
    base_url: "..."
    model: "..."
    # same keys as main

agent:
  system_prompt: "..."          # string or list of file paths
  max_context_tokens: 24000
  reasoning_history: drop       # drop | exchange | keep
  max_turns: 10
  mid_conversation_system_role: system  # system | user — role for non-leading system messages
  immutable_settings:           # keys the agent must not change (system_prompt always added)
    - max_turns

tools:
  max_result_chars: 100000      # truncation limit for tool result strings (default 100_000)
  shell_timeout: 30             # shell tool timeout in seconds (default 30)
  web_fetch_timeout: 15         # web_fetch timeout in seconds (default 15)
  web_max_response_bytes: 50000 # web_fetch response truncation (default 50_000)
  max_file_read_bytes: 1048576  # read_file size limit in bytes (default 1MB)
  web_search_max_results: 8     # web_search result count (default 8)

memory:
  idle_consolidate_after: 1800  # seconds a channel must be quiet before idle consolidation
  half_life_days: 30            # retrieval recency decay (e-folding time, days)
  retrieval:
    k: 8                        # KNN candidate count
    band_on: score              # score | similarity — what the bands apply to
    recency_from: created       # created | last_used — where ranking age runs from
    bands:
      strong: 0.75              # assert confidently at or above this value (score or similarity, per band_on)
      moderate: 0.60            # hedge at or above this value
  channel_groups: {}            # {group: [channel ids or * ? globs]} shared memory scopes

scheduler:
  timezone: UTC                 # IANA name; resolves offset-less times (default UTC)
  catch_up_window: 3600         # seconds a missed one-shot may be late and still fire
  max_reminders_per_channel: 50 # pending tool-created reminders per channel
  jobs:                         # operator-declared recurring jobs, synced by name
    - name: morning-checkin
      channel: irc:#lex
      every: 1d                 # required
      at: "08:00"               # optional anchor: HH:MM or ISO datetime
      prompt: "..."             # or prompt_file: a path relative to agent.yaml

heartbeat:
  after: 3600                   # seconds of system quiet before a beat; absent = asleep

funnel:
  default_budget: 512           # tokens per admission-funnel call
  budgets:
    memory: 512                 # optional per-source overrides

channels:
  "irc:#channel":
    system_prompt:              # per-channel override
      - prompts/SOUL.md
      - prompts/IRC.md
    max_context_tokens: 8000
    max_turns: 5

irc:
  host: irc.lan
  port: 6667
  nick: agent
  allow: [alice]                # default-deny; nick, or services account where the network has them
  channels:
    - "#channel"

discord:                        # optional; omit to leave the transport inert
  token: "..."                  # or set DISCORD_TOKEN in the environment
  allow:                        # Discord user ids; default-deny
    - 123456789012345678
  message_chunk_size: 2000      # 100-2000

web:                            # optional; omit to leave the transport inert
  host: 127.0.0.1
  port: 8765
  token: null                   # or CORVIDAE_WEB_TOKEN; else generated and logged
  allowed_origins: null         # extra origins, e.g. https://box.tailnet.ts.net

logging:                        # optional, falls back to defaults
  level: INFO                   # DEBUG | INFO | WARNING | ERROR | CRITICAL
  file: corvidae.log            # omit for stderr; CLI mode defaults to corvidae.log
```

When `system_prompt` is a list, each path is read and concatenated.
Relative paths resolve against the directory containing `agent.yaml`.

## Plugin Registration Order

`Runtime.start()` (`runtime.py`) registers plugins in two stages, not a
fixed hand-written order:

1. Three kernel objects are registered by hand: `ChannelRegistry` as
   `"registry"`, the `Clock` as `"clock"` and the `IdleTracker` as `"idle"`
   (`self.pm.register(..., name=...)` in `Runtime.start()`).
2. Every other plugin loads from the `corvidae` setuptools entry-point
   group: `self.pm.load_setuptools_entrypoints("corvidae")`
   (`runtime.py`), reading `[project.entry-points.corvidae]` in
   `pyproject.toml`. Names listed in `plugins.disabled` are blocked
   via `pm.set_blocked(name)` before this call, so disabled plugins are
   never instantiated.

The kernel's entry points (`[project.entry-points.corvidae]` in the root
`pyproject.toml`): `persistence`, `jsonl_log`, `core_tools`, `cli`, `irc`,
`signal`, `discord`, `web`, `acp`, `task`, `subagent`, `mcp`, `llm`,
`compaction`, `thinking`, `runtime_settings`, `tools`, `metrics`,
`usage_log`, `metrics_jsonl`, `scheduler`, `agent`,
`config_watcher`. The cognition extra adds `memory`, `memory_tools`,
`funnel`, `outcome_log`, `appraisal`, `critique` and `goals` from
`packages/corvidae-cognition/pyproject.toml`.

`Runtime.start()` loads them with pluggy's `load_setuptools_entrypoints`,
which does not catch errors: an entry point that fails to import stops
startup. That is why a transport with an optional SDK (ACP, Discord)
imports it lazily rather than at module load.

Entry-point loading does not guarantee a meaningful cross-plugin order;
plugins that need another plugin's state resolve it lazily via
`get_dependency`/`on_start` ordering (`trylast=True`), not registration
sequence.

Immediately after entry points load, `Runtime.start()` logs `"plugins
loaded"` with `loaded` (the sorted names `pm.list_name_plugin()` returns
with a plugin object; pluggy keeps each blocked name as `(name, None)`, and
those are left out) and `blocked` (the configured `plugins.disabled` list, verbatim
— not narrowed to names that matched a real plugin). A misspelled disable
entry shows up under `blocked` while the plugin it was meant to block
still shows up under `loaded`; that mismatch is the operator-visible
signal of the typo. It is not an error and does not stop the daemon.

Next, `install_plugin_hookspecs(pm)` installs the hookspecs plugins
declare ([Plugin-declared hookspecs](#plugin-declared-hookspecs)), and
the hook arg-binding guard checks every impl against the full set of
specs. Then `validate_dependencies(pm)` runs
to verify every plugin's `depends_on` set names a registered plugin.
Startup aborts with `RuntimeError` if any dependency is missing.
`report_orphan_hookimpls(pm)` then logs any hookimpl left with no spec,
before the `on_init` broadcast.

## Plugin Dependencies

Plugins declare dependencies using a class-level `depends_on` attribute (a
set of plugin name strings). `validate_dependencies(pm)` in `hooks.py`
iterates all registered plugins and raises `RuntimeError` if any declared
dependency is not found in the PM. It runs in `Runtime.start()`
(`runtime.py`) after all plugins are registered, before `on_start`.

**`get_dependency(pm, name, expected_type)`** — typed lookup via
`pm.get_plugin(name)`. Raises `RuntimeError` if the plugin is not
registered; raises `TypeError` if the plugin is not an instance of
`expected_type`. Returns the plugin cast to `expected_type`. Defined in
`hooks.py`.

**`validate_dependencies(pm)`** — iterates every registered plugin, checks
for a `depends_on` attribute, and raises `RuntimeError` naming the missing
dependency and the plugin that declared it. Runs once at startup.

**`optional_depends_on`** — a class-level set of plugin names the plugin
uses when they are loaded. `validate_dependencies` ignores it, so a
missing name is fine and it takes no part in cycle detection. Its only
reader is `report_orphan_hookimpls`, which logs at DEBUG rather than
WARNING for an impl of a hook owned by an absent optional dependency.

### Current dependency graph

```
Agent               → "registry", "task", "llm", "tools"
CLIPlugin           → "registry"    (ChannelRegistry)
IRCPlugin           → "registry"    (ChannelRegistry)
WebPlugin           → "registry"    (ChannelRegistry); looks up "agent" and "persistence" lazily
AcpPlugin           → "registry"    (ChannelRegistry)
SubagentPlugin      → "llm", "tools"
McpClientPlugin     → "clock"       (background reconnect loop)
CompactionPlugin    → "llm"         (LLMPlugin); soft-uses "agent" for forks
ThinkingPlugin      → "registry"    (ChannelRegistry)
MemoryPlugin        → "persistence", "llm", "clock", "idle"; soft-uses "agent", "registry" for forks  (cognition extra)
MemoryToolsPlugin   → "memory"      (MemoryPlugin)                                (cognition extra)
CritiquePlugin      → "task", "llm"; soft-uses "agent" for forks                  (cognition extra)
OutcomeLogPlugin    → "persistence"                                               (cognition extra)
AppraisalPlugin     → "persistence"; soft-uses "agent" for forks                  (cognition extra)
HeartbeatPlugin     → "idle", "registry"; needs "memory" only when awake          (cognition extra)
UsageLogPlugin      → "persistence"
SchedulerPlugin     → "registry", "persistence", "clock"
```

Plugins with `depends_on = set()` (declared but empty):
`LLMPlugin`, `ToolCollectionPlugin`, `PersistencePlugin`,
`CoreToolsPlugin`, `McpClientPlugin`, `RuntimeSettingsPlugin`, `JsonlLogPlugin`,
`TaskPlugin`, `MetricsPlugin`, `MetricsJsonlPlugin`, `ConfigWatcherPlugin`,
`FunnelPlugin` (cognition extra).

"Soft-uses" means `fork_agent(pm, stage)` (`pm.get_plugin("agent")`,
duck-typed): with the agent plugin absent or forks off, the stage runs
standalone.

Transport plugins use `get_dependency(pm, "registry", ChannelRegistry)` in
`on_start` to retrieve the shared `ChannelRegistry`. `SubagentPlugin` calls
`get_dependency(pm, "tools", ToolCollectionPlugin)` and
`get_dependency(pm, "llm", LLMPlugin)` at tool-call time inside `_launch`.

## Directory Layout

```
corvidae/
├── hooks.py              # AgentSpec, hookimpl, create_plugin_manager
├── tool.py               # Tool, ToolRegistry, tool_to_schema, ToolContext
├── tool_collection.py    # ToolCollectionPlugin (collects tools at startup)
├── channel.py            # Channel, ChannelConfig, ChannelRegistry, resolve_system_prompt
├── queue.py              # SerialQueue (is_empty property)
├── llm.py                # LLMClient
├── llm_plugin.py         # LLMPlugin (LLM client lifecycle)
├── turn.py               # run_agent_turn(), AgentTurnResult
├── context.py            # ContextWindow, MessageType, DEFAULT_CHARS_PER_TOKEN
├── jsonl_log.py          # JsonlLogPlugin (persist_message, on_compaction)
├── logging.py            # StructuredFormatter, configure_logging()
├── agent.py              # Agent (single-turn dispatch)
├── persistence.py        # PersistencePlugin (DB lifecycle, conversation init)
├── clock.py              # Clock, Timer (kernel clock: now, sleep, call_later, call_at)
├── idle.py               # IdleTracker, IdleWatch (per-channel and system quiet)
├── thinking.py           # ThinkingPlugin, strip_thinking(), strip_reasoning_content()
├── compaction.py         # CompactionPlugin
├── fork.py               # fork prompt builder, cut snap, agent.forks parser (#51)
├── task.py               # Task, TaskQueue, TaskPlugin
├── mcp_client.py         # McpClientPlugin (MCP server bridge)
├── main.py               # daemon entry point
├── channels/
│   ├── cli.py            # CLIPlugin
│   ├── irc.py            # IRCPlugin
│   ├── signal.py         # SignalPlugin (signal-cli JSON-RPC transport)
│   ├── discord.py        # DiscordPlugin (discord.py gateway transport)
│   ├── web.py            # WebPlugin (browser chat UI over aiohttp)
│   ├── web_static/       # its page, script, styles, vendored marked + DOMPurify
│   ├── acp.py            # AcpPlugin + corvidae acp command
│   └── split.py          # split_message() — shared outbound chunking
├── commands/
│   └── redact.py         # corvidae redact CLI (operator-only)
└── tools/
    ├── __init__.py       # CoreToolsPlugin
    ├── shell.py
    ├── files.py
    ├── web.py            # web_fetch, web_search
    ├── subagent.py       # SubagentPlugin, subagent tool, run_agent_loop()
    ├── settings.py       # RuntimeSettingsPlugin, set_settings tool
    └── task_pipeline.py  # TaskPipelinePlugin, task_pipeline tool

packages/corvidae-cognition/   # the cognition extra (docs/cognition.md)
├── pyproject.toml        # distribution corvidae-cognition; its entry points
└── corvidae_cognition/
    ├── memory.py         # MemoryPlugin (consolidation + retrieval + message_fts schema)
    ├── retention.py      # run_retention_job(), retention_score()
    ├── memory_tools.py   # MemoryToolsPlugin, search_memory, recall_raw
    ├── memory_command.py # corvidae memory: stamps, reconsolidate (operator CLI)
    ├── funnel.py         # FunnelPlugin (context-admission funnel)
    ├── appraisal.py      # AppraisalPlugin
    ├── critique.py       # CritiquePlugin
    ├── goals.py          # GoalsPlugin (goals table, add/update/retire_goal)
    └── outcome_log.py    # OutcomeLogPlugin (exchange_log)
```

## Known Risks

**Tool result ordering.** When the LLM requests multiple tool calls,
results arrive independently as task completions. Tool results are
batched via `channel.pending_tool_call_ids`: each result is appended
to the conversation as it arrives, but the LLM call is deferred until
the last pending result clears the set. This means the LLM sees all
results from a batch in a single turn. Partial failure (one tool errors)
still delivers all results; the failed tool returns an error string.
User messages can interleave mid-batch without disrupting the pending set.

**User message interleaving.** If a user sends a new message while tool
results are still pending, the conversation may interleave user
messages with tool results. The SerialQueue serializes processing (no
races), but ordering depends on arrival time.

**Tool sandboxing.** The `shell` tool runs commands without any
sandbox or privilege restriction, and `task_pipeline` (which calls the
same shell implementation), `read_file`, `write_file`, and `web_fetch`
are likewise unconfined. This was a scoping decision for the MVP:
Corvidae is a personal agent daemon on the user's own machine, and
sandboxing was not a priority for getting the loop working. It still
matters. The agent reads text it did not write (IRC, web pages, MCP
results), and an injected instruction runs with the daemon's full
privilege — the attacker is whoever can post in a channel the bot sits
in, not another tenant. [threat-model.md](threat-model.md) lays out the
exposure and an ordered fix list; `tools.disabled` and
`plugins.disabled` are the current per-deployment valves.
`harness/buster.yaml.in` is a partially narrowed profile (no subagent,
no MCP, no `shell`) but still carries `task_pipeline` and `write_file`.

## Unimplemented

The following items appear in earlier design documents but are not yet
implemented. Each needs discussion before proceeding.

### Hot-loading

The original design included a `ComponentLoader` with watchdog
filesystem watcher for hot-reloading plugin modules from a
`components/` directory. Not implemented. Transport plugins load via
setuptools entry points, like every other built-in plugin.

### Double-buffer compaction

An optimization where a new compacted prompt is built from the full log
in a second llama-server KV cache slot, warmed, and swapped in
seamlessly. Not implemented; current compaction is stop-the-world.

### Additional transports

BlueSky and other transports are future work. The hook system supports
them; the implemented transports are CLI, IRC, Signal, Discord, and Web.
