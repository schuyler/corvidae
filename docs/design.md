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

3. **Transport plugins** — IRC, CLI. Each transport converts
   platform-specific messages to/from a common Channel abstraction.

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
list of every hook (including lifecycle hooks like `on_init` and the
outbound-display hooks `send_thinking`/`send_tool_status`/`send_progress`
not shown here).

```python
class AgentSpec:
    async def on_start(self, config: dict) -> None
    async def on_stop(self) -> None
    async def on_message(self, channel: Channel, sender: str, text: str) -> None
    async def send_message(self, channel: Channel, text: str, latency_ms: float | None = None) -> None
    def register_tools(self, tool_registry: list) -> None  # sync; plugins append Tool instances or bare callables
    async def on_agent_response(self, channel: Channel, request_text: str, response_text: str, correlation_id: str | None, meta: dict, logprobs: dict | None) -> None
    async def on_notify(self, channel: Channel, source: str, text: str, tool_call_id: str | None, meta: dict | None) -> None
    async def should_process_message(self, channel: Channel, sender: str, text: str, correlation_id: str | None) -> bool | None
    async def on_message_admitted(self, channel: Channel, correlation_id: str, sender: str, text: str) -> None
    async def on_message_rejected(self, channel: Channel, correlation_id: str, sender: str, text: str) -> None
    async def on_message_persisted(self, channel: Channel, correlation_id: str, rowid: int | None, text: str, meta: dict) -> None
    async def on_llm_error(self, channel: Channel, error: Exception) -> str | None
    async def compact_conversation(self, channel: Channel, conversation: ContextWindow, max_tokens: int) -> None
    async def process_tool_result(self, tool_name: str, result: str, channel: Channel | None) -> str | None
    async def before_agent_turn(self, channel: Channel, correlation_id: str | None, meta: dict) -> None
    async def after_persist_assistant(self, channel: Channel, message: dict) -> None
    async def transform_display_text(self, channel: Channel, text: str, result_message: dict) -> str | None
    async def on_idle(self) -> None
    async def load_conversation(self, channel: Channel) -> list[dict] | None
    async def resolve_channel(self, channel_id: str) -> Channel | None
    async def on_conversation_event(self, channel: Channel, message: dict, message_type: MessageType) -> int | None
    async def before_compaction(self, channel: Channel, compacted_ids: list[int], cut: int) -> None
    async def on_compaction(self, channel: Channel, summary_msg: dict, retain_count: int, compacted_ids: list[int]) -> None
```

`create_plugin_manager()` in `hooks.py` creates the manager and adds
hookspecs.

### Broadcast dispatch and result resolution

All hooks are called via `pm.ahook.<hook_name>(...)`, which broadcasts to
every registered implementation and returns a list of results. Gate hooks
that need a single resolved value pass the result list to
`resolve_reject_wins(results)` from `hooks.py`: any `False` in the results
returns `False`; otherwise any `True` returns `True`; otherwise `None`.
`should_process_message` is the only hook resolved this way.

Hooks marked `firstresult=True` in the hookspec (`on_llm_error`,
`compact_conversation`, `process_tool_result`, `transform_display_text`,
`load_conversation`, `resolve_channel`) use pluggy's own first-non-None-wins dispatch — the
first handler to return a non-None value stops the chain; no separate
resolver function is involved. `process_tool_result` and
`transform_display_text` are wrapper chains: `@hookimpl(wrapper=True)`
implementations compose transforms over an identity seed.

Pure broadcast hooks (e.g. `on_start`, `on_idle`) resolve neither way;
their return values are ignored.

### Hook reference

| Hook | Semantics | Call site |
|------|-----------|-----------|
| `on_init` | broadcast | `Runtime.start()`, after all plugins are registered, before `on_start`; store `pm`/`config`, resolve other plugins — no runtime resources |
| `on_start` | broadcast | daemon startup, after config load |
| `on_stop` | broadcast | SIGINT/SIGTERM |
| `on_message` | broadcast | inbound message from any transport |
| `send_message` | broadcast | outbound message delivery |
| `send_thinking` | broadcast | `Agent`, when the LLM response carries `reasoning_content` |
| `send_tool_status` | broadcast | `Agent`/`TaskPlugin`, tool call dispatched and completed |
| `send_progress` | broadcast | `Agent`, intermediate LLM text alongside tool calls |
| `register_tools` | broadcast (sync) | startup tool collection |
| `on_agent_response` | broadcast | after agent loop produces a text response; no default implementation |
| `on_notify` | broadcast | inject a notification into a channel queue |
| `should_process_message` | broadcast / resolve_reject_wins | `on_message`, before enqueue; no default implementation; None from empty broadcast allows all messages |
| `on_message_admitted` | broadcast | `on_message`, after a USER message is admitted; fires exactly once per admitted message |
| `on_message_rejected` | broadcast | `on_message`, after a USER message is vetoed; fires exactly once per rejected message |
| `on_message_persisted` | broadcast | dequeue, after the correlation-originating row is persisted; fires once per originating item regardless of outcome (`rowid` may be None) |
| `on_llm_error` | firstresult=True | `_run_turn`, after LLM exception; return error string or None for default |
| `compact_conversation` | broadcast | `_process_queue_item`, step 5; all implementations run for side effects |
| `process_tool_result` | firstresult=True wrapper chain | `dispatch_tool_call` (both subagent and main agent paths), after tool execution; return replacement string or None for default |
| `before_agent_turn` | broadcast | `_process_queue_item`, before LLM call; plugins inject context into conversation log |
| `after_persist_assistant` | broadcast | `_process_queue_item`, after assistant message is persisted; plugins may mutate the in-memory dict |
| `transform_display_text` | firstresult=True wrapper chain | `_resolve_display_text`, before `send_message`; return transformed text or None to leave unchanged |
| `on_idle` | broadcast | `IdleMonitor`, when all queues empty and cooldown elapsed |
| `load_conversation` | firstresult=True | `_process_queue_item`, when `channel.conversation is None`; return list of tagged message dicts or None to defer |
| `resolve_channel` | firstresult=True | `SchedulerPlugin`, for a persisted channel id the registry does not hold (e.g. a reminder due after a restart); a transport returns the live Channel for ids it owns, created with transport-specific config, or None to defer. `SignalPlugin` implements it; callers fall back to `registry.get_or_create` from the split id |
| `on_conversation_event` | broadcast | `_process_queue_item`, after every `conv.append()`; side effects only (persistence, JSONL logging) |
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
per-channel YAML → runtime overrides (set by `set_settings` tool).
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
```

`channel.id` returns `"{transport}:{scope}"`.

**ChannelRegistry** — lifecycle management. `get_or_create(transport,
scope, config)` returns an existing channel or creates one.
`resolve_config(channel)` calls `channel.config.resolve(agent_defaults,
runtime_overrides=channel.runtime_overrides)` so runtime overrides are
factored into every resolution.

`load_channel_config(config, registry)` pre-registers channels from
YAML before `on_start`.

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
```

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
`is_model_not_loaded(error)` recognizes the unloaded-model case.

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
(`stage`, `channel_id`, `correlation_id`, `meta`, plus stage-specific
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
    correlation_id TEXT,        -- NULL when the call ran outside any correlated turn
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
    exchange_key TEXT PRIMARY KEY,  -- the correlation id (outcome_log.py keeps its own vocabulary; see plugin-guide.md's Correlation section)
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
calls the `process_tool_result` hook after each tool execution. When
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
- `remove_by_type(message_type)` — removes all entries of the given `MessageType`
  from the in-memory list. Returns the number removed. Raises `ValueError` if called
  with `MessageType.MESSAGE` or `MessageType.SUMMARY`. Plugins use this to clean up
  stale `CONTEXT` entries from an earlier turn before re-injecting fresh ones.

After each `conv.append()`, `Agent` fires `on_conversation_event` (broadcast)
so persistence plugins can write to their storage. After `replace_with_summary()`,
`CompactionPlugin` fires `on_compaction` (broadcast) so persistence plugins can
update the DB boundary.

### Append-only log

Conversation history is irreplaceable and must never be deleted. The
`message_log` table in `PersistencePlugin` is strictly append-only — no
rows are ever deleted or updated. The DB is a complete audit log
including `reasoning_content`. This invariant applies to all persistence
paths: SQLite, JSONL, and any future storage backends.

`on_compaction` inserts a summary row whose timestamp encodes the boundary:
`oldest_retained_message.timestamp - 1e-6`. `load_conversation` filters with
`WHERE timestamp > summary_ts`, returning only the summary plus retained and new
messages. Old rows remain in the DB but are invisible to the working set.

`remove_by_type()` operates on the in-memory list only. Old CONTEXT rows in the DB
become invisible after the next compaction (their timestamps fall below the boundary).

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
```

`message_type` is a persistence category, orthogonal to conversational role
(`user`, `assistant`, `tool`, `system`). `CompactionPlugin` only summarizes `MESSAGE`
entries; `SUMMARY` and `CONTEXT` rows survive compaction.

In-memory message dicts carry a `_message_type` metadata key (set by `append()`).
`build_prompt()` strips it before returning messages to the LLM.

The `before_agent_turn` hook gives plugins a chance to inject entries (e.g., `CONTEXT`
rows) into the conversation before each LLM call. `Agent` fires
`on_conversation_event` for each injected message after the hook returns.

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
friends ([MemoryPlugin](#memoryplugin)), `exchange_log` (OutcomeLogPlugin),
`usage_log` (UsageLogPlugin), and `schedule`
([SchedulerPlugin](#schedulerplugin)).

## Agent

`agent.py` — the central plugin. Implements `on_start`, `on_message`,
`on_notify`, `on_stop`. Declares `depends_on = {"registry", "task", "llm", "tools"}`
so `validate_dependencies(pm)` verifies those dependencies are registered at startup.
Retrieves `ChannelRegistry` during `on_start` via
`get_dependency(pm, "registry", ChannelRegistry)` from `hooks.py`.

DB lifecycle (open/close) is delegated to `PersistencePlugin`. Conversation
initialization is handled directly in `Agent._process_queue_item`: a
`ContextWindow` is created, then `load_conversation` (firstresult=True) is
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

`Agent.queues` is a public `dict[str, SerialQueue]` (keyed by channel ID).
Queues added after startup are visible to other plugins that hold a reference
to this dict because the reference is live.

### Message processing

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
   `load_conversation` (firstresult=True) to populate history, and assigns
   `channel.conversation`
3. Resets `turn_counter` to 0 when the item originates a new correlation
   (`item.originating`, set during correlation resolution: every USER
   item, and a notification with no inherited `meta["correlation_id"]`)
   **and** no tool cycle is in flight (`channel.pending_tool_call_ids` is
   empty). An item arriving mid-cycle is charged to the running cycle
   instead (R4), so a chatty channel cannot extend `max_turns`. Keying on
   `originating` rather than the USER role makes `max_turns` a
   per-exchange budget for clock-driven exchanges too: a channel fed only
   by scheduler notifications would otherwise exhaust it after ten
   responses and have every later tool call suppressed.
4. Resolves channel config (including `max_turns`)
5. Appends the inbound message to conversation; fires `on_conversation_event`
   (broadcast, side effects only — persistence plugins write to storage)
5b. **Tool result batching** — if the item is a tool-result notification
    (`role=NOTIFICATION` with `tool_call_id` set), removes the ID from
    `channel.pending_tool_call_ids`. If the set is non-empty after the
    removal, returns immediately (more results still pending). The LLM call
    is deferred until the last result clears the set. User messages can
    interleave during this window — they don't clear the pending set.
6. Broadcasts `compact_conversation`; all implementations run for side
   effects; return values are not used
7. Calls `before_agent_turn` hook (broadcast); fires `on_conversation_event`
   for any messages injected by the hook
8. **`_run_turn`** — calls `run_agent_turn` (single LLM invocation).
   On exception, broadcasts `on_llm_error` (`firstresult=True`), sends
   the error message to the channel, and returns `None` to abort
   processing. `LLMPlugin` implements it `trylast`: for an unloaded
   backend model it returns a message naming the model, and it defers
   (`None`) on every other error, so the generic message still applies.
9. Appends the assistant message to conversation; fires `on_conversation_event`
10. Calls `after_persist_assistant` hook (broadcast); `ThinkingPlugin`
    uses this to strip `reasoning_content` from the in-memory copy
11. **`_handle_response`** — decision point:
    - **Tool calls, under limit** (`turn_counter < max_turns`):
      increment counter, dispatch tool calls as Tasks, return without
      sending a response
    - **Tool calls, at limit**: suppresses tool dispatch. Calls
      `_resolve_display_text` with `MAX_TURNS_FALLBACK_MESSAGE` as
      fallback, fires `on_agent_response`, sends text response.
    - **No tool calls**: increments counter, calls
      `_resolve_display_text`, fires `on_agent_response`, sends text
      response

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

After `_handle_response` completes, `_process_queue_item` calls
`_maybe_fire_idle()`. This is the push-based idle detection mechanism:
after each queue item finishes, `Agent` checks whether all `SerialQueue`
instances in `self.queues` are empty and `TaskQueue.is_idle` is `True`
(skipped if `TaskPlugin` is not registered). If so, and if at least
`idle_cooldown_seconds` have elapsed since the last `on_idle` firing,
it broadcasts the `on_idle` hook.

**`_resolve_display_text`** broadcasts `transform_display_text`
(`firstresult=True` wrapper chain). If the hook returns a value,
that value is used; otherwise falls back to `result.text`. When a
`fallback` is provided and the resolved text is falsy, the fallback is
returned instead. Hook exceptions are caught, logged, and the input text
is returned.

### Tool call dispatch

`_dispatch_tool_calls` records all tool call IDs in
`channel.pending_tool_call_ids`, then creates a `Task` for each tool call
and enqueues it on the `TaskQueue` (accessed via
`getattr(pm.get_plugin("task"), "task_queue", None)`). Each task's `work`
closure calls `dispatch_tool_call()` from `tool.py`, which handles JSON
parsing, unknown-tool detection, invocation, error wrapping, logging, and
the `process_tool_result` hook. The closure returns `result.content`; the
TaskPlugin delivers it via `on_notify`.

`TaskPlugin` is declared in `Agent.depends_on`. Tool dispatch logs an error
and returns without enqueuing if the task queue is unavailable.

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

### load_conversation

Implements `load_conversation` (firstresult=True). Called by `Agent` when
`channel.conversation is None`. Queries `message_log` for the channel, applying
the timestamp filter so only the compaction boundary and newer rows are returned.
Returns a list of tagged message dicts (with `_message_type` set). Logs an INFO
event when history is loaded.

### on_conversation_event

Implements `on_conversation_event` (broadcast). Called after every `conv.append()`.
Strips `_message_type` from the message dict before inserting into `message_log`.

### on_compaction

Implements `on_compaction` (broadcast). Called after `replace_with_summary()`.
Determines the timestamp boundary from the most-recent non-retained row, inserts
the summary row with `timestamp = boundary - 1e-6`, and commits.

**Graceful degradation:** without `PersistencePlugin`, `load_conversation` returns
no history and conversation events are not persisted.

## JsonlLogPlugin

`jsonl_log.py` — writes an append-only JSONL log alongside the SQLite
conversation store. Registered as the `"jsonl_log"` entry point.

Configured via `daemon.jsonl_log_dir` in `agent.yaml`. If the key is
absent, the plugin is a complete no-op — all hookimpls return early.
When configured, `on_start` resolves the directory relative to
`_base_dir` and creates it if necessary.

### on_conversation_event

Implements `on_conversation_event` (broadcast). Strips `_message_type`
from the message dict and writes a JSON line with `ts`, `channel`,
`type`, and `message` fields to a per-channel `.jsonl` file.

### on_compaction

Implements `on_compaction` (broadcast). Writes the compaction summary as
a JSON line with `type: "summary"`.

### File handles

Per-channel file handles are opened lazily in `_get_handle` and held
open for the plugin lifetime. Channel IDs are sanitized (`/` and `:` →
`_`) for filenames. All handles are flushed and closed in `on_stop`.

## IdleMonitorPlugin

`idle.py` — a no-op stub that declares `depends_on = set()`. It implements
the `on_idle` hook with a pass-through body. Registered as the
`"idle_monitor"` entry point.

Idle detection is push-based, not polling-based. After each queue item
finishes, `Agent._maybe_fire_idle()` checks whether all queues are empty and
the cooldown has elapsed, then broadcasts `on_idle` directly. See the Agent
section for the full idle detection logic.

`IdleMonitorPlugin` exists as a registration point for future idle-triggered
behavior. Other plugins that want to react to idle events implement the
`on_idle` hook directly.

**Graceful degradation:** without `IdleMonitorPlugin`, the `on_idle` hook
may still fire if other plugins implement it, since `Agent._maybe_fire_idle()`
broadcasts to all registered `on_idle` implementations regardless.

**Config:**
```yaml
daemon:
  idle_cooldown_seconds: 30   # minimum seconds between on_idle firings (default 30)
```

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

`compaction.py` — implements the `compact_conversation` hook to keep
conversation history within the configured token budget. Registered as
the `"compaction"` entry point. Declares `depends_on = {"llm"}`.
Logger name: `corvidae.compaction`.

**ContextCompactPlugin** (`context_compact.py`) has been removed: superseded
by `MemoryPlugin` (compaction-time memory consolidation plus retrieval), and
per-turn token stats now live in the `usage_log` table.

### compact_conversation

Implements the `compact_conversation` broadcast hook. The hook is called
for side effects; its return value is not used by the caller.

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
   cycle, if the retained tail stays at or below the midpoint between
   `compaction_retention` and `compaction_threshold`. Otherwise (a
   long-running call near the head with chat interleaved behind it) move
   forward with `snap_cut_forward` to drop the cycle whole, if at least
   half the retention budget survives. Otherwise keep the walk's cut, as
   before #51. Backing up without that bound could leave compaction
   dropping almost nothing and re-running every few messages. In-flight
   calls have no result yet and never straddle; one dropped by compaction
   leaves its later result orphaned, as before.
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
`compact_conversation` hook returns `None` on every turn and compaction
does not run. The conversation history will grow without bound until the
LLM context window is exceeded.

## Forks

`fork.py` plus `Agent.fork` — background processes run as branches of the
main window (GitHub issue #51; plan of record `plans/forks.md`). Off by
default: `agent.forks.enabled`.

On a single llama-server KV slot, every separate background prompt evicts
the main context, and compaction re-prefills the whole window. A **fork**
instead renders the channel's window exactly as the main turn does, up to
a cut, and appends one `user` instruction. The prompt is then a prefix of
what the server already holds, so it reuses the cached KV, and it carries
the agent's own system prompt, so first-person records and summaries are
written by the agent they describe.

```
fork_prompt = build_prompt(window[:cut]) + [{"role": "user", "content": instruction}]
```

**Prefix parity.** `Agent.fork(channel, instruction, *, cut=None,
extra_body=None)` builds with exactly what step 7 of the agent loop uses:
the `llm.main` client (its `extra_body`, e.g. `id_slot`), `_tool_schemas`,
`build_prompt(pending_ids, mid_conversation_system_role)`, and the same
runtime-override filter (`_llm_overrides`). `reasoning_effort` is left
alone, because the chat template renders it into the system block.
On top of that a fork sends `tool_choice: "none"` (tools stay in the
render), `reasoning_budget_tokens` from `agent.forks.reasoning_budget_tokens`
(a per-request sampler budget that never touches the template), and the
stage's `extra_body` (`response_format`). Those keys are applied after the
runtime overrides, so `set_settings` can't re-enable tool calls on a fork.

**Invariants.** A fork mutates nothing, persists nothing, fires no hooks,
and never dispatches tools: a reply carrying `tool_calls`, or a tool call
written out as text (`<tool_call>`, `<function…`, a bare
`{"name", "arguments"}` object), raises `ForkRefused`. No live window or
no main client raises `ForkUnavailable`.
Every caller treats any fork exception, or an empty or unparseable reply,
as a miss: it logs at WARNING and makes its standalone call instead.

**Which process forks from where.** Plugins reach the Agent through
`fork_agent(pm, stage)`, which returns it only when that stage is enabled.
Stage names are opaque to core (it never names cognition plugins).

| Stage | Cut | Runs | Standalone when |
|---|---|---|---|
| `compaction` | dropped prefix | inline in `compact_conversation` | forks off for the stage; fork miss |
| `consolidation` (compaction trigger) | dropped prefix | inline in `before_compaction`; the record is stashed and the `on_compaction` task consumes it | also: a range dialog row is not in `messages[:cut]`; the watermark moved since the fork |
| `consolidation` (idle trigger) | full window | in the idle task | also: no registered channel or live window; a range row not in the window |
| `appraisal` (stage 2) | full window at run time | silent TaskQueue task | also: the exchange's originating rowid has left the window (or was never persisted) |
| `critique` (each lens) | full window at run time | silent TaskQueue task | same as appraisal; a lens whose fork misses falls back alone |

Seeding, imports, `consolidate_replay.py` and the `eval_*` scripts have no
live window and always run standalone. Subagents are unchanged: workers
with their own persona-neutral prompts.

The compaction and consolidation forks share one prefix: the second
fork's prefill is just its instruction. Containment for consolidation is
exact: every `message`-type row in the range must be a `_db_id` in the
fork's window (summary rows are skipped; a live window's summary has no
rowid and summaries are never consolidated). Appraisal and critique check
the originating rowid recorded by their own `on_message_persisted`
(critique per lens, right before each fork, since an earlier lens's fork
is an await during which compaction can drop the exchange), and read the
window at run time, so a fork after the user has said more still sees the
exchange. The instruction names it by quoting its opening message and the
response the user was sent, which can differ from the window's last
assistant message (the max-turns fallback).

**Instructions.** Each stage has a fork instruction beside its standalone
prompt, written to the agent in the second person:
`agent.compaction_fork_prompt`, `memory.consolidation_fork_prompt`,
`appraisal.stage2_fork_prompt`, `critique.fork_prompt.<lens>` (documented
copies in `prompts/*_fork.md`). They are read from static config with
`tuning.config_value`, never `resolve_tunable`, so `set_settings` can't
write them. Consolidation appends a sentence naming where the range
starts ("the whole conversation above", "what follows the summary at its
top", or a quote of the first new message when rows below the watermark
are still in view). Appraisal appends the exchange quotes and the
retrieval profile. Critique appends the exchange quotes, and the
provenance lens still passes the CONTEXT snapshot inline.

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

**Not yet settled.** Whether `response_format` is enforced under
`tool_choice: "none"` with tools present, and `/apply-template` parity with
and without `tool_choice`, need the live checks in #73 (K2, K3); the
harness probes and the flip of `enabled` are tracked there.

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
  per-plugin opt-outs.

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

# Legacy (deprecated):
agent:
  max_tool_result_chars: 100000   # use tools.max_result_chars instead
```

**Graceful degradation:** without `ToolCollectionPlugin`, `Agent.depends_on`
validation fails at startup.

## MemoryPlugin

`memory.py` — autobiographical memory. Entry point
name: `"memory"`; `depends_on = {"persistence", "llm"}`. Absorbs the former
`DreamPlugin` embryo (removed): compaction becomes memory formation instead
of file-based fact scraping.

### Consolidation (write path)

Two triggers, one watermarked code path:

- **`on_compaction(channel, summary_msg, retain_count, compacted_ids)`** —
  the moment dialog leaves the active window it enters long-term memory.
  The compacted `message_log` id range rides the hook payload (rowid
  threading, below); consolidation never depends on the summary row being
  in the DB.
- **`on_idle`** — channels inactive past `memory.idle_consolidate_after`
  seconds get their un-consolidated tail consolidated.

Both spawn a tracked plugin-owned `asyncio.create_task` (never the
TaskQueue — every TaskQueue completion wakes the main model). The task
reads the per-channel **consolidation watermark**, fetches the working
range, produces the record (first-person summary with epistemic framing,
JSON out) — from a fork of the live window when one is available (see
[Forks](#forks)), else one standalone `llm.background` call
(`prompts/memory_consolidation.md`) —
scores an importance prior through the pluggable
`MemoryPlugin.importance_prior` (default: cheap-model `RubricPrior`,
0.5 on failure; `AppraisalPlugin` wraps it in `AppraisalPrior` when
appraisal is enabled), embeds the summary via
`llm.embedding`, and commits the memory row together with the watermark
advance in one compare-and-set section. The row's `created_at` is the
`message_log.timestamp` of the last row in the source range, i.e. when
the conversation happened, not when consolidation ran, so history
consolidated long after the fact is aged correctly by retrieval and
retention.

**Near-duplicate merge.** Before inserting the new vector, consolidation
looks for the nearest indexed record with cosine similarity
≥ `memory.dup_threshold` among the records in the channel's retrieval
scope (#58). A record on another channel qualifies only if that channel's
scope has the same reach (`ChannelScope.same_reach`), because groups are
not transitive: with groups {A, B} and {A, C}, folding C's record into one
owned by A, and later A's into one owned by B, would hide the fact from C.
Equal reach keeps the survivor visible to exactly the channels that could
see the superseded record. Under a single `*` group every channel
qualifies. The old record is superseded (`superseded_by`, `indexed=0`,
vector deleted); the survivor takes the summed retrieval count, the
higher importance, and the later of the two records' dates. Message
ranges fold only within one channel: `message_log` ids are global, and
`recall_raw` and `/redact` read a record's range on its own channel, so on
a cross-channel merge each record keeps its own range and the superseded
record still leads to its channel's source dialog. Overlapping triggers on the same
range discard the loser's record — no duplicates. Embedding failure
stores the record with `embedded=0` for later backfill; the summary text
is canonical and vectors are a rebuildable cache. All consolidation LLM
calls carry `stage="consolidation"` attribution, along with the `trigger`
that started the run (`"compaction"` or `"idle"`) — set once at
`_consolidate_range`'s entry, so it rides into the `on_llm_response`
attribution snapshot for that call, into `MetricsPlugin`'s emitted tags,
the `usage_log.trigger` column, and the `memory consolidated` completion
log line.

### Retrieval (read path)

`before_agent_turn` runs when the window tail is a plain user MESSAGE
(notification-triggered turns skip retrieval entirely). The inbound text is embedded and matched against `memory_vec`
(sqlite-vec exact KNN, cosine distance); candidates are filtered to
`indexed=1 AND redacted=0` and channel-compartmentalized (`channel_id`
match, or shared scope via `memory.channel_groups`, whose members may be
`*` / `?` globs matched with SQLite `GLOB`). Scores are
`cosine_similarity × exp(-age_days / memory.half_life_days)`, where
`age_days` runs from `created_at`, or from `max(created_at,
last_retrieved_at)` under `memory.retrieval.recency_from: last_used`.
Candidates rank by score and are annotated into relevance bands (strong
≥ 0.75, moderate ≥ 0.60, else weak; weak drops unless nothing else
matched). The bands apply to the score by default, or to raw similarity
under `memory.retrieval.band_on: similarity`, so an old but apt memory is
not presented as a guess. The `(age)` shown always runs from
`created_at`. Winners enter the window as
`[band] (age) summary` lines through the admission funnel — never by
direct CONTEXT append. Admitted memories get their access statistics
bumped (`retrieval_count`, `last_retrieved_at` — the write half of
reconsolidation), and every retrieval writes a `retrieval_log` row
(the top-ranked candidate's decayed score and raw similarity,
hit/admitted counts, degradation flag).

**Graceful degradation:** encoder down → FTS5 keyword retrieval over
`memory_fts` (flagged `degraded_to_fts=1`); sqlite-vec unloadable → FTS5
with one clear startup warning; no `llm.embedding` role → records stored
with `embedded=0`, retrieval FTS5-only; a stored encoder differing from
config → ERROR and embeddings disabled rather than silently mixing
encoders. Nothing in the memory path raises into the turn loop.

### Schema

`memory` (records, with `indexed`, `superseded_by`, `redacted`, and
`embedded` columns), `consolidation_watermark`, `embedding_meta`,
`memory_fts` (external-content FTS5 with insert/update sync triggers; no
delete trigger — memory rows are never deleted), `memory_vec` (vec0,
created only when the extension loads), `retrieval_log`, `retention_meta`
(one-row table persisting `last_run` for the retention job interval), and
`message_fts` (regular FTS5 table over `message_log` content, maintained by
`message_log_ai`/`message_log_au` triggers; queried and cleared by `corvidae
redact` — not by `search_memory`, which queries `memory_fts`). All on the persistence connection in
`sessions.db`; `message_log` stays append-only except for the
operator-only redaction tombstone.

The epistemic-calibration prompt fragment (`prompts/memory_calibration.md`)
is documented but NOT auto-injected — hedging vocabulary is persona, not
architecture.

### Operator redaction (`corvidae redact`)

`corvidae/commands/redact.py` — operator-only CLI for permanent content
removal. Never a registered agent tool; never reachable from any hook the
model can influence.

**Forms:**

```
corvidae redact --db sessions.db message <id> [<id2>...]   # specific message_log rows
corvidae redact --db sessions.db memory <memory_id>        # a record + its raw range
corvidae redact --db sessions.db range <start_id> <end_id> # message_log id range (inclusive)
```

`--dry-run` prints affected row counts without writing.

**WAL requirement.** The CLI opens its own connection (`PRAGMA busy_timeout
= 5000`, short transactions) and asserts WAL journal mode at startup. If
the mode is not WAL, it aborts with a message naming the current mode and
how to convert. The `corvidae` daemon configures WAL on startup; a
daemon-created DB is always WAL.

**Schema-presence probe (pre-memory-schema DBs).** Before touching any
surface, the CLI probes `sqlite_master` for the tables and triggers it
needs. If `message_fts` is absent, the message tombstone is still written
but the FTS surface is skipped with a printed notice — safe because the
`message_log_ai`/`message_log_au` triggers will index the tombstone (not
the original content) the next time the daemon runs its `message_fts`
backfill. If `memory` or `memory_vec` are absent, the memory cascade is
skipped with a notice. The CLI never creates schema; schema is
daemon-owned.

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

**Verification pass.** After each cascade, the CLI extracts a sample token
from the original content (before tombstoning) and runs FTS `MATCH` queries
against `message_fts` and `memory_fts`. It prints `"verified: 0 hits for
'<token>' in <tables>"` when clean or `"WARNING: FTS still contains hits for
'<token>': <table>: <N> hits"` when the cascade left searchable content — every invocation proves its own
completeness. Skipped when `--dry-run` is in effect or when no suitable
sample token can be extracted.

## FunnelPlugin

`funnel.py` — the context-admission funnel. Entry
point name: `"funnel"`. The single chokepoint for tail CONTEXT admission:

1. **Dedupe** against CONTEXT already in the window.
2. **Budget** per source (`funnel.default_budget`, default 512 tokens;
   `funnel.budgets.<source>` overrides), greedy in given order, drops
   logged rather than silent.
3. **Frame** every admitted batch in explicit data-not-instructions
   wrapping with a source label (injection defense, written once here).
4. **Append + persist** one CONTEXT message at the tail through
   `on_conversation_event`, with the rowid attached, so the window matches
   its reload. Stale CONTEXT retires by aging past the compaction boundary.

Callers: memory retrieval (and optionally date/time grounding). The
funnel also owns deferred registration: non-tool-call notification
payloads queue per `(channel, origin)` via `register_and_wake()`, one
stub notification wakes the channel per pending pair, and the drain in
`before_agent_turn` admits everything queued for the triggering
exchange's origin.

## SchedulerPlugin

`scheduler.py` — the clock-driven producer: reminders, operator-declared
periodic jobs, and the heartbeat (#59; design rationale in
`plans/scheduler.md`). Entry point name: `"scheduler"`;
`depends_on = {"registry", "persistence"}`. `on_idle` is no substitute:
it is push-based, so a daemon with no traffic never fires it.

**Rows.** One `schedule` table on the persistence connection. A row is a
due time, an optional interval, a target channel id, the text, and an
origin (`reminder` or `heartbeat`). Reminders come from the `remind_me`
tool and are pinned to the calling channel. Jobs come from
`scheduler.jobs` in `agent.yaml` and are synced by `name`.

```sql
CREATE TABLE schedule (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,            -- config jobs only; NULL for tool-created rows
  channel_id TEXT NOT NULL,    -- "transport:scope", the persistence key
  origin TEXT NOT NULL,        -- 'reminder' | 'heartbeat'
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
`meta["schedule_id"]` travels with the correlation.

**Timer.** One asyncio task sleeps until the earliest enabled row is due
(at most an hour, so a wall-clock jump is noticed). An `asyncio.Event`
wakes it early whenever a row is added, cancelled, or re-synced. Before
firing anything it waits until the persistence DB is open and
`Agent.started` is True. Clock and sleeper are constructor arguments, so
tests inject fakes and never sleep.

**Firing.** A due row is first advanced (recurring) or disabled as
`fired` (one-shot): at most once. Then the channel is resolved
(`registry.get`, then the `resolve_channel` hook, then
`registry.get_or_create` from the split id), `channel.touch()` runs so
MemoryPlugin's inactivity check does not consolidate mid-exchange, and
the plugin calls:

```python
await pm.ahook.on_notify(
    channel=channel, source=row.origin, text=framed_text,
    tool_call_id=None, meta={"origin": row.origin, "schedule_id": row.id},
)
```

From there the normal pipeline runs: serial queue, persistence,
origin-keyed consumers, attribution, and `send_message`, whose reply on
the row's channel is the unprompted message. A heartbeat on a channel no
transport owns (`internal:heartbeat`) reaches the log and persistence
only, because every transport's `send_*` hooks filter on
`matches_transport`. The originating notification resets `turn_counter`
(step 3 of [Message processing](#message-processing)).

**Framing and role.** Reminder text is framed as data:

> A reminder you set on 2026-10-23 14:02 (PDT) for this conversation is
> due. Reminder text: «...». Decide whether and how to act on it.

`«` and `»` inside the text are replaced, so the text cannot close the
quote. Config jobs read `Scheduled job "<name>" (set by the operator,
every <interval>) is due. Job prompt: «...». ...`. A late catch-up firing
is prefixed `[overdue by 42m, daemon was down]`. The row is stored as
`role: system` with the `[reminder]`/`[heartbeat]` prefix like any
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
the full tool set minus `subagent` itself.

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
  calls `_connect_server` for each. Connections are entered into an
  `AsyncExitStack` so sessions and transports stay alive for the daemon's
  lifetime. After all connections, calls `_build_tool_list()` and caches the
  result. Connection failures per server are caught, logged as warnings, and
  skipped; a failed server does not abort startup.
- `register_tools` — extends `tool_registry` with the cached `Tool` instances
  (sync; called after `on_start` completes).
- `on_stop` — closes all sessions and transports via `AsyncExitStack.aclose()`.

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
and `definitions` from the top level of `mcp_tool.inputSchema` before wrapping
it in the OpenAI function-call envelope. Sanitization is shallow (top-level
only); nested occurrences in `properties` are not removed.

### Configuration

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

- LLM inference keys: `temperature`, `top_p`, `top_k`, `frequency_penalty`,
  `presence_penalty`, `max_tokens`
- Framework keys: `max_turns`, `max_context_tokens`, `reasoning_history`
- Dotted plugin tunables (`critique.*`, `appraisal.*`; see configuration.md)

Keys are not validated, except `reasoning_history`, which is rejected
(with no overrides applied) unless it is `drop`, `exchange`, or `keep` — a
stored bad value would make every later config resolve raise. See the
Blocklist note below for what the lack of validation means.

Pass `null` for a key to clear that override and revert to the static config value.

### Return value

On success: a confirmation string listing current overrides (or noting that all
overrides have been cleared). On blocked keys: an error string naming the
blocked keys.

### Blocklist

`system_prompt` is always blocked regardless of operator config. Additional
keys are blocked via `agent.immutable_settings` in `agent.yaml`:

```yaml
agent:
  immutable_settings:   # keys the agent must not change (system_prompt always added)
    - max_turns
```

Note the limit of what this buys: any undotted key that is neither a
framework key nor blocklisted is forwarded verbatim into the LLM request
body, where it can override `model` and `messages`, so blocking
`system_prompt` does not by itself protect the system prompt. Blocklisting
`messages`, `model`, `tools` and `tool_choice` covers the known cases; see
[threat-model.md](threat-model.md), finding 1.

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
backoff retry (10s→300s cap). Joins configured channels on connect.
Forwards both channel and private messages. `split_message` splits
outgoing text into 400-byte chunks preserving paragraph/sentence
boundaries.

`split_message` itself lives in `channels/split.py`, not in the IRC
plugin, so a transport can size its own chunks without importing IRC.

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
forward resolution. It returns None for ids it does not own. Known gap:
forward resolution starts only after a connect, so a call that lands
before the first connect finds no alias and creates the channel with the
default config, which the registry then keeps for the process lifetime.

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
vetoes and the one place that logs the attempt.

**Liveness.** Two indicators, both gated on the allowlist — an
unauthorized sender gets no evidence that the account is live and
monitored.

- *Typing.* On the first outstanding message for a channel the transport
  sends `sendTyping` immediately, covering queueing delay, and starts a
  refresh loop that re-sends every 10s while the channel's outstanding
  count stays positive (Signal clients expire an indicator after roughly
  15s). `send_message` and `on_message_rejected` each give one outstanding
  slot back; a second gate plugin vetoing a message this transport already
  counted is why the rejection path decrements too. No explicit stop is
  ever sent — once the loop exits, the indicator expires on its own.
- *Read receipt.* Sent from `on_message_persisted` — processing start —
  not on arrival. A message can sit queued behind contended local
  inference, and a "read" marker followed by a long silence is worse than
  no marker. Envelope timestamps are held FIFO per channel and popped when
  the matching turn starts; only correlation ids this transport itself
  admitted are eligible, since `on_message_persisted` also fires for
  notification-originated turns that never pushed a timestamp. That
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

Group conversations are out of scope; agent-initiated (unprompted) sends
and any privacy handling specific to disappearing messages are not
implemented.

## Configuration

```yaml
daemon:
  session_db: sessions.db
  idle_cooldown_seconds: 30   # minimum seconds between on_idle firings (default 30)

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
  idle_consolidate_after: 1800  # seconds of channel inactivity before idle consolidation
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
    - name: heartbeat
      channel: internal:heartbeat
      origin: heartbeat         # reminder (default) | heartbeat
      every: 6h                 # required
      at: "08:00"               # optional anchor: HH:MM or ISO datetime
      prompt: "..."             # or prompt_file: prompts/heartbeat.md

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
  channels:
    - "#channel"

logging:                        # optional, falls back to defaults
  level: INFO                   # DEBUG | INFO | WARNING | ERROR | CRITICAL
  file: corvidae.log            # omit for stderr; CLI mode defaults to corvidae.log
```

When `system_prompt` is a list, each path is read and concatenated.
Relative paths resolve against the directory containing `agent.yaml`.

## Plugin Registration Order

`Runtime.start()` (`runtime.py`) registers plugins in two stages, not a
fixed hand-written order:

1. `ChannelRegistry` is the only manually registered plugin —
   `self.pm.register(self.registry, name="registry")` (runtime.py:128).
2. Every other plugin loads from the `corvidae` setuptools entry-point
   group: `self.pm.load_setuptools_entrypoints("corvidae")`
   (runtime.py:141), reading `[project.entry-points.corvidae]` in
   `pyproject.toml:35-62`. Names listed in `plugins.disabled` are blocked
   via `pm.set_blocked(name)` before this call, so disabled plugins are
   never instantiated.

The full entry-point plugin set (`pyproject.toml:36-62`): `persistence`,
`jsonl_log`, `core_tools`, `cli`, `irc`, `signal`, `task`, `subagent`, `mcp`, `llm`,
`compaction`, `memory`, `memory_tools`, `funnel`, `thinking`,
`runtime_settings`, `tools`, `metrics`, `usage_log`, `metrics_jsonl`,
`outcome_log`, `appraisal`, `critique`, `scheduler`, `agent`,
`idle_monitor`, `config_watcher`.

Entry-point loading does not guarantee a meaningful cross-plugin order;
plugins that need another plugin's state resolve it lazily via
`get_dependency`/`on_start` ordering (`trylast=True`), not registration
sequence.

Immediately after entry points load, `Runtime.start()` logs `"plugins
loaded"` with `loaded` (the sorted names `pm.list_name_plugin()` actually
returns) and `blocked` (the configured `plugins.disabled` list, verbatim
— not narrowed to names that matched a real plugin). A misspelled disable
entry shows up under `blocked` while the plugin it was meant to block
still shows up under `loaded`; that mismatch is the operator-visible
signal of the typo. It is not an error and does not stop the daemon.

After entry points load, `validate_dependencies(pm)` runs (runtime.py:159)
to verify every plugin's `depends_on` set names a registered plugin.
Startup aborts with `RuntimeError` if any dependency is missing.

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

### Current dependency graph

```
Agent               → "registry", "task", "llm", "tools"
CLIPlugin           → "registry"    (ChannelRegistry)
IRCPlugin           → "registry"    (ChannelRegistry)
SubagentPlugin      → "llm", "tools"
CompactionPlugin    → "llm"         (LLMPlugin); soft-uses "agent" for forks
ThinkingPlugin      → "registry"    (ChannelRegistry)
MemoryPlugin        → "persistence", "llm"; soft-uses "agent", "registry" for forks
MemoryToolsPlugin   → "memory"      (MemoryPlugin)
CritiquePlugin      → "task", "llm"; soft-uses "agent" for forks
OutcomeLogPlugin    → "persistence"
AppraisalPlugin     → "persistence"; soft-uses "agent" for forks
UsageLogPlugin      → "persistence"
SchedulerPlugin     → "registry", "persistence"
```

Plugins with `depends_on = set()` (declared but empty):
`LLMPlugin`, `ToolCollectionPlugin`, `PersistencePlugin`, `IdleMonitorPlugin`,
`CoreToolsPlugin`, `McpClientPlugin`, `RuntimeSettingsPlugin`, `JsonlLogPlugin`,
`TaskPlugin`, `MetricsPlugin`, `MetricsJsonlPlugin`, `ConfigWatcherPlugin`,
`FunnelPlugin`.

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
├── jsonl_log.py          # JsonlLogPlugin (on_conversation_event, on_compaction)
├── logging.py            # StructuredFormatter, configure_logging()
├── agent.py              # Agent (single-turn dispatch)
├── persistence.py        # PersistencePlugin (DB lifecycle, conversation init)
├── idle.py               # IdleMonitorPlugin (no-op stub; idle detection is push-based in Agent)
├── thinking.py           # ThinkingPlugin, strip_thinking(), strip_reasoning_content()
├── compaction.py         # CompactionPlugin
├── fork.py               # fork prompt builder, cut snap, agent.forks parser (#51)
├── task.py               # Task, TaskQueue, TaskPlugin
├── mcp_client.py         # McpClientPlugin (MCP server bridge)
├── memory.py             # MemoryPlugin (consolidation + retrieval + message_fts schema)
├── retention.py          # run_retention_job(), retention_score()
├── funnel.py             # FunnelPlugin (context-admission funnel)
├── main.py               # daemon entry point
├── channels/
│   ├── cli.py            # CLIPlugin
│   ├── irc.py            # IRCPlugin
│   ├── signal.py         # SignalPlugin (signal-cli JSON-RPC transport)
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
    ├── memory_tools.py   # MemoryToolsPlugin, search_memory, recall_raw
    └── task_pipeline.py  # TaskPipelinePlugin, task_pipeline tool
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
them; the implemented transports are CLI, IRC, and Signal.
