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
except `register_tools` which is sync.

```python
class AgentSpec:
    async def on_start(self, config: dict) -> None
    async def on_stop(self) -> None
    async def on_message(self, channel: Channel, sender: str, text: str) -> None
    async def send_message(self, channel: Channel, text: str, latency_ms: float | None = None) -> None
    def register_tools(self, tool_registry: list) -> None  # sync; plugins append Tool instances or bare callables
    async def on_agent_response(self, channel: Channel, request_text: str, response_text: str) -> None
    async def on_notify(self, channel: Channel, source: str, text: str, tool_call_id: str | None, meta: dict | None) -> None
    async def should_process_message(self, channel: Channel, sender: str, text: str) -> bool | None
    async def on_llm_error(self, channel: Channel, error: Exception) -> str | None
    async def compact_conversation(self, channel: Channel, conversation: ContextWindow, max_tokens: int) -> None
    async def process_tool_result(self, tool_name: str, result: str, channel: Channel | None) -> str | None
    async def before_agent_turn(self, channel: Channel) -> None
    async def after_persist_assistant(self, channel: Channel, message: dict) -> None
    async def transform_display_text(self, channel: Channel, text: str, result_message: dict) -> str | None
    async def on_idle(self) -> None
    async def load_conversation(self, channel: Channel) -> list[dict] | None
    async def on_conversation_event(self, channel: Channel, message: dict, message_type: MessageType) -> None
    async def on_compaction(self, channel: Channel, summary_msg: dict, retain_count: int) -> None
```

`create_plugin_manager()` in `hooks.py` creates the manager and adds
hookspecs.

### Broadcast dispatch and result resolution

All hooks are called via `pm.ahook.<hook_name>(...)`, which broadcasts to
every registered implementation and returns a list of results. For hooks
that need a single resolved value, the caller passes the result list to
`resolve_hook_results(results, hook_name, strategy, pm=pm)` from `hooks.py`.

`HookStrategy` defines three resolution strategies:

- **REJECT_WINS** — any `False` in the results returns `False`; otherwise
  any `True` returns `True`; otherwise `None`.
- **ACCEPT_WINS** — any `True` in the results returns `True`; otherwise `None`.
- **VALUE_FIRST** — returns the first non-`None` result. If multiple plugins
  return non-`None`, the alphabetically-first plugin name's result is used
  and a warning is logged.

Pure broadcast hooks (e.g. `on_start`, `on_idle`) do not call
`resolve_hook_results`; their return values are ignored.

### Hook reference

| Hook | Semantics | Call site |
|------|-----------|-----------|
| `on_start` | broadcast | daemon startup, after config load |
| `on_stop` | broadcast | SIGINT/SIGTERM |
| `on_message` | broadcast | inbound message from any transport |
| `send_message` | broadcast | outbound message delivery |
| `register_tools` | broadcast (sync) | startup tool collection |
| `on_agent_response` | broadcast | after agent loop produces a text response; no default implementation |
| `on_notify` | broadcast | inject a notification into a channel queue |
| `should_process_message` | broadcast / REJECT_WINS | `on_message`, before enqueue; no default implementation; None from empty broadcast allows all messages |
| `on_llm_error` | broadcast / VALUE_FIRST | `_run_turn`, after LLM exception; return error string or None for default |
| `compact_conversation` | broadcast | `_process_queue_item`, step 5; all implementations run for side effects |
| `process_tool_result` | broadcast / VALUE_FIRST | `dispatch_tool_call` (both subagent and main agent paths), after tool execution; return replacement string or None for default |
| `before_agent_turn` | broadcast | `_process_queue_item`, before LLM call; plugins inject context into conversation log |
| `after_persist_assistant` | broadcast | `_process_queue_item`, after assistant message is persisted; plugins may mutate the in-memory dict |
| `transform_display_text` | broadcast / VALUE_FIRST | `_resolve_display_text`, before `send_message`; return transformed text or None to leave unchanged |
| `on_idle` | broadcast | `IdleMonitor`, when all queues empty and cooldown elapsed |
| `load_conversation` | broadcast / VALUE_FIRST | `_process_queue_item`, when `channel.conversation is None`; return list of tagged message dicts or None to defer |
| `on_conversation_event` | broadcast | `_process_queue_item`, after every `conv.append()`; side effects only (persistence, JSONL logging) |
| `on_compaction` | broadcast | `CompactionPlugin`, after `replace_with_summary()`; side effects only (persistence) |

## Channel System

`channel.py` provides multi-transport, multi-scope channel management.

**ChannelConfig** — per-channel config with agent-level fallback:

```python
@dataclass
class ChannelConfig:
    system_prompt: str | list[str] | None = None
    max_context_tokens: int | None = None
    keep_thinking_in_history: bool | None = None
    max_turns: int | None = None
```

`resolve(agent_defaults, runtime_overrides=None)` returns a dict with all
values resolved. Resolution order: built-in defaults → agent-level YAML →
per-channel YAML → runtime overrides (set by `set_settings` tool).
Defaults: `max_context_tokens=24000`, `keep_thinking_in_history=False`,
`max_turns=10`.

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

## Agent Loop

`agent_loop.py` contains two functions:

**`run_agent_turn`** — single LLM invocation. Used by Agent for
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

**`run_agent_loop`** — multi-turn blocking loop. Used by subagent
execution. Calls the LLM, executes tool calls inline, repeats until a
text response or max_turns. Injects `ToolContext` into tools that
declare a `_ctx` parameter. Accepts an optional `pm` keyword argument;
when provided, passes it to `dispatch_tool_call()`, which calls the
`process_tool_result` hook after each tool execution. When `pm` is
`None`, the hook is skipped.

Also in `agent_loop.py`:
- `strip_thinking(text)` — removes `<think>...</think>` blocks
- `strip_reasoning_content(messages)` — removes `reasoning_content`
  from message dicts in place
- `_truncate(text, max_len)` — truncation utility for logging

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
- `build_prompt()` — returns `[system_msg, *messages]` with `_message_type` stripped
- `token_estimate()` — rough count via `chars / chars_per_token`
- `replace_with_summary(summary_msg, retain_count)` — replaces older messages
  with a summary in-memory, retaining the `retain_count` most-recent entries.
  Raises `ValueError` if `retain_count` exceeds `len(messages)`.
- `remove_by_type(message_type)` — removes all entries of the given `MessageType`
  from the in-memory list. Returns the number removed. Raises `ValueError` if called
  with `MessageType.MESSAGE` or `MessageType.SUMMARY`. Plugins use this to clean up
  previously injected `CONTEXT` entries before re-injecting fresh ones.

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
- Active prompt: `ThinkingPlugin.after_persist_assistant` calls
  `strip_reasoning_content()` on the in-memory message when
  `keep_thinking_in_history=false`

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

## Agent

`agent.py` — the central plugin. Implements `on_start`, `on_message`,
`on_notify`, `on_stop`. Declares `depends_on = {"registry", "task", "llm", "tools"}`
so `validate_dependencies(pm)` verifies those dependencies are registered at startup.
Retrieves `ChannelRegistry` during `on_start` via
`get_dependency(pm, "registry", ChannelRegistry)` from `hooks.py`.

DB lifecycle (open/close) is delegated to `PersistencePlugin`. Conversation
initialization is handled directly in `Agent._process_queue_item`: a
`ContextWindow` is created, then `load_conversation` (VALUE_FIRST) is called
so persistence plugins can populate history. `Agent` reads `_chars_per_token`
and `_base_dir` from config in `_start_plugin`.

The LLM client is borrowed from `LLMPlugin` (`get_dependency(pm, "llm", LLMPlugin)`)
in `_start_plugin`. The tool registry and `max_result_chars` are borrowed from
`ToolCollectionPlugin` (`get_dependency(pm, "tools", ToolCollectionPlugin)`).
Agent does not own either lifecycle.

`Agent.queues` is a public `dict[str, SerialQueue]` (keyed by channel ID).
Queues added after startup are visible to other plugins that hold a reference
to this dict because the reference is live.

### Message processing

`on_message` broadcasts `should_process_message` and passes the result
list to `resolve_hook_results` with `HookStrategy.REJECT_WINS`. If the
resolved result is `False` (identity check, not falsiness), the message
is dropped. Any other result (`True` or `None`) proceeds to enqueue.

`on_message` and `on_notify` both enqueue a `QueueItem` onto a
per-channel `SerialQueue`. The queue's consumer calls
`_process_queue_item`, which delegates to four helper methods:

1. **`_build_conversation_message`** — converts a `QueueItem` into a
   conversation message dict and `request_text`. Returns `None` on
   unrecognized role (logged, item dropped).
2. When `channel.conversation is None`, creates a `ContextWindow`, calls
   `load_conversation` (VALUE_FIRST) to populate history, and assigns
   `channel.conversation`
3. Resets `turn_counter` to 0 on user messages
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
   On exception, broadcasts `on_llm_error` (resolved with
   `HookStrategy.VALUE_FIRST`), sends the error message to the channel,
   and returns `None` to abort processing.
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

After `_handle_response` completes, `_process_queue_item` calls
`_maybe_fire_idle()`. This is the push-based idle detection mechanism:
after each queue item finishes, `Agent` checks whether all `SerialQueue`
instances in `self.queues` are empty and `TaskQueue.is_idle` is `True`
(skipped if `TaskPlugin` is not registered). If so, and if at least
`idle_cooldown_seconds` have elapsed since the last `on_idle` firing,
it broadcasts the `on_idle` hook. This replaces the former polling-based
`IdleMonitor` background task.

**`_resolve_display_text`** broadcasts `transform_display_text` and
resolves with `HookStrategy.VALUE_FIRST`. If the hook returns a value,
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
conversation, and calls `_run_turn` again.

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
persistence. Registered as `"persistence"` in `main.py`, immediately after
`ChannelRegistry`. Declares `depends_on = set()`.

`PersistencePlugin.on_start` reads `daemon.session_db` from config (default
`"sessions.db"`), opens the database with `aiosqlite.connect`, and runs schema
creation. `on_stop` closes the connection.

`PersistencePlugin.db` is public for test injection: setting `persistence.db`
before `on_start` causes the `if self.db is None:` guard to skip the open.

### load_conversation

Implements `load_conversation` (VALUE_FIRST). Called by `Agent` when
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
conversation store. Registered as `"jsonl_log"` in `main.py`.

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
the `on_idle` hook with a pass-through body. Registered as `"idle_monitor"`
in `main.py` after `Agent`.

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
display. Registered as `"thinking"` in `main.py` before `agent`.

Implements two hooks:

- `after_persist_assistant`: reads `keep_thinking_in_history` from the
  resolved channel config. If `False`, calls `strip_reasoning_content`
  on the in-memory message dict. The DB copy is already written; this
  only affects subsequent prompt builds.
- `transform_display_text`: calls `strip_thinking` on the response text.
  Returns the stripped string if it differs from the input, or `None`
  if no `<think>` tags were present.

**Graceful degradation:** without `ThinkingPlugin`, `<think>` blocks
pass through to the channel and `reasoning_content` remains in the
in-memory history regardless of `keep_thinking_in_history`.

## CompactionPlugin

`compaction.py` — implements the `compact_conversation` hook to keep
conversation history within the configured token budget. Registered as
`"compaction"` in `main.py` before `Agent`. Declares `depends_on = {"llm"}`.
Logger name: `corvidae.compaction`.

**ContextCompactPlugin** (`context_compact.py`) is an alternative compaction
strategy that is currently disabled in `main.py`. It is commented out because
it conflicts with `CompactionPlugin` when both operate on the same conversation.

### compact_conversation

Implements the `compact_conversation` broadcast hook. The hook is called
for side effects; its return value is not used by the caller.

**Algorithm:**

1. If `conversation.token_estimate() < 80% of max_tokens`, return `None` (skip).
2. If `len(conversation.messages) <= 5`, return `None` (skip — too few messages to compact).
3. Backward walk: starting from the most recent message, accumulate messages
   until their token estimate would exceed 50% of `max_tokens`. The count
   of accumulated messages is `retain_count`.
4. If `retain_count >= len(conversation.messages)`, return `None` (all messages fit — no-op).
5. Filter the older (non-retained) messages to `MESSAGE` type only, excluding
   any `SUMMARY` entries. Strip `_message_type` metadata before passing to LLM.
6. Call `_summarize(older_clean)` — sends the older messages to the LLM
   with a system prompt asking for a concise summary. Resolves the LLM
   client lazily via `get_dependency(pm, "llm", LLMPlugin)` on first call
   (not in `on_start`, because `on_start` hooks run in LIFO order and
   `CompactionPlugin.on_start` fires before `LLMPlugin.on_start`). Caps
   input to 100 messages (first 50 + last 50 with a truncation marker).
   This method is a separate public method so tests can patch it via
   `patch.object`.
7. Call `conversation.replace_with_summary(summary_msg, retain_count)`.
8. Fire `on_compaction(channel, summary_msg, retain_count)` (broadcast) so
   persistence plugins can update the DB.
9. Return `True`.

**Graceful degradation:** without `CompactionPlugin`, the
`compact_conversation` hook returns `None` on every turn and compaction
does not run. The conversation history will grow without bound until the
LLM context window is exceeded.

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

`llm_plugin.py` — owns `LLMClient` instance lifecycle. Registered as `"llm"`
in `main.py` after `McpClientPlugin`. Declares `depends_on = set()`.

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
`"tools"` in `main.py`. Declares `depends_on = set()`. Its `on_start` is decorated
with `@hookimpl(trylast=True)` so it fires after all other `on_start` hooks have
run, ensuring every tool-providing plugin has had a chance to finish setup.

### Lifecycle

- `on_start` — reads `tools.max_result_chars` from config (default 100,000).
  Falls back to `agent.max_tool_result_chars` with a deprecation warning.
  Calls `pm.hook.register_tools(tool_registry=collected)` (sync broadcast) to
  collect tools from all plugins, then builds a `ToolRegistry`.

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

## DreamPlugin

`tools/dream.py` — background memory consolidation. Registered as `"dream"` in
`main.py` after `ToolCollectionPlugin`. Does not use `hookimpl` decorators —
its `on_start` and `on_idle` are plain async methods registered via the standard
`pm.register()` path.

### What it does

On each `on_idle` firing, if `interval_seconds` have elapsed since the last
dream cycle, queries `sessions.db` for recent assistant messages (up to 40
rows, filtered to `role=assistant`, capped at 20 per channel), extracts
sentences from the content, deduplicates against the existing `## Long-term
Memory` section of `MEMORY.md`, and appends new facts (up to 20 per cycle).

### DB discovery

Searches for `sessions.db` in the workspace root at startup, checking
`workspace/corvidae/sessions.db` and `workspace/sessions.db` first, then
recursively up to depth 3. If not found, the dream cycle is a no-op.

### Configuration

```yaml
dream:
  interval_seconds: 300   # minimum seconds between dream cycles (default 300)
```

**Graceful degradation:** if `sessions.db` is not found, or if `MEMORY.md`
is absent, the plugin skips the cycle without error.

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

The `settings` dict accepts:

- LLM inference keys: `temperature`, `top_p`, `top_k`, `frequency_penalty`,
  `presence_penalty`, `max_tokens`
- Framework keys: `max_turns`, `max_context_tokens`, `keep_thinking_in_history`

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

The following tool files exist but are not registered in the current `main.py`:

| File | Status |
|------|--------|
| `tools/goal_tracker.py` | Experimental; not registered |
| `tools/perf_mon.py` | Experimental; not registered |
| `tools/local_indexer.py` | Experimental; not registered |
| `tools/index.py` | Disabled via commented-out registration in `main.py` |

## Transports

### CLI (`channels/cli.py`)

`CLIPlugin` — stdin/stdout transport. Routes to `cli:local` channel.
Reads stdin line-by-line in a background task. Prints responses to
stdout with latency timing.

### IRC (`channels/irc.py`)

`IRCPlugin` — IRC transport via pydle. Connects with exponential
backoff retry (10s→300s cap). Joins configured channels on connect.
Forwards both channel and private messages. `split_message` splits
outgoing text into 400-byte chunks preserving paragraph/sentence
boundaries.

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
    timeout: null               # HTTP timeout per request; null = aiohttp default (300s)
  background:                   # optional separate LLM for subagent tasks
    base_url: "..."
    model: "..."
    # same keys as main

agent:
  system_prompt: "..."          # string or list of file paths
  max_context_tokens: 24000
  keep_thinking_in_history: false
  max_turns: 10
  immutable_settings:           # keys the agent must not change (system_prompt always added)
    - max_turns

tools:
  max_result_chars: 100000      # truncation limit for tool result strings (default 100_000)
  shell_timeout: 30             # shell tool timeout in seconds (default 30)
  web_fetch_timeout: 15         # web_fetch timeout in seconds (default 15)
  web_max_response_bytes: 50000 # web_fetch response truncation (default 50_000)
  max_file_read_bytes: 1048576  # read_file size limit in bytes (default 1MB)
  web_search_max_results: 8     # web_search result count (default 8)

dream:
  interval_seconds: 300         # minimum seconds between dream cycles (default 300)

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
  version: 1
  ...
```

When `system_prompt` is a list, each path is read and concatenated.
Relative paths resolve against the directory containing `agent.yaml`.

## Plugin Registration Order

Defined in `main.py`:

1. `ChannelRegistry` — registered as a named plugin (`"registry"`) on the PM
2. `PersistencePlugin` — DB lifecycle and conversation persistence (after registry)
3. `JsonlLogPlugin` — JSONL conversation logging (peer persistence hook consumer)
4. `CoreToolsPlugin` — registers core tools (shell, read_file, write_file, web_fetch, web_search, task_pipeline)
5. `CLIPlugin` — stdin/stdout transport
6. `IRCPlugin` — IRC transport
7. `TaskPlugin` — task queue
8. `SubagentPlugin` — registers the `subagent` tool
9. `McpClientPlugin` — MCP server connections and tool forwarding
10. `LLMPlugin` — owns LLM client lifecycle (after mcp, before compaction and agent)
11. `CompactionPlugin` — provides default `compact_conversation` implementation
12. `ThinkingPlugin` — handles `<think>` stripping and `reasoning_content` removal
13. `RuntimeSettingsPlugin` — registers the `set_settings` tool
14. `ToolCollectionPlugin` — collects tools from all plugins via `register_tools` hook; `on_start` is `trylast=True` so it fires after all other `on_start` hooks
15. `DreamPlugin` — background memory consolidation via `on_idle`
16. `Agent` — agent loop (after all tools, transports, and support plugins)
17. `IdleMonitorPlugin` — no-op stub; registered after `Agent`

After all registrations, `validate_dependencies(pm)` runs to verify that
every plugin's `depends_on` set names a registered plugin. Startup aborts
with `RuntimeError` if any dependency is missing.

## Plugin Dependencies

Plugins declare dependencies using a class-level `depends_on` attribute (a
set of plugin name strings). `validate_dependencies(pm)` in `hooks.py`
iterates all registered plugins and raises `RuntimeError` if any declared
dependency is not found in the PM. It runs in `main.py` after all plugins
are registered, before `on_start`.

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
CompactionPlugin    → "llm"         (LLMPlugin)
ThinkingPlugin      → "registry"    (ChannelRegistry)
```

Plugins with `depends_on = set()` (declared but empty):
`LLMPlugin`, `ToolCollectionPlugin`, `PersistencePlugin`, `IdleMonitorPlugin`,
`CoreToolsPlugin`, `McpClientPlugin`, `RuntimeSettingsPlugin`, `JsonlLogPlugin`,
`TaskPlugin`.

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
├── agent_loop.py         # run_agent_turn(), run_agent_loop(), strip_thinking
├── context.py            # ContextWindow, MessageType, DEFAULT_CHARS_PER_TOKEN
├── context_compact.py    # ContextCompactPlugin (disabled — conflicts with CompactionPlugin)
├── jsonl_log.py          # JsonlLogPlugin (on_conversation_event, on_compaction)
├── logging.py            # StructuredFormatter, _DEFAULT_LOGGING
├── agent.py              # Agent (single-turn dispatch)
├── persistence.py        # PersistencePlugin (DB lifecycle, conversation init)
├── idle.py               # IdleMonitorPlugin (no-op stub; idle detection is push-based in Agent)
├── thinking.py           # ThinkingPlugin
├── compaction.py         # CompactionPlugin
├── task.py               # Task, TaskQueue, TaskPlugin
├── mcp_client.py         # McpClientPlugin (MCP server bridge)
├── main.py               # daemon entry point
├── channels/
│   ├── cli.py            # CLIPlugin
│   └── irc.py            # IRCPlugin
└── tools/
    ├── __init__.py       # CoreToolsPlugin
    ├── shell.py
    ├── files.py
    ├── web.py            # web_fetch, web_search
    ├── subagent.py       # SubagentPlugin, subagent tool
    ├── settings.py       # RuntimeSettingsPlugin, set_settings tool
    ├── task_pipeline.py  # TaskPipelinePlugin, task_pipeline tool
    ├── dream.py          # DreamPlugin (background memory consolidation)
    ├── goal_tracker.py   # experimental; not registered
    ├── perf_mon.py       # experimental; not registered
    ├── local_indexer.py  # experimental; not registered
    └── index.py          # WorkspaceIndexerPlugin (disabled in main.py)
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

**Shell sandboxing.** The `shell` tool runs commands without any
sandbox or privilege restriction. This is intentional: Corvidae is a
personal agent daemon running on the user's own machine, not a
multi-tenant service. Sandboxing would add complexity while providing
no security benefit in the single-user deployment model. This would
need to change before deploying Corvidae in any shared or untrusted
environment.

## Unimplemented

The following items appear in earlier design documents but are not yet
implemented. Each needs discussion before proceeding.

### Hot-loading

The original design included a `ComponentLoader` with watchdog
filesystem watcher for hot-reloading plugin modules from a
`components/` directory. Not implemented. The current transport plugins
are registered statically in `main.py`.

### Memory retrieval

The original design described a retrieval system using embeddings
(nomic-embed-text or similar) with sqlite-vec for vector similarity
search. The system would inject `<memory>` blocks into the conversation
stream between turns. Prerequisites exist (the message log with durable
compaction), but the retrieval pipeline is not built.

### Double-buffer compaction

An optimization where a new compacted prompt is built from the full log
in a second llama-server KV cache slot, warmed, and swapped in
seamlessly. Not implemented; current compaction is stop-the-world.

### Additional transports

Signal, BlueSky, and other transports were noted as future work. The
hook system supports them; none are implemented beyond IRC and CLI.
