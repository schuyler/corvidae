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
    async def compact_conversation(self, conversation: ConversationLog, client: LLMClient, max_tokens: int) -> None
    async def process_tool_result(self, tool_name: str, result: str, channel: Channel | None) -> str | None
    async def before_agent_turn(self, channel: Channel) -> None
    async def after_persist_assistant(self, channel: Channel, message: dict) -> None
    async def transform_display_text(self, channel: Channel, text: str, result_message: dict) -> str | None
    async def on_idle(self) -> None
    async def ensure_conversation(self, channel: Channel) -> bool | None
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
| `process_tool_result` | broadcast / VALUE_FIRST | `run_agent_loop` (subagent path only, not interactive messages), after tool execution; return replacement string or None for default |
| `before_agent_turn` | broadcast | `_process_queue_item`, before LLM call; plugins inject context into conversation log |
| `after_persist_assistant` | broadcast | `_process_queue_item`, after assistant message is persisted; plugins may mutate the in-memory dict |
| `transform_display_text` | broadcast / VALUE_FIRST | `_resolve_display_text`, before `send_message`; return transformed text or None to leave unchanged |
| `on_idle` | broadcast | `IdleMonitor`, when all queues empty and cooldown elapsed |
| `ensure_conversation` | broadcast / ACCEPT_WINS | `_process_queue_item`, before agent turn when `channel.conversation is None`; return True=initialized, None=defer |

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

`resolve(agent_defaults)` returns a dict with all values resolved
(channel value if set, else agent default, else hardcoded fallback).
Defaults: `max_context_tokens=24000`, `keep_thinking_in_history=False`,
`max_turns=10`.

**Channel** — identifies a transport + scope combination:

```python
@dataclass
class Channel:
    transport: str
    scope: str
    config: ChannelConfig
    conversation: ConversationLog | None = None
    created_at: float
    last_active: float
    turn_counter: int = 0  # consecutive LLM turns without user message
```

`channel.id` returns `"{transport}:{scope}"`.

**ChannelRegistry** — lifecycle management. `get_or_create(transport,
scope, config)` returns an existing channel or creates one.
`resolve_config(channel)` calls `channel.config.resolve(agent_defaults)`.

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
    def __init__(self, base_url: str, model: str, api_key: str | None = None, extra_body: dict | None = None)
    async def start(self)
    async def stop(self)
    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict
```

## Agent Loop

`agent_loop.py` contains two functions:

**`run_agent_turn`** — single LLM invocation. Used by AgentPlugin for
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
when provided, calls the `process_tool_result` hook after each tool
execution. When `pm` is `None`, the hook is skipped.

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

`conversation.py` provides persistence and prompt construction.

**ConversationLog** — per-channel conversation with SQLite persistence:

- `load()` — loads history from DB
- `append(message, message_type=MessageType.MESSAGE)` — appends to in-memory list and persists to DB
- `build_prompt()` — returns `[system_msg, *messages]`
- `token_estimate()` — rough count via `chars / 3.5`
- `replace_with_summary(summary_msg, retain_count)` — replaces older
  messages with a summary, retaining the `retain_count` most-recent
  entries. Updates the in-memory list and the DB atomically. The
  summary is stored as a `SUMMARY`-typed row whose timestamp encodes
  the compaction boundary; old rows remain in the DB but become
  invisible to `load()` via the timestamp filter. Raises
  `ValueError` if `retain_count` exceeds `len(messages)`.
- `remove_by_type(message_type)` — removes all entries of the given
  `MessageType` from the in-memory list only. Returns the
  number of entries removed. Raises `ValueError` if called with
  `MessageType.MESSAGE` or `MessageType.SUMMARY` — those types are
  managed by compaction. Plugins use this to clean up previously
  injected `CONTEXT` entries before re-injecting fresh ones.

### Append-only log

The `message_log` table is append-only. No rows are ever deleted or
updated. The DB serves as a complete, immutable audit log — including
`reasoning_content`.

Compaction (`replace_with_summary`) inserts a new summary row whose
timestamp encodes the boundary: `oldest_retained_message.timestamp -
1e-6`. `load()` filters with `WHERE timestamp > summary_ts`, returning
only the summary plus retained and new messages. Old rows remain in the
DB but are invisible to the working set.

`remove_by_type()` operates on the in-memory message list only. Old
CONTEXT rows remain in the DB but become invisible after the next
compaction (their timestamps fall below the summary boundary).

**Thinking token handling** — three layers:
- Display: `ThinkingPlugin.transform_display_text` calls `strip_thinking()`
  to remove `<think>` blocks before the text is sent to the channel
- Persistent log: full message dict preserved (including
  `reasoning_content`)
- Active prompt: `ThinkingPlugin.after_persist_assistant` calls
  `strip_reasoning_content()` on the in-memory message when
  `keep_thinking_in_history=false`

If `ThinkingPlugin` is not registered, `<think>` blocks pass through
to the channel and `reasoning_content` remains in the in-memory history.

### Message types

```python
class MessageType(str, enum.Enum):
    MESSAGE = "message"    # ordinary conversation turn (user or assistant)
    SUMMARY = "summary"    # compaction summary replacing older messages
    CONTEXT = "context"    # plugin-injected context (memory, notes, etc.)
```

`message_type` is a persistence category, orthogonal to conversational
role (`user`, `assistant`, `tool`, `system`). It controls how compaction
treats a row: `CompactionPlugin` only summarizes `MESSAGE` entries;
`SUMMARY` and `CONTEXT` rows are retained through compaction.

In-memory message dicts carry a `_message_type` metadata key (set during
`load()` and `append()`). `build_prompt()` strips it before returning
messages to the LLM. `append()` accepts an optional `message_type`
parameter (default `MessageType.MESSAGE`), allowing plugins to inject
non-message entries directly into the conversation log.

The `before_agent_turn` hook gives plugins a chance to inject entries
(e.g., `CONTEXT` rows) into the conversation log before each LLM call.

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

## AgentPlugin

`agent.py` — the central plugin. Implements `on_start`, `on_message`,
`on_notify`, `on_stop`. Declares `depends_on = {"registry"}` so
`validate_dependencies(pm)` can verify its dependency is registered at
startup. Retrieves `ChannelRegistry` during `on_start` via
`get_dependency(pm, "registry", ChannelRegistry)` from `hooks.py`.

DB lifecycle (open/close) and conversation initialization are delegated to
`PersistencePlugin` via the `ensure_conversation` hook. AgentPlugin does not
manage a database connection or a `base_dir`.

`AgentPlugin.queues` is a public `dict[str, SerialQueue]` (keyed by
channel ID). `IdleMonitorPlugin` holds a reference to this dict; queues
added after `IdleMonitorPlugin.on_start` are included automatically
because the reference is live.

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
2. Broadcasts `ensure_conversation` when `channel.conversation is None`
   and resolves with `HookStrategy.ACCEPT_WINS`; if no plugin returns
   `True`, logs an error and drops the message
3. Resets `turn_counter` to 0 on user messages
4. Resolves channel config (including `max_turns`)
5. Appends the inbound message to conversation log
6. Broadcasts `compact_conversation`; all implementations run for side
   effects; return values are not used
7. Calls `before_agent_turn` hook (broadcast)
8. **`_run_turn`** — calls `run_agent_turn` (single LLM invocation).
   On exception, broadcasts `on_llm_error` (resolved with
   `HookStrategy.VALUE_FIRST`), sends the error message to the channel,
   and returns `None` to abort processing.
9. Persists the assistant message
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

**`_resolve_display_text`** broadcasts `transform_display_text` and
resolves with `HookStrategy.VALUE_FIRST`. If the hook returns a value,
that value is used; otherwise falls back to `result.text`. When a
`fallback` is provided and the resolved text is falsy, the fallback is
returned instead. Hook exceptions are caught, logged, and the input text
is returned.

### Tool call dispatch

`_dispatch_tool_calls` creates a `Task` for each tool call and enqueues
it on the `TaskQueue` (accessed via
`getattr(pm.get_plugin("task"), "task_queue", None)`). `TaskPlugin` is
intentionally not declared in `AgentPlugin.depends_on` — it is treated
as optional, and tool dispatch degrades gracefully (logs an error) if
the task queue is unavailable. Each task's `work` closure:

- Parses the tool call arguments
- Looks up the tool function in `self.tools`
- Injects `ToolContext` for tools declaring `_ctx`
- Calls the tool, returns the string result
- On unknown tool or exception, returns an error string (no crash)

Closure capture uses default argument binding to avoid the
loop-variable capture bug.

### Notification-driven continuation

When a tool task completes, `TaskPlugin` fires `on_notify` with the
result and `tool_call_id`. `AgentPlugin.on_notify` enqueues a
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

`persistence.py` — manages the SQLite database lifecycle and
per-channel conversation initialization. Registered as `"persistence"` in
`main.py`, immediately after `ChannelRegistry`.

`PersistencePlugin` depends on `"registry"`. Its `on_start` retrieves
`ChannelRegistry` via `get_dependency(pm, "registry", ChannelRegistry)`,
reads `daemon.session_db` from config (default `"sessions.db"`), opens the
database with `aiosqlite.connect`, and calls `init_db` to create the schema.
`on_stop` closes the connection.

`PersistencePlugin.db` is public for test injection: setting `persistence.db`
before `on_start` causes the `if self.db is None:` guard to skip the open.
`PersistencePlugin.base_dir` holds the directory containing `agent.yaml` and
is used to resolve relative system prompt file paths.

### ensure_conversation

Implements the `ensure_conversation` broadcast hook (resolved with
`HookStrategy.ACCEPT_WINS`). Called by `AgentPlugin._process_queue_item` (step 2)
when `channel.conversation is None`.

Steps:
1. If `channel.conversation` is already set, returns `True` immediately
2. Creates a `ConversationLog` bound to the plugin's DB connection and
   the channel ID
3. Resolves the channel config via `ChannelRegistry.resolve_config` and
   calls `resolve_system_prompt` with `self.base_dir`; assigns
   `conv.system_prompt` (**must** happen before `load()`)
4. Calls `conv.load()` to populate history from the DB
5. Assigns `channel.conversation = conv`
6. Returns `True`

Initialization events log to `corvidae.persistence`.

**Graceful degradation:** without `PersistencePlugin`, the `ensure_conversation`
hook returns `None`. `AgentPlugin` logs an error and drops the message; no crash.

## IdleMonitorPlugin

`idle.py` — monitors system idle state and fires `on_idle` when all
queues are quiescent. Registered as `"idle_monitor"` in `main.py`.

`IdleMonitorPlugin` depends on `"agent_loop"`. Its `on_start` is
decorated with `@hookimpl(trylast=True)` so it runs after
`AgentPlugin.on_start` has fully initialized.

`IdleMonitorPlugin.on_start` calls `get_dependency(pm, "agent_loop",
AgentPlugin)` to retrieve a reference to `AgentPlugin.queues`, then
creates and starts an `IdleMonitor`.

`IdleMonitorPlugin.on_stop` cancels the monitor before queue teardown.

**Graceful degradation:** without `IdleMonitorPlugin`, the `on_idle`
hook is never fired.

### IdleMonitor

`IdleMonitor` (`idle.py`) is a background asyncio task that polls for
idle state and fires `on_idle` when conditions are met.

**Idle condition:** all `SerialQueue` instances in `AgentPlugin.queues`
have `is_empty=True`, `TaskQueue.is_idle` is `True` (no queued or
active tasks; skipped if `TaskPlugin` is not registered), and at least
`idle_cooldown_seconds` have elapsed since the last `on_idle` firing.

**Polling:** checks the idle condition every `idle_poll_interval`
seconds. Exceptions from `on_idle` implementations are caught and
logged as warnings; the monitor continues running.

**Config:**
```yaml
daemon:
  idle_cooldown_seconds: 30   # minimum seconds between on_idle firings (default 30)
  idle_poll_interval: 2       # seconds between idle checks (default 2)
```

## ThinkingPlugin

`thinking.py` — strips `<think>` blocks and `reasoning_content` for
display. Registered as `"thinking"` in `main.py` before `agent_loop`.

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
`"compaction"` in `main.py` before `AgentPlugin`. Logger name:
`corvidae.compaction`.

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
6. Call `_summarize(client, older_clean)` — sends the older messages to the
   LLM with a system prompt asking for a concise summary. This method is a
   separate public method so tests can patch it via `patch.object`.
7. Call `conversation.replace_with_summary(summary_msg, retain_count)`.
8. Return `True`.

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
`daemon.max_task_workers` from config (default 4) to set concurrency.
Starts/stops the worker on lifecycle hooks. Registers the `task_status`
tool. On task completion, fires `on_notify`.

## Subagent Tool

`tools/subagent.py` — `SubagentPlugin` registers the `subagent` tool, which
launches a background agent with its own LLM session and the full tool set.

### Tool signature

```python
async def subagent(instructions: str, description: str, _ctx: ToolContext) -> str
```

`_ctx` is system-injected (excluded from the LLM-visible schema). Returns a
confirmation string with the enqueued task ID on success, or an error string
if the task queue or channel context is unavailable.

### How it works

1. At tool-call time (not `on_start`), retrieves `AgentPlugin` via
   `get_dependency(self.pm, "agent_loop", AgentPlugin)` and reads
   `AgentPlugin._max_tool_result_chars` and `AgentPlugin.tool_registry`.
2. Calls `registry.exclude("subagent")` to remove itself from the subagent's tool set.
3. Builds a `messages` list (system prompt + user instructions) and a
   `work` coroutine that captures it, creates a fresh `LLMClient`, calls
   `run_agent_loop` with `channel` and `task_queue` from `_ctx` (enabling
   the subagent to enqueue nested tasks), strips thinking from the result,
   then shuts the client down.
4. Wraps `work` in a `Task` with `tool_call_id` from `_ctx`, enqueues it on
   `ctx.task_queue`.
5. Returns immediately; result is delivered via `TaskPlugin → on_notify →
   AgentPlugin` (same path as all other tasks).

### LLM configuration

Uses `llm.background` if present in config, otherwise `llm.main`. Configured
at `on_start` time; subagent calls after startup use the value captured then.

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
| `shell(command)` | `tools/shell.py` | Execute shell command, 30s timeout |
| `read_file(path)` | `tools/files.py` | Read file (<1MB) |
| `write_file(path, content)` | `tools/files.py` | Write file, creates parent dirs |
| `web_fetch(url)` | `tools/web.py` | Fetch URL, 15s timeout, 50KB truncation |

Additionally, `SubagentPlugin` registers `subagent` and `TaskPlugin`
registers `task_status` as tool closures during `on_start`.

| Tool | File | Purpose |
|------|------|---------|
| `subagent(instructions, description)` | `tools/subagent.py` | Launch background subagent with own LLM session |
| `task_status()` | `task.py` | Report task queue status |

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
  idle_poll_interval: 2       # seconds between idle checks (default 2)

llm:
  main:
    base_url: "http://host:8080"
    model: "model-name"
    api_key: "optional"
    extra_body: {}              # optional, passed through to LLM
  background:                   # optional separate LLM for subagent tasks
    base_url: "..."
    model: "..."

agent:
  system_prompt: "..."          # string or list of file paths
  max_context_tokens: 24000
  keep_thinking_in_history: false
  max_turns: 10

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
2. `PersistencePlugin` — DB lifecycle and conversation initialization (after registry, before everything else)
3. `CoreToolsPlugin` — registers core tools
4. `CLIPlugin` — stdin/stdout transport
5. `IRCPlugin` — IRC transport
6. `TaskPlugin` — task queue
7. `SubagentPlugin` — registers the `subagent` tool
8. `McpClientPlugin` — MCP server connections and tool forwarding
9. `CompactionPlugin` — provides default `compact_conversation` implementation
10. `ThinkingPlugin` — handles `<think>` stripping and `reasoning_content` removal
11. `RuntimeSettingsPlugin` — registers the `set_settings` tool
12. `AgentPlugin` — agent loop (after all tools, transports, and support plugins)
13. `IdleMonitorPlugin` — idle monitor (after `AgentPlugin`, depends on `agent_loop`)

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
AgentPlugin         → "registry"    (ChannelRegistry)
CLIPlugin           → "registry"    (ChannelRegistry)
IRCPlugin           → "registry"    (ChannelRegistry)
PersistencePlugin   → "registry"    (ChannelRegistry)
SubagentPlugin      → "agent_loop"  (AgentPlugin)
IdleMonitorPlugin   → "agent_loop"  (AgentPlugin)
```

Plugins with no `depends_on` attribute declared: `CoreToolsPlugin`,
`McpClientPlugin`, `CompactionPlugin`, `ThinkingPlugin`, `RuntimeSettingsPlugin`.

Transport plugins use `get_dependency(pm, "registry", ChannelRegistry)` in
`on_start` to retrieve the shared `ChannelRegistry`. `SubagentPlugin` calls
`get_dependency(pm, "agent_loop", AgentPlugin)` at tool-call time (inside
`_launch`) to access `AgentPlugin.tool_registry`.

`AgentPlugin` also accesses `TaskPlugin` via `pm.get_plugin("task")`, but
this is intentionally not declared in `depends_on` — `TaskPlugin` is
treated as optional, and `_dispatch_tool_calls` degrades gracefully
(logs an error and returns) when the task queue is unavailable.

## Directory Layout

```
corvidae/
├── hooks.py              # AgentSpec, hookimpl, create_plugin_manager
├── tool.py               # Tool, ToolRegistry, tool_to_schema, ToolContext
├── channel.py            # Channel, ChannelConfig, ChannelRegistry, resolve_system_prompt
├── queue.py              # SerialQueue (is_empty property)
├── llm.py                # LLMClient
├── agent_loop.py         # run_agent_turn(), run_agent_loop(), strip_thinking
├── conversation.py       # ConversationLog, init_db
├── logging.py            # StructuredFormatter, _DEFAULT_LOGGING
├── agent.py              # AgentPlugin (single-turn dispatch)
├── persistence.py        # PersistencePlugin (DB lifecycle, conversation init)
├── idle.py               # IdleMonitor, IdleMonitorPlugin
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
    ├── web.py
    ├── subagent.py       # SubagentPlugin, subagent tool
    └── settings.py       # RuntimeSettingsPlugin, set_settings tool
```

## Known Risks

**Tool result ordering.** When the LLM requests multiple tool calls,
each completes independently and triggers a separate agent turn. The
LLM sees results one at a time, not batched. This may cause chattier
behavior or premature responses. Batching is intentionally not
implemented: it adds timeout and partial-failure complexity (what
happens when one tool in a batch hangs or errors?) that is not worth
the cost for a personal daemon where correctness and simplicity matter
more than throughput. Can be revisited if chattiness becomes a real
problem.

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
