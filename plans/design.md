# Sherman Design Document

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
┌─────────────────────────────────────────────┐
│              Plugin Manager                 │
│          (apluggy.PluginManager)            │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │   IRC   │  │  Agent   │  │   Task    │  │
│  │Transport│  │  Plugin  │  │  Plugin   │  │
│  │ Plugin  │  │          │  │           │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│                     │                       │
│          ┌──────────┴──────────┐            │
│          │  llama-server       │            │
│          │  (OpenAI-compat)    │            │
│          └─────────────────────┘            │
└─────────────────────────────────────────────┘
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
    def register_tools(self, tool_registry: ToolRegistry) -> None  # sync; use Tool.from_function(fn) to register
    async def on_agent_response(self, channel: Channel, request_text: str, response_text: str) -> None
    async def on_notify(self, channel: Channel, source: str, text: str, tool_call_id: str | None, meta: dict | None) -> None
```

`create_plugin_manager()` in `hooks.py` creates the manager and adds
hookspecs.

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
) -> AgentTurnResult
```

`run_agent_turn` appends the assistant message to `messages` in place
before returning. Callers should not append it again.

**`run_agent_loop`** — multi-turn blocking loop. Used by subagent
execution. Calls the LLM, executes tool calls inline, repeats until a
text response or max_turns. Injects `ToolContext` into tools that
declare a `_ctx` parameter.

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
- `append(message)` — appends to in-memory list and persists to DB
- `build_prompt()` — returns `[system_msg, *messages]`
- `token_estimate()` — rough count via `chars / 3.5`
- `compact_if_needed(client, max_tokens)` — when token estimate reaches
  80% of limit, summarizes older messages via an LLM call. Retention
  uses a token-budget backward walk: starting from the most recent
  message, messages are kept until they would exceed 50% of
  `max_tokens`; the rest are replaced by a summary. Compaction is
  durable: summaries are persisted to the DB and summarized messages
  are deleted, so the working set survives restarts.

**Thinking token handling** — three layers:
- Display: `strip_thinking()` removes `<think>` blocks from content
- Persistent log: full message dict preserved (including
  `reasoning_content`)
- Active prompt: `strip_reasoning_content()` removes
  `reasoning_content` from in-memory messages when
  `keep_thinking_in_history=false`

### Database schema

```sql
CREATE TABLE message_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message TEXT NOT NULL,      -- JSON
    timestamp REAL NOT NULL,
    is_summary INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_log_channel ON message_log(channel_id, timestamp);
```

`is_summary` marks rows written by compaction (the synthetic summary
message). Added via `ALTER TABLE message_log ADD COLUMN is_summary
INTEGER NOT NULL DEFAULT 0` on existing databases.

## AgentPlugin

`agent.py` — the central plugin. Implements `on_start`, `on_message`,
`on_notify`, `on_stop`. Declares `depends_on = {"registry"}` so
`validate_dependencies(pm)` can verify its dependency is registered at
startup. Retrieves `ChannelRegistry` during `on_start` via
`get_dependency(pm, "registry", ChannelRegistry)` from `hooks.py`.

### Message processing

`on_message` and `on_notify` both enqueue a `QueueItem` onto a
per-channel `SerialQueue`. The queue's consumer calls
`_process_queue_item`, which:

1. Lazy-initializes conversation on the channel
2. Resets `turn_counter` to 0 on user messages
3. Resolves channel config (including `max_turns`)
4. Appends the inbound message to conversation log
5. Compacts if approaching context limit
6. Calls `run_agent_turn` (single LLM invocation)
7. Persists the assistant message
8. Strips `reasoning_content` from in-memory copy if configured
9. Decision point:
   - **Tool calls, under limit** (`turn_counter < max_turns`):
     increment counter, dispatch tool calls as Tasks, return without
     sending a response
   - **Tool calls, at limit**: send fallback text
     `"(max tool-calling rounds reached)"`
   - **No tool calls**: increment counter, send text response

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
notification `QueueItem`. The next `_process_queue_item` call formats
it as a `role: "tool"` message (if `tool_call_id` is set) or a system
message, appends it to conversation, and calls `run_agent_turn` again.

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
   `get_dependency(self.pm, "agent_loop", AgentPlugin)` and accesses
   its `tool_registry`.
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
2. CoreToolsPlugin — registers tools
3. CLIPlugin — stdin/stdout transport
4. IRCPlugin — IRC transport
5. TaskPlugin — task queue
6. SubagentPlugin — registers the subagent tool
7. AgentPlugin — agent loop (last, so all tools and queues are ready)

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
AgentPlugin      → "registry"   (ChannelRegistry)
CLIPlugin        → "registry"   (ChannelRegistry)
IRCPlugin        → "registry"   (ChannelRegistry)
SubagentPlugin   → "agent_loop" (AgentPlugin)
```

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
sherman/
├── hooks.py              # AgentSpec, hookimpl, create_plugin_manager
├── tool.py               # Tool, ToolRegistry, tool_to_schema, ToolContext
├── channel.py            # Channel, ChannelConfig, ChannelRegistry, resolve_system_prompt
├── queue.py              # SerialQueue
├── llm.py                # LLMClient
├── agent_loop.py         # run_agent_turn(), run_agent_loop(), strip_thinking
├── conversation.py       # ConversationLog, init_db
├── logging.py            # StructuredFormatter, _DEFAULT_LOGGING
├── agent.py              # AgentPlugin (single-turn dispatch)
├── task.py               # Task, TaskQueue, TaskPlugin
├── main.py               # daemon entry point
├── channels/
│   ├── cli.py            # CLIPlugin
│   └── irc.py            # IRCPlugin
└── tools/
    ├── __init__.py       # CoreToolsPlugin
    ├── shell.py
    ├── files.py
    ├── web.py
    └── subagent.py       # SubagentPlugin, subagent tool
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
sandbox or privilege restriction. This is intentional: Sherman is a
personal agent daemon running on the user's own machine, not a
multi-tenant service. Sandboxing would add complexity while providing
no security benefit in the single-user deployment model. This would
need to change before deploying Sherman in any shared or untrusted
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
