# Corvidae Plugin Guide

Corvidae plugins extend the agent daemon using [apluggy](https://pypi.org/project/apluggy/) (async pluggy). A plugin is a class with `@hookimpl`-decorated methods corresponding to the hooks it wants to handle.

```python
from corvidae.hooks import hookimpl

class GreetPlugin:
    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        if text.strip().lower() == "hello":
            await self.pm.ahook.send_message(channel=channel, text=f"Hello, {sender}!")
```

## Plugin anatomy

A plugin is a plain Python class. It only needs to implement the hooks it cares about — all hooks are optional. The `@hookimpl` decorator marks each implementation.

```python
from corvidae.hooks import hookimpl

class MyPlugin:
    @hookimpl
    async def on_start(self, config: dict) -> None:
        self.setting = config.get("my_plugin", {}).get("setting", "default")

    @hookimpl
    async def on_stop(self) -> None:
        # clean up resources
        pass
```

## Registering a plugin

### Built-in (in main.py)

Add a `pm.register()` call before `validate_dependencies()`:

```python
# corvidae/main.py
my_plugin = MyPlugin(pm)
pm.register(my_plugin, name="my_plugin")

validate_dependencies(pm)
```

Registration order matters: tool-providing and transport plugins must be registered before `agent_loop` so their tools are collected during `on_start`.

### External (setuptools entry points)

External plugins are loaded automatically via `pm.load_setuptools_entrypoints("corvidae")`. Declare them in `pyproject.toml`:

```toml
[project.entry-points.corvidae]
my_plugin = "my_package:MyPlugin"
```

The entry point value must be importable and instantiable without arguments, or be a factory function returning the plugin instance.

## Available hooks

### Lifecycle

| Hook | Type | When |
|------|------|------|
| `on_start(config: dict)` | async broadcast | Once at startup, after config is loaded |
| `on_stop()` | async broadcast | On SIGINT/SIGTERM, before process exits |

`config` is the full parsed `agent.yaml` dict. The key `_base_dir` (a `Path`) is injected by `main.py` pointing to the config file's directory.

### Messaging

| Hook | Type | When |
|------|------|------|
| `on_message(channel, sender: str, text: str)` | async broadcast | Inbound message arrives |
| `send_message(channel, text: str, latency_ms: float \| None)` | async broadcast | Outbound delivery request |
| `on_notify(channel, source: str, text: str, tool_call_id: str \| None, meta: dict \| None)` | async broadcast | Notification injected into channel |

`send_message` is broadcast to all plugins. Transport plugins filter for their own channels:

```python
@hookimpl
async def send_message(self, channel, text: str) -> None:
    if not channel.matches_transport("mytransport"):
        return
    # deliver text to this channel
```

`latency_ms` is optional in `send_message` — omit it from your hookimpl signature if your transport doesn't use it. Pluggy tolerates missing optional parameters.

### Extension points

| Hook | Type | When |
|------|------|------|
| `register_tools(tool_registry: list)` | sync broadcast | During `on_start`, to collect tools |
| `on_agent_response(channel, request_text: str, response_text: str)` | async broadcast | After agent produces a response |
| `before_agent_turn(channel)` | async broadcast | Before each LLM invocation |
| `after_persist_assistant(channel, message: dict)` | async broadcast | After assistant message is written to DB; plugins may mutate the in-memory dict |
| `on_idle()` | async broadcast | All queues empty and cooldown elapsed |

`after_persist_assistant` — the DB row is already written when this
hook fires. Mutations to `message` affect in-memory prompt construction
only; they do not update the persisted record.

### Hook result resolution

These hooks are broadcast to all plugins. The caller uses `resolve_hook_results` to reduce the result list to a single value according to a per-hook strategy.

| Hook | Strategy | Returns | Behavior |
|------|----------|---------|----------|
| `should_process_message(channel, sender, text)` | `REJECT_WINS` | `bool \| None` | Any `False` vetoes the message; any `True` (with no `False`) accepts; `None` if all defer |
| `on_llm_error(channel, error)` | `VALUE_FIRST` | `str \| None` | Non-None string replaces the default error message; multiple non-None → alphabetically-first plugin wins with a warning |
| `compact_conversation(conversation, client, max_tokens)` | broadcast only | `None` | Called for side effects; return value is not used by the caller |
| `process_tool_result(tool_name, result, channel)` | `VALUE_FIRST` | `str \| None` | Non-None string replaces the tool result in the conversation; multiple non-None → alphabetically-first plugin wins with a warning |
| `transform_display_text(channel, text, result_message)` | `VALUE_FIRST` | `str \| None` | Non-None string replaces the response text before it is sent to the channel; multiple non-None → alphabetically-first plugin wins with a warning |
| `ensure_conversation(channel)` | `ACCEPT_WINS` | `bool \| None` | Any `True` means the conversation was initialized; `None` if none handled |

`compact_conversation` — all registered implementations run. The return value is ignored by the caller. Use this hook for side effects (e.g., custom summarization that mutates the conversation in-place).

`process_tool_result` only fires during subagent execution (`run_agent_loop`), not during interactive message processing.

`transform_display_text` — `result_message` is the raw assistant message dict from the LLM response. It may contain `reasoning_content` if the model produces thinking tokens. `text` is the string content extracted from that message. Return `None` to leave `text` unchanged.

Example hook returning a value:

```python
@hookimpl
async def should_process_message(self, channel, sender: str, text: str) -> bool | None:
    if sender in self.blocklist:
        return False
    return None  # no opinion
```

## Tool registration

Tools are async functions with type-annotated parameters. The docstring's first line becomes the tool description for the LLM. Register them in `register_tools`:

```python
from corvidae.hooks import hookimpl
from corvidae.tool import Tool

class WeatherPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def get_weather(city: str, units: str = "metric") -> str:
            """Get current weather for a city."""
            return await fetch_weather(city, units)

        tool_registry.append(Tool.from_function(get_weather))
```

`Tool.from_function()` infers the tool name from `fn.__name__` and generates a JSON schema from the type annotations. Parameters prefixed with `_` are excluded from the schema — they are injected at call time, not supplied by the LLM.

## Context injection (ToolContext)

Tools that need channel context, the task queue, or the tool call ID declare a `_ctx: ToolContext` parameter. Corvidae injects it automatically; the LLM never sees it.

```python
from corvidae.tool import Tool, ToolContext

class MyPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def enqueue_work(instructions: str, _ctx: ToolContext) -> str:
            """Enqueue background work."""
            from corvidae.task import Task

            async def work():
                return await do_something(instructions)

            task = Task(
                work=work,
                channel=_ctx.channel,
                tool_call_id=_ctx.tool_call_id,
                description=instructions[:80],
            )
            await _ctx.task_queue.enqueue(task)
            return f"Enqueued task {task.task_id}"

        tool_registry.append(Tool.from_function(enqueue_work))
```

`ToolContext` attributes:
- `channel: Channel | None` — the channel this tool call is executing on
- `tool_call_id: str` — the LLM-assigned call ID for this invocation
- `task_queue: TaskQueue | None` — the task queue (None if TaskPlugin is not registered)

## Plugin dependencies

Declare `depends_on` as a class attribute (a set of plugin names). `validate_dependencies()` raises `RuntimeError` at startup if any declared dependency is not registered, or if the dependency graph contains a cycle. The error message includes the full cycle path, e.g. `Dependency cycle detected: a -> b -> a`.

```python
class MyPlugin:
    depends_on = {"agent_loop", "task"}

    def __init__(self, pm) -> None:
        self.pm = pm
```

To get a typed reference to a dependency:

```python
from corvidae.hooks import get_dependency
from corvidae.agent import AgentPlugin

agent = get_dependency(self.pm, "agent_loop", AgentPlugin)
registry = agent.tool_registry
```

`get_dependency` raises `RuntimeError` if the plugin is not found and `TypeError` if it is the wrong type.

## Injecting context before agent turns

`before_agent_turn` fires before every LLM call. Use it to inject contextual information (memory retrieval, current state, etc.) into the conversation:

```python
from corvidae.conversation import MessageType

class MemoryPlugin:
    @hookimpl
    async def before_agent_turn(self, channel) -> None:
        notes = await self.fetch_relevant_notes(channel.id)
        if notes:
            channel.conversation.append(
                {"role": "user", "content": f"[Context]\n{notes}"},
                message_type=MessageType.CONTEXT,
            )
```

`MessageType.CONTEXT` entries survive compaction — compaction only summarizes `MESSAGE` entries.

## Channels

A channel is a `transport:scope` pair like `irc:#general` or `cli:local`. Each channel has its own `ConversationLog` with SQLite persistence.

```python
channel.id              # "irc:#general"
channel.transport       # "irc"
channel.scope           # "#general"
channel.conversation    # ConversationLog
channel.matches_transport("irc")  # True
```

Channels are created on-demand when messages arrive, or pre-registered in `agent.yaml`:

```yaml
channels:
  irc:#general:
    system_prompt: "You are the channel bot."
    max_context_tokens: 16000
```

**IRC transport config:**

```yaml
irc:
  host: irc.libera.chat       # IRC server (default irc.libera.chat)
  port: 6667                   # IRC port (default 6667)
  nick: corvidae               # Bot nickname (default corvidae)
  tls: false                   # Use TLS (default false)
  channels: ["#general"]       # Channels to join
  message_chunk_size: 400      # Max UTF-8 bytes per IRC message (default 400)
```

## Registration order

The current registration sequence in `main.py`:

```
registry       (ChannelRegistry)
persistence    (PersistencePlugin)
core_tools     (CoreToolsPlugin)
cli            (CLIPlugin)
irc            (IRCPlugin)
task           (TaskPlugin)
subagent       (SubagentPlugin)
mcp            (McpClientPlugin)
compaction     (CompactionPlugin)
thinking       (ThinkingPlugin)
agent_loop     (AgentPlugin)
idle_monitor   (IdleMonitorPlugin)
```

Tool-providing plugins and transport plugins register before `agent_loop`. The `agent_loop` plugin collects tools during `on_start`, so anything appending to `tool_registry` must be registered first.

**Startup order:** `main.py` calls `pm.ahook.on_start(config=config)` first, which runs all plugins' `on_start` hooks concurrently via `asyncio.gather`. Then it calls `agent_loop.on_start(config=config)` explicitly. This guarantees that all plugins (including `McpClientPlugin`) have completed initialization before `AgentPlugin` collects tools via `register_tools`. `AgentPlugin.on_start` does not have `@hookimpl` — it is called only by `main.py`.

**Shutdown order:** `main.py` calls `agent_loop.on_stop()` first (drains queues, closes LLM client), then `pm.ahook.on_stop()` to tear down all other plugins.

`idle_monitor` registers after `agent_loop` because it depends on `"agent_loop"`. Its `on_start` uses `@hookimpl(trylast=True)` to run late in the broadcast. Because `AgentPlugin.on_start` is called after the broadcast completes, `idle_monitor` is always initialized before `AgentPlugin` starts.

## Hook exception safety

Broadcast hook calls from `AgentPlugin` are wrapped in `try/except`. If a plugin raises an exception from `before_agent_turn`, `after_persist_assistant`, `on_agent_response`, or `send_message`, the exception is logged at WARNING or ERROR level and processing continues. Plugins do not need to catch their own exceptions to protect the queue consumer.

Hooks using `resolve_hook_results` are not wrapped — exceptions propagate to the call site. The exception is `compact_conversation`, whose broadcast invocation is wrapped: a compaction failure is logged at WARNING and the turn continues without compaction.

## Async considerations

All broadcast hooks are `async`. Corvidae uses apluggy's `pm.ahook.*` for async dispatch.

For hooks that return a value, call `pm.ahook.<hook>(...)` and pass the result list to `resolve_hook_results` with the appropriate strategy:

```python
from corvidae.hooks import resolve_hook_results, HookStrategy

results = await pm.ahook.some_hook(channel=channel, ...)
result = resolve_hook_results(results, "some_hook", HookStrategy.VALUE_FIRST, pm=pm)
```

`@hookimpl(tryfirst=True)` and `@hookimpl(trylast=True)` markers are respected by apluggy's broadcast dispatch and affect the order in which results are collected.

## Stock tools

These tools are registered by built-in plugins. They are available to the LLM in every standard Corvidae deployment.

All tool results are truncated at `MAX_TOOL_RESULT_CHARS` (default 100,000 characters) by `execute_tool_call` in `corvidae/tool.py`. The truncation appends `[truncated — N chars total]` so the LLM knows output was cut. Override via config:

```yaml
agent:
  max_tool_result_chars: 100000  # read by AgentPlugin; SubagentPlugin reads it from AgentPlugin at launch time
```

### CoreToolsPlugin tools

Registered by `CoreToolsPlugin` (registered as `core_tools` in `main.py`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `shell` | `command: str` | Runs a shell command and returns combined stdout/stderr. Times out after `tools.shell_timeout` seconds (default 30). Returns `"(no output)"` if the command produces none. Non-zero exit codes are appended to the output. |
| `read_file` | `path: str` | Reads a file and returns its text content. Returns an error string for missing files, directories, unreadable files, or files larger than `tools.max_file_read_bytes` bytes (default 1 MB). |
| `write_file` | `path: str`, `content: str` | Writes `content` to `path`, creating parent directories as needed. Returns a confirmation with the byte count, or an error string on failure. |
| `web_fetch` | `url: str` | Fetches a URL via HTTP GET and returns the response body as text. Times out after `tools.web_fetch_timeout` seconds (default 15). Truncates responses at `tools.web_max_response_bytes` characters (default 50,000, independent of `MAX_TOOL_RESULT_CHARS`). |
| `web_search` | `query: str`, `max_results: int` (optional) | Searches the web via DuckDuckGo and returns formatted results with titles, URLs, and snippets. Defaults to 8 results per page. |

**CoreToolsPlugin config:**

```yaml
tools:
  shell_timeout: 30              # seconds before shell command is killed
  web_fetch_timeout: 15          # seconds before web request is aborted
  web_max_response_bytes: 50000  # response body truncation limit
  max_file_read_bytes: 1048576   # file size limit (1 MB)
```

### SubagentPlugin tools

Registered by `SubagentPlugin` (registered as `subagent` in `main.py`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `subagent` | `instructions: str`, `description: str`, `_ctx: ToolContext` | Enqueues a background task that runs a full agent loop with `instructions` as its prompt. Returns immediately with the enqueued task ID. Results are delivered via `on_notify` when the task completes. |

`subagent` requires `ToolContext` (injected automatically — not an LLM parameter). It excludes itself from the subagent's tool registry to prevent recursion. The subagent uses the `llm.background` config block if present, falling back to `llm.main`.

## Bundled plugins

These plugins are registered by `main.py` and are part of the default
daemon. They can be omitted if the application does not need their
functionality; each degrades gracefully when absent.

### McpClientPlugin (`corvidae/mcp_client.py`)

Connects to external [MCP](https://modelcontextprotocol.io/) servers and
exposes their tools to the agent loop. Registered as `"mcp"` before
`agent_loop`.

Implements three hooks:

- `on_start` — connects to all servers listed under `mcp.servers`
  in `agent.yaml`, fetches their tool lists, and builds cached `Tool` instances.
  Runs in the broadcast; completes before `AgentPlugin.on_start` calls
  `register_tools`.
- `register_tools` — appends the cached tools to `tool_registry`.
- `on_stop` — closes all MCP sessions and transports via `AsyncExitStack`.

**Config:**
```yaml
mcp:
  servers:
    filesystem:
      transport: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
      env:                     # optional: extra env vars for the subprocess
        NODE_ENV: production
      tool_prefix: "fs"        # optional: prefix for tool names; defaults to server name
      timeout_seconds: 30      # optional: per-call timeout; default 30

    remote_api:
      transport: sse
      url: "http://localhost:8001/sse"
      timeout_seconds: 60
```

Tool names are prefixed to avoid collisions: a server named `filesystem`
produces tools like `filesystem__read_file`. Set `tool_prefix: ""` to
disable prefixing. If two servers produce the same tool name, the first
wins and the duplicate is skipped with a WARNING.

**Without this plugin:** MCP servers cannot be used. No tools are affected
if the `mcp:` config key is absent — the plugin no-ops.

### ThinkingPlugin (`corvidae/thinking.py`)

Strips `<think>...</think>` blocks and `reasoning_content` from LLM
output. Registered as `"thinking"` before `agent_loop`.

Implements two hooks:

- `after_persist_assistant` — reads `keep_thinking_in_history` from the
  resolved channel config. If `False`, calls `strip_reasoning_content`
  on the in-memory message dict. The DB copy is already written; this
  only affects subsequent prompt builds.
- `transform_display_text` — calls `strip_thinking` on the response
  text. Returns the stripped string if it differs from the input, or
  `None` if no `<think>` tags were present.

**Without this plugin:** `<think>` blocks pass through to the channel
verbatim, and `reasoning_content` remains in in-memory history
regardless of the `keep_thinking_in_history` config value.

### CompactionPlugin (`corvidae/compaction.py`)

Compacts conversation history when it approaches the channel's
`max_context_tokens` limit. Registered as `"compaction"` before `agent_loop`.

Implements one hook:

- `compact_conversation` — fires before each LLM call. Checks whether the
  token estimate exceeds `compaction_threshold * max_tokens`. If so, and if
  the conversation has more than `min_messages_to_compact` messages,
  summarizes older messages to fit within `compaction_retention * max_tokens`.
  Returns `True` when compaction ran; `None` when the threshold was not met.

Token estimation divides total character count by `chars_per_token`. The same
`chars_per_token` value must be used when constructing `ConversationLog`
instances (done by `PersistencePlugin`); configure both via `agent.chars_per_token`.

**Config:**
```yaml
agent:
  compaction_threshold: 0.8      # compact when token estimate exceeds this fraction of max_context_tokens
  compaction_retention: 0.5      # retain this fraction of max_context_tokens after compaction
  min_messages_to_compact: 5     # skip compaction if conversation has this many messages or fewer
  chars_per_token: 3.5           # character-to-token ratio used for token estimation
```

**Without this plugin:** conversations grow without bound. The LLM will
receive an error from the API when the context limit is exceeded.

### PersistencePlugin (`corvidae/persistence.py`)

Opens the SQLite database, runs schema migrations via `init_db`, and sets
the journal mode. Registered as `"persistence"` before `agent_loop`.
Implements `on_start`, `on_stop`, and `ensure_conversation`.

**Config:**
```yaml
daemon:
  session_db: sessions.db       # path to SQLite database file (default "sessions.db")
  sqlite_journal_mode: wal      # SQLite journal mode (default "wal");
                                # allowed values: delete, truncate, persist, memory, wal, off
```

**Without this plugin:** `ensure_conversation` is never fulfilled; `AgentPlugin`
logs an error and drops every message.

### TaskPlugin (`corvidae/task.py`)

Owns the `TaskQueue` and delivers task results via the `on_notify` hook.
Registered as `"task"` before `agent_loop`.

**Config:**
```yaml
daemon:
  max_task_workers: 4       # concurrent task workers (default 4)
  completed_task_buffer: 100  # number of completed task records to retain in memory
```

**Without this plugin:** tool calls that return asynchronously (e.g., `subagent`)
cannot complete. `AgentPlugin` logs an error when tool dispatch is attempted
without a `TaskQueue`.

### IdleMonitorPlugin (`corvidae/idle.py`)

Fires the `on_idle` broadcast hook when all queues are quiescent.
Registered as `"idle_monitor"` after `agent_loop`.

Depends on `"agent_loop"`. Its `on_start` uses `@hookimpl(trylast=True)`
to run late in the broadcast. `AgentPlugin.on_start` is called by
`main.py` after the broadcast completes, so the idle monitor is always
initialized before the agent starts. It
retrieves `AgentPlugin.queues` (a `dict[str, SerialQueue]`) by reference,
so queues created after `IdleMonitorPlugin.on_start` are included
automatically.

**Idle condition:** all `SerialQueue` instances have `is_empty=True`,
`TaskQueue.is_idle` is `True` (skipped if `TaskPlugin` is not
registered), and at least `idle_cooldown_seconds` have elapsed since
the last firing.

**Config:**
```yaml
daemon:
  idle_cooldown_seconds: 30   # minimum seconds between on_idle firings (default 30)
  idle_poll_interval: 2       # seconds between idle checks (default 2)
```

**Without this plugin:** the `on_idle` hook is never fired.
