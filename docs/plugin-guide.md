# Corvidae Plugin Guide

Corvidae plugins extend the agent daemon using [apluggy](https://pypi.org/project/apluggy/) (async pluggy). A plugin subclasses `CorvidaePlugin` and decorates methods with `@hookimpl` for each hook it handles.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl

class GreetPlugin(CorvidaePlugin):
    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        if text.strip().lower() == "hello":
            await self.pm.ahook.send_message(channel=channel, text=f"Hello, {sender}!")
```

`CorvidaePlugin.on_init` stores `pm` and `config` as instance attributes. Subclasses that need extra initialization override `on_init` and call `super().on_init(pm, config)` first.

## Plugin anatomy

A plugin subclasses `CorvidaePlugin` from `corvidae.hooks`. It only needs to implement the hooks it cares about — all hooks are optional. The `@hookimpl` decorator marks each implementation. Plugins use no-argument constructors; `pm` and `config` are received via the `on_init` hook.

```python
from corvidae.hooks import CorvidaePlugin, hookimpl

class MyPlugin(CorvidaePlugin):
    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.setting = config.get("my_plugin", {}).get("setting", "default")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # open runtime resources (connections, file handles, asyncio tasks)
        pass

    @hookimpl
    async def on_stop(self) -> None:
        # clean up resources
        pass
```

## Registering a plugin

Plugins are loaded via setuptools entry points. Declare them in `pyproject.toml`:

```toml
[project.entry-points.corvidae]
my_plugin = "my_package:MyPlugin"
```

`Runtime.start()` calls `pm.load_setuptools_entrypoints("corvidae")`, which instantiates each entry point class with no arguments and registers it. `validate_dependencies(pm)` runs next, then `pm.ahook.on_init(pm=pm, config=config)` broadcasts to all registered plugins.

All plugin classes must be instantiable with no arguments. `pm` and `config` are delivered via `on_init`.

**ChannelRegistry** is the exception: it is not an entry-point plugin. It is a plain class with no `@hookimpl` decorators, constructed explicitly in `Runtime.start()` and populated from config before the entry point plugins are loaded.

## Writing replacement plugins

To replace a built-in hook handler, implement the hook at default priority. Built-in handlers use `trylast=True` and serve as fallbacks.

```python
class FancyCompactionPlugin(CorvidaePlugin):
    @hookimpl
    async def compact_conversation(self, channel, conversation, max_tokens):
        # Custom strategy
        await self.my_compaction(conversation, max_tokens)
        return True  # non-None stops the chain; built-in never runs
```

To observe without replacing, use `tryfirst=True` and return None:

```python
class CompactionObserver(CorvidaePlugin):
    @hookimpl(tryfirst=True)
    async def compact_conversation(self, channel, conversation, max_tokens):
        self.record_metrics(conversation)
        return None  # chain continues to the actual compaction handler
```

This pattern applies to any `firstresult=True` hook. For broadcast hooks (where all handlers run regardless), `tryfirst` and `trylast` affect only execution order, not whether subsequent handlers run.

## Registering CLI subcommands

Plugins can add subcommands to the `corvidae` CLI by registering entries in the `corvidae.commands` entry point group. Each entry point maps a subcommand name to a `click.Command` (or `click.Group` for nested subcommands).

```toml
[project.entry-points."corvidae.commands"]
scaffold = "corvidae_scaffold:scaffold_command"
```

The entry point value must be a `click.Command` or `click.Group` object. `corvidae.main` discovers all registered entries at import time and adds them to the top-level `corvidae` group. If an entry point fails to load, a warning is logged and the subcommand is skipped — other subcommands are unaffected.

Subcommands are standard click commands managed by the `corvidae` click Group. Subcommands that need the agent runtime construct a `Runtime` instance and call `asyncio.run(runtime.run())`:

```python
import asyncio
import click
from corvidae.runtime import Runtime

@click.command("myplugin")
@click.option("--config", default="agent.yaml", help="Path to config file")
def myplugin_command(config):
    """Start corvidae with myplugin active."""
    runtime = Runtime(
        config_path=config,
        overrides={"channels": {"cli:local": {}}},
    )
    asyncio.run(runtime.run())
```

Utility subcommands that do not need the agent runtime omit the `Runtime` import entirely. They can perform their work synchronously without booting the plugin system.

Entry point modules in `corvidae.commands` are imported at dispatcher load time, before `Runtime.start()` configures logging or creates an event loop. These modules must not have module-level side effects that depend on logging configuration or a running event loop. Standard library imports (`asyncio`, `click`) and `corvidae.runtime` are safe at module level. Avoid importing modules with side effects (network connections, file I/O, logging configuration) at the top level — use lazy imports inside the command function for those.

The existing `corvidae` entry point group (for plugins) and the `corvidae.commands` group (for subcommands) are distinct setuptools groups. Registering in one does not affect the other.

The built-in subcommand `scaffold` is registered by corvidae itself under `corvidae.commands`. Do not register an entry point named `scaffold` in the `corvidae.commands` group — a collision will silently overwrite the built-in command, causing unpredictable behavior.

## Available hooks

### Lifecycle

| Hook | Type | When |
|------|------|------|
| `on_init(pm, config: dict)` | async broadcast | After all plugins are registered, before `on_start`. Use to store `pm`, read config values, and resolve references to other plugins. Do not create runtime resources here. |
| `on_start(config: dict)` | async broadcast | Once at startup, after `on_init`. Use to open runtime resources: DB connections, network clients, file handles, asyncio tasks. |
| `on_stop()` | async broadcast | On SIGINT/SIGTERM, before process exits |

`config` is the full parsed `agent.yaml` dict. The key `_base_dir` (a `Path`) is set by `Runtime.start()` pointing to the config file's directory.

`CorvidaePlugin.on_init` stores `pm` and `config` as instance attributes. Subclasses that override `on_init` must call `await super().on_init(pm, config)` to preserve this behavior.

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
| `on_conversation_event(channel, message: dict, message_type: MessageType)` | async broadcast | After every `conv.append()` call; message is the untagged dict (no `_message_type` key) |
| `on_compaction(channel, summary_msg: dict, retain_count: int)` | async broadcast | After compaction replaces older messages with a summary; `summary_msg` is the untagged summary dict |
| `on_idle()` | async broadcast | All queues empty and cooldown elapsed; only fires when `IdleMonitorPlugin` is registered |

`after_persist_assistant` — the DB row is already written when this
hook fires. Mutations to `message` affect in-memory prompt construction
only; they do not update the persisted record.

`before_agent_turn` — messages injected via `channel.conversation.append()` inside this hook are passed through `on_conversation_event` and persisted to the DB.

### Hook result resolution

These hooks return a value. Hooks marked `firstresult=True` (sequential) stop at the first non-None return. Wrapper chain hooks use `firstresult=True` with an identity seed; implementations use `@hookimpl(wrapper=True)` to compose transforms. Broadcast hooks use `resolve_hook_results` for result resolution.

| Hook | Strategy | Returns | Behavior |
|------|----------|---------|----------|
| `should_process_message(channel, sender, text)` | `REJECT_WINS` (broadcast) | `bool \| None` | Any `False` vetoes the message; any `True` (with no `False`) accepts; `None` if all defer |
| `on_llm_error(channel, error)` | `firstresult=True` | `str \| None` | First non-None string wins; chain stops. If all return None, default error message is used. |
| `compact_conversation(channel, conversation, max_tokens)` | `firstresult=True` | `bool \| None` | First non-None return stops the chain. `ContextCompactPlugin` uses `tryfirst` (returns None). `CompactionPlugin` uses `trylast` (returns True). Third-party plugins run at default priority. |
| `process_tool_result(tool_name, result, channel)` | `firstresult=True` wrapper chain | `str \| None` | Wrappers compose transforms in LIFO order. The seed returns the input unchanged. Non-wrapper hookimpls short-circuit the seed via firstresult. |
| `transform_display_text(channel, text, result_message)` | `firstresult=True` wrapper chain | `str \| None` | Wrappers compose transforms in LIFO order. The seed returns the input text unchanged. Non-wrapper hookimpls short-circuit the seed via firstresult. |
| `load_conversation(channel)` | `firstresult=True` | `list[dict] \| None` | First non-None result wins. `PersistencePlugin` uses `trylast` as fallback. Called once when a channel's conversation is first initialized. |

`compact_conversation` — sequential hook (`firstresult=True`). The first handler returning a non-None value stops the chain. `ContextCompactPlugin` runs first (`tryfirst=True`) to generate background blocks and returns None (does not stop the chain). `CompactionPlugin` runs last (`trylast=True`) as the default strategy and returns True. A third-party plugin at default priority runs between them — if it returns non-None, `CompactionPlugin` is never called.

`process_tool_result` fires for all tool calls — it is invoked from `dispatch_tool_call` in `corvidae/tool.py`, which is used by both the main agent loop and background subagent loops. It does not fire for pre-dispatch errors (JSON parse failure or unknown tool name).

`transform_display_text` — `result_message` is the raw assistant message dict from the LLM response. It may contain `reasoning_content` if the model produces thinking tokens. `text` is the string content extracted from that message. Wrapper implementations receive the chain result (the output of inner wrappers or the seed) via `yield`. Non-wrapper implementations that return `None` defer to the seed, which returns `text` unchanged.

Example hook returning a value:

```python
@hookimpl
async def should_process_message(self, channel, sender: str, text: str) -> bool | None:
    if sender in self.blocklist:
        return False
    return None  # no opinion
```

## Tool registration

Tools **must** be coroutine functions (`async def`) with type-annotated parameters. `Tool.from_function()` enforces this at registration time and raises `TypeError` if given a sync callable. The docstring's first line becomes the tool description for the LLM. See [Avoid blocking the event loop](#avoid-blocking-the-event-loop) for what `async def` means in practice — the check rejects sync functions, but it cannot detect blocking I/O called from inside an async one. Register tools in `register_tools`:

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

## Avoid blocking the event loop

Every tool runs on the single main asyncio event loop. A synchronous blocking call inside a tool — `requests.get`, `time.sleep`, a slow `open(...).read()`, a sync database driver — stalls the entire process: the main agent, every task-queue worker, and every channel queue all wait until the call returns.

The `Tool.from_function()` async check rejects functions defined with `def` instead of `async def`, but it cannot see what your `async def` does internally. If your code calls a blocking library, you are responsible for moving that call off the event loop with `asyncio.to_thread()`. The pattern used by the built-in file tools (`corvidae/tools/files.py`) is the model: a small sync helper does the I/O, the async tool awaits it via `to_thread`.

```python
import asyncio
from corvidae.hooks import hookimpl
from corvidae.tool import Tool

def _lookup_sync(symbol: str) -> str:
    # Blocking call — sync HTTP client, sync DB driver, slow file read, etc.
    import requests
    return requests.get(f"https://api.example.com/{symbol}", timeout=5).text

class QuotePlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def lookup(symbol: str) -> str:
            """Look up a stock quote."""
            return await asyncio.to_thread(_lookup_sync, symbol)

        tool_registry.append(Tool.from_function(lookup))
```

For common cases the standard library and ecosystem offer native async alternatives — prefer them when available:

- HTTP — use `aiohttp` (see `corvidae/tools/web.py`).
- Subprocesses — use `asyncio.create_subprocess_shell` / `create_subprocess_exec` (see `corvidae/tools/shell.py`).

As a last-resort safety net, `execute_tool_call` detects a sync callable registered as a bare `Tool(name=..., fn=sync_fn, schema=...)` (bypassing `Tool.from_function()`) and wraps the call in `asyncio.to_thread()`, logging a warning. Treat that warning as a bug to fix in the plugin, not a supported registration path.

## Plugin dependencies

Declare `depends_on` as a class attribute (a set of plugin names). `validate_dependencies()` raises `RuntimeError` at startup if any declared dependency is not registered, or if the dependency graph contains a cycle. The error message includes the full cycle path, e.g. `Dependency cycle detected: a -> b -> a`.

```python
class MyPlugin(CorvidaePlugin):
    depends_on = {"agent", "task"}
```

To get a typed reference to a dependency:

```python
from corvidae.hooks import get_dependency
from corvidae.tool_collection import ToolCollectionPlugin

tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
registry = tools_plugin.get_registry()
```

`get_dependency` raises `RuntimeError` if the plugin is not found and `TypeError` if it is the wrong type.

## Injecting context before agent turns

`before_agent_turn` fires before every LLM call. Use it to inject contextual information (memory retrieval, current state, etc.) into the conversation:

```python
from corvidae.context import MessageType
from corvidae.hooks import CorvidaePlugin, hookimpl

class MemoryPlugin(CorvidaePlugin):
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

A channel is a `transport:scope` pair like `irc:#general` or `cli:local`. Each channel has its own `ContextWindow` initialized on first message.

```python
channel.id              # "irc:#general"
channel.transport       # "irc"
channel.scope           # "#general"
channel.conversation    # ContextWindow | None
channel.matches_transport("irc")  # True
```

Channels are created on-demand when messages arrive, or pre-registered in `agent.yaml`:

```yaml
channels:
  irc:#general:
    system_prompt: "You are the channel bot."
    max_context_tokens: 16000
```

The `irc` transport is provided by `IRCPlugin` (registered via the `corvidae` entry point group). Its config block:

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

Plugins are loaded via `pm.load_setuptools_entrypoints("corvidae")`. The entry point loading order is non-deterministic — it is not guaranteed to match any specific sequence. The following plugins are registered:

```
registry          (ChannelRegistry)     — explicit, before entry points load
persistence       (PersistencePlugin)   — entry point
jsonl_log         (JsonlLogPlugin)      — entry point
core_tools        (CoreToolsPlugin)     — entry point
cli               (CLIPlugin)           — entry point
irc               (IRCPlugin)           — entry point
task              (TaskPlugin)          — entry point
subagent          (SubagentPlugin)      — entry point
mcp               (McpClientPlugin)     — entry point
llm               (LLMPlugin)           — entry point
compaction        (CompactionPlugin)    — entry point
context_compact   (ContextCompactPlugin) — entry point
thinking          (ThinkingPlugin)      — entry point
runtime_settings  (RuntimeSettingsPlugin) — entry point
tools             (ToolCollectionPlugin) — entry point
dream             (DreamPlugin)         — entry point
agent             (Agent)               — entry point
idle_monitor      (IdleMonitorPlugin)   — entry point
```

`ChannelRegistry` is constructed and registered explicitly before entry points load. Its `agent_defaults` attribute and channel config are populated from config before the `on_init` broadcast runs.

**Tool collection:** `ToolCollectionPlugin.on_start` uses `@hookimpl(trylast=True)` so it fires after all other `on_start` hooks, then calls `register_tools` to collect tools from every registered plugin. The non-deterministic loading order does not affect tool collection because `trylast=True` ensures `ToolCollectionPlugin.on_start` always runs last in the broadcast.

**Startup order:** `Runtime.start()` broadcasts `on_init`, then `on_start`. `ToolCollectionPlugin.on_start` runs last (trylast) to collect tools. `Runtime.start()` then calls `agent.on_start(config=config)` explicitly after the broadcast completes. `Agent.on_start` does not have `@hookimpl` — it is called only by `Runtime.start()`. Adding `@hookimpl` to `Agent.on_start` would cause double initialization.

**Shutdown order:** `Runtime.stop()` calls `agent.on_stop()` first (drains queues), then `pm.ahook.on_stop()` to tear down all other plugins.

`idle_monitor` depends on `"agent"`. Its `on_start` uses `@hookimpl(trylast=True)` to run late in the broadcast. Because `Agent.on_start` is called after the broadcast completes, `idle_monitor` is always initialized before `Agent` starts.

## Plugin disable

Disable entry-point plugins via the `plugins.disabled` config key:

```yaml
plugins:
  disabled:
    - compaction          # disable the built-in compaction plugin
```

Names must match the entry-point name from `[project.entry-points.corvidae]` in `pyproject.toml`. `Runtime.start()` calls `pm.set_blocked(name)` for each entry before loading entry points. Blocked plugins are not instantiated.

Only entry-point plugins can be disabled this way. Manually registered plugins (e.g., `ChannelRegistry`) cannot be blocked via config.

This is a startup-time mechanism. It does not interact with hot-reload.

## Hook exception safety

Broadcast hook calls from `Agent` are wrapped in `try/except`. If a plugin raises an exception from `before_agent_turn`, `after_persist_assistant`, `on_agent_response`, or `send_message`, the exception is logged at WARNING or ERROR level and processing continues. Plugins do not need to catch their own exceptions to protect the queue consumer.

For `firstresult=True` hooks (`compact_conversation`, `load_conversation`, `on_llm_error`), pluggy propagates the first handler's exception immediately — the chain stops and no further handlers run. The call sites in `Agent` handle these as follows:

- `compact_conversation` — wrapped in try/except; failure is logged at WARNING and the turn continues without compaction.
- `load_conversation` — not wrapped; an exception propagates to `_process_queue_item`, which will fail the queue item.
- `on_llm_error` — called inside the existing try/except in `_run_turn`; an exception from the hook itself is not separately caught (it propagates up from `_run_turn`).

For `should_process_message` (broadcast with `resolve_hook_results`), exceptions propagate to the call site.

For `transform_display_text` and `process_tool_result` (wrapper chain hooks), pluggy propagates exceptions from wrapper implementations to the call site. `transform_display_text` is wrapped in try/except in `_resolve_display_text`.

## Async considerations

All broadcast hooks are `async`. Corvidae uses apluggy's `pm.ahook.*` for async dispatch.

For hooks with `firstresult=True` (`compact_conversation`, `load_conversation`, `on_llm_error`), `pm.ahook.<hook>(...)` returns a single value (or None) directly:

```python
result = await pm.ahook.load_conversation(channel=channel)
```

For `transform_display_text` and `process_tool_result`, use `@hookimpl(wrapper=True)` to participate in the wrapper chain. The seed plugin returns the input value unchanged as the innermost result; each wrapper receives that result, optionally transforms it, and returns the modified value:

```python
from corvidae.hooks import hookimpl

@hookimpl(wrapper=True)
def transform_display_text(self, **kwargs):
    result = yield
    if result is not None:
        return my_transform(result)
    return result
```

For the `should_process_message` broadcast hook, call `pm.ahook.<hook>(...)` and pass the result list to `resolve_hook_results`:

```python
from corvidae.hooks import resolve_hook_results, HookStrategy

results = await pm.ahook.should_process_message(channel=channel, ...)
result = resolve_hook_results(results, "should_process_message", HookStrategy.REJECT_WINS)
```

`@hookimpl(tryfirst=True)` and `@hookimpl(trylast=True)` markers are respected by apluggy's dispatch and affect execution order. For `firstresult=True` hooks, `tryfirst` handlers run before default-priority handlers, which run before `trylast` handlers. The chain stops at the first non-None return.

The async-only rule extends to **tool functions** as well as hooks: `Tool.from_function()` enforces it at registration time, and any blocking I/O inside a tool must be wrapped with `asyncio.to_thread()`. See [Avoid blocking the event loop](#avoid-blocking-the-event-loop).

## Stock tools

These tools are registered by built-in plugins. They are available to the LLM in every standard Corvidae deployment.

All tool results are truncated at `MAX_TOOL_RESULT_CHARS` (default 100,000 characters) by `execute_tool_call` in `corvidae/tool.py`. The truncation appends `[truncated — N chars total]` so the LLM knows output was cut. Override via config:

```yaml
tools:
  max_result_chars: 100000  # read by ToolCollectionPlugin during on_start
```

The legacy key `agent.max_tool_result_chars` is still accepted but deprecated; a warning is logged at startup when it is present.

### CoreToolsPlugin tools

Registered by `CoreToolsPlugin` (entry point name: `core_tools`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `shell` | `command: str` | Runs a shell command and returns combined stdout/stderr. Times out after `tools.shell_timeout` seconds (default 30). Returns `"(no output)"` if the command produces none. Non-zero exit codes are appended to the output. |
| `read_file` | `path: str` | Reads a file and returns its text content. Returns an error string for missing files, directories, unreadable files, or files larger than `tools.max_file_read_bytes` bytes (default 1 MB). |
| `write_file` | `path: str`, `content: str` | Writes `content` to `path`, creating parent directories as needed. Returns a confirmation with the byte count, or an error string on failure. |
| `web_fetch` | `url: str` | Fetches a URL via HTTP GET and returns the response body as text. Times out after `tools.web_fetch_timeout` seconds (default 15). Truncates responses at `tools.web_max_response_bytes` characters (default 50,000, independent of `MAX_TOOL_RESULT_CHARS`). |
| `web_search` | `query: str`, `max_results: int` (optional) | Searches the web via DuckDuckGo and returns formatted results with titles, URLs, and snippets. Defaults to 8 results per page. |
| `task_pipeline` | `definition: str` | Executes a task graph defined in YAML or JSON. The definition must contain a `tasks` key with a list of objects, each having `name`, `command`, and optionally `depends_on`. Tasks run in topological order; failed tasks block their dependents. Returns a status summary. |

**CoreToolsPlugin config:**

```yaml
tools:
  shell_timeout: 30                  # seconds before shell command is killed
  web_fetch_timeout: 15              # seconds before web request is aborted
  web_max_response_bytes: 50000      # response body truncation limit
  web_search_max_results: 8          # max results returned per search query
  max_file_read_bytes: 1048576   # file size limit (1 MB)
```

### SubagentPlugin tools

Registered by `SubagentPlugin` (entry point name: `subagent`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `subagent` | `instructions: str`, `description: str`, `_ctx: ToolContext` | Enqueues a background task that runs a full agent loop with `instructions` as its prompt. Returns immediately with the enqueued task ID. Results are delivered via `on_notify` when the task completes. |

`subagent` requires `ToolContext` (injected automatically — not an LLM parameter). It excludes itself from the subagent's tool registry to prevent recursion. The subagent uses the `llm.background` config block if present, falling back to `llm.main`.

### RuntimeSettingsPlugin tools

Registered by `RuntimeSettingsPlugin` (entry point name: `runtime_settings`).

| Tool | Parameters | What it does |
|------|------------|--------------|
| `set_settings` | `settings: dict` | Updates per-channel runtime settings. Accepts LLM inference parameters and framework parameters. Pass `null` for a key to revert it to the static config value. Returns the current overrides after the update. |

## Bundled plugins

These plugins ship with Corvidae and are registered via entry points as part of the default daemon. They can be omitted if the application does not need their functionality; each degrades gracefully when absent.

### McpClientPlugin (`corvidae/mcp_client.py`)

Connects to external [MCP](https://modelcontextprotocol.io/) servers and
exposes their tools to the agent loop. Entry point name: `"mcp"`.

Implements three hooks:

- `on_start` — connects to all servers listed under `mcp.servers`
  in `agent.yaml`, fetches their tool lists, and builds cached `Tool` instances.
  Runs in the broadcast; completes before `ToolCollectionPlugin.on_start`
  (trylast) calls `register_tools`.
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
output. Entry point name: `"thinking"`.

Implements two hooks:

- `after_persist_assistant` — reads `keep_thinking_in_history` from the
  resolved channel config. If `False`, calls `strip_reasoning_content`
  on the in-memory message dict. The DB copy is already written; this
  only affects subsequent prompt builds.
- `transform_display_text` (`@hookimpl(wrapper=True)`) — receives the
  chain result and calls `strip_thinking` on it unconditionally. Returns
  the stripped string. Operates as a sync wrapper; does not take
  `channel`, `text`, or `result_message` parameters directly.

**Without this plugin:** `<think>` blocks pass through to the channel
verbatim, and `reasoning_content` remains in in-memory history
regardless of the `keep_thinking_in_history` config value.

### CompactionPlugin (`corvidae/compaction.py`)

Compacts conversation history when it approaches the channel's
`max_context_tokens` limit. Entry point name: `"compaction"`.

Implements one hook:

- `compact_conversation` (`trylast=True`) — fires before each LLM call. Checks whether the token estimate exceeds `compaction_threshold * max_tokens`. If so, and if the conversation has more than `min_messages_to_compact` messages, summarizes older messages to fit within `compaction_retention * max_tokens`. Returns True to stop the `firstresult` chain. Runs after `ContextCompactPlugin` (tryfirst) and any third-party handlers at default priority.

Token estimation divides total character count by `chars_per_token`. The same
`chars_per_token` value is used when constructing `ContextWindow` instances
(done by `Agent`); configure via `agent.chars_per_token`.

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

### ContextCompactPlugin (`corvidae/context_compact.py`)

Generates persistent background blocks from older conversation segments and injects them before each agent turn. Background blocks capture summarized context from messages older than the most recent compaction boundary, preserving historical knowledge across turns without growing the foreground context window. Entry point name: `"context_compact"`.

Implements five hooks:

- `on_start` — loads config from `agent.context_compact`.
- `register_tools` — registers the `context_stats` tool for observability (turn counts, last block timestamps per channel).
- `compact_conversation` (`tryfirst=True`) — generates background blocks from pre-compaction messages. Returns None (does not stop the chain; `CompactionPlugin` runs after).
- `before_agent_turn` — injects the most recent background block as a CONTEXT entry if none exists in the current conversation.
- `on_agent_response` — tracks per-channel turn counts.

**Config:**
```yaml
agent:
  context_compact:
    enabled: true
    bg_block_threshold: 20        # generate a block after this many turns
    bg_compaction_threshold: 0.75 # compact when token budget exceeds this fraction
    min_background_blocks: 1      # minimum blocks to retain in prompt context
    max_background_block_chars: 2048  # max characters per background block
```

`context_compact` depends on `compaction` and `llm` (declared via `depends_on`). `validate_dependencies()` raises a `RuntimeError` at startup if either is absent.

**Without this plugin:** no background blocks are generated. Compaction still works via `CompactionPlugin`.

### PersistencePlugin (`corvidae/persistence.py`)

Opens the SQLite database, runs schema migrations, and sets the journal mode.
Entry point name: `"persistence"`. Implements `on_start`,
`on_stop`, `load_conversation`, `on_conversation_event`, and `on_compaction`.

**Config:**
```yaml
daemon:
  session_db: sessions.db       # path to SQLite database file (default "sessions.db")
  sqlite_journal_mode: wal      # SQLite journal mode (default "wal");
                                # allowed values: delete, truncate, persist, memory, wal, off
```

**Without this plugin:** `load_conversation` returns no history; conversation
history is not persisted across restarts.

### JsonlLogPlugin (`corvidae/jsonl_log.py`)

Writes an append-only JSONL log of conversation events alongside the SQLite
store. Each `on_conversation_event` and `on_compaction` call produces one
JSON line in a per-channel file. Entry point name: `"jsonl_log"`.

Implements three hooks:

- `on_start` — reads `daemon.jsonl_log_dir` and creates the directory.
  If the key is absent the plugin is a no-op for the rest of its lifetime.
- `on_conversation_event` — writes a record with fields `ts`, `channel`,
  `type` (message type string), and `message` (the untagged message dict).
- `on_compaction` — writes a record with `type: "summary"` and the untagged
  summary dict.

File names are derived from the channel ID with `/` and `:` replaced by `_`.
File handles are opened in append mode and kept open until `on_stop`.

**Config:**
```yaml
daemon:
  jsonl_log_dir: logs/   # path relative to the config file; omit to disable
```

**Without this plugin:** no JSONL log is written. Conversation history is
still persisted in SQLite by `PersistencePlugin`.

### LLMPlugin (`corvidae/llm_plugin.py`)

Owns the `LLMClient` instance lifecycle. Entry point name: `"llm"`. Other plugins retrieve clients via
`get_dependency(pm, "llm", LLMPlugin)`.

Implements two hooks:

- `on_start` — reads `llm.main` (required) and `llm.background` (optional),
  creates `LLMClient` instances, and starts their aiohttp sessions.
- `on_stop` — closes all aiohttp sessions.

`get_client(role)` returns the client for `"main"` or `"background"`. If no
background client is configured, `get_client("background")` falls back to
the main client.

**Config:**
```yaml
llm:
  main:                        # required
    base_url: https://api.openai.com/v1
    model: gpt-4o
    api_key: sk-...            # optional; can also use environment variable
    extra_body: {}             # optional: extra fields merged into request body
    max_retries: 3             # optional (default 3)
    retry_base_delay: 2.0      # optional (default 2.0)
    retry_max_delay: 60.0      # optional (default 60.0)
    timeout: 120               # optional: request timeout in seconds
  background:                  # optional — absent means use llm.main
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
```

**Without this plugin:** `Agent` and `CompactionPlugin` cannot create LLM
clients and will raise `RuntimeError` during startup.

### TaskPlugin (`corvidae/task.py`)

Owns the `TaskQueue` and delivers task results via the `on_notify` hook.
Entry point name: `"task"`.

**Config:**
```yaml
daemon:
  max_task_workers: 4       # concurrent task workers (default 4)
  completed_task_buffer: 100  # number of completed task records to retain in memory
```

**Without this plugin:** tool calls that return asynchronously (e.g., `subagent`)
cannot complete. `Agent` logs an error when tool dispatch is attempted
without a `TaskQueue`.

### RuntimeSettingsPlugin (`corvidae/tools/settings.py`)

Registers the `set_settings` tool, which lets the agent update per-channel
LLM inference parameters and framework settings at runtime. Entry point name: `"runtime_settings"`.

Implements one hook:

- `register_tools` — appends the `set_settings` tool to `tool_registry`.

The `set_settings` tool accepts a `settings` dict. Supported keys include
LLM inference parameters (`temperature`, `top_p`, `top_k`,
`frequency_penalty`, `presence_penalty`, `max_tokens`) and framework
parameters (`max_turns`, `max_context_tokens`, `keep_thinking_in_history`).
Pass `null` for a key to clear the override and revert to the static config
value. `system_prompt` is always blocked. Additional keys can be blocked via
`agent.immutable_settings` in `agent.yaml`.

**Config:**
```yaml
agent:
  immutable_settings: [temperature]   # keys the agent cannot change at runtime
```

**Without this plugin:** the `set_settings` tool is not available. Channel
settings can only be configured statically in `agent.yaml`.

### ToolCollectionPlugin (`corvidae/tool_collection.py`)

Collects tools from all registered plugins and owns the `ToolRegistry`.
Entry point name: `"tools"`.

Implements one hook:

- `on_start` (trylast=True) — calls the sync `register_tools` broadcast after
  all other `on_start` hooks have completed. Builds a `ToolRegistry` from the
  collected items. Reads `tools.max_result_chars` (or the deprecated
  `agent.max_tool_result_chars`) to configure the per-call result truncation
  limit.

`Agent` retrieves the registry via
`get_dependency(pm, "tools", ToolCollectionPlugin)`. The `trylast=True`
marker guarantees all tool providers have fully initialized before collection
runs.

Other plugins can access the registry:
```python
from corvidae.hooks import get_dependency
from corvidae.tool_collection import ToolCollectionPlugin

tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
registry = tools_plugin.get_registry()
tools_dict, schemas = tools_plugin.get_tools()
```

**Without this plugin:** `Agent` cannot retrieve tools and will raise
`RuntimeError` during startup.

### DreamPlugin (`corvidae/tools/dream.py`)

Periodically reviews recent conversation history and appends extracted facts
to `MEMORY.md`. Entry point name: `"dream"`.

Implements two hooks:

- `on_start` — locates `sessions.db` within the workspace tree.
- `on_idle` — runs a dream cycle if `interval_seconds` have elapsed since
  the last cycle. Queries the last 40 rows from `message_log`, filters for
  assistant messages, strips `<think>` blocks, and appends new sentences to
  the `## Long-term Memory` section of `MEMORY.md`. Skips sentences already
  present (deduplication by normalized text). No-ops if `sessions.db` is not
  found.

**Config:**
```yaml
dream:
  interval_seconds: 300   # minimum seconds between dream cycles (default 300)
```

**Without this plugin:** `MEMORY.md` is not updated automatically. The
`on_idle` hook fires for other listeners regardless.

### IdleMonitorPlugin (`corvidae/idle.py`)

Fires the `on_idle` broadcast hook when all queues are quiescent.
Entry point name: `"idle_monitor"`.

Currently a stub — implements `on_idle` as a no-op. The idle
detection logic itself lives in `Agent._maybe_fire_idle`, which
checks queue quiescence and fires the `on_idle` broadcast.
Plugins that implement `on_idle` (such as `DreamPlugin`) receive
those calls.

**Idle condition:** all `SerialQueue` instances have `is_empty=True`,
`TaskQueue.is_idle` is `True` (skipped if `TaskPlugin` is not
registered), and at least `idle_cooldown_seconds` have elapsed since
the last firing.

**Config:**
```yaml
daemon:
  idle_cooldown_seconds: 30   # minimum seconds between on_idle firings (default 30)
```

**Without this plugin:** the `on_idle` hook is never fired. Plugins that implement `on_idle` (such as `DreamPlugin`) will not receive calls.
