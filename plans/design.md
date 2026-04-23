# Agent Daemon Design Document

An asyncio-based agent daemon that connects to IRC (and later Signal,
BlueSky, etc.), routes messages through a local LLM via the OpenAI Agents
SDK, and supports hot-loadable plugin components via pluggy.

## Architecture Overview

The daemon is a single Python asyncio process running in an LXC container
on the desktop (i7-8700K / RTX 3060 Ti / 128GB DDR4-2666). It connects
to a Qwen3.6-35B-A3B model served by llama-server on the host via
OpenAI-compatible API.

Three layers:

1. **Plugin system (pluggy + apluggy)** — defines lifecycle hooks and
   extension points. Components register/unregister at runtime for
   hot-loading.

2. **Agent loop (hand-rolled)** — manages prompt construction, tool
   calling, LLM interaction via aiohttp against llama-server's
   OpenAI-compatible Chat Completions API. Owns conversation state
   and memory.

3. **Transport plugins** — IRC (first), Signal (later). Each transport
   converts platform-specific messages to/from a common format and
   calls hooks on the plugin manager.

```
┌─────────────────────────────────────────────┐
│              Plugin Manager                 │
│          (apluggy.PluginManager)            │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │   IRC   │  │  Agent   │  │  (future)  │  │
│  │Transport│  │   Loop   │  │  BlueSky,  │  │
│  │ Plugin  │  │  Plugin  │  │  LinkRoll  │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│                     │                       │
│              ┌──────┴───────┐               │
│              │ OpenAI Agents│               │
│              │     SDK      │               │
│              │   Runner     │               │
│              └──────┬───────┘               │
│                     │                       │
│          ┌──────────┴──────────┐            │
│          │  llama-server       │            │
│          │  (Qwen3.6 on host)  │            │
│          └─────────────────────┘            │
└─────────────────────────────────────────────┘
```

## Dependencies

```
# Core
apluggy          # async-aware pluggy wrapper
pydantic         # data models, tool schema generation
aiohttp          # async HTTP client (LLM API + web_fetch tool)

# IRC transport
pydle            # asyncio IRC client

# Persistence
aiosqlite        # async SQLite for session/log storage

# Hot-loading
watchdog         # filesystem watcher for component changes

# Config
pyyaml           # YAML config parsing
```

## Hook Specifications

The plugin manager defines these hookspecs. All hooks are async (called
via `pm.ahook`). Plugins implement whichever hooks they need.

```python
# hooks.py
from __future__ import annotations

from typing import TYPE_CHECKING

import apluggy as pluggy

if TYPE_CHECKING:
    from sherman.channel import Channel

hookspec = pluggy.HookspecMarker("sherman")
hookimpl = pluggy.HookimplMarker("sherman")

class AgentSpec:
    """Hook specifications for the agent daemon."""

    @hookspec
    async def on_start(self, config: dict) -> None:
        """Called when the daemon starts. Initialize connections."""

    @hookspec
    async def on_stop(self) -> None:
        """Called on shutdown. Clean up resources."""

    @hookspec
    async def on_message(self, channel: Channel, sender: str, text: str) -> None:
        """A message arrived from a transport.

        channel: Channel object identifying transport and scope
            (e.g. transport="irc", scope="#lex")
        sender: nick, phone number, or other identifier
        text: message content
        """

    @hookspec
    async def send_message(self, channel: Channel, text: str) -> None:
        """Send a message to a specific channel.

        Each transport plugin checks channel.transport and sends if it
        matches. Non-matching transports ignore the call.
        """

    @hookspec
    def register_tools(self, tool_registry: list) -> None:
        """Append tool functions to the tool registry.

        Called during agent construction, before the agent loop starts.
        Sync because tool registration happens at startup/reload, not
        during async message handling.
        """

    @hookspec
    async def on_agent_response(
        self, channel: Channel, request_text: str, response_text: str
    ) -> None:
        """Called after the agent produces a response, before it is sent.

        Plugins can use this for logging, analytics, or response
        modification (though modification is not the intended pattern).
        """

    @hookspec
    async def on_task_complete(
        self, channel: Channel, task_id: str, result: str
    ) -> None:
        """A background task finished. The result should be posted
        to the originating channel."""
```

## Plugin Manager Setup

```python
# plugin_manager.py
import apluggy as pluggy
from hooks import AgentSpec

def create_plugin_manager() -> pluggy.PluginManager:
    pm = pluggy.PluginManager("sherman")
    pm.add_hookspecs(AgentSpec)
    return pm
```

## Agent Loop Plugin

The agent loop plugin subscribes to `on_message`, calls llama-server
directly via aiohttp, handles tool dispatch, and manages conversation
state.

### LLM Client

A thin wrapper around aiohttp that speaks the Chat Completions API.
No SDK, no client library.

```python
class LLMClient:
    """Async client for OpenAI-compatible Chat Completions API."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session: aiohttp.ClientSession | None = None

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat completion request. Returns the raw response dict."""
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
```

### Agent Loop

The core loop: call LLM, check for tool calls, execute them, feed
results back, repeat until the model produces a plain text response.

```python
async def run_agent_loop(
    client: LLMClient,
    messages: list[dict],
    tools: dict[str, callable],
    tool_schemas: list[dict],
    max_turns: int = 10,
) -> str:
    """Run the agent loop to completion.

    Args:
        client: LLM client
        messages: Conversation history (mutated in place — new turns
            are appended)
        tools: Map of tool name -> async callable
        tool_schemas: JSON schemas for tool definitions
        max_turns: Safety limit on tool-calling rounds

    Returns:
        The final text response from the model.
    """
    for _ in range(max_turns):
        response = await client.chat(messages, tools=tool_schemas or None)
        choice = response["choices"][0]
        msg = choice["message"]

        # Append the assistant's message to history
        messages.append(msg)

        # If no tool calls, we're done — return raw content
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "")

        # Execute each tool call and append results
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = json.loads(call["function"]["arguments"])
            fn = tools.get(fn_name)
            if fn:
                try:
                    result = await fn(**fn_args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Error: unknown tool '{fn_name}'"

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": str(result),
            })

    return "(max tool-calling rounds reached)"


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text content.

    Defensive fallback — with the current llama-server + Qwen3.6 setup,
    thinking tokens appear in the separate `reasoning_content` field,
    not in `content`, so this should be a no-op. Retained in case a
    different model or server version puts <think> tags in content.
    """
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

### Tool Schema Generation

Tools are Python async functions with type hints and docstrings. Schemas
are generated at registration time using Pydantic.

```python
from pydantic import TypeAdapter
import inspect

def tool_to_schema(fn: callable) -> dict:
    """Generate a Chat Completions tool schema from a function.

    Uses the function's type hints for parameter types and its
    docstring for descriptions. Pydantic handles the JSON schema
    generation.
    """
    sig = inspect.signature(fn)
    fields = {}
    for name, param in sig.parameters.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            annotation = str
        fields[name] = (annotation, ...)

    # Build a Pydantic model dynamically
    from pydantic import create_model
    ParamsModel = create_model(f"{fn.__name__}_params", **{
        name: (ann, ...) for name, (ann, _) in fields.items()
    })

    schema = ParamsModel.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": (fn.__doc__ or "").strip().split("\n")[0],
            "parameters": schema,
        },
    }
```

### Thinking Token Handling

Qwen3.6 emits `<think>...</think>` blocks before responses when in
thinking mode. llama-server parses these and returns them in a separate
`reasoning_content` field on the message object — they do **not** appear
in `content`. This was confirmed by the Phase 0 smoke test (see
`plans/smoke-test.md`).

This means:

- **No `strip_thinking()` needed at the API level.** The `content` field
  contains only the visible response. The `reasoning_content` field
  contains the chain-of-thought. They are already separated by the
  server.

- **Tool calling is clean.** When the model makes a tool call, thinking
  goes into `reasoning_content`, the tool call goes into `tool_calls`,
  and `content` is empty. No interleaving.

Handling at three layers:

**Display:** Just use `content` directly — it's already clean.

**Persistent log:** Preserve the full message dict including
`reasoning_content`. The log is the ground truth; nothing is discarded.

**Active prompt history:** Omit `reasoning_content` from assistant
messages when constructing the prompt. The smoke test confirmed that
multi-turn tool calling works identically with or without reasoning
in history, so there's no benefit to keeping it, and omitting it
saves significant context (reasoning is often 5-10x the visible
response).

```yaml
# agent.yaml
agent:
  keep_thinking_in_history: false  # reasoning_content omitted from prompt
```

Note: `strip_thinking()` is retained as a defensive fallback in case
a different model or server version puts `<think>` tags in `content`
directly, but it should be a no-op with the current llama-server +
Qwen3.6 setup.

### Conversation State and Memory

#### Phase 1 (implement now): Stop-the-world compaction

Conversation state is an ordered list of message dicts. Every message
(user, assistant, tool, and memory) is appended to both the active
prompt and a persistent log.

```python
class ConversationLog:
    """Append-only persistent log + active prompt management."""

    def __init__(self, db: aiosqlite.Connection, channel_id: str):
        self.db = db
        self.channel_id = channel_id
        self.messages: list[dict] = []   # active prompt messages
        self.system_prompt: str = ""

    async def load(self):
        """Load conversation history from the database."""
        ...

    async def append(self, message: dict):
        """Append a message to both active list and persistent log."""
        self.messages.append(message)
        await self._persist(message)

    async def _persist(self, message: dict):
        """Write a single message to the append-only log."""
        await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
            (self.channel_id, json.dumps(message), time.time())
        )
        await self.db.commit()

    def token_estimate(self) -> int:
        """Rough token count estimate (chars / 3.5)."""
        total = len(self.system_prompt)
        for msg in self.messages:
            total += len(msg.get("content", ""))
        return int(total / 3.5)

    async def compact_if_needed(self, client: LLMClient, max_tokens: int):
        """If approaching context limit, summarize older messages.

        Keeps the most recent N messages intact, summarizes the rest
        into a single assistant message. The full history remains in
        the persistent log — compaction only affects the active prompt.
        """
        if self.token_estimate() < max_tokens * 0.8:
            return

        # Keep the last ~20 messages, summarize the rest
        keep = 20
        if len(self.messages) <= keep:
            return

        to_summarize = self.messages[:-keep]
        to_keep = self.messages[-keep:]

        summary = await self._summarize(client, to_summarize)

        self.messages = [
            {"role": "assistant", "content": f"[Summary of earlier conversation]\n{summary}"},
            *to_keep,
        ]

    async def _summarize(self, client: LLMClient, messages: list[dict]) -> str:
        """Ask the LLM to summarize a block of messages."""
        summary_prompt = [
            {"role": "system", "content": "Summarize the following conversation concisely, preserving key facts, decisions, and context that would be needed to continue the conversation."},
            {"role": "user", "content": json.dumps(messages, indent=2)},
        ]
        response = await client.chat(summary_prompt)
        return response["choices"][0]["message"]["content"]

    def build_prompt(self) -> list[dict]:
        """Construct the full prompt for the LLM."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages,
        ]
```

The persistent log table:

```sql
CREATE TABLE message_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message TEXT NOT NULL,      -- JSON
    timestamp REAL NOT NULL
);
CREATE INDEX idx_log_channel ON message_log(channel_id, timestamp);
```

This is deliberately simple. Every message ever sent or received is in
the log, forever. Compaction only affects the in-memory active prompt.
The log is the foundation that future memory retrieval builds on.

#### Future: Append-only memory injection

The background process will search the log (via embeddings), find
relevant prior context, and *append* `<memory>` blocks into the
conversation stream between turns. These are sequential entries in the
message list, not a preamble — they sit where they chronologically
occurred (after the most recent turn, before the next user message).
This preserves the KV cache for everything above the injection point.

#### Future: Double-buffer compaction

When context approaches the limit, the background slot builds a new
compacted prompt from the full log, warms it in a second llama-server
KV cache slot, and swaps it in. The foreground slot eats one swap
latency instead of a full re-ingestion pause.

Neither of these future features is in scope for the first
implementation, but the append-only log and the separation between
"active prompt" and "persistent log" make them possible without
rearchitecting.

### Agent Loop Plugin Implementation

```python
class AgentLoopPlugin:
    def __init__(self, pm, config):
        self.pm = pm
        self.config = config
        self.client: LLMClient | None = None
        self.conversations: dict[str, ConversationLog] = {}
        self.tools: dict[str, callable] = {}
        self.tool_schemas: list[dict] = []
        self.db = None

    @hookimpl
    async def on_start(self, config):
        llm_config = config.get("llm", {})
        self.client = LLMClient(
            base_url=llm_config["base_url"],
            model=llm_config["model"],
        )
        await self.client.start()

        self.db = await aiosqlite.connect(
            config.get("daemon", {}).get("session_db", "sessions.db")
        )
        await self._init_db()

        # Collect tools from all plugins
        tool_fns = []
        self.pm.hook.register_tools(tool_registry=tool_fns)
        for fn in tool_fns:
            self.tools[fn.__name__] = fn
            self.tool_schemas.append(tool_to_schema(fn))

        self.system_prompt = config.get("agent", {}).get(
            "system_prompt", "You are a helpful assistant."
        )
        self.max_context_tokens = config.get("agent", {}).get(
            "max_context_tokens", 24000
        )
        self.keep_thinking = config.get("agent", {}).get(
            "keep_thinking_in_history", True
        )

    @hookimpl
    async def on_message(self, channel, sender, text):
        conv = await self._get_conversation(channel.id)
        await conv.append({"role": "user", "content": text})
        await conv.compact_if_needed(self.client, self.max_context_tokens)

        messages = conv.build_prompt()
        raw_response = await run_agent_loop(
            self.client, messages, self.tools, self.tool_schemas
        )

        # Persist the full response (including thinking) to the log
        await conv.append({"role": "assistant", "content": raw_response})

        # Optionally strip thinking from the active prompt history
        if not self.keep_thinking:
            conv.strip_thinking_from_last()

        # Always strip thinking for display
        display_response = strip_thinking(raw_response)

        await self.pm.ahook.on_agent_response(
            channel=channel,
            request_text=text,
            response_text=display_response,
        )
        await self.pm.ahook.send_message(
            channel=channel,
            text=display_response,
        )

    async def _get_conversation(self, channel_id: str) -> ConversationLog:
        if channel_id not in self.conversations:
            conv = ConversationLog(self.db, channel_id)
            conv.system_prompt = self.system_prompt
            await conv.load()
            self.conversations[channel_id] = conv
        return self.conversations[channel_id]

    async def _init_db(self):
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS message_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_log_channel ON message_log(channel_id, timestamp)"
        )
        await self.db.commit()

    @hookimpl
    async def on_stop(self):
        if self.client:
            await self.client.stop()
        if self.db:
            await self.db.close()
```

## IRC Transport Plugin

Uses pydle as the asyncio IRC client. The plugin manages the pydle
client lifecycle and translates between IRC events and agent hooks.

### Channel ID Format

`irc:{channel}` — e.g., `irc:#lex`

### Message Flow

1. pydle receives a PRIVMSG in a joined channel
2. IRC plugin calls `pm.ahook.on_message(channel=<Channel irc:#lex>, ...)`
3. Agent loop plugin handles it, calls `pm.ahook.send_message(...)`
4. IRC plugin's `send_message` hook checks `channel.matches_transport("irc")`, sends PRIVMSG

### IRC Message Splitting

IRC has a ~512 byte message limit per line. Long agent responses must be
split into multiple messages. Split on paragraph boundaries first, then
on sentence boundaries, then hard-wrap.

### Implementation Sketch

```python
class IRCClient(pydle.Client):
    """Thin pydle subclass that forwards messages to the plugin."""

    def __init__(self, plugin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plugin = plugin

    async def on_connect(self):
        for channel in self.plugin.channels:
            await self.join(channel)

    async def on_message(self, target, source, message):
        if target in self.plugin.channels and source != self.nickname:
            channel = self.plugin.pm.registry.get_or_create("irc", target)
            await self.plugin.pm.ahook.on_message(
                channel=channel,
                sender=source,
                text=message,
            )


class IRCPlugin:
    def __init__(self, pm):
        self.pm = pm
        self.client = None
        self.channels = []

    @hookimpl
    async def on_start(self, config):
        irc_config = config.get("irc", {})
        self.channels = irc_config.get("channels", ["#lex"])
        nick = irc_config.get("nick", "agent")
        server = irc_config.get("server", "irc.lan")
        port = irc_config.get("port", 6667)

        self.client = IRCClient(
            self,
            nickname=nick,
            username=nick,
            realname="Agent Daemon",
        )
        # Connect in background — don't block other plugins starting
        asyncio.create_task(self.client.connect(server, port))

    @hookimpl
    async def send_message(self, channel, text):
        if not channel.matches_transport("irc"):
            return
        for chunk in self._split_message(text):
            await self.client.message(channel.scope, chunk)

    @hookimpl
    async def on_stop(self):
        if self.client:
            await self.client.quit("Shutting down")

    def _split_message(self, text, max_len=400):
        """Split long messages for IRC.

        Prefer splitting on paragraph breaks, then sentences, then
        hard-wrap. max_len leaves room for IRC protocol overhead.
        """
        if len(text) <= max_len:
            return [text]

        chunks = []
        # Split on double newlines first (paragraphs)
        paragraphs = text.split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_len:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current.strip())
                # If single paragraph exceeds max_len, hard-wrap
                while len(para) > max_len:
                    chunks.append(para[:max_len])
                    para = para[max_len:]
                current = para
        if current:
            chunks.append(current.strip())
        return chunks if chunks else [text]
```

## Background Task System

The daemon runs llama-server with `--parallel 2` (or higher). One slot
is reserved for foreground work (interactive message responses), and
background tasks consume the other slot(s) from a queue.

### Design

The agent loop has two modes:

**Foreground (interactive):** When a message arrives, the agent does a
quick LLM call to decide how to respond. If the task is simple, it
responds inline. If the task is complex or long-running, the agent calls
the `background_task` tool, which enqueues the work and returns an
acknowledgment immediately. The foreground `Runner.run()` completes
quickly either way.

**Background (worker):** A single asyncio task pulls work from an
`asyncio.Queue`, executes it (which may involve multiple LLM calls,
tool use, file I/O, etc.), and posts the result to the originating
channel via `pm.ahook.send_message()`. One task at a time, FIFO order.

```
Message arrives
  → Foreground LLM call (fast)
  → Agent decides: respond inline or spawn background work
  
  Inline path:
    → Response sent immediately
    → Foreground slot freed
  
  Background path:
    → "Working on it" sent immediately
    → Task enqueued
    → Foreground slot freed
    → Background worker picks up task (when idle)
    → Worker does LLM calls, tool use, etc.
    → Result posted to channel
```

### Task Queue

```python
@dataclass
class BackgroundTask:
    task_id: str
    channel: Channel
    description: str
    instructions: str
    created_at: float

class TaskQueue:
    def __init__(self):
        self.queue: asyncio.Queue[BackgroundTask] = asyncio.Queue()
        self.active_task: BackgroundTask | None = None
        self.completed: dict[str, str] = {}  # task_id -> result

    async def enqueue(self, task: BackgroundTask):
        await self.queue.put(task)

    async def run_worker(self, execute_fn):
        """Pull tasks and execute them one at a time.

        execute_fn(task) -> str is the function that does the work,
        typically an agent run with its own Runner.run() call.
        """
        while True:
            task = await self.queue.get()
            self.active_task = task
            try:
                result = await execute_fn(task)
                self.completed[task.task_id] = result
            except Exception as e:
                result = f"Task {task.task_id} failed: {e}"
            finally:
                self.active_task = None
                self.queue.task_done()
            yield task, result
```

### Background Task Tool

The agent calls this tool when it decides a request needs background
processing. The tool returns immediately with a confirmation.

```python

async def background_task(
    description: str,
    instructions: str,
) -> str:
    """Launch a long-running task in the background.

    Use this for any request that requires research, multi-step work,
    file generation, or anything that would take more than a few
    seconds. Returns immediately with a task ID; results will be
    posted to the channel when complete.

    Args:
        description: Brief summary of what this task does.
        instructions: Detailed step-by-step instructions for the
            background worker to follow.
    """
    # The actual enqueue happens in the agent loop plugin,
    # which wraps this tool and injects the channel.
    ...
```

### Agent Loop Changes for Background Support

The agent loop plugin starts the background worker as an asyncio task
during `on_start`. When a background task completes, it calls
`on_task_complete`, which triggers `send_message`.

The background worker uses a *separate* Agent instance (possibly with
a different system prompt optimized for task execution rather than
conversation) but the same model backend. This keeps the background
agent's context clean — it doesn't carry the conversation history,
just the task instructions.

```python
class AgentLoopPlugin:
    @hookimpl
    async def on_start(self, config):
        # ... existing setup ...
        self.task_queue = TaskQueue()
        self._worker_task = asyncio.create_task(
            self._run_background_worker()
        )

    async def _run_background_worker(self):
        async for task, result in self.task_queue.run_worker(
            self._execute_background_task
        ):
            await self.pm.ahook.on_task_complete(
                channel=task.channel,
                task_id=task.task_id,
                result=result,
            )
            await self.pm.ahook.send_message(
                channel=task.channel,
                text=f"[Task {task.task_id[:8]}] {result}",
            )

    async def _execute_background_task(self, task):
        """Run a background task with its own conversation context."""
        messages = [
            {"role": "system", "content": "You are executing a background task. "
             "Work through the instructions step by step. Be thorough."},
            {"role": "user", "content": task.instructions},
        ]
        return await run_agent_loop(
            self.client, messages, self.tools, self.tool_schemas
        )

    @hookimpl
    async def on_stop(self):
        if self._worker_task:
            self._worker_task.cancel()
        # ... existing cleanup ...
```

## Starter Tools

These tools are available to the agent from the start. They let the
agent do useful work inside the container without external credentials.

All tools are registered via the `register_tools` hook from a core
tools plugin (not hot-loadable, since they're part of the base
capability set).

Tools are plain async functions. The `tool_to_schema()` function in
`agent_loop.py` generates Chat Completions tool schemas from their
type hints and docstrings at registration time. No decorator needed.

### shell

Run a command inside the container and return stdout/stderr.

```python
async def shell(command: str) -> str:
    """Execute a shell command and return the output.

    Args:
        command: The command to run (via /bin/sh -c).
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(), timeout=30
    )
    output = stdout.decode() if stdout else ""
    if stderr:
        output += f"\nSTDERR:\n{stderr.decode()}"
    return output or "(no output)"
```

**Security note:** The container IS the sandbox. The agent can run
whatever it wants inside the container — that's the point. Credentials
and access are scoped by what's mounted/available in the LXC. No
allowlist or command filtering needed.

### read_file / write_file

Read from and write to the container filesystem.

```python

async def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Absolute or relative path to the file.
    """
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist"
    if p.stat().st_size > 1_000_000:
        return f"Error: {path} is too large ({p.stat().st_size} bytes)"
    return p.read_text()



async def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.

    Args:
        path: Absolute or relative path to the file.
        content: The content to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"
```

### web_fetch

Fetch a URL and return the content. Useful for research tasks.

```python

async def web_fetch(url: str) -> str:
    """Fetch a URL and return its text content.

    Args:
        url: The URL to fetch.
    """
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return f"HTTP {resp.status}"
            text = await resp.text()
            # Truncate very large responses
            if len(text) > 50_000:
                return text[:50_000] + "\n\n[truncated]"
            return text
```

### Task Status

Let the agent (and the user) check on background tasks.

```python

async def task_status() -> str:
    """Check the status of background tasks.

    Returns information about the currently running task and the
    number of tasks in the queue.
    """
    # Injected by the agent loop plugin with access to TaskQueue
    ...
```

### Tools Plugin

```python
class CoreToolsPlugin:
    @hookimpl
    def register_tools(self, tool_registry):
        tool_registry.extend([
            shell,
            read_file,
            write_file,
            web_fetch,
            task_status,
            background_task,
        ])
```

## Hot-Loading

A file watcher monitors the `components/` directory. When a Python file
changes, the loader:

1. Calls `on_stop` on the affected plugin (if it was registered)
2. Unregisters the old plugin instance
3. Reloads the module via `importlib.reload()`
4. Instantiates the new plugin class
5. Registers the new instance
6. Calls `on_start` with the current config

```python
class ComponentLoader:
    def __init__(self, pm, config, components_dir="components"):
        self.pm = pm
        self.config = config
        self.components_dir = components_dir
        self.loaded: dict[str, object] = {}  # module_name -> plugin instance

    async def load_all(self):
        """Load all Python files in the components directory."""
        for path in Path(self.components_dir).glob("*.py"):
            if path.name.startswith("_"):
                continue
            await self.load_module(path.stem)

    async def load_module(self, module_name: str):
        """Load or reload a single component module."""
        await self.unload_module(module_name)

        module_path = f"components.{module_name}"
        if module_path in sys.modules:
            module = importlib.reload(sys.modules[module_path])
        else:
            module = importlib.import_module(module_path)

        # Convention: each module has a create_plugin(pm, config) function
        if hasattr(module, "create_plugin"):
            plugin = module.create_plugin(self.pm, self.config)
            self.pm.register(plugin, name=module_name)
            self.loaded[module_name] = plugin
            await self.pm.ahook.on_start(config=self.config)

    async def unload_module(self, module_name: str):
        """Unload a component if it was previously loaded."""
        if module_name in self.loaded:
            old_plugin = self.loaded.pop(module_name)
            # Per-plugin stop if the plugin has the method
            if hasattr(old_plugin, "on_stop"):
                try:
                    await old_plugin.on_stop()
                except Exception:
                    pass
            self.pm.unregister(old_plugin)
```

**Note on hot-loading edge cases:** Python's `importlib.reload` replaces
the module object in `sys.modules`, but existing references to old
classes persist. Since we unregister the old plugin instance and create
a new one from the reloaded module, this is safe. The hookimpl
decorators are re-evaluated on the new class. However, if a reloaded
module changes a Pydantic model that other modules reference, those
other modules will still hold the old class. For this project, that
shouldn't matter — plugins don't share model types directly.

## Configuration

YAML config file. Loaded at startup, passed to all plugins via
`on_start(config=...)`.

```yaml
# agent.yaml

daemon:
  components_dir: components
  session_db: sessions.db
  watch_components: true   # enable hot-loading file watcher

llm:
  base_url: "http://192.168.1.88:8080/v1"
  api_key: "not-needed"
  model: "qwen3.6-35b-a3b"

agent:
  max_context_tokens: 24000
  keep_thinking_in_history: false  # reasoning_content omitted from prompt; see plans/smoke-test.md
  system_prompt: |
    You are a personal assistant. Be concise and direct.
    You are communicating over IRC so keep responses short.

irc:
  server: irc.lan
  port: 6667
  nick: agent
  channels:
    - "#lex"
```

## Daemon Entry Point

```python
# main.py
import asyncio
import signal
import yaml
from pathlib import Path

from plugin_manager import create_plugin_manager
from agent_loop import AgentLoopPlugin
from tools import CoreToolsPlugin
from component_loader import ComponentLoader


async def main():
    # Load config
    config_path = Path("agent.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create plugin manager
    pm = create_plugin_manager()

    # Register core plugins (not hot-loadable)
    core_tools = CoreToolsPlugin()
    pm.register(core_tools, name="core_tools")

    agent_loop = AgentLoopPlugin(pm, config)
    pm.register(agent_loop, name="agent_loop")

    # Load component plugins (hot-loadable)
    loader = ComponentLoader(pm, config, config["daemon"]["components_dir"])
    await loader.load_all()

    # Start all plugins
    await pm.ahook.on_start(config=config)

    # Set up graceful shutdown
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    # Optionally start the file watcher for hot-loading
    if config["daemon"].get("watch_components", False):
        loader.start_watching()

    # Wait for shutdown signal
    await stop_event.wait()

    # Teardown
    await pm.ahook.on_stop()


if __name__ == "__main__":
    asyncio.run(main())
```

## Directory Layout

```
agent-daemon/
├── main.py                 # Entry point
├── hooks.py                # AgentSpec hookspecs
├── plugin_manager.py       # create_plugin_manager()
├── channel.py              # Channel, ChannelConfig, ChannelRegistry, load_channel_config
├── llm.py                  # LLMClient (aiohttp wrapper)
├── agent_loop.py           # run_agent_loop(), tool schema generation
├── conversation.py         # ConversationLog (append-only + compaction)
├── background.py           # TaskQueue, BackgroundTask
├── tools.py                # CoreToolsPlugin (shell, files, web_fetch, etc.)
├── component_loader.py     # Hot-loading with watchdog
├── components/
│   ├── __init__.py
│   └── irc.py              # IRC transport plugin
├── agent.yaml              # Configuration
├── pyproject.toml           # Project metadata and dependencies
├── tests/
│   ├── test_hooks.py        # Hook registration and dispatch
│   ├── test_agent_loop.py   # Agent loop with mocked LLM
│   ├── test_conversation.py # Conversation log and compaction
│   ├── test_background.py   # Background task queue and worker
│   ├── test_irc.py          # IRC plugin with mocked pydle
│   ├── test_tools.py        # Starter tools
│   └── test_loader.py       # Component hot-loading
└── README.md
```

## Implementation Sequence

### Phase 0: Smoke Test ✓

Completed. See `scripts/smoke_test.py` and `plans/smoke-test.md` for
full results. All steps passed:

- aiohttp ↔ llama-server connectivity works
- Tool calling works (correct function name, valid JSON arguments,
  round-trip produces final text response)
- Thinking tokens appear in `reasoning_content` field, not `content`
  — no interleaving with tool calls
- Multi-turn tool calling works with and without reasoning in history
- Recommendation: `keep_thinking_in_history: false`

### Phase 1: Core Framework ✓

Completed. 39 tests pass. All deliverables implemented:

- Hook definitions (`hooks.py`) — `AgentSpec` with 7 lifecycle hooks
- Plugin manager setup (`plugin_manager.py`) — `create_plugin_manager()`
- LLM client (`llm.py`) — async `LLMClient` wrapping OpenAI-compatible API
- Agent loop (`agent_loop.py`) — tool dispatch, schema generation via pydantic,
  `strip_thinking()` for `<think>` block removal
- Conversation log (`conversation.py`) — append-only SQLite persistence,
  LLM-based compaction when token estimate reaches 80% of context limit
- Daemon entry point (`main.py`) — loads YAML config, fires hooks, waits for
  SIGINT/SIGTERM, shuts down cleanly
- Tests for hooks, agent loop (mocked HTTP), conversation log

### Phase 1.5: Channel Abstraction ✓

Completed. 68 tests pass (39 existing + 29 new). All deliverables implemented:

- `channel.py` — `ChannelConfig` (per-channel config with agent-level fallback),
  `Channel` (transport + scope + config + conversation reference),
  `ChannelRegistry` (lifecycle management, get_or_create, by_transport),
  `load_channel_config` (pre-registers channels from YAML before `on_start`)
- Hook signature migration — four hookspecs changed from `channel_id: str` to
  `channel: Channel`: `on_message`, `send_message`, `on_agent_response`,
  `on_task_complete`
- Registry injection — `pm.registry` set as a plain attribute on the plugin
  manager in `main.py`; plugins access it via `self.pm.registry`
- YAML channel config loading — `channels:` top-level key, keys in
  `"transport:scope"` format, per-channel overrides for `system_prompt`,
  `max_context_tokens`, `keep_thinking_in_history`

### Phase 2: Agent Loop Plugin

Channel objects are now available via `pm.registry`. The agent loop plugin
should use `pm.registry.get_or_create(channel.transport, channel.scope)` (or
the channel passed via `on_message`) and attach a `ConversationLog` to
`channel.conversation` on first use:

```python
if channel.conversation is None:
    channel.conversation = ConversationLog(self.db, channel.id)
    resolved = self.pm.registry.resolve_config(channel)
    channel.conversation.system_prompt = resolved["system_prompt"]
    await channel.conversation.load()
```

1. Agent loop plugin (`agent_loop.py` plugin class)
2. LLM configuration from YAML
3. Thinking token handling — three-layer strategy (strip for display,
   preserve in log, configurable for active history)
4. ConversationLog integration via `channel.conversation` — append-only
   persistence, stop-the-world compaction when approaching context limit;
   per-channel config resolved via `pm.registry.resolve_config(channel)`
5. Wire up `register_tools` hook
6. Tests with mocked LLM responses (mock aiohttp)

### Phase 3: Starter Tools and Background Tasks

1. Core tools plugin (`tools.py`) — shell, read_file, write_file,
   web_fetch
2. Background task queue (`background.py`)
3. `background_task` and `task_status` tools
4. Background worker in agent loop plugin with its own system prompt
5. Tests for tools (mock subprocess for shell, temp dirs for files,
   mock aiohttp for web_fetch)
6. Tests for background task queue

### Phase 4: IRC Transport

1. IRC plugin (`components/irc.py`) with pydle
2. Message splitting for long responses
3. Wire up to agent loop via hooks
4. End-to-end test: send IRC message, get response from local LLM
5. Tests with mocked pydle client

### Phase 5: Hot-Loading

1. Component loader (`component_loader.py`)
2. Watchdog filesystem watcher integration
3. Test: modify a component file, verify it reloads
4. Test: add a new component file, verify it loads

## Open Questions

These should be resolved during implementation, not upfront:

1. **Thinking tokens and tool calling**: ✓ Resolved by smoke test.
   Qwen3.6 produces valid tool call JSON with thinking mode enabled.
   Thinking tokens go into a separate `reasoning_content` field, not
   `content`. No interleaving. Multi-turn works with or without
   reasoning in history. Default: `keep_thinking_in_history: false`.

2. **pydle maintenance status**: pydle's last release may be dated.
   If it doesn't work with current Python, `irc` (jaraco) with an
   asyncio adapter is the fallback.

3. **apluggy `on_start` fan-out**: When `pm.ahook.on_start()` is called,
   does apluggy call all implementations concurrently or sequentially?
   If concurrent, the IRC plugin's `asyncio.create_task` for connection
   may be redundant. If sequential, it's needed to avoid blocking other
   plugins' startup.

4. **llama-server slot assignment**: Can requests be pinned to specific
   KV cache slots? This matters for the future double-buffer compaction
   pattern. Not blocking for Phase 1 but worth investigating early.

5. **Compaction quality**: The stop-the-world summarization uses the
   same LLM. Quality of the summary determines how much context is
   lost. May need prompt tuning or a dedicated summarization prompt.

6. **Token counting**: The `chars / 3.5` estimate is rough. For
   accurate compaction triggers, investigate whether llama-server
   returns token counts in the response (it should, in the `usage`
   field) and track actual consumption.

7. **Handoff to Claude/xAI**: Not in scope for this phase. Will be
   added later as a tool that dispatches requests to external APIs.
   The tool system already supports this — it's just a new function
   in the tool registry.

## Memory Retrieval (Future)

The conversation log stores every message permanently. The retrieval
system will use this log to inject relevant prior context into the
active prompt as the conversation progresses.

### Architecture Sketch

**Embedding model:** A small CPU-only model (e.g., nomic-embed-text or
bge-small-en-v1.5 via sentence-transformers, ~130MB). Runs in-process,
generates embeddings in milliseconds. Does not touch llama-server.

**Vector store:** sqlite-vec — adds vector similarity search as a
virtual table on top of the existing SQLite database. No separate
service.

**Retrieval trigger:** After each turn, the background process embeds
the recent exchange, searches the vector index for similar past
messages, and appends relevant fragments as `<memory>` blocks in the
conversation stream — between the most recent assistant response and
the next user message.

**Prompt structure (at maturity):**

```
[System instructions]                    ← stable, cached
[Identity / personality]                 ← stable, cached
[Compacted history summary]              ← stable between compactions, cached
[Turn N: user message]                   ← cached
[Turn N: assistant response]             ← cached
<memory>relevant fragment from log</memory>  ← appended by background
[Turn N+1: user message]                 ← new
```

Memory blocks are append-only. Once injected, they become part of the
cached history and are never moved or edited. This preserves the KV
cache for everything above the injection point.

**Compaction** remains a separate, less frequent operation triggered
by context overflow. It rebuilds the prompt from the full log
(potentially in a second llama-server slot for seamless swap), keeping
relevant memories and summarizing the rest.

**What this requires before implementation:**

- The message log and conversation system from Phase 1
- sqlite-vec integration
- An embedding pipeline (sentence-transformers or a small GGUF
  embedding model via llama.cpp on a separate port)
- A background asyncio task that watches for new turns and runs
  retrieval without blocking the foreground
- Tuning: similarity thresholds, how many fragments to inject,
  how to cluster/deduplicate related memories

None of this is in scope for the initial implementation. The
append-only log and the separation between active prompt and
persistent log are the prerequisites, and those are built in Phase 1.

## Non-Goals (For Now)

- Signal transport
- BlueSky integration
- Link roll / semantic indexing
- LXC container management tools
- Autobus cross-process messaging
- Web UI
- Scheduled / proactive agent behavior (background tasks are reactive,
  not proactive — the agent doesn't wake up on its own yet)
- Persistent memory system (see Memory Retrieval section above for
  the planned architecture — prerequisites are built in Phase 1 but
  retrieval itself is deferred)

These are all future components that plug in via the same hook system.
The architecture supports them; the first implementation doesn't include
them.