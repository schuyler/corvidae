# Task System Redesign

## Context

The current background task system conflates three things:

1. A task queue (general async work dispatch + result delivery)
2. An agent loop executor (runs `run_agent_loop` with tools and an LLM)
3. Tool registration for `background_task` and `task_status`

`BackgroundPlugin._execute_task` always runs `run_agent_loop`. It reaches
into `AgentPlugin` via `pm.agent_plugin` for tools and client.
`AgentPlugin._process_queue_item` reaches into `BackgroundPlugin` via
`pm.background` for the task queue. The coupling is bidirectional.

Meanwhile, the agent loop itself (`run_agent_loop`) is a tight synchronous
loop that blocks the channel queue for its entire duration — including all
tool calls, no matter how long they take. A 5-minute shell command blocks
the channel.

This redesign separates task infrastructure from agent logic, introduces
a `ToolContext` so tools can interact with the system, and replaces the
blocking agent loop with a notification-driven turn model.

## Design

### Task

A task is an async callable that returns a string, plus delivery context.

```python
@dataclass
class Task:
    work: Callable[[], Awaitable[str]]
    channel: Channel
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time)
    tool_call_id: str | None = None
    description: str = ""  # human-readable, for status display
```

The queue doesn't know what `work` does. It calls `await task.work()`,
catches exceptions, and delivers the result string to `task.channel` via
`on_notify` with `task.tool_call_id`.

### TaskQueue

Remains in `background.py` (renamed conceptually — it's general
infrastructure now, not "background" specific). Owns the worker loop.
No LLM knowledge, no tool knowledge, no reference to AgentPlugin.

```python
class TaskQueue:
    async def enqueue(self, task: Task) -> None: ...
    async def run_worker(self, on_complete: Callable) -> None: ...
    def status(self) -> str: ...
```

`on_complete` calls `pm.ahook.on_notify(channel, source, text,
tool_call_id, meta)` to deliver the result. The TaskQueue plugin
registers a `task_status` tool (no LLM context needed — just reads
queue state).

### ToolContext

Replaces the `_`-prefix injection convention. A single context object
injected into tool calls that gives tools access to the system.

```python
@dataclass
class ToolContext:
    channel: Channel
    tool_call_id: str
    task_queue: TaskQueue
```

Tools that need system access declare a `_ctx: ToolContext` parameter
(single underscore prefix preserved as the "injected, not LLM-supplied"
convention — but now it's one parameter, not N). The agent turn
dispatcher constructs the context and injects it.

Tools that don't need context (e.g., `read_file`, `web_fetch`) don't
declare `_ctx` and work exactly as today.

A tool that wants to background work uses the context:

```python
async def shell(command: str, _ctx: ToolContext) -> str:
    """Execute a shell command."""
    async def work():
        proc = await asyncio.create_subprocess_shell(...)
        stdout, stderr = await proc.communicate()
        return stdout.decode()

    task = Task(
        work=work,
        channel=_ctx.channel,
        tool_call_id=_ctx.tool_call_id,
        description=f"shell: {command[:60]}",
    )
    await _ctx.task_queue.enqueue(task)
    return f"Task {task.task_id} enqueued: {command[:60]}"
```

Whether a tool runs inline or enqueues a task is the tool's decision.
The system doesn't distinguish.

### Agent turn (replaces agent loop)

The current `run_agent_loop` is a multi-turn blocking loop:

```
for _ in range(max_turns):
    response = LLM(messages)
    if no tool_calls: return
    for call in tool_calls:
        result = await tool(args)      # blocks
        messages.append(result)
```

This becomes a single-turn function:

```python
async def run_agent_turn(
    client: LLMClient,
    messages: list[dict],
    tool_schemas: list[dict],
) -> AgentTurnResult:
    """Single LLM invocation. Returns the response and any tool calls."""
    response = await client.chat(messages, tools=tool_schemas or None)
    msg = response["choices"][0]["message"]
    msg.setdefault("role", "assistant")
    return AgentTurnResult(
        message=msg,
        tool_calls=msg.get("tool_calls", []),
        text=msg.get("content", ""),
    )
```

No tool execution. No loop. The caller (AgentPlugin) handles dispatch:

1. Append assistant message to conversation
2. If tool_calls: for each call, dispatch a Task to the queue
3. If no tool_calls: send text response to channel
4. Return — channel queue is free

Tool results arrive later as notifications, which trigger another
agent turn. Multi-turn reasoning emerges from the notification cycle.

### max_turns

Per-channel counter on the Channel object. Tracks consecutive LLM turns
without a user message.

- User message: reset counter to 0
- LLM turn (from notification or any other trigger): increment counter
- Counter >= limit: force text response, do not dispatch tool calls,
  log a warning

The limit comes from channel config (falls back to agent defaults),
same as `max_context_tokens`.

### subagent tool

A tool, not a plugin. Registered via `register_tools` like any other.
Constructs a `run_agent_loop` coroutine (the old multi-turn loop,
preserved for sub-agent use) with its own LLMClient and enqueues it
as a Task.

```python
async def subagent(
    instructions: str,
    _ctx: ToolContext,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Launch a sub-agent in the background."""
    # Build LLM client from args or fall back to defaults
    client = _build_client(model, base_url, defaults)

    async def work():
        await client.start()
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instructions},
            ]
            return await run_agent_loop(client, messages, tools, schemas)
        finally:
            await client.stop()

    task = Task(
        work=work,
        channel=_ctx.channel,
        tool_call_id=_ctx.tool_call_id,
        description=f"subagent: {instructions[:60]}",
    )
    await _ctx.task_queue.enqueue(task)
    return f"Task {task.task_id} enqueued"
```

`run_agent_loop` is preserved as-is for sub-agent use — sub-agents
run their own tight loop in a background task, which is fine because
they don't hold any channel queue. The blocking-loop problem only
exists when the loop runs inline on the channel queue.

The `subagent` tool needs access to the available tool set and
default LLM config. These could come from ToolContext (extended to
carry them) or from a closure at registration time.

### What stays the same

- `LLMClient` — unchanged
- `ConversationLog` — unchanged
- `Channel`, `ChannelConfig`, `ChannelRegistry` — unchanged
  (max_turns counter added to Channel)
- `SerialQueue` — unchanged (per-channel serial processing)
- `Tool`, `ToolRegistry`, `tool_to_schema` — unchanged
- `hookspecs` — unchanged (on_message, on_notify, send_message, etc.)
- `strip_thinking`, `strip_reasoning_content` — unchanged
- `channels/`, `tools/` — unchanged (tools may adopt `_ctx` but don't
  have to)
- `resolve_system_prompt` — unchanged

### What changes

| Current | New |
|---|---|
| `run_agent_loop()` — multi-turn blocking loop | `run_agent_turn()` — single LLM call, returns tool calls |
| `run_agent_loop()` executes tools inline | Tool calls dispatched as Tasks |
| `BackgroundPlugin` — owns queue, executor, tools | `TaskPlugin` — owns queue only, no LLM knowledge |
| `BackgroundTask` — carries LLM instructions | `Task` — carries `work: async () -> str` |
| `AgentPlugin._process_queue_item` — runs full agent loop | Runs one agent turn, dispatches tasks, returns |
| `_tool_call_id` injection via inspect | `_ctx: ToolContext` injection (one param) |
| `background_task` tool — LLM-specific closure | `subagent` tool — constructs agent loop, enqueues as Task |
| `pm.agent_plugin` / `pm.background` cross-refs | TaskQueue available via ToolContext; no cross-refs |
| No max_turns per channel | `Channel.turn_counter` reset on user message |

### Target layout

```
sherman/
├── hooks.py              # AgentSpec, hookimpl, create_plugin_manager
├── tool.py               # Tool, ToolRegistry, tool_to_schema, ToolContext
├── channel.py            # Channel (+ turn_counter), ChannelConfig, ChannelRegistry
├── queue.py              # SerialQueue
├── llm.py                # LLMClient
├── agent_loop.py         # run_agent_turn() + run_agent_loop() (kept for subagent)
├── conversation.py       # ConversationLog, init_db, resolve_system_prompt
├── logging.py            # StructuredFormatter, _DEFAULT_LOGGING
├── agent.py              # AgentPlugin (single-turn dispatch, no blocking loop)
├── task.py               # Task, TaskQueue, TaskPlugin
├── main.py
├── channels/
│   ├── cli.py
│   └── irc.py
└── tools/
    ├── __init__.py        # CoreToolsPlugin
    ├── shell.py
    ├── files.py
    ├── web.py
    └── subagent.py        # subagent tool
```

### Deleted

- `background.py` — replaced by `task.py`
- Legacy delegation stubs in `agent.py`
- `_truncate` private import (make it public or move to shared util)
- `tool_to_schema` re-export from `agent_loop.py`
- `_DEFAULT_LOGGING` re-export from `main.py`
- `QueueItem` re-export hack in `queue.py`

### Migration path

The change to single-turn dispatch is significant — all tests that mock
`run_agent_loop` or assert on multi-turn behavior in AgentPlugin will
need rewriting. Tests for `run_agent_loop` itself stay (it's preserved
for subagent use).

Suggested phases:

1. **Task + ToolContext**: Create `task.py` with Task, TaskQueue, TaskPlugin.
   Create ToolContext in `tool.py`. Wire ToolContext injection into
   `run_agent_loop` (replacing `_tool_call_id` inspection). All existing
   tests pass — ToolContext is additive.

2. **run_agent_turn**: Add `run_agent_turn()` alongside `run_agent_loop()`.
   No callers yet — just the function + its own tests.

3. **AgentPlugin single-turn**: Switch AgentPlugin from `run_agent_loop`
   to `run_agent_turn` + task dispatch. Update/rewrite AgentPlugin tests.
   This is the big phase.

4. **subagent tool**: Create `tools/subagent.py`. Register via
   CoreToolsPlugin or its own plugin. Tests.

5. **Cleanup**: Delete BackgroundPlugin, legacy stubs, stale re-exports.
   Update MEMORY.md.

### Risk assessment

**Highest risk: Phase 3.** AgentPlugin test rewrite. The existing tests
assert on `run_agent_loop` call args and multi-turn behavior. The new
tests need to assert on task dispatch and notification-driven continuation.
~40 tests affected.

**Medium risk: Tool result ordering.** Without batching, the LLM sees
tool results one at a time. If it requested 3 tool calls, it gets 3
separate notification-triggered turns. This may confuse some models or
produce chattier behavior. Accepted as a YOLO — add batching later if
needed.

**Medium risk: subagent tool needs default LLM config.** The tool needs
to know what model/base_url to use when the LLM doesn't specify. This
config needs to flow from `agent.yaml` to the tool somehow — either via
ToolContext or closure at registration time.

**Low risk: ToolContext injection.** Same mechanism as `_tool_call_id`
but simpler (one param instead of N). Backward compatible — tools
without `_ctx` work unchanged.
