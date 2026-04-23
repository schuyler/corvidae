# Phase 3 Design: Starter Tools and Background Tasks

## 1. Summary

Phase 3 adds two new modules (`sherman/tools.py`, `sherman/background.py`), modifies two existing
files (`sherman/agent_loop_plugin.py`, `sherman/main.py`), and adds two new test files
(`tests/test_tools.py`, `tests/test_background.py`).

Deliverables:
- Four core tools: `shell`, `read_file`, `write_file`, `web_fetch`
- A background task system: `BackgroundTask` dataclass, `TaskQueue` class
- Two task-management tools: `background_task`, `task_status`
- A background worker in `AgentLoopPlugin` that executes queued tasks via `run_agent_loop()` with a
  dedicated system prompt

**New files:**
- `sherman/tools.py` — CoreToolsPlugin with four stateless tool functions
- `sherman/background.py` — BackgroundTask dataclass, TaskQueue class
- `tests/test_tools.py` — tests for core tools
- `tests/test_background.py` — tests for background system

**Modified files:**
- `sherman/agent_loop_plugin.py` — add TaskQueue, background worker, task tool closures
- `sherman/main.py` — register CoreToolsPlugin before AgentLoopPlugin

**Unchanged files:** `hooks.py`, `plugin_manager.py`, `channel.py`, `llm.py`, `agent_loop.py`,
`conversation.py`

## 2. Discrepancies with design.md

### 1. `run_worker` async generator pattern (design.md lines 760–777)

The sketch uses `yield task, result` inside a `while True` loop. `asyncio.create_task()` cannot run
an async generator. The `async for` consumer pattern works syntactically, but exception handling has
gaps — error results can be lost.

**Resolution:** Make `run_worker` a plain coroutine that accepts an `on_complete` callback. Same
FIFO-with-callback semantics, no async generator complexity.

### 2. design.md references `Runner.run()` (line 714)

The project uses hand-rolled `run_agent_loop()`, not the OpenAI Agents SDK.

**Resolution:** Use `run_agent_loop()` for background task execution.

### 3. CoreToolsPlugin registers background_task and task_status (design.md lines 982–992)

Both tools need `TaskQueue` access, which lives on `AgentLoopPlugin`. The design.md itself
acknowledges this ("Injected by the agent loop plugin", line 975; "The actual enqueue happens in
the agent loop plugin", line 804).

**Resolution:** `CoreToolsPlugin` registers only the four stateless tools (`shell`, `read_file`,
`write_file`, `web_fetch`). `AgentLoopPlugin` creates and registers the two task tools
(`background_task`, `task_status`) during `on_start`, after `TaskQueue` initialization. They are
closure functions that capture `self`.

### 4. Channel context injection for background_task

The LLM calls `background_task(description, instructions)` but the tool needs to know which channel
the request originated from.

**Resolution:** `AgentLoopPlugin.on_message` creates a local `background_task` closure that captures
the local `channel` variable. It builds a `local_tools` dict — a copy of `self.tools` with this
per-call closure replacing the shared placeholder — and passes `local_tools` to `run_agent_loop()`.

The `background_task` registered during `on_start` is a placeholder that is always overridden in
`on_message`. Each call to `on_message` produces its own independent closure, so concurrent calls on
different channels cannot interfere.

**Concurrency safety:** `run_agent_loop()` has multiple `await` points — `client.chat()` and each
`await tools[fn_name](**args)` call. The event loop can interleave concurrent `on_message` calls at
any of these points. Sharing `self._current_channel` across calls would therefore produce incorrect
channel routing. Per-call closures capture `channel` by value at call time and are not shared between
concurrent invocations, so no locking is needed.

### 5. design.md main.py shows AgentLoopPlugin(pm, config) taking two args

Current code takes only `pm`.

**Resolution:** Keep current signature `AgentLoopPlugin(pm)`.

## 3. Core Tools Plugin (`sherman/tools.py`)

### Module structure

Four module-level async tool functions + a `CoreToolsPlugin` class with a `register_tools`
hookimpl.

### Tool functions

```python
async def shell(command: str) -> str:
    """Execute a shell command and return the output."""
    # Uses asyncio.create_subprocess_shell with 30s timeout
    # On timeout: kills process, returns error message
    # Includes STDERR section if stderr is non-empty
    # Includes exit code if non-zero
    # Returns "(no output)" if both stdout and stderr are empty

async def read_file(path: str) -> str:
    """Read the contents of a file."""
    # Returns error string if: path doesn't exist, is not a file, or is >1MB
    # Returns file contents on success

async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    # Creates parent dirs with mkdir(parents=True, exist_ok=True)
    # Returns "Wrote N chars to path" on success
    # Returns error string on failure

async def web_fetch(url: str) -> str:
    """Fetch a URL and return its text content."""
    # Creates a new aiohttp.ClientSession per call (15s timeout)
    # Returns "HTTP {status}" for non-200 responses
    # Truncates responses >50000 chars with "[truncated]" marker
    # Catches asyncio.TimeoutError and aiohttp.ClientError
```

### CoreToolsPlugin

```python
class CoreToolsPlugin:
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        tool_registry.extend([shell, read_file, write_file, web_fetch])
```

Design decisions:
- Error conditions return error strings rather than raising. `run_agent_loop()` already catches
  exceptions, but structured error messages are clearer for the LLM.
- `shell` kills the process on timeout rather than leaving it orphaned.
- `read_file` adds an `is_file()` check to prevent confusing errors on directories.
- `web_fetch` catches `aiohttp.ClientError` for DNS failures, connection refused, etc.
- `background_task` and `task_status` are NOT registered here — they go in `AgentLoopPlugin`.

## 4. Background Task System (`sherman/background.py`)

### BackgroundTask dataclass

```python
@dataclass
class BackgroundTask:
    channel: Channel
    description: str
    instructions: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time)
```

### TaskQueue class

```python
class TaskQueue:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[BackgroundTask] = asyncio.Queue()
        self.active_task: BackgroundTask | None = None
        self.completed: dict[str, str] = {}  # task_id -> result

    async def enqueue(self, task: BackgroundTask) -> None:
        """Add a task to the queue."""

    async def run_worker(
        self,
        execute_fn: Callable[[BackgroundTask], Awaitable[str]],
        on_complete: Callable[[BackgroundTask, str], Awaitable[None]],
    ) -> None:
        """Pull tasks from the queue and execute them one at a time.

        Runs forever (while True). Cancelled via asyncio.Task.cancel()
        during shutdown.
        """

    def status(self) -> str:
        """Return a human-readable status summary.

        Shows active task, pending count, and last 3 completed results.
        """
```

Design decisions:
- `run_worker` is a plain coroutine with `on_complete` callback (not an async generator).
- `task_id` defaults to `uuid.uuid4().hex[:12]` — 12 hex chars.
- `status()` is a method on `TaskQueue`. The `task_status` tool closure calls it.
- Error results are stored in `completed` dict so `task_status` can report failures.
- No cap on completed dict — acceptable for single-instance daemon.

### run_worker error handling

When `execute_fn` raises an exception:
1. Store `f"Task {task.task_id} failed: {exc}"` in `self.completed[task.task_id]`.
2. Call `on_complete(task, error_string)`.
3. Continue the loop (process the next queued task).

## 5. Task Tools (background_task, task_status)

Both are closure functions. `task_status` is created in `AgentLoopPlugin.on_start`, capturing
`self`. `background_task` is created per-call in `on_message`, capturing the local `channel`.

```python
# In AgentLoopPlugin.on_start, after TaskQueue init:

async def background_task(description: str, instructions: str) -> str:
    """Launch a long-running task in the background."""
    # Placeholder — overridden per-call in on_message with a channel-capturing closure.
    raise RuntimeError("background_task placeholder called outside on_message context")

async def task_status() -> str:
    """Check the status of background tasks."""
    return self.task_queue.status()

self.tools["background_task"] = background_task
self.tool_schemas.append(tool_to_schema(background_task))
self.tools["task_status"] = task_status
self.tool_schemas.append(tool_to_schema(task_status))
```

```python
# In AgentLoopPlugin.on_message, before calling run_agent_loop():

async def background_task(description: str, instructions: str) -> str:
    """Launch a long-running task in the background."""
    task = BackgroundTask(
        channel=channel,  # captures local variable, not shared state
        description=description,
        instructions=instructions,
    )
    await self.task_queue.enqueue(task)
    return f"Task {task.task_id} enqueued: {description}"

local_tools = {**self.tools, "background_task": background_task}
# Pass local_tools (not self.tools) to run_agent_loop()
```

Channel context injection: each `on_message` call creates its own `background_task` closure that
captures the local `channel` variable. Concurrent calls on different channels each get an independent
closure, so channel routing is safe regardless of `await` interleaving within `run_agent_loop()`.

## 6. AgentLoopPlugin Modifications

### New instance attributes

```python
self.task_queue: TaskQueue | None = None
self._worker_task: asyncio.Task | None = None
```

### on_start additions

After existing tool collection:
1. Initialize `self.task_queue = TaskQueue()`
2. Create `background_task` and `task_status` closures
3. Add them to `self.tools` and `self.tool_schemas`
4. Start worker: `self._worker_task = asyncio.create_task(self.task_queue.run_worker(...))`

### on_message modification

Before calling `run_agent_loop()`, create a per-call `background_task` closure that captures the
local `channel` variable. Build `local_tools = {**self.tools, "background_task": background_task}`
and pass `local_tools` to `run_agent_loop()` instead of `self.tools`. No shared mutable state.

### New methods

```python
async def _execute_background_task(self, task: BackgroundTask) -> str:
    """Run a background task with its own conversation context."""
    # Exclude background_task to prevent unbounded recursive task creation.
    bg_tools = {k: v for k, v in self.tools.items() if k != "background_task"}
    bg_schemas = [s for s in self.tool_schemas if s["function"]["name"] != "background_task"]
    messages = [
        {"role": "system", "content": "You are executing a background task. "
         "Work through the instructions step by step. Be thorough."},
        {"role": "user", "content": task.instructions},
    ]
    return await run_agent_loop(
        self.client, messages, bg_tools, bg_schemas
    )
```

Background worker uses `self.tools` (with `background_task` filtered out), not `local_tools`, because
background tasks run outside the context of any `on_message` call.

async def _on_task_complete(self, task: BackgroundTask, result: str) -> None:
    """Handle a completed background task."""
    display_result = strip_thinking(result)
    await self.pm.ahook.on_task_complete(
        channel=task.channel, task_id=task.task_id, result=display_result,
    )
    await self.pm.ahook.send_message(
        channel=task.channel, text=f"[Task {task.task_id}] {display_result}",
    )
```

`_on_task_complete` is responsible for calling `send_message`. The `on_task_complete` hook is a
notification only — the same pattern as `on_agent_response` + `send_message` in the foreground path.
Transport plugins implementing `on_task_complete` must NOT call `send_message`.

### on_stop modification

Cancel worker task before closing client:

```python
if self._worker_task:
    self._worker_task.cancel()
    try:
        await self._worker_task
    except asyncio.CancelledError:
        pass
# Then existing client/db cleanup
```

Worker cancellation during an in-flight `client.chat()` is safe — aiohttp handles `CancelledError`
gracefully and `session.close()` is idempotent.

### Note on recursive task creation

`_execute_background_task` filters `background_task` out of the tools dict it passes to
`run_agent_loop()`. This prevents unbounded recursive task creation:

```python
async def _execute_background_task(self, task: BackgroundTask) -> str:
    bg_tools = {k: v for k, v in self.tools.items() if k != "background_task"}
    ...
    return await run_agent_loop(
        self.client, messages, bg_tools, ...
    )
```

Background tasks can still use all other tools, including `task_status`.

## 7. main.py Modifications

```python
from sherman.tools import CoreToolsPlugin

# In main():
# Register CoreToolsPlugin BEFORE AgentLoopPlugin
core_tools = CoreToolsPlugin()
pm.register(core_tools, name="core_tools")

# Then register AgentLoopPlugin (collects tools from already-registered plugins)
agent_loop = AgentLoopPlugin(pm)
pm.register(agent_loop, name="agent_loop")
```

## 8. Test Plan

### tests/test_tools.py

**TestShell:**
1. `test_shell_simple_command` — `echo hello`, verify "hello" in output
2. `test_shell_returns_stderr` — command writing to stderr, verify STDERR section
3. `test_shell_nonzero_exit_code` — `exit 1`, verify exit code in output
4. `test_shell_timeout` — mock `asyncio.wait_for` to raise `TimeoutError`, verify message is
   `"Error: command timed out after 30 seconds"`
5. `test_shell_no_output` — `true`, verify "(no output)"

Mock strategy: Tests 1–3, 5 use real subprocess. Test 4 mocks to avoid real wait.

**TestReadFile:**
1. `test_read_file_success` — tmp file, verify contents
2. `test_read_file_not_found` — nonexistent path, verify error
3. `test_read_file_too_large` — mock stat >1MB, verify error
4. `test_read_file_directory` — directory path, verify error

Mock strategy: `tmp_path` fixture.

**TestWriteFile:**
1. `test_write_file_success` — write to tmp dir, read back
2. `test_write_file_creates_parents` — nested path, verify parents created
3. `test_write_file_overwrites` — write twice, verify second wins
4. `test_write_file_permission_error` — read-only directory, verify error

Mock strategy: `tmp_path` fixture.

**TestWebFetch:**
1. `test_web_fetch_success` — mock 200, verify body
2. `test_web_fetch_non_200` — mock 404, verify "HTTP 404"
3. `test_web_fetch_truncates_large_response` — mock >50000 chars, verify truncation
4. `test_web_fetch_timeout` — mock `TimeoutError`, verify message
5. `test_web_fetch_connection_error` — mock `ClientError`, verify message

Mock strategy: Patch `aiohttp.ClientSession` with `AsyncMock`.

**TestCoreToolsPlugin:**
1. `test_register_tools_adds_four_tools` — verify 4 functions added
2. `test_registered_tool_names` — verify names: `shell`, `read_file`, `write_file`, `web_fetch`

### tests/test_background.py

**TestToolToSchema:**
1. `test_tool_to_schema_zero_params` — call `tool_to_schema` on a parameterless async function
   (e.g., `async def noop() -> str: ...`); verify the schema has
   `"parameters": {"properties": {}, "type": "object"}` and no `"required"` key. This covers
   `task_status`, which has no parameters.

**TestBackgroundTask:**
1. `test_task_id_auto_generated` — verify 12-char hex string
2. `test_task_fields` — verify all fields with explicit values
3. `test_created_at_auto_set` — verify close to `time.time()`

**TestTaskQueue:**
1. `test_enqueue_and_dequeue` — enqueue, start worker, verify `execute_fn` receives task
2. `test_fifo_ordering` — enqueue A, B, C; verify order
3. `test_active_task_tracking` — verify `active_task` during execution, `None` after
4. `test_completed_dict` — verify result stored in `completed`
5. `test_execute_fn_error` — raise in `execute_fn`, verify error stored, `on_complete` called,
   worker continues
6. `test_worker_cancellation` — cancel worker, verify clean `CancelledError`
7. `test_status_no_tasks` — empty queue, verify "No tasks."
8. `test_status_with_active` — during execution, status shows active task
9. `test_status_with_completed` — after completion, status lists results

Mock strategy: `AsyncMock` for `execute_fn` and `on_complete`. `asyncio.Event` for timing control.

### Required existing test updates

1. `test_on_start_collects_tools` in `tests/test_agent_loop_plugin.py` currently asserts
   `len(plugin.tool_schemas) == 1`. After Phase 3, `on_start` adds `background_task` and `task_status`
   closures, making the expected count 3. Update the assertion to reflect the new tool count.

2. `test_on_message_calls_run_agent_loop` in `tests/test_agent_loop_plugin.py` asserts
   `call_kwargs[0][2] is plugin.tools` (identity check). After Phase 3, `on_message` passes `local_tools`
   (a new dict copy created per call), not `plugin.tools`, so the identity check will fail. Update the
   test to verify the tools dict contains the expected tools (e.g., checking that `local_tools` contains
   the core tools and the per-call `background_task` closure) rather than checking object identity.

**TestBackgroundTaskTool (integration):**
1. `test_background_task_enqueues` — call closure, verify task in queue via
   `queue.queue.get_nowait()` after enqueue, checking the task's fields
2. `test_background_task_uses_current_channel` — verify correct channel
3. `test_task_status_reports_correctly` — verify status output

**TestBackgroundWorkerIntegration:**
1. `test_worker_executes_with_agent_loop` — mock `run_agent_loop`, verify `send_message` called
2. `test_worker_posts_to_correct_channel` — two tasks, different channels, verify routing;
   use `on_complete` callback to signal an `asyncio.Event`, await the event before asserting
3. `test_on_task_complete_hook_fired` — verify hook called with correct args
4. `test_on_stop_cancels_worker` — verify worker cancelled on stop

## 9. Assumptions

1. Hardcoded background system prompt is acceptable for Phase 3.
2. Session-per-call for `web_fetch`. Shared session deferred.
3. No cap on completed task dict. Acceptable for single-instance daemon.
4. `background_task` is filtered out of tools passed to `_execute_background_task`; sub-tasks are not supported in Phase 3.
5. Per-call `background_task` closures capture `channel` by value; no shared mutable channel state.

## 10. Implementation Sequence

1. `sherman/background.py` — `BackgroundTask`, `TaskQueue` (no deps on new code)
2. `sherman/tools.py` — four tool functions + `CoreToolsPlugin` (no deps on background)
3. `sherman/agent_loop_plugin.py` — `TaskQueue` init, task tool closures, per-call `local_tools`
   in `on_message`, background worker
4. `sherman/main.py` — `CoreToolsPlugin` registration
5. Tests for all of the above
