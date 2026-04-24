# Phase 1 Design: Task + ToolContext

## 1. Scope

Create `task.py` with Task dataclass, TaskQueue, and TaskPlugin. Create
ToolContext in `tool.py`. Wire ToolContext injection into `run_agent_loop`
(replacing `_tool_call_id` inspection). All existing tests continue to
pass — ToolContext is additive, no deletions.

## 2. File Changes

### New files

| File | Contents |
|------|----------|
| `sherman/task.py` | `Task` dataclass, `TaskQueue` class, `TaskPlugin` class |
| `tests/test_task.py` | Tests for Task, TaskQueue, TaskPlugin |

### Modified files

| File | Change |
|------|--------|
| `sherman/tool.py` | Add `ToolContext` dataclass |
| `sherman/agent_loop.py` | Add `channel`/`task_queue` keyword-only params to `run_agent_loop`; inject `_ctx: ToolContext`; keep `_tool_call_id` injection |
| `sherman/agent.py` | Pass `channel` and `task_queue` to `run_agent_loop` in `_process_queue_item` |
| `sherman/main.py` | Register `TaskPlugin` before `AgentPlugin` |

### Unchanged files

`background.py`, `hooks.py`, all existing tests, all tools, all channels.

## 3. ToolContext

**Location**: bottom of `sherman/tool.py`, after `ToolRegistry`.

Add `from __future__ import annotations` at the top of `tool.py` (enables
string-based annotation evaluation, avoids runtime circular imports).

Add TYPE_CHECKING block:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sherman.channel import Channel
    from sherman.task import TaskQueue
```

The dataclass:

```python
@dataclass
class ToolContext:
    """Context injected into tools that declare a ``_ctx`` parameter.

    Constructed per tool call by run_agent_loop. Tools without ``_ctx``
    work exactly as before.

    Attributes:
        channel: The channel this tool call is executing on. None when
            run_agent_loop is called without channel context (e.g.,
            background task sub-agent loops in Phase 1).
        tool_call_id: The LLM-assigned call ID for this invocation.
        task_queue: The TaskQueue for enqueueing background work. None
            when no TaskPlugin is registered.
    """
    channel: Channel | None
    tool_call_id: str
    task_queue: TaskQueue | None
```

**Schema exclusion**: `_ctx` starts with `_`, so `tool_to_schema()` already
skips it (line 38 of `tool.py`). No change needed.

## 4. Task Dataclass

**Location**: `sherman/task.py`

### Imports for task.py

```python
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sherman.channel import Channel

from sherman.hooks import hookimpl
```

`Channel` is imported under `TYPE_CHECKING` only. With `from __future__ import
annotations`, the `Task.channel: Channel` annotation is a deferred string —
no runtime import needed. `Channel` objects are passed in by callers at
runtime, so attribute access (e.g., `task.channel.id`) works without the
import being resolved at module load time.

### Task

```python
@dataclass
class Task:
    """A unit of async work with delivery context.

    The queue calls ``await task.work()``, catches exceptions, and
    delivers the result string via the completion callback.

    Attributes:
        work: Async callable returning a result string.
        channel: Channel to deliver results to.
        task_id: Unique identifier (auto-generated 12-char hex).
        created_at: Unix timestamp of creation.
        tool_call_id: LLM tool call ID for deferred result delivery.
            None if not triggered by a tool call.
        description: Human-readable label for status display.
    """
    work: Callable[[], Awaitable[str]]
    channel: Channel
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time)
    tool_call_id: str | None = None
    description: str = ""
```

Key difference from `BackgroundTask`: carries `work` (async callable) instead
of `instructions` (string). The queue doesn't know what `work` does.

## 5. TaskQueue

**Location**: `sherman/task.py`

```python
class TaskQueue:
    """Async worker queue that processes Tasks one at a time.

    Unlike background.TaskQueue which takes external execute_fn and
    on_complete callbacks, this one calls ``await task.work()`` directly.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.active_task: Task | None = None
        self.completed: dict[str, str] = {}  # task_id -> result

    async def enqueue(self, task: Task) -> None: ...
    async def run_worker(
        self,
        on_complete: Callable[[Task, str], Awaitable[None]],
    ) -> None: ...
    def status(self) -> str: ...
```

### enqueue

Log DEBUG "task enqueued" with extra `task_id`, `channel` (task.channel.id),
`description`. Then `await self.queue.put(task)`.

### run_worker

```
while True:
    task = await self.queue.get()
    self.active_task = task
    log DEBUG "task started" (task_id, description)
    result = f"Task {task.task_id} failed: (unknown error)"
    try:
        result = await task.work()
        log DEBUG "task completed" (task_id, result_length)
    except asyncio.CancelledError:
        self.active_task = None
        raise
    except Exception as exc:
        log WARNING "task failed" (task_id, exc_info=True)
        result = f"Task {task.task_id} failed: {exc}"
    finally:
        self.queue.task_done()
    self.completed[task.task_id] = result
    self.active_task = None
    await on_complete(task, result)
```

This is structurally identical to `background.TaskQueue.run_worker` but calls
`task.work()` instead of `execute_fn(task)`.

### status

Same implementation as `background.TaskQueue.status()`.

### Logger

`logging.getLogger(__name__)` — resolves to `sherman.task`.

## 6. TaskPlugin

**Location**: `sherman/task.py`

```python
class TaskPlugin:
    """Plugin owning the new TaskQueue.

    In Phase 1, coexists with BackgroundPlugin. TaskPlugin registers no
    tools (avoids name collision with BackgroundPlugin's task_status).
    Tool registration deferred to Phase 3+.
    """

    def __init__(self, pm) -> None:
        self.pm = pm
        self.task_queue: TaskQueue | None = None
        self._worker_task: asyncio.Task | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        self.task_queue = TaskQueue()
        self.pm.task_plugin = self  # attach for discovery

        async def _complete_wrapper(task: Task, result: str) -> None:
            return await self._on_task_complete(task, result)

        self._worker_task = asyncio.create_task(
            self.task_queue.run_worker(_complete_wrapper)
        )
        logger.debug("TaskPlugin started")

    @hookimpl
    async def on_stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.debug("TaskPlugin stopped")

    async def _on_task_complete(self, task: Task, result: str) -> None:
        """Deliver completed task result via hooks."""
        logger.debug(
            "task complete, dispatching notification",
            extra={
                "task_id": task.task_id,
                "channel": task.channel.id,
                "result_length": len(result),
            },
        )
        await self.pm.ahook.on_notify(
            channel=task.channel,
            source="task",
            text=f"[Task {task.task_id}] {result}",
            tool_call_id=task.tool_call_id,
            meta={"task_id": task.task_id},
        )
```

**No tools registered in Phase 1.** BackgroundPlugin already registers
`task_status`. Adding a second one would cause a name collision in the
ToolRegistry. Deferred to Phase 5 when BackgroundPlugin is deleted.

**No register_tools hookimpl.** TaskPlugin does not implement register_tools.

**No on_task_complete hook.** The design spec's `on_task_complete` hookspec
doesn't exist yet. In Phase 1, TaskPlugin only fires the existing `on_notify`
hook. Adding new hookspecs is deferred to later phases when consumers exist.

## 7. ToolContext Injection in run_agent_loop

### Signature change

```python
async def run_agent_loop(
    client: LLMClient,
    messages: list[dict],
    tools: dict[str, Callable],
    tool_schemas: list[dict],
    max_turns: int = 10,
    *,
    channel: "Channel | None" = None,
    task_queue: "TaskQueue | None" = None,
) -> str:
```

New keyword-only params with None defaults. All existing callers pass
positional/keyword args only for the first 5 params — fully backward
compatible.

### Import additions to agent_loop.py

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sherman.channel import Channel
    from sherman.task import TaskQueue

from sherman.tool import ToolContext, tool_to_schema  # modify existing line 23 import to add ToolContext
```

**Note**: This modifies the existing `from sherman.tool import tool_to_schema`
import at line 23 — it is NOT a new import line.

### Tool dispatch change (replaces lines 124-128 only — the injection block inside the existing try)

The surrounding `try:`, `tool_start = time.monotonic()`, `tool_fn = tools[fn_name]`
(lines 118-123) and the timing/logging/`except` block (lines 129-145) are
**preserved unchanged**. Only the parameter injection and call lines are replaced:

```python
tool_sig = inspect.signature(tool_fn)
call_kwargs = dict(args)

# Inject ToolContext for tools that declare _ctx
if "_ctx" in tool_sig.parameters:
    call_kwargs["_ctx"] = ToolContext(
        channel=channel,
        tool_call_id=call_id,
        task_queue=task_queue,
    )

# Backward compat: inject _tool_call_id (removed in Phase 5)
if "_tool_call_id" in tool_sig.parameters:
    call_kwargs["_tool_call_id"] = call_id

content = await tool_fn(**call_kwargs)
```

Both injections coexist. A tool declaring both `_ctx` and `_tool_call_id`
gets both (redundant but harmless).

## 8. AgentPlugin Wiring

In `_process_queue_item` (around line 238 of `agent.py`), change the
`run_agent_loop` call:

```python
# Resolve new task queue from TaskPlugin
task_queue_ref = getattr(
    getattr(self.pm, "task_plugin", None), "task_queue", None
)

raw_response = await run_agent_loop(
    self.client, messages, local_tools, self.tool_schemas,
    channel=channel,
    task_queue=task_queue_ref,
)
```

The `channel` variable is already available (line 169). The `task_queue`
is resolved from `pm.task_plugin` using the same `getattr` pattern as
the existing `pm.background` lookup (line 205).

**The existing `task_queue` variable at line 205 and the `background_task`
closure (lines 207-228) are left unchanged.** `task_queue_ref` is a new
variable added immediately before the `run_agent_loop` call at line 238.
The two variables refer to different queues: `task_queue` is the
BackgroundPlugin queue (used by the `background_task` closure),
`task_queue_ref` is the new TaskPlugin queue (passed to `run_agent_loop`
for ToolContext injection).

**BackgroundPlugin._execute_task**: Does NOT pass channel/task_queue.
Sub-agent loops run without ToolContext. No tools in the sub-agent tool
set declare `_ctx`, so no injection occurs.

## 9. main.py Registration

Add after BackgroundPlugin registration (line 93), before AgentPlugin
(line 96):

```python
from sherman.task import TaskPlugin

# Register TaskPlugin before AgentPlugin (provides new task queue)
task_plugin = TaskPlugin(pm)
pm.register(task_plugin, name="task")
```

## 10. Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Tool without `_ctx` | No injection. Works as before. |
| Tool with `_tool_call_id` | Still injected. Backward compat preserved. |
| `tool_to_schema` on tool with `_ctx` | `_ctx` excluded (starts with `_`). |
| `run_agent_loop` called without channel/task_queue | Works as before. `_ctx` gets None fields. |
| Existing tests | All pass unchanged. New params are keyword-only with defaults. |

## 11. Risks and Open Questions

### Import circularity (low risk)
Runtime import graph: `task.py` -> `channel.py`, `agent_loop.py` -> `tool.py`.
TYPE_CHECKING only: `tool.py` -> `channel.py`, `task.py`.
No cycles. Verified by tracing.

### ToolContext with None fields (low risk, Phase 1 only)
When `run_agent_loop` called without channel/task_queue (e.g., from
`BackgroundPlugin._execute_task`), any tool declaring `_ctx` gets None
fields. In Phase 1, no existing tools declare `_ctx`. In later phases,
tools using `_ctx` must handle None or callers must always provide values.

### TaskPlugin + BackgroundPlugin coexistence (accepted)
Both plugins own separate queues. They don't interact. The old queue
handles `background_task` tool calls. The new queue is available via
ToolContext but unused until Phase 3 when tools start using `_ctx` to
enqueue Tasks.

### _tool_call_id removal timeline
Kept for backward compat in Phase 1. The `background_task` closure in
`agent.py` (line 208) uses `_tool_call_id`. Remove in Phase 5 after
`background_task` is replaced by the `subagent` tool.

### ToolContext field growth
The redesign spec mentions subagent tool needing LLM config and tool set.
These could be added to ToolContext in Phase 4. For Phase 1, three fields
suffice: channel, tool_call_id, task_queue.

## 12. Test Plan

### tests/test_task.py (new)

**TestTask**:
- `test_task_id_auto_generated` — 12-char hex string
- `test_task_fields` — all fields set correctly
- `test_created_at_auto_set` — timestamp between before/after

**TestTaskQueue**:
- `test_enqueue_and_dequeue` — worker receives task, calls task.work()
- `test_fifo_ordering` — 3 tasks processed in order
- `test_active_task_tracking` — set during execution, None before/after
- `test_completed_dict` — result stored by task_id
- `test_work_error` — exception stored as error string, worker continues
- `test_worker_cancellation` — CancelledError raised cleanly
- `test_status_no_tasks` — "no tasks" message
- `test_status_with_active` — shows active task
- `test_status_with_completed` — shows completed results

**TestTaskQueueLogging**:
- `test_enqueue_logs_debug` — "task enqueued" with task_id, channel, description
- `test_run_worker_logs_task_started`
- `test_run_worker_logs_task_completed` — with result_length
- `test_run_worker_logs_task_failed_warning` — with exc_info

**TestTaskPlugin**:
- `test_on_start_creates_queue_and_worker`
- `test_on_stop_cancels_worker`
- `test_on_task_complete_fires_on_notify` — on_notify called with correct args
- `test_on_task_complete_does_not_fire_on_task_complete_hook` — intentional: TaskPlugin fires only on_notify, not on_task_complete (deferred to later phases)
- `test_pm_task_plugin_attached` — pm.task_plugin set during on_start

### tests/test_agent_loop.py (additions)

**ToolContext injection tests** (appended to existing file):
- `test_ctx_injected_when_declared` — tool with `_ctx: ToolContext` receives it
- `test_ctx_not_injected_when_not_declared` — tool without `_ctx` unchanged
- `test_ctx_has_correct_tool_call_id` — matches the call's ID
- `test_ctx_has_correct_channel` — matches passed channel
- `test_ctx_has_correct_task_queue` — matches passed task_queue
- `test_ctx_channel_and_task_queue_none_when_not_provided` — channel=None, task_queue=None (tool_call_id is always set)
- `test_ctx_excluded_from_schema` — `_ctx` not in tool schema
- `test_tool_call_id_still_injected` — backward compat with _ctx coexisting

### No modifications to existing tests

All 267 existing tests pass without changes.

## 13. Implementation Sequence

1. Add `ToolContext` to `sherman/tool.py` (+ `from __future__` and TYPE_CHECKING imports)
2. Create `sherman/task.py` with `Task`, `TaskQueue`, `TaskPlugin`
3. Modify `sherman/agent_loop.py`: add keyword-only params, add `_ctx` injection
4. Modify `sherman/agent.py`: pass `channel` and `task_queue` to `run_agent_loop`
5. Modify `sherman/main.py`: import and register `TaskPlugin`
6. Write `tests/test_task.py`
7. Add ToolContext injection tests to `tests/test_agent_loop.py`
8. Run full test suite — all 267+ tests must pass

---

--- Design Report ---

Design ready for review: **yes**

Phase 1 is additive and backward-compatible. All existing code paths
preserved. New components wired in but not consumed by existing tools
until later phases.

---

## Design Review Report

Reviewed by: Chico (code review agent)

### Findings

**Critical (2) — both fixed inline above:**

- **C1** (Section 7): "replaces lines 118-128" corrected to "replaces lines
  124-128 only." Lines 118-123 (try, tool_start, tool_fn lookup) and 129-145
  (timing/logging/except) are preserved. ✅ Fixed.

- **C2** (Section 8): Added explicit statement that the existing `task_queue`
  variable at line 205 and the `background_task` closure are untouched.
  `task_queue_ref` is a new, separate variable. ✅ Fixed.

**Important (4) — all fixed inline above:**

- **I1**: Added imports section for `task.py` with explicit `Channel` under
  `TYPE_CHECKING`. ✅ Fixed.

- **I2**: Renamed `test_ctx_none_fields_when_not_provided` to
  `test_ctx_channel_and_task_queue_none_when_not_provided`. ✅ Fixed.

- **I3**: Added `test_on_task_complete_does_not_fire_on_task_complete_hook`
  to make the intentional omission explicit. ✅ Fixed.

- **I4**: Clarified that the `ToolContext` import modifies the existing line 23
  import, not a new line. ✅ Fixed.

**Cosmetic (2) — not fixed, acceptable as-is:**

- M1: `_complete_wrapper` comment explaining monkey-patching support.
- M2: Test name `test_work_error` vs `test_task_work_error`.

### Verdict after fixes: **PASS**

All critical and important findings have been addressed in the design document.

---

--- Documentation Report ---

Updated `.claude/MEMORY.md`:
- Added `task.py` to the file map
- Added `test_task.py` and `test_tool_context.py` to the test file map
- Updated test count from 268 to 298
- Updated Current Status to reflect Phase 1 completion and reframe next work as Phase 2+
- Updated `_`-prefixed tool parameters decision to cover ToolContext injection alongside `_tool_call_id`
- Added ToolContext architectural decision entry

Proceed: **yes**
