"""Task system for the corvidae agent daemon.

Contains:
  - Task: dataclass representing a unit of async work with delivery context
  - TaskQueue: async worker queue that processes Tasks one at a time;
    optionally persists each completion to a JSONL file (results_path)
    so results survive deque eviction and daemon restarts
  - TaskPlugin: plugin that owns the TaskQueue and delivers results via on_notify
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.channel import Channel

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.outbound import OutboundToolStatus
from corvidae.tool import Tool

logger = logging.getLogger(__name__)

# Result string used when a task raises an unhandled exception.
# Format with task_id=<str> and error=<exc or "(unknown error)">.
TASK_FAILURE_TEMPLATE = "Task {task_id} failed: {error}"
# Number of completed task records shown in TaskQueue.status() output.
STATUS_HISTORY_COUNT = 3


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
        ctx: contextvars.Context captured at Task creation time (i.e. in
            the enqueuing caller's context). The worker runs the work
            inside this context so attribution set by the enqueuer is
            visible to the task body — worker coroutines are created once
            at startup and would otherwise never see it.
        trace_id: The trace this task belongs to. Stamped by
            Agent._dispatch_tool_calls from the current turn's attribution
            so the tool cycle carries its trace forward.
        annotations: Extensible annotations, opaque to core. Stamped alongside
            trace_id and merged into the completion notification's
            annotations by TaskPlugin.
        deliver: When False, the task is silent: completion fires no
            tool-status event, no on_notify, and therefore no main-model
            turn. Failures are still logged by the worker.
    """

    work: Callable[[], Awaitable[str]]
    channel: Channel
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time)
    tool_call_id: str | None = None
    description: str = ""
    ctx: contextvars.Context = field(default_factory=contextvars.copy_context)
    trace_id: str | None = None
    annotations: dict = field(default_factory=dict)
    deliver: bool = True

    def __post_init__(self) -> None:
        # Trap #5: a silent task holding a tool-call id would leave the
        # channel's pending_tool_call_ids never clearing, stalling its tool
        # batch forever — reject the combination at construction.
        if not self.deliver and self.tool_call_id is not None:
            raise ValueError(
                "deliver=False requires tool_call_id=None: a silent task "
                "must not hold a tool-call id (it would never clear from "
                "pending_tool_call_ids)"
            )


def _is_valid_results_path(value: object) -> bool:
    """True if ``value`` is a usable results path.

    Canonical empty/whitespace -> invalid validation (review #594-9). Both
    on_init (plugin path, after base_dir prepend) and TaskQueue.__init__
    (direct construction) use this, so the check lives in one place.
    """
    return value is not None and str(value).strip() != ""


class TaskQueue:
    """Async worker queue that processes Tasks with configurable concurrency."""

    def __init__(
        self,
        max_workers: int = 1,
        completed_buffer: int = 100,
        results_path: str | Path | None = None,
    ) -> None:
        self.max_workers = max_workers
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self._active_tasks: list[Task] = []
        self.completed: collections.deque[tuple[str, str]] = collections.deque(maxlen=completed_buffer)
        # When set, every completion (success or failure) appends one JSON
        # line to this file, so results survive the in-memory deque's
        # eviction and a daemon restart. Audit side channel: a write
        # failure is logged and swallowed, never breaking completion.
        # An empty/whitespace path is treated as unset (Path("") would
        # silently become the current directory). Shared validation with
        # on_init (review #594-9).
        self.results_path: Path | None = (
            Path(results_path) if _is_valid_results_path(results_path) else None
        )

    @property
    def active_task(self) -> Task | None:
        """Return one active task (or None).

        Provided for backward compatibility with single-worker usage.
        When max_workers > 1, multiple tasks may be active simultaneously;
        this property returns an arbitrary one.
        """
        return next(iter(self._active_tasks), None)

    @property
    def is_idle(self) -> bool:
        """True when no tasks are queued or actively being processed."""
        return self.queue.qsize() == 0 and not self._active_tasks

    async def enqueue(self, task: Task) -> None:
        """Add a task to the queue."""
        logger.info(
            "task enqueued",
            extra={
                "task_id": task.task_id,
                "channel": task.channel.id,
                "description": task.description,
            },
        )
        await self.queue.put(task)

    async def run_worker(
        self,
        on_complete: Callable[[Task, str], Awaitable[None]],
    ) -> None:
        """Pull tasks from the queue and execute up to max_workers concurrently.

        Spawns max_workers worker coroutines that all pull from the shared
        queue. Runs forever until cancelled via asyncio.Task.cancel()
        during shutdown.
        """
        workers: list[asyncio.Task] = []
        try:
            for _ in range(self.max_workers):
                workers.append(asyncio.create_task(
                    self._run_one_worker(on_complete)
                ))
            # Block until all workers end (they loop forever, so this
            # only returns when cancelled).
            await asyncio.gather(*workers)
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def _run_one_worker(
        self,
        on_complete: Callable[[Task, str], Awaitable[None]],
    ) -> None:
        """Single worker loop — pulls and executes tasks from the shared queue."""
        while True:
            task = await self.queue.get()
            self._active_tasks.append(task)
            logger.info(
                "task started",
                extra={"task_id": task.task_id, "description": task.description},
            )
            result = TASK_FAILURE_TEMPLATE.format(task_id=task.task_id, error="(unknown error)")
            # Run the work inside the context captured at Task creation so
            # attribution contextvars set by the enqueuer are visible here.
            # The extra task indirection is the accepted cost of entering a
            # contextvars.Context for a coroutine.
            work_task = asyncio.create_task(task.work(), context=task.ctx)
            try:
                result = await work_task
                logger.info(
                    "task completed",
                    extra={"task_id": task.task_id, "result_length": len(result)},
                )
            except asyncio.CancelledError:
                # Worker was cancelled while awaiting; don't orphan the work.
                work_task.cancel()
                try:
                    self._active_tasks.remove(task)
                except ValueError:
                    pass
                logger.info(
                    "task cancelled",
                    extra={"task_id": task.task_id, "description": task.description},
                )
                raise
            except Exception as exc:
                logger.warning(
                    "task failed",
                    extra={"task_id": task.task_id},
                    exc_info=True,
                )
                result = TASK_FAILURE_TEMPLATE.format(task_id=task.task_id, error=exc)
            finally:
                self.queue.task_done()
            self.completed.append((task.task_id, result))
            try:
                self._active_tasks.remove(task)
            except ValueError:
                pass
            # Synchronous, and BEFORE on_complete, deliberately (review
            # #594-2): (a) the append -> on_complete ordering (no await
            # between them) is load-bearing for burst coalescing (#398 R5/R6);
            # (b) a synchronous write means on_complete firing implies the
            # record is durable, which the audit contract and the tests rely
            # on. Tradeoff: a stalled disk (NFS) blocks the event loop for the
            # write's duration. Mitigations: the write is one small local JSON
            # line; a full disk fails fast (open() raises, caught in
            # _persist_result); NFS is not the expected deployment for a local
            # agent's audit log. A fire-and-forget writer is the longer-term
            # option if that tradeoff changes.
            self._persist_result(task, result)
            await on_complete(task, result)

    def _persist_result(self, task: Task, result: str) -> None:
        """Append one JSONL record for a completed task (sync — see call site).

        Record: {"ts", "task_id", "channel", "description", "result"}.
        A disk failure is logged and swallowed: persistence is an audit
        side channel and must never break task completion or delivery.
        """
        if self.results_path is None:
            return
        record = {
            "ts": time(),
            "task_id": task.task_id,
            "channel": task.channel.id,
            "description": task.description,
            "result": result,
        }
        try:
            with open(self.results_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            logger.warning(
                "task result persist failed for task %s at %s",
                task.task_id,
                self.results_path,
                exc_info=True,
            )

    def get_result_from_deque(self, task_id: str) -> str | None:
        """Return the result from the in-memory completed deque, or None.

        MUST run on the event-loop thread: the worker appends to (and, at
        maxlen, evicts from) self.completed on the loop thread, and CPython's
        deque iterator raises RuntimeError if the deque is mutated during
        iteration. Calling this off-loop (e.g. via to_thread) races that
        mutation (review #594-1).
        """
        for tid, res in self.completed:
            if tid == task_id:
                return res
        return None

    def get_result_from_file(self, task_id: str) -> str | None:
        """Return the result from the persisted JSONL file, or None.

        Safe off the loop thread (via to_thread): it reads only the file,
        never the deque. The last matching record wins. A partial or corrupt
        line (the file is appended to while this scans) is skipped, not
        allowed to discard earlier records; a non-UTF-8 byte sequence raises
        while iterating and is caught. Only string results honor the tool's
        contract.
        """
        if self.results_path is None:
            return None
        try:
            with open(self.results_path, "r", encoding="utf-8") as fh:
                found: str | None = None
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and rec.get("task_id") == task_id:
                        result = rec.get("result")
                        if isinstance(result, str):
                            found = result
            return found
        except (OSError, UnicodeDecodeError):
            return None

    def status(self) -> str:
        """Return a human-readable status summary.

        Shows active task(s), pending count, and last 3 completed results.
        """
        parts = []

        if self._active_tasks:
            for t in self._active_tasks:
                parts.append(f"Active: [{t.task_id}] {t.description}")

        pending = self.queue.qsize()
        if pending:
            parts.append(f"Pending: {pending} task(s)")

        if self.completed:
            recent = list(self.completed)[-STATUS_HISTORY_COUNT:]  # already (task_id, result) tuples
            completed_lines = []
            for tid, res in recent:
                completed_lines.append(f"  [{tid}] {res}")
            parts.append("Completed (last 3):\n" + "\n".join(completed_lines))

        if not parts:
            return "No tasks."

        return "\n".join(parts)


class TaskPlugin(CorvidaePlugin):
    """Plugin owning the TaskQueue and the task_status / task_result tools."""

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.task_queue: TaskQueue | None = None
        self._worker_task: asyncio.Task | None = None
        self._max_workers: int = 4
        self._completed_buffer: int = 100
        self._results_path: Path | None = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        daemon_config = config.get("daemon", {})
        self._max_workers = daemon_config.get("max_task_workers", 4)
        self._completed_buffer = daemon_config.get("completed_task_buffer", 100)
        # Reset first: a later on_init without the key must not keep a stale
        # path (consistent with _max_workers/_completed_buffer, which are
        # unconditionally overwritten). on_init prepends base_dir; the
        # empty/whitespace validation is shared with the constructor via
        # _is_valid_results_path (review #594-9).
        self._results_path = None
        results_path = daemon_config.get("task_results_path")
        if _is_valid_results_path(results_path):
            base_dir = config.get("_base_dir", Path("."))
            self._results_path = Path(base_dir) / results_path

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self  # captured by closure

        async def task_status() -> str:
            """Return the current status of the task queue."""
            if plugin.task_queue is None:
                return "Task queue not initialized."
            return plugin.task_queue.status()

        async def task_result(task_id: str) -> str:
            """Return the stored result of a completed task by its task id."""
            if plugin.task_queue is None:
                return "Task queue not initialized."
            q = plugin.task_queue
            # Deque leg on the loop thread (no race with the worker's
            # append/evict); file leg off the loop thread (review #594-1).
            result = q.get_result_from_deque(task_id)
            if result is None:
                result = await asyncio.to_thread(q.get_result_from_file, task_id)
            if result is None:
                if q.results_path is None:
                    return (
                        f"No stored result for task {task_id}. Result "
                        "persistence is disabled; set daemon.task_results_path "
                        "to record results across deque eviction and restarts."
                    )
                return (
                    f"No stored result for task {task_id}. It may have "
                    "completed before result persistence was enabled, or "
                    "the id is wrong."
                )
            return result

        tool_registry.append(Tool.from_function(task_status))
        tool_registry.append(Tool.from_function(task_result))

    @hookimpl
    async def on_start(self, config: dict) -> None:
        if self._results_path is not None:
            await asyncio.to_thread(
                self._results_path.parent.mkdir, parents=True, exist_ok=True
            )
        self.task_queue = TaskQueue(
            max_workers=self._max_workers,
            completed_buffer=self._completed_buffer,
            results_path=self._results_path,
        )

        self._worker_task = asyncio.create_task(
            self.task_queue.run_worker(self._on_task_complete)
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
                "deliver": task.deliver,
            },
        )
        # Silent tasks (task.deliver=False): log and stop — no tool-status event,
        # no on_notify, no main-model turn.
        if not task.deliver:
            return
        # Notify transports that a tool call has completed
        if task.tool_call_id:
            try:
                await self.pm.ahook.send(
                    channel=task.channel,
                    event=OutboundToolStatus(
                        tool_name=task.description.split(":", 1)[-1] if task.description and task.description.startswith("tool:") else "unknown",
                        status="completed",
                        result_summary=result[:200] if result else None,
                        tool_call_id=task.tool_call_id,
                    ),
                )
            except Exception:
                logger.warning("send hook failed for tool status", exc_info=True, extra={"channel": task.channel.id})

        text = f"[Task {task.task_id}] {result}"
        annotations = {
            "task_id": task.task_id,
            "trace_id": task.trace_id,
            **task.annotations,
        }
        try:
            # A result that answers no tool call is cut here (#744); one that
            # does is cut by the Agent when it stores the result row (#575).
            if not (task.tool_call_id is None and await self._notify_cut(task, text, annotations)):
                await self.pm.ahook.on_notify(
                    channel=task.channel,
                    source="task",
                    text=text,
                    tool_call_id=task.tool_call_id,
                    annotations=annotations,
                )
        except Exception as exc:
            logger.warning(
                "on_notify hook failed in _on_task_complete",
                exc_info=True,
                extra={
                    "channel": task.channel.id,
                    "task_id": task.task_id,
                    "tool_call_id": task.tool_call_id,
                },
            )
            # A tool call must still get a result, or the channel's tool
            # cycle waits forever (#173). Deliver an error result straight
            # to the Agent's queue; it skips this if the real one got there.
            if task.tool_call_id:
                await self._deliver_tool_error(task, exc)

    async def _notify_cut(self, task: Task, text: str, annotations: dict) -> bool:
        """Deliver a background result through the spill service, cut at the cap.

        The cap is ``tools.max_result_chars``, read at delivery so a settings
        change applies to the next result. Returns True when the notification
        was delivered; False when there is no cap or no service to cut with, and
        the caller delivers it whole.
        """
        # Imported here: spill imports the tool collection, which this module
        # must not pull in at import time.
        from corvidae.spill import NotificationSpill
        from corvidae.tool_collection import ToolCollectionPlugin

        tools = self.pm.get_plugin("tools")
        if not isinstance(tools, ToolCollectionPlugin):
            return False
        limit = tools.max_result_chars
        spill = self.pm.get_plugin("spill")
        if not isinstance(spill, NotificationSpill):
            if len(text) > limit:
                # shortcut: a bare plugin manager has no spill service, so the
                # result goes out whole; the Runtime always registers one.
                logger.warning(
                    "task %s on channel %s: result of %d characters is over "
                    "tools.max_result_chars (%d) but no spill service is registered, "
                    "so it is delivered whole; register NotificationSpill as 'spill' "
                    "to cut it",
                    task.task_id, task.channel.id, len(text), limit,
                    extra={"channel": task.channel.id, "task_id": task.task_id},
                )
            return False
        await spill.notify(
            [task.channel], source="task", text=text, annotations=annotations, limit=limit,
        )
        return True

    async def _deliver_tool_error(self, task: Task, exc: Exception) -> None:
        """Hand the Agent an error result for a task whose on_notify raised."""
        text = (
            f"[Task {task.task_id}] Error: the result of this tool call could "
            f"not be delivered ({type(exc).__name__}); the tool may have run. "
            "Check its effects before retrying."
        )
        # Duck-typed: task.py does not import the Agent. A missing Agent
        # raises AttributeError here and is logged like any other failure.
        try:
            agent = self.pm.get_plugin("agent")
            await agent.deliver_tool_error(
                task.channel, task.tool_call_id, text,
                {"task_id": task.task_id, "trace_id": task.trace_id, **task.annotations},
            )
        except Exception:
            logger.error(
                "fallback delivery of an error result for tool call %s on "
                "channel %s raised; if no result was queued the channel waits "
                "on it until restart",
                task.tool_call_id, task.channel.id,
                exc_info=True,
                extra={"channel": task.channel.id, "task_id": task.task_id,
                       "tool_call_id": task.tool_call_id},
            )
