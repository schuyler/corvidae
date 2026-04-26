"""Task system for the corvidae agent daemon.

Contains:
  - Task: dataclass representing a unit of async work with delivery context
  - TaskQueue: async worker queue that processes Tasks one at a time
  - TaskPlugin: plugin that owns the TaskQueue and delivers results via on_notify
"""

from __future__ import annotations

import asyncio
import collections
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.channel import Channel

from corvidae.hooks import hookimpl
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
    """

    work: Callable[[], Awaitable[str]]
    channel: Channel
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time)
    tool_call_id: str | None = None
    description: str = ""


class TaskQueue:
    """Async worker queue that processes Tasks with configurable concurrency."""

    def __init__(self, max_workers: int = 1, completed_buffer: int = 100) -> None:
        self.max_workers = max_workers
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self._active_tasks: list[Task] = []
        self.completed: collections.deque[tuple[str, str]] = collections.deque(maxlen=completed_buffer)

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
        logger.debug(
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
            logger.debug(
                "task started",
                extra={"task_id": task.task_id, "description": task.description},
            )
            result = TASK_FAILURE_TEMPLATE.format(task_id=task.task_id, error="(unknown error)")
            try:
                result = await task.work()
                logger.debug(
                    "task completed",
                    extra={"task_id": task.task_id, "result_length": len(result)},
                )
            except asyncio.CancelledError:
                try:
                    self._active_tasks.remove(task)
                except ValueError:
                    pass
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
            await on_complete(task, result)

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


class TaskPlugin:
    """Plugin owning the TaskQueue and task_status tool."""

    def __init__(self, pm) -> None:
        self.pm = pm
        self.task_queue: TaskQueue | None = None
        self._worker_task: asyncio.Task | None = None

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self  # captured by closure

        async def task_status() -> str:
            """Return the current status of the task queue."""
            if plugin.task_queue is None:
                return "Task queue not initialized."
            return plugin.task_queue.status()

        tool_registry.append(Tool.from_function(task_status))

    @hookimpl
    async def on_start(self, config: dict) -> None:
        daemon_config = config.get("daemon", {})
        max_workers = daemon_config.get("max_task_workers", 4)
        completed_buffer = daemon_config.get("completed_task_buffer", 100)
        self.task_queue = TaskQueue(max_workers=max_workers, completed_buffer=completed_buffer)

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
            },
        )
        try:
            await self.pm.ahook.on_notify(
                channel=task.channel,
                source="task",
                text=f"[Task {task.task_id}] {result}",
                tool_call_id=task.tool_call_id,
                meta={"task_id": task.task_id},
            )
        except Exception:
            logger.warning(
                "on_notify hook failed in _on_task_complete",
                exc_info=True,
                extra={"channel": task.channel.id, "task_id": task.task_id},
            )
