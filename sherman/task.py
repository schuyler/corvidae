"""Task system for the sherman agent daemon.

Contains:
  - Task: dataclass representing a unit of async work with delivery context
  - TaskQueue: async worker queue that processes Tasks one at a time
  - TaskPlugin: plugin that owns the TaskQueue and delivers results via on_notify
"""

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
from sherman.tool import Tool

logger = logging.getLogger(__name__)


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
    """Async worker queue that processes Tasks one at a time."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.active_task: Task | None = None
        self.completed: dict[str, str] = {}  # task_id -> result

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
        """Pull tasks from the queue and execute them one at a time.

        Runs forever (while True). Cancelled via asyncio.Task.cancel()
        during shutdown.
        """
        while True:
            task = await self.queue.get()
            self.active_task = task
            logger.debug(
                "task started",
                extra={"task_id": task.task_id, "description": task.description},
            )
            result = f"Task {task.task_id} failed: (unknown error)"
            try:
                result = await task.work()
                logger.debug(
                    "task completed",
                    extra={"task_id": task.task_id, "result_length": len(result)},
                )
            except asyncio.CancelledError:
                self.active_task = None
                raise
            except Exception as exc:
                logger.warning(
                    "task failed",
                    extra={"task_id": task.task_id},
                    exc_info=True,
                )
                result = f"Task {task.task_id} failed: {exc}"
            finally:
                self.queue.task_done()
            self.completed[task.task_id] = result
            self.active_task = None
            await on_complete(task, result)

    def status(self) -> str:
        """Return a human-readable status summary.

        Shows active task, pending count, and last 3 completed results.
        """
        parts = []

        if self.active_task:
            parts.append(
                f"Active: [{self.active_task.task_id}] {self.active_task.description}"
            )

        pending = self.queue.qsize()
        if pending:
            parts.append(f"Pending: {pending} task(s)")

        if self.completed:
            recent = list(self.completed.items())[-3:]
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
        await self.pm.ahook.on_task_complete(
            channel=task.channel,
            task_id=task.task_id,
            result=result,
        )
