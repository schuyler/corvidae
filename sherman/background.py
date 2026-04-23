"""Background task system for the sherman agent daemon."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import time

from sherman.channel import Channel


@dataclass
class BackgroundTask:
    channel: Channel
    description: str
    instructions: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time)


class TaskQueue:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[BackgroundTask] = asyncio.Queue()
        self.active_task: BackgroundTask | None = None
        self.completed: dict[str, str] = {}  # task_id -> result

    async def enqueue(self, task: BackgroundTask) -> None:
        """Add a task to the queue."""
        await self.queue.put(task)

    async def run_worker(
        self,
        execute_fn: Callable[[BackgroundTask], Awaitable[str]],
        on_complete: Callable[[BackgroundTask, str], Awaitable[None]],
    ) -> None:
        """Pull tasks from the queue and execute them one at a time.

        Runs forever (while True). Cancelled via asyncio.Task.cancel()
        during shutdown.
        """
        while True:
            task = await self.queue.get()
            self.active_task = task
            result = f"Task {task.task_id} failed: (unknown error)"
            try:
                result = await execute_fn(task)
            except asyncio.CancelledError:
                self.active_task = None
                raise
            except Exception as exc:
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
