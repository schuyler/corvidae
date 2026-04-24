"""Background task system for the sherman agent daemon.

Contains:
  - BackgroundTask: dataclass representing a single background task
  - TaskQueue: async worker queue that processes tasks one at a time
  - BackgroundPlugin: plugin that owns the task queue and worker, registers
    background_task and task_status tools, and delivers results via on_notify
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import time

from sherman.agent_loop import _truncate, run_agent_loop, strip_thinking
from sherman.channel import Channel
from sherman.hooks import hookimpl
from sherman.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class BackgroundTask:
    channel: Channel
    description: str
    instructions: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time)
    tool_call_id: str | None = None


class TaskQueue:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[BackgroundTask] = asyncio.Queue()
        self.active_task: BackgroundTask | None = None
        self.completed: dict[str, str] = {}  # task_id -> result

    async def enqueue(self, task: BackgroundTask) -> None:
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
            logger.debug(
                "task started",
                extra={"task_id": task.task_id, "description": task.description},
            )
            result = f"Task {task.task_id} failed: (unknown error)"
            try:
                result = await execute_fn(task)
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


class BackgroundPlugin:
    """Plugin that owns the background task queue and worker.

    Registers background_task (placeholder) and task_status tools.
    Delivers task results via pm.ahook.on_notify() so they flow back
    through AgentPlugin's notification path.

    AgentPlugin's _process_queue_item replaces the background_task
    placeholder with a per-call closure that captures the current channel
    and enqueues to self.task_queue.

    Attach pattern: sets pm.background = self during on_start so that
    AgentPlugin can locate this plugin's task_queue.
    """

    def __init__(self, pm) -> None:
        self.pm = pm
        self.task_queue: TaskQueue | None = None
        self._worker_task: asyncio.Task | None = None
        self.bg_client = None  # set during on_start if llm.background is configured

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Optionally create a dedicated background LLM client.
        llm_config = config.get("llm", {})
        bg_config = llm_config.get("background")
        if bg_config is not None:
            self.bg_client = LLMClient(
                base_url=bg_config["base_url"],
                model=bg_config["model"],
                api_key=bg_config.get("api_key"),
                extra_body=bg_config.get("extra_body"),
            )
            await self.bg_client.start()

        # Initialize task queue.
        self.task_queue = TaskQueue()

        # Attach self to pm so AgentPlugin can find task_queue.
        self.pm.background = self

        # Start background worker.
        # Use wrapper closures so monkey-patching _execute_task / _on_task_complete
        # on this instance (e.g., in tests) is picked up at call time.
        async def _execute_wrapper(task: BackgroundTask) -> str:
            return await self._execute_task(task)

        async def _complete_wrapper(task: BackgroundTask, result: str) -> None:
            return await self._on_task_complete(task, result)

        self._worker_task = asyncio.create_task(
            self.task_queue.run_worker(
                _execute_wrapper,
                _complete_wrapper,
            )
        )

        logger.debug("BackgroundPlugin started")

    @hookimpl
    async def on_stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self.bg_client:
            await self.bg_client.stop()

        logger.debug("BackgroundPlugin stopped")

    @hookimpl
    def register_tools(self, tool_registry) -> None:
        """Register background_task placeholder and task_status tool."""
        async def background_task(description: str, instructions: str) -> str:
            """Launch a long-running task in the background."""
            raise RuntimeError("background_task placeholder called outside processing context")

        async def task_status() -> str:
            """Check the status of background tasks."""
            if self.task_queue is None:
                return "Background task system not initialized."
            return self.task_queue.status()

        tool_registry.append(background_task)
        tool_registry.append(task_status)

    async def _execute_task(self, task: BackgroundTask) -> str:
        """Run a background task with its own conversation context."""
        logger.debug(
            "executing background task",
            extra={
                "task_id": task.task_id,
                "description": task.description,
                "instructions": _truncate(task.instructions or ""),
            },
        )

        # Get agent plugin's tools (excluding background_task to prevent nesting)
        agent = getattr(self.pm, "agent_plugin", None)
        if agent is not None:
            bg_tools = {k: v for k, v in agent.tools.items() if k != "background_task"}
            bg_schemas = [s for s in agent.tool_schemas if s["function"]["name"] != "background_task"]
            client = self.bg_client or agent.client
        else:
            bg_tools = {}
            bg_schemas = []
            client = self.bg_client

        messages = [
            {"role": "system", "content": "You are executing a background task. "
             "Work through the instructions step by step. Be thorough."},
            {"role": "user", "content": task.instructions},
        ]
        return await run_agent_loop(client, messages, bg_tools, bg_schemas)

    async def _on_task_complete(self, task: BackgroundTask, result: str) -> None:
        """Handle a completed background task.

        Routes through on_notify so the LLM sees the task result in the
        conversation and can react to it.
        """
        logger.debug(
            "task complete, dispatching notification",
            extra={
                "task_id": task.task_id,
                "channel": task.channel.id,
                "result_length": len(result),
            },
        )
        display_result = strip_thinking(result)
        await self.pm.ahook.on_task_complete(
            channel=task.channel, task_id=task.task_id, result=display_result,
        )
        await self.pm.ahook.on_notify(
            channel=task.channel,
            source="background_task",
            text=f"[Task {task.task_id}] {display_result}",
            tool_call_id=task.tool_call_id,
            meta={"task_id": task.task_id},
        )
