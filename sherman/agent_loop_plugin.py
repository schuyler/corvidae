"""AgentLoopPlugin — wires the agent loop into the hook system."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

import aiosqlite

from sherman.agent_loop import run_agent_loop, strip_reasoning_content, strip_thinking, tool_to_schema
from sherman.background import BackgroundTask, TaskQueue
from sherman.conversation import ConversationLog, init_db
from sherman.hooks import hookimpl
from sherman.llm import LLMClient
from sherman.prompt import resolve_system_prompt

logger = logging.getLogger(__name__)


class AgentLoopPlugin:
    """Plugin that wires the agent loop into the hook system."""

    def __init__(self, pm) -> None:  # pm is untyped: pluggy.PluginManager has no typed interface for .registry
        self.pm = pm
        self.client: LLMClient | None = None
        self.db: aiosqlite.Connection | None = None
        self.tools: dict[str, Callable] = {}
        self.tool_schemas: list[dict] = []
        self.base_dir: Path = Path(".")
        self.task_queue: TaskQueue | None = None
        self._worker_task: asyncio.Task | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # 1. Create and start LLM client.
        # Missing 'llm' key uses empty dict, causing KeyError on required
        # fields (base_url, model) — fail-fast on misconfiguration.
        self.tools = {}
        self.tool_schemas = []
        self.base_dir = config.get("_base_dir", Path("."))
        llm_config = config.get("llm", {})
        self.client = LLMClient(
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config.get("api_key"),
        )
        await self.client.start()

        # 2. Open SQLite database (only if not already injected for testing)
        if self.db is None:
            db_path = config.get("daemon", {}).get("session_db", "sessions.db")
            self.db = await aiosqlite.connect(db_path)
            await init_db(self.db)

        # 3. Collect tools from all plugins via register_tools hook (sync).
        tool_fns: list = []
        self.pm.hook.register_tools(tool_registry=tool_fns)
        for fn in tool_fns:
            self.tools[fn.__name__] = fn
            self.tool_schemas.append(tool_to_schema(fn))

        # 4. Initialize task queue and create task tool closures.
        self.task_queue = TaskQueue()

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

        # 5. Start background worker.
        # Use lambda wrappers so tests can monkey-patch _execute_background_task
        # and _on_task_complete on the plugin instance after on_start returns.
        async def _execute_wrapper(task: BackgroundTask) -> str:
            return await self._execute_background_task(task)

        async def _complete_wrapper(task: BackgroundTask, result: str) -> None:
            return await self._on_task_complete(task, result)

        self._worker_task = asyncio.create_task(
            self.task_queue.run_worker(
                _execute_wrapper,
                _complete_wrapper,
            )
        )

    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        if not self.client:
            return

        # 1. Lazy-initialize conversation on the channel
        await self._ensure_conversation(channel)
        conv = channel.conversation
        resolved = self.pm.registry.resolve_config(channel)

        # 2. Append user message to conversation log (persisted)
        await conv.append({"role": "user", "content": text})

        # 3. Compact if approaching context limit
        await conv.compact_if_needed(self.client, resolved["max_context_tokens"])

        # 4. Build per-call background_task closure capturing local channel.
        # The closure guards self.task_queue so on_message works even if
        # on_start hasn't initialized the background task system yet.
        async def background_task(description: str, instructions: str) -> str:
            """Launch a long-running task in the background."""
            if not self.task_queue:
                return "Error: background task system not initialized"
            task = BackgroundTask(
                channel=channel,
                description=description,
                instructions=instructions,
            )
            await self.task_queue.enqueue(task)
            return f"Task {task.task_id} enqueued: {description}"

        local_tools = {**self.tools, "background_task": background_task}

        # 5. Build prompt and run agent loop
        messages = conv.build_prompt()
        # NOTE: messages_before must use len(messages) from build_prompt(), NOT
        # len(conv.messages). build_prompt() prepends the system message, so
        # len(messages) == len(conv.messages) + 1. Using len(conv.messages) would
        # cause an off-by-one and double-persist the last user message.
        messages_before = len(messages)

        try:
            raw_response = await run_agent_loop(
                self.client, messages, local_tools, self.tool_schemas
            )
        except Exception:
            logger.exception("run_agent_loop failed for channel %s", channel.id)
            await self.pm.ahook.send_message(
                channel=channel,
                text="Sorry, I encountered an error and could not process your message.",
            )
            return

        # 6. Persist new messages appended by run_agent_loop
        new_messages = messages[messages_before:]
        # Persist BEFORE stripping — conv.append serializes the dict at call
        # time, so the full message (including reasoning_content) is saved to
        # disk before in-memory stripping occurs below.
        for msg in new_messages:
            await conv.append(msg)

        # 7. Thinking token handling for active history.
        # Only strip the newly appended messages — earlier messages were
        # already stripped on prior turns.
        if not resolved["keep_thinking_in_history"]:
            strip_reasoning_content(new_messages)

        # 8. Strip thinking for display and send response
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

    @hookimpl
    async def on_stop(self) -> None:
        # NOTE: If a background task is in-flight when shutdown occurs, it is
        # cancelled without notification. The task result is lost. A future
        # improvement could send a "task cancelled" message to the originating
        # channel before cancelling.
        # Cancel worker task before closing client/db.
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self.client:
            await self.client.stop()
        if self.db:
            await self.db.close()

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

    async def _on_task_complete(self, task: BackgroundTask, result: str) -> None:
        """Handle a completed background task."""
        display_result = strip_thinking(result)
        await self.pm.ahook.on_task_complete(
            channel=task.channel, task_id=task.task_id, result=display_result,
        )
        await self.pm.ahook.send_message(
            channel=task.channel, text=f"[Task {task.task_id}] {display_result}",
        )

    async def _ensure_conversation(self, channel) -> None:
        """Lazy-initialize ConversationLog on a channel if not present."""
        if channel.conversation is not None:
            return
        conv = ConversationLog(self.db, channel.id)
        resolved = self.pm.registry.resolve_config(channel)
        conv.system_prompt = resolve_system_prompt(resolved["system_prompt"], self.base_dir)
        await conv.load()
        channel.conversation = conv
