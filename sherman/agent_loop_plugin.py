"""AgentLoopPlugin — wires the agent loop into the hook system.

This plugin implements the core agent message processing flow. On each message:
1. Enqueue into the channel's per-channel serial queue (fire-and-enqueue)
2. Consumer: lazy-initialize conversation for the channel
3. Consumer: append message to conversation log
4. Consumer: compact if approaching context limit
5. Consumer: run agent loop (LLM + tool calls)
6. Consumer: persist new messages
7. Consumer: strip thinking tokens if configured
8. Consumer: send response via transport

Per-channel queues serialize all processing for a channel so concurrent
arrivals never race through the agent loop.

Config change (breaking): llm block is now structured:
    llm:
      main:           # required
        base_url: ...
        model: ...
        extra_body: ...  # optional
      background:     # optional — absent means use llm.main
        base_url: ...
        model: ...
        extra_body: ...

Logging:
    - INFO: on_start complete, on_message received, agent response sent,
      conversation initialized
    - ERROR: LLM client not initialized, agent loop failures
    - Latency is tracked via time.monotonic() around run_agent_loop
"""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

import aiosqlite

from sherman.agent_loop import _truncate, run_agent_loop, strip_reasoning_content, strip_thinking, tool_to_schema
from sherman.background import BackgroundTask, TaskQueue
from sherman.channel_queue import ChannelQueue, QueueItem
from sherman.conversation import ConversationLog, init_db
from sherman.hooks import hookimpl
from sherman.llm import LLMClient
from sherman.prompt import resolve_system_prompt

logger = logging.getLogger(__name__)


class AgentLoopPlugin:
    """Plugin that wires the agent loop into the hook system.

    Attributes:
        pm: Plugin manager instance (untyped due to pluggy limitations)
        client: LLM client for chat completions (main)
        bg_client: Optional LLM client for background tasks
        bg_extra_body: Optional extra_body for background task LLM calls
        db: SQLite connection for conversation persistence
        tools: Dict mapping tool names to async callable functions
        tool_schemas: List of tool schemas for LLM function calling
        base_dir: Base path for resolving relative system prompt files
        task_queue: Background task queue for long-running operations
        _worker_task: Background worker task that processes the queue
        _queues: Per-channel serial queues (dict[channel_id, ChannelQueue])

    The plugin collects tools from all registered plugins via the register_tools
    hook during on_start, then provides them to the agent loop during processing.
    """

    def __init__(self, pm) -> None:  # pm is untyped: pluggy.PluginManager has no typed interface for .registry
        self.pm = pm
        self.client: LLMClient | None = None
        self.bg_client: LLMClient | None = None
        self.bg_extra_body: dict | None = None
        self.db: aiosqlite.Connection | None = None
        self.tools: dict[str, Callable] = {}
        self.tool_schemas: list[dict] = []
        self.base_dir: Path = Path(".")
        self.task_queue: TaskQueue | None = None
        self._worker_task: asyncio.Task | None = None
        self.extra_body: dict | None = None
        self._queues: dict[str, ChannelQueue] = {}

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # 1. Create and start main LLM client.
        # llm.main is required — raises KeyError if absent.
        self.tools = {}
        self.tool_schemas = []
        self.base_dir = config.get("_base_dir", Path("."))
        llm_config = config.get("llm", {})

        # Breaking change: llm.main is now required.
        # Raises KeyError if 'main' is absent, giving a clear error message.
        main_config = llm_config["main"]
        self.extra_body = main_config.get("extra_body")
        self.client = LLMClient(
            base_url=main_config["base_url"],
            model=main_config["model"],
            api_key=main_config.get("api_key"),
        )
        await self.client.start()

        # 2. Optionally create background LLM client.
        bg_config = llm_config.get("background")
        if bg_config is not None:
            self.bg_extra_body = bg_config.get("extra_body")
            self.bg_client = LLMClient(
                base_url=bg_config["base_url"],
                model=bg_config["model"],
                api_key=bg_config.get("api_key"),
            )
            await self.bg_client.start()

        # 3. Open SQLite database (only if not already injected for testing)
        if self.db is None:
            db_path = config.get("daemon", {}).get("session_db", "sessions.db")
            self.db = await aiosqlite.connect(db_path)
            await init_db(self.db)

        # 4. Collect tools from all plugins via register_tools hook (sync).
        tool_fns: list = []
        self.pm.hook.register_tools(tool_registry=tool_fns)
        for fn in tool_fns:
            self.tools[fn.__name__] = fn
            self.tool_schemas.append(tool_to_schema(fn))

        # 5. Initialize task queue and create task tool closures.
        self.task_queue = TaskQueue()

        async def background_task(description: str, instructions: str) -> str:
            """Launch a long-running task in the background."""
            # Placeholder — overridden per-call in _process_queue_item with
            # a channel-capturing closure.
            raise RuntimeError("background_task placeholder called outside processing context")

        async def task_status() -> str:
            """Check the status of background tasks."""
            return self.task_queue.status()

        self.tools["background_task"] = background_task
        self.tool_schemas.append(tool_to_schema(background_task))
        self.tools["task_status"] = task_status
        self.tool_schemas.append(tool_to_schema(task_status))

        # 6. Start background worker.
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

        logger.info(
            "on_start complete",
            extra={
                "tool_count": len(self.tools),
                "channel_count": len(self.pm.registry.all()),
            },
        )

    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        if not self.client:
            logger.error("on_message: LLM client not initialized")
            return

        logger.info(
            "on_message received",
            extra={"channel": channel.id, "sender": sender},
        )

        item = QueueItem(
            role="user",
            content=text,
            channel=channel,
            sender=sender,
        )
        queue = self._get_or_create_queue(channel)
        await queue.enqueue(item)

    @hookimpl
    async def on_notify(
        self,
        channel,
        source: str,
        text: str,
        tool_call_id: str | None,
        meta: dict | None,
    ) -> None:
        """Enqueue a notification item on the channel's queue."""
        item = QueueItem(
            role="notification",
            content=text,
            channel=channel,
            source=source,
            tool_call_id=tool_call_id,
            meta=meta or {},
        )
        logger.debug(
            "on_notify received",
            extra={
                "channel": item.channel.id,
                "source": item.source,
                "tool_call_id": item.tool_call_id,
                "content_length": len(item.content),
            },
        )
        queue = self._get_or_create_queue(channel)
        await queue.enqueue(item)

    def _get_or_create_queue(self, channel) -> ChannelQueue:
        """Return existing ChannelQueue for channel or create and start a new one."""
        channel_id = channel.id
        if channel_id not in self._queues:
            q = ChannelQueue()
            q.start(self._process_queue_item)
            self._queues[channel_id] = q
        return self._queues[channel_id]

    async def _process_queue_item(self, item: QueueItem) -> None:
        """Process one item from the channel queue.

        This is the extracted body of the old on_message, generalized to
        handle both user messages and notifications.

        Dispatches on item.role:
          "user"         → {"role": "user", "content": item.content}
          "notification" with tool_call_id
                         → {"role": "tool", "tool_call_id": "...", "content": item.content}
          "notification" without tool_call_id
                         → {"role": "system", "content": "[source]\n\ncontent"}
        """
        logger.debug(
            "processing queue item",
            extra={
                "channel": item.channel.id,
                "role": item.role,
                "source": item.source,
                "has_tool_call_id": item.tool_call_id is not None,
            },
        )
        channel = item.channel

        # Build the conversation message based on item role
        if item.role == "user":
            conversation_message = {"role": "user", "content": item.content}
            request_text = item.content
        elif item.role == "notification":
            if item.tool_call_id:
                conversation_message = {
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": item.content,
                }
            else:
                conversation_message = {
                    "role": "system",
                    "content": f"[{item.source}]\n\n{item.content}",
                }
            request_text = item.content
        else:
            logger.error("_process_queue_item: unknown item role %r", item.role)
            return

        # 1. Lazy-initialize conversation on the channel
        await self._ensure_conversation(channel)
        conv = channel.conversation
        resolved = self.pm.registry.resolve_config(channel)

        # 2. Append message to conversation log (persisted)
        await conv.append(conversation_message)

        # 3. Compact if approaching context limit
        await conv.compact_if_needed(self.client, resolved["max_context_tokens"])

        # 4. Build per-call background_task closure capturing local channel.
        # The closure guards self.task_queue so processing works even if
        # on_start hasn't initialized the background task system yet.
        async def background_task(
            description: str, instructions: str, _tool_call_id: str | None = None
        ) -> str:
            """Launch a long-running task in the background."""
            if not self.task_queue:
                return "Error: background task system not initialized"
            task = BackgroundTask(
                channel=channel,
                description=description,
                instructions=instructions,
                tool_call_id=_tool_call_id,
            )
            await self.task_queue.enqueue(task)
            logger.debug(
                "background_task enqueued",
                extra={
                    "task_id": task.task_id,
                    "channel": channel.id,
                    "description": _truncate(description),
                },
            )
            return f"Task {task.task_id} enqueued: {description}"

        local_tools = {**self.tools, "background_task": background_task}

        # 5. Build prompt and run agent loop
        messages = conv.build_prompt()
        # NOTE: messages_before must use len(messages) from build_prompt(), NOT
        # len(conv.messages). build_prompt() prepends the system message, so
        # len(messages) == len(conv.messages) + 1. Using len(conv.messages) would
        # cause an off-by-one and double-persist the last user message.
        messages_before = len(messages)

        start = time.monotonic()
        try:
            if self.extra_body is not None:
                raw_response = await run_agent_loop(
                    self.client, messages, local_tools, self.tool_schemas, extra_body=self.extra_body
                )
            else:
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

        latency_ms = round((time.monotonic() - start) * 1000, 1)

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

        logger.info(
            "agent response sent",
            extra={"channel": channel.id, "latency_ms": latency_ms},
        )

        await self.pm.ahook.on_agent_response(
            channel=channel,
            request_text=request_text,
            response_text=display_response,
        )
        await self.pm.ahook.send_message(
            channel=channel,
            text=display_response,
            latency_ms=latency_ms,
        )

    @hookimpl
    async def on_stop(self) -> None:
        # NOTE: If a background task is in-flight when shutdown occurs, it is
        # cancelled without notification. The task result is lost. A future
        # improvement could send a "task cancelled" message to the originating
        # channel before cancelling.
        # NOTE: Channel queue items in flight or queued at stop time are also
        # dropped silently — this is consistent with background task behavior.

        # Cancel all channel queue consumers.
        for queue in self._queues.values():
            await queue.stop()

        # Cancel background worker task before closing client/db.
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self.bg_client:
            await self.bg_client.stop()
        if self.client:
            await self.client.stop()
        if self.db:
            await self.db.close()

    async def _execute_background_task(self, task: BackgroundTask) -> str:
        """Run a background task with its own conversation context."""
        logger.debug(
            "executing background task",
            extra={
                "task_id": task.task_id,
                "description": task.description,
                "instructions": _truncate(task.instructions or ""),
            },
        )
        # Exclude background_task to prevent unbounded recursive task creation.
        bg_tools = {k: v for k, v in self.tools.items() if k != "background_task"}
        bg_schemas = [s for s in self.tool_schemas if s["function"]["name"] != "background_task"]
        messages = [
            {"role": "system", "content": "You are executing a background task. "
             "Work through the instructions step by step. Be thorough."},
            {"role": "user", "content": task.instructions},
        ]
        # Use bg_client if configured, otherwise fall back to main client.
        client = self.bg_client or self.client
        extra_body = self.bg_extra_body if self.bg_client else self.extra_body
        if extra_body is not None:
            return await run_agent_loop(
                client, messages, bg_tools, bg_schemas, extra_body=extra_body
            )
        else:
            return await run_agent_loop(
                client, messages, bg_tools, bg_schemas
            )

    async def _on_task_complete(self, task: BackgroundTask, result: str) -> None:
        """Handle a completed background task.

        Routes through on_notify so the LLM sees the task result in the
        conversation and can react to it, rather than sending directly.
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

    async def _ensure_conversation(self, channel) -> None:
        """Lazy-initialize ConversationLog on a channel if not present."""
        if channel.conversation is not None:
            return
        conv = ConversationLog(self.db, channel.id)
        resolved = self.pm.registry.resolve_config(channel)
        conv.system_prompt = resolve_system_prompt(resolved["system_prompt"], self.base_dir)
        await conv.load()
        channel.conversation = conv

        logger.info(
            "conversation initialized for channel",
            extra={"channel": channel.id},
        )
