"""AgentPlugin — wires the agent loop into the hook system.

Merged from orchestrator.py, lifecycle.py, and processing.py.
File boundaries now match domain responsibility rather than execution phase.

AgentPlugin is the central plugin that:
  - Manages per-channel serial queues
  - Initializes LLM clients, DB, and tools on startup
  - Processes inbound messages and notifications through the agent loop

Config:
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
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

from sherman.agent_loop import _truncate, run_agent_loop, strip_reasoning_content, strip_thinking
from sherman.background import BackgroundTask
from sherman.channel import Channel
from sherman.conversation import ConversationLog, init_db, resolve_system_prompt
from sherman.hooks import hookimpl
from sherman.llm import LLMClient
from sherman.queue import SerialQueue
from sherman.tool import Tool, ToolRegistry

logger = logging.getLogger("sherman.agent")


@dataclass
class QueueItem:
    """A single item in the per-channel processing queue.

    Attributes:
        role: "user" for inbound messages, "notification" for injected events.
        content: The text content to process.
        channel: The Channel this item belongs to.
        sender: For user messages, the sender identity; None for notifications.
        source: For notifications, the origin (e.g. "background_task"); None for user messages.
        tool_call_id: For deferred tool results (background task completions).
        meta: Extensible metadata (task_id, etc.).
    """

    role: str
    content: str
    channel: Channel
    sender: str | None = None
    source: str | None = None
    tool_call_id: str | None = None
    meta: dict = field(default_factory=dict)


class AgentPlugin:
    """Plugin that wires the agent loop into the hook system.

    Attributes:
        pm: Plugin manager instance (untyped due to pluggy limitations)
        client: LLM client for chat completions (main)
        db: SQLite connection for conversation persistence
        tools: Dict mapping tool names to async callable functions
        tool_schemas: List of tool schemas for LLM function calling
        base_dir: Base path for resolving relative system prompt files
        _queues: Per-channel serial queues (dict[channel_id, SerialQueue])
    """

    def __init__(self, pm) -> None:
        self.pm = pm
        self.client: LLMClient | None = None
        self.db: aiosqlite.Connection | None = None
        self.tools: dict[str, Callable] = {}
        self.tool_schemas: list[dict] = []
        self.base_dir: Path = Path(".")
        self._queues: dict[str, SerialQueue] = {}

    @hookimpl
    async def on_start(self, config: dict) -> None:
        await self._start_plugin(config)

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

    def _get_or_create_queue(self, channel) -> SerialQueue:
        """Return existing SerialQueue for channel or create and start a new one."""
        channel_id = channel.id
        if channel_id not in self._queues:
            q = SerialQueue()
            q.start(self._process_queue_item)
            self._queues[channel_id] = q
        return self._queues[channel_id]

    async def _process_queue_item(self, item: QueueItem) -> None:
        """Process one item from the channel queue."""
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
        # Resolve task_queue from BackgroundPlugin if registered.
        task_queue = getattr(getattr(self.pm, "background", None), "task_queue", None)

        async def background_task(
            description: str, instructions: str, _tool_call_id: str | None = None
        ) -> str:
            """Launch a long-running task in the background."""
            if not task_queue:
                return "Error: background task system not initialized"
            task = BackgroundTask(
                channel=channel,
                description=description,
                instructions=instructions,
                tool_call_id=_tool_call_id,
            )
            await task_queue.enqueue(task)
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
        messages_before = len(messages)

        # Resolve new task queue from TaskPlugin
        task_queue_ref = getattr(
            getattr(self.pm, "task_plugin", None), "task_queue", None
        )

        start = time.monotonic()
        try:
            raw_response = await run_agent_loop(
                self.client, messages, local_tools, self.tool_schemas,
                channel=channel,
                task_queue=task_queue_ref,
            )
        except Exception:  # broad catch: aiohttp, KeyError, TimeoutError, etc.
            logger.exception("run_agent_loop failed for channel %s", channel.id)
            await self.pm.ahook.send_message(
                channel=channel,
                text="Sorry, I encountered an error and could not process your message.",
            )
            return

        latency_ms = round((time.monotonic() - start) * 1000, 1)

        # 6. Persist new messages appended by run_agent_loop
        new_messages = messages[messages_before:]
        for msg in new_messages:
            await conv.append(msg)

        # 7. Thinking token handling for active history.
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
        # Cancel all channel queue consumers.
        for queue in self._queues.values():
            await queue.stop()

        await self._stop_plugin()

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

    async def _start_plugin(self, config: dict) -> None:
        """Initialize LLM client, database, and tools."""
        self.tools = {}
        self.tool_schemas = []
        self.base_dir = config.get("_base_dir", Path("."))
        llm_config = config.get("llm", {})

        # Breaking change: llm.main is required.
        main_config = llm_config["main"]
        self.client = LLMClient(
            base_url=main_config["base_url"],
            model=main_config["model"],
            api_key=main_config.get("api_key"),
            extra_body=main_config.get("extra_body"),
        )
        await self.client.start()

        # Open SQLite database (only if not already injected for testing)
        if self.db is None:
            db_path = config.get("daemon", {}).get("session_db", "sessions.db")
            self.db = await aiosqlite.connect(db_path)
            await init_db(self.db)

        # Collect tools from all plugins via register_tools hook (sync).
        collected: list = []
        self.pm.hook.register_tools(tool_registry=collected)
        tool_registry = ToolRegistry()
        for item in collected:
            if isinstance(item, Tool):
                tool_registry.add(item)
            else:
                tool_registry.add(Tool.from_function(item))
        self.tools = tool_registry.as_dict()
        self.tool_schemas = tool_registry.schemas()

        logger.info(
            "on_start complete",
            extra={
                "tool_count": len(self.tools),
                "channel_count": len(self.pm.registry.all()),
            },
        )

    async def _stop_plugin(self) -> None:
        """Close LLM client and database."""
        if self.client:
            await self.client.stop()
        if self.db:
            await self.db.close()

    # ---------------------------------------------------------------------------
    # Legacy delegation methods — kept for test compatibility during transition
    # Background task execution is now owned by BackgroundPlugin, but these
    # delegation stubs let existing tests that monkey-patch _on_task_complete
    # and _execute_background_task on the plugin instance continue to work.
    # ---------------------------------------------------------------------------

    async def _execute_background_task(self, task) -> str:
        """Execute a background task (delegates to BackgroundPlugin).

        Legacy delegation stub — kept for test compatibility. BackgroundPlugin
        owns background task execution; this method routes to it.
        """
        bg = getattr(self.pm, "background", None)
        if bg is not None:
            return await bg._execute_task(task)
        raise RuntimeError("No BackgroundPlugin registered")

    async def _on_task_complete(self, task, result: str) -> None:
        """Handle a completed background task (delegates to BackgroundPlugin).

        Legacy delegation stub — kept for test compatibility.
        """
        bg = getattr(self.pm, "background", None)
        if bg is not None:
            return await bg._on_task_complete(task, result)
        raise RuntimeError("No BackgroundPlugin registered")
