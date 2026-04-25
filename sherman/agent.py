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
    - ERROR: LLM client not initialized, run_agent_turn failures, missing TaskQueue
    - Latency is tracked via AgentTurnResult.latency_ms
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import aiosqlite

from sherman.agent_loop import run_agent_turn, strip_reasoning_content, strip_thinking
from sherman.channel import Channel, ChannelRegistry, resolve_system_prompt
from sherman.conversation import ConversationLog, init_db
from sherman.hooks import get_dependency, hookimpl
from sherman.llm import LLMClient
from sherman.queue import SerialQueue
from sherman.task import Task
from sherman.tool import Tool, ToolContext, ToolRegistry, execute_tool_call

logger = logging.getLogger("sherman.agent")


class QueueItemRole(Enum):
    USER = "user"
    NOTIFICATION = "notification"


@dataclass
class QueueItem:
    """A single item in the per-channel processing queue.

    Attributes:
        role: QueueItemRole.USER for inbound messages, QueueItemRole.NOTIFICATION
            for injected events.
        content: The text content to process.
        channel: The Channel this item belongs to.
        sender: For user messages, the sender identity; None for notifications.
        source: For notifications, the origin (e.g. "task"); None for user messages.
        tool_call_id: For deferred tool results (background task completions).
        meta: Extensible metadata (task_id, etc.).
    """

    role: QueueItemRole
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
        _registry: ChannelRegistry, resolved in on_start via get_dependency
    """

    depends_on = {"registry"}

    def __init__(self, pm) -> None:
        self.pm = pm
        self.client: LLMClient | None = None
        self.db: aiosqlite.Connection | None = None
        self.tools: dict[str, Callable] = {}
        self.tool_schemas: list[dict] = []
        self.base_dir: Path = Path(".")
        self._queues: dict[str, SerialQueue] = {}
        self._registry: ChannelRegistry | None = None

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
            role=QueueItemRole.USER,
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
            role=QueueItemRole.NOTIFICATION,
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
        """Process one item from the channel queue.

        This method is the center of the agent loop. It is called once per
        user message and once per tool result. The tool-calling cycle is not
        a loop in this code — it is implemented via re-entry through the
        channel's serial queue:

            _process_queue_item (LLM call, tool calls detected)
              → _dispatch_tool_calls (enqueue Tasks on TaskQueue)
              → TaskQueue.run_worker (execute tool in background)
              → _on_task_complete → on_notify (enqueue notification)
              → _process_queue_item again (with tool result)

        This re-entrant design keeps the serial queue unblocked during tool
        execution, allowing user messages to interleave mid-cycle. A literal
        loop (as run_agent_loop uses for subagents) would block the queue
        for the entire tool-calling sequence.
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
        if item.role == QueueItemRole.USER:
            conversation_message = {"role": "user", "content": item.content}
            request_text = item.content
        elif item.role == QueueItemRole.NOTIFICATION:
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

        # 2. Reset turn_counter on user messages
        if item.role == QueueItemRole.USER:
            channel.turn_counter = 0

        # 3. Resolve config and read max_turns_limit
        resolved = self._registry.resolve_config(channel)
        max_turns_limit = resolved["max_turns"]

        # 4. Append message to conversation log (persisted)
        await conv.append(conversation_message)

        # 5. Compact if approaching context limit
        try:
            await conv.compact_if_needed(self.client, resolved["max_context_tokens"])
        except Exception:
            logger.warning("compaction failed, skipping", exc_info=True)

        # 6. Build prompt and call run_agent_turn (single LLM invocation)
        messages = conv.build_prompt()

        try:
            result = await run_agent_turn(self.client, messages, self.tool_schemas)
        except Exception:
            logger.exception("run_agent_turn failed for channel %s", channel.id)
            await self.pm.ahook.send_message(
                channel=channel,
                text="Sorry, I encountered an error and could not process your message.",
            )
            return

        # 7. Persist assistant message (run_agent_turn already appended to messages in place)
        await conv.append(result.message)

        # 8. Strip reasoning_content from in-memory copy if configured
        if not resolved["keep_thinking_in_history"]:
            strip_reasoning_content([result.message])

        # 9. Dispatch tool calls or send text response
        # Check-before-increment: turn_counter < max_turns_limit allows dispatch
        if result.tool_calls and channel.turn_counter < max_turns_limit:
            channel.turn_counter += 1
            await self._dispatch_tool_calls(result.tool_calls, channel)
            # Do NOT send text response; return and wait for tool results
            return
        elif result.tool_calls:
            # Max turns reached — suppress tool dispatch, send fallback text
            logger.warning(
                "max_turns reached, suppressing tool calls",
                extra={"channel": channel.id, "turn_counter": channel.turn_counter},
            )
            display_response = strip_thinking(result.text) or "(max tool-calling rounds reached)"
        else:
            # No tool calls — normal text response
            channel.turn_counter += 1
            display_response = strip_thinking(result.text)

        logger.info(
            "agent response sent",
            extra={"channel": channel.id, "latency_ms": result.latency_ms},
        )

        await self.pm.ahook.on_agent_response(
            channel=channel,
            request_text=request_text,
            response_text=display_response,
        )
        await self.pm.ahook.send_message(
            channel=channel,
            text=display_response,
            latency_ms=result.latency_ms,
        )

    async def _dispatch_tool_calls(
        self, tool_calls: list[dict], channel: Channel
    ) -> None:
        """Dispatch LLM tool calls as Tasks to the TaskQueue."""
        task_queue = getattr(self.pm.get_plugin("task"), "task_queue", None)
        if task_queue is None:
            logger.error("tool calls requested but no TaskQueue available")
            return

        for call in tool_calls:
            call_id = call["id"]
            fn_name = call["function"]["name"]
            raw_args = call["function"]["arguments"]

            async def make_work(fn_name=fn_name, raw_args=raw_args, call_id=call_id):
                """Execute a single tool call. Captures fn_name, raw_args, call_id via defaults."""
                args = json.loads(raw_args)
                if fn_name not in self.tools:
                    logger.warning("unknown tool called: %s", fn_name)
                    return f"Error: unknown tool '{fn_name}'"
                try:
                    tool_fn = self.tools[fn_name]
                    return await execute_tool_call(
                        tool_fn,
                        args,
                        channel=channel,
                        tool_call_id=call_id,
                        task_queue=task_queue,
                    )
                except Exception:
                    logger.warning(
                        "tool %s raised exception", fn_name, exc_info=True
                    )
                    return f"Error: tool '{fn_name}' failed"

            task = Task(
                work=make_work,
                channel=channel,
                tool_call_id=call_id,
                description=f"tool:{fn_name}",
            )
            await task_queue.enqueue(task)

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
        resolved = self._registry.resolve_config(channel)
        conv.system_prompt = resolve_system_prompt(resolved["system_prompt"], self.base_dir)
        await conv.load()
        channel.conversation = conv

        logger.info(
            "conversation initialized for channel",
            extra={"channel": channel.id},
        )

    async def _start_plugin(self, config: dict) -> None:
        """Initialize LLM client, database, and tools."""
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
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
        self.tool_registry = tool_registry
        self.tools = tool_registry.as_dict()
        self.tool_schemas = tool_registry.schemas()

        logger.info(
            "on_start complete",
            extra={
                "tool_count": len(self.tools),
                "channel_count": len(self._registry.all()),
            },
        )

    async def _stop_plugin(self) -> None:
        """Close LLM client and database."""
        if self.client:
            await self.client.stop()
        if self.db:
            await self.db.close()

