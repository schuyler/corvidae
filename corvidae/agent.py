"""AgentPlugin — wires the agent loop into the hook system.

Merged from orchestrator.py, lifecycle.py, and processing.py.
File boundaries now match domain responsibility rather than execution phase.

AgentPlugin is the central plugin that:
  - Manages per-channel serial queues
  - Initializes LLM clients and tools on startup
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
    agent:
      max_tool_result_chars: 100000  # optional — tool result truncation limit

Logging:
    - INFO: on_start complete, on_message received, agent response sent
    - ERROR: LLM client not initialized, run_agent_turn failures, missing TaskQueue
    - Latency is tracked via AgentTurnResult.latency_ms
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from corvidae.agent_loop import run_agent_turn
from corvidae.channel import Channel, ChannelRegistry
from corvidae.hooks import resolve_hook_results, HookStrategy, get_dependency, hookimpl
from corvidae.llm import LLMClient
from corvidae.queue import SerialQueue
from corvidae.task import Task
from corvidae.tool import Tool, ToolContext, ToolRegistry, execute_tool_call

logger = logging.getLogger("corvidae.agent")

DEFAULT_LLM_ERROR_MESSAGE = "Sorry, I encountered an error and could not process your message."
# Note: same text as MAX_ROUNDS_REACHED_MESSAGE in agent_loop.py — kept in sync.
MAX_TURNS_FALLBACK_MESSAGE = "(max tool-calling rounds reached)"


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
        tools: Dict mapping tool names to async callable functions
        tool_schemas: List of tool schemas for LLM function calling
        queues: Per-channel serial queues (dict[channel_id, SerialQueue]).
            Public attribute; IdleMonitorPlugin references this dict directly.
        _registry: ChannelRegistry, resolved in on_start via get_dependency
    """

    depends_on = {"registry"}

    def __init__(self, pm) -> None:
        self.pm = pm
        self.client: LLMClient | None = None
        self.tools: dict[str, Callable] = {}
        self.tool_schemas: list[dict] = []
        self.queues: dict[str, SerialQueue] = {}
        self._registry: ChannelRegistry | None = None
        self._max_tool_result_chars: int = 100_000

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

        # Hook: should_process_message (broadcast, reject-wins)
        results = await self.pm.ahook.should_process_message(
            channel=channel, sender=sender, text=text,
        )
        gate_result = resolve_hook_results(results, "should_process_message", HookStrategy.REJECT_WINS)
        if gate_result is False:
            logger.info(
                "message rejected by should_process_message hook",
                extra={"channel": channel.id, "sender": sender},
            )
            return

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
        if channel_id not in self.queues:
            q = SerialQueue()
            q.start(self._process_queue_item)
            self.queues[channel_id] = q
        return self.queues[channel_id]

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
        if channel.conversation is None:
            results = await self.pm.ahook.ensure_conversation(channel=channel)
            result = resolve_hook_results(results, "ensure_conversation", HookStrategy.ACCEPT_WINS)
            if result is None:
                logger.error(
                    "no persistence plugin initialized conversation for %s",
                    channel.id,
                )
                return
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
            await self.pm.ahook.compact_conversation(
                conversation=conv, client=self.client,
                max_tokens=resolved["max_context_tokens"],
            )
        except Exception:
            logger.warning("compaction failed, skipping", exc_info=True)

        # 6. Let plugins inject context before the LLM call
        try:
            await self.pm.ahook.before_agent_turn(channel=channel)
        except Exception:
            logger.warning(
                "before_agent_turn hook failed, skipping", exc_info=True
            )

        # 7. Build prompt and call run_agent_turn (single LLM invocation)
        messages = conv.build_prompt()

        try:
            result = await run_agent_turn(self.client, messages, self.tool_schemas)
        except Exception as exc:
            logger.exception("run_agent_turn failed for channel %s", channel.id)
            results = await self.pm.ahook.on_llm_error(channel=channel, error=exc)
            error_msg = resolve_hook_results(
                results, "on_llm_error", HookStrategy.VALUE_FIRST, pm=self.pm,
            )
            if error_msg is None:
                error_msg = DEFAULT_LLM_ERROR_MESSAGE
            try:
                await self.pm.ahook.send_message(
                    channel=channel,
                    text=error_msg,
                )
            except Exception:
                logger.warning(
                    "send_message hook failed on error path",
                    exc_info=True,
                    extra={"channel": channel.id},
                )
            return

        # 8. Persist assistant message (run_agent_turn already appended to messages in place)
        await conv.append(result.message)

        # 9. Let plugins post-process the in-memory assistant message (e.g., strip reasoning_content)
        try:
            await self.pm.ahook.after_persist_assistant(channel=channel, message=conv.messages[-1])
        except Exception:
            logger.warning(
                "after_persist_assistant hook failed",
                exc_info=True,
                extra={"channel": channel.id},
            )

        # 10. Dispatch tool calls or send text response
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
            results = await self.pm.ahook.transform_display_text(
                channel=channel, text=result.text, result_message=result.message,
            )
            transformed = resolve_hook_results(
                results, "transform_display_text", HookStrategy.VALUE_FIRST, pm=self.pm,
            )
            display_response = (transformed if transformed is not None else result.text) or MAX_TURNS_FALLBACK_MESSAGE
        else:
            # No tool calls — normal text response
            channel.turn_counter += 1
            results = await self.pm.ahook.transform_display_text(
                channel=channel, text=result.text, result_message=result.message,
            )
            transformed = resolve_hook_results(
                results, "transform_display_text", HookStrategy.VALUE_FIRST, pm=self.pm,
            )
            display_response = transformed if transformed is not None else result.text

        logger.info(
            "agent response sent",
            extra={"channel": channel.id, "latency_ms": result.latency_ms},
        )

        try:
            await self.pm.ahook.on_agent_response(
                channel=channel,
                request_text=request_text,
                response_text=display_response,
            )
        except Exception:
            logger.warning(
                "on_agent_response hook failed",
                exc_info=True,
                extra={"channel": channel.id},
            )
        try:
            await self.pm.ahook.send_message(
                channel=channel,
                text=display_response,
                latency_ms=result.latency_ms,
            )
        except Exception:
            logger.error(
                "send_message hook failed, response not delivered",
                exc_info=True,
                extra={"channel": channel.id},
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
                        max_result_chars=self._max_tool_result_chars,
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

    async def on_stop(self) -> None:
        # Cancel all channel queue consumers.
        for queue in self.queues.values():
            await queue.stop()

        await self._stop_plugin()

    async def _start_plugin(self, config: dict) -> None:
        """Initialize LLM client and tools."""
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        self.tools = {}
        self.tool_schemas = []
        llm_config = config.get("llm", {})
        agent_config = config.get("agent", {})
        self._max_tool_result_chars = agent_config.get("max_tool_result_chars", 100_000)

        # Breaking change: llm.main is required.
        main_config = llm_config["main"]
        self.client = LLMClient(
            base_url=main_config["base_url"],
            model=main_config["model"],
            api_key=main_config.get("api_key"),
            extra_body=main_config.get("extra_body"),
        )
        await self.client.start()

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
        """Close LLM client."""
        if self.client:
            await self.client.stop()
