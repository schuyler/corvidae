"""Agent — wires the agent loop into the hook system.

Merged from orchestrator.py, lifecycle.py, and processing.py.
File boundaries now match domain responsibility rather than execution phase.

Agent is the central plugin that:
  - Manages per-channel serial queues
  - Borrows the LLM client from LLMPlugin on startup
  - Borrows the tool registry and max_result_chars from ToolCollectionPlugin on startup
  - Processes inbound messages and notifications through the agent loop

Config:
    agent:
      chars_per_token: 4.0  # optional — deprecated; retained for config compatibility only

LLM client configuration is owned by LLMPlugin (llm: main:, llm: background:).
Tool result truncation configuration is owned by ToolCollectionPlugin (tools: max_result_chars:).

Logging:
    - INFO: on_start complete, on_message received, agent response sent
    - ERROR: LLM client not initialized, run_agent_turn failures, missing TaskQueue
    - Latency is tracked via AgentTurnResult.latency_ms
"""

import asyncio
import collections
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, fields as dc_fields
from enum import Enum

from corvidae.turn import AgentTurnResult, run_agent_turn
from corvidae.attribution import get_attribution, reset_attribution, set_attribution
from corvidae.channel import Channel, ChannelConfig, ChannelRegistry, resolve_system_prompt
from corvidae.context import ContextWindow, MessageType, DEFAULT_CHARS_PER_TOKEN
from corvidae.hooks import (
    CorvidaePlugin,
    HookStrategy,
    get_dependency,
    hookimpl,
    resolve_hook_results,
    resolve_single_result,
)
from corvidae.queue import SerialQueue
from corvidae.task import Task
from corvidae.tool import dispatch_tool_call

logger = logging.getLogger("corvidae.agent")

# Keys that belong to the framework config layer (ChannelConfig fields).
# Derived at module load time so it stays in sync when new fields are added.
FRAMEWORK_KEYS = {f.name for f in dc_fields(ChannelConfig)}

DEFAULT_LLM_ERROR_MESSAGE = "Sorry, I encountered an error and could not process your message."
# Note: same text as MAX_ROUNDS_REACHED_MESSAGE in tools/subagent.py — kept in sync.
MAX_TURNS_FALLBACK_MESSAGE = "(max tool-calling rounds reached)"

# Bound on the originating-text LRU (Agent._originating_text). Keyed by
# exchange_key, not per-channel — user messages interleave mid-cycle by
# design (§3.3), so a single per-channel slot would be clobbered.
ORIGINATING_TEXT_LRU_MAXSIZE = 512


def mint_exchange_key() -> str:
    """Mint a time-sortable, globally-unique exchange key.

    Shape: ``<hex-time>-<hex12>`` (§3.1 timestamp-prefixed-hex convention).
    The hex-time prefix is monotone-nondecreasing across calls (wall-clock
    seconds); the uuid4 suffix guarantees distinctness within the same
    second.
    """
    return f"{int(time.time()):x}-{uuid.uuid4().hex[:12]}"


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
        exchange_key: The exchange this item belongs to (Phase 2). None for
            notifications until dequeue-time resolution runs.
        origin: 'user'|'reminder'|'critique'|'heartbeat'|'task'. None until
            dequeue-time resolution runs (for notifications).
        originates_exchange: True when this item mints/owns the exchange
            (USER items, and dequeue-minted notifications). False for
            tool-result notifications that inherited a key from a prior
            item. Tracked on the item, never re-derived, so
            on_message_persisted firing discipline stays correct.
    """

    role: QueueItemRole
    content: str
    channel: Channel
    sender: str | None = None
    source: str | None = None
    tool_call_id: str | None = None
    meta: dict = field(default_factory=dict)
    exchange_key: str | None = None
    origin: str | None = None
    originates_exchange: bool = False


class Agent(CorvidaePlugin):
    """Plugin that wires the agent loop into the hook system.

    Attributes:
        pm: Plugin manager instance (untyped due to pluggy limitations)
        _client: LLM client for chat completions (borrowed from LLMPlugin)
        _tools: Dict mapping tool names to async callable functions (from ToolCollectionPlugin)
        _tool_schemas: List of tool schemas for LLM function calling (from ToolCollectionPlugin)
        queues: Per-channel serial queues (dict[channel_id, SerialQueue]).
            Public attribute; IdleMonitorPlugin references this dict directly.
        _registry: ChannelRegistry, resolved in on_start via get_dependency
    """

    depends_on = frozenset({"registry", "task", "llm", "tools"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._client = None
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: list[dict] = []
        self.queues: dict[str, SerialQueue] = {}
        self._registry: ChannelRegistry | None = None
        self._max_tool_result_chars: int = 100_000
        self._chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
        self._base_dir = None
        self._idle_cooldown: float = 30.0
        self._request_logprobs: bool = False
        self._last_idle_fire: float = 0.0
        self._idle_firing: bool = False
        # Exchange-keyed LRU of originating text (the exchange's true
        # originating USER/notification text), bounded so long-running
        # exchanges don't leak memory. Never a per-channel slot — user
        # messages interleave mid-cycle by design (§3.3).
        self._originating_text: collections.OrderedDict[str, str] = collections.OrderedDict()

    @property
    def tools(self) -> dict[str, Callable]:
        """Public read-only access to the tool dict."""
        return self._tools

    @property
    def tool_schemas(self) -> list[dict]:
        """Public read-only access to tool schemas."""
        return self._tool_schemas

    async def _maybe_fire_idle(self) -> None:
        """Fire on_idle if all queues are empty and cooldown has elapsed.

        Uses a boolean guard (_idle_firing) rather than a lock. This works
        because asyncio is single-threaded: there is no await between the
        flag check and the flag set, so no other task can interleave and
        pass the check before the flag is raised.
        """
        if self._idle_firing:
            return
        for q in self.queues.values():
            if not q.is_empty:
                return
        # Check task queue
        task_plugin = self.pm.get_plugin("task")
        if task_plugin is not None:
            tq = getattr(task_plugin, "task_queue", None)
            if tq is not None and not tq.is_idle:
                return
        if time.monotonic() - self._last_idle_fire < self._idle_cooldown:
            return
        self._idle_firing = True
        try:
            await self.pm.ahook.on_idle()
            self._last_idle_fire = time.monotonic()
        except Exception:
            logger.warning("on_idle hook raised exception", exc_info=True)
        finally:
            self._idle_firing = False

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        agent_config = config.get("agent", {})
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        # Static operator-only flag (not a per-channel runtime tunable, so it
        # does not go through resolve_tunable). Agent retains no config dict —
        # on_init extracts derived scalars only — so step 7 reads this scalar.
        self._request_logprobs = agent_config.get("request_logprobs", False)
        daemon_config = config.get("daemon", {})
        self._idle_cooldown = daemon_config.get("idle_cooldown_seconds", 30.0)

    async def on_start(self, config: dict) -> None:
        await self._start_plugin(config)

    @hookimpl
    async def on_message(self, channel, sender: str, text: str) -> None:
        if not self._client:
            logger.error("on_message: LLM client not initialized")
            return

        logger.info(
            "on_message received",
            extra={"channel": channel.id, "sender": sender},
        )

        # Mint the exchange key BEFORE the gate fires, so gate plugins can
        # correlate their decision with the exchange (Phase 2 §4).
        exchange_key = mint_exchange_key()

        # Hook: should_process_message (broadcast, reject-wins)
        results = await self.pm.ahook.should_process_message(
            channel=channel, sender=sender, text=text, exchange_key=exchange_key,
        )
        gate_result = resolve_hook_results(results, "should_process_message", HookStrategy.REJECT_WINS)
        if gate_result is False:
            logger.info(
                "message rejected by should_process_message hook",
                extra={"channel": channel.id, "sender": sender},
            )
            try:
                await self.pm.ahook.on_message_rejected(
                    channel=channel, exchange_key=exchange_key, sender=sender, text=text,
                )
            except Exception:
                logger.warning("on_message_rejected hook failed", exc_info=True, extra={"channel": channel.id})
            return

        try:
            await self.pm.ahook.on_message_admitted(
                channel=channel, exchange_key=exchange_key, sender=sender, text=text,
            )
        except Exception:
            logger.warning("on_message_admitted hook failed", exc_info=True, extra={"channel": channel.id})

        item = QueueItem(
            role=QueueItemRole.USER,
            content=text,
            channel=channel,
            sender=sender,
            exchange_key=exchange_key,
            origin="user",
            originates_exchange=True,
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

    def _build_conversation_message(
        self, item: QueueItem
    ) -> tuple[dict, str] | None:
        """Build the conversation message dict and request_text from a QueueItem.

        Returns (conversation_message, request_text), or None if the item role
        is unrecognized (logs an error in that case).
        """
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
            logger.error("_build_conversation_message: unknown item role %r", item.role)
            return None
        return conversation_message, request_text

    async def _run_turn(
        self,
        channel: Channel,
        messages: list[dict],
        tool_schemas: list[dict],
        llm_overrides: dict | None,
    ) -> AgentTurnResult | None:
        """Call run_agent_turn and handle LLM errors.

        Returns the AgentTurnResult on success, or None on error (error message
        already sent to the channel via send_message hook).
        """
        try:
            return await run_agent_turn(self._client, messages, tool_schemas, extra_body=llm_overrides)
        except Exception as exc:
            logger.exception("run_agent_turn failed for channel %s", channel.id)
            error_msg = await self.pm.ahook.on_llm_error(channel=channel, error=exc)
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
            return None

    async def _resolve_display_text(
        self,
        channel: Channel,
        result: AgentTurnResult,
        fallback: str | None,
    ) -> str:
        """Call transform_display_text hook and resolve the result.

        Returns the transformed text if the hook returns a non-None value,
        otherwise returns the hook input text. If fallback is not None,
        uses fallback when the resolved text is falsy (empty string or None).
        When fallback is None, an empty resolved text is returned as-is.
        """
        try:
            resolved = await self.pm.ahook.transform_display_text(
                channel=channel, text=result.text, result_message=result.message,
            )
        except Exception:
            logger.warning(
                "transform_display_text hook failed",
                exc_info=True,
                extra={"channel": channel.id},
            )
            resolved = result.text
        if resolved is None:
            resolved = result.text
        if fallback is not None:
            return resolved or fallback
        return resolved

    async def _handle_response(
        self,
        result: AgentTurnResult,
        channel: Channel,
        max_turns_limit: int,
        request_text: str,
        item: QueueItem | None = None,
    ) -> None:
        """Dispatch tool calls or send text response.

        Implements the decision point at step 10 of _process_queue_item:
        - Tool calls under limit: increment counter, dispatch, return.
        - Tool calls at limit: NO counter increment, resolve display text
          with MAX_TURNS_FALLBACK_MESSAGE, fire on_agent_response, send message.
        - No tool calls: increment counter, resolve display text, fire
          on_agent_response, send message.

        ``item`` provides the current exchange's key/origin (for the
        enriched on_agent_response call) and, via _dispatch_tool_calls,
        stamps the tool cycle's Tasks with the same exchange.
        """
        if result.tool_calls and channel.turn_counter < max_turns_limit:
            channel.turn_counter += 1
            # Show intermediate text (e.g. "Let me look that up...") before dispatching tools
            if result.text:
                try:
                    await self.pm.ahook.send_progress(channel=channel, text=result.text)
                except Exception:
                    logger.warning("send_progress hook failed", exc_info=True, extra={"channel": channel.id})
            await self._dispatch_tool_calls(result.tool_calls, channel)
            # Do NOT send text response; return and wait for tool results
            return
        elif result.tool_calls:
            # Max turns reached — suppress tool dispatch, send fallback text
            logger.warning(
                "max_turns reached, suppressing tool calls",
                extra={"channel": channel.id, "turn_counter": channel.turn_counter},
            )
            display_response = await self._resolve_display_text(
                channel, result, fallback=MAX_TURNS_FALLBACK_MESSAGE
            )
        else:
            # No tool calls — normal text response
            channel.turn_counter += 1
            display_response = await self._resolve_display_text(channel, result, fallback=None)

        logger.info(
            "agent response sent",
            extra={
                "channel": channel.id,
                "latency_ms": result.latency_ms,
                "tool_calls_count": len(result.tool_calls) if result.tool_calls else 0,
                "turn_counter": channel.turn_counter,
                "pending_tools": len(channel.pending_tool_call_ids),
            },
        )

        exchange_key = item.exchange_key if item is not None else None
        origin = item.origin if item is not None else None
        originating_text = (
            self._originating_text.get(exchange_key)
            if exchange_key is not None
            else None
        )
        try:
            await self.pm.ahook.on_agent_response(
                channel=channel,
                request_text=request_text,
                response_text=display_response,
                exchange_key=exchange_key,
                origin=origin,
                originating_text=originating_text,
                logprobs=result.logprobs,
                withheld=False,
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

        # Exchange key/origin resolution MUST happen here, before the
        # set_attribution call below, so the widened attribution carries
        # the key into every LLM call in the turn (including tool Tasks
        # created during dispatch, which snapshot the context at creation).
        # If resolution ran inside _process_queue_item_attributed instead
        # (after attribution is set), usage_log.exchange_key would be null
        # for the whole turn.
        if item.exchange_key is None:
            inherited_key = item.meta.get("exchange_key") if item.meta else None
            if inherited_key is not None:
                item.exchange_key = inherited_key
                item.origin = item.meta.get("origin")
                item.originates_exchange = False
            else:
                item.exchange_key = mint_exchange_key()
                item.origin = (item.meta.get("origin") if item.meta else None) or "task"
                item.originates_exchange = True

        # Attribute all LLM activity in this turn (including tool Tasks
        # created during dispatch, which capture the context at creation)
        # to the turn stage and this channel. Compaction re-labels itself
        # around its own LLM call and restores this attribution after.
        attribution_token = set_attribution(
            stage="turn", channel_id=channel.id,
            exchange_key=item.exchange_key, origin=item.origin,
        )
        try:
            await self._process_queue_item_attributed(item, channel)
        finally:
            reset_attribution(attribution_token)

    async def _process_queue_item_attributed(
        self, item: QueueItem, channel: Channel
    ) -> None:
        """Body of _process_queue_item, run with turn attribution set."""
        import time as _time
        _t0 = _time.monotonic()
        _phases: dict[str, float] = {}

        # Build the conversation message based on item role
        msg_result = self._build_conversation_message(item)
        if msg_result is None:
            return
        conversation_message, request_text = msg_result

        _t_conv_init_start = _time.monotonic()

        # 1. Lazy-initialize conversation on the channel
        if channel.conversation is None:
            resolved_cfg = self._registry.resolve_config(channel)
            conv = ContextWindow(channel.id, chars_per_token=self._chars_per_token)
            base_dir = self._base_dir
            from pathlib import Path
            if base_dir is None:
                base_dir = Path(".")
            conv.system_prompt = resolve_system_prompt(resolved_cfg["system_prompt"], base_dir)
            history = await self.pm.ahook.load_conversation(channel=channel)
            if history:
                conv.messages = history
            channel.conversation = conv
        conv = channel.conversation
        _phases["conv_init"] = _time.monotonic() - _t_conv_init_start

        # 2. Reset turn_counter on user messages
        if item.role == QueueItemRole.USER:
            channel.turn_counter = 0

        # 3. Resolve config and read max_turns_limit
        resolved = self._registry.resolve_config(channel)
        max_turns_limit = resolved["max_turns"]

        # 4. Append message to conversation log and fire persistence event.
        # The resolved rowid attaches to the WINDOW copy (conv.append
        # shallow-copies, so the local dict would not update the window).
        _t_persist_start = _time.monotonic()
        conv.append(conversation_message)
        rowid = None
        try:
            results = await self.pm.ahook.on_conversation_event(
                channel=channel,
                message=conversation_message,
                message_type=MessageType.MESSAGE,
            )
            rowid = resolve_single_result(results, "on_conversation_event")
            if rowid is not None:
                conv.messages[-1]["_db_id"] = rowid
        except Exception:
            logger.warning("on_conversation_event hook failed", exc_info=True)
        _phases["persist"] = _time.monotonic() - _t_persist_start

        # 4c. Fire on_message_persisted only when this item originates its
        # exchange (USER items, and notification items whose key was minted
        # at dequeue). Mid-exchange tool-result rows, injected CONTEXT, and
        # assistant rows never fire it — per-row firing under one key would
        # overwrite the rowid with each successive row (§4.5). Also record
        # the exchange's true originating text in the bounded LRU.
        if item.originates_exchange and item.exchange_key is not None:
            self._originating_text[item.exchange_key] = request_text
            self._originating_text.move_to_end(item.exchange_key)
            while len(self._originating_text) > ORIGINATING_TEXT_LRU_MAXSIZE:
                self._originating_text.popitem(last=False)
            if rowid is not None:
                try:
                    await self.pm.ahook.on_message_persisted(
                        channel=channel, exchange_key=item.exchange_key,
                        rowid=rowid, origin=item.origin,
                    )
                except Exception:
                    logger.warning("on_message_persisted hook failed", exc_info=True, extra={"channel": channel.id})

        # 4b. Batch tool results: if this is a tool-result notification and
        # there are still pending tool calls from the current batch, skip the
        # LLM call. The serial queue ensures results arrive one at a time; the
        # last result to arrive clears pending_tool_call_ids and triggers the
        # LLM call. User messages can interleave — they don't clear the pending
        # set, so the agent remains responsive while tools run.
        if item.role == QueueItemRole.NOTIFICATION and item.tool_call_id:
            channel.pending_tool_call_ids.discard(item.tool_call_id)
            if channel.pending_tool_call_ids:
                logger.debug(
                    "tool result buffered, pending: %d",
                    len(channel.pending_tool_call_ids),
                    extra={"channel": channel.id, "tool_call_id": item.tool_call_id},
                )
                return

        # 5. Compact if approaching context limit
        _t_compact_start = _time.monotonic()
        try:
            await self.pm.ahook.compact_conversation(
                channel=channel, conversation=conv,
                max_tokens=resolved["max_context_tokens"],
            )
        except Exception:
            logger.warning("compaction failed, skipping", exc_info=True)
        _phases["compact"] = _time.monotonic() - _t_compact_start

        # 6. Let plugins inject context before the LLM call
        _t_before_turn_start = _time.monotonic()
        msg_count_before = len(conv.messages)
        try:
            await self.pm.ahook.before_agent_turn(
                channel=channel, exchange_key=item.exchange_key, origin=item.origin,
            )
        except Exception:
            logger.warning(
                "before_agent_turn hook failed, skipping", exc_info=True
            )
        # Fire on_conversation_event for any messages injected by
        # before_agent_turn. Messages already carrying a _db_id were
        # persisted by their producer (e.g. the admission funnel) — firing
        # again would double-store them.
        for msg in conv.messages[msg_count_before:]:
            if isinstance(msg.get("_db_id"), int):
                continue
            mt = msg.get("_message_type", MessageType.MESSAGE)
            clean = {k: v for k, v in msg.items() if not k.startswith("_")}
            try:
                results = await self.pm.ahook.on_conversation_event(
                    channel=channel, message=clean, message_type=mt,
                )
                rowid = resolve_single_result(results, "on_conversation_event")
                if rowid is not None:
                    msg["_db_id"] = rowid
            except Exception:
                logger.warning("on_conversation_event hook failed", exc_info=True)
        _phases["before_turn"] = _time.monotonic() - _t_before_turn_start

        # 7. Build prompt and call run_agent_turn (single LLM invocation)
        _t_llm_start = _time.monotonic()
        messages = conv.build_prompt()
        # Dotted keys are plugin tunables (see corvidae.tuning), never LLM
        # inference params — they must not leak into the request body.
        llm_overrides = {
            k: v
            for k, v in channel.runtime_overrides.items()
            if k not in FRAMEWORK_KEYS and "." not in k
        }
        if self._request_logprobs:
            llm_overrides["logprobs"] = True

        result = await self._run_turn(channel, messages, self._tool_schemas, llm_overrides or None)
        if result is None:
            return

        # 8. Append assistant message and fire persistence event, attaching
        # the resolved rowid to the window copy (same rule as step 4).
        conv.append(result.message)
        try:
            results = await self.pm.ahook.on_conversation_event(
                channel=channel,
                message=result.message,
                message_type=MessageType.MESSAGE,
            )
            rowid = resolve_single_result(results, "on_conversation_event")
            if rowid is not None:
                conv.messages[-1]["_db_id"] = rowid
        except Exception:
            logger.warning("on_conversation_event hook failed", exc_info=True)

        # 8b. Fire send_thinking if the LLM produced reasoning_content
        reasoning_text = result.message.get("reasoning_content")
        if reasoning_text:
            try:
                await self.pm.ahook.send_thinking(channel=channel, text=reasoning_text)
            except Exception:
                logger.warning("send_thinking hook failed", exc_info=True, extra={"channel": channel.id})

        # 9. Let plugins post-process the in-memory assistant message (e.g., strip reasoning_content)
        try:
            await self.pm.ahook.after_persist_assistant(channel=channel, message=conv.messages[-1])
        except Exception:
            logger.warning(
                "after_persist_assistant hook failed",
                exc_info=True,
                extra={"channel": channel.id},
            )
        _phases["llm"] = _time.monotonic() - _t_llm_start

        # 10. Dispatch tool calls or send text response
        await self._handle_response(result, channel, max_turns_limit, request_text, item)

        # 11. Push-based idle detection: check if the system became idle
        await self._maybe_fire_idle()

        # 12. Log phase timing breakdown
        _total = _time.monotonic() - _t0
        _phases["other"] = max(0, _total - sum(_phases.values()))
        logger.info(
            "queue item timing",
            extra={
                "channel": item.channel.id,
                "role": item.role.value,
                "total_ms": round(_total * 1000),
                **{f"{k}_ms": round(v * 1000) for k, v in _phases.items()},
                "prompt_tokens": result.message.get("usage", {}).get("prompt_tokens") if hasattr(result, "message") and isinstance(result.message, dict) else None,
                "message_count": len(conv.messages) if conv else 0,
            },
        )

    async def _dispatch_tool_calls(
        self, tool_calls: list[dict], channel: Channel
    ) -> None:
        """Enqueue each tool call as a Task on the TaskQueue.

        Each task's work closure calls dispatch_tool_call(), which handles
        JSON parsing, unknown-tool detection, invocation, error wrapping,
        logging, and the process_tool_result hook. Returns the result
        string; the TaskPlugin delivers it via on_notify.

        Logs an error and returns without enqueuing if the TaskQueue is
        unavailable (TaskPlugin not registered).

        Stamps each Task with the current exchange's key/origin, read from
        the attribution contextvar set by _process_queue_item at dequeue
        (this coroutine runs under that attribution). Every turn of a tool
        cycle inherits its exchange's key and origin this way, however many
        hops (§3.3).
        """
        task_queue = getattr(self.pm.get_plugin("task"), "task_queue", None)
        if task_queue is None:
            logger.error("tool calls requested but no TaskQueue available")
            return

        current_attribution = get_attribution()
        exchange_key = current_attribution.get("exchange_key")
        origin = current_attribution.get("origin")

        # Record all call IDs so we can wait for every result before the next LLM call.
        channel.pending_tool_call_ids = {call["id"] for call in tool_calls}

        for call in tool_calls:
            call_id = call["id"]
            fn_name = call["function"]["name"]

            async def make_work(call=call):
                """Execute a single tool call via dispatch_tool_call."""
                result = await dispatch_tool_call(
                    call, self._tools,
                    channel=channel,
                    task_queue=task_queue,
                    max_result_chars=self._max_tool_result_chars,
                    pm=self.pm,
                )
                return result.content

            # Notify transports that a tool call has been dispatched
            try:
                await self.pm.ahook.send_tool_status(
                    channel=channel,
                    tool_name=fn_name,
                    status="dispatched",
                    args_summary=call["function"].get("arguments", "")[:200],
                )
            except Exception:
                logger.warning("send_tool_status hook failed", exc_info=True, extra={"channel": channel.id})

            task = Task(
                work=make_work,
                channel=channel,
                tool_call_id=call_id,
                description=f"tool:{fn_name}",
                exchange_key=exchange_key,
                origin=origin,
            )
            await task_queue.enqueue(task)

    async def on_stop(self) -> None:
        # Cancel all channel queue consumers.
        for queue in self.queues.values():
            await queue.stop()

        await self._stop_plugin()

    async def _start_plugin(self, config: dict) -> None:
        """Initialize LLM client and tools."""
        from pathlib import Path
        from corvidae.llm_plugin import LLMPlugin
        from corvidae.tool_collection import ToolCollectionPlugin
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        self._tools = {}
        self._tool_schemas = []
        self._base_dir = config.get("_base_dir", Path("."))

        # Get the LLM client from LLMPlugin (which owns the lifecycle).
        llm = get_dependency(self.pm, "llm", LLMPlugin)
        self._client = llm.main_client

        # Get tool registry and config from ToolCollectionPlugin.
        tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
        self._tools, self._tool_schemas = tools_plugin.get_tools()
        self._max_tool_result_chars = tools_plugin.max_result_chars

        logger.info(
            "on_start complete",
            extra={
                "tool_count": len(self._tools),
                "channel_count": len(self._registry.all()),
            },
        )

    async def _stop_plugin(self) -> None:
        """Release LLM client reference (lifecycle owned by LLMPlugin)."""
        self._client = None

    def refresh_tools(self) -> None:
        """Re-borrow tools from ToolCollectionPlugin after a hot-reload."""
        from corvidae.tool_collection import ToolCollectionPlugin
        tools_plugin = get_dependency(self.pm, "tools", ToolCollectionPlugin)
        self._tools, self._tool_schemas = tools_plugin.get_tools()
        self._max_tool_result_chars = tools_plugin.max_result_chars

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Re-read agent config and re-borrow the LLM client from LLMPlugin.

        Re-borrows _client because LLMPlugin may have swapped its main_client
        during this same reload cycle (Key Design Decision #3 in design doc).
        System prompt changes take effect only on new conversations (existing
        conversations retain their current ContextWindow.system_prompt).
        """
        # Re-read agent config values.
        agent_config = config.get("agent", {})
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        self._request_logprobs = agent_config.get("request_logprobs", False)
        daemon_config = config.get("daemon", {})
        self._idle_cooldown = daemon_config.get("idle_cooldown_seconds", 30.0)

        # Re-borrow the LLM client from LLMPlugin after a possible swap.
        # Avoids isinstance check so MagicMock(spec=LLMPlugin) works in tests.
        llm = self.pm.get_plugin("llm")
        if llm is not None and hasattr(llm, "get_client"):
            self._client = llm.get_client()

        logger.info("on_config_reload: agent config updated")

    @hookimpl(trylast=True)
    async def on_plugin_added(self, name: str, plugin: object) -> None:
        """Refresh tools when a plugin is added at runtime.

        trylast=True ensures ToolCollectionPlugin.rebuild_registry() runs
        first, same ordering rationale as on_start.
        """
        self.refresh_tools()

    @hookimpl(trylast=True)
    async def on_plugin_removed(self, name: str) -> None:
        """Refresh tools when a plugin is removed at runtime.

        trylast=True ensures ToolCollectionPlugin.rebuild_registry() runs
        first, same ordering rationale as on_start.
        """
        self.refresh_tools()


# Backward-compatible alias; will be deprecated in a future release.
AgentPlugin = Agent
