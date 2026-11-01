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
      forks:                # background processes as forks of the window (#51)
        enabled: false      # see corvidae.fork for the full section

LLM client configuration is owned by LLMPlugin (llm: main:, llm: background:).
Tool result truncation configuration is owned by ToolCollectionPlugin (tools: max_result_chars:).

Logging:
    - INFO: on_start complete, on_message received, agent response sent
    - ERROR: LLM client not initialized, run_agent_turn failures, missing TaskQueue
    - Latency is tracked via AgentTurnResult.latency_ms
"""

import asyncio
import logging
from typing import TYPE_CHECKING

import aiohttp
from collections.abc import Callable
from dataclasses import dataclass, field, fields as dc_fields, replace
from enum import Enum

from corvidae.clock import Clock, Timer
from corvidae.llm import LLMHTTPError, is_backend_unreachable

if TYPE_CHECKING:
    from corvidae.tool_collection import ToolCollectionPlugin
from corvidae.turn import AgentTurnResult, run_agent_turn
from corvidae.attribution import get_attribution, reset_attribution, set_attribution
from corvidae.channel import (
    Channel,
    ChannelConfig,
    ChannelRegistry,
    INFERENCE_KEYS,
    resolve_system_prompt,
)
from corvidae.context import (
    DEFAULT_CHARS_PER_TOKEN,
    UNSENT_MARKER_LABEL,  # noqa: F401 (re-exported for tests and plugins)
    ContextWindow,
    count_tokens,
    message_tokens,
    MessageType,
    unsent_marker_text,
)
# Re-exported: mint_trace_id lives in a leaf module so transports can
# import it without this module; existing imports from here keep working.
from corvidae.trace import mint_trace_id  # noqa: F401
from corvidae.fork import (
    ForkRefused,
    ForkResult,
    ForksConfig,
    ForkUnavailable,
    build_fork_prompt,
    looks_like_tool_call,
    parse_forks_config,
)
from corvidae.idle import IdleTracker
from corvidae.hooks import (
    CorvidaePlugin,
    get_dependency,
    hookimpl,
    resolve_reject_wins,
    resolve_single_result,
)
from corvidae.outbound import (
    OutboundMessage,
    OutboundProgress,
    OutboundThinking,
    OutboundToolStatus,
)
from corvidae.queue import SerialQueue
from corvidae.task import Task
from corvidae.tool import DEFAULT_TOOL_TIMEOUTS, ToolTimeouts, dispatch_tool_call
from corvidae.tool_policy import (
    ToolView,
    build_tool_view,
    disabled_on,
    disabling_keys,
    refusal_message,
)

logger = logging.getLogger("corvidae.agent")

# Keys that belong to the framework config layer (ChannelConfig fields).
# Derived at module load time so it stays in sync when new fields are added.
FRAMEWORK_KEYS = {f.name for f in dc_fields(ChannelConfig)}

# User-facing wording: names the failure class in calm language and gives the
# user a concrete next step. The template is private because every call site
# must supply the failure it caught; there is no bare default to send.
_LLM_ERROR_TEMPLATE = (
    "Sorry — {cause}, so I couldn't process your message. "
    "Please try again in a moment."
)
# Note: same text as MAX_ROUNDS_REACHED_MESSAGE in tools/subagent.py — kept in sync.
MAX_TURNS_FALLBACK_MESSAGE = "(max tool-calling rounds reached)"



def _describe_llm_failure(exc: Exception) -> str:
    """Map an LLM failure to a calm, human-readable cause for the user.

    Keeps Python type names out of user-facing text while still telling the
    user which class of failure happened.
    """
    # A timeout is the most common transient failure: say so plainly.
    if isinstance(exc, TimeoutError):
        return "the model request timed out"
    # Connection failures mean the model server could not be reached at all.
    if isinstance(exc, aiohttp.ClientConnectionError):
        return "I couldn't reach the model server"
    # HTTP failures from the provider are still request failures to the user.
    if isinstance(exc, LLMHTTPError):
        return f"the model server returned an error (HTTP {exc.status})"
    # Anything else: name the class without guessing at a more specific cause.
    return "the model request failed"


def _llm_error_message(exc: Exception) -> str:
    """Build the complete user-facing message for a specific LLM failure.

    The exception is required: callers always know which failure happened, so
    omitting the cause is a programming error rather than a supported default.
    """
    return _LLM_ERROR_TEMPLATE.format(cause=_describe_llm_failure(exc))


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
        meta: Extensible metadata, opaque to core. Plugins stamp their own
            keys (e.g. ``meta["origin"]``) and read them back downstream.
        trace_id: The trace this item belongs to. None for
            notifications until dequeue-time resolution runs.
        originating: True when this item mints/owns the trace id
            (USER items, and dequeue-minted notifications). False for
            tool-result notifications that inherited an id from a prior
            item. Tracked on the item, never re-derived, so
            on_message_persisted firing discipline stays correct.
        retry: True for the re-run of a turn whose LLM call could not reach
            the backend (#281). The item is a copy of the failed one with
            ``originating=False``; its message is already in the window and
            the log, so steps 2 and 4-6 of processing skip it.
    """

    role: QueueItemRole
    content: str
    channel: Channel
    sender: str | None = None
    source: str | None = None
    tool_call_id: str | None = None
    meta: dict = field(default_factory=dict)
    trace_id: str | None = None
    originating: bool = False
    retry: bool = False


#: Backoff for re-running a turn the backend could not answer (#281): the
#: delay for attempt n is min(BASE * 2**n, MAX) seconds (5, 10, 20, 40, 60...).
TURN_RETRY_BASE_DELAY = 5.0
TURN_RETRY_MAX_DELAY = 60.0


@dataclass
class _TurnRetry:
    """The one pending retry of a channel: its item, attempt number and timer."""

    item: QueueItem
    attempt: int
    timer: Timer



#: Roles agent.mid_conversation_system_role accepts (GitHub issue #49).
MID_CONVERSATION_SYSTEM_ROLES = ("system", "user")


def _resolve_mid_conversation_system_role(agent_config: dict) -> str:
    """Read agent.mid_conversation_system_role, warning on and ignoring an
    unsupported value so a typo degrades to today's behavior."""
    role = agent_config.get("mid_conversation_system_role", "system")
    if role not in MID_CONVERSATION_SYSTEM_ROLES:
        logger.warning(
            "agent.mid_conversation_system_role %r is not one of %s; using 'system'",
            role, ", ".join(MID_CONVERSATION_SYSTEM_ROLES),
        )
        return "system"
    return role


class Agent(CorvidaePlugin):
    """Plugin that wires the agent loop into the hook system.

    Attributes:
        pm: Plugin manager instance (untyped due to pluggy limitations)
        _client: LLM client for chat completions (borrowed from LLMPlugin)
        _tools: Dict mapping tool names to async callable functions (from ToolCollectionPlugin)
        _tool_schemas: List of tool schemas for LLM function calling (from ToolCollectionPlugin)
        queues: Per-channel serial queues (dict[channel_id, SerialQueue]).
            Idle detection does not read them: Agent pushes activity and
            settle events to the kernel IdleTracker.
        _registry: ChannelRegistry, resolved in on_start via get_dependency
    """

    depends_on = frozenset({"registry", "task", "llm", "tools"})

    # Class-level default so an Agent built without __init__ (as some tests do)
    # still reads as having no tracker.
    _idle: IdleTracker | None = None

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._client = None
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: list[dict] = []
        # The ToolCollectionPlugin and the registry generation the two
        # fields above were read at (#109); None until _start_plugin runs.
        self._tools_plugin: "ToolCollectionPlugin | None" = None
        self._tools_generation: int = 0
        self.queues: dict[str, SerialQueue] = {}
        # (channel_id, tool_call_id) of tool results queued but not yet
        # processed, so deliver_tool_error never adds a second result (#173).
        self._queued_tool_results: set[tuple[str, str]] = set()
        self._registry: ChannelRegistry | None = None
        self._max_tool_result_chars: int = 100_000
        # Per-call tool time bounds, copied from ToolCollectionPlugin at start.
        self._tool_call_timeouts: ToolTimeouts = DEFAULT_TOOL_TIMEOUTS
        self._chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
        self._base_dir = None
        self._request_logprobs: bool = False
        self._mid_conversation_system_role: str = "system"
        self._forks: ForksConfig = ForksConfig()
        # Kernel idle tracker (registered as "idle"), resolved in _start_plugin;
        # None when the plugin manager was built without one.
        self._idle: IdleTracker | None = None
        # True between a completed on_start and on_stop; see `started`.
        self._started: bool = False
        # Set when the Agent has started or has stopped; on_message awaits it
        # while no client exists yet (#71). _stopping tells a waiter which.
        self._started_event: asyncio.Event = asyncio.Event()
        self._stopping: bool = False
        # Channel id -> the pending retry of its unreachable-backend turn.
        self._turn_retries: dict[str, _TurnRetry] = {}

    @property
    def tools(self) -> dict[str, Callable]:
        """Public read-only access to the tool dict."""
        return self._tools

    @property
    def tool_schemas(self) -> list[dict]:
        """Public read-only access to tool schemas."""
        return self._tool_schemas

    @property
    def started(self) -> bool:
        """True once on_start has completed, False again after on_stop.

        Runtime broadcasts on_start to every plugin concurrently and only
        then starts Agent, so a producer whose background task begins in
        its own on_start (SchedulerPlugin) can run before the queues have
        a registry and client. Such producers wait for this before
        calling on_notify.
        """
        return self._started

    def _note_activity(self, channel_id: str) -> None:
        """Tell the idle tracker that ``channel_id`` has an inbound item."""
        # No tracker means idle signals are off; nothing to report.
        if self._idle is None:
            return
        self._idle.note_activity(channel_id)

    def _note_settled(self, channel: Channel) -> None:
        """Tell the idle tracker ``channel`` is quiet, if no work remains on it."""
        if self._idle is None:
            return
        # Another queued item means the channel still has work to do.
        queue = self.queues.get(channel.id)
        if queue is not None and not queue.is_empty:
            return
        # Outstanding tool calls (subagent tasks included) are in-flight work;
        # their results arrive via on_notify and settle the channel again.
        # Known edge: ACP's cancel_prompt clears pending_tool_call_ids without
        # processing an item, so that channel stays busy until its next item.
        if channel.pending_tool_call_ids:
            return
        self._idle.note_settled(channel.id)

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        agent_config = config.get("agent", {})
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        # Static operator-only flag (not a per-channel runtime tunable, so it
        # does not go through resolve_tunable). Agent retains no config dict —
        # on_init extracts derived scalars only — so step 7 reads this scalar.
        self._request_logprobs = agent_config.get("request_logprobs", False)
        # Role for non-leading system messages at prompt build (issue #49).
        self._mid_conversation_system_role = _resolve_mid_conversation_system_role(agent_config)
        # agent.forks: static operator flags (validated at startup by
        # validate_agent_config, so a bad section never gets this far).
        self._forks = parse_forks_config(agent_config)

    async def on_start(self, config: dict) -> None:
        await self._start_plugin(config)

    @hookimpl
    async def on_message(self, channel, sender: str, text: str, trace_id: str) -> None:
        # The transport mints the id. A missing one would silently break the
        # transport's link between this message and its lifecycle hooks, so
        # fail loudly -- before the readiness check, so a mis-wired transport
        # is caught even while the agent is starting.
        if not trace_id:
            raise ValueError("on_message requires a non-empty trace_id from the transport")

        # A transport can deliver from its own on_start (step 11), before the
        # Agent starts (step 12). Hold here, inside on_message, so every
        # transport is covered at one point and admission still fires before
        # on_message returns once the Agent starts (transports release their
        # typing slot in a finally if it hasn't).
        # Gate on _started, not _client: on_config_reload sets _client even
        # before the Agent has started, while _registry/_tools are still unset.
        if not self._started:
            await self._started_event.wait()
            if self._stopping or not self._started:
                # Agent is stopping or never started: nothing will process this.
                logger.warning(
                    "message on channel %s (trace_id %s) was not processed: "
                    "the agent was stopping or never started; resend it after the "
                    "daemon is running",
                    channel.id, trace_id,
                    extra={"channel": channel.id, "trace_id": trace_id},
                )
                return

        logger.info(
            "on_message received",
            extra={"channel": channel.id, "sender": sender},
        )

        # Hook: should_process_message (broadcast, reject-wins)
        results = await self.pm.ahook.should_process_message(
            channel=channel, sender=sender, text=text, trace_id=trace_id,
        )
        gate_result = resolve_reject_wins(results)
        if gate_result is False:
            logger.info(
                "message rejected by should_process_message hook",
                extra={"channel": channel.id, "sender": sender},
            )
            try:
                await self.pm.ahook.on_message_rejected(
                    channel=channel, trace_id=trace_id, sender=sender, text=text,
                )
            except Exception:
                logger.warning("on_message_rejected hook failed", exc_info=True, extra={"channel": channel.id})
            return

        try:
            await self.pm.ahook.on_message_admitted(
                channel=channel, trace_id=trace_id, sender=sender, text=text,
            )
        except Exception:
            logger.warning("on_message_admitted hook failed", exc_info=True, extra={"channel": channel.id})

        item = QueueItem(
            role=QueueItemRole.USER,
            content=text,
            channel=channel,
            sender=sender,
            trace_id=trace_id,
            originating=True,
        )
        # The gate admitted the item: the channel is busy from here.
        self._note_activity(channel.id)
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
        # Notifications (scheduler, task completions) are inbound work too.
        self._note_activity(channel.id)
        queue = self._get_or_create_queue(channel)
        await queue.enqueue(item)
        # Remember a queued tool result until it is processed. No await may
        # come before this point: deliver_tool_error relies on the record
        # existing by the time a failed on_notify broadcast returns.
        if tool_call_id:
            self._queued_tool_results.add((channel.id, tool_call_id))

    async def deliver_tool_error(
        self, channel, tool_call_id: str, text: str, meta: dict | None
    ) -> bool:
        """Queue an error tool result when on_notify failed to deliver one (#173).

        Enqueues straight onto the channel's serial queue, not through the
        on_notify broadcast that just failed. Does nothing when a result for
        ``tool_call_id`` is already queued (another plugin's hookimpl raised
        after this Agent's succeeded) or the id is no longer pending, so the
        model never sees two results for one call. Returns True if it queued.
        """
        if (channel.id, tool_call_id) in self._queued_tool_results:
            return False
        if tool_call_id not in channel.pending_tool_call_ids:
            return False
        item = QueueItem(
            role=QueueItemRole.NOTIFICATION,
            content=text,
            channel=channel,
            source="task",
            tool_call_id=tool_call_id,
            meta=meta or {},
        )
        queue = self._get_or_create_queue(channel)
        await queue.enqueue(item)
        self._queued_tool_results.add((channel.id, tool_call_id))
        # The error result is inbound work like any notification.
        self._note_activity(channel.id)
        return True

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

    def _warn_over_budget(
        self, channel: Channel, conv: ContextWindow, max_ctx: int,
        dropped: list[dict], truncated: bool,
    ) -> None:
        """Log the one WARNING for a fit_to_budget application (#308)."""
        # Message_log id range of the dropped rows, when they carry ids.
        ids = [m["_db_id"] for m in dropped if isinstance(m.get("_db_id"), int)]
        if not dropped:
            what = "dropped no rows"
        elif ids:
            what = (
                f"dropped {len(dropped)} oldest rows from memory (message_log "
                f"ids {min(ids)}-{max(ids)}; the log keeps them)"
            )
        else:
            what = (
                f"dropped {len(dropped)} oldest rows from memory (no "
                "message_log ids (rows not saved))"
            )
        if truncated:
            newest_id = conv.messages[-1].get("_db_id")
            what += (
                "; truncated the newest row (message_log id "
                f"{newest_id if isinstance(newest_id, int) else 'unsaved'}) to fit"
            )
        # Still over budget: say so with the numbers, nothing more to remove.
        remedy = "Raise max_context_tokens (agent-wide or this channel's override)"
        if conv.token_estimate() > max_ctx:
            sys_tokens = count_tokens(conv.system_prompt)
            if sys_tokens > max_ctx:
                what += (
                    f"; still over budget: the system prompt alone ({sys_tokens} "
                    f"tokens) exceeds the {max_ctx} budget; sent as is"
                )
                remedy += ", shorten the system prompt"
            else:
                newest_tokens = message_tokens(conv.messages[-1]) if conv.messages else 0
                what += (
                    f"; still over budget: system prompt {sys_tokens} tokens, "
                    f"newest row {newest_tokens} tokens, budget {max_ctx}; sent as is"
                )
        logger.warning(
            "context window for %s was over the %d token budget after "
            "compaction: %s. %s, or check why compaction did not bring the "
            "window down (see earlier log lines).",
            channel.id, max_ctx, what, remedy,
            extra={"channel": channel.id},
        )


    async def _run_turn(
        self,
        channel: Channel,
        messages: list[dict],
        tool_schemas: list[dict],
        llm_overrides: dict | None,
        item: QueueItem | None = None,
    ) -> AgentTurnResult | None:
        """Call run_agent_turn and handle LLM errors.

        Returns the AgentTurnResult on success, or None on error (error message
        already sent to the channel via the send hook, unless a
        should_send_response gate vetoed the "error" emission). ``item``
        supplies the trace id and meta for that gate.
        """
        try:
            result = await run_agent_turn(self._client, messages, tool_schemas, extra_body=llm_overrides)
        except Exception as exc:
            unreachable = is_backend_unreachable(exc)
            if item is not None and item.retry:
                # A failed retry sends nothing (the fallback went out on the
                # first failure).
                if unreachable:
                    # Still unreachable: back off further; the schedule logs
                    # the one WARNING.
                    self._schedule_turn_retry(channel, item, exc)
                else:
                    # Some other failure ends the retries.
                    logger.warning(
                        "retry of turn on channel %s (trace_id %s) failed with %s; "
                        "not retrying — the user can resend the message",
                        channel.id, item.trace_id, type(exc).__name__,
                        exc_info=True,
                        extra={"channel": channel.id, "trace_id": item.trace_id},
                    )
                    self._turn_retries.pop(channel.id, None)
                return None
            logger.exception("run_agent_turn failed for channel %s", channel.id)
            # Schedule the retry before the error gate: a vetoed fallback
            # returns early below and must not cancel the retry.
            if unreachable and item is not None:
                self._schedule_turn_retry(channel, item, exc)
            # The error hook is optional policy: if it raises, fall back to the
            # default message rather than leaving the user with silence.
            try:
                error_msg = await self.pm.ahook.resolve_llm_error(channel=channel, error=exc)
            except Exception:
                logger.warning(
                    "resolve_llm_error hook failed",
                    exc_info=True,
                    extra={"channel": channel.id},
                )
                error_msg = None
            if error_msg is None:
                # Build a complete message with the human-readable failure class.
                error_msg = _llm_error_message(exc)
            # Per-emission gate: a veto suppresses this unpersisted send and
            # nothing else — no assistant message exists on this path.
            if not await self._gate_send(channel, error_msg, "error", item):
                logger.info(
                    "error fallback not sent: vetoed by should_send_response",
                    extra={"channel": channel.id},
                )
                return None
            try:
                await self.pm.ahook.send(
                    channel=channel,
                    event=OutboundMessage(text=error_msg),
                )
            except Exception:
                logger.warning(
                    "send hook failed on error path",
                    exc_info=True,
                    extra={"channel": channel.id},
                )
            return None
        # Any successful turn makes a pending retry of this channel redundant.
        self._cancel_turn_retry(channel.id)
        return result

    def _cancel_turn_retry(self, channel_id: str) -> None:
        """Drop the channel's pending retry and cancel its timer (no-op if fired)."""
        state = self._turn_retries.pop(channel_id, None)
        if state is not None:
            state.timer.cancel()

    def _schedule_turn_retry(
        self, channel: Channel, item: QueueItem, exc: Exception
    ) -> None:
        """Arm a timer that re-enqueues ``item`` as a retry after a backoff delay."""
        # Resolved lazily: test helpers mark the Agent started without _start_plugin.
        # shortcut: hand-built plugin managers omit the clock; make it strict
        # (depends_on) if a non-Runtime entry point appears
        clock = self.pm.get_plugin("clock")
        if not isinstance(clock, Clock):
            logger.warning(
                "turn on channel %s (trace_id %s) will not be retried: "
                "no kernel clock registered (name 'clock'); Runtime registers one",
                channel.id, item.trace_id,
                extra={"channel": channel.id, "trace_id": item.trace_id},
            )
            return
        state = self._turn_retries.get(channel.id)
        if state is not None and state.item is item:
            # Another failure of the same retry: back off further.
            attempt = state.attempt + 1
            retry_item = state.item
        else:
            # A newer failed turn supersedes any older pending retry; its
            # window already holds the older message.
            if state is not None:
                state.timer.cancel()
            attempt = 0
            retry_item = replace(item, originating=False, retry=True)
        delay = min(TURN_RETRY_BASE_DELAY * 2**attempt, TURN_RETRY_MAX_DELAY)

        async def fire() -> None:
            # Never enqueue after on_stop: the queues are gone and so is the client.
            if not self._started:
                return
            # Enqueue (not call) to keep the channel's serial ordering.
            await self._get_or_create_queue(channel).enqueue(retry_item)
            self._note_activity(channel.id)

        timer = clock.call_later(delay, fire, name=f"turn-retry:{channel.id}")
        self._turn_retries[channel.id] = _TurnRetry(retry_item, attempt, timer)
        logger.warning(
            "turn on channel %s (trace_id %s) could not reach the LLM backend (%s); "
            "retrying in %g s — check that the server at llm.main.base_url is "
            "running and has its model loaded",
            channel.id, item.trace_id, type(exc).__name__, delay,
            extra={
                "channel": channel.id, "trace_id": item.trace_id,
                "attempt": attempt, "delay": delay,
            },
        )

    async def _gate_send(
        self,
        channel: Channel,
        text: str,
        emission: str,
        item: QueueItem | None,
    ) -> bool:
        """Fire should_send_response; return True when ``text`` may be sent.

        Reject-wins over every implementation. A raising hook fails open:
        an outbound gate that breaks must never silence the agent, so the
        failure is logged and the emission goes out.
        """
        trace_id = item.trace_id if item is not None else None
        meta = item.meta if item is not None else {}
        try:
            results = await self.pm.ahook.should_send_response(
                channel=channel, text=text, emission=emission,
                trace_id=trace_id, meta=meta,
            )
            verdict = resolve_reject_wins(results)
        except Exception:
            logger.warning(
                "should_send_response hook failed for the %s emission on %s; "
                "sending it anyway",
                emission, channel.id,
                exc_info=True,
                extra={"channel": channel.id, "emission": emission, "trace_id": trace_id},
            )
            return True
        return verdict is not False

    async def _append_unsent_marker(
        self, channel: Channel, emission: str, text: str
    ) -> None:
        """Append and persist the not-sent marker for a vetoed emission.

        A CONTEXT system row, persisted like any other, so it reloads with
        the window and the model never cites the unsent text as said. Core
        appends it directly rather than through a plugin, so every gate's
        veto gets one.
        """
        conv = channel.conversation
        if conv is None:
            return
        marker = {"role": "system", "content": unsent_marker_text(emission, text)}
        conv.append(marker, MessageType.CONTEXT)
        try:
            results = await self.pm.ahook.persist_message(
                channel=channel, message=marker, message_type=MessageType.CONTEXT,
            )
            rowid = resolve_single_result(results, "persist_message")
            if rowid is not None:
                conv.messages[-1]["_db_id"] = rowid
        except Exception:
            logger.warning(
                "persist_message hook failed for the not-sent marker on %s",
                channel.id, exc_info=True, extra={"channel": channel.id},
            )

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
        *,
        display_text: str | None = None,
        unsent: bool = False,
    ) -> None:
        """Dispatch tool calls or send text response.

        Implements the decision point at step 10 of _process_queue_item:
        - Tool calls under limit: increment counter, gate and send progress
          text, dispatch, return.
        - Tool calls at limit: NO counter increment, resolve display text
          with MAX_TURNS_FALLBACK_MESSAGE, fire on_agent_response, send message.
        - No tool calls: increment counter, resolve display text, fire
          on_agent_response, send message.

        ``item`` provides the current turn's trace id/meta (for the
        enriched on_agent_response call and the send gate) and, via
        _dispatch_tool_calls, stamps the tool cycle's Tasks with the same
        trace. ``display_text`` is the text the step-7 final gate already
        resolved and checked (None resolves it here); ``unsent`` is that
        gate's veto, which skips the transport send but still fires
        on_agent_response.
        """
        if result.tool_calls and channel.turn_counter < max_turns_limit:
            channel.turn_counter += 1
            # Show intermediate text (e.g. "Let me look that up...") before
            # dispatching tools, unless the progress gate vetoes it; a veto
            # leaves the row alone and marks the window instead.
            if result.text:
                if await self._gate_send(channel, result.text, "progress", item):
                    try:
                        await self.pm.ahook.send(
                            channel=channel, event=OutboundProgress(text=result.text)
                        )
                    except Exception:
                        logger.warning("send hook failed for progress", exc_info=True, extra={"channel": channel.id})
                else:
                    await self._append_unsent_marker(channel, "progress", result.text)
            await self._dispatch_tool_calls(result.tool_calls, channel)
            # Do NOT send text response; return and wait for tool results
            return
        elif result.tool_calls:
            # Max turns reached — suppress tool dispatch, send fallback text
            logger.warning(
                "max_turns reached, suppressing tool calls",
                extra={"channel": channel.id, "turn_counter": channel.turn_counter},
            )
            display_response = display_text if display_text is not None else (
                await self._resolve_display_text(
                    channel, result, fallback=MAX_TURNS_FALLBACK_MESSAGE
                )
            )
        else:
            # No tool calls — normal text response
            channel.turn_counter += 1
            display_response = display_text if display_text is not None else (
                await self._resolve_display_text(channel, result, fallback=None)
            )

        logger.info(
            "agent response not sent: vetoed by should_send_response"
            if unsent else "agent response sent",
            extra={
                "channel": channel.id,
                "latency_ms": result.latency_ms,
                "tool_calls_count": len(result.tool_calls) if result.tool_calls else 0,
                "turn_counter": channel.turn_counter,
                "pending_tools": len(channel.pending_tool_call_ids),
            },
        )

        trace_id = item.trace_id if item is not None else None
        meta = item.meta if item is not None else {}
        try:
            await self.pm.ahook.on_agent_response(
                channel=channel,
                request_text=request_text,
                response_text=display_response,
                trace_id=trace_id,
                meta=meta,
                logprobs=result.logprobs,
            )
        except Exception:
            logger.warning(
                "on_agent_response hook failed",
                exc_info=True,
                extra={"channel": channel.id},
            )
        # A vetoed final never reaches a transport; the row is UNSENT and the
        # window already carries its marker.
        if unsent:
            return
        try:
            await self.pm.ahook.send(
                channel=channel,
                event=OutboundMessage(text=display_response, latency_ms=result.latency_ms),
            )
        except Exception:
            logger.error(
                "send hook failed, response not delivered",
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
        channel = item.channel
        # The whole body is guarded so a failed or early-returning item still
        # reports the channel as settled.
        try:
            logger.debug(
                "processing queue item",
                extra={
                    "channel": item.channel.id,
                    "role": item.role,
                    "source": item.source,
                    "has_tool_call_id": item.tool_call_id is not None,
                },
            )
            # Trace resolution MUST happen here, before the
            # set_attribution call below, so the widened attribution carries
            # the id into every LLM call in the turn (including tool Tasks
            # created during dispatch, which snapshot the context at creation).
            # If resolution ran inside _process_queue_item_attributed instead
            # (after attribution is set), usage_log.trace_id would be null
            # for the whole turn.
            if item.trace_id is None:
                inherited = item.meta.get("trace_id") if item.meta else None
                if inherited is not None:
                    item.trace_id = inherited
                    item.originating = False
                else:
                    item.trace_id = mint_trace_id()
                    item.originating = True

            # Attribute all LLM activity in this turn (including tool Tasks
            # created during dispatch, which capture the context at creation)
            # to the turn stage and this channel. Compaction re-labels itself
            # around its own LLM call and restores this attribution after.
            attribution_token = set_attribution(
                stage="turn", channel_id=channel.id,
                trace_id=item.trace_id, meta=item.meta,
            )
            try:
                await self._process_queue_item_attributed(item, channel)
            finally:
                reset_attribution(attribution_token)
        finally:
            self._note_settled(channel)

    async def _process_queue_item_attributed(
        self, item: QueueItem, channel: Channel
    ) -> None:
        """Body of _process_queue_item, run with turn attribution set."""
        # A retry only runs while it is still the channel's pending one; a
        # newer successful turn may have dropped it after its timer fired.
        if item.retry:
            state = self._turn_retries.get(channel.id)
            if state is None or state.item is not item:
                logger.debug(
                    "retry superseded on channel %s (trace_id %s)",
                    channel.id, item.trace_id,
                    extra={"channel": channel.id, "trace_id": item.trace_id},
                )
                return
        import time as _time
        _t0 = _time.monotonic()
        _phases: dict[str, float] = {}

        # Build the conversation message based on item role
        msg_result = self._build_conversation_message(item)
        if msg_result is None:
            return
        conversation_message, request_text = msg_result

        # Pick up registry changes (e.g. a reconnected MCP server) before
        # any conversation is initialised or the turn's tools are read.
        self._refresh_tools_if_changed()

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
            # The tool set is fixed for the conversation, like the system
            # prompt it sits beside at the head of the prompt (principle 2).
            conv.tool_view = self._build_tool_view(channel)
            history = await self.pm.ahook.resolve_conversation(channel=channel)
            if history:
                conv.messages = history
            channel.conversation = conv
        conv = channel.conversation
        _phases["conv_init"] = _time.monotonic() - _t_conv_init_start

        # 2. Reset turn_counter when this item originates a new trace
        # (USER items, and notifications whose id was minted at dequeue in
        # _process_queue_item) — but only when it starts a cycle. An item
        # arriving while a tool cycle is in flight is charged against that
        # cycle's budget instead (R4), so a chatty channel can't extend
        # max_turns. Keying on `originating` rather than USER role lets a
        # notification-only channel (scheduler reminders, heartbeat) get a
        # fresh per-exchange budget instead of exhausting it permanently.
        if not item.retry and item.originating and not channel.pending_tool_call_ids:
            channel.turn_counter = 0

        # 3. Resolve config and read max_turns_limit
        resolved = self._registry.resolve_config(channel)
        max_turns_limit = resolved["max_turns"]

        # Steps 4-6 already ran on the first attempt of a retried turn (its
        # output is in the window), so a retry goes straight to the LLM call.
        if not item.retry:
            # 4. Append message to conversation log and fire persistence event.
            # The resolved rowid attaches to the WINDOW copy (conv.append
            # shallow-copies, so the local dict would not update the window).
            _t_persist_start = _time.monotonic()
            conv.append(conversation_message)
            rowid = None
            try:
                results = await self.pm.ahook.persist_message(
                    channel=channel,
                    message=conversation_message,
                    message_type=MessageType.MESSAGE,
                )
                rowid = resolve_single_result(results, "persist_message")
                if rowid is not None:
                    conv.messages[-1]["_db_id"] = rowid
            except Exception:
                logger.warning("persist_message hook failed", exc_info=True)
            _phases["persist"] = _time.monotonic() - _t_persist_start

            # 4c. Fire on_message_persisted only when this item originates its
            # trace (USER items, and notification items whose id was
            # minted at dequeue). Mid-exchange tool-result rows, injected
            # CONTEXT, and assistant rows never fire it — per-row firing under
            # one id would overwrite the rowid with each successive row. Fires
            # regardless of persistence outcome (rowid may be None).
            if item.originating and item.trace_id is not None:
                try:
                    await self.pm.ahook.on_message_persisted(
                        channel=channel, trace_id=item.trace_id,
                        rowid=rowid, text=request_text, meta=item.meta,
                    )
                except Exception:
                    logger.warning("on_message_persisted hook failed", exc_info=True, extra={"channel": channel.id})

            # 4b. Batch tool results: if this is a tool-result notification and
            # there are still pending tool calls, skip the LLM call.
            # pending_tool_call_ids is unioned across every in-flight dispatch
            # batch (R3), not scoped to one batch — so the LLM call fires after
            # the last result of the last outstanding batch, not the current
            # one. The serial queue ensures results arrive one at a time. User
            # messages can interleave — they don't clear the pending set, so the
            # agent remains responsive while tools run.
            if item.role == QueueItemRole.NOTIFICATION and item.tool_call_id:
                channel.pending_tool_call_ids.discard(item.tool_call_id)
                # Same step as the discard: a result is either queued or no
                # longer pending, never neither (see deliver_tool_error).
                self._queued_tool_results.discard((channel.id, item.tool_call_id))
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
                await self.pm.ahook.resolve_compaction(
                    channel=channel, conversation=conv,
                    max_tokens=resolved["max_context_tokens"],
                )
            except Exception:
                logger.warning("compaction failed, skipping", exc_info=True)
            _phases["compact"] = _time.monotonic() - _t_compact_start

        # 5b. Hard ceiling (#308, PRD §3.3.2): whatever compaction did or
        # skipped, a window over max_context_tokens loses its oldest rows from
        # memory (message_log keeps them). Pure list arithmetic, so not wrapped.
        # Outside the retry guard: a retried turn (#281) is a main-turn call
        # too, and tool results may have joined its window since.
        max_ctx = resolved["max_context_tokens"]
        dropped, truncated = conv.fit_to_budget(max_ctx)
        still_over = conv.token_estimate() > max_ctx
        if dropped or truncated or still_over:
            self._warn_over_budget(channel, conv, max_ctx, dropped, truncated)

        if not item.retry:
            # 6. Let plugins inject context before the LLM call
            _t_before_turn_start = _time.monotonic()
            msg_count_before = len(conv.messages)
            try:
                await self.pm.ahook.before_agent_turn(
                    channel=channel, trace_id=item.trace_id, meta=item.meta,
                )
            except Exception:
                logger.warning(
                    "before_agent_turn hook failed, skipping", exc_info=True
                )
            # Fire persist_message for any messages injected by
            # before_agent_turn. Messages already carrying a _db_id were
            # persisted by their producer — firing again would double-store
            # them.
            for msg in conv.messages[msg_count_before:]:
                if isinstance(msg.get("_db_id"), int):
                    continue
                mt = msg.get("_message_type", MessageType.MESSAGE)
                clean = {k: v for k, v in msg.items() if not k.startswith("_")}
                try:
                    results = await self.pm.ahook.persist_message(
                        channel=channel, message=clean, message_type=mt,
                    )
                    rowid = resolve_single_result(results, "persist_message")
                    if rowid is not None:
                        msg["_db_id"] = rowid
                except Exception:
                    logger.warning("persist_message hook failed", exc_info=True)
            _phases["before_turn"] = _time.monotonic() - _t_before_turn_start

        # 7. Build prompt and call run_agent_turn (single LLM invocation)
        _t_llm_start = _time.monotonic()
        messages = conv.build_prompt(
            frozenset(channel.pending_tool_call_ids),
            mid_conversation_system_role=self._mid_conversation_system_role,
        )
        llm_overrides = self._llm_overrides(channel)
        if self._request_logprobs:
            llm_overrides["logprobs"] = True

        result = await self._run_turn(
            channel, messages, self._tool_view(channel).schemas, llm_overrides or None,
            item,
        )
        if result is None:
            return

        # 7b. The persistence-controlling send gate (trap #6). It fires only
        # when this result's tool calls will NOT dispatch — a text result,
        # or the max-turns branch — because hiding an assistant message
        # whose calls dispatch would orphan its tool rows on reload. The
        # dispatch condition is computed once here and _handle_response
        # takes the same branch (turn_counter does not move in between).
        # The gate checks exactly what would be sent: the display text, or
        # the max-turns fallback on that branch.
        will_dispatch = bool(result.tool_calls) and channel.turn_counter < max_turns_limit
        assistant_message = result.message
        display_text: str | None = None
        unsent = False
        if not will_dispatch:
            if result.tool_calls:
                # Max-turns: these calls never dispatch, so a row that kept
                # them would dangle on reload. Strip them from the row AND
                # the window copy (one dict feeds both), whatever the verdict.
                assistant_message = {
                    k: v for k, v in result.message.items() if k != "tool_calls"
                }
                if assistant_message.get("content") is None:
                    assistant_message["content"] = ""
            display_text = await self._resolve_display_text(
                channel, result,
                fallback=MAX_TURNS_FALLBACK_MESSAGE if result.tool_calls else None,
            )
            unsent = not await self._gate_send(channel, display_text, "final", item)

        # 8. Append assistant message and fire persistence event, attaching
        # the resolved rowid to the window copy (same rule as step 4). A
        # vetoed final persists as UNSENT: in the window, never sent.
        message_type = MessageType.UNSENT if unsent else MessageType.MESSAGE
        conv.append(assistant_message, message_type)
        assistant_in_window = conv.messages[-1]
        try:
            results = await self.pm.ahook.persist_message(
                channel=channel,
                message=assistant_message,
                message_type=message_type,
            )
            rowid = resolve_single_result(results, "persist_message")
            if rowid is not None:
                assistant_in_window["_db_id"] = rowid
        except Exception:
            logger.warning("persist_message hook failed", exc_info=True)

        # 8b. Send an OutboundThinking event if the LLM produced
        # reasoning_content, unless the thinking gate vetoes it. The marker
        # waits until after step 9, which works on the assistant message.
        unsent_markers: list[tuple[str, str]] = []
        if unsent:
            unsent_markers.append(("final", display_text or ""))
        reasoning_text = result.message.get("reasoning_content")
        # A vetoed final withholds its reasoning with it: the reasoning
        # often carries the answer, and the final marker already covers
        # the whole response, so the thinking gate is not consulted.
        if reasoning_text and not unsent:
            if await self._gate_send(channel, reasoning_text, "thinking", item):
                try:
                    await self.pm.ahook.send(
                        channel=channel, event=OutboundThinking(text=reasoning_text)
                    )
                except Exception:
                    logger.warning("send hook failed for thinking", exc_info=True, extra={"channel": channel.id})
            else:
                unsent_markers.append(("thinking", reasoning_text))

        # 9. Let plugins post-process the in-memory assistant message (e.g., strip reasoning_content)
        try:
            await self.pm.ahook.after_persist_assistant(channel=channel, message=assistant_in_window)
        except Exception:
            logger.warning(
                "after_persist_assistant hook failed",
                exc_info=True,
                extra={"channel": channel.id},
            )

        # 9b. Mark each vetoed emission in the window (persisted, so the
        # marker reloads beside the row it describes).
        for emission, text in unsent_markers:
            await self._append_unsent_marker(channel, emission, text)
        _phases["llm"] = _time.monotonic() - _t_llm_start

        # 10. Dispatch tool calls or send text response
        await self._handle_response(
            result, channel, max_turns_limit, request_text, item,
            display_text=display_text, unsent=unsent,
        )

        # 11. Log phase timing breakdown
        _total = _time.monotonic() - _t0
        _phases["other"] = max(0, _total - sum(_phases.values()))
        logger.info(
            "queue item timing",
            extra={
                "channel": item.channel.id,
                "role": item.role.value,
                "total_ms": round(_total * 1000),
                **{f"{k}_ms": round(v * 1000) for k, v in _phases.items()},
                "prompt_tokens": result.usage.get("prompt_tokens") if result.usage else None,
                "message_count": len(conv.messages) if conv else 0,
            },
        )

    @staticmethod
    def _llm_overrides(channel: Channel) -> dict:
        """The channel's runtime overrides that go into the request body.

        Only INFERENCE_KEYS are forwarded (GitHub issue #206). set_settings
        already refuses other undotted keys; this filter keeps framework
        keys, dotted plugin tunables (see corvidae.tuning), transport
        metadata, and anything else that reaches runtime_overrides out of
        the body. Shared by the main turn and forks, which must send the
        same body to keep the cached prefix.
        """
        return {
            k: v
            for k, v in channel.runtime_overrides.items()
            if k in INFERENCE_KEYS
        }

    def fork_enabled(self, stage: str) -> bool:
        """Whether ``stage`` runs as a fork (agent.forks.enabled and stages)."""
        return self._forks.stage_enabled(stage)

    async def fork(
        self,
        channel: Channel,
        instruction: str,
        *,
        cut: int | None = None,
        extra_body: dict | None = None,
    ) -> ForkResult:
        """Run one fork of the channel's window (GitHub issue #51).

        The prompt is ``build_prompt(window[:cut])`` plus ``instruction`` as
        a user turn, built with exactly what step 7 of the agent loop uses
        (main client, tool schemas, pending ids, mid-conversation role,
        override filter, reasoning effort untouched), so the server reuses
        the cached prefix. ``cut`` None forks from the whole window.

        On top of the main turn's body a fork sends ``tool_choice: "none"``,
        the configured ``reasoning_budget_tokens``, and ``extra_body`` (a
        stage's ``response_format``). Those keys are applied after the
        runtime overrides so ``set_settings`` can't re-enable tool calls.

        Mutates nothing, persists nothing, fires no hooks, dispatches no
        tools. Attribution gains ``mode="fork"`` for the call; the caller
        sets ``stage``.

        Raises:
            ForkUnavailable: no live window or no main client.
            ForkRefused: the model answered with tool calls, structured or
                written out as text.
        """
        conv = channel.conversation
        if conv is None:
            raise ForkUnavailable(f"no live window on {channel.id}")
        if self._client is None:
            raise ForkUnavailable("LLM client not initialized")

        messages = build_fork_prompt(
            conv,
            cut,
            instruction,
            frozenset(channel.pending_tool_call_ids),
            self._mid_conversation_system_role,
        )
        # Fork-specific keys last: a stage's body, then the invariants.
        body = {**self._llm_overrides(channel), **(extra_body or {}), "tool_choice": "none"}
        if self._forks.reasoning_budget_tokens is not None:
            body["reasoning_budget_tokens"] = self._forks.reasoning_budget_tokens

        # Bound once: a config reload or plugin stop can replace or clear
        # self._client during the await, and the stamp must name the model
        # this request was actually sent with.
        client = self._client
        token = set_attribution(channel_id=channel.id, mode="fork")
        try:
            # The channel's own schemas: a fork must share the turn's prefix.
            response = await client.chat(
                messages, tools=self._tool_view(channel).schemas or None, extra_body=body
            )
        finally:
            reset_attribution(token)

        message = response["choices"][0]["message"]
        if message.get("tool_calls"):
            raise ForkRefused(
                f"fork on {channel.id} answered with "
                f"{len(message['tool_calls'])} tool call(s)"
            )
        if looks_like_tool_call(message.get("content") or ""):
            raise ForkRefused(f"fork on {channel.id} wrote a tool call as text")
        return ForkResult(
            text=message.get("content") or "",
            message=message,
            usage=response.get("usage"),
            # What the request was sent with, so records can be stamped.
            model=client.request_model(body),
        )

    def _warn_if_forks_change_model(self) -> None:
        """Warn when forks are on and llm.background is a different model.

        Forks always run on llm.main (only it shares the window's cache),
        so enabling them moves the forked background stages off a
        separately configured background model.
        """
        if not self._forks.enabled:
            return
        llm = self.pm.get_plugin("llm")
        background = getattr(llm, "_clients", {}).get("background") if llm is not None else None
        main_model = getattr(self._client, "model", None)
        background_model = getattr(background, "model", None)
        if background is None or background_model == main_model:
            return
        logger.warning(
            "agent.forks.enabled: forked stages run on llm.main (%s), not "
            "llm.background (%s); their cost moves to the main model",
            main_model, background_model,
        )

    def _build_tool_view(self, channel: Channel) -> ToolView:
        """The daemon's tools narrowed to ``channel``'s disabled list.

        A per-channel name that matches no loaded tool is logged, since a
        typo there silently disables nothing. An MCP tool whose server has
        not connected yet reads the same way; the warning says so.
        """
        disabled = channel.disabled_tools
        view = build_tool_view(
            self._tools, self._tool_schemas, disabled, self._tools_generation
        )
        unknown = sorted(name for name in disabled if name not in self._tools)
        if unknown:
            # Name the config keys that carry the unknown names: a typo in a
            # transport-wide entry is fixed under its own key, not the channel's.
            keys = sorted({
                key for name in unknown
                for key in disabling_keys(name, channel, expand=False)
            })
            logger.warning(
                "%s names no loaded tool: %s (channel %s); check the spelling "
                "(loaded: %s), or ignore if an MCP server has not connected yet",
                ", ".join(keys), ", ".join(unknown), channel.id,
                ", ".join(sorted(self._tools)),
                extra={"channel": channel.id},
            )
        return view

    def _tool_view(self, channel: Channel) -> ToolView:
        """The conversation's tool view, resolving it on first use for a
        window built outside conversation init (a test, an import)."""
        conv = channel.conversation
        if conv is None:
            return self._build_tool_view(channel)
        if conv.tool_view is None:
            conv.tool_view = self._build_tool_view(channel)
        elif conv.tool_view.generation != self._tools_generation:
            # The registry moved since this view was built: re-narrow the
            # new tool set by the view's own disabled set, not the live
            # config, so a reloaded channels.<id>.tools.disabled still
            # waits for the compaction boundary (#149, #109).
            conv.tool_view = build_tool_view(
                self._tools, self._tool_schemas,
                conv.tool_view.disabled, self._tools_generation,
            )
        return conv.tool_view

    def _snapshot_tools(self, tools_plugin: "ToolCollectionPlugin") -> None:
        """Read the tool dict, schemas and generation from the registry.

        Plain attribute reads with no await between them, so a background
        rebuild_registry cannot tear the three apart.
        """
        self._tools_plugin = tools_plugin
        self._tools, self._tool_schemas = tools_plugin.get_tools()
        self._tools_generation = tools_plugin.generation

    def _refresh_tools_if_changed(self) -> None:
        """Re-read the registry when its generation has moved (#109).

        A no-op while the generation is unchanged, so the schema lists stay
        the same objects turn after turn and the cached prompt prefix holds.
        Each conversation's view is rebuilt lazily by _tool_view.
        """
        tools_plugin = self._tools_plugin
        if tools_plugin is None or tools_plugin.generation == self._tools_generation:
            return
        previous = set(self._tools)
        self._snapshot_tools(tools_plugin)
        current = set(self._tools)
        # One INFO line per change: it alters the head of the prompt.
        logger.info(
            "tool registry changed (generation %d): added [%s], removed [%s]",
            self._tools_generation,
            ", ".join(sorted(current - previous)),
            ", ".join(sorted(previous - current)),
        )

    @hookimpl
    async def on_compaction(
        self,
        channel: Channel,
        summary_msg: dict,
        retain_count: int,
        compacted_ids: list[int],
    ) -> None:
        """Re-resolve the channel's tool set at the compaction boundary.

        Compaction already rebuilds the KV cache past the system prompt, so
        this is where a reloaded channels.<id>.tools.disabled reaches the
        schemas (issue #149); between boundaries they never move. A call
        dispatched before the boundary runs to completion.
        """
        conv = channel.conversation
        if conv is None:
            return
        view = self._build_tool_view(channel)
        previous = conv.tool_view
        if previous is not None and previous.disabled != view.disabled:
            logger.info(
                "tool set changed at compaction boundary",
                extra={"channel": channel.id, "disabled": sorted(view.disabled)},
            )
        conv.tool_view = view

    async def _dispatch_tool_calls(
        self, tool_calls: list[dict], channel: Channel
    ) -> None:
        """Enqueue each tool call as a Task on the TaskQueue.

        Each task's work closure calls dispatch_tool_call(), which handles
        JSON parsing, unknown-tool detection, invocation, error wrapping,
        logging, and the transform_tool_result hook. Returns the result
        string; the TaskPlugin delivers it via on_notify.

        Logs an error and delivers an error tool result directly (via
        on_notify) when the TaskQueue is unavailable (TaskPlugin not
        registered), so the model can finish the turn instead of waiting
        forever for a result that will never arrive.

        Stamps each Task with the current turn's trace id/meta, read
        from the attribution contextvar set by _process_queue_item at
        dequeue (this coroutine runs under that attribution). Every turn of
        a tool cycle inherits its trace this way, however many hops.
        """
        task_queue = getattr(self.pm.get_plugin("task"), "task_queue", None)

        current_attribution = get_attribution()
        trace_id = current_attribution.get("trace_id")
        meta = current_attribution.get("meta") or {}

        # Union in this batch's call IDs — a second dispatch while an earlier
        # batch is still outstanding must not clobber its ids (R3). The LLM
        # call fires once the pending set empties, i.e. after the last
        # result of the last outstanding batch.
        channel.pending_tool_call_ids |= {call["id"] for call in tool_calls}

        if task_queue is None:
            logger.error(
                "Tool calls %s requested on %s but no task queue is available "
                "— the `task` plugin is registered but has no task_queue: check "
                "that TaskPlugin.on_start ran before dispatch, and that the "
                "plugin registered as 'task' is TaskPlugin",
                [call["function"]["name"] for call in tool_calls],
                channel.id,
                extra={"channel": channel.id},
            )
            # No queue means no TaskPlugin to deliver results; inject an error
            # result per call so the tool cycle can complete.
            for call in tool_calls:
                call_id = call["id"]
                fn_name = call["function"]["name"]
                text = (
                    f"[Task unavailable] Error: tool '{fn_name}' was not "
                    "executed because the task queue is unavailable"
                )
                notify_meta = {"trace_id": trace_id, **meta}
                # One call's failed delivery must not stop the rest.
                try:
                    await self.pm.ahook.on_notify(
                        channel=channel,
                        source="task",
                        text=text,
                        tool_call_id=call_id,
                        meta=notify_meta,
                    )
                except Exception:
                    logger.warning(
                        "on_notify failed delivering no-task-queue error for "
                        "tool %s call %s on %s; queueing it directly",
                        fn_name, call_id, channel.id,
                        exc_info=True,
                        extra={"channel": channel.id, "tool_call_id": call_id, "tool": fn_name},
                    )
                    # Safe after a partial broadcast: deliver_tool_error
                    # skips ids already queued or no longer pending, so the
                    # model never sees two results for one call.
                    await self.deliver_tool_error(channel, call_id, text, notify_meta)
            return

        for call in tool_calls:
            call_id = call["id"]
            fn_name = call["function"]["name"]

            async def make_work(call=call):
                """Execute a single tool call via dispatch_tool_call, unless
                the tool is disabled on this channel (issue #149)."""
                # Checked when the task runs, against the conversation's
                # view and the live config: hiding a tool is not access
                # control, since the model can name one it saw earlier or
                # one injected text names.
                name = call["function"]["name"]
                if name in disabled_on(channel):
                    logger.warning(
                        "refused tool %s on %s: disabled by %s",
                        name, channel.id, ", ".join(disabling_keys(name, channel)),
                        extra={"channel": channel.id, "tool": name, "tool_call_id": call["id"]},
                    )
                    return refusal_message(name, channel)
                result = await dispatch_tool_call(
                    call, self._tools,
                    channel=channel,
                    task_queue=task_queue,
                    max_result_chars=self._max_tool_result_chars,
                    pm=self.pm,
                    call_timeouts=self._tool_call_timeouts,
                )
                return result.content

            # Notify transports that a tool call has been dispatched
            try:
                await self.pm.ahook.send(
                    channel=channel,
                    event=OutboundToolStatus(
                        tool_name=fn_name,
                        status="dispatched",
                        args_summary=call["function"].get("arguments", "")[:200],
                        tool_call_id=call_id,
                    ),
                )
            except Exception:
                logger.warning("send hook failed for tool status", exc_info=True, extra={"channel": channel.id})

            task = Task(
                work=make_work,
                channel=channel,
                tool_call_id=call_id,
                description=f"tool:{fn_name}",
                trace_id=trace_id,
                meta=meta,
            )
            await task_queue.enqueue(task)

    async def on_stop(self) -> None:
        # Stop advertising readiness first, so producers stop injecting
        # notifications into queues that are about to be cancelled.
        self._started = False
        # Drop pending turn retries: a timer firing after this would run a
        # turn with no client (#281).
        for state in self._turn_retries.values():
            state.timer.cancel()
        self._turn_retries.clear()
        # Release any message still held waiting for a start that won't come.
        self.release_held_messages()
        # Cancel all channel queue consumers.
        for queue in self.queues.values():
            await queue.stop()

        await self._stop_plugin()

    def release_held_messages(self) -> None:
        """Wake every on_message held for startup so it returns unprocessed.

        Called by on_stop and by Runtime when startup aborts (#71); each
        held caller logs one warning naming its channel and trace id.
        """
        self._stopping = True
        self._started_event.set()

    def _mark_started(self) -> None:
        """Record that the Agent can take work and wake held on_message calls."""
        self._started = True
        self._stopping = False
        self._started_event.set()

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
        self._snapshot_tools(tools_plugin)
        self._max_tool_result_chars = tools_plugin.max_result_chars
        self._tool_call_timeouts = tools_plugin.call_timeouts

        # The idle tracker is optional: a hand-built plugin manager may lack it.
        # shortcut: hand-built plugin managers omit the idle tracker; make it strict if a non-Runtime entry point appears
        self._idle = self.pm.get_plugin("idle")
        if self._idle is None:
            logger.warning(
                "no idle tracker registered (name 'idle'); idle signals are off — "
                "Runtime registers one, so this plugin manager was built by hand"
            )

        self._warn_if_forks_change_model()

        # Everything _process_queue_item needs is in place.
        self._mark_started()

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
        self._mid_conversation_system_role = _resolve_mid_conversation_system_role(agent_config)
        self._forks = parse_forks_config(agent_config)

        # Re-borrow the LLM client from LLMPlugin after a possible swap.
        # Avoids isinstance check so MagicMock(spec=LLMPlugin) works in tests.
        llm = self.pm.get_plugin("llm")
        if llm is not None and hasattr(llm, "get_client"):
            self._client = llm.get_client()

        logger.info("on_config_reload: agent config updated")
