"""CompactionPlugin — token-budget conversation compaction.

Extracts compaction logic from ContextWindow into a standalone plugin that
implements the resolve_compaction hookspec. Registers in main.py before
Agent so it runs as the default compaction strategy.

Algorithm:
    1. Skip if token_estimate < 80% of max_tokens.
    2. Skip if len(messages) <= 5.
    3. Backward walk retaining messages within 50% of max_tokens, then move
       the cut so no tool cycle straddles it: back to keep the cycle when
       the tail stays well under the trigger, else forward to drop it
       whole if enough of the tail survives, else keep the walk's cut
       (see _clean_cut). In-flight calls have no result yet and never
       straddle.
    4. Skip if all messages fit (retain_count >= len(messages)).
    5. Summarize. With forks on for "compaction" (agent.forks, #51), fork
       the window at the cut and append agent.compaction_fork_prompt: the
       server reuses the cached prefix and the agent summarizes in its own
       voice. Otherwise, or if the fork fails, the same view is sent as a
       plain call on the compaction stage's client with no tools (_view_summary). If that
       fails too, the over-budget fallback in the Agent applies (#308).
    6. Summary role is assistant, or user if the retained tail holds no
       user message (#56).
    7. Fire before_compaction (awaited) while the window is still whole.
    8. Call conversation.replace_with_summary(summary_msg, retain_count) (sync).
    9. Fire on_compaction hook if pm is set.
    10. Return True.
"""

import logging
import time
from typing import TYPE_CHECKING

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.context import DEFAULT_CHARS_PER_TOKEN, MessageType, count_tokens, message_tokens
from corvidae.prompts import resolve_prompt
from corvidae.fork import (
    build_fork_prompt,
    fork_agent,
    snap_cut_forward,
    snap_cut_to_tool_boundary,
    view_agent,
    view_call,
)
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.llm import is_backend_unreachable

if TYPE_CHECKING:
    # Only for annotations; compaction must not import these at runtime.
    from corvidae.channel import Channel
    from corvidae.context import ContextWindow

logger = logging.getLogger(__name__)


class BackendUnreachable(Exception):
    """The LLM backend could not be reached during compaction (#400).

    Distinct from a summary miss: a miss tries the background view, while an
    outage skips it and defers compaction to the turn's retry loop.
    """

    def __init__(self, channel_id: str) -> None:
        super().__init__(f"channel {channel_id}: LLM backend unreachable")
        self.channel_id = channel_id


class CompactionPlugin(CorvidaePlugin):
    """Plugin that implements default token-budget conversation compaction."""

    depends_on = frozenset({"llm"})

    # The instruction a compaction fork appends to the dropped prefix.
    # prompts/compaction_fork.md is the documented copy; the static config
    # key agent.compaction_fork_prompt overrides it. Written to the agent in
    # the second person: in a fork it is the agent summarizing its own
    # conversation, not a neutral summarizer reading a transcript.
    DEFAULT_FORK_PROMPT = (
        "[Compaction] Pause the conversation here. Everything above this "
        "message is about to leave your context window. Only your summary of "
        "it will remain, followed by the more recent conversation, so write "
        "the summary that lets you carry on naturally from it.\n\n"
        "Explain at a high level what you were working on or trying to "
        "achieve. If the conversation above opens with a summary of earlier "
        "conversation, carry its key information forward; do not discard it. "
        "Preserve specific details: names, file paths, variable names, error "
        "messages, decisions, commitments, what you were asked to do or not "
        "do, discoveries made, and the current line of investigation. Keep "
        "what you were told apart from what you inferred.\n\n"
        "In 1000 words or less. Reply with the summary only. Do not call tools."
    )

    # Reserve for the LLM's own summary response (prompt caps it at ~1000
    # words, ~1300-1500 tokens for English) plus JSON structure overhead.
    _SUMMARY_RESPONSE_RESERVE_TOKENS = 2000

    def __init__(self, pm=None) -> None:
        self.pm = pm  # May be None; on_init will set it via CorvidaePlugin.on_init
        self._compaction_threshold: float = 0.8
        self._compaction_retention: float = 0.5
        self._min_messages: int = 5
        self._chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
        self._fork_prompt: str = self.DEFAULT_FORK_PROMPT
        self._min_messages_between_compactions: int = 6
        self._last_compaction_msg_count: dict[str, int] = {}  # channel_id -> message_count at last compaction
        self._failed_compaction_cooldown: float = 30.0  # seconds
        self._last_failed_compaction: dict[str, float] = {}  # channel_id -> timestamp

    @hookimpl
    def register_llm_stages(self, stages: dict[str, str]) -> None:
        stages["compaction"] = "chat"

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        agent_config = config.get("agent", {})
        self._compaction_threshold = agent_config.get("compaction_threshold", 0.8)
        self._compaction_retention = agent_config.get("compaction_retention", 0.5)
        self._min_messages = agent_config.get("min_messages_to_compact", 5)
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        self._fork_prompt = resolve_prompt(config, "agent.compaction_fork_prompt", self.DEFAULT_FORK_PROMPT)

        # The standalone summary prompt is gone; say so once at startup (O2).
        if "compaction_prompt" in agent_config:
            logger.warning(
                "agent.compaction_prompt is no longer used and is ignored: "
                "compaction summaries are now written from the agent's own "
                "view with agent.compaction_fork_prompt. Move any custom "
                "instruction there (it is sent as a user instruction after "
                "the conversation, not as a system prompt) and remove "
                "agent.compaction_prompt from agent.yaml."
            )

        self._validate_config()

    def _validate_config(self) -> None:
        """Warn if compaction config values are likely ineffective.

        Two checks:
        1. If compaction_threshold <= compaction_retention, compaction is a
           no-op: the backward walk retains all messages within the retention
           budget, which equals or exceeds the trigger threshold.
        2. If the gap between threshold and retention is < 0.1, compaction
           will only summarize a tiny fraction of messages and may actually
           grow the context.
        """
        threshold = self._compaction_threshold
        retention = self._compaction_retention
        gap = threshold - retention

        if threshold <= retention:
            logger.warning(
                "compaction_threshold (%s) <= compaction_retention (%s): "
                "compaction will be ineffective (retains as many tokens as "
                "it triggers on). Increase compaction_threshold or decrease "
                "compaction_retention so that threshold > retention.",
                threshold, retention,
            )
        elif gap < 0.1:
            logger.warning(
                "compaction config has a very small gap between "
                "compaction_threshold (%s) and compaction_retention (%s) "
                "(gap=%.2f). Compaction will be fragile — it may summarize "
                "very few messages and could grow the context. "
                "A gap of at least 0.1 is recommended (e.g., threshold=0.8, "
                "retention=0.5).",
                threshold, retention, gap,
            )

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Re-read compaction thresholds and the fork prompt.

        Resolves into locals first so a bad prompt file leaves no partial
        state (§3.11.3: a malformed reload changes nothing).
        """
        agent_config = config.get("agent", {})
        threshold = agent_config.get("compaction_threshold", 0.8)
        retention = agent_config.get("compaction_retention", 0.5)
        min_msgs = agent_config.get("min_messages_to_compact", 5)
        cpt = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        prompt = resolve_prompt(config, "agent.compaction_fork_prompt", self.DEFAULT_FORK_PROMPT)

        # All resolved; commit atomically.
        self._compaction_threshold = threshold
        self._compaction_retention = retention
        self._min_messages = min_msgs
        self._chars_per_token = cpt
        self._fork_prompt = prompt

        self._validate_config()

        logger.info("on_config_reload: compaction thresholds updated")

    @hookimpl(trylast=True)
    async def resolve_compaction(self, channel, conversation, max_tokens):
        """Compact the conversation if it exceeds 80% of max_tokens.

        Returns True if compaction was performed, None otherwise.
        """
        channel_id = conversation.channel_id

        if conversation.token_estimate() < self._compaction_threshold * max_tokens:
            return None

        if len(conversation.messages) <= self._min_messages:
            return None

        # Cooldown after a failed compaction: don't retry immediately.
        # A channel with no recorded failure is never in cooldown. (Defaulting
        # to timestamp 0 put every channel in cooldown for the first 30 s
        # after boot, since time.monotonic() counts from boot.)
        last_fail = self._last_failed_compaction.get(channel_id)
        if last_fail is not None and time.monotonic() - last_fail < self._failed_compaction_cooldown:
            logger.debug(
                "compaction skipped (cooldown after recent failure)",
                extra={"channel_id": channel_id},
            )
            return None

        # Minimum messages between compactions: don't re-compact too aggressively.
        last_msg_count = self._last_compaction_msg_count.get(channel_id, 0)
        messages_since = len(conversation.messages) - last_msg_count
        if last_msg_count > 0 and messages_since < self._min_messages_between_compactions:
            logger.debug(
                "compaction skipped (too soon since last compaction)",
                extra={"channel_id": channel_id, "messages_since": messages_since},
            )
            return None

        logger.warning(
            "compaction triggered (approaching context limit)",
            extra={"channel_id": channel_id},
        )

        retain_budget = int(max_tokens * self._compaction_retention)
        retain_count = 0
        retain_tokens = 0
        for msg in reversed(conversation.messages):
            msg_tokens = message_tokens(msg)
            if retain_tokens + msg_tokens > retain_budget and retain_count > 0:
                break
            retain_tokens += msg_tokens
            retain_count += 1

        # Move the cut so no tool cycle straddles it: dropping a call while
        # keeping its results orphans them, and a fork of the prefix would
        # render the call differently from the main turn (#51).
        cut = self._clean_cut(
            conversation, len(conversation.messages) - retain_count, max_tokens
        )
        retain_count = len(conversation.messages) - cut

        # All-fit guard: no-op compaction would accomplish nothing.
        if retain_count >= len(conversation.messages):
            return None

        older = conversation.messages[:cut]

        # Collect the message_log rowids of everything about to be removed —
        # the on_compaction payload consolidation consumes (§3.1, §4.8).
        # Messages that were never persisted carry no _db_id and are skipped.
        compacted_ids = [
            m["_db_id"] for m in older if isinstance(m.get("_db_id"), int)
        ]

        # Attribute the summary LLM call to the compaction stage. Set/reset
        # around the call only, so it shadows the enclosing turn attribution
        # and restores it afterwards.
        attribution_token = set_attribution(stage="compaction", channel_id=channel_id)
        try:
            summary_text = await self._fork_summary(channel, cut)
            if summary_text is None:
                # Fork miss or forks off: the same view on the stage's client.
                summary_text = await self._view_summary(channel, conversation, cut)
        except BackendUnreachable:
            # The backend is down; the view would wait on the same dead
            # endpoint. Defer to the turn's retry loop (#400).
            self._last_failed_compaction[channel_id] = time.monotonic()
            return None
        except Exception:
            # Record failure time for cooldown
            self._last_failed_compaction[channel_id] = time.monotonic()
            raise
        finally:
            reset_attribution(attribution_token)

        # Reject blank/empty summaries — they're worse than no summary
        # because they erase context without preserving any information.
        if not summary_text or not summary_text.strip():
            logger.warning(
                "compaction produced empty summary, aborting",
                extra={"channel_id": channel_id},
            )
            self._last_failed_compaction[channel_id] = time.monotonic()
            return None

        # The summary is normally an assistant message. If the retained tail
        # holds no user message (a long tool loop), render it as user
        # instead: strict chat templates (Swift-1.5 / Qwen 3.8) reject a
        # prompt with no user query (GitHub issue #56). The role is stored
        # with the summary, so a reload renders the same prompt.
        retained = conversation.messages[-retain_count:]
        summary_role = (
            "assistant" if any(m.get("role") == "user" for m in retained) else "user"
        )
        summary_msg = {
            "role": summary_role,
            "content": f"[Summary of earlier conversation]\n{summary_text}",
        }

        # The window still holds everything about to be dropped, and the
        # server still holds its prefix: last chance for a fork over it.
        if self.pm is not None:
            try:
                await self.pm.ahook.before_compaction(
                    channel=channel, compacted_ids=compacted_ids, cut=cut,
                )
            except Exception:
                logger.warning("before_compaction hook failed", exc_info=True)

        conversation.replace_with_summary(summary_msg, retain_count)

        # Record successful compaction for cooldown tracking.
        self._last_compaction_msg_count[channel_id] = len(conversation.messages)
        # Clear any previous failure cooldown.
        self._last_failed_compaction.pop(channel_id, None)

        if self.pm is not None:
            try:
                await self.pm.ahook.on_compaction(
                    channel=channel,
                    summary_msg=summary_msg,
                    retain_count=retain_count,
                    compacted_ids=compacted_ids,
                )
            except Exception:
                logger.warning("on_compaction hook failed", exc_info=True)

        return True

    def _clean_cut(self, conversation, planned: int, max_tokens: int) -> int:
        """The cut to compact at, given the retain walk's ``planned`` cut.

        Prefers backing up to keep a straddling tool cycle whole, but only
        while the window after compaction stays well under the trigger
        (halfway from compaction_retention to compaction_threshold), counted
        the way the trigger counts it: system prompt, a summary allowance,
        and the kept messages. The back-up must also drop some dialog, not
        just the previous summary. A long-running call
        near the head with chat interleaved behind it would otherwise pull
        the cut back so far that compaction drops almost nothing and runs
        again every few messages. Then the cut moves forward past the
        cycle's last result instead, dropping it whole, but only while at
        least half the retention budget survives. Otherwise (the result is
        near the newest message) the planned cut stands, orphaned result
        and all, as before #51: never worse than the unsnapped walk.
        """
        messages = conversation.messages
        back = snap_cut_to_tool_boundary(messages, planned)
        if back == planned:
            return planned

        def tail_tokens(cut: int) -> int:
            return sum(message_tokens(m) for m in messages[cut:])

        ceiling = max_tokens * (self._compaction_retention + self._compaction_threshold) / 2
        fixed = count_tokens(conversation.system_prompt) + self._SUMMARY_RESPONSE_RESERVE_TOKENS
        drops_dialog = any(
            m.get("_message_type", MessageType.MESSAGE) == MessageType.MESSAGE
            for m in messages[:back]
        )
        if drops_dialog and fixed + tail_tokens(back) <= ceiling:
            return back
        forward = snap_cut_forward(messages, planned)
        if forward < len(messages) and tail_tokens(forward) >= max_tokens * self._compaction_retention / 2:
            return forward
        return planned

    async def _fork_summary(self, channel, cut: int) -> str | None:
        """The summary from a fork of the window at ``cut``, or None.

        None means use the background view (_view_summary): forks are off for
        compaction, or the fork failed or came back empty (logged). The
        fork's prefix already holds any prior summary at its head, so the
        instruction carries it forward without a bounded copy.

        Raises:
            BackendUnreachable: the fork failed on a connection-class error.
                The view call goes out on the same dead backend, so it is not
                attempted (#400, PRD §3.1's unreachable-backend rule).
        """
        agent = fork_agent(self.pm, "compaction")
        if agent is None:
            return None
        try:
            result = await agent.fork(channel, self._fork_prompt, cut=cut)
        except Exception as exc:
            if is_backend_unreachable(exc):
                # One wait already spent; the turn's own retry loop waits for
                # the backend, and compaction runs first once it is back.
                logger.warning(
                    "compaction deferred: channel %s could not reach the LLM "
                    "backend (%s); not trying the background view",
                    channel.id, type(exc).__name__,
                    extra={"channel_id": channel.id},
                )
                raise BackendUnreachable(channel.id) from exc
            logger.warning(
                "compaction fork failed; falling back to the background view",
                exc_info=True, extra={"channel_id": channel.id},
            )
            return None
        if not result.text.strip():
            logger.warning(
                "compaction fork returned no text; falling back to the background view",
                extra={"channel_id": channel.id},
            )
            return None
        return result.text

    async def _view_summary(self, channel: "Channel", conversation: "ContextWindow", cut: int) -> str:
        """The summary written from the live window at ``cut`` on llm.compaction.

        Renders the window prefix exactly as the fork does (compaction fork
        prompt, the channel's pending tool-call ids, the Agent's mid role)
        and sends it as a plain call with no tools. The render is the
        uncapped prefix (the window was over budget, which is why compaction
        runs), so this call carries the summary's cost on the compaction
        stage's profile.

        Raises:
            RuntimeError: no Agent can supply a view; the over-budget
                fallback then applies.
        """
        agent = view_agent(self.pm)
        if agent is None:
            raise RuntimeError(
                f"no agent plugin can render a view for channel {channel.id}; "
                "the over-budget fallback applies"
            )
        # The same render a fork would send, headed by the system prompt.
        messages = build_fork_prompt(
            conversation,
            cut,
            self._fork_prompt,
            frozenset(channel.pending_tool_call_ids),
            agent.mid_conversation_system_role,
        )
        from corvidae.llm_plugin import LLMPlugin
        llm = get_dependency(self.pm, "llm", LLMPlugin)
        result = await view_call(llm.get_client("compaction"), channel.id, messages)
        return result.text
