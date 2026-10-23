"""CompactionPlugin — token-budget conversation compaction.

Extracts compaction logic from ContextWindow into a standalone plugin that
implements the compact_conversation hookspec. Registers in main.py before
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
       voice. Otherwise, or if the fork fails, the standalone call: existing
       summaries as context + the older MESSAGE entries.
    6. Summary role is assistant, or user if the retained tail holds no
       user message (#56).
    7. Fire before_compaction (awaited) while the window is still whole.
    8. Call conversation.replace_with_summary(summary_msg, retain_count) (sync).
    9. Fire on_compaction hook if pm is set.
    10. Return True.
"""

import json
import logging
import time

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.context import DEFAULT_CHARS_PER_TOKEN, MessageType, count_tokens, message_tokens
from corvidae.fork import fork_agent, snap_cut_forward, snap_cut_to_tool_boundary
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger(__name__)


class CompactionPlugin(CorvidaePlugin):
    """Plugin that implements default token-budget conversation compaction."""

    depends_on = frozenset({"llm"})

    DEFAULT_SUMMARY_PROMPT = (
        "Please summarize the first half of this conversation, so that the second half "
        "flows naturally verbatim. The summary should include a high-level explanation of "
        "what the agent was working on or attempting to achieve. The agent's next prompt "
        "will contain your summary followed by the second half of the conversation to this point.\n\n"
        "In 1000 words or less. Preserve specific details: file paths, variable names, "
        "error messages, discoveries made, and the current line of investigation."
    )

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

    # Cap on the carried-forward prior-summary text, as a fraction of
    # max_tokens — bounds growth across repeated compaction rounds, since
    # the summary length limit lives only in the (configurable) prompt text
    # and a custom agent.compaction_prompt need not set one.
    _PRIOR_SUMMARIES_MAX_FRACTION = 0.5

    def __init__(self, pm=None) -> None:
        self.pm = pm  # May be None; on_init will set it via CorvidaePlugin.on_init
        self._compaction_threshold: float = 0.8
        self._compaction_retention: float = 0.5
        self._min_messages: int = 5
        self._chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
        self._llm_client = None
        self._summary_prompt: str = self.DEFAULT_SUMMARY_PROMPT
        self._fork_prompt: str = self.DEFAULT_FORK_PROMPT
        self._min_messages_between_compactions: int = 6
        self._last_compaction_msg_count: dict[str, int] = {}  # channel_id -> message_count at last compaction
        self._failed_compaction_cooldown: float = 30.0  # seconds
        self._last_failed_compaction: dict[str, float] = {}  # channel_id -> timestamp

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        agent_config = config.get("agent", {})
        self._compaction_threshold = agent_config.get("compaction_threshold", 0.8)
        self._compaction_retention = agent_config.get("compaction_retention", 0.5)
        self._min_messages = agent_config.get("min_messages_to_compact", 5)
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        self._summary_prompt = agent_config.get("compaction_prompt", self.DEFAULT_SUMMARY_PROMPT)
        self._fork_prompt = agent_config.get("compaction_fork_prompt", self.DEFAULT_FORK_PROMPT)

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
        """Re-read compaction thresholds and invalidate the cached LLM client.

        Invalidates _llm_client so it is re-resolved from LLMPlugin on the
        next compaction call (necessary after LLMPlugin swaps the client on reload).
        """
        agent_config = config.get("agent", {})
        self._compaction_threshold = agent_config.get("compaction_threshold", 0.8)
        self._compaction_retention = agent_config.get("compaction_retention", 0.5)
        self._min_messages = agent_config.get("min_messages_to_compact", 5)
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        self._summary_prompt = agent_config.get("compaction_prompt", self.DEFAULT_SUMMARY_PROMPT)
        self._fork_prompt = agent_config.get("compaction_fork_prompt", self.DEFAULT_FORK_PROMPT)

        # Invalidate cached LLM client so it is re-borrowed from LLMPlugin.
        # LLMPlugin may have swapped the client during this same reload cycle.
        self._llm_client = None

        self._validate_config()

        logger.info("on_config_reload: compaction thresholds updated")

    @hookimpl(trylast=True)
    async def compact_conversation(self, channel, conversation, max_tokens):
        """Compact the conversation if it exceeds 80% of max_tokens.

        Returns True if compaction was performed, None otherwise.
        """
        channel_id = conversation.channel_id

        if conversation.token_estimate() < self._compaction_threshold * max_tokens:
            return None

        if len(conversation.messages) <= self._min_messages:
            return None

        # Cooldown after a failed compaction: don't retry immediately.
        last_fail = self._last_failed_compaction.get(channel_id, 0)
        if time.monotonic() - last_fail < self._failed_compaction_cooldown:
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

        # Separate existing summaries from new messages.
        # Summaries are carried forward as context so that repeated compaction
        # doesn't lose accumulated knowledge (the "death spiral" where
        # compacting tool-only messages with no user input produces a summary
        # saying "no user instructions").
        prior_summaries = [
            m for m in older
            if m.get("_message_type", MessageType.MESSAGE) == MessageType.SUMMARY
        ]
        new_messages = [
            m for m in older
            if m.get("_message_type", MessageType.MESSAGE) == MessageType.MESSAGE
        ]
        # Strip every internal _-prefixed tag before passing to the LLM
        # (bootstrap-mapping §4.8 — the rowid tag must not leak into requests).
        prior_summaries_clean = [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in prior_summaries
        ]
        new_messages_clean = [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in new_messages
        ]

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
                standalone_token = set_attribution(mode="standalone")
                try:
                    summary_text = await self._summarize(
                        new_messages_clean, prior_summaries_clean, max_tokens=max_tokens
                    )
                finally:
                    reset_attribution(standalone_token)
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

        None means use the standalone summarizer: forks are off for
        compaction, or the fork failed or came back empty (logged). The
        fork's prefix already holds any prior summary at its head, so the
        instruction carries it forward without a bounded copy.
        """
        agent = fork_agent(self.pm, "compaction")
        if agent is None:
            return None
        try:
            result = await agent.fork(channel, self._fork_prompt, cut=cut)
        except Exception:
            logger.warning(
                "compaction fork failed; falling back to the standalone summary",
                exc_info=True, extra={"channel_id": channel.id},
            )
            return None
        if not result.text.strip():
            logger.warning(
                "compaction fork returned no text; falling back to the standalone summary",
                extra={"channel_id": channel.id},
            )
            return None
        return result.text

    def _bound_prior_summaries_text(self, prior_summaries: list[dict], max_tokens: int) -> str:
        """Join prior-summary content and cap it to a fixed fraction of
        max_tokens, keeping the most recent (tail) portion. Without this, the
        single carried-forward summary can grow across compaction rounds until
        it alone exceeds the budget."""
        text = "\n".join(s.get("content", "") for s in prior_summaries)
        cap_tokens = int(max_tokens * self._PRIOR_SUMMARIES_MAX_FRACTION)
        if count_tokens(text) <= cap_tokens:
            return text
        cap_chars = int(cap_tokens * DEFAULT_CHARS_PER_TOKEN)
        return "[...earlier summary content truncated...]\n" + text[-cap_chars:]

    async def _summarize(self, messages: list[dict], prior_summaries: list[dict] | None = None, *, max_tokens: int) -> str:
        """Ask the LLM to summarize a list of messages.

        Sends the messages to the LLM with a system prompt asking for a
        concise summary that preserves key facts, decisions, and context.

        If prior_summaries is provided, they are prepended as context so
        the new summary can build on accumulated knowledge rather than
        producing an independent (and potentially contradictory) summary.

        This is a separate method so tests can patch it via
        patch.object(plugin, "_summarize", ...).

        Args:
            messages: List of message dicts (role/content, no _message_type).
            prior_summaries: Optional list of prior summary message dicts to
                carry forward.
            max_tokens: The channel's max context tokens, used to derive the
                truncation budget for this summarization call.

        Returns:
            Summary text string from the LLM.
        """
        # Resolve the LLM client lazily — on_start hooks run in LIFO order
        # (last registered = first called), so CompactionPlugin.on_start fires
        # before LLMPlugin.on_start. The client is not available until all
        # on_start hooks have completed, so we resolve it here at compaction
        # time instead.
        if self._llm_client is None and self.pm is not None:
            from corvidae.llm_plugin import LLMPlugin
            llm = get_dependency(self.pm, "llm", LLMPlugin)
            self._llm_client = llm.main_client
        if self._llm_client is None:
            raise RuntimeError("LLM client not available for compaction")

        # Build system prompt before truncation — the token budget depends on
        # its size. The configured prompt (agent.compaction_prompt) leads in
        # every round; with prior summaries, an instruction to incorporate
        # their (bounded) content follows it.
        if prior_summaries:
            system_content = (
                self._summary_prompt + "\n\n"
                "Below are PRIOR SUMMARIES from earlier compaction rounds, followed by "
                "NEW MESSAGES that need to be incorporated. Your summary MUST preserve "
                "ALL key information from the prior summaries AND add any new facts, "
                "decisions, discoveries, or user instructions from the new messages. "
                "Do NOT discard information from the prior summaries — carry it forward.\n\n"
                "PRIOR SUMMARIES:\n"
                + self._bound_prior_summaries_text(prior_summaries, max_tokens)
            )
        else:
            system_content = self._summary_prompt

        budget = max(max_tokens - count_tokens(system_content) - self._SUMMARY_RESPONSE_RESERVE_TOKENS, 0)

        # Cap the summarization input by both message count (cheap first
        # filter) and half the token budget, head and tail, in one pass over
        # the original list — avoids double-processing and head/tail overlap
        # bookkeeping.
        max_head_tail = 50
        half_budget = budget // 2

        head, head_tokens, i = [], 0, 0
        while i < len(messages) and len(head) < max_head_tail:
            t = message_tokens(messages[i])
            if head and head_tokens + t > half_budget:
                break
            head.append(messages[i])
            head_tokens += t
            i += 1

        tail, tail_tokens, j = [], 0, len(messages) - 1
        while j >= i and len(tail) < max_head_tail:
            t = message_tokens(messages[j])
            if tail and tail_tokens + t > half_budget:
                break
            tail.append(messages[j])
            tail_tokens += t
            j -= 1
        tail.reverse()

        omitted = len(messages) - len(head) - len(tail)
        if omitted > 0:
            marker = {"role": "user", "content": f"[...{omitted} messages omitted...]"}
            messages = head + [marker] + tail

        response = await self._llm_client.chat([
            {"role": "system", "content": system_content},
            {"role": "user", "content": json.dumps(messages)},
        ])
        return response["choices"][0]["message"]["content"]
