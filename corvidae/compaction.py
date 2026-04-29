"""CompactionPlugin — token-budget conversation compaction.

Extracts compaction logic from ContextWindow into a standalone plugin that
implements the compact_conversation hookspec. Registers in main.py before
Agent so it runs as the default compaction strategy.

Algorithm:
    1. Skip if token_estimate < 80% of max_tokens.
    2. Skip if len(messages) <= 5.
    3. Backward walk retaining messages within 50% of max_tokens.
    4. Skip if all messages fit (retain_count >= len(messages)).
    5. Filter older messages to MESSAGE type, strip _message_type metadata.
    6. Summarize via LLM (_summarize method, patchable in tests).
    7. Call conversation.replace_with_summary(summary_msg, retain_count) (sync).
    8. Fire on_compaction hook if pm is set.
    9. Return True.
"""

import json
import logging
import time

from corvidae.context import DEFAULT_CHARS_PER_TOKEN, MessageType
from corvidae.hooks import get_dependency, hookimpl

logger = logging.getLogger(__name__)


class CompactionPlugin:
    """Plugin that implements default token-budget conversation compaction."""

    depends_on = {"llm"}

    DEFAULT_SUMMARY_PROMPT = (
        "Please summarize the first half of this conversation, so that the second half "
        "flows naturally verbatim. The summary should include a high-level explanation of "
        "what the agent was working on or attempting to achieve. The agent's next prompt "
        "will contain your summary followed by the second half of the conversation to this point.\n\n"
        "In 1000 words or less. Preserve specific details: file paths, variable names, "
        "error messages, discoveries made, and the current line of investigation."
    )

    def __init__(self, pm) -> None:
        self.pm = pm
        self._compaction_threshold: float = 0.8
        self._compaction_retention: float = 0.5
        self._min_messages: int = 5
        self._chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
        self._llm_client = None
        self._summary_prompt: str = self.DEFAULT_SUMMARY_PROMPT
        self._min_messages_between_compactions: int = 6
        self._last_compaction_msg_count: dict[str, int] = {}  # channel_id -> message_count at last compaction
        self._failed_compaction_cooldown: float = 30.0  # seconds
        self._last_failed_compaction: dict[str, float] = {}  # channel_id -> timestamp

    @hookimpl
    async def on_start(self, config: dict) -> None:
        agent_config = config.get("agent", {})
        self._compaction_threshold = agent_config.get("compaction_threshold", 0.8)
        self._compaction_retention = agent_config.get("compaction_retention", 0.5)
        self._min_messages = agent_config.get("min_messages_to_compact", 5)
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        self._summary_prompt = agent_config.get("compaction_prompt", self.DEFAULT_SUMMARY_PROMPT)

    @hookimpl
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
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = ""
            msg_tokens = int(len(content) / self._chars_per_token)
            if retain_tokens + msg_tokens > retain_budget and retain_count > 0:
                break
            retain_tokens += msg_tokens
            retain_count += 1

        # All-fit guard: no-op compaction would accomplish nothing.
        if retain_count >= len(conversation.messages):
            return None

        older = conversation.messages[:-retain_count]

        # Filter older to MESSAGE type only; exclude SUMMARY entries from summarizer input.
        older = [m for m in older if m.get("_message_type", MessageType.MESSAGE) == MessageType.MESSAGE]
        # Strip _message_type before passing to LLM to avoid serializing internal metadata.
        older_clean = [{k: v for k, v in m.items() if k != "_message_type"} for m in older]

        try:
            summary_text = await self._summarize(older_clean)
        except Exception:
            # Record failure time for cooldown
            self._last_failed_compaction[channel_id] = time.monotonic()
            raise

        # Reject blank/empty summaries — they're worse than no summary
        # because they erase context without preserving any information.
        if not summary_text or not summary_text.strip():
            logger.warning(
                "compaction produced empty summary, aborting",
                extra={"channel_id": channel_id},
            )
            self._last_failed_compaction[channel_id] = time.monotonic()
            return None

        summary_msg = {
            "role": "assistant",
            "content": f"[Summary of earlier conversation]\n{summary_text}",
        }

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
                )
            except Exception:
                logger.warning("on_compaction hook failed", exc_info=True)

        return True

    async def _summarize(self, messages: list[dict]) -> str:
        """Ask the LLM to summarize a list of messages.

        Sends the messages to the LLM with a system prompt asking for a
        concise summary that preserves key facts, decisions, and context.

        This is a separate method so tests can patch it via
        patch.object(plugin, "_summarize", ...).

        Args:
            messages: List of message dicts (role/content, no _message_type).

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

        # Cap the summarization input to avoid sending hundreds of thousands
        # of tokens to the LLM. Take the first 50 messages (early context)
        # and the last 50 messages (recent context), with a truncation marker.
        max_messages = 100
        if len(messages) > max_messages:
            head = messages[:50]
            tail = messages[-50:]
            truncated_count = len(messages) - max_messages
            truncated_marker = {
                "role": "user",
                "content": f"[...{truncated_count} messages omitted...]",
            }
            messages = head + [truncated_marker] + tail

        response = await self._llm_client.chat([
            {
                "role": "system",
                "content": self._summary_prompt,
            },
            {"role": "user", "content": json.dumps(messages)},
        ])
        return response["choices"][0]["message"]["content"]
