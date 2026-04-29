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

from corvidae.context import DEFAULT_CHARS_PER_TOKEN, MessageType
from corvidae.hooks import get_dependency, hookimpl

logger = logging.getLogger(__name__)


class CompactionPlugin:
    """Plugin that implements default token-budget conversation compaction."""

    depends_on = {"llm"}

    def __init__(self, pm=None) -> None:
        self.pm = pm
        self._compaction_threshold: float = 0.8
        self._compaction_retention: float = 0.5
        self._min_messages: int = 5
        self._chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
        self._llm_client = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        from corvidae.llm_plugin import LLMPlugin
        agent_config = config.get("agent", {})
        self._compaction_threshold = agent_config.get("compaction_threshold", 0.8)
        self._compaction_retention = agent_config.get("compaction_retention", 0.5)
        self._min_messages = agent_config.get("min_messages_to_compact", 5)
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)
        if self.pm is not None:
            llm = get_dependency(self.pm, "llm", LLMPlugin)
            self._llm_client = llm.main_client

    @hookimpl
    async def compact_conversation(self, channel, conversation, max_tokens):
        """Compact the conversation if it exceeds 80% of max_tokens.

        Returns True if compaction was performed, None otherwise.
        """
        if conversation.token_estimate() < self._compaction_threshold * max_tokens:
            return None

        if len(conversation.messages) <= self._min_messages:
            return None

        logger.warning(
            "compaction triggered (approaching context limit)",
            extra={"channel_id": conversation.channel_id},
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

        summary_text = await self._summarize(older_clean)
        summary_msg = {
            "role": "assistant",
            "content": f"[Summary of earlier conversation]\n{summary_text}",
        }

        conversation.replace_with_summary(summary_msg, retain_count)

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
        response = await self._llm_client.chat([
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation concisely, "
                    "preserving key facts, decisions, and context that "
                    "would be needed to continue the conversation."
                ),
            },
            {"role": "user", "content": json.dumps(messages)},
        ])
        return response["choices"][0]["message"]["content"]
