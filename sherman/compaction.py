"""CompactionPlugin — token-budget conversation compaction.

Extracts compaction logic from ConversationLog into a standalone plugin that
implements the compact_conversation hookspec. Registers in main.py before
AgentPlugin so it runs as the default compaction strategy.

Algorithm:
    1. Skip if token_estimate < 80% of max_tokens.
    2. Skip if len(messages) <= 5.
    3. Backward walk retaining messages within 50% of max_tokens.
    4. Skip if all messages fit (retain_count >= len(messages)).
    5. Filter older messages to MESSAGE type, strip _message_type metadata.
    6. Summarize via LLM (_summarize method, patchable in tests).
    7. Call conversation.replace_with_summary(summary_msg, retain_count).
    8. Return True.
"""

import json
import logging

from sherman.hooks import hookimpl

logger = logging.getLogger(__name__)


class CompactionPlugin:
    """Plugin that implements default token-budget conversation compaction."""

    @hookimpl
    async def compact_conversation(self, conversation, client, max_tokens):
        """Compact the conversation if it exceeds 80% of max_tokens.

        Returns True if compaction was performed, None otherwise.
        """
        from sherman.conversation import MessageType

        if conversation.token_estimate() < 0.8 * max_tokens:
            return None

        if len(conversation.messages) <= 5:
            return None

        logger.warning(
            "compaction triggered (approaching context limit)",
            extra={"channel_id": conversation.channel_id},
        )

        retain_budget = int(max_tokens * 0.5)
        retain_count = 0
        retain_tokens = 0
        for msg in reversed(conversation.messages):
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = ""
            msg_tokens = int(len(content) / 3.5)
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

        summary_text = await self._summarize(client, older_clean)
        summary_msg = {
            "role": "assistant",
            "content": f"[Summary of earlier conversation]\n{summary_text}",
        }

        await conversation.replace_with_summary(summary_msg, retain_count)
        return True

    async def _summarize(self, client, messages: list[dict]) -> str:
        """Ask the LLM to summarize a list of messages.

        Sends the messages to the LLM with a system prompt asking for a
        concise summary that preserves key facts, decisions, and context.

        This is a separate method so tests can patch it via
        patch.object(plugin, "_summarize", ...).

        Args:
            client: LLMClient instance.
            messages: List of message dicts (role/content, no _message_type).

        Returns:
            Summary text string from the LLM.
        """
        response = await client.chat([
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
