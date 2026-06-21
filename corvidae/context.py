"""ContextWindow — in-memory conversation context.

Provides the ContextWindow class, MessageType enum, count_tokens function, and
DEFAULT_CHARS_PER_TOKEN constant. Token counting uses tiktoken (cl100k_base
encoding) when available, falling back to character-based estimation if tiktoken
cannot be imported. This module is purely in-memory: no database access, no
async I/O. Persistence is handled externally via the on_conversation_event and
on_compaction hook infrastructure.

Logging:
    - WARNING: logged once at import time if tiktoken is unavailable
    - DEBUG: entries removed by type
"""

import enum
import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoder = None
    logger.warning(
        "tiktoken unavailable; falling back to character-based token estimation"
    )

_FALLBACK_CHARS_PER_TOKEN: float = 3.5


def count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken, or fall back to char estimate."""
    if not text:
        return 0
    if _encoder is not None:
        return len(_encoder.encode(text))
    return int(len(text) / _FALLBACK_CHARS_PER_TOKEN)


class MessageType(str, enum.Enum):
    """Persistence category for a message log entry.

    Controls storage and filtering behavior, not conversational role.
    Values match the ``message_type`` TEXT column in the ``message_log`` table.

    MESSAGE: an ordinary conversation message (user or assistant turn).
    SUMMARY: a compaction summary that replaces a range of older messages.
    CONTEXT: plugin-injected contextual information (memory, notes, retrieved
             documents, etc.) that should appear in the prompt but is not part
             of the conversational turn history.
    """

    MESSAGE = "message"
    SUMMARY = "summary"
    CONTEXT = "context"


#: Default characters-per-token estimate for rough token counting.
DEFAULT_CHARS_PER_TOKEN: float = 3.5


class ContextWindow:
    """In-memory conversation context window.

    Maintains a list of tagged message dicts. All operations are synchronous
    and purely in-memory. Persistence is handled externally by plugins that
    implement the on_conversation_event and on_compaction hooks.

    Attributes:
        channel_id: Identifies the channel this window belongs to.
        messages: In-memory list of tagged message dicts.
        system_prompt: The system prompt prepended by ``build_prompt``.
        chars_per_token: Retained for interface and config compatibility. Has no
            effect on token counting — ``count_tokens()`` uses the module-level
            ``_FALLBACK_CHARS_PER_TOKEN`` constant, not this attribute.
    """

    def __init__(self, channel_id: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN):
        self.channel_id = channel_id
        self.messages: list[dict] = []
        self.system_prompt: str = ""
        self.chars_per_token: float = chars_per_token

    def append(self, message: dict, message_type: MessageType = MessageType.MESSAGE) -> None:
        """Append a message to self.messages with _message_type tagging.

        Makes a shallow copy of ``message`` before adding ``_message_type``
        so the caller's dict is not mutated.

        Args:
            message: The message dict (e.g. ``{"role": "user", "content": "..."}``).
            message_type: Persistence category. Defaults to MESSAGE.
        """
        tagged = dict(message)
        tagged["_message_type"] = message_type
        self.messages.append(tagged)

    def replace_with_summary(self, summary_msg: dict, retain_count: int) -> None:
        """Replace older messages with a summary in-memory.

        Args:
            summary_msg: Untagged summary dict (role/content). Tagged with
                _message_type=SUMMARY in the in-memory list.
            retain_count: Number of most-recent messages to keep alongside the
                summary. Must not exceed len(self.messages).

        Raises:
            ValueError: If retain_count > len(self.messages).
        """
        if retain_count > len(self.messages):
            raise ValueError(
                f"retain_count ({retain_count}) exceeds len(messages) ({len(self.messages)})"
            )
        retained = self.messages[-retain_count:] if retain_count > 0 else []
        tagged = {**summary_msg, "_message_type": MessageType.SUMMARY}
        self.messages = [tagged] + retained

    def build_prompt(self) -> list[dict]:
        """Return [system_message, *self.messages] with _message_type stripped.

        Does not modify self.messages. Strips internal _message_type metadata
        from each message dict before returning.
        """
        cleaned = []
        for msg in self.messages:
            if "_message_type" in msg:
                msg = {k: v for k, v in msg.items() if k != "_message_type"}
            cleaned.append(msg)
        return [{"role": "system", "content": self.system_prompt}] + cleaned

    def token_estimate(self) -> int:
        """Token count using tiktoken (cl100k_base), with character-based fallback.

        Includes system prompt plus all message content. Non-string content
        (None, lists) is treated as 0 tokens. Falls back to character-based
        estimation if tiktoken is unavailable.
        """
        total = count_tokens(self.system_prompt)
        for msg in self.messages:
            content = msg.get("content") or ""
            if not isinstance(content, str):
                continue
            total += count_tokens(content)
        return total

    def remove_by_type(self, message_type: MessageType) -> int:
        """Remove all in-memory entries of a given type.

        Only allowed for non-MESSAGE, non-SUMMARY types (use compaction for those).

        Args:
            message_type: The type to remove. Must not be MESSAGE or SUMMARY.

        Returns:
            The number of in-memory entries removed.

        Raises:
            ValueError: If message_type is MESSAGE or SUMMARY.
        """
        if message_type in (MessageType.MESSAGE, MessageType.SUMMARY):
            raise ValueError(
                f"Cannot remove {message_type.value!r} entries — "
                f"use compaction for MESSAGE and SUMMARY lifecycle"
            )
        before = len(self.messages)
        self.messages = [
            m for m in self.messages
            if m.get("_message_type") != message_type
        ]
        removed = before - len(self.messages)
        logger.debug(
            "removed entries by type",
            extra={
                "channel_id": self.channel_id,
                "message_type": message_type.value,
                "count": removed,
            },
        )
        return removed
