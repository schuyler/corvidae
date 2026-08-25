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

TOOL_RESULT_PENDING = "[still running — the result will arrive in a later message]"
TOOL_RESULT_ABANDONED = "[did not complete — no result was returned]"


def _visible(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if not k.startswith("_")}


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

    def build_prompt(self, pending_tool_call_ids: frozenset = frozenset()) -> list[dict]:
        """Return [system_message, *self.messages] with internal tags stripped.

        Does not modify self.messages. Strips every _-prefixed key
        (_message_type, _db_id, ...) from each message dict before
        returning — internal tags must never reach the LLM
        (bootstrap-mapping §4.8).

        An assistant tool_calls message is deferred to the arrival index of
        the last of its results, so call+results emit as one contiguous
        block there (synthesizing a placeholder for any call with no result
        yet). If any call in the message is unanswered, the block emits in
        place instead. A tool message whose call is not in the window (e.g.
        compaction dropped it) stays where it is.
        """
        result_at = {}
        for i, m in enumerate(self.messages):
            if m.get("role") == "tool" and m.get("tool_call_id") and m["tool_call_id"] not in result_at:
                result_at[m["tool_call_id"]] = i

        emit_at = {}
        for i, msg in enumerate(self.messages):
            calls = msg.get("tool_calls") or []
            if not calls:
                continue
            if all(c["id"] in result_at for c in calls):
                emit_at[i] = max(result_at[c["id"]] for c in calls)
            else:
                emit_at[i] = i
        anchored = {j: i for i, j in emit_at.items()}
        claimed = {c["id"] for m in self.messages for c in (m.get("tool_calls") or [])}

        def block(i: int) -> list[dict]:
            msg = self.messages[i]
            out = [_visible(msg)]
            for call in msg.get("tool_calls") or []:
                result = self.messages[result_at[call["id"]]] if call["id"] in result_at else None
                out.append(_visible(result) if result is not None else {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": (
                        TOOL_RESULT_PENDING
                        if call["id"] in pending_tool_call_ids
                        else TOOL_RESULT_ABANDONED
                    ),
                })
            return out

        out = [{"role": "system", "content": self.system_prompt}]
        for i, msg in enumerate(self.messages):
            if i in anchored:
                out += block(anchored[i])
            if msg.get("role") == "tool" and msg.get("tool_call_id") in claimed:
                continue
            if i in emit_at:
                continue
            out.append(_visible(msg))
        return out

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
