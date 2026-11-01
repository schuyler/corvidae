"""ContextWindow — in-memory conversation context.

Provides the ContextWindow class, MessageType enum, count_tokens function, and
DEFAULT_CHARS_PER_TOKEN constant. Token counting uses tiktoken (cl100k_base
encoding) when available, falling back to character-based estimation if tiktoken
cannot be imported. This module is purely in-memory: no database access, no
async I/O. Persistence is handled externally via the persist_message and
on_compaction hook infrastructure.

Logging:
    - WARNING: logged once at import time if tiktoken is unavailable
    - DEBUG: entries removed by type
"""

import enum
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.tool_policy import ToolView

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
    UNSENT: an assistant message a should_send_response gate vetoed. It
            stays in the window (and reloads into it) but no transport ever
            sees it, including history replay. Compaction and consolidation
            read MESSAGE rows only, so it is never summarized as said.
    """

    MESSAGE = "message"
    SUMMARY = "summary"
    CONTEXT = "context"
    UNSENT = "unsent"


#: Default characters-per-token estimate for rough token counting.
DEFAULT_CHARS_PER_TOKEN: float = 3.5

TOOL_RESULT_PENDING = "[still running — the result will arrive in a later message]"
TOOL_RESULT_ABANDONED = "[did not complete — no result was returned]"
#: Frame for a tool result whose call is no longer in the window (#79).
ORPHAN_RESULT_LABEL = "result of an earlier tool call that is no longer shown"

#: Label the not-sent marker opens with. The Agent appends the marker (a
#: CONTEXT system row) right after the assistant row whose emission a
#: should_send_response gate vetoed, so it reloads beside that row.
UNSENT_MARKER_LABEL = "not sent"
_UNSENT_MARKER_SUBJECTS = {
    "final": "Your response above was not sent",
    "progress": "Your progress text beginning «{excerpt}» was not shown",
    "thinking": "Your reasoning on your latest turn was not shown",
}
#: The progress marker quotes this much of the text: build_prompt moves a
#: tool-calls block to its last result, so the marker renders before the
#: block and its position alone can't say which text it means. The final
#: and thinking markers quote nothing (the final sits right above it;
#: reasoning must not outlive reasoning_history: drop in a marker).
_UNSENT_MARKER_EXCERPT_CHARS = 40


def unsent_marker_text(emission: str, text: str) -> str:
    """The not-sent marker content for a vetoed ``emission`` of ``text``."""
    excerpt = " ".join(text.split())
    if len(excerpt) > _UNSENT_MARKER_EXCERPT_CHARS:
        excerpt = excerpt[:_UNSENT_MARKER_EXCERPT_CHARS].rstrip() + "…"
    subject = _UNSENT_MARKER_SUBJECTS.get(emission, f"Your {emission} was not sent")
    return (
        f"[{UNSENT_MARKER_LABEL}: {emission}]\n\n"
        f"{subject.format(excerpt=excerpt)}; the channel did not see it. "
        "Don't refer to it as something you said."
    )


def unsent_marker_emission(content: object) -> str | None:
    """The emission a not-sent marker's ``content`` names, or None when the
    content is not a marker. History replay uses it to hold back what a
    gate suppressed live."""
    if not isinstance(content, str):
        return None
    prefix = f"[{UNSENT_MARKER_LABEL}: "
    if not content.startswith(prefix):
        return None
    end = content.find("]", len(prefix))
    if end == -1:
        return None
    return content[len(prefix):end]


def _visible(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if not k.startswith("_")}


def message_tokens(msg: dict) -> int:
    """Token count of a message the way it actually gets sent: the
    JSON-serialized visible dict, not just `content`. Catches size hiding in
    `tool_calls` and `reasoning_content` that content-only counting misses.
    Accepts messages in tagged or already-clean form — `_visible()` strips
    any `_`-prefixed keys itself, and is a no-op on a dict with none.
    """
    return count_tokens(json.dumps(_visible(msg)))


class ContextWindow:
    """In-memory conversation context window.

    Maintains a list of tagged message dicts. All operations are synchronous
    and purely in-memory. Persistence is handled externally by plugins that
    implement the persist_message and on_compaction hooks.

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
        # The conversation's tool set (corvidae/tool_policy.py), resolved
        # with the system prompt and fixed until the next compaction.
        self.tool_view: "ToolView | None" = None
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

    def build_prompt(
        self,
        pending_tool_call_ids: frozenset = frozenset(),
        mid_conversation_system_role: str = "system",
    ) -> list[dict]:
        """Return [system_message, *self.messages] with internal tags stripped.

        Does not modify self.messages. Strips every _-prefixed key
        (_message_type, _db_id, ...) from each message dict before
        returning — internal tags must never reach the LLM
        (bootstrap-mapping §4.8).

        An assistant tool_calls message is deferred to the arrival index of
        the last of its results, so call+results emit as one contiguous
        block there (synthesizing a placeholder for any call with no result
        yet). If any call in the message is unanswered, the block emits in
        place instead. A tool message whose call is not in the window
        (compaction dropped the call, whether it was answered or still in
        flight, #79) keeps its place but renders as a framed system
        notification without ``tool_call_id``, like an untargeted
        notification: strict chat templates reject a tool result with no
        preceding call. It depends only on whether the call is in the
        window; calls precede their results, so a fork's prefix renders it
        identically.

        mid_conversation_system_role renders every system message after the
        leading one under that role (``agent.mid_conversation_system_role``).
        Strict chat templates reject a system message that is not first;
        "user" satisfies them. Only the returned copy changes — the window
        and the DB keep ``role: system``.
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
            # Orphaned result: its call is gone, so frame it as a notification.
            if msg.get("role") == "tool":
                out.append({
                    "role": "system",
                    "content": f"[{ORPHAN_RESULT_LABEL}]\n\n{msg.get('content', '')}",
                })
                continue
            out.append(_visible(msg))

        # Re-role non-leading system messages (CONTEXT, untargeted
        # notifications) for strict chat templates. A pure function of the
        # message, so the rendered prefix stays stable across turns.
        if mid_conversation_system_role != "system":
            out[1:] = [
                {**m, "role": mid_conversation_system_role} if m.get("role") == "system" else m
                for m in out[1:]
            ]
        return out

    def token_estimate(self) -> int:
        """Token count using tiktoken (cl100k_base), with character-based fallback.

        Includes system prompt plus every message's JSON-serialized visible
        fields (see message_tokens) — not just content. Falls back to
        character-based estimation if tiktoken is unavailable.
        """
        total = count_tokens(self.system_prompt)
        for msg in self.messages:
            total += message_tokens(msg)
        return total

    def fit_to_budget(self, max_tokens: int) -> tuple[list[dict], bool]:
        """Bring the window within ``max_tokens`` without any LLM call (#308).

        The over-budget fallback behind PRD §3.3.2: drops the oldest rows from
        the head of the in-memory window (``message_log`` is never touched),
        snapping the cut forward past a straddling tool cycle when that can be
        done without dropping the newest row. The newest row is always kept;
        if it alone does not fit, its content is truncated in memory only,
        with a visible marker. Pure and synchronous; it cannot fail.

        Returns:
            ``(dropped, truncated)``: the rows removed (oldest first) and
            whether the newest row's content was truncated. ``([], False)``
            when the window already fits, or is empty.
        """
        # Nothing to do when the window already fits (or there is no row to cut).
        if self.token_estimate() <= max_tokens or not self.messages:
            return [], False

        # Function-local import: corvidae.fork imports ContextWindow from this
        # module, so a module-level import would be circular.
        from corvidae.fork import snap_cut_forward

        fixed = count_tokens(self.system_prompt)
        toks = [message_tokens(m) for m in self.messages]
        n = len(self.messages)

        # Smallest head cut whose tail fits; never cut away the newest row.
        tail = sum(toks)
        cut = n - 1
        for c in range(n):
            if fixed + tail <= max_tokens:
                cut = c
                break
            tail -= toks[c]
        cut = min(cut, n - 1)

        # Snap forward past a straddling tool cycle; moving forward only drops
        # more, so the tail still fits. If that would drop the newest row, keep
        # the raw cut and let build_prompt frame the orphaned result (§3.3.4).
        snapped = snap_cut_forward(self.messages, cut)
        if snapped <= n - 1:
            cut = snapped

        # Drop in place so any holder of the list sees the change.
        dropped = self.messages[:cut]
        del self.messages[:cut]

        # The newest row alone may still not fit: truncate it in memory only.
        truncated = False
        newest = self.messages[-1]
        if fixed + message_tokens(newest) > max_tokens:
            content = str(newest.get("content") or "")
            total = len(content)
            room = max_tokens - fixed

            def candidate(k: int) -> dict:
                """The newest row cut to its first k characters plus marker."""
                marker = (
                    "\n\n[truncated to fit the context window: showing the "
                    f"first {k} of {total} characters]"
                )
                return {**newest, "content": content[:k] + marker}

            # Binary search the largest k measured to fit; lo always holds a
            # k already known to fit, since token counts are not monotone in k.
            if message_tokens(candidate(0)) <= room:
                lo, hi = 0, total
                while lo < hi - 1:
                    mid = (lo + hi) // 2
                    if message_tokens(candidate(mid)) <= room:
                        lo = mid
                    else:
                        hi = mid
                self.messages[-1] = candidate(lo)
                truncated = True
            # Else nothing more can be removed; the caller reports the overage.
        return dropped, truncated
