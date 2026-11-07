"""ContextWindow — in-memory conversation context.

Provides the ContextWindow class, MessageType enum, count_tokens function, and
DEFAULT_CHARS_PER_TOKEN constant. Token counting uses tiktoken (cl100k_base
encoding) when available, falling back to character-based estimation if tiktoken
cannot be imported. This module is purely in-memory: no database access, no
async I/O. Persistence is handled externally via the persist_message and
on_compaction hook infrastructure.

Logging:
    - WARNING: logged once at import time if tiktoken is unavailable
    - WARNING: logged per call when tiktoken fails and count_tokens falls back
      to the character-based estimate
    - DEBUG: entries removed by type
"""

import enum
import json
import logging
from typing import TYPE_CHECKING

from corvidae.provenance import frame

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
    """Count tokens in a string using tiktoken, or fall back to char estimate.

    Never raises: any tiktoken failure is logged and the character-based
    estimate is returned instead (PRD 3.3.1; #352).
    """
    # Empty text costs nothing.
    if not text:
        return 0
    if _encoder is not None:
        try:
            # tiktoken's default error stops special-token strings in user text
            # being encoded as real control tokens when the IDs go to a model.
            # We only measure length and never send IDs, so the check protects
            # nothing, and would send every message containing such a string
            # (tool output, pasted logs) down the fallback below.
            return len(_encoder.encode(text, disallowed_special=()))
        except Exception as exc:
            # Rescue: log it, then fall through to the character estimate.
            # shortcut: ~3.5 chars/token is approximate; upgrade if tiktoken
            # failures turn out to be routine rather than rare.
            logger.warning(
                "tiktoken failed on %d-char text (%s); using character-based "
                "estimate instead",
                len(text),
                type(exc).__name__,
                exc_info=True,
            )
    # Character-based estimate: tiktoken is unavailable or just failed.
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


def unanswered_user_messages(messages: list[dict]) -> list[dict]:
    """Return the user MESSAGE rows after the window's last assistant row.

    These are the messages no reply has answered yet (a coalesced burst, #398),
    in window order; all user MESSAGE rows when there is no assistant row.
    CONTEXT rows and notification/tool rows are skipped by type and role. A
    row with no ``_message_type`` counts as MESSAGE.
    """
    # Find where the last assistant row sits (-1 when there is none).
    last_assistant = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_assistant = i
    # Keep only user-role MESSAGE rows after that point.
    return [
        msg for msg in messages[last_assistant + 1:]
        if msg.get("role") == "user"
        and msg.get("_message_type", MessageType.MESSAGE) == MessageType.MESSAGE
    ]


def well_formed_calls(msg: object) -> list[dict]:
    """The tool calls in ``msg`` that can be rendered: dicts with a string id.

    A backend can omit ``id``, and ``turn.py`` passes ``tool_calls`` through
    unnormalized (#392), so every reader of the field filters through this one
    rule rather than trusting the shape it was given.
    """
    if not isinstance(msg, dict):
        return []
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [c for c in calls if isinstance(c, dict) and isinstance(c.get("id"), str)]


def result_call_id(msg: object) -> str | None:
    """The ``tool_call_id`` of a tool-result row, or None when it has none.

    Only a non-empty string can match a call: the field arrives from
    ``message_log`` unvalidated, and an unhashable value (a list, a dict)
    would raise at set membership (#392).
    """
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return None
    call_id = msg.get("tool_call_id")
    return call_id if isinstance(call_id, str) and call_id else None


def message_tokens(msg: dict) -> int:
    """Token count of a message the way it actually gets sent: the
    JSON-serialized visible dict, not just `content`. Catches size hiding in
    `tool_calls` and `reasoning_content` that content-only counting misses.
    Accepts messages in tagged or already-clean form — `_visible()` strips
    any `_`-prefixed keys itself, and is a no-op on a dict with none.

    Never raises on a row's content (PRD §3.1.18.1): ``default=str`` counts a
    value JSON cannot encode by its text, and a row that is not a mapping at
    all is counted by its repr.
    """
    if not isinstance(msg, dict):
        return count_tokens(repr(msg))
    return count_tokens(json.dumps(_visible(msg), default=str))


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
        # The key fenced payloads' nonces derive from (corvidae/provenance.py);
        # None until the Agent sets it, so a window that missed it fails loudly.
        self.fence_key: bytes | None = None
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

        Every row the window can hold is rendered (PRD §3.1.18.1). A row that
        is not a message object is left out; a malformed tool call in an
        otherwise renderable row is left out of it, and a tool result whose
        call had no usable id renders as an orphan result. Each omission logs
        one WARNING naming the channel and the row id.
        """
        result_at = {}
        for i, m in enumerate(self.messages):
            call_id = result_call_id(m)
            if call_id is not None and call_id not in result_at:
                result_at[call_id] = i

        # Only well-formed calls are placed or claimed; the rest are dropped
        # from the render below, where each row's loss is logged once.
        emit_at = {}
        for i, msg in enumerate(self.messages):
            calls = well_formed_calls(msg)
            if not calls:
                continue
            if all(c["id"] in result_at for c in calls):
                emit_at[i] = max(result_at[c["id"]] for c in calls)
            else:
                emit_at[i] = i
        anchored = {j: i for i, j in emit_at.items()}
        claimed = {c["id"] for m in self.messages for c in well_formed_calls(m)}

        def block(i: int) -> list[dict]:
            msg = self.messages[i]
            out = [self._renderable(msg)]
            for call in well_formed_calls(msg):
                result = self.messages[result_at[call["id"]]] if call["id"] in result_at else None
                out.append(frame(result, _visible(result), self.channel_id, self.fence_key)[1] if result is not None else {
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
        # The header computed for the previous row when it was a person's
        # message: a burst from one sender carries one header.
        prev_header = None
        for i, msg in enumerate(self.messages):
            before, prev_header = prev_header, None
            # A row that is not a message object can't be rendered at all.
            if not isinstance(msg, dict):
                self._warn_unrenderable(
                    i, f"is a {type(msg).__name__}, not a message object",
                    "left out of the prompt",
                )
                continue
            # A row whose only payload is calls that can't carry an id has
            # nothing left to render, and reporting it once is enough.
            malformed = self._malformed_call_count(msg)
            if malformed and not well_formed_calls(msg) and not msg.get("content"):
                self._warn_unrenderable(
                    i, f"carries {malformed} tool call(s) with no usable id",
                    "left out of the prompt",
                )
                continue
            # One warning per row, before either render path so the deferred
            # block path reports it too.
            if malformed:
                self._warn_unrenderable(
                    i, f"carries {malformed} tool call(s) with no usable id",
                    "rendered without them",
                )
            if i in anchored:
                out += block(anchored[i])
            # A result whose call is in the window was emitted inside its
            # call's block above. Compare through result_call_id: the raw
            # field can be unhashable (#392).
            call_id = result_call_id(msg)
            if call_id is not None and call_id in claimed:
                continue
            if i in emit_at:
                continue
            # Orphaned result: its call is gone, so frame it as a notification.
            if msg.get("role") == "tool":
                _, framed = frame(msg, _visible(msg), self.channel_id, self.fence_key)
                out.append({
                    "role": "system",
                    "content": f"[{ORPHAN_RESULT_LABEL}]\n\n{framed.get('content', '')}",
                })
                continue
            header, rendered = frame(msg, self._renderable(msg), self.channel_id, self.fence_key)
            if msg.get("role") == "user":
                prev_header = header
            if header is not None and header != before:
                out.append(header)
            out.append(rendered)

        # Re-role non-leading system messages (CONTEXT, untargeted
        # notifications) for strict chat templates. A pure function of each
        # message and the rows before it, so the rendered prefix stays stable
        # across turns.
        if mid_conversation_system_role != "system":
            out[1:] = [
                {**m, "role": mid_conversation_system_role} if m.get("role") == "system" else m
                for m in out[1:]
            ]
        return out

    @staticmethod
    def _malformed_call_count(msg: dict) -> int:
        """How many of ``msg``'s tool calls have no usable id.

        A truthy or falsy non-list value (``""``, ``0``, ``{}``, a string) is
        one unusable call: testing truthiness would let ``""`` through
        unwarned (#392).
        """
        if "tool_calls" not in msg:
            return 0
        raw = msg["tool_calls"]
        if raw is None:
            return 0
        if not isinstance(raw, list):
            return 1
        return len(raw) - len(well_formed_calls(msg))

    @staticmethod
    def _renderable(msg: dict) -> dict:
        """``msg`` as it goes to the model: visible fields, well-formed calls
        only, and no ``tool_calls`` key at all when none survive (an empty
        list is not the same as an absent one to a chat template)."""
        out = _visible(msg)
        if "tool_calls" not in out:
            return out
        kept = well_formed_calls(msg)
        if kept:
            out["tool_calls"] = kept
        else:
            del out["tool_calls"]
        return out

    def _warn_unrenderable(self, index: int, what: str, consequence: str) -> None:
        """Log one WARNING for content left out of the prompt (PRD §3.1.18.1).

        Names the channel and the row: ``_db_id`` when the row was persisted,
        its position when it was not. Never the row's content (it may be
        unrenderable for that very reason).
        """
        row = self.messages[index]
        row_id = row.get("_db_id") if isinstance(row, dict) else None
        where = (
            f"message_log id {row_id}" if isinstance(row_id, int)
            else f"window position {index} (not persisted)"
        )
        logger.warning(
            "channel %s: %s at %s; %s — the turn continues without it "
            "(inspect or redact the row)",
            self.channel_id, what, where, consequence,
            extra={"channel": self.channel_id},
        )

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
        # A row that is not a mapping has no content to truncate; it stands
        # as it is and the caller reports the overage (#392).
        truncated = False
        newest = self.messages[-1]
        if isinstance(newest, dict) and fixed + message_tokens(newest) > max_tokens:
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
