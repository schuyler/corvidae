"""The pre-send hold-and-revise state machine (#1149, #1169 part 2).

``PreSendHold`` is a helper class, not a plugin: it has no entry point and
fires no hooks of its own. Its owner (appraisal, or critique's fallback when
appraisal is absent) passes it ``pm`` and calls it from its own hookimpls.
The owner's check returns ``await hold.refuse(...)``; the kernel decides
whether that refusal was in time. ``on_final`` then learns the outcome: an
in-time refusal is held, the objections go to the ``held`` store with a
wake-up, and exactly one revising turn follows, unchecked.
See docs/1149-presend-hold-design.md for the full account.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass

from corvidae.held import HeldContext
from corvidae.hooks import Refusal, final_was_unsent, get_dependency

logger = logging.getLogger("corvidae.presend")

#: The held batch's source and its frame label.
SOURCE = "pre-send check"

#: The revising turn's stub text. It names no NO_REPLY: a revision is never held.
REVISION_STUB = (
    "Your last reply was held back before it was sent, for the reasons in the "
    "CONTEXT above. Write a revised reply now."
)

#: Bound on the pending-refusal LRU; a stale entry ages out.
PENDING_LRU_MAXSIZE = 64


@dataclass
class _Pending:
    """A refusal returned whose outcome is not yet known."""

    channel_id: str
    origin: str
    #: The refused text; ``on_final`` matches it against the final's text.
    draft: str
    objections: list[str]


@dataclass
class _Revision:
    """A hold is outstanding on a channel."""

    origin: str
    #: The current turn's origin equals the hold's (set at ``before_agent_turn``).
    turn_matches: bool = False
    #: The current turn's trace (set at ``before_agent_turn``).
    turn_trace: str | None = None
    #: Trace of the turn at which our batch was offered; None until then.
    offered_trace: str | None = None
    #: Our entries admitted at that turn.
    admitted: int = 0


class PreSendHold:
    """Hold-and-revise state for one owner (``reporter``: "appraisal" or "critique")."""

    def __init__(self, pm, reporter: str) -> None:
        self.pm = pm
        self.reporter = reporter
        # Trace id -> refusal awaiting its outcome, oldest first.
        self._pending: collections.OrderedDict[str, _Pending] = collections.OrderedDict()
        # Channel id -> the outstanding hold on that channel.
        self._revisions: dict[str, _Revision] = {}

    def suppressed(self, channel_id: str) -> bool:
        """True while a hold is outstanding on the channel: do not check."""
        return channel_id in self._revisions

    async def refuse(
        self, channel, trace_id: str, text: str, annotations: dict, objections: list[str],
    ) -> Refusal:
        """Remember the refused draft and answer the refusal for the kernel.

        The reason is what the kernel's late path prints when the refusal
        arrives after the send (R5).
        """
        # The hold's origin: the annotation, else the trace's recorded one, else "user".
        origin = (annotations or {}).get("origin")
        if not origin:
            origin = await self.pm.ahook.resolve_trace_origin(trace_id=trace_id)
        origin = origin or "user"
        # Store the pending refusal, bounded, newest last.
        self._pending[trace_id] = _Pending(channel.id, origin, text, list(objections))
        self._pending.move_to_end(trace_id)
        while len(self._pending) > PENDING_LRU_MAXSIZE:
            self._pending.popitem(last=False)
        return Refusal(reason="\n\n".join(objections))

    def on_turn(self, channel, trace_id: str | None, annotations: dict) -> None:
        """Note which turn is running (``before_agent_turn``).

        Invariant: ``before_agent_turn`` runs before every admit in the same
        queue item, so the turn recorded here is the turn ``on_admitted`` sees.
        """
        revision = self._revisions.get(channel.id)
        if revision is None:
            return
        revision.turn_matches = (annotations or {}).get("origin") == revision.origin
        revision.turn_trace = trace_id

    async def on_admitted(
        self, channel, source: str, admitted_keys: list[str], duplicate_keys: list[str],
    ) -> None:
        """Record the offer of our batch (``on_held_admitted``).

        The revision is the turn the held store offered our batch to: a turn
        of the hold's origin, the first one or any later batch at the same turn.
        """
        revision = self._revisions.get(channel.id)
        if revision is None or source != SOURCE or not revision.turn_matches:
            return
        # Only the offering turn counts; a later turn's batch is another hold's.
        if revision.offered_trace is not None and revision.turn_trace != revision.offered_trace:
            return
        revision.offered_trace = revision.turn_trace
        revision.admitted = len(admitted_keys)

    async def on_final(
        self, channel, trace_id: str | None, response_text: str, annotations: dict,
    ) -> bool:
        """Learn a final's outcome; True when its trace was held (skip idle work)."""
        # Clear a due revision first: its own final, or nothing outstanding.
        revision = self._revisions.get(channel.id)
        if (
            revision is not None and revision.offered_trace is not None
            and (trace_id == revision.offered_trace or not channel.pending_tool_call_ids)
        ):
            del self._revisions[channel.id]
            if revision.admitted == 0:
                logger.warning(
                    "pre-send hold on channel %s: the held objections were not "
                    "admitted at the revising turn (duplicates, or no room), so "
                    "the revision may have gone out without them",
                    channel.id, extra={"channel": channel.id, "trace_id": trace_id},
                )
        # Take the pending refusal only when it is this very draft.
        pending = self._pending.get(trace_id) if trace_id is not None else None
        if pending is None or pending.draft != response_text:
            return False
        del self._pending[trace_id]
        # A sent final means the refusal was late: the kernel's late path delivers it.
        if not final_was_unsent(channel):
            return False
        # Held in time: store the objections, wake the revising turn, record it.
        try:
            held = get_dependency(self.pm, "held", HeldContext)
            await held.hold_and_wake(
                channel, pending.origin, SOURCE, pending.objections,
                stub_text=REVISION_STUB,
            )
        except Exception:
            logger.warning(
                "pre-send hold on channel %s, trace %s: the draft was held with "
                "no revising turn because the held store failed; its objections "
                "are lost", channel.id, trace_id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return True
        # State is set only after the wake-up succeeded.
        self._revisions[channel.id] = _Revision(origin=pending.origin)
        await self.pm.ahook.on_trace_record(
            trace_id=trace_id, channel_id=channel.id, source=self.reporter,
            origin=None, record={"held": {"objections": pending.objections}},
        )
        return True
