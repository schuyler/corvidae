"""Goals notify: a goal's state changes go to the channel that created it (#841).

Implements ``on_goals_state_changed`` (owned by the goals plugin in
corvidae-cognition) and turns each change into one ``notify_other_channel``
call to the goal's ``origin_channel``. Delivery, the notification fields
(``source="agent_send from <actor>"``, ``origin: "agent_send"``,
``source_channel``, ``source_trace_id``) and the ``hop_count`` loop guard all
come from cross_channel_send, which reads them from the changing turn's
attribution: the goals plugin fires the hook inside the tool call that made
the change. Every change wakes the origin channel; its turn and the decide
gate decide whether anything is said.

Skipped: a goal with no origin (seeded with ``corvidae goals seed``) and a
change made on the origin channel itself (which includes heartbeat-created
goals changed by the heartbeat). One channel per goal.

Ordering: the hook fires after the goals lock is released, so two changes to
one goal close together can arrive out of commit order. Each message names
``old → new`` so the receiver can tell; real ordering is #937's.
"""

from __future__ import annotations

import logging

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from cross_channel_send import CrossChannelSendPlugin

logger = logging.getLogger("goals_notify")


class GoalsNotifyPlugin(CorvidaePlugin):
    """Sends each goal state change to the goal's origin channel."""

    depends_on = frozenset({"cross_channel_send"})
    # Without goals the hookimpl is inactive; the kernel logs that at DEBUG.
    optional_depends_on = frozenset({"goals"})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._send: CrossChannelSendPlugin | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Resolve the delivery plugin once; depends_on guarantees it is loaded.
        self._send = get_dependency(self.pm, "cross_channel_send", CrossChannelSendPlugin)

    @hookimpl
    async def on_goals_state_changed(
        self,
        goal_id: int,
        kind: str,
        old_status: str,
        new_status: str,
        origin_channel: str | None,
        actor_channel: str,
        conclusion: str | None,
    ) -> None:
        """Tell the goal's origin channel about a committed state change."""
        # No origin, or the origin made the change itself: nobody to tell.
        if origin_channel is None or origin_channel == actor_channel:
            return
        # Build the message; each names old → new so a reordering shows.
        if kind == "retired":
            text = f"Goal #{goal_id} retired (was {old_status}). Conclusion: {conclusion}"
        else:
            text = f"Goal #{goal_id} changed status: {old_status} → {new_status}."
        assert self._send is not None  # set in on_start, before any turn
        result = await self._send.notify_other_channel(origin_channel, text)
        # notify_other_channel reports a refusal as text, never raises.
        if not result.startswith("Queued on "):
            logger.warning(
                "goals_notify: goal #%d change not sent to %s: %s",
                goal_id, origin_channel, result,
            )
