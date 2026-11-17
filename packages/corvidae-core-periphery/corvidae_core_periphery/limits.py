"""LimitsPlugin: bounds on waiting work and inbound messages (issue #1050).

``limits.max_waiting`` refuses new internal work (a subagent, a fork, a note
to another channel) once the accepted, unfinished work across all channels
reaches the ceiling; results of work already started are never refused.
``limits.inbound_per_channel`` rejects a channel's next message while that
many of its USER messages are still accepted and unfinished. With no
``limits`` block the plugin gives no verdicts.
"""

from __future__ import annotations

import logging

from corvidae.admission import WorkRefusal
from corvidae.agent import Agent
from corvidae.channel import Channel
from corvidae.clock import Clock
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger("corvidae.limits")

# Defaults for keys absent from a present ``limits`` block.
DEFAULTS: dict[str, int] = {
    "inbound_per_channel": 50,
    "notice_interval": 300,
    "max_waiting": 200,
}


def parse_limits(config: dict) -> dict[str, int] | None:
    """Validate the ``limits`` block; None when it is absent.

    Raises ValueError naming the key, the rejected value and the accepted
    values, ending with the file to check.
    """
    block = config.get("limits")
    if block is None:
        return None
    accepted = ", ".join(DEFAULTS)
    # `limits: 5` would otherwise silently mean "no limits".
    if not isinstance(block, dict):
        raise ValueError(
            f"Config error: limits is {block!r}; expected a mapping with "
            f"{accepted} — check agent.yaml"
        )
    settings = dict(DEFAULTS)
    for key, value in block.items():
        # A typo (or a key from a later phase) must not be ignored.
        if key not in DEFAULTS:
            raise ValueError(
                f"Config error: limits.{key} is not a known key; accepted keys "
                f"are {accepted} — check agent.yaml"
            )
        # bool is an int in Python; `true` is a typo, not a limit of one.
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Config error: limits.{key} is {value!r}; expected a positive "
                f"integer (a whole number, at least 1) — check agent.yaml"
            )
        settings[key] = value
    return settings


class LimitsPlugin(CorvidaePlugin):
    """Refuses work past ``limits.max_waiting``; rejects inbound past its cap."""

    depends_on = frozenset({"agent", "clock"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._settings: dict[str, int] | None = None
        self._agent: Agent | None = None
        self._clock: Clock | None = None
        # Channel id -> monotonic time of its last rejection warning.
        self._last_warned: dict[str, float] = {}

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        # Validate at startup so a bad block aborts rather than misleads.
        self.pm = pm
        self._settings = parse_limits(config)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Resolve the agent (depth reads) and the kernel clock (warning pacing).
        self._agent = get_dependency(self.pm, "agent", Agent)
        self._clock = get_dependency(self.pm, "clock", Clock)

    @hookimpl
    async def should_start_work(
        self,
        channel: Channel,
        kind: str,
        source_channel: str | None,
        trace_id: str | None,
        annotations: dict,
    ) -> WorkRefusal | None:
        # No block, no verdicts; no agent yet means on_start has not run.
        if self._settings is None or self._agent is None:
            return None
        total = sum(self._agent.waiting_work().values())
        ceiling = self._settings["max_waiting"]
        if total < ceiling:
            return None
        return WorkRefusal(
            "limits.max_waiting",
            f"{total} items of accepted work are waiting; wait for some to "
            "finish, or raise limits.max_waiting in agent.yaml",
        )

    # shortcut: concurrent arrivals on one channel can each pass this gate
    # before either is enqueued (Agent.on_message awaits between the gate and
    # enqueue), overshooting the cap by a few; upgrade to a per-channel
    # admission counter if the overshoot matters.
    @hookimpl
    async def should_process_message(
        self, channel: Channel, sender: str, text: str, trace_id: str | None
    ) -> bool | None:
        # No block, no verdicts.
        if self._settings is None or self._agent is None or self._clock is None:
            return None
        waiting = self._agent.waiting_messages().get(channel.id, 0)
        cap = self._settings["inbound_per_channel"]
        if waiting < cap:
            return None
        # Reject; warn at most once per channel per notice_interval.
        now = self._clock.monotonic()
        interval = self._settings["notice_interval"]
        last = self._last_warned.get(channel.id)
        if last is None or now - last >= interval:
            self._last_warned[channel.id] = now
            logger.warning(
                "limits: rejected a message from %s on %s: %d messages already "
                "waiting (limits.inbound_per_channel); further rejections on "
                "this channel are not logged for %ds",
                sender,
                channel.id,
                waiting,
                interval,
            )
        return False
