"""HeartbeatPlugin — a turn the agent starts on system idle (#134).

When no channel has had inbound or in-flight work for ``heartbeat.after``
seconds, the plugin sends one notification to ``internal:heartbeat``, a
channel no transport owns. The notification is the sections plugins add
through ``on_heartbeat_collect_prompts`` (#1084, #1094) -- the memory plugin's
recent memories and, when goals is loaded, its goals sections and pick rule --
followed by ``heartbeat.prompt``, last. The heartbeat itself adds none. The turn runs
the ordinary notification pipeline with the agent's normal tools, and acts only if the review warrants it
(PRD §3.7 Heartbeat). Nothing it writes reaches a person: that needs a
cross-channel send (PRD §3.5.9), which goes through its own gate.

The trigger is the kernel idle tracker's system watch, not a timer. A beat
is itself inbound work, so the system is busy until the beat's turn
settles and the next beat is due ``after`` seconds later: at most one beat
per ``after`` seconds of quiet, and none during traffic.

Config:
    heartbeat:
      after: 3600    # seconds of system-wide quiet; absent = asleep

The sections' own settings live with their plugins: ``memory.heartbeat_limit``,
``goals.heartbeat_limit`` and ``goals.heartbeat_instruction``
(``heartbeat.memory_limit`` and ``heartbeat.goals_instruction`` still work
for one release, with a warning; ``goals.heartbeat_limit`` never had a
``heartbeat.*`` spelling).

Sleep is durable because it lives in agent.yaml: without
``heartbeat.after`` the plugin registers nothing, at every start.

The beat channel's window grows beat by beat and is bounded by ordinary
compaction, like any other channel's.

Logging:
    - INFO: asleep at start, each beat sent
    - WARNING: a contributor raised (logged by the fail-soft hook dispatch;
      its section is skipped and the beat still goes out)
"""

from __future__ import annotations

import logging
import math

from corvidae.channel import ChannelRegistry
from corvidae.prompts import resolve_prompt
from corvidae.hooks import (
    CorvidaePlugin,
    get_dependency,
    hookimpl,
    hookspec,
)
from corvidae.idle import IdleTracker, IdleWatch

logger = logging.getLogger("corvidae.heartbeat")

#: The channel beats run on. No transport owns ``internal``, so replies go
#: to sessions.db and the logs only. A persistence key: don't change it.
HEARTBEAT_CHANNEL = "internal:heartbeat"

#: The beat's instruction. prompts/heartbeat.md is the documented copy.
DEFAULT_HEARTBEAT_PROMPT = (
    "This is your heartbeat. It becomes active when nothing else is "
    "occupying your attention. This is your opportunity to decide what to "
    "do next. Nobody reads this channel, and nothing you write here reaches "
    "anyone.\n\n"
    "Review the sections above, if there are any, and ask whether anything "
    "in them needs something from you now: a promise you made and haven't "
    "kept, something you said you would check, a mistake you should "
    "correct.\n\n"
    "If something does, act on it with your tools, then reply with one line "
    "saying what you did and why. If nothing does, say so in one line and "
    "stop. Don't invent work to look busy, and don't redo something you "
    "already did."
)


def _heartbeat_block(config: dict) -> dict:
    """Return the ``heartbeat`` config mapping ({} when absent).

    Raises ValueError, naming the key and the accepted shape, when the key
    holds something other than a mapping.
    """
    block = config.get("heartbeat") or {}
    # A scalar or list under heartbeat: can't hold keys, so fail loudly.
    if not isinstance(block, dict):
        raise ValueError(
            f"heartbeat: got {block!r}; expected a mapping such as "
            "'heartbeat: {after: 3600}' — check agent.yaml"
        )
    return block


def parse_after(config: dict) -> float | None:
    """Return ``heartbeat.after`` in seconds, or None when the heartbeat sleeps.

    Raises ValueError, naming the key, the value and what is accepted, for
    anything but a positive finite number: a zero period would mean
    back-to-back beats.
    """
    value = _heartbeat_block(config).get("after")
    if value is None:
        return None
    # bool is an int subclass; True is not a period. NaN would arm at zero
    # delay (back-to-back beats) and infinity would never beat.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"heartbeat.after: got {value!r}; expected a positive finite number of "
            "seconds — check agent.yaml, or remove the key to keep the "
            "heartbeat asleep"
        )
    return float(value)


class HeartbeatSpec:
    """Hooks the heartbeat plugin owns."""

    @hookspec
    async def on_heartbeat_collect_prompts(self, channel) -> str | None:
        """Contribute one section of the heartbeat's beat (#1084).

        Called by the beat on ``internal:heartbeat``. Return the section's
        text, or None to add nothing. Sections come in hook order
        (``tryfirst`` first, ``trylast`` last), separated by blank lines,
        and ``heartbeat.prompt`` follows them, last. A raising
        implementation is logged and skipped (observe hooks fail soft).
        """


class HeartbeatPlugin(CorvidaePlugin):
    """Starts a reviewing turn on internal:heartbeat after system-wide quiet.

    Attributes:
        _after: Seconds of quiet before a beat; None while asleep.
        _watch: The system idle watch, live from on_start to on_stop.
    """

    hookspecs = HeartbeatSpec
    depends_on = frozenset({"idle", "registry"})
    model_visible_annotations = frozenset({"origin"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._after: float | None = None
        self._idle: IdleTracker | None = None
        # The kernel ChannelRegistry, resolved in on_start.
        self._registry: ChannelRegistry | None = None
        self._watch: IdleWatch | None = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read heartbeat.after and the prompt override."""
        await super().on_init(pm, config)
        # Validated before the asleep return: a bad value fails startup either way.
        self._after = parse_after(config)
        if self._after is None:
            return
        # Prompt overrides (#411): file, inline, or built-in default.
        self._prompt = resolve_prompt(config, "heartbeat.prompt", DEFAULT_HEARTBEAT_PROMPT)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Register the system idle watch, unless asleep."""
        if self._after is None:
            logger.info("heartbeat: asleep (no heartbeat.after in config)")
            return
        self._idle = get_dependency(self.pm, "idle", IdleTracker)
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        self._watch = self._idle.watch_system(self._after, self._beat, name="heartbeat")

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the watch; a beat already enqueued drains with its queue."""
        if self._watch is not None:
            self._watch.cancel()
            self._watch = None

    async def _beat(self) -> None:
        """The system watch callback: gather the beat's sections, re-check
        quiet, notify.

        Never raises for a failed contributor: the fail-soft dispatch logs
        and skips it, and an exception escaping the callback would stop the
        tracker re-arming the watch.
        """
        transport, _, scope = HEARTBEAT_CHANNEL.partition(":")
        channel = self._registry.get_or_create(transport, scope)
        # Each plugin adds its section (None adds nothing); the beat is those
        # sections followed by the prompt, so the instruction comes last.
        contributions = await self.pm.ahook.on_heartbeat_collect_prompts(channel=channel)
        sections = [s for s in contributions if s is not None]
        # The gather awaited; a message may have arrived. Never beat while busy.
        if not self._idle.system_quiet():
            return
        logger.info(
            "heartbeat: beat", extra={"channel": channel.id, "sections": len(sections)}
        )
        text = "\n\n".join([*map(str, sections), self._prompt])
        await self.pm.ahook.on_notify(
            channel=channel,
            source="heartbeat",
            text=text,
            tool_call_id=None,
            annotations={"origin": "heartbeat"},
        )
