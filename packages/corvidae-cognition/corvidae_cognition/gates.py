"""GatePlugin — the decide gate's quiet veto and emission switches (PRD §3.7).

What reaches the channel, decided from origin and operator switches
(stage-1 appraisal and the salience floor it fed were removed, #1185):

  * ``gate.send.allow_progress`` / ``allow_thinking`` suppress those
    emissions wholesale.
  * The **quiet veto** (#790). An operator lists origins in
    ``gate.send.quiet_origins``, a per-channel tunable (#928: ``set_settings``
    can override it on one channel; ``agent.immutable_settings`` locks it);
    a ``final``, ``progress`` or ``thinking`` emission from a listed origin
    is withheld, and so is the turn's ``error`` fallback — an outage during a
    burst of quiet notifications is the same flood in error text (#936); the
    failure is logged by the kernel before this hook, so nothing is lost.
    ``user`` and ``reminder`` can never be quiet; an invalid value fails
    open. The plugin always offers a ``speak_up`` tool, whatever the list
    holds, so a hot reload that adds a quiet origin never leaves the veto
    without its hatch (tools are collected only at startup, #925): calling
    it marks the trace surfaced, which lifts the quiet veto for the trace's
    later emissions. With no quiet origin it lifts nothing.

This plugin withholds no final from an origin that is not quiet.

Origin is a plugin-side convention (R1): ``annotations["origin"]`` when a
producer stamped it, else ``user`` for a trace admitted through
``on_message_admitted``, else the origin ``exchange_log`` holds, else
``task`` (outcome_log's default for background exchanges).

Every veto is recorded into ``exchange_log.outcomes`` through
``upsert_exchange`` from a fire-and-forget task, so the gate path never
waits on the shared write connection. Every tunable resolves through
``resolve_tunable`` at decision time, so both runtime tuning surfaces
apply on the next exchange without a restart.
"""

import asyncio
import collections
import logging

from corvidae.attribution import get_attribution
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tool import Tool, ToolContext, tool_to_schema
from corvidae.tuning import resolve_tunable
from corvidae_cognition.config_keys import warn_removed_keys

logger = logging.getLogger("corvidae.gates")

# Origins that can never be quiet: the text was asked for. A user reply
# answers the person in front of us; a reminder is text the operator
# scheduled, and withholding it silently is the gate overruling an instruction.
_NEVER_WITHHELD_ORIGINS = ("user", "reminder")

SEND_ALLOW_PROGRESS_DEFAULT = True     # gate.send.allow_progress
SEND_ALLOW_THINKING_DEFAULT = True     # gate.send.allow_thinking

# Keys #1185 removed with stage-1 appraisal: each is warned by name at startup
# and reload, and otherwise ignored.
_REMOVED_KEYS = {
    "gate.engagement": "the engagement gate was removed with stage-1 appraisal (#1185)",
    "gate.send.min_salience": (
        "the decide gate's salience floor was removed with stage-1 appraisal "
        "(#1185); "
        "gate.send.quiet_origins still withholds a source's replies"
    ),
    "gate.send.enforce": (
        "the decide gate's salience floor was removed with stage-1 appraisal "
        "(#1185); "
        "gate.send.quiet_origins still withholds a source's replies"
    ),
}

# Bound on the admitted-trace set that defines "user" origin.
ADMITTED_LRU_MAXSIZE = 1024

# Emissions the quiet veto covers (#936): everything a quiet turn might make
# noise with. `error` belongs here because a model outage during a burst of
# quiet notifications otherwise sends one "model call failed" per turn, which is
# the flood the veto exists to stop, in error text. The failure itself is still
# logged at ERROR by the kernel before this gate runs.
# The final is not here: `_decide_final` owns it, because a quiet final records
# the richer `outcomes.send` row (#790).
_QUIET_VETOABLE = frozenset({"progress", "thinking", "error"})

# The per-emission switches, by emission name.
_EMISSION_SWITCHES = {
    "progress": ("gate.send.allow_progress", SEND_ALLOW_PROGRESS_DEFAULT),
    "thinking": ("gate.send.allow_thinking", SEND_ALLOW_THINKING_DEFAULT),
}


_TRUE_STRINGS = frozenset({"true", "yes", "on", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "off", "0"})


def _bool_tunable(channel, config: dict, key: str, default: bool) -> bool:
    """Resolve ``key`` as a bool. A value that isn't a bool or a boolean
    word falls back to ``default``, logged: ``bool("false")`` is True, and a
    set_settings call can store any JSON value."""
    value = resolve_tunable(channel, config, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE_STRINGS | _FALSE_STRINGS:
        return value.strip().lower() in _TRUE_STRINGS
    logger.warning(
        "gate tunable %s=%r is not a boolean (accepted: true/false); using the default %r",
        key, value, default, extra={"key": key, "channel": getattr(channel, "id", None)},
    )
    return default


class GatePlugin(CorvidaePlugin):
    """The decide gate: the quiet veto and the progress/thinking switches."""

    # Reads outcome_log and persistence fail-soft; neither is required.
    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.config: dict = {}
        # Trace ids admitted via on_message_admitted: the plugin-side
        # definition of "user" origin, bounded so it can't grow forever.
        self._admitted: collections.OrderedDict[str, None] = collections.OrderedDict()
        # Traces the agent surfaced through speak_up, bounded like _admitted.
        # Never discarded explicitly: a final is gated before the same
        # response's thinking, so the LRU bound evicts entries instead.
        self._surfaced: collections.OrderedDict[str, None] = collections.OrderedDict()
        # Strong refs to fire-and-forget outcome writes, so a write is never
        # GC-dropped mid-flight and on_stop can drain them.
        self._record_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.config = config
        # A key #1185 removed configures nothing: say so by name.
        warn_removed_keys(config, _REMOVED_KEYS)

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        # Tunables resolve at decision time; swapping the reference is all
        # reload needs, apart from reporting a removed key it brings back.
        self.config = config
        warn_removed_keys(config, _REMOVED_KEYS)

    @hookimpl
    async def on_stop(self) -> None:
        # Let in-flight outcome writes land rather than abandoning them.
        while self._record_tasks:
            await asyncio.gather(*set(self._record_tasks), return_exceptions=True)

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Offer speak_up whatever gate.send.quiet_origins holds.

        Tools are collected once at startup while the quiet list is re-read
        on every decision, so registering only for a non-empty list would
        leave a reload-added quiet origin with no hatch until restart (#925).
        """
        plugin = self

        async def speak_up(reason: str = "", _ctx: ToolContext | None = None) -> str:
            """Call this when a turn started by a quiet source (a webhook or other automated notification whose replies are normally withheld) has something that genuinely needs the human's attention. After it, your final reply of this turn is sent. reason: one short phrase on why."""
            # Without a channel there is nothing to record against.
            channel = _ctx.channel if _ctx is not None else None
            if channel is None:
                return (
                    "Error: no channel context available; speak_up did nothing, "
                    "so your reply stays withheld"
                )
            # Without a trace id the turn cannot be marked surfaced.
            trace_id = get_attribution().get("trace_id")
            if trace_id is None:
                logger.warning(
                    "gate: speak_up on %s has no trace id; the reply stays withheld",
                    channel.id, extra={"channel": channel.id},
                )
                return (
                    "Error: this turn has no trace id; speak_up did nothing, "
                    "so your reply stays withheld"
                )
            # Mark the trace surfaced (bounded LRU) and record the call.
            plugin._surfaced[trace_id] = None
            plugin._surfaced.move_to_end(trace_id)
            while len(plugin._surfaced) > ADMITTED_LRU_MAXSIZE:
                plugin._surfaced.popitem(last=False)
            plugin._record(trace_id, channel, {"speak_up": str(reason)[:200]})
            return (
                "Your final reply of this turn will be sent, subject to the "
                "usual checks."
            )

        tool_registry.append(Tool(name="speak_up", fn=speak_up, schema=tool_to_schema(speak_up)))

    @hookimpl
    async def on_message_admitted(self, channel, trace_id: str, sender: str, text: str) -> None:
        """Remember an admitted trace: this is what "user" origin means."""
        self._admitted[trace_id] = None
        self._admitted.move_to_end(trace_id)
        while len(self._admitted) > ADMITTED_LRU_MAXSIZE:
            self._admitted.popitem(last=False)

    # ------------------------------------------------------------------
    # The decide gate
    # ------------------------------------------------------------------

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations):
        """Decide whether an outbound emission reaches the channel.

        "progress"/"thinking": the allow switches. "final", "progress",
        "thinking" and "error" are withheld outright when their origin is a
        quiet one (#790, #936). Anything else is no opinion. Any unexpected
        error fails open: a broken gate never silences the agent.
        """
        try:
            # The quiet veto comes first. With a quiet set configured, the
            # origin is resolved for progress, thinking and the error
            # fallback alike.
            quiet = self._quiet_origins(channel)
            surfaced = trace_id is not None and trace_id in self._surfaced
            # The quiet veto covers every emission a quiet turn can make,
            # including its error fallback (#936). Logging happens in the kernel
            # before this hook, so withholding the message loses nothing.
            if emission in _QUIET_VETOABLE and quiet and not surfaced:
                origin = await self._origin(trace_id, annotations)
                if origin in quiet:
                    if trace_id is not None:
                        self._record(trace_id, channel, {
                            "suppressed": {emission: True},
                            "quiet_origin": origin,
                        })
                    return False
            if emission in _EMISSION_SWITCHES:
                key, default = _EMISSION_SWITCHES[emission]
                if _bool_tunable(channel, self.config, key, default):
                    return None
                # The vetoing plugin records the suppression; core only
                # marks the window.
                if trace_id is not None:
                    self._record(trace_id, channel, {"suppressed": {emission: True}})
                return False
            # A final is only ever withheld by the quiet veto: with nothing
            # quiet there is no decision to make, and no origin to look up.
            if emission != "final" or not quiet:
                return None
            origin = await self._origin(trace_id, annotations)
            return self._decide_final(channel, trace_id, origin, surfaced=surfaced, quiet=quiet)
        except Exception:
            logger.warning(
                "decide gate failed for the %s emission of %s on %s; sending it",
                emission, trace_id, channel.id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id, "emission": emission},
            )
            return None

    def _decide_final(
        self, channel, trace_id, origin, surfaced: bool, quiet: frozenset[str],
    ) -> bool | None:
        """The final-emission policy; returns False to withhold."""
        # Not a quiet origin: sent, with nothing to record.
        if origin not in quiet:
            return None
        # A surfaced quiet final is sent; say why it got through. The reason
        # overwrites a quiet_origin an earlier final of the trace merged in.
        if surfaced:
            if trace_id is not None:
                self._record(trace_id, channel, {"send": {
                    "emission": "final", "origin": origin, "vetoed": False,
                    "reason": "surfaced", "surfaced": True,
                }})
            return None
        # Quiet origin, not surfaced: withheld.
        if trace_id is not None:
            self._record(trace_id, channel, {"send": {
                "emission": "final", "origin": origin, "vetoed": True,
                "reason": "quiet_origin", "surfaced": False,
            }})
        logger.info(
            "decide gate withheld a quiet-origin final (%s) on %s",
            origin, channel.id,
            extra={"channel": channel.id, "trace_id": trace_id},
        )
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _quiet_origins(self, channel) -> frozenset[str]:
        """The validated ``gate.send.quiet_origins`` set for ``channel``:
        its runtime override, else config, else empty (#928). Invalid values
        fail open to empty; origins that are never withheld are dropped.
        Both are logged with the channel, since an override is per channel."""
        key = "gate.send.quiet_origins"
        value = resolve_tunable(channel, self.config, key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            logger.warning(
                "%s=%r on %s is not valid (accepted: a list of origin strings); "
                "no origin is quiet", key, value, channel.id,
                extra={"key": key, "channel": channel.id},
            )
            return frozenset()
        # Drop origins whose finals are never withheld, naming each.
        for origin in value:
            if origin in _NEVER_WITHHELD_ORIGINS:
                logger.warning(
                    "%s=%r on %s: ignoring %r, because %s finals are never "
                    "withheld (a direct reply or an operator-scheduled reminder)",
                    key, value, channel.id, origin, origin,
                    extra={"key": key, "channel": channel.id},
                )
        return frozenset(v for v in value if v not in _NEVER_WITHHELD_ORIGINS)

    async def _origin(self, trace_id, annotations) -> str:
        """The exchange's origin, read from propagated state, never inferred
        from text or source strings (trap #3).

        ``annotations["origin"]`` first, then the admitted-trace LRU. A trace the
        LRU has evicted (a long tool cycle under heavy traffic) falls back
        to the exchange_log row's origin, so an old user exchange is never
        mistaken for background traffic.
        """
        origin = (annotations or {}).get("origin")
        if origin:
            return origin
        if trace_id is None:
            return "task"
        if trace_id in self._admitted:
            return "user"
        recorded = await self._recorded_origin(trace_id)
        return recorded or "task"

    async def _recorded_origin(self, trace_id: str) -> str | None:
        """The origin exchange_log holds for ``trace_id``, or None."""
        persistence = self.pm.get_plugin("persistence")
        db = getattr(persistence, "db", None)
        if db is None or self.pm.get_plugin("outcome_log") is None:
            return None
        try:
            async with db.execute(
                "SELECT origin FROM exchange_log WHERE trace_id = ?", (trace_id,)
            ) as cursor:
                row = await cursor.fetchone()
        except Exception:
            logger.warning(
                "gate: could not read the recorded origin for %s; treating it as task",
                trace_id, exc_info=True, extra={"trace_id": trace_id},
            )
            return None
        return row[0] if row else None

    def _record(self, trace_id, channel, outcomes: dict) -> None:
        """Fire-and-forget merge of ``outcomes`` into the exchange row.

        No origin is passed: upsert_exchange patches origin onto an existing
        row, so a derived origin passed here could relabel it (2B review A-4).
        """
        outcome_log = self.pm.get_plugin("outcome_log")
        if outcome_log is None:
            return

        async def _write() -> None:
            try:
                await outcome_log.upsert_exchange(trace_id, channel.id, None, outcomes=outcomes)
            except Exception:
                logger.warning(
                    "gate outcome write failed for %s", trace_id, exc_info=True,
                    extra={"trace_id": trace_id, "channel": channel.id},
                )

        task = asyncio.create_task(_write())
        self._record_tasks.add(task)
        task.add_done_callback(self._record_tasks.discard)
