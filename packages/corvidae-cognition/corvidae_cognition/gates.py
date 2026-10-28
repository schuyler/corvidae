"""GatePlugin — the engagement and decide gates (WP2.9; PRD §3.7).

The agent's right to silence, decided from the appraisal vector:

  * The **engagement gate** (``should_process_message``) pulls the inbound
    stage-1 vector and rejects a message whose salience is below
    ``gate.engagement.threshold``, but only once ``gate.engagement.enforce``
    is set. Until then it runs in shadow mode: it computes and records the
    would-have-rejected verdict and passes everything. Engagement is never
    live-explored below threshold (a rejected message is user-visible); it
    calibrates offline from the recorded verdicts (trap #7).
  * The **decide gate** (``should_send_response``) pulls the outbound
    vector over the final response text. A direct reply to a user is never
    withheld by defaults; a final from any other origin (reminder,
    heartbeat, task, critique) is withheld when its salience is below
    ``gate.send.min_salience`` and ``gate.send.enforce`` is set — "adds
    nothing, stays silent". ``gate.send.allow_progress`` /
    ``allow_thinking`` suppress those emissions wholesale.

Both gates PULL their vectors through AppraisalPlugin's compute API
rather than relying on hook order: apluggy dispatches broadcast hooks
concurrently, and the pull API's per-(trace, direction) dedup makes the
compute run once whoever asks first. A missing AppraisalPlugin, or a
compute that fails, fails open: the gate never rejects or withholds on a
missing signal.

Origin is a plugin-side convention (R1): ``meta["origin"]`` when a
producer stamped it, else ``user`` for a trace admitted through
``on_message_admitted``, else ``task`` (outcome_log's default for
background exchanges).

Every decision is recorded into ``exchange_log.outcomes`` through
``upsert_exchange`` from a fire-and-forget task, so the gate path never
waits on the shared write connection. Every tunable resolves through
``resolve_tunable`` at decision time, so both runtime tuning surfaces
apply on the next exchange without a restart.
"""

import asyncio
import collections
import logging

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tuning import resolve_tunable

logger = logging.getLogger("corvidae.gates")

# Best-guess defaults (operator directive 1); the outcome log fits them
# over time.
ENGAGEMENT_ENFORCE_DEFAULT = False     # gate.engagement.enforce — shadow mode
ENGAGEMENT_THRESHOLD_DEFAULT = 0.2     # gate.engagement.threshold
SEND_ENFORCE_DEFAULT = True            # gate.send.enforce
SEND_MIN_SALIENCE_DEFAULT = 0.15       # gate.send.min_salience
SEND_ALLOW_PROGRESS_DEFAULT = True     # gate.send.allow_progress
SEND_ALLOW_THINKING_DEFAULT = True     # gate.send.allow_thinking

# Bound on the admitted-trace set that defines "user" origin.
ADMITTED_LRU_MAXSIZE = 1024

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
    if isinstance(value, str) and value.strip().lower() in _TRUE_STRINGS | _FALSE_STRINGS:
        return value.strip().lower() in _TRUE_STRINGS
    logger.warning(
        "gate tunable %s=%r is not a boolean (accepted: true/false); using the default %r",
        key, value, default, extra={"key": key, "channel": getattr(channel, "id", None)},
    )
    return default


def _float_tunable(channel, config: dict, key: str, default: float) -> float:
    """Resolve ``key`` as a number. A non-numeric value (a list, a word, a
    bool) falls back to ``default``, logged, so a bad override can never
    raise inside the gate."""
    value = resolve_tunable(channel, config, key, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    logger.warning(
        "gate tunable %s=%r is not a number (accepted: a number such as %r); using the default",
        key, value, default, extra={"key": key, "channel": getattr(channel, "id", None)},
    )
    return default


class GatePlugin(CorvidaePlugin):
    """The engagement (inbound) and decide (outbound) gates."""

    # Reads appraisal and outcome_log fail-soft; neither is required.
    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.config: dict = {}
        # Trace ids admitted via on_message_admitted: the plugin-side
        # definition of "user" origin, bounded so it can't grow forever.
        self._admitted: collections.OrderedDict[str, None] = collections.OrderedDict()
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

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        # Tunables resolve at decision time; swapping the reference is all
        # reload needs.
        self.config = config

    @hookimpl
    async def on_stop(self) -> None:
        # Let in-flight outcome writes land rather than abandoning them.
        while self._record_tasks:
            await asyncio.gather(*set(self._record_tasks), return_exceptions=True)

    @hookimpl
    async def on_message_admitted(self, channel, trace_id: str, sender: str, text: str) -> None:
        """Remember an admitted trace: this is what "user" origin means."""
        self._admitted[trace_id] = None
        self._admitted.move_to_end(trace_id)
        while len(self._admitted) > ADMITTED_LRU_MAXSIZE:
            self._admitted.popitem(last=False)

    # ------------------------------------------------------------------
    # The engagement gate
    # ------------------------------------------------------------------

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        """Shadow-first engagement gate over the inbound stage-1 vector.

        Returns False only when enforcement is on and salience is below
        the threshold; otherwise None. No AppraisalPlugin, or a failed
        compute, returns None with nothing recorded (unchanged behavior).
        Any unexpected error fails open: this broadcast runs on the
        transport read path, and a raise would drop the message.
        """
        try:
            return await self._engagement(channel, text, trace_id)
        except Exception:
            logger.warning(
                "engagement gate failed for %s on %s; admitting the message",
                trace_id, channel.id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return None

    async def _engagement(self, channel, text, trace_id) -> bool | None:
        """The engagement decision; returns False to reject."""
        if trace_id is None:
            return None
        appraisal = self.pm.get_plugin("appraisal")
        if appraisal is None:
            return None
        vector = await self._pull(
            appraisal.get_or_compute, channel, trace_id, text, direction="in",
        )
        salience = (vector or {}).get("salience")
        if salience is None:
            return None

        threshold = _float_tunable(
            channel, self.config, "gate.engagement.threshold", ENGAGEMENT_THRESHOLD_DEFAULT
        )
        enforce = _bool_tunable(
            channel, self.config, "gate.engagement.enforce", ENGAGEMENT_ENFORCE_DEFAULT
        )
        would_reject = salience < threshold
        # The row may not exist yet (the admission insert comes after this
        # hook), hence the upsert; origin is "user" by construction here.
        self._record(trace_id, channel, origin="user", outcomes={"engagement": {
            "salience": salience,
            "threshold": threshold,
            "would_reject": would_reject,
            "enforced": enforce,
        }})
        if enforce and would_reject:
            logger.info(
                "engagement gate rejected a message on %s (salience %.3f < %.3f)",
                channel.id, salience, threshold,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return False
        return None

    # ------------------------------------------------------------------
    # The decide gate
    # ------------------------------------------------------------------

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, meta):
        """Decide whether an outbound emission reaches the channel.

        "final": origin-conditional salience policy over the outbound
        vector. "progress"/"thinking": the allow switches. Anything else
        (including "error") is no opinion. Any unexpected error fails open:
        a broken gate never silences the agent.
        """
        try:
            if emission in _EMISSION_SWITCHES:
                key, default = _EMISSION_SWITCHES[emission]
                if _bool_tunable(channel, self.config, key, default):
                    return None
                # The vetoing plugin records the suppression; core only
                # marks the window.
                if trace_id is not None:
                    self._record(trace_id, channel, {"suppressed": {emission: True}})
                return False
            if emission != "final":
                return None
            origin = await self._origin(trace_id, meta)
            return await self._decide_final(channel, text, trace_id, origin)
        except Exception:
            logger.warning(
                "decide gate failed for the %s emission of %s on %s; sending it",
                emission, trace_id, channel.id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id, "emission": emission},
            )
            return None

    async def _decide_final(self, channel, text, trace_id, origin) -> bool | None:
        """The final-emission policy; returns False to withhold."""
        min_salience = _float_tunable(
            channel, self.config, "gate.send.min_salience", SEND_MIN_SALIENCE_DEFAULT
        )
        enforce = _bool_tunable(
            channel, self.config, "gate.send.enforce", SEND_ENFORCE_DEFAULT
        )
        record: dict = {
            "emission": "final",
            "origin": origin,
            "min_salience": min_salience,
            "enforced": enforce,
        }

        # Pull the outbound vector over the text that would be sent, plus
        # advisory context: the input-side stage 1 and the previous
        # exchange's stage 2 (the current one is unfinished by construction).
        appraisal = self.pm.get_plugin("appraisal")
        vector = None
        if appraisal is not None and trace_id is not None:
            vector = await self._pull(
                appraisal.get_or_compute_out, channel, trace_id, text, direction="out",
            )
            if origin != "user":
                # Calibration context for decisions that can bite; a user
                # final is never withheld, so it skips the extra reads.
                record["advisory"] = await self._advisory(appraisal, channel, trace_id)
        salience = (vector or {}).get("salience")

        if salience is None:
            # Fail open: never withhold on a missing signal, but record the
            # miss so calibration can see how often the gate ran blind. No
            # salience key: a null in a json_patch fragment deletes the key.
            record.update(vetoed=False, would_veto=False, reason="no_vector")
            verdict = None
        elif origin == "user":
            # A direct reply to a human is never swallowed by a guessed
            # threshold.
            record.update(salience=salience, vetoed=False, would_veto=False, reason="user_origin")
            verdict = None
        else:
            would_veto = salience < min_salience
            vetoed = would_veto and enforce
            record.update(
                salience=salience, vetoed=vetoed, would_veto=would_veto,
                reason="below_threshold" if would_veto else "at_or_above_threshold",
            )
            verdict = False if vetoed else None

        if trace_id is not None:
            self._record(trace_id, channel, {"send": record})
        if verdict is False:
            logger.info(
                "decide gate withheld a %s-origin final on %s (salience %.3f < %.3f)",
                origin, channel.id, salience, min_salience,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
        return verdict

    async def _advisory(self, appraisal, channel, trace_id) -> dict:
        """Input-side stage 1 and the previous stage 2, best effort."""
        advisory: dict = {}
        try:
            # Absent values are left out, not nulled: a null in a
            # json_patch fragment deletes the key.
            stage1 = await appraisal.get_appraisal(trace_id)
            if stage1 is not None and stage1.get("salience") is not None:
                advisory["input_salience"] = stage1["salience"]
            get_last = getattr(appraisal, "get_last_stage2", None)
            previous = get_last(channel.id) if callable(get_last) else None
            if previous is not None:
                advisory["previous_stage2"] = previous
        except Exception:
            logger.warning(
                "decide gate: advisory appraisal read failed for %s",
                trace_id, exc_info=True, extra={"trace_id": trace_id},
            )
        return advisory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _origin(self, trace_id, meta) -> str:
        """The exchange's origin, read from propagated state, never inferred
        from text or source strings (trap #3).

        ``meta["origin"]`` first, then the admitted-trace LRU. A trace the
        LRU has evicted (a long tool cycle under heavy traffic) falls back
        to the exchange_log row's origin, so an old user exchange is never
        mistaken for background traffic and withheld.
        """
        origin = (meta or {}).get("origin")
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

    async def _pull(self, compute, channel, trace_id, text, *, direction: str) -> dict | None:
        """Pull a vector through the appraisal compute API; None on failure.

        A CancelledError from a compute this task shares with another
        caller means the OWNER was cancelled (WP2.4's cross-task warning):
        that is a compute failure here, unless this task itself is being
        cancelled, which must propagate.
        """
        try:
            return await compute(channel, trace_id, text)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            logger.warning(
                "gate: %s appraisal compute was cancelled for %s; failing open",
                direction, trace_id, extra={"trace_id": trace_id, "direction": direction},
            )
        except Exception:
            logger.warning(
                "gate: %s appraisal compute failed for %s; failing open",
                direction, trace_id, exc_info=True,
                extra={"trace_id": trace_id, "direction": direction},
            )
        return None

    def _record(self, trace_id, channel, outcomes: dict, origin: str | None = None) -> None:
        """Fire-and-forget merge of ``outcomes`` into the exchange row.

        ``origin`` is passed only where it is certain (the inbound gate:
        "user"). upsert_exchange patches origin onto an existing row, so a
        derived origin passed here could relabel it (2B review A-4).
        """
        outcome_log = self.pm.get_plugin("outcome_log")
        if outcome_log is None:
            return

        async def _write() -> None:
            try:
                await outcome_log.upsert_exchange(trace_id, channel.id, origin, outcomes=outcomes)
            except Exception:
                logger.warning(
                    "gate outcome write failed for %s", trace_id, exc_info=True,
                    extra={"trace_id": trace_id, "channel": channel.id},
                )

        task = asyncio.create_task(_write())
        self._record_tasks.add(task)
        task.add_done_callback(self._record_tasks.discard)
