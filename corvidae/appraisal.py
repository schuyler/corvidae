"""AppraisalPlugin stage 1 — gate appraisal, FTS5 probe, direction-keyed store.

Cheap, always-on perception that gates expensive cognition (bootstrap-mapping
§3.2; plans/implementation/phase-2.md WP2.4). Stage 1 is surface heuristics
plus an FTS5 familiarity probe on a **dedicated read-only** SQLite connection
(WAL permits concurrent readers; borrowing the persistence connection would
queue the gate behind consolidation writes), under a hard latency budget,
failing open — no probe result within budget means the vector is built from
surface heuristics alone (trap #1). Stage 1 never blocks and never calls a
model.

The stage-1 vector is produced by a pull-based compute
(:meth:`AppraisalPlugin.get_or_compute`), NOT by cross-hook ordering: apluggy
dispatches broadcast async hooks concurrently via ``asyncio.gather``, so a
push model would race the gate consumer against this plugin's write. Callers
pull; the compute runs once per ``(exchange_key, direction)`` and caches.

Every parameter here is a commented best-guess constant, runtime-tunable
through both surfaces via ``corvidae.tuning.resolve_tunable`` (operator
directive 2, 2026-07-06).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import re

import aiosqlite

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tuning import resolve_tunable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Best-guess defaults (§6-tunable; every one resolves through resolve_tunable
# at decision time — never cached at init).
# --------------------------------------------------------------------------
PROBE_BUDGET_MS_DEFAULT = 50          # appraisal.probe.budget_ms
PROBE_MAX_TOKENS_DEFAULT = 12         # appraisal.probe.max_tokens
PROBE_RANK_SCALE_DEFAULT = 10.0       # appraisal.probe.rank_scale (bm25 ranks are negative)
NOVELTY_NO_PROBE_DEFAULT = 0.5        # appraisal.novelty.no_probe_default — a value, never a null
WEIGHT_NOVELTY_DEFAULT = 0.35         # appraisal.weights.novelty
WEIGHT_QUESTION_DEFAULT = 0.15        # appraisal.weights.question
WEIGHT_DISAGREEMENT_DEFAULT = 0.20    # appraisal.weights.disagreement
WEIGHT_COMMITMENT_DEFAULT = 0.20      # appraisal.weights.commitment
WEIGHT_IMPERATIVE_DEFAULT = 0.10      # appraisal.weights.imperative

# In-memory store bound — constant, not tunable (stage 1 runs before enqueue,
# outside SerialQueue serialization, so per-channel slots would race; §3.2).
CACHE_MAXSIZE = 512

# Surface-heuristic marker sets (tier 1c). Deliberately crude: these are
# density signals, not NLP — the §6 standing experiment is what tunes the
# blend, not the marker lists.
_NEGATION_WORDS = frozenset({
    "no", "not", "never", "none", "nothing", "neither", "nor", "cannot",
    "without", "nowhere", "nobody",
})
_DISAGREEMENT_WORDS = frozenset({
    "wrong", "incorrect", "disagree", "actually", "however", "but", "nope",
    "false", "mistaken", "untrue", "error", "misremember", "misremembered",
})
_IMPERATIVE_LEADS = frozenset({
    "please", "do", "don't", "stop", "go", "make", "remember", "write",
    "check", "run", "tell", "give", "let", "take", "try", "use", "call",
    "send", "get", "put", "keep", "find", "look", "wait", "help", "add",
    "remove", "fix", "update", "list", "show", "explain",
})
_COMMITMENT_WORDS = frozenset({
    "will", "must", "promise", "deadline", "commit", "shall", "guarantee",
    "tomorrow", "tonight", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "always", "definitely",
})

# Density → score scaling factors: how many marker hits per unit of text
# saturate the signal at 1.0. Best-guess constants (unit-tested at the
# boundaries, tuned by the §6 experiment if ever).
_NEGATION_SCALE = 4.0      # 25% negation tokens → 1.0
_DISAGREEMENT_SCALE = 5.0  # 20% disagreement tokens → 1.0
_COMMITMENT_SCALE = 5.0    # 20% commitment/number tokens → 1.0


def clamp01(x: float) -> float:
    """Clamp a float into [0.0, 1.0]."""
    return max(0.0, min(1.0, x))


def _sentences(text: str) -> list[str]:
    """Split text into sentence-ish segments, keeping their terminators."""
    return [s.strip() for s in re.findall(r"[^.?!\n]+[.?!]?", text) if s.strip()]


def surface_signals(text: str) -> dict:
    """Score the tier-1c surface heuristics for a message, 0–1 each.

    Pure function: no model, no I/O. Returns
    ``{negation, question, imperative, disagreement, commitment}``.
    Empty/whitespace text scores all zeros (no division by zero).
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    sentences = _sentences(text)
    if not words or not sentences:
        return {
            "negation": 0.0, "question": 0.0, "imperative": 0.0,
            "disagreement": 0.0, "commitment": 0.0,
        }

    n_words = len(words)
    n_sentences = len(sentences)

    # Negation density: negation words plus n't contractions, per token.
    negation_hits = sum(
        1 for w in words if w in _NEGATION_WORDS or w.endswith("n't")
    )
    negation = clamp01(_NEGATION_SCALE * negation_hits / n_words)

    # Question density: fraction of sentences that end with a question mark.
    question_sentences = sum(1 for s in sentences if s.endswith("?"))
    question = clamp01(question_sentences / n_sentences)

    # Imperative markers: fraction of sentences opening with an
    # imperative-ish lead word.
    def _lead(s: str) -> str:
        m = re.match(r"[a-z']+", s.lower())
        return m.group(0) if m else ""

    imperative_sentences = sum(1 for s in sentences if _lead(s) in _IMPERATIVE_LEADS)
    imperative = clamp01(imperative_sentences / n_sentences)

    # Disagreement markers per token.
    disagreement_hits = sum(1 for w in words if w in _DISAGREEMENT_WORDS)
    disagreement = clamp01(_DISAGREEMENT_SCALE * disagreement_hits / n_words)

    # Numbers/commitment density: numeric tokens plus commitment words.
    commitment_hits = sum(
        1 for w in words if w in _COMMITMENT_WORDS or any(c.isdigit() for c in w)
    )
    commitment = clamp01(_COMMITMENT_SCALE * commitment_hits / n_words)

    return {
        "negation": negation,
        "question": question,
        "imperative": imperative,
        "disagreement": disagreement,
        "commitment": commitment,
    }


class _LRUDict(collections.OrderedDict):
    """A bounded dict that evicts its least-recently-used entry."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._maxsize:
            self.popitem(last=False)


class AppraisalPlugin(CorvidaePlugin):
    """Stage-1 appraisal: heuristics + FTS5 probe behind a pull API.

    Soft-uses the memory FTS surface (via its own read-only connection) and
    the outcome-log writer; fail-soft when either is absent.
    """

    depends_on = frozenset({"persistence"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.config: dict = {}
        # Both maps keyed (exchange_key, direction); direction ∈ {"in","out"}.
        # get_or_compute_out (WP2.9) will share this discipline — the two
        # directions compute over different text and must never alias.
        self._cache: _LRUDict = _LRUDict(maxsize=CACHE_MAXSIZE)
        self._inflight: dict[tuple[str, str], asyncio.Future] = {}
        # Strong refs to fire-and-forget persist tasks — prevents a
        # GC-dropped write mid-flight (trap #10).
        self._persist_tasks: set[asyncio.Task] = set()
        self._probe_db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self.config = config

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        # Swap the config reference; every tunable resolves through
        # resolve_tunable at decision time, so this is all reload needs.
        self.config = config

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Open the dedicated read-only probe connection.

        The DB path comes from the same config key PersistencePlugin reads.
        Missing DB or missing memory_fts degrades to no-probe with one
        WARNING — the vector is then built from surface heuristics alone.
        """
        db_path = config.get("daemon", {}).get("session_db", "sessions.db")
        try:
            self._probe_db = await aiosqlite.connect(
                f"file:{db_path}?mode=ro", uri=True
            )
            # Verify the FTS table is queryable up front so per-message
            # probes don't discover the miss one WARNING at a time.
            async with self._probe_db.execute(
                "SELECT name FROM sqlite_master WHERE name = 'memory_fts'"
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                await self._probe_db.close()
                self._probe_db = None
                logger.warning(
                    "appraisal probe disabled: memory_fts missing in %s", db_path
                )
        except Exception:
            self._probe_db = None
            logger.warning(
                "appraisal probe disabled: cannot open %s read-only",
                db_path, exc_info=True,
            )

    @hookimpl
    async def on_stop(self) -> None:
        if self._probe_db is not None:
            await self._probe_db.close()
            self._probe_db = None
        # Let in-flight persists finish rather than abandoning writes.
        if self._persist_tasks:
            await asyncio.gather(*self._persist_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # The thin gate trigger (computes; never decides)
    # ------------------------------------------------------------------

    @hookimpl
    async def should_process_message(self, channel, sender, text, exchange_key):
        """Thin trigger: compute the inbound stage-1 vector, return None.

        The try/except is load-bearing, not defensive dressing: this is a
        plain broadcast hook, dispatched concurrently via asyncio.gather
        WITHOUT return_exceptions — a raw exception here would propagate to
        the transport read path, discard sibling impls' results, and
        prevent REJECT_WINS from resolving. An appraisal failure never
        rejects or crashes the inbound path (traps #1/#10).
        """
        if exchange_key is None:
            return None
        try:
            await self.get_or_compute(channel, exchange_key, text)
        except Exception:
            logger.warning(
                "stage-1 appraisal failed at the gate (failing open)",
                exc_info=True,
                extra={"channel": channel.id, "exchange_key": exchange_key},
            )
        return None

    # ------------------------------------------------------------------
    # Pull API
    # ------------------------------------------------------------------

    async def get_or_compute(self, channel, exchange_key: str, text: str) -> dict:
        """Return the INBOUND stage-1 vector for this exchange, computing once.

        Idempotent and concurrency-safe: concurrent callers for the same key
        await a single shared in-flight future, so the probe runs exactly
        once per (exchange_key, direction) regardless of how many hookimpls
        request it or what order they fire in. On first compute,
        fire-and-forget persists probe_score + the vector under the
        appraisal envelope's "stage1" key.

        Returned vectors are the SAME dict object that lives in the cache —
        treat them as immutable; copy on write if mutation is ever needed.
        """
        return await self._get_or_compute(channel, exchange_key, text, "in")

    async def get_appraisal(self, exchange_key: str) -> dict | None:
        """Pure reader for the inbound stage-1 vector (no compute)."""
        return await self._read_vector(exchange_key, "in", "stage1")

    async def get_appraisal_out(self, exchange_key: str) -> dict | None:
        """Pure reader for the outbound stage-1 vector (no compute)."""
        return await self._read_vector(exchange_key, "out", "stage1_out")

    async def get_stage2(self, exchange_key: str) -> dict | None:
        """Pure reader for the stage-2 vector (WP2.5 writes it)."""
        return await self._read_vector(exchange_key, None, "stage2")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_or_compute(self, channel, exchange_key: str, text: str, direction: str) -> dict:
        k = (exchange_key, direction)
        if k in self._cache:                    # 1. hit — return before any await
            return self._cache[k]
        if (fut := self._inflight.get(k)):      # 2. compute in progress: share it.
            # SHIELD is load-bearing: a cancelled waiter cancels only its
            # own wrapper, never the shared future; siblings are unharmed.
            return await asyncio.shield(fut)
        fut = asyncio.get_running_loop().create_future()
        # 3. Registered BEFORE the first await — THIS is the dedup:
        # concurrent callers for k now take branch 2; the probe runs once
        # per (key, direction).
        self._inflight[k] = fut
        try:
            vector, probe_score = await self._compute(channel, text, direction)
        except BaseException as exc:
            # 4. EVICT, never cache a failure — the next request retries.
            # BaseException, not Exception: OWNER cancellation
            # (CancelledError) must also evict and wake waiters, or they
            # hang forever on an abandoned future. The fut.exception() call
            # marks it retrieved (no GC-time warning). Fail-open is the
            # CALLERS' job (thin hookimpls, gates — traps #1/#10), never
            # this method's.
            del self._inflight[k]
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()
            raise
        # 5. Cache, clear in-flight, THEN wake waiters — a waking waiter
        # observes a populated cache. done() guard: a cancelled future must
        # never raise InvalidStateError into the owner's return path.
        self._cache[k] = vector
        del self._inflight[k]
        if not fut.done():
            fut.set_result(vector)
        # 6. Fire-and-forget persist — the gate path never awaits it; the
        # strong ref prevents a GC-dropped write.
        t = asyncio.create_task(
            self._persist_stage1(channel, exchange_key, direction, vector, probe_score)
        )
        self._persist_tasks.add(t)
        t.add_done_callback(self._persist_tasks.discard)
        return vector

    async def _compute(self, channel, text: str, direction: str) -> tuple[dict, float | None]:
        """Blend surface heuristics with the probe into the stage-1 vector.

        Applies the probe timeout internally and degrades to probe-less
        blending — a probe timeout is NOT a failure and never evicts the
        in-flight future (trap #1); only unexpected exceptions propagate.

        Returns (vector, probe_score) where probe_score is the familiarity
        the probe produced, or None when no probe result arrived in budget.
        """
        signals = surface_signals(text)
        cfg = self.config

        familiarity: float | None = None
        if self._probe_db is not None:
            budget_ms = resolve_tunable(
                channel, cfg, "appraisal.probe.budget_ms", PROBE_BUDGET_MS_DEFAULT
            )
            try:
                familiarity = await asyncio.wait_for(
                    self._probe_query(channel, text), timeout=budget_ms / 1000
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.debug("appraisal probe timed out; failing open")
            except asyncio.CancelledError:
                raise  # owner cancellation is not a probe degradation
            except Exception:
                logger.warning("appraisal probe failed; failing open", exc_info=True)

        # Signal→vector mapping (pinned in the plan's Stage-1 constants):
        # probe absent → no_probe_default, a value, never a null.
        if familiarity is not None:
            novelty = clamp01(1.0 - familiarity)
        else:
            novelty = resolve_tunable(
                channel, cfg, "appraisal.novelty.no_probe_default", NOVELTY_NO_PROBE_DEFAULT
            )
        question = signals["question"]
        disagreement = max(signals["disagreement"], signals["negation"])
        commitment = signals["commitment"]
        imperative = signals["imperative"]  # feeds salience only

        w_nov = resolve_tunable(channel, cfg, "appraisal.weights.novelty", WEIGHT_NOVELTY_DEFAULT)
        w_q = resolve_tunable(channel, cfg, "appraisal.weights.question", WEIGHT_QUESTION_DEFAULT)
        w_dis = resolve_tunable(channel, cfg, "appraisal.weights.disagreement", WEIGHT_DISAGREEMENT_DEFAULT)
        w_com = resolve_tunable(channel, cfg, "appraisal.weights.commitment", WEIGHT_COMMITMENT_DEFAULT)
        w_imp = resolve_tunable(channel, cfg, "appraisal.weights.imperative", WEIGHT_IMPERATIVE_DEFAULT)

        salience = clamp01(
            w_nov * novelty
            + w_q * question
            + w_dis * disagreement
            + w_com * commitment
            + w_imp * imperative
        )
        vector = {
            "novelty": novelty,
            "commitment_density": commitment,
            "disagreement": disagreement,
            "question": question,
            "salience": salience,
        }
        return vector, familiarity

    async def _probe_query(self, channel, text: str) -> float | None:
        """Run the FTS5 familiarity probe. Returns familiarity or None.

        Raw user text in MATCH is a syntax-error generator — each token is
        quoted and OR-joined, capped at appraisal.probe.max_tokens. The
        probe is FTS5-only by design: sqlite-vec is brute-force exact KNN,
        so a "coarse vector probe" would cost the same as full retrieval.
        """
        if self._probe_db is None:
            return None
        max_tokens = resolve_tunable(
            channel, self.config, "appraisal.probe.max_tokens", PROBE_MAX_TOKENS_DEFAULT
        )
        tokens = re.findall(r"\w+", text)[:max_tokens]
        if not tokens:
            return None
        match_expr = " OR ".join(f'"{t}"' for t in tokens)
        async with self._probe_db.execute(
            "SELECT rank FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT 3",
            (match_expr,),
        ) as cursor:
            rows = await cursor.fetchall()
        # Familiarity = bounded transform of top bm25 rank and hit count.
        # bm25 ranks are negative; more-negative = stronger match.
        if not rows:
            return 0.0
        hits = len(rows)
        top_rank = rows[0][0]
        rank_scale = resolve_tunable(
            channel, self.config, "appraisal.probe.rank_scale", PROBE_RANK_SCALE_DEFAULT
        )
        return clamp01((min(hits, 3) / 3) * min(1.0, -top_rank / rank_scale))

    async def _persist_stage1(
        self, channel, exchange_key: str, direction: str, vector: dict,
        probe_score: float | None,
    ) -> None:
        """Fire-and-forget stage-1 persist via the atomic-merge upsert.

        At gate time the on_message_admitted/rejected insert has not fired
        yet, and a plain update_exchange would silently no-op on the
        missing row — hence the upsert. Catches and LOGS its own
        exceptions: a persist failure must never propagate anywhere near
        the gate path, but it must be visible in the log (trap #10).
        """
        try:
            outcome_log = self.pm.get_plugin("outcome_log")
            if outcome_log is None:
                logger.debug("stage-1 persist skipped: outcome_log not registered")
                return
            envelope_key = "stage1" if direction == "in" else "stage1_out"
            columns: dict = {"appraisal": {envelope_key: vector}}
            if probe_score is not None:
                columns["probe_score"] = probe_score
            await outcome_log.upsert_exchange(
                exchange_key, channel.id, "user", **columns
            )
        except Exception:
            logger.warning(
                "stage-1 appraisal persist failed",
                exc_info=True,
                extra={"exchange_key": exchange_key, "direction": direction},
            )

    async def _read_vector(
        self, exchange_key: str, direction: str | None, envelope_key: str
    ) -> dict | None:
        """Read a vector: in-memory store first, then the appraisal envelope
        in exchange_log. Returns None when neither has it."""
        if direction is not None:
            cached = self._cache.get((exchange_key, direction))
            if cached is not None:
                return cached
        db = self._probe_db
        if db is None:
            # Fall back to the persistence connection for the read; the
            # pure readers are not latency-critical.
            persistence = self.pm.get_plugin("persistence") if hasattr(self, "pm") else None
            db = getattr(persistence, "db", None)
            if db is None:
                return None
        try:
            async with db.execute(
                "SELECT appraisal FROM exchange_log WHERE exchange_key = ?",
                (exchange_key,),
            ) as cursor:
                row = await cursor.fetchone()
        except Exception:
            # Table may not exist yet (no exchange recorded) — not an error.
            return None
        if row is None or row[0] is None:
            return None
        try:
            return json.loads(row[0]).get(envelope_key)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "malformed appraisal envelope for exchange %s", exchange_key
            )
            return None
