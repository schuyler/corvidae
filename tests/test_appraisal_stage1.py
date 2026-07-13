"""RED tests for WP2.4 — AppraisalPlugin stage 1: gate appraisal, FTS5 probe, store.

Per plans/implementation/phase-2.md WP2.4:
- surface heuristics (module-level pure functions) at boundaries;
- FTS5 probe on a dedicated read-only connection, sanitized MATCH, hard
  latency budget, fail-open;
- the direction-keyed pull API get_or_compute: idempotent, concurrency-safe
  (single shared in-flight future per (exchange_key, direction)),
  evict-on-failure, fire-and-forget stage-1 persist via upsert_exchange;
- the thin should_process_message trigger: load-bearing try/except, always
  returns None (this plugin computes; the gate plugin decides).
"""

import asyncio
import json
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import create_plugin_manager
from corvidae.persistence import PersistencePlugin, init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(scope="appraisal") -> Channel:
    return Channel(transport="test", scope=scope, config=ChannelConfig())


async def _seed_probe_db(path, summaries):
    """Create a session DB file at path with a content-full memory_fts table
    seeded with the given summaries. The probe only ever queries memory_fts,
    so a standalone FTS5 table (same name/column as production's
    external-content one) is sufficient and keeps the fixture focused."""
    db = await aiosqlite.connect(path)
    await db.execute("CREATE VIRTUAL TABLE memory_fts USING fts5(summary)")
    for s in summaries:
        await db.execute("INSERT INTO memory_fts(summary) VALUES (?)", (s,))
    await db.commit()
    await db.close()


async def build_appraisal(tmp_path, summaries=(), config=None, with_outcome_log=False):
    """Register an AppraisalPlugin against a file-backed session DB whose
    memory_fts is seeded with `summaries`.

    Returns (plugin, channel, db_path, pm). When with_outcome_log is True, a
    real OutcomeLogPlugin + PersistencePlugin pair is registered against the
    SAME file DB so upsert_exchange writes land somewhere observable.
    """
    from corvidae.appraisal import AppraisalPlugin

    db_path = str(tmp_path / "sessions.db")
    await _seed_probe_db(db_path, summaries)

    pm = create_plugin_manager()

    if with_outcome_log:
        from corvidae.outcome_log import OutcomeLogPlugin

        db = await aiosqlite.connect(db_path)
        await init_db(db)
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        outcome = OutcomeLogPlugin()
        pm.register(outcome, name="outcome_log")
        await outcome.on_init(pm=pm, config={})
        await outcome.on_start(config={})

    plugin = AppraisalPlugin()
    pm.register(plugin, name="appraisal")
    cfg = config or {}
    cfg.setdefault("daemon", {})["session_db"] = db_path
    await plugin.on_init(pm=pm, config=cfg)
    await plugin.on_start(config=cfg)

    channel = _make_channel()
    return plugin, channel, db_path, pm


async def _teardown(plugin, pm):
    """Stop the appraisal plugin and close any test-owned persistence DB."""
    await plugin.on_stop()
    persistence = pm.get_plugin("persistence")
    if persistence is not None and persistence.db is not None:
        await persistence.db.close()


# ---------------------------------------------------------------------------
# 1. Surface heuristics at boundaries (pure functions, no I/O)
# ---------------------------------------------------------------------------


class TestSurfaceSignals:
    def test_empty_text_scores_all_zero(self):
        from corvidae.appraisal import surface_signals

        signals = surface_signals("")
        assert set(signals) == {
            "negation", "question", "imperative", "disagreement", "commitment",
        }
        assert all(v == 0.0 for v in signals.values())

    def test_all_questions_scores_question_one(self):
        from corvidae.appraisal import surface_signals

        signals = surface_signals("Where were you? What happened? Who said that?")
        assert signals["question"] == 1.0

    def test_plain_statement_scores_question_zero(self):
        from corvidae.appraisal import surface_signals

        signals = surface_signals("The sky was clear this morning.")
        assert signals["question"] == 0.0

    def test_dense_negation_scores_high(self):
        from corvidae.appraisal import surface_signals

        dense = "No, that is not right. I never said that. Nothing about this is true."
        sparse = "The meeting went well and everyone agreed on the plan."
        assert surface_signals(dense)["negation"] > 0.5
        assert surface_signals(sparse)["negation"] == 0.0

    def test_scores_are_clamped_to_unit_interval(self):
        from corvidae.appraisal import surface_signals

        pathological = "no not never none nothing neither nor " * 20 + "?"
        signals = surface_signals(pathological)
        for name, value in signals.items():
            assert 0.0 <= value <= 1.0, f"{name} escaped [0,1]: {value}"


# ---------------------------------------------------------------------------
# 2. FTS5 probe: familiarity, hostile input, fail-open timeout
# ---------------------------------------------------------------------------


class TestProbe:
    async def test_familiar_text_high_familiarity_low_novelty(self, tmp_path):
        plugin, channel, _, pm = await build_appraisal(
            tmp_path,
            summaries=[
                "I helped Schuyler debug the corvidae funnel budget yesterday",
                "We discussed the corvidae funnel design at length",
            ],
        )
        familiar = await plugin.get_or_compute(
            channel, "k-fam", "tell me about the corvidae funnel budget"
        )
        unseen = await plugin.get_or_compute(
            channel, "k-new", "quantum harpsichord marmalade festival"
        )
        assert familiar["novelty"] < unseen["novelty"]
        # Unseen text has zero hits → familiarity 0.0 → novelty 1.0.
        assert unseen["novelty"] == 1.0
        await _teardown(plugin, pm)

    async def test_hostile_fts5_operators_do_not_raise(self, tmp_path):
        plugin, channel, _, pm = await build_appraisal(
            tmp_path, summaries=["an innocuous memory"]
        )
        vector = await plugin.get_or_compute(
            channel, "k-hostile", 'ignore this "AND ( NEAR" OR NOT (*)'
        )
        assert 0.0 <= vector["novelty"] <= 1.0
        assert 0.0 <= vector["salience"] <= 1.0
        await _teardown(plugin, pm)

    async def test_probe_timeout_fails_open_with_default_novelty(self, tmp_path, monkeypatch):
        """A probe slower than the budget must not block the gate: the
        vector still arrives, novelty at the no-probe default."""
        plugin, channel, _, pm = await build_appraisal(
            tmp_path,
            summaries=["some memory"],
            config={"appraisal": {"probe": {"budget_ms": 20}}},
        )

        async def slow_probe(*args, **kwargs):
            await asyncio.sleep(5.0)

        monkeypatch.setattr(plugin, "_probe_query", slow_probe)
        vector = await asyncio.wait_for(
            plugin.get_or_compute(channel, "k-slow", "anything at all"),
            timeout=2.0,
        )
        assert vector["novelty"] == 0.5  # appraisal.novelty.no_probe_default
        await _teardown(plugin, pm)

    async def test_missing_db_degrades_to_no_probe(self, tmp_path):
        """No session DB on disk → plugin starts, warns once, vectors use
        the no-probe novelty default (fail-open, trap #1)."""
        from corvidae.appraisal import AppraisalPlugin

        pm = create_plugin_manager()
        plugin = AppraisalPlugin()
        pm.register(plugin, name="appraisal")
        cfg = {"daemon": {"session_db": str(tmp_path / "does-not-exist.db")}}
        await plugin.on_init(pm=pm, config=cfg)
        await plugin.on_start(config=cfg)

        channel = _make_channel()
        vector = await plugin.get_or_compute(channel, "k-nodb", "hello there")
        assert vector["novelty"] == 0.5
        await plugin.on_stop()


# ---------------------------------------------------------------------------
# 3. Direction-keyed store: distinct keys, single probe under concurrency
# ---------------------------------------------------------------------------


class TestPullApi:
    async def test_racing_messages_keep_distinct_vectors_under_distinct_keys(self, tmp_path):
        plugin, channel, _, pm = await build_appraisal(
            tmp_path, summaries=["the corvidae funnel budget design"]
        )
        v1, v2 = await asyncio.gather(
            plugin.get_or_compute(channel, "k-a", "corvidae funnel budget?"),
            plugin.get_or_compute(channel, "k-b", "zebra xylophone unrelated"),
        )
        assert v1 is not v2
        assert v1["novelty"] != v2["novelty"] or v1["question"] != v2["question"]
        # Each key reads back its own vector.
        assert await plugin.get_appraisal("k-a") == v1
        assert await plugin.get_appraisal("k-b") == v2
        await _teardown(plugin, pm)

    async def test_concurrent_callers_share_one_probe(self, tmp_path, monkeypatch):
        """The dedup guarantee: N concurrent callers for one key run the
        probe exactly once (spy on the probe query)."""
        plugin, channel, _, pm = await build_appraisal(
            tmp_path, summaries=["seeded memory text"]
        )
        calls = 0
        original = plugin._probe_query

        async def counting_probe(*args, **kwargs):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)  # widen the race window
            return await original(*args, **kwargs)

        monkeypatch.setattr(plugin, "_probe_query", counting_probe)
        results = await asyncio.gather(
            *[plugin.get_or_compute(channel, "k-one", "seeded memory text") for _ in range(5)]
        )
        assert calls == 1
        assert all(r == results[0] for r in results)
        await _teardown(plugin, pm)

    async def test_ordering_independence_consumer_before_and_after(self, tmp_path):
        """A consumer's should_process_message pulling get_or_compute
        observes the SAME vector whether registered before or after
        AppraisalPlugin, including under one concurrent gather broadcast."""
        from corvidae.appraisal import AppraisalPlugin
        from corvidae.hooks import CorvidaePlugin, hookimpl

        observed = {}

        class Consumer(CorvidaePlugin):
            depends_on = frozenset()

            def __init__(self, appraisal):
                self._appraisal = appraisal

            @hookimpl
            async def should_process_message(self, channel, sender, text, exchange_key):
                observed[exchange_key] = await self._appraisal.get_or_compute(
                    channel, exchange_key, text
                )
                return None

        for consumer_first in (True, False):
            db_path = str(tmp_path / f"order-{consumer_first}.db")
            await _seed_probe_db(db_path, ["a seeded memory"])
            pm = create_plugin_manager()
            plugin = AppraisalPlugin()
            consumer = Consumer(plugin)
            if consumer_first:
                pm.register(consumer, name="consumer")
                pm.register(plugin, name="appraisal")
            else:
                pm.register(plugin, name="appraisal")
                pm.register(consumer, name="consumer")
            cfg = {"daemon": {"session_db": db_path}}
            await plugin.on_init(pm=pm, config=cfg)
            await plugin.on_start(config=cfg)
            consumer.pm = pm

            channel = _make_channel(scope=f"order-{consumer_first}")
            key = f"k-order-{consumer_first}"
            # Concurrent broadcast: both hookimpls fire in one gather.
            await pm.ahook.should_process_message(
                channel=channel, sender="user", text="a seeded memory", exchange_key=key
            )
            assert observed[key] == await plugin.get_appraisal(key)
            await plugin.on_stop()

    async def test_returned_vector_has_pinned_keys(self, tmp_path):
        plugin, channel, _, pm = await build_appraisal(tmp_path)
        vector = await plugin.get_or_compute(channel, "k-shape", "will you commit to 3 items by Monday?")
        assert set(vector) == {
            "novelty", "commitment_density", "disagreement", "question", "salience",
        }
        assert all(0.0 <= v <= 1.0 for v in vector.values())
        await _teardown(plugin, pm)


# ---------------------------------------------------------------------------
# 4. Stage-1 persist: upsert lands in exchange_log in either interleaving
# ---------------------------------------------------------------------------


class TestStage1Persist:
    async def _read_row(self, db_path, key):
        db = await aiosqlite.connect(db_path)
        async with db.execute(
            "SELECT appraisal, outcomes, probe_score FROM exchange_log WHERE exchange_key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
        await db.close()
        return row

    async def _drain_persists(self, plugin):
        """Await the fire-and-forget persist tasks the plugin spawned."""
        for _ in range(20):
            pending = set(getattr(plugin, "_persist_tasks", set()))
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)

    async def test_rejected_message_keeps_stage1_row_persist_first(self, tmp_path):
        """Persist lands BEFORE on_message_rejected's insert — the row ends
        with both the stage-1 vector and the rejection outcome."""
        plugin, channel, db_path, pm = await build_appraisal(
            tmp_path, summaries=["something"], with_outcome_log=True
        )
        key = "k-rej-1"
        vector = await plugin.get_or_compute(channel, key, "a rejected message")
        await self._drain_persists(plugin)
        await pm.ahook.on_message_rejected(
            channel=channel, exchange_key=key, sender="user", text="a rejected message"
        )
        row = await self._read_row(db_path, key)
        assert row is not None
        assert json.loads(row[0])["stage1"] == vector
        assert json.loads(row[1]) == {"gate": "rejected"}
        await _teardown(plugin, pm)

    async def test_rejected_message_keeps_stage1_row_persist_second(self, tmp_path):
        """Persist lands AFTER the rejection insert — same converged row
        (write-order independence)."""
        plugin, channel, db_path, pm = await build_appraisal(
            tmp_path, summaries=["something"], with_outcome_log=True
        )
        key = "k-rej-2"
        await pm.ahook.on_message_rejected(
            channel=channel, exchange_key=key, sender="user", text="a rejected message"
        )
        vector = await plugin.get_or_compute(channel, key, "a rejected message")
        await self._drain_persists(plugin)
        row = await self._read_row(db_path, key)
        assert row is not None
        assert json.loads(row[0])["stage1"] == vector
        assert json.loads(row[1]) == {"gate": "rejected"}
        await _teardown(plugin, pm)

    async def test_probe_score_persisted_when_probe_ran(self, tmp_path):
        plugin, channel, db_path, pm = await build_appraisal(
            tmp_path, summaries=["the corvidae funnel budget"], with_outcome_log=True
        )
        key = "k-probe-score"
        await plugin.get_or_compute(channel, key, "corvidae funnel budget")
        await self._drain_persists(plugin)
        row = await self._read_row(db_path, key)
        assert row is not None
        assert row[2] is not None  # probe_score
        assert 0.0 <= row[2] <= 1.0
        await _teardown(plugin, pm)


# ---------------------------------------------------------------------------
# 5. The thin trigger: fail-open, eviction-not-poisoning
# ---------------------------------------------------------------------------


class TestThinTrigger:
    async def test_gate_hook_computes_and_returns_none(self, tmp_path):
        plugin, channel, _, pm = await build_appraisal(tmp_path, summaries=["x"])
        result = await plugin.should_process_message(
            channel=channel, sender="user", text="hello", exchange_key="k-thin"
        )
        assert result is None
        assert await plugin.get_appraisal("k-thin") is not None
        await _teardown(plugin, pm)

    async def test_compute_failure_fails_open_and_recomputes(self, tmp_path, monkeypatch):
        """Blend raising → the thin trigger returns None (no exception to
        the transport read path); the failed in-flight future is evicted so
        a subsequent call recomputes successfully."""
        plugin, channel, _, pm = await build_appraisal(tmp_path, summaries=["x"])

        real_compute = plugin._compute
        boom = {"armed": True}

        async def failing_compute(*args, **kwargs):
            if boom["armed"]:
                raise RuntimeError("compute exploded")
            return await real_compute(*args, **kwargs)

        monkeypatch.setattr(plugin, "_compute", failing_compute)

        result = await plugin.should_process_message(
            channel=channel, sender="user", text="hello", exchange_key="k-fail"
        )
        assert result is None  # fail-open: never rejects, never raises
        assert await plugin.get_appraisal("k-fail") is None

        boom["armed"] = False
        vector = await plugin.get_or_compute(channel, "k-fail", "hello")
        assert vector is not None
        assert await plugin.get_appraisal("k-fail") == vector
        await _teardown(plugin, pm)


# ---------------------------------------------------------------------------
# 6. Cancellation regressions on the shared in-flight future
# ---------------------------------------------------------------------------


class TestCancellation:
    async def test_cancelling_one_waiter_leaves_siblings_unharmed(self, tmp_path, monkeypatch):
        plugin, channel, _, pm = await build_appraisal(tmp_path, summaries=["x"])

        started = asyncio.Event()
        release = asyncio.Event()
        original = plugin._probe_query

        async def gated_probe(*args, **kwargs):
            started.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(plugin, "_probe_query", gated_probe)

        owner = asyncio.create_task(plugin.get_or_compute(channel, "k-c1", "text"))
        await started.wait()
        waiter_a = asyncio.create_task(plugin.get_or_compute(channel, "k-c1", "text"))
        waiter_b = asyncio.create_task(plugin.get_or_compute(channel, "k-c1", "text"))
        await asyncio.sleep(0)  # let waiters attach to the in-flight future

        waiter_a.cancel()
        release.set()

        vector_owner = await owner
        vector_b = await waiter_b
        with pytest.raises(asyncio.CancelledError):
            await waiter_a
        assert vector_owner == vector_b
        # The fire-and-forget persist fired (tasks were spawned).
        assert await plugin.get_appraisal("k-c1") == vector_owner
        await _teardown(plugin, pm)

    async def test_cancelling_owner_evicts_and_wakes_waiters(self, tmp_path, monkeypatch):
        plugin, channel, _, pm = await build_appraisal(tmp_path, summaries=["x"])

        started = asyncio.Event()
        release = asyncio.Event()
        original = plugin._probe_query

        async def gated_probe(*args, **kwargs):
            started.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(plugin, "_probe_query", gated_probe)

        owner = asyncio.create_task(plugin.get_or_compute(channel, "k-c2", "text"))
        await started.wait()
        waiter = asyncio.create_task(plugin.get_or_compute(channel, "k-c2", "text"))
        await asyncio.sleep(0)

        owner.cancel()
        # The waiter must be woken promptly with the cancellation — no hang
        # on an abandoned future.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, timeout=2.0)
        with pytest.raises(asyncio.CancelledError):
            await owner

        # A subsequent call recomputes (the in-flight entry was evicted).
        release.set()
        monkeypatch.setattr(plugin, "_probe_query", original)
        vector = await plugin.get_or_compute(channel, "k-c2", "text")
        assert vector is not None
        await _teardown(plugin, pm)


class TestPersistSpyGaps:
    """2B review-gate test-adequacy follow-ups: (1) the waiter-cancel test
    asserted the persist via get_appraisal, which the in-memory cache
    satisfies — spy on _persist_stage1 itself; (2) probe_score must be
    OMITTED (not nulled) when the probe didn't run."""

    async def test_persist_actually_fires_after_compute(self, tmp_path, monkeypatch):
        plugin, channel, _, pm = await build_appraisal(tmp_path, summaries=["x"])

        persisted = []
        original = plugin._persist_stage1

        async def spying_persist(*args, **kwargs):
            persisted.append(args)
            return await original(*args, **kwargs)

        monkeypatch.setattr(plugin, "_persist_stage1", spying_persist)

        await plugin.get_or_compute(channel, "k-spy", "some text")
        for _ in range(20):
            pending = set(plugin._persist_tasks)
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)

        assert len(persisted) == 1
        # Second call is a cache hit — no second persist.
        await plugin.get_or_compute(channel, "k-spy", "some text")
        assert len(persisted) == 1
        await _teardown(plugin, pm)

    async def test_probe_score_omitted_when_probe_did_not_run(self, tmp_path):
        """No probe (missing DB) → the stage-1 upsert carries NO probe_score
        column: the row's probe_score stays NULL rather than being written,
        and the appraisal envelope still lands."""
        from corvidae.appraisal import AppraisalPlugin
        from corvidae.outcome_log import OutcomeLogPlugin

        # Session DB exists (for outcome_log/persistence) but has NO
        # memory_fts — the probe degrades at on_start.
        db_path = str(tmp_path / "sessions.db")
        db = await aiosqlite.connect(db_path)
        await init_db(db)

        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        outcome = OutcomeLogPlugin()
        pm.register(outcome, name="outcome_log")
        await outcome.on_init(pm=pm, config={})
        await outcome.on_start(config={})

        plugin = AppraisalPlugin()
        pm.register(plugin, name="appraisal")
        cfg = {"daemon": {"session_db": db_path}}
        await plugin.on_init(pm=pm, config=cfg)
        await plugin.on_start(config=cfg)
        assert plugin._probe_db is None  # probe degraded (no memory_fts)

        channel = _make_channel(scope="no-probe-score")
        vector = await plugin.get_or_compute(channel, "k-noprobe", "hello")
        for _ in range(20):
            pending = set(plugin._persist_tasks)
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)

        import json as _json
        async with db.execute(
            "SELECT appraisal, probe_score FROM exchange_log WHERE exchange_key = ?",
            ("k-noprobe",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert _json.loads(row[0])["stage1"] == vector
        assert row[1] is None  # probe_score never written

        await plugin.on_stop()
        await db.close()
