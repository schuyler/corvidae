"""Retention's memory-table writes run as write units on the route (#257, #1211).

Demotion, re-promotion and the vec backfill each are one unit
(``persistence.write(fn, owner="memory")``: one ``BEGIN IMMEDIATE`` ..
``COMMIT``), so a mid-batch failure leaves no row changed and another
writer's commit cannot land retention's partial work.
"""

import logging
import time

import pytest

from corvidae_cognition.retention import run_retention_job

from .test_memory_consolidation import build_consolidation_env
from .test_memory_retention import fetch_memory, vec_row_exists
from .test_memory_retention import seed_memory_row as _seed
from .test_memory_write_conn import _FailingCursor, count, stop

DAY = 86400.0
NOW = time.time()
OLD = NOW - 400 * DAY  # far past grace, decayed well below demote_below


async def env(tmp_path):
    """File-DB memory env (the write route needs a file) with vec."""
    memory, persistence, stub, channel, db = await build_consolidation_env(
        db_path=str(tmp_path / "sessions.db")
    )
    if not memory._vec_available:
        pytest.skip("sqlite-vec not installed")
    return memory, stub, channel, db


async def seed_memory_row(db, channel_id, summary, **kw):
    """Seed a row; an embedded one gets a 4-dim vec row (the env's schema)."""
    embedded = kw.get("embedded", 1)
    mid = await _seed(db, channel_id, summary, vec_available=False, **kw)
    if embedded:
        import sqlite_vec

        await db.execute(
            "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
            (mid, sqlite_vec.serialize_float32([1.0, 0.0, 0.0, 0.0])),
        )
    return mid


def fail_nth_write(persistence, nth, prefix=None):
    """Make the nth (1-based) statement starting with ``prefix`` (any
    statement when None) inside a unit raise; ``state["units"]`` counts the
    units that went through the route."""
    state = {"seen": 0, "units": 0}

    class _Nth(_FailingCursor):
        def _check(self, sql):
            if prefix is None or sql.startswith(prefix):
                state["seen"] += 1
                if state["seen"] == nth:
                    raise RuntimeError("injected failure")

    real = persistence.write

    async def write(fn, **kwargs):
        state["units"] += 1
        return await real(lambda cur: fn(_Nth(cur, None)), **kwargs)

    persistence.write = write
    return state


class TestDemotionAtomic:
    async def test_failure_mid_batch_demotes_nothing(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"old {i}", importance=0.3,
                                  created_at=OLD)
            for i in range(3)
        ]
        state = fail_nth_write(memory._persistence(), 2, "UPDATE memory SET indexed = 0")

        with pytest.raises(RuntimeError, match="injected"):
            await run_retention_job(memory, now=NOW)

        assert state["units"] >= 1  # the write ran as a unit on the route
        for mid in ids:
            row = await fetch_memory(db, mid)
            assert (row["indexed"], row["embedded"]) == (1, 1)
            assert await vec_row_exists(db, mid)
        await stop(memory, db)

    async def test_success_through_the_route_is_complete(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"old {i}", importance=0.3,
                                  created_at=OLD)
            for i in range(3)
        ]
        state = fail_nth_write(memory._persistence(), 10_000)

        await run_retention_job(memory, now=NOW)

        assert state["units"] >= 1
        for mid in ids:
            assert (await fetch_memory(db, mid))["indexed"] == 0
            assert not await vec_row_exists(db, mid)
        await stop(memory, db)


class TestRepromotionAtomic:
    async def test_failure_mid_batch_repromotes_nothing(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"fresh {i}", importance=0.9,
                                  indexed=0, embedded=0, created_at=NOW)
            for i in range(3)
        ]
        state = fail_nth_write(memory._persistence(), 2, "UPDATE memory SET indexed = 1")

        with pytest.raises(RuntimeError, match="injected"):
            await run_retention_job(memory, now=NOW)

        assert state["units"] >= 1
        for mid in ids:
            assert (await fetch_memory(db, mid))["indexed"] == 0
        await stop(memory, db)


class TestBackfillAtomic:
    async def test_failure_mid_batch_backfills_nothing(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"fresh {i}", importance=0.9,
                                  embedded=0, created_at=NOW)
            for i in range(3)
        ]
        stub.embed_result = [[1.0, 0.0, 0.0, 0.0]] * 3
        before_vec = await count(db, "memory_vec")
        state = fail_nth_write(memory._persistence(), 2, "INSERT INTO memory_vec")

        await run_retention_job(memory, now=NOW)  # backfill logs its failure

        assert state["units"] >= 1
        assert await count(db, "memory_vec") == before_vec
        for mid in ids:
            assert (await fetch_memory(db, mid))["embedded"] == 0
        await stop(memory, db)

    async def test_embedding_is_computed_outside_the_unit(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        await seed_memory_row(db, channel.id, "fresh", importance=0.9,
                              embedded=0, created_at=NOW)
        persistence = memory._persistence()
        real_write = persistence.write
        in_unit = {"open": False}

        async def write(fn, **kwargs):
            # True from the moment a unit is submitted until it returns.
            in_unit["open"] = True
            try:
                return await real_write(fn, **kwargs)
            finally:
                in_unit["open"] = False

        persistence.write = write
        seen = []
        real_embed = stub.embed

        async def embed(texts, kind=None):
            seen.append(in_unit["open"])  # no write lock held over the await
            return await real_embed(texts, kind=kind)

        stub.embed = embed

        await run_retention_job(memory, now=NOW)

        assert seen == [False]
        await stop(memory, db)

    async def test_success_through_the_route_is_complete(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"fresh {i}", importance=0.9,
                                  embedded=0, created_at=NOW)
            for i in range(2)
        ]
        stub.embed_result = [[1.0, 0.0, 0.0, 0.0]] * 2
        fail_nth_write(memory._persistence(), 10_000)

        await run_retention_job(memory, now=NOW)

        for mid in ids:
            assert (await fetch_memory(db, mid))["embedded"] == 1
            assert await vec_row_exists(db, mid)
        await stop(memory, db)


def mutate_before_txn(memory, db, nth, sql, params=()):
    """Wrap ``persistence.write`` so ``sql`` commits on the seeding connection
    just before the nth (1-based) unit is submitted: after the pass's
    candidate SELECT, before its write, and never inside the unit (which
    holds the write lock)."""
    persistence = memory._persistence()
    real = persistence.write
    state = {"calls": 0, "applied": False}

    async def write(fn, **kwargs):
        # Count units; mutate only ahead of the one under test.
        state["calls"] += 1
        if state["calls"] == nth:
            await db.execute(sql, params)
            state["applied"] = True
        return await real(fn, **kwargs)

    persistence.write = write
    return state


class TestDemotionRechecksSelection:
    async def test_row_retrieved_meanwhile_is_not_demoted(self, tmp_path, caplog):
        memory, stub, channel, db = await env(tmp_path)
        mid = await seed_memory_row(db, channel.id, "old", importance=0.3,
                                    created_at=OLD)
        # A retrieval lands after the candidate SELECT, before the write.
        # The stamp is old enough that the row would still score below the
        # threshold, so a (wrong) demotion is not undone by re-promotion.
        state = mutate_before_txn(
            memory, db, 1,
            "UPDATE memory SET last_retrieved_at = ? WHERE id = ?", (OLD + 1, mid),
        )

        with caplog.at_level(logging.INFO):
            await run_retention_job(memory, now=NOW)

        assert state["applied"]
        # Nothing was written, so no "demoted" line is logged.
        assert not [r for r in caplog.records if "retention: demoted" in r.message]

        row = await fetch_memory(db, mid)
        assert (row["indexed"], row["embedded"]) == (1, 1)
        assert await vec_row_exists(db, mid)
        await stop(memory, db)

    async def test_row_redacted_meanwhile_is_not_demoted(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        mid = await seed_memory_row(db, channel.id, "old", importance=0.3,
                                    created_at=OLD)
        # A redaction lands between the SELECT and the write.
        state = mutate_before_txn(
            memory, db, 1, "UPDATE memory SET redacted = 1 WHERE id = ?", (mid,)
        )

        await run_retention_job(memory, now=NOW)

        assert state["applied"]
        row = await fetch_memory(db, mid)
        assert (row["indexed"], row["embedded"], row["redacted"]) == (1, 1, 1)
        assert await vec_row_exists(db, mid)
        await stop(memory, db)

    async def test_log_reports_rows_actually_demoted(self, tmp_path, caplog):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"old {i}", importance=0.3,
                                  created_at=OLD)
            for i in range(2)
        ]
        # One of the two candidates is redacted before the write.
        state = mutate_before_txn(
            memory, db, 1, "UPDATE memory SET redacted = 1 WHERE id = ?", (ids[0],)
        )

        with caplog.at_level(logging.INFO):
            await run_retention_job(memory, now=NOW)

        assert state["applied"]
        assert "retention: demoted 1 records" in caplog.text
        assert (await fetch_memory(db, ids[1]))["indexed"] == 0
        await stop(memory, db)


# Mutations that make a row ineligible; each takes the row id as its param.
SUPERSEDE = "UPDATE memory SET superseded_by = id + 1000 WHERE id = ?"
REDACT = "UPDATE memory SET redacted = 1 WHERE id = ?"


class TestRepromotionRechecksSelection:
    @pytest.mark.parametrize("mutation", [SUPERSEDE, REDACT], ids=["superseded", "redacted"])
    async def test_row_ineligible_meanwhile_stays_unindexed(self, tmp_path, mutation, caplog):
        memory, stub, channel, db = await env(tmp_path)
        mid = await seed_memory_row(db, channel.id, "fresh", importance=0.9,
                                    indexed=0, embedded=0, created_at=NOW)
        # Supersession or redaction lands between the SELECT and the write.
        state = mutate_before_txn(memory, db, 1, mutation, (mid,))

        with caplog.at_level(logging.INFO):
            await run_retention_job(memory, now=NOW)

        assert state["applied"]
        row = await fetch_memory(db, mid)
        assert row["indexed"] == 0
        # Nothing was written, so no count line is logged.
        assert not [r for r in caplog.records if "retention: re-promoted" in r.message]
        await stop(memory, db)


class TestBackfillRechecksSelection:
    @pytest.mark.parametrize("mutation", [SUPERSEDE, REDACT], ids=["superseded", "redacted"])
    async def test_row_ineligible_meanwhile_gets_no_vec_row(self, tmp_path, mutation, caplog):
        memory, stub, channel, db = await env(tmp_path)
        mid = await seed_memory_row(db, channel.id, "fresh", importance=0.9,
                                    embedded=0, created_at=NOW)
        stub.embed_result = [[1.0, 0.0, 0.0, 0.0]]
        # No demotion/re-promotion candidates: backfill's unit is the 1st.
        state = mutate_before_txn(memory, db, 1, mutation, (mid,))

        with caplog.at_level(logging.INFO):
            await run_retention_job(memory, now=NOW)

        assert state["applied"]
        row = await fetch_memory(db, mid)
        assert row["embedded"] == 0
        assert not await vec_row_exists(db, mid)
        # Nothing was written, so no count line is logged.
        assert not [r for r in caplog.records if "retention: backfilled" in r.message]
        await stop(memory, db)

    async def test_row_demoted_meanwhile_gets_no_vec_row(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        mid = await seed_memory_row(db, channel.id, "fresh", importance=0.9,
                                    embedded=0, created_at=NOW)
        stub.embed_result = [[1.0, 0.0, 0.0, 0.0]]
        # A demotion lands between the SELECT and the backfill write.
        state = mutate_before_txn(
            memory, db, 1, "UPDATE memory SET indexed = 0 WHERE id = ?", (mid,)
        )

        await run_retention_job(memory, now=NOW)

        assert state["applied"]
        row = await fetch_memory(db, mid)
        assert (row["indexed"], row["embedded"]) == (0, 0)
        assert not await vec_row_exists(db, mid)
        await stop(memory, db)
