"""Retention's memory-table writes run in memory's write transaction (#257).

Demotion, re-promotion and the vec backfill each are one
``BEGIN IMMEDIATE`` .. ``COMMIT`` (``MemoryPlugin._write_txn``) on memory's
dedicated write connection, so another plugin's commit on the persistence
connection cannot land retention's partial work and a mid-batch failure
leaves no row changed.
"""

import time

import pytest

from corvidae_cognition.retention import run_retention_job

from .test_memory_consolidation import build_consolidation_env
from .test_memory_retention import fetch_memory, vec_row_exists
from .test_memory_retention import seed_memory_row as _seed
from .test_memory_write_conn import count, hook_execute, is_write_sql, stop

DAY = 86400.0
NOW = time.time()
OLD = NOW - 400 * DAY  # far past grace, decayed well below demote_below


async def env(tmp_path):
    """File-DB memory env (a write connection needs a file) with vec."""
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
        await db.commit()
    return mid


def fail_nth_write(db, nth, prefix=None):
    """Hook that commits on the shared connection before every write
    statement and raises at the nth (1-based) matching write."""
    state = {"seen": 0, "stray": 0}

    async def before(sql):
        if not is_write_sql(sql):
            return
        await db.commit()  # a stray commit on the shared connection
        state["stray"] += 1
        if prefix is None or sql.startswith(prefix):
            state["seen"] += 1
            if state["seen"] == nth:
                raise RuntimeError("injected failure")

    return before, state


class TestDemotionAtomic:
    async def test_failure_mid_batch_demotes_nothing(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"old {i}", importance=0.3,
                                  created_at=OLD)
            for i in range(3)
        ]
        before, state = fail_nth_write(db, 2, "UPDATE memory SET indexed = 0")
        hook_execute(await memory._write_conn(), before)

        with pytest.raises(RuntimeError, match="injected"):
            await run_retention_job(memory, now=NOW)

        assert state["stray"] >= 1  # the write ran on the write connection
        for mid in ids:
            row = await fetch_memory(db, mid)
            assert (row["indexed"], row["embedded"]) == (1, 1)
            assert await vec_row_exists(db, mid)
        await stop(memory, db)

    async def test_success_with_interleaved_commits_is_complete(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"old {i}", importance=0.3,
                                  created_at=OLD)
            for i in range(3)
        ]
        before, state = fail_nth_write(db, 10_000)
        hook_execute(await memory._write_conn(), before)

        await run_retention_job(memory, now=NOW)

        assert state["stray"] >= 3
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
        before, state = fail_nth_write(db, 2, "UPDATE memory SET indexed = 1")
        hook_execute(await memory._write_conn(), before)

        with pytest.raises(RuntimeError, match="injected"):
            await run_retention_job(memory, now=NOW)

        assert state["stray"] >= 1
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
        before, state = fail_nth_write(db, 2, "INSERT INTO memory_vec")
        hook_execute(await memory._write_conn(), before)

        await run_retention_job(memory, now=NOW)  # backfill logs its failure

        assert state["stray"] >= 1
        assert await count(db, "memory_vec") == before_vec
        for mid in ids:
            assert (await fetch_memory(db, mid))["embedded"] == 0
        await stop(memory, db)

    async def test_embedding_is_computed_outside_the_transaction(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        await seed_memory_row(db, channel.id, "fresh", importance=0.9,
                              embedded=0, created_at=NOW)
        wdb = await memory._write_conn()
        seen = []
        real_embed = stub.embed

        async def embed(texts, kind=None):
            seen.append(wdb.in_transaction)  # no write lock held over the await
            return await real_embed(texts, kind=kind)

        stub.embed = embed

        await run_retention_job(memory, now=NOW)

        assert seen == [False]
        await stop(memory, db)

    async def test_success_with_interleaved_commits_is_complete(self, tmp_path):
        memory, stub, channel, db = await env(tmp_path)
        ids = [
            await seed_memory_row(db, channel.id, f"fresh {i}", importance=0.9,
                                  embedded=0, created_at=NOW)
            for i in range(2)
        ]
        stub.embed_result = [[1.0, 0.0, 0.0, 0.0]] * 2
        before, state = fail_nth_write(db, 10_000)
        hook_execute(await memory._write_conn(), before)

        await run_retention_job(memory, now=NOW)

        for mid in ids:
            assert (await fetch_memory(db, mid))["embedded"] == 1
            assert await vec_row_exists(db, mid)
        await stop(memory, db)
