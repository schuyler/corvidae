"""Memory record writes run on a dedicated write connection (#221, folds #196).

Consolidation's insert + vec row + near-dup merge + watermark advance, and
``write_record`` (goal retirement), are one SQLite transaction
(``BEGIN IMMEDIATE`` .. ``COMMIT``) on a connection only MemoryPlugin writes
to, so no other plugin's commit can land mid-section and a rollback cannot
discard another plugin's pending row.
"""

import asyncio
import logging

import aiosqlite
import pytest

from corvidae.channel import Channel
from corvidae.context import MessageType
from corvidae_cognition import memory as memory_module

from .test_memory_consolidation import (
    build_consolidation_env,
    fetch_memories,
    fetch_watermark,
    seed_dialog,
)

DIALOG = [("user", "one"), ("assistant", "two"), ("user", "three")]
LOGGER = "corvidae.memory"
PENDING_JSON = '{"role": "user", "content": "pending row"}'


class _Hooked:
    """Awaitable async-context result of a hooked ``execute``.

    Runs ``before(sql)`` first, then the real statement; mirrors aiosqlite's
    execute() result, which is both awaitable and an async context manager.
    """

    def __init__(self, real, before, sql, args):
        self._real, self._before, self._sql, self._args = real, before, sql, args
        self._cursor = None

    async def _run(self):
        # The hook may raise (injected failure) or interleave another writer.
        await self._before(self._sql)
        return await self._real(self._sql, *self._args)

    def __await__(self):
        return self._run().__await__()

    async def __aenter__(self):
        self._cursor = await self._run()
        return self._cursor

    async def __aexit__(self, *exc):
        await self._cursor.close()


def hook_execute(conn, before):
    """Route every ``conn.execute`` through ``before(sql)`` (async)."""
    real = conn.execute
    conn.execute = lambda sql, *args: _Hooked(real, before, sql, args)


def is_write_sql(sql: str) -> bool:
    """True for the statements of the section, not BEGIN/COMMIT/ROLLBACK."""
    head = sql.lstrip().split(None, 1)[0].upper()
    return head in {"INSERT", "UPDATE", "DELETE"}


async def count(db, table: str) -> int:
    async with db.execute(f"SELECT count(*) FROM {table}") as cursor:
        return (await cursor.fetchone())[0]


async def consolidate(memory, channel, rowids, trigger="idle"):
    await memory._consolidate_range(channel.id, rowids[-1], trigger)


async def stop(memory, db):
    await memory.on_stop()
    await db.close()


class TestAtomicSection:
    """R1: every effect of a consolidation is visible, or none, whatever
    another writer commits on the persistence connection meanwhile."""

    @pytest.mark.parametrize(
        "fail_on",
        [
            "INSERT INTO memory (",
            "INSERT INTO memory_vec",
            "UPDATE memory SET superseded_by",
            "INSERT INTO consolidation_watermark",
        ],
        ids=["memory-insert", "vec-insert", "dup-merge", "watermark"],
    )
    async def test_failure_at_each_step_leaves_nothing(self, tmp_path, fail_on):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        if not memory._vec_available:
            pytest.skip("sqlite-vec not installed")
        first = await seed_dialog(persistence, channel, DIALOG)
        await consolidate(memory, channel, first)
        # A second range with the same embedding so the dup merge runs.
        second = await seed_dialog(persistence, channel, DIALOG)
        before_memories = await count(db, "memory")
        before_vec = await count(db, "memory_vec")

        # Another plugin commits on the persistence connection between every
        # statement of the section, and the chosen step fails.
        await db.execute("PRAGMA busy_timeout=20")
        stray = {"commits": 0}

        async def before(sql):
            if not is_write_sql(sql):
                return
            await db.commit()  # a stray commit on the shared connection
            stray["commits"] += 1
            if sql.startswith(fail_on):
                raise RuntimeError(f"injected failure at {fail_on}")

        hook_execute(await memory._write_conn(), before)
        await consolidate(memory, channel, second)

        assert stray["commits"] >= 1
        assert await count(db, "memory") == before_memories
        assert await count(db, "memory_vec") == before_vec
        async with db.execute(
            "SELECT count(*) FROM memory WHERE superseded_by IS NOT NULL"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        assert await fetch_watermark(db, channel.id) == first[-1]
        await stop(memory, db)

    async def test_success_with_interleaved_commits_is_complete(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)

        async def before(sql):
            if is_write_sql(sql):
                await db.commit()

        hook_execute(await memory._write_conn(), before)
        await consolidate(memory, channel, rowids)

        assert len(await fetch_memories(db)) == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        await stop(memory, db)


class TestIsolationFromOtherWriters:
    """R2: no other plugin shares memory's write transaction."""

    async def test_pending_persist_message_survives_failed_consolidation(
        self, tmp_path
    ):
        path = str(tmp_path / "sessions.db")
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=path
        )
        if not memory._vec_available:
            pytest.skip("sqlite-vec not installed")
        rowids = await seed_dialog(persistence, channel, DIALOG)

        async def fail_vec(sql):
            if sql.startswith("INSERT INTO memory_vec"):
                raise RuntimeError("injected vec failure")

        hook_execute(await memory._write_conn(), fail_vec)

        # A persist_message insert pending on the persistence connection
        # (executed, not yet committed), as it is between its two awaits.
        cursor = await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, "
            "message_type) VALUES (?, ?, 1.0, 'message')",
            (channel.id, PENDING_JSON),
        )
        pending_id = cursor.lastrowid
        task = asyncio.create_task(consolidate(memory, channel, rowids))
        await asyncio.sleep(0.2)  # consolidation waits on the write lock
        await db.commit()  # persist_message's commit
        await task

        async with aiosqlite.connect(path) as other:
            async with other.execute(
                "SELECT message FROM message_log WHERE id = ?", (pending_id,)
            ) as cur:
                row = await cur.fetchone()
        assert row == (PENDING_JSON,)
        assert await fetch_memories(db) == []
        await stop(memory, db)

    async def test_write_connection_is_not_the_persistence_connection(
        self, tmp_path
    ):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        assert await memory._write_conn() is not db
        await stop(memory, db)


class TestRollbackAndRetry:
    """R3, R4: a failed write leaves no row anywhere; the retry stores one."""

    async def test_vec_failure_leaves_no_row_then_retry_stores_one(
        self, tmp_path
    ):
        path = str(tmp_path / "sessions.db")
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=path
        )
        if not memory._vec_available:
            pytest.skip("sqlite-vec not installed")
        rowids = await seed_dialog(persistence, channel, DIALOG)
        wdb = await memory._write_conn()
        real_execute = wdb.execute

        async def fail_vec(sql):
            if sql.startswith("INSERT INTO memory_vec"):
                raise RuntimeError("injected vec failure")

        hook_execute(wdb, fail_vec)
        await consolidate(memory, channel, rowids)

        # Later commits on either connection must not resurrect anything.
        await db.commit()
        await wdb.commit()
        for table in ("memory", "memory_vec", "memory_fts"):
            assert await count(db, table) == 0, table
        assert await fetch_watermark(db, channel.id) is None

        # Restore the vec insert; the next trigger retries the same range.
        wdb.execute = real_execute
        await consolidate(memory, channel, rowids)
        records = await fetch_memories(db)
        assert len(records) == 1
        assert await count(db, "memory_vec") == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        await stop(memory, db)


class TestFailureLogging:
    """R5: failures are logged with their remedy and traceback."""

    async def test_failure_warning_names_channel_trigger_range_and_retry(
        self, tmp_path, caplog
    ):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)

        async def fail(sql):
            if sql.startswith("INSERT INTO consolidation_watermark"):
                raise RuntimeError("injected watermark failure")

        hook_execute(await memory._write_conn(), fail)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            await consolidate(memory, channel, rowids, trigger="compaction")

        records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "consolidation failed" in r.getMessage()
        ]
        assert len(records) == 1
        text = records[0].getMessage()
        assert channel.id in text
        assert "compaction" in text
        assert f"{rowids[0]}" in text and f"{rowids[-1]}" in text
        assert "retried" in text
        assert records[0].exc_info is not None
        await stop(memory, db)

    async def test_failed_rollback_is_logged(self, tmp_path, caplog):
        path = str(tmp_path / "sessions.db")
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=path
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)

        async def fail(sql):
            if sql.startswith("INSERT INTO consolidation_watermark"):
                raise RuntimeError("injected watermark failure")
            if sql.strip().upper() == "ROLLBACK":
                raise RuntimeError("injected rollback failure")

        wdb = await memory._write_conn()
        hook_execute(wdb, fail)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            await consolidate(memory, channel, rowids)

        texts = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "rollback" in r.getMessage().lower()
        ]
        assert texts, "a failed rollback must be logged at WARNING"
        assert path in texts[0] or "sessions.db" in texts[0]
        await stop(memory, db)

    async def test_busy_timeout_is_logged_and_range_is_retried(
        self, tmp_path, caplog, monkeypatch
    ):
        path = str(tmp_path / "sessions.db")
        monkeypatch.setattr(memory_module, "WRITE_BUSY_TIMEOUT_MS", 100)
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=path
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)

        # A third connection holds the write lock past the busy timeout.
        async with aiosqlite.connect(path, isolation_level=None) as holder:
            await holder.execute("BEGIN IMMEDIATE")
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                await consolidate(memory, channel, rowids)
            await holder.execute("ROLLBACK")

        texts = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "busy" in r.getMessage().lower()
        ]
        assert texts, "a blocked write must log a WARNING"
        assert "sessions.db" in texts[0]
        assert "100" in texts[0]
        assert await fetch_memories(db) == []
        assert await fetch_watermark(db, channel.id) is None

        # The range was not lost: the next trigger stores it.
        await consolidate(memory, channel, rowids)
        assert len(await fetch_memories(db)) == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        await stop(memory, db)


class TestWriteRecord:
    """R6: write_record is one transaction on the write connection."""

    ARGS = dict(
        channel_id="irc:#test", summary="I retired a goal.", created_at=1.0,
        importance=0.8, msg_id_start=1, msg_id_end=1,
    )

    async def test_failing_write_record_leaves_no_row(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        if not memory._vec_available:
            pytest.skip("sqlite-vec not installed")

        async def fail_vec(sql):
            if sql.startswith("INSERT INTO memory_vec"):
                raise RuntimeError("injected vec failure")

        wdb = await memory._write_conn()
        real_execute = wdb.execute
        hook_execute(wdb, fail_vec)
        with pytest.raises(RuntimeError, match="injected vec failure"):
            await memory.write_record(**self.ARGS)
        await wdb.commit()
        await db.commit()
        for table in ("memory", "memory_vec", "memory_fts"):
            assert await count(db, table) == 0, table

        # A later successful call still works on the same connection.
        wdb.execute = real_execute
        memory_id = await memory.write_record(**self.ARGS)
        assert (await fetch_memories(db))[0]["id"] == memory_id
        await stop(memory, db)

    async def test_write_record_commits_on_the_write_connection(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        statements: list[str] = []

        async def record(sql):
            statements.append(sql.strip().split(None, 1)[0].upper())

        wdb = await memory._write_conn()
        assert wdb is not db
        hook_execute(wdb, record)
        await memory.write_record(**self.ARGS)

        assert statements[0] == "BEGIN"
        assert statements[-1] == "COMMIT"
        assert len(await fetch_memories(db)) == 1
        await stop(memory, db)


class TestWriteConnectionLifecycle:
    """R7: set up like the persistence connection; opened and closed with
    the plugin; its commits are visible on the persistence connection."""

    async def test_opens_in_wal_with_bounded_busy_timeout_and_vec(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        wdb = await memory._write_conn()
        async with wdb.execute("PRAGMA journal_mode") as cursor:
            assert (await cursor.fetchone())[0] == "wal"
        async with wdb.execute("PRAGMA busy_timeout") as cursor:
            assert (await cursor.fetchone())[0] == memory_module.WRITE_BUSY_TIMEOUT_MS
        if memory._vec_available:
            async with wdb.execute("SELECT count(*) FROM memory_vec") as cursor:
                assert (await cursor.fetchone())[0] == 0
        await stop(memory, db)

    async def test_on_start_opens_and_on_stop_closes(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        # on_start already ran in the builder, after the schema existed.
        wdb = memory._write_db
        assert wdb is not None and wdb is not db
        await memory.on_stop()
        assert memory._write_db is None
        assert wdb._connection is None or not wdb._running
        await db.close()

    async def test_committed_record_is_visible_on_persistence_connection(
        self, tmp_path
    ):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)
        await consolidate(memory, channel, rowids)
        # No commit or refresh on db: a read sees the record at once.
        assert len(await fetch_memories(db)) == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        await stop(memory, db)
