"""Memory record writes run as units on the session write route (#221, #1211).

Consolidation's insert + vec row + near-dup merge + watermark advance, and
``write_record`` (goal retirement), are one write unit
(``persistence.write(fn, owner="memory")``): one ``BEGIN IMMEDIATE`` ..
``COMMIT`` on the kernel's writer, so another plugin's write can't land
mid-section and a rollback cannot discard another plugin's row. These tests
inject failures at each statement of a unit and check atomicity through the
route.
"""

import asyncio
import logging

import aiosqlite
import pytest

from corvidae.context import MessageType

from .test_memory_consolidation import (
    EMBED_CONFIG,
    build_consolidation_env,
    fetch_memories,
    fetch_watermark,
    seed_dialog,
)

DIALOG = [("user", "one"), ("assistant", "two"), ("user", "three")]
LOGGER = "corvidae.memory"


class _FailingCursor:
    """A write cursor that raises on statements starting with ``fail_on``.

    Everything else reaches the real cursor, so the unit runs exactly as in
    the daemon until the chosen statement.
    """

    def __init__(self, real, fail_on):
        self._real, self._fail_on = real, fail_on

    def _check(self, sql):
        if self._fail_on and sql.startswith(self._fail_on):
            raise RuntimeError(f"injected failure at {self._fail_on}")

    def execute(self, sql, params=()):
        self._check(sql)
        return self._real.execute(sql, params)

    def executemany(self, sql, seq):
        self._check(sql)
        return self._real.executemany(sql, seq)


class Injector:
    """Wraps ``persistence.write`` so units see a ``_FailingCursor``.

    ``fail_on`` can be changed or cleared between calls; ``calls`` records
    the owner of every unit that went through.
    """

    def __init__(self, persistence, fail_on=None):
        self.fail_on = fail_on
        self.calls: list[str | None] = []
        real = persistence.write

        async def write(fn, **kwargs):
            self.calls.append(kwargs.get("owner"))
            return await real(lambda cur: fn(_FailingCursor(cur, self.fail_on)), **kwargs)

        persistence.write = write


async def count(db, table: str) -> int:
    async with db.execute(f"SELECT count(*) FROM {table}") as cursor:
        return (await cursor.fetchone())[0]


async def consolidate(memory, channel, rowids, trigger="idle"):
    await memory._consolidate_range(channel.id, rowids[-1], trigger)


async def stop(memory, db):
    await memory.on_stop()
    await db.close()


class TestAtomicSection:
    """R1: every effect of a consolidation is visible, or none."""

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
        before_vec = await count(persistence.db, "memory_vec")

        # The chosen statement of the second consolidation's unit fails.
        Injector(persistence, fail_on)
        await consolidate(memory, channel, second)

        assert await count(db, "memory") == before_memories
        assert await count(persistence.db, "memory_vec") == before_vec
        async with db.execute(
            "SELECT count(*) FROM memory WHERE superseded_by IS NOT NULL"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        assert await fetch_watermark(db, channel.id) == first[-1]
        await stop(memory, db)

    async def test_success_with_a_concurrent_persist_is_complete(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)

        # Another plugin's write lands while consolidation runs: the route
        # queues the units, so both complete whole.
        message = {"role": "user", "content": "concurrent"}
        _, persisted = await asyncio.gather(
            consolidate(memory, channel, rowids),
            persistence.persist_message(channel, message, MessageType.MESSAGE),
        )

        assert persisted is not None
        assert len(await fetch_memories(db)) == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        await stop(memory, db)


class TestIsolationFromOtherWriters:
    """R2: no other plugin shares memory's write transaction."""

    async def test_concurrent_persist_message_survives_failed_consolidation(
        self, tmp_path
    ):
        path = str(tmp_path / "sessions.db")
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=path
        )
        if not memory._vec_available:
            pytest.skip("sqlite-vec not installed")
        rowids = await seed_dialog(persistence, channel, DIALOG)
        Injector(persistence, "INSERT INTO memory_vec")

        # A persist_message issued while consolidation's unit is failing and
        # rolling back: its own unit must commit regardless.
        message = {"role": "user", "content": "pending row"}
        results = await asyncio.gather(
            consolidate(memory, channel, rowids),
            persistence.persist_message(channel, message, MessageType.MESSAGE),
        )
        pending_id = results[1]
        assert pending_id is not None

        async with aiosqlite.connect(path) as other:
            async with other.execute(
                "SELECT message FROM message_log WHERE id = ?", (pending_id,)
            ) as cur:
                row = await cur.fetchone()
        assert row is not None and "pending row" in row[0]
        assert await fetch_memories(db) == []
        await stop(memory, db)

    async def test_memory_writes_go_through_the_route_as_owner_memory(
        self, tmp_path
    ):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)
        injector = Injector(persistence)
        await consolidate(memory, channel, rowids)
        # The commit unit is attributed to memory (R3 names the owner on a revoke).
        assert injector.calls and set(injector.calls) == {"memory"}
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
        injector = Injector(persistence, "INSERT INTO memory_vec")
        await consolidate(memory, channel, rowids)

        for table in ("memory", "memory_fts"):
            assert await count(db, table) == 0, table
        assert await count(persistence.db, "memory_vec") == 0
        assert await fetch_watermark(db, channel.id) is None

        # Restore the vec insert; the next trigger retries the same range.
        injector.fail_on = None
        await consolidate(memory, channel, rowids)
        records = await fetch_memories(db)
        assert len(records) == 1
        assert await count(persistence.db, "memory_vec") == 1
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
        Injector(persistence, "INSERT INTO consolidation_watermark")
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

    @pytest.mark.timeout(30)
    async def test_blocked_write_is_logged_and_range_is_retried(
        self, tmp_path, caplog
    ):
        path = str(tmp_path / "sessions.db")
        # The wait is config-resolved (I1), so a short one is asked for the way
        # an operator would set it rather than by patching a module constant.
        config = {**EMBED_CONFIG, "daemon": {"sqlite_busy_timeout_ms": 100}}
        memory, persistence, stub, channel, db = await build_consolidation_env(
            config=config, db_path=path
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)

        # An out-of-process writer holds the write lock past the writer's
        # busy timeout: the unit's BEGIN fails and consolidation warns.
        async with aiosqlite.connect(path, isolation_level=None) as holder:
            await holder.execute("BEGIN IMMEDIATE")
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                await consolidate(memory, channel, rowids)
            await holder.execute("ROLLBACK")

        texts = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "consolidation failed" in r.getMessage()
        ]
        assert texts, "a blocked write must log a WARNING"
        assert channel.id in texts[0] and "retried" in texts[0]
        assert await fetch_memories(db) == []
        assert await fetch_watermark(db, channel.id) is None

        # The range was not lost: the next trigger stores it.
        await consolidate(memory, channel, rowids)
        assert len(await fetch_memories(db)) == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        await stop(memory, db)


class TestWriteRecord:
    """R6: write_record is one unit on the write route."""

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
        injector = Injector(persistence, "INSERT INTO memory_vec")
        with pytest.raises(RuntimeError, match="injected vec failure|injected failure"):
            await memory.write_record(**self.ARGS)
        for table in ("memory", "memory_fts"):
            assert await count(db, table) == 0, table
        assert await count(persistence.db, "memory_vec") == 0

        # A later successful call still works.
        injector.fail_on = None
        memory_id = await memory.write_record(**self.ARGS)
        assert (await fetch_memories(db))[0]["id"] == memory_id
        await stop(memory, db)

    async def test_write_record_is_one_unit_owned_by_memory(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        injector = Injector(persistence)
        await memory.write_record(**self.ARGS)

        # Insert, vec row and dup check are one unit, not several.
        assert injector.calls == ["memory"]
        assert len(await fetch_memories(db)) == 1
        await stop(memory, db)


class TestWriterSetup:
    """R7: the writer runs in WAL with sqlite-vec loaded; committed rows are
    visible on the reader at once."""

    async def test_writer_is_wal_and_has_vec(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        async with persistence.db.execute("PRAGMA journal_mode") as cursor:
            assert (await cursor.fetchone())[0] == "wal"
        if memory._vec_available:
            # Only a writer with vec0 loaded can query the virtual table.
            result = await persistence.write(
                lambda cur: cur.execute("SELECT count(*) FROM memory_vec").rows,
                owner="test",
            )
            assert result == [(0,)]
        await stop(memory, db)

    async def test_committed_record_is_visible_on_reader(self, tmp_path):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        rowids = await seed_dialog(persistence, channel, DIALOG)
        await consolidate(memory, channel, rowids)
        # No commit or refresh on the reader: a read sees the record at once.
        async with persistence.db.execute("SELECT count(*) FROM memory") as cursor:
            assert (await cursor.fetchone())[0] == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        await stop(memory, db)

    async def test_writer_vec_load_failure_runs_fts_only_and_warns(
        self, tmp_path, caplog
    ):
        memory, persistence, stub, channel, db = await build_consolidation_env(
            db_path=str(tmp_path / "sessions.db")
        )
        if not memory._vec_available:
            pytest.skip("sqlite-vec not installed")

        async def fail(path, *, owner):
            raise RuntimeError("extension loading disabled")

        # Re-run the schema pass with the writer's extension load failing.
        persistence.load_extension = fail
        memory._schema_ready = False
        memory._vec_available = False
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            await memory._ensure_schema()

        assert memory._vec_available is False
        messages = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert any("writer" in m and "sqlite-vec" in m and "check" in m
                   for m in messages), messages
        await stop(memory, db)
