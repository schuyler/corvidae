"""Acceptance tests for the kernel write route to sessions.db (#1211).

One test per approved criterion: R1/R2 (a stalled write unit is rolled back,
in Python and in a long SQL statement), R3 (the ERROR line naming the owner
and deadline) and R6 (startup refuses a non-serialized SQLite build).
"""

import asyncio
import logging
import sqlite3
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from corvidae.context import MessageType
from corvidae.persistence import PersistencePlugin
from corvidae.session_writer import (
    SessionWriter,
    WriteCursor,
    WriteRevoked,
    open_session_store,
)

# Deadline given to every stalling unit; short so the tests stay fast.
STALL_DEADLINE_S = 0.2
# Owner string the stalling unit presents; R3 looks for it in the ERROR line.
STALL_OWNER = "stall-test"
# Margin allowed past the deadline for a concurrent kernel write to land.
MARGIN_S = 1.0
# Recursive CTE that would run for minutes; only interrupt() can end it.
LONG_SQL = (
    "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r) "
    "SELECT count(*) FROM (SELECT i FROM r LIMIT 10000000000)"
)


@dataclass
class Outcome:
    """What one stall-and-collide run observed."""

    write_error: BaseException | None  # what the stalled ``write`` raised
    after_revoke: list[BaseException]  # what the unit's follow-up statement raised
    row_id: int | None  # persist_message's return value
    elapsed: float  # seconds persist_message took


@dataclass
class Store:
    """The route as the daemon wires it: reader, writer and the plugin."""

    reader: aiosqlite.Connection
    writer: SessionWriter
    persistence: PersistencePlugin


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[Store]:
    """A real file-backed store with the injection pair set on persistence."""
    reader, writer = await open_session_store(tmp_path / "sessions.db")
    persistence = PersistencePlugin()
    # Inject the pair the way on_start expects test code to.
    persistence.db, persistence.writer = reader, writer
    try:
        yield Store(reader, writer, persistence)
    finally:
        # Close both halves even if the test failed, so no thread hangs exit.
        try:
            await reader.close()
        finally:
            await writer.close()


async def _scratch_rows(reader: aiosqlite.Connection) -> int:
    """Count rows in the scratch table through the read-only reader."""
    async with reader.execute("SELECT count(*) FROM scratch") as cur:
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


async def _stall_and_collide(
    store: Store, stall: Callable[[WriteCursor, threading.Event], None]
) -> Outcome:
    """Run a stalling unit and a concurrent kernel persist_message.

    ``stall`` runs inside the unit after its INSERT; an exception it raises
    (an interrupted statement) is swallowed. The unit then tries one more
    statement and records what that raised.
    """
    # Scratch table made in its own successful unit, so the stalled unit
    # itself creates nothing.
    await store.writer.write(
        lambda cur: cur.execute("CREATE TABLE scratch (v INTEGER)"),
        owner=STALL_OWNER,
    )

    in_unit = threading.Event()  # set once the unit has inserted its row
    release = threading.Event()  # lets a Python-stalled unit continue
    finished = threading.Event()  # set when the unit's thread is done
    after_revoke: list[BaseException] = []

    def unit(cur: WriteCursor) -> None:
        """Insert a row, stall, then try to write once more."""
        try:
            cur.execute("INSERT INTO scratch (v) VALUES (1)")
            in_unit.set()
            try:
                stall(cur, release)
            except Exception:  # noqa: BLE001 - an interrupted stall is expected
                pass
            try:
                # The revoked unit's next call must raise, not write.
                cur.execute("INSERT INTO scratch (v) VALUES (2)")
            except BaseException as exc:  # noqa: BLE001 - captured for assertion
                after_revoke.append(exc)
        finally:
            finished.set()

    # Start the stalling unit as a task so the kernel write can race it.
    stalled = asyncio.create_task(
        store.writer.write(unit, owner=STALL_OWNER, deadline_s=STALL_DEADLINE_S)
    )
    try:
        # Wait until the unit actually holds the write lock.
        for _ in range(200):
            if in_unit.is_set():
                break
            await asyncio.sleep(0.01)
        assert in_unit.is_set(), "stalling unit never started"

        # The kernel's own write, issued while the lock is held.
        channel = SimpleNamespace(id="test:scope")
        started = time.monotonic()
        row_id = await store.persistence.persist_message(
            channel, {"role": "user", "content": "hi"}, MessageType.MESSAGE
        )
        elapsed = time.monotonic() - started

        # Collect what the stalled write raised to its caller.
        write_error: BaseException | None = None
        try:
            await stalled
        except WriteRevoked as exc:
            write_error = exc
        # Let a Python-stalled unit run on and record its follow-up error.
        release.set()
        await asyncio.to_thread(finished.wait, 5)
    finally:
        # Never leave the unit's thread blocked, whatever failed above.
        release.set()
        if not stalled.done():
            stalled.cancel()
    return Outcome(write_error, after_revoke, row_id, elapsed)


def _python_stall(cur: WriteCursor, release: threading.Event) -> None:
    """Block in plain Python; the safety timeout keeps a broken route bounded."""
    release.wait(timeout=10)


def _sql_stall(cur: WriteCursor, release: threading.Event) -> None:
    """Block inside one long SQL statement that only interrupt() can end."""
    cur.execute(LONG_SQL)


@pytest.mark.parametrize(
    "stall", [_python_stall, _sql_stall], ids=["python-stall", "long-sql"]
)
async def test_r1_r2_stalled_unit_is_rolled_back_and_revoked(
    store: Store, stall: Callable[[WriteCursor, threading.Event], None]
) -> None:
    """R1/R2: a stalled unit is revoked, the kernel write lands, no rows stay."""
    out = await _stall_and_collide(store, stall)

    # R1: the caller of the stalled write is told it was revoked.
    assert isinstance(out.write_error, WriteRevoked)
    # R1: the kernel's persist_message succeeded within deadline plus margin.
    assert isinstance(out.row_id, int)
    assert out.elapsed < STALL_DEADLINE_S + MARGIN_S
    # R2: the stalled unit's next call raised WriteRevoked rather than writing.
    assert len(out.after_revoke) == 1
    assert isinstance(out.after_revoke[0], WriteRevoked)
    # R2: none of the unit's rows (neither the pre-stall nor the post-revoke
    # insert) are visible in the table.
    assert await _scratch_rows(store.reader) == 0


async def test_r3_error_names_owner_and_deadline(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    """R3: one ERROR from the writer names the owner plugin and the deadline."""
    with caplog.at_level(logging.ERROR):
        await _stall_and_collide(store, _python_stall)

    # Only ERROR records from the writer's own logger count.
    errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and r.name == "corvidae.session_writer"
    ]
    assert len(errors) == 1
    message = errors[0].getMessage()
    # The formatted line carries the owner and the deadline (0.2 s).
    assert STALL_OWNER in message
    assert "0.2" in message


def test_r6_refuses_non_serialized_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R6: SessionWriter.open refuses threadsafety != 3, naming value and remedy."""
    # Pretend the SQLite build is only thread-safe at level 1.
    monkeypatch.setattr(sqlite3, "threadsafety", 1)

    with pytest.raises(RuntimeError) as excinfo:
        SessionWriter.open(tmp_path / "sessions.db")

    message = str(excinfo.value)
    # The message names the value found and the remedy to apply.
    assert "threadsafety == 1" in message
    assert "SQLITE_THREADSAFE=1" in message
