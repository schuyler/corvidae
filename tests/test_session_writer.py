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
    WriteAbandoned,
    WriteClosed,
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
    stall_errors: list[BaseException] = []  # what the stall itself raised

    def unit(cur: WriteCursor) -> None:
        """Insert a row, stall, then try to write once more."""
        try:
            cur.execute("INSERT INTO scratch (v) VALUES (1)")
            in_unit.set()
            try:
                stall(cur, release)
            except Exception as exc:  # noqa: BLE001 - an interrupted stall is expected
                # Recorded, not swallowed: the long-SQL stall ends this way.
                stall_errors.append(exc)
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
        # Bounded so a route that never revokes fails this test instead of
        # tripping pytest's whole-run timeout.
        row_id = await asyncio.wait_for(
            store.persistence.persist_message(
                channel, {"role": "user", "content": "hi"}, MessageType.MESSAGE
            ),
            STALL_DEADLINE_S + MARGIN_S + 1,
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
    # The structured fields name the deadline itself, not an elapsed time.
    assert errors[0].owner == STALL_OWNER
    assert errors[0].deadline_s == STALL_DEADLINE_S


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


async def test_close_is_bounded_when_a_statement_ignores_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """#1315: a statement that never yields to ``interrupt()`` must not hang close.

    Simulated from the revoker's side: the unit's lock is held by a thread that
    never releases it, which is the observable a statement that ignores
    ``interrupt()`` presents — ``acquire`` never succeeds however often the loop
    interrupts (``sqlite3.Connection.interrupt`` is read-only in C, so the
    interrupt itself cannot be stubbed). Past the bound, ``close()`` must log
    and leave the connection open rather than closing it under the worker.
    """
    import corvidae.session_writer as sw

    monkeypatch.setattr(sw, "REVOKE_BOUND_S", 0.2)
    writer = SessionWriter.open(tmp_path / "sessions.db")
    unit = sw._Unit("stuck-owner", 0.2)
    unit.state = sw._RUNNING

    holding = threading.Event()
    release = threading.Event()

    def hold_the_lock() -> None:
        unit.lock.acquire()
        holding.set()
        release.wait(5.0)
        unit.lock.release()

    holder = threading.Thread(target=hold_the_lock, daemon=True)
    holder.start()
    holding.wait(2.0)

    writer._last_unit = unit

    try:
        started = time.monotonic()
        with caplog.at_level(logging.ERROR):
            await asyncio.wait_for(writer.close(), timeout=3.0)
        elapsed = time.monotonic() - started
        # Left open: the connection is still usable after close() returned.
        still_open = writer._conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        release.set()
        holder.join(2.0)
        writer._conn.close()

    # Bounded: it returned on its own bound, not on the unit finishing.
    assert elapsed < 1.0, elapsed
    errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and r.name == "corvidae.session_writer"
    ]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "stuck-owner" in message
    assert "ignores interrupt" in message
    assert still_open
    assert writer._closed is True


async def test_close_is_bounded_when_the_route_lock_itself_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """#1315 review: the lock wait has to be bounded, not only the revoke.

    A statement that ignores ``interrupt()`` keeps its shielded unit task — and
    therefore the route lock — alive forever, so ``close()`` would block before
    reaching any bound behind the lock. This holds the route lock the way that
    case does and requires ``close()`` to return on its own bound.
    """
    import corvidae.session_writer as sw

    monkeypatch.setattr(sw, "REVOKE_BOUND_S", 0.2)
    writer = SessionWriter.open(tmp_path / "sessions.db")

    lock = writer._lock()
    await lock.acquire()  # the stuck unit's shielded task would hold this
    try:
        started = time.monotonic()
        with caplog.at_level(logging.ERROR):
            await asyncio.wait_for(writer.close(), timeout=3.0)
        elapsed = time.monotonic() - started
        still_open = writer._conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        lock.release()
        writer._conn.close()

    assert elapsed < 1.0, elapsed
    errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and r.name == "corvidae.session_writer"
    ]
    assert len(errors) == 1
    assert "write route was still held" in errors[0].getMessage()
    assert still_open
    assert writer._closed is True


def test_outcome_does_not_block_when_the_unit_lock_is_held(
    tmp_path: Path,
) -> None:
    """#1315 critical: ``_outcome`` must not acquire the lock on the loop.

    After a revocation that gave up, the worker holds the unit's lock inside a C
    call. ``_outcome`` runs on the event loop; a plain ``with unit.lock`` there
    wedges the loop, which also disarms shutdown's force-exit.
    """
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    unit = sw._Unit("wedged-owner", 0.2)
    # The writer has abandoned a unit: the lock is held inside a C call, so the
    # wait here would be unbounded.
    writer._abandoned = True
    writer._abandoned_owner = unit.owner
    writer._last_unit = unit
    holding = threading.Event()
    release = threading.Event()

    def hold() -> None:
        unit.lock.acquire()
        holding.set()
        release.wait(5.0)
        unit.lock.release()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    try:
        assert holding.wait(2.0)
        started = time.monotonic()
        with pytest.raises(WriteAbandoned):
            writer._outcome(unit)
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        holder.join(2.0)
        writer._conn.close()


def test_the_revoke_bound_fits_inside_the_configured_budget(tmp_path: Path) -> None:
    """#1315: the bound is derived per writer, not a constant above the budget.

    A bound longer than the graceful phase never elapses inside it, so the
    diagnostic is dead and the force-exit lands mid-wait.
    """
    import corvidae.session_writer as sw
    from corvidae.shutdown import (
        DEFAULT_SHUTDOWN_TIMEOUTS,
        parse_shutdown_timeouts,
        stop_bound_seconds,
    )

    assert sw.REVOKE_BOUND_S < DEFAULT_SHUTDOWN_TIMEOUTS.graceful

    small = parse_shutdown_timeouts({"shutdown_timeout": 1}).graceful
    writer = SessionWriter.open(tmp_path / "sessions.db", graceful_s=small)
    try:
        assert writer._revoke_bound_s == stop_bound_seconds(small)
        assert writer._revoke_bound_s < small
    finally:
        writer._conn.close()


async def test_a_stalled_unit_through_write_leaves_the_loop_responsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1315 critical, driven through the real path: write() → _revoke → outcome.

    The unit function takes the unit's own lock and then sleeps, which is what a
    statement stuck in a C call presents to the revoker. Revocation gives up at
    the bound; the awaiting loop must keep running (a wedged loop disarms
    shutdown's force-exit) and the caller must see WriteRevoked.
    """
    import corvidae.session_writer as sw

    monkeypatch.setattr(sw, "REVOKE_BOUND_S", 0.2)
    writer = SessionWriter.open(tmp_path / "sessions.db", graceful_s=None)
    monkeypatch.setattr(writer, "_revoke_bound_s", 0.2)
    release = threading.Event()

    def stall(cursor: WriteCursor) -> None:
        # Hold the unit's lock across the stall, as a statement in C does.
        cursor._unit.lock.acquire()
        release.wait(5.0)
        cursor._unit.lock.release()

    ticks = 0

    async def count_ticks(stop: asyncio.Future[None]) -> None:
        nonlocal ticks
        while not stop.done():
            ticks += 1
            await asyncio.sleep(0.05)

    stop = asyncio.get_running_loop().create_future()
    ticker = asyncio.create_task(count_ticks(stop))
    try:
        with pytest.raises(WriteRevoked):
            await asyncio.wait_for(
                writer.write(stall, owner="stall-test", deadline_s=0.2),
                timeout=3.0,
            )
        # The loop kept turning while the stalled unit was abandoned.
        assert ticks >= 4, ticks
    finally:
        stop.set_result(None)
        release.set()
        await ticker
        writer._conn.close()


async def test_close_does_not_close_the_connection_after_an_abandonment(
    tmp_path: Path,
) -> None:
    """#1315, third recurrence: after a give-up, close() must not close on the loop.

    ``sqlite3.Connection.close()`` waits for a statement still inside the
    connection; with an abandoned unit that statement never returns, so closing
    here blocks the event loop — the hang the issue exists to remove.
    """
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    unit = sw._Unit("abandoned-owner", 0.2)
    unit.state = sw._REVOKED
    writer._last_unit = unit
    writer._abandoned = True

    real_conn = writer._conn
    closed: list[bool] = []

    def record_close() -> None:
        closed.append(True)

    # A stand-in, because sqlite3.Connection.close is read-only in C; reaching it
    # at all is what the assertion detects.
    writer._conn = SimpleNamespace(close=record_close)  # type: ignore[assignment]
    try:
        started = time.monotonic()
        await asyncio.wait_for(writer.close(), timeout=3.0)
        assert time.monotonic() - started < 0.5
        assert closed == []
        assert writer._closed is True
    finally:
        writer._conn = real_conn  # type: ignore[assignment]
        real_conn.close()


async def test_a_real_abandonment_keeps_the_loop_free_through_close(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """#1315, end to end: write() stalls, the unit is abandoned, close() returns.

    The unit function takes the unit's own lock and sleeps — what a statement
    inside a C call presents to the revoker — so revocation gives up on the real
    path and the flag is set by the code, not by the test. The loop must keep
    turning while `close()` runs, and the connection must be left open.
    """
    writer = SessionWriter.open(tmp_path / "sessions.db", graceful_s=0.5)
    release = threading.Event()

    def stall(cursor: WriteCursor) -> None:
        cursor._unit.lock.acquire()
        release.wait(5.0)
        cursor._unit.lock.release()

    ticks = 0

    async def count_ticks(stop: asyncio.Future[None]) -> None:
        nonlocal ticks
        while not stop.done():
            ticks += 1
            await asyncio.sleep(0.05)

    stop = asyncio.get_running_loop().create_future()
    ticker = asyncio.create_task(count_ticks(stop))
    try:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(WriteRevoked):
                await asyncio.wait_for(
                    writer.write(stall, owner="stall-test", deadline_s=0.2), timeout=4.0
                )
        # One ERROR for the abandonment, naming the owner (#1328 follow-up 5).
        errors = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR and r.name == "corvidae.session_writer"
        ]
        assert len(errors) == 1, [r.getMessage() for r in errors]
        assert errors[0].owner == "stall-test"
        # Set by _revoke on its give-up branch, not by this test; the behaviour
        # assertions below (not this line) are what pin the fix.
        assert writer._abandoned is True
        started = time.monotonic()
        await asyncio.wait_for(writer.close(), timeout=2.0)
        elapsed = time.monotonic() - started
        # A wedged loop could neither finish close() nor run the ticker after it.
        assert elapsed < 1.0, elapsed
        ticks_before = ticks
        await asyncio.sleep(0.15)
        assert ticks > ticks_before, "the loop stopped turning after close()"
        # Left open: the connection still answers after close(). Off the loop,
        # because on a real abandonment this call is the block probe A measured.
        assert (
            await asyncio.to_thread(lambda: writer._conn.execute("SELECT 1").fetchone())
        ) == (1,)
        # A later write is refused by name rather than failing inside SQLite
        # with "cannot start a transaction within a transaction".
        def healthy(cursor: WriteCursor) -> None:
            cursor.execute("SELECT 1")

        with pytest.raises(WriteAbandoned) as excinfo:
            await writer.write(healthy, owner="healthy", deadline_s=5.0)
        assert "stall-test" in str(excinfo.value)
        assert "plugins.disabled" in str(excinfo.value)
    finally:
        stop.set_result(None)
        release.set()
        await ticker
        writer._conn.close()


async def test_an_abandoned_unit_never_hands_back_a_result(tmp_path: Path) -> None:
    """#1315: the give-up branch writes no state, so state alone must not decide.

    If the worker's own COMMIT lands after the caller was told nothing, the unit
    reads `_COMMITTED`; the caller must still see `WriteAbandoned` rather than a
    silent `None`.
    """
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    unit = sw._Unit("abandoned-owner", 0.2)
    unit.state = sw._COMMITTED  # what the worker leaves behind if its COMMIT lands
    writer._last_unit = unit
    writer._abandoned = True
    writer._abandoned_owner = unit.owner
    try:
        with pytest.raises(WriteAbandoned):
            writer._outcome(unit)
    finally:
        writer._conn.close()


async def test_an_abandoned_unit_is_not_reported_as_rolled_back(tmp_path: Path) -> None:
    """#1315: a `_REVOKED` unit on an abandoned writer must not read as a rollback.

    The give-up branch writes no state, so a unit from an earlier, successful
    revocation can still carry `_REVOKED`; the caller must see the abandonment,
    because on a busy wait its write may have landed.
    """
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    unit = sw._Unit("abandoned-owner", 0.2)
    unit.state = sw._REVOKED
    writer._last_unit = unit
    writer._abandoned = True
    writer._abandoned_owner = unit.owner
    try:
        with pytest.raises(WriteAbandoned):
            writer._outcome(unit)
    finally:
        writer._conn.close()


async def test_a_healthy_writer_reports_the_workers_own_error(
    tmp_path: Path,
) -> None:
    """#1328: the lock-unavailable branch must not blame an abandonment that
    never happened.

    A worker that fails its BEGIN signals `done` from inside the unit lock, so the
    loop can see the outcome before the lock is released. The caller must get the
    real error, not "this writer refuses further writes until restart".
    """
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    unit = sw._Unit("healthy-owner", 0.2)
    unit.state = sw._ROLLED_BACK
    unit.exc = sqlite3.OperationalError("cannot start a transaction within a transaction")
    holding = threading.Event()
    release = threading.Event()

    def hold_briefly() -> None:
        unit.lock.acquire()
        holding.set()
        release.wait(5.0)
        unit.lock.release()

    holder = threading.Thread(target=hold_briefly, daemon=True)
    holder.start()
    try:
        assert holding.wait(2.0)
        timer = threading.Timer(0.2, release.set)
        timer.start()
        with pytest.raises(sqlite3.OperationalError):
            await asyncio.wait_for(asyncio.to_thread(writer._outcome, unit), timeout=3.0)
        assert writer._abandoned is False
    finally:
        release.set()
        holder.join(2.0)
        writer._conn.close()


async def test_a_queued_unit_is_refused_by_name_when_the_route_frees(
    tmp_path: Path,
) -> None:
    """#1328 follow-up 1: the refusal must not blame a plugin that never ran.

    A second `write()` that starts while the writer is healthy passes the entry
    check and queues on the route lock; if the unit ahead of it is abandoned, the
    queued unit must be refused by name once the route is free, naming itself and
    the abandoned unit, not just the other plugin's owner.
    """
    writer = SessionWriter.open(tmp_path / "sessions.db", graceful_s=0.5)
    lock = writer._lock()
    await lock.acquire()  # the unit ahead holds the route

    ran: list[str] = []

    def body(cursor: WriteCursor) -> None:
        ran.append("ran")
        raise AssertionError("a refused unit must not run")

    try:
        queued = asyncio.create_task(
            writer.write(body, owner="plugin-b", deadline_s=1.0)
        )
        await asyncio.sleep(0.1)  # plugin-b passed the entry check and is queued
        assert writer._abandoned is False
        # Now the unit ahead of it is abandoned.
        writer._abandoned = True
        writer._abandoned_owner = "plugin-a"
        lock.release()
        with pytest.raises(WriteAbandoned) as excinfo:
            await asyncio.wait_for(queued, timeout=2.0)
        message = str(excinfo.value)
        assert "plugin-b" in message
        assert "plugin-a" in message
        assert excinfo.value.owner == "plugin-b"
        assert excinfo.value.abandoned_owner == "plugin-a"
        assert ran == [], "the refused unit was allowed to run"
    finally:
        if lock.locked():
            lock.release()
        await writer.close()
        writer._conn.close()


async def test_an_already_abandoned_writer_names_the_caller(tmp_path: Path) -> None:
    """#1330 round, M5: the entry-path refusal names the caller, not the old owner."""
    writer = SessionWriter.open(tmp_path / "sessions.db")
    writer._abandoned = True
    writer._abandoned_owner = "plugin-a"
    # Closed as well as abandoned: the abandonment message must win, because it
    # names the unit and the remedy; "start a new one" alone loses that (#1328).
    await writer.close()

    def body(cursor: WriteCursor) -> None:
        raise AssertionError("a refused unit must not run")

    try:
        with pytest.raises(WriteAbandoned) as excinfo:
            await writer.write(body, owner="plugin-b", deadline_s=1.0)
        message = str(excinfo.value)
        assert "plugin-b" in message
        assert "plugin-a" in message
        assert excinfo.value.owner == "plugin-b"
    finally:
        writer._conn.close()
class _RollbackFails:
    """Connection proxy whose ROLLBACK raises, then delegates everything else."""

    def __init__(self, conn, failures: int) -> None:
        self._conn = conn
        self._failures = failures
        self.rollbacks = 0

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, sql, *args):
        if "ROLLBACK" in sql.upper() and self._failures > 0:
            self._failures -= 1
            self.rollbacks += 1
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)


async def test_a_rollback_that_fails_once_is_retried_and_the_writer_survives(
    tmp_path: Path,
) -> None:
    """#1314: a transient ROLLBACK failure must not wedge the write route."""
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    proxy = _RollbackFails(writer._conn, failures=1)
    writer._conn = proxy
    unit = sw._Unit("revoked-owner", 1.0)
    writer._conn.execute("BEGIN IMMEDIATE")
    try:
        writer._rollback_after_revocation(writer._conn, unit)
        assert proxy.rollbacks == 1
        assert not writer._abandoned
        # The next unit commits: the connection is not stuck in a transaction.
        await writer.write(lambda cur: cur.execute("SELECT 1"), owner="plugin-b")
    finally:
        writer._conn._conn.close()


async def test_a_rollback_that_never_succeeds_abandons_the_writer_by_name(
    tmp_path: Path,
) -> None:
    """#1314: failing twice must give WriteAbandoned, not 'cannot start a
    transaction within a transaction' on every later write."""
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    proxy = _RollbackFails(writer._conn, failures=2)
    writer._conn = proxy
    unit = sw._Unit("revoked-owner", 1.0)
    writer._conn.execute("BEGIN IMMEDIATE")
    try:
        writer._rollback_after_revocation(writer._conn, unit)
        assert proxy.rollbacks == 2
        assert writer._abandoned
        assert writer._abandoned_owner == "revoked-owner"
        with pytest.raises(WriteAbandoned) as excinfo:
            await writer.write(lambda cur: cur.execute("SELECT 1"), owner="plugin-b")
        assert "revoked-owner" in str(excinfo.value)
    finally:
        writer._conn._conn.close()


async def test_one_error_per_revocation_when_the_rollback_fails(
    tmp_path: Path, caplog
) -> None:
    """#1314: the R3 "rolled back and revoked" line must not follow the truth."""
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    proxy = _RollbackFails(writer._conn, failures=2)
    writer._conn = proxy
    unit = sw._Unit("plugin-b", 0.05)
    writer._conn.execute("BEGIN IMMEDIATE")
    try:
        caplog.set_level(logging.ERROR)
        writer._revoke(unit)
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1, errors
        assert "restart the daemon" in errors[0]
        assert "rolled back and revoked" not in errors[0]
        # The caller is told the right cause and the right remedy.
        with pytest.raises(WriteAbandoned) as excinfo:
            await writer.write(lambda cur: cur.execute("SELECT 1"), owner="plugin-c")
        message = str(excinfo.value)
        assert "restart the daemon" in message
        assert "plugins.disabled" not in message
        assert "ignores interrupt()" not in message
    finally:
        writer._conn._conn.close()


async def test_every_abandoned_refusal_carries_the_writers_cause(tmp_path: Path) -> None:
    """#1343 review: `cause` must reach all three raise sites, not one."""
    import corvidae.session_writer as sw

    writer = SessionWriter.open(tmp_path / "sessions.db")
    writer._abandoned_owner = "plugin-b"
    writer._abandon_cause = "rollback"
    writer._abandoned = True
    unit = sw._Unit("plugin-b", 0.05)
    try:
        # Entry path (`write()` -> `_refuse_if_abandoned`).
        with pytest.raises(WriteAbandoned) as entry:
            await writer.write(lambda cur: cur.execute("SELECT 1"), owner="plugin-b")
        # Outcome path (`_outcome`), reached by a unit already inside the writer.
        with pytest.raises(WriteAbandoned) as outcome:
            writer._outcome(unit)
    finally:
        writer._conn.close()
    for excinfo in (entry, outcome):
        assert "restart the daemon" in str(excinfo.value)
        assert "plugins.disabled" not in str(excinfo.value)


async def test_a_write_after_close_is_refused_by_name(tmp_path: Path) -> None:
    """#1328 item 3: a late write must name the writer, not fail inside SQLite."""
    writer = SessionWriter.open(tmp_path / "sessions.db")
    await writer.close()

    ran: list[str] = []

    def body(cursor: WriteCursor) -> None:
        ran.append("ran")

    with pytest.raises(WriteClosed) as excinfo:
        await writer.write(body, owner="plugin-b")
    message = str(excinfo.value)
    assert "plugin-b" in message
    assert str(tmp_path / "sessions.db") in message
    assert "closed" in message
    assert ran == [], "a refused unit must not run"


async def test_a_queued_write_is_refused_when_close_wins_the_route(
    tmp_path: Path,
) -> None:
    """#1328 item 3: a unit that queued before close() must not begin on a
    closed connection — SQLite's "Cannot operate on a closed database" is the
    text WriteClosed exists to replace."""
    writer = SessionWriter.open(tmp_path / "sessions.db")
    lock = writer._lock()
    await lock.acquire()  # a unit ahead holds the route

    ran: list[str] = []

    def body(cursor: WriteCursor) -> None:
        ran.append("ran")

    try:
        queued = asyncio.create_task(
            writer.write(body, owner="plugin-b", deadline_s=1.0)
        )
        await asyncio.sleep(0.1)  # plugin-b passed the entry check and queued
        assert writer._closed is False
        writer._closed = True  # what close() sets before it releases the route
        lock.release()
        with pytest.raises(WriteClosed) as excinfo:
            await asyncio.wait_for(queued, timeout=2.0)
        assert "plugin-b" in str(excinfo.value)
        assert ran == [], "a refused unit was allowed to run"
    finally:
        if lock.locked():
            lock.release()
        writer._conn.close()


async def test_load_extension_after_close_is_refused_by_name(tmp_path: Path) -> None:
    """#1355: a load after close names the writer, not SQLite's closed-db text."""
    writer = SessionWriter.open(tmp_path / "sessions.db")
    await writer.close()

    with pytest.raises(WriteClosed) as excinfo:
        await writer.load_extension("libnonexistent.so", owner="plugin-c")

    message = str(excinfo.value)
    assert "plugin-c" in message
    assert str(tmp_path / "sessions.db") in message
    assert "closed" in message


async def test_set_journal_mode_after_close_is_refused_by_name(tmp_path: Path) -> None:
    """#1355: a mode change after close refuses by name, like the other two."""
    writer = SessionWriter.open(tmp_path / "sessions.db")
    await writer.close()

    with pytest.raises(WriteClosed) as excinfo:
        await writer.set_journal_mode("WAL")

    message = str(excinfo.value)
    # No caller-supplied owner on this method, so it names itself.
    assert "set_journal_mode" in message
    assert str(tmp_path / "sessions.db") in message
    assert "closed" in message


async def test_a_queued_extension_load_is_refused_when_close_wins_the_route(
    tmp_path: Path,
) -> None:
    """#1355 review, F3: the guard has to be *under* the route lock, not before
    it. A call that queued ahead of close() must not reach a closed connection —
    and a guard hoisted outside the lock leaves every after-close test green."""
    writer = SessionWriter.open(tmp_path / "sessions.db")
    lock = writer._lock()
    await lock.acquire()

    try:
        queued = asyncio.create_task(
            writer.load_extension("libnonexistent.so", owner="plugin-d")
        )
        await asyncio.sleep(0.1)  # queued behind the route
        assert writer._closed is False
        writer._closed = True  # what close() sets before it releases the route
        lock.release()
        with pytest.raises(WriteClosed) as excinfo:
            await asyncio.wait_for(queued, timeout=2.0)
        assert "plugin-d" in str(excinfo.value)
    finally:
        if lock.locked():
            lock.release()
        writer._conn.close()


async def test_a_queued_journal_mode_is_refused_when_close_wins_the_route(
    tmp_path: Path,
) -> None:
    """The same placement check for set_journal_mode."""
    writer = SessionWriter.open(tmp_path / "sessions.db")
    lock = writer._lock()
    await lock.acquire()

    try:
        queued = asyncio.create_task(writer.set_journal_mode("WAL"))
        await asyncio.sleep(0.1)
        assert writer._closed is False
        writer._closed = True
        lock.release()
        with pytest.raises(WriteClosed) as excinfo:
            await asyncio.wait_for(queued, timeout=2.0)
        assert "set_journal_mode" in str(excinfo.value)
    finally:
        if lock.locked():
            lock.release()
        writer._conn.close()
