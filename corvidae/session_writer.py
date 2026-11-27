"""Kernel-owned write route for ``sessions.db`` (#1211).

Every write to the session database is a *unit*: a synchronous function
``fn(cursor) -> T`` that :class:`SessionWriter` runs on a fresh daemon worker
thread inside ``BEGIN IMMEDIATE ... COMMIT``, one unit at a time. The event
loop awaits it with a deadline. A unit that overruns is revoked: its cursor is
poisoned, any running statement is interrupted, the transaction is rolled back
from a kernel thread, an ERROR names the owner and the deadline, and the
caller gets :class:`WriteRevoked`. Because ``fn`` is synchronous it cannot
await a model call while holding the write lock.

Only stdlib ``sqlite3`` plus ``aiosqlite`` (for the reader that
:func:`open_session_store` returns) are used; no plugin is imported.
"""

import asyncio
import enum
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote

from corvidae.shutdown import DEFAULT_SHUTDOWN_TIMEOUTS, stop_bound_seconds

import aiosqlite

logger = logging.getLogger(__name__)

# Deadline a unit gets when the caller names none.
DEFAULT_WRITE_DEADLINE_S: float = 5.0
# Largest deadline any caller may ask for (schema units use it).
MAX_WRITE_DEADLINE_S: float = 30.0
#: How long a revocation may take before the unit is abandoned (#1315).
#: A statement that never reaches SQLite's interrupt check (a read on a hung
#: device, the busy wait on ``COMMIT`` outside WAL) makes the interrupt-and-
#: acquire loop spin with no bound; that loop holds the write route, so shutdown
#: waits with it. Derived from the graceful shutdown budget less a reserve, the
#: way ``mcp_client`` derives its stop bound: a constant above that budget would
#: never elapse inside it, and the diagnostic below would never reach the log.
REVOKE_BOUND_S: float = stop_bound_seconds(DEFAULT_SHUTDOWN_TIMEOUTS.graceful)

# Unit lifecycle states, written only under the unit's lock.
_RUNNING = "running"
_COMMITTED = "committed"
_ROLLED_BACK = "rolled_back"
_REVOKED = "revoked"

T = TypeVar("T")


class WriteRevoked(Exception):
    """A write unit overran its deadline and was rolled back."""

    def __init__(self, owner: str, deadline_s: float) -> None:
        self.owner = owner
        self.deadline_s = deadline_s
        super().__init__(
            f"write unit from {owner} overran its {deadline_s:g} s deadline "
            "and was rolled back"
        )


class WriteAbandoned(WriteRevoked):
    """A unit could not be revoked, so its outcome is unknown (#1315).

    Its statement is not reaching SQLite's interrupt check, so the transaction it
    opened is still open and the connection cannot be used again. The caller is
    told the outcome is unknown rather than that the write was rolled back: on a
    busy-wait abandonment the worker's own ``COMMIT`` may still land.
    """

    def __init__(self, owner: str, path: str) -> None:
        self.owner = owner
        self.path = path
        self.deadline_s = float("nan")
        Exception.__init__(
            self,
            f"write unit from {owner} on {path} could not be revoked: its "
            "statement ignores interrupt(), so whether its write landed is "
            "unknown and this writer refuses further writes until restart — add "
            "the plugin to plugins.disabled and report it",
        )


@dataclass(frozen=True)
class WriteResult:
    """What one statement returned, fetched eagerly."""

    rows: list[tuple]
    lastrowid: int | None
    rowcount: int


class _Unit:
    """Per-unit record shared by the worker, the revoker and the loop."""

    def __init__(self, owner: str, deadline_s: float) -> None:
        self.owner = owner
        self.deadline_s = deadline_s
        # Guards every connection access of this unit and all fields below.
        self.lock = threading.Lock()
        self.revoke_requested = False
        self.state = _RUNNING
        self.result: Any = None
        self.exc: BaseException | None = None


def _normalize_value(owner: str, where: str, value: Any) -> Any:
    """Return ``value`` as an exact type sqlite3 binds without adapting.

    Runs before the unit lock is taken so no plugin-controlled Python runs
    while the worker holds it. Enum members bind by ``.value`` (as sqlite3
    does today); subclasses of the plain types are copied to the base type.
    """
    # Exact plain types pass through untouched.
    if value is None or type(value) in (int, float, str, bytes):
        return value
    # An Enum member binds by its value, normalized again.
    if isinstance(value, enum.Enum):
        return _normalize_value(owner, where, value.value)
    # Buffers copy to bytes.
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    # Subclasses copy to their base type (bool is an int subclass).
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, bytes):
        return bytes(value)
    raise TypeError(
        f"write unit from {owner}: parameter {where} has unsupported type "
        f"{type(value).__name__}; convert it to None, int, float, str or "
        "bytes before the call"
    )


def _plain_str(owner: str, value: Any, what: str) -> str:
    """Copy a str (or str subclass) to a plain str, or raise naming the owner."""
    # A non-str here would fail inside str.__str__ with a message naming
    # neither the unit's owner nor the remedy.
    if not isinstance(value, str):
        raise TypeError(
            f"write unit from {owner}: {what} must be a str, got "
            f"{type(value).__name__}; pass SQL text and string parameter names"
        )
    return str.__str__(value)


def _normalize_params(owner: str, params: Any) -> Any:
    """Materialize one parameter set as a tuple or plain dict of plain values."""
    if isinstance(params, Mapping):
        return {
            _plain_str(owner, k, "a parameter name"): _normalize_value(owner, repr(k), v)
            for k, v in params.items()
        }
    return tuple(
        _normalize_value(owner, str(i), v) for i, v in enumerate(params)
    )


class WriteCursor:
    """What a unit receives: statements only, never the raw connection."""

    def __init__(self, conn: sqlite3.Connection, unit: _Unit) -> None:
        self._conn = conn
        self._unit = unit

    def _enter(self) -> None:
        """Acquire the unit lock and check the unit is still live."""
        # Caller releases the lock; checks run under it.
        self._unit.lock.acquire()
        unit = self._unit
        if unit.revoke_requested:
            unit.lock.release()
            raise WriteRevoked(unit.owner, unit.deadline_s)
        if unit.state != _RUNNING:
            unit.lock.release()
            raise RuntimeError(
                f"write cursor from {unit.owner} used after its unit ended; "
                "do all writes inside the unit function"
            )

    def execute(self, sql: str, params: Sequence | Mapping = ()) -> WriteResult:
        """Run one statement and return every row it produced."""
        # Normalize before the lock so only SQLite runs under it.
        owner = self._unit.owner
        sql = _plain_str(owner, sql, "the SQL statement")
        norm = _normalize_params(owner, params)
        self._enter()
        try:
            cur = self._conn.execute(sql, norm)
            rows = [tuple(r) for r in cur.fetchall()]
            return WriteResult(rows, cur.lastrowid, cur.rowcount)
        finally:
            self._unit.lock.release()

    def executemany(
        self, sql: str, seq: Iterable[Sequence | Mapping]
    ) -> WriteResult:
        """Run one DML statement per parameter set."""
        # Materialize the iterable and every set before the lock.
        owner = self._unit.owner
        sql = _plain_str(owner, sql, "the SQL statement")
        sets = [_normalize_params(owner, p) for p in list(seq)]
        self._enter()
        try:
            cur = self._conn.executemany(sql, sets)
            return WriteResult([], cur.lastrowid, cur.rowcount)
        finally:
            self._unit.lock.release()


class SessionWriter:
    """The one read-write connection to ``sessions.db``."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        path: str | Path = "sessions.db",
        *,
        revoke_bound_s: float = REVOKE_BOUND_S,
    ) -> None:
        self._conn = conn
        # Named in the revocation ERRORs, where the issue asks for the path.
        self._path = str(path)
        #: How long a revocation may take for this writer. Derived from the
        #: configured ``daemon.shutdown_timeout`` at open time, so the bound
        #: always fits inside the phase it sits in (#1315).
        self._revoke_bound_s = revoke_bound_s
        # Serializes units FIFO on the loop; created lazily on first use so
        # the writer can be opened from synchronous code (R6 test).
        self._alock: asyncio.Lock | None = None
        self._closed = False
        # The most recent unit; close() ends it if a cancelled loop left it running.
        self._last_unit: _Unit | None = None
        #: Set when a revocation gave up: the worker still holds the connection
        #: inside a statement, so closing it here would block the loop (#1315).
        self._abandoned = False
        self._abandoned_owner = ""

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int | None = None,
        graceful_s: float | None = None,
    ) -> "SessionWriter":
        """Open the writer connection, refusing a non-serialized SQLite (R6).

        ``busy_timeout_ms`` is how long a unit's BEGIN waits for an
        out-of-process writer's lock; None means
        ``corvidae.persistence.DEFAULT_BUSY_TIMEOUT_MS``.
        """
        # Read at call time so tests can patch it.
        level = sqlite3.threadsafety
        if level != 3:
            raise RuntimeError(
                f"SQLite is not built serialized (sqlite3.threadsafety == "
                f"{level}, need 3); the session write route cannot roll back "
                "a stalled plugin safely — run corvidae on a Python whose "
                "sqlite3 module is built with SQLITE_THREADSAFE=1"
            )
        # Local import: persistence imports this module at load time.
        revoke_bound_s = (
            REVOKE_BOUND_S if graceful_s is None else stop_bound_seconds(graceful_s)
        )
        if busy_timeout_ms is None:
            from corvidae.persistence import DEFAULT_BUSY_TIMEOUT_MS

            busy_timeout_ms = DEFAULT_BUSY_TIMEOUT_MS
        conn = sqlite3.connect(
            str(path),
            isolation_level=None,
            check_same_thread=False,
            timeout=busy_timeout_ms / 1000,
        )
        # The wait is the first statement on the connection, ahead of
        # journal_mode and the schema, which are writes themselves (#1291).
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        except BaseException:
            conn.close()
            raise
        return cls(conn, path, revoke_bound_s=revoke_bound_s)

    def _lock(self) -> asyncio.Lock:
        """The asyncio lock, created on first use inside a running loop."""
        if self._alock is None:
            self._alock = asyncio.Lock()
        return self._alock

    def _refuse_if_abandoned(self) -> None:
        """Refuse every write after an abandonment, naming why (#1315).

        Left to SQLite, the next unit fails with "cannot start a transaction
        within a transaction", which names neither the abandonment nor the owner.
        """
        if self._abandoned:
            raise WriteAbandoned(self._abandoned_owner, self._path)

    async def write(
        self,
        fn: Callable[[WriteCursor], T],
        *,
        owner: str,
        deadline_s: float = DEFAULT_WRITE_DEADLINE_S,
    ) -> T:
        """Run ``fn`` as one transactional unit with a deadline."""
        self._refuse_if_abandoned()
        # Reject a deadline outside the accepted range.
        if not 0 < deadline_s <= MAX_WRITE_DEADLINE_S:
            raise ValueError(
                f"write unit from {owner}: deadline_s={deadline_s!r} is "
                f"outside (0, {MAX_WRITE_DEADLINE_S:g}] seconds; pass a "
                "smaller deadline or split the work into several units"
            )
        # The internal task owns the lock for the whole unit and runs under
        # shield, so a cancelled caller neither cuts the unit short nor frees
        # the lock while a worker still holds the transaction.
        async def locked() -> T:
            async with self._lock():
                return await self._run_unit(fn, owner, deadline_s)

        return await self._shielded(locked())

    @staticmethod
    async def _shielded(coro: Any) -> Any:
        """Await ``coro`` as a task a cancelled caller cannot cut short.

        The task keeps the writer lock until its blocking work ends. If the
        caller was cancelled, nobody awaits the task, so a done-callback
        retrieves its exception (already logged where it arose) instead of
        leaving "Task exception was never retrieved" noise.
        """
        task = asyncio.ensure_future(coro)

        # Consume an outcome nobody is left to await.
        def _retrieve(t: asyncio.Future) -> None:
            if not t.cancelled() and t.exception() is not None:
                logger.debug("write task ended with %r after its caller left", t.exception())

        task.add_done_callback(_retrieve)
        return await asyncio.shield(task)

    async def _run_unit(
        self, fn: Callable[[WriteCursor], T], owner: str, deadline_s: float
    ) -> T:
        """Start the worker, wait for it, revoke it at the deadline."""
        loop = asyncio.get_running_loop()
        unit = _Unit(owner, deadline_s)
        self._last_unit = unit
        began: asyncio.Future[None] = loop.create_future()
        done: asyncio.Future[None] = loop.create_future()

        def signal(fut: asyncio.Future[None]) -> None:
            """Wake the loop; ignore a closed loop or a resolved future."""
            def _set() -> None:
                if not fut.done():
                    fut.set_result(None)
            try:
                loop.call_soon_threadsafe(_set)
            except RuntimeError:
                logger.debug("loop closed before write unit signal (%s)", owner)

        threading.Thread(
            target=self._worker,
            args=(fn, unit, began, done, signal),
            name=f"session-write-{owner}",
            daemon=True,
        ).start()

        # Wait for BEGIN to succeed or the unit to end (a failed BEGIN).
        await asyncio.wait({began, done}, return_when=asyncio.FIRST_COMPLETED)
        if not began.done():
            return self._outcome(unit)
        # The deadline clock starts at "began".
        await asyncio.wait({done}, timeout=deadline_s)
        if done.done():
            return self._outcome(unit)
        # Overrun: revoke on a dedicated daemon thread.
        revoked: asyncio.Future[None] = loop.create_future()

        def revoke_thread() -> None:
            try:
                self._revoke(unit)
            except BaseException as exc:  # noqa: BLE001 - relayed to the awaiting loop
                logger.exception("revoking write unit from %s failed", owner)
                unit.exc = exc
            finally:
                signal(revoked)

        threading.Thread(
            target=revoke_thread, name=f"session-revoke-{owner}", daemon=True
        ).start()
        await revoked
        return self._outcome(unit)

    def _outcome(self, unit: _Unit) -> T:
        """Return the unit's recorded result or raise its recorded outcome.

        Never waits unboundedly for the unit's lock. On an abandoned writer it
        does not wait at all: the worker holds that lock inside a C call, so a
        plain ``with unit.lock`` here would wedge the event loop and disarm
        shutdown's force-exit, turning a bounded ugly exit into an unbounded hang
        (#1315). On a healthy writer it does wait — bounded in practice, because
        the only reason the lock is unavailable there is a worker that has
        already finished and is about to release it (#1328).
        """
        if not unit.lock.acquire(blocking=False):
            if not self._abandoned:
                # A healthy writer reaches here when the worker signalled `done`
                # from inside the lock (a failed BEGIN) and has not released it
                # yet: wait for that release and report the worker's real error,
                # not "this writer refuses writes until restart" (#1328).
                unit.lock.acquire()
            else:
                # Abandoned: the worker holds this lock inside a C call, so the
                # wait would be unbounded. Its outcome is unknown, not rolled
                # back — its own COMMIT may still land.
                raise WriteAbandoned(unit.owner, self._path)
        try:
            state, exc, result = unit.state, unit.exc, unit.result
        finally:
            unit.lock.release()
        # Abandonment decides first, whatever state the unit reads: the give-up
        # branch writes none, so this unit can read _RUNNING, _COMMITTED if the
        # worker's own COMMIT later landed, or _REVOKED from an earlier unit on
        # the same writer. None of those may hand the caller a result or tell it
        # the write was rolled back (#1315). No identity test, and this is why it
        # is safe to drop rather than because it could never matter: `_outcome`
        # is reachable only from `_run_unit`, which assigns `_last_unit` while it
        # holds the route lock, so no other unit can be `_last_unit` here. A unit
        # queued behind the one that was abandoned does reach this method on a
        # writer that was healthy when it called `write()` — the refusal check at
        # entry runs before the route lock — and for that unit the flag is still
        # the right answer: its own answer is unknown too.
        if self._abandoned:
            raise WriteAbandoned(unit.owner, self._path)
        if state == _REVOKED:
            raise WriteRevoked(unit.owner, unit.deadline_s)
        if exc is not None:
            raise exc
        return result

    def _worker(
        self,
        fn: Callable[[WriteCursor], Any],
        unit: _Unit,
        began: "asyncio.Future[None]",
        done: "asyncio.Future[None]",
        signal: Callable[["asyncio.Future[None]"], None],
    ) -> None:
        """Run BEGIN, ``fn`` and COMMIT, touching the connection only under lock."""
        conn = self._conn
        # Step 1: BEGIN IMMEDIATE under the lock.
        with unit.lock:
            # Revoked before it began (shutdown): never touch the connection.
            if unit.revoke_requested:
                return
            try:
                conn.execute("BEGIN IMMEDIATE")
            except BaseException as exc:  # noqa: BLE001 - recorded and returned to caller
                unit.state, unit.exc = _ROLLED_BACK, exc
                signal(done)
                return
        signal(began)
        # Step 2: the unit's own work; exceptions are recorded below.
        result: Any = None
        error: BaseException | None = None
        try:
            result = fn(WriteCursor(conn, unit))
        except BaseException as exc:  # noqa: BLE001 - rolled back and propagated
            error = exc
        # Step 3: end the transaction unless the revoker owns it.
        with unit.lock:
            if unit.revoke_requested:
                return
            try:
                if error is None:
                    conn.execute("COMMIT")
                    unit.state, unit.result = _COMMITTED, result
                else:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    unit.state, unit.exc = _ROLLED_BACK, error
            except BaseException as exc:  # noqa: BLE001 - see below
                if error is None:
                    # COMMIT failed: it did not commit; roll back.
                    try:
                        if conn.in_transaction:
                            conn.execute("ROLLBACK")
                    except BaseException:  # noqa: BLE001 - revoker finishes it
                        logger.debug(
                            "worker ROLLBACK failed for %s; revoker ends it",
                            unit.owner, exc_info=True,
                        )
                        return
                    unit.state, unit.exc = _ROLLED_BACK, exc
                else:
                    # Worker's own ROLLBACK raised: leave RUNNING for the revoker.
                    logger.debug(
                        "worker ROLLBACK failed for %s; revoker ends it",
                        unit.owner, exc_info=True,
                    )
                    return
        signal(done)

    def _end_transaction(
        self, unit: _Unit, *, bound_s: float | None = None
    ) -> bool | None:
        """Interrupt and roll back ``unit`` unless it already finished.

        Returns True when this call revoked it, False when the unit had already
        finished, and None when ``bound_s`` expired first — the unit's statement
        is not reaching SQLite's interrupt check, so waiting longer cannot help
        (#1315). Runs on a daemon thread: it loops ``interrupt()`` until it holds
        the unit lock, so it works even while the unit is inside a long statement.
        """
        conn = self._conn
        # Step 1: flag first so the worker's next lock acquisition raises.
        unit.revoke_requested = True
        # Step 2: interrupt until the lock is ours, or the caller's bound ends.
        started = time.monotonic()
        while True:
            conn.interrupt()
            if unit.lock.acquire(timeout=0.05):
                break
            if bound_s is not None and time.monotonic() - started >= bound_s:
                return None
        try:
            # Step 3: a terminal state means the unit finished at the wire,
            # unless the rollback was caused by our own interrupt on COMMIT.
            caused_by_us = (
                unit.state == _ROLLED_BACK
                and isinstance(unit.exc, sqlite3.OperationalError)
                and "interrupted" in str(unit.exc)
            )
            if unit.state in (_COMMITTED, _ROLLED_BACK) and not caused_by_us:
                return False
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            unit.state, unit.exc = _REVOKED, None
            return True
        finally:
            unit.lock.release()

    def _revoke(self, unit: _Unit) -> None:
        """Revoke an overrun unit and log the R3 ERROR (daemon thread)."""
        outcome = self._end_transaction(unit, bound_s=self._revoke_bound_s)
        if outcome is None:
            # The statement is not reaching interrupt(); the loop would hold the
            # write route forever, which is what made shutdown hang (#1315).
            logger.error(
                "write unit from %r did not yield to revocation within %g s on "
                "%s; its statement ignores interrupt(), so it is abandoned and "
                "its transaction is left open — this writer now refuses every "
                "further write (WriteAbandoned) and the connection is left to "
                "process exit rather than closed under a running worker",
                unit.owner, self._revoke_bound_s, self._path,
                extra={"owner": unit.owner},
            )
            # No state write here: the lock is held by the worker, and this
            # module writes unit state only under it. The flag is the record.
            self._abandoned_owner = unit.owner
            self._abandoned = True
            return
        if not outcome:
            return
        # Step 4: R3 ERROR naming the owner and the deadline.
        logger.error(
            "write unit from %r overran its %g s deadline on %s; rolled back "
            "and revoked, so its write was discarded — if this recurs, add "
            "the plugin to plugins.disabled and report it",
            unit.owner, unit.deadline_s, self._path,
            extra={"owner": unit.owner, "deadline_s": unit.deadline_s},
        )

    async def _blocking(self, fn: Callable[[], T]) -> T:
        """Run a blocking connection call on a fresh daemon thread."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()

        def run() -> None:
            # Relay result or exception to the loop, tolerating a closed loop.
            def deliver(value: Any = None, error: BaseException | None = None) -> None:
                if fut.done():
                    return
                if error is not None:
                    fut.set_exception(error)
                else:
                    fut.set_result(value)

            try:
                value = fn()
            except BaseException as exc:  # noqa: BLE001 - relayed to the caller
                relay = lambda e=exc: deliver(error=e)  # noqa: E731
            else:
                relay = lambda v=value: deliver(v)  # noqa: E731
            try:
                loop.call_soon_threadsafe(relay)
            except RuntimeError:
                logger.debug("loop closed before blocking call finished")

        threading.Thread(target=run, daemon=True, name="session-writer-call").start()
        return await fut

    async def load_extension(self, path: str, *, owner: str) -> None:
        """Load a SQLite extension on the writer, outside any transaction."""
        def load() -> None:
            self._conn.enable_load_extension(True)
            try:
                self._conn.load_extension(path)
            finally:
                self._conn.enable_load_extension(False)

        # Shielded like write(): a cancelled caller must not free the lock
        # while the blocking load still runs on the connection.
        async def locked() -> None:
            async with self._lock():
                logger.debug("loading sqlite extension %s for %s", path, owner)
                await self._blocking(load)

        await self._shielded(locked())

    async def set_journal_mode(self, mode: str) -> str:
        """Apply a journal mode on the writer and return the mode in effect."""
        def apply() -> str:
            row = self._conn.execute(f"PRAGMA journal_mode={mode}").fetchone()
            return str(row[0]) if row else mode

        # Shielded for the same reason as load_extension.
        async def locked() -> str:
            async with self._lock():
                return await self._blocking(apply)

        return await self._shielded(locked())

    async def close(self) -> None:
        """Wait for the in-flight unit, then close the connection.

        Bounded: a unit whose statement ignores ``interrupt()`` keeps its
        shielded task — and therefore the route lock — alive forever, so the
        wait for the lock has to be bounded too, not only the revoke behind it
        (#1315). Past the bound the writer logs, marks itself closed and leaves
        the connection to process exit.
        """
        lock = self._lock()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self._revoke_bound_s)
        except (asyncio.TimeoutError, TimeoutError):
            self._closed = True
            owner = self._last_unit.owner if self._last_unit else "unknown"
            logger.error(
                "the write route was still held %g s into shutdown on %s by a "
                "unit from %r; that unit is not yielding to interrupt(), so the "
                "connection is left open for process exit — if this recurs, add "
                "the plugin to plugins.disabled and report it",
                self._revoke_bound_s, self._path, owner, extra={"owner": owner},
            )
            return
        try:
            if self._closed:
                return
            self._closed = True
            if self._abandoned:
                # A previous revocation gave up: the connection is still inside a
                # statement the worker cannot leave, and sqlite3.close() waits for
                # it. Closing here blocks the loop and disarms shutdown's
                # force-exit (#1315); the log line from _revoke already named the
                # owner, so leave the connection to process exit.
                return
            # A unit still running here means its awaiting task was cancelled
            # (shutdown): end it first, since closing the connection under a
            # running worker crashes the interpreter.
            unit = self._last_unit
            if unit is not None and unit.state == _RUNNING:
                outcome = await self._blocking(
                    lambda: self._end_transaction(unit, bound_s=self._revoke_bound_s)
                )
                if outcome is None:
                    # The statement ignores interrupt(); waiting cannot end it.
                    # Leave the connection to process exit rather than close it
                    # under a running worker (#1315, and #1310's crash).
                    logger.error(
                        "write unit from %r did not yield to revocation within "
                        "%g s at shutdown on %s; its statement ignores "
                        "interrupt(), so the connection is left open for process "
                        "exit instead of being closed under a running worker",
                        unit.owner, self._revoke_bound_s, self._path,
                        extra={"owner": unit.owner},
                    )
                    return
                if outcome:
                    logger.warning(
                        "write unit from %r was still running at shutdown; "
                        "rolled back, its write was discarded", unit.owner,
                    )
            # Off the loop: sqlite3.Connection.close() waits for a statement
            # still inside the connection, so a wrong guard here must not be able
            # to wedge the event loop (#1315, three rounds of it).
            await self._blocking(self._conn.close)
        finally:
            lock.release()


async def open_session_store(
    path: str | Path,
    *,
    journal_mode: str = "wal",
    busy_timeout_ms: int | None = None,
    graceful_s: float | None = None,
) -> tuple[aiosqlite.Connection, SessionWriter]:
    """Open writer, apply journal mode, run schema, then open the reader.

    The writer's busy timeout (``busy_timeout_ms``, None for the default) is
    set first. The mode is applied before the reader exists because leaving
    WAL while another connection is open fails with "database is locked".
    """
    # Local import: persistence imports this module for its constants.
    from corvidae.persistence import MESSAGE_LOG_SCHEMA

    writer = SessionWriter.open(
        path, busy_timeout_ms=busy_timeout_ms, graceful_s=graceful_s
    )
    try:
        await writer.set_journal_mode(journal_mode)

        # Schema unit: one execute per statement.
        def schema(cur: WriteCursor) -> None:
            for stmt in MESSAGE_LOG_SCHEMA:
                cur.execute(stmt)

        await writer.write(
            schema, owner="persistence", deadline_s=MAX_WRITE_DEADLINE_S
        )
        # Reader: read-only URI, autocommit so a stray DML pins no snapshot.
        uri = f"file:{quote(str(Path(path).resolve()))}?mode=ro"
        reader = await aiosqlite.connect(uri, uri=True, isolation_level=None)
    except BaseException:
        await writer.close()
        raise
    return reader, writer
