"""Lazily opened, rate-limited read-only SQLite probe (#159, #237).

CritiquePlugin (message_fts) and AppraisalPlugin (memory_fts) each read a
table that another plugin creates. Opening the connection once at on_start
meant a table that appeared later (created by another plugin's own startup,
or after the first run) stayed invisible until a restart. ReadOnlyProbe
opens on first use instead and retries at a bounded rate until the table
exists.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import aiosqlite

# Named under "corvidae": corvidae/logging.py configures that logger, and
# __name__ would be corvidae_cognition.probe.
logger = logging.getLogger("corvidae.probe")

# Minimum seconds between open attempts while the table is unavailable.
PROBE_RETRY_SECONDS: float = 30.0


class ReadOnlyProbe:
    """Lazily opened, rate-limited read-only SQLite connection that requires one table."""

    def __init__(
        self,
        path: str,
        table: str,
        owner: str,
        consequence: str,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self.table = table
        self.owner = owner
        self.consequence = consequence
        # Read at call time so tests can swap in a stepping fake clock.
        self.clock = clock
        self._conn: aiosqlite.Connection | None = None
        # None, not 0.0, so a fake clock starting at 0 still attempts first.
        self._last_attempt: float | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._failed_before = False

    @property
    def connection(self) -> aiosqlite.Connection | None:
        """The cached connection, or None. Never opens one."""
        return self._conn

    def _throttled(self) -> bool:
        """True when the last attempt is too recent to try again."""
        return (
            self._last_attempt is not None
            and self.clock() - self._last_attempt < PROBE_RETRY_SECONDS
        )

    async def get(self) -> aiosqlite.Connection | None:
        """Return the connection, opening it if due; None while unavailable."""
        # Fast path: closed, cached, or attempted too recently.
        if self._closed:
            return None
        if self._conn is not None:
            return self._conn
        # While an attempt is in flight (lock held) its stamp is already set,
        # but callers must wait for its outcome rather than return None.
        if not self._lock.locked() and self._throttled():
            return None
        async with self._lock:
            # Re-check under the lock: callers that failed the fast path
            # together make one attempt per interval, not one each.
            if self._closed:
                return None
            if self._conn is not None:
                return self._conn
            if self._throttled():
                return None
            # Recorded at attempt start. A get() cancelled mid-attempt keeps
            # this stamp (no WARNING); the probe recovers on the next interval.
            self._last_attempt = self.clock()
            return await self._attempt()

    async def _attempt(self) -> aiosqlite.Connection | None:
        """One open + table check; caches and returns the connection on success."""
        conn: aiosqlite.Connection | None = None
        try:
            try:
                conn = await aiosqlite.connect(
                    f"file:{self.path}?mode=ro", uri=True
                )
                async with conn.execute(
                    "SELECT name FROM sqlite_master WHERE name = ?",
                    (self.table,),
                ) as cursor:
                    row = await cursor.fetchone()
            except BaseException:
                # Cancellation or any failure: never leave a half-open
                # connection behind (its worker thread would outlive us).
                if conn is not None:
                    await conn.close()
                raise
            if row is None:
                await conn.close()
                self._failed_before = True
                logger.warning(
                    "%s probe unavailable: %s missing in %s; %s; "
                    "retrying automatically every %gs",
                    self.owner, self.table, self.path, self.consequence,
                    PROBE_RETRY_SECONDS,
                )
                return None
        except Exception as exc:
            # Open or query failure: warn with the reason and retry later.
            self._failed_before = True
            logger.warning(
                "%s probe unavailable: cannot open %s read-only (%s); %s; "
                "retrying automatically every %gs",
                self.owner, self.path, type(exc).__name__, self.consequence,
                PROBE_RETRY_SECONDS, exc_info=True,
            )
            return None
        # Success: cache, and announce recovery once if we had failed before.
        self._conn = conn
        if self._failed_before:
            self._failed_before = False
            logger.info(
                "%s probe recovered: %s available in %s",
                self.owner, self.table, self.path,
            )
        return conn

    async def close(self) -> None:
        """Close the connection; later get() calls return None and open nothing."""
        async with self._lock:
            self._closed = True
            if self._conn is not None:
                conn, self._conn = self._conn, None
                await conn.close()
