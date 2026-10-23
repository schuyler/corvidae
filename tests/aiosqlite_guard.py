"""Close aiosqlite connections a test leaves open (GitHub issue #64).

aiosqlite runs each connection on a non-daemon worker thread that exits only
when the connection is closed or stopped. pytest keeps the last failing
test's traceback in sys.last_traceback, so a connection that test left open
is never garbage-collected, and the interpreter waits on its thread forever
at exit. pytest-timeout does not cover interpreter shutdown.

The fixture records every connection opened through aiosqlite.connect during
a test and closes any still open at teardown, on the test's own event loop.
"""

import logging

import aiosqlite
import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(autouse=True)
async def close_leaked_aiosqlite_connections(monkeypatch: pytest.MonkeyPatch):
    """Track aiosqlite connections opened by the test; close leftovers at teardown."""
    opened: list[aiosqlite.Connection] = []
    real_connect = aiosqlite.connect

    # Record each Connection. It is both awaitable and an async context
    # manager, so returning the same object keeps every call style working.
    def tracking_connect(*args, **kwargs) -> aiosqlite.Connection:
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    # Every call site uses the module attribute (aiosqlite.connect), so
    # patching it covers both test code and corvidae code under test.
    monkeypatch.setattr(aiosqlite, "connect", tracking_connect)

    yield

    # Close connections the test left open. _connection is aiosqlite's
    # handle to the sqlite3 connection; it is None once closed (or if the
    # connection was never awaited), and there is no public equivalent.
    for conn in opened:
        if conn._connection is None:
            continue
        try:
            await conn.close()
        except Exception:
            # Closing failed (e.g. a statement still running): log it and
            # stop the worker thread anyway so the process can exit.
            logger.warning("could not close leaked aiosqlite connection; stopping it", exc_info=True)
            conn.stop()
