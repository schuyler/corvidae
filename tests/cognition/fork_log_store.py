"""A ForkLog over the real write route, for the fork_log tests (#1211)."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from corvidae.persistence import PersistencePlugin
from corvidae_cognition.fork_log import ForkLog
from helpers import open_test_store


async def open_log(path: str | Path, **kwargs) -> ForkLog:
    """A ForkLog on ``path`` through a hand-built persistence plugin.

    The log is not opened; ``log.seed_db`` is a read-write autocommit
    connection with named-row access, for seeding and for raw reads. Closing
    ``log.seed_db`` closes the shared reader and writer too, so the teardown
    is ``await log.close(); await log.seed_db.close()``.
    """
    db, reader, writer = await open_test_store(path)
    db.row_factory = aiosqlite.Row
    persistence = PersistencePlugin()
    persistence.db, persistence.writer = reader, writer
    log = ForkLog(lambda: persistence, **kwargs)
    log.seed_db = db  # type: ignore[attr-defined]
    return log


async def close_log(log: ForkLog) -> None:
    """Close the log and every connection ``open_log`` made for it."""
    await log.close()
    await log.seed_db.close()  # type: ignore[attr-defined]
