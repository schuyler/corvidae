"""Outcome log — the per-exchange record table and its writer API.

Phase 0 ships the ``exchange_log`` schema and the writer methods only.
The retrieval-profile columns are populated from Phase 1a (there is no
retrieval to profile before then); origin/appraisal/provenance/outcomes
columns are populated from Phase 2. Nothing writes rows in Phase 0.

The table accumulates one row per exchange so later phases can correlate
retrieval quality, appraisal, and critique/engagement outcomes — the
calibration data for the memory system's evals.
"""

import logging
import time

import aiosqlite

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger(__name__)

EXCHANGE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS exchange_log (
    exchange_key TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    origin TEXT,
    message_rowid INTEGER,
    created_at REAL NOT NULL,
    retrieval_top_score REAL,
    retrieval_hit_count INTEGER,
    probe_score REAL,
    appraisal TEXT,
    provenance_snapshot TEXT,
    outcomes TEXT
)
"""
EXCHANGE_LOG_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_exchange_channel "
    "ON exchange_log (channel_id, created_at)"
)

# Columns update_exchange may touch. The identity columns (exchange_key,
# channel_id, created_at) are immutable after record_exchange; everything
# else is a nullable profile/outcome column filled in by later phases.
UPDATABLE_COLUMNS = frozenset(
    {
        "origin",
        "message_rowid",
        "retrieval_top_score",
        "retrieval_hit_count",
        "probe_score",
        "appraisal",
        "provenance_snapshot",
        "outcomes",
    }
)


class OutcomeLogPlugin(CorvidaePlugin):
    """Owns the exchange_log table and its guarded writer API."""

    depends_on = frozenset({"persistence"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._table_ready = False

    def _resolve_db(self) -> aiosqlite.Connection | None:
        """Return the persistence plugin's DB connection, or None if not open yet."""
        from corvidae.persistence import PersistencePlugin
        persistence = get_dependency(self.pm, "persistence", PersistencePlugin)
        return persistence.db

    async def _ensure_table(self) -> aiosqlite.Connection:
        """Create the exchange_log table on first use and return the DB.

        on_start hook ordering is LIFO across plugins, so the persistence
        DB may not exist yet during our on_start — creation is retried
        lazily on first write. Raises RuntimeError if the DB is missing at
        write time (writer calls are explicit API, not fail-soft hooks).
        """
        db = self._resolve_db()
        if db is None:
            raise RuntimeError("persistence DB not available for exchange_log")
        if not self._table_ready:
            await db.execute(EXCHANGE_LOG_DDL)
            await db.execute(EXCHANGE_LOG_INDEX_DDL)
            await db.commit()
            self._table_ready = True
        return db

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Best-effort early DDL; falls back to lazy creation on first write
        # if the persistence DB hasn't opened yet (hook-ordering dependent).
        try:
            await self._ensure_table()
        except Exception:
            logger.warning("exchange_log table creation deferred", exc_info=True)

    async def record_exchange(
        self,
        exchange_key: str,
        channel_id: str,
        origin: str | None = None,
        message_rowid: int | None = None,
    ) -> None:
        """Insert a new exchange row. Idempotent (INSERT OR IGNORE).

        Args:
            exchange_key: Unique key for the exchange (minted in Phase 2).
            channel_id: The channel the exchange happened on.
            origin: 'user'|'reminder'|'critique'|'heartbeat'|'task'; None
                until Phase 2 populates it.
            message_rowid: message_log.id of the originating message; None
                for gate-rejected exchanges.
        """
        db = await self._ensure_table()
        await db.execute(
            "INSERT OR IGNORE INTO exchange_log "
            "(exchange_key, channel_id, origin, message_rowid, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (exchange_key, channel_id, origin, message_rowid, time.time()),
        )
        await db.commit()

    async def update_exchange(self, exchange_key: str, **columns) -> None:
        """Update named nullable columns of an existing exchange row.

        Only the columns in UPDATABLE_COLUMNS are accepted — SQL is never
        built from arbitrary kwargs.

        Raises:
            ValueError: If a kwarg is not an updatable column.
        """
        unknown = set(columns) - UPDATABLE_COLUMNS
        if unknown:
            raise ValueError(
                f"update_exchange: unknown or immutable column(s): {sorted(unknown)}"
            )
        if not columns:
            return
        db = await self._ensure_table()
        # Column names are validated against the frozen allowlist above, so
        # interpolating them into the SET clause is safe; values are bound.
        set_clause = ", ".join(f"{name} = ?" for name in columns)
        await db.execute(
            f"UPDATE exchange_log SET {set_clause} WHERE exchange_key = ?",
            (*columns.values(), exchange_key),
        )
        await db.commit()
