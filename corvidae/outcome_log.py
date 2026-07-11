"""Outcome log — the per-exchange record table and its writer API.

Phase 0 ships the ``exchange_log`` schema and the writer methods only.
The retrieval-profile columns are populated from Phase 1a (there is no
retrieval to profile before then); origin/appraisal/provenance/outcomes
columns are populated from Phase 2. Nothing writes rows in Phase 0.

The table accumulates one row per exchange so later phases can correlate
retrieval quality, appraisal, and critique/engagement outcomes — the
calibration data for the memory system's evals.
"""

import json
import logging
import time

import aiosqlite

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger(__name__)

# Columns whose values are JSON envelopes merged via SQLite's json_patch()
# rather than overwritten. Concurrent fire-and-forget writers touch these
# columns with disjoint or overlapping top-level keys; the merge must be a
# single atomic SQL statement (no read-then-write) so no writer's key is
# lost to a lost update (WP2.1 point 7).
MERGE_COLUMNS = frozenset({"outcomes", "appraisal"})

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

    # ------------------------------------------------------------------
    # Hook consumers (WP2.1 point 7) — all fail-soft (log + continue),
    # since these are hooks now, not explicit writer calls; exceptions
    # must not propagate into the turn.
    # ------------------------------------------------------------------

    @hookimpl
    async def on_message_admitted(self, channel, exchange_key: str, sender: str, text: str) -> None:
        try:
            await self.record_exchange(exchange_key, channel.id, origin="user")
        except Exception:
            logger.warning("on_message_admitted: record_exchange failed", exc_info=True)

    @hookimpl
    async def on_message_rejected(self, channel, exchange_key: str, sender: str, text: str) -> None:
        try:
            await self.record_exchange(exchange_key, channel.id, origin="user")
            await self.update_exchange(exchange_key, outcomes={"gate": "rejected"})
        except Exception:
            logger.warning("on_message_rejected: record_exchange failed", exc_info=True)

    @hookimpl
    async def on_message_persisted(
        self, channel, exchange_key: str, rowid: int, origin: str | None
    ) -> None:
        try:
            # INSERT OR IGNORE covers notification-born exchanges (no prior
            # on_message_admitted for those) — origin lands on first insert.
            await self.record_exchange(exchange_key, channel.id, origin=origin)
            await self.update_exchange(exchange_key, message_rowid=rowid)
        except Exception:
            logger.warning("on_message_persisted: record_exchange failed", exc_info=True)

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

    def _validate_columns(self, columns: dict) -> None:
        """Shared validation for update_exchange/upsert_exchange column kwargs.

        Raises:
            ValueError: If a kwarg is not an updatable column, if a
                merge-column (outcomes/appraisal) value is not a dict, or
                if a plain-set column value is a dict.
        """
        unknown = set(columns) - UPDATABLE_COLUMNS
        if unknown:
            raise ValueError(
                f"unknown or immutable column(s): {sorted(unknown)}"
            )
        for name, value in columns.items():
            if name in MERGE_COLUMNS:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{name!r} is a merge column (JSON envelope) and "
                        f"requires a dict fragment, got {type(value).__name__}"
                    )
            elif isinstance(value, dict):
                raise ValueError(
                    f"{name!r} is a plain-set column and cannot take a "
                    f"dict value (only outcomes/appraisal merge)"
                )

    def _build_set_clause(self, columns: dict) -> tuple[str, list]:
        """Build the SET clause and bound params for an UPDATE over columns.

        Merge columns (outcomes/appraisal) are patched atomically via
        SQLite's json_patch() in a single statement — never read into
        Python first. Plain-set columns bind normally. Column names are
        validated against the frozen allowlist by _validate_columns, so
        interpolating them into the SET clause is safe.
        """
        set_parts = []
        params: list = []
        for name, value in columns.items():
            if name in MERGE_COLUMNS:
                set_parts.append(f"{name} = json_patch(COALESCE({name}, '{{}}'), ?)")
                params.append(json.dumps(value))
            else:
                set_parts.append(f"{name} = ?")
                params.append(value)
        return ", ".join(set_parts), params

    async def update_exchange(self, exchange_key: str, **columns) -> None:
        """Update named nullable columns of an existing exchange row.

        Only the columns in UPDATABLE_COLUMNS are accepted — SQL is never
        built from arbitrary kwargs. Merge-columns (outcomes, appraisal)
        are dicts merged atomically in-database via json_patch — a single
        UPDATE statement, no SELECT (RFC 7386 deep-merge: disjoint keys
        from concurrent writers all survive; same-key writers deep-merge).

        Raises:
            ValueError: If a kwarg is not an updatable column, a merge
                column is given a scalar, or a plain-set column is given
                a dict.
        """
        self._validate_columns(columns)
        if not columns:
            return
        db = await self._ensure_table()
        set_clause, params = self._build_set_clause(columns)
        await db.execute(
            f"UPDATE exchange_log SET {set_clause} WHERE exchange_key = ?",
            (*params, exchange_key),
        )
        await db.commit()

    async def upsert_exchange(
        self,
        exchange_key: str,
        channel_id: str,
        origin: str | None = None,
        **columns,
    ) -> None:
        """Create-or-update an exchange row in one call.

        Semantics: INSERT OR IGNORE the identity row (like record_exchange),
        then a single guarded UPDATE applying **columns (same atomic
        json_patch merge as update_exchange for outcomes/appraisal). For
        gate-time writers that run before or race on_message_admitted /
        on_message_persisted's inserts — without the upsert, those inserts'
        INSERT OR IGNORE would silently no-op against a row that doesn't
        exist yet, and a plain UPDATE would no-op too, losing the gate-path
        columns. Idempotent and write-order independent: the hook-driven
        INSERT OR IGNORE that runs later stays correct against a row this
        upsert already created.

        Raises:
            ValueError: Same as update_exchange.
        """
        self._validate_columns(columns)
        db = await self._ensure_table()
        await db.execute(
            "INSERT OR IGNORE INTO exchange_log "
            "(exchange_key, channel_id, origin, message_rowid, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (exchange_key, channel_id, origin, None, time.time()),
        )
        merged_columns = dict(columns)
        if origin is not None:
            merged_columns.setdefault("origin", origin)
        if merged_columns:
            set_clause, params = self._build_set_clause(merged_columns)
            await db.execute(
                f"UPDATE exchange_log SET {set_clause} WHERE exchange_key = ?",
                (*params, exchange_key),
            )
        await db.commit()
