"""Outcome log — the per-exchange record table and its writer API.

``OutcomeLogPlugin`` owns the ``exchange_log`` schema and writes the
identity row (origin, channel, originating message rowid) via its
message-lifecycle hookimpls. The retrieval-profile, appraisal, and
provenance/outcomes columns are filled in by ``MemoryPlugin``,
``AppraisalPlugin``, and ``CritiquePlugin`` through the writer API below.

The table accumulates one row per exchange so retrieval quality,
appraisal, and critique/engagement outcomes can be correlated — the
calibration data for the memory system's evals.
"""

import json
import logging
import time

import aiosqlite

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.persistence import add_column_if_missing

# Named explicitly: corvidae/logging.py configures the "corvidae" logger,
# and __name__ would be corvidae_cognition.outcome_log.
logger = logging.getLogger("corvidae.outcome_log")

# Columns whose values are JSON envelopes merged via SQLite's json_patch()
# rather than overwritten. Concurrent fire-and-forget writers touch these
# columns with disjoint or overlapping top-level keys; the merge must be a
# single atomic SQL statement (no read-then-write) so no writer's key is
# lost to a lost update.
MERGE_COLUMNS = frozenset({"outcomes", "appraisal"})

EXCHANGE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS exchange_log (
    trace_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    origin TEXT,
    message_rowid INTEGER,
    created_at REAL NOT NULL,
    retrieval_top_score REAL,
    retrieval_top_similarity REAL,
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

# Columns update_exchange may touch. The identity columns (trace_id,
# channel_id, created_at) are immutable after record_exchange; everything
# else is a nullable profile/outcome column filled in by other plugins.
UPDATABLE_COLUMNS = frozenset(
    {
        "origin",
        "message_rowid",
        "retrieval_top_score",
        "retrieval_top_similarity",
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
        # Remember the configured DB path and whether on_start has run, so a
        # missing DB can be diagnosed as startup ordering or a real failure.
        self._db_path = "sessions.db"
        self._started = False

    def _resolve_db(self) -> aiosqlite.Connection | None:
        """Return the persistence plugin's DB connection, or None if not open yet."""
        from corvidae.persistence import PersistencePlugin
        persistence = get_dependency(self.pm, "persistence", PersistencePlugin)
        return persistence.db

    def _channel_is_known(self, channel_id: str, trace_id: str) -> bool:
        """Return False when the registry never held channel_id (#261).

        A transport gate refuses strangers on a transient channel the
        registry never held. A row keyed on that id would grow exchange_log
        without bound and keep a trace of a refused sender, and it feeds no
        calibration. Every insert path (record_exchange, upsert_exchange)
        asks this, so no writer (appraisal, gates, hooks) can leak one.
        With no registry plugin, fail open and record, so a calibration row
        is never lost.
        """
        registry = self.pm.get_plugin("registry")
        if registry is not None and registry.get(channel_id) is None:
            logger.debug(
                "exchange_log: no row for unregistered channel %s (trace %s)",
                channel_id, trace_id,
            )
            return False
        return True

    async def _ensure_table(self) -> aiosqlite.Connection:
        """Create the exchange_log table on first use and return the DB.

        on_start hook ordering is LIFO across plugins, so the persistence
        DB may not exist yet during our on_start — creation is retried
        lazily on first write. Raises RuntimeError if the DB is missing at
        write time (writer calls are explicit API, not fail-soft hooks).
        """
        db = self._resolve_db()
        if db is None:
            raise RuntimeError(
                "persistence DB not available for exchange_log: "
                f"daemon.session_db={self._db_path!r}, "
                f"on_start completed: {self._started}"
            )
        if not self._table_ready:
            await db.execute(EXCHANGE_LOG_DDL)
            # Schema upgrade: exchange_log tables from before #66 gain
            # retrieval_top_similarity as a nullable column.
            await add_column_if_missing(
                db, "exchange_log", "retrieval_top_similarity", "REAL"
            )
            await db.execute(EXCHANGE_LOG_INDEX_DDL)
            await db.commit()
            self._table_ready = True
        return db

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Record the configured path before attempting the early DDL so the
        # guard message can name it even if the persistence DB is not open.
        self._db_path = config.get("daemon", {}).get("session_db", "sessions.db")
        # Best-effort early DDL; falls back to lazy creation on first write
        # if the persistence DB hasn't opened yet (hook-ordering dependent).
        try:
            # The usual case at boot: persistence starts after us (on_start
            # is LIFO). Expected, so one plain line, no traceback (#163).
            if self._resolve_db() is None:
                logger.warning(
                    "exchange_log table creation deferred until the first "
                    "write: persistence DB not open yet (daemon.session_db=%r)",
                    self._db_path,
                )
                return
            await self._ensure_table()
        except Exception:
            logger.warning("exchange_log table creation deferred", exc_info=True)
        finally:
            self._started = True

    # ------------------------------------------------------------------
    # Hook consumers — all fail-soft (log + continue), since these are
    # hooks, not explicit writer calls; exceptions must not propagate
    # into the turn.
    # ------------------------------------------------------------------

    @hookimpl
    async def on_message_admitted(self, channel, trace_id: str, sender: str, text: str) -> None:
        try:
            await self.record_exchange(trace_id, channel.id, origin="user")
        except Exception:
            logger.warning("on_message_admitted: record_exchange failed", exc_info=True)

    @hookimpl
    async def on_message_rejected(self, channel, trace_id: str, sender: str, text: str) -> None:
        # Unregistered (transport-refused) channels are skipped by the
        # writers themselves; the update below is a no-op without a row.
        try:
            await self.record_exchange(trace_id, channel.id, origin="user")
            await self.update_exchange(trace_id, outcomes={"gate": "rejected"})
        except Exception:
            logger.warning("on_message_rejected: record_exchange failed", exc_info=True)

    @hookimpl
    async def on_message_persisted(
        self, channel, trace_id: str, rowid: int | None, text: str, meta: dict,
    ) -> None:
        try:
            # INSERT OR IGNORE covers notification-born exchanges (no prior
            # on_message_admitted for those) — origin lands on first insert.
            # meta["origin"] normalizes to "task" for the background-origin
            # default (reproduces core's old dequeue-mint default); the
            # admitted-time INSERT OR IGNORE row (origin="user") already
            # exists for user messages by the time this hook fires.
            origin = meta.get("origin") or "task"
            await self.record_exchange(trace_id, channel.id, origin=origin)
            if rowid is not None:
                await self.update_exchange(trace_id, message_rowid=rowid)
        except Exception:
            logger.warning("on_message_persisted: record_exchange failed", exc_info=True)

    async def record_exchange(
        self,
        trace_id: str,
        channel_id: str,
        origin: str | None = None,
        message_rowid: int | None = None,
    ) -> None:
        """Insert a new exchange row. Idempotent (INSERT OR IGNORE).

        Args:
            trace_id: The trace id minted for this exchange.
            channel_id: The channel the exchange happened on.
            origin: 'user'|'reminder'|'critique'|'heartbeat'|'task', or
                None if the producer didn't stamp one.
            message_rowid: message_log.id of the originating message; None
                for gate-rejected exchanges.
        """
        if not self._channel_is_known(channel_id, trace_id):
            return
        db = await self._ensure_table()
        await db.execute(
            "INSERT OR IGNORE INTO exchange_log "
            "(trace_id, channel_id, origin, message_rowid, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (trace_id, channel_id, origin, message_rowid, time.time()),
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

    async def update_exchange(self, trace_id: str, **columns) -> None:
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
            f"UPDATE exchange_log SET {set_clause} WHERE trace_id = ?",
            (*params, trace_id),
        )
        await db.commit()

    async def upsert_exchange(
        self,
        trace_id: str,
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
        if not self._channel_is_known(channel_id, trace_id):
            return
        db = await self._ensure_table()
        await db.execute(
            "INSERT OR IGNORE INTO exchange_log "
            "(trace_id, channel_id, origin, message_rowid, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (trace_id, channel_id, origin, None, time.time()),
        )
        merged_columns = dict(columns)
        if origin is not None:
            merged_columns.setdefault("origin", origin)
        if merged_columns:
            set_clause, params = self._build_set_clause(merged_columns)
            await db.execute(
                f"UPDATE exchange_log SET {set_clause} WHERE trace_id = ?",
                (*params, trace_id),
            )
        await db.commit()


# ------------------------------------------------------------------
# Operator redaction: the exchange_log scrubber for `corvidae redact`
# ------------------------------------------------------------------


def _scrub_past_claims(appraisal_json: str | None, tombstone: str) -> str | None:
    """Return appraisal JSON with stage2.past_claims text tombstoned, or None
    when there is nothing to change.

    Each claim and each of its queries becomes ``tombstone``, so the number
    of claims and queries survives for calibration while the words do not.
    No appraisal, no stage 2, or claims already tombstoned → None.
    """
    # NULL is the common case: most exchanges never get a stage-2 appraisal.
    # The column is written by outcome_log's JSON merge, so it parses, and
    # appraisal normalizes past_claims to a list of {claim, queries}.
    if appraisal_json is None:
        return None
    appraisal = json.loads(appraisal_json)
    stage2 = appraisal.get("stage2") or {}
    claims = stage2.get("past_claims")
    if not claims:
        return None
    scrubbed = [
        {"claim": tombstone, "queries": [tombstone] * len(claim["queries"])}
        for claim in claims
    ]
    # Already scrubbed by an earlier redaction: report no change.
    if scrubbed == claims:
        return None
    stage2["past_claims"] = scrubbed
    return json.dumps(appraisal)


async def redact_exchange_log(
    db: aiosqlite.Connection,
    channel_id: str,
    msg_ids: list[int],
    dry_run: bool,
    tombstone: str,
) -> int | None:
    """Scrub exchange_log text derived from redacted messages in one channel.

    For each redacted id X: the exchange that owns X (in the channel, the
    largest message_rowid <= X) gets its stage-2 past_claims tombstoned.
    Every exchange in the channel with message_rowid >= the smallest X loses
    its provenance_snapshot, since any later window may have carried X's
    text as CONTEXT. Exchanges with a NULL message_rowid are not reached.

    Registered as the ``exchange_log`` scrubber in the ``corvidae.redact``
    entry-point group, which ``corvidae redact`` runs per channel over the
    ids it tombstoned (#160). Returns the number of exchanges changed (or
    that would be, in dry-run mode), or None when the table is absent.
    """
    # A DB whose daemon never ran outcome_log has no table to scrub.
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'exchange_log'"
    ) as cursor:
        if await cursor.fetchone() is None:
            return None
    if not msg_ids:
        return 0
    changed: set[str] = set()

    # Find each redacted id's owning exchange, once per exchange.
    owners: set[str] = set()
    for msg_id in msg_ids:
        async with db.execute(
            "SELECT trace_id FROM exchange_log "
            "WHERE channel_id = ? AND message_rowid IS NOT NULL "
            "AND message_rowid <= ? ORDER BY message_rowid DESC LIMIT 1",
            (channel_id, msg_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            owners.add(row[0])

    # Tombstone the owners' past_claims text.
    for trace_id in owners:
        async with db.execute(
            "SELECT appraisal FROM exchange_log WHERE trace_id = ?", (trace_id,)
        ) as cursor:
            row = await cursor.fetchone()
        scrubbed = _scrub_past_claims(row[0], tombstone)
        if scrubbed is None:
            continue
        changed.add(trace_id)
        if not dry_run:
            await db.execute(
                "UPDATE exchange_log SET appraisal = ? WHERE trace_id = ?",
                (scrubbed, trace_id),
            )

    # Drop every snapshot in the channel from the first redacted id on.
    snapshot_where = (
        "WHERE channel_id = ? AND message_rowid >= ? "
        "AND provenance_snapshot IS NOT NULL"
    )
    snapshot_args = (channel_id, min(msg_ids))
    async with db.execute(
        f"SELECT trace_id FROM exchange_log {snapshot_where}", snapshot_args
    ) as cursor:
        changed.update(row[0] for row in await cursor.fetchall())
    if not dry_run:
        await db.execute(
            f"UPDATE exchange_log SET provenance_snapshot = NULL {snapshot_where}",
            snapshot_args,
        )
        await db.commit()
    return len(changed)
