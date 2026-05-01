"""PersistencePlugin — SQLite database lifecycle and conversation persistence.

Manages the SQLite database connection. Implements load_conversation,
on_conversation_event, and on_compaction hooks for persisting conversation
history to SQLite.

Config:
    daemon:
      session_db: sessions.db       # path to SQLite database
      sqlite_journal_mode: wal      # SQLite journal mode (default "wal");
                                    # allowed values: delete, truncate, persist,
                                    # memory, wal, off
"""

import json
import logging
import time
import aiosqlite

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.migrations import MIGRATIONS

logger = logging.getLogger(__name__)

_ALLOWED_JOURNAL_MODES = {"delete", "truncate", "persist", "memory", "wal", "off"}


# ---------------------------------------------------------------------------
# Schema versioning helpers
# ---------------------------------------------------------------------------

_CREATE_SCHEMA_VERSION_TABLE = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
)"""


async def _ensure_schema_version_table(db: aiosqlite.Connection) -> None:
    """Create the schema_version meta table if absent and seed it with version 0."""
    await db.execute(_CREATE_SCHEMA_VERSION_TABLE)

    async with db.execute("SELECT COUNT(*) FROM schema_version") as cursor:
        row = await cursor.fetchone()
    if row[0] == 0:
        await db.execute("INSERT INTO schema_version (version) VALUES (0)")
    await db.commit()


async def get_schema_version(db: aiosqlite.Connection) -> int:
    """Return the current schema version from the database.

    Returns 0 if the ``schema_version`` table does not exist yet (fresh DB
    or legacy database created before the migration system).
    """
    try:
        async with db.execute("SELECT version FROM schema_version") as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0
    except aiosqlite.OperationalError:
        return 0


async def _apply_pending_migrations(db: aiosqlite.Connection) -> None:
    """Apply every migration whose number is greater than the stored version.

    Each migration runs via ``executescript``. On success the version is
    advanced; on failure the transaction is rolled back and the version
    remains where it was.
    """
    async with db.execute("SELECT version FROM schema_version") as cursor:
        row = await cursor.fetchone()
    current_version = row[0]

    pending = MIGRATIONS[current_version:]

    if not pending:
        logger.info(
            "Schema is up to date at version %d, no pending migrations",
            current_version,
        )
        return

    for i, migration_sql in enumerate(pending):
        migration_number = current_version + i + 1
        try:
            await db.executescript(migration_sql)
            await db.execute(
                "UPDATE schema_version SET version = ?", (migration_number,)
            )
            await db.commit()
            logger.info(
                "Applied migration #%d (version %d/%d)",
                migration_number,
                migration_number,
                len(MIGRATIONS),
            )
        except Exception:
            await db.rollback()
            logger.exception(
                "Migration #%d failed, schema_version remains at %d",
                migration_number,
                current_version,
            )
            raise


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

async def init_db(db: aiosqlite.Connection) -> None:
    """Apply pending schema migrations and update the version tracker.

    On a fresh database this creates the ``schema_version`` meta table and
    runs all migrations in order. On an existing database it checks the
    current version and applies only the pending migrations.
    """
    await _ensure_schema_version_table(db)
    await _apply_pending_migrations(db)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _strip_internal_tags(message: dict) -> dict:
    """Return a copy of *message* with internal metadata keys removed."""
    return {k: v for k, v in message.items() if not k.startswith("_")}


def _parse_message_rows(rows: list[tuple]) -> list[dict]:
    """Parse (message_json, message_type) rows into tagged message dicts."""
    from corvidae.context import MessageType
    result = []
    for row in rows:
        msg = json.loads(row[0])
        msg["_message_type"] = MessageType(row[1])
        result.append(msg)
    return result


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class PersistencePlugin(CorvidaePlugin):
    """Plugin that manages SQLite database lifecycle and conversation persistence.

    Attributes:
        pm: Plugin manager instance (set in on_init via CorvidaePlugin).
        db: aiosqlite.Connection, opened in on_start, closed in on_stop.
            Public for test injection (same pattern as former Agent.db).
    """

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.db: aiosqlite.Connection | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Open SQLite database (only if not already injected for testing)
        if self.db is None:
            db_path = config.get("daemon", {}).get("session_db", "sessions.db")
            self.db = await aiosqlite.connect(db_path)
            await init_db(self.db)

        # Set journal mode (WAL by default for concurrent access)
        journal_mode = config.get("daemon", {}).get("sqlite_journal_mode", "wal").lower()
        if journal_mode not in _ALLOWED_JOURNAL_MODES:
            raise ValueError(f"Invalid sqlite_journal_mode: {journal_mode!r}")
        async with self.db.execute(f"PRAGMA journal_mode={journal_mode}") as cursor:
            row = await cursor.fetchone()
        actual_mode = row[0] if row else journal_mode
        logger.info(
            "SQLite journal mode set to %s", actual_mode,
            extra={"journal_mode": actual_mode},
        )

    @hookimpl
    async def on_stop(self) -> None:
        if self.db:
            await self.db.close()

    @hookimpl(trylast=True)
    async def load_conversation(self, channel) -> list[dict] | None:
        """Load conversation history from SQLite for the channel.

        Returns a list of tagged message dicts if rows exist, or None if the
        channel has no history.

        If a summary row exists, returns that summary plus non-summary rows
        with timestamp > summary_ts. Otherwise returns all non-summary rows
        ordered by timestamp, id.
        """
        async with self.db.execute(
            "SELECT id, message, timestamp FROM message_log "
            "WHERE channel_id = ? AND message_type = 'summary' "
            "ORDER BY id DESC LIMIT 1",
            (channel.id,),
        ) as cursor:
            summary_row = await cursor.fetchone()

        from corvidae.context import MessageType

        if summary_row:
            _, summary_message, summary_ts = summary_row
            summary_msg = json.loads(summary_message)
            summary_msg["_message_type"] = MessageType.SUMMARY
            # Load non-summary rows after the summary boundary
            async with self.db.execute(
                "SELECT message, message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "AND timestamp > ? ORDER BY id",
                (channel.id, summary_ts),
            ) as cursor:
                rows = await cursor.fetchall()
            loaded = _parse_message_rows(rows)
            result = [summary_msg] + loaded
        else:
            async with self.db.execute(
                "SELECT message, message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "ORDER BY timestamp, id",
                (channel.id,),
            ) as cursor:
                rows = await cursor.fetchall()
            result = _parse_message_rows(rows)

        if not result:
            return None
        logger.info(
            "conversation loaded for channel",
            extra={"channel_id": channel.id, "count": len(result)},
        )
        return result

    @hookimpl
    async def on_conversation_event(self, channel, message: dict, message_type) -> None:
        """Persist a conversation message to SQLite."""
        ts = time.time()
        clean = _strip_internal_tags(message)
        await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            (channel.id, json.dumps(clean), ts, message_type),
        )
        await self.db.commit()

    @hookimpl
    async def on_compaction(self, channel, summary_msg: dict, retain_count: int) -> None:
        """Persist a compaction summary to SQLite with timestamp boundary logic."""
        if retain_count > 0:
            async with self.db.execute(
                "SELECT timestamp FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "ORDER BY id DESC LIMIT 1 OFFSET ?",
                (channel.id, retain_count - 1),
            ) as cursor:
                row = await cursor.fetchone()
            summary_ts = row[0] - 1e-6 if row else time.time()
        else:
            summary_ts = time.time()

        clean = _strip_internal_tags(summary_msg)
        await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            (channel.id, json.dumps(clean), summary_ts, "summary"),
        )
        await self.db.commit()
