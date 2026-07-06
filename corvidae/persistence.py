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

logger = logging.getLogger(__name__)

_ALLOWED_JOURNAL_MODES = {"delete", "truncate", "persist", "memory", "wal", "off"}


async def init_db(db: aiosqlite.Connection) -> None:
    """Create message_log table and index.

    Creates the message_log table if it doesn't exist, plus an index on
    (channel_id, timestamp) for efficient per-channel queries ordered by time.
    """
    await db.execute(
        """CREATE TABLE IF NOT EXISTS message_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp REAL NOT NULL,
            message_type TEXT NOT NULL DEFAULT 'message'
        )"""
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_log_channel ON message_log (channel_id, timestamp)"
    )
    await db.commit()


def _strip_internal_keys(message: dict) -> dict:
    """Drop every _-prefixed key before serialization.

    Internal window tags (_message_type, _db_id, ...) must never reach the
    DB (bootstrap-mapping §4.8).
    """
    return {k: v for k, v in message.items() if not k.startswith("_")}


def _parse_message_rows(rows: list[tuple]) -> list[dict]:
    """Parse (id, message_json, message_type) rows into tagged message dicts.

    Re-attaches the rowid as _db_id so the reload path restores the rowid
    threading that consolidation depends on (bootstrap-mapping §4.8).
    """
    from corvidae.context import MessageType
    result = []
    for row in rows:
        msg = json.loads(row[1])
        msg["_message_type"] = MessageType(row[2])
        msg["_db_id"] = row[0]
        result.append(msg)
    return result


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
            summary_id, summary_message, summary_ts = summary_row
            summary_msg = json.loads(summary_message)
            summary_msg["_message_type"] = MessageType.SUMMARY
            summary_msg["_db_id"] = summary_id
            # Load non-summary rows after the summary boundary
            async with self.db.execute(
                "SELECT id, message, message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "AND timestamp > ? ORDER BY id",
                (channel.id, summary_ts),
            ) as cursor:
                rows = await cursor.fetchall()
            loaded = _parse_message_rows(rows)
            result = [summary_msg] + loaded
        else:
            async with self.db.execute(
                "SELECT id, message, message_type FROM message_log "
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
    async def on_conversation_event(
        self, channel, message: dict, message_type
    ) -> int | None:
        """Persist a conversation message to SQLite and return its rowid.

        Strips every _-prefixed key from the message dict before writing to
        the DB to avoid serializing internal metadata. This is the single
        implementation of this hook allowed to return non-None
        (bootstrap-mapping §4.8).
        """
        ts = time.time()
        # Strip internal window tags (must not be written to the JSON column)
        clean = _strip_internal_keys(message)
        cursor = await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            (channel.id, json.dumps(clean), ts, message_type),
        )
        rowid = cursor.lastrowid
        await self.db.commit()
        return rowid

    @hookimpl
    async def on_compaction(
        self, channel, summary_msg: dict, retain_count: int, compacted_ids: list[int]
    ) -> None:
        """Persist a compaction summary to SQLite with timestamp boundary logic.

        The summary timestamp is set just before the oldest retained message
        so that load_conversation correctly excludes pre-compaction messages.
        The compacted_ids parameter is ignored here — it exists for
        consolidation consumers (MemoryPlugin).
        """
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

        # Strip internal window tags if present
        clean = _strip_internal_keys(summary_msg)
        await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            (channel.id, json.dumps(clean), summary_ts, "summary"),
        )
        await self.db.commit()
