"""PersistencePlugin — SQLite database lifecycle and conversation persistence.

Manages the SQLite database connection. Implements resolve_conversation,
persist_message, and on_compaction hooks for persisting conversation
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
import sqlite3
import time
from pathlib import Path

import aiosqlite

from corvidae.hooks import CorvidaePlugin, hookimpl

logger = logging.getLogger(__name__)

_ALLOWED_JOURNAL_MODES = {"delete", "truncate", "persist", "memory", "wal", "off"}


async def add_column_if_missing(
    db: aiosqlite.Connection, table: str, column: str, decl: str
) -> None:
    """Add ``column`` to an existing ``table`` unless it is already there.

    Upgrades a table created before the column existed; old rows read the
    column's default. Two first callers can race between the check and the
    ALTER, so the loser's "duplicate column name" error means the column is
    in place, not a failure. ``table``, ``column``, and ``decl`` are
    interpolated into SQL and must be code constants, never input.
    """
    async with db.execute(f"PRAGMA table_info({table})") as cursor:
        existing = {row[1] for row in await cursor.fetchall()}
    if column in existing:
        return
    try:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc):
            raise
        logger.debug("%s.%s added concurrently by another caller", table, column)


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


def decode_message_row(
    row_id: int,
    raw: bytes | str,
    channel_id: str,
    *,
    consequence: str = "the rest of the history still loads",
    redacting: bool = False,
) -> dict | None:
    """Decode one ``message_log.message`` value, or None if it is corrupt.

    The single decoder every reader of the column shares. ``raw`` is the value
    selected as ``CAST(message AS BLOB)`` so invalid UTF-8 fails here, where it
    can be caught, rather than inside the sqlite3 fetch. ValueError covers
    JSONDecodeError, UnicodeDecodeError and the int digit limit; deeply nested
    arrays raise RecursionError. A value that is not a JSON object is corrupt
    too. A corrupt row logs one WARNING naming the row id and channel (never
    the row's bytes) and says what happens next via ``consequence``.
    ``redacting=True`` is for the redact tool, where corruption is expected and
    handled: the warning says the row is being tombstoned and carries no
    traceback or "inspect or redact" advice.
    """
    # Decode the JSON payload
    try:
        msg = json.loads(raw)
    except (ValueError, RecursionError):
        # Redact path: plain wording, no traceback
        if redacting:
            logger.warning(
                "row %s for channel %s is corrupt; tombstoning it as role "
                "'unknown'", row_id, channel_id,
            )
            return None
        logger.warning(
            "skipping unparseable message row %s for channel %s; %s — "
            "inspect or redact the row",
            row_id, channel_id, consequence, exc_info=True,
        )
        return None
    # Only a JSON object can carry the role/content keys and window tags
    if not isinstance(msg, dict):
        # Redact path: plain wording
        if redacting:
            logger.warning(
                "row %s for channel %s is corrupt (decoded to %s, not an "
                "object); tombstoning it as role 'unknown'",
                row_id, channel_id, type(msg).__name__,
            )
            return None
        logger.warning(
            "skipping message row %s for channel %s: decoded to %s, "
            "not an object; %s — inspect or redact the row",
            row_id, channel_id, type(msg).__name__, consequence,
        )
        return None
    return msg


def parse_message_type(
    row_id: int,
    value: str,
    channel_id: str,
    *,
    consequence: str = "the rest of the history still loads",
):
    """Map a stored message_type string onto MessageType, or None if unknown.

    An unknown value logs one WARNING naming the row, channel, the rejected
    value and the accepted ones; the caller skips (or treats as non-dialog)
    the row.
    """
    from corvidae.context import MessageType
    try:
        return MessageType(value)
    except ValueError:
        accepted = ", ".join(t.value for t in MessageType)
        logger.warning(
            "skipping message row %s for channel %s: unknown message_type "
            "%r (accepted: %s); %s — inspect or correct the row",
            row_id, channel_id, value, accepted, consequence,
        )
        return None


def _parse_message_rows(rows: list[tuple], channel_id: str) -> list[dict]:
    """Parse (id, message_bytes, message_type) rows into tagged message dicts.

    Re-attaches the rowid as _db_id so the reload path restores the rowid
    threading that consolidation depends on (bootstrap-mapping §4.8).

    One corrupt record must not abort the whole conversation load, so a row
    that fails to decode, or has a message_type outside MessageType, is
    skipped with a warning naming it and the channel.
    """
    result = []
    for row in rows:
        msg = decode_message_row(row[0], row[1], channel_id)
        if msg is None:
            continue
        # Map the stored type string back onto the enum
        message_type = parse_message_type(row[0], row[2], channel_id)
        if message_type is None:
            continue
        msg["_message_type"] = message_type
        msg["_db_id"] = row[0]
        result.append(msg)
    return result


def _parse_summary_row(row: tuple, channel_id: str) -> dict | None:
    """Decode an (id, message_bytes, timestamp) summary row, or None if corrupt.

    The message arrives as bytes (the query casts it to BLOB) so that invalid
    UTF-8 fails here, where it can be caught, rather than inside the sqlite3
    fetch. ValueError covers JSONDecodeError, UnicodeDecodeError and the int
    digit limit; deeply nested arrays raise RecursionError. A value that
    decodes to anything but a JSON object is corrupt too.
    """
    try:
        msg = json.loads(row[1])
    except (ValueError, RecursionError):
        logger.warning(
            "skipping unparseable summary row %s for channel %s; "
            "falling back to an older summary or the full history",
            row[0], channel_id, exc_info=True,
        )
        return None
    if not isinstance(msg, dict):
        logger.warning(
            "skipping summary row %s for channel %s: decoded to %s, not an "
            "object; falling back to an older summary or the full history",
            row[0], channel_id, type(msg).__name__,
        )
        return None
    return msg


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
            base_dir = config.get("_base_dir", Path("."))
            self.db = await aiosqlite.connect(str(Path(base_dir) / db_path))
            await init_db(self.db)

        # Set journal mode (WAL by default for concurrent access)
        journal_mode = config.get("daemon", {}).get("sqlite_journal_mode", "wal").lower()
        if journal_mode not in _ALLOWED_JOURNAL_MODES:
            allowed = ", ".join(sorted(_ALLOWED_JOURNAL_MODES))
            raise ValueError(
                f"Invalid sqlite_journal_mode: {journal_mode!r} "
                f"(accepted values: {allowed})"
            )
        async with self.db.execute(f"PRAGMA journal_mode={journal_mode}") as cursor:
            row = await cursor.fetchone()
        actual_mode = row[0] if row else journal_mode
        logger.info(
            "SQLite journal mode set to %s", actual_mode,
            extra={"journal_mode": actual_mode},
        )

        # Seed the idle tracker: every channel with history is quiet since its
        # last row, so channel watches fire after a restart with no traffic.
        # pm is absent when the plugin was built without one (some tests do).
        # shortcut: hand-built plugin managers omit the idle tracker; make it strict if a non-Runtime entry point appears
        pm = getattr(self, "pm", None)
        idle = pm.get_plugin("idle") if pm is not None else None
        if idle is not None:
            async with self.db.execute(
                "SELECT channel_id, MAX(timestamp) FROM message_log GROUP BY channel_id"
            ) as cursor:
                rows = await cursor.fetchall()
            for channel_id, last_ts in rows:
                idle.note_restored(channel_id, last_ts)
        else:
            logger.warning(
                "no plugin manager or idle tracker (name 'idle'); restart seed "
                "skipped — Runtime provides both, so this plugin was built by hand"
            )

    @hookimpl
    async def on_stop(self) -> None:
        if self.db:
            await self.db.close()

    @hookimpl(trylast=True)
    async def resolve_conversation(self, channel) -> list[dict] | None:
        """Load conversation history from SQLite for the channel.

        Returns a list of tagged message dicts if rows exist, or None if the
        channel has no history.

        If a summary row exists, returns that summary plus non-summary rows
        with timestamp > summary_ts. Otherwise returns all non-summary rows
        ordered by timestamp, id.

        A summary row that fails to decode is skipped with a warning and the
        next-newest summary is tried; message_log is append-only, so an older
        summary plus the rows after its boundary still covers the whole
        conversation. If no summary decodes, all non-summary rows load.

        A corrupt ordinary row is skipped with a warning; the rows around it
        still load. Every query casts the message to BLOB so invalid UTF-8
        is caught while decoding instead of raising inside the fetch.
        """
        # Find the newest summary that decodes. The message is cast to BLOB
        # so invalid UTF-8 surfaces in _parse_summary_row instead of raising
        # inside the fetch, where the cursor cannot step past it.
        summary_id = summary_msg = summary_ts = None
        async with self.db.execute(
            "SELECT id, CAST(message AS BLOB), timestamp FROM message_log "
            "WHERE channel_id = ? AND message_type = 'summary' "
            "ORDER BY id DESC",
            (channel.id,),
        ) as cursor:
            async for row in cursor:
                summary_msg = _parse_summary_row(row, channel.id)
                if summary_msg is not None:
                    summary_id, _, summary_ts = row
                    break

        from corvidae.context import MessageType

        if summary_msg is not None:
            summary_msg["_message_type"] = MessageType.SUMMARY
            summary_msg["_db_id"] = summary_id
            # Load non-summary rows after the summary boundary
            async with self.db.execute(
                "SELECT id, CAST(message AS BLOB), message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "AND timestamp > ? ORDER BY id",
                (channel.id, summary_ts),
            ) as cursor:
                rows = await cursor.fetchall()
            loaded = _parse_message_rows(rows, channel.id)
            result = [summary_msg] + loaded
        else:
            async with self.db.execute(
                "SELECT id, CAST(message AS BLOB), message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "ORDER BY timestamp, id",
                (channel.id,),
            ) as cursor:
                rows = await cursor.fetchall()
            result = _parse_message_rows(rows, channel.id)

        if not result:
            return None
        logger.info(
            "conversation loaded for channel",
            extra={"channel_id": channel.id, "count": len(result)},
        )
        return result

    @hookimpl
    async def persist_message(
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
        try:
            cursor = await self.db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, ?)",
                (channel.id, json.dumps(clean), ts, message_type),
            )
            rowid = cursor.lastrowid
            await self.db.commit()
        except Exception:
            # A locked or failing DB must not stall the turn; the window still
            # holds the message, and the next event retries the write.
            logger.warning(
                "failed to persist conversation event for channel %s",
                channel.id,
                exc_info=True,
                extra={"channel_id": channel.id},
            )
            return None
        return rowid

    @hookimpl
    async def on_compaction(
        self, channel, summary_msg: dict, retain_count: int, compacted_ids: list[int]
    ) -> None:
        """Persist a compaction summary to SQLite with timestamp boundary logic.

        The summary timestamp is set just before the oldest retained message
        so that resolve_conversation correctly excludes pre-compaction messages.
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
