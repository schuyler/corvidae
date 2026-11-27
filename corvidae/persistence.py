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
      sqlite_busy_timeout_ms: 30000 # ms the write route's writer waits for an
                                    # out-of-process writer's lock before giving up
                                    # (default 30000); also the markdown index's
                                    # wait. Must be 1..600000.
"""

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import aiosqlite

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.provenance import PROVENANCE_KEY, decode as decode_provenance
from corvidae.shutdown import parse_shutdown_timeouts
from corvidae.session_writer import (
    DEFAULT_WRITE_DEADLINE_S,
    SessionWriter,
    WriteCursor,
    open_session_store,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)

_ALLOWED_JOURNAL_MODES = {"delete", "truncate", "persist", "memory", "wal", "off"}

#: Default write-lock wait for the session write route's writer connection (the
#: only in-daemon writer on sessions.db, #1211): how long a unit's BEGIN waits
#: for an out-of-process writer such as an offline CLI. The markdown index uses
#: the same value on its own index.db. Out-of-process tools (the goals and
#: redact CLIs, reconsolidate) connect on their own.
DEFAULT_BUSY_TIMEOUT_MS = 30_000

#: A wait outside this window is a mistake, not a policy: 0 or a negative value
#: disables busy handling entirely (worse than the inherited 5 s), and SQLite
#: clamps anything past INT_MAX.
BUSY_TIMEOUT_BOUNDS = (1, 600_000)


def busy_timeout_ms_from_config(config: dict) -> int:
    """Resolve ``daemon.sqlite_busy_timeout_ms``, or the default when unusable.

    Applied to the session write route's writer connection, which carries every
    in-daemon write to the session database (#1211), and to the markdown index's
    own index.db, so one config key sets one wait. Out-of-process tools connect
    on their own and are not covered.

    Types are checked before conversion: ``int()`` would accept ``True`` as 1 ms
    and truncate ``1.5``, and ``int(float("inf"))`` raises OverflowError, which
    must not abort startup. A digit string is accepted because that is what a
    YAML scalar yields. 0 and negatives are rejected: they disable busy
    handling, which is worse than the 5 s default this replaces.
    """
    daemon = config.get("daemon")
    raw = (daemon if isinstance(daemon, dict) else {}).get(
        "sqlite_busy_timeout_ms", DEFAULT_BUSY_TIMEOUT_MS
    )
    low, high = BUSY_TIMEOUT_BOUNDS
    value: int | None
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        # bool first: True is an int subclass, and True is not a timeout.
        value = None
    elif isinstance(raw, str):
        # Let int() decide: str.isdigit() is true for "²", and lstrip("+-")
        # accepts "++5"; both then raise ValueError inside us, and this runs
        # in on_start with no caller to catch it.
        try:
            value = int(raw.strip())
        except ValueError:
            value = None
    else:
        value = raw
    if value is None or not low <= value <= high:
        logger.warning(
            "daemon.sqlite_busy_timeout_ms %r is unusable (want an integer from "
            "%d to %d ms); using %d ms",
            raw, low, high, DEFAULT_BUSY_TIMEOUT_MS,
        )
        return DEFAULT_BUSY_TIMEOUT_MS
    return value


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


def add_column_if_missing_sync(
    cur: WriteCursor, table: str, column: str, decl: str
) -> None:
    """Add ``column`` to ``table`` unless present; the write-unit version.

    Same contract as :func:`add_column_if_missing`, for use inside a unit.
    Inside the unit nothing else writes, so no duplicate-column race exists.
    """
    existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table})").rows}
    if column in existing:
        return
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# The message_log schema as one statement per entry (units run one execute
# each; a script would commit the kernel's transaction).
MESSAGE_LOG_SCHEMA: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS message_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp REAL NOT NULL,
            message_type TEXT NOT NULL DEFAULT 'message'
        )""",
    "CREATE INDEX IF NOT EXISTS idx_log_channel ON message_log (channel_id, timestamp)",
)


async def init_db(db: aiosqlite.Connection) -> None:
    """Create message_log table and index on a read-write connection.

    Runs :data:`MESSAGE_LOG_SCHEMA`. Kept for tests that seed a read-write
    connection; the daemon creates the schema through the write route.
    """
    for stmt in MESSAGE_LOG_SCHEMA:
        await db.execute(stmt)
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


async def view_rows(
    db: aiosqlite.Connection,
    channel_id: str,
    after_ts: float | None,
    through_id: int | None = None,
) -> list[tuple[float, dict]]:
    """The channel's non-summary rows as (timestamp, tagged message) pairs.

    With ``after_ts`` set: rows newer than it, ordered by id (what reload
    loads under a summary). With None: every row, ordered by timestamp then
    id. ``through_id`` caps the row id (None: no cap). Shared by
    ``resolve_conversation`` and the catch-up view so both read rows one way.
    """
    # Build the query: the summary-boundary form or the whole-history form.
    sql = (
        "SELECT id, CAST(message AS BLOB), message_type, timestamp FROM message_log "
        "WHERE channel_id = ? AND message_type != 'summary'"
    )
    params: list = [channel_id]
    if after_ts is not None:
        sql += " AND timestamp > ?"
        params.append(after_ts)
    if through_id is not None:
        sql += " AND id <= ?"
        params.append(through_id)
    sql += " ORDER BY id" if after_ts is not None else " ORDER BY timestamp, id"
    async with db.execute(sql, params) as cursor:
        rows = await cursor.fetchall()
    # Parse, then re-pair each surviving message with its stored timestamp.
    stamps = {row[0]: row[3] for row in rows}
    return [
        (stamps[msg["_db_id"]], msg)
        for msg in _parse_message_rows([row[:3] for row in rows], channel_id)
    ]


async def decodable_summaries(
    db: aiosqlite.Connection, channel_id: str
) -> list[tuple[int, float, dict]]:
    """Every summary row that decodes, as (id, timestamp, message), newest first.

    Each message is tagged as a summary with its ``_db_id``. A corrupt row is
    skipped with ``_parse_summary_row``'s warning.
    """
    from corvidae.context import MessageType

    # Read newest first; the BLOB cast lets bad UTF-8 fail in the parser.
    async with db.execute(
        "SELECT id, CAST(message AS BLOB), timestamp FROM message_log "
        "WHERE channel_id = ? AND message_type = 'summary' ORDER BY id DESC",
        (channel_id,),
    ) as cursor:
        rows = await cursor.fetchall()
    found: list[tuple[int, float, dict]] = []
    for row in rows:
        msg = _parse_summary_row(row, channel_id)
        if msg is None:
            continue
        msg["_message_type"] = MessageType.SUMMARY
        msg["_db_id"] = row[0]
        found.append((row[0], row[2], msg))
    return found


class PersistencePlugin(CorvidaePlugin):
    """Plugin that manages SQLite database lifecycle and conversation persistence.

    Attributes:
        pm: Plugin manager instance (set in on_init via CorvidaePlugin).
        db: read-only aiosqlite.Connection (the reader), opened in on_start,
            closed in on_stop. Public for test injection.
        writer: the SessionWriter, the only read-write connection. Public for
            test injection; inject it together with ``db``.
    """

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.db: aiosqlite.Connection | None = None
        self.writer: SessionWriter | None = None

    async def write(
        self,
        fn: Callable[[WriteCursor], T],
        *,
        owner: str,
        deadline_s: float = DEFAULT_WRITE_DEADLINE_S,
    ) -> T:
        """Run ``fn`` as a write unit on the session database."""
        if self.writer is None:
            raise RuntimeError(
                f"session writer is not open (write from {owner}); "
                "PersistencePlugin.on_start has not run yet"
            )
        return await self.writer.write(fn, owner=owner, deadline_s=deadline_s)

    async def load_extension(self, path: str, *, owner: str) -> None:
        """Load a SQLite extension on the writer connection."""
        if self.writer is None:
            raise RuntimeError(
                f"session writer is not open (load_extension from {owner}); "
                "PersistencePlugin.on_start has not run yet"
            )
        await self.writer.load_extension(path, owner=owner)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Injection is a pair: a half-injected store would be silently
        # replaced by one opened from config.
        if (self.db is None) != (self.writer is None):
            missing = "writer" if self.writer is None else "db"
            raise RuntimeError(
                f"PersistencePlugin.{missing} is not set but the other half "
                "is; inject both (db, writer) from open_session_store or neither"
            )
        if self.db is None:
            # Validate the journal mode before touching the file.
            daemon_cfg = config.get("daemon", {})
            journal_mode = daemon_cfg.get("sqlite_journal_mode", "wal").lower()
            if journal_mode not in _ALLOWED_JOURNAL_MODES:
                allowed = ", ".join(sorted(_ALLOWED_JOURNAL_MODES))
                raise ValueError(
                    f"Invalid sqlite_journal_mode: {journal_mode!r} "
                    f"(accepted values: {allowed})"
                )
            db_path = daemon_cfg.get("session_db", "sessions.db")
            # Two connections cannot share a private in-memory database.
            if str(db_path) == ":memory:":
                raise ValueError(
                    "Config error: daemon.session_db is ':memory:', which the "
                    "session write route cannot use (the reader and writer "
                    "are separate connections) — set daemon.session_db to a "
                    "file path in agent.yaml"
                )
            base_dir = config.get("_base_dir", Path("."))
            # A writer that inherits sqlite3.connect's 5.0 s default gives up
            # while an out-of-process writer (an offline CLI) holds the WAL
            # write lock (corvidae#1211). The wait is bounded, not removed,
            # and open_session_store sets it before journal_mode and the
            # schema, both of which are writes.
            busy_ms = busy_timeout_ms_from_config(config)
            # The writer's revoke bound is derived from the configured graceful
            # phase, so it always fits inside the shutdown it sits in (#1315).
            graceful_s = parse_shutdown_timeouts(
                config.get("daemon")
            ).graceful
            self.db, self.writer = await open_session_store(
                Path(base_dir) / db_path,
                journal_mode=journal_mode,
                busy_timeout_ms=busy_ms,
                graceful_s=graceful_s,
            )
            logger.info(
                "SQLite busy timeout set to %d ms", busy_ms,
                extra={"busy_timeout_ms": busy_ms},
            )
            # Report the mode in effect, which SQLite may refuse to change.
            async with self.db.execute("PRAGMA journal_mode") as cursor:
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
        # Close the reader, then the writer (waits for the in-flight unit).
        try:
            if self.db:
                await self.db.close()
        finally:
            if self.writer:
                await self.writer.close()

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
            loaded = [m for _, m in await view_rows(self.db, channel.id, summary_ts)]
            result = [summary_msg] + loaded
        else:
            result = [m for _, m in await view_rows(self.db, channel.id, None)]

        if not result:
            return None
        logger.info(
            "conversation loaded for channel",
            extra={"channel_id": channel.id, "count": len(result)},
        )
        return result

    async def find_spill_source(self, name: str) -> str | None:
        """The source recorded when the spill file ``name`` was written, or None.

        ``name`` is the file's name (32 hex digits and ``.txt``). The row that
        recorded the cut holds the file's path and source in its provenance
        record (#576). None means no such row, or one from before sources were
        recorded; the caller says which gap it was.

        shortcut: this scans the rows that mention ``name`` by a substring
        match, on every read of a spill file; add an index on the spill path
        if message_log grows enough for that to show. It decodes each
        candidate rather than using json_extract, which fails the whole query
        on one corrupt row.
        """
        if self.db is None:
            return None
        async with self.db.execute(
            "SELECT id, channel_id, CAST(message AS BLOB) FROM message_log "
            "WHERE instr(CAST(message AS BLOB), CAST(? AS BLOB)) > 0 ORDER BY id",
            (name,),
        ) as cursor:
            rows = await cursor.fetchall()
        for row_id, channel_id, raw in rows:
            msg = decode_message_row(
                row_id, raw, channel_id,
                consequence="it is skipped when looking for a spill file's source",
            )
            decoded = decode_provenance(msg.get(PROVENANCE_KEY)) if msg else None
            if decoded is None:
                continue
            meta = decoded[0]
            if meta.spill_path and Path(meta.spill_path).name == name and meta.cut_source:
                return meta.cut_source
        return None

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
        payload = json.dumps(clean)

        # One unit: the insert, returning its rowid.
        def insert(cur: WriteCursor) -> int | None:
            return cur.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, ?)",
                (channel.id, payload, ts, message_type),
            ).lastrowid

        try:
            rowid = await self.write(insert, owner="persistence")
        except Exception:
            # A locked, revoked or failing write must not stall the turn.
            # Nothing retries it: the window keeps the message in memory, but
            # the row is lost, so it will not survive a restart.
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
        # Strip internal window tags if present
        clean = _strip_internal_keys(summary_msg)
        payload = json.dumps(clean)

        # One unit: find the boundary timestamp and insert the summary.
        def insert(cur: WriteCursor) -> None:
            if retain_count > 0:
                rows = cur.execute(
                    "SELECT timestamp FROM message_log "
                    "WHERE channel_id = ? AND message_type != 'summary' "
                    "ORDER BY id DESC LIMIT 1 OFFSET ?",
                    (channel.id, retain_count - 1),
                ).rows
                summary_ts = rows[0][0] - 1e-6 if rows else time.time()
            else:
                summary_ts = time.time()
            cur.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, ?)",
                (channel.id, payload, summary_ts, "summary"),
            )

        await self.write(insert, owner="persistence")
