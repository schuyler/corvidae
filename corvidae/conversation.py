"""Conversation log with SQLite persistence.

This module provides per-channel conversation history management. Messages are
persisted to SQLite and kept in-memory for efficient access. Compaction is
handled externally via CompactionPlugin, which calls replace_with_summary()
when the estimated token count approaches the context limit.

Logging:
    - DEBUG: messages loaded, message appended
    - INFO: compaction completed (before/after counts)
"""

import enum
import json
import logging
import time

import aiosqlite

logger = logging.getLogger(__name__)


class MessageType(str, enum.Enum):
    """Persistence category for a message log entry.

    Controls storage and filtering behavior, not conversational role.
    Values match the ``message_type`` TEXT column in the ``message_log`` table.

    MESSAGE: an ordinary conversation message (user or assistant turn).
    SUMMARY: a compaction summary that replaces a range of older messages.
    CONTEXT: plugin-injected contextual information (memory, notes, retrieved
             documents, etc.) that should appear in the prompt but is not part
             of the conversational turn history.
    """

    MESSAGE = "message"
    SUMMARY = "summary"
    CONTEXT = "context"


#: Default characters-per-token estimate for rough token counting.
#: Used by ConversationLog, CompactionPlugin, and PersistencePlugin.
DEFAULT_CHARS_PER_TOKEN: float = 3.5


def _parse_message_rows(rows: list[tuple]) -> list[dict]:
    """Parse (message_json, message_type) rows into tagged message dicts."""
    result = []
    for row in rows:
        msg = json.loads(row[0])
        msg["_message_type"] = MessageType(row[1])
        result.append(msg)
    return result


class ConversationLog:
    """Per-channel conversation history with SQLite persistence and compaction.

    Maintains an in-memory message list that is kept in sync with a
    ``message_log`` table in the provided database. When the estimated
    token count approaches the configured context limit, older messages
    are summarized via the LLM and replaced with a single summary entry
    so the history stays within bounds.

    Attributes:
        db: The aiosqlite connection used for persistence.
        channel_id: Identifies the channel this log belongs to.
        messages: In-memory list of message dicts (role/content pairs).
        system_prompt: The system prompt prepended by ``build_prompt``.
    """

    def __init__(self, db: aiosqlite.Connection, channel_id: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN):
        self.db = db
        self.channel_id = channel_id
        self.messages: list[dict] = []
        self.system_prompt: str = ""
        self.chars_per_token: float = chars_per_token

    async def load(self) -> None:
        """Load messages from DB for this channel, ordered by id.

        If a summary row exists, loads only that summary plus non-summary rows
        whose timestamp is after the summary boundary (timestamp > summary_ts).
        Otherwise loads all rows ordered by timestamp then id.
        Logs the message count at DEBUG level.
        """
        async with self.db.execute(
            "SELECT id, message, timestamp FROM message_log "
            "WHERE channel_id = ? AND message_type = 'summary' "
            "ORDER BY id DESC LIMIT 1",
            (self.channel_id,),
        ) as cursor:
            summary_row = await cursor.fetchone()

        if summary_row:
            _, summary_message, summary_ts = summary_row
            summary_msg = json.loads(summary_message)
            summary_msg["_message_type"] = MessageType.SUMMARY
            # Load non-summary rows — rows with timestamp <= summary_ts are excluded.
            async with self.db.execute(
                "SELECT message, message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "AND timestamp > ? ORDER BY id",
                (self.channel_id, summary_ts),
            ) as cursor:
                rows = await cursor.fetchall()
            loaded = _parse_message_rows(rows)
            self.messages = [summary_msg] + loaded
        else:
            async with self.db.execute(
                "SELECT message, message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "ORDER BY timestamp, id",
                (self.channel_id,),
            ) as cursor:
                rows = await cursor.fetchall()
            self.messages = _parse_message_rows(rows)

        logger.debug(
            "messages loaded from DB",
            extra={"count": len(self.messages)},
        )

    async def append(self, message: dict, message_type: MessageType = MessageType.MESSAGE) -> None:
        """Append to both self.messages and persistent log.

        Args:
            message: The message dict to store (e.g. ``{"role": "user", "content": "..."}``).
            message_type: Persistence category for the entry. Defaults to
                ``MessageType.MESSAGE``.

        A shallow copy of ``message`` is made before adding ``_message_type``
        to the in-memory entry, so the caller's dict is not mutated. The
        original (untagged) dict is written to the DB.

        Logs the role and content length at DEBUG level.
        """
        tagged = dict(message)
        tagged["_message_type"] = message_type
        self.messages.append(tagged)
        await self._persist(message, message_type)

        logger.debug(
            "message appended",
            extra={
                "role": message.get("role"),
                "content_length": len(message.get("content", "")),
            },
        )

    def token_estimate(self) -> int:
        """Rough token count: int(total_chars / chars_per_token).

        Uses a simple character-based heuristic (~3.5 characters per token by
        default). Includes system prompt length plus all message content lengths.
        Non-string content (None, lists) is treated as 0 characters.
        """
        total_chars = len(self.system_prompt)
        for msg in self.messages:
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = ""
            total_chars += len(content)
        return int(total_chars / self.chars_per_token)

    async def replace_with_summary(self, summary_msg: dict, retain_count: int) -> None:
        """Replace older messages with a summary, retaining the most recent entries.

        Updates the in-memory message list and the DB atomically. The in-memory
        list is updated BEFORE DB writes because the DB logic derives
        num_retained_total from len(self.messages) - 1.

        Args:
            summary_msg: Untagged summary dict (role/content). Will be tagged
                with _message_type=SUMMARY in-memory; stored without the tag in DB.
            retain_count: Number of most-recent messages to keep alongside the
                summary. Must not exceed len(self.messages).

        Raises:
            ValueError: If retain_count > len(self.messages).

        Logs compaction completion with before/after counts at INFO level.
        """
        if retain_count > len(self.messages):
            raise ValueError(
                f"retain_count ({retain_count}) exceeds len(messages) ({len(self.messages)})"
            )

        retained = self.messages[-retain_count:] if retain_count > 0 else []
        tagged = {**summary_msg, "_message_type": MessageType.SUMMARY}
        messages_before = len(self.messages)

        # Set in-memory state BEFORE DB writes — DB logic reads len(self.messages) - 1.
        self.messages = [tagged] + retained

        # --- DB persistence ---
        # Count all non-summary entries in the retained set.
        # self.messages = [summary_msg] + retained after the assignment above.
        num_retained_total = len(self.messages) - 1

        if num_retained_total > 0:
            # Find the timestamp of the oldest retained message.
            async with self.db.execute(
                "SELECT timestamp FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "ORDER BY id DESC LIMIT 1 OFFSET ?",
                (self.channel_id, num_retained_total - 1),
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                summary_ts = row[0] - 1e-6
            else:
                summary_ts = time.time()
        else:
            # Everything compacted — summary timestamp = now, so nothing
            # passes the filter until new messages arrive.
            summary_ts = time.time()

        await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            (self.channel_id, json.dumps(summary_msg), summary_ts, MessageType.SUMMARY),
        )
        await self.db.commit()

        logger.info(
            "compaction completed",
            extra={
                "channel_id": self.channel_id,
                "messages_before": messages_before,
                "messages_after": len(self.messages),
            },
        )

    def build_prompt(self) -> list[dict]:
        """Return [system_message, *self.messages].

        The system message is prepended to the in-memory message list.
        This does not modify self.messages. Strips internal _message_type
        metadata from each message dict before returning.
        """
        cleaned = []
        for msg in self.messages:
            # Strip _message_type: it is internal metadata and must not be sent to the LLM.
            if "_message_type" in msg:
                msg = {k: v for k, v in msg.items() if k != "_message_type"}
            cleaned.append(msg)
        return [{"role": "system", "content": self.system_prompt}] + cleaned

    async def remove_by_type(self, message_type: MessageType) -> int:
        """Remove all entries of a given type from in-memory state only.

        Returns the number of entries removed. Plugins use this to clean up
        injected entries before re-injecting fresh ones, preventing
        unbounded accumulation. Old rows remain in the DB but become
        invisible after the next compaction (their timestamps fall below
        the summary boundary).

        Raises ValueError if message_type is MESSAGE or SUMMARY — those types
        are managed by compaction, not by this method.

        Args:
            message_type: The type to remove. Must not be MESSAGE or SUMMARY.

        Returns:
            The number of in-memory entries removed.
        """
        if message_type in (MessageType.MESSAGE, MessageType.SUMMARY):
            raise ValueError(
                f"Cannot remove {message_type.value!r} entries — "
                f"use compaction for MESSAGE and SUMMARY lifecycle"
            )
        before = len(self.messages)
        self.messages = [
            m for m in self.messages
            if m.get("_message_type") != message_type
        ]
        removed = before - len(self.messages)
        logger.debug(
            "removed entries by type",
            extra={
                "channel_id": self.channel_id,
                "message_type": message_type.value,
                "count": removed,
            },
        )
        return removed

    async def _persist(self, message: dict, message_type: MessageType = MessageType.MESSAGE) -> None:
        """INSERT a single row into message_log.

        Args:
            message: The message dict to serialize. Must not contain
                ``_message_type``; callers are responsible for stripping it.
            message_type: Value written to the ``message_type`` column.
                Defaults to ``MessageType.MESSAGE``.

        Serializes ``message`` to JSON and stores it with the current timestamp.
        """
        await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, ?)",
            (self.channel_id, json.dumps(message), time.time(), message_type),
        )
        await self.db.commit()


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


