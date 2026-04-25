"""Conversation log with SQLite persistence and LLM-based compaction.

This module provides per-channel conversation history management. Messages are
persisted to SQLite and kept in-memory for efficient access. When the estimated
token count approaches the context limit, older messages are summarized via the
LLM and replaced with a single summary entry.

Logging:
    - DEBUG: messages loaded, message appended
    - WARNING: compaction triggered (approaching context limit)
    - INFO: compaction completed (before/after counts)
"""

import enum
import json
import logging
import time

import aiosqlite

from sherman.llm import LLMClient

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

    def __init__(self, db: aiosqlite.Connection, channel_id: str):
        self.db = db
        self.channel_id = channel_id
        self.messages: list[dict] = []
        self.system_prompt: str = ""

    async def load(self) -> None:
        """Load messages from DB for this channel, ordered by id.

        If a summary row exists, loads only that summary plus non-summary rows
        with a higher id. Otherwise loads all rows ordered by timestamp then id.
        Logs the message count at DEBUG level.
        """
        async with self.db.execute(
            "SELECT id, message FROM message_log "
            "WHERE channel_id = ? AND message_type = 'summary' "
            "ORDER BY id DESC LIMIT 1",
            (self.channel_id,),
        ) as cursor:
            summary_row = await cursor.fetchone()

        if summary_row:
            summary_id, summary_message = summary_row
            summary_msg = json.loads(summary_message)
            summary_msg["_message_type"] = MessageType.SUMMARY
            # Load all non-summary rows — summarized messages are deleted
            # during compaction, so only retained + new messages remain.
            async with self.db.execute(
                "SELECT message, message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "ORDER BY id",
                (self.channel_id,),
            ) as cursor:
                rows = await cursor.fetchall()
            loaded = []
            for row in rows:
                msg = json.loads(row[0])
                msg["_message_type"] = MessageType(row[1])
                loaded.append(msg)
            self.messages = [summary_msg] + loaded
        else:
            async with self.db.execute(
                "SELECT message, message_type FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "ORDER BY timestamp, id",
                (self.channel_id,),
            ) as cursor:
                rows = await cursor.fetchall()
            loaded = []
            for row in rows:
                msg = json.loads(row[0])
                msg["_message_type"] = MessageType(row[1])
                loaded.append(msg)
            self.messages = loaded

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
        """Rough token count: int(total_chars / 3.5).

        Uses a simple character-based heuristic: ~3.5 characters per token.
        Includes system prompt length plus all message content lengths.
        Non-string content (None, lists) is treated as 0 characters.
        """
        total_chars = len(self.system_prompt)
        for msg in self.messages:
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = ""
            total_chars += len(content)
        return int(total_chars / 3.5)

    async def compact_if_needed(self, client: LLMClient, max_tokens: int) -> None:
        """If token_estimate >= 80% of max_tokens AND len(messages) > 5,
        summarize older messages using a token-budget backward walk.

        Retains the most-recent messages that fit within 50% of max_tokens,
        summarizes all older messages, and replaces them with a single summary
        entry. Non-string content is treated as 0 tokens.

        Logs a WARNING when compaction is triggered and INFO when completed
        with before/after message counts.
        """
        if self.token_estimate() < max_tokens * 0.8:
            return
        if len(self.messages) <= 5:
            return

        logger.warning(
            "compaction triggered (approaching context limit)",
            extra={"channel_id": self.channel_id},
        )

        retain_budget = int(max_tokens * 0.5)
        retain_count = 0
        retain_tokens = 0
        for msg in reversed(self.messages):
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = ""
            msg_tokens = int(len(content) / 3.5)
            if retain_tokens + msg_tokens > retain_budget and retain_count > 0:
                break
            retain_tokens += msg_tokens
            retain_count += 1

        if retain_count >= len(self.messages):
            return

        older = self.messages[:-retain_count]
        retained = self.messages[-retain_count:]

        # Filter older to MESSAGE-type only; exclude SUMMARY entries from summarizer input.
        older = [m for m in older if m.get("_message_type", MessageType.MESSAGE) == MessageType.MESSAGE]
        # Strip _message_type before passing to LLM to avoid serializing internal metadata.
        older_clean = [{k: v for k, v in m.items() if k != "_message_type"} for m in older]

        summary_text = await self._summarize(client, older_clean)
        summary_msg_untagged = {
            "role": "assistant",
            "content": f"[Summary of earlier conversation]\n{summary_text}",
        }
        summary_msg = {**summary_msg_untagged, "_message_type": MessageType.SUMMARY}
        messages_before = len(self.messages)
        # Update self.messages before _persist_summary — _persist_summary derives
        # num_retained from len(self.messages) - 1, so the list must reflect the
        # post-compaction state (summary + retained) before persistence runs.
        self.messages = [summary_msg] + retained
        # Pass the untagged dict to persistence to avoid serializing _message_type into the DB.
        await self._persist_summary(summary_msg_untagged)

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
        """Remove all entries of a given type from both memory and DB.

        Returns the number of entries removed. Plugins use this to clean up
        injected entries before re-injecting fresh ones, preventing
        unbounded accumulation.

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
        await self.db.execute(
            "DELETE FROM message_log "
            "WHERE channel_id = ? AND message_type = ?",
            (self.channel_id, message_type),
        )
        await self.db.commit()
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

    async def _persist_summary(self, summary_msg: dict) -> None:
        """Persist a compaction summary to the DB.

        Deletes summarized (non-retained) messages and old summary rows,
        then inserts the new summary. After this, the DB contains only
        retained messages and the summary row. load() finds the summary
        and loads all remaining non-summary rows.

        Invariant: self.messages must reflect the current DB state — i.e.,
        len(self.messages) - 1 retained messages must correspond to the
        most recent non-summary rows in the DB. This holds because append()
        keeps in-memory and DB state in sync, and compact_if_needed is the
        only caller.
        """
        # Count all non-summary entries in the retained set.
        # self.messages = [summary_msg] + retained after compact_if_needed runs.
        # retained can include MESSAGE, CONTEXT, and any future non-summary types.
        num_retained_total = len(self.messages) - 1

        if num_retained_total == 0:
            # No non-summary entries in the retained set.
            # Delete all non-summary rows for this channel unconditionally.
            await self.db.execute(
                "DELETE FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary'",
                (self.channel_id,),
            )
        else:
            # Find the id of the oldest retained non-summary row.
            # Since ids are assigned in insertion order and self.messages
            # is loaded/maintained in id order, this corresponds to
            # retained[0] (the oldest entry kept after compaction).
            async with self.db.execute(
                "SELECT id FROM message_log "
                "WHERE channel_id = ? AND message_type != 'summary' "
                "ORDER BY id DESC LIMIT 1 OFFSET ?",
                (self.channel_id, num_retained_total - 1),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                oldest_retained_id = row[0]
                # Delete all non-summary rows before the retained window.
                # This covers MESSAGE rows (summarized away) and CONTEXT rows
                # (they annotated conversation that has been summarized).
                await self.db.execute(
                    "DELETE FROM message_log "
                    "WHERE channel_id = ? AND message_type != 'summary' AND id < ?",
                    (self.channel_id, oldest_retained_id),
                )
            else:
                logger.warning(
                    "_persist_summary: boundary query returned no row; "
                    "skipping pre-retained deletion (possible SUMMARY-in-retained edge case)",
                    extra={"channel_id": self.channel_id, "num_retained_total": num_retained_total},
                )

        # Delete old summary rows
        await self.db.execute(
            "DELETE FROM message_log "
            "WHERE channel_id = ? AND message_type = 'summary'",
            (self.channel_id,),
        )
        # Insert the new summary
        await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            (self.channel_id, json.dumps(summary_msg), time.time(), MessageType.SUMMARY),
        )
        await self.db.commit()

    async def _summarize(self, client: LLMClient, messages: list[dict]) -> str:
        """Ask LLM to summarize messages.

        Sends the messages to the LLM with a system prompt asking for
        a concise summary that preserves key facts, decisions, and context.
        """
        response = await client.chat([
            {"role": "system", "content": "Summarize the following conversation concisely, preserving key facts, decisions, and context that would be needed to continue the conversation."},
            {"role": "user", "content": json.dumps(messages)},
        ])
        return response["choices"][0]["message"]["content"]


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


