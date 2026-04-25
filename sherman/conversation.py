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

import json
import logging
import time

import aiosqlite

from sherman.llm import LLMClient

logger = logging.getLogger(__name__)


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
        """Load messages from DB for this channel, ordered by timestamp.

        Logs the message count at DEBUG level.
        """
        async with self.db.execute(
            "SELECT message FROM message_log WHERE channel_id = ? ORDER BY timestamp",
            (self.channel_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        self.messages = [json.loads(row[0]) for row in rows]

        logger.debug(
            "messages loaded from DB",
            extra={"count": len(self.messages)},
        )

    async def append(self, message: dict) -> None:
        """Append to both self.messages and persistent log.

        Logs the role and content length at DEBUG level.
        """
        self.messages.append(message)
        await self._persist(message)

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
        """
        total_chars = len(self.system_prompt)
        for msg in self.messages:
            total_chars += len(msg.get("content", ""))
        return int(total_chars / 3.5)

    async def compact_if_needed(self, client: LLMClient, max_tokens: int) -> None:
        """If token_estimate >= 80% of max_tokens AND len(messages) > 20,
        summarize older messages. Keep last 20, replace rest with summary.

        Logs a WARNING when compaction is triggered and INFO when completed
        with before/after message counts.
        """
        if self.token_estimate() < max_tokens * 0.8:
            return
        if len(self.messages) <= 20:
            return

        logger.warning(
            "compaction triggered (approaching context limit)",
            extra={"channel_id": self.channel_id},
        )

        older = self.messages[:-20]
        last_20 = self.messages[-20:]

        summary_text = await self._summarize(client, older)
        summary_msg = {
            "role": "assistant",
            "content": f"[Summary of earlier conversation]\n{summary_text}",
        }
        messages_before = len(self.messages)
        self.messages = [summary_msg] + last_20

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
        This does not modify self.messages.
        """
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    async def _persist(self, message: dict) -> None:
        """INSERT into message_log.

        Serializes the message dict to JSON and stores with timestamp.
        """
        await self.db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
            (self.channel_id, json.dumps(message), time.time()),
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
            timestamp REAL NOT NULL
        )"""
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_log_channel ON message_log (channel_id, timestamp)"
    )
    await db.commit()


