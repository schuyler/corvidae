"""Tests for sherman.conversation.ConversationLog and init_db."""

import json
import time
from unittest.mock import AsyncMock

import aiosqlite

from sherman.conversation import ConversationLog, init_db


class TestInitDb:
    async def test_init_db_creates_table(self):
        async with aiosqlite.connect(":memory:") as conn:
            await init_db(conn)

            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_log'"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None, "message_log table should exist"

            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_log_channel'"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None, "idx_log_channel index should exist"


class TestConversationLogPersistence:
    async def test_append_persists_message(self, db):
        conv = ConversationLog(db, channel_id="chan1")
        message = {"role": "user", "content": "hello"}

        await conv.append(message)

        async with db.execute(
            "SELECT channel_id, message FROM message_log WHERE channel_id = ?",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None, "Row should exist in message_log"
        assert row[0] == "chan1"
        assert json.loads(row[1]) == message

    async def test_load_restores_messages(self, db):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        base_ts = time.time()
        for i, msg in enumerate(messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert conv.messages == messages

    async def test_load_orders_by_timestamp(self, db):
        msg_early = {"role": "user", "content": "early"}
        msg_late = {"role": "assistant", "content": "late"}
        base_ts = time.time()

        # Insert in reverse order
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
            ("chan1", json.dumps(msg_late), base_ts + 10),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
            ("chan1", json.dumps(msg_early), base_ts),
        )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert conv.messages == [msg_early, msg_late]

    async def test_load_filters_by_channel(self, db):
        msg_chan1 = {"role": "user", "content": "for chan1"}
        msg_chan2 = {"role": "user", "content": "for chan2"}
        base_ts = time.time()

        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
            ("chan1", json.dumps(msg_chan1), base_ts),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
            ("chan2", json.dumps(msg_chan2), base_ts + 1),
        )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert conv.messages == [msg_chan1]


class TestConversationLogTokenEstimate:
    async def test_token_estimate(self, db):
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = "You are helpful."  # 16 chars
        conv.messages = [
            {"role": "user", "content": "Hello there!"},   # 12 chars
            {"role": "assistant", "content": "Hi!"},        # 3 chars
        ]
        # total = 16 + 12 + 3 = 31 chars; int(31 / 3.5) = 8
        assert conv.token_estimate() == int(31 / 3.5)


class TestConversationLogBuildPrompt:
    async def test_build_prompt(self, db):
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = "You are a helpful assistant."
        conv.messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        prompt = conv.build_prompt()

        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        assert prompt == expected


class TestConversationLogCompaction:
    async def test_compact_if_needed_below_threshold(self, db):
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        # Small messages that won't exceed 80% of max_tokens
        conv.messages = [{"role": "user", "content": "hi"} for _ in range(5)]
        original_messages = list(conv.messages)

        mock_client = AsyncMock()
        await conv.compact_if_needed(mock_client, max_tokens=10000)

        assert conv.messages == original_messages
        mock_client.chat.assert_not_called()

    async def test_compact_if_needed_triggers(self, db):
        conv = ConversationLog(db, channel_id="chan1")
        # Set system_prompt and messages so token_estimate() exceeds 80% of max_tokens
        # Each message has 100 chars of content; 25 messages = 2500 chars total
        # token_estimate = int(2500 / 3.5) = 714; max_tokens = 100; threshold = 80
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "x" * 100}
            for _ in range(25)
        ]
        last_20 = conv.messages[-20:]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "mock summary"}}]}
        )

        await conv.compact_if_needed(mock_client, max_tokens=100)

        assert len(conv.messages) == 21  # summary msg + last 20
        assert conv.messages[0] == {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nmock summary",
        }
        assert conv.messages[1:] == last_20

    async def test_compact_if_needed_few_messages(self, db):
        conv = ConversationLog(db, channel_id="chan1")
        # 15 messages exceeding token threshold but <= 20 messages -> no compaction
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "x" * 100}
            for _ in range(15)
        ]
        original_messages = list(conv.messages)

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "mock summary"}}]}
        )

        await conv.compact_if_needed(mock_client, max_tokens=100)

        assert conv.messages == original_messages
        mock_client.chat.assert_not_called()
