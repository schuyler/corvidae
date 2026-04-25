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


class TestDurableCompaction:
    async def test_init_db_adds_is_summary_column(self):
        """init_db() must create the is_summary column on message_log."""
        async with aiosqlite.connect(":memory:") as conn:
            await init_db(conn)

            async with conn.execute("PRAGMA table_info(message_log)") as cursor:
                columns = await cursor.fetchall()

            column_names = [col[1] for col in columns]
            assert "is_summary" in column_names, (
                f"is_summary column not found; got columns: {column_names}"
            )

    async def test_init_db_idempotent_with_existing_column(self):
        """Calling init_db() twice must not raise an error."""
        async with aiosqlite.connect(":memory:") as conn:
            await init_db(conn)
            # Second call must complete without exception
            await init_db(conn)

            async with conn.execute("PRAGMA table_info(message_log)") as cursor:
                columns = await cursor.fetchall()

            column_names = [col[1] for col in columns]
            assert "is_summary" in column_names

    async def test_compact_persists_summary_row(self, db):
        """After compaction, a row with is_summary=1 must exist in the DB."""
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        # 25 messages to trigger compaction (last 20 retained)
        base_ts = time.time()
        for i in range(25):
            msg = {"role": "user", "content": "x" * 100}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        await db.commit()
        conv.messages = [{"role": "user", "content": "x" * 100} for _ in range(25)]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "mock summary"}}]}
        )

        await conv.compact_if_needed(mock_client, max_tokens=100)

        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND is_summary = 1",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1, f"Expected 1 summary row, got {row[0]}"

        # Verify the persisted message content is the summary
        async with db.execute(
            "SELECT message FROM message_log WHERE channel_id = ? AND is_summary = 1",
            ("chan1",),
        ) as cursor:
            summary_row = await cursor.fetchone()
        assert json.loads(summary_row[0]) == {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nmock summary",
        }

    async def test_load_skips_pre_summary_messages(self, db):
        """load() must return only the summary row + messages after it."""
        base_ts = time.time()

        old_msg = {"role": "user", "content": "old message"}
        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nthe summary"}
        new_msg1 = {"role": "user", "content": "new message 1"}
        new_msg2 = {"role": "assistant", "content": "new message 2"}

        # Old message before the summary
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 0)",
            ("chan1", json.dumps(old_msg), base_ts),
        )
        # Summary row
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 1)",
            ("chan1", json.dumps(summary_msg), base_ts + 1),
        )
        # Newer non-summary messages after the summary
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 0)",
            ("chan1", json.dumps(new_msg1), base_ts + 2),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 0)",
            ("chan1", json.dumps(new_msg2), base_ts + 3),
        )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert conv.messages == [summary_msg, new_msg1, new_msg2], (
            f"Expected [summary, new1, new2], got: {conv.messages}"
        )

    async def test_load_without_summary_loads_all(self, db):
        """When no summary rows exist, load() loads everything (existing behavior)."""
        base_ts = time.time()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        for i, msg in enumerate(messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert conv.messages == messages

    async def test_repeated_compaction_load_correct(self, db):
        """After two compactions, load() returns only the newest summary + its retained messages."""
        base_ts = time.time()

        # Simulate first compaction: insert 25 original messages + first summary row
        original_messages = [
            {"role": "user", "content": f"original {i}"}
            for i in range(25)
        ]
        for i, msg in enumerate(original_messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 0)",
                ("chan1", json.dumps(msg), base_ts + i),
            )

        # First summary row: timestamp just before the oldest retained message
        # (retained = last 20 of originals, oldest retained = original_messages[5] at base_ts+5)
        first_summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nfirst summary"}
        first_summary_ts = base_ts + 5 - 0.001
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 1)",
            ("chan1", json.dumps(first_summary_msg), first_summary_ts),
        )

        # Add 6 more messages after first compaction (so we have >20 messages again
        # when counting summary + 20 retained + 6 new = 27 messages in memory)
        extra_messages = [
            {"role": "user", "content": f"extra {i}"}
            for i in range(6)
        ]
        extra_base_ts = base_ts + 25
        for i, msg in enumerate(extra_messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 0)",
                ("chan1", json.dumps(msg), extra_base_ts + i),
            )

        await db.commit()

        # Load from DB to exercise the load() → compact_if_needed() integration path
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        await conv.load()

        # After load(), conv.messages should be:
        # [first_summary] + original[5:25] + extra[0:6] = 27 messages
        # (first_summary + 20 retained originals + 6 extras)
        assert len(conv.messages) == 27
        assert conv.messages[0] == first_summary_msg

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "second summary"}}]}
        )

        # Capture last 20 before compaction mutates conv.messages
        expected_retained = conv.messages[-20:]

        # Trigger compaction: 27 messages, use low max_tokens so token estimate
        # exceeds 80% threshold
        await conv.compact_if_needed(mock_client, max_tokens=10)

        second_summary_msg = {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nsecond summary",
        }
        # After compaction: [second_summary] + last 20 of the original 27
        assert conv.messages[0] == second_summary_msg
        assert conv.messages[1:] == expected_retained
        assert len(conv.messages) == 21

        # Now verify load() returns only the newest summary + its retained messages
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()

        assert conv2.messages[0] == second_summary_msg, (
            f"First message after load() should be second summary, got: {conv2.messages[0]}"
        )
        assert len(conv2.messages) == 21, (
            f"Expected 21 messages (1 summary + 20 retained), got: {len(conv2.messages)}"
        )
        # The first summary must NOT appear in the loaded messages
        assert first_summary_msg not in conv2.messages, (
            "First (old) summary must not appear after loading with second summary present"
        )
