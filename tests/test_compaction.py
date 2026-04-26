"""Tests for sherman.compaction.CompactionPlugin.

These tests are the RED phase: they fail until sherman/compaction.py is created
and ConversationLog.replace_with_summary() is implemented.
"""

import json
import time
from unittest.mock import AsyncMock, patch

import aiosqlite

from sherman.conversation import ConversationLog, init_db


class TestConversationLogCompaction:
    async def test_compact_below_threshold(self, db):
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        # Small messages that won't exceed 80% of max_tokens
        conv.messages = [{"role": "user", "content": "hi"} for _ in range(5)]
        original_messages = list(conv.messages)

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()

        mock_client = AsyncMock()
        result = await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=10000
        )

        assert result is None
        assert conv.messages == original_messages
        mock_client.chat.assert_not_called()

    async def test_compact_triggers(self, db):
        # Token math: 10 messages x 35 chars, max_tokens=50.
        # token_estimate = int(350/3.5) = 100; threshold = 50*0.8 = 40 → triggers.
        # retain_budget = int(50*0.5) = 25; each msg = int(35/3.5) = 10 tokens.
        # Walk: msg1→10≤25 count=1; msg2→20≤25 count=2; msg3→30>25 AND count>0 → break.
        # retain_count = 2 → conv.messages = [summary, msgs[-2], msgs[-1]]
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(10)
        ]
        expected_retained = conv.messages[-2:]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "mock summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        result = await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        from sherman.conversation import MessageType
        assert result is True
        assert len(conv.messages) == 3  # summary msg + 2 retained
        assert conv.messages[0] == {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nmock summary",
            "_message_type": MessageType.SUMMARY,
        }
        assert conv.messages[1:] == expected_retained

    async def test_compact_few_messages(self, db):
        # 5 messages exceeding token threshold but len <= 5 -> no compaction (new guard).
        # 5 messages x 100 chars = 500 chars; token_estimate = int(500/3.5) = 142;
        # max_tokens=100; threshold=80 → outer guard passes.
        # len(messages) = 5 → inner guard (<= 5) blocks compaction.
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "x" * 100}
            for _ in range(5)
        ]
        original_messages = list(conv.messages)

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "mock summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        result = await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=100
        )

        assert result is None
        assert conv.messages == original_messages
        mock_client.chat.assert_not_called()

    async def test_compact_token_budget_large_messages(self, db):
        # Large messages (350 chars each = 100 tokens each) → fewer retained.
        # 10 messages, max_tokens=1000.
        # token_estimate = int(3500/3.5) = 1000; threshold = 800 → triggers; len=10>5.
        # retain_budget = int(1000*0.5) = 500.
        # Walk: cumulative: 100, 200, 300, 400, 500, 600>500 AND count>0 → break.
        # retain_count = 5.
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "a" * 350}
            for _ in range(10)
        ]
        expected_retained = conv.messages[-5:]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "large summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=1000
        )

        assert len(conv.messages) == 6  # summary + 5 retained
        assert conv.messages[1:] == expected_retained

    async def test_compact_token_budget_small_messages(self, db):
        # Small messages (35 chars each = 10 tokens each) → more retained.
        # 15 messages, max_tokens=100.
        # token_estimate = int(525/3.5) = 150; threshold = 80 → triggers; len=15>5.
        # retain_budget = int(100*0.5) = 50.
        # Walk: 10,20,30,40,50 (count=5), next: 60>50 AND count>0 → break.
        # retain_count = 5.
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(15)
        ]
        expected_retained = conv.messages[-5:]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "small summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=100
        )

        assert len(conv.messages) == 6  # summary + 5 retained
        assert conv.messages[1:] == expected_retained

    async def test_compact_none_content_no_crash(self, db):
        # A message with content=None must not raise TypeError in token_estimate()
        # or in the backward walk. The None is treated as "" (0 tokens).
        # Setup: 9 messages with 35-char content + 1 message with content=None.
        # The None-content message contributes 0 tokens to both token_estimate()
        # and the walk.
        # token_estimate (with fix): int((9*35 + 0)/3.5) = int(315/3.5) = 90;
        # max_tokens=100; threshold=80 → triggers; len=10>5 → proceeds.
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(9)
        ] + [{"role": "user", "content": None}]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "none-content summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        # Must not raise TypeError
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=100
        )

        from sherman.conversation import MessageType
        assert conv.messages[0] == {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nnone-content summary",
            "_message_type": MessageType.SUMMARY,
        }

    async def test_compact_list_content_treated_as_zero_tokens(self, db):
        # A message with list content is treated as 0 tokens in the backward walk.
        # Setup: 6 messages with 35-char content + 1 message with list content.
        # max_tokens=50; retain_budget=25.
        # The list-content message is the last message (position -1 in reversed walk).
        # Walk: list-msg → 0+0=0≤25 count=1; msg6→0+10=10≤25 count=2;
        #       msg5→10+10=20≤25 count=3; msg4→20+10=30>25 AND count>0 → break.
        # retain_count=3; retained=[msgs[4], msgs[5], list_msg].
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        string_messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(6)
        ]
        list_msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        conv.messages = string_messages + [list_msg]

        # token_estimate: list content → 0 chars → int(6*35/3.5) = 60 ≥ 40 → triggers
        # len=7>5 → proceeds.
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "list-content summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        # retain_count=3: [string_messages[4], string_messages[5], list_msg]
        assert len(conv.messages) == 4  # summary + 3 retained
        assert conv.messages[-1] == list_msg

    async def test_compact_all_fit_no_compaction(self, db):
        # When all messages fit within the retain budget, compaction is skipped
        # even if the outer token threshold fires (driven by system_prompt size).
        # 10 messages with 1-char content: int(1/3.5)=0 tokens each.
        # system_prompt of 1000 chars drives token_estimate over 80%:
        #   token_estimate = int((1000+10)/3.5) = int(1010/3.5) = 288 ≥ 160 → triggers.
        # retain_budget = int(200*0.5) = 100; each msg = 0 tokens.
        # Walk: all 10 msgs accumulate 0 tokens → retain_count=10=len(messages) → early return.
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = "a" * 1000
        conv.messages = [
            {"role": "user", "content": "x"}
            for _ in range(10)
        ]
        original_messages = list(conv.messages)

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "should not summarize"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        result = await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=200
        )

        assert result is None
        assert conv.messages == original_messages
        mock_client.chat.assert_not_called()

    async def test_compact_large_single_message_retained(self, db):
        # A single message that exceeds the entire retain budget is still retained
        # (the "at least one" rule: break fires only when retain_count > 0).
        # Setup: 8 small messages (35 chars each = 10 tokens) + 1 huge message (350 chars = 100 tokens).
        # max_tokens=100; retain_budget=50.
        # token_estimate = int((8*35+350)/3.5) = int(630/3.5) = 180 ≥ 80 → triggers; len=9>5.
        # Walk backward: huge_msg: 0+100>50 BUT retain_count=0 → don't break → count=1, retain=100.
        #   small8: 100+10>50 AND count=1>0 → break.
        # retain_count=1; retained=[huge_msg].
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        small_messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(8)
        ]
        huge_message = {"role": "user", "content": "a" * 350}
        conv.messages = small_messages + [huge_message]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "huge summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=100
        )

        assert len(conv.messages) == 2  # summary + huge_message
        assert conv.messages[1] == huge_message

    async def test_compact_backward_walk_boundary(self, db):
        # Construct messages of known sizes to verify retain_count and which messages
        # are retained at the walk boundary.
        # Messages: A-F (35 chars = 10 tokens each), G (105 chars = 30 tokens).
        # max_tokens=100; retain_budget=50.
        # token_estimate = int((6*35+105)/3.5) = int(315/3.5) = 90 ≥ 80 → triggers; len=7>5.
        # Walk backward (msgs ordered A,B,C,D,E,F,G → reversed: G,F,E,D,...):
        #   G: 0+30=30≤50 → count=1, retain=30
        #   F: 30+10=40≤50 → count=2, retain=40
        #   E: 40+10=50≤50 → count=3, retain=50
        #   D: 50+10=60>50 AND count=3>0 → break
        # retain_count=3; retained=[E, F, G].
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        msg_a = {"role": "user", "content": "A" * 35}
        msg_b = {"role": "user", "content": "B" * 35}
        msg_c = {"role": "user", "content": "C" * 35}
        msg_d = {"role": "user", "content": "D" * 35}
        msg_e = {"role": "user", "content": "E" * 35}
        msg_f = {"role": "user", "content": "F" * 35}
        msg_g = {"role": "user", "content": "G" * 105}
        conv.messages = [msg_a, msg_b, msg_c, msg_d, msg_e, msg_f, msg_g]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "boundary summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=100
        )

        assert len(conv.messages) == 4  # summary + 3 retained
        assert conv.messages[1] == msg_e
        assert conv.messages[2] == msg_f
        assert conv.messages[3] == msg_g


class TestDurableCompaction:
    async def test_init_db_has_message_type_column(self):
        """init_db() must create the message_type column on message_log."""
        async with aiosqlite.connect(":memory:") as conn:
            await init_db(conn)

            async with conn.execute("PRAGMA table_info(message_log)") as cursor:
                columns = await cursor.fetchall()

            column_names = [col[1] for col in columns]
            assert "message_type" in column_names, (
                f"message_type column not found; got columns: {column_names}"
            )

    async def test_init_db_idempotent(self):
        """Calling init_db() twice must not raise an error."""
        async with aiosqlite.connect(":memory:") as conn:
            await init_db(conn)
            # Second call must complete without exception
            await init_db(conn)

            async with conn.execute("PRAGMA table_info(message_log)") as cursor:
                columns = await cursor.fetchall()

            column_names = [col[1] for col in columns]
            assert "message_type" in column_names

    async def test_compact_persists_summary_row(self, db):
        """After compaction, exactly one message_type='summary' row exists; retained count matches token-walk."""
        # Token math: 10 messages x 35 chars, max_tokens=50.
        # token_estimate = int(350/3.5) = 100 ≥ 40 → triggers; len=10>5 → proceeds.
        # retain_budget=25; each msg = 10 tokens.
        # Walk: count=1(10), count=2(20), 30>25 AND count>0 → break. retain_count=2.
        # After compaction: conv.messages = [summary, msgs[-2], msgs[-1]] → 3 messages.
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        base_ts = time.time()
        for i in range(10):
            msg = {"role": "user", "content": "a" * 35}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        await db.commit()
        conv.messages = [{"role": "user", "content": "a" * 35} for _ in range(10)]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "mock summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1, f"Expected 1 summary row, got {row[0]}"

        # Verify the persisted message content is the summary
        async with db.execute(
            "SELECT message FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            summary_row = await cursor.fetchone()
        assert json.loads(summary_row[0]) == {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nmock summary",
        }

        # Verify retained count: token-based walk retains 2 messages (not 20)
        assert len(conv.messages) == 3, (
            f"Expected 3 messages (1 summary + 2 retained), got {len(conv.messages)}"
        )

    async def test_load_returns_summary_plus_remaining_messages(self, db):
        """load() must return the summary row + all non-summary rows.

        After compaction, replace_with_summary deletes summarized messages, so
        only retained + new messages remain alongside the summary row.
        """
        base_ts = time.time()

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nthe summary"}
        retained_msg = {"role": "user", "content": "retained message"}
        new_msg = {"role": "assistant", "content": "new message"}

        # Retained message (lower id than summary — this is normal after compaction)
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(retained_msg), base_ts),
        )
        # Summary row
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'summary')",
            ("chan1", json.dumps(summary_msg), base_ts + 1),
        )
        # New message added after compaction
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(new_msg), base_ts + 2),
        )
        await db.commit()

        from sherman.conversation import MessageType
        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        expected_summary = {**summary_msg, "_message_type": MessageType.SUMMARY}
        expected_retained = {**retained_msg, "_message_type": MessageType.MESSAGE}
        expected_new = {**new_msg, "_message_type": MessageType.MESSAGE}
        assert conv.messages == [expected_summary, expected_retained, expected_new], (
            f"Expected [summary, retained, new], got: {conv.messages}"
        )

    async def test_load_without_summary_loads_all(self, db):
        """When no summary rows exist, load() loads everything (existing behavior)."""
        from sherman.conversation import MessageType
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

        expected = [{**msg, "_message_type": MessageType.MESSAGE} for msg in messages]
        assert conv.messages == expected

    async def test_repeated_compaction_load_correct(self, db):
        """After two compactions, load() returns only the newest summary + its retained messages.

        Simulates post-first-compaction DB state (summarized messages already
        deleted by replace_with_summary), then triggers a second compaction and
        verifies load() correctness.

        Token math:
          - "a" * 35 → int(35/3.5) = 10 tokens each.
          - max_tokens=50: retain_budget=25; threshold=40.
          - Walk: count=1(10), count=2(20), 30>25 AND count>0 → break. retain_count=2.

        Post-first-compaction DB (simulated):
          - 2 retained messages (ids 1-2) + first_summary (id 3) + 4 extra messages (ids 4-7).
          - load() returns [first_summary, retained[0], retained[1], extra[0..3]] = 7 msgs.
          - token_estimate = int((48 + 6*35)/3.5) = 73 ≥ 40 → triggers; len=7>5.

        Second compaction:
          - Walk: extra[3](10), extra[2](20), extra[1](30>25) → break. retain_count=2.
          - replace_with_summary deletes old rows, inserts second_summary.
          - load() returns [second_summary, extra[2], extra[3]] = 3 msgs.
        """
        base_ts = time.time()

        from sherman.conversation import MessageType
        # 2 retained messages from first compaction (simulating cleaned-up state)
        retained_messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(2)
        ]
        for i, msg in enumerate(retained_messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )

        # First summary row
        first_summary_msg = {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nfirst summary",
        }
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'summary')",
            ("chan1", json.dumps(first_summary_msg), base_ts + 5),
        )

        # 4 extra messages added after first compaction
        extra_messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(4)
        ]
        extra_base_ts = base_ts + 10
        for i, msg in enumerate(extra_messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), extra_base_ts + i),
            )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        await conv.load()

        # load() returns: [first_summary] + [retained[0], retained[1], extra[0..3]] = 7 msgs
        assert len(conv.messages) == 7, (
            f"Expected 7 messages after load(), got {len(conv.messages)}"
        )
        first_summary_msg_tagged = {**first_summary_msg, "_message_type": MessageType.SUMMARY}
        assert conv.messages[0] == first_summary_msg_tagged

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "second summary"}}]}
        )

        expected_retained = conv.messages[-2:]

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        second_summary_msg = {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nsecond summary",
        }
        second_summary_msg_tagged = {**second_summary_msg, "_message_type": MessageType.SUMMARY}
        assert conv.messages[0] == second_summary_msg_tagged
        assert conv.messages[1:] == expected_retained
        assert len(conv.messages) == 3

        # Verify load() on a fresh ConversationLog
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()

        assert conv2.messages[0] == second_summary_msg_tagged
        assert len(conv2.messages) == 3, (
            f"Expected 3 messages (1 summary + 2 retained), got: {len(conv2.messages)}"
        )
        assert first_summary_msg_tagged not in conv2.messages


class TestIdBasedSummaryOrdering:
    async def test_load_uses_id_not_timestamp_for_summary_cutoff(self, db):
        """load() must use id-based cutoff, not timestamp, when ordering after a summary.

        Insert messages with identical timestamps but different IDs. Insert a summary
        row with the same timestamp as those messages. With the current timestamp-based
        code, load() uses ``timestamp > summary_ts``, which means messages with
        ``timestamp == summary_ts`` are excluded. The fix uses ``id > summary_id`` so
        messages inserted after the summary are correctly included regardless of their
        timestamp.
        """
        shared_ts = 1_000_000.0  # fixed timestamp — all rows share it

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nthe summary"}
        msg_after_1 = {"role": "user", "content": "after summary 1"}
        msg_after_2 = {"role": "assistant", "content": "after summary 2"}

        from sherman.conversation import MessageType
        # Insert the summary row first so it gets a lower id
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'summary')",
            ("chan1", json.dumps(summary_msg), shared_ts),
        )
        # Insert two non-summary messages with the *same* timestamp as the summary.
        # These have higher ids and should be returned by load() after the fix.
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(msg_after_1), shared_ts),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(msg_after_2), shared_ts),
        )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        # After the fix (id-based): [summary, msg_after_1, msg_after_2]
        expected = [
            {**summary_msg, "_message_type": MessageType.SUMMARY},
            {**msg_after_1, "_message_type": MessageType.MESSAGE},
            {**msg_after_2, "_message_type": MessageType.MESSAGE},
        ]
        assert conv.messages == expected, (
            f"Expected [summary, msg_after_1, msg_after_2], got: {conv.messages}"
        )

    async def test_replace_with_summary_no_timestamp_arithmetic(self, db):
        """replace_with_summary must not rely on row[0] - 0.001 timestamp arithmetic.

        After compaction where all messages share the same timestamp, the summary row
        stored by the current code has timestamp = shared_ts - 0.001. A fresh load()
        then uses ``timestamp > shared_ts - 0.001``, which includes ALL original
        messages because shared_ts > shared_ts - 0.001. The fix stores the summary
        with its own auto-assigned id (no timestamp manipulation), and load() uses
        ``id > summary_id`` so only the truly retained messages are returned.
        """
        shared_ts = 1_000_000.0

        # Insert 10 messages all with the same timestamp
        messages = [{"role": "user", "content": "a" * 35} for _ in range(10)]
        for msg in messages:
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), shared_ts),
            )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        conv.messages = list(messages)

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "mock summary"}}]}
        )

        # Token math: 10 msgs x 35 chars → int(350/3.5)=100 ≥ 40 → triggers; len=10>5.
        # retain_count=2; conv.messages = [summary, msgs[-2], msgs[-1]].
        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        # Verify the summary row: with the fix the timestamp should NOT be shared_ts - 0.001.
        # The current code DOES write shared_ts - 0.001 — detecting this is the failure signal.
        async with db.execute(
            "SELECT timestamp FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            summary_row = await cursor.fetchone()

        assert summary_row is not None, "Summary row must exist after compaction"
        # With the fix: summary row's timestamp is NOT shared_ts - 0.001.
        assert summary_row[0] != shared_ts - 0.001, (
            f"Summary timestamp should not use timestamp arithmetic (row[0] - 0.001); "
            f"got {summary_row[0]}, expected something other than {shared_ts - 0.001}"
        )

    async def test_rapid_messages_with_same_timestamp(self, db):
        """Compaction followed by load() must return correct messages when all messages
        share the same timestamp.

        This is the specific bug the fix addresses. With timestamp arithmetic, the
        summary is stored at shared_ts - 0.001. When load() runs, it fetches non-summary
        rows with ``timestamp > shared_ts - 0.001``. Because all 10 original messages
        have timestamp == shared_ts > shared_ts - 0.001, ALL of them are returned —
        even the ones that were supposed to be summarized. The loaded message list is
        then 11 items (summary + 10) instead of the correct 3 (summary + 2 retained).

        After the fix (id-based): only the 2 truly retained messages follow the summary.
        """
        shared_ts = 1_000_000.0

        from sherman.conversation import MessageType
        # Insert 10 messages all with the exact same timestamp
        messages = [{"role": "user", "content": "a" * 35} for _ in range(10)]
        for msg in messages:
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), shared_ts),
            )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        conv.messages = list(messages)

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "rapid summary"}}]}
        )

        # Token math: 10 msgs x 35 chars → int(350/3.5)=100 ≥ 40 → triggers; len=10>5.
        # retain_budget=25; each msg=10 tokens.
        # Walk: count=1(10), count=2(20), 30>25 AND count>0 → break. retain_count=2.
        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        expected_summary_msg = {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nrapid summary",
            "_message_type": MessageType.SUMMARY,
        }
        # In-memory state should be correct regardless of persistence
        assert conv.messages[0] == expected_summary_msg
        assert len(conv.messages) == 3  # summary + 2 retained

        # Now verify that a fresh load() returns the correct 3 messages, not 11.
        conv2 = ConversationLog(db, channel_id="chan1")
        conv2.system_prompt = ""
        await conv2.load()

        assert conv2.messages[0] == expected_summary_msg, (
            f"First message after load() should be the summary, got: {conv2.messages[0]}"
        )
        assert len(conv2.messages) == 3, (
            f"Expected 3 messages (summary + 2 retained), got {len(conv2.messages)}. "
            f"If this is 11, the timestamp arithmetic bug is confirmed."
        )


class TestCompactionWithContext:
    """Tests for compaction behavior when CONTEXT entries are present."""

    async def test_compact_preserves_retained_context(self, db):
        """CONTEXT entries in the retained window must survive compaction.

        After compact_conversation:
        - in-memory: [summary, retained_msg, retained_ctx]
        - DB: summary row + retained non-summary rows present

        Token math: 10 messages x 35 chars → int(350/3.5)=100 ≥ 40 → triggers;
        len=10>5 → proceeds. retain_budget=25; each msg=10 tokens.
        Walk: count=1(10), count=2(20), 30>25 AND count>0 → break. retain_count=2.

        We make the last 2 entries a MESSAGE + CONTEXT pair so both end up retained.
        """
        from sherman.conversation import MessageType

        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""

        base_ts = time.time()
        # Insert 8 ordinary messages and 1 context entry and 1 message to DB
        for i in range(8):
            msg = {"role": "user", "content": "a" * 35}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        retained_msg = {"role": "user", "content": "b" * 35}
        retained_ctx = {"role": "system", "content": "c" * 35}
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(retained_msg), base_ts + 8),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'context')",
            ("chan1", json.dumps(retained_ctx), base_ts + 9),
        )
        await db.commit()

        # Set up in-memory state to match DB
        conv.messages = (
            [{"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}] * 8
            + [
                {"role": "user", "content": "b" * 35, "_message_type": MessageType.MESSAGE},
                {"role": "system", "content": "c" * 35, "_message_type": MessageType.CONTEXT},
            ]
        )

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "compacted summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        # In-memory: summary + 2 retained entries (msg + ctx)
        assert len(conv.messages) == 3, (
            f"Expected 3 messages (summary + 2 retained), got {len(conv.messages)}"
        )
        retained_types = {m["_message_type"] for m in conv.messages[1:]}
        assert MessageType.CONTEXT in retained_types, (
            "CONTEXT entry in retained window was not preserved after compaction"
        )

        # DB: context row must still exist for the channel
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] >= 1, (
            "CONTEXT row in retained window was deleted from DB during compaction"
        )

    async def test_compact_deletes_older_context(self, db):
        """CONTEXT entries in the older (summarized) window must be deleted from DB.

        After compaction, CONTEXT rows with id < oldest_retained_id must be gone.

        Setup: 8 MESSAGE + 1 CONTEXT (older window) + 2 MESSAGE (retained).
        retain_count=2 (last 2 message entries fit in token budget).
        After compaction: only the 2 retained MESSAGE rows + summary row should remain.
        The CONTEXT in the older window must be deleted.

        Token math: 9 messages x 35 chars + 1 x 35 chars = 10 x 35 = 350 chars;
        int(350/3.5)=100 ≥ 40 → triggers; len=10>5.
        retain_budget=25; each msg=10 tokens.
        Walk backward: last msg(10≤25, count=1), second-to-last(20≤25, count=2),
        third(30>25 AND count>0) → break. retain_count=2.

        We place the CONTEXT entry at position -3 (in the older window after retain_count=2).
        """
        from sherman.conversation import MessageType

        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""

        base_ts = time.time()
        # 7 MESSAGE entries (older window)
        for i in range(7):
            msg = {"role": "user", "content": "a" * 35}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        # 1 CONTEXT entry (also in older window — position -3)
        older_ctx = {"role": "system", "content": "old context that should be deleted"}
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'context')",
            ("chan1", json.dumps(older_ctx), base_ts + 7),
        )
        # 2 retained MESSAGE entries
        for i in range(2):
            msg = {"role": "user", "content": "b" * 35}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + 8 + i),
            )
        await db.commit()

        # Set up in-memory state to match DB (7 msg + 1 ctx + 2 msg = 10 total)
        conv.messages = (
            [{"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}] * 7
            + [{"role": "system", "content": "old context that should be deleted",
                "_message_type": MessageType.CONTEXT}]
            + [{"role": "user", "content": "b" * 35, "_message_type": MessageType.MESSAGE}] * 2
        )

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "summary with older ctx deleted"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        # DB: context row in the older window must be gone
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 0, (
            f"Expected 0 CONTEXT rows after compaction (older context should be deleted), "
            f"got {row[0]}"
        )


class TestCompactFiltersNonMessage:
    """Tests for compaction filtering behavior from TestMessageType."""

    async def test_compact_filters_non_message_from_older(self, db):
        """compact_conversation must exclude SUMMARY-typed entries from the older
        portion passed to the summarizer.

        Setup: 10 messages that trigger compaction, with the first message
        (index 0, which ends up in 'older') typed as SUMMARY. The summarizer
        must only receive MESSAGE-typed entries, so the SUMMARY-typed dict
        must not appear in the messages list sent to the LLM.
        """
        from sherman.conversation import MessageType

        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""

        # Build 10 messages. The first one is typed SUMMARY to simulate a
        # pre-existing summary entry in the older portion.
        # Token math: 10 msgs x 35 chars → int(350/3.5)=100 ≥ 40 → triggers; len=10>5.
        # retain_count=2 → older = messages[0:8], retained = messages[-2:].
        # The SUMMARY-typed message at index 0 falls in older and must be filtered out.
        conv.messages = [
            {"role": "assistant", "content": "a" * 35, "_message_type": MessageType.SUMMARY},
        ] + [
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}
            for _ in range(9)
        ]

        captured_older = []

        async def capture_summarize(client, messages):
            captured_older.extend(messages)
            return "filtered summary"

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "filtered summary"}}]}
        )

        from sherman.compaction import CompactionPlugin
        plugin = CompactionPlugin()

        # Patch the plugin's _summarize to capture what older list is passed in
        with patch.object(plugin, "_summarize", side_effect=capture_summarize):
            await plugin.compact_conversation(
                conversation=conv, client=mock_client, max_tokens=50
            )

        # The SUMMARY-typed dict (index 0) must not appear in older passed to summarizer
        for msg in captured_older:
            assert msg.get("_message_type") != MessageType.SUMMARY, (
                f"SUMMARY-typed message must be filtered from older before summarization: {msg!r}"
            )
            # _message_type must also be stripped before LLM serialization
            assert "_message_type" not in msg, (
                f"_message_type key must be stripped before passing to _summarize: {msg!r}"
            )
