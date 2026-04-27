"""Tests for corvidae.compaction.CompactionPlugin.

These tests are the RED phase: they fail until corvidae/compaction.py is created
and ConversationLog.replace_with_summary() is implemented.
"""

import json
import time
from unittest.mock import AsyncMock, patch

import aiosqlite

from corvidae.conversation import ConversationLog, init_db


class TestConversationLogCompaction:
    async def test_compact_below_threshold(self, db):
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        # Small messages that won't exceed 80% of max_tokens
        conv.messages = [{"role": "user", "content": "hi"} for _ in range(5)]
        original_messages = list(conv.messages)

        from corvidae.compaction import CompactionPlugin
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

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        result = await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        from corvidae.conversation import MessageType
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

        from corvidae.compaction import CompactionPlugin
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

        from corvidae.compaction import CompactionPlugin
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

        from corvidae.compaction import CompactionPlugin
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

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        # Must not raise TypeError
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=100
        )

        from corvidae.conversation import MessageType
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

        from corvidae.compaction import CompactionPlugin
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

        from corvidae.compaction import CompactionPlugin
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

        from corvidae.compaction import CompactionPlugin
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

        from corvidae.compaction import CompactionPlugin
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
        """After compaction, 11 total rows (10 original + 1 summary); load() returns 3.

        Append-only: original rows are NOT deleted. The summary row is added
        alongside the originals. load() uses timestamp filtering to return only
        the working set: summary + 2 retained.

        Token math: 10 messages x 35 chars, max_tokens=50.
        token_estimate = int(350/3.5) = 100 ≥ 40 → triggers; len=10>5 → proceeds.
        retain_budget=25; each msg = 10 tokens.
        Walk: count=1(10), count=2(20), 30>25 AND count>0 → break. retain_count=2.
        After compaction: conv.messages = [summary, msgs[-2], msgs[-1]] → 3 messages.
        """
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

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        # Append-only: all 10 original rows + 1 summary = 11 total rows in DB
        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ?",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 11, f"Expected 11 total rows (10 original + 1 summary), got {row[0]}"

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

        # Verify in-memory state
        assert len(conv.messages) == 3, (
            f"Expected 3 messages (1 summary + 2 retained), got {len(conv.messages)}"
        )

        # Verify load() on a fresh ConversationLog returns the correct 3 messages
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()
        assert len(conv2.messages) == 3, (
            f"Expected load() to return 3 messages (summary + 2 retained), got {len(conv2.messages)}"
        )
        from corvidae.conversation import MessageType
        assert conv2.messages[0]["_message_type"] == MessageType.SUMMARY

    async def test_load_returns_summary_plus_remaining_messages(self, db):
        """load() must return summary + messages with timestamp > summary_ts.

        Append-only: the summary's timestamp is set to oldest_retained.ts - 1e-6.
        load() uses WHERE timestamp > summary_ts to filter. The retained message
        has a timestamp GREATER than summary_ts so it passes the filter. An extra
        "old" message below the summary boundary must be excluded.

        Setup:
          - old_msg at base_ts + 0.5  (below summary_ts → excluded)
          - retained_msg at base_ts + 2.0  (timestamp > summary_ts → included)
          - summary at summary_ts = base_ts + 2.0 - 1e-6  (i.e. base_ts + 1.999999)
          - new_msg at base_ts + 3.0  (timestamp > summary_ts → included)

        load() must return [summary, retained_msg, new_msg] — 3 items, NOT 4.
        Current code (no timestamp filter) returns 4, making this test RED.
        After implementation (timestamp filter), old_msg is excluded → passes.
        """
        base_ts = time.time()

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nthe summary"}
        old_msg = {"role": "user", "content": "old message that should be excluded"}
        retained_msg = {"role": "user", "content": "retained message"}
        new_msg = {"role": "assistant", "content": "new message"}

        # In append-only, the summary timestamp = oldest_retained.ts - 1e-6.
        # retained_msg is the oldest retained, so summary_ts < retained_msg timestamp.
        retained_ts = base_ts + 2.0
        summary_ts = retained_ts - 1e-6  # boundary: exactly one microsecond before retained

        # Old message BELOW the summary boundary — must be excluded by timestamp filter
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(old_msg), base_ts + 0.5),
        )
        # Retained message inserted before summary (lower id), but timestamp > summary_ts
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(retained_msg), retained_ts),
        )
        # Summary row with boundary timestamp
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'summary')",
            ("chan1", json.dumps(summary_msg), summary_ts),
        )
        # New message added after compaction
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(new_msg), base_ts + 3.0),
        )
        await db.commit()

        from corvidae.conversation import MessageType
        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        # Must return 3 (summary + retained + new), NOT 4 (current no-filter code returns 4).
        expected_summary = {**summary_msg, "_message_type": MessageType.SUMMARY}
        expected_retained = {**retained_msg, "_message_type": MessageType.MESSAGE}
        expected_new = {**new_msg, "_message_type": MessageType.MESSAGE}
        assert conv.messages == [expected_summary, expected_retained, expected_new], (
            f"Expected [summary, retained, new] (3 items, old_msg excluded), got: {conv.messages}"
        )

    async def test_load_without_summary_loads_all(self, db):
        """When no summary rows exist, load() loads everything (existing behavior)."""
        from corvidae.conversation import MessageType
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
        """After two compactions, both summary rows accumulate in DB; load() uses only the latest.

        Append-only: old summary rows are NOT deleted. Each compaction INSERTs a
        new summary. load() selects the summary with the highest id (ORDER BY id DESC
        LIMIT 1) and uses its timestamp as the filter boundary.

        Token math:
          - "a" * 35 → int(35/3.5) = 10 tokens each.
          - max_tokens=50: retain_budget=25; threshold=40.
          - Walk: count=1(10), count=2(20), 30>25 AND count>0 → break. retain_count=2.

        Post-first-compaction DB (simulated):
          - 2 retained messages at base_ts+0, base_ts+1
          - first_summary at first_summary_ts = base_ts+0 - 1e-6 (boundary before retained[0])
          - 4 extra messages at extra_base_ts+0 .. extra_base_ts+3
          load() returns [first_summary, retained[0], retained[1], extra[0..3]] = 7 msgs.
          token_estimate = int((48 + 6*35)/3.5) = 73 ≥ 40 → triggers; len=7>5.

        Second compaction:
          - Walk: extra[3](10), extra[2](20), extra[1](30>25) → break. retain_count=2.
          - Inserts second_summary (append-only, first_summary stays).
          - load() uses second_summary's timestamp as boundary → returns [second_summary, extra[2], extra[3]] = 3 msgs.
        """
        base_ts = time.time()

        from corvidae.conversation import MessageType
        # 2 retained messages from first compaction
        retained_messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(2)
        ]
        for i, msg in enumerate(retained_messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )

        # First summary row: boundary = oldest_retained.ts - 1e-6 = base_ts - 1e-6
        first_summary_msg = {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nfirst summary",
        }
        first_summary_ts = base_ts - 1e-6
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'summary')",
            ("chan1", json.dumps(first_summary_msg), first_summary_ts),
        )

        # 4 extra messages added after first compaction (timestamps after retained)
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
        # (all have timestamp > first_summary_ts)
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

        from corvidae.compaction import CompactionPlugin
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

        # Append-only: both summary rows now exist in DB
        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 2, (
            f"Expected 2 summary rows in DB (append-only accumulation), got {row[0]}"
        )

        # Verify load() on a fresh ConversationLog uses only the latest summary
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()

        assert conv2.messages[0] == second_summary_msg_tagged, (
            f"Expected latest summary as first message, got: {conv2.messages[0]}"
        )
        assert len(conv2.messages) == 3, (
            f"Expected 3 messages (latest summary + 2 retained), got: {len(conv2.messages)}"
        )
        assert first_summary_msg_tagged not in conv2.messages


class TestTimestampBasedSummaryOrdering:
    """Tests verifying that load() uses timestamp-based filtering for append-only compaction.

    In the append-only design, the summary row's timestamp encodes the compaction
    boundary: summary_ts = oldest_retained.timestamp - 1e-6. load() uses
    WHERE timestamp > summary_ts to return the correct working set. Old rows
    (timestamp <= summary_ts) remain in the DB but are excluded by the filter.
    """

    async def test_load_uses_timestamp_for_summary_cutoff(self, db):
        """load() must use timestamp-based filtering (timestamp > summary_ts).

        Setup: summary with timestamp = shared_ts - 1e-6; two messages with
        timestamp = shared_ts (pass filter); one OLD message at shared_ts - 2e-6
        (below summary boundary, must be excluded).

        load() must return [summary, msg1, msg2] — 3 items, NOT 4.
        Current code (no timestamp filter) returns 4, making this test RED.
        After implementation (timestamp filter), old_msg is excluded → passes.
        """
        shared_ts = 1_000_000.0
        summary_ts = shared_ts - 1e-6

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nthe summary"}
        old_msg = {"role": "user", "content": "old message below summary boundary"}
        msg_after_1 = {"role": "user", "content": "after summary 1"}
        msg_after_2 = {"role": "assistant", "content": "after summary 2"}

        from corvidae.conversation import MessageType
        # Insert an old message BELOW the summary boundary — must be excluded by filter
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(old_msg), shared_ts - 2e-6),
        )
        # Insert the summary with boundary timestamp
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'summary')",
            ("chan1", json.dumps(summary_msg), summary_ts),
        )
        # Insert two messages with timestamp = shared_ts — these should pass the filter
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

        # Timestamp-based: old_msg (ts < summary_ts) excluded; both after-messages included.
        # Must return 3 (summary + 2 messages), NOT 4 (which current no-filter code returns).
        expected = [
            {**summary_msg, "_message_type": MessageType.SUMMARY},
            {**msg_after_1, "_message_type": MessageType.MESSAGE},
            {**msg_after_2, "_message_type": MessageType.MESSAGE},
        ]
        assert conv.messages == expected, (
            f"Expected [summary, msg_after_1, msg_after_2] (3 items, old_msg excluded), "
            f"got: {conv.messages}"
        )

    async def test_replace_with_summary_uses_timestamp_arithmetic(self, db):
        """replace_with_summary must store the summary at oldest_retained.timestamp - 1e-6.

        After compaction where all messages share the same timestamp (shared_ts),
        the summary row should be stored at shared_ts - 1e-6. This is the designed
        boundary mechanism for timestamp-based filtering.
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
        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        # Verify the summary row uses timestamp arithmetic: summary_ts = shared_ts - 1e-6
        async with db.execute(
            "SELECT timestamp FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            summary_row = await cursor.fetchone()

        assert summary_row is not None, "Summary row must exist after compaction"
        # With the append-only design: summary_ts = oldest_retained.ts - 1e-6 = shared_ts - 1e-6
        assert summary_row[0] == shared_ts - 1e-6, (
            f"Summary timestamp should be oldest_retained.ts - 1e-6 = {shared_ts - 1e-6}; "
            f"got {summary_row[0]}"
        )

    async def test_rapid_messages_with_same_timestamp(self, db):
        """When all 10 messages share shared_ts and summary is at shared_ts - 1e-6,
        load() returns ALL 10 messages + summary = 11 total.

        This is the correct behavior for append-only: erring toward inclusion is
        better than data loss. All 10 messages have timestamp = shared_ts >
        shared_ts - 1e-6 = summary_ts, so they all pass the filter. The extra
        8 messages are harmless redundant context that the LLM handles fine.
        """
        shared_ts = 1_000_000.0

        from corvidae.conversation import MessageType
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
        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        expected_summary_msg = {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nrapid summary",
            "_message_type": MessageType.SUMMARY,
        }
        # In-memory state reflects only compacted view: summary + 2 retained
        assert conv.messages[0] == expected_summary_msg
        assert len(conv.messages) == 3  # summary + 2 retained (in-memory)

        # After load(): timestamp filter passes ALL 10 original messages (shared_ts > summary_ts).
        # Correct behavior: 11 messages (summary + all 10). Extra messages are redundant context.
        conv2 = ConversationLog(db, channel_id="chan1")
        conv2.system_prompt = ""
        await conv2.load()

        assert conv2.messages[0] == expected_summary_msg, (
            f"First message after load() should be the summary, got: {conv2.messages[0]}"
        )
        assert len(conv2.messages) == 11, (
            f"Expected 11 messages (summary + all 10 original) when all share shared_ts, "
            f"got {len(conv2.messages)}. "
            f"Timestamp filter summary_ts={shared_ts - 1e-6} passes all messages with ts={shared_ts}."
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
        from corvidae.conversation import MessageType

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

        from corvidae.compaction import CompactionPlugin
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

        # Append-only: all 10 original rows + 1 summary = 11 total rows in DB
        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ?",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 11, (
            f"Expected 11 total rows (10 original + 1 summary, append-only), got {row[0]}"
        )

        # DB: context row must still exist (not deleted)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] >= 1, (
            "CONTEXT row was deleted from DB during compaction (should be append-only)"
        )

        # Verify load() returns only summary + 2 retained (timestamp filter excludes older rows)
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()
        assert len(conv2.messages) == 3, (
            f"Expected load() to return 3 messages (summary + 2 retained), got {len(conv2.messages)}"
        )
        from corvidae.conversation import MessageType
        assert conv2.messages[0]["_message_type"] == MessageType.SUMMARY

    async def test_compact_older_context_remains_in_db_excluded_by_load(self, db):
        """CONTEXT entries in the older (summarized) window remain in DB (append-only).

        Append-only: no rows are deleted. The CONTEXT row in the older window stays in DB
        but becomes invisible to load() because its timestamp is below the summary boundary.

        Setup: 7 MESSAGE + 1 CONTEXT (older window, base_ts+7) + 2 MESSAGE (retained, base_ts+8/9).
        retain_count=2 (last 2 message entries fit in token budget).
        Summary stored at oldest_retained.ts - 1e-6 = base_ts + 8 - 1e-6.
        load() uses WHERE timestamp > base_ts + 8 - 1e-6:
          - older 7 MESSAGE rows (base_ts+0..6): excluded
          - CONTEXT row (base_ts+7): excluded (7 < 8 - 1e-6 is FALSE for ts=base_ts+7
            since base_ts+7 < base_ts+8-1e-6) → excluded
          - 2 retained MESSAGE rows (base_ts+8, base_ts+9): included

        Token math: 9 messages x 35 chars + 1 x 35 chars = 10 x 35 = 350 chars;
        int(350/3.5)=100 ≥ 40 → triggers; len=10>5.
        retain_budget=25; each msg=10 tokens.
        Walk backward: last msg(10≤25, count=1), second-to-last(20≤25, count=2),
        third(30>25 AND count>0) → break. retain_count=2.

        We place the CONTEXT entry at position -3 (in the older window after retain_count=2).
        """
        from corvidae.conversation import MessageType

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
        older_ctx = {"role": "system", "content": "old context that should be excluded by filter"}
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
            + [{"role": "system", "content": "old context that should be excluded by filter",
                "_message_type": MessageType.CONTEXT}]
            + [{"role": "user", "content": "b" * 35, "_message_type": MessageType.MESSAGE}] * 2
        )

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "summary with older ctx excluded"}}]}
        )

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin()
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        # Append-only: CONTEXT row must still exist in DB (not deleted)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, (
            f"Expected CONTEXT row to remain in DB (append-only, no deletes), "
            f"got {row[0]} rows"
        )

        # Verify load() excludes the CONTEXT row via timestamp filter:
        # summary_ts = base_ts+8-1e-6; CONTEXT ts = base_ts+7 < summary_ts → excluded.
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()
        assert len(conv2.messages) == 3, (
            f"Expected load() to return 3 messages (summary + 2 retained, CONTEXT excluded "
            f"by timestamp filter), got {len(conv2.messages)}"
        )
        for msg in conv2.messages[1:]:
            assert msg.get("_message_type") != MessageType.CONTEXT, (
                f"CONTEXT row should be excluded by timestamp filter, but found in load(): {msg}"
            )


class TestMultipleCompactionAccumulation:
    """Tests verifying that multiple compactions accumulate summary rows (append-only)."""

    async def test_multiple_compactions_accumulate_summaries(self, db):
        """Running compaction twice accumulates 2 summary rows in DB; load() uses the latest.

        Append-only: each compaction INSERTs a new summary row. Old summaries stay.
        load() selects the summary with the highest id (ORDER BY id DESC LIMIT 1) and
        uses its timestamp as the filter boundary.

        Token math for each compaction: 10 messages x 35 chars → int(350/3.5)=100 ≥ 40 →
        triggers; len=10>5. retain_budget=25; each msg=10 tokens.
        Walk: count=1(10), count=2(20), 30>25 AND count>0 → break. retain_count=2.
        """
        from corvidae.conversation import MessageType
        from corvidae.compaction import CompactionPlugin

        base_ts = time.time()
        plugin = CompactionPlugin()

        # --- First compaction ---
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""

        # Insert 10 messages for first compaction
        first_batch = [{"role": "user", "content": "a" * 35} for _ in range(10)]
        for i, msg in enumerate(first_batch):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        await db.commit()
        conv.messages = list(first_batch)

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "first summary"}}]}
        )
        await plugin.compact_conversation(
            conversation=conv, client=mock_client, max_tokens=50
        )

        # After first compaction: 1 summary row in DB
        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, f"Expected 1 summary after first compaction, got {row[0]}"

        # Intermediate load check: fresh load after first compaction must return
        # summary + 2 retained = 3 messages (timestamp filter excludes the 8 older rows).
        conv_mid = ConversationLog(db, channel_id="chan1")
        conv_mid.system_prompt = ""
        await conv_mid.load()
        assert len(conv_mid.messages) == 3, (
            f"Expected 3 messages (summary + 2 retained) after first compaction load, "
            f"got {len(conv_mid.messages)}"
        )
        assert conv_mid.messages[0]["_message_type"] == MessageType.SUMMARY, (
            f"First message after first compaction load should be SUMMARY, "
            f"got: {conv_mid.messages[0]}"
        )

        # --- Insert more messages after first compaction ---
        second_base_ts = base_ts + 20
        second_batch = [{"role": "user", "content": "b" * 35} for _ in range(10)]
        for i, msg in enumerate(second_batch):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), second_base_ts + i),
            )
        await db.commit()

        # Load into fresh conv for second compaction
        conv2 = ConversationLog(db, channel_id="chan1")
        conv2.system_prompt = ""
        await conv2.load()

        # --- Second compaction ---
        mock_client2 = AsyncMock()
        mock_client2.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "second summary"}}]}
        )
        await plugin.compact_conversation(
            conversation=conv2, client=mock_client2, max_tokens=50
        )

        # Append-only: both summary rows accumulate in DB
        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 2, (
            f"Expected 2 summary rows in DB after two compactions (append-only), got {row[0]}"
        )

        # Verify load() uses only the latest summary
        conv3 = ConversationLog(db, channel_id="chan1")
        conv3.system_prompt = ""
        await conv3.load()

        assert conv3.messages[0]["_message_type"] == MessageType.SUMMARY, (
            f"First message should be SUMMARY, got: {conv3.messages[0]}"
        )
        second_summary_content = "[Summary of earlier conversation]\nsecond summary"
        assert conv3.messages[0]["content"] == second_summary_content, (
            f"Expected latest (second) summary, got: {conv3.messages[0]['content']}"
        )

        # load() returns latest summary + 2 retained (from second compaction)
        assert len(conv3.messages) == 3, (
            f"Expected 3 messages (latest summary + 2 retained), got {len(conv3.messages)}"
        )

        # First summary must not appear in the loaded messages
        first_summary_content = "[Summary of earlier conversation]\nfirst summary"
        for msg in conv3.messages:
            assert msg.get("content") != first_summary_content or msg.get("_message_type") != MessageType.SUMMARY, (
                f"Old (first) summary should not appear in load() results: {msg}"
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
        from corvidae.conversation import MessageType

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

        from corvidae.compaction import CompactionPlugin
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
