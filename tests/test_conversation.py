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

        await conv.compact_if_needed(mock_client, max_tokens=50)

        assert len(conv.messages) == 3  # summary msg + 2 retained
        assert conv.messages[0] == {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nmock summary",
        }
        assert conv.messages[1:] == expected_retained

    async def test_compact_if_needed_few_messages(self, db):
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

        await conv.compact_if_needed(mock_client, max_tokens=100)

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

        await conv.compact_if_needed(mock_client, max_tokens=1000)

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

        await conv.compact_if_needed(mock_client, max_tokens=100)

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

        # Must not raise TypeError
        await conv.compact_if_needed(mock_client, max_tokens=100)

        assert conv.messages[0] == {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nnone-content summary",
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

        # token_estimate (with None/non-string guard): list content → 0 chars → int(6*35/3.5) = int(210/3.5) = 60 ≥ 40 → triggers
        # len=7>5 → proceeds.
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "list-content summary"}}]}
        )

        await conv.compact_if_needed(mock_client, max_tokens=50)

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

        await conv.compact_if_needed(mock_client, max_tokens=200)

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

        await conv.compact_if_needed(mock_client, max_tokens=100)

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

        await conv.compact_if_needed(mock_client, max_tokens=100)

        assert len(conv.messages) == 4  # summary + 3 retained
        assert conv.messages[1] == msg_e
        assert conv.messages[2] == msg_f
        assert conv.messages[3] == msg_g


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
        """After compaction, exactly one is_summary=1 row exists; retained count matches token-walk."""
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

        await conv.compact_if_needed(mock_client, max_tokens=50)

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

        # Verify retained count: token-based walk retains 2 messages (not 20)
        assert len(conv.messages) == 3, (
            f"Expected 3 messages (1 summary + 2 retained), got {len(conv.messages)}"
        )

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
        """After two compactions, load() returns only the newest summary + its retained messages.

        Fixture design (token-based walk):
          - Message content: "a" * 35 → int(35/3.5) = 10 tokens each.
          - max_tokens=50: retain_budget = int(50*0.5) = 25; threshold = 50*0.8 = 40.
          - Walk retains messages until cumulative tokens exceed 25. With 10-token msgs:
            count=1(10≤25), count=2(20≤25), 30>25 AND count>0 → break. retain_count=2.

        First compaction state (pre-loaded into DB):
          - 6 original messages at timestamps base_ts+0..5.
          - token_estimate for 6 msgs = int(6*35/3.5) = 60 ≥ 40 → would trigger, len=6>5.
          - retain_count=2 → retained = original[4], original[5].
          - first_summary_ts = base_ts + 4 - 0.001 (just before original[4]).
          - DB contains: original[0..5] (is_summary=0) + first_summary row.

        Post-load state (before second compaction):
          - load() sees first_summary at base_ts+4-0.001, fetches non-summary rows
            with ts > base_ts+4-0.001 → original[4] and original[5].
          - Then 3 extra messages (new[0..2]) added at extra_base_ts+0..2.
          - conv.messages after load = [first_summary, orig[4], orig[5], new[0], new[1], new[2]] = 6 msgs.
          - token_estimate = int((48 + 5*35)/3.5) = int(223/3.5) = 63 ≥ 40 → triggers; len=6>5.
            (48 = len("[Summary of earlier conversation]\\nfirst summary"))

        Second compaction walk (on 6-message list):
          - Backward: new[2](10), new[1](20), new[0](30>25 AND count>0) → break.
          - retain_count=2; retained=[new[1], new[2]].
          - After second compaction: [second_summary, new[1], new[2]] = 3 msgs.

        Post second-compaction load (conv2):
          - load() finds second_summary, fetches non-summary rows after its timestamp.
          - conv2.messages = [second_summary, new[1], new[2]] = 3 msgs.
        """
        base_ts = time.time()

        # 6 original messages representing state before the first compaction
        original_messages = [
            {"role": "user", "content": "a" * 35}
            for i in range(6)
        ]
        for i, msg in enumerate(original_messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 0)",
                ("chan1", json.dumps(msg), base_ts + i),
            )

        # First summary row: timestamp just before original[4] (oldest retained after 1st compaction).
        # _persist_summary uses OFFSET (num_retained-1) = OFFSET 1 over non-summary rows
        # ordered by ts DESC → that gives original[4] at base_ts+4.
        # summary_ts = base_ts + 4 - 0.001.
        first_summary_msg = {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nfirst summary",
        }
        first_summary_ts = base_ts + 4 - 0.001
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 1)",
            ("chan1", json.dumps(first_summary_msg), first_summary_ts),
        )

        # 3 extra messages added after the first compaction
        extra_messages = [
            {"role": "user", "content": "a" * 35}
            for i in range(3)
        ]
        extra_base_ts = base_ts + 10
        for i, msg in enumerate(extra_messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, is_summary) VALUES (?, ?, ?, 0)",
                ("chan1", json.dumps(msg), extra_base_ts + i),
            )

        await db.commit()

        # Load to get the post-first-compaction in-memory state
        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        await conv.load()

        # load() returns: [first_summary] + original[4] + original[5] + extra[0..2] = 6 msgs
        assert len(conv.messages) == 6, (
            f"Expected 6 messages after load(), got {len(conv.messages)}"
        )
        assert conv.messages[0] == first_summary_msg

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "second summary"}}]}
        )

        # Capture the expected retained messages before second compaction mutates conv.messages.
        # Walk retains retain_count=2 (new[1] and new[2] = extra_messages[1] and extra_messages[2]).
        expected_retained = conv.messages[-2:]

        await conv.compact_if_needed(mock_client, max_tokens=50)

        second_summary_msg = {
            "role": "assistant",
            "content": "[Summary of earlier conversation]\nsecond summary",
        }
        # After second compaction: [second_summary, new[1], new[2]] = 3 messages
        assert conv.messages[0] == second_summary_msg
        assert conv.messages[1:] == expected_retained
        assert len(conv.messages) == 3, (
            f"Expected 3 messages (1 summary + 2 retained), got {len(conv.messages)}"
        )

        # verify load() on a fresh ConversationLog returns only the newest summary + its retained msgs
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()

        assert conv2.messages[0] == second_summary_msg, (
            f"First message after load() should be second summary, got: {conv2.messages[0]}"
        )
        assert len(conv2.messages) == 3, (
            f"Expected 3 messages (1 summary + 2 retained), got: {len(conv2.messages)}"
        )
        # The first summary must NOT appear in the loaded messages
        assert first_summary_msg not in conv2.messages, (
            "First (old) summary must not appear after loading with second summary present"
        )
