"""Tests for corvidae.conversation.ConversationLog and init_db."""

import json
import time
from unittest.mock import AsyncMock, patch

import aiosqlite

from corvidae.conversation import ConversationLog, init_db


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

        from corvidae.conversation import MessageType
        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        expected = [{**msg, "_message_type": MessageType.MESSAGE} for msg in messages]
        assert conv.messages == expected

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

        from corvidae.conversation import MessageType
        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert conv.messages == [
            {**msg_early, "_message_type": MessageType.MESSAGE},
            {**msg_late, "_message_type": MessageType.MESSAGE},
        ]

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

        from corvidae.conversation import MessageType
        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert conv.messages == [{**msg_chan1, "_message_type": MessageType.MESSAGE}]


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


class TestReplaceWithSummary:
    """Tests for ConversationLog.replace_with_summary(summary_msg, retain_count).

    These tests fail until replace_with_summary() is added to ConversationLog.
    """

    async def test_basic_replace_with_summary(self, db):
        """10 messages, replace_with_summary with retain_count=2.

        Verify messages = [summary+tag, retained[-2], retained[-1]].
        """
        from corvidae.conversation import MessageType

        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": f"msg {i}", "_message_type": MessageType.MESSAGE}
            for i in range(10)
        ]
        expected_retained = conv.messages[-2:]

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\ntest"}
        await conv.replace_with_summary(summary_msg, retain_count=2)

        assert len(conv.messages) == 3
        assert conv.messages[0] == {**summary_msg, "_message_type": MessageType.SUMMARY}
        assert conv.messages[1:] == expected_retained

    async def test_replace_with_summary_zero_retain(self, db):
        """retain_count=0 — messages = [summary+tag] only."""
        from corvidae.conversation import MessageType

        conv = ConversationLog(db, channel_id="chan1")
        conv.messages = [
            {"role": "user", "content": f"msg {i}", "_message_type": MessageType.MESSAGE}
            for i in range(5)
        ]

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nzero retain"}
        await conv.replace_with_summary(summary_msg, retain_count=0)

        assert len(conv.messages) == 1
        assert conv.messages[0] == {**summary_msg, "_message_type": MessageType.SUMMARY}

    async def test_replace_with_summary_raises_on_too_many_retained(self, db):
        """retain_count > len(messages) raises ValueError."""
        import pytest
        from corvidae.conversation import MessageType

        conv = ConversationLog(db, channel_id="chan1")
        conv.messages = [
            {"role": "user", "content": "msg", "_message_type": MessageType.MESSAGE}
            for _ in range(3)
        ]

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nerr"}
        with pytest.raises(ValueError):
            await conv.replace_with_summary(summary_msg, retain_count=10)

    async def test_replace_with_summary_db_persistence(self, db):
        """After replace_with_summary, DB retains all rows (append-only).

        All 5 original message rows remain in the DB — no rows are deleted.
        A new summary row is inserted. load() on a fresh ConversationLog returns
        only the summary + 2 retained messages (the working set).
        """
        from corvidae.conversation import MessageType

        base_ts = time.time()
        conv = ConversationLog(db, channel_id="chan1")

        # Insert 5 rows into DB
        for i in range(5):
            msg = {"role": "user", "content": f"msg {i}"}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        await db.commit()

        # Set in-memory state to match DB
        conv.messages = [
            {"role": "user", "content": f"msg {i}", "_message_type": MessageType.MESSAGE}
            for i in range(5)
        ]

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\npersist test"}
        await conv.replace_with_summary(summary_msg, retain_count=2)

        # DB: exactly 1 summary row
        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, f"Expected 1 summary row, got {row[0]}"

        # DB: all 5 original message rows still exist (append-only — no deletion)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND message_type = 'message'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 5, f"Expected 5 message rows (append-only, no deletion), got {row[0]}"

        # In-memory: 3 total (summary + 2 retained)
        assert len(conv.messages) == 3

        # load() on a fresh instance returns working set: summary + 2 retained messages
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()
        assert len(conv2.messages) == 3, (
            f"Expected load() to return 3 messages (summary + 2 retained), got {len(conv2.messages)}"
        )
        assert conv2.messages[0]["_message_type"] == MessageType.SUMMARY
        assert all(
            m["_message_type"] == MessageType.MESSAGE for m in conv2.messages[1:]
        ), "Expected retained messages to be MESSAGE type"


class TestMessageType:
    """Tests for MessageType enum, in-memory tagging, and message_type DB column.

    All tests import MessageType from corvidae.conversation. This import will
    fail (ImportError) until Phase 1 is implemented — that is the intended
    red state.
    """

    async def test_message_type_enum_values(self):
        """MessageType enum must expose MESSAGE='message' and SUMMARY='summary',
        and round-trip correctly from a DB string via MessageType(value)."""
        from corvidae.conversation import MessageType  # fails until Phase 1

        assert MessageType.MESSAGE == "message", (
            f"Expected MessageType.MESSAGE == 'message', got {MessageType.MESSAGE!r}"
        )
        assert MessageType.SUMMARY == "summary", (
            f"Expected MessageType.SUMMARY == 'summary', got {MessageType.SUMMARY!r}"
        )
        # Round-trip: constructing from the string value must give the enum member
        assert MessageType("message") is MessageType.MESSAGE
        assert MessageType("summary") is MessageType.SUMMARY

    async def test_persist_with_explicit_message_type(self, db):
        """_persist(msg, MessageType.SUMMARY) must write message_type='summary' to DB."""
        from corvidae.conversation import MessageType  # fails until Phase 1

        conv = ConversationLog(db, channel_id="chan1")
        msg = {"role": "assistant", "content": "a summary"}

        await conv._persist(msg, MessageType.SUMMARY)

        async with db.execute(
            "SELECT message_type FROM message_log WHERE channel_id = ?",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None, "Row should exist after _persist"
        assert row[0] == "summary", (
            f"Expected message_type='summary', got {row[0]!r}"
        )

    async def test_persist_default_message_type(self, db):
        """_persist(msg) with no message_type arg must write message_type='message' to DB."""
        from corvidae.conversation import MessageType  # fails until Phase 1

        conv = ConversationLog(db, channel_id="chan1")
        msg = {"role": "user", "content": "hello"}

        await conv._persist(msg)

        async with db.execute(
            "SELECT message_type FROM message_log WHERE channel_id = ?",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None, "Row should exist after _persist"
        assert row[0] == "message", (
            f"Expected message_type='message', got {row[0]!r}"
        )

    async def test_load_tags_messages_with_message_type(self, db):
        """load() must tag every in-memory message dict with a '_message_type' key."""
        from corvidae.conversation import MessageType  # fails until Phase 1

        base_ts = time.time()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        for i, msg in enumerate(messages):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert len(conv.messages) == 2, (
            f"Expected 2 messages, got {len(conv.messages)}"
        )
        for i, loaded_msg in enumerate(conv.messages):
            assert "_message_type" in loaded_msg, (
                f"Message at index {i} is missing '_message_type' key: {loaded_msg!r}"
            )
            assert loaded_msg["_message_type"] == MessageType.MESSAGE, (
                f"Expected MessageType.MESSAGE at index {i}, got {loaded_msg['_message_type']!r}"
            )

    async def test_build_prompt_strips_message_type(self, db):
        """build_prompt() must return dicts with no '_message_type' key,
        even when self.messages contains tagged dicts."""
        from corvidae.conversation import MessageType  # fails until Phase 1

        conv = ConversationLog(db, channel_id="chan1")
        conv.system_prompt = "You are helpful."
        conv.messages = [
            {"role": "user", "content": "hi", "_message_type": MessageType.MESSAGE},
            {"role": "assistant", "content": "hello", "_message_type": MessageType.MESSAGE},
        ]

        prompt = conv.build_prompt()

        # The system message is index 0; message dicts start at index 1
        for i, msg in enumerate(prompt):
            assert "_message_type" not in msg, (
                f"Prompt message at index {i} still contains '_message_type': {msg!r}"
            )

    async def test_append_tags_in_memory_message(self, db):
        """append(msg) must tag the in-memory copy with _message_type=MessageType.MESSAGE."""
        from corvidae.conversation import MessageType  # fails until Phase 1

        conv = ConversationLog(db, channel_id="chan1")
        msg = {"role": "user", "content": "hello"}

        await conv.append(msg)

        assert len(conv.messages) == 1, (
            f"Expected 1 message in conv.messages, got {len(conv.messages)}"
        )
        last = conv.messages[-1]
        assert "_message_type" in last, (
            f"In-memory message after append() is missing '_message_type': {last!r}"
        )
        assert last["_message_type"] == MessageType.MESSAGE, (
            f"Expected MessageType.MESSAGE, got {last['_message_type']!r}"
        )
        # Original dict must not be mutated
        assert "_message_type" not in msg, (
            "append() must not mutate the original message dict"
        )


# ---------------------------------------------------------------------------
# Phase 2: CONTEXT MessageType, remove_by_type, load/compaction changes
# ---------------------------------------------------------------------------


class TestMessageTypeContext:
    """Tests for the CONTEXT MessageType variant added in Phase 2.

    Tests here will fail until CONTEXT is added to the MessageType enum.
    """

    async def test_message_type_context_value(self):
        """MessageType.CONTEXT must equal the string 'context'."""
        from corvidae.conversation import MessageType

        assert MessageType.CONTEXT == "context", (
            f"Expected MessageType.CONTEXT == 'context', got {MessageType.CONTEXT!r}"
        )

    async def test_append_with_context_type(self, db):
        """append() with message_type=MessageType.CONTEXT must persist a row
        with message_type='context' in the DB."""
        from corvidae.conversation import MessageType

        conv = ConversationLog(db, channel_id="chan1")
        msg = {"role": "system", "content": "remembered: user prefers terse replies"}

        await conv.append(msg, message_type=MessageType.CONTEXT)

        async with db.execute(
            "SELECT message_type FROM message_log WHERE channel_id = ?",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None, "Row should exist after append()"
        assert row[0] == "context", (
            f"Expected message_type='context', got {row[0]!r}"
        )

    async def test_load_includes_context_entries(self, db):
        """Regression guard: load() returns both MESSAGE and CONTEXT rows when no
        summary exists. Verifies the WHERE clause uses message_type != 'summary'
        so CONTEXT entries are not silently dropped.
        """
        from corvidae.conversation import MessageType

        base_ts = time.time()
        msg_msg = {"role": "user", "content": "hello"}
        msg_ctx = {"role": "system", "content": "context: user is in Paris"}

        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(msg_msg), base_ts),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'context')",
            ("chan1", json.dumps(msg_ctx), base_ts + 1),
        )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        assert len(conv.messages) == 2, (
            f"Expected 2 messages (MESSAGE + CONTEXT), got {len(conv.messages)}: {conv.messages}"
        )
        types_loaded = {m["_message_type"] for m in conv.messages}
        assert MessageType.MESSAGE in types_loaded, "MESSAGE entry not loaded"
        assert MessageType.CONTEXT in types_loaded, (
            "CONTEXT entry not loaded — load() is filtering it out"
        )

    async def test_load_with_summary_includes_context(self, db):
        """Regression guard: when a summary row exists, load() returns the summary
        plus all MESSAGE and CONTEXT rows that follow it (not just MESSAGE rows).
        Verifies the summary-branch WHERE clause uses message_type != 'summary'.
        """
        from corvidae.conversation import MessageType

        base_ts = time.time()
        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nold stuff"}
        msg_message = {"role": "user", "content": "retained message"}
        msg_context = {"role": "system", "content": "context: user timezone is UTC+1"}

        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'summary')",
            ("chan1", json.dumps(summary_msg), base_ts),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            ("chan1", json.dumps(msg_message), base_ts + 1),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'context')",
            ("chan1", json.dumps(msg_context), base_ts + 2),
        )
        await db.commit()

        conv = ConversationLog(db, channel_id="chan1")
        await conv.load()

        # Must have: summary + message + context
        assert len(conv.messages) == 3, (
            f"Expected 3 messages (SUMMARY + MESSAGE + CONTEXT), got {len(conv.messages)}: {conv.messages}"
        )
        assert conv.messages[0]["_message_type"] == MessageType.SUMMARY
        types_after_summary = {m["_message_type"] for m in conv.messages[1:]}
        assert MessageType.MESSAGE in types_after_summary, "MESSAGE entry not loaded"
        assert MessageType.CONTEXT in types_after_summary, (
            "CONTEXT entry not loaded — summary-branch WHERE filters it out"
        )


class TestPersistSummaryWithContext:
    """Tests for replace_with_summary edge cases introduced by CONTEXT entries.

    These test the unified non-summary boundary fix in replace_with_summary.
    Tests call conv.replace_with_summary(summary_msg, retain_count) directly.
    """

    async def test_replace_with_summary_correct_offset_with_context(self, db):
        """When the retained set contains MESSAGE + CONTEXT entries interspersed,
        replace_with_summary must use a timestamp boundary so that no DB rows are
        deleted (append-only). load() returns only the correct working set.

        Setup: older window = [MSG1, MSG2, MSG3, MSG4, MSG5]; retained window =
        [CTX6, MSG7, MSG8] (3 non-summary entries). All 7 MESSAGE rows and 1 CONTEXT
        row remain in the DB. load() returns summary + CTX6 + MSG7 + MSG8.
        """
        from corvidae.conversation import MessageType

        base_ts = time.time()

        # Insert 5 MESSAGE rows (will be in older window after compaction)
        older_msgs = []
        for i in range(5):
            msg = {"role": "user", "content": f"older {i}"}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )
            older_msgs.append(msg)

        # Insert CONTEXT row (will be oldest retained entry)
        ctx_msg = {"role": "system", "content": "retained context"}
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'context')",
            ("chan1", json.dumps(ctx_msg), base_ts + 5),
        )
        # Insert 2 more MESSAGE rows (also retained)
        retained_msgs = []
        for i in range(2):
            msg = {"role": "user", "content": f"retained {i}"}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + 6 + i),
            )
            retained_msgs.append(msg)
        await db.commit()

        # Build summary message
        summary_content = "[Summary of earlier conversation]\ntest summary"
        summary_msg = {"role": "assistant", "content": summary_content}

        # Set conv.messages to PRE-compaction state: all 8 messages
        conv = ConversationLog(db, channel_id="chan1")
        conv.messages = (
            [{**m, "_message_type": MessageType.MESSAGE} for m in older_msgs]
            + [{**ctx_msg, "_message_type": MessageType.CONTEXT}]
            + [{**m, "_message_type": MessageType.MESSAGE} for m in retained_msgs]
        )

        # Call replace_with_summary with retain_count=3 (ctx + 2 retained msgs)
        await conv.replace_with_summary(summary_msg, retain_count=3)

        # All 7 MESSAGE rows remain in DB (append-only — no deletion)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'message'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 7, (
            f"Expected 7 MESSAGE rows (append-only, no deletion), got {row[0]}"
        )

        # The CONTEXT row still exists (append-only)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, (
            f"Expected 1 CONTEXT row in DB (append-only), got {row[0]}"
        )

        # load() on a fresh instance returns working set: summary + 3 retained entries
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()
        assert len(conv2.messages) == 4, (
            f"Expected load() to return 4 messages (summary + 3 retained), got {len(conv2.messages)}"
        )
        assert conv2.messages[0]["_message_type"] == MessageType.SUMMARY
        retained_types = {m["_message_type"] for m in conv2.messages[1:]}
        assert MessageType.CONTEXT in retained_types, "CONTEXT entry missing from load() working set"
        assert MessageType.MESSAGE in retained_types, "MESSAGE entry missing from load() working set"

    async def test_replace_with_summary_context_only_retained(self, db):
        """When retained set contains only CONTEXT entries (no MESSAGE),
        all rows remain in the DB (append-only). load() returns summary + 1 CONTEXT.

        Setup: older window = [MSG1..MSG5]; retained = [CTX6] (1 non-summary, 0 MESSAGE).
        All 5 MESSAGE rows and 1 CONTEXT row remain in DB after compaction.
        """
        from corvidae.conversation import MessageType

        base_ts = time.time()

        # Insert 5 MESSAGE rows (older window)
        for i in range(5):
            msg = {"role": "user", "content": f"older {i}"}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        # Insert 1 CONTEXT row (the only retained entry)
        ctx_msg = {"role": "system", "content": "sole retained context"}
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'context')",
            ("chan1", json.dumps(ctx_msg), base_ts + 5),
        )
        await db.commit()

        summary_content = "[Summary of earlier conversation]\nzero-message summary"
        summary_msg = {"role": "assistant", "content": summary_content}

        # PRE-compaction state: 5 older messages + 1 context
        conv = ConversationLog(db, channel_id="chan1")
        conv.messages = (
            [
                {"role": "user", "content": f"older {i}", "_message_type": MessageType.MESSAGE}
                for i in range(5)
            ]
            + [{**ctx_msg, "_message_type": MessageType.CONTEXT}]
        )

        # retain_count=1 (only the CONTEXT entry)
        await conv.replace_with_summary(summary_msg, retain_count=1)

        # All 5 MESSAGE rows still exist (append-only — no deletion)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'message'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 5, (
            f"Expected 5 MESSAGE rows (append-only, no deletion), got {row[0]}"
        )

        # CONTEXT row still exists (append-only)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, (
            f"Expected 1 CONTEXT row in DB (append-only), got {row[0]}"
        )

        # load() on a fresh instance returns summary + 1 CONTEXT (the working set)
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()
        assert len(conv2.messages) == 2, (
            f"Expected load() to return 2 messages (summary + 1 CONTEXT), got {len(conv2.messages)}"
        )
        assert conv2.messages[0]["_message_type"] == MessageType.SUMMARY
        assert conv2.messages[1]["_message_type"] == MessageType.CONTEXT

    async def test_replace_with_summary_zero_retained(self, db):
        """When retain_count == 0 (retained set is completely empty),
        all rows remain in the DB (append-only). load() returns only the summary.

        The summary timestamp is set to time.time() so no existing rows pass the
        timestamp > summary_ts filter until new messages arrive.
        """
        from corvidae.conversation import MessageType

        # Use timestamps well in the past so time.time() in replace_with_summary
        # (used as summary_ts when retain_count=0) is greater than all of them.
        base_ts = time.time() - 100

        # Insert 5 MESSAGE rows and 2 CONTEXT rows — all remain in DB (append-only)
        for i in range(5):
            msg = {"role": "user", "content": f"older msg {i}"}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        for i in range(2):
            ctx = {"role": "system", "content": f"older ctx {i}"}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'context')",
                ("chan1", json.dumps(ctx), base_ts + 5 + i),
            )
        await db.commit()

        summary_content = "[Summary of earlier conversation]\nzero-retained summary"
        summary_msg = {"role": "assistant", "content": summary_content}

        # PRE-compaction state: all 7 messages
        conv = ConversationLog(db, channel_id="chan1")
        conv.messages = (
            [
                {"role": "user", "content": f"older msg {i}", "_message_type": MessageType.MESSAGE}
                for i in range(5)
            ]
            + [
                {"role": "system", "content": f"older ctx {i}", "_message_type": MessageType.CONTEXT}
                for i in range(2)
            ]
        )

        # Must not raise any SQLite error
        await conv.replace_with_summary(summary_msg, retain_count=0)

        # All 5 MESSAGE rows still exist in DB (append-only — no deletion)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'message'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 5, (
            f"Expected 5 MESSAGE rows (append-only, no deletion), got {row[0]}"
        )

        # Both CONTEXT rows still exist in DB (append-only)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 2, (
            f"Expected 2 CONTEXT rows (append-only, no deletion), got {row[0]}"
        )

        # load() on a fresh instance returns only the summary (nothing passes
        # the timestamp filter because summary_ts = time.time() > all old rows)
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()
        assert len(conv2.messages) == 1, (
            f"Expected load() to return only 1 message (summary), got {len(conv2.messages)}"
        )
        assert conv2.messages[0]["_message_type"] == MessageType.SUMMARY

    async def test_replace_with_summary_correct_offset_four_args(self, db):
        """Verify replace_with_summary with a non-trivial retained set including
        mixed types. All rows remain in the DB (append-only); load() returns the
        correct working set: summary + 2 retained CONTEXT entries.

        Setup: older window = [MSG1..MSG4]; retained = [CTX5, CTX6].
        All 4 MESSAGE rows and 2 CONTEXT rows remain in DB after compaction.
        """
        from corvidae.conversation import MessageType

        base_ts = time.time()

        # Insert 4 MESSAGE rows (older window)
        for i in range(4):
            msg = {"role": "user", "content": f"older {i}"}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                ("chan1", json.dumps(msg), base_ts + i),
            )
        # Insert 2 CONTEXT rows (retained)
        ctx_msgs = []
        for i in range(2):
            ctx = {"role": "system", "content": f"retained ctx {i}"}
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'context')",
                ("chan1", json.dumps(ctx), base_ts + 4 + i),
            )
            ctx_msgs.append(ctx)
        await db.commit()

        summary_content = "[Summary of earlier conversation]\nmixed retained"
        summary_msg = {"role": "assistant", "content": summary_content}

        # PRE-compaction state: 4 older + 2 context retained
        conv = ConversationLog(db, channel_id="chan1")
        conv.messages = (
            [
                {"role": "user", "content": f"older {i}", "_message_type": MessageType.MESSAGE}
                for i in range(4)
            ]
            + [
                {**c, "_message_type": MessageType.CONTEXT}
                for c in ctx_msgs
            ]
        )

        # retain_count=2 (the 2 CONTEXT entries)
        await conv.replace_with_summary(summary_msg, retain_count=2)

        # All 4 MESSAGE rows still exist (append-only — no deletion)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'message'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 4, (
            f"Expected 4 MESSAGE rows (append-only, no deletion), got {row[0]}"
        )

        # Both CONTEXT rows still exist (append-only)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 2, (
            f"Expected 2 CONTEXT rows in DB (append-only), got {row[0]}"
        )

        # Exactly 1 summary row
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'summary'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, (
            f"Expected 1 summary row after compaction, got {row[0]}"
        )

        # load() on a fresh instance returns working set: summary + 2 retained CONTEXT entries
        conv2 = ConversationLog(db, channel_id="chan1")
        await conv2.load()
        assert len(conv2.messages) == 3, (
            f"Expected load() to return 3 messages (summary + 2 CONTEXT), got {len(conv2.messages)}"
        )
        assert conv2.messages[0]["_message_type"] == MessageType.SUMMARY
        assert all(
            m["_message_type"] == MessageType.CONTEXT for m in conv2.messages[1:]
        ), "Expected retained entries to be CONTEXT type"


class TestRemoveByType:
    """Tests for the new ConversationLog.remove_by_type() method (Phase 2).

    All tests here fail until remove_by_type() is added to ConversationLog.
    """

    async def test_remove_by_type_removes_context(self, db):
        """remove_by_type(CONTEXT) removes CONTEXT entries from in-memory state only.

        The method no longer touches the DB (append-only). CONTEXT rows remain
        in the DB after remove_by_type(); only the in-memory working set is updated.
        """
        from corvidae.conversation import MessageType

        conv = ConversationLog(db, channel_id="chan1")

        # Append a MESSAGE and two CONTEXT entries
        msg = {"role": "user", "content": "hello"}
        ctx1 = {"role": "system", "content": "context: user is in Paris"}
        ctx2 = {"role": "system", "content": "context: user prefers terse replies"}

        await conv.append(msg, message_type=MessageType.MESSAGE)
        await conv.append(ctx1, message_type=MessageType.CONTEXT)
        await conv.append(ctx2, message_type=MessageType.CONTEXT)

        assert len(conv.messages) == 3

        removed = await conv.remove_by_type(MessageType.CONTEXT)

        # Returns count of removed entries
        assert removed == 2, f"Expected 2 removed, got {removed}"

        # In-memory: only the MESSAGE entry remains
        assert len(conv.messages) == 1, (
            f"Expected 1 message remaining in memory, got {len(conv.messages)}"
        )
        assert conv.messages[0]["_message_type"] == MessageType.MESSAGE

        # DB: both CONTEXT rows still exist (remove_by_type is in-memory only)
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'context'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 2, (
            f"Expected 2 CONTEXT rows still in DB (remove_by_type is in-memory only), got {row[0]}"
        )

        # DB: MESSAGE row still present
        async with db.execute(
            "SELECT COUNT(*) FROM message_log "
            "WHERE channel_id = ? AND message_type = 'message'",
            ("chan1",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, (
            f"Expected 1 MESSAGE row in DB, got {row[0]}"
        )

    async def test_remove_by_type_rejects_message(self, db):
        """remove_by_type(MESSAGE) must raise ValueError — MESSAGE lifecycle
        is managed by compaction, not by remove_by_type."""
        from corvidae.conversation import MessageType
        import pytest

        conv = ConversationLog(db, channel_id="chan1")

        with pytest.raises(ValueError):
            await conv.remove_by_type(MessageType.MESSAGE)

    async def test_remove_by_type_rejects_summary(self, db):
        """remove_by_type(SUMMARY) must raise ValueError — SUMMARY lifecycle
        is managed by compaction, not by remove_by_type."""
        from corvidae.conversation import MessageType
        import pytest

        conv = ConversationLog(db, channel_id="chan1")

        with pytest.raises(ValueError):
            await conv.remove_by_type(MessageType.SUMMARY)


# ---------------------------------------------------------------------------
# Simplification tests (Red TDD phase)
# ---------------------------------------------------------------------------


class TestParseMessageRows:
    """Tests for the _parse_message_rows module-level helper (Item 3).

    These tests fail until _parse_message_rows is extracted from ConversationLog.load().
    """

    def test_parse_message_rows_import(self):
        """_parse_message_rows must be importable from corvidae.conversation."""
        from corvidae.conversation import _parse_message_rows  # noqa: F401

    def test_parse_message_rows_basic(self):
        """_parse_message_rows correctly parses (json, message_type) rows and tags _message_type."""
        import json
        from corvidae.conversation import MessageType, _parse_message_rows

        msg1 = {"role": "user", "content": "hello"}
        msg2 = {"role": "assistant", "content": "hi"}
        rows = [
            (json.dumps(msg1), "message"),
            (json.dumps(msg2), "message"),
        ]

        result = _parse_message_rows(rows)

        assert len(result) == 2
        assert result[0] == {**msg1, "_message_type": MessageType.MESSAGE}
        assert result[1] == {**msg2, "_message_type": MessageType.MESSAGE}

    def test_parse_message_rows_mixed_types(self):
        """_parse_message_rows handles rows with different message_type values."""
        import json
        from corvidae.conversation import MessageType, _parse_message_rows

        msg_msg = {"role": "user", "content": "a message"}
        msg_ctx = {"role": "system", "content": "some context"}
        rows = [
            (json.dumps(msg_msg), "message"),
            (json.dumps(msg_ctx), "context"),
        ]

        result = _parse_message_rows(rows)

        assert len(result) == 2
        assert result[0]["_message_type"] == MessageType.MESSAGE
        assert result[1]["_message_type"] == MessageType.CONTEXT

    def test_parse_message_rows_empty(self):
        """_parse_message_rows returns an empty list for empty input."""
        from corvidae.conversation import _parse_message_rows

        result = _parse_message_rows([])

        assert result == []


class TestDefaultCharsPerToken:
    """Tests for the DEFAULT_CHARS_PER_TOKEN module-level constant (Item 6).

    These tests fail until DEFAULT_CHARS_PER_TOKEN is added to corvidae.conversation.
    """

    def test_default_chars_per_token_import(self):
        """DEFAULT_CHARS_PER_TOKEN must be importable from corvidae.conversation."""
        from corvidae.conversation import DEFAULT_CHARS_PER_TOKEN  # noqa: F401

    def test_default_chars_per_token_value(self):
        """DEFAULT_CHARS_PER_TOKEN must equal 3.5."""
        from corvidae.conversation import DEFAULT_CHARS_PER_TOKEN

        assert DEFAULT_CHARS_PER_TOKEN == 3.5, (
            f"Expected DEFAULT_CHARS_PER_TOKEN == 3.5, got {DEFAULT_CHARS_PER_TOKEN!r}"
        )
