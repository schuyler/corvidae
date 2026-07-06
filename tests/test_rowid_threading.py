"""Tests for rowid threading (Phase 1a WP1a.2, bootstrap-mapping §4.8).

Window messages carry their message_log rowid as an internal ``_db_id`` tag:
- PersistencePlugin.on_conversation_event returns the inserted rowid.
- The agent resolves the broadcast results and attaches the rowid to the
  in-window copy of the message.
- Every serialization boundary strips all ``_``-prefixed keys.
- load_conversation re-attaches ids on reload.
- Compaction fires on_compaction with the ids of exactly the compacted
  messages.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.channel import Channel
from corvidae.compaction import CompactionPlugin
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import resolve_single_result
from corvidae.persistence import PersistencePlugin, init_db

from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response


@pytest.fixture
async def plugin_and_channel():
    """(plugin, channel, db) with TaskPlugin teardown."""
    plugin, channel, db = await build_plugin_and_channel()
    yield plugin, channel, db
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


class TestRowidAttachment:
    async def test_user_and_assistant_messages_carry_db_ids(self, plugin_and_channel):
        """After a user turn, both window messages carry ints matching message_log rows."""
        plugin, channel, db = plugin_and_channel
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hi there"))
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        conv = channel.conversation
        user_msg, assistant_msg = conv.messages[-2], conv.messages[-1]
        assert isinstance(user_msg["_db_id"], int)
        assert isinstance(assistant_msg["_db_id"], int)

        # The attached ids point at the rows that hold these messages.
        for tagged in (user_msg, assistant_msg):
            async with db.execute(
                "SELECT message FROM message_log WHERE id = ?", (tagged["_db_id"],)
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert json.loads(row[0])["content"] == tagged["content"]

    async def test_persistence_returns_rowid(self, db):
        """on_conversation_event returns the inserted message_log rowid."""
        persistence = PersistencePlugin()
        persistence.db = db
        channel = Channel(transport="test", scope="rowid")
        rowid = await persistence.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "x"},
            message_type=MessageType.MESSAGE,
        )
        assert isinstance(rowid, int)
        async with db.execute(
            "SELECT message FROM message_log WHERE id = ?", (rowid,)
        ) as cursor:
            row = await cursor.fetchone()
        assert json.loads(row[0])["content"] == "x"


class TestUnderscoreStrips:
    def test_build_prompt_strips_all_underscore_keys(self):
        """build_prompt output contains no _-prefixed keys."""
        conv = ContextWindow("test:strip")
        conv.system_prompt = "sys"
        conv.append({"role": "user", "content": "hello"})
        conv.messages[-1]["_db_id"] = 42
        conv.messages[-1]["_anything_else"] = "internal"

        prompt = conv.build_prompt()
        for msg in prompt:
            assert not any(k.startswith("_") for k in msg)
        # The real keys survive.
        assert prompt[-1] == {"role": "user", "content": "hello"}

    async def test_persisted_json_contains_no_underscore_keys(self, db):
        """Persistence write paths strip every _-prefixed key."""
        persistence = PersistencePlugin()
        persistence.db = db
        channel = Channel(transport="test", scope="strip")

        message = {
            "role": "user",
            "content": "hello",
            "_message_type": MessageType.MESSAGE,
            "_db_id": 7,
        }
        rowid = await persistence.on_conversation_event(
            channel=channel, message=message, message_type=MessageType.MESSAGE
        )
        async with db.execute(
            "SELECT message FROM message_log WHERE id = ?", (rowid,)
        ) as cursor:
            row = await cursor.fetchone()
        assert not any(k.startswith("_") for k in json.loads(row[0]))

        summary = {
            "role": "assistant",
            "content": "summary text",
            "_message_type": MessageType.SUMMARY,
            "_db_id": 9,
        }
        await persistence.on_compaction(
            channel=channel, summary_msg=summary, retain_count=0, compacted_ids=[1]
        )
        async with db.execute(
            "SELECT message FROM message_log WHERE channel_id = ? "
            "AND message_type = 'summary'",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert not any(k.startswith("_") for k in json.loads(row[0]))


class TestReloadReattachesIds:
    async def test_load_conversation_returns_db_ids(self, db):
        """load_conversation re-attaches the id column as _db_id."""
        persistence = PersistencePlugin()
        persistence.db = db
        channel = Channel(transport="test", scope="reload")

        rowids = []
        for content in ("one", "two"):
            rowids.append(
                await persistence.on_conversation_event(
                    channel=channel,
                    message={"role": "user", "content": content},
                    message_type=MessageType.MESSAGE,
                )
            )

        loaded = await persistence.load_conversation(channel=channel)
        assert [m["_db_id"] for m in loaded] == rowids

    async def test_load_conversation_after_summary_carries_ids(self, db):
        """The post-summary reload path also re-attaches ids."""
        persistence = PersistencePlugin()
        persistence.db = db
        channel = Channel(transport="test", scope="reload2")

        await persistence.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "old"},
            message_type=MessageType.MESSAGE,
        )
        await persistence.on_compaction(
            channel=channel,
            summary_msg={"role": "assistant", "content": "[Summary] old stuff"},
            retain_count=1,
            compacted_ids=[],
        )
        recent_id = await persistence.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "recent"},
            message_type=MessageType.MESSAGE,
        )

        loaded = await persistence.load_conversation(channel=channel)
        # Summary first, then the retained messages, all carrying _db_id.
        assert all(isinstance(m.get("_db_id"), int) for m in loaded)
        assert loaded[-1]["_db_id"] == recent_id


class TestCompactionCarriesCompactedIds:
    async def test_on_compaction_receives_compacted_ids(self):
        """Compaction fires on_compaction with the ids of exactly the removed messages."""
        plugin = CompactionPlugin()
        plugin.pm = MagicMock()
        plugin.pm.ahook.on_compaction = AsyncMock()
        plugin._summarize = AsyncMock(return_value="a summary")
        plugin._min_messages = 2
        plugin._compaction_threshold = 0.0  # always trigger
        plugin._compaction_retention = 0.0  # retain minimum

        conv = ContextWindow("test:compact")
        channel = Channel(transport="test", scope="compact")
        channel.conversation = conv
        for i in range(6):
            conv.append({"role": "user", "content": f"message number {i} " * 10})
            conv.messages[-1]["_db_id"] = 100 + i

        result = await plugin.compact_conversation(channel, conv, max_tokens=10)
        assert result is True

        call_kwargs = plugin.pm.ahook.on_compaction.call_args.kwargs
        retain_count = call_kwargs["retain_count"]
        expected_ids = [100 + i for i in range(6 - retain_count)]
        assert call_kwargs["compacted_ids"] == expected_ids

    async def test_summarizer_input_carries_no_underscore_keys(self):
        """The message prep for the summarizer strips all _-prefixed keys."""
        plugin = CompactionPlugin()
        plugin.pm = MagicMock()
        plugin.pm.ahook.on_compaction = AsyncMock()
        seen: list[list[dict]] = []

        async def fake_summarize(messages, prior_summaries=None):
            seen.append(messages)
            return "a summary"

        plugin._summarize = fake_summarize
        plugin._min_messages = 2
        plugin._compaction_threshold = 0.0
        plugin._compaction_retention = 0.0

        conv = ContextWindow("test:strip2")
        channel = Channel(transport="test", scope="strip2")
        for i in range(6):
            conv.append({"role": "user", "content": f"message number {i} " * 10})
            conv.messages[-1]["_db_id"] = 200 + i

        await plugin.compact_conversation(channel, conv, max_tokens=10)
        assert seen, "summarizer was not called"
        for msg in seen[0]:
            assert not any(k.startswith("_") for k in msg)


class TestResolveSingleResult:
    def test_empty_returns_none(self):
        assert resolve_single_result([], "hook") is None

    def test_all_none_returns_none(self):
        assert resolve_single_result([None, None], "hook") is None

    def test_single_value_returned(self):
        assert resolve_single_result([None, 5, None], "hook") == 5

    def test_multiple_values_logs_error_and_uses_first(self, caplog):
        with caplog.at_level(logging.ERROR, logger="corvidae.hooks"):
            result = resolve_single_result([None, 5, 7], "my_hook")
        assert result == 5
        assert any("my_hook" in rec.message for rec in caplog.records)
