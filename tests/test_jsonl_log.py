"""Tests for the observer mechanism on ConversationLog and JsonlLogPlugin.

ConversationLog tests (Section 1):
- Tests reference self.observers, which does not exist yet — these fail until
  the observer mechanism is added to ConversationLog.

JsonlLogPlugin tests (Section 2):
- Tests import from corvidae.jsonl_log, which does not exist yet — all
  JsonlLogPlugin tests fail at import until the module is created.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.conversation import ConversationLog, MessageType


# ---------------------------------------------------------------------------
# Section 1: Observer mechanism on ConversationLog
# ---------------------------------------------------------------------------


class TestConversationLogObservers:
    """Tests for self.observers list and observer dispatch in ConversationLog."""

    async def test_init_observers_is_empty_list(self, db):
        """__init__ must initialize self.observers as an empty list."""
        conv = ConversationLog(db, channel_id="chan1")

        # Fails until self.observers is added to __init__
        assert hasattr(conv, "observers"), "ConversationLog must have an 'observers' attribute"
        assert conv.observers == []

    async def test_observers_type_annotation(self, db):
        """self.observers must be a list (not None or other type)."""
        conv = ConversationLog(db, channel_id="chan1")

        assert isinstance(conv.observers, list)

    async def test_append_calls_observer(self, db):
        """append() must call each observer after persisting, with correct args."""
        conv = ConversationLog(db, channel_id="chan1")
        observer = AsyncMock()
        conv.observers.append(observer)

        message = {"role": "user", "content": "hello"}
        await conv.append(message)

        observer.assert_awaited_once()
        args = observer.call_args
        channel_id_arg, message_arg, message_type_arg, ts_arg = args[0]

        assert channel_id_arg == "chan1"
        assert message_arg == message
        assert message_type_arg == MessageType.MESSAGE
        assert isinstance(ts_arg, float)

    async def test_append_passes_timestamp_to_observer(self, db):
        """Observer receives a float timestamp close to time.time()."""
        conv = ConversationLog(db, channel_id="chan1")
        observer = AsyncMock()
        conv.observers.append(observer)

        before = time.time()
        await conv.append({"role": "user", "content": "ts check"})
        after = time.time()

        _, _, _, ts_arg = observer.call_args[0]
        assert before <= ts_arg <= after

    async def test_append_calls_multiple_observers(self, db):
        """append() must call all observers, not just the first one."""
        conv = ConversationLog(db, channel_id="chan1")
        obs1 = AsyncMock()
        obs2 = AsyncMock()
        conv.observers.extend([obs1, obs2])

        await conv.append({"role": "user", "content": "multi"})

        obs1.assert_awaited_once()
        obs2.assert_awaited_once()

    async def test_append_calls_observer_with_explicit_message_type(self, db):
        """append() with explicit message_type passes that type to observers."""
        conv = ConversationLog(db, channel_id="chan1")
        observer = AsyncMock()
        conv.observers.append(observer)

        msg = {"role": "system", "content": "ctx"}
        await conv.append(msg, message_type=MessageType.CONTEXT)

        _, _, message_type_arg, _ = observer.call_args[0]
        assert message_type_arg == MessageType.CONTEXT

    async def test_observer_exception_is_logged_not_raised(self, db, caplog):
        """An observer that raises must not prevent append or other observers."""
        import logging

        conv = ConversationLog(db, channel_id="chan1")

        async def bad_observer(channel_id, message, message_type, ts):
            raise RuntimeError("observer failed")

        good_observer = AsyncMock()
        conv.observers.extend([bad_observer, good_observer])

        with caplog.at_level(logging.ERROR, logger="corvidae"):
            # Must not raise
            await conv.append({"role": "user", "content": "ok"})

        # The good observer still ran
        good_observer.assert_awaited_once()
        # An error was logged
        assert any("observer" in r.message.lower() or "observer" in str(r.exc_info).lower()
                   for r in caplog.records
                   if r.levelno >= logging.ERROR), (
            "Expected an error-level log record about the failing observer"
        )
        # Verify persistence succeeded despite observer failure
        async with conv.db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ?", ("chan1",)
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, "Message must be persisted to DB even when an observer raises"

    async def test_replace_with_summary_calls_observer(self, db):
        """replace_with_summary() must call each observer with the summary message."""
        conv = ConversationLog(db, channel_id="chan1")
        conv.messages = [
            {"role": "user", "content": f"msg {i}", "_message_type": MessageType.MESSAGE}
            for i in range(5)
        ]
        observer = AsyncMock()
        conv.observers.append(observer)

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\ntest"}
        await conv.replace_with_summary(summary_msg, retain_count=2)

        observer.assert_awaited_once()
        channel_id_arg, message_arg, message_type_arg, ts_arg = observer.call_args[0]

        assert channel_id_arg == "chan1"
        assert message_arg == summary_msg
        assert message_type_arg == MessageType.SUMMARY
        assert isinstance(ts_arg, float)

    async def test_replace_with_summary_passes_summary_ts_to_observer(self, db):
        """Observer timestamp from replace_with_summary matches the DB summary_ts."""
        conv = ConversationLog(db, channel_id="chan1")
        conv.messages = [
            {"role": "user", "content": "msg", "_message_type": MessageType.MESSAGE}
            for _ in range(3)
        ]
        captured_ts = []

        async def capturing_observer(channel_id, message, message_type, ts):
            captured_ts.append(ts)

        conv.observers.append(capturing_observer)

        before = time.time()
        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\nts test"}
        await conv.replace_with_summary(summary_msg, retain_count=1)
        after = time.time()

        assert len(captured_ts) == 1
        # The summary_ts passed to the observer should be close to now
        # (it may be slightly before 'before' due to the -1e-6 offset in the retain path,
        # so we allow a small margin)
        assert before - 0.001 <= captured_ts[0] <= after

    async def test_persist_returns_timestamp(self, db):
        """_persist() must return the float timestamp it wrote to the DB."""
        conv = ConversationLog(db, channel_id="chan1")
        msg = {"role": "user", "content": "hello"}

        before = time.time()
        result = await conv._persist(msg)
        after = time.time()

        # Fails until _persist() is changed to return its timestamp
        assert result is not None, "_persist() must return a timestamp, got None"
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert before <= result <= after

    async def test_observer_fires_after_persist(self, db):
        """Observer sees the DB row when it runs, confirming _persist completed first."""
        conv = ConversationLog(db, channel_id="chan1")
        db_count_at_observer_time = []

        async def counting_observer(channel_id, message, message_type, ts):
            async with conv.db.execute(
                "SELECT COUNT(*) FROM message_log WHERE channel_id = ?", (channel_id,)
            ) as cursor:
                row = await cursor.fetchone()
            db_count_at_observer_time.append(row[0])

        conv.observers.append(counting_observer)
        await conv.append({"role": "user", "content": "ordering check"})

        assert db_count_at_observer_time == [1], (
            "DB row must exist before observer fires"
        )


# ---------------------------------------------------------------------------
# Section 2: JsonlLogPlugin
# ---------------------------------------------------------------------------


def _make_channel(db, channel_id: str) -> object:
    """Build a minimal mock channel with a real ConversationLog."""
    conv = ConversationLog(db, channel_id=channel_id)
    channel = MagicMock()
    channel.id = channel_id
    channel.conversation = conv
    return channel


class TestJsonlPluginNoOp:
    async def test_jsonl_plugin_noop_without_config(self, db, tmp_path):
        """Plugin with no jsonl_log_dir in config must be inert with no errors."""
        from corvidae.jsonl_log import JsonlLogPlugin
        plugin = JsonlLogPlugin()
        config = {"daemon": {}}  # no jsonl_log_dir key

        # Must not raise
        await plugin.on_start(config=config)

        channel = _make_channel(db, "cli:default")
        # ensure_conversation: must not raise even though plugin is a no-op
        await plugin.ensure_conversation(channel=channel)

        # Appending through the observer must not raise
        await channel.conversation.append({"role": "user", "content": "hi"})

        # on_stop must not raise
        await plugin.on_stop()


class TestJsonlPluginCreatesDirectory:
    async def test_jsonl_plugin_creates_directory(self, db, tmp_path):
        """on_start must create the jsonl_log_dir if it does not exist."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "jsonl_logs"
        assert not log_dir.exists()

        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        assert log_dir.exists(), f"Expected {log_dir} to be created by on_start"


class TestJsonlPluginWritesOnAppend:
    async def test_jsonl_plugin_writes_on_append(self, db, tmp_path):
        """After ensure_conversation + append, the JSONL file exists."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        channel = _make_channel(db, "cli:default")
        await plugin.ensure_conversation(channel=channel)

        msg = {"role": "user", "content": "hello"}
        await channel.conversation.append(msg)

        log_file = log_dir / "cli_default.jsonl"
        assert log_file.exists(), f"Expected JSONL file at {log_file}"

    async def test_jsonl_plugin_write_content_on_append(self, db, tmp_path):
        """JSONL file contains a record with correct ts, channel, type, and message."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        channel = _make_channel(db, "cli:default")
        await plugin.ensure_conversation(channel=channel)

        msg = {"role": "user", "content": "hello"}
        before = time.time()
        await channel.conversation.append(msg)
        after = time.time()

        log_file = log_dir / "cli_default.jsonl"
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert before <= record["ts"] <= after
        assert record["channel"] == "cli:default"
        assert record["type"] == "message"
        assert record["message"] == msg


class TestJsonlPluginWritesOnSummary:
    async def test_jsonl_plugin_writes_on_summary(self, db, tmp_path):
        """After replace_with_summary, JSONL file contains a summary record."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        channel = _make_channel(db, "cli:default")
        await plugin.ensure_conversation(channel=channel)

        # Populate some messages
        conv = channel.conversation
        conv.messages = [
            {"role": "user", "content": f"msg {i}", "_message_type": MessageType.MESSAGE}
            for i in range(5)
        ]

        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\ntest summary"}
        await conv.replace_with_summary(summary_msg, retain_count=2)

        log_file = log_dir / "cli_default.jsonl"
        assert log_file.exists(), f"Expected JSONL file at {log_file}"
        lines = log_file.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]

        summary_records = [r for r in records if r["type"] == "summary"]
        assert len(summary_records) == 1
        assert summary_records[0]["message"] == summary_msg


class TestJsonlPluginSanitizesChannelId:
    async def test_jsonl_plugin_sanitizes_channel_id(self, db, tmp_path):
        """Channel IDs with '/' and ':' produce filenames with '_'."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        # Channel ID with both ':' and '/'
        channel = _make_channel(db, "irc:#general/topic")
        await plugin.ensure_conversation(channel=channel)
        await channel.conversation.append({"role": "user", "content": "hi"})

        expected_file = log_dir / "irc_#general_topic.jsonl"
        assert expected_file.exists(), (
            f"Expected sanitized filename {expected_file.name} in {log_dir}"
        )

    async def test_jsonl_plugin_sanitizes_colon(self, db, tmp_path):
        """Colon in channel ID is replaced with underscore in filename."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        channel = _make_channel(db, "cli:default")
        await plugin.ensure_conversation(channel=channel)
        await channel.conversation.append({"role": "user", "content": "test"})

        expected_file = log_dir / "cli_default.jsonl"
        assert expected_file.exists()


class TestJsonlPluginStop:
    async def test_jsonl_plugin_stop_flushes(self, db, tmp_path):
        """on_stop must flush and close all open file handles without raising."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        channel = _make_channel(db, "cli:default")
        await plugin.ensure_conversation(channel=channel)
        await channel.conversation.append({"role": "user", "content": "before stop"})

        # on_stop must not raise
        await plugin.on_stop()

        # File content must be readable and flushed after stop
        log_file = log_dir / "cli_default.jsonl"
        content = log_file.read_text()
        assert content.strip(), "Log file must contain flushed content after on_stop"


class TestJsonlPluginValidJson:
    async def test_jsonl_plugin_valid_json_lines(self, db, tmp_path):
        """Every line written to the JSONL file must be valid JSON."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        channel = _make_channel(db, "cli:default")
        await plugin.ensure_conversation(channel=channel)

        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        for msg in messages:
            await channel.conversation.append(msg)

        log_file = log_dir / "cli_default.jsonl"
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == len(messages)

        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {line!r} — {exc}")

    async def test_jsonl_record_has_required_fields(self, db, tmp_path):
        """Each JSONL record must have ts, channel, type, and message fields."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin()
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_start(config=config)

        channel = _make_channel(db, "cli:default")
        await plugin.ensure_conversation(channel=channel)
        await channel.conversation.append({"role": "user", "content": "check fields"})

        log_file = log_dir / "cli_default.jsonl"
        record = json.loads(log_file.read_text().strip())

        for field_name in ("ts", "channel", "type", "message"):
            assert field_name in record, f"Record is missing required field {field_name!r}: {record}"
