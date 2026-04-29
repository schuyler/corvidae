"""Tests for corvidae.jsonl_log.JsonlLogPlugin — hook-based dispatch.

JsonlLogPlugin implements on_conversation_event and on_compaction as peer
persistence hooks, writing JSONL logs alongside SQLite persistence.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(channel_id: str) -> object:
    """Build a minimal mock channel (no ContextWindow needed)."""
    channel = MagicMock()
    channel.id = channel_id
    return channel


# ---------------------------------------------------------------------------
# TestJsonlPluginNoOp
# ---------------------------------------------------------------------------


class TestJsonlPluginNoOp:
    async def test_jsonl_plugin_noop_without_config(self, tmp_path):
        """Plugin with no jsonl_log_dir in config is inert — no errors."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        plugin = JsonlLogPlugin(None)
        config = {"daemon": {}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")

        # on_conversation_event must not raise when plugin is disabled
        await plugin.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "hi"},
            message_type=MessageType.MESSAGE,
        )

        # on_compaction must not raise when plugin is disabled
        await plugin.on_compaction(
            channel=channel,
            summary_msg={"role": "assistant", "content": "[Summary]"},
            retain_count=0,
        )

        await plugin.on_stop()

    async def test_no_depends_on_attribute(self):
        """JsonlLogPlugin must not declare depends_on (no longer needs persistence first)."""
        from corvidae.jsonl_log import JsonlLogPlugin
        plugin = JsonlLogPlugin(None)
        # depends_on should be absent or empty — it no longer relies on persistence
        depends_on = getattr(plugin, "depends_on", None)
        assert not depends_on, (
            f"JsonlLogPlugin should not have depends_on, got {depends_on!r}"
        )

    async def test_no_ensure_conversation_method(self):
        """JsonlLogPlugin must not implement ensure_conversation (observer model removed)."""
        from corvidae.jsonl_log import JsonlLogPlugin
        plugin = JsonlLogPlugin(None)
        assert not hasattr(plugin, "ensure_conversation"), (
            "JsonlLogPlugin must not implement ensure_conversation after refactor"
        )


# ---------------------------------------------------------------------------
# TestJsonlPluginCreatesDirectory
# ---------------------------------------------------------------------------


class TestJsonlPluginCreatesDirectory:
    async def test_jsonl_plugin_creates_directory(self, tmp_path):
        """on_start must create the jsonl_log_dir if it does not exist."""
        from corvidae.jsonl_log import JsonlLogPlugin
        log_dir = tmp_path / "jsonl_logs"
        assert not log_dir.exists()

        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        assert log_dir.exists()


# ---------------------------------------------------------------------------
# TestOnConversationEvent
# ---------------------------------------------------------------------------


class TestOnConversationEvent:
    """Tests for JsonlLogPlugin.on_conversation_event hookimpl.

    RED phase: fails because on_conversation_event does not exist yet.
    """

    async def test_on_conversation_event_writes_jsonl_record(self, tmp_path):
        """on_conversation_event writes a JSONL record to the channel's log file."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        msg = {"role": "user", "content": "hello"}

        before = time.time()
        await plugin.on_conversation_event(
            channel=channel, message=msg, message_type=MessageType.MESSAGE
        )
        after = time.time()

        log_file = log_dir / "cli_default.jsonl"
        assert log_file.exists(), f"Expected JSONL file at {log_file}"

        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert before <= record["ts"] <= after
        assert record["channel"] == "cli:default"
        assert record["type"] == "message"
        assert record["message"] == msg

    async def test_on_conversation_event_context_type(self, tmp_path):
        """on_conversation_event with CONTEXT type writes type='context' in record."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")

        await plugin.on_conversation_event(
            channel=channel,
            message={"role": "system", "content": "ctx"},
            message_type=MessageType.CONTEXT,
        )

        log_file = log_dir / "cli_default.jsonl"
        record = json.loads(log_file.read_text().strip())
        assert record["type"] == "context"

    async def test_on_conversation_event_multiple_messages(self, tmp_path):
        """Multiple on_conversation_event calls append lines in order."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]

        for msg in messages:
            await plugin.on_conversation_event(
                channel=channel, message=msg, message_type=MessageType.MESSAGE
            )

        log_file = log_dir / "cli_default.jsonl"
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 3

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["message"]["content"] == messages[i]["content"]

    async def test_on_conversation_event_noop_when_disabled(self, tmp_path):
        """on_conversation_event must be a no-op when log_dir is not configured."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        plugin = JsonlLogPlugin(None)
        config = {"daemon": {}}  # no jsonl_log_dir
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        # Must not raise
        await plugin.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "hi"},
            message_type=MessageType.MESSAGE,
        )

        # No files should have been created
        assert not list(tmp_path.glob("**/*.jsonl"))

    async def test_on_conversation_event_strips_message_type_tag(self, tmp_path):
        """on_conversation_event must not write _message_type into the JSONL record's message."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        # Pass a message that has _message_type already tagged (shouldn't appear in output)
        msg = {"role": "user", "content": "hello", "_message_type": MessageType.MESSAGE}

        await plugin.on_conversation_event(
            channel=channel, message=msg, message_type=MessageType.MESSAGE
        )

        log_file = log_dir / "cli_default.jsonl"
        record = json.loads(log_file.read_text().strip())
        assert "_message_type" not in record["message"], (
            "JSONL record must not contain _message_type in the message field"
        )


# ---------------------------------------------------------------------------
# TestOnCompaction
# ---------------------------------------------------------------------------


class TestOnCompaction:
    """Tests for JsonlLogPlugin.on_compaction hookimpl.

    RED phase: fails because on_compaction does not exist yet.
    """

    async def test_on_compaction_writes_summary_record(self, tmp_path):
        """on_compaction writes a JSONL record with type='summary'."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        summary_msg = {"role": "assistant", "content": "[Summary of earlier conversation]\ntest"}

        before = time.time()
        await plugin.on_compaction(
            channel=channel, summary_msg=summary_msg, retain_count=2
        )
        after = time.time()

        log_file = log_dir / "cli_default.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]

        summary_records = [r for r in records if r["type"] == "summary"]
        assert len(summary_records) == 1

        rec = summary_records[0]
        assert rec["channel"] == "cli:default"
        assert rec["message"] == summary_msg
        assert before <= rec["ts"] <= after

    async def test_on_compaction_noop_when_disabled(self, tmp_path):
        """on_compaction must be a no-op when log_dir is not configured."""
        from corvidae.jsonl_log import JsonlLogPlugin

        plugin = JsonlLogPlugin(None)
        config = {"daemon": {}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        # Must not raise
        await plugin.on_compaction(
            channel=channel,
            summary_msg={"role": "assistant", "content": "[Summary]"},
            retain_count=0,
        )

        assert not list(tmp_path.glob("**/*.jsonl"))


# ---------------------------------------------------------------------------
# TestJsonlPluginSanitizesChannelId
# ---------------------------------------------------------------------------


class TestJsonlPluginSanitizesChannelId:
    async def test_sanitizes_colon(self, tmp_path):
        """Colon in channel ID is replaced with underscore in filename."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        await plugin.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "test"},
            message_type=MessageType.MESSAGE,
        )

        expected_file = log_dir / "cli_default.jsonl"
        assert expected_file.exists()

    async def test_sanitizes_slash_and_colon(self, tmp_path):
        """Channel IDs with '/' and ':' produce filenames with '_'."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("irc:#general/topic")
        await plugin.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "hi"},
            message_type=MessageType.MESSAGE,
        )

        expected_file = log_dir / "irc_#general_topic.jsonl"
        assert expected_file.exists(), (
            f"Expected sanitized filename {expected_file.name} in {log_dir}"
        )


# ---------------------------------------------------------------------------
# TestJsonlPluginStop
# ---------------------------------------------------------------------------


class TestJsonlPluginStop:
    async def test_on_stop_flushes_and_closes(self, tmp_path):
        """on_stop must flush and close all open file handles without raising."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        await plugin.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "before stop"},
            message_type=MessageType.MESSAGE,
        )

        await plugin.on_stop()

        log_file = log_dir / "cli_default.jsonl"
        content = log_file.read_text()
        assert content.strip(), "Log file must contain flushed content after on_stop"


# ---------------------------------------------------------------------------
# TestJsonlPluginValidJson
# ---------------------------------------------------------------------------


class TestJsonlPluginValidJson:
    async def test_all_lines_valid_json(self, tmp_path):
        """Every line in the JSONL file must be valid JSON."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        for msg in messages:
            await plugin.on_conversation_event(
                channel=channel, message=msg, message_type=MessageType.MESSAGE
            )

        log_file = log_dir / "cli_default.jsonl"
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == len(messages)

        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {line!r} — {exc}")

    async def test_record_has_required_fields(self, tmp_path):
        """Each JSONL record must have ts, channel, type, and message fields."""
        from corvidae.jsonl_log import JsonlLogPlugin
        from corvidae.context import MessageType

        log_dir = tmp_path / "logs"
        plugin = JsonlLogPlugin(None)
        config = {"daemon": {"jsonl_log_dir": str(log_dir)}}
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)

        channel = _make_channel("cli:default")
        await plugin.on_conversation_event(
            channel=channel,
            message={"role": "user", "content": "check fields"},
            message_type=MessageType.MESSAGE,
        )

        log_file = log_dir / "cli_default.jsonl"
        record = json.loads(log_file.read_text().strip())

        for field_name in ("ts", "channel", "type", "message"):
            assert field_name in record, f"Record missing field {field_name!r}: {record}"
