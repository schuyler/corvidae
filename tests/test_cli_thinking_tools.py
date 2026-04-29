"""Tests for CLIPlugin send_thinking and send_tool_status hooks."""

from unittest.mock import AsyncMock

import pytest

from corvidae.channel import ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.channels.cli import CLIPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
}


def _make_cli_plugin():
    """Create a CLIPlugin with a cli:local channel registered."""
    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")
    pm.ahook.on_message = AsyncMock()
    pm.ahook.send_message = AsyncMock()

    channel = registry.get_or_create("cli", "local")
    plugin = CLIPlugin(pm)
    pm.register(plugin, name="cli")
    return plugin, channel


def _make_non_cli_channel():
    """Create a plugin with an irc channel (not cli)."""
    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")

    channel = registry.get_or_create("irc", "#general")
    plugin = CLIPlugin(pm)
    pm.register(plugin, name="cli")
    return plugin, channel


# ---------------------------------------------------------------------------
# send_thinking
# ---------------------------------------------------------------------------


class TestSendThinking:
    async def test_cli_channel_prints_thinking_in_blue(self, capsys):
        """send_thinking on a cli channel prints text in bright blue ANSI."""
        plugin, channel = _make_cli_plugin()
        await plugin.send_thinking(channel=channel, text="I am reasoning about this")
        captured = capsys.readouterr()
        assert "I am reasoning about this" in captured.out
        assert "\033[94m" in captured.out
        assert "\033[0m" in captured.out

    async def test_non_cli_channel_ignored(self, capsys):
        """send_thinking on a non-cli channel produces no output."""
        plugin, channel = _make_non_cli_channel()
        await plugin.send_thinking(channel=channel, text="should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""

    async def test_long_thinking_truncated(self, capsys):
        """send_thinking truncates text longer than 500 chars with '...'."""
        plugin, channel = _make_cli_plugin()
        long_text = "x" * 600
        await plugin.send_thinking(channel=channel, text=long_text)
        captured = capsys.readouterr()
        assert "xxx..." in captured.out
        # Should not contain the full 600 chars
        assert "x" * 600 not in captured.out
        # Should contain first 500 chars worth of content
        assert "x" * 497 in captured.out  # 500 - 3 for "..."


# ---------------------------------------------------------------------------
# send_tool_status — dispatched
# ---------------------------------------------------------------------------


class TestSendToolStatusDispatched:
    async def test_cli_channel_prints_dispatched_tool(self, capsys):
        """send_tool_status dispatched on a cli channel prints in bright magenta with ⚙."""
        plugin, channel = _make_cli_plugin()
        await plugin.send_tool_status(
            channel=channel,
            tool_name="shell",
            status="dispatched",
            args_summary='{"command": "ls"}',
        )
        captured = capsys.readouterr()
        assert "⚙" in captured.out
        assert "shell" in captured.out
        assert "\033[95m" in captured.out
        assert "\033[0m" in captured.out

    async def test_dispatched_includes_args_summary(self, capsys):
        """send_tool_status dispatched shows pretty-printed JSON args."""
        plugin, channel = _make_cli_plugin()
        await plugin.send_tool_status(
            channel=channel,
            tool_name="read_file",
            status="dispatched",
            args_summary='{"path": "/tmp/test.txt"}',
        )
        captured = capsys.readouterr()
        assert "read_file" in captured.out
        assert "path" in captured.out

    async def test_dispatched_without_args_summary(self, capsys):
        """send_tool_status dispatched with no args_summary shows just the tool name."""
        plugin, channel = _make_cli_plugin()
        await plugin.send_tool_status(
            channel=channel,
            tool_name="shell",
            status="dispatched",
            args_summary=None,
        )
        captured = capsys.readouterr()
        assert "⚙" in captured.out
        assert "shell" in captured.out

    async def test_non_cli_channel_ignored(self, capsys):
        """send_tool_status dispatched on a non-cli channel produces no output."""
        plugin, channel = _make_non_cli_channel()
        await plugin.send_tool_status(
            channel=channel,
            tool_name="shell",
            status="dispatched",
        )
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# send_tool_status — completed
# ---------------------------------------------------------------------------


class TestSendToolStatusCompleted:
    async def test_cli_channel_prints_completed_tool(self, capsys):
        """send_tool_status completed on a cli channel prints ✓ in bright magenta."""
        plugin, channel = _make_cli_plugin()
        await plugin.send_tool_status(
            channel=channel,
            tool_name="shell",
            status="completed",
            result_summary="hello world",
        )
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "shell" in captured.out
        assert "hello world" in captured.out
        assert "\033[95m" in captured.out

    async def test_completed_without_result_summary(self, capsys):
        """send_tool_status completed with no result_summary shows just ✓ and name."""
        plugin, channel = _make_cli_plugin()
        await plugin.send_tool_status(
            channel=channel,
            tool_name="shell",
            status="completed",
            result_summary=None,
        )
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "shell" in captured.out
        assert "→" not in captured.out

    async def test_completed_truncates_long_result(self, capsys):
        """send_tool_status completed truncates result_summary to 80 chars."""
        plugin, channel = _make_cli_plugin()
        long_result = "x" * 200
        await plugin.send_tool_status(
            channel=channel,
            tool_name="shell",
            status="completed",
            result_summary=long_result,
        )
        captured = capsys.readouterr()
        assert "→" in captured.out
        # Should not contain full 200-char result
        assert "x" * 200 not in captured.out


# ---------------------------------------------------------------------------
# send_progress
# ---------------------------------------------------------------------------


class TestSendProgress:
    async def test_cli_channel_prints_progress_in_grey(self, capsys):
        """send_progress on a cli channel prints text in grey (bright black)."""
        plugin, channel = _make_cli_plugin()
        await plugin.send_progress(channel=channel, text="Let me look that up...")
        captured = capsys.readouterr()
        assert "Let me look that up..." in captured.out
        assert "\033[90m" in captured.out
        assert "\033[0m" in captured.out

    async def test_non_cli_channel_ignored(self, capsys):
        """send_progress on a non-cli channel produces no output."""
        plugin, channel = _make_non_cli_channel()
        await plugin.send_progress(channel=channel, text="should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""
