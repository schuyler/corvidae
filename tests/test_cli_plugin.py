"""Tests for sherman.channels.cli.CLIPlugin."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sherman.channel import Channel, ChannelConfig, ChannelRegistry
from sherman.hooks import create_plugin_manager

from sherman.channels.cli import CLIPlugin


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
}

BASE_CONFIG: dict = {}


def _make_pm_with_registry(transport: str | None = None):
    """Create a plugin manager with a ChannelRegistry.

    If transport is given, pre-register one channel with that transport
    so that registry.by_transport(transport) returns a non-empty list.

    Returns (pm, registry). Callers that bypass on_start must set
    plugin._registry = registry on the CLIPlugin instance.
    """
    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")

    pm.ahook.on_message = AsyncMock()
    pm.ahook.send_message = AsyncMock()

    if transport is not None:
        registry.get_or_create(transport, "local")

    return pm, registry


# ---------------------------------------------------------------------------
# Section 1 — on_start
# ---------------------------------------------------------------------------


class TestOnStart:
    async def test_on_start_no_cli_channels_skips_task(self):
        """When the registry has no cli channels, on_start leaves _task as None."""
        pm, _registry = _make_pm_with_registry()  # no cli channel registered

        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        await plugin.on_start(config=BASE_CONFIG)

        assert plugin._task is None

    async def test_on_start_with_cli_channel_creates_task(self):
        """When registry has a cli:local channel, on_start creates a task."""
        pm, _registry = _make_pm_with_registry(transport="cli")

        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        # Patch _read_loop to avoid actually reading stdin.
        async def fake_read_loop():
            # Block indefinitely so the task stays alive long enough to inspect.
            await asyncio.sleep(9999)

        plugin._read_loop = fake_read_loop

        await plugin.on_start(config=BASE_CONFIG)

        try:
            assert plugin._task is not None
            assert isinstance(plugin._task, asyncio.Task)
        finally:
            plugin._task.cancel()
            try:
                await plugin._task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Section 2 — _read_loop behaviour
# ---------------------------------------------------------------------------


class TestReadLoop:
    async def test_read_loop_dispatches_on_message(self):
        """readline returns a line then "" (EOF).

        Verify on_message is called with the correct channel, sender="user",
        and the stripped line text.
        """
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")
        plugin._registry = registry  # bypass on_start

        lines = iter(["hello world\n", ""])

        def fake_readline():
            return next(lines)

        loop = asyncio.get_running_loop()

        async def fake_executor(executor, fn, *args):
            return fn(*args)

        with patch.object(loop, "run_in_executor", side_effect=fake_executor), \
             patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"):
            mock_stdin.readline = fake_readline
            # Run the read loop directly (not via create_task) so we can await it.
            await plugin._read_loop()

        expected_channel = registry.get_or_create("cli", "local")
        pm.ahook.on_message.assert_awaited_once_with(
            channel=expected_channel,
            sender="user",
            text="hello world",
        )

    async def test_read_loop_skips_blank_lines(self):
        """readline returns a blank line then "".

        Verify on_message is NOT called.
        """
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")
        plugin._registry = registry  # bypass on_start

        lines = iter(["\n", ""])

        def fake_readline():
            return next(lines)

        loop = asyncio.get_running_loop()

        async def fake_executor(executor, fn, *args):
            return fn(*args)

        with patch.object(loop, "run_in_executor", side_effect=fake_executor), \
             patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"):
            mock_stdin.readline = fake_readline
            await plugin._read_loop()

        pm.ahook.on_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Section 3 — send_message
# ---------------------------------------------------------------------------


class TestSendMessage:
    async def test_send_message_cli_channel_prints(self, capsys):
        """send_message with a cli channel prints the text to stdout."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        channel = registry.get_or_create("cli", "local")
        await plugin.send_message(channel=channel, text="hello from agent")

        captured = capsys.readouterr()
        assert "hello from agent" in captured.out

    async def test_send_message_non_cli_channel_ignored(self, capsys):
        """send_message with a non-cli channel produces no stdout output."""
        pm, registry = _make_pm_with_registry()
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        irc_channel = registry.get_or_create("irc", "#general")
        await plugin.send_message(channel=irc_channel, text="should not appear")

        captured = capsys.readouterr()
        assert "should not appear" not in captured.out

    async def test_send_message_with_latency_ms_shows_dim_duration(self, capsys):
        """send_message with latency_ms=1500.0 outputs the duration as '(1.5s)'
        wrapped in ANSI dim codes."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        channel = registry.get_or_create("cli", "local")
        await plugin.send_message(channel=channel, text="hello", latency_ms=1500.0)

        captured = capsys.readouterr()
        assert "(1.5s)" in captured.out
        assert "\033[2m" in captured.out
        assert "\033[0m" in captured.out

    async def test_send_message_without_latency_ms_no_ansi(self, capsys):
        """send_message with latency_ms=None produces no ANSI escape codes."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        channel = registry.get_or_create("cli", "local")
        await plugin.send_message(channel=channel, text="hello", latency_ms=None)

        captured = capsys.readouterr()
        assert "\033[2m" not in captured.out
        assert "\033[0m" not in captured.out


# ---------------------------------------------------------------------------
# Section 4 — on_stop
# ---------------------------------------------------------------------------


class TestOnStop:
    async def test_on_stop_cancels_task(self):
        """After on_start creates a task, on_stop cancels it and sets _task to None."""
        pm, _registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        async def fake_read_loop():
            await asyncio.sleep(9999)

        plugin._read_loop = fake_read_loop

        await plugin.on_start(config=BASE_CONFIG)
        assert plugin._task is not None

        task = plugin._task
        await plugin.on_stop()

        assert task.cancelled()
        assert plugin._task is None

    async def test_on_stop_no_task_is_noop(self):
        """on_stop when _task is None does not raise."""
        pm, _registry = _make_pm_with_registry()
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        assert plugin._task is None
        # Should not raise.
        await plugin.on_stop()
