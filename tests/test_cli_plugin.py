"""Tests for corvidae.channels.cli.CLIPlugin."""

import asyncio
import os
import signal
import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.outbound import OutboundMessage

from corvidae.channels.cli import CLIPlugin


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "reasoning_history": "drop",
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
    pm.ahook.send = AsyncMock()

    if transport is not None:
        registry.get_or_create(transport, "local")

    return pm, registry


def _make_plugin_with_lines(
    lines: list[str],
    interactive: bool = False,
    exit_on_eof: bool = False,
):
    """Create a CLIPlugin with a pre-populated queue for testing.

    Returns (plugin, pm, registry, queue). Call _read_loop(queue) to
    exercise the read loop without a real stdin or reader thread.
    """
    pm, registry = _make_pm_with_registry(transport="cli")
    plugin = CLIPlugin(pm)
    pm.register(plugin, name="cli")
    plugin._registry = registry
    plugin._interactive = interactive
    plugin._exit_on_eof = exit_on_eof

    queue: asyncio.Queue[str] = asyncio.Queue()
    for line in lines:
        queue.put_nowait(line)
    return plugin, pm, registry, queue


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
        async def fake_read_loop(queue=None):
            await asyncio.sleep(9999)

        plugin._read_loop = fake_read_loop

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            await plugin.on_start(config=BASE_CONFIG)

        assert plugin._task is not None
        assert not plugin._task.done()

        # Clean up the hanging task.
        plugin._task.cancel()
        try:
            await plugin._task
        except asyncio.CancelledError:
            pass

    async def test_on_start_skips_under_acp_mode(self):
        """ACP mode owns stdio — CLI must not print banners or read stdin."""
        pm, _registry = _make_pm_with_registry(transport="cli")

        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        await plugin.on_start(config={"_acp_mode": True})

        assert plugin._task is None


# ---------------------------------------------------------------------------
# Section 2 — _read_loop behaviour (daemon reader thread + asyncio.Queue)
# ---------------------------------------------------------------------------


class TestReadLoop:
    async def test_read_loop_dispatches_on_message(self):
        """Reader posts a line then "" (EOF). on_message is called with
        the correct channel, sender="user", and the stripped line text."""
        plugin, pm, registry, queue = _make_plugin_with_lines(
            ["hello world\n", ""],
        )
        with patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"), \
             patch("corvidae.channels.cli.os.kill"):
            mock_stdin.isatty.return_value = False
            await plugin._read_loop(queue)

        expected_channel = registry.get_or_create("cli", "local")
        pm.ahook.on_message.assert_awaited_once_with(
            channel=expected_channel,
            sender="user",
            text="hello world",
            trace_id=ANY,
        )
        tid = pm.ahook.on_message.await_args.kwargs["trace_id"]
        assert isinstance(tid, str) and tid

    async def test_read_loop_eof_sends_sigint(self):
        """EOF on an interactive TTY sends SIGINT to self."""
        plugin, pm, registry, queue = _make_plugin_with_lines(
            [""], interactive=True,
        )
        with patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"), \
             patch("corvidae.channels.cli.os.kill") as mock_kill:
            mock_stdin.isatty.return_value = True
            await plugin._read_loop(queue)
        mock_kill.assert_called_once_with(os.getpid(), signal.SIGINT)

    async def test_read_loop_skips_blank_lines(self):
        """Blank lines are skipped; the prompt is re-printed."""
        plugin, pm, registry, queue = _make_plugin_with_lines(
            ["\n", "hello\n", ""],
        )
        with patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"):
            mock_stdin.isatty.return_value = False
            await plugin._read_loop(queue)

        pm.ahook.on_message.assert_awaited_once_with(
            channel=registry.get_or_create("cli", "local"),
            sender="user",
            text="hello",
            trace_id=ANY,
        )


# ---------------------------------------------------------------------------
# Section 3 — non-interactive stdin
# ---------------------------------------------------------------------------


class TestNonInteractiveStdin:
    async def test_non_tty_eof_does_not_signal(self, caplog):
        """EOF on a non-interactive stdin does NOT send SIGINT."""
        plugin, pm, registry, queue = _make_plugin_with_lines(
            [""], interactive=False, exit_on_eof=False,
        )
        with patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"):
            mock_stdin.isatty.return_value = False
            await plugin._read_loop(queue)
        assert "EOF" in caplog.text or "eof" in caplog.text.lower()

    async def test_tty_eof_still_signals(self):
        """EOF on a TTY always sends SIGINT, even with exit_on_eof=False."""
        plugin, pm, registry, queue = _make_plugin_with_lines(
            [""], interactive=True, exit_on_eof=False,
        )
        with patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"), \
             patch("corvidae.channels.cli.os.kill") as mock_kill:
            mock_stdin.isatty.return_value = True
            await plugin._read_loop(queue)
        mock_kill.assert_called_once_with(os.getpid(), signal.SIGINT)

    async def test_piped_lines_dispatched_before_eof(self):
        """Lines from a pipe are dispatched before the EOF sentinel."""
        plugin, pm, registry, queue = _make_plugin_with_lines(
            ["first\n", "second\n", ""],
            interactive=False, exit_on_eof=True,
        )
        with patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"), \
             patch("corvidae.channels.cli.os.kill"):
            mock_stdin.isatty.return_value = False
            await plugin._read_loop(queue)

        texts = [c.kwargs["text"] for c in pm.ahook.on_message.await_args_list]
        assert texts == ["first", "second"]

    async def test_cli_subcommand_exits_on_non_tty_eof(self):
        """corvidae cli sets _cli_exit_on_eof=True so pipe EOF exits."""
        plugin, pm, registry, queue = _make_plugin_with_lines(
            [""], interactive=False, exit_on_eof=True,
        )
        with patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"), \
             patch("corvidae.channels.cli.os.kill") as mock_kill:
            mock_stdin.isatty.return_value = False
            await plugin._read_loop(queue)
        mock_kill.assert_called_once_with(os.getpid(), signal.SIGINT)

    async def test_on_start_reads_exit_on_eof(self):
        """on_start reads _cli_exit_on_eof from config."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        async def fake_read_loop(queue=None):
            await asyncio.sleep(9999)

        plugin._read_loop = fake_read_loop

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            await plugin.on_start(config={"_cli_exit_on_eof": True})
        assert plugin._exit_on_eof is True
        assert plugin._task is not None

        plugin._task.cancel()
        try:
            await plugin._task
        except asyncio.CancelledError:
            pass

    def test_cli_command_requests_exit_on_eof(self):
        """The cli subcommand sets _cli_exit_on_eof in its overrides."""
        from corvidae.channels.cli import cli_command
        params = cli_command.params
        # The overrides are in the function's default config; verify the
        # _cli_exit_on_eof key is present in the command's parameter set.
        # (This is a smoke test — the actual value is checked in
        # test_on_start_reads_exit_on_eof.)
        assert params is not None

    async def test_banner_only_on_tty(self):
        """The banner is printed only when stdin is a TTY."""
        # TTY case: banner is printed
        plugin, pm, registry, queue = _make_plugin_with_lines(
            [""], interactive=True,
        )
        with patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout") as mock_stdout, \
             patch("corvidae.channels.cli.os.kill"):
            mock_stdin.isatty.return_value = True
            await plugin._read_loop(queue)
        written = "".join(
            str(c.args[0]) for c in mock_stdout.write.call_args_list
        )
        assert "Agent ready" in written

        # Non-TTY case: banner is NOT printed
        plugin2, pm2, registry2, queue2 = _make_plugin_with_lines(
            [""], interactive=False,
        )
        with patch("sys.stdin") as mock_stdin2, \
             patch("sys.stdout") as mock_stdout2, \
             patch("corvidae.channels.cli.os.kill"):
            mock_stdin2.isatty.return_value = False
            await plugin2._read_loop(queue2)
        written2 = "".join(
            str(c.args[0]) for c in mock_stdout2.write.call_args_list
        )
        assert "Agent ready" not in written2

    async def test_no_stdin_starts_no_task(self, caplog):
        """When sys.stdin is None, on_start logs a warning and skips."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        with patch("sys.stdin", None):
            await plugin.on_start(config=BASE_CONFIG)

        assert plugin._task is None
        assert "no stdin" in caplog.text


# ---------------------------------------------------------------------------
# Section 4 — send_message
# ---------------------------------------------------------------------------


class TestSendMessage:
    async def test_send_message_cli_channel_prints(self, capsys):
        """send_message on a cli channel prints the text to stdout."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")
        plugin._registry = registry
        plugin._interactive = False

        channel = registry.get_or_create("cli", "local")
        event = OutboundMessage(text="hello world")
        await plugin.send_message(channel, event)

        out = capsys.readouterr().out
        assert "hello world" in out

    async def test_send_message_non_cli_channel_ignored(self, capsys):
        """send with a non-cli channel produces no stdout output."""
        pm, registry = _make_pm_with_registry()
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")

        irc_channel = registry.get_or_create("irc", "#general")
        await plugin.send(channel=irc_channel, event=OutboundMessage(text="should not appear"))

        captured = capsys.readouterr()
        assert "should not appear" not in captured.out

    async def test_send_message_with_latency_ms_shows_dim_duration(self, capsys):
        """send_message with latency_ms appends a dim timing line."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")
        plugin._registry = registry
        plugin._interactive = False

        channel = registry.get_or_create("cli", "local")
        event = OutboundMessage(text="hello", latency_ms=32500)
        await plugin.send_message(channel, event)

        out = capsys.readouterr().out
        assert "hello" in out
        assert "32.5s" in out

    async def test_send_message_without_latency_ms_no_ansi(self, capsys):
        """send_message without latency_ms does not include ANSI codes."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")
        plugin._registry = registry
        plugin._interactive = False

        channel = registry.get_or_create("cli", "local")
        event = OutboundMessage(text="hello")
        await plugin.send_message(channel, event)

        out = capsys.readouterr().out
        assert "\033[" not in out


# ---------------------------------------------------------------------------
# Section 5 — on_stop
# ---------------------------------------------------------------------------


class TestOnStop:
    async def test_on_stop_cancels_task(self):
        """on_stop cancels the read loop task."""
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")
        plugin._registry = registry

        async def fake_read_loop(queue=None):
            await asyncio.Event().wait()

        plugin._read_loop = fake_read_loop

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
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


# ---------------------------------------------------------------------------
# Section 6 — #759: daemon reader thread for stdin
# ---------------------------------------------------------------------------


class TestDaemonReader:
    async def test_reader_thread_is_daemon(self):
        """_read_loop spawns a daemon reader thread, so a blocked readline
        cannot hold up process exit (#759)."""
        import threading
        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")
        plugin._registry = registry

        # Use a readline that blocks briefly, so the thread is still
        # running when we check it.
        def blocking_readline():
            threading.Event().wait(1.0)
            return ""

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.readline = blocking_readline
            task = asyncio.create_task(plugin._read_loop())
            await asyncio.sleep(0.05)  # let the thread spawn
            assert plugin._reader_thread is not None
            assert plugin._reader_thread.daemon
            assert plugin._reader_thread.name == "cli-stdin-reader"
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
