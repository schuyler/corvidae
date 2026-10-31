"""Tests for corvidae.channels.irc.IRCPlugin."""

import asyncio
import logging
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.outbound import OutboundMessage, OutboundProgress

# Mock pydle before importing irc_plugin since it may not be installed
# Create a more sophisticated mock that allows real IRCClient instantiation
class MockClient:
    def __init__(self, *args, **kwargs):
        self.nickname = kwargs.get('nickname', args[0] if args else 'TestBot')
        self.connected = False

    async def connect(self, *args, **kwargs):
        self.connected = True
        # Block forever to simulate holding connection
        await asyncio.sleep(9999)

    async def on_connect(self):
        pass

    async def on_channel_message(self, target, by, message):
        pass

    async def on_private_message(self, target, by, message):
        pass

    async def join(self, channel):
        pass

    async def message(self, target, message):
        pass

    async def quit(self, message=""):
        self.connected = False

# Create a mock pydle module
mock_pydle = MagicMock()
mock_pydle.Client = MockClient

with patch.dict('sys.modules', {'pydle': mock_pydle}):
    from corvidae.channels import irc as _irc_module
    from corvidae.channels.irc import IRCPlugin, IRCClient, split_message


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "reasoning_history": "drop",
}

BASE_CONFIG: dict = {}

IRC_CONFIG = {
    "irc": {
        "host": "irc.example.com",
        "port": 6667,
        "nick": "TestBot",
        "channels": ["#test", "#general"],
    }
}


def _make_pm_with_registry():
    """Create a plugin manager with a ChannelRegistry."""
    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")
    pm.ahook.on_message = AsyncMock()
    pm.ahook.send = AsyncMock()
    return pm, registry


# ---------------------------------------------------------------------------
# Section 1 — on_start (3 tests)
# ---------------------------------------------------------------------------

class TestOnStart:
    async def test_on_start_no_irc_config_skips(self):
        """When config has no 'irc' key, client is None and task is None."""
        pm, _registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        await plugin.on_start(config=BASE_CONFIG)

        assert plugin.client is None
        assert plugin._connect_task is None

    async def test_on_start_with_irc_config_creates_client_and_task(self):
        """When config has 'irc' key, creates IRCClient and background task."""
        pm, _registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        # The real connect_with_retry will call the mocked connect (which blocks forever)
        await plugin.on_start(config=IRC_CONFIG)

        try:
            assert plugin.client is not None
            assert plugin._connect_task is not None
            assert isinstance(plugin._connect_task, asyncio.Task)
        finally:
            if plugin._connect_task:
                plugin._connect_task.cancel()
                try:
                    await plugin._connect_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def test_on_start_reads_channels_from_config(self):
        """self.channels populated from config irc.channels list."""
        pm, _registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        # on_init now reads channel config; must be called before on_start
        await plugin.on_init(pm=pm, config=IRC_CONFIG)
        # The real connect_with_retry will call the mocked connect (which blocks forever)
        await plugin.on_start(config=IRC_CONFIG)

        try:
            assert plugin.channels == ["#test", "#general"]
        finally:
            if plugin._connect_task:
                plugin._connect_task.cancel()
                try:
                    await plugin._connect_task
                except (asyncio.CancelledError, Exception):
                    pass


# ---------------------------------------------------------------------------
# Section 2 — IRCClient events (3 tests)
# ---------------------------------------------------------------------------

class TestIRCClientEvents:
    async def test_on_channel_message_forwards_to_hook(self):
        """Calls pm.ahook.on_message with channel=registry.get_or_create('irc', target)."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")
        plugin._registry = registry  # bypass on_start

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.nickname = "TestBot"

            # Simulate channel message
            await plugin.on_channel_message(
                target="#test",
                by="user",
                message="hello bot"
            )

            expected_channel = registry.get_or_create("irc", "#test")
            pm.ahook.on_message.assert_awaited_once_with(
                channel=expected_channel,
                sender="user",
                text="hello bot",
                trace_id=ANY,
            )
            # The transport mints a real (non-empty string) trace id.
            tid = pm.ahook.on_message.await_args.kwargs["trace_id"]
            assert isinstance(tid, str) and tid

    async def test_on_channel_message_ignores_self(self):
        """Messages from own nick are filtered out."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")
        plugin._registry = registry  # bypass on_start

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.nickname = "TestBot"

            # Simulate message from self
            await plugin.on_channel_message(
                target="#test",
                by="TestBot",
                message="hello from myself"
            )

            pm.ahook.on_message.assert_not_awaited()

    async def test_on_private_message_forwards_to_hook(self):
        """Private message uses sender nick as scope."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")
        plugin._registry = registry  # bypass on_start

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.nickname = "TestBot"

            # Simulate private message
            await plugin.on_private_message(
                target="TestBot",
                by="user",
                message="private message"
            )

            expected_channel = registry.get_or_create("irc", "user")
            pm.ahook.on_message.assert_awaited_once_with(
                channel=expected_channel,
                sender="user",
                text="private message",
                trace_id=ANY,
            )
            tid = pm.ahook.on_message.await_args.kwargs["trace_id"]
            assert isinstance(tid, str) and tid


# ---------------------------------------------------------------------------
# Section 3 — send_message (3 tests)
# ---------------------------------------------------------------------------

class TestSendMessage:
    async def test_send_message_irc_channel_sends_privmsg(self):
        """Calls client.message(channel.scope, chunk) for IRC channels."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client

            plugin.client.message = AsyncMock()

            channel = registry.get_or_create("irc", "#test")
            await plugin.send(channel=channel, event=OutboundMessage(text="hello"))

            plugin.client.message.assert_awaited_once_with("#test", "hello")

    async def test_send_message_non_irc_channel_ignored(self):
        """Non-IRC channels produce no client.message() call."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.message = AsyncMock()

            cli_channel = registry.get_or_create("cli", "local")
            await plugin.send(channel=cli_channel, event=OutboundMessage(text="hello"))

            plugin.client.message.assert_not_awaited()

    async def test_send_message_long_text_is_split(self):
        """Text exceeding max_len gets split into multiple client.message() calls."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.message = AsyncMock()

            channel = registry.get_or_create("irc", "#test")
            # Create text longer than default max_len (400)
            long_text = "a" * 500
            await plugin.send(channel=channel, event=OutboundMessage(text=long_text))

            # Should be called twice due to splitting
            assert plugin.client.message.await_count == 2

    async def test_send_message_skips_whitespace_only_chunks(self):
        """Chunks that are only whitespace are not sent to IRC."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.message = AsyncMock()

            channel = registry.get_or_create("irc", "#test")
            # Text that splits into chunks including whitespace-only ones
            await plugin.send(channel=channel, event=OutboundMessage(text="\n"))
            plugin.client.message.assert_not_awaited()

            # Also test bare whitespace
            await plugin.send(channel=channel, event=OutboundMessage(text="   "))
            plugin.client.message.assert_not_awaited()

            # But real content still sends
            await plugin.send(channel=channel, event=OutboundMessage(text="hello"))
            plugin.client.message.assert_awaited_once_with("#test", "hello")

        # Whitespace-only chunk in a multi-chunk list (defensive filter)
        plugin.client = MagicMock()
        plugin.client.message = AsyncMock()
        channel = registry.get_or_create("irc", "#test")
        with patch.object(_irc_module, 'split_message',
                          return_value=["para1", "\n\n", "para2"]):
            await plugin.send(channel=channel, event=OutboundMessage(text="ignored"))
        calls = [c.args[1] for c in plugin.client.message.await_args_list]
        assert calls == ["para1", "para2"]

        # Embedded newlines within a chunk are split into separate lines
        plugin.client.message = AsyncMock()
        with patch.object(_irc_module, 'split_message',
                          return_value=["line1\n\nline2\n\n"]):
            await plugin.send(channel=channel, event=OutboundMessage(text="ignored"))
        calls = [c.args[1] for c in plugin.client.message.await_args_list]
        assert calls == ["line1", "line2"]

        # CRLF line endings: trailing \r is stripped
        plugin.client.message = AsyncMock()
        with patch.object(_irc_module, 'split_message',
                          return_value=["line1\r\nline2\r\n"]):
            await plugin.send(channel=channel, event=OutboundMessage(text="ignored"))
        calls = [c.args[1] for c in plugin.client.message.await_args_list]
        assert calls == ["line1", "line2"]

    async def test_send_message_warns_when_disconnected(self, caplog):
        """Queued output is logged with the channel, length and queue depth."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        plugin.client = MagicMock()
        plugin.client.connected = False
        plugin.client.message = AsyncMock()

        channel = registry.get_or_create("irc", "#test")
        with caplog.at_level(logging.WARNING, logger="corvidae.irc_plugin"):
            await plugin.send(channel=channel, event=OutboundMessage(text="hello world"))

        plugin.client.message.assert_not_awaited()
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            "irc:#test" in m and "11" in m and "queued" in m and "depth=1" in m
            for m in warnings
        ), warnings

    async def test_send_message_real_split_no_blank_lines(self):
        """Integration: real split_message + send_message produces no blank PRIVMSGs."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        plugin.client = MagicMock()
        plugin.client.connected = True
        plugin.client.message = AsyncMock()

        channel = registry.get_or_create("irc", "#test")
        # Text long enough that split_message preserves \n\n in chunks
        para = "x" * 300
        text = para + "\n\n" + para
        await plugin.send(channel=channel, event=OutboundMessage(text=text))

        sent = [c.args[1] for c in plugin.client.message.await_args_list]
        assert len(sent) >= 2
        assert all(line.strip() for line in sent)


# ---------------------------------------------------------------------------
# Section 3b — send_progress (R2: progress text must not be silently
# dropped — the progress event must reach the client, not vanish as the base-class
# no-op would make it.)
# ---------------------------------------------------------------------------


class TestSendProgress:
    async def test_send_progress_irc_channel_writes_to_client(self):
        """send_progress on an IRC channel delivers the text like send_message."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.connected = True
            plugin.client.message = AsyncMock()

            channel = registry.get_or_create("irc", "#test")
            await plugin.send(channel=channel, event=OutboundProgress(text="let me look that up"))

            plugin.client.message.assert_awaited_once_with("#test", "let me look that up")

    async def test_send_progress_non_irc_channel_writes_nothing(self):
        """send_progress on a non-IRC channel is a no-op, like send_message."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.connected = True
            plugin.client.message = AsyncMock()

            cli_channel = registry.get_or_create("cli", "local")
            await plugin.send(channel=cli_channel, event=OutboundProgress(text="ignored"))

            plugin.client.message.assert_not_awaited()


# ---------------------------------------------------------------------------
# split_message's own algorithm (paragraph/sentence/word ladder, UTF-8
# byte-length behavior) is tested against its relocated home in
# tests/test_message_split.py. What stays here is IRC-specific: that
# IRCPlugin.send_message correctly consumes split_message's output (chunk
# delivery, whitespace-only-chunk filtering) — see TestSendMessage above.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Section 5 — on_stop (2 tests)
# ---------------------------------------------------------------------------

class TestOnStop:
    async def test_on_stop_cancels_connect_task(self):
        """Cancels task and calls client.quit()."""
        pm, _registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        # The real connect_with_retry will call the mocked connect (which blocks forever)
        await plugin.on_start(config=IRC_CONFIG)
        assert plugin._connect_task is not None

        task = plugin._connect_task
        await plugin.on_stop()

        assert task.cancelled()
        assert plugin._connect_task is None

    async def test_on_stop_no_task_is_noop(self):
        """No task means no exception."""
        pm, _registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        assert plugin._connect_task is None
        # Should not raise
        await plugin.on_stop()


# ---------------------------------------------------------------------------
# Section 6 — reconnection (1 test)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RED PHASE: send_message latency_ms parameter (architecture critique)
# ---------------------------------------------------------------------------


class TestSendMessageLatencyMs:
    async def test_send_accepts_latency_ms_on_event(self):
        """OutboundMessage carries latency_ms and IRC delivery must accept it."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('corvidae.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.message = AsyncMock()

            channel = registry.get_or_create("irc", "#test")
            # Must not raise TypeError for unexpected keyword argument
            await plugin.send(channel=channel, event=OutboundMessage(text="hello", latency_ms=42.5))

            # The message should still be sent normally
            plugin.client.message.assert_awaited_once_with("#test", "hello")


class TestReconnection:
    async def test_connect_with_retry_exponential_backoff(self):
        """Delay doubles each failure, caps at 300s."""
        pm, _registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        # Create a real IRCClient (our mock pydle.Client allows real instantiation)
        client = IRCClient(plugin, "TestBot", server="irc.example.com", port=6667, tls=False)
        plugin.client = client

        # Make connection fail repeatedly
        attempts = [0]

        async def failing_connect(*args, **kwargs):
            attempts[0] += 1
            raise Exception("Connection failed")

        client.connect = AsyncMock(side_effect=failing_connect)

        # Track delays
        delays = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(delay):
            delays.append(delay)
            # Shorten long delays for test speed
            if delay > 1:
                await original_sleep(0.01)

        # Set up _connect_task so connect_with_retry will run
        plugin._connect_task = asyncio.get_event_loop().create_future()

        with patch('asyncio.sleep', side_effect=tracking_sleep):
            task = asyncio.create_task(client.connect_with_retry())

            # Wait for a few retry attempts
            await asyncio.sleep(0.15)

            # Signal shutdown
            plugin._connect_task = None

            # Cancel the task
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_unexpected_exception_reenters_backoff(self, caplog):
        """A non-pydle exception must retry, not end the connect task."""
        pm, _registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        client = IRCClient(plugin, "TestBot", server="irc.example.com", port=6667)
        plugin.client = client
        plugin._connect_task = MagicMock()

        calls = []

        async def flaky_connect(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("unexpected handshake failure")
            plugin._connect_task = None  # shutdown after the retry

        client.connect = flaky_connect

        delays = []

        async def fake_sleep(delay):
            delays.append(delay)

        with caplog.at_level(logging.WARNING, logger="corvidae.irc_plugin"), \
                patch("corvidae.channels.irc.asyncio.sleep", fake_sleep):
            await client.connect_with_retry()

        assert len(calls) == 2, "connect_with_retry did not retry after ValueError"
        assert delays and delays[0] == client._INITIAL_BACKOFF
        assert any(
            "unexpected" in r.getMessage().lower()
            for r in caplog.records if r.levelno >= logging.WARNING
        ), [r.getMessage() for r in caplog.records]

        # Verify exponential backoff: delays should increase (10, 20, 40, ...)
        # Filter out the 86400 sleep (holds connection)
        retry_delays = [d for d in delays if d < 1000]
        if len(retry_delays) >= 2:
            assert retry_delays[1] >= retry_delays[0] * 2
        if len(retry_delays) >= 3:
            assert retry_delays[2] >= retry_delays[1] * 2
        # All delays should be <= 300s cap
        assert all(d <= 300 for d in retry_delays)


async def test_retry_warning_names_host_port_and_nick(caplog):
    """The IRC retry warning names the connection target it failed to reach."""
    import logging

    pm, _registry = _make_pm_with_registry()
    plugin = IRCPlugin(pm)
    pm.register(plugin, name="irc")

    client = IRCClient(
        plugin, "TestBot", server="irc.example.com", port=6667, tls=False
    )
    plugin.client = client
    plugin._connect_task = asyncio.get_event_loop().create_future()

    async def failing_connect(*args, **kwargs):
        plugin._connect_task = None
        raise OSError("connection refused")

    client.connect = AsyncMock(side_effect=failing_connect)

    # The suite mocks pydle, so its Error base is a MagicMock; give the
    # handler a real exception class for the except tuple.
    with patch.object(_irc_module.pydle, "Error", Exception), caplog.at_level(
        logging.WARNING, logger="corvidae.irc_plugin"
    ):
        await client.connect_with_retry()

    assert "irc.example.com" in caplog.text
    assert "6667" in caplog.text
    assert "TestBot" in caplog.text


# ---------------------------------------------------------------------------
# #106 — outbound queue while disconnected, flushed after reconnect
# ---------------------------------------------------------------------------


def _make_queue_fixture():
    """Build a plugin with a real IRCClient whose join/message share one mock.

    Returns (plugin, client, parent, channel); `parent.mock_calls` is a single
    ordered call list covering both join and message.
    """
    pm, registry = _make_pm_with_registry()
    plugin = IRCPlugin(pm)
    pm.register(plugin, name="irc")
    plugin.channels = ["#test"]
    client = IRCClient(plugin, "TestBot")
    parent = MagicMock()
    parent.join = AsyncMock()
    parent.message = AsyncMock()
    client.join = parent.join
    client.message = parent.message
    client.connected = False
    plugin.client = client
    return plugin, client, parent, registry.get_or_create("irc", "#test")


class TestOutboundQueue:
    async def test_send_while_disconnected_queues_then_flushes_on_connect(self):
        """R1: queued output is delivered in order, after the joins."""
        plugin, client, parent, channel = _make_queue_fixture()
        await plugin.send(channel=channel, event=OutboundMessage(text="first"))
        await plugin.send_progress(channel, OutboundProgress(text="second"))
        parent.message.assert_not_awaited()

        client.connected = True
        await client.on_connect()

        names = [c[0] for c in parent.mock_calls]
        assert names == ["join", "message", "message"]
        sent = [c.args for c in parent.message.await_args_list]
        assert sent == [("#test", "first"), ("#test", "second")]

    async def test_queue_overflow_drops_oldest_and_logs(self, caplog):
        """R2: the oldest entries go first; WARNING names channel and count."""
        plugin, client, parent, channel = _make_queue_fixture()
        with patch.object(_irc_module, "_OUTBOUND_QUEUE_MAX", 4):
            with caplog.at_level(logging.WARNING, logger="corvidae.irc_plugin"):
                for i in range(7):
                    await plugin.send(
                        channel=channel, event=OutboundMessage(text=f"m{i}")
                    )
            client.connected = True
            await client.on_connect()

        sent = [c.args[1] for c in parent.message.await_args_list]
        assert sent == ["m3", "m4", "m5", "m6"]
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            "irc:#test" in m and "dropped" in m and "3" in m for m in warnings
        ), warnings

    async def test_flushed_message_splits_like_connected_send(self):
        """R4: a queued multi-line message flushes as the connected send does."""
        text = "line one\n\nline two\n" + "y" * 500
        plugin, client, parent, channel = _make_queue_fixture()
        client.connected = True
        await plugin.send(channel=channel, event=OutboundMessage(text=text))
        expected = [c.args for c in parent.message.await_args_list]
        assert len(expected) >= 3

        plugin2, client2, parent2, channel2 = _make_queue_fixture()
        await plugin2.send(channel=channel2, event=OutboundMessage(text=text))
        client2.connected = True
        await client2.on_connect()
        assert [c.args for c in parent2.message.await_args_list] == expected

    async def test_send_during_flush_keeps_order(self):
        """R5: a send made mid-flush is delivered after older queued ones."""
        plugin, client, parent, channel = _make_queue_fixture()
        await plugin.send(channel=channel, event=OutboundMessage(text="old1"))
        await plugin.send(channel=channel, event=OutboundMessage(text="old2"))
        client.connected = True

        sent: list[str] = []
        injected = False

        async def slow_message(target, line):
            """Record the line; on the first one, inject a new send."""
            nonlocal injected
            sent.append(line)
            if not injected:
                injected = True
                await plugin.send(
                    channel=channel, event=OutboundMessage(text="new")
                )

        client.message = slow_message
        await plugin._flush_outbound()
        assert sent == ["old1", "old2", "new"]
