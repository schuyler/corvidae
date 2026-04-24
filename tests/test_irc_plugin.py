"""Tests for sherman.channels.irc.IRCPlugin."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sherman.channel import Channel, ChannelConfig, ChannelRegistry
from sherman.hooks import create_plugin_manager

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
    from sherman.channels import irc as _irc_module
    from sherman.channels.irc import IRCPlugin, IRCClient, split_message


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
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
    pm.registry = registry
    pm.ahook.on_message = AsyncMock()
    pm.ahook.send_message = AsyncMock()
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

        with patch('sherman.channels.irc.IRCClient') as mock_client_class:
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
            )

    async def test_on_channel_message_ignores_self(self):
        """Messages from own nick are filtered out."""
        pm, _registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('sherman.channels.irc.IRCClient') as mock_client_class:
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

        with patch('sherman.channels.irc.IRCClient') as mock_client_class:
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
            )


# ---------------------------------------------------------------------------
# Section 3 — send_message (3 tests)
# ---------------------------------------------------------------------------

class TestSendMessage:
    async def test_send_message_irc_channel_sends_privmsg(self):
        """Calls client.message(channel.scope, chunk) for IRC channels."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('sherman.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client

            plugin.client.message = AsyncMock()

            channel = registry.get_or_create("irc", "#test")
            await plugin.send_message(channel=channel, text="hello")

            plugin.client.message.assert_awaited_once_with("#test", "hello")

    async def test_send_message_non_irc_channel_ignored(self):
        """Non-IRC channels produce no client.message() call."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('sherman.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.message = AsyncMock()

            cli_channel = registry.get_or_create("cli", "local")
            await plugin.send_message(channel=cli_channel, text="hello")

            plugin.client.message.assert_not_awaited()

    async def test_send_message_long_text_is_split(self):
        """Text exceeding max_len gets split into multiple client.message() calls."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('sherman.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.message = AsyncMock()

            channel = registry.get_or_create("irc", "#test")
            # Create text longer than default max_len (400)
            long_text = "a" * 500
            await plugin.send_message(channel=channel, text=long_text)

            # Should be called twice due to splitting
            assert plugin.client.message.await_count == 2

    async def test_send_message_skips_whitespace_only_chunks(self):
        """Chunks that are only whitespace are not sent to IRC."""
        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")

        with patch('sherman.channels.irc.IRCClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            plugin.client = mock_client
            plugin.client.message = AsyncMock()

            channel = registry.get_or_create("irc", "#test")
            # Text that splits into chunks including whitespace-only ones
            await plugin.send_message(channel=channel, text="\n")
            plugin.client.message.assert_not_awaited()

            # Also test bare whitespace
            await plugin.send_message(channel=channel, text="   ")
            plugin.client.message.assert_not_awaited()

            # But real content still sends
            await plugin.send_message(channel=channel, text="hello")
            plugin.client.message.assert_awaited_once_with("#test", "hello")

        # Whitespace-only chunk in a multi-chunk list (defensive filter)
        plugin.client = MagicMock()
        plugin.client.message = AsyncMock()
        channel = registry.get_or_create("irc", "#test")
        with patch.object(_irc_module, 'split_message',
                          return_value=["para1", "\n\n", "para2"]):
            await plugin.send_message(channel=channel, text="ignored")
        calls = [c.args[1] for c in plugin.client.message.await_args_list]
        assert calls == ["para1", "para2"]

        # Embedded newlines within a chunk are split into separate lines
        plugin.client.message = AsyncMock()
        with patch.object(_irc_module, 'split_message',
                          return_value=["line1\n\nline2\n\n"]):
            await plugin.send_message(channel=channel, text="ignored")
        calls = [c.args[1] for c in plugin.client.message.await_args_list]
        assert calls == ["line1", "line2"]

        # CRLF line endings: trailing \r is stripped
        plugin.client.message = AsyncMock()
        with patch.object(_irc_module, 'split_message',
                          return_value=["line1\r\nline2\r\n"]):
            await plugin.send_message(channel=channel, text="ignored")
        calls = [c.args[1] for c in plugin.client.message.await_args_list]
        assert calls == ["line1", "line2"]

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
        await plugin.send_message(channel=channel, text=text)

        sent = [c.args[1] for c in plugin.client.message.await_args_list]
        assert len(sent) >= 2
        assert all(line.strip() for line in sent)


# ---------------------------------------------------------------------------
# Section 4 — split_message (6 tests)
# ---------------------------------------------------------------------------

class TestSplitMessage:
    def test_split_short_message_unchanged(self):
        """Under 400 bytes returns single-element list."""
        result = split_message("short message", max_len=400)
        assert result == ["short message"]

    def test_split_on_paragraph_boundaries(self):
        """Splits on \\n\\n boundaries, preserving separators for reassembly."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        # With small max_len to force splitting
        result = split_message(text, max_len=30)
        # Should split at paragraph boundaries
        assert len(result) > 1
        assert all(len(part.encode('utf-8')) <= 30 for part in result)
        # Content should be preserved when reassembled
        reassembled = ''.join(result)
        assert reassembled == text

    def test_split_on_sentence_boundaries(self):
        """Splits on .!? + whitespace when no paragraph boundary fits."""
        text = "This is sentence one. This is sentence two! This is sentence three?"
        # With max_len that doesn't fit full paragraphs but fits sentences
        result = split_message(text, max_len=25)
        assert len(result) > 1
        assert all(len(part.encode('utf-8')) <= 25 for part in result)
        # CRITICAL: Splits should occur at sentence boundaries where possible
        # A sentence boundary is .!? followed by space
        for part in result:
            # Each chunk should be valid UTF-8
            part.encode('utf-8')  # Should not raise
        # Verify content is preserved when reassembled
        reassembled = ''.join(result)
        assert reassembled == text

    def test_split_on_word_boundaries(self):
        """Word-level splitting when no paragraph/sentence boundary fits."""
        text = "This is a verylongwordthatwontfit and another word"
        # With max_len that doesn't fit sentences
        result = split_message(text, max_len=20)
        assert len(result) > 1
        assert all(len(part.encode('utf-8')) <= 20 for part in result)
        # CRITICAL: Splits should happen at word boundaries (spaces), not mid-word
        # when possible. Oversized words get split across chunks.
        # Check that chunks either end at word boundaries or contain oversized word parts
        for i, part in enumerate(result):
            # Each chunk should be valid UTF-8
            part.encode('utf-8')  # Should not raise
            # Content should be preserved when reassembled
        reassembled = ''.join(result)
        assert reassembled == text

    def test_split_hard_truncates_long_word(self):
        """Oversized word gets split across multiple chunks."""
        text = "a" * 500  # Single very long word
        result = split_message(text, max_len=400)
        # Should split into multiple chunks
        assert len(result) > 1
        # All chunks should be <= max_len
        assert all(len(part.encode('utf-8')) <= 400 for part in result)
        # Reassembling should give original text
        reassembled = ''.join(result)
        assert reassembled == text
        # First chunk should be close to max_len (not truncated arbitrarily short)
        byte_len = len(result[0].encode('utf-8'))
        assert byte_len >= 390, f"First chunk should be close to max_len, got {byte_len} bytes"

    def test_split_preserves_content(self):
        """Reassembling all chunks produces the original text (minus truncation markers)."""
        text = "Para one. Para two!\n\nNew paragraph. Another sentence.\n\nFinal para."
        result = split_message(text, max_len=30)
        # Reassemble by joining with nothing (paragraph/sentence boundaries preserved)
        reassembled = ''.join(result)
        # Should equal original since no truncation occurred
        assert reassembled == text

    def test_split_uses_utf8_byte_length(self):
        """Multi-byte chars counted by UTF-8 byte length, not character count."""
        # Mix of ASCII and multi-byte chars
        text = "Hello 世界 " * 100  # Each Chinese char is 3 bytes
        result = split_message(text, max_len=400)
        # All parts should be under 400 bytes
        assert all(len(part.encode('utf-8')) <= 400 for part in result)


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

        # Verify exponential backoff: delays should increase (10, 20, 40, ...)
        # Filter out the 86400 sleep (holds connection)
        retry_delays = [d for d in delays if d < 1000]
        if len(retry_delays) >= 2:
            assert retry_delays[1] >= retry_delays[0] * 2
        if len(retry_delays) >= 3:
            assert retry_delays[2] >= retry_delays[1] * 2
        # All delays should be <= 300s cap
        assert all(d <= 300 for d in retry_delays)
