"""Tests for the cross_channel_send plugin (#234)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cross_channel_send import CrossChannelSendPlugin, DEFAULT_MAX_HOPS


def _build_plugin(max_hops=None):
    """Build a CrossChannelSendPlugin with a mock agent."""
    plugin = CrossChannelSendPlugin()
    mock_agent = AsyncMock()
    mock_agent.list_channels = MagicMock(return_value=["signal:alice", "web:main"])
    plugin._agent = mock_agent
    if max_hops is not None:
        plugin._max_hops = max_hops
    return plugin, mock_agent


class TestRegisterTools:
    async def test_registers_two_tools(self):
        """The plugin registers send_message and list_channels."""
        plugin, _ = _build_plugin()

        tool_registry = []
        plugin.register_tools(tool_registry)

        assert len(tool_registry) == 2
        names = [t.__name__ for t in tool_registry]
        assert "send_message" in names
        assert "list_channels" in names


class TestSendMessage:
    async def test_sends_message_on_success(self):
        """send_message calls enqueue_agent_send with attribution and returns confirmation."""
        plugin, mock_agent = _build_plugin()

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="signal:alice",
            trace_id="test-trace-123", meta={},
        )
        try:
            result = await plugin.send_message("web:main", "hello")
        finally:
            reset_attribution(token)

        mock_agent.enqueue_agent_send.assert_awaited_once()
        call_args = mock_agent.enqueue_agent_send.call_args
        assert call_args[0][0] == "web:main"
        assert call_args[0][1] == "hello"
        assert call_args.kwargs["source_channel_id"] == "signal:alice"
        assert call_args.kwargs["trace_id"] == "test-trace-123"
        assert call_args.kwargs["hop_count"] == 1
        assert result == "Queued on web:main."

    async def test_returns_error_for_unregistered_channel(self):
        """send_message returns an error naming the channel when it's not registered."""
        plugin, mock_agent = _build_plugin()

        mock_agent.enqueue_agent_send.side_effect = KeyError(
            "channel 'nope:xyz' is not registered"
        )

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="signal:alice",
            trace_id="t1", meta={},
        )
        try:
            result = await plugin.send_message("nope:xyz", "hello")
        finally:
            reset_attribution(token)

        assert "nope:xyz" in result
        assert "not registered" in result

    async def test_returns_error_when_agent_not_started(self):
        """send_message returns an error when the agent is not available."""
        plugin = CrossChannelSendPlugin()
        plugin._agent = None

        result = await plugin.send_message("signal:alice", "hello")

        assert "not available" in result

    async def test_returns_error_for_empty_text(self):
        """send_message returns an error for empty text (via kernel ValueError)."""
        plugin, mock_agent = _build_plugin()

        mock_agent.enqueue_agent_send.side_effect = ValueError(
            "text must be a non-empty string"
        )

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="signal:alice",
            trace_id="t1", meta={},
        )
        try:
            result = await plugin.send_message("signal:alice", "")
        finally:
            reset_attribution(token)

        assert "non-empty" in result

    async def test_returns_error_without_attribution(self):
        """send_message returns an error when no attribution context is set."""
        plugin, mock_agent = _build_plugin()

        result = await plugin.send_message("web:main", "hello")

        assert "no source channel" in result
        mock_agent.enqueue_agent_send.assert_not_awaited()

    async def test_passes_trace_id_from_attribution(self):
        """send_message passes the current trace_id and channel_id to enqueue_agent_send."""
        plugin, mock_agent = _build_plugin()

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="signal:alice",
            trace_id="trace-abc-456", meta={},
        )
        try:
            await plugin.send_message("discord:dm:123", "meeting at 3")
        finally:
            reset_attribution(token)

        mock_agent.enqueue_agent_send.assert_awaited_once()
        call_kwargs = mock_agent.enqueue_agent_send.call_args.kwargs
        assert call_kwargs["trace_id"] == "trace-abc-456"
        assert call_kwargs["source_channel_id"] == "signal:alice"


class TestHopCount:
    """Loop prevention: hop_count increments across bounces, max_hops caps it."""

    async def test_first_send_has_hop_count_1(self):
        """A send from a user turn (no prior hops) has hop_count=1."""
        plugin, mock_agent = _build_plugin()

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="signal:alice",
            trace_id="t1", meta={},
        )
        try:
            await plugin.send_message("web:main", "hello")
        finally:
            reset_attribution(token)

        assert mock_agent.enqueue_agent_send.call_args.kwargs["hop_count"] == 1

    async def test_second_send_has_hop_count_2(self):
        """A send from a context that received an AGENT_SEND (hop 1) has hop_count=2."""
        plugin, mock_agent = _build_plugin()

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="web:main",
            trace_id="t1",
            meta={"origin": "agent_send", "source_channel": "signal:alice", "hop_count": 1},
        )
        try:
            await plugin.send_message("signal:alice", "reply")
        finally:
            reset_attribution(token)

        assert mock_agent.enqueue_agent_send.call_args.kwargs["hop_count"] == 2

    async def test_third_send_has_hop_count_3(self):
        """A send from a context at hop 2 has hop_count=3 (default max)."""
        plugin, mock_agent = _build_plugin()

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="signal:alice",
            trace_id="t1",
            meta={"origin": "agent_send", "source_channel": "web:main", "hop_count": 2},
        )
        try:
            await plugin.send_message("web:main", "reply again")
        finally:
            reset_attribution(token)

        assert mock_agent.enqueue_agent_send.call_args.kwargs["hop_count"] == 3

    async def test_exceeds_max_hops_refused(self):
        """A send that would exceed max_hops is refused."""
        plugin, mock_agent = _build_plugin(max_hops=3)

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="web:main",
            trace_id="t1",
            meta={"origin": "agent_send", "source_channel": "signal:alice", "hop_count": 3},
        )
        try:
            result = await plugin.send_message("signal:alice", "loop!")
        finally:
            reset_attribution(token)

        assert "loop detected" in result
        assert "hop 4" in result
        mock_agent.enqueue_agent_send.assert_not_awaited()

    async def test_custom_max_hops(self):
        """A custom max_hops config is respected."""
        plugin, mock_agent = _build_plugin(max_hops=1)

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="web:main",
            trace_id="t1",
            meta={"origin": "agent_send", "source_channel": "signal:alice", "hop_count": 1},
        )
        try:
            result = await plugin.send_message("signal:alice", "bounce")
        finally:
            reset_attribution(token)

        assert "loop detected" in result
        mock_agent.enqueue_agent_send.assert_not_awaited()

    async def test_default_max_hops_is_3(self):
        """The default max_hops is 3."""
        assert DEFAULT_MAX_HOPS == 3


class TestListChannels:
    async def test_lists_channels(self):
        """list_channels returns the registered channel IDs."""
        plugin, mock_agent = _build_plugin()

        result = await plugin.list_channels()

        assert "signal:alice" in result
        assert "web:main" in result

    async def test_lists_no_channels(self):
        """list_channels returns a message when no channels are registered."""
        plugin, mock_agent = _build_plugin()
        mock_agent.list_channels.return_value = []

        result = await plugin.list_channels()

        assert "No channels" in result

    async def test_agent_not_available(self):
        """list_channels returns an error when the agent is not available."""
        plugin = CrossChannelSendPlugin()
        plugin._agent = None

        result = await plugin.list_channels()

        assert "No channels" in result
