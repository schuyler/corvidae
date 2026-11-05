"""Tests for the cross_channel_send plugin (#234)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.hooks import create_plugin_manager
from cross_channel_send import CrossChannelSendPlugin


def _build_plugin():
    """Build a CrossChannelSendPlugin with a mock agent."""
    plugin = CrossChannelSendPlugin()
    mock_agent = AsyncMock()
    plugin._agent = mock_agent
    return plugin, mock_agent


class TestRegisterTools:
    async def test_registers_send_message_tool(self):
        """The plugin registers the send_message tool."""
        plugin, _ = _build_plugin()

        tool_registry = []
        plugin.register_tools(tool_registry)

        assert len(tool_registry) == 1
        assert tool_registry[0].__name__ == "send_message"


class TestSendMessage:
    async def test_sends_message_on_success(self):
        """send_message calls enqueue_agent_send and returns a confirmation."""
        plugin, mock_agent = _build_plugin()

        result = await plugin.send_message("signal:+1555", "hello")

        mock_agent.enqueue_agent_send.assert_awaited_once()
        call_args = mock_agent.enqueue_agent_send.call_args
        assert call_args[0][0] == "signal:+1555"
        assert call_args[0][1] == "hello"
        assert "signal:+1555" in result
        assert "Queued" in result

    async def test_returns_error_for_unregistered_channel(self):
        """send_message returns an error naming the channel when it's not registered."""
        plugin, mock_agent = _build_plugin()

        mock_agent.enqueue_agent_send.side_effect = KeyError(
            "channel 'nope:xyz' is not registered"
        )

        result = await plugin.send_message("nope:xyz", "hello")

        assert "nope:xyz" in result
        assert "not registered" in result

    async def test_returns_error_when_agent_not_started(self):
        """send_message returns an error when the agent is not started."""
        plugin, mock_agent = _build_plugin()

        mock_agent.enqueue_agent_send.side_effect = RuntimeError("Agent is not started")

        result = await plugin.send_message("signal:+1555", "hello")

        assert "not started" in result

    async def test_returns_error_for_empty_text(self):
        """send_message returns an error for empty text."""
        plugin, mock_agent = _build_plugin()

        mock_agent.enqueue_agent_send.side_effect = ValueError("text must not be empty")

        result = await plugin.send_message("signal:+1555", "")

        assert "empty" in result

    async def test_passes_trace_id_from_attribution(self):
        """send_message passes the current trace_id to enqueue_agent_send."""
        plugin, mock_agent = _build_plugin()

        from corvidae.attribution import set_attribution, reset_attribution
        token = set_attribution(
            stage="turn", channel_id="signal:schuyler",
            trace_id="test-trace-123", meta={},
        )
        try:
            result = await plugin.send_message("discord:dm:123", "meeting at 3")
        finally:
            reset_attribution(token)

        mock_agent.enqueue_agent_send.assert_awaited_once()
        call_kwargs = mock_agent.enqueue_agent_send.call_args.kwargs
        assert call_kwargs["trace_id"] == "test-trace-123"
