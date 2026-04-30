"""Tests for agent firing send_thinking and send_tool_status hooks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.turn import AgentTurnResult
from corvidae.channel import ChannelConfig, ChannelRegistry
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.hooks import create_plugin_manager
from corvidae.task import TaskPlugin

from helpers import build_plugin_and_channel, drain, drain_task_queue
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call_response,
    _make_tool_call,
)


# ---------------------------------------------------------------------------
# send_thinking hook
# ---------------------------------------------------------------------------


class TestSendThinkingHook:
    async def test_fired_when_reasoning_content_present(self):
        """Agent fires send_thinking when LLM response includes reasoning_content."""
        plugin, channel, db = await build_plugin_and_channel()

        # Mock send_thinking to track calls
        plugin.pm.ahook.send_thinking = AsyncMock()

        # Mock LLM to return reasoning_content
        reasoning_text = "I need to think about this carefully"
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=_make_text_response("Here is my answer", reasoning=reasoning_text)
        )
        plugin._client = mock_client
        plugin._tools = {}
        plugin._tool_schemas = []
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        plugin.pm.ahook.send_thinking.assert_awaited_once_with(
            channel=channel,
            text=reasoning_text,
        )
        await db.close()

    async def test_not_fired_when_no_reasoning_content(self):
        """Agent does not fire send_thinking when LLM response has no reasoning_content."""
        plugin, channel, db = await build_plugin_and_channel()

        plugin.pm.ahook.send_thinking = AsyncMock()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=_make_text_response("Just a plain answer")
        )
        plugin._client = mock_client
        plugin._tools = {}
        plugin._tool_schemas = []
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        plugin.pm.ahook.send_thinking.assert_not_awaited()
        await db.close()

    async def test_failure_does_not_break_agent(self):
        """If send_thinking hook raises, the agent still delivers the response."""
        plugin, channel, db = await build_plugin_and_channel()

        plugin.pm.ahook.send_thinking = AsyncMock(side_effect=RuntimeError("boom"))

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=_make_text_response("My answer", reasoning="some thinking")
        )
        plugin._client = mock_client
        plugin._tools = {}
        plugin._tool_schemas = []
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="hello")
        await drain(plugin, channel)

        # Response should still be delivered via send_message
        plugin.pm.ahook.send_message.assert_awaited_once()
        call_kwargs = plugin.pm.ahook.send_message.call_args[1]
        assert "My answer" in call_kwargs["text"]
        await db.close()


# ---------------------------------------------------------------------------
# send_tool_status hook
# ---------------------------------------------------------------------------


class TestSendToolStatusHook:
    async def test_dispatched_fired_on_tool_call(self):
        """Agent fires send_tool_status(dispatched) when LLM requests a tool call."""
        plugin, channel, db = await build_plugin_and_channel()

        plugin.pm.ahook.send_tool_status = AsyncMock()

        tool_call = _make_tool_call("tc1", "shell", {"command": "echo hi"})

        plugin._client = MagicMock()
        plugin._client.chat = AsyncMock(
            return_value=_make_tool_call_response([tool_call])
        )

        # Register the tool
        async def fake_shell(command, **kwargs):
            return "hi"

        plugin._tools = {"shell": fake_shell}
        plugin._tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="run echo")
        await drain(plugin, channel)
        await drain_task_queue(plugin)

        # Check dispatched was called
        dispatched_calls = [
            c for c in plugin.pm.ahook.send_tool_status.call_args_list
            if c[1].get("status") == "dispatched"
        ]
        assert len(dispatched_calls) == 1
        assert dispatched_calls[0][1]["tool_name"] == "shell"
        assert dispatched_calls[0][1]["args_summary"] is not None
        await db.close()

    async def test_dispatched_failure_does_not_break_agent(self):
        """If send_tool_status raises on dispatch, tool still gets enqueued."""
        plugin, channel, db = await build_plugin_and_channel()

        plugin.pm.ahook.send_tool_status = AsyncMock(side_effect=RuntimeError("boom"))

        tool_call = _make_tool_call("tc1", "shell", {"command": "echo hi"})

        plugin._client = MagicMock()
        plugin._client.chat = AsyncMock(
            return_value=_make_tool_call_response([tool_call])
        )

        async def fake_shell(command, **kwargs):
            return "hi"

        plugin._tools = {"shell": fake_shell}
        plugin._tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="run echo")
        await drain(plugin, channel)
        await drain_task_queue(plugin)

        # Tool should still have been dispatched — task queue should drain
        # without error
        await db.close()

    async def test_completed_fired_on_tool_result(self):
        """send_tool_status(completed) fires when tool result is delivered."""
        plugin, channel, db = await build_plugin_and_channel()

        plugin.pm.ahook.send_tool_status = AsyncMock()

        tool_call = _make_tool_call("tc1", "shell", {"command": "echo hi"})

        # First call: tool call. Second call: text response after tool result.
        plugin._client = MagicMock()
        plugin._client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response([tool_call]),
                _make_text_response("Done!"),
            ]
        )

        async def fake_shell(command, **kwargs):
            return "hi"

        plugin._tools = {"shell": fake_shell}
        plugin._tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="run echo")
        await drain(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)  # drain again for the tool-result turn

        # Check completed was called
        completed_calls = [
            c for c in plugin.pm.ahook.send_tool_status.call_args_list
            if c[1].get("status") == "completed"
        ]
        assert len(completed_calls) == 1
        assert completed_calls[0][1]["tool_name"] == "shell"
        assert completed_calls[0][1]["result_summary"] is not None
        await db.close()
