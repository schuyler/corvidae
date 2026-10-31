"""Tests for the agent sending OutboundThinking and OutboundToolStatus events."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.trace import mint_trace_id
from corvidae.turn import AgentTurnResult
from corvidae.channel import ChannelConfig, ChannelRegistry
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.hooks import create_plugin_manager
from corvidae.task import TaskPlugin

from corvidae.outbound import OutboundMessage, OutboundThinking, OutboundToolStatus
from helpers import build_plugin_and_channel, drain, drain_task_queue, sent
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call_response,
    _make_tool_call,
)


def _raise_for(kind: type, recorded: list) -> "AsyncMock":
    """Build a ``send`` mock that raises for events of ``kind``.

    Events of other kinds are recorded in ``recorded`` and succeed, so a
    test can check that one failing event kind does not stop the others.
    """

    # Side effect: raise on the chosen event kind, record the rest.
    async def _send(channel, event):
        if isinstance(event, kind):
            raise RuntimeError("boom")
        recorded.append((channel, event))

    return AsyncMock(side_effect=_send)


# ---------------------------------------------------------------------------
# OutboundThinking event
# ---------------------------------------------------------------------------


class TestSendThinkingHook:
    async def test_fired_when_reasoning_content_present(self):
        """Agent sends OutboundThinking when LLM response includes reasoning_content."""
        plugin, channel, db = await build_plugin_and_channel()

        # Mock LLM to return reasoning_content
        reasoning_text = "I need to think about this carefully"
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=_make_text_response("Here is my answer", reasoning=reasoning_text)
        )
        plugin._client = mock_client
        plugin._mark_started()
        plugin._tools = {}
        plugin._tool_schemas = []
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert sent(plugin.pm, OutboundThinking) == [
            (channel, OutboundThinking(text=reasoning_text)),
        ]
        await db.close()

    async def test_not_fired_when_no_reasoning_content(self):
        """Agent does not send OutboundThinking when LLM response has no reasoning_content."""
        plugin, channel, db = await build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=_make_text_response("Just a plain answer")
        )
        plugin._client = mock_client
        plugin._mark_started()
        plugin._tools = {}
        plugin._tool_schemas = []
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert sent(plugin.pm, OutboundThinking) == []
        await db.close()

    async def test_failure_does_not_break_agent(self):
        """If sending OutboundThinking raises, the agent still delivers the response."""
        plugin, channel, db = await build_plugin_and_channel()

        delivered: list = []
        plugin.pm.ahook.send = _raise_for(OutboundThinking, delivered)

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=_make_text_response("My answer", reasoning="some thinking")
        )
        plugin._client = mock_client
        plugin._mark_started()
        plugin._tools = {}
        plugin._tool_schemas = []
        plugin._max_tool_result_chars = 10000

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        await drain(plugin, channel)

        # Response should still be delivered as an OutboundMessage
        messages = [e for _c, e in delivered if isinstance(e, OutboundMessage)]
        assert len(messages) == 1
        assert "My answer" in messages[0].text
        await db.close()


# ---------------------------------------------------------------------------
# OutboundToolStatus event
# ---------------------------------------------------------------------------


class TestSendToolStatusHook:
    async def test_dispatched_fired_on_tool_call(self):
        """Agent sends OutboundToolStatus(dispatched) when LLM requests a tool call."""
        plugin, channel, db = await build_plugin_and_channel()

        tool_call = _make_tool_call("tc1", "shell", {"command": "echo hi"})

        plugin._client = MagicMock()
        plugin._mark_started()
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

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="run echo", trace_id=mint_trace_id())
        await drain(plugin, channel)
        await drain_task_queue(plugin)

        # Check dispatched was called
        dispatched = [
            e for _c, e in sent(plugin.pm, OutboundToolStatus)
            if e.status == "dispatched"
        ]
        assert len(dispatched) == 1
        assert dispatched[0].tool_name == "shell"
        assert dispatched[0].args_summary is not None
        # The dispatch event carries the LLM's id for the call.
        assert dispatched[0].tool_call_id == "tc1"
        await db.close()

    async def test_dispatched_failure_does_not_break_agent(self):
        """If sending OutboundToolStatus raises on dispatch, tool still gets enqueued."""
        plugin, channel, db = await build_plugin_and_channel()

        plugin.pm.ahook.send = _raise_for(OutboundToolStatus, [])

        tool_call = _make_tool_call("tc1", "shell", {"command": "echo hi"})

        plugin._client = MagicMock()
        plugin._mark_started()
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

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="run echo", trace_id=mint_trace_id())
        await drain(plugin, channel)
        await drain_task_queue(plugin)

        # Tool should still have been dispatched — task queue should drain
        # without error
        await db.close()

    async def test_completed_fired_on_tool_result(self):
        """OutboundToolStatus(completed) is sent when tool result is delivered."""
        plugin, channel, db = await build_plugin_and_channel()

        tool_call = _make_tool_call("tc1", "shell", {"command": "echo hi"})

        # First call: tool call. Second call: text response after tool result.
        plugin._client = MagicMock()
        plugin._mark_started()
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

        await plugin.pm.ahook.on_message(channel=channel, sender="user", text="run echo", trace_id=mint_trace_id())
        await drain(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)  # drain again for the tool-result turn

        # Check completed was called
        completed = [
            e for _c, e in sent(plugin.pm, OutboundToolStatus)
            if e.status == "completed"
        ]
        assert len(completed) == 1
        assert completed[0].tool_name == "shell"
        assert completed[0].result_summary is not None
        # The completion event pairs with the dispatch via the LLM call id.
        assert completed[0].tool_call_id == "tc1"
        await db.close()
