"""Failing tests for extracted helper methods in AgentPlugin (Red TDD).

These tests specify the interface contracts of four methods that do not yet
exist on AgentPlugin:

    _build_conversation_message(self, item) -> tuple[dict, str] | None
    _run_turn(self, channel, messages, tool_schemas, llm_overrides) -> AgentTurnResult | None
    _resolve_display_text(self, channel, result, fallback) -> str
    _handle_response(self, result, channel, max_turns_limit, request_text) -> None

All tests are expected to fail with AttributeError until the helpers are
implemented (Item 2 of corvidae-cleanup.md).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from corvidae.agent import (
    AgentPlugin,
    MAX_TURNS_FALLBACK_MESSAGE,
    QueueItem,
    QueueItemRole,
)
from corvidae.agent_loop import AgentTurnResult

from helpers import build_plugin_and_channel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def plugin_and_channel():
    """Minimal plugin + channel with in-memory DB. Caller handles teardown."""
    plugin, channel, db = await build_plugin_and_channel()
    yield plugin, channel, db
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


def _make_turn_result(text="hello", tool_calls=None):
    """Build an AgentTurnResult for test use."""
    tc = tool_calls or []
    return AgentTurnResult(
        message={"role": "assistant", "content": text, "tool_calls": tc},
        tool_calls=tc,
        text=text,
        latency_ms=42.0,
    )


# ===========================================================================
# _build_conversation_message
# ===========================================================================


class TestBuildConversationMessage:
    """Tests for _build_conversation_message."""

    async def test_user_item_returns_user_message(self, plugin_and_channel):
        """USER role -> {"role": "user", "content": ...}, request_text == content."""
        plugin, channel, db = plugin_and_channel
        item = QueueItem(
            role=QueueItemRole.USER,
            content="hello world",
            channel=channel,
            sender="alice",
        )

        result = plugin._build_conversation_message(item)

        assert result is not None
        msg, request_text = result
        assert msg == {"role": "user", "content": "hello world"}
        assert request_text == "hello world"

    async def test_notification_with_tool_call_id_returns_tool_message(
        self, plugin_and_channel
    ):
        """NOTIFICATION with tool_call_id -> role='tool' message."""
        plugin, channel, db = plugin_and_channel
        item = QueueItem(
            role=QueueItemRole.NOTIFICATION,
            content="tool output",
            channel=channel,
            source="task",
            tool_call_id="call-abc",
        )

        result = plugin._build_conversation_message(item)

        assert result is not None
        msg, request_text = result
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call-abc"
        assert msg["content"] == "tool output"
        assert request_text == "tool output"

    async def test_notification_without_tool_call_id_returns_system_message(
        self, plugin_and_channel
    ):
        """NOTIFICATION without tool_call_id -> role='system' message with source prefix."""
        plugin, channel, db = plugin_and_channel
        item = QueueItem(
            role=QueueItemRole.NOTIFICATION,
            content="something happened",
            channel=channel,
            source="scheduler",
        )

        result = plugin._build_conversation_message(item)

        assert result is not None
        msg, request_text = result
        assert msg["role"] == "system"
        assert "[scheduler]" in msg["content"]
        assert "something happened" in msg["content"]
        assert request_text == "something happened"

    async def test_unknown_role_returns_none(self, plugin_and_channel):
        """Unrecognized role -> None (caller should return early)."""
        plugin, channel, db = plugin_and_channel

        # Construct an item with an invalid role by bypassing the enum
        item = QueueItem(
            role=QueueItemRole.USER,  # will be overridden below
            content="sneaky",
            channel=channel,
        )
        # Bypass enum validation by setting the attribute directly
        object.__setattr__(item, "role", "totally_unknown")

        result = plugin._build_conversation_message(item)

        assert result is None

    async def test_unknown_role_logs_error(self, plugin_and_channel, caplog):
        """Unrecognized role -> error is logged."""
        import logging

        plugin, channel, db = plugin_and_channel
        item = QueueItem(
            role=QueueItemRole.USER,
            content="sneaky",
            channel=channel,
        )
        object.__setattr__(item, "role", "totally_unknown")

        with caplog.at_level(logging.ERROR, logger="corvidae.agent"):
            plugin._build_conversation_message(item)

        assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ===========================================================================
# _run_turn
# ===========================================================================


class TestRunTurn:
    """Tests for _run_turn."""

    async def test_success_returns_agent_turn_result(self, plugin_and_channel):
        """Successful run_agent_turn call -> AgentTurnResult returned."""
        plugin, channel, db = plugin_and_channel

        expected = _make_turn_result(text="success")

        with patch(
            "corvidae.agent.run_agent_turn", new=AsyncMock(return_value=expected)
        ):
            result = await plugin._run_turn(
                channel=channel,
                messages=[{"role": "user", "content": "hi"}],
                tool_schemas=[],
                llm_overrides=None,
            )

        assert result is expected

    async def test_llm_error_returns_none(self, plugin_and_channel):
        """run_agent_turn raises -> _run_turn returns None."""
        plugin, channel, db = plugin_and_channel

        # Provide a mock client so the plugin doesn't blow up before the call
        plugin.client = MagicMock()

        with patch(
            "corvidae.agent.run_agent_turn",
            new=AsyncMock(side_effect=RuntimeError("LLM exploded")),
        ):
            result = await plugin._run_turn(
                channel=channel,
                messages=[{"role": "user", "content": "hi"}],
                tool_schemas=[],
                llm_overrides=None,
            )

        assert result is None

    async def test_llm_error_sends_error_message(self, plugin_and_channel):
        """run_agent_turn raises -> error message sent to channel via send_message hook."""
        plugin, channel, db = plugin_and_channel
        plugin.client = MagicMock()

        with patch(
            "corvidae.agent.run_agent_turn",
            new=AsyncMock(side_effect=RuntimeError("LLM exploded")),
        ):
            await plugin._run_turn(
                channel=channel,
                messages=[{"role": "user", "content": "hi"}],
                tool_schemas=[],
                llm_overrides=None,
            )

        plugin.pm.ahook.send_message.assert_called_once()
        call_kwargs = plugin.pm.ahook.send_message.call_args.kwargs
        assert call_kwargs["channel"] is channel
        # The error message should be a non-empty string
        assert isinstance(call_kwargs["text"], str)
        assert call_kwargs["text"]

    async def test_llm_error_fires_on_llm_error_hook(self, plugin_and_channel):
        """run_agent_turn raises -> on_llm_error hook is called with channel and error."""
        plugin, channel, db = plugin_and_channel
        plugin.client = MagicMock()
        plugin.pm.ahook.on_llm_error = AsyncMock(return_value=[])

        exc = RuntimeError("bad model")
        with patch(
            "corvidae.agent.run_agent_turn",
            new=AsyncMock(side_effect=exc),
        ):
            await plugin._run_turn(
                channel=channel,
                messages=[{"role": "user", "content": "hi"}],
                tool_schemas=[],
                llm_overrides=None,
            )

        plugin.pm.ahook.on_llm_error.assert_called_once()
        call_kwargs = plugin.pm.ahook.on_llm_error.call_args.kwargs
        assert call_kwargs["channel"] is channel
        assert call_kwargs["error"] is exc

    async def test_passes_llm_overrides_to_run_agent_turn(self, plugin_and_channel):
        """llm_overrides dict -> forwarded to run_agent_turn as extra_body."""
        plugin, channel, db = plugin_and_channel
        expected = _make_turn_result()

        with patch(
            "corvidae.agent.run_agent_turn", new=AsyncMock(return_value=expected)
        ) as mock_turn:
            await plugin._run_turn(
                channel=channel,
                messages=[],
                tool_schemas=[],
                llm_overrides={"temperature": 0.7},
            )

        mock_turn.assert_called_once()
        call_kwargs = mock_turn.call_args.kwargs
        # extra_body should contain the overrides (or they may be positional —
        # accept either kwargs["extra_body"] or args[3])
        extra_body = call_kwargs.get("extra_body") or (
            mock_turn.call_args.args[3] if len(mock_turn.call_args.args) > 3 else None
        )
        assert extra_body == {"temperature": 0.7}


# ===========================================================================
# _resolve_display_text
# ===========================================================================


class TestResolveDisplayText:
    """Tests for _resolve_display_text."""

    async def test_hook_returns_value_uses_hook_result(self, plugin_and_channel):
        """transform_display_text returns a value -> that value is used."""
        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="original")
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=["transformed"])

        text = await plugin._resolve_display_text(channel, result, fallback=None)

        assert text == "transformed"

    async def test_hook_returns_none_uses_input_text(self, plugin_and_channel):
        """transform_display_text returns None -> falls back to result.text."""
        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="original text")
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        text = await plugin._resolve_display_text(channel, result, fallback=None)

        assert text == "original text"

    async def test_fallback_used_when_resolved_text_is_empty_and_fallback_given(
        self, plugin_and_channel
    ):
        """Resolved text is empty + fallback provided -> fallback returned."""
        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="")
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        text = await plugin._resolve_display_text(
            channel, result, fallback=MAX_TURNS_FALLBACK_MESSAGE
        )

        assert text == MAX_TURNS_FALLBACK_MESSAGE

    async def test_fallback_none_empty_resolved_text_returned_as_is(
        self, plugin_and_channel
    ):
        """fallback=None + resolved text is empty -> empty string returned as-is."""
        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="")
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        text = await plugin._resolve_display_text(channel, result, fallback=None)

        assert text == ""

    async def test_hook_exception_does_not_propagate(self, plugin_and_channel):
        """transform_display_text raises -> exception caught, falls back to result.text."""
        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="safe text")
        plugin.pm.ahook.transform_display_text = AsyncMock(
            side_effect=RuntimeError("hook crash")
        )

        # Must not raise
        text = await plugin._resolve_display_text(channel, result, fallback=None)

        assert text == "safe text"

    async def test_hook_exception_logs_warning(self, plugin_and_channel, caplog):
        """transform_display_text raises -> warning logged with channel context."""
        import logging

        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="safe text")
        plugin.pm.ahook.transform_display_text = AsyncMock(
            side_effect=RuntimeError("hook crash")
        )

        with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
            await plugin._resolve_display_text(channel, result, fallback=None)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records


# ===========================================================================
# _handle_response
# ===========================================================================


class TestHandleResponse:
    """Tests for _handle_response."""

    async def test_text_response_sends_message(self, plugin_and_channel):
        """No tool calls -> send_message called with the text."""
        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="the answer")

        # Ensure transform_display_text returns nothing so we use result.text
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        await plugin._handle_response(
            result=result,
            channel=channel,
            max_turns_limit=5,
            request_text="the question",
        )

        plugin.pm.ahook.send_message.assert_called_once()
        call_kwargs = plugin.pm.ahook.send_message.call_args.kwargs
        assert "the answer" in call_kwargs["text"]

    async def test_text_response_increments_turn_counter(self, plugin_and_channel):
        """No tool calls -> channel.turn_counter incremented by 1."""
        plugin, channel, db = plugin_and_channel
        channel.turn_counter = 2
        result = _make_turn_result(text="done")
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        await plugin._handle_response(
            result=result,
            channel=channel,
            max_turns_limit=5,
            request_text="go",
        )

        assert channel.turn_counter == 3

    async def test_text_response_fires_on_agent_response(self, plugin_and_channel):
        """No tool calls -> on_agent_response hook fired with request and response text."""
        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="the answer")
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        await plugin._handle_response(
            result=result,
            channel=channel,
            max_turns_limit=5,
            request_text="the question",
        )

        plugin.pm.ahook.on_agent_response.assert_called_once()
        call_kwargs = plugin.pm.ahook.on_agent_response.call_args.kwargs
        assert call_kwargs["channel"] is channel
        assert call_kwargs["request_text"] == "the question"
        assert "the answer" in call_kwargs["response_text"]

    async def test_tool_calls_under_limit_dispatched_no_send_message(
        self, plugin_and_channel
    ):
        """Tool calls + counter < max_turns -> dispatch, NO send_message."""
        plugin, channel, db = plugin_and_channel
        channel.turn_counter = 0
        tool_call = {
            "id": "call-001",
            "function": {"name": "my_tool", "arguments": '{"x": "v"}'},
        }
        result = _make_turn_result(text="", tool_calls=[tool_call])

        async def my_tool(x: str) -> str:
            """A tool."""
            return "done"

        plugin.tools = {"my_tool": my_tool}

        await plugin._handle_response(
            result=result,
            channel=channel,
            max_turns_limit=5,
            request_text="use the tool",
        )

        # send_message must NOT be called when tool calls are dispatched
        plugin.pm.ahook.send_message.assert_not_called()

    async def test_tool_calls_under_limit_increment_turn_counter(
        self, plugin_and_channel
    ):
        """Tool calls + counter < max_turns -> turn_counter incremented."""
        plugin, channel, db = plugin_and_channel
        channel.turn_counter = 1
        tool_call = {
            "id": "call-002",
            "function": {"name": "my_tool", "arguments": '{"x": "v"}'},
        }
        result = _make_turn_result(text="", tool_calls=[tool_call])

        async def my_tool(x: str) -> str:
            """A tool."""
            return "done"

        plugin.tools = {"my_tool": my_tool}

        await plugin._handle_response(
            result=result,
            channel=channel,
            max_turns_limit=5,
            request_text="use the tool",
        )

        assert channel.turn_counter == 2

    async def test_tool_calls_at_max_turns_sends_fallback_text(
        self, plugin_and_channel
    ):
        """Tool calls + counter == max_turns -> fallback text sent, no dispatch."""
        plugin, channel, db = plugin_and_channel
        channel.turn_counter = 3  # already at the limit
        tool_call = {
            "id": "call-003",
            "function": {"name": "my_tool", "arguments": '{"x": "v"}'},
        }
        result = _make_turn_result(text="", tool_calls=[tool_call])
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        enqueued_tasks = []
        task_plugin = plugin.pm.get_plugin("task")
        original_enqueue = task_plugin.task_queue.enqueue

        async def spy_enqueue(task):
            enqueued_tasks.append(task)
            await original_enqueue(task)

        task_plugin.task_queue.enqueue = spy_enqueue

        await plugin._handle_response(
            result=result,
            channel=channel,
            max_turns_limit=3,  # limit == counter -> at max
            request_text="use the tool",
        )

        # No tasks dispatched
        assert not enqueued_tasks
        # Fallback message sent
        plugin.pm.ahook.send_message.assert_called_once()
        sent_text = plugin.pm.ahook.send_message.call_args.kwargs["text"]
        assert "max tool-calling rounds reached" in sent_text

    async def test_tool_calls_at_max_turns_does_not_increment_counter(
        self, plugin_and_channel
    ):
        """Tool calls at max_turns -> counter NOT incremented (dispatch skipped)."""
        plugin, channel, db = plugin_and_channel
        channel.turn_counter = 3
        tool_call = {
            "id": "call-004",
            "function": {"name": "my_tool", "arguments": '{"x": "v"}'},
        }
        result = _make_turn_result(text="", tool_calls=[tool_call])
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        await plugin._handle_response(
            result=result,
            channel=channel,
            max_turns_limit=3,
            request_text="use the tool",
        )

        # Counter stays at 3 — not incremented because dispatch was suppressed
        assert channel.turn_counter == 3

    async def test_latency_ms_passed_to_send_message(self, plugin_and_channel):
        """send_message receives latency_ms from the AgentTurnResult."""
        plugin, channel, db = plugin_and_channel
        result = _make_turn_result(text="done")
        result = AgentTurnResult(
            message={"role": "assistant", "content": "done"},
            tool_calls=[],
            text="done",
            latency_ms=123.4,
        )
        plugin.pm.ahook.transform_display_text = AsyncMock(return_value=[])

        await plugin._handle_response(
            result=result,
            channel=channel,
            max_turns_limit=5,
            request_text="go",
        )

        plugin.pm.ahook.send_message.assert_called_once()
        call_kwargs = plugin.pm.ahook.send_message.call_args.kwargs
        assert call_kwargs.get("latency_ms") == 123.4
