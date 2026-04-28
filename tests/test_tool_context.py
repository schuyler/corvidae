"""Tests for ToolContext injection in run_agent_loop."""

import json
from unittest.mock import AsyncMock, MagicMock

from corvidae.agent_loop import run_agent_loop
from corvidae.tool import tool_to_schema
from corvidae.tool import ToolContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call_response(calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": calls,
                }
            }
        ]
    }


def _make_tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


# ---------------------------------------------------------------------------
# ToolContext injection tests
# ---------------------------------------------------------------------------


async def test_ctx_injected_when_declared():
    """A tool declaring _ctx receives a ToolContext instance."""
    received_ctx = []

    async def ctx_tool(_ctx: ToolContext) -> str:
        """A tool that accepts context."""
        received_ctx.append(_ctx)
        return "ok"

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_ctx", "ctx_tool", {})]
            ),
            {"choices": [{"message": {"content": "done"}}]},
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    await run_agent_loop(
        client,
        messages,
        tools={"ctx_tool": ctx_tool},
        tool_schemas=[],
    )

    assert len(received_ctx) == 1
    assert isinstance(received_ctx[0], ToolContext)


async def test_ctx_not_injected_when_not_declared():
    """A tool without _ctx is called without any ToolContext argument."""
    call_kwargs_received = []

    async def plain_tool(x: str) -> str:
        """A plain tool without context."""
        call_kwargs_received.append({"x": x})
        return "ok"

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_plain", "plain_tool", {"x": "hello"})]
            ),
            {"choices": [{"message": {"content": "done"}}]},
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    await run_agent_loop(
        client,
        messages,
        tools={"plain_tool": plain_tool},
        tool_schemas=[],
    )

    assert call_kwargs_received == [{"x": "hello"}]


async def test_ctx_has_correct_tool_call_id():
    """The ToolContext injected contains the LLM-assigned tool call ID."""
    received_ctx = []

    async def ctx_tool(_ctx: ToolContext) -> str:
        """Tool that captures context."""
        received_ctx.append(_ctx)
        return "ok"

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("my_call_id_42", "ctx_tool", {})]
            ),
            {"choices": [{"message": {"content": "done"}}]},
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    await run_agent_loop(
        client,
        messages,
        tools={"ctx_tool": ctx_tool},
        tool_schemas=[],
    )

    assert received_ctx[0].tool_call_id == "my_call_id_42"


async def test_ctx_has_correct_channel():
    """The ToolContext injected contains the channel passed to run_agent_loop."""
    from corvidae.channel import Channel, ChannelConfig

    received_ctx = []

    async def ctx_tool(_ctx: ToolContext) -> str:
        """Tool that captures context."""
        received_ctx.append(_ctx)
        return "ok"

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_ch", "ctx_tool", {})]
            ),
            {"choices": [{"message": {"content": "done"}}]},
        ]
    )

    channel = Channel(transport="test", scope="scope1", config=ChannelConfig())
    messages = [{"role": "user", "content": "go"}]
    await run_agent_loop(
        client,
        messages,
        tools={"ctx_tool": ctx_tool},
        tool_schemas=[],
        channel=channel,
    )

    assert received_ctx[0].channel is channel


async def test_ctx_has_correct_task_queue():
    """The ToolContext injected contains the task_queue passed to run_agent_loop."""
    from corvidae.task import TaskQueue

    received_ctx = []

    async def ctx_tool(_ctx: ToolContext) -> str:
        """Tool that captures context."""
        received_ctx.append(_ctx)
        return "ok"

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_tq", "ctx_tool", {})]
            ),
            {"choices": [{"message": {"content": "done"}}]},
        ]
    )

    task_queue = TaskQueue()
    messages = [{"role": "user", "content": "go"}]
    await run_agent_loop(
        client,
        messages,
        tools={"ctx_tool": ctx_tool},
        tool_schemas=[],
        task_queue=task_queue,
    )

    assert received_ctx[0].task_queue is task_queue


async def test_ctx_channel_and_task_queue_none_when_not_provided():
    """When run_agent_loop is called without channel/task_queue, those fields
    are None in the injected ToolContext. tool_call_id is always set."""
    received_ctx = []

    async def ctx_tool(_ctx: ToolContext) -> str:
        """Tool that captures context."""
        received_ctx.append(_ctx)
        return "ok"

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_none", "ctx_tool", {})]
            ),
            {"choices": [{"message": {"content": "done"}}]},
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    await run_agent_loop(
        client,
        messages,
        tools={"ctx_tool": ctx_tool},
        tool_schemas=[],
        # channel and task_queue intentionally omitted
    )

    ctx = received_ctx[0]
    assert ctx.channel is None
    assert ctx.task_queue is None
    assert ctx.tool_call_id == "call_none"


async def test_ctx_excluded_from_schema():
    """_ctx does not appear in the tool schema generated by tool_to_schema."""

    async def ctx_tool(name: str, _ctx: ToolContext) -> str:
        """A tool with a context parameter."""
        return name

    schema = tool_to_schema(ctx_tool)
    params = schema["function"]["parameters"]
    properties = params.get("properties", {})
    assert "_ctx" not in properties, "_ctx must be excluded from the tool schema"
    assert "name" in properties


