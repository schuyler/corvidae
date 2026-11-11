"""Tests for execute_tool_call() in corvidae/tool.py.

These tests are expected to FAIL with ImportError until execute_tool_call
is implemented.
"""

import pytest

from corvidae.tool import execute_tool_call, ToolContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _plain_tool(x: str, y: int) -> str:
    """A plain async tool with no _ctx."""
    return f"{x}-{y}"


async def _ctx_tool(x: str, _ctx: ToolContext) -> str:
    """A tool that declares _ctx."""
    return f"{x}-ctx"


async def _int_tool(n: int) -> int:
    """A tool that returns an int."""
    return n * 2


async def _raising_tool(message: str) -> str:
    """A tool that always raises ValueError."""
    raise ValueError(message)


async def _ctx_only_tool(_ctx: ToolContext) -> str:
    """A tool with only _ctx and no regular args."""
    return "ctx-only"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_execute_tool_call_basic():
    """Plain async tool with args and no _ctx returns str(result)."""
    result = await execute_tool_call(
        _plain_tool,
        {"x": "hello", "y": 3},
        tool_call_id="call_1",
    )
    assert result == "hello-3"


async def test_execute_tool_call_injects_ctx():
    """Tool declaring _ctx receives a ToolContext with correct fields."""
    from unittest.mock import MagicMock
    from corvidae.task import TaskQueue
    from corvidae.channel import Channel, ChannelConfig

    channel = Channel(transport="test", scope="s1", config=ChannelConfig())
    task_queue = TaskQueue()

    received: list[ToolContext] = []

    async def capturing_tool(name: str, _ctx: ToolContext) -> str:
        received.append(_ctx)
        return name

    result = await execute_tool_call(
        capturing_tool,
        {"name": "alice"},
        channel=channel,
        tool_call_id="call_ctx_42",
        task_queue=task_queue,
    )

    assert len(received) == 1
    ctx = received[0]
    assert isinstance(ctx, ToolContext)
    assert ctx.channel is channel
    assert ctx.tool_call_id == "call_ctx_42"
    assert ctx.task_queue is task_queue


async def test_execute_tool_call_no_ctx_when_not_declared():
    """Tool without _ctx param is called without any ToolContext argument."""
    call_kwargs: list[dict] = []

    async def spy_tool(a: str, b: str) -> str:
        call_kwargs.append({"a": a, "b": b})
        return a + b

    from corvidae.channel import Channel, ChannelConfig
    channel = Channel(transport="test", scope="s2", config=ChannelConfig())

    await execute_tool_call(
        spy_tool,
        {"a": "foo", "b": "bar"},
        channel=channel,
        tool_call_id="call_no_ctx",
    )

    assert call_kwargs == [{"a": "foo", "b": "bar"}]


async def test_execute_tool_call_returns_string():
    """When the tool returns a non-str value, execute_tool_call returns str() of it."""
    result = await execute_tool_call(
        _int_tool,
        {"n": 7},
        tool_call_id="call_int",
    )
    assert result == "14"
    assert isinstance(result, str)


async def test_execute_tool_call_propagates_exception():
    """Exceptions raised by the tool propagate out of execute_tool_call."""
    with pytest.raises(ValueError, match="boom"):
        await execute_tool_call(
            _raising_tool,
            {"message": "boom"},
            tool_call_id="call_exc",
        )


async def test_execute_tool_call_ctx_defaults_none():
    """When channel and task_queue are omitted, ToolContext has None for those fields."""
    received: list[ToolContext] = []

    async def capturing_tool(_ctx: ToolContext) -> str:
        received.append(_ctx)
        return "ok"

    await execute_tool_call(
        capturing_tool,
        {},
        tool_call_id="call_defaults",
    )

    assert len(received) == 1
    ctx = received[0]
    assert ctx.channel is None
    assert ctx.task_queue is None
    assert ctx.tool_call_id == "call_defaults"


# ---------------------------------------------------------------------------
# execute_tool_call returns the whole result: the Agent cuts it at
# tools.max_result_chars when it is delivered, and spills the rest (#575)
# ---------------------------------------------------------------------------


class TestToolResultNotCutAtDispatch:
    """The result string leaves execute_tool_call whole."""

    async def test_large_result_is_returned_whole(self):
        """A 150k-character result is returned unchanged, with no marker."""
        output = "z" * 150_000

        async def large_tool() -> str:
            """Returns 150k chars."""
            return output

        result = await execute_tool_call(large_tool, {}, tool_call_id="call_large")
        assert result == output

    def test_execute_tool_call_takes_no_cap(self):
        """The cap is not execute_tool_call's: it has no max_result_chars."""
        import inspect

        assert "max_result_chars" not in inspect.signature(execute_tool_call).parameters


# ---------------------------------------------------------------------------
# Tests for async enforcement (item 3)
# ---------------------------------------------------------------------------


def test_from_function_rejects_sync_function():
    from corvidae.tool import Tool

    def sync_fn(x: str) -> str:
        """A sync tool."""
        return x

    with pytest.raises(TypeError, match="must be async"):
        Tool.from_function(sync_fn)


async def test_execute_tool_call_wraps_sync_callable():
    """execute_tool_call wraps a sync callable in asyncio.to_thread."""

    def sync_tool(x: str) -> str:
        return f"sync-{x}"

    result = await execute_tool_call(
        sync_tool,
        {"x": "hello"},
        tool_call_id="call_sync",
    )
    assert result == "sync-hello"
