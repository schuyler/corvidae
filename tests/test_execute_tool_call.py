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
# Tests for tool result truncation (item D — red phase, expected to FAIL)
# ---------------------------------------------------------------------------


class TestToolResultTruncation:
    """Tests for MAX_TOOL_RESULT_CHARS truncation in execute_tool_call.

    These tests MUST FAIL against current code because truncation is not
    implemented yet.
    """

    async def test_result_within_limit_unchanged(self):
        """A tool returning 100 chars returns that result unchanged."""
        from corvidae.tool import MAX_TOOL_RESULT_CHARS  # noqa: F401 — import confirms constant exists

        output = "x" * 100

        async def small_tool() -> str:
            """Returns 100 chars."""
            return output

        result = await execute_tool_call(small_tool, {}, tool_call_id="call_small")
        assert result == output
        assert len(result) == 100

    async def test_result_at_limit_unchanged(self):
        """A tool returning exactly MAX_TOOL_RESULT_CHARS returns that result unchanged."""
        from corvidae.tool import MAX_TOOL_RESULT_CHARS

        output = "y" * MAX_TOOL_RESULT_CHARS

        async def exact_tool() -> str:
            """Returns exactly MAX_TOOL_RESULT_CHARS chars."""
            return output

        result = await execute_tool_call(exact_tool, {}, tool_call_id="call_exact")
        assert result == output
        assert len(result) == MAX_TOOL_RESULT_CHARS

    async def test_result_exceeds_limit_truncated(self):
        """A tool returning 150_000 chars is truncated to MAX_TOOL_RESULT_CHARS."""
        from corvidae.tool import MAX_TOOL_RESULT_CHARS

        output = "z" * 150_000

        async def large_tool() -> str:
            """Returns 150k chars."""
            return output

        result = await execute_tool_call(large_tool, {}, tool_call_id="call_large")
        assert len(result) <= MAX_TOOL_RESULT_CHARS + 200  # allow for truncation notice
        assert result[:MAX_TOOL_RESULT_CHARS] == output[:MAX_TOOL_RESULT_CHARS]
        assert "truncated" in result.lower()

    async def test_truncation_notice_format(self):
        """Truncation notice includes the original length."""
        from corvidae.tool import MAX_TOOL_RESULT_CHARS

        original_len = 150_000
        output = "a" * original_len

        async def big_tool() -> str:
            """Returns 150k chars."""
            return output

        result = await execute_tool_call(big_tool, {}, tool_call_id="call_notice")
        assert str(original_len) in result
