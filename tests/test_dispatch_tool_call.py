"""Tests for dispatch_tool_call and ToolCallResult in corvidae.tool."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.tool import dispatch_tool_call, ToolCallResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def _make_call_raw(call_id: str, name: str, raw_args: str) -> dict:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": raw_args,
        },
    }


# ---------------------------------------------------------------------------
# 1. Successful execution — tool returns a result string
# ---------------------------------------------------------------------------


async def test_successful_execution_returns_tool_result():
    """dispatch_tool_call returns a ToolCallResult with the tool's return value."""
    tool_fn = AsyncMock(return_value="hello from tool")
    call = _make_call("call_1", "my_tool", {"x": "value"})

    result = await dispatch_tool_call(call, {"my_tool": tool_fn})

    assert isinstance(result, ToolCallResult)
    assert result.content == "hello from tool"
    assert result.error is False
    assert result.tool_call_id == "call_1"
    assert result.tool_name == "my_tool"


async def test_successful_execution_calls_tool_with_parsed_args():
    """dispatch_tool_call passes parsed JSON args to the tool function."""
    tool_fn = AsyncMock(return_value="ok")
    call = _make_call("call_2", "my_tool", {"a": 1, "b": "two"})

    await dispatch_tool_call(call, {"my_tool": tool_fn})

    tool_fn.assert_awaited_once_with(a=1, b="two")


# ---------------------------------------------------------------------------
# 2. JSON parse error — malformed arguments
# ---------------------------------------------------------------------------


async def test_json_parse_error_returns_error_result():
    """Malformed JSON in tool call arguments returns an error ToolCallResult."""
    tool_fn = AsyncMock(return_value="should not be called")
    call = _make_call_raw("call_bad", "my_tool", "{not valid json")

    result = await dispatch_tool_call(call, {"my_tool": tool_fn})

    assert isinstance(result, ToolCallResult)
    assert result.error is True
    assert "malformed arguments" in result.content
    assert "my_tool" in result.content


async def test_json_parse_error_does_not_call_tool():
    """When JSON parsing fails, the tool function is not invoked."""
    tool_fn = AsyncMock(return_value="should not be called")
    call = _make_call_raw("call_bad", "my_tool", "{not valid json")

    await dispatch_tool_call(call, {"my_tool": tool_fn})

    tool_fn.assert_not_awaited()


async def test_json_parse_error_preserves_call_id_and_name():
    """ToolCallResult includes tool_call_id and tool_name even on JSON parse error."""
    call = _make_call_raw("id_xyz", "some_tool", "{broken")

    result = await dispatch_tool_call(call, {"some_tool": AsyncMock()})

    assert result.tool_call_id == "id_xyz"
    assert result.tool_name == "some_tool"


# ---------------------------------------------------------------------------
# 3. Unknown tool — fn_name not in tools dict
# ---------------------------------------------------------------------------


async def test_unknown_tool_returns_error_result():
    """Unknown tool name returns an error ToolCallResult."""
    call = _make_call("call_ghost", "nonexistent_tool", {})

    result = await dispatch_tool_call(call, {})

    assert isinstance(result, ToolCallResult)
    assert result.error is True
    assert "unknown tool" in result.content
    assert "nonexistent_tool" in result.content


async def test_unknown_tool_latency_ms_is_none():
    """Latency is None for pre-dispatch errors (unknown tool)."""
    call = _make_call("call_ghost", "ghost_tool", {})

    result = await dispatch_tool_call(call, {})

    assert result.latency_ms is None


async def test_unknown_tool_preserves_call_id_and_name():
    """ToolCallResult includes tool_call_id and tool_name on unknown tool error."""
    call = _make_call("id_abc", "missing_tool", {})

    result = await dispatch_tool_call(call, {})

    assert result.tool_call_id == "id_abc"
    assert result.tool_name == "missing_tool"


# ---------------------------------------------------------------------------
# 4. Tool exception — execute_tool_call raises
# ---------------------------------------------------------------------------


async def test_tool_exception_returns_error_result():
    """When the tool raises an exception, an error ToolCallResult is returned."""
    async def bad_tool(**kwargs):
        raise ValueError("something went wrong")

    call = _make_call("call_err", "bad_tool", {})

    result = await dispatch_tool_call(call, {"bad_tool": bad_tool})

    assert isinstance(result, ToolCallResult)
    assert result.error is True
    assert "bad_tool" in result.content


async def test_tool_exception_error_content_starts_with_error():
    """Tool exception error message starts with 'Error:'."""
    async def bad_tool(**kwargs):
        raise RuntimeError("boom")

    call = _make_call("call_err", "bad_tool", {})

    result = await dispatch_tool_call(call, {"bad_tool": bad_tool})

    assert result.content.startswith("Error:")


async def test_tool_exception_latency_ms_is_set():
    """Latency is measured even when the tool raises (execute_tool_call was invoked)."""
    async def bad_tool(**kwargs):
        raise RuntimeError("boom")

    call = _make_call("call_err", "bad_tool", {})

    result = await dispatch_tool_call(call, {"bad_tool": bad_tool})

    assert result.latency_ms is not None
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# 5. process_tool_result hook fires on successful execution
# ---------------------------------------------------------------------------


async def test_hook_fires_on_success():
    """process_tool_result hook is called when the tool executes successfully."""
    from corvidae.hooks import create_plugin_manager, hookimpl

    tool_fn = AsyncMock(return_value="raw result")
    call = _make_call("call_hook", "my_tool", {})

    pm = create_plugin_manager()
    hook_calls = []

    class RecordingPlugin:
        @hookimpl
        async def process_tool_result(self, tool_name, result, channel):
            hook_calls.append({"tool_name": tool_name, "result": result})
            return None

    pm.register(RecordingPlugin(), name="recorder")

    await dispatch_tool_call(call, {"my_tool": tool_fn}, pm=pm)

    assert len(hook_calls) == 1
    assert hook_calls[0]["tool_name"] == "my_tool"
    # The tool returned a plain string; str() on a str is a no-op, so the
    # value passed to the hook is identical to what the tool returned.
    assert hook_calls[0]["result"] == "raw result"


async def test_hook_replacement_used_on_success():
    """When process_tool_result returns a string, it replaces the tool result content."""
    from corvidae.hooks import create_plugin_manager, hookimpl

    tool_fn = AsyncMock(return_value="original output")
    call = _make_call("call_hook", "my_tool", {})

    pm = create_plugin_manager()

    class TransformPlugin:
        @hookimpl
        async def process_tool_result(self, tool_name, result, channel):
            return "transformed: " + result

    pm.register(TransformPlugin(), name="transform")

    result = await dispatch_tool_call(call, {"my_tool": tool_fn}, pm=pm)

    assert result.content == "transformed: original output"
    assert result.error is False


async def test_hook_none_return_keeps_original_result():
    """When process_tool_result returns None, the original content is kept."""
    from corvidae.hooks import create_plugin_manager, hookimpl

    tool_fn = AsyncMock(return_value="keep this")
    call = _make_call("call_hook", "my_tool", {})

    pm = create_plugin_manager()

    class PassthroughPlugin:
        @hookimpl
        async def process_tool_result(self, tool_name, result, channel):
            return None

    pm.register(PassthroughPlugin(), name="passthrough")

    result = await dispatch_tool_call(call, {"my_tool": tool_fn}, pm=pm)

    assert result.content == "keep this"


# ---------------------------------------------------------------------------
# 6. process_tool_result hook fires on tool exception
# ---------------------------------------------------------------------------


async def test_hook_fires_on_tool_exception():
    """process_tool_result hook is called even when the tool raises an exception."""
    from corvidae.hooks import create_plugin_manager, hookimpl

    async def bad_tool(**kwargs):
        raise RuntimeError("deliberate failure")

    call = _make_call("call_err", "bad_tool", {})

    pm = create_plugin_manager()
    hook_calls = []

    class RecordingPlugin:
        @hookimpl
        async def process_tool_result(self, tool_name, result, channel):
            hook_calls.append({"tool_name": tool_name, "result": result})
            return None

    pm.register(RecordingPlugin(), name="recorder")

    await dispatch_tool_call(call, {"bad_tool": bad_tool}, pm=pm)

    assert len(hook_calls) == 1
    assert hook_calls[0]["tool_name"] == "bad_tool"


# ---------------------------------------------------------------------------
# 7. process_tool_result hook does NOT fire on JSON parse error
# ---------------------------------------------------------------------------


async def test_hook_does_not_fire_on_json_parse_error():
    """process_tool_result hook is NOT called for JSON parse errors."""
    from corvidae.hooks import create_plugin_manager, hookimpl

    call = _make_call_raw("call_bad", "my_tool", "{not valid json")

    pm = create_plugin_manager()
    hook_calls = []

    class RecordingPlugin:
        @hookimpl
        async def process_tool_result(self, tool_name, result, channel):
            hook_calls.append(True)
            return None

    pm.register(RecordingPlugin(), name="recorder")

    await dispatch_tool_call(call, {"my_tool": AsyncMock()}, pm=pm)

    assert hook_calls == [], "Hook must not fire for JSON parse errors"


# ---------------------------------------------------------------------------
# 8. process_tool_result hook does NOT fire on unknown tool
# ---------------------------------------------------------------------------


async def test_hook_does_not_fire_on_unknown_tool():
    """process_tool_result hook is NOT called for unknown tool errors."""
    from corvidae.hooks import create_plugin_manager, hookimpl

    call = _make_call("call_ghost", "nonexistent_tool", {})

    pm = create_plugin_manager()
    hook_calls = []

    class RecordingPlugin:
        @hookimpl
        async def process_tool_result(self, tool_name, result, channel):
            hook_calls.append(True)
            return None

    pm.register(RecordingPlugin(), name="recorder")

    await dispatch_tool_call(call, {}, pm=pm)

    assert hook_calls == [], "Hook must not fire for unknown tool errors"


# ---------------------------------------------------------------------------
# 9. Latency measurement
# ---------------------------------------------------------------------------


async def test_latency_ms_set_on_successful_execution():
    """latency_ms is a non-negative float when the tool executes successfully."""
    tool_fn = AsyncMock(return_value="result")
    call = _make_call("call_lat", "my_tool", {})

    result = await dispatch_tool_call(call, {"my_tool": tool_fn})

    assert result.latency_ms is not None
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0


async def test_latency_ms_none_for_json_parse_error():
    """latency_ms is None when the tool is never invoked due to JSON parse error."""
    call = _make_call_raw("call_bad", "my_tool", "not json at all")

    result = await dispatch_tool_call(call, {"my_tool": AsyncMock()})

    assert result.latency_ms is None


async def test_latency_ms_none_for_unknown_tool():
    """latency_ms is None when the tool is never invoked due to unknown tool name."""
    call = _make_call("call_ghost", "ghost_tool", {})

    result = await dispatch_tool_call(call, {})

    assert result.latency_ms is None


# ---------------------------------------------------------------------------
# 10. Logging records
# ---------------------------------------------------------------------------


async def test_successful_tool_logs_info_dispatched(caplog):
    """INFO 'tool call dispatched' is logged with fn_name and arg_keys on success."""
    tool_fn = AsyncMock(return_value="result")
    call = _make_call("call_log", "my_tool", {"x": "v"})

    with caplog.at_level(logging.INFO, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"my_tool": tool_fn})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [r for r in records if r.getMessage() == "tool call dispatched"]
    assert matching, "Expected INFO 'tool call dispatched' record from corvidae.tool"
    rec = matching[0]
    assert hasattr(rec, "arg_keys"), "'tool call dispatched' log must include arg_keys"
    assert rec.arg_keys == ["x"]


async def test_successful_tool_logs_info_result(caplog):
    """INFO 'tool call result' is logged with result_length and latency_ms on success."""
    tool_fn = AsyncMock(return_value="tool output")
    call = _make_call("call_log", "my_tool", {})

    with caplog.at_level(logging.INFO, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"my_tool": tool_fn})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [r for r in records if r.getMessage() == "tool call result"]
    assert matching, "Expected INFO 'tool call result' record from corvidae.tool"
    rec = matching[0]
    assert hasattr(rec, "latency_ms"), "'tool call result' log must have latency_ms"
    assert isinstance(rec.latency_ms, (int, float)), "latency_ms must be numeric"
    assert hasattr(rec, "result_length"), "'tool call result' log must have result_length"


async def test_successful_tool_logs_debug_arguments(caplog):
    """DEBUG 'tool call arguments' is logged with tool name and truncated args JSON."""
    tool_fn = AsyncMock(return_value="result")
    call = _make_call("call_log", "my_tool", {"key": "value"})

    with caplog.at_level(logging.DEBUG, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"my_tool": tool_fn})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [r for r in records if r.getMessage() == "tool call arguments"]
    assert matching, "Expected DEBUG 'tool call arguments' record from corvidae.tool"
    rec = matching[0]
    assert hasattr(rec, "tool"), "'tool call arguments' must have tool attribute"
    assert rec.tool == "my_tool"
    assert hasattr(rec, "arguments"), "'tool call arguments' must have arguments attribute"
    assert isinstance(rec.arguments, str)


async def test_successful_tool_logs_debug_result_content(caplog):
    """DEBUG 'tool call result content' is logged with tool name and content on success."""
    tool_fn = AsyncMock(return_value="my result")
    call = _make_call("call_log", "my_tool", {})

    with caplog.at_level(logging.DEBUG, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"my_tool": tool_fn})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [r for r in records if r.getMessage() == "tool call result content"]
    assert matching, "Expected DEBUG 'tool call result content' record from corvidae.tool"
    rec = matching[0]
    assert hasattr(rec, "tool"), "'tool call result content' must have tool attribute"
    assert rec.tool == "my_tool"
    assert hasattr(rec, "content"), "'tool call result content' must have content attribute"


async def test_json_parse_error_logs_warning(caplog):
    """WARNING is logged when tool call arguments are malformed JSON."""
    call = _make_call_raw("call_bad", "my_tool", "{bad json")

    with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"my_tool": AsyncMock()})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    warning_records = [r for r in records if r.levelno == logging.WARNING]
    assert warning_records, "Expected WARNING record from corvidae.tool for malformed JSON"


async def test_unknown_tool_logs_warning(caplog):
    """WARNING is logged when an unknown tool name is encountered."""
    call = _make_call("call_ghost", "nonexistent_tool", {})

    with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
        await dispatch_tool_call(call, {})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    warning_records = [r for r in records if r.levelno == logging.WARNING]
    assert warning_records, "Expected WARNING record from corvidae.tool for unknown tool"


async def test_tool_exception_logs_warning_with_exc_info(caplog):
    """WARNING with exc_info is logged when the tool raises an exception."""
    async def bad_tool(**kwargs):
        raise RuntimeError("failure")

    call = _make_call("call_err", "bad_tool", {})

    with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"bad_tool": bad_tool})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    warning_records = [r for r in records if r.levelno == logging.WARNING]
    assert warning_records, "Expected WARNING record from corvidae.tool for tool exception"
    # exc_info should be set (tuple of (type, value, traceback))
    assert warning_records[0].exc_info is not None


async def test_tool_exception_does_not_log_result_info(caplog):
    """INFO 'tool call result' is NOT logged when the tool raises an exception."""
    async def bad_tool(**kwargs):
        raise RuntimeError("boom")

    call = _make_call("call_err", "bad_tool", {})

    with caplog.at_level(logging.INFO, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"bad_tool": bad_tool})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    result_records = [r for r in records if r.getMessage() == "tool call result"]
    assert not result_records, "'tool call result' must NOT be logged when tool raises"


async def test_tool_exception_does_not_log_result_content_debug(caplog):
    """DEBUG 'tool call result content' is NOT logged when the tool raises."""
    async def bad_tool(**kwargs):
        raise RuntimeError("boom")

    call = _make_call("call_err", "bad_tool", {})

    with caplog.at_level(logging.DEBUG, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"bad_tool": bad_tool})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    content_records = [r for r in records if r.getMessage() == "tool call result content"]
    assert not content_records, "'tool call result content' must NOT be logged when tool raises"


async def test_tool_exception_still_logs_dispatched(caplog):
    """INFO 'tool call dispatched' is logged even when the tool raises an exception."""
    async def bad_tool(**kwargs):
        raise RuntimeError("deliberate failure")

    call = _make_call("call_err", "bad_tool", {"a": 1, "b": 2})

    with caplog.at_level(logging.INFO, logger="corvidae.tool"):
        await dispatch_tool_call(call, {"bad_tool": bad_tool})

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [r for r in records if r.getMessage() == "tool call dispatched"]
    assert matching, "Expected INFO 'tool call dispatched' record even when tool raises"


# ---------------------------------------------------------------------------
# ToolCallResult dataclass shape
# ---------------------------------------------------------------------------


async def test_tool_call_result_has_required_fields():
    """ToolCallResult has all required fields: tool_call_id, tool_name, content,
    latency_ms, error."""
    tool_fn = AsyncMock(return_value="x")
    call = _make_call("id1", "t", {})

    result = await dispatch_tool_call(call, {"t": tool_fn})

    assert hasattr(result, "tool_call_id")
    assert hasattr(result, "tool_name")
    assert hasattr(result, "content")
    assert hasattr(result, "latency_ms")
    assert hasattr(result, "error")
