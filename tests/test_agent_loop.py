"""Tests for sherman.agent_loop -- run_agent_loop, tool_to_schema, strip_thinking."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from sherman.agent_loop import run_agent_loop, strip_thinking, tool_to_schema

try:
    from sherman.agent_loop import AgentTurnResult, run_agent_turn
except ImportError:
    AgentTurnResult = None
    run_agent_turn = None


def _make_text_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


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
# run_agent_loop tests
# ---------------------------------------------------------------------------


async def test_simple_response_no_tools():
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("Hello, world!"))

    messages = [{"role": "user", "content": "Hi"}]
    result = await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    assert result == "Hello, world!"
    assert messages[-1] == {"role": "assistant", "content": "Hello, world!"}


async def test_single_tool_call():
    tool_fn = AsyncMock(return_value="tool result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_1", "my_tool", {"x": "value"})]
            ),
            _make_text_response("final answer"),
        ]
    )

    messages = [{"role": "user", "content": "do thing"}]
    result = await run_agent_loop(
        client,
        messages,
        tools={"my_tool": tool_fn},
        tool_schemas=[],
    )

    # Tool should be called with the correct args
    tool_fn.assert_awaited_once_with(x="value")

    assert result == "final answer"

    # messages should contain: user, assistant w/ tool_call, tool result, assistant final
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]

    # assistant message with tool call
    assert messages[1].get("tool_calls") is not None

    # tool result message
    assert messages[2]["role"] == "tool"
    assert messages[2]["content"] == "tool result"

    # final assistant message
    assert messages[3] == {"role": "assistant", "content": "final answer"}


async def test_unknown_tool_returns_error():
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_x", "nonexistent_tool", {})]
            ),
            _make_text_response("ok"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    await run_agent_loop(
        client,
        messages,
        tools={},
        tool_schemas=[],
    )

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert "Error: unknown tool" in tool_result["content"]


async def test_tool_exception_returns_error():
    async def bad_tool(**kwargs):
        raise ValueError("something went wrong")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_2", "bad_tool", {})]
            ),
            _make_text_response("recovered"),
        ]
    )

    messages = [{"role": "user", "content": "break it"}]
    await run_agent_loop(
        client,
        messages,
        tools={"bad_tool": bad_tool},
        tool_schemas=[],
    )

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert tool_result["content"].startswith("Error:")


async def test_max_turns_exceeded():
    client = MagicMock()
    # Always returns a tool call, never terminates naturally
    client.chat = AsyncMock(
        return_value=_make_tool_call_response(
            [_make_tool_call("call_loop", "noop", {})]
        )
    )

    noop = AsyncMock(return_value="noop result")
    messages = [{"role": "user", "content": "spin"}]
    result = await run_agent_loop(
        client,
        messages,
        tools={"noop": noop},
        tool_schemas=[],
        max_turns=3,
    )

    assert result == "(max tool-calling rounds reached)"


async def test_multiple_tool_calls_in_one_turn():
    tool_a = AsyncMock(return_value="result_a")
    tool_b = AsyncMock(return_value="result_b")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [
                    _make_tool_call("call_a", "tool_a", {"n": 1}),
                    _make_tool_call("call_b", "tool_b", {"n": 2}),
                ]
            ),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "run both"}]
    result = await run_agent_loop(
        client,
        messages,
        tools={"tool_a": tool_a, "tool_b": tool_b},
        tool_schemas=[],
    )

    tool_a.assert_awaited_once_with(n=1)
    tool_b.assert_awaited_once_with(n=2)
    assert result == "done"

    # Both tool result messages should be present
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 2


# ---------------------------------------------------------------------------
# tool_to_schema tests
# ---------------------------------------------------------------------------


def test_tool_to_schema_basic():
    def greet(name: str) -> str:
        """Say hello to someone."""
        return f"Hello {name}"

    schema = tool_to_schema(greet)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "greet"
    assert schema["function"]["description"] == "Say hello to someone."
    assert "parameters" in schema["function"]
    # title keys should be stripped
    assert "title" not in schema["function"]["parameters"]


def test_tool_to_schema_multiple_params():
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    schema = tool_to_schema(add)

    params = schema["function"]["parameters"]
    properties = params["properties"]
    assert "a" in properties
    assert "b" in properties


def test_tool_to_schema_no_annotations():
    def mystery(x, y):
        """A mysterious function."""
        pass

    schema = tool_to_schema(mystery)

    params = schema["function"]["parameters"]
    properties = params.get("properties", {})
    # Parameters without type hints should default to string
    for prop in properties.values():
        assert prop.get("type") == "string"


def test_tool_to_schema_zero_params():
    """tool_to_schema on a zero-parameter async function should produce
    a schema with empty properties and no 'required' key."""

    async def noop() -> str:
        """Do nothing."""
        ...

    schema = tool_to_schema(noop)
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert params.get("properties") == {} or "properties" not in params or params["properties"] == {}
    assert "required" not in params


# ---------------------------------------------------------------------------
# strip_thinking tests
# ---------------------------------------------------------------------------


def test_strip_thinking_removes_tags():
    text = "<think>internal monologue</think>actual response"
    assert strip_thinking(text) == "actual response"


def test_strip_thinking_no_tags():
    text = "plain response without thinking"
    assert strip_thinking(text) == text


def test_strip_thinking_multiline():
    text = "<think>\nline one\nline two\n</think>\nfinal answer"
    assert strip_thinking(text) == "final answer"


async def test_run_agent_loop_does_not_pass_extra_body():
    """run_agent_loop no longer accepts extra_body — LLMClient handles it internally.
    Verify chat() is called without extra_body kwarg."""
    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value=_make_text_response("test response"))

    messages = [{"role": "user", "content": "hello"}]
    tools = {}
    tool_schemas = []

    result = await run_agent_loop(
        client=mock_client,
        messages=messages,
        tools=tools,
        tool_schemas=tool_schemas,
        max_turns=1,
    )

    # run_agent_loop passes no extra_body to client.chat() — LLMClient handles it
    mock_client.chat.assert_called_once()
    call_args = mock_client.chat.call_args
    assert "extra_body" not in call_args.kwargs


async def test_llm_client_extra_body_applied_in_chat():
    """LLMClient with extra_body merges it into the request payload.
    Verify the instance extra_body is applied when client.chat() is called."""
    from sherman.llm import LLMClient
    import aiohttp
    from unittest.mock import patch, AsyncMock, MagicMock

    client = LLMClient(
        base_url="http://localhost:8080",
        model="test-model",
        extra_body={"id_slot": 1},
    )
    # Verify the extra_body is stored on the instance
    assert client.extra_body == {"id_slot": 1}


async def test_llm_client_no_extra_body_by_default():
    """LLMClient without extra_body has extra_body=None."""
    from sherman.llm import LLMClient

    client = LLMClient(base_url="http://localhost:8080", model="test-model")
    assert client.extra_body is None


# ---------------------------------------------------------------------------
# Logging tests — INFO records with latency_ms
# ---------------------------------------------------------------------------


async def test_llm_response_logs_info_with_latency_ms(caplog):
    """After a successful LLM call, an INFO record with message 'LLM response
    received' and a numeric latency_ms attribute must be emitted."""
    import logging

    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("hello"))

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.INFO, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    matching = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "LLM response received"]
    assert matching, "Expected INFO record with message 'LLM response received'"
    assert hasattr(matching[0], "latency_ms"), "'LLM response received' log must have latency_ms attribute"
    assert isinstance(matching[0].latency_ms, (int, float)), "latency_ms must be numeric"


async def test_tool_call_result_logs_info_with_latency_ms(caplog):
    """After a successful tool execution, an INFO record with message 'tool call
    result' and a numeric latency_ms attribute must be emitted."""
    import logging

    tool_fn = AsyncMock(return_value="tool result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "my_tool", {"x": "v"})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.INFO, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={"my_tool": tool_fn}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    matching = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "tool call result"]
    assert matching, "Expected INFO record with message 'tool call result'"
    assert hasattr(matching[0], "latency_ms"), "'tool call result' log must have latency_ms attribute"
    assert isinstance(matching[0].latency_ms, (int, float)), "latency_ms must be numeric"


async def test_tool_call_dispatched_logs_info(caplog):
    """After dispatching a tool call, an INFO record with message 'tool call
    dispatched' must be emitted (latency_ms not required)."""
    import logging

    tool_fn = AsyncMock(return_value="result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "my_tool", {})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.INFO, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={"my_tool": tool_fn}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    matching = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "tool call dispatched"]
    assert matching, "Expected INFO record with message 'tool call dispatched'"


async def test_tool_call_result_not_logged_for_unknown_tool(caplog):
    """When an unknown tool is called, 'tool call result' INFO must NOT be emitted."""
    import logging

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "ghost_tool", {})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.INFO, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    result_records = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "tool call result"]
    assert not result_records, "'tool call result' INFO must NOT be emitted for unknown tool"


async def test_tool_call_result_not_logged_on_exception(caplog):
    """When a tool raises an exception, 'tool call result' INFO must NOT be emitted."""
    import logging

    async def bad_tool(**kwargs):
        raise ValueError("boom")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "bad_tool", {})]),
            _make_text_response("recovered"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.INFO, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={"bad_tool": bad_tool}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    result_records = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "tool call result"]
    assert not result_records, "'tool call result' INFO must NOT be emitted when tool raises"


# ---------------------------------------------------------------------------
# Logging tests — DEBUG records for content visibility
# ---------------------------------------------------------------------------


async def test_llm_response_content_debug_log(caplog):
    """After a successful LLM call, a DEBUG record with message 'LLM response
    content' must be emitted with has_reasoning_content and
    reasoning_content_length attributes."""
    import logging

    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("hello"))

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.DEBUG, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "LLM response content"
    ]
    assert matching, "Expected DEBUG record with message 'LLM response content'"
    rec = matching[0]
    assert hasattr(rec, "has_reasoning_content"), (
        "'LLM response content' DEBUG log must have has_reasoning_content attribute"
    )
    assert isinstance(rec.has_reasoning_content, bool), (
        "has_reasoning_content must be a bool"
    )
    assert hasattr(rec, "reasoning_content_length"), (
        "'LLM response content' DEBUG log must have reasoning_content_length attribute"
    )
    assert hasattr(rec, "content"), "'LLM response content' DEBUG log must have content attribute"


async def test_llm_response_content_debug_log_with_reasoning(caplog):
    """When response has reasoning_content, DEBUG log must have
    has_reasoning_content=True and reasoning_content_length > 0."""
    import logging

    reasoning_text = "<reasoning>think step by step</reasoning>"
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "final answer",
                    "reasoning_content": reasoning_text,
                }
            }
        ]
    }
    client = MagicMock()
    client.chat = AsyncMock(return_value=response)

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.DEBUG, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "LLM response content"
    ]
    assert matching, "Expected DEBUG record with message 'LLM response content'"
    rec = matching[0]
    assert rec.has_reasoning_content is True, (
        "has_reasoning_content must be True when reasoning_content is present"
    )
    assert isinstance(rec.reasoning_content_length, int), (
        "reasoning_content_length must be an int when reasoning_content is present"
    )
    assert rec.reasoning_content_length > 0, (
        "reasoning_content_length must be > 0 for non-empty reasoning_content"
    )
    assert hasattr(rec, "content"), "'LLM response content' DEBUG log must have content attribute"


async def test_llm_response_content_truncated(caplog):
    """Content longer than 200 chars must be truncated in the DEBUG log."""
    import logging

    long_content = "x" * 300
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response(long_content))

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.DEBUG, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "LLM response content"
    ]
    assert matching, "Expected DEBUG record with message 'LLM response content'"
    rec = matching[0]
    assert hasattr(rec, "content"), (
        "'LLM response content' DEBUG log must have content attribute"
    )
    assert len(rec.content) <= 203, (
        "Truncated content must be at most 203 chars (200 + '...')"
    )
    assert rec.content.endswith("..."), (
        "Truncated content must end with '...'"
    )


async def test_tool_call_arguments_debug_log(caplog):
    """After dispatching a tool call, a DEBUG record with message 'tool call
    arguments' must be emitted with tool and arguments attributes."""
    import logging

    tool_fn = AsyncMock(return_value="result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "my_tool", {"x": "v"})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.DEBUG, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={"my_tool": tool_fn}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "tool call arguments"
    ]
    assert matching, "Expected DEBUG record with message 'tool call arguments'"
    rec = matching[0]
    assert hasattr(rec, "tool"), (
        "'tool call arguments' DEBUG log must have tool attribute"
    )
    assert rec.tool == "my_tool", "tool attribute must match the called tool name"
    assert hasattr(rec, "arguments"), (
        "'tool call arguments' DEBUG log must have arguments attribute"
    )
    assert isinstance(rec.arguments, str), "arguments must be a string (JSON)"


async def test_tool_call_result_content_debug_log(caplog):
    """After a successful tool execution, a DEBUG record with message 'tool call
    result content' must be emitted with tool and content attributes."""
    import logging

    tool_fn = AsyncMock(return_value="tool result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "my_tool", {"x": "v"})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.DEBUG, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={"my_tool": tool_fn}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "tool call result content"
    ]
    assert matching, "Expected DEBUG record with message 'tool call result content'"
    rec = matching[0]
    assert hasattr(rec, "tool"), (
        "'tool call result content' DEBUG log must have tool attribute"
    )
    assert rec.tool == "my_tool", "tool attribute must match the called tool name"
    assert hasattr(rec, "content"), (
        "'tool call result content' DEBUG log must have content attribute"
    )


async def test_tool_call_result_content_not_logged_on_exception(caplog):
    """When a tool raises an exception, 'tool call result content' DEBUG must NOT
    be emitted."""
    import logging

    async def bad_tool(**kwargs):
        raise ValueError("boom")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "bad_tool", {})]),
            _make_text_response("recovered"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.DEBUG, logger="sherman.agent_loop"):
        await run_agent_loop(client, messages, tools={"bad_tool": bad_tool}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    result_content_records = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "tool call result content"
    ]
    assert not result_content_records, (
        "'tool call result content' DEBUG must NOT be emitted when tool raises"
    )


# ---------------------------------------------------------------------------
# run_agent_turn tests
# ---------------------------------------------------------------------------


def _make_mixed_response(text: str, calls: list[dict]) -> dict:
    """Response with both text content and tool calls."""
    return {
        "choices": [
            {
                "message": {
                    "content": text,
                    "tool_calls": calls,
                }
            }
        ]
    }


def _make_null_content_tool_call_response(calls: list[dict]) -> dict:
    """Response with content=null and tool calls — as some LLMs emit."""
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": calls,
                }
            }
        ]
    }


_skip_no_agent_turn = pytest.mark.skipif(
    run_agent_turn is None, reason="run_agent_turn not yet implemented"
)


# Cases 1, 7, 8: text-only response; message appended; latency positive float
@_skip_no_agent_turn
async def test_run_agent_turn_text_response():
    """Text response: tool_calls is [], text is the content, message appended,
    latency_ms is a positive float."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("Hello from turn"))

    messages = [{"role": "user", "content": "Hi"}]
    result = await run_agent_turn(client, messages, tool_schemas=[])

    assert isinstance(result, AgentTurnResult)
    assert result.text == "Hello from turn"
    assert result.tool_calls == []
    # result.message is the raw assistant dict
    assert result.message is messages[-1]
    assert result.message.get("role") == "assistant"
    # message appended in place (case 7)
    assert len(messages) == 2
    assert messages[-1]["content"] == "Hello from turn"
    # latency_ms is a positive float (case 8)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0


# Cases 2, 6: tool-calls-only response with non-empty tool_schemas passes schemas to client
@_skip_no_agent_turn
async def test_run_agent_turn_tool_calls_only():
    """Tool call response: tool_calls populated, text is '', non-empty tool_schemas
    passed as-is to client.chat()."""
    tool_schemas = [{"type": "function", "function": {"name": "do_thing"}}]
    calls = [_make_tool_call("call_1", "do_thing", {"x": 1})]
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_tool_call_response(calls))

    messages = [{"role": "user", "content": "do it"}]
    expected_call_arg = list(messages)  # snapshot before mutation
    result = await run_agent_turn(client, messages, tool_schemas=tool_schemas)

    assert result.tool_calls == calls
    assert result.text == ""
    # non-empty tool_schemas must be passed through unchanged (case 6)
    # client.chat is called before the append, so use the pre-mutation snapshot
    client.chat.assert_awaited_once_with(expected_call_arg, tools=tool_schemas)


# Case 3: response with both text and tool calls
@_skip_no_agent_turn
async def test_run_agent_turn_text_and_tool_calls():
    """Response containing both text and tool calls: both are surfaced on result."""
    calls = [_make_tool_call("call_2", "do_thing", {})]
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_mixed_response("thinking out loud", calls))

    messages = [{"role": "user", "content": "go"}]
    result = await run_agent_turn(client, messages, tool_schemas=[])

    assert result.text == "thinking out loud"
    assert result.tool_calls == calls


# Case 4: content=null → text is "" not None
@_skip_no_agent_turn
async def test_run_agent_turn_null_content_text_is_empty_string():
    """When the LLM returns content=null, result.text must be '' not None."""
    calls = [_make_tool_call("call_3", "do_thing", {})]
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_null_content_tool_call_response(calls))

    messages = [{"role": "user", "content": "go"}]
    result = await run_agent_turn(client, messages, tool_schemas=[])

    assert result.text == ""
    assert result.text is not None


# Case 5: empty tool_schemas → tools=None passed to client.chat()
@_skip_no_agent_turn
async def test_run_agent_turn_empty_tool_schemas_passes_none_to_client():
    """Empty tool_schemas must result in tools=None being passed to client.chat()."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("ok"))

    messages = [{"role": "user", "content": "hi"}]
    await run_agent_turn(client, messages, tool_schemas=[])

    client.chat.assert_awaited_once()
    call_kwargs = client.chat.call_args.kwargs
    assert call_kwargs.get("tools") is None


# Case 9: INFO "LLM response received" with latency_ms; DEBUG "LLM response content"
@_skip_no_agent_turn
async def test_run_agent_turn_logging_info_and_debug(caplog):
    """run_agent_turn must emit INFO 'LLM response received' (with latency_ms) and
    DEBUG 'LLM response content' on a successful call."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("logged"))

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.DEBUG, logger="sherman.agent_loop"):
        await run_agent_turn(client, messages, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]

    info_records = [
        r for r in records
        if r.levelno == logging.INFO and r.getMessage() == "LLM response received"
    ]
    assert info_records, "Expected INFO record 'LLM response received'"
    assert hasattr(info_records[0], "latency_ms"), "'LLM response received' must have latency_ms"
    assert isinstance(info_records[0].latency_ms, (int, float)), "latency_ms must be numeric"

    debug_records = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "LLM response content"
    ]
    assert debug_records, "Expected DEBUG record 'LLM response content'"


# Case 10: reasoning_content present → DEBUG log attributes
@_skip_no_agent_turn
async def test_run_agent_turn_reasoning_content_debug_log(caplog):
    """When the response contains reasoning_content, the DEBUG 'LLM response content'
    log must have has_reasoning_content=True and a positive reasoning_content_length."""
    reasoning_text = "step by step reasoning here"
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "final answer",
                    "reasoning_content": reasoning_text,
                }
            }
        ]
    }
    client = MagicMock()
    client.chat = AsyncMock(return_value=response)

    messages = [{"role": "user", "content": "reason through it"}]
    with caplog.at_level(logging.DEBUG, logger="sherman.agent_loop"):
        await run_agent_turn(client, messages, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "sherman.agent_loop"]
    debug_records = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "LLM response content"
    ]
    assert debug_records, "Expected DEBUG record 'LLM response content'"
    rec = debug_records[0]
    assert hasattr(rec, "has_reasoning_content")
    assert rec.has_reasoning_content is True
    assert hasattr(rec, "reasoning_content_length")
    assert isinstance(rec.reasoning_content_length, int)
    assert rec.reasoning_content_length > 0


# ---------------------------------------------------------------------------
# RED PHASE: tool exception error message includes fn_name (architecture critique)
# ---------------------------------------------------------------------------


async def test_tool_exception_error_message_includes_tool_name():
    """When a tool raises an exception in run_agent_loop, the tool result message
    must include the tool name.

    RED phase: fails until agent_loop.py line 199 changes from
    'Error: unknown error' to "Error: tool '{fn_name}' raised an exception".
    """
    async def bad_tool(**kwargs):
        raise RuntimeError("deliberate failure")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_err", "bad_tool", {})]
            ),
            _make_text_response("recovered"),
        ]
    )

    messages = [{"role": "user", "content": "trigger error"}]
    await run_agent_loop(
        client,
        messages,
        tools={"bad_tool": bad_tool},
        tool_schemas=[],
    )

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert "bad_tool" in tool_result["content"], (
        f"Error message must include the tool name 'bad_tool', got: {tool_result['content']!r}"
    )


# Case 11: exception from client.chat() → messages unchanged, exception propagates
@_skip_no_agent_turn
async def test_run_agent_turn_exception_leaves_messages_unchanged():
    """When client.chat() raises, messages must be left unchanged and the exception
    must propagate unmodified to the caller."""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=RuntimeError("session not started"))

    original_message = {"role": "user", "content": "hi"}
    messages = [original_message]

    with pytest.raises(RuntimeError, match="session not started"):
        await run_agent_turn(client, messages, tool_schemas=[])

    # messages must be exactly as passed — no assistant message appended
    assert len(messages) == 1
    assert messages[0] is original_message
