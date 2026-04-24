"""Tests for sherman.agent_loop -- run_agent_loop, tool_to_schema, strip_thinking."""

import json
from unittest.mock import AsyncMock, MagicMock

from sherman.agent_loop import run_agent_loop, strip_thinking, tool_to_schema


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


async def test_run_agent_loop_with_extra_body():
    """run_agent_loop should accept and pass extra_body to client.chat()."""
    extra_body = {"id_slot": 1}

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
        extra_body=extra_body
    )

    # Verify extra_body was passed to client.chat
    mock_client.chat.assert_called_once()
    call_args = mock_client.chat.call_args
    assert call_args.kwargs["extra_body"] == extra_body


async def test_run_agent_loop_with_extra_body_none():
    """extra_body=None should not be passed to client.chat()."""
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
        extra_body=None
    )

    # Verify extra_body=None was not passed to client.chat
    mock_client.chat.assert_called_once()
    call_args = mock_client.chat.call_args
    assert "extra_body" not in call_args.kwargs


async def test_run_agent_loop_with_extra_body_empty():
    """extra_body={} should not add extra fields to client.chat() call."""
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
        extra_body={}
    )

    # Verify extra_body was passed but empty
    mock_client.chat.assert_called_once()
    call_args = mock_client.chat.call_args
    assert call_args.kwargs.get("extra_body") == {}
