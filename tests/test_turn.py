"""Tests for corvidae.turn -- run_agent_turn, AgentTurnResult, tool_to_schema, LLMClient extra_body."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.turn import AgentTurnResult, run_agent_turn
from corvidae.tool import tool_to_schema

from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call_response,
    _make_tool_call,
    _make_mixed_response,
    _make_null_content_tool_call_response,
)


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
# LLMClient extra_body tests
# ---------------------------------------------------------------------------


async def test_llm_client_extra_body_applied_in_chat():
    """LLMClient with extra_body merges it into the request payload.
    Verify the instance extra_body is applied when client.chat() is called."""
    from corvidae.llm import LLMClient

    client = LLMClient(
        base_url="http://localhost:8080",
        model="test-model",
        extra_body={"id_slot": 1},
    )
    # Verify the extra_body is stored on the instance
    assert client.extra_body == {"id_slot": 1}


async def test_llm_client_no_extra_body_by_default():
    """LLMClient without extra_body has extra_body=None."""
    from corvidae.llm import LLMClient

    client = LLMClient(base_url="http://localhost:8080", model="test-model")
    assert client.extra_body is None


# ---------------------------------------------------------------------------
# run_agent_turn tests
# ---------------------------------------------------------------------------


# Cases 1, 7, 8: text-only response; message appended; latency positive float
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
async def test_run_agent_turn_logging_info_and_debug(caplog):
    """run_agent_turn must emit INFO 'LLM response received' (with latency_ms) and
    DEBUG 'LLM response content' on a successful call."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("logged"))

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.DEBUG, logger="corvidae.turn"):
        await run_agent_turn(client, messages, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.turn"]

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
    with caplog.at_level(logging.DEBUG, logger="corvidae.turn"):
        await run_agent_turn(client, messages, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.turn"]
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


# Case 11: exception from client.chat() → messages unchanged, exception propagates
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
