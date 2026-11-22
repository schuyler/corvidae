"""Tool arguments are validated against the generated schema (#1161)."""

from __future__ import annotations

from corvidae.tool import check_tool_args

SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


def test_good_arguments_pass():
    assert check_tool_args(SCHEMA, {"query": "hello"}) is None


def test_no_schema_means_no_check():
    assert check_tool_args(None, {"anything": 1}) is None


def test_an_unknown_key_is_named():
    message = check_tool_args(SCHEMA, {"query": "x", "qury": "y"})
    assert message is not None
    assert "qury" in message
    assert "accepted" in message


def test_a_missing_required_key_is_named():
    message = check_tool_args(SCHEMA, {})
    assert message is not None
    assert "query" in message
    assert "required" in message


def test_a_wrong_type_names_the_value_and_what_is_accepted():
    message = check_tool_args(SCHEMA, {"query": ["a", "b"]})
    assert message is not None
    assert "query" in message
    assert "['a', 'b']" in message
    assert "string" in message


async def test_a_bad_call_never_reaches_the_tool():
    """The tool function is not called when the arguments are rejected (#1161)."""
    from corvidae.tool import dispatch_tool_call

    called: list[dict] = []

    async def search(query: str) -> str:
        called.append({"query": query})
        return "ok"

    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    result = await dispatch_tool_call(
        {"id": "c1", "function": {"name": "search", "arguments": '{"qury": "x"}'}},
        {"search": search},
        schemas={"search": schema},
        call_timeouts=__import__("corvidae.tool", fromlist=["ToolTimeouts"]).DEFAULT_TOOL_TIMEOUTS,
    )

    assert called == []
    assert result.error is True
    assert "qury" in result.content


async def test_a_good_call_still_reaches_the_tool():
    from corvidae.tool import DEFAULT_TOOL_TIMEOUTS, dispatch_tool_call

    calls: list[str] = []

    async def search(query: str) -> str:
        calls.append(query)
        return "ok"

    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    result = await dispatch_tool_call(
        {"id": "c2", "function": {"name": "search", "arguments": '{"query": "x"}'}},
        {"search": search},
        schemas={"search": schema},
        call_timeouts=DEFAULT_TOOL_TIMEOUTS,
    )

    assert calls == ["x"]
    assert result.error is False


def test_a_generated_schema_rejects_an_unknown_key():
    """The schema this code generates must reject it, not just a hand-written one."""
    from corvidae.tool import tool_to_schema

    async def search(query: str, limit: int = 10) -> str:
        return "ok"

    schema = tool_to_schema(search)["function"]["parameters"]
    message = check_tool_args(schema, {"query": "x", "qury": "y"})

    assert message is not None
    assert "qury" in message


def test_a_generated_schema_does_not_require_an_optional_parameter():
    from corvidae.tool import tool_to_schema

    async def search(query: str, limit: int = 10) -> str:
        return "ok"

    params = tool_to_schema(search)["function"]["parameters"]
    assert params["required"] == ["query"]
    assert check_tool_args(params, {"query": "x"}) is None


def test_the_agent_maps_its_schemas_by_tool_name():
    """The map the Agent hands the dispatcher is name -> parameters (#1161)."""
    from corvidae.agent import Agent

    agent = object.__new__(Agent)
    agent._tool_schemas = [
        {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "broken"}},  # no parameters
    ]

    assert agent._tool_arg_schemas() == {"search": {"type": "object"}}
