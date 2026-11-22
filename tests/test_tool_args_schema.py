"""Tool arguments are validated against the generated schema (#1161)."""

from __future__ import annotations

from corvidae.tool import check_tool_args

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
        "offset": {"type": "integer"},
    },
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
    assert "tool call rejected" in result.content


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


def test_a_kwargs_tool_keeps_an_open_schema_and_accepts_extras():
    """A variadic parameter is not a required property (#1161 round 2)."""
    from typing import Any

    from corvidae.tool import tool_to_schema

    async def k(alpha: str, **kwargs: Any) -> str:
        return "ok"

    params = tool_to_schema(k)["function"]["parameters"]
    assert "kwargs" not in params["properties"]
    assert params["required"] == ["alpha"]
    assert check_tool_args(params, {"alpha": "a", "extra": 1}) is None


def test_the_message_names_the_key_and_the_value_it_got():
    message = check_tool_args(SCHEMA, {"query": "x", "qury": "y"})
    assert message is not None
    assert "qury = 'y'" in message
    assert "accepted: limit, offset, query" in message  # every key it takes


def test_the_message_names_a_missing_required_key_by_name():
    message = check_tool_args(SCHEMA, {})
    assert message is not None
    assert "query is missing" in message
    assert "accepted: string" in message


def test_the_message_reports_a_choice_parameters_allowed_values():
    """A $ref property's allowed values are named, not just "the declared shape"."""
    schema = {
        "type": "object",
        "properties": {"mode": {"$ref": "#/$defs/Speed"}},
        "required": ["mode"],
        "$defs": {"Speed": {"enum": ["fast", "slow"]}},
    }

    message = check_tool_args(schema, {"mode": "medium"})

    assert message is not None
    assert "accepted: one of 'fast', 'slow'" in message


def test_a_nested_choice_reports_the_choice_not_the_parent_type():
    """A $ref inside a property resolves to the leaf (#1170 B1)."""
    schema = {
        "type": "object",
        "properties": {"opts": {"$ref": "#/$defs/Opts"}},
        "required": ["opts"],
        "$defs": {
            "Opts": {
                "type": "object",
                "properties": {"mode": {"enum": ["fast", "slow"]}},
            }
        },
    }

    message = check_tool_args(schema, {"opts": {"mode": "medium"}})

    assert message is not None
    assert "accepted: one of 'fast', 'slow'" in message


def test_a_property_named_items_is_a_property_not_an_array_step():
    """A string path part is never an array index (#1170 round 4)."""
    schema = {
        "type": "object",
        "properties": {"items": {"type": "string"}},
        "required": ["items"],
    }

    message = check_tool_args(schema, {"items": 5})

    assert message is not None
    assert "accepted: string" in message


def test_a_nested_unknown_key_names_its_value_and_that_objects_keys():
    schema = {
        "type": "object",
        "properties": {
            "opts": {
                "type": "object",
                "properties": {"mode": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "required": ["opts"],
    }

    message = check_tool_args(schema, {"opts": {"mdoe": "fast"}})

    assert message is not None
    assert "mdoe = 'fast'" in message
    assert "accepted: mode" in message
