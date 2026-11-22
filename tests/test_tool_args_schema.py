"""Tool arguments are validated against the generated schema (#1161).

The message's wording of *what is accepted* is jsonschema's own (#1170): five
review rounds of deriving it by hand produced wrong answers for nested paths,
$ref levels, properties named `items` and choice parameters. What these tests
pin is the part the code adds -- the path prefix, the key list for an unknown
key, and that a rejected call never reaches the tool.
"""

from __future__ import annotations

from typing import Any

from corvidae.tool import DEFAULT_TOOL_TIMEOUTS, check_tool_args, dispatch_tool_call, tool_to_schema

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


def test_an_unknown_key_lists_the_keys_the_tool_takes():
    message = check_tool_args(SCHEMA, {"query": "x", "qury": "y"})

    assert message is not None
    assert "'qury' was unexpected" in message
    assert "the tool takes: limit, offset, query" in message


def test_a_missing_required_key_is_named():
    message = check_tool_args(SCHEMA, {})

    assert message is not None
    assert "'query' is a required property" in message


def test_a_wrong_type_names_the_value_and_the_type():
    message = check_tool_args(SCHEMA, {"query": ["a", "b"]})

    assert message is not None
    assert "query: ['a', 'b'] is not of type 'string'" in message


def test_a_nested_choice_names_its_path_and_values():
    schema = {
        "type": "object",
        "properties": {"opts": {"$ref": "#/$defs/Opts"}},
        "required": ["opts"],
        "$defs": {
            "Opts": {
                "type": "object",
                "properties": {"mode": {"enum": ["fast", "slow"]}},
                "required": ["mode"],
            }
        },
    }

    message = check_tool_args(schema, {"opts": {"mode": "medium"}})

    assert message is not None
    assert "opts.mode: 'medium' is not one of ['fast', 'slow']" in message


def test_a_missing_key_under_a_ref_names_its_path():
    schema = {
        "type": "object",
        "properties": {"opts": {"$ref": "#/$defs/Opts"}},
        "required": ["opts"],
        "$defs": {
            "Opts": {
                "type": "object",
                "properties": {"mode": {"type": "string"}},
                "required": ["mode"],
            }
        },
    }

    message = check_tool_args(schema, {"opts": {}})

    assert message is not None
    assert "opts: 'mode' is a required property" in message


def test_a_property_named_items_is_not_an_array_step():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "string"}},
        "required": ["items"],
    }

    message = check_tool_args(schema, {"items": 5})

    assert message is not None
    assert "items: 5 is not of type 'string'" in message


def test_a_generated_schema_rejects_an_unknown_key():
    """The schema this code generates must reject it, not just a hand-written one."""

    async def search(query: str, limit: int = 10) -> str:
        return "ok"

    params = tool_to_schema(search)["function"]["parameters"]
    message = check_tool_args(params, {"query": "x", "qury": "y"})

    assert message is not None
    assert "qury" in message


def test_a_generated_schema_does_not_require_an_optional_parameter():
    async def search(query: str, limit: int = 10) -> str:
        return "ok"

    params = tool_to_schema(search)["function"]["parameters"]

    assert params["required"] == ["query"]
    assert check_tool_args(params, {"query": "x"}) is None


def test_a_kwargs_tool_keeps_an_open_schema_and_accepts_extras():
    """A variadic parameter is not a required property (#1161 round 2)."""

    async def k(alpha: str, **kwargs: Any) -> str:
        return "ok"

    params = tool_to_schema(k)["function"]["parameters"]

    assert "kwargs" not in params["properties"]
    assert params["required"] == ["alpha"]
    assert check_tool_args(params, {"alpha": "a", "extra": 1}) is None


def test_the_agent_maps_its_schemas_by_tool_name():
    """The map the Agent hands the dispatcher is name -> parameters."""
    from corvidae.agent import Agent

    agent = object.__new__(Agent)
    agent._tool_schemas = [
        {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "broken"}},
    ]

    assert agent._tool_arg_schemas() == {"search": {"type": "object"}}


async def test_a_bad_call_never_reaches_the_tool():
    called: list[dict] = []

    async def search(query: str) -> str:
        called.append({"query": query})
        return "ok"

    result = await dispatch_tool_call(
        {"id": "c1", "function": {"name": "search", "arguments": '{"qury": "x"}'}},
        {"search": search},
        schemas={"search": SCHEMA},
        call_timeouts=DEFAULT_TOOL_TIMEOUTS,
    )

    assert called == []
    assert result.error is True
    assert "tool call rejected" in result.content


async def test_a_good_call_still_reaches_the_tool():
    calls: list[str] = []

    async def search(query: str) -> str:
        calls.append(query)
        return "ok"

    result = await dispatch_tool_call(
        {"id": "c2", "function": {"name": "search", "arguments": '{"query": "x"}'}},
        {"search": search},
        schemas={"search": SCHEMA},
        call_timeouts=DEFAULT_TOOL_TIMEOUTS,
    )

    assert calls == ["x"]
    assert result.error is False
