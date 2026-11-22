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
