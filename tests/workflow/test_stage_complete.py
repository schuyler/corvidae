"""Tests for workflow.tools.stage_complete — StageCompleteHolder, make_stage_complete_tool."""

import pytest

from corvidae.tool import Tool
from workflow.tools.stage_complete import StageCompleteHolder, make_stage_complete_tool


def test_holder_starts_empty():
    """A fresh holder has no result."""
    holder = StageCompleteHolder()
    assert holder.outcome is None
    assert holder.summary == ""
    assert holder.escape_target is None
    assert holder.called is False


async def test_stage_complete_tool_populates_holder():
    """Calling the stage_complete tool writes to the holder."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    result = await tool_fn(outcome="proceed", summary="Task done")

    assert holder.called is True
    assert holder.outcome.value == "proceed"
    assert holder.summary == "Task done"
    assert holder.escape_target is None
    assert "proceed" in result


async def test_stage_complete_tool_with_escape_target():
    """stage_complete can specify an escape target."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    await tool_fn(
        outcome="need_clarification",
        summary="Unclear requirements",
        escape_target="comment_on_issue",
    )

    assert holder.outcome.value == "need_clarification"
    assert holder.escape_target == "comment_on_issue"


async def test_stage_complete_tool_invalid_outcome():
    """stage_complete returns an error string for invalid outcome."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    result = await tool_fn(outcome="invalid", summary="oops")

    # Should NOT raise — returns a helpful error for the LLM to retry.
    assert "Error" in result
    assert "'proceed'" in result
    assert holder.called is False


async def test_stage_complete_tool_invalid_outcome_sentence():
    """stage_complete returns an error when the LLM passes a sentence."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    result = await tool_fn(
        outcome="Comment posted on GitHub issue #6 describing the plan.",
        summary="Done",
    )

    assert "Error" in result
    assert "'proceed'" in result
    assert holder.called is False
    assert holder.outcome is None


async def test_stage_complete_normalizes_null_escape_target():
    """stage_complete normalizes 'null' escape_target to None."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    await tool_fn(outcome="proceed", summary="done", escape_target="null")

    assert holder.outcome.value == "proceed"
    assert holder.escape_target is None


async def test_stage_complete_normalizes_none_escape_target():
    """stage_complete normalizes 'none' escape_target to None."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    await tool_fn(outcome="proceed", summary="done", escape_target="none")

    assert holder.escape_target is None


async def test_stage_complete_normalizes_empty_escape_target():
    """stage_complete normalizes '' escape_target to None."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    await tool_fn(outcome="proceed", summary="done", escape_target="")

    assert holder.escape_target is None


async def test_stage_complete_preserves_real_escape_target():
    """stage_complete preserves a real escape target value."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    await tool_fn(
        outcome="need_clarification",
        summary="unclear",
        escape_target="comment_on_issue",
    )

    assert holder.escape_target == "comment_on_issue"


# ---------------------------------------------------------------------------
# Schema enum constraint tests
# ---------------------------------------------------------------------------


def test_stage_complete_schema_has_enum_for_outcome():
    """The tool schema must include an enum constraint for the outcome parameter.

    This is the core fix: the LLM sees the enum in the schema and is forced
    to pick one of the four valid values instead of passing free-form text.
    """
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)
    tool = Tool.from_function(tool_fn)

    outcome_prop = tool.schema["function"]["parameters"]["properties"]["outcome"]
    assert "enum" in outcome_prop, "outcome property should have enum constraint"
    assert set(outcome_prop["enum"]) == {
        "proceed",
        "need_clarification",
        "scope_changed",
        "blocked",
    }


def test_stage_complete_schema_outcome_is_required():
    """The outcome parameter must be in the required list."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)
    tool = Tool.from_function(tool_fn)

    params = tool.schema["function"]["parameters"]
    assert "outcome" in params["required"]


async def test_stage_complete_error_message_explicit_about_sentences():
    """When the LLM passes a sentence, the error message must be very explicit."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    result = await tool_fn(
        outcome="Created and checked out branch fix/issue-6",
        summary="Done",
    )

    # The error must tell the LLM it passed a sentence and list valid values.
    assert "Error" in result
    assert "'proceed'" in result
    assert holder.called is False
    assert holder.outcome is None
