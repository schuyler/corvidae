"""Tests for workflow.tools.stage_complete — StageCompleteHolder, make_stage_complete_tool."""

import pytest

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
