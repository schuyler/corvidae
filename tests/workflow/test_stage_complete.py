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
    """stage_complete raises ValueError for invalid outcome."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    with pytest.raises(ValueError):
        await tool_fn(outcome="invalid", summary="oops")
