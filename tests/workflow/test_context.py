"""Tests for workflow.context — WorkflowContext."""

from workflow.types import ToolCallRecord
from workflow.context import WorkflowContext


def test_empty_context_builds_empty_string():
    """A fresh context produces an empty context string."""
    ctx = WorkflowContext()
    assert ctx.build_context_string("any_stage") == ""


def test_record_stage_adds_summary():
    """record_stage stores the stage summary."""
    ctx = WorkflowContext()
    ctx.record_stage("implement", "Done", [])
    assert ctx.summaries["implement"] == "Done"


def test_record_stage_accumulates_tool_calls():
    """record_stage appends tool call records to the log."""
    ctx = WorkflowContext()
    calls = [ToolCallRecord(tool_name="shell", arguments={}, result="ok")]
    ctx.record_stage("implement", "Done", calls)
    assert len(ctx.tool_logs) == 1
    assert ctx.tool_logs[0].tool_name == "shell"

    ctx.record_stage("refactor", "Clean", [])
    assert len(ctx.tool_logs) == 1  # No new calls appended


def test_build_context_string_excludes_current_stage():
    """build_context_string omits the named stage from context."""
    ctx = WorkflowContext()
    ctx.summaries["a"] = "Summary A"
    ctx.summaries["b"] = "Summary B"
    result = ctx.build_context_string("a")
    assert "Summary A" not in result
    assert "Summary B" in result


def test_build_context_string_includes_all_other_stages():
    """build_context_string includes summaries from all other stages."""
    ctx = WorkflowContext()
    ctx.summaries["a"] = "Summary A"
    ctx.summaries["b"] = "Summary B"
    ctx.summaries["c"] = "Summary C"
    result = ctx.build_context_string("b")
    assert "Summary A" in result
    assert "Summary C" in result
    assert "Summary B" not in result


def test_facts_store_arbitrary_values():
    """WorkflowContext.facts holds arbitrary orchestrator-gathered data."""
    ctx = WorkflowContext()
    ctx.facts["issue_url"] = "https://github.com/o/r/issues/1"
    ctx.facts["branch"] = "feat/issue-1"
    assert ctx.facts["issue_url"] == "https://github.com/o/r/issues/1"
    assert ctx.facts["branch"] == "feat/issue-1"
