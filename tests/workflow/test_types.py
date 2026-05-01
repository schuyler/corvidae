"""Tests for workflow.types — StageOutcome, ToolCallRecord, StageResult."""

from workflow.types import StageOutcome, ToolCallRecord, StageResult


def test_stage_outcome_values():
    """StageOutcome has the four expected values."""
    assert StageOutcome.PROCEED == "proceed"
    assert StageOutcome.NEED_CLARIFICATION == "need_clarification"
    assert StageOutcome.SCOPE_CHANGED == "scope_changed"
    assert StageOutcome.BLOCKED == "blocked"


def test_tool_call_record_construction():
    """ToolCallRecord stores tool call details."""
    record = ToolCallRecord(
        tool_name="shell",
        arguments={"command": "ls"},
        result="file.py",
        error=False,
    )
    assert record.tool_name == "shell"
    assert record.arguments == {"command": "ls"}
    assert record.result == "file.py"
    assert record.error is False


def test_tool_call_record_error_default():
    """ToolCallRecord.error defaults to False."""
    record = ToolCallRecord(tool_name="x", arguments={}, result="ok")
    assert record.error is False


def test_stage_result_construction():
    """StageResult stores outcome and metadata from a completed stage."""
    result = StageResult(
        stage_name="implement",
        outcome=StageOutcome.PROCEED,
        summary="Implemented the feature",
    )
    assert result.stage_name == "implement"
    assert result.outcome == StageOutcome.PROCEED
    assert result.summary == "Implemented the feature"
    assert result.escape_target is None
    assert result.tool_calls == []


def test_stage_result_with_escape_target():
    """StageResult can specify an escape target for back-tracking."""
    result = StageResult(
        stage_name="gate",
        outcome=StageOutcome.NEED_CLARIFICATION,
        summary="Need more info",
        escape_target="comment_on_issue",
    )
    assert result.escape_target == "comment_on_issue"


def test_stage_result_with_tool_calls():
    """StageResult includes a record of all tool calls made during the stage."""
    calls = [
        ToolCallRecord(tool_name="shell", arguments={"command": "ls"}, result="out"),
        ToolCallRecord(tool_name="write_file", arguments={"path": "a.py"}, result="ok"),
    ]
    result = StageResult(
        stage_name="test",
        outcome=StageOutcome.PROCEED,
        summary="done",
        tool_calls=calls,
    )
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "shell"
