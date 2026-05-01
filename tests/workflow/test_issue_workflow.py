"""Tests for workflow.workflows.issue — build_issue_workflow."""

from workflow.types import StageOutcome
from workflow.stage import Stage
from workflow.workflows.issue import build_issue_workflow, parse_issue_url


def test_parse_issue_url():
    """parse_issue_url extracts owner/repo and issue number."""
    repo, num = parse_issue_url(
        "https://github.com/owner/repo/issues/42"
    )
    assert repo == "owner/repo"
    assert num == 42


def test_parse_issue_url_trailing_slash():
    """parse_issue_url handles trailing slashes."""
    repo, num = parse_issue_url(
        "https://github.com/owner/repo/issues/42/"
    )
    assert repo == "owner/repo"
    assert num == 42


def test_parse_issue_url_invalid():
    """parse_issue_url raises ValueError for non-issue URLs."""
    import pytest
    with pytest.raises(ValueError):
        parse_issue_url("https://github.com/owner/repo/pull/42")


def test_workflow_has_expected_stages():
    """build_issue_workflow returns stages with expected names."""
    stages = build_issue_workflow(
        repo="owner/repo",
        issue_number=1,
        repo_path="/repo",
        issue_body="Fix the bug",
    )
    names = [s.name for s in stages]
    assert "comment_on_issue" in names
    assert "add_label" in names
    assert "create_branch" in names
    assert "gate_ready" in names
    assert "write_tests" in names
    assert "implement" in names
    assert "refactor" in names
    assert "update_docs" in names
    assert "submit_pr" in names


def test_gate_ready_has_escape_target():
    """gate_ready stage can escape back to comment_on_issue."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    gate = next(s for s in stages if s.name == "gate_ready")
    assert StageOutcome.NEED_CLARIFICATION in gate.escape_targets
    assert gate.escape_targets[StageOutcome.NEED_CLARIFICATION] == "comment_on_issue"


def test_mechanical_stages_are_marked():
    """add_label is a mechanical stage."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    label_stage = next(s for s in stages if s.name == "add_label")
    assert label_stage.is_mechanical is True
    assert label_stage.mechanical_fn is not None


def test_llm_stages_have_instructions():
    """Every non-mechanical stage has a non-empty instruction."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    for stage in stages:
        if not stage.is_mechanical:
            assert len(stage.instruction) > 20, (
                f"Stage {stage.name} has no instruction"
            )


def test_llm_stages_have_tools():
    """Stages that need file/shell/git/GitHub tools have them."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    impl = next(s for s in stages if s.name == "implement")
    tool_names = [t.name for t in impl.tools]
    # implement needs write_file, read_file, and a shell tool
    assert any("write" in n for n in tool_names)
    assert any("read" in n for n in tool_names)


def test_update_docs_is_conditional():
    """update_docs has a should_run callback."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    docs = next(s for s in stages if s.name == "update_docs")
    assert docs.should_run is not None
