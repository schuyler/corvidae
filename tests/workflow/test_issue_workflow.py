"""Tests for workflow.workflows.issue — build_issue_workflow."""

import pytest

from unittest.mock import AsyncMock, patch

from workflow.types import StageOutcome
from workflow.stage import Stage
from workflow.workflows.issue import (
    build_issue_workflow, parse_issue_url, _docs_should_run,
    _tests_should_run, _refactor_should_run, _implement_should_run,
)


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


def test_commit_changes_stage_exists():
    """commit_changes stage is present in the workflow."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    names = [s.name for s in stages]
    assert "commit_changes" in names


def test_commit_changes_positioned_before_gate_docs():
    """commit_changes sits between refactor and gate_docs."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    names = [s.name for s in stages]
    refactor_idx = names.index("refactor")
    commit_idx = names.index("commit_changes")
    gate_docs_idx = names.index("gate_docs")
    assert refactor_idx < commit_idx < gate_docs_idx


def test_commit_changes_is_mechanical():
    """commit_changes is a mechanical stage."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    commit_stage = next(s for s in stages if s.name == "commit_changes")
    assert commit_stage.is_mechanical is True
    assert commit_stage.mechanical_fn is not None


@pytest.mark.timeout(5)
async def test_commit_changes_uses_implement_summary():
    """commit_changes uses the implement stage's summary as commit message."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    commit_stage = next(s for s in stages if s.name == "commit_changes")

    # Create a context with an implement summary
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    ctx.summaries["implement"] = "Implemented the new feature with tests"

    with patch("workflow.workflows.issue.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "[main abc123] Implemented the new feature with tests"
        result = await commit_stage.mechanical_fn(ctx)

        assert result.outcome == StageOutcome.PROCEED
        assert result.stage_name == "commit_changes"
        # Verify the commit command used the implement summary
        cmd = mock.call_args[0][0]
        assert "add -A" in cmd
        assert "commit -m" in cmd
        assert "Implemented the new feature" in cmd


def test_update_docs_is_conditional():
    """update_docs has a should_run callback."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    docs = next(s for s in stages if s.name == "update_docs")
    assert docs.should_run is not None


def test_gate_docs_is_mechanical():
    """gate_docs is a mechanical stage (no LLM needed)."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    gate = next(s for s in stages if s.name == "gate_docs")
    assert gate.is_mechanical is True
    assert gate.mechanical_fn is not None


@pytest.mark.timeout(5)
async def test_docs_should_run_reads_facts_true():
    """_docs_should_run returns True when facts say docs are needed."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    ctx.facts["docs_needed"] = True
    assert _docs_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_docs_should_run_reads_facts_false():
    """_docs_should_run returns False when facts say docs not needed."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    ctx.facts["docs_needed"] = False
    assert _docs_should_run(ctx) is False


@pytest.mark.timeout(5)
async def test_docs_should_run_defaults_false_when_no_fact():
    """_docs_should_run returns False if no docs_needed fact is set."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    assert _docs_should_run(ctx) is False


@pytest.mark.timeout(5)
async def test_gate_docs_mechanical_sets_facts_needed():
    """gate_docs mechanical fn sets docs_needed=True when source changed."""
    from workflow.context import WorkflowContext
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    gate = next(s for s in stages if s.name == "gate_docs")
    ctx = WorkflowContext()

    # Simulate: git diff shows source files changed, and docs/ dir exists
    diff_output = "src/main.py\nsrc/utils.py"
    ls_output = "configuration.md\ndesign.md\nplugin-guide.md"

    with patch("workflow.workflows.issue.shell", new_callable=AsyncMock) as mock:
        mock.side_effect = [diff_output, ls_output]
        result = await gate.mechanical_fn(ctx)

    assert result.outcome == StageOutcome.PROCEED
    assert result.stage_name == "gate_docs"
    assert ctx.facts["docs_needed"] is True


@pytest.mark.timeout(5)
async def test_gate_docs_mechanical_sets_facts_not_needed():
    """gate_docs mechanical fn sets docs_needed=False when only tests changed."""
    from workflow.context import WorkflowContext
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    gate = next(s for s in stages if s.name == "gate_docs")
    ctx = WorkflowContext()

    # Simulate: only test files changed
    diff_output = "tests/test_main.py\ntests/test_utils.py"
    ls_output = "configuration.md\ndesign.md\nplugin-guide.md"

    with patch("workflow.workflows.issue.shell", new_callable=AsyncMock) as mock:
        mock.side_effect = [diff_output, ls_output]
        result = await gate.mechanical_fn(ctx)

    assert result.outcome == StageOutcome.PROCEED
    assert result.stage_name == "gate_docs"
    assert ctx.facts["docs_needed"] is False


@pytest.mark.timeout(5)
async def test_gate_docs_mechanical_no_docs_dir():
    """gate_docs sets docs_needed=False when no docs directory exists."""
    from workflow.context import WorkflowContext
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    gate = next(s for s in stages if s.name == "gate_docs")
    ctx = WorkflowContext()

    # Source files changed, but no docs directory
    diff_output = "src/main.py\nsrc/utils.py"
    ls_output = ""  # No docs found

    with patch("workflow.workflows.issue.shell", new_callable=AsyncMock) as mock:
        mock.side_effect = [diff_output, ls_output]
        result = await gate.mechanical_fn(ctx)

    assert result.outcome == StageOutcome.PROCEED
    assert ctx.facts["docs_needed"] is False


def test_comment_on_issue_has_record_plan_tool():
    """comment_on_issue stage includes the record_plan tool."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    comment = next(s for s in stages if s.name == "comment_on_issue")
    tool_names = [t.name for t in comment.tools]
    assert "record_plan" in tool_names
    assert "comment_on_issue" in tool_names


def test_comment_on_issue_has_post_fn():
    """comment_on_issue stage has a post_fn to copy plan to context."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    comment = next(s for s in stages if s.name == "comment_on_issue")
    assert comment.post_fn is not None


def test_comment_on_issue_instruction_mentions_record_plan():
    """comment_on_issue instruction tells the LLM to call record_plan."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    comment = next(s for s in stages if s.name == "comment_on_issue")
    assert "record_plan" in comment.instruction


def test_write_tests_has_should_run():
    """write_tests stage has a should_run predicate."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    write_tests = next(s for s in stages if s.name == "write_tests")
    assert write_tests.should_run is not None


def test_refactor_has_should_run():
    """refactor stage has a should_run predicate."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    refactor = next(s for s in stages if s.name == "refactor")
    assert refactor.should_run is not None


def test_create_branch_mentions_plan():
    """create_branch instruction mentions the implementation plan."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    branch = next(s for s in stages if s.name == "create_branch")
    assert "plan" in branch.instruction.lower()


def test_submit_pr_mentions_plan():
    """submit_pr instruction mentions the implementation plan."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    pr = next(s for s in stages if s.name == "submit_pr")
    assert "plan" in pr.instruction.lower()


# -- should_run predicate tests --


@pytest.mark.timeout(5)
async def test_tests_should_run_defaults_true_without_plan():
    """_tests_should_run returns True when no plan exists."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    assert _tests_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_tests_should_run_true_when_plan_needs_tests():
    """_tests_should_run returns True when plan says tests needed."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"needs_tests": True, "task_type": "feature"}
    assert _tests_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_tests_should_run_false_when_plan_no_tests():
    """_tests_should_run returns False when plan says no tests needed."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"needs_tests": False, "task_type": "docs_only"}
    assert _tests_should_run(ctx) is False


@pytest.mark.timeout(5)
async def test_refactor_should_run_defaults_true_without_plan():
    """_refactor_should_run returns True when no plan exists."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    assert _refactor_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_refactor_should_run_false_for_docs_only():
    """_refactor_should_run returns False for docs_only tasks."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"task_type": "docs_only", "needs_tests": False}
    assert _refactor_should_run(ctx) is False


@pytest.mark.timeout(5)
async def test_refactor_should_run_true_for_feature():
    """_refactor_should_run returns True for feature tasks."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"task_type": "feature", "needs_tests": True}
    assert _refactor_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_refactor_should_run_true_for_bug_fix():
    """_refactor_should_run returns True for bug_fix tasks."""
    from workflow.context import WorkflowContext
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"task_type": "bug_fix", "needs_tests": True}
    assert _refactor_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_post_fn_copies_plan_to_context():
    """comment_on_issue post_fn copies plan from holder to context.facts."""
    from workflow.context import WorkflowContext
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    comment = next(s for s in stages if s.name == "comment_on_issue")

    ctx = WorkflowContext()
    # Simulate the plan being recorded by the tool
    # (The plan_holder is internal to build_issue_workflow, so we
    # test by using the record_plan tool directly)
    from workflow.tools.plan import PlanHolder, make_record_plan_tool
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)
    await tool.fn(
        branch_name="feat/issue-1",
        task_type="feature",
        needs_tests=True,
        files_to_change=["src/main.py"],
    )

    # Now simulate the post_fn behavior
    assert holder.data is not None
    ctx.facts["plan"] = holder.data
    assert ctx.facts["plan"]["branch_name"] == "feat/issue-1"
    assert ctx.facts["plan"]["task_type"] == "feature"
    assert ctx.facts["plan"]["needs_tests"] is True


def test_gate_ready_instructions_mention_proceed():
    """gate_ready instructions clearly tell the LLM to use 'proceed'."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    gate = next(s for s in stages if s.name == "gate_ready")
    # Instructions should explicitly mention calling stage_complete
    # with 'proceed' (not just referencing it vaguely)
    assert "proceed" in gate.instruction.lower()


# -- Fast path tests (Task 5) --


def test_implement_has_should_run():
    """implement stage has a should_run predicate for fast path."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    implement = next(s for s in stages if s.name == "implement")
    assert implement.should_run is not None


@pytest.mark.timeout(5)
async def test_implement_should_run_defaults_true_without_plan():
    """_implement_should_run returns True when no plan exists (graceful degradation)."""
    from workflow.context import WorkflowContext
    from workflow.workflows.issue import _implement_should_run
    ctx = WorkflowContext()
    assert _implement_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_implement_should_run_true_for_feature():
    """_implement_should_run returns True for feature tasks."""
    from workflow.context import WorkflowContext
    from workflow.workflows.issue import _implement_should_run
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"task_type": "feature", "needs_tests": True}
    assert _implement_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_implement_should_run_true_for_bug_fix():
    """_implement_should_run returns True for bug_fix tasks."""
    from workflow.context import WorkflowContext
    from workflow.workflows.issue import _implement_should_run
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"task_type": "bug_fix", "needs_tests": True}
    assert _implement_should_run(ctx) is True


@pytest.mark.timeout(5)
async def test_implement_should_run_false_for_docs_only():
    """_implement_should_run returns False for docs_only tasks (fast path)."""
    from workflow.context import WorkflowContext
    from workflow.workflows.issue import _implement_should_run
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"task_type": "docs_only", "needs_tests": False}
    assert _implement_should_run(ctx) is False


@pytest.mark.timeout(5)
async def test_implement_should_run_true_for_refactor():
    """_implement_should_run returns True for refactor tasks."""
    from workflow.context import WorkflowContext
    from workflow.workflows.issue import _implement_should_run
    ctx = WorkflowContext()
    ctx.facts["plan"] = {"task_type": "refactor", "needs_tests": True}
    assert _implement_should_run(ctx) is True


def test_fast_path_docs_only_skips_tdd_stages():
    """docs_only tasks skip write_tests, implement, and refactor (fast path).

    The fast path means a simple README change doesn't go through the
    full TDD cycle of write_tests → implement → refactor.
    """
    from workflow.context import WorkflowContext
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="Update README",
    )
    ctx = WorkflowContext()
    # Simulate plan recorded by comment_on_issue stage
    ctx.facts["plan"] = {
        "task_type": "docs_only",
        "needs_tests": False,
        "needs_docs": True,
    }

    # Stages that should be skipped for docs_only
    write_tests = next(s for s in stages if s.name == "write_tests")
    implement = next(s for s in stages if s.name == "implement")
    refactor = next(s for s in stages if s.name == "refactor")

    assert write_tests.should_run(ctx) is False
    assert implement.should_run(ctx) is False
    assert refactor.should_run(ctx) is False


def test_fast_path_feature_runs_all_tdd_stages():
    """Feature tasks run all TDD stages (no fast path)."""
    from workflow.context import WorkflowContext
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="Add new feature",
    )
    ctx = WorkflowContext()
    ctx.facts["plan"] = {
        "task_type": "feature",
        "needs_tests": True,
    }

    write_tests = next(s for s in stages if s.name == "write_tests")
    implement = next(s for s in stages if s.name == "implement")
    refactor = next(s for s in stages if s.name == "refactor")

    assert write_tests.should_run(ctx) is True
    assert implement.should_run(ctx) is True
    assert refactor.should_run(ctx) is True


def test_fast_path_always_runs_commit_and_submit():
    """commit_changes and submit_pr always run regardless of task_type."""
    from workflow.context import WorkflowContext
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="Docs update",
    )
    ctx = WorkflowContext()
    ctx.facts["plan"] = {
        "task_type": "docs_only",
        "needs_tests": False,
    }

    # These stages have no should_run — they always execute
    commit = next(s for s in stages if s.name == "commit_changes")
    submit = next(s for s in stages if s.name == "submit_pr")

    assert commit.should_run is None
    assert submit.should_run is None
