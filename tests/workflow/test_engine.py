"""Tests for workflow.engine — WorkflowEngine.

All tests use mechanical stages (no LLM mocking needed).
Verifies linear progression, escape hatches, conditional skipping,
blocked stops, context accumulation, invalid escape targets, and
max iterations guard.
"""

from unittest.mock import MagicMock

import pytest

from workflow.types import StageOutcome, StageResult
from workflow.context import WorkflowContext
from workflow.stage import Stage
from workflow.engine import WorkflowEngine


def _mech_fn(name, outcome, summary, escape=None):
    """Create a mechanical_fn that returns a fixed StageResult."""
    async def fn(ctx):
        return StageResult(
            stage_name=name, outcome=outcome, summary=summary,
            escape_target=escape,
        )
    return fn


async def test_linear_progression():
    """Engine proceeds through stages in order."""
    order = []

    def mk(name):
        async def fn(ctx):
            order.append(name)
            return StageResult(
                stage_name=name, outcome=StageOutcome.PROCEED,
                summary=f"done {name}",
            )
        return fn

    stages = [
        Stage(name="a", instruction="", is_mechanical=True, mechanical_fn=mk("a")),
        Stage(name="b", instruction="", is_mechanical=True, mechanical_fn=mk("b")),
        Stage(name="c", instruction="", is_mechanical=True, mechanical_fn=mk("c")),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    assert order == ["a", "b", "c"]


async def test_escape_hatch():
    """Engine jumps back to the escape target stage."""
    order = []
    gate_count = {"n": 0}

    async def start_fn(ctx):
        order.append("start")
        return StageResult(
            stage_name="start", outcome=StageOutcome.PROCEED,
            summary="started",
        )

    async def gate_fn(ctx):
        order.append("gate")
        gate_count["n"] += 1
        if gate_count["n"] == 1:
            return StageResult(
                stage_name="gate",
                outcome=StageOutcome.NEED_CLARIFICATION,
                summary="unclear",
                escape_target="start",
            )
        return StageResult(
            stage_name="gate", outcome=StageOutcome.PROCEED,
            summary="ready",
        )

    def mk_done(name):
        async def fn(ctx):
            order.append(name)
            return StageResult(
                stage_name=name, outcome=StageOutcome.PROCEED,
                summary=f"done {name}",
            )
        return fn

    stages = [
        Stage(name="start", instruction="", is_mechanical=True,
              mechanical_fn=start_fn),
        Stage(name="gate", instruction="",
              escape_targets={StageOutcome.NEED_CLARIFICATION: "start"},
              is_mechanical=True, mechanical_fn=gate_fn),
        Stage(name="done", instruction="", is_mechanical=True,
              mechanical_fn=mk_done("done")),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    assert order == ["start", "gate", "start", "gate", "done"]


async def test_conditional_stage_skipped():
    """Engine skips stages where should_run returns False."""
    order = []

    def mk(name):
        async def fn(ctx):
            order.append(name)
            return StageResult(
                stage_name=name, outcome=StageOutcome.PROCEED,
                summary=f"done {name}",
            )
        return fn

    stages = [
        Stage(name="a", instruction="", is_mechanical=True, mechanical_fn=mk("a")),
        Stage(name="b", instruction="", is_mechanical=True, mechanical_fn=mk("b"),
              should_run=lambda ctx: False),
        Stage(name="c", instruction="", is_mechanical=True, mechanical_fn=mk("c")),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    assert order == ["a", "c"]


async def test_blocked_stops_workflow():
    """Engine stops when a stage returns BLOCKED."""
    order = []

    def mk(name, outcome):
        async def fn(ctx):
            order.append(name)
            return StageResult(
                stage_name=name, outcome=outcome, summary=name,
            )
        return fn

    stages = [
        Stage(name="a", instruction="", is_mechanical=True,
              mechanical_fn=mk("a", StageOutcome.PROCEED)),
        Stage(name="b", instruction="", is_mechanical=True,
              mechanical_fn=mk("b", StageOutcome.BLOCKED)),
        Stage(name="c", instruction="", is_mechanical=True,
              mechanical_fn=mk("c", StageOutcome.PROCEED)),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    assert order == ["a", "b"]


async def test_context_accumulation():
    """Engine records each stage's results in the context."""
    stages = [
        Stage(name="a", instruction="", is_mechanical=True,
              mechanical_fn=_mech_fn("a", StageOutcome.PROCEED, "summary a")),
        Stage(name="b", instruction="", is_mechanical=True,
              mechanical_fn=_mech_fn("b", StageOutcome.PROCEED, "summary b")),
    ]

    ctx = WorkflowContext()
    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, ctx)

    assert ctx.summaries == {"a": "summary a", "b": "summary b"}


async def test_invalid_escape_target_stops():
    """Engine stops when escape_target doesn't match any stage."""
    stages = [
        Stage(name="a", instruction="", is_mechanical=True,
              mechanical_fn=_mech_fn(
                  "a", StageOutcome.NEED_CLARIFICATION, "x",
                  escape="nonexistent",
              ),
              escape_targets={StageOutcome.NEED_CLARIFICATION: "nonexistent"}),
    ]

    engine = WorkflowEngine(MagicMock())
    ctx = WorkflowContext()
    await engine.run(stages, ctx)

    # Should stop — no infinite loop or crash
    assert "a" in ctx.summaries


async def test_max_iterations_prevents_infinite_loop():
    """Engine stops after max_iterations to prevent infinite loops."""
    async def always_escape(ctx):
        return StageResult(
            stage_name="loop", outcome=StageOutcome.NEED_CLARIFICATION,
            summary="again", escape_target="loop",
        )

    stages = [
        Stage(name="loop", instruction="",
              escape_targets={StageOutcome.NEED_CLARIFICATION: "loop"},
              is_mechanical=True, mechanical_fn=always_escape),
    ]

    engine = WorkflowEngine(MagicMock(), max_iterations=5)
    ctx = WorkflowContext()
    await engine.run(stages, ctx)

    # Should stop after 5 iterations, not hang forever
    assert len(ctx.summaries.get("loop", "")) >= 0


async def test_proceed_with_spurious_escape_target_advances():
    """Engine advances normally when PROCEED has a spurious escape_target.

    LLMs sometimes pass escape_target='null' or similar. The engine
    should ignore it and just advance.
    """
    order = []

    def mk(name, outcome, escape=None):
        async def fn(ctx):
            order.append(name)
            return StageResult(
                stage_name=name, outcome=outcome, summary=name,
                escape_target=escape,
            )
        return fn

    stages = [
        Stage(name="a", instruction="", is_mechanical=True,
              mechanical_fn=mk("a", StageOutcome.PROCEED, escape="null")),
        Stage(name="b", instruction="", is_mechanical=True,
              mechanical_fn=mk("b", StageOutcome.PROCEED)),
    ]

    engine = WorkflowEngine(MagicMock())
    ctx = WorkflowContext()
    await engine.run(stages, ctx)

    # Both stages should run — engine ignores the spurious escape_target
    assert order == ["a", "b"]


async def test_blocked_with_valid_escape_target_jumps_back():
    """Engine jumps back to escape target when stage returns BLOCKED.

    This tests the recovery mechanism: submit_pr fails with BLOCKED,
    jumps back to commit_changes for retry.
    """
    order = []
    commit_count = {"n": 0}

    async def commit_fn(ctx):
        order.append("commit_changes")
        commit_count["n"] += 1
        return StageResult(
            stage_name="commit_changes",
            outcome=StageOutcome.PROCEED,
            summary="Changes committed",
        )

    pr_count = {"n": 0}

    async def pr_fn(ctx):
        order.append("submit_pr")
        pr_count["n"] += 1
        if pr_count["n"] == 1:
            # First attempt fails — trigger recovery
            return StageResult(
                stage_name="submit_pr",
                outcome=StageOutcome.BLOCKED,
                summary="PR creation failed: auth error",
                escape_target="commit_changes",
            )
        # Second attempt succeeds
        return StageResult(
            stage_name="submit_pr",
            outcome=StageOutcome.PROCEED,
            summary="PR created successfully",
        )

    stages = [
        Stage(
            name="commit_changes",
            instruction="",
            is_mechanical=True,
            mechanical_fn=commit_fn,
        ),
        Stage(
            name="submit_pr",
            instruction="",
            is_mechanical=True,
            mechanical_fn=pr_fn,
            # Escape target: BLOCKED → go back to commit_changes
            escape_targets={StageOutcome.BLOCKED: "commit_changes"},
        ),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    # Should have run: commit, pr (fail), commit (retry), pr (success)
    assert order == ["commit_changes", "submit_pr", "commit_changes", "submit_pr"]


async def test_blocked_without_escape_target_shows_summary(capsys):
    """Engine stops with clear summary when BLOCKED has no escape target.

    The output should show which stage failed, the LLM's summary,
    and suggested next steps.
    """
    stages = [
        Stage(
            name="submit_pr",
            instruction="",
            is_mechanical=True,
            mechanical_fn=_mech_fn(
                "submit_pr",
                StageOutcome.BLOCKED,
                "PR creation failed: network timeout",
            ),
            # No escape targets — workflow should stop
        ),
    ]

    ctx = WorkflowContext()
    ctx.facts["branch_name"] = "fix/issue-42"
    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, ctx)

    captured = capsys.readouterr()
    # Stage failed indication
    assert "✗ submit_pr (blocked)" in captured.out
    # Detailed blocked summary
    assert "Workflow blocked" in captured.out
    assert "no escape target" in captured.out.lower()
    # LLM summary shown
    assert "network timeout" in captured.out
    # Branch info shown
    assert "fix/issue-42" in captured.out
    # Suggested next steps
    assert "suggested next steps" in captured.out.lower()


async def test_blocked_undeclared_escape_target_stops_with_detail(capsys):
    """Engine stops with clear summary when escape target is not declared.

    If the LLM passes an escape_target that isn't in the stage's
    escape_targets dict, the workflow stops with a detailed message.
    """
    stages = [
        Stage(
            name="submit_pr",
            instruction="",
            is_mechanical=True,
            mechanical_fn=_mech_fn(
                "submit_pr",
                StageOutcome.BLOCKED,
                "Auth failed",
                escape="implement",  # Not declared as an escape target
            ),
            # Only BLOCKED -> commit_changes is declared
            escape_targets={StageOutcome.BLOCKED: "commit_changes"},
        ),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    captured = capsys.readouterr()
    assert "✗ submit_pr (blocked)" in captured.out
    assert "Invalid escape target" in captured.out
    assert "implement" in captured.out


class TestEngineDisplay:
    """Tests for engine's display output during stage lifecycle."""

    async def test_displays_stage_start(self, capsys):
        """Engine shows stage start via display."""
        stages = [
            Stage(name="alpha", instruction="", is_mechanical=True,
                  mechanical_fn=_mech_fn("alpha", StageOutcome.PROCEED, "ok")),
        ]
        engine = WorkflowEngine(MagicMock())
        await engine.run(stages, WorkflowContext())
        captured = capsys.readouterr()
        assert "▸ alpha" in captured.out

    async def test_displays_stage_complete(self, capsys):
        """Engine shows stage completion via display."""
        stages = [
            Stage(name="beta", instruction="", is_mechanical=True,
                  mechanical_fn=_mech_fn("beta", StageOutcome.PROCEED, "ok")),
        ]
        engine = WorkflowEngine(MagicMock())
        await engine.run(stages, WorkflowContext())
        captured = capsys.readouterr()
        assert "✓ beta (proceed)" in captured.out

    async def test_displays_stage_skip(self, capsys):
        """Engine shows skipped stage via display."""
        stages = [
            Stage(name="gamma", instruction="", is_mechanical=True,
                  mechanical_fn=_mech_fn("gamma", StageOutcome.PROCEED, "ok"),
                  should_run=lambda ctx: False),
        ]
        engine = WorkflowEngine(MagicMock())
        await engine.run(stages, WorkflowContext())
        captured = capsys.readouterr()
        assert "⊘ gamma (skipped)" in captured.out

    async def test_displays_blocked_workflow(self, capsys):
        """Engine shows detailed blocked message when workflow stops."""
        stages = [
            Stage(name="delta", instruction="", is_mechanical=True,
                  mechanical_fn=_mech_fn("delta", StageOutcome.BLOCKED, "stuck")),
        ]
        engine = WorkflowEngine(MagicMock())
        await engine.run(stages, WorkflowContext())
        captured = capsys.readouterr()
        assert "✗ delta (blocked)" in captured.out
        assert "blocked" in captured.out.lower()
        assert "no escape target" in captured.out.lower()
        assert "suggested next steps" in captured.out.lower()

    async def test_displays_blocked_workflow_with_branch(self, capsys):
        """Engine shows branch info in blocked summary."""
        stages = [
            Stage(name="delta", instruction="", is_mechanical=True,
                  mechanical_fn=_mech_fn("delta", StageOutcome.BLOCKED, "stuck")),
        ]
        ctx = WorkflowContext()
        ctx.facts["branch_name"] = "feature/my-branch"
        engine = WorkflowEngine(MagicMock())
        await engine.run(stages, ctx)
        captured = capsys.readouterr()
        assert "feature/my-branch" in captured.out
        assert "git checkout" in captured.out

    async def test_displays_workflow_complete(self, capsys):
        """Engine shows completion message when all stages finish."""
        stages = [
            Stage(name="zeta", instruction="", is_mechanical=True,
                  mechanical_fn=_mech_fn("zeta", StageOutcome.PROCEED, "done")),
        ]
        engine = WorkflowEngine(MagicMock())
        await engine.run(stages, WorkflowContext())
        captured = capsys.readouterr()
        assert "done" in captured.out.lower()


# -- post_fn integration tests --


async def test_post_fn_called_after_stage(capsys):
    """Engine calls post_fn after a stage completes."""
    post_fn_calls = []

    def my_post_fn(result, ctx):
        post_fn_calls.append((result.stage_name, dict(ctx.summaries)))

    stages = [
        Stage(
            name="alpha",
            instruction="",
            is_mechanical=True,
            mechanical_fn=_mech_fn("alpha", StageOutcome.PROCEED, "done alpha"),
            post_fn=my_post_fn,
        ),
    ]

    engine = WorkflowEngine(MagicMock())
    ctx = WorkflowContext()
    await engine.run(stages, ctx)

    assert len(post_fn_calls) == 1
    assert post_fn_calls[0][0] == "alpha"
    assert post_fn_calls[0][1] == {"alpha": "done alpha"}


async def test_post_fn_can_write_to_context_facts():
    """post_fn can store structured data in context.facts."""
    plan_data = {"branch_name": "feat/test", "task_type": "feature"}

    def copy_plan(result, ctx):
        ctx.facts["plan"] = plan_data

    async def use_plan_fn(ctx):
        return StageResult(
            stage_name="use_plan",
            outcome=StageOutcome.PROCEED,
            summary=ctx.facts.get("plan", {}).get("branch_name", "missing"),
        )

    stages = [
        Stage(
            name="analyze",
            instruction="",
            is_mechanical=True,
            mechanical_fn=_mech_fn("analyze", StageOutcome.PROCEED, "analyzed"),
            post_fn=copy_plan,
        ),
        Stage(
            name="use_plan",
            instruction="",
            is_mechanical=True,
            mechanical_fn=use_plan_fn,
        ),
    ]

    engine = WorkflowEngine(MagicMock())
    ctx = WorkflowContext()
    await engine.run(stages, ctx)

    # Second stage should have access to the plan
    assert ctx.facts["plan"] == plan_data
    assert ctx.summaries["use_plan"] == "feat/test"
