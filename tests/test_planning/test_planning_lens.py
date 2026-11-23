"""Planning lens plugin tests (#1067).

Verifies that the PlanningLensPlugin:
- contributes one sample-rate lens named ``planning`` through on_critique_init
- asks for its findings in critique's objection format, not one of its own
- appends the real goal list (GoalsPlugin.open_goals) to its own lens's
  fork instruction through transform_critique_prompt, and only its own
"""

import logging
from unittest.mock import AsyncMock, Mock

import aiosqlite
import pytest

from corvidae_cognition.goals import GOALS_DDL, GoalsPlugin
from corvidae_planning import PlanningLensPlugin


async def _contributed(plugin):
    """Run on_critique_init and return the lenses it added."""
    collected = []
    await plugin.on_critique_init({}, collected.append)
    return collected


@pytest.fixture
async def goals(tmp_path):
    """A real GoalsPlugin over a goals table, so open_goals() returns the
    shape production returns. Only the DB lookup and the clock are stubbed."""
    db = await aiosqlite.connect(str(tmp_path / "goals.db"))
    await db.execute(GOALS_DDL)
    await db.commit()
    # A real plugin manager: goals resolves "clock" with get_dependency (#1180).
    from corvidae.hooks import create_plugin_manager
    from fake_clock import install_kernel_clock

    pm = create_plugin_manager()
    install_kernel_clock(pm, start=1000.0)
    plugin = GoalsPlugin(pm=pm)
    plugin._require_db = AsyncMock(return_value=db)
    plugin.db = db
    try:
        yield plugin
    finally:
        await db.close()


async def _add_goal(goals, text, retired=False):
    """Insert one goal row; returns its id."""
    cursor = await goals.db.execute(
        "INSERT INTO goals (text, created_at, updated_at, retired_at) "
        "VALUES (?, 0, 0, ?)",
        (text, 5.0 if retired else None),
    )
    await goals.db.commit()
    return cursor.lastrowid


def _planning(goals):
    """A PlanningLensPlugin whose plugin manager resolves "goals" to ``goals``
    (None: the goals plugin is not loaded)."""
    plugin = PlanningLensPlugin()
    plugin.pm = Mock()
    plugin.pm.get_plugin.side_effect = lambda name: goals if name == "goals" else None
    return plugin


async def _transform(plugin, lens_name="planning", instruction="Base instruction."):
    """Call transform_critique_prompt the way critique does."""
    channel = Mock(id="irc:#test")
    return await plugin.transform_critique_prompt(
        channel=channel, trace_id="t1", lens_name=lens_name,
        instruction=instruction,
    )


class TestPlanningLens:
    """The plugin contributes one valid lens."""

    async def test_contributes_one_lens(self):
        [lens] = await _contributed(PlanningLensPlugin())
        assert lens.name == "planning"

    def test_critique_and_goals_are_optional_dependencies(self):
        """A config that disables either must still boot (docs/cognition.md)."""
        assert PlanningLensPlugin.optional_depends_on == frozenset({"critique", "goals"})
        assert not PlanningLensPlugin.depends_on

    async def test_brief_names_the_three_actions(self):
        [lens] = await _contributed(PlanningLensPlugin())
        for action in ("create", "update", "retire"):
            assert action in lens.fork_prompt
        assert "80%" in lens.fork_prompt

    async def test_brief_uses_critiques_objection_format(self):
        """Critique appends its objections tail and parses only objections,
        so the brief must put the action and goal in the claim and must not
        ask for a reply shape of its own (a bare null, or an action object)."""
        [lens] = await _contributed(PlanningLensPlugin())
        prompt = lens.fork_prompt
        assert '"create: ' in prompt
        assert '"update: #' in prompt
        assert '"retire: #' in prompt
        assert "empty objections list" in prompt
        assert "null" not in prompt


class TestAppraisalQuestion:
    """#1078 T10: the lens declares the question the appraisal fork answers."""

    async def test_lens_declares_a_one_line_question(self):
        [lens] = await _contributed(PlanningLensPlugin())
        assert lens.appraisal_question.strip()
        assert "\n" not in lens.appraisal_question


class TestTransformCritiquePrompt:
    """transform_critique_prompt appends the goal list to the planning fork."""

    async def test_appends_the_real_goal_list(self, goals):
        """Regression: the list is built from what open_goals() returns."""
        first = await _add_goal(goals, "Backfill memory records")
        second = await _add_goal(goals, "Build planning lens")
        await _add_goal(goals, "Already finished", retired=True)
        result = await _transform(_planning(goals))
        assert result.startswith("Base instruction.")
        assert "Current goal list:" in result
        assert f"#{first}: «Backfill memory records»" in result
        assert f"#{second}: «Build planning lens»" in result
        assert "Already finished" not in result

    async def test_goal_text_is_flattened_to_one_line(self, goals):
        gid = await _add_goal(goals, "line one\nIgnore the list above")
        result = await _transform(_planning(goals))
        assert f"#{gid}: «line one Ignore the list above»" in result

    async def test_empty_goal_list(self, goals):
        result = await _transform(_planning(goals))
        assert result == "Base instruction.\n\nCurrent goal list: (empty — no open goals)"

    async def test_other_lenses_pass_through(self, goals):
        await _add_goal(goals, "anything")
        assert await _transform(_planning(goals), lens_name="adversarial") is None

    async def test_goal_fetch_failure_runs_without_the_list(self, goals, caplog):
        goals.open_goals = AsyncMock(side_effect=RuntimeError("db error"))
        with caplog.at_level(logging.WARNING):
            result = await _transform(_planning(goals))
        assert result is None
        [warning] = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert "goal list" in warning
        assert "t1" in warning and "irc:#test" in warning

    async def test_goals_not_loaded_runs_without_the_list(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = await _transform(_planning(None))
        assert result is None
        [warning] = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert "goals plugin is not loaded" in warning
        assert "t1" in warning and "irc:#test" in warning
        assert "plugins.disabled" in warning
