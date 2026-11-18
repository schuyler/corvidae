"""Planning lens plugin tests.

Verifies that the PlanningLensPlugin:
- contributes a valid lens through the on_critique_init hook
- has the expected shape (name, sample_rate, no patterns, no thresholds)
- implements transform_critique_prompt to append the goal list
- the fork prompt mentions the action categories and null
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from corvidae_planning import PlanningLensPlugin


def _run_init(plugin):
    """Run on_critique_init and return the collected lenses."""
    async def _run():
        collected = []

        def add_lens(lens):
            collected.append(lens)

        await plugin.on_critique_init({}, add_lens)
        return collected

    return asyncio.run(_run())


def _make_plugin(goals=None):
    """Create a PlanningLensPlugin with a mocked GoalsPlugin."""
    plugin = PlanningLensPlugin()
    mock_goals = AsyncMock()
    if goals is not None:
        mock_goals.open_goals.return_value = goals
    else:
        mock_goals.open_goals.return_value = []
    plugin._goals = mock_goals
    return plugin


class TestPlanningLensPlugin:
    """The planning lens plugin contributes a valid lens."""

    def test_contributes_a_lens(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert len(lenses) == 1

    def test_lens_name(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert lenses[0].name == "planning"

    def test_depends_on_critique_and_goals(self):
        assert PlanningLensPlugin.depends_on == frozenset({"critique", "goals"})

    def test_lens_has_no_reply_patterns(self):
        """Sample-rate trigger: no keyword gate."""
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert lenses[0].reply_patterns == ()

    def test_lens_has_no_thresholds(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert lenses[0].thresholds == ()

    def test_lens_sample_rate(self):
        """10% sample rate, like the Grice lens."""
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert lenses[0].sample_rate == 0.1

    def test_lens_fork_prompt_is_nonempty(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert lenses[0].fork_prompt.strip()

    def test_lens_fork_prompt_mentions_goal_actions(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        prompt = lenses[0].fork_prompt
        assert "create" in prompt
        assert "update" in prompt
        assert "retire" in prompt

    def test_lens_fork_prompt_mentions_null(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert "null" in lenses[0].fork_prompt

    def test_lens_fork_prompt_requires_json(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert "JSON" in lenses[0].fork_prompt

    def test_lens_fork_prompt_mentions_goal_list(self):
        """The fork prompt references the goal list."""
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert "goal list" in lenses[0].fork_prompt

    def test_lens_fork_prompt_has_confidence_threshold(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert "80%" in lenses[0].fork_prompt


class TestTransformCritiquePrompt:
    """The transform_critique_prompt hook appends the goal list."""

    def test_appends_goal_list(self):
        plugin = _make_plugin(goals=[
            {"id": 181, "text": "Backfill memory records"},
            {"id": 186, "text": "Build planning lens"},
        ])

        async def _run():
            return await plugin.transform_critique_prompt(
                channel=None, trace_id="test",
                lens_name="planning",
                instruction="Base instruction.",
            )

        result = asyncio.run(_run())
        assert result is not None
        assert "Base instruction." in result
        assert "#181: Backfill memory records" in result
        assert "#186: Build planning lens" in result
        assert "Current goal list:" in result

    def test_empty_goal_list(self):
        plugin = _make_plugin(goals=[])

        async def _run():
            return await plugin.transform_critique_prompt(
                channel=None, trace_id="test",
                lens_name="planning",
                instruction="Base instruction.",
            )

        result = asyncio.run(_run())
        assert result is not None
        assert "(empty" in result

    def test_non_planning_lens_returns_none(self):
        plugin = _make_plugin(goals=[{"id": 1, "text": "test"}])

        async def _run():
            return await plugin.transform_critique_prompt(
                channel=None, trace_id="test",
                lens_name="grice",
                instruction="Base instruction.",
            )

        result = asyncio.run(_run())
        assert result is None

    def test_goals_fetch_failure_returns_none(self):
        plugin = PlanningLensPlugin()
        mock_goals = AsyncMock()
        mock_goals.open_goals.side_effect = Exception("db error")
        plugin._goals = mock_goals

        async def _run():
            return await plugin.transform_critique_prompt(
                channel=None, trace_id="test",
                lens_name="planning",
                instruction="Base instruction.",
            )

        result = asyncio.run(_run())
        assert result is None

    def test_preserves_base_instruction(self):
        plugin = _make_plugin(goals=[{"id": 42, "text": "test goal"}])

        async def _run():
            return await plugin.transform_critique_prompt(
                channel=None, trace_id="test",
                lens_name="planning",
                instruction="Original text here.",
            )

        result = asyncio.run(_run())
        assert result.startswith("Original text here.")
