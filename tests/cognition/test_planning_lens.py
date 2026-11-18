"""Planning lens plugin tests.

Verifies that the PlanningLensPlugin contributes a valid lens through
the on_critique_init hook, and that the lens has the expected shape:
- name is "planning"
- no reply_patterns (sample-rate trigger, not keyword gate)
- no thresholds (doesn't use the appraisal vector)
- sample_rate is 0.1 (10% of turns, like Grice)
- fork_prompt is non-empty and mentions the action categories
"""

import asyncio

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

    def test_depends_on_critique(self):
        assert PlanningLensPlugin.depends_on == frozenset({"critique"})

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

    def test_lens_fork_prompt_no_goal_list(self):
        """The fork does not have access to the goal list."""
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert "goal list" in lenses[0].fork_prompt

    def test_lens_fork_prompt_has_confidence_threshold(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert "80%" in lenses[0].fork_prompt
