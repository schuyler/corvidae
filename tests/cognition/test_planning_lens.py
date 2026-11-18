"""Planning lens plugin tests.

Verifies that the PlanningLensPlugin contributes a valid lens through
the on_critique_init hook, and that the lens has the expected shape:
- name is "planning"
- reply_patterns are the goal-signal regexes
- no thresholds (doesn't use the appraisal vector)
- sample_rate is 0.0 (pattern-only sentinel)
- fork_prompt is non-empty and mentions the action categories
"""

import asyncio
import re

import pytest

from corvidae_planning import PlanningLensPlugin, _GOAL_SIGNALS


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

    def test_lens_has_reply_patterns(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert len(lenses[0].reply_patterns) > 0
        # All patterns should compile
        for pattern in lenses[0].reply_patterns:
            re.compile(pattern)

    def test_lens_has_no_thresholds(self):
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert lenses[0].thresholds == ()

    def test_lens_sample_rate_is_pattern_only_sentinel(self):
        """sample_rate=0.0 means the lens runs only when a reply pattern
        matches; the sampling draw (which applies when no pattern matches)
        is at 0.0, so the lens never runs without a pattern match."""
        plugin = PlanningLensPlugin()
        lenses = _run_init(plugin)
        assert lenses[0].sample_rate == 0.0

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


class TestGoalSignalPatterns:
    """The goal-signal regexes match the expected phrases."""

    @pytest.mark.parametrize("text,should_match", [
        # Commitments
        ("I'll pick up the backfill tomorrow", True),
        ("I will start working on the lens", True),
        ("We'll handle the deployment", True),
        ("I'm going to write the tests", True),
        ("Next step: run the tests", True),
        ("TODO: fix the bug", True),
        # Completions
        ("The PR is done", True),
        ("Goal #181 was retired", True),
        ("The branch was merged", True),
        ("We shipped it yesterday", True),
        ("The issue has been resolved", True),
        # State changes
        ("The blocker is unblocked", True),
        ("We're blocked on the review", True),
        ("No longer pending", True),
        ("The scope has shifted", True),
        # Non-goal turns
        ("What's the weather like?", False),
        ("That's a good point", False),
        ("I agree with you", False),
        ("The memory import looks good", False),
    ])
    def test_pattern_matching(self, text, should_match):
        matched = any(re.search(p, text) for p in _GOAL_SIGNALS)
        assert matched == should_match, f"{text!r} should {'match' if should_match else 'not match'}"

    @pytest.mark.parametrize("text", [
        "i'll pick up the backfill tomorrow",
        "i will start working on the lens",
        "next step: run the tests",
        "todo: fix the bug",
        "the pr is done",
        "now unblocked",
        "goal retired",
    ])
    def test_patterns_are_case_insensitive(self, text):
        assert any(re.search(p, text) for p in _GOAL_SIGNALS), \
            f"lowercase {text!r} should match"
