"""Planning lens plugin tests.

Verifies that the PlanningLensPlugin contributes a valid lens through
the on_critique_init hook, and that the lens has the expected shape:
- name is "planning"
- reply_patterns are the goal-signal regexes
- no thresholds (doesn't use the appraisal vector)
- sample_rate is 0.0 (fires only on pattern match)
- fork_prompt is non-empty
"""

import re

import pytest

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.critique import Lens
from corvidae_planning import PlanningLensPlugin, _FORK_PROMPT, _GOAL_SIGNALS


def _run_init(plugin):
    """Run on_critique_init and return the collected lenses."""
    collected = []

    def add_lens(lens):
        collected.append(lens)

    import asyncio
    asyncio.get_event_loop().run_until_complete(
        plugin.on_critique_init({}, add_lens)
    )
    return collected


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

    def test_lens_sample_rate_is_zero(self):
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
        assert "CREATE" in prompt
        assert "UPDATE" in prompt
        assert "RETIRE" in prompt

    def test_goal_signals_include_commitment(self):
        # Check that at least one pattern matches "I'll"
        assert any(re.search(p, "I'll do it") for p in _GOAL_SIGNALS)

    def test_goal_signals_include_completion(self):
        assert any(re.search(p, "it is done") for p in _GOAL_SIGNALS)
        assert any(re.search(p, "goal retired") for p in _GOAL_SIGNALS)

    def test_goal_signals_include_state_change(self):
        assert any(re.search(p, "now unblocked") for p in _GOAL_SIGNALS)


class TestGoalSignalPatterns:
    """The goal-signal regexes match the expected phrases."""

    @pytest.mark.parametrize("text,should_match", [
        # Commitments
        ("I'll pick up the backfill tomorrow", True),
        ("I will start working on the lens", True),
        ("Next step: run the tests", True),
        ("TODO: fix the bug", True),
        # Completions
        ("The PR is done", True),
        ("Goal #181 was retired", True),
        ("The branch was merged", True),
        ("We shipped it yesterday", True),
        # State changes
        ("The blocker is unblocked", True),
        ("We're blocked on the review", True),
        ("No longer pending", True),
        # Non-goal turns
        ("What's the weather like?", False),
        ("That's a good point", False),
        ("I agree with you", False),
        ("The memory import looks good", False),
    ])
    def test_pattern_matching(self, text, should_match):
        matched = any(re.search(p, text) for p in _GOAL_SIGNALS)
        assert matched == should_match, f"{text!r} should {'match' if should_match else 'not match'}"
