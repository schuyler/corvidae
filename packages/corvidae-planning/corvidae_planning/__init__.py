"""Planning critique lens: flags turns that warrant a goal action.

A standalone plugin that contributes a single lens through the
``on_critique_init`` hook. The lens fires at a sample rate on
conversation-channel turns and asks a cheap fork whether the turn
warrants a goal create/update/retire.

The plugin also implements ``transform_critique_prompt`` to append the
current goal list to the fork instruction, so the fork can reference
specific goal ids and avoid suggesting goals that already exist.

Design note: ``docs/design/planning-lens-2026-11-17.md`` in the lex workspace.
"""

from __future__ import annotations

import logging

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae_cognition.critique import Lens
from corvidae_cognition.goals import GoalsPlugin

logger = logging.getLogger(__name__)

# The fork prompt: asks the model to evaluate whether the turn warrants
# a goal action. High-precision threshold (>=80% confidence).
# The fork sees the conversation window plus the goal list (appended by
# transform_critique_prompt). It produces a specific suggestion with
# goal ids; the agent decides whether to act.
_FORK_PROMPT = """\
[Self-critique: planning] Review the most recent exchange. Does it warrant a goal action?

You are evaluating the conversation text and the current goal list.

Actions:
- create: the agent committed to a multi-step task ("I'll do X", "next step: Y") that should be tracked as a goal but is not in the goal list. "goal" should be a short new goal title.
- update: an existing goal's state changed (a blocker cleared, a new finding landed, the scope shifted) and the goal text is now stale. "goal" should be the goal id (e.g. "#181").
- retire: an existing goal is done (the work is complete) but not yet retired. "goal" should be the goal id (e.g. "#181").

If the suggested goal is already in the goal list, do not suggest it. If you would assign less than 80% probability that a goal action is needed, return null. If the goal action is trivial (a single-beat task that doesn't need tracking), return null. If the turn is casual conversation with no planning content, return null.

Return exactly one JSON value. No markdown.
If no action is warranted, return null.
Otherwise return an object with fields:
- action: one of "create", "update", "retire"
- goal: string (a title for create, a goal id for update/retire)
- rationale: string (one sentence)"""


class PlanningLensPlugin(CorvidaePlugin):
    """Contributes the planning lens to the critique plugin.

    Sample-rate trigger (like the Grice lens): fires on a fraction of
    turns and lets the fork model decide whether a goal action is
    warranted. The fork sees the goal list (appended via
    ``transform_critique_prompt``) so it can reference specific goal ids
    and avoid suggesting goals that already exist.
    """

    depends_on = frozenset({"critique", "goals"})

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self._goals = get_dependency(pm, "goals", GoalsPlugin)

    @hookimpl
    async def on_critique_init(self, config: dict, add_lens) -> None:
        add_lens(Lens(
            name="planning",
            fork_prompt=_FORK_PROMPT,
            thresholds=(),
            reply_patterns=(),
            # Start at 10% (like Grice); tune after a week of observation.
            sample_rate=0.1,
        ))

    @hookimpl
    async def transform_critique_prompt(
        self, channel, trace_id: str, lens_name: str, instruction: str,
    ) -> str | None:
        """Append the current goal list to the planning lens fork instruction."""
        if lens_name != "planning":
            return None
        try:
            goals = await self._goals.open_goals()
        except Exception:
            logger.warning(
                "planning lens: failed to fetch goal list; "
                "fork will run without it",
                exc_info=True,
            )
            return None
        if not goals:
            instruction += (
                "\n\nCurrent goal list: (empty — no open goals)"
            )
        else:
            lines = [f"  #{g['id']}: {g['text']}" for g in goals]
            instruction += (
                "\n\nCurrent goal list:\n" + "\n".join(lines)
            )
        return instruction
