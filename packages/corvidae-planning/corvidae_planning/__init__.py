"""Planning critique lens: flags turns that warrant a goal action.

A standalone plugin that contributes a single lens through the
``on_critique_init`` hook. The lens fires at a sample rate on the turns
critique judges and asks a fork whether the turn warrants a goal
create, update or retire. A finding is an ordinary critique objection: the
agent reads it at its next turn and decides whether to act.

The plugin also implements ``transform_critique_prompt`` to append the
current open goal list to the fork instruction, so the fork can name
specific goal ids and avoid suggesting goals that already exist.
"""

from __future__ import annotations

import logging

from corvidae.fork import quote
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.critique import Lens
from corvidae_cognition.goals import GoalsPlugin

logger = logging.getLogger(__name__)

# The lens's name: it keys the fork prompt override, the tunables and the
# outcome entry (critique.fork_prompt.planning, critique.lens.planning.*).
LENS_NAME = "planning"

# The brief. Critique appends its objections JSON tail, so the reply format
# is critique's: one objection per goal action, none when nothing is needed.
# The goal list is appended by transform_critique_prompt.
_FORK_PROMPT = """\
[Self-critique: planning] Review the most recent exchange. Does it warrant a goal action?

You are evaluating the conversation text and the current goal list, which follows the exchange below.

Actions:
- create: the agent committed to a multi-step task ("I'll do X", "next step: Y") that should be tracked as a goal but is not in the goal list.
- update: an existing goal's state changed (a blocker cleared, a new finding landed, the scope shifted) and the goal text is now stale.
- retire: an existing goal is done (the work is complete) but not yet retired.

If the suggested goal is already in the goal list, do not suggest it. If you would assign less than 80% probability that a goal action is needed, do not suggest it. If the goal action is trivial (a single-beat task that doesn't need tracking), do not suggest it. If the turn is casual conversation with no planning content, suggest nothing.

Report each goal action as one objection. Its "claim" is the action and the goal: "create: <a short new goal title>", "update: #<goal id>" or "retire: #<goal id>". Its "objection" is a one-sentence rationale. If no goal action is warranted, return an empty objections list."""


class PlanningLensPlugin(CorvidaePlugin):
    """Contributes the planning lens to the critique plugin.

    Sample-rate trigger: fires on a fraction of judged turns and lets the
    fork model decide whether a goal action is warranted. The fork sees the
    goal list (appended via ``transform_critique_prompt``) so it can name
    specific goal ids and avoid suggesting goals that already exist.
    """

    # Optional, as for any lens contributor (docs/cognition.md): with
    # critique disabled the hookimpls are orphans, reported at debug; with
    # goals disabled the fork runs without the goal list. A hard dependency
    # would stop the daemon from starting on a config that disables either.
    optional_depends_on = frozenset({"critique", "goals"})

    @hookimpl
    async def on_critique_init(self, config: dict, add_lens) -> None:
        # One lens, no thresholds or patterns: it runs at its sample rate.
        add_lens(Lens(
            name=LENS_NAME,
            fork_prompt=_FORK_PROMPT,
            thresholds=(),
            reply_patterns=(),
            # Start at 10%; tunable as critique.lens.planning.sample_rate.
            sample_rate=0.1,
        ))

    @hookimpl
    async def transform_critique_prompt(
        self, channel, trace_id: str, lens_name: str, instruction: str,
    ) -> str | None:
        """Append the current goal list to the planning lens fork instruction.

        Answers only for the planning lens and returns None for every other
        lens, so it never shadows another plugin's transform of its own lens.
        """
        if lens_name != LENS_NAME:
            return None
        # Fetch the open goals; without them the fork still runs, unlisted.
        goals_plugin = self.pm.get_plugin("goals")
        if not isinstance(goals_plugin, GoalsPlugin):
            logger.warning(
                "planning lens: the goals plugin is not loaded, so the fork "
                "for trace %s on channel %s runs without the goal list; "
                "enable goals or list corvidae_planning under plugins.disabled",
                trace_id, getattr(channel, "id", channel),
            )
            return None
        try:
            goals = await goals_plugin.open_goals()
        except Exception:
            logger.warning(
                "planning lens: could not fetch the goal list for trace %s "
                "on channel %s; the fork runs without it",
                trace_id, getattr(channel, "id", channel), exc_info=True,
            )
            return None
        # Render one line per goal; goal text is flattened and capped by
        # quote() so a goal can't break the list's format.
        if not goals:
            return instruction + "\n\nCurrent goal list: (empty — no open goals)"
        lines = [f"  #{g['id']}: «{quote(g['summary'])}»" for g in goals]
        return instruction + "\n\nCurrent goal list:\n" + "\n".join(lines)
