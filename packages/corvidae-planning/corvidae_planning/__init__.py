"""Planning critique lens: flags turns that warrant a goal action.

A standalone plugin that contributes a single lens through the
``on_critique_init`` hook. The appraisal fork scores the lens's
``appraisal_question`` on the turns critique judges; when it wins, the
lens asks a fork whether the turn warrants a goal
create, update or retire. A finding is an ordinary critique objection: the
agent reads it at its next turn and decides whether to act.

The plugin also implements ``transform_critique_prompt`` to append the
current open goal list to the fork instruction, so the fork can name
specific goal ids and avoid suggesting goals that already exist. It asks for
the list through a hook it owns, ``resolve_corvidae_planning_goals``, which
the goals plugin implements (#1181), so planning never looks goals up.
"""

from __future__ import annotations

import logging

from corvidae.fork import quote
from corvidae.hooks import CorvidaePlugin, hookimpl, hookspec
from corvidae_cognition.critique import Lens

logger = logging.getLogger(__name__)

# The lens's name: it keys the fork prompt override, the tunables and the
# outcome entry (critique.lens.planning.fork_prompt and
# critique.lens.planning.appraisal_question).
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


class PlanningSpec:
    """Hooks the planning plugin owns."""

    @hookspec(firstresult=True)
    async def resolve_corvidae_planning_goals(self) -> list[dict] | None:
        """The open goals, front of the queue first, or None when no goals
        plugin answers.

        Each dict has at least "id" (int) and "summary" (the goal's text,
        str). An empty list means goals answered and none are open.
        """


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
    # The goal list is read through this hook; goals implements it (#1181).
    hookspecs = PlanningSpec

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # An absent goals plugin is a configuration the operator chose:
        # report the fallback once here, not on every fork (plugin-guide,
        # "When nobody answers").
        if not self.pm.has_plugin("goals"):
            logger.info(
                "planning lens: no goals plugin loaded, so planning forks "
                "run without the goal list",
            )

    @hookimpl
    async def on_critique_init(self, config: dict, add_lens) -> None:
        # One lens. The appraisal fork scores this question against the
        # other lenses on each exchange; tune it as
        # critique.lens.planning.appraisal_question.
        add_lens(Lens(
            name=LENS_NAME,
            fork_prompt=_FORK_PROMPT,
            appraisal_question=(
                "Does this turn call for creating, updating or retiring a goal: "
                "did the plan or next step drift from, or lose track of, the "
                "agent's open goals?"
            ),
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
        # Ask for the open goals; with no goals plugin to answer (None) the
        # fork runs unlisted, already reported at startup.
        try:
            goals: list[dict] | None = (
                await self.pm.ahook.resolve_corvidae_planning_goals()
            )
        except Exception:
            logger.warning(
                "planning lens: could not fetch the goal list for trace %s "
                "on channel %s; the fork runs without it",
                trace_id, getattr(channel, "id", channel), exc_info=True,
            )
            return None
        if goals is None:
            return None
        # Render one line per goal; goal text is flattened and capped by
        # quote() so a goal can't break the list's format.
        if not goals:
            return instruction + "\n\nCurrent goal list: (empty — no open goals)"
        lines = [f"  #{g['id']}: «{quote(g['summary'])}»" for g in goals]
        return instruction + "\n\nCurrent goal list:\n" + "\n".join(lines)
