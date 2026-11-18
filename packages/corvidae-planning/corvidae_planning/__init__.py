"""Planning critique lens: flags turns that warrant a goal action.

A standalone plugin that contributes a single lens through the
``on_critique_init`` hook. The lens fires at a sample rate on
conversation-channel turns and asks a cheap fork whether the turn
warrants a goal create/update/retire.

Same shape as the Grice lens: sample-rate trigger, fork on the
conversation window, high-precision threshold, finding in context.

Design note: ``docs/design/planning-lens-2026-11-17.md`` in the lex workspace.
"""

from __future__ import annotations

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.critique import Lens

# The fork prompt: asks the model to evaluate whether the turn warrants
# a goal action. High-precision threshold (>=80% confidence).
# The fork sees the conversation window; it does not have the goal list.
# It produces a goal description; the agent resolves it to a goal id.
_FORK_PROMPT = """\
[Self-critique: planning] Review the most recent exchange. Does it warrant a goal action?

You are evaluating the conversation text only. You do not have access to the current goal list.

Actions:
- create: the agent committed to a multi-step task ("I'll do X", "next step: Y") that should be tracked as a goal but likely isn't one yet. "goal" should be a short new goal title.
- update: an existing goal's state changed (a blocker cleared, a new finding landed, the scope shifted) and the goal text is now stale. "goal" should be a short description of which goal (the agent will resolve it to an id).
- retire: an existing goal is done (the work is complete) but not yet retired. "goal" should be a short description of which goal (the agent will resolve it to an id).

If you would assign less than 80% probability that a goal action is needed, return null. If the goal action is trivial (a single-beat task that doesn't need tracking), return null. If the turn is casual conversation with no planning content, return null.

Return exactly one JSON value. No markdown.
If no action is warranted, return null.
Otherwise return an object with fields:
- action: one of "create", "update", "retire"
- goal: string (a title for create, a description for update/retire)
- rationale: string (one sentence)"""


class PlanningLensPlugin(CorvidaePlugin):
    """Contributes the planning lens to the critique plugin.

    Sample-rate trigger (like the Grice lens): fires on a fraction of
    turns and lets the fork model decide whether a goal action is
    warranted. No thresholds (doesn't use the appraisal vector), no
    reply patterns (no keyword gate).
    """

    depends_on = frozenset({"critique"})

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
