"""Planning critique lens: flags turns that warrant a goal action.

A standalone plugin that contributes a single lens through the
``on_critique_init`` hook. The lens fires when the agent's reply contains
a goal-relevant signal (commitment, completion, or state-change language)
and asks a cheap fork whether the turn warrants a goal create/update/retire.

Design note: ``docs/design/planning-lens-2026-11-17.md`` in the lex workspace.
"""

from __future__ import annotations

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.critique import Lens

# Goal-relevant signal patterns: commitment, completion, state-change.
# These are the pre-filter regexes (reply_patterns on the Lens). The lens
# fires only when one of these matches the agent's reply.
# All patterns are case-insensitive (?i).
_GOAL_SIGNALS = (
    # Commitment: "I'll do X", "next step: Y", "TODO: Z"
    r"(?i)\bI(?:['\u2019]ll|\s+will)\b",
    r"(?i)\bwe(?:['\u2019]ll|\s+will)\b",
    r"(?i)\bI(?:['\u2019]m|\s+am)\s+going\s+to\b",
    r"(?i)\bnext\s+step\b",
    r"(?i)\bTODO\b",
    # Completion: "done", "retired", "merged", "shipped"
    r"(?i)\b(?:is|are|was|were|has|have|had|got)\s+(?:been\s+)?(?:done|complete|completed|finished|closed|resolved)\b",
    r"(?i)\bretired\b",
    r"(?i)\bmerged\b",
    r"(?i)\bshipped\b",
    # State-change: blocker clearing, scope shift
    r"(?i)\bunblocked\b",
    r"(?i)\bblocked\s+on\b",
    r"(?i)\bno\s+longer\s+(?:blocked|pending|waiting)\b",
    r"(?i)\b(?:scope|priority)\s+(?:has\s+)?(?:shift|shifted|change|changed)\b",
)

# The fork prompt: asks the model to evaluate whether the turn warrants
# a goal action. High-precision threshold (>=80% confidence).
# The fork does NOT have access to the goal list — it produces a goal
# description, and the agent resolves it to a goal id.
_FORK_PROMPT = """\
[Self-critique: planning] Review the most recent agent reply. Does it warrant a goal action?

You do not have access to the current goal list. You are evaluating the reply text only.

Actions:
- create: the agent committed to a multi-step task that is not already a goal. "goal" should be a short new goal title.
- update: an existing goal's state changed (a blocker cleared, a new finding landed, the scope shifted) and the goal text is now stale. "goal" should be a short description of which goal (the agent will resolve it to an id).
- retire: an existing goal is done (the work is complete) but not yet retired. "goal" should be a short description of which goal (the agent will resolve it to an id).

If you would assign less than 80% probability that a goal action is needed, return null. If the goal action is trivial (a single-beat task that doesn't need tracking), return null.

Return exactly one JSON value. No markdown.
If no action is warranted, return null.
Otherwise return an object with fields:
- action: one of "create", "update", "retire"
- goal: string (a title for create, a description for update/retire)
- rationale: string (one sentence)"""


class PlanningLensPlugin(CorvidaePlugin):
    """Contributes the planning lens to the critique plugin.

    The lens has no thresholds (it doesn't use the appraisal vector).
    ``sample_rate=0.0`` is the pattern-only sentinel: when a reply pattern
    matches, the lens runs immediately (no sampling); when no pattern
    matches, the sampling draw at 0.0 means it never runs.
    """

    depends_on = frozenset({"critique"})

    @hookimpl
    async def on_critique_init(self, config: dict, add_lens) -> None:
        add_lens(Lens(
            name="planning",
            fork_prompt=_FORK_PROMPT,
            thresholds=(),
            reply_patterns=_GOAL_SIGNALS,
            # Pattern-only: runs when a reply pattern matches, never otherwise.
            sample_rate=0.0,
        ))
