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
    r"(?i)\bI(?:'ll| will)\b",
    r"(?i)\bnext step\b",
    r"(?i)\bTODO\b",
    r"(?i)\bI(?:'ll| will) (?:pick up|start|work on|build|write|set up|create|add|file|open)\b",
    # Completion: "done", "retired", "merged", "shipped"
    r"(?i)\b(?:is|are|was|were|got) (?:done|complete|finished)\b",
    r"(?i)\bretired\b",
    r"(?i)\bmerged\b",
    r"(?i)\bshipped\b",
    # State-change: blocker clearing, scope shift
    r"(?i)\bunblocked\b",
    r"(?i)\bblocked on\b",
    r"(?i)\bno longer (?:blocked|pending|waiting)\b",
    r"(?i)\b(?:scope|priority) (?:shift|change|changed)\b",
)

# The fork prompt: asks the model to evaluate whether the turn warrants
# a goal action. High-precision threshold (>=80% confidence).
_FORK_PROMPT = """\
[Self-critique: planning] Review the most recent exchange. Does it warrant a goal action?

- CREATE: the agent committed to a multi-step task ("I'll do X", "next step: Y") that is not already a goal.
- UPDATE: an existing goal's state changed (a blocker cleared, a new finding landed, the scope shifted) and the goal text is now stale.
- RETIRE: an existing goal is done (the work is complete) but not yet retired.

Fire ONLY if you are confident (>=80%) that a goal action is warranted. If the turn is ambiguous, or the goal action is trivial (a single-beat task that doesn't need tracking), don't fire.

Output JSON: {"action": "create"|"update"|"retire", "goal": "<description or goal id>", "rationale": "<one sentence>"} or null if no goal action is warranted."""


class PlanningLensPlugin(CorvidaePlugin):
    """Contributes the planning lens to the critique plugin.

    The lens has no thresholds (it doesn't use the appraisal vector) and
    a sample_rate of 0.0 (it fires only when a reply pattern matches).
    """

    depends_on = frozenset({"critique"})

    @hookimpl
    async def on_critique_init(self, config: dict, add_lens) -> None:
        add_lens(Lens(
            name="planning",
            fork_prompt=_FORK_PROMPT,
            thresholds=(),
            reply_patterns=_GOAL_SIGNALS,
            sample_rate=0.0,
        ))
