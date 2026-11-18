"""Grice alignment lens: flags replies that violate Grice's maxims.

A standalone plugin that contributes a single lens through the
``on_critique_init`` hook. The lens fires at a sample rate on the turns
critique judges and asks a fork whether the reply violates any of
Grice's four cooperative maxims (Quantity, Quality, Relation, Manner).
A finding is an ordinary critique objection: the agent reads it at its
next turn and decides whether to act.

No ``transform_critique_prompt``: the fork needs no external context
beyond the exchange text (the critique plugin provides that).
"""

from __future__ import annotations

import logging

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae_cognition.critique import Lens

logger = logging.getLogger(__name__)

# The lens's name: it keys the fork prompt override, the tunables and the
# outcome entry (critique.lens.grice.*).
LENS_NAME = "grice"

_FORK_PROMPT = """\
[Self-critique: grice] Review the agent's most recent reply against
Grice's four maxims of cooperative communication.

Maxims:
- Quantity: be as informative as required, no more. Flag over-answering
  (answering questions the user didn't ask) or under-answering (omitting
  information the user needs).
- Quality: be truthful and evidence-based. Flag assertions presented as
  fact without verification, or claims that contradict available evidence.
- Relation: be relevant. Flag content that is off-topic, tangential, or
  unconnected to the user's question.
- Manner: be clear, brief, and orderly. Flag ambiguity, unnecessary
  complexity, or disorganized presentation.

Rules:
- Flag a violation only when it is clear and material. Borderline cases
  are not violations.
- One objection per violated maxim (max 4).
- If no maxim is clearly violated, return an empty objections list.
- The "claim" is the maxim name and what was violated. The "objection"
  is a one-sentence rationale."""


class GriceLensPlugin(CorvidaePlugin):
    """Contributes the Grice alignment lens to the critique plugin.

    Sample-rate trigger: fires on a fraction of judged turns and lets the
    fork model decide whether a Grice maxim is violated. No external
    context is needed beyond the exchange text.
    """

    # Optional, as for any lens contributor (docs/cognition.md): with
    # critique disabled the hookimpls are orphans, reported at debug.
    optional_depends_on = frozenset({"critique"})

    @hookimpl
    async def on_critique_init(self, config: dict, add_lens) -> None:
        # One lens, no thresholds or patterns: it runs at its sample rate.
        add_lens(Lens(
            name=LENS_NAME,
            fork_prompt=_FORK_PROMPT,
            thresholds=(),
            reply_patterns=(),
            # Start at 10%; tunable as critique.lens.grice.sample_rate.
            sample_rate=0.1,
        ))
