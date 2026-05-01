"""stage_complete tool — the mechanism for LLMs to signal stage completion.

The LLM calls this tool when it believes the current stage is done.
The tool writes to a mutable holder that the StageRunner inspects
after each tool dispatch. This decouples the LLM's completion signal
from the orchestrator's control flow.

The holder is created per-stage by the StageRunner and captured by
closure in the tool function. When exit criteria fail, the holder
is reset and the LLM gets another chance within the same stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from workflow.types import StageOutcome

logger = logging.getLogger(__name__)

# Pre-compute valid outcome values for fast lookup and error messages.
_VALID_OUTCOMES = [e.value for e in StageOutcome]
_VALID_OUTCOMES_STR = ", ".join(repr(v) for v in _VALID_OUTCOMES)


@dataclass
class StageCompleteHolder:
    """Mutable holder for stage_complete results.

    The stage_complete tool writes to this; the StageRunner reads it
    after each tool dispatch round.
    """

    outcome: StageOutcome | None = None
    summary: str = ""
    escape_target: str | None = None
    called: bool = False


def make_stage_complete_tool(holder: StageCompleteHolder):
    """Create a stage_complete tool that writes to the given holder.

    Returns an async function suitable for Tool.from_function().
    The function captures the holder via closure so the StageRunner
    can inspect it after dispatch.
    """

    async def stage_complete(
        outcome: str,
        summary: str,
        escape_target: str | None = None,
    ) -> str:
        """Signal that this stage is complete.

        Args:
            outcome: EXACTLY one of these values: 'proceed',
                'need_clarification', 'scope_changed', 'blocked'.
                Do NOT pass a sentence or description — pass only one of
                those four literal strings.
            summary: Your reasoning and decisions. Describe what you did and why.
                     Include approaches you tried and rejected. Do NOT repeat facts
                     the orchestrator can verify directly (file contents, test output).
            escape_target: Name of the stage to return to (only for
                           need_clarification or scope_changed).
        """
        # Validate outcome before constructing the enum so we can return a
        # helpful error message instead of letting the ValueError propagate.
        if outcome not in _VALID_OUTCOMES:
            logger.warning(
                "stage_complete received invalid outcome %r", outcome,
            )
            return (
                f"Error: 'outcome' must be one of {_VALID_OUTCOMES_STR}, "
                f"but got {outcome!r}. "
                f"Please call stage_complete again with a valid outcome."
            )

        holder.outcome = StageOutcome(outcome)
        holder.summary = summary
        holder.escape_target = escape_target
        holder.called = True
        return f"Stage complete: {outcome}"

    return stage_complete
