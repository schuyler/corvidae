"""WorkflowEngine — orchestrates stage progression.

Manages the linear backbone with escape hatches:
- PROCEED → next stage (default forward path)
- Escape outcome → jump to the named target stage
- BLOCKED or invalid escape → stop the workflow

Also handles:
- Conditional stages (should_run → skip)
- Human gates (stdin prompt for approval)
- Max iterations guard (prevents infinite escape loops)
"""

from __future__ import annotations

import logging

from corvidae.llm import LLMClient

from workflow.types import StageOutcome
from workflow.context import WorkflowContext
from workflow.stage import Stage, StageRunner

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Orchestrates a workflow by running stages in sequence.

    The engine manages stage transitions, context accumulation,
    human gates, and the max-iterations safety guard.
    """

    def __init__(
        self,
        client: LLMClient,
        max_turns: int = 20,
        max_iterations: int = 50,
    ) -> None:
        self.runner = StageRunner(client, max_turns)
        self.max_iterations = max_iterations

    async def run(
        self, stages: list[Stage], context: WorkflowContext
    ) -> None:
        """Run the workflow to completion.

        Iterates through stages linearly, handling escape hatches
        by jumping to the target stage. Stops when:
        - All stages complete (end of list)
        - A stage returns BLOCKED without an escape target
        - A non-PROCEED outcome has no matching escape target
        - Max iterations is reached (safety guard)
        """
        current_idx = 0
        iterations = 0

        while current_idx < len(stages):
            # Guard against infinite loops from escape cycles
            iterations += 1
            if iterations > self.max_iterations:
                logger.error(
                    "Max iterations (%d) reached, stopping workflow",
                    self.max_iterations,
                )
                return

            stage = stages[current_idx]

            # Check should_run — skip stage if predicate returns False
            if stage.should_run is not None and not stage.should_run(context):
                logger.info(
                    "Skipping stage: %s (should_run=False)", stage.name
                )
                current_idx += 1
                continue

            # Run the stage via StageRunner
            result = await self.runner.run(stage, context)

            # Record results in the shared context
            context.record_stage(
                stage.name, result.summary, result.tool_calls
            )

            # Human gate — pause for approval if enabled and stage succeeded
            if stage.human_gate and result.outcome == StageOutcome.PROCEED:
                approved = await self._human_approval(stage, result)
                if not approved:
                    logger.info(
                        "Human gate rejected for stage: %s", stage.name
                    )
                    return

            # Determine next stage based on outcome
            if result.escape_target:
                # Validate that the escape is allowed from this stage
                if result.outcome not in stage.escape_targets:
                    logger.error(
                        "Escape target %s not allowed for outcome %s "
                        "in stage %s",
                        result.escape_target,
                        result.outcome.value,
                        stage.name,
                    )
                    return
                target_idx = self._find_stage(stages, result.escape_target)
                if target_idx is None:
                    logger.error(
                        "Invalid escape target: %s", result.escape_target
                    )
                    return
                logger.info(
                    "Escaping from %s to %s (outcome=%s)",
                    stage.name, result.escape_target,
                    result.outcome.value,
                )
                current_idx = target_idx

            elif result.outcome == StageOutcome.PROCEED:
                current_idx += 1

            else:
                # Non-PROCEED outcome with no escape target — stop
                logger.warning(
                    "Stage %s ended with %s and no escape target, "
                    "stopping workflow",
                    stage.name, result.outcome.value,
                )
                return

    @staticmethod
    def _find_stage(stages: list[Stage], name: str) -> int | None:
        """Find a stage's index by name. Returns None if not found."""
        for i, stage in enumerate(stages):
            if stage.name == name:
                return i
        return None

    @staticmethod
    async def _human_approval(stage: Stage, result) -> bool:
        """Prompt for human approval via stdin."""
        print(f"\n{'=' * 60}")
        print(f"Stage: {stage.name}")
        print(f"Summary: {result.summary}")
        print(f"{'=' * 60}")
        response = input("Proceed? [y/N] ").strip().lower()
        return response in ("y", "yes")
