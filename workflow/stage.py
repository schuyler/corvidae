"""Stage definition and StageRunner — the core stage execution machinery.

Stage: dataclass defining a single workflow stage.
StageRunner: runs one stage (mechanical or LLM-driven), handling the
tool-calling loop, stage_complete detection, and exit criteria.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from corvidae.llm import LLMClient
from corvidae.tool import Tool as CorvidaeTool, ToolRegistry, dispatch_tool_call
from corvidae.turn import run_agent_turn

from workflow.types import StageOutcome, StageResult, ToolCallRecord
from workflow.context import WorkflowContext
from workflow.tools.stage_complete import StageCompleteHolder, make_stage_complete_tool

logger = logging.getLogger(__name__)

# System prompt for all LLM stages
SYSTEM_PROMPT = (
    "You are a workflow agent executing a specific stage of a workflow. "
    "You have a focused task and a limited set of tools. "
    "Work through the task methodically. "
    "When finished, call stage_complete with your outcome and a summary "
    "of your reasoning."
)

# Type aliases for stage callback signatures
ExitCriteriaFn = Callable[[StageResult, WorkflowContext], Awaitable[bool]]
MechanicalFn = Callable[[WorkflowContext], Awaitable[StageResult]]
ShouldRunFn = Callable[[WorkflowContext], bool]


@dataclass
class Stage:
    """Definition of a single workflow stage.

    Attributes:
        name: Unique stage identifier (used for transitions and logging).
        instruction: The prompt describing what the LLM should do.
            For mechanical stages, a description for logging.
        tools: Tool instances available to the LLM in this stage.
        escape_targets: Maps StageOutcome to target stage name.
        exit_criteria: Async function verifying the stage's exit conditions.
            Receives (StageResult, WorkflowContext), returns True if met.
        should_run: Optional predicate — return False to skip this stage.
        human_gate: Whether to pause for human approval before proceeding.
        is_mechanical: If True, run mechanical_fn instead of LLM.
        mechanical_fn: Async function for mechanical stages.
    """

    name: str
    instruction: str
    tools: list = field(default_factory=list)
    escape_targets: dict[StageOutcome, str] = field(default_factory=dict)
    exit_criteria: ExitCriteriaFn | None = None
    should_run: ShouldRunFn | None = None
    human_gate: bool = False
    is_mechanical: bool = False
    mechanical_fn: MechanicalFn | None = None


class StageRunner:
    """Runs a single workflow stage.

    For mechanical stages: calls mechanical_fn directly.
    For LLM stages: builds prompt, runs tool-calling loop until
    stage_complete, verifies exit criteria.
    """

    def __init__(self, client: LLMClient, max_turns: int = 20) -> None:
        self.client = client
        self.max_turns = max_turns

    async def run(self, stage: Stage, context: WorkflowContext) -> StageResult:
        """Run a single stage and return the result."""
        logger.info("Starting stage: %s", stage.name)

        if stage.is_mechanical:
            result = await self._run_mechanical(stage, context)
        else:
            result = await self._run_llm(stage, context)

        logger.info(
            "Stage complete: %s (outcome=%s)",
            stage.name, result.outcome.value,
        )
        return result

    async def _run_mechanical(
        self, stage: Stage, context: WorkflowContext
    ) -> StageResult:
        """Run a mechanical (no-LLM) stage."""
        if stage.mechanical_fn is None:
            raise ValueError(
                f"Mechanical stage '{stage.name}' has no mechanical_fn"
            )
        return await stage.mechanical_fn(context)

    async def _run_llm(
        self, stage: Stage, context: WorkflowContext
    ) -> StageResult:
        """Run an LLM stage with the tool-calling loop.

        Builds a scoped tool registry (stage tools + stage_complete),
        runs the LLM in a loop until stage_complete is called or
        max_turns is reached, then verifies exit criteria.
        """
        # Build scoped tool registry: stage tools + stage_complete
        holder = StageCompleteHolder()
        stage_complete_fn = make_stage_complete_tool(holder)
        stage_complete_tool = CorvidaeTool.from_function(stage_complete_fn)

        registry = ToolRegistry()
        registry.add(stage_complete_tool)
        for tool in stage.tools:
            registry.add(tool)

        tools_dict = registry.as_dict()
        tool_schemas = registry.schemas()

        # Build initial messages with prior context
        prior_context = context.build_context_string(stage.name)
        user_content = stage.instruction
        if prior_context:
            user_content += (
                f"\n\n## Context from previous stages\n{prior_context}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tool_records: list[ToolCallRecord] = []
        exit_criteria_failures = 0

        # Tool-calling loop: run until stage_complete or max_turns
        for _ in range(self.max_turns):
            turn = await run_agent_turn(
                self.client, messages, tool_schemas
            )

            # No tool calls — LLM returned text only
            if not turn.tool_calls:
                if holder.called:
                    break  # stage_complete was called in a prior turn
                # Prompt the LLM to call stage_complete
                messages.append({
                    "role": "user",
                    "content": (
                        "Please call stage_complete to signal you are done."
                    ),
                })
                continue

            # Dispatch each tool call
            for call in turn.tool_calls:
                result = await dispatch_tool_call(call, tools_dict)
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                })

                # Capture arguments for the record
                try:
                    args = json.loads(call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                tool_records.append(ToolCallRecord(
                    tool_name=result.tool_name,
                    arguments=args,
                    result=result.content,
                    error=result.error,
                ))

                if holder.called:
                    break  # stage_complete was dispatched

            if not holder.called:
                continue  # LLM hasn't called stage_complete yet

            # stage_complete was called — check exit criteria
            if stage.exit_criteria is not None:
                stage_result = StageResult(
                    stage_name=stage.name,
                    outcome=holder.outcome or StageOutcome.PROCEED,
                    summary=holder.summary,
                    escape_target=holder.escape_target,
                    tool_calls=tool_records,
                )
                passed = await stage.exit_criteria(stage_result, context)
                if not passed:
                    exit_criteria_failures += 1
                    if exit_criteria_failures >= 3:
                        logger.warning(
                            "Exit criteria failed %d times for stage %s",
                            exit_criteria_failures, stage.name,
                        )
                        return StageResult(
                            stage_name=stage.name,
                            outcome=StageOutcome.BLOCKED,
                            summary=(
                                f"Exit criteria failed after "
                                f"{exit_criteria_failures} attempts"
                            ),
                            tool_calls=tool_records,
                        )
                    # Reset holder and let LLM retry
                    holder.outcome = None
                    holder.summary = ""
                    holder.escape_target = None
                    holder.called = False
                    messages.append({
                        "role": "user",
                        "content": (
                            "Exit criteria not met. Please address the "
                            "issue and call stage_complete again."
                        ),
                    })
                    continue

            # Exit criteria passed or no exit criteria defined
            break

        # Check if we exited without stage_complete
        if not holder.called:
            logger.error(
                "Stage %s ended without stage_complete", stage.name
            )
            return StageResult(
                stage_name=stage.name,
                outcome=StageOutcome.BLOCKED,
                summary="Stage ended without calling stage_complete",
                tool_calls=tool_records,
            )

        return StageResult(
            stage_name=stage.name,
            outcome=holder.outcome or StageOutcome.PROCEED,
            summary=holder.summary,
            escape_target=holder.escape_target,
            tool_calls=tool_records,
        )
