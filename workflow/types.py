"""Workflow engine core types.

Defines the data structures used across all workflow components:
- StageOutcome: possible outcomes when a stage completes
- ToolCallRecord: captured I/O from a single tool invocation
- StageResult: everything the orchestrator needs from a completed stage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StageOutcome(str, Enum):
    """Possible outcomes when a stage completes.

    The orchestrator maps these to transitions. The LLM never chooses
    the next stage directly — it signals an outcome and the orchestrator
    decides what happens next.
    """

    PROCEED = "proceed"
    NEED_CLARIFICATION = "need_clarification"
    SCOPE_CHANGED = "scope_changed"
    BLOCKED = "blocked"


@dataclass
class ToolCallRecord:
    """Captured I/O from a single tool invocation within a stage.

    The orchestrator logs these as facts. The LLM doesn't need to
    summarise what it did — the orchestrator already knows.
    """

    tool_name: str
    arguments: dict
    result: str
    error: bool = False


@dataclass
class StageResult:
    """Everything the orchestrator needs from a completed stage.

    Two channels:
    1. summary — the LLM's reasoning and decisions (judgment only)
    2. tool_calls — observable facts the orchestrator gathered
    """

    stage_name: str
    outcome: StageOutcome
    summary: str
    escape_target: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
