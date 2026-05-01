"""WorkflowContext — state accumulator across workflow stages.

Two channels, as required by the design doc:
1. summaries — per-stage LLM reasoning (judgment, decisions, rejected approaches)
2. facts — orchestrator-gathered data (file diffs, test results, GitHub state)
3. tool_logs — captured I/O from all tool invocations across all stages

The LLM never redundantly reports what the orchestrator can verify directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workflow.types import ToolCallRecord


@dataclass
class WorkflowContext:
    """Accumulated state across workflow stages.

    Passed to each stage so the LLM has access to prior decisions and
    the orchestrator can track progress.
    """

    summaries: dict[str, str] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    tool_logs: list[ToolCallRecord] = field(default_factory=list)

    def record_stage(
        self,
        stage_name: str,
        summary: str,
        tool_calls: list[ToolCallRecord],
    ) -> None:
        """Record results from a completed stage.

        Stores the summary and appends tool call records to the log.
        Only non-empty tool_calls lists are appended (mechanical stages
        pass an empty list).
        """
        self.summaries[stage_name] = summary
        if tool_calls:
            self.tool_logs.extend(tool_calls)

    def build_context_string(self, current_stage: str) -> str:
        """Build context string for inclusion in an LLM prompt.

        Includes summaries from all prior stages, excluding the current
        stage (it hasn't completed yet). Ordered by insertion order.
        """
        parts = []
        for name, summary in self.summaries.items():
            if name != current_stage:
                parts.append(f"## {name}\n{summary}")
        return "\n\n".join(parts)
