"""Plan recording tool — lets the LLM save a structured plan for later stages.

The record_plan tool captures structured data about the implementation plan
(branch name, task type, files to change) and stores it in a PlanHolder.
After the stage completes, the engine copies the plan to context.facts["plan"]
so later stages can consume it.

This avoids redundant re-analysis: the comment_on_issue stage analyzes the issue
once, records the plan, and downstream stages (create_branch, write_tests,
submit_pr) read from the plan instead of re-deriving the same information.
"""

import logging
from dataclasses import dataclass

from corvidae.tool import Tool

logger = logging.getLogger(__name__)

# Valid task types for the plan artifact.
VALID_TASK_TYPES = ("docs_only", "bug_fix", "feature", "refactor")
_VALID_TASK_TYPES_STR = ", ".join(repr(t) for t in VALID_TASK_TYPES)


@dataclass
class PlanHolder:
    """Mutable holder for the structured plan.

    The record_plan tool writes to this; the engine reads it after
    the stage completes and copies the data to context.facts["plan"].
    """

    data: dict | None = None


def make_record_plan_tool(holder: PlanHolder) -> Tool:
    """Create a record_plan tool that writes to the given holder.

    Returns a Tool wrapping an async function. The function captures
    the holder via closure so the engine can inspect it after the
    comment_on_issue stage completes.
    """

    async def record_plan(
        branch_name: str,
        task_type: str,
        needs_tests: bool,
        needs_docs: bool = True,
        files_to_change: list[str] | None = None,
        pr_title: str = "",
        summary: str = "",
    ) -> str:
        """Record a structured implementation plan for later stages.

        Call this after analyzing the issue to save your plan. Later stages
        will use this plan to avoid redundant analysis.

        Args:
            branch_name: Suggested git branch name (e.g. 'feat/issue-42-add-logging').
            task_type: Classification of the task. Must be EXACTLY one of:
                'docs_only', 'bug_fix', 'feature', 'refactor'.
            needs_tests: Whether this task requires writing tests.
            needs_docs: Whether documentation needs updating. Defaults to True.
            files_to_change: List of files that will likely be modified.
            pr_title: Suggested PR title.
            summary: Brief summary of the implementation plan.
        """
        # Validate task_type before storing
        if task_type not in VALID_TASK_TYPES:
            return (
                f"Error: Invalid task_type {task_type!r}. "
                f"Must be one of: {_VALID_TASK_TYPES_STR}."
            )

        # Build the plan dict with defaults for optional fields
        plan = {
            "branch_name": branch_name,
            "task_type": task_type,
            "needs_tests": bool(needs_tests),
            "needs_docs": bool(needs_docs),
            "files_to_change": files_to_change or [],
            "pr_title": pr_title,
            "summary": summary,
        }

        holder.data = plan
        logger.info(
            "Plan recorded: task_type=%s, branch=%s",
            task_type, branch_name,
        )

        return (
            f"Plan recorded: {task_type} task, branch '{branch_name}', "
            f"needs_tests={plan['needs_tests']}, needs_docs={plan['needs_docs']}, "
            f"{len(plan['files_to_change'])} files to change."
        )

    return Tool.from_function(record_plan)
