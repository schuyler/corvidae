"""GitHub issue implementation workflow.

Defines the stages for implementing a GitHub issue:
1. Comment on issue with implementation plan
2. Add "in-progress" label (mechanical)
3. Create branch (LLM suggests name)
4. Gate: ready to implement? (escape to stage 1)
5. Write tests (red)
6. Implement (green)
7. Refactor
8. Update docs (conditional — skipped if gate_docs decides not needed)
9. Submit PR

Stages 10-12 (respond to PR feedback, close issue) are deferred —
they require waiting for external events (PR review, PR merge).
"""

from __future__ import annotations

import re

from corvidae.tool import Tool
from corvidae.tools.files import read_file, write_file
from corvidae.tools.shell import shell

from workflow.types import StageOutcome, StageResult
from workflow.context import WorkflowContext
from workflow.stage import Stage
from workflow.tools.github import (
    make_github_comment_tool,
    make_add_label_tool,
    make_create_pr_tool,
)
from workflow.tools.git import (
    make_create_branch_tool,
    make_create_worktree_tool,
)


def parse_issue_url(url: str) -> tuple[str, int]:
    """Parse a GitHub issue URL into (owner/repo, issue_number).

    Raises:
        ValueError: If the URL is not a valid GitHub issue URL.
    """
    url = url.rstrip("/")
    match = re.match(
        r"https://github\.com/([^/]+/[^/]+)/issues/(\d+)$", url
    )
    if not match:
        raise ValueError(f"Not a valid GitHub issue URL: {url}")
    return match.group(1), int(match.group(2))


# -- Shell and file tool wrappers for workflow use --

async def run_command(command: str) -> str:
    """Execute a shell command and return the output."""
    return await shell(command, timeout=30)


async def read_source(path: str) -> str:
    """Read the contents of a file."""
    return await read_file(path)


async def write_source(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    return await write_file(path, content)


# -- Mechanical stage helpers --

def _make_add_label_fn(repo: str, issue_number: int):
    """Create a mechanical_fn that adds the 'in-progress' label."""
    async def fn(ctx: WorkflowContext) -> StageResult:
        import shlex
        cmd = (
            f"gh issue edit {issue_number} "
            f"--repo {shlex.quote(repo)} "
            f"--add-label in-progress"
        )
        await shell(cmd, timeout=30)
        return StageResult(
            stage_name="add_label",
            outcome=StageOutcome.PROCEED,
            summary="Added in-progress label",
        )
    return fn


# -- Docs conditional check --

def _docs_should_run(context: WorkflowContext) -> bool:
    """Check if the update_docs stage should run.

    The gate_docs stage's summary starts with 'YES' or 'NO'
    to signal whether docs are needed. If 'YES', run the stage.
    Otherwise skip it.
    """
    gate_summary = context.summaries.get("gate_docs", "")
    return gate_summary.strip().upper().startswith("YES")


# -- Workflow builder --

def build_issue_workflow(
    repo: str,
    issue_number: int,
    repo_path: str,
    issue_body: str,
) -> list[Stage]:
    """Build the GitHub issue implementation workflow.

    Args:
        repo: GitHub repo in owner/name format.
        issue_number: The issue number.
        repo_path: Local path to the git repository.
        issue_body: The issue body text (fetched via gh CLI).

    Returns:
        Ordered list of Stage instances.
    """
    # Create scoped tools
    github_comment = make_github_comment_tool(repo, issue_number)
    create_branch = make_create_branch_tool(repo_path)
    create_pr = make_create_pr_tool(repo, "HEAD")

    # File and shell tools (shared across stages)
    shell_tool = Tool.from_function(run_command)
    file_tools = [
        Tool.from_function(read_source),
        Tool.from_function(write_source),
    ]

    return [
        # 1. Comment on issue with implementation plan
        Stage(
            name="comment_on_issue",
            instruction=(
                f"Read the GitHub issue #{issue_number} in {repo}.\n\n"
                f"The issue body:\n```\n{issue_body}\n```\n\n"
                "Write a comment describing your implementation plan. "
                "Include:\n"
                "- Requirements extracted from the issue\n"
                "- Proposed approach\n"
                "- Any questions or ambiguities\n\n"
                "Post the comment using the comment_on_issue tool."
            ),
            tools=[github_comment],
        ),

        # 2. Add "in-progress" label (mechanical)
        Stage(
            name="add_label",
            instruction="Add in-progress label",
            is_mechanical=True,
            mechanical_fn=_make_add_label_fn(repo, issue_number),
        ),

        # 3. Create branch (LLM suggests name)
        Stage(
            name="create_branch",
            instruction=(
                "Suggest a descriptive branch name for implementing "
                "this issue, then create the branch using the "
                "create_branch tool."
            ),
            tools=[create_branch],
        ),

        # 4. Gate: ready to implement?
        Stage(
            name="gate_ready",
            instruction=(
                "Review the issue and context from previous stages. "
                "Do you have enough information to implement this?\n\n"
                "If yes, call stage_complete with outcome 'proceed'.\n"
                "If not, call stage_complete with outcome "
                "'need_clarification' and list your questions."
            ),
            escape_targets={
                StageOutcome.NEED_CLARIFICATION: "comment_on_issue",
            },
        ),

        # 5. Write tests (red)
        Stage(
            name="write_tests",
            instruction=(
                "Write failing tests for the implementation described "
                "in the issue. Follow TDD: write tests that capture the "
                "requirements but do NOT yet pass.\n\n"
                "Use write_source to create test files, then run_command "
                "to run the tests and verify they fail."
            ),
            tools=file_tools + [shell_tool],
        ),

        # 6. Implement (green)
        Stage(
            name="implement",
            instruction=(
                "Write the implementation to make the tests pass.\n\n"
                "Use write_source to create or edit source files, "
                "read_source to inspect existing code, and "
                "run_command to run the tests."
            ),
            tools=file_tools + [shell_tool],
        ),

        # 7. Refactor
        Stage(
            name="refactor",
            instruction=(
                "Review the code and tests. Clean up:\n"
                "- Improve naming\n"
                "- Extract functions\n"
                "- Remove duplication\n\n"
                "Ensure tests still pass after refactoring."
            ),
            tools=file_tools + [shell_tool],
        ),

        # 8. Gate: docs needed? (merged into update_docs for simplicity)
        Stage(
            name="gate_docs",
            instruction=(
                "Decide whether documentation updates are needed for "
                "this change.\n\n"
                "If YES, start your summary with 'YES' and describe "
                "what needs documenting.\n"
                "If NO, start your summary with 'NO' and explain why."
            ),
        ),

        # 9. Update docs (conditional)
        Stage(
            name="update_docs",
            instruction=(
                "Update documentation based on the changes made.\n\n"
                "Edit the relevant documentation files."
            ),
            tools=file_tools,
            should_run=_docs_should_run,
        ),

        # 10. Submit PR
        Stage(
            name="submit_pr",
            instruction=(
                "Write a clear PR description summarizing the changes, "
                "then create the PR using the create_pr tool.\n\n"
                "The PR body should reference the issue number and "
                "describe what was changed and why."
            ),
            tools=[create_pr],
        ),
    ]
