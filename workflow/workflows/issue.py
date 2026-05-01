"""GitHub issue implementation workflow.

Defines the stages for implementing a GitHub issue:
1. Comment on issue with implementation plan (+ record structured plan artifact)
2. Add "in-progress" label (mechanical)
3. Create branch (uses plan's branch name if available)
4. Gate: ready to implement? (escape to stage 1)
5. Write tests (red) — skipped if plan says no tests needed
6. Implement (green)
7. Refactor — skipped for docs_only tasks
8. Commit changes (mechanical)
9. Gate: docs needed? (mechanical — checks diffs and docs/ dir)
10. Update docs (conditional — skipped if gate_docs decides not needed)
11. Submit PR (uses plan's pr_title/summary if available)

Stages 12-14 (respond to PR feedback, close issue) are deferred —
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
from workflow.tools.plan import (
    PlanHolder,
    make_record_plan_tool,
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


def _make_commit_fn(repo_path: str):
    """Create a mechanical_fn that stages and commits all changes.

    Uses the implement stage's summary as the commit message.
    Falls back to a generic message if no implement summary exists.
    """
    async def fn(ctx: WorkflowContext) -> StageResult:
        import shlex
        # Derive commit message from the implement stage's summary
        message = ctx.summaries.get("implement", "Implement changes").strip()
        # Truncate to first line to avoid multi-line commit message issues
        message = message.split("\n")[0][:200]
        cmd = (
            f"git -C {shlex.quote(repo_path)} add -A && "
            f"git -C {shlex.quote(repo_path)} "
            f"commit -m {shlex.quote(message)}"
        )
        await shell(cmd, timeout=30)
        return StageResult(
            stage_name="commit_changes",
            outcome=StageOutcome.PROCEED,
            summary=f"Committed changes: {message}",
        )
    return fn


# -- Docs conditional check --

# -- Should-run predicates for stages that can be skipped --


def _tests_should_run(context: WorkflowContext) -> bool:
    """Check if the write_tests stage should run.

    Reads from context.facts["plan"]["needs_tests"].
    Defaults to True if no plan exists (graceful degradation).
    """
    plan = context.facts.get("plan")
    if plan is None:
        return True
    return bool(plan.get("needs_tests", True))


def _implement_should_run(context: WorkflowContext) -> bool:
    """Check if the implement stage should run.

    Skips implement for docs_only tasks — the update_docs stage handles
    doc changes instead. Defaults to True if no plan exists (graceful
    degradation).
    """
    plan = context.facts.get("plan")
    if plan is None:
        return True
    return plan.get("task_type") != "docs_only"


def _refactor_should_run(context: WorkflowContext) -> bool:
    """Check if the refactor stage should run.

    Skips refactor for docs_only tasks since there's nothing to refactor.
    Defaults to True if no plan exists (graceful degradation).
    """
    plan = context.facts.get("plan")
    if plan is None:
        return True
    return plan.get("task_type") != "docs_only"


def _docs_should_run(context: WorkflowContext) -> bool:
    """Check if the update_docs stage should run.

    Reads the deterministic decision from context.facts["docs_needed"],
    which is set by the mechanical gate_docs stage.
    Defaults to False if no decision has been recorded.
    """
    return bool(context.facts.get("docs_needed", False))


def _make_gate_docs_fn(repo_path: str):
    """Create a mechanical_fn that decides if docs need updating.

    Checks git diff for changed files and whether the project has
    a docs/ directory. Sets context.facts["docs_needed"] accordingly.
    Docs are needed when non-test source files changed AND the project
    has documentation files that might need updating.
    """

    async def fn(ctx: WorkflowContext) -> StageResult:
        import shlex

        # Get list of files changed on this branch vs main
        diff_cmd = (
            f"git -C {shlex.quote(repo_path)} "
            f"diff --name-only main...HEAD"
        )
        diff_output = await shell(diff_cmd, timeout=15)
        changed_files = [
            f.strip()
            for f in diff_output.strip().splitlines()
            if f.strip()
        ]

        # Check if the project has a docs/ directory with files
        ls_cmd = (
            f"git -C {shlex.quote(repo_path)} "
            f"ls-files docs/"
        )
        ls_output = await shell(ls_cmd, timeout=15)
        has_docs = bool(ls_output.strip())

        # Determine if docs need updating: non-test source files changed
        # AND the project has documentation files
        test_patterns = ("tests/", "test_", "_test.")
        source_files = [
            f for f in changed_files
            if not any(p in f for p in test_patterns)
        ]

        docs_needed = bool(source_files) and has_docs

        # Store the decision in context facts for _docs_should_run
        ctx.facts["docs_needed"] = docs_needed

        if docs_needed:
            summary = (
                f"Docs update needed: {len(source_files)} source files "
                f"changed and project has docs/ directory."
            )
        else:
            reason = (
                "no source files changed"
                if not source_files
                else "no docs/ directory found"
            )
            summary = f"Docs update not needed: {reason}."

        return StageResult(
            stage_name="gate_docs",
            outcome=StageOutcome.PROCEED,
            summary=summary,
        )

    return fn


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
    # Create plan holder — the record_plan tool writes to this,
    # and a post_fn copies the data to context.facts["plan"] after
    # the comment_on_issue stage completes.
    plan_holder = PlanHolder()
    record_plan_tool = make_record_plan_tool(plan_holder)

    # Post-stage function: copy plan from holder to context.facts.
    # This runs after comment_on_issue completes.
    def _copy_plan_to_context(result, ctx: WorkflowContext) -> None:
        if plan_holder.data is not None:
            ctx.facts["plan"] = plan_holder.data

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
                "Post the comment using the comment_on_issue tool.\n\n"
                "Then call record_plan to save your structured plan with:\n"
                "- branch_name: a descriptive git branch name\n"
                "- task_type: one of 'docs_only', 'bug_fix', 'feature', 'refactor'\n"
                "- needs_tests: whether tests are needed\n"
                "- needs_docs: whether docs need updating\n"
                "- files_to_change: list of files to modify\n"
                "- pr_title: suggested PR title\n"
                "- summary: brief plan summary\n"
            ),
            tools=[github_comment, record_plan_tool],
            post_fn=_copy_plan_to_context,
        ),

        # 2. Add "in-progress" label (mechanical)
        Stage(
            name="add_label",
            instruction="Add in-progress label",
            is_mechanical=True,
            mechanical_fn=_make_add_label_fn(repo, issue_number),
        ),

        # 3. Create branch (uses plan's branch name if available)
        Stage(
            name="create_branch",
            instruction=(
                "Create a branch for implementing this issue. "
                "If the implementation plan includes a branch_name, "
                "use that name. Otherwise suggest a descriptive one. "
                "Create the branch using the create_branch tool."
            ),
            tools=[create_branch],
        ),

        # 4. Gate: ready to implement?
        Stage(
            name="gate_ready",
            instruction=(
                "Review the issue and context from previous stages. "
                "Do you have enough information to implement this?\n\n"
                "Call stage_complete with:\n"
                '- outcome="proceed" if you have enough info, OR\n'
                '- outcome="need_clarification" if you need more info.\n\n'
                "Always use one of these exact outcome values. "
                "Put your reasoning in the summary parameter, not the outcome."
            ),
            escape_targets={
                StageOutcome.NEED_CLARIFICATION: "comment_on_issue",
            },
        ),

        # 5. Write tests (red) — skipped if plan says no tests needed
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
            should_run=_tests_should_run,
        ),

        # 6. Implement (green) — skipped for docs_only tasks
        Stage(
            name="implement",
            instruction=(
                "Write the implementation to make the tests pass.\n\n"
                "Use write_source to create or edit source files, "
                "read_source to inspect existing code, and "
                "run_command to run the tests."
            ),
            tools=file_tools + [shell_tool],
            should_run=_implement_should_run,
        ),

        # 7. Refactor — skipped for docs_only tasks
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
            should_run=_refactor_should_run,
        ),

        # 8. Commit changes (mechanical)
        Stage(
            name="commit_changes",
            instruction="Commit all changes to the branch",
            is_mechanical=True,
            mechanical_fn=_make_commit_fn(repo_path),
        ),

        # 9. Gate: docs needed? (mechanical — checks diffs and docs/ dir)
        Stage(
            name="gate_docs",
            instruction="Check if documentation updates are needed",
            is_mechanical=True,
            mechanical_fn=_make_gate_docs_fn(repo_path),
        ),

        # 10. Update docs (conditional)
        Stage(
            name="update_docs",
            instruction=(
                "Update documentation based on the changes made.\n\n"
                "Edit the relevant documentation files."
            ),
            tools=file_tools,
            should_run=_docs_should_run,
        ),

        # 11. Submit PR (uses plan's pr_title/summary if available)
        Stage(
            name="submit_pr",
            instruction=(
                "Write a clear PR description summarizing the changes, "
                "then create the PR using the create_pr tool.\n\n"
                "If the implementation plan includes a pr_title or "
                "summary, use those as a starting point.\n\n"
                "The PR body should reference the issue number and "
                "describe what was changed and why.\n\n"
                "If PR creation fails (e.g. due to auth, network, or "
                "branch issues), call stage_complete with "
                "outcome='blocked' and escape_target='commit_changes' "
                "so the workflow can retry."
            ),
            tools=[create_pr],
            # Recovery: if PR submission fails, jump back to commit
            # so changes can be re-committed or fixed before retrying.
            escape_targets={
                StageOutcome.BLOCKED: "commit_changes",
            },
        ),
    ]
