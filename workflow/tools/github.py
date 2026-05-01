"""GitHub tools — factory functions that create scoped gh CLI tools.

Each factory captures repo/issue context via closure, so the LLM
doesn't need to know the repo or issue number — it just calls the
tool with the content. This prevents the LLM from targeting a
different repo.

Uses the gh CLI (must be installed and authenticated) for all
GitHub operations. Shell arguments are properly quoted to prevent
injection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex

from corvidae.tool import Tool
from corvidae.tools.shell import shell

logger = logging.getLogger(__name__)

# Hidden HTML comment appended to workflow comments so we can detect
# them on re-runs and avoid posting duplicates.
_WORKFLOW_MARKER = "<!-- corvidae-workflow -->"


def make_github_comment_tool(repo: str, issue_number: int) -> Tool:
    """Create a tool for posting comments on a GitHub issue.

    Includes deduplication: before posting, the tool checks whether
    the issue already has a comment tagged with the corvidae-workflow
    marker. If one is found the comment is skipped and a message
    is returned instead.
    """

    async def _existing_workflow_comment() -> str | None:
        """Return the body of an existing workflow comment, or None.

        Uses subprocess directly (not the shared shell() helper) so
        we can parse stdout separately from stderr. The shell() helper
        concatenates both, which breaks JSON parsing when gh emits
        warnings to stderr.
        """
        cmd = (
            f"gh issue view {issue_number} "
            f"--repo {shlex.quote(repo)} "
            f"--json comments -q .comments"
        )
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
        except (OSError, asyncio.TimeoutError):
            # Can't run the check — fail open.
            logger.debug(
                "Failed to check existing comments", exc_info=True,
            )
            return None

        raw = stdout_bytes.decode(errors="replace").strip()
        try:
            comments = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # If we can't parse comments, assume none exist and
            # let the post proceed — fail open rather than block.
            logger.debug("Could not parse comments JSON: %s", raw[:200])
            return None
        for comment in comments:
            body = comment.get("body", "")
            if _WORKFLOW_MARKER in body:
                return body
        return None

    async def comment_on_issue(body: str) -> str:
        """Post a comment on the GitHub issue.

        Args:
            body: The comment text to post.
        """
        # Deduplication: if a workflow comment already exists, return
        # its body so the LLM can review it and decide whether to
        # update it or proceed without changes.
        existing = await _existing_workflow_comment()
        if existing is not None:
            logger.info(
                "Workflow comment already exists on %s#%d, returning for review",
                repo, issue_number,
            )
            return (
                "A workflow comment already exists on this issue. "
                "Here is its current content:\n\n"
                + existing
                + "\n\nReview this comment. If it is still accurate and "
                "complete, call stage_complete with outcome='proceed' "
                "without posting a new comment. If it needs changes, "
                "post an updated comment."
            )

        # Append hidden marker so future runs can detect this comment.
        tagged_body = body + "\n\n" + _WORKFLOW_MARKER
        cmd = (
            f"gh issue comment {issue_number} "
            f"--repo {shlex.quote(repo)} "
            f"--body {shlex.quote(tagged_body)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(comment_on_issue)


def make_add_label_tool(repo: str, issue_number: int) -> Tool:
    """Create a tool for adding labels to a GitHub issue."""

    async def add_label(label: str) -> str:
        """Add a label to the GitHub issue.

        Args:
            label: The label to add (e.g. 'in-progress').
        """
        cmd = (
            f"gh issue edit {issue_number} "
            f"--repo {shlex.quote(repo)} "
            f"--add-label {shlex.quote(label)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(add_label)


def make_create_pr_tool(repo: str, branch: str | None = None) -> Tool:
    """Create a tool for opening a pull request.

    If branch is None, the tool dynamically reads the current branch
    from git at invocation time. This ensures the correct branch name
    is used even when the tool is created before the branch exists.
    """

    async def create_pr(title: str, body: str) -> str:
        """Create a pull request on GitHub.

        Args:
            title: The PR title.
            body: The PR description.
        """
        # Resolve branch name dynamically if not provided
        head_branch = branch
        if head_branch is None:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=15,
            )
            head_branch = result.stdout.strip()

        cmd = (
            f"gh pr create "
            f"--repo {shlex.quote(repo)} "
            f"--head {shlex.quote(head_branch)} "
            f"--title {shlex.quote(title)} "
            f"--body {shlex.quote(body)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(create_pr)
