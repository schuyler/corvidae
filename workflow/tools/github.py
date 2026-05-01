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

import shlex

from corvidae.tool import Tool
from corvidae.tools.shell import shell


def make_github_comment_tool(repo: str, issue_number: int) -> Tool:
    """Create a tool for posting comments on a GitHub issue."""

    async def comment_on_issue(body: str) -> str:
        """Post a comment on the GitHub issue.

        Args:
            body: The comment text to post.
        """
        cmd = (
            f"gh issue comment {issue_number} "
            f"--repo {shlex.quote(repo)} "
            f"--body {shlex.quote(body)}"
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


def make_create_pr_tool(repo: str, branch: str) -> Tool:
    """Create a tool for opening a pull request."""

    async def create_pr(title: str, body: str) -> str:
        """Create a pull request on GitHub.

        Args:
            title: The PR title.
            body: The PR description.
        """
        cmd = (
            f"gh pr create "
            f"--repo {shlex.quote(repo)} "
            f"--head {shlex.quote(branch)} "
            f"--title {shlex.quote(title)} "
            f"--body {shlex.quote(body)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(create_pr)
