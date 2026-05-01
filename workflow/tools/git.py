"""Git tools — factory functions that create scoped git CLI tools.

Each factory captures the repo path via closure so tools operate on
the correct repository. Uses git -C to specify the working directory
instead of cd, which is cleaner in async contexts.
"""

from __future__ import annotations

import shlex

from corvidae.tool import Tool
from corvidae.tools.shell import shell


def make_create_branch_tool(repo_path: str) -> Tool:
    """Create a tool for creating and checking out a git branch."""

    async def create_branch(name: str) -> str:
        """Create and checkout a new git branch.

        Args:
            name: The branch name (e.g. 'feat/issue-42').
        """
        cmd = (
            f"git -C {shlex.quote(repo_path)} "
            f"checkout -b {shlex.quote(name)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(create_branch)


def make_create_worktree_tool(repo_path: str) -> Tool:
    """Create a tool for adding a git worktree."""

    async def create_worktree(branch: str, path: str) -> str:
        """Create a git worktree for the branch.

        Args:
            branch: The branch to check out in the worktree.
            path: The directory path for the new worktree.
        """
        cmd = (
            f"git -C {shlex.quote(repo_path)} "
            f"worktree add {shlex.quote(path)} "
            f"{shlex.quote(branch)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(create_worktree)
