"""Tests for workflow.tools.git — git CLI tool factories."""

from unittest.mock import AsyncMock, patch

from workflow.tools.git import (
    make_create_branch_tool,
    make_create_worktree_tool,
)


async def test_create_branch():
    """create_branch runs git checkout -b in the repo directory."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "Switched to a new branch 'feat/issue-1'"
        tool = make_create_branch_tool("/repo")
        result = await tool.fn(name="feat/issue-1")

        assert "Switched" in result
        cmd = mock.call_args[0][0]
        assert "checkout -b" in cmd
        assert "feat/issue-1" in cmd
        assert "/repo" in cmd


async def test_create_worktree():
    """create_worktree runs git worktree add."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "Preparing worktree"
        tool = make_create_worktree_tool("/repo")
        result = await tool.fn(branch="feat/issue-1", path="/repo-wt")

        cmd = mock.call_args[0][0]
        assert "worktree add" in cmd
        assert "feat/issue-1" in cmd
        assert "/repo-wt" in cmd


async def test_create_branch_quotes_name():
    """create_branch shell-escapes the branch name."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_create_branch_tool("/repo")
        await tool.fn(name="feat/issue; rm -rf /")

        cmd = mock.call_args[0][0]
        assert "rm -rf" not in cmd or "'" in cmd
