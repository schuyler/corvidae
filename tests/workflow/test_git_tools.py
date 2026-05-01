"""Tests for workflow.tools.git — git CLI tool factories."""

import pytest
from unittest.mock import AsyncMock, patch

from workflow.tools.git import (
    make_commit_tool,
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


@pytest.mark.timeout(5)
async def test_commit_tool_adds_and_commits():
    """commit tool runs git add -A then git commit with message."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "[main abc123] Fix the bug"
        tool = make_commit_tool("/repo")
        result = await tool.fn(message="Fix the bug")

        # shell is called once with both commands chained
        assert mock.call_count == 1
        cmd = mock.call_args[0][0]
        assert "add -A" in cmd
        assert "commit -m" in cmd
        assert "Fix the bug" in cmd
        assert "/repo" in cmd


@pytest.mark.timeout(5)
async def test_commit_tool_quotes_message():
    """commit tool shell-escapes the commit message."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_commit_tool("/repo")
        await tool.fn(message="fix; rm -rf /")

        cmd = mock.call_args[0][0]
        # The semicolon should be inside quotes, not a real shell separator
        assert "rm -rf" not in cmd or "'" in cmd


@pytest.mark.timeout(5)
async def test_create_branch_quotes_name():
    """create_branch shell-escapes the branch name."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_create_branch_tool("/repo")
        await tool.fn(name="feat/issue; rm -rf /")

        cmd = mock.call_args[0][0]
        assert "rm -rf" not in cmd or "'" in cmd


# -- Co-authored-by trailer tests --


@pytest.mark.timeout(5)
async def test_commit_includes_co_authored_by():
    """commit tool appends Co-authored-by trailer when agent_name is set."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "[main abc123] Fix the bug"
        tool = make_commit_tool(
            "/repo",
            agent_name="corvidae-workflow",
        )
        await tool.fn(message="Fix the bug")

        cmd = mock.call_args[0][0]
        assert "Co-authored-by" in cmd
        assert "corvidae-workflow" in cmd


@pytest.mark.timeout(5)
async def test_commit_co_authored_by_with_custom_email():
    """commit tool uses custom email in Co-authored-by trailer."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_commit_tool(
            "/repo",
            agent_name="corvidae-bot",
            agent_email="bot@corvidae.dev",
        )
        await tool.fn(message="Fix")

        cmd = mock.call_args[0][0]
        assert "Co-authored-by: corvidae-bot <bot@corvidae.dev>" in cmd


@pytest.mark.timeout(5)
async def test_commit_no_co_authored_by_by_default():
    """commit tool does NOT add Co-authored-by when agent_name is not set."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_commit_tool("/repo")
        await tool.fn(message="Fix")

        cmd = mock.call_args[0][0]
        assert "Co-authored-by" not in cmd
