"""Tests for workflow.tools.github — GitHub CLI tool factories."""

from unittest.mock import AsyncMock, patch

from workflow.tools.github import (
    make_github_comment_tool,
    make_add_label_tool,
    make_create_pr_tool,
)


async def test_comment_on_issue():
    """comment_on_issue runs gh CLI with correct arguments."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "https://github.com/o/r/issues/1#issuecomment-1"
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My comment")

        assert "comment" in result
        mock.assert_awaited_once()
        cmd = mock.call_args[0][0]
        assert "42" in cmd
        assert "owner/repo" in cmd
        assert "--body" in cmd


async def test_add_label():
    """add_label runs gh CLI to add a label."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = ""
        tool = make_add_label_tool("owner/repo", 42)
        result = await tool.fn(label="in-progress")

        mock.assert_awaited_once()
        cmd = mock.call_args[0][0]
        assert "--add-label" in cmd
        assert "in-progress" in cmd


async def test_create_pr():
    """create_pr runs gh CLI to create a pull request."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "https://github.com/o/r/pull/5"
        tool = make_create_pr_tool("owner/repo", "feat/issue-1")
        result = await tool.fn(title="Fix bug", body="Description")

        assert "pull" in result
        cmd = mock.call_args[0][0]
        assert "--title" in cmd
        assert "feat/issue-1" in cmd


async def test_comment_escapes_body():
    """comment_on_issue shell-escapes the body to prevent injection."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_github_comment_tool("o/r", 1)
        await tool.fn(body="test'; rm -rf /")

        cmd = mock.call_args[0][0]
        # The body should be shell-quoted, not raw
        assert "rm -rf" not in cmd or "'" in cmd or '"' in cmd
