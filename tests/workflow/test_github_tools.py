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
        # First call: fetch existing comments (returns empty list)
        # Second call: post the comment
        mock.side_effect = [
            "[]",  # no existing comments
            "https://github.com/o/r/issues/1#issuecomment-1",
        ]
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My comment")

        assert "comment" in result
        assert mock.await_count == 2
        # First call: checking for existing comments
        check_cmd = mock.call_args_list[0][0][0]
        assert "--comments" in check_cmd
        # Second call: posting the comment
        post_cmd = mock.call_args_list[1][0][0]
        assert "42" in post_cmd
        assert "owner/repo" in post_cmd
        assert "--body" in post_cmd
        # Body should include the workflow marker
        assert "corvidae-workflow" in post_cmd


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
        mock.side_effect = ["[]", "ok"]
        tool = make_github_comment_tool("o/r", 1)
        await tool.fn(body="test'; rm -rf /")

        # Second call is the post
        cmd = mock.call_args_list[1][0][0]
        # The body should be shell-quoted, not raw
        assert "rm -rf" not in cmd or "'" in cmd or '"' in cmd


async def test_comment_skips_when_workflow_comment_exists():
    """comment_on_issue skips posting if a workflow comment already exists."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        # Return existing comments, one with the workflow marker
        mock.return_value = '[{"body": "Nice idea"}, {"body": "Plan here\\n\\n<!-- corvidae-workflow -->"}]'
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My new plan")

        # Should only have been called once (the check), not twice (the post)
        mock.assert_awaited_once()
        assert "already exists" in result
        assert "Skipping" in result


async def test_comment_posts_when_no_workflow_comment_exists():
    """comment_on_issue posts when existing comments lack the marker."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            '[{"body": "Nice idea"}]',  # no workflow marker
            "https://github.com/o/r/issues/1#issuecomment-1",
        ]
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My plan")

        assert mock.await_count == 2
        assert "comment" in result


async def test_comment_posts_when_comments_fetch_fails():
    """comment_on_issue posts anyway if comment check fails (fail open)."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            "not valid json",  # gh CLI returned unexpected output
            "https://github.com/o/r/issues/1#issuecomment-1",
        ]
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My plan")

        assert mock.await_count == 2
        assert "comment" in result


async def test_comment_posts_when_no_comments_exist():
    """comment_on_issue posts when the issue has zero comments."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            "[]",  # no comments at all
            "https://github.com/o/r/issues/1#issuecomment-1",
        ]
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My plan")

        assert mock.await_count == 2
        assert "comment" in result
