"""Tests for workflow.tools.github — GitHub CLI tool factories."""

from unittest.mock import AsyncMock, MagicMock, patch

from workflow.tools.github import (
    make_github_comment_tool,
    make_add_label_tool,
    make_create_pr_tool,
)


def _mock_subprocess(stdout: str, rc: int = 0):
    """Create a mock asyncio.subprocess.Process that yields the given stdout."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(
        stdout.encode(), b"",
    ))
    proc.returncode = rc
    return proc


async def test_comment_on_issue():
    """comment_on_issue runs gh CLI with correct arguments."""
    with patch("workflow.tools.github.asyncio.create_subprocess_shell",
               return_value=_mock_subprocess("[]")) as mock_sp, \
         patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock_shell:
        mock_shell.return_value = (
            "https://github.com/o/r/issues/1#issuecomment-1"
        )
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My comment")

        assert "comment" in result
        # subprocess called for dedup check (no --comments flag)
        check_cmd = mock_sp.call_args[0][0]
        assert "--json comments" in check_cmd
        # shell called for posting
        post_cmd = mock_shell.call_args[0][0]
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


async def test_create_pr_dynamic_branch():
    """create_pr reads current branch dynamically when branch is None."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock_shell:
        mock_shell.return_value = "https://github.com/o/r/pull/6"
        # Patch subprocess.run inside the function's local import
        import subprocess as real_subprocess
        from unittest.mock import MagicMock, patch as local_patch
        cp = MagicMock()
        cp.stdout = "fix/dynamic-branch\n"

        with local_patch("subprocess.run", return_value=cp):
            tool = make_create_pr_tool("owner/repo")
            result = await tool.fn(title="Fix bug", body="Description")

            cmd = mock_shell.call_args[0][0]
            assert "fix/dynamic-branch" in cmd


async def test_comment_escapes_body():
    """comment_on_issue shell-escapes the body to prevent injection."""
    with patch("workflow.tools.github.asyncio.create_subprocess_shell",
               return_value=_mock_subprocess("[]")), \
         patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_github_comment_tool("o/r", 1)
        await tool.fn(body="test'; rm -rf /")

        # The post call should have shell-quoted body
        cmd = mock.call_args[0][0]
        assert "rm -rf" not in cmd or "'" in cmd or '"' in cmd


async def test_comment_skips_when_workflow_comment_exists():
    """comment_on_issue skips posting if a workflow comment already exists."""
    comments_json = (
        '[{"body": "Nice idea"},'
        ' {"body": "Plan here\\n\\n<!-- corvidae-workflow -->"}]'
    )
    with patch("workflow.tools.github.asyncio.create_subprocess_shell",
               return_value=_mock_subprocess(comments_json)), \
         patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock_shell:
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My new plan")

        # Should NOT have called shell to post — dedup found the marker
        mock_shell.assert_not_awaited()
        assert "already exists" in result
        assert "Skipping" in result


async def test_comment_posts_when_no_workflow_comment_exists():
    """comment_on_issue posts when existing comments lack the marker."""
    with patch("workflow.tools.github.asyncio.create_subprocess_shell",
               return_value=_mock_subprocess('[{"body": "Nice idea"}]')), \
         patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock_shell:
        mock_shell.return_value = (
            "https://github.com/o/r/issues/1#issuecomment-1"
        )
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My plan")

        mock_shell.assert_awaited_once()
        assert "comment" in result


async def test_comment_posts_when_comments_fetch_fails():
    """comment_on_issue posts anyway if comment check fails (fail open)."""
    with patch("workflow.tools.github.asyncio.create_subprocess_shell",
               return_value=_mock_subprocess("not valid json")), \
         patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock_shell:
        mock_shell.return_value = (
            "https://github.com/o/r/issues/1#issuecomment-1"
        )
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My plan")

        mock_shell.assert_awaited_once()
        assert "comment" in result


async def test_comment_posts_when_no_comments_exist():
    """comment_on_issue posts when the issue has zero comments."""
    with patch("workflow.tools.github.asyncio.create_subprocess_shell",
               return_value=_mock_subprocess("[]")), \
         patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock_shell:
        mock_shell.return_value = (
            "https://github.com/o/r/issues/1#issuecomment-1"
        )
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My plan")

        mock_shell.assert_awaited_once()
        assert "comment" in result
