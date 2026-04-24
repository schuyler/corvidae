"""Tests for sherman.tools — CoreToolsPlugin and its four tool functions.

All tests are expected to FAIL until sherman/tools.py is implemented.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The import under test — will raise ImportError until the file exists.
from sherman.tools import CoreToolsPlugin, read_file, shell, web_fetch, write_file


# ---------------------------------------------------------------------------
# TestShell
# ---------------------------------------------------------------------------


class TestShell:
    async def test_shell_simple_command(self):
        result = await shell("echo hello")
        assert "hello" in result

    async def test_shell_returns_stderr(self):
        # Write to stderr via shell redirection
        result = await shell("echo errtext >&2")
        assert "STDERR" in result
        assert "errtext" in result

    async def test_shell_nonzero_exit_code(self):
        result = await shell("exit 1")
        assert "1" in result

    async def test_shell_timeout(self):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await shell("sleep 9999")
        assert result == "Error: command timed out after 30 seconds"

    async def test_shell_no_output(self):
        result = await shell("true")
        assert result == "(no output)"


# ---------------------------------------------------------------------------
# TestReadFile
# ---------------------------------------------------------------------------


class TestReadFile:
    async def test_read_file_success(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("file contents here")
        result = await read_file(str(f))
        assert result == "file contents here"

    async def test_read_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        result = await read_file(str(missing))
        assert "error" in result.lower() or "not found" in result.lower() or "no such" in result.lower()

    async def test_read_file_too_large(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x")
        # Mock stat to return size > 1MB
        mock_stat = MagicMock()
        mock_stat.st_size = 1024 * 1024 + 1
        with patch("pathlib.Path.stat", return_value=mock_stat):
            result = await read_file(str(f))
        assert "error" in result.lower() or "too large" in result.lower() or "1mb" in result.lower()

    async def test_read_file_directory(self, tmp_path):
        result = await read_file(str(tmp_path))
        assert "error" in result.lower() or "directory" in result.lower() or "not a file" in result.lower()


# ---------------------------------------------------------------------------
# TestWriteFile
# ---------------------------------------------------------------------------


class TestWriteFile:
    async def test_write_file_success(self, tmp_path):
        target = tmp_path / "out.txt"
        result = await write_file(str(target), "hello world")
        assert "wrote" in result.lower() or str(target) in result
        assert target.read_text() == "hello world"

    async def test_write_file_creates_parents(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.txt"
        result = await write_file(str(target), "nested")
        assert target.exists()
        assert target.read_text() == "nested"

    async def test_write_file_overwrites(self, tmp_path):
        target = tmp_path / "file.txt"
        await write_file(str(target), "first")
        result = await write_file(str(target), "second")
        assert target.read_text() == "second"

    async def test_write_file_permission_error(self, tmp_path):
        target = tmp_path / "file.txt"
        with patch("pathlib.Path.write_text", side_effect=PermissionError("denied")):
            result = await write_file(str(target), "content")
        assert "error" in result.lower() or "permission" in result.lower() or "denied" in result.lower()


# ---------------------------------------------------------------------------
# TestWebFetch
# ---------------------------------------------------------------------------


def _make_mock_session(status: int, text: str) -> MagicMock:
    """Build a mock aiohttp.ClientSession that returns the given status/text."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.text = AsyncMock(return_value=text)

    # Response is used as an async context manager: async with session.get(url) as resp
    mock_response_ctx = AsyncMock()
    mock_response_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response_ctx)

    # Session itself is used as an async context manager
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_session_ctx


class TestWebFetch:
    async def test_web_fetch_success(self):
        mock_session_ctx = _make_mock_session(200, "page content")
        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com")
        assert "page content" in result

    async def test_web_fetch_non_200(self):
        mock_session_ctx = _make_mock_session(404, "not found body")
        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/missing")
        assert "404" in result

    async def test_web_fetch_truncates_large_response(self):
        large_body = "x" * 60000
        mock_session_ctx = _make_mock_session(200, large_body)
        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/big")
        assert len(result) < 60000
        assert "[truncated]" in result

    async def test_web_fetch_timeout(self):
        import aiohttp

        mock_session = MagicMock()
        mock_response_ctx = AsyncMock()
        mock_response_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_session.get = MagicMock(return_value=mock_response_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/slow")
        assert "timed out" in result.lower() or "timeout" in result.lower()

    async def test_web_fetch_connection_error(self):
        import aiohttp

        mock_session = MagicMock()
        mock_response_ctx = AsyncMock()
        mock_response_ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("connection failed"))
        mock_session.get = MagicMock(return_value=mock_response_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await web_fetch("http://example.com/broken")
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# TestCoreToolsPlugin
# ---------------------------------------------------------------------------


class TestCoreToolsPlugin:
    def test_register_tools_adds_four_tools(self):
        plugin = CoreToolsPlugin()
        registry = []
        plugin.register_tools(tool_registry=registry)
        assert len(registry) == 4

    def test_registered_tool_names(self):
        from sherman.tool import Tool
        plugin = CoreToolsPlugin()
        registry = []
        plugin.register_tools(tool_registry=registry)
        # Items are Tool instances after Step 4
        names = {item.name if isinstance(item, Tool) else item.__name__ for item in registry}
        assert names == {"shell", "read_file", "write_file", "web_fetch"}
