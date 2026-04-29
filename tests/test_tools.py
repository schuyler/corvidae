"""Tests for corvidae.tools — CoreToolsPlugin and its four tool functions.

All tests are expected to FAIL until corvidae/tools.py is implemented.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The import under test — will raise ImportError until the file exists.
from corvidae.tools import CoreToolsPlugin, read_file, shell, web_fetch, write_file, web_search


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


def _make_mock_session(status: int, text: str, max_response_bytes: int = 50_000) -> MagicMock:
    """Build a mock aiohttp.ClientSession that returns the given status/text.

    Short responses (those that fit within max_response_bytes) are modelled by
    having readexactly raise IncompleteReadError so the caller receives the
    partial bytes without a truncation indicator.  Large responses are modelled
    by returning exactly max_response_bytes bytes from readexactly so the caller
    appends the truncation indicator — the caller never sees the remainder.

    For simplicity this helper always encodes text as UTF-8 and sets the mock
    encoding to "utf-8".
    """
    text_bytes = text.encode("utf-8")

    mock_content = MagicMock()
    if len(text_bytes) >= max_response_bytes:
        # Simulate a full read: readexactly returns exactly max_response_bytes bytes,
        # which causes the production code to append TRUNCATION_INDICATOR.
        mock_content.readexactly = AsyncMock(return_value=text_bytes[:max_response_bytes])
    else:
        # Simulate a response shorter than the limit: readexactly raises
        # IncompleteReadError carrying all the bytes.
        incomplete_error = asyncio.IncompleteReadError(text_bytes, None)
        mock_content.readexactly = AsyncMock(side_effect=incomplete_error)

    mock_response = MagicMock()
    mock_response.status = status
    mock_response.content = mock_content
    mock_response.get_encoding = MagicMock(return_value="utf-8")

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
# TestWebFetchStreaming — Fix 2: stream-limit response body via readexactly
# ---------------------------------------------------------------------------


def _make_mock_session_streaming(
    status: int,
    *,
    readexactly_return: bytes | None = None,
    readexactly_error: asyncio.IncompleteReadError | None = None,
    encoding: str = "utf-8",
) -> MagicMock:
    """Build a mock aiohttp.ClientSession using readexactly/get_encoding.

    Pass exactly one of readexactly_return (bytes) or readexactly_error.
    """
    mock_content = MagicMock()
    if readexactly_error is not None:
        mock_content.readexactly = AsyncMock(side_effect=readexactly_error)
    else:
        mock_content.readexactly = AsyncMock(return_value=readexactly_return)

    mock_response = MagicMock()
    mock_response.status = status
    mock_response.content = mock_content
    mock_response.get_encoding = MagicMock(return_value=encoding)

    mock_response_ctx = AsyncMock()
    mock_response_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response_ctx)

    return mock_session


class TestWebFetchStreaming:
    """Tests for Fix 2: streaming response body with readexactly."""

    async def test_web_fetch_uses_readexactly(self):
        """web_fetch_with_session must call response.content.readexactly(max_response_bytes)."""
        from corvidae.tools.web import web_fetch_with_session

        partial_data = b"short response"
        error = asyncio.IncompleteReadError(partial_data, None)
        mock_session = _make_mock_session_streaming(
            200, readexactly_error=error
        )
        await web_fetch_with_session(mock_session, "http://example.com", max_response_bytes=50_000)
        mock_session.get.return_value.__aenter__.return_value.content.readexactly.assert_called_once_with(50_000)

    async def test_web_fetch_truncation_on_full_read(self):
        """When readexactly returns exactly max_response_bytes, result ends with TRUNCATION_INDICATOR."""
        from corvidae.tools.web import TRUNCATION_INDICATOR, web_fetch_with_session

        max_bytes = 100
        full_data = b"a" * max_bytes
        mock_session = _make_mock_session_streaming(
            200, readexactly_return=full_data
        )
        result = await web_fetch_with_session(
            mock_session, "http://example.com", max_response_bytes=max_bytes
        )
        assert result.endswith(TRUNCATION_INDICATOR), (
            f"Expected result to end with {TRUNCATION_INDICATOR!r}, got: {result!r}"
        )

    async def test_web_fetch_no_truncation_on_short_response(self):
        """When readexactly raises IncompleteReadError with partial data, no truncation indicator."""
        from corvidae.tools.web import TRUNCATION_INDICATOR, web_fetch_with_session

        partial_data = b"short response"
        error = asyncio.IncompleteReadError(partial_data, None)
        mock_session = _make_mock_session_streaming(
            200, readexactly_error=error
        )
        result = await web_fetch_with_session(
            mock_session, "http://example.com", max_response_bytes=50_000
        )
        assert TRUNCATION_INDICATOR not in result, (
            f"Expected no truncation indicator for short response, got: {result!r}"
        )
        assert "short response" in result

    async def test_web_fetch_decode_error_replaced(self):
        """Incomplete UTF-8 sequence in partial data must not raise; replacement char present."""
        from corvidae.tools.web import web_fetch_with_session

        # b"\xe2\x82" is an incomplete 3-byte UTF-8 sequence (euro sign is \xe2\x82\xac)
        incomplete_utf8 = b"hello " + b"\xe2\x82"
        error = asyncio.IncompleteReadError(incomplete_utf8, None)
        mock_session = _make_mock_session_streaming(
            200, readexactly_error=error
        )
        # Must not raise UnicodeDecodeError
        result = await web_fetch_with_session(
            mock_session, "http://example.com", max_response_bytes=50_000
        )
        assert "\ufffd" in result, (
            f"Expected Unicode replacement character in result, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# TestCoreToolsPlugin
# ---------------------------------------------------------------------------


class TestCoreToolsPlugin:
    def test_register_tools_adds_six_tools(self):
        plugin = CoreToolsPlugin(None)
        registry = []
        plugin.register_tools(tool_registry=registry)
        assert len(registry) == 6

    def test_registered_tool_names(self):
        from corvidae.tool import Tool
        plugin = CoreToolsPlugin(None)
        registry = []
        plugin.register_tools(tool_registry=registry)
        names = {item.name if isinstance(item, Tool) else item.__name__ for item in registry}
        assert names == {"task_pipeline", "shell", "read_file", "write_file", "web_fetch", "web_search"}

    def test_web_search_max_results_config(self):
        plugin = CoreToolsPlugin(None)
        assert plugin._web_search_max_results == 8

        plugin2 = CoreToolsPlugin(None)
        import asyncio
        asyncio.run(plugin2.on_start(config={"tools": {"web_search_max_results": 15}}))
        try:
            assert plugin2._web_search_max_results == 15
        finally:
            asyncio.run(plugin2.on_stop())


# ---------------------------------------------------------------------------
# TestWebFetchSessionReuse
# ---------------------------------------------------------------------------


class TestWebFetchSessionReuse:
    def test_core_tools_plugin_has_on_start(self):
        plugin = CoreToolsPlugin(None)
        assert hasattr(plugin, "on_start"), "CoreToolsPlugin must have an on_start method"
        assert callable(plugin.on_start)

    def test_core_tools_plugin_has_on_stop(self):
        plugin = CoreToolsPlugin(None)
        assert hasattr(plugin, "on_stop"), "CoreToolsPlugin must have an on_stop method"
        assert callable(plugin.on_stop)

    async def test_session_created_on_start(self):
        import aiohttp
        plugin = CoreToolsPlugin(None)
        await plugin.on_start(config={})
        try:
            assert hasattr(plugin, "_session"), "CoreToolsPlugin must have a _session attribute after on_start"
            assert isinstance(plugin._session, aiohttp.ClientSession)
        finally:
            if hasattr(plugin, "_session") and plugin._session is not None:
                await plugin._session.close()

    async def test_session_closed_on_stop(self):
        plugin = CoreToolsPlugin(None)
        await plugin.on_start(config={})
        session = plugin._session
        await plugin.on_stop()
        assert session.closed, "Session must be closed after on_stop"


# ---------------------------------------------------------------------------
# TestFileToolsAsyncIO
# ---------------------------------------------------------------------------


class TestFileToolsAsyncIO:
    async def test_read_file_uses_thread(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("contents")
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            await read_file(str(f))
        mock_to_thread.assert_called()

    async def test_write_file_uses_thread(self, tmp_path):
        target = tmp_path / "out.txt"
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            await write_file(str(target), "data")
        mock_to_thread.assert_called()


# ---------------------------------------------------------------------------
# TestWebSearch — Mocked DuckDuckGo integration
# ---------------------------------------------------------------------------


def _make_ddgs_mock(search_results):
    """Return a MagicMock that mimics DDGS().text() yielding the given results.

    search_results should be an iterable of dicts with optional keys:
      title (str), href (str), body (str)
    """
    mock_ddgs = MagicMock()
    mock_ctx = MagicMock()
    mock_ddgs.text.return_value = search_results
    mock_ctx.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


class TestWebSearch:
    async def test_web_search_returns_formatted_results(self):
        results = [
            {"title": "First Result", "href": "http://example.com/1", "body": "snippet one"},
            {"title": "Second Result", "href": "http://example.com/2", "body": "snippet two"},
        ]
        mock_ctx = _make_ddgs_mock(results)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("test query")
        assert "test query" in output
        assert "First Result" in output
        assert "http://example.com/1" in output
        assert "snippet one" in output
        assert "Second Result" in output

    async def test_web_search_skips_results_without_url(self):
        results = [
            {"title": "Bad Entry", "href": "", "body": "no url"},
            {"title": "Good Entry", "href": "http://good.com", "body": "has url"},
        ]
        mock_ctx = _make_ddgs_mock(results)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("test")
        assert "Bad Entry" not in output
        assert "Good Entry" in output

    async def test_web_search_no_results(self):
        mock_ctx = _make_ddgs_mock([])
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("zxcvbnmqwerty no results ever")
        assert "No results found." in output

    async def test_web_search_custom_max_results(self):
        # The web_search function passes max_results to ddgs.text(). We verify
        # the count by checking the output only contains 3 result blocks when
        # the mock yields more than 3 results total.
        all_results = [
            {"title": f"Result {i}", "href": f"http://example.com/{i}", "body": f"snippet {i}"}
            for i in range(20)
        ]
        mock_ddgs = MagicMock()
        # ddgs.text(query, max_results=3) should only return the first 3 items
        mock_ddgs.text.return_value = all_results[:3]
        mock_ctx = MagicMock()
        mock_ddgs.text.return_value = all_results[:3]
        mock_ctx.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        async def _mock_to_thread(fn):
            return fn()

        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            with patch("asyncio.to_thread", new=_mock_to_thread):
                output = await web_search("test", max_results=3)
        # Count how many separator blocks appear (3 results = 2 separators)
        assert output.count("=" * 60) == 2, f"Expected 2 separators for 3 results, got {output[:200]}"

    async def test_web_search_error_handling(self):
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=RuntimeError("network down"))
        mock_ctx.__exit__ = MagicMock(return_value=False)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("error test")
        assert "Error searching the web" in output

    async def test_web_search_separator(self):
        results = [
            {"title": "R1", "href": "http://a.com", "body": "s1"},
            {"title": "R2", "href": "http://b.com", "body": "s2"},
        ]
        mock_ctx = _make_ddgs_mock(results)
        with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
            output = await web_search("sep test")
        assert "=" * 60 in output

    async def test_web_search_uses_asyncio_to_thread(self, tmp_path):
        """web_search must run ddgs via asyncio.to_thread to avoid blocking."""
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            mock_ctx = _make_ddgs_mock([])
            with patch("corvidae.tools.web.DDGS", return_value=mock_ctx):
                await web_search("to thread test")
        mock_to_thread.assert_called_once()
