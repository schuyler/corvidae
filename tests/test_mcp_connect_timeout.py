"""McpClientPlugin connect timeout (#72).

A configured MCP server that never answers ``initialize`` must not block
daemon startup: it times out after ``connect_timeout_seconds``, is logged
as unavailable, and the remaining servers still connect. These tests use
real stdio subprocesses — ``sleep`` as the server that never answers, and
``tests/fixtures/mcp_echo_server.py`` as a working one — because the
failure being guarded against lives in the transport, not in our mocks.
"""

import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corvidae.mcp_client import McpClientPlugin

ECHO_SERVER = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"

# Skip rather than fail where the hung-server stand-in is unavailable.
pytestmark = pytest.mark.skipif(shutil.which("sleep") is None, reason="needs `sleep`")


def _hung_server(connect_timeout: float | None = 0.5, duration: str = "1000") -> dict:
    """A stdio "server" that never speaks MCP, so initialize never returns.
    ``duration`` lets a test give its process a unique command line."""
    cfg = {"transport": "stdio", "command": "sleep", "args": [duration]}
    if connect_timeout is not None:
        cfg["connect_timeout_seconds"] = connect_timeout
    return cfg


def _echo_server() -> dict:
    """A real stdio MCP server exposing one `echo` tool."""
    return {"transport": "stdio", "command": sys.executable, "args": [str(ECHO_SERVER)]}


async def _start(servers: dict) -> McpClientPlugin:
    """Init and start a plugin with the given server configs."""
    plugin = McpClientPlugin(None)
    config = {"mcp": {"servers": servers}}
    await plugin.on_init(pm=MagicMock(), config=config)
    try:
        await plugin.on_start(config=config)
    except BaseException:
        # Close whatever did connect, or its stdio subprocess outlives the
        # test and hangs event-loop teardown past the pytest timeout.
        await plugin.on_stop()
        raise
    return plugin


class TestConnectTimeout:
    async def test_hung_server_times_out_and_startup_completes(self, caplog):
        """on_start returns promptly, the hung server is skipped, and the
        warning names the timeout."""
        caplog.set_level(logging.WARNING, logger="corvidae.mcp_client")
        started = time.monotonic()
        plugin = await _start({"hung": _hung_server(0.5)})
        try:
            elapsed = time.monotonic() - started
            assert plugin._servers == []
            assert plugin._cached_tools == []
            # Bounded: the 0.5s connect timeout plus subprocess teardown.
            assert elapsed < 10
            timeout_records = [
                r for r in caplog.records
                if "timed out" in r.getMessage() and getattr(r, "server", None) == "hung"
            ]
            assert timeout_records, [r.getMessage() for r in caplog.records]
            # Logged once: the timeout does not also reach on_start's skip.
            assert not any("failed to connect" in r.getMessage() for r in caplog.records)
        finally:
            await plugin.on_stop()

    async def test_good_server_still_connects_alongside_hung_one(self):
        """A hung server does not stop a working one from registering tools."""
        plugin = await _start({"hung": _hung_server(0.5), "echo": _echo_server()})
        try:
            assert [s.name for s in plugin._servers] == ["echo"]
            assert [t.name for t in plugin._cached_tools] == ["echo__echo"]
        finally:
            await plugin.on_stop()

    async def test_hung_server_subprocess_is_terminated_on_timeout(self):
        """The timed-out server's process is reaped at timeout, not left
        running until shutdown: this test's own `sleep` child (unique
        duration, so other tests' servers cannot match) is gone after on_start."""
        if shutil.which("pgrep") is None:
            pytest.skip("needs `pgrep`")
        duration = "1000.4271"
        plugin = await _start({"hung": _hung_server(0.5, duration)})
        try:
            proc = await asyncio.create_subprocess_exec(
                # Only this process's own children, so a stray process left
                # by another run cannot fail the test.
                "pgrep", "-P", str(os.getpid()), "-f", f"^sleep {duration}$",
                stdout=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            assert out.strip() == b"", f"hung server still running: {out!r}"
        finally:
            await plugin.on_stop()

    async def test_connect_timeout_defaults_to_30_seconds(self, monkeypatch):
        """With no connect_timeout_seconds, the bound passed to the connect
        is 30s."""
        seen: list[float] = []
        real_timeout = asyncio.timeout

        def recording_timeout(delay):
            seen.append(delay)
            return real_timeout(0.5)

        monkeypatch.setattr("corvidae.mcp_client.asyncio.timeout", recording_timeout)
        plugin = await _start({"hung": _hung_server(None)})
        try:
            assert seen == [30.0]
            assert plugin._servers == []
        finally:
            await plugin.on_stop()


class TestConnectTimeoutValidation:
    async def test_non_positive_connect_timeout_skips_server(self, caplog):
        """connect_timeout_seconds must be positive: 0 is a config error, so
        the server is skipped with the usual failed-to-connect warning."""
        caplog.set_level(logging.WARNING, logger="corvidae.mcp_client")
        plugin = await _start({"bad": _hung_server(0)})
        try:
            assert plugin._servers == []
            assert any(
                "failed to connect" in r.getMessage() and r.exc_info
                and "must be positive" in str(r.exc_info[1])
                for r in caplog.records
            )
        finally:
            await plugin.on_stop()


class TestRealServer:
    async def test_real_server_tool_registers_and_calls(self):
        """Against a real mcp 2.x server, the tool's schema translates and a
        call returns its text (mcp 2 renamed inputSchema/isError)."""
        plugin = await _start({"echo": _echo_server()})
        try:
            (tool,) = plugin._cached_tools
            params = tool.schema["function"]["parameters"]
            assert params["properties"]["text"]["type"] == "string"
            assert await tool.fn(text="hi there") == "hi there"
        finally:
            await plugin.on_stop()
