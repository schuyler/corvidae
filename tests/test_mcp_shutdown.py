"""McpClientPlugin shutdown with per-server owner tasks (#213).

apluggy's ``gather`` runs ``on_start`` and ``on_stop`` in different tasks, and
anyio requires a cancel scope to exit in the task that entered it. These
tests run the two hooks in separate ``create_task`` calls against real stdio
and streamable-HTTP servers, and check that shutdown is clean (R1), closes
each transport normally (R2), survives a dead server (R4), and stays inside
Runtime's 3 s stop budget (R5).
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corvidae import mcp_client
from corvidae.mcp_client import McpClientPlugin

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "mcp_echo_server.py")


def _stdio_cfg() -> dict:
    """Config for the real stdio echo server fixture."""
    return {"transport": "stdio", "command": sys.executable,
            "args": [ECHO_SERVER], "connect_timeout_seconds": 20}


def _child_pids() -> list[int]:
    """PIDs of this process's direct children (the stdio server subprocesses)."""
    out = subprocess.run(["pgrep", "-P", str(os.getpid())],
                         capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


async def _wait_no_children(timeout: float = 5) -> list[int]:
    """Poll until this process has no child processes; return what is left."""
    deadline = time.monotonic() + timeout
    while _child_pids() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    return _child_pids()


def _make_plugin(servers: dict) -> tuple[McpClientPlugin, dict]:
    """A plugin initialised with the given server configs."""
    plugin = McpClientPlugin(None)
    return plugin, {"mcp": {"servers": servers}}


async def _start_in_own_task(plugin: McpClientPlugin, config: dict) -> None:
    """Run init and on_start in one fresh task, as the broadcast does."""
    async def run() -> None:
        await plugin.on_init(pm=MagicMock(), config=config)
        await plugin.on_start(config=config)
    await asyncio.create_task(run())


async def _stop_in_own_task(plugin: McpClientPlugin) -> None:
    """Run on_stop in a different fresh task."""
    await asyncio.create_task(plugin.on_stop())


@pytest.fixture(params=["stdio", "streamable_http"])
def transport_cfg(request):
    """Server config for each real transport (the HTTP URL is resolved lazily)."""
    if request.param == "stdio":
        return request.param, _stdio_cfg()
    from mcp_http_server import TOKEN
    url = request.getfixturevalue("mcp_url")
    return request.param, {"transport": "streamable_http", "url": url,
                           "headers": {"Authorization": f"Bearer {TOKEN}"},
                           "connect_timeout_seconds": 10}


class TestCrossTaskShutdown:
    async def test_stop_in_other_task_is_clean_and_closes_transport(
        self, transport_cfg, mcp_request_log, caplog
    ):
        # R1 + R2: start and stop in different tasks; nothing warns, the
        # stdio subprocess is gone, the HTTP session got its DELETE.
        kind, cfg = transport_cfg
        plugin, config = _make_plugin({"srv": cfg})
        caplog.set_level(logging.DEBUG)
        await _start_in_own_task(plugin, config)
        assert [s.name for s in plugin._servers] == ["srv"]
        if kind == "stdio":
            assert _child_pids()

        await _stop_in_own_task(plugin)

        bad = [r for r in caplog.records
               if r.levelno >= logging.WARNING
               or "different task" in r.getMessage()
               or "exit stack" in r.getMessage()]
        assert bad == [], [r.getMessage() for r in bad]
        if kind == "stdio":
            assert await _wait_no_children() == []
        else:
            assert mcp_request_log.issued_sessions
            assert mcp_request_log.deletes() == mcp_request_log.issued_sessions[:1]
        assert plugin._servers == [] and plugin._cached_tools == []


class TestDeadServer:
    async def test_server_killed_after_connect_does_not_break_stop(self, caplog):
        # R4: a stdio server killed after connect; on_stop completes.
        plugin, config = _make_plugin({"srv": _stdio_cfg(), "other": _stdio_cfg()})
        await _start_in_own_task(plugin, config)
        victim = _child_pids()[0]
        os.kill(victim, signal.SIGKILL)
        await asyncio.sleep(0.3)
        caplog.set_level(logging.DEBUG)

        await asyncio.wait_for(_stop_in_own_task(plugin), 3)

        # No error escaped; any warning names a server; the survivor is gone.
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) <= 1
        assert await _wait_no_children() == []

    async def test_owner_failure_after_ready_logs_one_warning(self, caplog):
        # R4: an exception raised while the owner closes its transport is
        # logged once with the server, type and traceback, not raised.
        plugin, config = _make_plugin({"srv": _stdio_cfg(), "other": _stdio_cfg()})
        await _start_in_own_task(plugin, config)

        # Make the "srv" owner's teardown raise.
        owner = plugin._owners["srv"]
        owner.stack.push_async_callback(_raise_on_close)
        caplog.set_level(logging.DEBUG)

        await _stop_in_own_task(plugin)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "srv" in warnings[0].getMessage() or warnings[0].__dict__.get("server") == "srv"
        assert "ZeroDivisionError" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None
        # The other server was still torn down.
        assert await _wait_no_children() == []


async def _raise_on_close() -> None:
    """Teardown callback that fails, standing in for a broken transport."""
    raise ZeroDivisionError("boom")


class TestStopBudget:
    async def test_two_stdio_servers_stop_concurrently(self):
        # R5: both owners are signalled at once; stop is well under 3 s.
        plugin, config = _make_plugin({"a": _stdio_cfg(), "b": _stdio_cfg()})
        await _start_in_own_task(plugin, config)
        assert len(plugin._servers) == 2

        started = time.monotonic()
        await _stop_in_own_task(plugin)

        assert time.monotonic() - started < 2.0
        assert await _wait_no_children() == []

    async def test_hung_teardown_is_cancelled_and_logged(self, caplog, monkeypatch):
        # R5: a teardown that hangs past the bound is cancelled and logged
        # at WARNING with the server name and the by-hand remedy.
        monkeypatch.setattr(mcp_client, "_STOP_BOUND_SECONDS", 0.3)
        plugin, config = _make_plugin({"srv": _stdio_cfg()})
        await _start_in_own_task(plugin, config)

        async def hang() -> None:
            await asyncio.sleep(60)

        plugin._owners["srv"].stack.push_async_callback(hang)
        caplog.set_level(logging.DEBUG)

        started = time.monotonic()
        await _stop_in_own_task(plugin)

        assert time.monotonic() - started < 2.0
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "srv" in message
        assert "killed by hand" in message


class TestOwnerLifecycle:
    async def test_server_started_from_tool_call_is_stopped_by_on_stop(self, caplog):
        # A server pending at startup connects on the first tool call (also
        # through an owner task) and is closed by on_stop from another task.
        good = _stdio_cfg()
        plugin, config = _make_plugin({
            "up": good,
            "late": {**good, "command": "definitely-not-a-real-binary-213"},
        })
        await _start_in_own_task(plugin, config)
        assert list(plugin._pending_servers) == ["late"]
        plugin._pending_servers["late"] = good  # now it can connect
        caplog.set_level(logging.DEBUG)
        caplog.clear()  # drop the expected startup failure of "late"

        tool = next(t for t in plugin._cached_tools if t.name == "up__echo")
        assert await asyncio.create_task(tool.fn(text="hi")) == "hi"
        assert sorted(s.name for s in plugin._servers) == ["late", "up"]
        assert len(_child_pids()) == 2

        await _stop_in_own_task(plugin)

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        assert await _wait_no_children() == []

    async def test_cancelled_on_start_stops_started_owners(self):
        # A CancelledError delivered mid-connect must stop and await the
        # owners already started, so no subprocess outlives the cancel.
        hang = {"transport": "stdio", "command": sys.executable,
                "args": ["-c", "import time; time.sleep(60)"],
                "connect_timeout_seconds": 30}
        plugin, config = _make_plugin({"up": _stdio_cfg(), "hang": hang})
        await plugin.on_init(pm=MagicMock(), config=config)
        task = asyncio.create_task(plugin.on_start(config=config))
        # Wait until the first server is up and the second is mid-handshake.
        async with asyncio.timeout(10):
            while len(_child_pids()) < 2:
                await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await _wait_no_children() == []
        await plugin.on_stop()
