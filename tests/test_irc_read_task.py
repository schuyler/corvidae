"""A dead IRC read task is treated as a disconnect (#275).

pydle's ``handle_forever`` is spawned with a bare ``asyncio.create_task`` and
never tracked. If ``connection.recv`` raises anything other than
``TimeoutError`` — a ``ConnectionResetError`` from ``readline`` on a reset
socket, say — the task dies, ``on_disconnect`` never runs, and
``client.connected`` keeps reporting True because pydle only checks that the
reader and writer still exist. ``connect_with_retry`` polls ``connected``, so
it never reconnects and outbound output goes to a dead socket instead of the
#106 queue.

``IRCClient`` keeps a reference to its own read task and the reconnect poll
treats a finished task as a drop.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pydle
import pytest

from corvidae.channels.irc import IRCClient

pytestmark = pytest.mark.timeout(15)


class _Plugin:
    """Minimal stand-in for IRCPlugin, which IRCClient reaches through."""

    def __init__(self) -> None:
        self._connect_task = object()  # non-None: not shutting down


def _client() -> IRCClient:
    client = IRCClient(nickname="TestBot", plugin=_Plugin())
    client._server, client._port = "irc.example", 6667
    return client


async def test_a_dead_read_task_is_not_alive(caplog):
    """When pydle's read loop raises, the poll stops trusting the connection."""
    client = _client()
    assert client._read_loop_alive() is True  # never started: assume live

    async def boom(self):
        raise ConnectionResetError("peer reset")

    with patch.object(pydle.Client, "handle_forever", boom):
        with caplog.at_level(logging.WARNING, logger="corvidae.irc_plugin"):
            task = asyncio.create_task(client.handle_forever())
            with pytest.raises(ConnectionResetError):
                await task

    assert client._read_loop_alive() is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("read task" in w.lower() for w in warnings), warnings
    assert any("irc.example" in w for w in warnings), warnings


async def test_a_running_read_task_is_alive():
    """A task that is merely running keeps the connection trusted."""
    client = _client()
    task = asyncio.create_task(asyncio.sleep(60))
    try:
        client._read_task = task
        assert client._read_loop_alive() is True
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_a_cancelled_read_task_is_not_a_disconnect(caplog):
    """Shutdown cancels the task; that is deliberate, not a drop."""
    client = _client()
    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    client._read_task = task

    with caplog.at_level(logging.WARNING, logger="corvidae.irc_plugin"):
        assert client._read_loop_alive() is True

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("read task" in w.lower() for w in warnings), warnings
