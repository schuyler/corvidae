"""A connect failure that arrives as an exception group is a failed connect (#596).

A group holding a ``CancelledError`` is a ``BaseExceptionGroup`` that is
neither an ``Exception`` nor a ``CancelledError``, so the connect handlers'
``except (Exception, asyncio.CancelledError)`` missed it. It escaped
``on_start`` and aborted startup for every server after it, and the failed
server never reached ``_pending_servers`` for the retry loop (PRD §3.5.8:
fail-soft per server; a server that fails recovers without a restart).
"""

import asyncio
import logging

import pytest
from fake_clock import clock_pm

import corvidae.mcp_client as mcp_client_mod

# The kind of failure a TaskGroup or anyio scope raises when a connect's child
# task was cancelled: a group whose only member is a CancelledError.
def _grouped_cancel() -> BaseExceptionGroup:
    return BaseExceptionGroup("connect failed", [asyncio.CancelledError()])


SERVERS = {
    "first": {"transport": "streamable_http", "url": "http://127.0.0.1:1/a"},
    "second": {"transport": "streamable_http", "url": "http://127.0.0.1:1/b"},
}


async def _plugin(servers: dict, connect) -> "mcp_client_mod.McpClientPlugin":
    """An initialised plugin whose ``_connect_server`` is ``connect``."""
    plugin = mcp_client_mod.McpClientPlugin(None)
    await plugin.on_init(pm=clock_pm(), config={"mcp": {"servers": servers}})
    plugin._connect_server = connect
    return plugin


async def test_on_start_skips_a_server_whose_connect_raises_a_grouped_cancel(caplog):
    """#596: the first server fails as a group; the second is still tried, and
    both are left for the retry loop."""
    attempted: list[str] = []

    async def connect(name, cfg):
        attempted.append(name)
        if name == "first":
            raise _grouped_cancel()
        raise ConnectionError("refused")

    plugin = await _plugin(SERVERS, connect)
    with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
        await plugin.on_start({"mcp": {"servers": SERVERS}})

    assert attempted == ["first", "second"], "startup must go on past the failed server"
    assert set(plugin._pending_servers) == {"first", "second"}
    skip = next(
        r.getMessage() for r in caplog.records if "'first'" in r.getMessage()
    )
    assert "CancelledError" in skip, "the warning names the types in the group"
    assert "background retry will try again" in skip, "and says what happens next"
    # The retry task is a daemon-lifetime task; stop it so the test ends clean.
    await plugin.on_stop()


async def test_retry_round_keeps_a_server_pending_on_a_grouped_cancel(caplog):
    """#596: the same group inside the retry loop leaves the server pending and
    the round returns, instead of escaping the loop."""

    async def connect(name, cfg):
        raise _grouped_cancel()

    plugin = await _plugin({}, connect)
    plugin._pending_servers = {"first": SERVERS["first"]}

    with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
        reconnected = await plugin._retry_round()

    assert reconnected is False
    assert list(plugin._pending_servers) == ["first"]
    warning = next(r.getMessage() for r in caplog.records if "'first'" in r.getMessage())
    assert "CancelledError" in warning


async def test_a_real_cancel_wrapped_in_a_group_still_propagates_and_stops_owners():
    """#596: when someone is cancelling the task, a group wrapping that cancel
    is a shutdown, not a failed connect: it propagates and ``_stop_owners``
    runs, so no started server outlives it."""
    entered = asyncio.Event()

    async def connect(name, cfg):
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError as cancel:
            # What a task group does to the cancel that reached its child.
            raise BaseExceptionGroup("wrapped", [cancel]) from None

    plugin = await _plugin({"first": SERVERS["first"]}, connect)
    stopped: list[bool] = []

    async def stop_owners():
        stopped.append(True)

    plugin._stop_owners = stop_owners

    task = asyncio.create_task(plugin.on_start({"mcp": {"servers": SERVERS}}))
    await entered.wait()
    task.cancel()
    with pytest.raises(BaseExceptionGroup):
        await task

    assert stopped == [True], "a wrapped shutdown cancel must still close started owners"


async def test_a_real_bare_cancel_still_propagates_and_stops_owners():
    """#596: the existing behaviour is unchanged for a bare CancelledError."""
    entered = asyncio.Event()

    async def connect(name, cfg):
        entered.set()
        await asyncio.sleep(30)

    plugin = await _plugin({"first": SERVERS["first"]}, connect)
    stopped: list[bool] = []

    async def stop_owners():
        stopped.append(True)

    plugin._stop_owners = stop_owners

    task = asyncio.create_task(plugin.on_start({"mcp": {"servers": SERVERS}}))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stopped == [True]


async def test_a_group_holding_a_keyboard_interrupt_is_not_swallowed():
    """#596: only a group of ordinary failures and cancels is a failed connect.
    A group carrying ``KeyboardInterrupt`` must keep propagating, as a bare
    ``KeyboardInterrupt`` already does."""

    async def connect(name, cfg):
        raise BaseExceptionGroup("interrupted", [KeyboardInterrupt()])

    plugin = await _plugin({"first": SERVERS["first"]}, connect)

    with pytest.raises(BaseExceptionGroup):
        await plugin.on_start({"mcp": {"servers": SERVERS}})
