"""A connect failure that arrives as an exception group is a failed connect (#596).

A group holding a CancelledError is neither an Exception nor a CancelledError,
so the connect handlers missed it: startup aborted for later servers and the
failed one never reached _pending_servers (PRD §3.5.8: fail-soft per server).
"""

import asyncio
import logging

import pytest
from fake_clock import clock_pm

import corvidae.mcp_client as mcp_client_mod

SERVERS = {
    "first": {"transport": "streamable_http", "url": "http://127.0.0.1:1/a"},
    "second": {"transport": "streamable_http", "url": "http://127.0.0.1:1/b"},
}


def _grouped_cancel() -> BaseExceptionGroup:
    return BaseExceptionGroup("connect failed", [asyncio.CancelledError()])


async def _plugin(servers: dict, connect) -> "mcp_client_mod.McpClientPlugin":
    plugin = mcp_client_mod.McpClientPlugin(None)
    await plugin.on_init(pm=clock_pm(), config={"mcp": {"servers": servers}})
    plugin._connect_server = connect
    return plugin


async def test_on_start_skips_a_grouped_cancel_and_tries_the_next_server(caplog):
    attempted: list[str] = []

    async def connect(name, cfg):
        attempted.append(name)
        raise _grouped_cancel() if name == "first" else ConnectionError("refused")

    plugin = await _plugin(SERVERS, connect)
    with caplog.at_level(logging.WARNING, logger="corvidae.mcp_client"):
        await plugin.on_start({"mcp": {"servers": SERVERS}})
    await plugin.on_stop()  # ends the retry task on_start created

    assert attempted == ["first", "second"]
    assert set(plugin._pending_servers) == {"first", "second"}
    warning = next(r.getMessage() for r in caplog.records if "'first'" in r.getMessage())
    assert "CancelledError" in warning and "background retry will try again" in warning


async def test_retry_round_keeps_a_server_pending_on_a_grouped_cancel():
    async def connect(name, cfg):
        raise _grouped_cancel()

    plugin = await _plugin({}, connect)
    plugin._pending_servers = {"first": SERVERS["first"]}

    assert await plugin._retry_round() is False
    assert list(plugin._pending_servers) == ["first"]


async def test_a_real_cancel_wrapped_in_a_group_propagates_and_stops_owners():
    """With someone cancelling the task, a group around that cancel is a shutdown."""
    entered = asyncio.Event()
    stopped: list[bool] = []

    async def connect(name, cfg):
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError as cancel:
            raise BaseExceptionGroup("wrapped", [cancel]) from None  # what a task group does

    async def stop_owners():
        stopped.append(True)

    plugin = await _plugin({"first": SERVERS["first"]}, connect)
    plugin._stop_owners = stop_owners
    task = asyncio.create_task(plugin.on_start({"mcp": {"servers": SERVERS}}))
    await entered.wait()
    task.cancel()

    with pytest.raises(BaseExceptionGroup):
        await task
    assert stopped == [True]


async def test_a_group_holding_a_keyboard_interrupt_is_not_swallowed():
    async def connect(name, cfg):
        raise BaseExceptionGroup("interrupted", [KeyboardInterrupt()])

    plugin = await _plugin({"first": SERVERS["first"]}, connect)
    with pytest.raises(BaseExceptionGroup):
        await plugin.on_start({"mcp": {"servers": SERVERS}})
