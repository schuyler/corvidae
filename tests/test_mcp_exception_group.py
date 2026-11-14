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


# ---------------------------------------------------------------------------
# #604: a shutdown mixed with connect errors, and a bare KeyboardInterrupt,
# must both still close the owners.
# ---------------------------------------------------------------------------


async def test_a_cancel_mixed_with_connect_errors_closes_the_owners():
    """A cancel arriving while a connect's task group holds child errors raises
    a group of Exceptions and re-arms the cancel. That is a shutdown: the
    owners are closed before it propagates, and no later code runs."""
    entered = asyncio.Event()
    stopped: list[bool] = []
    reached_end: list[bool] = []

    async def connect(name, cfg):
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError as cancel:
            # What asyncio.TaskGroup does in 3.13: the child error and the
            # cancel together, as an ExceptionGroup, with the cancel re-armed.
            raise ExceptionGroup(
                "unhandled errors in a TaskGroup", [ConnectionError("refused")]
            ) from cancel

    async def stop_owners():
        stopped.append(True)

    plugin = await _plugin({"first": SERVERS["first"]}, connect)
    plugin._stop_owners = stop_owners

    async def rebuild(*args, **kwargs):
        reached_end.append(True)

    plugin._rebuild_shared_registry = rebuild
    task = asyncio.create_task(plugin.on_start({"mcp": {"servers": SERVERS}}))
    await entered.wait()
    task.cancel()

    with pytest.raises(BaseException):
        await task
    assert stopped == [True], "a mixed shutdown must still close the owners"
    assert reached_end == [], "nothing after the try may run in a shutdown"


async def test_a_bare_keyboard_interrupt_closes_the_owners():
    """The same shutdown without a group around it: the owners still close."""
    stopped: list[bool] = []

    async def connect(name, cfg):
        raise KeyboardInterrupt()

    async def stop_owners():
        stopped.append(True)

    plugin = await _plugin({"first": SERVERS["first"]}, connect)
    plugin._stop_owners = stop_owners

    with pytest.raises(KeyboardInterrupt):
        await plugin.on_start({"mcp": {"servers": SERVERS}})
    assert stopped == [True]


async def test_a_bare_system_exit_closes_the_owners():
    stopped: list[bool] = []

    async def connect(name, cfg):
        raise SystemExit(1)

    async def stop_owners():
        stopped.append(True)

    plugin = await _plugin({"first": SERVERS["first"]}, connect)
    plugin._stop_owners = stop_owners

    with pytest.raises(SystemExit):
        await plugin.on_start({"mcp": {"servers": SERVERS}})
    assert stopped == [True]


async def test_a_connect_error_while_cancelling_is_a_shutdown_not_a_skip():
    """The rule itself: an ordinary Exception stops being a failed connect once
    this task is being cancelled (#604)."""
    assert mcp_client_mod._is_connect_failure(ConnectionError("refused")) is True
    assert mcp_client_mod._is_connect_failure(
        ExceptionGroup("g", [ConnectionError("refused")])
    ) is True

    judged: list[bool] = []

    async def victim():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            # cancelling() is nonzero here: this task is being shut down.
            judged.append(mcp_client_mod._is_connect_failure(ConnectionError("refused")))
            judged.append(
                mcp_client_mod._is_connect_failure(
                    ExceptionGroup("g", [ConnectionError("refused")])
                )
            )
            raise

    task = asyncio.create_task(victim())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert judged == [False, False], "a connect error during a cancel must propagate"
