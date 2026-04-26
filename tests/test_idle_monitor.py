"""Tests for IdleMonitor class (red phase).

IdleMonitor is a new class in sherman.agent that polls for system idle state
and fires the on_idle hook when all queues are empty and the cooldown has elapsed.
"""

import asyncio
import time
import logging
from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock, patch

import pytest

from sherman.hooks import create_plugin_manager, hookimpl
from sherman.queue import SerialQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pm_with_on_idle_mock():
    """Create a plugin manager with ahook.on_idle mocked."""
    pm = create_plugin_manager()
    pm.ahook.on_idle = AsyncMock()
    return pm


def _empty_queues():
    """Return a dict of empty SerialQueues."""
    return {}


def _non_empty_queues():
    """Return a dict with a SerialQueue that has a pending item."""
    q = SerialQueue()
    q._queue.put_nowait("pending")
    return {"ch1": q}


# ---------------------------------------------------------------------------
# Import guard — IdleMonitor does not exist yet
# ---------------------------------------------------------------------------


def _import_idle_monitor():
    """Import IdleMonitor from sherman.idle; skip if not yet implemented."""
    try:
        from sherman.idle import IdleMonitor
        return IdleMonitor
    except ImportError:
        pytest.fail("IdleMonitor not found in sherman.idle — implement it")


# ---------------------------------------------------------------------------
# Construction and config propagation
# ---------------------------------------------------------------------------


def test_idle_monitor_config_propagated():
    """IdleMonitor stores cooldown_seconds and poll_interval from constructor."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    monitor = IdleMonitor(
        pm=pm,
        queues={},
        cooldown_seconds=60.0,
        poll_interval=5.0,
    )

    assert monitor._cooldown == 60.0
    assert monitor._poll_interval == 5.0


def test_idle_monitor_default_config():
    """IdleMonitor defaults: cooldown=30, poll_interval=2."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    monitor = IdleMonitor(pm=pm, queues={})

    assert monitor._cooldown == 30.0
    assert monitor._poll_interval == 2.0


# ---------------------------------------------------------------------------
# _is_idle logic
# ---------------------------------------------------------------------------


def test_is_idle_true_when_queues_empty_and_cooldown_elapsed():
    """_is_idle returns True when all queues are empty and cooldown has passed."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()
    # No task plugin registered — that path is safely skipped.

    monitor = IdleMonitor(pm=pm, queues={}, cooldown_seconds=0.0, poll_interval=1.0)
    monitor._last_fired = 0.0  # long ago

    assert monitor._is_idle() is True


def test_is_idle_false_when_queue_non_empty():
    """_is_idle returns False when any SerialQueue has pending items."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    queues = _non_empty_queues()
    monitor = IdleMonitor(pm=pm, queues=queues, cooldown_seconds=0.0, poll_interval=1.0)
    monitor._last_fired = 0.0

    assert monitor._is_idle() is False


def test_is_idle_false_during_cooldown():
    """_is_idle returns False when the cooldown period has not elapsed."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    monitor = IdleMonitor(pm=pm, queues={}, cooldown_seconds=3600.0, poll_interval=1.0)
    monitor._last_fired = time.monotonic()  # just fired

    assert monitor._is_idle() is False


def test_is_idle_false_when_task_queue_non_empty():
    """_is_idle returns False when the task plugin's TaskQueue has items."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    # Simulate a task plugin with a non-idle task queue
    mock_tq = MagicMock()
    mock_tq.is_idle = False  # queue has pending items

    mock_task_plugin = NonCallableMagicMock()
    mock_task_plugin.task_queue = mock_tq
    pm.register(mock_task_plugin, name="task")

    monitor = IdleMonitor(pm=pm, queues={}, cooldown_seconds=0.0, poll_interval=1.0)
    monitor._last_fired = 0.0

    assert monitor._is_idle() is False


def test_is_idle_false_when_task_plugin_has_active_tasks():
    """_is_idle returns False when the task plugin has active (running) tasks."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    mock_tq = MagicMock()
    mock_tq.is_idle = False  # active tasks running

    mock_task_plugin = NonCallableMagicMock()
    mock_task_plugin.task_queue = mock_tq
    pm.register(mock_task_plugin, name="task")

    monitor = IdleMonitor(pm=pm, queues={}, cooldown_seconds=0.0, poll_interval=1.0)
    monitor._last_fired = 0.0

    assert monitor._is_idle() is False


def test_is_idle_true_when_no_task_plugin():
    """_is_idle is True when there is no task plugin registered (no task queue to check)."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()
    # No task plugin registered.

    monitor = IdleMonitor(pm=pm, queues={}, cooldown_seconds=0.0, poll_interval=1.0)
    monitor._last_fired = 0.0

    assert monitor._is_idle() is True


# ---------------------------------------------------------------------------
# Firing behavior
# ---------------------------------------------------------------------------


async def test_idle_monitor_fires_on_idle_when_idle():
    """IdleMonitor fires on_idle when all queues empty and cooldown elapsed."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    fired = asyncio.Event()

    async def on_idle_side_effect():
        fired.set()

    pm.ahook.on_idle.side_effect = on_idle_side_effect

    monitor = IdleMonitor(
        pm=pm,
        queues={},
        cooldown_seconds=0.0,
        poll_interval=0.01,  # very fast polling for tests
    )
    monitor._last_fired = 0.0  # cooldown already elapsed

    monitor.start()
    try:
        await asyncio.wait_for(fired.wait(), timeout=1.0)
    finally:
        await monitor.stop()

    pm.ahook.on_idle.assert_awaited()


async def test_idle_monitor_does_not_fire_during_cooldown():
    """IdleMonitor does not fire on_idle during the cooldown period."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    monitor = IdleMonitor(
        pm=pm,
        queues={},
        cooldown_seconds=3600.0,  # very long cooldown
        poll_interval=0.01,
    )
    monitor._last_fired = time.monotonic()  # just fired

    monitor.start()
    # Wait long enough for a few poll cycles.
    await asyncio.sleep(0.05)
    await monitor.stop()

    pm.ahook.on_idle.assert_not_awaited()


async def test_idle_monitor_does_not_fire_when_queues_non_empty():
    """IdleMonitor does not fire on_idle when queues have pending items."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    queues = _non_empty_queues()
    monitor = IdleMonitor(
        pm=pm,
        queues=queues,
        cooldown_seconds=0.0,
        poll_interval=0.01,
    )
    monitor._last_fired = 0.0

    monitor.start()
    await asyncio.sleep(0.05)
    await monitor.stop()

    pm.ahook.on_idle.assert_not_awaited()


async def test_idle_monitor_stop_cancels_polling_task():
    """stop() cancels the background polling task."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    monitor = IdleMonitor(pm=pm, queues={}, cooldown_seconds=3600.0, poll_interval=0.1)
    monitor._last_fired = time.monotonic()

    monitor.start()
    task = monitor._task
    assert task is not None
    assert not task.done()

    await monitor.stop()

    assert task.done()
    assert monitor._task is None


async def test_idle_monitor_exception_in_hook_caught_and_monitor_continues(caplog):
    """An exception in the on_idle hook is caught and logged; the monitor
    continues polling and can fire again after the cooldown."""
    IdleMonitor = _import_idle_monitor()
    pm = create_plugin_manager()

    fire_count = [0]

    class OnIdlePlugin:
        @hookimpl
        async def on_idle(self):
            fire_count[0] += 1
            if fire_count[0] == 1:
                raise RuntimeError("on_idle hook exploded")

    pm.register(OnIdlePlugin(), name="on_idle_plugin")

    monitor = IdleMonitor(
        pm=pm,
        queues={},
        cooldown_seconds=0.0,
        poll_interval=0.01,
    )
    monitor._last_fired = 0.0

    with caplog.at_level(logging.WARNING, logger="sherman.idle"):
        monitor.start()
        # Wait for the hook to fire at least twice (first raises, second succeeds)
        deadline = time.monotonic() + 2.0
        while fire_count[0] < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        await monitor.stop()

    # First fire raised — warning should be logged
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "on_idle" in r.getMessage().lower()
    ]
    assert warning_records, "Expected WARNING log about on_idle exception"

    # Monitor must have continued and fired a second time
    assert fire_count[0] >= 2, (
        f"Expected monitor to fire at least twice, got {fire_count[0]}"
    )


async def test_idle_monitor_updates_last_fired_after_hook():
    """After firing on_idle, _last_fired is updated to prevent immediate re-firing."""
    IdleMonitor = _import_idle_monitor()
    pm = _make_pm_with_on_idle_mock()

    fired_event = asyncio.Event()

    async def on_idle_side_effect():
        fired_event.set()

    pm.ahook.on_idle.side_effect = on_idle_side_effect

    monitor = IdleMonitor(
        pm=pm,
        queues={},
        cooldown_seconds=0.0,
        poll_interval=0.01,
    )
    monitor._last_fired = 0.0

    before = time.monotonic()
    monitor.start()
    await asyncio.wait_for(fired_event.wait(), timeout=1.0)
    await monitor.stop()

    assert monitor._last_fired >= before, (
        "_last_fired must be updated after on_idle fires"
    )


# ---------------------------------------------------------------------------
# IdleMonitorPlugin tests (Phase C — will fail with ImportError until
# sherman/idle.py is created)
# ---------------------------------------------------------------------------


class TestIdleMonitorPlugin:
    """Tests for IdleMonitorPlugin.

    These tests import from sherman.idle which does not exist yet.
    They are expected to fail with ImportError (red phase).
    """

    async def test_idle_monitor_plugin_on_start_creates_monitor(self):
        """on_start creates an IdleMonitor and starts it.

        Register AgentPlugin and IdleMonitorPlugin, call on_start on each using
        the new explicit sequencing: broadcast first, then agent.on_start() explicitly.
        Verify plugin._monitor is not None and has a running background task.
        """
        from sherman.idle import IdleMonitorPlugin
        from sherman.agent import AgentPlugin
        from sherman.channel import ChannelRegistry
        from sherman.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "", "max_context_tokens": 8000})
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        persistence = PersistencePlugin(pm)
        pm.register(persistence, name="persistence")

        agent = AgentPlugin(pm)
        pm.register(agent, name="agent_loop")

        idle_plugin = IdleMonitorPlugin(pm)
        pm.register(idle_plugin, name="idle_monitor")

        config = {
            "llm": {"main": {"base_url": "http://localhost:8080", "model": "test"}},
            "daemon": {"session_db": ":memory:"},
        }

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client), \
             patch("sherman.persistence.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.persistence.init_db", new_callable=AsyncMock):
            mock_connect.return_value = AsyncMock()
            # Mirrors new main.py sequencing: broadcast first, then explicit agent start.
            # AgentPlugin.on_start no longer has @hookimpl after the race-condition fix,
            # so the broadcast alone is not sufficient to initialize the agent.
            await pm.ahook.on_start(config=config)
            await agent.on_start(config=config)

        try:
            assert idle_plugin._monitor is not None, (
                "IdleMonitorPlugin._monitor must be set after on_start"
            )
            assert idle_plugin._monitor._task is not None, (
                "IdleMonitor background task must be started after on_start"
            )
            assert not idle_plugin._monitor._task.done(), (
                "IdleMonitor background task must still be running"
            )
        finally:
            if idle_plugin._monitor:
                await idle_plugin._monitor.stop()

    async def test_idle_monitor_plugin_on_stop_stops_monitor(self):
        """on_stop stops the IdleMonitor background task.

        Uses the new explicit sequencing: broadcast on_start, then agent.on_start();
        agent.on_stop() before broadcast on_stop().  Verify the monitor's background
        task is done after shutdown.
        """
        from sherman.idle import IdleMonitorPlugin
        from sherman.agent import AgentPlugin
        from sherman.channel import ChannelRegistry
        from sherman.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "", "max_context_tokens": 8000})
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        persistence = PersistencePlugin(pm)
        pm.register(persistence, name="persistence")

        agent = AgentPlugin(pm)
        pm.register(agent, name="agent_loop")

        idle_plugin = IdleMonitorPlugin(pm)
        pm.register(idle_plugin, name="idle_monitor")

        config = {
            "llm": {"main": {"base_url": "http://localhost:8080", "model": "test"}},
            "daemon": {"session_db": ":memory:"},
        }

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client), \
             patch("sherman.persistence.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.persistence.init_db", new_callable=AsyncMock):
            mock_connect.return_value = AsyncMock()
            # New sequencing: broadcast first, then explicit agent start.
            await pm.ahook.on_start(config=config)
            await agent.on_start(config=config)

        monitor = idle_plugin._monitor
        assert monitor is not None
        task = monitor._task
        assert task is not None

        # New shutdown sequencing: agent stops first, then broadcast.
        await agent.on_stop()
        await pm.ahook.on_stop()

        assert monitor._task is None, (
            "IdleMonitor._task must be None after on_stop"
        )
        assert task.done(), (
            "IdleMonitor background task must be done after on_stop"
        )

    async def test_idle_monitor_plugin_uses_agent_queues(self):
        """IdleMonitorPlugin passes the agent's queues dict reference to IdleMonitor.

        After on_start (using new explicit sequencing), verify that
        idle_plugin._monitor._queues is the same object as agent.queues
        (same dict reference, not a copy).
        """
        from sherman.idle import IdleMonitorPlugin
        from sherman.agent import AgentPlugin
        from sherman.channel import ChannelRegistry
        from sherman.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "", "max_context_tokens": 8000})
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        persistence = PersistencePlugin(pm)
        pm.register(persistence, name="persistence")

        agent = AgentPlugin(pm)
        pm.register(agent, name="agent_loop")

        idle_plugin = IdleMonitorPlugin(pm)
        pm.register(idle_plugin, name="idle_monitor")

        config = {
            "llm": {"main": {"base_url": "http://localhost:8080", "model": "test"}},
            "daemon": {"session_db": ":memory:"},
        }

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client), \
             patch("sherman.persistence.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.persistence.init_db", new_callable=AsyncMock):
            mock_connect.return_value = AsyncMock()
            # New sequencing: broadcast first, then explicit agent start.
            await pm.ahook.on_start(config=config)
            await agent.on_start(config=config)

        try:
            assert idle_plugin._monitor is not None
            assert idle_plugin._monitor._queues is agent.queues, (
                "IdleMonitor must hold a reference to agent.queues, not a copy"
            )
        finally:
            if idle_plugin._monitor:
                await idle_plugin._monitor.stop()

    async def test_idle_monitor_plugin_not_registered_no_crash(self):
        """When no IdleMonitorPlugin is registered, calling on_idle does not crash.

        on_idle is a broadcast hook; zero implementations means a no-op.
        """
        from sherman.idle import IdleMonitorPlugin  # noqa: F401

        pm = create_plugin_manager()
        from sherman.agent import AgentPlugin
        agent = AgentPlugin(pm)
        pm.register(agent, name="agent_loop")
        # IdleMonitorPlugin is intentionally NOT registered.

        # Should not raise even with no on_idle implementations.
        await pm.ahook.on_idle()

