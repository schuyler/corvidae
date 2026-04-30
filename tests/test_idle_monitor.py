"""Tests for idle monitoring.

Part 2 of Agent decomposition: push-based idle detection.
IdleMonitor polling class has been removed. Agent now detects
idle state and broadcasts on_idle after each queue item completes.
"""

import asyncio
import time
import logging
from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

import pytest

from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.queue import SerialQueue


# ---------------------------------------------------------------------------
# TestIdleMonitorPlugin — surviving tests (no _monitor assertions)
# ---------------------------------------------------------------------------


class TestIdleMonitorPlugin:
    """Tests for IdleMonitorPlugin that do not depend on _monitor."""

    async def test_idle_monitor_plugin_not_registered_no_crash(self):
        """When no IdleMonitorPlugin is registered, calling on_idle does not crash.

        on_idle is a broadcast hook; zero implementations means a no-op.
        """
        from corvidae.idle import IdleMonitorPlugin  # noqa: F401

        pm = create_plugin_manager()
        from corvidae.agent import Agent
        agent = Agent(pm)
        pm.register(agent, name="agent")
        # IdleMonitorPlugin is intentionally NOT registered.

        # Should not raise even with no on_idle implementations.
        await pm.ahook.on_idle()


# ---------------------------------------------------------------------------
# TestPushBasedIdle — tests for the new push-based idle detection in Part 2
#
# These tests exercise:
#   - Agent._maybe_fire_idle() (new private method)
#   - Agent._idle_cooldown / _last_idle_fire (new state)
#   - IdleMonitorPlugin.depends_on no longer containing agent_loop/task
#
# All tests are expected to FAIL (red phase) until Part 2 is implemented.
# ---------------------------------------------------------------------------


class TestPushBasedIdle:
    """Tests for push-based idle detection (Part 2 of Agent decomposition)."""

    # -----------------------------------------------------------------------
    # _maybe_fire_idle — basic attribute presence
    # -----------------------------------------------------------------------

    def test_agent_plugin_has_maybe_fire_idle(self):
        """Agent must have a _maybe_fire_idle method after Part 2."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        agent = Agent(pm)
        assert hasattr(agent, "_maybe_fire_idle"), (
            "Agent must have _maybe_fire_idle after Part 2 implementation"
        )
        assert callable(agent._maybe_fire_idle), (
            "_maybe_fire_idle must be callable"
        )

    def test_agent_plugin_has_idle_cooldown_attribute(self):
        """Agent must initialise _idle_cooldown in __init__."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        agent = Agent(pm)
        assert hasattr(agent, "_idle_cooldown"), (
            "Agent must have _idle_cooldown attribute"
        )
        assert agent._idle_cooldown == 30.0, (
            "_idle_cooldown default must be 30.0 seconds"
        )

    def test_agent_plugin_has_last_idle_fire_attribute(self):
        """Agent must initialise _last_idle_fire to 0.0 in __init__."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        agent = Agent(pm)
        assert hasattr(agent, "_last_idle_fire"), (
            "Agent must have _last_idle_fire attribute"
        )
        assert agent._last_idle_fire == 0.0, (
            "_last_idle_fire must initialise to 0.0"
        )

    # -----------------------------------------------------------------------
    # _maybe_fire_idle — firing conditions
    # -----------------------------------------------------------------------

    async def test_maybe_fire_idle_fires_when_queues_empty_and_cooldown_elapsed(self):
        """_maybe_fire_idle fires on_idle when all queues empty and cooldown elapsed."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        pm.ahook.on_idle = AsyncMock()

        agent = Agent(pm)
        # No queues at all — all (zero) queues are empty
        agent._idle_cooldown = 0.0
        agent._last_idle_fire = 0.0

        await agent._maybe_fire_idle()

        pm.ahook.on_idle.assert_awaited_once()

    async def test_maybe_fire_idle_does_not_fire_when_queue_has_items(self):
        """_maybe_fire_idle does NOT fire on_idle when a queue has pending items."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        pm.ahook.on_idle = AsyncMock()

        agent = Agent(pm)
        agent._idle_cooldown = 0.0
        agent._last_idle_fire = 0.0

        # Add a non-empty queue
        q = SerialQueue()
        q._queue.put_nowait("pending_item")
        agent.queues["ch1"] = q

        await agent._maybe_fire_idle()

        pm.ahook.on_idle.assert_not_awaited()

    async def test_maybe_fire_idle_does_not_fire_during_cooldown(self):
        """_maybe_fire_idle does NOT fire on_idle when cooldown has not elapsed."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        pm.ahook.on_idle = AsyncMock()

        agent = Agent(pm)
        agent._idle_cooldown = 3600.0       # very long cooldown
        agent._last_idle_fire = time.monotonic()  # just fired

        await agent._maybe_fire_idle()

        pm.ahook.on_idle.assert_not_awaited()

    async def test_maybe_fire_idle_does_not_fire_when_task_queue_not_idle(self):
        """_maybe_fire_idle does NOT fire on_idle when the task queue is not idle."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        pm.ahook.on_idle = AsyncMock()

        agent = Agent(pm)
        agent._idle_cooldown = 0.0
        agent._last_idle_fire = 0.0

        # Register a mock task plugin whose task queue is not idle
        mock_tq = MagicMock()
        mock_tq.is_idle = False
        mock_task_plugin = NonCallableMagicMock()
        mock_task_plugin.task_queue = mock_tq
        pm.register(mock_task_plugin, name="task")

        await agent._maybe_fire_idle()

        pm.ahook.on_idle.assert_not_awaited()

    async def test_maybe_fire_idle_updates_last_idle_fire_after_firing(self):
        """_last_idle_fire is updated to current time after on_idle fires."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        pm.ahook.on_idle = AsyncMock()

        agent = Agent(pm)
        agent._idle_cooldown = 0.0
        agent._last_idle_fire = 0.0

        before = time.monotonic()
        await agent._maybe_fire_idle()

        assert agent._last_idle_fire >= before, (
            "_last_idle_fire must be updated to current time after firing"
        )

    async def test_maybe_fire_idle_does_not_update_last_fired_when_not_firing(self):
        """_last_idle_fire is NOT updated when _maybe_fire_idle does not fire."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        pm.ahook.on_idle = AsyncMock()

        agent = Agent(pm)
        agent._idle_cooldown = 3600.0
        saved = time.monotonic()
        agent._last_idle_fire = saved  # just fired — cooldown active

        await agent._maybe_fire_idle()

        assert agent._last_idle_fire == saved, (
            "_last_idle_fire must not change when idle was not fired"
        )

    async def test_maybe_fire_idle_fires_once_then_respects_cooldown(self):
        """After firing, a second call within cooldown does not fire again."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        pm.ahook.on_idle = AsyncMock()

        agent = Agent(pm)
        agent._idle_cooldown = 3600.0  # long cooldown
        # Set last-fire two cooldowns in the past so the first call definitely
        # passes the cooldown check, regardless of the absolute monotonic clock
        # value (which is environment-dependent — small in fresh sandboxes).
        agent._last_idle_fire = time.monotonic() - 2 * agent._idle_cooldown

        # First call should fire (cooldown has elapsed)
        await agent._maybe_fire_idle()
        assert pm.ahook.on_idle.await_count == 1

        # Second immediate call — cooldown now active
        await agent._maybe_fire_idle()
        assert pm.ahook.on_idle.await_count == 1, (
            "on_idle must not fire a second time within the cooldown period"
        )

    # -----------------------------------------------------------------------
    # _maybe_fire_idle — race condition / concurrent calls
    # -----------------------------------------------------------------------

    async def test_maybe_fire_idle_concurrent_calls_fire_once(self):
        """Two concurrent _maybe_fire_idle calls only fire on_idle once."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        agent = Agent(pm)

        fired = asyncio.Event()
        resume = asyncio.Event()

        async def slow_on_idle():
            fired.set()
            await resume.wait()

        pm.ahook.on_idle = AsyncMock(side_effect=slow_on_idle)

        agent._idle_cooldown = 0.0
        agent._last_idle_fire = 0.0

        t1 = asyncio.create_task(agent._maybe_fire_idle())
        await fired.wait()  # ensure first call is inside on_idle
        t2 = asyncio.create_task(agent._maybe_fire_idle())
        await asyncio.sleep(0)  # yield to let t2 run and bail
        resume.set()  # release t1
        await asyncio.gather(t1, t2)

        assert pm.ahook.on_idle.await_count == 1

    async def test_maybe_fire_idle_clears_flag_on_exception_and_retries(self):
        """_idle_firing flag is cleared and timestamp not updated on exception; retry fires."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        pm.ahook.on_idle = AsyncMock(side_effect=RuntimeError("boom"))
        agent = Agent(pm)

        agent._idle_cooldown = 0.0
        agent._last_idle_fire = 0.0
        original_ts = agent._last_idle_fire

        await agent._maybe_fire_idle()

        assert not agent._idle_firing, "_idle_firing must be cleared after exception"
        assert agent._last_idle_fire == original_ts, "_last_idle_fire must not advance on failure"
        assert pm.ahook.on_idle.await_count == 1

        # Retry should fire again since timestamp was not updated
        pm.ahook.on_idle.side_effect = None
        await agent._maybe_fire_idle()
        assert pm.ahook.on_idle.await_count == 2, "on_idle should fire on retry after exception"

    # -----------------------------------------------------------------------
    # IdleMonitorPlugin.depends_on — should be empty after Part 2
    # -----------------------------------------------------------------------

    def test_idle_monitor_plugin_depends_on_is_empty_after_part2(self):
        """After Part 2, IdleMonitorPlugin.depends_on must not include agent_loop or task.

        IdleMonitorPlugin becomes a pure on_idle consumer; it no longer needs
        to access Agent.queues directly.
        """
        from corvidae.idle import IdleMonitorPlugin

        pm = create_plugin_manager()
        plugin = IdleMonitorPlugin(pm)

        depends_on = getattr(plugin, "depends_on", set())
        assert "agent" not in depends_on, (
            "IdleMonitorPlugin must not depend on agent after Part 2"
        )
        assert "task" not in depends_on, (
            "IdleMonitorPlugin must not depend on task after Part 2"
        )

    # -----------------------------------------------------------------------
    # IdleMonitorPlugin — no longer has an IdleMonitor instance after Part 2
    # -----------------------------------------------------------------------

    def test_idle_monitor_plugin_has_no_monitor_attribute(self):
        """After Part 2, IdleMonitorPlugin has no _monitor attribute.

        The IdleMonitor polling class is removed; IdleMonitorPlugin is a
        pure on_idle consumer and has no internal polling state.
        """
        from corvidae.idle import IdleMonitorPlugin

        pm = create_plugin_manager()
        plugin = IdleMonitorPlugin(pm)

        assert not hasattr(plugin, "_monitor"), (
            "IdleMonitorPlugin must not have _monitor after Part 2 "
            "(IdleMonitor class is removed)"
        )
