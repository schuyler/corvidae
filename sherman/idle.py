"""IdleMonitorPlugin — standalone idle-state monitoring plugin.

Moves IdleMonitor and its lifecycle wiring out of AgentPlugin into a
separate plugin that can be registered or unregistered independently.

When registered, IdleMonitorPlugin.on_start creates an IdleMonitor that
polls AgentPlugin.queues and the TaskPlugin's task queue. When all queues
are empty and the cooldown has elapsed, it fires the on_idle broadcast hook.
"""

import asyncio
import logging
import time

from sherman.agent import AgentPlugin
from sherman.hooks import get_dependency, hookimpl
from sherman.queue import SerialQueue

logger = logging.getLogger(__name__)


class IdleMonitor:
    """Polls for system idle state and fires the on_idle hook.

    Runs a background asyncio task that checks every poll_interval seconds
    whether all SerialQueues are empty and the TaskQueue is idle. When both
    conditions are true and at least cooldown_seconds have elapsed since the
    last firing, broadcasts the on_idle hook to all registered plugins.

    Exceptions raised by on_idle implementations are caught and logged as
    warnings; the monitor continues running.

    Created by IdleMonitorPlugin.on_start and stopped in IdleMonitorPlugin.on_stop
    before queue teardown.
    """

    def __init__(
        self,
        pm,
        queues: dict[str, SerialQueue],
        cooldown_seconds: float = 30.0,
        poll_interval: float = 2.0,
    ) -> None:
        """
        Args:
            pm: The plugin manager instance.
            queues: Reference to AgentPlugin.queues (dict[channel_id, SerialQueue]).
                IdleMonitor checks all queues on each poll; new queues added after
                construction are included automatically because this is a reference.
            cooldown_seconds: Minimum seconds between on_idle firings.
            poll_interval: Seconds between idle state checks.
        """
        self._pm = pm
        self._queues = queues
        self._cooldown = cooldown_seconds
        self._poll_interval = poll_interval
        self._last_fired: float = 0.0
        self._task: asyncio.Task | None = None

    def _is_idle(self) -> bool:
        """Return True if the system is in an idle state.

        Checks: all SerialQueues have is_empty=True, TaskQueue.is_idle is True
        (if TaskPlugin is registered), and cooldown_seconds have elapsed since
        the last on_idle firing.
        """
        for q in self._queues.values():
            if not q.is_empty:
                return False
        task_plugin = self._pm.get_plugin("task")
        if task_plugin is not None:
            tq = getattr(task_plugin, "task_queue", None)
            if tq is not None:
                if not tq.is_idle:
                    return False
        if time.monotonic() - self._last_fired < self._cooldown:
            return False
        return True

    async def _run(self) -> None:
        """Consumer loop: polls idle state and fires on_idle when conditions are met."""
        while True:
            await asyncio.sleep(self._poll_interval)
            if self._is_idle():
                try:
                    await self._pm.ahook.on_idle()
                except Exception:
                    logger.warning("on_idle hook raised exception", exc_info=True)
                self._last_fired = time.monotonic()

    def start(self) -> None:
        """Launch the background polling task. No-op if already running."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the background polling task and wait for it to finish."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None


class IdleMonitorPlugin:
    """Plugin that monitors system idle state and fires the on_idle hook.

    Depends on agent_loop being registered. Uses trylast=True on on_start
    so it fires after AgentPlugin.on_start has fully initialized.
    """

    depends_on = {"agent_loop"}

    def __init__(self, pm) -> None:
        self.pm = pm
        self._monitor: IdleMonitor | None = None

    @hookimpl(trylast=True)
    async def on_start(self, config: dict) -> None:
        agent = get_dependency(self.pm, "agent_loop", AgentPlugin)
        daemon_config = config.get("daemon", {})
        self._monitor = IdleMonitor(
            pm=self.pm,
            queues=agent.queues,
            cooldown_seconds=daemon_config.get("idle_cooldown_seconds", 30),
            poll_interval=daemon_config.get("idle_poll_interval", 2),
        )
        self._monitor.start()

    @hookimpl
    async def on_stop(self) -> None:
        if self._monitor:
            await self._monitor.stop()
