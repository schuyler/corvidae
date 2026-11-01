"""Kernel clock: wall and monotonic time, interruptible sleep, one-shot timers.

Runtime builds one Clock and registers it as "clock". The scheduler, the idle
tracker and memory's retention timer all run on it, so time passing is a
kernel input and tests can swap in a fake (tests/fake_clock.py).

Callbacks must let ``asyncio.CancelledError`` through (catch ``Exception``,
never ``BaseException``): ``Clock.close()`` relies on that to stop a running
callback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

#: Longest a wall-clock wait sleeps in one go, so a wall-clock jump
#: (suspend/resume, NTP step) is noticed within the hour.
MAX_WALL_SLEEP_SECONDS = 3600.0

ClockCallback = Callable[[], Awaitable[None]]


class Timer:
    """Handle for one scheduled callback. One-shot: it never re-arms itself."""

    def __init__(self, name: str) -> None:
        # Name the timer was created with, without the "clock:" task prefix.
        self.name = name
        # The asyncio task that waits and then runs the callback.
        self._task: asyncio.Task | None = None
        # Set just before the callback is awaited; cancel() is a no-op after.
        self._started = False

    def cancel(self) -> None:
        """Cancel the timer while it is still waiting; no-op once it has started."""
        if self._started or self._task is None:
            return
        self._task.cancel()


class Clock:
    """Real clock backed by ``time`` and asyncio."""

    def __init__(self) -> None:
        # Live timer tasks mapped to their handles; each removes itself on finish.
        self._timers: dict[asyncio.Task, Timer] = {}

    def now(self) -> float:
        """Return wall-clock time in seconds since the epoch."""
        return time.time()

    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        return time.monotonic()

    async def sleep(self, delay: float, wake: asyncio.Event | None = None) -> None:
        """Sleep ``delay`` monotonic seconds, or until ``wake`` is set.

        A negative delay counts as 0.
        """
        delay = max(0.0, delay)
        if wake is None:
            # Nothing can interrupt this sleep.
            await asyncio.sleep(delay)
            return
        try:
            # Wait for the wake event, giving up after the delay.
            await asyncio.wait_for(wake.wait(), delay)
        except TimeoutError:
            # The delay elapsed without a wake: the normal way out.
            pass

    def call_later(self, delay: float, callback: ClockCallback, *, name: str) -> Timer:
        """Run ``callback`` once, ``delay`` monotonic seconds from now."""

        async def wait() -> None:
            # Sleep out the delay on the clock's own sleep.
            await self.sleep(delay)

        return self._schedule(wait, callback, name)

    def call_soon(self, callback: ClockCallback, *, name: str) -> Timer:
        """Run ``callback`` once, on the next loop step, without waiting on time."""

        async def wait() -> None:
            # Nothing to wait for; returns without suspending.
            return

        return self._schedule(wait, callback, name)

    def call_at(self, when: float, callback: ClockCallback, *, name: str) -> Timer:
        """Run ``callback`` once, when wall time reaches ``when``."""

        async def wait() -> None:
            # Sleep in capped steps so a wall-clock jump is noticed.
            while self.now() < when:
                await self.sleep(min(when - self.now(), MAX_WALL_SLEEP_SECONDS))

        return self._schedule(wait, callback, name)

    def _schedule(
        self,
        wait: Callable[[], Awaitable[None]],
        callback: ClockCallback,
        name: str,
    ) -> Timer:
        """Create the task that waits, then runs the callback, and track it."""
        timer = Timer(name)

        async def run() -> None:
            # Wait out the delay; cancel() can still land here.
            await wait()
            # From here on cancel() is a no-op.
            timer._started = True
            try:
                await callback()
            except Exception:
                # Log with the timer name and traceback so the error is not lost.
                logger.warning("clock timer %s failed", name, exc_info=True)

        task = asyncio.create_task(run(), name=f"clock:{name}")
        timer._task = task
        self._timers[task] = timer
        # Forget the task once it finishes, however it finishes.
        task.add_done_callback(lambda t: self._timers.pop(t, None))
        return timer

    def live_timer_names(self) -> list[str]:
        """Names of timers not yet finished, for tests."""
        return [timer.name for timer in self._timers.values()]

    async def close(self) -> None:
        """Cancel every live timer task, running callbacks included, and await them."""
        tasks = list(self._timers)
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                # Expected: we just cancelled it.
                pass
