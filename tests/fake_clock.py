"""FakeClock: a deterministic stand-in for the kernel clock (R7).

No test sleeps real time. ``FakeClock`` keeps its own wall and monotonic
time, parks every ``sleep`` until ``advance`` passes its deadline, and
records every task that sleeps on it so ``advance`` can wait for each one
to park again (or finish) before time moves on.

Kernel tests only. Nothing here imports ``corvidae_cognition``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from corvidae.clock import Clock, ClockCallback, Timer

if TYPE_CHECKING:
    from corvidae.idle import IdleTracker

#: Default fake wall-clock start. Integral, so every sum in a test is exact.
DEFAULT_START = 1_800_000_000.0

#: Fake monotonic time at construction. Integral for the same reason.
MONOTONIC_START = 1_000.0


@dataclass
class _Park:
    """One task parked in ``FakeClock.sleep``."""

    # Monotonic time at which ``advance`` releases the sleeper.
    deadline: float
    # Set to let the sleeper return.
    release: asyncio.Event
    # The sleeping task, so its settled event can be cleared on release.
    task: asyncio.Task


class FakeClock(Clock):
    """Clock whose time moves only when a test calls ``advance`` or ``jump``."""

    def __init__(self, start: float = DEFAULT_START) -> None:
        """Start fake wall time at ``start`` and fake monotonic time at 1000."""
        super().__init__()
        # Fake wall time and fake monotonic time; advance moves both.
        self._wall = start
        self._mono = MONOTONIC_START
        # Sleepers currently parked, in the order they parked.
        self._parked: list[_Park] = []
        # Per recorded task: set while the task is parked or finished,
        # cleared while it is runnable. settle() waits on the cleared ones.
        self._settled: dict[asyncio.Task, asyncio.Event] = {}
        # Every delay passed to sleep(), for tests that read the schedule.
        self.sleeps: asyncio.Queue[float] = asyncio.Queue()

    def now(self) -> float:
        """Return fake wall time."""
        return self._wall

    def monotonic(self) -> float:
        """Return fake monotonic time."""
        return self._mono

    def jump(self, seconds: float) -> None:
        """Step wall time only, as a suspend/resume or NTP step would."""
        self._wall += seconds

    # ------------------------------------------------------------------
    # Task tracking
    # ------------------------------------------------------------------

    def _track(self, task: asyncio.Task) -> asyncio.Event:
        """Record ``task`` (once) and return its parked-or-finished event."""
        event = self._settled.get(task)
        if event is None:
            # A new task starts out runnable: its event is not set.
            event = asyncio.Event()
            self._settled[task] = event
            task.add_done_callback(lambda t, ev=event: self._on_done(t, ev))
        return event

    def _on_done(self, task: asyncio.Task, event: asyncio.Event) -> None:
        """A finished task counts as settled and is forgotten."""
        event.set()
        self._settled.pop(task, None)

    def _record_timers(self, name: str) -> None:
        """Record every not-yet-recorded timer task named for ``name``."""
        found = False
        for task in asyncio.all_tasks():
            if task.get_name() == f"clock:{name}" and task not in self._settled:
                self._track(task)
                found = True
        # A timer task that is already recorded (a nested call) is fine; one
        # that cannot be found at all would make settle() miss it silently.
        if not found and not any(
            t.get_name() == f"clock:{name}" for t in self._settled
        ):
            raise AssertionError(
                f"FakeClock could not find the task for timer {name!r}; "
                "Clock.call_later/call_at/call_soon must name it 'clock:<name>'"
            )

    def call_later(
        self, delay: float, callback: ClockCallback, *, name: str
    ) -> Timer:
        """Schedule as the real clock does, and record the timer task."""
        timer = super().call_later(delay, callback, name=name)
        self._record_timers(name)
        return timer

    def call_soon(self, callback: ClockCallback, *, name: str) -> Timer:
        """Schedule as the real clock does, and record the timer task."""
        timer = super().call_soon(callback, name=name)
        self._record_timers(name)
        return timer

    def call_at(
        self, when: float, callback: ClockCallback, *, name: str
    ) -> Timer:
        """Schedule as the real clock does, and record the timer task."""
        timer = super().call_at(when, callback, name=name)
        self._record_timers(name)
        return timer

    # ------------------------------------------------------------------
    # Sleeping
    # ------------------------------------------------------------------

    async def sleep(
        self, delay: float, wake: asyncio.Event | None = None
    ) -> None:
        """Park until ``advance`` passes ``monotonic() + delay`` or ``wake`` is set."""
        delay = max(0.0, delay)
        task = asyncio.current_task()
        park = _Park(self._mono + delay, asyncio.Event(), task)
        # Record the delay and park with no await in between, so a test that
        # wakes on next_sleep() always finds the sleeper already parked.
        self.sleeps.put_nowait(delay)
        event = self._track(task)
        self._parked.append(park)
        event.set()
        helper: asyncio.Future | None = None
        try:
            if wake is not None:
                if wake.is_set():
                    self._release(park)
                else:
                    # A helper releases the park when the wake event fires.
                    helper = asyncio.ensure_future(self._release_on_wake(wake, park))
            await park.release.wait()
        finally:
            if helper is not None:
                helper.cancel()
            if park in self._parked:
                self._parked.remove(park)

    async def _release_on_wake(self, wake: asyncio.Event, park: _Park) -> None:
        """Release ``park`` once ``wake`` is set."""
        await wake.wait()
        self._release(park)

    def _release(self, park: _Park) -> None:
        """Let a parked sleeper run: mark it runnable and wake it."""
        if park in self._parked:
            self._parked.remove(park)
        self._track(park.task).clear()
        park.release.set()

    async def next_sleep(self, timeout: float = 2.0) -> float:
        """Return the next delay passed to ``sleep``, waiting for one."""
        return await asyncio.wait_for(self.sleeps.get(), timeout=timeout)

    # ------------------------------------------------------------------
    # Moving time
    # ------------------------------------------------------------------

    async def settle(self) -> None:
        """Wait until every recorded task has parked on the clock or finished."""
        me = asyncio.current_task()
        # Let freshly woken tasks and wake helpers take their first step.
        for _ in range(5):
            await asyncio.sleep(0)
        while True:
            waiting = [
                ev
                for task, ev in list(self._settled.items())
                if task is not me and not ev.is_set()
            ]
            if not waiting:
                return
            await asyncio.gather(*(ev.wait() for ev in waiting))
            await asyncio.sleep(0)

    async def advance(self, seconds: float) -> None:
        """Move both clocks forward, running every sleeper due on the way.

        Settles first so timers armed just before the call have registered
        their deadlines, then steps through each parked deadline at or
        before the target, earliest first, settling after each release.
        """
        await self.settle()
        target = self._mono + seconds
        while True:
            # Parked sleepers due at or before the target, earliest first;
            # ties release in the order they parked.
            due = [p for p in self._parked if p.deadline <= target]
            if not due:
                break
            park = min(due, key=lambda p: p.deadline)
            self._step_to(park.deadline)
            self._release(park)
            await self.settle()
        self._step_to(target)
        await self.settle()

    def _step_to(self, mono: float) -> None:
        """Move monotonic time to ``mono`` and wall time by the same amount."""
        self._wall += mono - self._mono
        self._mono = mono


def install_kernel_clock(
    pm, *, start: float = DEFAULT_START, started: bool = True
) -> tuple[FakeClock, "IdleTracker"]:
    """Register a FakeClock as "clock" and an IdleTracker on it as "idle".

    The tracker is started unless ``started`` is False.
    """
    # Imported here so a missing tracker fails the tests that need it, not
    # every test that only wants the clock.
    from corvidae.idle import IdleTracker

    clock = FakeClock(start)
    idle = IdleTracker(clock)
    pm.register(clock, name="clock")
    pm.register(idle, name="idle")
    if started:
        idle.start()
    return clock, idle


def clock_pm():
    """A stand-in plugin manager whose "clock" is a real Clock.

    For tests that start McpClientPlugin and don't drive time; every other
    plugin lookup returns None.
    """
    from unittest.mock import MagicMock

    pm = MagicMock()
    pm.get_plugin.side_effect = lambda n: Clock() if n == "clock" else None
    return pm
