"""Tests for the kernel clock (corvidae/clock.py; plans/kernel-clock.md).

All but one test drive a FakeClock, so none sleeps real time (R7). The
fake inherits call_later, call_at, cancel and close from the real Clock,
so these tests exercise the production timer code.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from fake_clock import FakeClock


@pytest_asyncio.fixture
async def clock():
    """A FakeClock, closed at teardown so no timer task outlives the test."""
    fake = FakeClock()
    yield fake
    await fake.close()


async def test_call_later_runs_callback_after_delay(clock):
    """R5: a timer runs its callback once its delay has passed, not before."""
    calls: list[float] = []

    async def callback() -> None:
        # Record the fake monotonic time of the call.
        calls.append(clock.monotonic())

    start = clock.monotonic()
    clock.call_later(5, callback, name="later")

    # One second short of the delay: nothing has run.
    await clock.advance(4)
    assert calls == []

    # The last second: the callback runs exactly once, at the deadline.
    await clock.advance(1)
    assert calls == [start + 5]

    # One-shot: it never re-arms itself.
    await clock.advance(100)
    assert calls == [start + 5]


async def test_cancel_before_due_never_runs(clock):
    """R5: cancelling a waiting timer means its callback never runs."""
    calls: list[int] = []

    async def callback() -> None:
        calls.append(1)

    timer = clock.call_later(5, callback, name="cancelled")
    await clock.advance(1)
    timer.cancel()

    # Well past the deadline: still nothing, and the task is gone.
    await clock.advance(100)
    assert calls == []
    assert clock.live_timer_names() == []


async def test_cancel_inside_running_callback_is_noop(clock):
    """R5: Timer.cancel() does nothing once the callback has started, so a
    callback can cancel its own handle without killing itself."""
    handles: list = []
    finished: list[int] = []

    async def callback() -> None:
        # Cancel our own handle, then yield so a cancellation would land.
        handles[0].cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        finished.append(1)

    handles.append(clock.call_later(5, callback, name="self-cancel"))
    await clock.advance(5)

    # The callback ran to the end despite cancelling itself.
    assert finished == [1]


async def test_call_at_runs_at_wall_time(clock):
    """R5: call_at fires when wall time reaches the requested instant."""
    calls: list[float] = []

    async def callback() -> None:
        # Record the fake wall time of the call.
        calls.append(clock.now())

    when = clock.now() + 30
    clock.call_at(when, callback, name="at")

    await clock.advance(29)
    assert calls == []

    await clock.advance(1)
    assert calls == [when]


async def test_call_at_notices_wall_jump_within_max_wall_sleep(clock):
    """R6: a call_at never sleeps longer than MAX_WALL_SLEEP_SECONDS, so a
    forward wall-clock step is noticed within the hour."""
    calls: list[float] = []

    async def callback() -> None:
        calls.append(clock.now())

    # Due 10000 s of wall time from now: far beyond the hour cap.
    when = clock.now() + 10_000
    clock.call_at(when, callback, name="far")
    await clock.advance(0)

    # The wall clock steps forward 2 h, as after a suspend. Without the cap
    # the timer would be parked for the full 10000 s of monotonic time.
    clock.jump(7200)
    await clock.advance(3600)

    assert len(calls) == 1
    assert calls[0] >= when


async def test_sleep_returns_when_wake_set(clock):
    """R6: sleep returns as soon as the wake event is set, before its delay."""
    wake = asyncio.Event()
    sleeper = asyncio.create_task(clock.sleep(1000, wake))

    # The sleeper has parked once its delay shows up on the schedule.
    assert await clock.next_sleep() == 1000
    await clock.settle()
    assert not sleeper.done()

    # Setting the event ends the sleep with no fake time having passed.
    before = clock.monotonic()
    wake.set()
    await asyncio.wait_for(sleeper, timeout=2)
    assert clock.monotonic() == before


async def test_close_cancels_pending_timers(clock):
    """R5: close() cancels every timer still waiting, so none ever runs."""
    calls: list[int] = []

    async def callback() -> None:
        calls.append(1)

    clock.call_later(10, callback, name="a")
    clock.call_later(20, callback, name="b")
    await clock.advance(1)
    assert sorted(clock.live_timer_names()) == ["a", "b"]

    await clock.close()
    # Let any done-callback that removes a finished timer run.
    await asyncio.sleep(0)
    assert clock.live_timer_names() == []

    # Time passing after close runs nothing.
    await clock.advance(100)
    assert calls == []


async def test_close_cancels_running_callback(clock):
    """R5: close() also cancels a callback that is already running, which
    must let CancelledError through."""
    cancelled: list[int] = []

    async def callback() -> None:
        try:
            # Park on the clock so the callback is running but blocked.
            await clock.sleep(1000)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    clock.call_later(1, callback, name="running")
    await clock.advance(1)

    # The callback has started and is parked inside its own sleep.
    assert clock.live_timer_names() == ["running"]

    await asyncio.wait_for(clock.close(), timeout=2)
    assert cancelled == [1]
    # Let any done-callback that removes a finished timer run.
    await asyncio.sleep(0)
    assert clock.live_timer_names() == []


async def test_real_clock_zero_delay_timer_runs():
    """R5: the real Clock runs a zero-delay timer. The only real-Clock test."""
    from corvidae.clock import Clock

    real = Clock()
    ran = asyncio.Event()

    async def callback() -> None:
        ran.set()

    real.call_later(0, callback, name="real")
    try:
        await asyncio.wait_for(ran.wait(), timeout=2)
    finally:
        await real.close()
