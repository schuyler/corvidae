"""Tests for the kernel idle tracker (corvidae/idle.py; plans/kernel-clock.md).

Time comes from a FakeClock, so no test sleeps real time (R7). Every
period, repeat and advance is an integral number of seconds, so timings
are asserted with ``==``. Callbacks record elapsed fake seconds since the
fixture's clock start.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from fake_clock import DEFAULT_START, install_kernel_clock


@pytest_asyncio.fixture
async def unstarted(pm):
    """A FakeClock and an IdleTracker that has not been started."""
    clock, idle = install_kernel_clock(pm, started=False)
    yield clock, idle
    idle.stop()
    await clock.close()


class Recorder:
    """Records each callback as (elapsed fake seconds, channel id or None)."""

    def __init__(self, clock) -> None:
        self.clock = clock
        self.calls: list[tuple[float, str | None]] = []

    async def channel(self, cid: str) -> None:
        """Channel-watch callback."""
        self.calls.append((self.clock.now() - DEFAULT_START, cid))

    async def system(self) -> None:
        """System-watch callback."""
        self.calls.append((self.clock.now() - DEFAULT_START, None))

    def times(self) -> list[float]:
        """Elapsed times of every recorded call."""
        return [t for t, _ in self.calls]


def _idle_timers(clock) -> list[str]:
    """Names of live clock timers that belong to idle watches."""
    return [n for n in clock.live_timer_names() if n.startswith("idle:")]


async def test_channel_watch_fires_period_after_settle(kernel_clock):
    """R3: a channel watch fires `period` seconds after the channel settles,
    with the channel id."""
    clock, idle = kernel_clock
    rec = Recorder(clock)
    idle.watch_channels(60, rec.channel, name="w")

    idle.note_activity("a")
    await clock.advance(10)
    idle.note_settled("a")

    # One second short of the period: no call yet.
    await clock.advance(59)
    assert rec.calls == []

    await clock.advance(1)
    assert rec.calls == [(70, "a")]


async def test_activity_cancels_and_rearms_from_next_settle(kernel_clock):
    """R3: activity cancels the pending call; the next settle restarts the
    period from that settle."""
    clock, idle = kernel_clock
    rec = Recorder(clock)
    idle.watch_channels(60, rec.channel, name="w")

    idle.note_activity("a")
    idle.note_settled("a")
    await clock.advance(30)

    # Activity at t=30 cancels the call that was due at t=60.
    idle.note_activity("a")
    await clock.advance(100)
    assert rec.calls == []

    # Settling at t=130 restarts the clock: due at t=190, not before.
    idle.note_settled("a")
    await clock.advance(59)
    assert rec.calls == []
    await clock.advance(1)
    assert rec.calls == [(190, "a")]


async def test_never_fires_while_channel_busy(kernel_clock):
    """R3: a channel that has work and never settles is never signalled."""
    clock, idle = kernel_clock
    rec = Recorder(clock)
    idle.watch_channels(60, rec.channel, name="w")

    idle.note_activity("a")
    # Ten periods of sustained work, with no settle.
    await clock.advance(600)

    assert rec.calls == []
    assert _idle_timers(clock) == []


async def test_repeats_at_bounded_rate_through_sustained_quiet(kernel_clock):
    """R3: through sustained quiet a watch repeats every `repeat` seconds:
    10 calls over 10 periods, and a longer repeat gives fewer."""
    clock, idle = kernel_clock
    fast = Recorder(clock)
    slow = Recorder(clock)
    idle.watch_channels(60, fast.channel, name="fast")
    idle.watch_channels(60, slow.channel, name="slow", repeat=120)

    idle.note_activity("a")
    idle.note_settled("a")
    await clock.advance(600)

    # repeat defaults to period: exactly one call per period.
    assert fast.times() == [60 * n for n in range(1, 11)]
    # The first call is at `period`; later ones are `repeat` apart.
    assert slow.times() == [60, 180, 300, 420, 540]


async def test_each_watch_keeps_its_own_period(kernel_clock):
    """R3: each plugin picks its own period; watches do not share one."""
    clock, idle = kernel_clock
    short = Recorder(clock)
    long = Recorder(clock)
    # Large repeats so only the first firing of each watch is in play.
    idle.watch_channels(30, short.channel, name="short", repeat=1000)
    idle.watch_channels(90, long.channel, name="long", repeat=1000)

    idle.note_activity("a")
    idle.note_settled("a")

    await clock.advance(30)
    assert short.times() == [30]
    assert long.times() == []

    await clock.advance(60)
    assert short.times() == [30]
    assert long.times() == [90]


async def test_system_watch_waits_for_every_channel(kernel_clock):
    """R3: the system is quiet only when no tracked channel is busy."""
    clock, idle = kernel_clock
    rec = Recorder(clock)
    idle.watch_system(60, rec.system, name="sys")

    idle.note_activity("a")
    idle.note_activity("b")
    idle.note_settled("a")

    # Channel b is still busy: ten periods pass with no system signal.
    await clock.advance(600)
    assert rec.calls == []

    # The last channel settles at t=600; the period runs from there.
    idle.note_settled("b")
    await clock.advance(59)
    assert rec.calls == []
    await clock.advance(1)
    assert rec.calls == [(660, None)]


async def test_system_quiet_measured_from_last_settle(kernel_clock):
    """R3: system quiet counts from the latest settle that left no channel
    busy, so later activity pushes the signal back."""
    clock, idle = kernel_clock
    rec = Recorder(clock)
    idle.watch_system(60, rec.system, name="sys")

    # Channel a settles at t=0: the system is quiet, due at t=60.
    idle.note_activity("a")
    idle.note_settled("a")
    await clock.advance(30)

    # Channel b works from t=30 to t=40, which cancels the t=60 call.
    idle.note_activity("b")
    await clock.advance(10)
    idle.note_settled("b")

    # Nothing at t=60 or t=99; the call lands at t=40+60.
    await clock.advance(59)
    assert rec.calls == []
    await clock.advance(1)
    assert rec.calls == [(100, None)]


async def test_restored_channel_quiet_since_last_activity(kernel_clock):
    """R2: a restored channel is quiet since its last recorded activity, so
    its watch fires `period` after that, not `period` after the restore."""
    clock, idle = kernel_clock
    rec = Recorder(clock)
    idle.watch_channels(60, rec.channel, name="w")

    # Last activity was 20 s ago (wall time): the call is due in 40 s.
    idle.note_restored("a", clock.now() - 20)

    await clock.advance(39)
    assert rec.calls == []
    await clock.advance(1)
    assert rec.calls == [(40, "a")]


async def test_restored_channel_past_period_fires_at_start(unstarted):
    """R2: a restored channel already quiet past its period is signalled as
    soon as the tracker starts, with no traffic at all."""
    clock, idle = unstarted
    rec = Recorder(clock)
    idle.watch_channels(60, rec.channel, name="w")

    # Last activity an hour ago, recorded before start().
    idle.note_restored("a", clock.now() - 3600)
    await clock.advance(0)
    assert rec.calls == []

    idle.start()
    # A zero-delay timer fires only inside advance.
    await clock.advance(0)
    assert rec.calls == [(0, "a")]


async def test_note_restored_ignored_for_tracked_channel(kernel_clock):
    """R2: note_restored does nothing for a channel the tracker already
    knows, whether it is quiet or busy."""
    clock, idle = kernel_clock
    rec = Recorder(clock)
    idle.watch_channels(60, rec.channel, name="w")

    # Quiet channel: settled at t=0. A restore claiming ancient activity
    # would make it overdue and fire at once if it were honoured.
    idle.note_activity("a")
    idle.note_settled("a")
    await clock.advance(30)
    idle.note_restored("a", clock.now() - 10_000)
    await clock.advance(0)
    assert rec.calls == []
    await clock.advance(30)
    assert rec.calls == [(60, "a")]

    # Busy channel: a restore must not mark it quiet.
    idle.note_activity("b")
    idle.note_restored("b", clock.now() - 10_000)
    await clock.advance(600)
    assert [c for _, c in rec.calls if c == "b"] == []


async def test_nothing_fires_before_start(unstarted):
    """R3: before start() state is recorded but nothing is armed or fired."""
    clock, idle = unstarted
    rec = Recorder(clock)
    idle.watch_channels(60, rec.channel, name="chan")
    idle.watch_system(60, rec.system, name="sys")

    idle.note_activity("a")
    idle.note_settled("a")
    await clock.advance(1000)

    assert rec.calls == []
    assert _idle_timers(clock) == []

    # start() arms the recorded state: the overdue channel fires at once,
    # and the system is quiet since start(), so it fires a period later.
    idle.start()
    await clock.advance(0)
    assert rec.calls == [(1000, "a")]
    await clock.advance(59)
    assert rec.calls == [(1000, "a")]
    await clock.advance(1)
    assert (1060, None) in rec.calls


async def test_activity_during_callback_suppresses_rearm(kernel_clock):
    """R3: activity and a settle landing while a callback runs stop the old
    callback from re-arming; the fresh settle's timer is the one that fires."""
    clock, idle = kernel_clock
    calls: list[float] = []

    async def callback(cid: str) -> None:
        calls.append(clock.now() - DEFAULT_START)
        # Only the first call interferes: new work arrives and settles.
        if len(calls) == 1:
            idle.note_activity(cid)
            idle.note_settled(cid)

    # A long repeat makes a wrongly re-armed old timer land far away from
    # the fresh one (and, replacing it, would erase the call at t=120).
    idle.watch_channels(60, callback, name="w", repeat=1000)
    idle.note_activity("a")
    idle.note_settled("a")

    await clock.advance(200)
    assert calls == [60, 120]


async def test_watch_cancel_stops_firings(kernel_clock):
    """R3: a cancelled watch never fires again, including one cancelled
    before its first call."""
    clock, idle = kernel_clock
    rec = Recorder(clock)
    watch = idle.watch_channels(60, rec.channel, name="w")
    never = idle.watch_channels(60, rec.channel, name="never")

    idle.note_activity("a")
    idle.note_settled("a")
    never.cancel()
    await clock.advance(60)
    assert rec.calls == [(60, "a")]

    watch.cancel()
    await clock.advance(600)
    assert rec.calls == [(60, "a")]

    # Fresh activity and a settle arm nothing for a cancelled watch.
    idle.note_activity("a")
    idle.note_settled("a")
    await clock.advance(600)
    assert rec.calls == [(60, "a")]


async def test_watch_cancelled_inside_own_callback_does_not_rearm(kernel_clock):
    """R3: a watch cancelled from inside its own callback fires once and is
    gone: no re-arm, no orphan idle timer left on the clock."""
    clock, idle = kernel_clock
    holder: list = []
    calls: list[float] = []

    async def callback() -> None:
        calls.append(clock.now() - DEFAULT_START)
        # Timer.cancel() is a no-op on a running timer; the watch's own
        # cancelled flag has to stop the re-arm.
        holder[0].cancel()

    holder.append(idle.watch_system(60, callback, name="sys", repeat=60))

    # The system is quiet since start(): it fires once at t=60.
    await clock.advance(60)
    assert calls == [60]

    await clock.advance(600)
    assert calls == [60]
    assert _idle_timers(clock) == []


@pytest.mark.parametrize(
    "key", ["idle_cooldown_seconds", "idle_poll_interval"]
)
def test_validate_daemon_config_rejects_removed_keys_with_remedy(key):
    """R12: each removed daemon key is a ValueError that names the key and
    says to delete it."""
    from corvidae.idle import validate_daemon_config

    with pytest.raises(ValueError) as excinfo:
        validate_daemon_config({key: 30})

    message = str(excinfo.value)
    assert f"daemon.{key}" in message
    assert "delete the key" in message

    # Config without the removed keys is accepted.
    validate_daemon_config({"session_db": "sessions.db"})
