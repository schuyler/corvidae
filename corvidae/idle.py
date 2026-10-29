"""Idle signal: per-channel and system quiet, on the kernel clock.

Runtime builds one IdleTracker and registers it as "idle". The Agent pushes
activity and settle events, PersistencePlugin seeds channels from history,
and plugins register watches that fire after a period of quiet and again at
a bounded rate while the quiet lasts. The tracker never reads Agent queues,
never enqueues and never starts a turn.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from corvidae.clock import Clock, Timer

logger = logging.getLogger(__name__)

ChannelIdleCallback = Callable[[str], Awaitable[None]]
SystemIdleCallback = Callable[[], Awaitable[None]]

#: Daemon keys removed with the on_idle hook (#165).
_REMOVED_DAEMON_KEYS = ("idle_cooldown_seconds", "idle_poll_interval")


def validate_daemon_config(daemon: dict) -> None:
    """Raise ValueError if ``daemon`` carries a removed idle key."""
    for key in _REMOVED_DAEMON_KEYS:
        if key in daemon:
            raise ValueError(
                f"daemon.{key} was removed (#165): idle work now runs on the "
                "kernel clock, and each plugin sets its own period "
                "(e.g. memory.idle_consolidate_after); delete the key"
            )


@dataclass
class _ChannelState:
    """What the tracker knows about one channel."""

    # True from note_activity until the next note_settled.
    busy: bool = False
    # Monotonic time of the last settle (or restored activity); None while busy.
    quiet_since: float | None = None
    # Bumped by every note_activity.
    epoch: int = 0


class IdleWatch:
    """A registered request to be called when a channel, or the system, is quiet."""

    def __init__(
        self,
        tracker: IdleTracker,
        name: str,
        period: float,
        repeat: float,
        callback: ChannelIdleCallback | SystemIdleCallback,
        system: bool,
    ) -> None:
        self.name = name
        self.period = period
        self.repeat = repeat
        self.cancelled = False
        self._tracker = tracker
        self._callback = callback
        self._system = system
        # Waiting timers, keyed by channel id, or None for the system.
        self._timers: dict[str | None, Timer] = {}

    def cancel(self) -> None:
        """Stop the watch for good: cancel waiting timers and unregister it."""
        self.cancelled = True
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        self._tracker._remove_watch(self)


class IdleTracker:
    """Per-channel and system quiet, fed by pushes from the Agent."""

    def __init__(self, clock: Clock) -> None:
        # The clock arms watch timers; nothing arms before start().
        self._clock = clock
        self._started = False
        # Per-channel activity and quiet state, keyed by channel id.
        self._channels: dict[str, _ChannelState] = {}
        # Monotonic time the system went quiet; None while any channel is busy.
        self._system_quiet_since: float | None = None
        # Bumped on any activity so a stale system callback doesn't re-arm.
        self._system_epoch = 0
        # Registered watches, in registration order.
        self._watches: list[IdleWatch] = []

    # ------------------------------------------------------------------
    # Lifecycle (Runtime only)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin firing: arm every watch for every quiet channel and the system."""
        self._started = True
        # The system is quiet from start() unless a channel is already busy.
        if not self._any_busy():
            self._system_quiet_since = self._clock.monotonic()
        for watch in list(self._watches):
            self._arm_watch_all(watch)

    def stop(self) -> None:
        """Stop firing and cancel every waiting timer. Watches stay registered."""
        self._started = False
        for watch in self._watches:
            for timer in watch._timers.values():
                timer.cancel()

    # ------------------------------------------------------------------
    # Activity feed (Agent only)
    # ------------------------------------------------------------------

    def note_activity(self, channel_id: str) -> None:
        """Mark ``channel_id`` busy and cancel its pending quiet timers."""
        state = self._channels.setdefault(channel_id, _ChannelState())
        state.busy = True
        state.quiet_since = None
        state.epoch += 1
        for watch in self._watches:
            if not watch._system:
                self._cancel_timer(watch, channel_id)
        # Any busy channel makes the system busy too.
        self._system_quiet_since = None
        self._system_epoch += 1
        for watch in self._watches:
            if watch._system:
                self._cancel_timer(watch, None)

    def note_settled(self, channel_id: str) -> None:
        """Mark ``channel_id`` drained and arm the watches that now apply."""
        state = self._channels.setdefault(channel_id, _ChannelState())
        now = self._clock.monotonic()
        state.busy = False
        state.quiet_since = now
        for watch in list(self._watches):
            if not watch._system:
                self._arm(watch, channel_id, now + watch.period)
        # The system goes quiet only when no channel is left busy.
        if not self._any_busy():
            self._system_quiet_since = now
            for watch in list(self._watches):
                if watch._system:
                    self._arm(watch, None, now + watch.period)

    # ------------------------------------------------------------------
    # Restart seed (PersistencePlugin only)
    # ------------------------------------------------------------------

    def note_restored(self, channel_id: str, last_activity: float) -> None:
        """Seed an untracked channel as quiet since ``last_activity`` (wall time)."""
        if channel_id in self._channels:
            return
        # Convert the wall-time age of the last activity to monotonic time.
        age = max(0.0, self._clock.now() - last_activity)
        quiet_since = self._clock.monotonic() - age
        self._channels[channel_id] = _ChannelState(quiet_since=quiet_since)
        for watch in list(self._watches):
            if not watch._system:
                self._arm(watch, channel_id, quiet_since + watch.period)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def watch_channels(
        self,
        period: float,
        callback: ChannelIdleCallback,
        *,
        name: str,
        repeat: float | None = None,
    ) -> IdleWatch:
        """Call ``callback(channel_id)`` once a channel has been quiet ``period`` seconds."""
        watch = IdleWatch(
            self, name, period, period if repeat is None else repeat, callback, False
        )
        self._watches.append(watch)
        self._arm_watch_all(watch)
        return watch

    def watch_system(
        self,
        period: float,
        callback: SystemIdleCallback,
        *,
        name: str,
        repeat: float | None = None,
    ) -> IdleWatch:
        """Call ``callback()`` once no channel has been busy for ``period`` seconds."""
        watch = IdleWatch(
            self, name, period, period if repeat is None else repeat, callback, True
        )
        self._watches.append(watch)
        self._arm_watch_all(watch)
        return watch

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _remove_watch(self, watch: IdleWatch) -> None:
        """Unregister ``watch`` (called by IdleWatch.cancel)."""
        if watch in self._watches:
            self._watches.remove(watch)

    def _any_busy(self) -> bool:
        """True if any tracked channel is busy."""
        return any(state.busy for state in self._channels.values())

    def _cancel_timer(self, watch: IdleWatch, key: str | None) -> None:
        """Cancel ``watch``'s waiting timer for ``key``, if any."""
        timer = watch._timers.pop(key, None)
        if timer is not None:
            timer.cancel()

    def _arm_watch_all(self, watch: IdleWatch) -> None:
        """Arm ``watch`` for every key that is currently quiet."""
        if watch._system:
            if self._system_quiet_since is not None:
                self._arm(watch, None, self._system_quiet_since + watch.period)
            return
        for cid, state in list(self._channels.items()):
            if not state.busy and state.quiet_since is not None:
                self._arm(watch, cid, state.quiet_since + watch.period)

    def _arm(self, watch: IdleWatch, key: str | None, t: float) -> None:
        """Arm ``watch`` for ``key`` to fire at monotonic time ``t``.

        The only place timers are created. No-op unless the tracker is
        started and the watch is not cancelled.
        """
        if not self._started or watch.cancelled:
            return
        # Replace any waiting timer for this key.
        self._cancel_timer(watch, key)
        label = "system" if key is None else key

        async def fire() -> None:
            # Timer callback: check quiet, call the watcher, maybe re-arm.
            await self._fire(watch, key)

        watch._timers[key] = self._clock.call_later(
            max(0.0, t - self._clock.monotonic()),
            fire,
            name=f"idle:{watch.name}:{label}",
        )

    def _is_quiet(self, key: str | None) -> bool:
        """True if ``key`` (a channel id, or None for the system) is quiet now."""
        if key is None:
            return self._system_quiet_since is not None
        state = self._channels.get(key)
        return state is not None and not state.busy

    def _epoch(self, key: str | None) -> int:
        """Current activity epoch for ``key``."""
        return self._system_epoch if key is None else self._channels[key].epoch

    async def _fire(self, watch: IdleWatch, key: str | None) -> None:
        """Run a watch's callback for ``key`` and re-arm if the quiet continues."""
        # Re-check that the key is still quiet.
        if not self._is_quiet(key):
            return
        epoch = self._epoch(key)
        # Exceptions propagate to the clock, which logs them.
        if key is None:
            await watch._callback()
        else:
            await watch._callback(key)
        # Re-arm only if nothing changed while the callback ran.
        if (
            self._epoch(key) == epoch
            and self._started
            and not watch.cancelled
        ):
            self._arm(watch, key, self._clock.monotonic() + watch.repeat)
