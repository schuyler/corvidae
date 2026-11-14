"""Shutdown bounds: the graceful-phase limit and the exit watchdog (#746).

Two operator-set keys in the ``daemon:`` section bound how long a stopping
process may take:

* ``daemon.shutdown_timeout`` (default 3 s, positive only): how long the
  graceful phase (``Runtime.stop()``) may run before a forced exit.
* ``daemon.exit_timeout`` (default 5 s, ``0``/``null`` disables): how long
  the process may take to exit after the graceful phase ends before the
  watchdog forces exit 1.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass

from corvidae.tool_collection import _parse_seconds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShutdownTimeouts:
    """Parsed shutdown bounds, in seconds."""

    # daemon.shutdown_timeout: the graceful-phase bound.
    graceful: float
    # daemon.exit_timeout: the exit-watchdog bound; None means disabled.
    exit: float | None


DEFAULT_SHUTDOWN_TIMEOUTS = ShutdownTimeouts(graceful=3.0, exit=5.0)


# Upper bound on how long a plugin's on_stop may take. Runtime gives the whole
# stop broadcast daemon.shutdown_timeout (3 s by default), so each slow plugin
# (MCP client, web transport) takes the graceful budget less a reserve for the
# rest of the broadcast (other plugins' on_stop, idle.stop, clock.close) and,
# for MCP, the cancel grace spent after the bound (#760). At the default 3 s
# that is the historical 2.5 s.
STOP_RESERVE_SECONDS = 0.5
# Smallest bound worth handing to a timeout: a value that rounds to zero turns
# the wait into a poll, and a config value can be arbitrarily small.
MIN_STOP_BOUND_SECONDS = 0.05
# Fallback for when the config cannot be read (on_stop without on_init).
DEFAULT_STOP_BOUND_SECONDS = 2.5


def stop_bound_seconds(shutdown_timeout: float) -> float:
    """A plugin's share of the graceful phase (``daemon.shutdown_timeout``).

    A reserve rather than a fraction, because a cancel grace is spent *after*
    the bound: a bound of 80% of a 1 s budget would cancel at 0.8 s and then
    wait up to 0.3 s more, which is the overrun this bound exists to prevent
    (#760). Never zero, so a tiny budget still gets a real timeout.
    """
    return max(MIN_STOP_BOUND_SECONDS, shutdown_timeout - STOP_RESERVE_SECONDS)


def parse_shutdown_timeouts(daemon: object) -> ShutdownTimeouts:
    """Parse ``daemon.shutdown_timeout`` and ``daemon.exit_timeout``.

    ``daemon`` is the raw ``daemon:`` section (None when absent). Raises
    ValueError naming the key, the rejected value and the accepted values.
    """
    # An absent section keeps both defaults; anything else must be a mapping.
    if daemon is None:
        return DEFAULT_SHUTDOWN_TIMEOUTS
    if not isinstance(daemon, dict):
        raise ValueError(
            f"daemon must be a mapping of daemon settings (e.g. "
            f"{{shutdown_timeout: 3}}); got {daemon!r} ({type(daemon).__name__})"
        )

    # exit_timeout: absent keeps the default; 0/null disables the watchdog.
    if "exit_timeout" in daemon:
        exit_timeout = _parse_seconds("daemon.exit_timeout", daemon["exit_timeout"])
    else:
        exit_timeout = DEFAULT_SHUTDOWN_TIMEOUTS.exit

    # shutdown_timeout: absent keeps the default; otherwise positive only.
    graceful = DEFAULT_SHUTDOWN_TIMEOUTS.graceful
    if "shutdown_timeout" in daemon:
        value = daemon["shutdown_timeout"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"daemon.shutdown_timeout must be a number of seconds > 0; "
                f"got {value!r} ({type(value).__name__})"
            )
        graceful = float(value)
    return ShutdownTimeouts(graceful=graceful, exit=exit_timeout)


def arm_exit_watchdog(timeout: float | None) -> None:
    """Start a daemon thread that forces exit 1 after ``timeout`` seconds.

    None disables the watchdog. CPython joins non-daemon threads before
    atexit, but a daemon thread keeps running during that join, so it can
    still end the process when a leaked thread would otherwise hang it.
    """
    # Disabled: arm nothing.
    if timeout is None:
        return

    def _watch() -> None:
        """Wait out the bound, log the survivors, flush logs and exit 1."""
        # Clamp: Event.wait raises OverflowError above TIMEOUT_MAX.
        threading.Event().wait(min(timeout, threading.TIMEOUT_MAX))
        # Name every surviving non-daemon thread except the main thread and
        # this watchdog (which is a daemon anyway).
        me = threading.current_thread()
        main = threading.main_thread()
        names = [
            t.name
            for t in threading.enumerate()
            if t is not me and t is not main and not t.daemon
        ]
        logger.warning(
            "process did not exit within %ss after graceful shutdown finished "
            "(daemon.exit_timeout); forcing exit 1. Non-daemon threads still "
            "running: %s. Something left a thread running or blocked, often a "
            "connection a plugin's on_stop did not close.",
            timeout,
            ", ".join(names) if names else "none",
        )
        # Flush (not close) the root handlers so the line survives os._exit.
        # The exit sits in a finally: a handler whose flush raises (a closed
        # stream, a broken pipe) must not kill this thread and bring back
        # the hang.
        try:
            for handler in logging.getLogger().handlers:
                handler.flush()
        finally:
            os._exit(1)

    threading.Thread(target=_watch, name="exit-watchdog", daemon=True).start()
