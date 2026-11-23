"""Child daemon for tests/test_exit_watchdog.py (#746); not collected by pytest.

Runs a real ``Runtime`` through ``corvidae.runtime.run_daemon`` (the entry
point that arms the exit watchdog) on an offline config, optionally leaking an
aiosqlite connection and/or registering a plugin whose ``on_stop`` raises.

Usage: python exit_watchdog_child.py CONFIG_PATH [--leak] [--raise-on-stop]
                                                  [--raise-in-start]
"""

from __future__ import annotations

import argparse
import asyncio

import aiosqlite

from corvidae.hooks import hookimpl
from corvidae.runtime import Runtime


# Strong references to leaked connections, alive until interpreter exit.
_LEAKED: list[aiosqlite.Connection] = []


class _LeaksThenFailsStart:
    """Plugin that opens a connection in on_start and then raises (#757).

    The connection is never closed, so its worker thread is a non-daemon
    survivor of the failed boot: before #757 the process then sat alive and
    silent under systemd, with no bound on interpreter exit.
    """

    @hookimpl
    async def on_start(self, config: dict) -> None:
        _LEAKED.append(await aiosqlite.connect(":memory:"))
        raise RuntimeError("leaker: on_start failed on purpose")


class _RaisingOnStop:
    """Plugin whose on_stop always raises, so stop() propagates an exception."""

    @hookimpl
    async def on_stop(self) -> None:
        # The raise is the point: apluggy's gather lets it escape stop().
        raise RuntimeError("raiser: on_stop failed on purpose")


class _ChildRuntime(Runtime):
    """Runtime with test hooks: an optional raising plugin and leaked connection."""

    def __init__(
        self,
        config_path: str,
        leak: bool,
        raise_on_stop: bool,
        raise_in_start: bool = False,
    ) -> None:
        super().__init__(config_path=config_path)
        self._leak = leak
        self._raise_on_stop = raise_on_stop
        self._raise_in_start = raise_in_start

    async def load_plugins(self, config: dict) -> None:
        """Load the real plugin set, then add the test plugins if asked."""
        await super().load_plugins(config)
        if self._raise_on_stop:
            self.pm.register(_RaisingOnStop(), name="raiser")
        if self._raise_in_start:
            self.pm.register(_LeaksThenFailsStart(), name="leaker")

    async def start(self) -> None:
        """Start for real, leak a connection if asked, then schedule READY."""
        await super().start()
        if self._leak:
            # Opened and never closed: its worker thread is non-daemon. Held
            # at module level, not on the runtime: the runtime is freed when
            # main() returns, and a collected connection stops its own thread.
            _LEAKED.append(await aiosqlite.connect(":memory:"))
        # call_soon, not a direct print: run() installs the SIGTERM handler
        # only after start() returns, and this callback runs once run() is
        # awaiting the stop event, i.e. after the handler is in place.
        asyncio.get_running_loop().call_soon(lambda: print("READY", flush=True))


def main() -> None:
    """Parse flags, build the runtime and run it as the daemon would."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--leak", action="store_true")
    parser.add_argument("--raise-on-stop", action="store_true")
    parser.add_argument("--raise-in-start", action="store_true")
    args = parser.parse_args()
    runtime = _ChildRuntime(
        args.config_path, args.leak, args.raise_on_stop, args.raise_in_start
    )
    # Imported here so a missing run_daemon fails with a clear traceback in
    # the child's output rather than at collection time.
    from corvidae.runtime import run_daemon

    run_daemon(runtime)


if __name__ == "__main__":
    main()
