"""Child for tests/corvidae_restart/test_preflight.py (#984); not collected.

The pre-flight must close what it built even when plugin load fails partway —
that is the path a broken config takes, and the path the feature exists for.
`tests/preflight_cleanup_child.py` cannot see it: it installs its watch after
`load_plugins` has returned successfully, so a failing load is invisible.

This child watches from `_cleanup_failed_start` instead, wrapping the relay the
pre-flight's own plugin manager holds before the real cleanup runs.

Usage: python preflight_failed_load_child.py CONFIG_PATH
Prints the pre-flight's exit code, whether cleanup ran, and whether `on_stop`
reached the plugins.
"""

from __future__ import annotations

import asyncio
import sys

from corvidae_restart import preflight

_INSTANCES: list["_WatchingRuntime"] = []


class _WatchingRuntime(preflight.Runtime):
    """The real Runtime, with the failed-load cleanup observed as it happens."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cleanup_called = False
        self.stop_calls: list[int] = []
        _INSTANCES.append(self)

    async def _cleanup_failed_start(self) -> None:
        self.cleanup_called = True
        if self.pm is not None:
            real = self.pm.ahook.on_stop

            def watch(*args, **kwargs):
                self.stop_calls.append(1)
                return real(*args, **kwargs)

            self.pm.ahook.on_stop = watch
        await super()._cleanup_failed_start()


def main() -> None:
    preflight.Runtime = _WatchingRuntime
    code = asyncio.run(preflight._run(sys.argv[1]))
    runtime = _INSTANCES[-1]
    print(f"exit {code}")
    print("cleanup ran" if runtime.cleanup_called else "cleanup not run")
    print(
        "on_stop broadcast"
        if runtime.stop_calls
        else "on_stop not broadcast"
    )


if __name__ == "__main__":
    main()
