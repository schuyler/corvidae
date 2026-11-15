"""Child for tests/corvidae_restart/test_preflight.py (#563); not collected.

Runs the pre-flight in a real process and reports whether it closed what it
built. ``_cleanup_failed_start`` is the only thing that broadcasts ``on_stop``
to a pre-flight's plugins, and nothing noticed when that call was removed: no
test went red.

The broadcast is watched by wrapping the hook relay the pre-flight's own
Runtime builds, the way the leak child wraps ``load_plugins``, rather than by
adding a test hook to production code.

Usage: python preflight_cleanup_child.py CONFIG_PATH
Prints the pre-flight's exit code, then ``on_stop broadcast`` or
``on_stop not broadcast``.
"""

from __future__ import annotations

import asyncio
import sys

from corvidae_restart import preflight

# Every runtime the pre-flight built, so the watch can be read afterwards.
_INSTANCES: list["_WatchingRuntime"] = []


class _WatchingRuntime(preflight.Runtime):
    """The real Runtime, with the on_stop broadcast recorded as it happens."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stop_calls: list[int] = []
        _INSTANCES.append(self)

    async def load_plugins(self, config: dict) -> None:
        await super().load_plugins(config)
        real = self.pm.ahook.on_stop

        def watch(*args, **kwargs):
            self.stop_calls.append(1)
            return real(*args, **kwargs)

        self.pm.ahook.on_stop = watch


def main() -> None:
    preflight.Runtime = _WatchingRuntime
    code = asyncio.run(preflight._run(sys.argv[1]))
    print(f"exit {code}")
    calls = _INSTANCES[-1].stop_calls if _INSTANCES else []
    print("on_stop broadcast" if calls else "on_stop not broadcast")


if __name__ == "__main__":
    main()
