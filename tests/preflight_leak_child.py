"""Child for tests/corvidae_restart/test_preflight.py (#757); not collected.

Runs the pre-flight in a real process with a plugin that leaks an aiosqlite
connection during ``on_init`` and never closes it, so the connection's
non-daemon worker thread outlives the pre-flight's cleanup. Before #757 the
process then sat alive until the restart tool's ``preflight_timeout_seconds``
(default 120 s) killed the group.

The leak is injected by subclassing the pre-flight's own ``Runtime`` rather
than by adding a test hook to the production code.

With a second argument ``raise``, the plugin load raises after leaking the
connection, so the pre-flight's failure path is what is under test.

Usage: python preflight_leak_child.py CONFIG_PATH [raise]
"""

from __future__ import annotations

import sys

import aiosqlite
from corvidae_restart import preflight

from corvidae.shutdown import parse_shutdown_timeouts

# Set by main() from argv: raise after leaking, to exercise the failure path.
_RAISE_AFTER_LEAK = False

# Strong reference: a collected connection closes its own thread.
_LEAKED: list[aiosqlite.Connection] = []


class _LeakyRuntime(preflight.Runtime):
    """Runtime whose plugin load leaks a connection and starts nothing else."""

    async def load_plugins(self, config: dict) -> None:
        # The real load_plugins parses the daemon bounds at step 7; this
        # override loads no plugins, so parse them here to keep the bound
        # under test the one the config sets rather than the default.
        self.shutdown_timeouts = parse_shutdown_timeouts(config.get("daemon"))
        _LEAKED.append(await aiosqlite.connect(":memory:"))
        # Failure mode: the plugin dies after opening the connection.
        if _RAISE_AFTER_LEAK:
            raise RuntimeError("plugin failed after opening a connection")


def main() -> None:
    # Patch before main() builds its Runtime, so the pre-flight's own code
    # path (load_config, setup_logging, cleanup, arm) is what runs.
    global _RAISE_AFTER_LEAK
    _RAISE_AFTER_LEAK = len(sys.argv) > 2 and sys.argv[2] == "raise"
    preflight.Runtime = _LeakyRuntime
    sys.argv = ["preflight", "--config", sys.argv[1]]
    sys.exit(preflight.main())


if __name__ == "__main__":
    main()
