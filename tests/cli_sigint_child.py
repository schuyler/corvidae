"""Child daemon for tests/test_cli_sigint.py (#759); not collected by pytest.

Runs a real ``Runtime`` through ``corvidae.runtime.run_daemon`` on an offline
config with one ``cli:local`` channel, so ``CLIPlugin`` starts its stdin read
loop. The test starts this process with stdin an **open pipe that never sends a
line**, waits for READY, and sends SIGINT: the process must exit 0 within the
graceful bound, with no exit-watchdog WARNING about a leaked thread.

Usage: python cli_sigint_child.py CONFIG_PATH
"""

from __future__ import annotations

import argparse
import asyncio

from corvidae.runtime import Runtime


class _ChildRuntime(Runtime):
    """Runtime that announces READY once start() has returned."""

    async def start(self) -> None:
        """Start for real, then print READY from the loop once running."""
        await super().start()
        # call_soon, not a direct print: run() installs the signal handlers
        # after start() returns, so this runs only once they are in place.
        asyncio.get_running_loop().call_soon(lambda: print("READY", flush=True))


def main() -> None:
    """Parse the config path, build the runtime and run it as the daemon would."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    runtime = _ChildRuntime(config_path=args.config_path)
    # Imported here so a missing run_daemon fails with a clear traceback in
    # the child's output rather than at collection time.
    from corvidae.runtime import run_daemon

    run_daemon(runtime)


if __name__ == "__main__":
    main()
