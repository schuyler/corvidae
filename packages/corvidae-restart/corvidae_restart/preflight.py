"""Pre-flight check for the restart plugin.

Runs the daemon's own startup through plugin on_init (logging setup,
section validation, channel config, plugin load, hook and dependency
checks, on_init), so a config that passes the pre-flight is one the
daemon can boot. on_start is not run and no agent is started. Run as a
subprocess by the restart tool before the daemon is killed.

Output: `pre-flight OK` on success. On failure, one stderr line: the
kernel's own `Config error:` line, or `pre-flight failed: ...` for any
other exception. Exit code 0 or 1.

Usage:
    python -P -m corvidae_restart.preflight --config <path>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import warnings

from corvidae.runtime import Runtime


async def _run(config_path: str) -> int:
    """Run the daemon's startup through on_init; return the exit code."""
    runtime = Runtime(config_path=config_path)
    try:
        config = runtime.load_config()
        runtime.setup_logging(config)
        try:
            await runtime.load_plugins(config)
        finally:
            # Close whatever load_plugins built; returns at once if it
            # never created a plugin manager.
            await runtime._cleanup_failed_start()
        return 0
    except SystemExit as exc:
        # The kernel already wrote its one-line Config error to stderr.
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        # Any other failure: one line naming the exception and the remedy.
        print(
            f"pre-flight failed: {exc} ({type(exc).__name__}); fix the config "
            f"or the plugin set it names, then retry the restart — "
            f"check {config_path}",
            file=sys.stderr,
        )
        return 1
    finally:
        # _cleanup_failed_start is best effort: a plugin that leaked a
        # non-daemon thread during on_init, whether the load then succeeded
        # or raised, used to hold this process open until the restart tool's
        # preflight_timeout killed it (default 120 s) (#757). Arm the same
        # bound the daemon uses on every path — armed here, inside the
        # coroutine, because the wait it bounds happens after this returns,
        # when asyncio shuts down and the interpreter joins non-daemon
        # threads. shutdown_timeouts holds the defaults until load_plugins
        # parses the config, so an early failure is bounded too.
        runtime.arm_exit_bound()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight check for restart")
    parser.add_argument(
        "--config", required=True, help="Path to the agent config file"
    )
    args = parser.parse_args()

    # Keep kernel logs and plugin import warnings out of stderr; kernel
    # problems are reported by direct stderr writes, which these do not touch.
    warnings.simplefilter("ignore")
    logging.disable(logging.CRITICAL)

    code = asyncio.run(_run(args.config))
    # Any non-daemon thread still running now is a plugin's: asyncio has
    # joined its own executor, which is what a check inside the coroutine
    # would have caught instead. Name it directly — kernel logging is off
    # here, so the watchdog's own WARNING would go nowhere the operator can
    # see — and refuse the restart, because the daemon would leak it too.
    # This runs on the failure path as well, after the failure's own stderr
    # line, so a leak that preceded a failure is named too.
    survivors = [
        t.name
        for t in threading.enumerate()
        if t is not threading.main_thread() and not t.daemon
    ]
    if survivors:
        print(
            f"pre-flight: {', '.join(survivors)} still running after the "
            f"plugins were stopped; a plugin opened something in on_init and "
            f"did not close it in on_stop. Refusing the restart; this process "
            f"exits within daemon.exit_timeout.",
            file=sys.stderr,
        )
        return 1
    # A failed pre-flight already said why on stderr; only success reports OK.
    if code != 0:
        return code
    print("pre-flight OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
