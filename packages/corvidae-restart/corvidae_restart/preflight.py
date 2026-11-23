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
from pathlib import Path

from corvidae.runtime import Runtime


async def _run(config_path: str) -> int:
    """Run the daemon's startup through on_init; return the exit code."""
    return (await _run_with_bound(config_path))[0]


async def _run_with_bound(config_path: str) -> tuple[int, float | None]:
    """The pre-flight's exit code, and the exit bound it armed (#854).

    The survivor line has to say whether the process has a bound at all, so the
    bound comes back with the code rather than leaking through module state.
    """
    runtime = Runtime(config_path=config_path)
    code = await _preflight_body(runtime, config_path)
    return code, runtime.shutdown_timeouts.exit


async def _preflight_body(runtime: Runtime, config_path: str) -> int:
    """The pre-flight's work, once the runtime and its bound are known."""
    try:
        config = runtime.load_config()
        runtime.setup_logging(config)
        try:
            await runtime.load_plugins(config)
            # A skipped plugin or a fallen-back replacement means the daemon
            # would boot a different plugin set than configured (R14).
            report = runtime.load_report
            if report.fallbacks or report.skipped:
                parts = [
                    f'plugin "{n}" failed to load (skipped)' for n in report.skipped
                ] + [
                    f'plugin "{t}" replacing "{s}" failed to load (bundled used)'
                    for s, t in report.fallbacks.items()
                ]
                print(
                    f"pre-flight failed: {'; '.join(parts)} — fix or reinstall "
                    f"them, or remove the plugins.replace entry for a fallen-back "
                    f"slot, before restarting",
                    file=sys.stderr,
                )
                return 1
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


def check_corvidae_location(expected: str | None) -> str | None:
    """Return a refusal when this process's corvidae is not the expected one.

    An editable install of a single workspace package repoints ``corvidae`` at
    that worktree, because every package declares the workspace source, so a
    restart would boot code nobody reviewed (#1008).
    """
    import corvidae

    if not expected:
        return None
    here = Path(corvidae.__file__).resolve().parent
    want = Path(expected).resolve()
    if here == want:
        return None
    return (
        f"this venv resolves corvidae to {here}, not to {want}. An editable "
        f"install of a workspace package from elsewhere repoints it; re-run the "
        f"deployment's requirements install before restarting (#1008)"
    )


def _exit_sentence(bound: float | None) -> str:
    """What happens to this process next, said truthfully (#854)."""
    if bound is None:
        return (
            "this process has no exit bound (daemon.exit_timeout is disabled), "
            "so the restart tool will kill it at its own preflight timeout."
        )
    if bound <= 0:
        return (
            "this process has no exit bound (daemon.exit_timeout is 0), so the "
            "restart tool will kill it at its own preflight timeout."
        )
    return f"this process exits within daemon.exit_timeout ({bound:g} s)."


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight check for restart")
    parser.add_argument(
        "--config", required=True, help="Path to the agent config file"
    )
    parser.add_argument(
        "--expect-corvidae",
        default=None,
        help=(
            "Directory the running process imports corvidae from; the "
            "pre-flight refuses when this one resolves elsewhere (#1008)"
        ),
    )
    args = parser.parse_args()

    mismatch = check_corvidae_location(args.expect_corvidae)
    if mismatch is not None:
        print(f"pre-flight failed: {mismatch}", file=sys.stderr)
        return 1

    # Keep kernel logs and plugin import warnings out of stderr; kernel
    # problems are reported by direct stderr writes, which these do not touch.
    warnings.simplefilter("ignore")
    logging.disable(logging.CRITICAL)

    code, exit_bound = asyncio.run(_run_with_bound(args.config))
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
            f"did not close it in on_stop. Refusing the restart; "
            f"{_exit_sentence(exit_bound)}",
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
