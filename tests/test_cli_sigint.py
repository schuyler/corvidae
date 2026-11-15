"""SIGINT with an open stdin exits cleanly (#759).

``CLIPlugin`` used to read stdin with ``run_in_executor(None, readline)``. An
executor thread blocked in ``readline`` is non-daemon, so ``asyncio.run``'s
executor shutdown and interpreter exit both waited for it: SIGINT at a TTY, or
with an open pipe that never sent a line, logged a watchdog WARNING naming
``ThreadPoolExecutor-0_0`` (to corvidae.log, not the terminal) and exited 1
after ``daemon.exit_timeout``.

Two tests: the small one asserts the shape directly (a daemon reader thread,
prompt cancellation), and the acceptance one runs the real daemon in a
subprocess with an open pipe on stdin and sends SIGINT.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX signals, pipes and process semantics"
)

CHILD = Path(__file__).parent / "cli_sigint_child.py"

logger = logging.getLogger(__name__)

# The minimal profile's disabled list: the cognition set, offline-safe.
MINIMAL_DISABLED = [
    "memory",
    "memory_tools",
    "funnel",
    "appraisal",
    "critique",
    "outcome_log",
    "gates",
    "silence",
    "skills",
    "heartbeat",
    "goals",
]

READY_DEADLINE = 10.0
# shutdown_timeout (1) plus a generous margin, far below an unbounded hang.
EXIT_DEADLINE = 8.0


# ---------------------------------------------------------------------------
# The shape: a daemon reader thread, not an executor thread
# ---------------------------------------------------------------------------


async def test_the_stdin_reader_is_a_daemon_thread_and_stop_is_prompt(monkeypatch):
    """on_stop cancels at once and leaves no non-daemon reader behind.

    With ``run_in_executor`` the reader was a ``ThreadPoolExecutor`` worker:
    non-daemon, and still blocked in ``readline`` after the loop stopped, so
    the process could not exit.
    """
    import os

    from corvidae.channel import ChannelRegistry
    from corvidae.channels.cli import CLIPlugin
    from corvidae.hooks import create_plugin_manager, hookimpl

    read_end, write_end = os.pipe()
    stdin = os.fdopen(read_end, "r", buffering=1)
    monkeypatch.setattr(sys, "stdin", stdin)

    pm = create_plugin_manager()
    registry = ChannelRegistry({})
    pm.register(registry, name="registry")
    # A cli channel is what makes on_start start the read loop at all.
    registry.get_or_create("cli", "local")

    plugin = CLIPlugin(pm=pm)
    await plugin.on_start({})
    # Whatever the body asserts, the reader must be unblocked and the plugin
    # stopped before the test returns: a reader left blocked in readline is a
    # non-daemon executor thread under the pre-#759 code, and asyncio.run's
    # executor shutdown would then wait on it forever, past pytest's timeout
    # (#898). Closing the write end gives readline EOF, which frees any reader.
    stopped = False
    try:
        reader = next(
            (t for t in threading.enumerate() if t.name == "cli-stdin-reader"), None
        )
        assert reader is not None, "no cli-stdin-reader thread"
        assert reader.daemon, "the stdin reader must not hold the process open"

        # A line still flows through the thread to the read loop, which passes
        # it to on_message exactly as before.
        received: list[str] = []

        class Recorder:
            @hookimpl
            async def on_message(self, channel, sender, text, trace_id) -> None:
                received.append(text)

        pm.register(Recorder(), name="recorder")
        os.write(write_end, b"hello\n")
        for _ in range(40):
            if received:
                break
            await asyncio.sleep(0.05)
        assert received == ["hello"], received

        # on_stop must be prompt even though the thread is blocked in readline:
        # the write end is still open here, so the reader has no EOF to see.
        # Prompt means it cancels the read task itself, well inside the bound;
        # wait_for's own cancellation at the bound would also end the task, so
        # finishing under the bound alone would not prove on_stop cancelled it.
        read_task = plugin._task
        assert read_task is not None, "on_start did not start the read loop"
        started = time.monotonic()
        await asyncio.wait_for(plugin.on_stop(), timeout=2)
        elapsed = time.monotonic() - started
        stopped = True
        assert read_task.done(), "on_stop returned with the read task still running"
        assert elapsed < 1.0, f"on_stop took {elapsed:.2f}s with the reader blocked"
    finally:
        # Unblock the reader first, so neither on_stop nor loop teardown can
        # wait on it.
        os.close(write_end)
        try:
            # On a failure path the body never stopped the plugin; stop it now,
            # bounded, so its read task does not outlive the test. The body has
            # already failed here, so a cleanup failure is logged rather than
            # raised: raising would replace the assertion that says what broke.
            if not stopped:
                try:
                    await asyncio.wait_for(plugin.on_stop(), timeout=2)
                except Exception:
                    logger.warning(
                        "CLIPlugin.on_stop failed during cleanup after the test "
                        "body failed; the body's failure is the one reported",
                        exc_info=True,
                    )
        finally:
            # Close the read end whatever cleanup did.
            stdin.close()


# ---------------------------------------------------------------------------
# The acceptance case: a real process, an open pipe, SIGINT
# ---------------------------------------------------------------------------


class _Child:
    """A daemon subprocess with its merged output collected on a thread."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self.proc = proc
        self.lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._lock:
                self.lines.append(line)

    @property
    def output(self) -> str:
        with self._lock:
            return "".join(self.lines)

    def wait_for_text(self, text: str, deadline: float) -> bool:
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            if text in self.output:
                return True
            if self.proc.poll() is not None:
                self._reader.join(timeout=1.0)
                return text in self.output
            time.sleep(0.05)
        return False

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=5)
        self._reader.join(timeout=2)


def _write_config(tmp_path: Path) -> Path:
    """An offline config with one cli channel and a 1 s graceful bound."""
    config = {
        "llm": {
            "providers": {"main": {"base_url": "http://127.0.0.1:9/v1"}},
            "profiles": {"main": {"model": "m"}},
        },
        "daemon": {
            "session_db": str(tmp_path / "sessions.db"),
            "shutdown_timeout": 1,
            "exit_timeout": 2,
        },
        "agent": {"system_prompt": "test"},
        "channels": {"cli:local": {"register_prompt": "a terminal"}},
        "plugins": {"disabled": MINIMAL_DISABLED},
    }
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


# The child's own deadlines (READY_DEADLINE + EXIT_DEADLINE, 18 s) exceed the
# global 15 s timeout; this test's bound sits above them so a slow child ends in
# the diagnostic pytest.fail and child.kill(), not pytest-timeout's os._exit
# (#898).
@pytest.mark.timeout(30)
def test_sigint_at_an_open_pipe_exits_0_without_the_watchdog(tmp_path):
    """#759: SIGINT with stdin an open pipe exits 0 within the bound.

    The pipe is never written to and never closed, so the reader is blocked
    in ``readline`` when the signal arrives — exactly the TTY case.
    """
    config = _write_config(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-P", "-u", str(CHILD), str(config)],
        cwd=tmp_path,
        stdin=subprocess.PIPE,  # open, and never written to
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    child = _Child(proc)
    try:
        if not child.wait_for_text("READY", READY_DEADLINE):
            pytest.fail(
                f"child never printed READY (exit status {proc.poll()}); output:\n{child.output}"
            )
        proc.send_signal(signal.SIGINT)
        try:
            status = proc.wait(timeout=EXIT_DEADLINE)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"still alive {EXIT_DEADLINE}s after SIGINT with stdin open "
                f"(the blocked reader is holding the process?); "
                f"output:\n{child.output}"
            )
        child._reader.join(timeout=2)
        output = child.output

        assert status == 0, f"expected exit status 0, got {status}; output:\n{output}"
        # The watchdog only speaks when it fires; a leaked reader would name
        # itself here.
        assert "daemon.exit_timeout" not in output, output
        assert "ThreadPoolExecutor" not in output, output
    finally:
        child.kill()
