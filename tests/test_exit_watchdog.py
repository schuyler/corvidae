"""RED tests for the exit watchdog (#746): bounded process exit after SIGTERM.

Each test runs a real daemon in a subprocess (tests/exit_watchdog_child.py)
on an offline config with no channels, sends SIGTERM once the child prints
READY, and checks how and when it exits. Every wait has a deadline well under
the 15 s pytest timeout, and the child is always killed and reaped, so a hang
(the failure mode without the feature) is reported as a test failure with the
child's output, never as an orphaned process.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX signals and process semantics"
)

CHILD = Path(__file__).parent / "exit_watchdog_child.py"

# The minimal profile's disabled list: the cognition set, offline-safe.
MINIMAL_DISABLED = [
    "memory", "memory_tools", "funnel", "appraisal", "critique", "outcome_log",
    "gates", "silence", "skills", "heartbeat", "goals",
]

# How long the child may take to boot and print READY.
READY_DEADLINE = 8.0
# Margin after SIGTERM: shutdown_timeout (3) + exit_timeout (0.5) is the
# worst case, but a clean on_stop returns at once, so the watchdog fires at
# about 0.5 s. 3.0 s is far above that yet far below an unbounded hang.
EXIT_DEADLINE = 3.0


class _Child:
    """A daemon subprocess with its merged output collected on a thread."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self.proc = proc
        self.lines: list[str] = []
        self._lock = threading.Lock()
        # Reader thread: drains stdout so the pipe never fills and lines
        # can be waited on with a deadline.
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        """Collect output lines until the pipe closes."""
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._lock:
                self.lines.append(line)

    @property
    def output(self) -> str:
        """Everything the child has printed so far."""
        with self._lock:
            return "".join(self.lines)

    def wait_for_text(self, text: str, deadline: float) -> bool:
        """Poll the output for ``text``; False on timeout or child exit."""
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            if text in self.output:
                return True
            if self.proc.poll() is not None:
                # Child exited; give the reader a moment to drain, then stop.
                self._reader.join(timeout=1.0)
                return text in self.output
            time.sleep(0.05)
        return False

    def kill(self) -> None:
        """Kill and reap the child if it is still running."""
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=5)
        self._reader.join(timeout=2)


def _write_config(tmp_path: Path, daemon_extra: dict) -> Path:
    """Write an offline, channel-less config with every path under tmp_path."""
    daemon = {"session_db": str(tmp_path / "sessions.db")}
    daemon.update(daemon_extra)
    config = {
        "llm": {"main": {"base_url": "http://127.0.0.1:9/v1", "model": "m"}},
        "daemon": daemon,
        "agent": {"system_prompt": "test"},
        "plugins": {"disabled": MINIMAL_DISABLED},
    }
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _spawn(tmp_path: Path, config: Path, *flags: str) -> _Child:
    """Start the child daemon in tmp_path with stdin closed."""
    proc = subprocess.Popen(
        [sys.executable, "-u", str(CHILD), str(config), *flags],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return _Child(proc)


def _boot(child: _Child) -> None:
    """Wait for READY or fail with the child's output."""
    if not child.wait_for_text("READY", READY_DEADLINE):
        pytest.fail(
            f"child never printed READY (exit status {child.proc.poll()}); "
            f"output:\n{child.output}"
        )


@pytest.mark.parametrize(
    "raise_on_stop",
    [False, True],
    ids=["clean_on_stop", "raising_on_stop"],
)
def test_leaked_thread_forces_exit_1_with_warning(tmp_path, raise_on_stop):
    """R1+R2+R3: a leaked non-daemon thread cannot wedge exit.

    SIGTERM -> exit status 1 within the bound, and a WARNING names the
    surviving thread and daemon.exit_timeout. Also holds when a plugin's
    on_stop raises and stop() propagates the exception.
    """
    config = _write_config(tmp_path, {"exit_timeout": 0.5})
    flags = ["--leak"] + (["--raise-on-stop"] if raise_on_stop else [])
    child = _spawn(tmp_path, config, *flags)
    try:
        _boot(child)
        child.proc.send_signal(signal.SIGTERM)
        started = time.monotonic()
        try:
            status = child.proc.wait(timeout=EXIT_DEADLINE)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"daemon still alive {EXIT_DEADLINE}s after SIGTERM with a "
                f"leaked thread (exit watchdog missing?); output:\n{child.output}"
            )
        elapsed = time.monotonic() - started
        child._reader.join(timeout=2)
        output = child.output

        # R3: abnormal status.
        assert status == 1, f"expected exit status 1, got {status}; output:\n{output}"
        # R1: well inside the bound plus margin.
        assert elapsed < EXIT_DEADLINE
        # R2: the line names the leaked thread and the config key.
        assert "_connection_worker_thread" in output, output
        assert "daemon.exit_timeout" in output, output
    finally:
        child.kill()


def test_clean_shutdown_exits_0_without_watchdog(tmp_path):
    """R4: no leak, default exit_timeout -> status 0, promptly, no watchdog line."""
    config = _write_config(tmp_path, {})
    child = _spawn(tmp_path, config)
    try:
        _boot(child)
        child.proc.send_signal(signal.SIGTERM)
        try:
            status = child.proc.wait(timeout=EXIT_DEADLINE)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"clean daemon still alive {EXIT_DEADLINE}s after SIGTERM; "
                f"output:\n{child.output}"
            )
        child._reader.join(timeout=2)
        output = child.output

        assert status == 0, f"expected exit status 0, got {status}; output:\n{output}"
        # The watchdog only speaks when it fires.
        assert "daemon.exit_timeout" not in output, output
        assert "forcing exit" not in output, output
    finally:
        child.kill()


@pytest.mark.parametrize("disabled_value", [0, None], ids=["zero", "null"])
def test_exit_timeout_disabled_leaves_leaking_process_alive(tmp_path, disabled_value):
    """R6: exit_timeout 0 or null disables the watchdog.

    With a leaked thread the process stays alive well past the 0.5 s that
    would have fired a watchdog (the test then kills it itself).
    """
    config = _write_config(tmp_path, {"exit_timeout": disabled_value})
    child = _spawn(tmp_path, config, "--leak")
    try:
        _boot(child)
        child.proc.send_signal(signal.SIGTERM)
        # Wait until shutdown has begun, so the clock below starts after it.
        if not child.wait_for_text("shutdown signal received, stopping", 5.0):
            pytest.fail(f"shutdown never began; output:\n{child.output}")
        # Clearly past 0.5 s (and past graceful shutdown, which is instant).
        time.sleep(2.0)
        assert child.proc.poll() is None, (
            f"process exited (status {child.proc.poll()}) although "
            f"exit_timeout is {disabled_value!r}; output:\n{child.output}"
        )
    finally:
        # SIGKILL and reap: the leaked thread would hold it forever.
        child.kill()
