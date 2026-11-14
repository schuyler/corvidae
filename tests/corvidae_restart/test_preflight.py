"""Tests for the restart pre-flight: it refuses what the daemon refuses at boot.

Each test runs the module as a subprocess, exactly as ``run_preflight``
does, and asserts the exit code and the whole of stderr.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Minimal llm section: Runtime.start requires a default chat profile.
MINIMAL_LLM = """\
llm:
  main:
    base_url: http://localhost:8080/v1
    model: test-model
"""

# The generic remedy carried by every "pre-flight failed:" line.
GENERIC_REMEDY = "fix the config or the plugin set it names, then retry the restart"


def run_preflight_cli(config_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the pre-flight module as the restart tool does and capture output."""
    return subprocess.run(
        [sys.executable, "-P", "-m", "corvidae_restart.preflight", "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def write_config(tmp_path: Path, extra: str = "") -> Path:
    """Write a config with a valid llm section plus ``extra`` YAML."""
    path = tmp_path / "agent.yaml"
    path.write_text(MINIMAL_LLM + extra, encoding="utf-8")
    return path


# How long the leak case may take: daemon.exit_timeout (0.5) plus a boot,
# far below the 120 s preflight_timeout the restart tool would otherwise wait.
LEAK_DEADLINE = 10.0

# The child that injects a leaking plugin into the real pre-flight.
LEAK_CHILD = Path(__file__).parent.parent / "preflight_leak_child.py"


def test_a_leak_in_the_preflight_is_bounded_and_named(tmp_path: Path) -> None:
    """#757: a plugin that leaks a thread during the pre-flight cannot hold it.

    The pre-flight used to sit alive with the leaked connection's non-daemon
    worker thread until the restart tool's ``preflight_timeout_seconds``
    (default 120 s) killed the process group. It now names the survivor,
    refuses the restart, and exits within ``daemon.exit_timeout``.
    """
    path = write_config(tmp_path, "daemon:\n  exit_timeout: 0.5\n")

    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-P", str(LEAK_CHILD), str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 1, result.stderr
    assert elapsed < LEAK_DEADLINE, f"took {elapsed:.1f}s; stderr:\n{result.stderr}"
    # It says what is holding the process, not just that something is.
    assert "still running after the plugins were stopped" in result.stderr, result.stderr
    assert "daemon.exit_timeout" in result.stderr, result.stderr
    assert "on_init" in result.stderr and "on_stop" in result.stderr, result.stderr
    # And it does not claim success on the way out.
    assert "pre-flight OK" not in result.stderr, result.stderr


def test_valid_minimal_config_passes(tmp_path: Path) -> None:
    """A config defining llm.main exits 0 with exactly 'pre-flight OK'."""
    result = run_preflight_cli(write_config(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stderr == "pre-flight OK\n"


def test_unknown_channel_key_names_key_and_accepted_keys(tmp_path: Path) -> None:
    """An unknown key under channels.<id> is one line naming it and the accepted keys."""
    path = write_config(tmp_path, "channels:\n  'irc:#x':\n    bogus: 1\n")

    result = run_preflight_cli(path)

    assert result.returncode == 1
    assert len(result.stderr.splitlines()) == 1, result.stderr
    assert "irc:#x" in result.stderr
    assert "bogus" in result.stderr
    assert "unknown key" in result.stderr
    assert "accepted:" in result.stderr
    assert f"— check {path}" in result.stderr


def test_restart_on_init_error_is_relayed(tmp_path: Path) -> None:
    """The restart plugin's on_init ValueError is one line naming the bad key."""
    path = write_config(tmp_path, "restart:\n  max_restarts: x\n")

    result = run_preflight_cli(path)

    assert result.returncode == 1
    assert len(result.stderr.splitlines()) == 1, result.stderr
    assert "restart config 'max_restarts'" in result.stderr
    assert f"— check {path}" in result.stderr


def test_disabled_dependency_plugin_fails_with_generic_remedy(tmp_path: Path) -> None:
    """Disabling a plugin others depend on fails with the generic remedy line."""
    path = write_config(tmp_path, "plugins:\n  disabled: [persistence]\n")

    result = run_preflight_cli(path)

    assert result.returncode == 1
    assert len(result.stderr.splitlines()) == 1, result.stderr
    assert result.stderr.startswith("pre-flight failed:")
    assert "persistence" in result.stderr
    assert "(RuntimeError)" in result.stderr
    assert GENERIC_REMEDY in result.stderr
    assert f"— check {path}" in result.stderr


def test_unconfigurable_logging_level_fails_with_generic_remedy(tmp_path: Path) -> None:
    """A logging level the daemon cannot configure names the level and the remedy."""
    path = write_config(tmp_path, "logging:\n  level: verbose\n")

    result = run_preflight_cli(path)

    assert result.returncode == 1
    assert len(result.stderr.splitlines()) == 1, result.stderr
    assert result.stderr.startswith("pre-flight failed:")
    assert "verbose" in result.stderr.lower()
    assert GENERIC_REMEDY in result.stderr
    assert f"— check {path}" in result.stderr


def test_missing_config_path_reports_config_error(tmp_path: Path) -> None:
    """A missing config file is one 'Config error:' line naming the path."""
    path = tmp_path / "absent.yaml"

    result = run_preflight_cli(path)

    assert result.returncode == 1
    assert len(result.stderr.splitlines()) == 1, result.stderr
    assert "Config error:" in result.stderr
    assert f"— check {path}" in result.stderr
