"""Tests for the restart pre-flight: it refuses what the daemon refuses at boot.

Each test runs the module as a subprocess, exactly as ``run_preflight``
does, and asserts the exit code and the whole of stderr.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Minimal llm section: Runtime.start requires a default chat profile.
MINIMAL_LLM = """\
llm:
  providers:
    main:
      base_url: http://localhost:8080/v1
  profiles:
    main:
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

# The child that records whether the pre-flight closed what it built (#563).
CLEANUP_CHILD = Path(__file__).parent.parent / "preflight_cleanup_child.py"

# The child that watches the failed-load cleanup (#984).
FAILED_LOAD_CHILD = Path(__file__).parent.parent / "preflight_failed_load_child.py"


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
        # The test's own deadline: subprocess.run kills the child on expiry,
        # so a regression fails here instead of outliving the pytest timeout.
        timeout=LEAK_DEADLINE,
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


def test_a_leak_before_a_failed_preflight_is_bounded_and_named(tmp_path: Path) -> None:
    """#757: a plugin that opens a connection and then raises cannot hold it.

    The failure path used to return 1 without arming the exit bound, so the
    leaked connection's non-daemon thread kept the interpreter alive. It now
    keeps its own failure line, names the survivor, and exits within
    ``daemon.exit_timeout``.
    """
    path = write_config(tmp_path, "daemon:\n  exit_timeout: 0.5\n")

    # subprocess.run kills the child when LEAK_DEADLINE expires.
    result = subprocess.run(
        [sys.executable, "-P", str(LEAK_CHILD), str(path), "raise"],
        capture_output=True,
        text=True,
        timeout=LEAK_DEADLINE,
    )

    assert result.returncode == 1, result.stderr
    # The failure's own line survives.
    assert "pre-flight failed:" in result.stderr, result.stderr
    assert "plugin failed after opening a connection" in result.stderr, result.stderr
    # And the leaked thread is named, with the remedy.
    assert "still running after the plugins were stopped" in result.stderr, result.stderr
    assert "daemon.exit_timeout" in result.stderr, result.stderr
    assert "pre-flight OK" not in result.stderr, result.stderr


def test_valid_minimal_config_passes(tmp_path: Path) -> None:
    """A config defining llm.main exits 0 with exactly 'pre-flight OK'."""
    result = run_preflight_cli(write_config(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stderr == "pre-flight OK\n"


def test_the_preflight_closes_what_it_built(tmp_path: Path) -> None:
    """The pre-flight broadcasts on_stop before it returns (#563).

    ``_cleanup_failed_start`` is the only thing that closes a pre-flight's
    plugin manager, and removing its call made no test fail: the leak tests
    cover the exit bound, which ``arm_exit_bound`` arms on every path. The
    child records the broadcast through the runtime the pre-flight builds.
    """
    # PYTHONPATH this tree first: an editable install points at the main
    # checkout, and the child would then test that tree's preflight, not the
    # one under test. The mutation that removes the call proves the difference.
    repo = CLEANUP_CHILD.parent.parent
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(repo)] + [str(p) for p in sorted((repo / "packages").glob("*"))]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-P", str(CLEANUP_CHILD), str(write_config(tmp_path))],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "exit 0" in result.stdout, (result.stdout, result.stderr)
    assert "on_stop broadcast" in result.stdout, (
        "the pre-flight returned without closing its plugins: " + result.stdout
    )


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


# The child that reports a LoadReport from the plugin load (#1107).
REPORT_CHILD = Path(__file__).parent.parent / "preflight_report_child.py"


def run_report_child(
    config_path: Path, fallbacks: str, skipped: str
) -> subprocess.CompletedProcess[str]:
    """Run the pre-flight with a load that reports the given fallbacks and skips."""
    return subprocess.run(
        [sys.executable, "-P", str(REPORT_CHILD), str(config_path), fallbacks, skipped],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "fallbacks, skipped, needles",
    [
        # Skipped alone: names the plugin and says it was skipped.
        ("{}", '["rss"]', ("rss", "skipped")),
        # Fallback alone: names slot and target and says bundled was used.
        ('{"llm": "my_llm"}', "[]", ('replacing "llm"', "my_llm", "bundled used")),
    ],
)
def test_preflight_fails_when_a_plugin_was_skipped_or_fell_back(
    tmp_path: Path, fallbacks: str, skipped: str, needles: tuple[str, ...]
) -> None:
    """R14: a skipped plugin, or a fallen-back replacement, each alone fails the pre-flight.

    The stderr line names each plugin and the remedy; logging is disabled in
    the pre-flight, so the failure must be written to stderr directly.
    """
    path = write_config(tmp_path)
    result = run_report_child(path, fallbacks, skipped)

    assert result.returncode == 1, result.stderr
    for needle in needles:
        assert needle in result.stderr, (needle, result.stderr)
    assert "fix or reinstall" in result.stderr, result.stderr
    assert "pre-flight OK" not in result.stderr, result.stderr


def test_preflight_passes_when_the_load_report_is_clean(tmp_path: Path) -> None:
    """R14 guard: an empty report does not fail the pre-flight."""
    result = run_report_child(write_config(tmp_path), "{}", "[]")

    assert result.returncode == 0, result.stderr
    assert "pre-flight OK" in result.stderr, result.stderr

def test_the_preflight_closes_what_it_built_when_the_load_fails(tmp_path: Path) -> None:
    """A failed plugin load still closes what was built (#984).

    The #959 review found cleanup covered only on the path where the plugins
    load. That is the path a broken config does not take, and closing what a
    failed boot built is what the pre-flight exists for. ``persistence``
    disabled while another plugin depends on it fails inside
    ``load_plugins`` after the manager exists.
    """
    repo = FAILED_LOAD_CHILD.parent.parent
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(repo)] + [str(p) for p in sorted((repo / "packages").glob("*"))]
        ),
    }
    path = write_config(tmp_path, "plugins:\n  disabled: [persistence]\n")

    result = subprocess.run(
        [sys.executable, "-P", str(FAILED_LOAD_CHILD), str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert "exit 1" in result.stdout, (result.stdout, result.stderr)
    assert "cleanup ran" in result.stdout, (
        "the failed load did not reach the cleanup: " + result.stdout
    )
    assert "on_stop broadcast" in result.stdout, (
        "the failed load closed nothing: " + result.stdout
    )


def test_the_survivor_sentence_says_the_bound_is_disabled():
    """#854: with exit_timeout off, the line must not claim the process exits."""
    from corvidae_restart.preflight import _exit_sentence

    for disabled in (None, 0):
        line = _exit_sentence(disabled)
        assert "no exit bound" in line
        assert "preflight timeout" in line
        assert "exits within" not in line


def test_the_survivor_sentence_names_the_bound_when_there_is_one():
    from corvidae_restart.preflight import _exit_sentence

    assert "exits within daemon.exit_timeout (2.5 s)" in _exit_sentence(2.5)


def test_a_mismatched_corvidae_location_is_refused():
    """#1008: a venv whose corvidae resolves elsewhere must not boot."""
    from corvidae_restart.preflight import check_corvidae_location

    import corvidae

    here = str(__import__("pathlib").Path(corvidae.__file__).resolve().parent)

    assert check_corvidae_location(here) is None
    refusal = check_corvidae_location("/somewhere/else/deployed")
    assert refusal is not None
    assert here in refusal
    assert "/somewhere/else/deployed" in refusal
    assert "#1008" in refusal
    # No expectation given: nothing to compare, which is the old behaviour.
    assert check_corvidae_location(None) is None
