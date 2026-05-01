"""Tests for workflow.__main__ — CLI entry point."""

import subprocess
import sys


def test_cli_help():
    """CLI --help exits with 0."""
    result = subprocess.run(
        [sys.executable, "-m", "workflow", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "workflow" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_cli_implement_help():
    """implement --help shows required arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "workflow", "implement", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--issue" in result.stdout


def test_cli_no_args_shows_help():
    """Running with no arguments shows help."""
    result = subprocess.run(
        [sys.executable, "-m", "workflow"],
        capture_output=True, text=True,
    )
    # Should print help and exit cleanly
    assert result.returncode == 0 or "usage" in result.stdout.lower()
