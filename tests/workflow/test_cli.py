"""Tests for workflow.__main__ — CLI entry point."""

import logging
import os
import subprocess
import sys
import tempfile

import pytest

from workflow.__main__ import configure_workflow_logging


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


class TestConfigureWorkflowLogging:
    """Tests for configure_workflow_logging — dual-handler setup."""

    def test_creates_log_file(self, tmp_path):
        """configure_workflow_logging creates a log file."""
        log_file = str(tmp_path / "workflow.log")
        configure_workflow_logging(log_file=log_file)
        # Write a log message to verify the file gets created
        logger = logging.getLogger("workflow")
        logger.info("test message")
        # Flush handlers
        for handler in logger.handlers:
            handler.flush()
        assert os.path.exists(log_file)

    def test_stdout_handler_is_human_readable(self, capsys):
        """STDOUT handler outputs human-readable format (no timestamps)."""
        configure_workflow_logging(log_file="/dev/null")
        logger = logging.getLogger("workflow")
        logger.info("test stage message")
        # Flush handlers
        for handler in logger.handlers:
            handler.flush()
        captured = capsys.readouterr()
        # Should NOT contain timestamp format like "2026-"
        assert "2026-" not in captured.out
        # Should NOT contain level name like "INFO"
        lines = captured.out.strip().split("\n")
        # The output should be clean — just the message or nothing
        # (the stdout handler only shows stage-level events, not all INFO)
        # For now, verify it doesn't dump raw log format to stdout
        for line in lines:
            if line:
                assert "workflow" not in line or "▸" in line or "✓" in line or "⚠" in line

    def test_file_handler_captures_debug(self, tmp_path):
        """File handler captures DEBUG-level messages."""
        log_file = str(tmp_path / "workflow.log")
        configure_workflow_logging(log_file=log_file)
        logger = logging.getLogger("workflow")
        logger.debug("debug detail message")
        # Flush handlers
        for handler in logger.handlers:
            handler.flush()
        with open(log_file) as f:
            content = f.read()
        assert "debug detail message" in content

    def test_file_handler_uses_structured_format(self, tmp_path):
        """File handler uses StructuredFormatter."""
        log_file = str(tmp_path / "workflow.log")
        configure_workflow_logging(log_file=log_file)
        logger = logging.getLogger("workflow")
        logger.info("structured test", extra={"tool_name": "read", "target_file": "file.py"})
        for handler in logger.handlers:
            handler.flush()
        with open(log_file) as f:
            content = f.read()
        # StructuredFormatter adds key=value pairs
        assert "tool_name='read'" in content
        assert "target_file='file.py'" in content

    def test_log_directory_created(self, tmp_path):
        """configure_workflow_logging creates the log directory if needed."""
        log_file = str(tmp_path / "logs" / "workflow.log")
        configure_workflow_logging(log_file=log_file)
        logger = logging.getLogger("workflow")
        logger.info("test")
        for handler in logger.handlers:
            handler.flush()
        assert os.path.isdir(tmp_path / "logs")
