"""Tests for workflow/display.py — human-readable progress display."""

import io
import sys
from unittest.mock import patch

import pytest

from workflow.display import (
    Display,
    display_stage_start,
    display_stage_complete,
    display_stage_skip,
    display_error,
    display_workflow_complete,
    display_workflow_blocked,
)


class TestDisplay:
    """Tests for the Display class."""

    def test_display_stage_start_prints_arrow(self, capsys):
        """display_stage_start prints '▸ stage_name'."""
        d = Display()
        d.stage_start("implement")
        captured = capsys.readouterr()
        assert "▸ implement" in captured.out

    def test_display_stage_complete_proceed(self, capsys):
        """display_stage_complete prints checkmark for proceed."""
        d = Display()
        d.stage_complete("implement", "proceed")
        captured = capsys.readouterr()
        assert "✓ implement" in captured.out
        assert "proceed" in captured.out

    def test_display_stage_complete_blocked(self, capsys):
        """display_stage_complete prints cross for blocked."""
        d = Display()
        d.stage_complete("submit_pr", "blocked")
        captured = capsys.readouterr()
        assert "✗ submit_pr" in captured.out
        assert "blocked" in captured.out

    def test_display_stage_skip(self, capsys):
        """display_stage_skip prints skip marker."""
        d = Display()
        d.stage_skip("refactor")
        captured = capsys.readouterr()
        assert "⊘ refactor" in captured.out
        assert "skipped" in captured.out

    def test_display_error(self, capsys):
        """display_error prints error with marker."""
        d = Display()
        d.error("Something went wrong")
        captured = capsys.readouterr()
        assert "⚠ Something went wrong" in captured.out

    def test_display_workflow_complete(self, capsys):
        """display_workflow_complete prints completion message."""
        d = Display()
        d.workflow_complete()
        captured = capsys.readouterr()
        assert "done" in captured.out.lower()

    def test_display_workflow_blocked(self, capsys):
        """display_workflow_blocked prints blocked summary."""
        d = Display()
        d.workflow_blocked("submit_pr", "Network error")
        captured = capsys.readouterr()
        assert "blocked" in captured.out.lower()
        assert "submit_pr" in captured.out
        assert "Network error" in captured.out


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_display_stage_start_function(self, capsys):
        """Module-level display_stage_start works."""
        display_stage_start("implement")
        captured = capsys.readouterr()
        assert "▸ implement" in captured.out

    def test_display_stage_complete_function(self, capsys):
        """Module-level display_stage_complete works."""
        display_stage_complete("implement", "proceed")
        captured = capsys.readouterr()
        assert "✓ implement" in captured.out

    def test_display_stage_skip_function(self, capsys):
        """Module-level display_stage_skip works."""
        display_stage_skip("refactor")
        captured = capsys.readouterr()
        assert "⊘ refactor" in captured.out

    def test_display_error_function(self, capsys):
        """Module-level display_error works."""
        display_error("fail")
        captured = capsys.readouterr()
        assert "⚠ fail" in captured.out

    def test_display_workflow_complete_function(self, capsys):
        """Module-level display_workflow_complete works."""
        display_workflow_complete()
        captured = capsys.readouterr()
        assert "done" in captured.out.lower()

    def test_display_workflow_blocked_function(self, capsys):
        """Module-level display_workflow_blocked works."""
        display_workflow_blocked("stage", "reason")
        captured = capsys.readouterr()
        assert "stage" in captured.out
        assert "reason" in captured.out
