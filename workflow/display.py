"""Human-readable progress display for workflow stages.

Provides simple print-based output for stage lifecycle events.
STDOUT shows concise one-line-per-stage progress using unicode symbols.
Machine-formatted logging goes to the log file instead.

This module keeps display concerns separate from engine logic.
"""

from __future__ import annotations

from typing import TextIO
import sys


class Display:
    """Displays human-readable workflow progress to STDOUT.

    Each stage lifecycle event gets a single line:
    - Start:   ▸ stage_name
    - Done:    ✓ stage_name (outcome)
    - Skipped: ⊘ stage_name (skipped)
    - Error:   ⚠ message
    """

    def __init__(self, out: TextIO | None = None) -> None:
        """Initialize display with an output stream.

        Args:
            out: Output stream. Defaults to sys.stdout.
        """
        self._out = out or sys.stdout

    def _print(self, message: str) -> None:
        """Print a message to the output stream."""
        print(message, file=self._out, flush=True)

    def stage_start(self, name: str) -> None:
        """Display stage start indicator."""
        self._print(f"▸ {name}")

    def stage_complete(self, name: str, outcome: str) -> None:
        """Display stage completion with outcome."""
        if outcome == "proceed":
            self._print(f"✓ {name} ({outcome})")
        else:
            self._print(f"✗ {name} ({outcome})")

    def stage_skip(self, name: str) -> None:
        """Display stage skip indicator."""
        self._print(f"⊘ {name} (skipped)")

    def error(self, message: str) -> None:
        """Display an error message."""
        self._print(f"⚠ {message}")

    def workflow_complete(self) -> None:
        """Display workflow completion message."""
        self._print("✓ Workflow done")

    def workflow_blocked(
        self,
        stage_name: str,
        reason: str,
        summary: str = "",
        branch: str = "",
    ) -> None:
        """Display workflow blocked summary with suggested next steps.

        Args:
            stage_name: The stage that failed.
            reason: Why the workflow stopped.
            summary: The LLM's summary from the failed stage.
            branch: Branch name if one was created (for manual cleanup).
        """
        self._print(f"\n{'─' * 60}")
        self._print(f"✗ Workflow blocked at stage: {stage_name}")
        if reason:
            self._print(f"  Reason: {reason}")
        if summary and summary != reason:
            # Show LLM summary if it differs from the reason
            self._print(f"  Summary: {summary}")
        if branch:
            self._print(f"  Branch: {branch}")
        self._print("  Suggested next steps:")
        self._print("    1. Check the log file for full details")
        if branch:
            self._print(f"    2. Switch to branch: git checkout {branch}")
            self._print("    3. Fix the issue and retry, or clean up:")
            self._print(f"       git checkout main && git branch -D {branch}")
        else:
            self._print("    2. Fix the issue and re-run the workflow")
        self._print(f"{'─' * 60}")


# Module-level convenience functions
# These use a fresh Display each call so they always use the
# current sys.stdout (important for test fixtures like capsys).


def display_stage_start(name: str) -> None:
    """Display stage start using the default display."""
    Display().stage_start(name)


def display_stage_complete(name: str, outcome: str) -> None:
    """Display stage completion using the default display."""
    Display().stage_complete(name, outcome)


def display_stage_skip(name: str) -> None:
    """Display stage skip using the default display."""
    Display().stage_skip(name)


def display_error(message: str) -> None:
    """Display error using the default display."""
    Display().error(message)


def display_workflow_complete() -> None:
    """Display workflow completion using the default display."""
    Display().workflow_complete()


def display_workflow_blocked(stage_name: str, reason: str) -> None:
    """Display workflow blocked using the default display."""
    Display().workflow_blocked(stage_name, reason)
