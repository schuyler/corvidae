"""Tests that listable system-prompt fragments contain only prompt text.

Files in ``prompts/`` that ``docs/prompt-guide.md`` tells operators to add to a
channel's ``system_prompt`` list are read whole by ``resolve_system_prompt``,
so any operator-facing header in them would be sent to the model (#207).
"""

from pathlib import Path

import pytest

from corvidae.channel import resolve_system_prompt

# Repository root, used as the base directory for relative prompt paths.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Fragments that operators are told to list in a channel's system_prompt.
FRAGMENTS = [
    "prompts/chat.md",
    "prompts/signal_dm.md",
    "prompts/discord_group.md",
    "prompts/memory_calibration.md",
]


@pytest.mark.parametrize("path", FRAGMENTS)
def test_fragment_has_no_documentation_header(path: str) -> None:
    """A resolved fragment starts with prompt text and has no `---` rule."""
    text = resolve_system_prompt([path], REPO_ROOT)
    # No Markdown title ahead of the instructions.
    assert not text.lstrip().startswith("#"), f"{path} starts with a heading"
    # No horizontal rule separating a doc header from the prompt.
    assert "---" not in text.splitlines(), f"{path} contains a bare --- line"


def test_memory_calibration_sends_only_instructions() -> None:
    """R1: the calibration fragment begins at the instructions themselves."""
    text = resolve_system_prompt(["prompts/memory_calibration.md"], REPO_ROOT)
    assert text.startswith("Your long-term memory works by retrieval")
    assert "Epistemic memory calibration" not in text
    assert "auto-injected" not in text
