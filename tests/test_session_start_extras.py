"""The session-start hook installs every extra the full suite needs (#699).

Claude Code on the web runs `.claude/hooks/session-start.sh` once per session
and nothing else installs dependencies there, so a subset of extras shows up as
test directories failing to import — which reads as a broken tree rather than a
missing extra. CLAUDE.md's "everything" line is the reference the hook has to
match, and this pair drifts whenever a workspace package gains an extra and
nobody remembers the hook.

These tests read both files as text. They cannot prove `uv sync` succeeds (the
extra names themselves are checked against pyproject below), but they pin the
thing that fails silently: the hook quietly installing less than the docs say.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / ".claude" / "hooks" / "session-start.sh"
CLAUDE_MD = ROOT / "CLAUDE.md"


def _extras(line: str) -> list[str]:
    """The extra names in a `uv sync --extra a --extra b` line, in order."""
    return re.findall(r"--extra\s+([A-Za-z0-9_-]+)", line)


def _sync_lines(text: str) -> list[str]:
    return [
        line for line in text.splitlines()
        if re.match(r"^\s*uv sync --extra", line)
    ]


def test_the_hook_installs_every_extra_the_docs_call_for() -> None:
    """The hook's `uv sync` line carries the same extras, in the same order."""
    hook_lines = _sync_lines(HOOK.read_text())
    assert len(hook_lines) == 1, (
        f"{HOOK.relative_to(ROOT)} has {len(hook_lines)} `uv sync --extra` lines; "
        "one, matching CLAUDE.md, is what the tests below can check"
    )
    # CLAUDE.md names two lines: the inline one in prose and the "everything"
    # block. The block is the one carrying the `# everything` comment.
    everything = [
        line for line in _sync_lines(CLAUDE_MD.read_text())
        if "# everything" in line
    ]
    assert len(everything) == 1, (
        "CLAUDE.md should hold exactly one `uv sync --extra ...  # everything` "
        "line for the hook to match; rewording it means updating this test too"
    )
    assert _extras(hook_lines[0]) == _extras(everything[0]), (
        "the session-start hook and CLAUDE.md's `# everything` line disagree "
        "about which extras the full suite needs; a test directory whose extra "
        "is named in only one of them fails to import in a web session (#699)"
    )


def test_every_named_extra_exists_in_pyproject() -> None:
    """A typo'd extra would make `uv sync` fail on the first session start."""
    declared = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        ["optional-dependencies"]
    )
    named = set(_extras(_sync_lines(HOOK.read_text())[0]))
    assert named <= declared, (
        f"the session-start hook installs {sorted(named - declared)}, which "
        f"pyproject.toml does not declare as extras; declared: {sorted(declared)}"
    )
