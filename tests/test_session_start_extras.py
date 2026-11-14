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
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def _extras(line: str) -> list[str]:
    """The extra names in a `uv sync --extra a --extra b` line, in order."""
    return re.findall(r"--extra\s+([A-Za-z0-9_-]+)", line)


def _sync_lines(text: str) -> list[str]:
    return [
        line for line in text.splitlines()
        if re.match(r"^\s*uv sync --extra", line)
    ]


def _ci_full_job_extras() -> set[str]:
    """The extras CI's full job installs: the `uv sync` line with the most.

    The kernel, cognition-fts and web-smoke jobs each install a strict subset,
    so the largest line is the full one. CI is the real reference — it is what
    actually runs every suite — so a name missing here is a bug however the
    docs read.
    """
    lines = [
        line for line in WORKFLOW.read_text().splitlines()
        if re.search(r"\buv sync\b.*--extra", line)
    ]
    return set(_extras(max(lines, key=lambda line: len(_extras(line)))))


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


def test_the_hook_installs_what_ci_installs() -> None:
    """CLAUDE.md can be wrong too: `rss` was missing from both while CI's full
    job needed it for the rss-reader tests, which is the same drift one level up.

    CI's full job is the definition of "the full suite", so the hook follows it
    rather than the docs.
    """
    hook_extras = set(_extras(_sync_lines(HOOK.read_text())[0]))
    ci_extras = _ci_full_job_extras()
    assert ci_extras, "no `uv sync` line with extras found in CI's workflow"
    assert hook_extras == ci_extras, (
        "the session-start hook and CI's full job disagree: hook has "
        f"{sorted(hook_extras - ci_extras)} too many and is missing "
        f"{sorted(ci_extras - hook_extras)}"
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
