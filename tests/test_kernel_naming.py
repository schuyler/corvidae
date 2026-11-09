"""Kernel naming guard (#16).

"Core" is reserved for the System Two context of the core/periphery work, so
no kernel identifier may use it to mean the kernel. This file is the one place
allowed to spell the retired names: the guard below skips it, and the negative
checks for the retired plugin class and entry point live here for that reason.
"""

import re
from importlib.metadata import entry_points
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Directories the guard scans (Python files only).
SCANNED_DIRS = [REPO / "corvidae", REPO / "tests"]

# Retired kernel-as-"core" identifiers: Core<Upper>..., core_tools, CORE_...
RETIRED_NAME = re.compile(r"\bCore[A-Z]\w*|\bcore_tools\b|\bCORE_[A-Z0-9_]+")

# The old class and entry-point names, assembled so this file's own source
# is the only place they appear literally (the guard skips this file anyway).
OLD_CLASS = "Core" + "ToolsPlugin"
OLD_ENTRY_POINT = "core" + "_tools"


def _python_files() -> list[Path]:
    """Every .py file under the scanned dirs, minus caches and this file."""
    this_file = Path(__file__).resolve()
    files: list[Path] = []
    for root in SCANNED_DIRS:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts or path.resolve() == this_file:
                continue
            files.append(path)
    return files


def test_no_retired_core_names_in_kernel_or_tests() -> None:
    """No Python file under corvidae/ or tests/ uses a retired name (R8)."""
    hits: list[str] = []
    # Collect every hit with its location so the failure names each site.
    for path in _python_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for match in RETIRED_NAME.finditer(line):
                hits.append(f"{path.relative_to(REPO)}:{lineno}: {match.group(0)}")
    assert not hits, (
        "retired kernel-as-'core' names found; rename them to 'kernel' or to "
        "the split tool plugins (#16):\n" + "\n".join(hits)
    )


def test_retired_tools_plugin_class_is_gone() -> None:
    """The combined tools plugin no longer exists, with no alias (R1)."""
    import corvidae.tools

    assert not hasattr(corvidae.tools, OLD_CLASS), (
        f"corvidae.tools.{OLD_CLASS} must be removed, not aliased; "
        "use ShellToolPlugin, FileToolsPlugin or WebToolsPlugin"
    )


def test_retired_tools_entry_point_is_gone() -> None:
    """No installed distribution registers the old entry point (R4)."""
    names = {ep.name for ep in entry_points(group="corvidae")}
    assert OLD_ENTRY_POINT not in names, (
        f"entry point {OLD_ENTRY_POINT!r} must be removed with no alias; "
        "if pyproject.toml no longer lists it, re-sync with `uv sync`"
    )


@pytest.mark.parametrize(
    "line, expected",
    [
        ("from x import " + OLD_CLASS, True),
        ("disabled: [" + OLD_ENTRY_POINT + "]", True),
        ("CORE" + "_LLM_STAGES = {}", True),
        ("KERNEL_LLM_STAGES = {}", False),
        ("core loop, scored", False),
    ],
)
def test_guard_pattern_matches_retired_names_only(line: str, expected: bool) -> None:
    """The guard regex catches each retired form and passes the new names,
    so the guard above can fail (R8)."""
    assert bool(RETIRED_NAME.search(line)) is expected
