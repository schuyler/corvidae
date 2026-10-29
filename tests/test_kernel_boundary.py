"""The kernel/cognition package boundary (plans/kernel-split.md, Part 2 §3).

Two static checks that read source text, so they run in the kernel-only CI
job without the cognition extra installed:

1. No module under ``corvidae/`` imports ``corvidae_cognition``.
2. If ``packages/corvidae-cognition/`` exists, every kernel module it
   imports is on the allowlist below: the kernel's plugin API, documented
   in docs/plugin-guide.md under "What the kernel exports to plugins".
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
KERNEL = REPO / "corvidae"
COGNITION = REPO / "packages" / "corvidae-cognition" / "corvidae_cognition"

# Kernel modules a cognition module may import: Part 2 §2's table plus
# corvidae.outbound, the event types for the send hook (added by WP3),
# and corvidae.clock / corvidae.idle, the kernel clock and idle tracker (#165).
# Keep in step with docs/plugin-guide.md, "What the kernel exports to plugins".
ALLOWED_KERNEL_MODULES = {
    "corvidae.outbound",
    "corvidae.hooks",
    "corvidae.fork",
    "corvidae.attribution",
    "corvidae.tuning",
    "corvidae.context",
    "corvidae.task",
    "corvidae.tool",
    "corvidae.persistence",
    "corvidae.llm_plugin",
    "corvidae.clock",
    "corvidae.idle",
}


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    """Return (line, module) for every import in a file, function-local
    imports included, since a lazy import is still a dependency."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[tuple[int, str]] = []
    # Walk every node, not just the module body, to reach local imports.
    # Relative imports (level > 0) stay within one package and are skipped.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return found


def test_kernel_does_not_import_cognition() -> None:
    """Check 1: nothing under corvidae/ depends on the cognition package."""
    offenders = [
        f"{path.relative_to(REPO)}:{line}: {module}"
        for path in sorted(KERNEL.rglob("*.py"))
        for line, module in _imported_modules(path)
        if module == "corvidae_cognition" or module.startswith("corvidae_cognition.")
    ]
    assert offenders == [], "kernel imports cognition:\n" + "\n".join(offenders)


def test_cognition_imports_only_the_kernel_api() -> None:
    """Check 2: the cognition package imports only allowlisted kernel modules."""
    if not COGNITION.is_dir():
        pytest.skip("packages/corvidae-cognition/ does not exist yet")
    # Collect every import of a corvidae.* module that is not on the list.
    offenders = [
        f"{path.relative_to(REPO)}:{line}: {module}"
        for path in sorted(COGNITION.rglob("*.py"))
        for line, module in _imported_modules(path)
        if (module == "corvidae" or module.startswith("corvidae."))
        and module not in ALLOWED_KERNEL_MODULES
    ]
    assert offenders == [], (
        "cognition imports kernel modules outside the plugin API "
        "(add to the allowlist and docs/plugin-guide.md only deliberately):\n"
        + "\n".join(offenders)
    )
