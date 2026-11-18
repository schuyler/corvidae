"""The kernel/cognition package boundary (plans/kernel-split.md, Part 2 §3).

A static check that reads source text, so it runs in the kernel-only CI job
without the cognition extra installed: no module under ``corvidae/`` imports
``corvidae_cognition`` (PRD principle 1).

What plugins import from the kernel is not checked here. Before a formal
release there is no declared plugin API; a plugin that imports a kernel
module takes the risk that it changes (#1026).
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNEL = REPO / "corvidae"


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
    """Nothing under corvidae/ depends on the cognition package."""
    offenders = [
        f"{path.relative_to(REPO)}:{line}: {module}"
        for path in sorted(KERNEL.rglob("*.py"))
        for line, module in _imported_modules(path)
        if module == "corvidae_cognition" or module.startswith("corvidae_cognition.")
    ]
    assert offenders == [], "kernel imports cognition:\n" + "\n".join(offenders)
