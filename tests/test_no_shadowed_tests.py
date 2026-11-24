"""Guard: no test module may define two tests with the same name (#1243).

Python keeps the last definition of a name, so the earlier test is never
collected and nothing says so. #1159 shipped exactly that in
``tests/test_config_watcher.py``: one reload test ran, its twin never did.
Ruff's F811 would catch it, but lint is advisory until #455, so the failure
has to come from a test.
"""

import ast
from collections import Counter
from pathlib import Path

# Repository root: this file lives in <root>/tests/.
ROOT = Path(__file__).resolve().parent.parent


def _test_files() -> list[Path]:
    """Every test module in the kernel and in the plugin packages."""
    files = list((ROOT / "tests").rglob("*.py"))
    files += list(ROOT.glob("packages/*/tests/**/*.py"))
    return sorted(files)


def shadowed_tests(source: str) -> list[str]:
    """Names of ``test*`` functions defined more than once in one scope.

    A scope is the module body or one class body. Only direct children count:
    a definition inside an ``if`` block is a deliberate alternative, not a
    shadow, and non-``test`` helpers are ruff's business.
    """
    tree = ast.parse(source)
    scopes: list[ast.AST] = [tree] + [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    found: list[str] = []
    for scope in scopes:
        names = Counter(
            node.name
            for node in scope.body  # type: ignore[attr-defined]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test")
        )
        found += [name for name, count in names.items() if count > 1]
    return found


def test_the_scan_sees_a_shadowed_test():
    """The scan itself works: a seeded duplicate is reported, a clean file is not."""
    assert shadowed_tests("def test_a(): ...\ndef test_a(): ...\n") == ["test_a"]
    # The same name in two different classes is two tests, not a shadow.
    assert shadowed_tests("class A:\n def test_a(self): ...\nclass B:\n def test_a(self): ...\n") == []


def test_no_test_module_defines_a_test_twice():
    """Every test is collected: no module defines the same test name twice in one scope."""
    offenders = {}
    for path in _test_files():
        names = shadowed_tests(path.read_text())
        if names:
            offenders[str(path.relative_to(ROOT))] = names
    assert not offenders, (
        f"These test functions are defined twice, so the first copy never runs: {offenders}. "
        "Rename one of each pair so both are collected."
    )
