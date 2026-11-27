"""R5 (#1211): no daemon module opens its own read-write connection to sessions.db.

An AST scan of the in-tree code finds every ``sqlite3.connect`` and
``aiosqlite.connect`` call. Each must be on the allowlist below, which names
the file, the enclosing function and why that connection is allowed.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (relative path, enclosing function's qualified name, reason)
ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    # The route itself: the one read-write connection and the read-only reader.
    ("corvidae/session_writer.py", "SessionWriter.open", "the kernel write route"),
    ("corvidae/session_writer.py", "open_session_store", "the read-only reader"),
    # Different files, not sessions.db.
    (
        "packages/corvidae-cognition/corvidae_cognition/markdown_recall/index.py",
        "MarkdownIndex._connect",
        "opens its own index.db",
    ),
    # Read-only connection (mode=ro).
    (
        "packages/corvidae-cognition/corvidae_cognition/probe.py",
        "ReadOnlyProbe._attempt",
        "opens the database mode=ro",
    ),
    # Offline CLIs (run with the daemon stopped), exempt by the requirement.
    ("corvidae/commands/redact.py", "_message_cmd._run", "offline redact CLI"),
    ("corvidae/commands/redact.py", "_memory_cmd._run", "offline redact CLI"),
    ("corvidae/commands/redact.py", "_range_cmd._run", "offline redact CLI"),
    (
        "packages/corvidae-cognition/corvidae_cognition/reconsolidate.py",
        "_backup",
        "offline reconsolidate CLI",
    ),
    (
        "packages/corvidae-cognition/corvidae_cognition/reconsolidate.py",
        "check_is_copy",
        "offline reconsolidate CLI",
    ),
    (
        "packages/corvidae-cognition/corvidae_cognition/reconsolidate.py",
        "reconsolidate_command._run",
        "offline reconsolidate CLI",
    ),
    (
        "packages/corvidae-cognition/corvidae_cognition/goals.py",
        "_repair_cmd._run",
        "offline goals CLI",
    ),
    (
        "packages/corvidae-cognition/corvidae_cognition/goals.py",
        "_seed_cmd._run",
        "offline goals CLI",
    ),
)

# Connect functions the scan looks for: (module name, attribute).
CONNECT_CALLS = {("sqlite3", "connect"), ("aiosqlite", "connect")}


def _source_files() -> list[Path]:
    """Every non-test, non-build .py file under corvidae/ and packages/*/."""
    roots = [REPO / "corvidae", *sorted((REPO / "packages").glob("*"))]
    files: list[Path] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            parts = path.relative_to(REPO).parts
            # Skip test trees and build output.
            if "tests" in parts or "build" in parts:
                continue
            files.append(path)
    return files


def _connect_sites(path: Path) -> list[tuple[int, str]]:
    """Return (line, qualified enclosing function) for each connect call."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites: list[tuple[int, str]] = []

    def visit(node: ast.AST, stack: list[str]) -> None:
        """Walk the tree keeping the stack of enclosing class/function names."""
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                visit(child, [*stack, child.name])
                continue
            # A call whose func is `<name>.connect` for a watched module.
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and (child.func.value.id, child.func.attr) in CONNECT_CALLS
            ):
                sites.append((child.lineno, ".".join(stack) or "<module>"))
            visit(child, stack)

    visit(tree, [])
    return sites


def test_r5_no_unrouted_session_db_connections() -> None:
    """R5: every connect call in daemon code is on the allowlist."""
    allowed = {(rel, func) for rel, func, _reason in ALLOWLIST}
    unlisted: list[str] = []
    # Check every connect site against the allowlist.
    for path in _source_files():
        rel = path.relative_to(REPO).as_posix()
        for line, func in _connect_sites(path):
            if (rel, func) not in allowed:
                unlisted.append(f"  {rel}:{line} in {func}")

    assert not unlisted, (
        "Unrouted database connection(s) found:\n"
        + "\n".join(unlisted)
        + "\nRoute writes through `persistence.write` (the kernel write route) "
        "or, if the connection opens a different file or is read-only, add an "
        "allowlist entry to ALLOWLIST in tests/test_session_db_writers.py "
        "naming the file it opens."
    )
