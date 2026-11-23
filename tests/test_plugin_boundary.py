"""The plugin boundary ratchet (#1178; the PRD §4 decision under principle 1).

A plugin holds a reference to another plugin only when the target is a
kernel plugin, named in its ``depends_on`` and resolved with
``get_dependency``; it reaches every other plugin through hooks
(docs/plugin-guide.md, "Plugin boundaries"). A bare ``pm.get_plugin()`` has
no use in a plugin, but the tree still has calls that predate the rule.

This test lists those calls in ``ALLOWED``, per file and per plugin name,
each with the issue that removes it. It fails on a ``get_plugin`` call under
``packages/`` (test directories excluded) beyond what ``ALLOWED`` lists, and
on an entry that lists more calls than remain, so the list only shrinks.

It reads source text and imports nothing from the packages, so it runs in
the kernel-only CI job, like tests/test_kernel_boundary.py. It is the one
static check #1178 accepted: it does not look at ``get_plugins()``, table
reads, fired hooks or annotations. It does not catch an indirect call either,
such as ``getattr(pm, "get_plugin")`` or the bound method under another
name.
"""

import ast
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGES = REPO / "packages"

# Where to look for the fix, named in every failure message.
GUIDE = 'docs/plugin-guide.md, "Plugin boundaries"'

# The name recorded for a call whose first argument is not a string literal.
DYNAMIC = "<dynamic>"

# The calls that predate the rule: file (relative to the repo) -> plugin name
# -> (how many calls, the issue that removes them). Lower a count, or delete
# the entry, in the change that removes a call.
ALLOWED: dict[str, dict[str, tuple[int, str]]] = {
    "packages/corvidae-cognition/corvidae_cognition/critique.py": {
        "appraisal": (2, "#1169"),
    },
    "packages/corvidae-planning/corvidae_planning/__init__.py": {
        "goals": (1, "#1181"),
    },
}


def _plugin_sources() -> list[Path]:
    """Every .py file under packages/ outside a tests directory."""
    return [
        path
        for path in sorted(PACKAGES.rglob("*.py"))
        if "tests" not in path.relative_to(PACKAGES).parts
    ]


def _get_plugin_calls(path: Path) -> list[tuple[int, str]]:
    """Return (line, plugin name) for every ``get_plugin(...)`` call in a file.

    Parsed, not grepped, so the name inside a string (a log message that
    quotes ``get_plugin('clock')``) is not a call. Both ``x.get_plugin(...)``
    and a bare ``get_plugin(...)`` count; ``get_plugins()`` does not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    # Walk every node so calls inside functions and methods are reached.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name != "get_plugin":
            continue
        # Record the looked-up name; anything but a string literal is dynamic.
        first = node.args[0] if node.args else None
        target = (
            first.value
            if isinstance(first, ast.Constant) and isinstance(first.value, str)
            else DYNAMIC
        )
        found.append((node.lineno, target))
    return found


def _calls_by_file() -> dict[str, list[tuple[int, str]]]:
    """Every get_plugin call under packages/, keyed by repo-relative path."""
    calls: dict[str, list[tuple[int, str]]] = {}
    for path in _plugin_sources():
        found = _get_plugin_calls(path)
        if found:
            calls[path.relative_to(REPO).as_posix()] = found
    return calls


def test_no_get_plugin_call_beyond_the_allowlist() -> None:
    """A plugin gains no new get_plugin lookup (#1178)."""
    offenders: list[str] = []
    # Compare each file's per-name count with what ALLOWED lists for it.
    for rel, calls in _calls_by_file().items():
        allowed = ALLOWED.get(rel, {})
        counts = Counter(target for _, target in calls)
        for target, count in sorted(counts.items()):
            limit = allowed.get(target, (0, ""))[0]
            if count > limit:
                lines = ", ".join(str(line) for line, t in calls if t == target)
                offenders.append(
                    f"{rel}:{lines}: {count} get_plugin({target!r}) call(s), {limit} allowed"
                )
    assert offenders == [], (
        "new get_plugin lookups in plugins:\n"
        + "\n".join(offenders)
        + "\nReach a non-kernel plugin through a hook; reach a kernel plugin "
        "by naming it in depends_on and resolving it with get_dependency "
        f"({GUIDE})."
    )


def test_allowlist_lists_only_calls_that_remain() -> None:
    """An entry shrinks with the calls it covers, so the list only shrinks."""
    calls = _calls_by_file()
    stale: list[str] = []
    # Compare each listed count with the calls still in the file.
    for rel, entries in sorted(ALLOWED.items()):
        counts = Counter(target for _, target in calls.get(rel, []))
        for target, (limit, issue) in sorted(entries.items()):
            if counts[target] < limit:
                stale.append(
                    f"{rel}: ALLOWED lists {limit} get_plugin({target!r}) "
                    f"call(s) ({issue}), {counts[target]} remain"
                )
    assert stale == [], (
        "stale allowlist entries:\n"
        + "\n".join(stale)
        + "\nLower the count, or delete the entry, in ALLOWED in "
        "tests/test_plugin_boundary.py."
    )
