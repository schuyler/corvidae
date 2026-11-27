"""The optional-extra guard must tell an absent extra from a broken package (#1287).

What these tests pin is the failure mode that makes the guard safe to have: a
skip line is only honest when it fires for "not installed" and never for
"installed but broken or shadowed". A guard that treats any import error as an
environment gap turns a real regression into a green line, which is a worse
outcome than the collection error it replaced.
"""

from __future__ import annotations

import ast
import collections
import os
import pathlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from _optional_extras import (
    _REASONS,
    extra_importable,
    optional_extra,
    requires_extra,
)

ABSENT = "corvidae_extra_definitely_not_installed_xyz"

_TESTS_DIR = pathlib.Path(__file__).resolve().parent


def _guarded_names(node: ast.With) -> set[str]:
    """The module names an ``optional_extra("…")`` block stands in for."""
    names: set[str] = set()
    for item in node.items:
        call = item.context_expr
        if (
            isinstance(call, ast.Call)
            and getattr(call.func, "id", "") == "optional_extra"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            names.add(call.args[0].value)
    return names


#: Directories holding test modules that import an extra's package at import
#: time with no guard, and how many such sites each holds today. These are
#: pre-existing debt from the same class as #1287, not exemptions: the scan
#: enforces a ceiling, so a NEW unguarded import anywhere in the tree fails, and
#: fixing one means lowering its ceiling in the same commit. Most of these
#: directories are also passed as ``--ignore`` by the extra-free kernel job
#: (14 such entries in ``.github/workflows/tests.yml``), which is why CI has
#: never shown them. A count is one import statement, however many names it
#: binds. Measured by this scan at 06:12:30Z; the debt is tracked in #1313.
_UNGUARDED_DEBT = {
    "cognition": 103,
    "test_outbound_cap": 2,
    "test_planning": 2,
    "corvidae_restart": 1,
    "test_alignment": 1,
    "test_goals_notify": 1,
    "test_inotify": 1,
    "test_timepiece": 1,
}


def _collected_test_files() -> list[pathlib.Path]:
    """Test modules pytest would collect, minus the directories CI ignores.

    Only collectible names are in scope: ``*_child.py`` helpers under ``tests/``
    are launched by a test as a subprocess and are never imported during
    collection, so a module-scope extra import there is not a collection hazard.
    """
    files = []
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if not (path.name.startswith("test_") or path.name.endswith("_test.py")):
            continue
        files.append(path)
    return files


def _extra_import_names() -> set[str]:
    """Every import name an optional extra provides, read from the tree.

    `_REASONS` covers the six plugin packages that had module-scope imports when
    the guard helper was written; the point of this scan is that the next extra
    does not need to be remembered by hand. Each ``packages/*`` distribution
    contributes its project name and any top-level package directory inside it,
    which is how the import name is recovered where it differs from the
    distribution (``corvidae-whatsapp`` ships ``corvidae_whatsapp``).
    """
    names = set(_REASONS)
    root = _TESTS_DIR.parent / "packages"
    for dist in sorted(root.iterdir()):
        manifest = dist / "pyproject.toml"
        if not manifest.is_file():
            continue
        data = tomllib.loads(manifest.read_text())
        names.add(data["project"]["name"].replace("-", "_"))
        for child in sorted(dist.iterdir()):
            if (child / "__init__.py").is_file():
                names.add(child.name)
    return names


def _is_type_checking(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:``, a block that never executes."""
    return "TYPE_CHECKING" in {getattr(test, "id", None), getattr(test, "attr", None)}


def _import_time_imports(tree: ast.Module):
    """Yield ``(lineno, top_level_name, guarded_names)`` for import-time imports.

    Descent is generic on purpose: whatever runs when a module is imported counts
    — module scope, class bodies, and compound statements such as ``with``,
    ``try``, ``if``, ``for``, ``while`` and ``match``. It stops at function and
    lambda bodies, whose imports run on call rather than at collection.
    ``if TYPE_CHECKING:`` is skipped because it never executes, and a relative
    import is skipped because an extra installs as a top-level package, so
    ``from .x import y`` cannot name one.
    """

    def walk(nodes, guarded):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.If) and _is_type_checking(node.test):
                continue
            if isinstance(node, ast.With):
                yield from walk(node.body, guarded | _guarded_names(node))
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level:
                    continue
                for alias in node.names:
                    if isinstance(node, ast.Import):
                        top = alias.name.split(".")[0]
                    else:
                        top = (node.module or "").split(".")[0]
                    yield node.lineno, top, guarded
                continue
            yield from walk(list(ast.iter_child_nodes(node)), guarded)

    yield from walk(tree.body, frozenset())


def _is_skip(marker) -> bool:
    """The condition a ``skipif`` marker carries."""
    assert marker.name == "skipif", marker
    return bool(marker.args[0])


def test_absent_extra_reports_not_importable():
    assert extra_importable(ABSENT) is False


def test_installed_module_reports_importable():
    assert extra_importable("pytest") is True


def test_guard_skips_the_module_when_the_extra_is_absent():
    """The guard must raise pytest.skip before the guarded import executes."""
    with pytest.raises(pytest.skip.Exception) as excinfo:
        with optional_extra(ABSENT):
            raise AssertionError("the guarded import must not run")

    assert ABSENT in str(excinfo.value)
    # Without this flag pytest treats the skip as a test-level one and collection
    # of the module still dies on the import below the guard.
    assert excinfo.value.allow_module_level is True


def test_guard_lets_the_guarded_import_run_when_present():
    with optional_extra("pytest") as _unused:
        import pytest as imported_here

    assert imported_here is pytest


def test_marker_skips_only_when_the_extra_is_absent():
    assert _is_skip(requires_extra(ABSENT)) is True
    assert _is_skip(requires_extra("pytest")) is False


def test_broken_package_error_is_not_swallowed(tmp_path, monkeypatch):
    """A present package whose own import fails must still fail the run.

    This is the acp helper's discipline, kept: only the module named in the
    probe means "not installed", so a missing transitive dependency propagates
    with its own ``name`` rather than becoming a skip.
    """
    pkg = tmp_path / "broken_extra_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import nonexistent_transitive_dep_xyz  # noqa: F401\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "broken_extra_pkg", raising=False)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        extra_importable("broken_extra_pkg")

    assert excinfo.value.name == "nonexistent_transitive_dep_xyz"

    # The guard runs at import time of a test module, so a broken package has to
    # break there rather than yield a skip whose reason is a lie.
    with pytest.raises(ModuleNotFoundError) as guard_excinfo:
        with optional_extra("broken_extra_pkg"):
            pass

    assert guard_excinfo.value.name == "nonexistent_transitive_dep_xyz"

    # The marker path shares the probe, so it must break the same way.
    with pytest.raises(ModuleNotFoundError) as mark_excinfo:
        requires_extra("broken_extra_pkg")

    assert mark_excinfo.value.name == "nonexistent_transitive_dep_xyz"


def test_namespace_shadow_is_not_reported_as_absent(tmp_path, monkeypatch):
    """An empty directory of the right name imports, so it must not skip.

    A directory under a ``pythonpath`` entry that matches a package name becomes
    a namespace package ahead of the real one, and the symptom is
    ``ImportError: cannot import name X from Y (unknown location)`` at the
    module's own ``from Y import Z``. The probe sees a namespace package that
    imports cleanly and reports present, so that failure stays visible.
    """
    shadow = tmp_path / "shadowed_extra_pkg"
    shadow.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "shadowed_extra_pkg", raising=False)

    assert extra_importable("shadowed_extra_pkg") is True
    assert _is_skip(requires_extra("shadowed_extra_pkg")) is False

    import shadowed_extra_pkg  # noqa: F401

    with pytest.raises(ImportError) as excinfo:
        from shadowed_extra_pkg import SomethingNotThere  # noqa: F401

    assert "unknown location" in str(excinfo.value)

    # And through the guard itself: the context manager stands aside for the
    # import it guards, so the shadow failure surfaces rather than skipping.
    with pytest.raises(ImportError) as guarded:
        with optional_extra("shadowed_extra_pkg"):
            from shadowed_extra_pkg import SomethingNotThere  # noqa: F401

    assert "unknown location" in str(guarded.value)


def test_each_named_extra_has_a_reason():
    """A skip line should name the extra to install, not just 'skipped'."""
    for module in (
        "cross_channel_send",
        "corvidae_whatsapp",
        "context_panel",
        "webhook_notify",
        "github_webhook",
        "goals_notify",
    ):
        assert module in _REASONS
        assert module in _REASONS[module], _REASONS[module]
        assert requires_extra(module).mark.kwargs["reason"].find(module) != -1


def test_every_module_scope_extra_import_is_guarded():
    """No collected test module may import an extra's package unguarded.

    The hole that makes this necessary (#1287):
    ``tests/test_context_panel/test_render_smoke.py`` stood itself down with
    ``pytest.importorskip("playwright.async_api")`` and then imported
    ``context_panel`` bare. Where Playwright is installed and the extra is not,
    that is a collection error; where Playwright is absent the earlier skip hides
    it, and the ``web-smoke`` job installs both, so CI could never see it.
    Repairing the one reported file would leave the class open, so the predicate
    is structural, and it reads its own scope from the tree instead of a list
    somebody has to remember:

    * the names come from ``packages/*`` plus ``_REASONS``, so a plugin added
      tomorrow is in force the day it lands;
    * the files are the ones pytest collects (``test_*.py`` / ``*_test.py``),
      since a ``*_child.py`` helper run as a subprocess is never imported during
      collection;
    * ``_UNGUARDED_DEBT`` records the sites that already exist, directory by
      directory, as a ceiling rather than an exemption — a new unguarded import
      fails anywhere in the tree, and clearing one means lowering the ceiling in
      the same commit.

    Two limits, stated rather than implied. A dynamic import
    (``importlib.import_module("context_panel")``) and an extra reached through a
    re-export elsewhere are invisible to a scan of this shape. And a guard counts
    only in the house spelling ``with optional_extra("<literal name>")``: a
    wrapper around the helper, or a name held in a constant, reads as unguarded.
    That strictness is deliberate — one spelling is what keeps a skip honest about
    the exact module that is missing. A relative import is not a gap: extras
    install as top-level packages, so ``from .x import y`` cannot name one.
    """
    extra_names = _extra_import_names()
    offenders = set()
    # Count distinct (file, line, package) triples, not AST yields: one
    # ``from x import a, b, c`` is a single statement and must move the ceiling
    # by 1, or a fixer cannot tell what they cleared.
    per_dir = collections.defaultdict(set)
    for path in _collected_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for lineno, top, guarded in _import_time_imports(tree):
            if top in extra_names and top not in guarded:
                offenders.add(
                    f"{path.relative_to(_TESTS_DIR)}:{lineno} imports {top} "
                    "outside an optional_extra() guard"
                )
                parts = path.relative_to(_TESTS_DIR).parts
                key = parts[0] if len(parts) > 1 else "(tests root)"
                per_dir[key].add((str(path.relative_to(_TESTS_DIR)), lineno, top))

    over_budget = [
        f"{name}: {count} unguarded import sites, ceiling {ceiling} — "
        "guard the new ones with optional_extra() or lower the ceiling in "
        "_UNGUARDED_DEBT if you fixed some"
        for name, count in sorted(((k, len(v)) for k, v in per_dir.items()))
        if (ceiling := _UNGUARDED_DEBT.get(name, 0)) < count
    ]
    assert not over_budget, "\n".join(over_budget)

_HIDE_PLUGIN = """
import sys

class _Hide:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "context_panel":
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return None

sys.meta_path.insert(0, _Hide())
"""


def test_render_smoke_skips_rather_than_errors_without_context_panel(tmp_path):
    """#1287: playwright being present is not evidence that context_panel is.

    ``test_render_smoke.py`` skipped cleanly for a missing playwright and then
    imported ``context_panel`` unguarded, so a venv with playwright but without
    the extra errored at collection -- the failure mode #1281 is about. This
    collects the real module with the extra hidden and requires a clean exit.
    """
    # Without Playwright the module skips at its own importorskip and collection
    # exits 5 for that reason instead: the kernel job would pass this test on
    # the wrong skip. Require the environment the assertion is about.
    pytest.importorskip("playwright.async_api")
    (tmp_path / "hide_context_panel.py").write_text(_HIDE_PLUGIN)
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), str(root), os.environ.get("PYTHONPATH", "")]
        ),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "hide_context_panel",
         "--collect-only", "-q", "-rs",
         "tests/test_context_panel/test_render_smoke.py"],
        cwd=root, env=env, capture_output=True, text=True, timeout=180,
    )
    assert "ERROR" not in proc.stdout, proc.stdout
    # 0 or 5 ("no tests collected"): the module-level skip is the wanted
    # outcome, and either way collection must not have errored.
    assert proc.returncode in (0, 5), proc.stdout + proc.stderr
    # Not just "collected nothing": the reason must be the missing extra.
    assert "optional context_panel extra not installed" in proc.stdout, proc.stdout
