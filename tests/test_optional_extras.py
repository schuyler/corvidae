"""The optional-extra guard must tell an absent extra from a broken package (#1287).

What these tests pin is the failure mode that makes the guard safe to have: a
skip line is only honest when it fires for "not installed" and never for
"installed but broken or shadowed". A guard that treats any import error as an
environment gap turns a real regression into a green line, which is a worse
outcome than the collection error it replaced.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import warnings
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


_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent


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


def _is_type_checking(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:``, a block that never executes."""
    return "TYPE_CHECKING" in {getattr(test, "id", None), getattr(test, "attr", None)}


def _re_raise_guard_names(node: ast.Try) -> frozenset[str]:
    """The module names a ``try``/``except ModuleNotFoundError`` block vouches for.

    The tree's own correct guard, written a PR before the helper existed and cited
    in #1287 as the exemplar, is::

        try:
            import acp as _acp
        except ModuleNotFoundError as exc:
            if exc.name != "acp":
                raise

    It skips when the extra is absent and re-raises when the package is present but
    broken, which is the absent-versus-broken rule the helper enforces. The literal
    it compares ``exc.name`` against is the name being vouched for.
    """
    vouched: set[str] = set()
    for handler in node.handlers:
        handler_type = handler.type
        kind = getattr(handler_type, "id", None) or getattr(handler_type, "attr", None)
        if kind not in {"ModuleNotFoundError", "ImportError"}:
            continue
        for sub in ast.walk(handler):
            if not isinstance(sub, ast.If) or not any(isinstance(n, ast.Raise) for n in ast.walk(sub)):
                continue
            comparisons = sub.test.values if isinstance(sub.test, ast.BoolOp) else (sub.test,)
            for cmp in comparisons:
                if not isinstance(cmp, ast.Compare):
                    continue
                sides = [cmp.left, *cmp.comparators]
                if not any(isinstance(x, ast.Attribute) and x.attr == "name" for x in sides):
                    continue
                vouched |= {
                    x.value
                    for x in sides
                    if isinstance(x, ast.Constant) and isinstance(x.value, str)
                }
    return frozenset(vouched)


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
            if isinstance(node, ast.Try):
                yield from walk(node.body, guarded | _re_raise_guard_names(node))
                for branch in [*[h.body for h in node.handlers], node.orelse, node.finalbody]:
                    yield from walk(branch, guarded)
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
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        names.add(data["project"]["name"].replace("-", "_"))
        for child in sorted(dist.iterdir()):
            if (child / "__init__.py").is_file():
                names.add(child.name)

    # Extras whose distribution is third-party rather than a workspace package
    # (``acp``, ``discord``, ``emoji``) are invisible to the walk above, so the
    # extra key is the only clue to their import root — and that clue is wrong in
    # two directions, hence 2 filters: ``signal`` is a stdlib module name, and
    # ``cognition`` is this tree's own test package. Where a key's root differs from
    # the key in a way no walk can see (``signal`` → ``emoji``), the root is listed
    # by hand below.
    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    keys = {k.replace("-", "_") for k in cfg["project"]["optional-dependencies"]} - {"dev"}
    keys -= set(sys.stdlib_module_names) | _tree_local_names()
    keys.add("emoji")
    return names | keys


def _tree_local_names() -> set[str]:
    """Top-level names already importable from pytest's own ``pythonpath``.

    An extra key is a guess at an import root, and this tree collides with the
    guess: ``tests/cognition/__init__.py`` is empty and ``pythonpath`` includes
    ``tests``, so ``from cognition.test_redact_cascade import …`` is a sibling-test
    import inside the test tree, not an extra. Guarding it would add a skip that can
    never fire. The ``packages/*`` walk is exempt from this filter because those are
    declared workspace packages — so ``corvidae_restart`` stays enforced even though
    ``pythonpath`` also lists ``packages/corvidae-restart`` and it cannot error
    today. That asymmetry is deliberate, not an oversight.
    """
    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = cfg["tool"]["pytest"]["ini_options"].get("pythonpath", [])
    found: set[str] = set()
    for entry in entries:
        directory = _REPO_ROOT / entry
        if not directory.is_dir():
            continue
        for child in directory.iterdir():
            if child.is_dir() and (child / "__init__.py").is_file():
                found.add(child.name)
            elif child.suffix == ".py" and child.name != "conftest.py":
                found.add(child.stem)
    return found


def _scanned_files() -> list[pathlib.Path]:
    """Every module the test tree can run at import time.

    Wider than pytest's collection pattern, in two directions: the
    ``tests/`` tree, and each workspace package's own ``tests/`` directory
    (``packages/*/tests/``), which CI runs under no job at all. ``conftest.py`` and
    package ``__init__.py`` are imported during collection whether or not they
    match ``test_*.py``; a ``*_child.py`` helper is only safe while nothing
    imports it in-process; and a package test dir holds the same shape of import
    (#1287 cites ``packages/corvidae-vision/tests/test_vision.py`` as already
    correct, and today it is not). Recording all of it in the ledger costs a few
    lines and leaves no silent hole. Paths are repo-root-relative. Only
    ``__pycache__`` is skipped.
    """
    return [
        path
        for path in sorted(_TESTS_DIR.rglob("*.py"))
        + sorted(_REPO_ROOT.glob("packages/*/tests/**/*.py"))
        if "__pycache__" not in path.parts
    ]


#: Every module-scope import of an optional extra's package that currently sits
#: outside an ``optional_extra()`` guard, as ``relative/path.py:package``. This is
#: a ledger of existing debt, not an exemption: an import whose file-and-package
#: pair is absent here fails the test, and fixing one means deleting its line in
#: the same commit. Keyed without line numbers because those shift with any
#: unrelated edit above them. Measured by this scan at 06:22:01Z; tracked in #1313.


_UNGUARDED_DEBT = frozenset({
    # cognition — 68
    "tests/cognition/test_appraisal_fork.py:corvidae_cognition",
    "tests/cognition/test_appraisal_no_vector.py:corvidae_cognition",
    "tests/cognition/test_appraisal_stage2.py:corvidae_cognition",
    "tests/cognition/test_catchup_late_results.py:corvidae_cognition",
    "tests/cognition/test_catchup_oversized_tool_call.py:corvidae_cognition",
    "tests/cognition/test_catchup_segmentation_linear.py:corvidae_cognition",
    "tests/cognition/test_critique.py:corvidae_cognition",
    "tests/cognition/test_critique_fork.py:corvidae_cognition",
    "tests/cognition/test_critique_lenses.py:corvidae_cognition",
    "tests/cognition/test_critique_one_fork.py:corvidae_cognition",
    "tests/cognition/test_critique_proactivity.py:corvidae_cognition",
    "tests/cognition/test_critique_vector.py:corvidae_cognition",
    "tests/cognition/test_default_prompts.py:corvidae_cognition",
    "tests/cognition/test_driver_memory_role.py:corvidae_cognition",
    "tests/cognition/test_embed_width.py:corvidae_cognition",
    "tests/cognition/test_entry_size_target.py:corvidae_cognition",
    "tests/cognition/test_fork_log.py:corvidae_cognition",
    "tests/cognition/test_fork_log_digest_granularity.py:corvidae_cognition",
    "tests/cognition/test_fork_log_read_guard.py:corvidae_cognition",
    "tests/cognition/test_fork_log_snapshot_capture.py:corvidae_cognition",
    "tests/cognition/test_funnel_policy.py:corvidae_cognition",
    "tests/cognition/test_funnel_ponder_and_reload.py:corvidae_cognition",
    "tests/cognition/test_goals.py:corvidae_cognition",
    "tests/cognition/test_goals_repair.py:corvidae_cognition",
    "tests/cognition/test_heartbeat.py:corvidae_cognition",
    "tests/cognition/test_heartbeat_goals.py:corvidae_cognition",
    "tests/cognition/test_heartbeat_key_aliases.py:corvidae_cognition",
    "tests/cognition/test_heartbeat_window.py:corvidae_cognition",
    "tests/cognition/test_lease_scopes.py:corvidae_cognition",
    "tests/cognition/test_markdown_recall_chunking.py:corvidae_cognition",
    "tests/cognition/test_markdown_recall_frontmatter.py:corvidae_cognition",
    "tests/cognition/test_markdown_recall_index.py:corvidae_cognition",
    "tests/cognition/test_markdown_recall_plugin.py:corvidae_cognition",
    "tests/cognition/test_markdown_recall_search_hook.py:corvidae_cognition",
    "tests/cognition/test_memory_catchup.py:corvidae_cognition",
    "tests/cognition/test_memory_consolidation.py:corvidae_cognition",
    "tests/cognition/test_memory_contribution.py:corvidae_cognition",
    "tests/cognition/test_memory_dedup.py:corvidae_cognition",
    "tests/cognition/test_memory_fork.py:corvidae_cognition",
    "tests/cognition/test_memory_hybrid_retrieval.py:corvidae_cognition",
    "tests/cognition/test_memory_idle_clock.py:corvidae_cognition",
    "tests/cognition/test_memory_integration.py:corvidae_cognition",
    "tests/cognition/test_memory_line_ids.py:corvidae_cognition",
    "tests/cognition/test_memory_query_context.py:corvidae_cognition",
    "tests/cognition/test_memory_raw_fallback.py:corvidae_cognition",
    "tests/cognition/test_memory_recall_identity.py:corvidae_cognition",
    "tests/cognition/test_memory_relative_gate.py:corvidae_cognition",
    "tests/cognition/test_memory_retention.py:corvidae_cognition",
    "tests/cognition/test_memory_retrieval.py:corvidae_cognition",
    "tests/cognition/test_memory_schema.py:corvidae_cognition",
    "tests/cognition/test_memory_stamps.py:corvidae_cognition",
    "tests/cognition/test_memory_tools.py:corvidae_cognition",
    "tests/cognition/test_memory_turn_backoff.py:corvidae_cognition",
    "tests/cognition/test_memory_turn_trigger.py:corvidae_cognition",
    "tests/cognition/test_memory_window_exclusion.py:corvidae_cognition",
    "tests/cognition/fork_log_store.py:corvidae_cognition",
    "tests/cognition/test_plugin_shutdown_wired.py:corvidae_cognition",
    "tests/cognition/test_probe.py:corvidae_cognition",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition",
    "tests/cognition/test_recall_contribution.py:corvidae_cognition",
    "tests/cognition/test_reconsolidate.py:corvidae_cognition",
    "tests/cognition/test_redact_cascade.py:corvidae_cognition",
    "tests/cognition/test_redact_exchange_log.py:corvidae_cognition",
    "tests/cognition/test_retention_write_txn.py:corvidae_cognition",
    "tests/cognition/test_retrieval_benchmark.py:corvidae_cognition",
    "tests/cognition/test_retrieval_candidates.py:corvidae_cognition",
    "tests/cognition/test_retrieval_log_prune.py:corvidae_cognition",
    "tests/cognition/test_skills.py:corvidae_cognition",
    # test_planning — 2
    "tests/test_planning/test_planning_lens.py:corvidae_cognition",
    "tests/test_planning/test_planning_lens.py:corvidae_planning",
    # corvidae_restart — 1
    "tests/corvidae_restart/test_restart.py:corvidae_restart",
    # packages/corvidae-vision — 1
    "packages/corvidae-vision/tests/test_vision.py:corvidae_vision",
    # packages/rss-reader — 1
    "packages/rss-reader/tests/test_rss_reader.py:rss_reader",
    # test_alignment — 1
    "tests/test_alignment/test_alignment_lens.py:corvidae_alignment",
    # test_goals_notify — 1
    "tests/test_goals_notify/test_goals_notify.py:corvidae_cognition",
    # test_inotify — 1
    "tests/test_inotify/test_inotify_ping.py:corvidae_inotify",
    # test_outbound_cap — 1
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap",
    # test_timepiece — 1
    "tests/test_timepiece/test_timepiece.py:corvidae_timepiece",
    # (tests root) — 4
    "tests/preflight_cleanup_child.py:corvidae_restart",
    "tests/preflight_failed_load_child.py:corvidae_restart",
    "tests/preflight_leak_child.py:corvidae_restart",
    "tests/preflight_report_child.py:corvidae_restart",
})


def test_no_new_unguarded_extra_imports():
    """No test-tree module may import an extra's package unguarded, beyond debt.

    The hole that makes this necessary (#1287):
    ``tests/test_context_panel/test_render_smoke.py`` stood itself down with
    ``pytest.importorskip("playwright.async_api")`` and then imported
    ``context_panel`` bare. Where Playwright is installed and the extra is not,
    that is a collection error; where Playwright is absent the earlier skip hides
    it, and the ``web-smoke`` job installs both, so CI could never see it.
    Repairing the one reported file would leave the class open, so the predicate
    is structural, and it reads its own scope from the tree instead of a list
    somebody has to remember: the enforced names come from ``packages/*`` plus the
    root ``[project.optional-dependencies]`` minus stdlib collisions plus
    ``_REASONS``, and the scanned files are everything under ``tests/`` that can
    be imported at collection time.

    What it cannot see, stated rather than implied: a dynamic import
    (``importlib.import_module("context_panel")``), an extra reached through a
    re-export elsewhere, and an extra whose import root differs from its key
    without being a workspace package. A relative import is not a gap — extras
    install as top-level packages, so ``from .x import y`` cannot name one. A
    guard counts only in the house spelling ``with optional_extra("<literal
    name>")``; a wrapper around the helper or a name held in a constant reads as
    unguarded. That strictness is deliberate — one spelling is what keeps a skip
    honest about the exact module that is missing.
    """
    extra_names = _extra_import_names()
    found = set()
    for path in _scanned_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(_REPO_ROOT).as_posix()
        for _lineno, top, guarded in _import_time_imports(tree):
            if top in extra_names and top not in guarded:
                found.add(f"{relative}:{top}")

    new_debt = sorted(found - _UNGUARDED_DEBT)
    cleared = sorted(_UNGUARDED_DEBT - found)
    message = []
    if new_debt:
        message.append(
            "new unguarded extra imports (wrap them in "
            '"with optional_extra("<name>"):" before the import):\n  '
            + "\n  ".join(new_debt)
        )
    assert not message, "\n".join(message)
    if cleared:
        # Drift in the ledger's other direction is not the fault of the branch
        # that runs this test: warn, so an unledgered entry is fixed on purpose
        # (#1319) instead of reddening every PR that touches tests/.
        warnings.warn(
            "stale _UNGUARDED_DEBT entries — the tree guards them or no longer "
            "imports them; delete these lines (not a failure):\n  "
            + "\n  ".join(cleared),
            stacklevel=2,
        )


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


def test_a_stale_ledger_entry_warns_rather_than_fails(monkeypatch):
    """#1319: ledger drift must not redden branches that did not cause it."""
    stale = frozenset({"tests/cognition/gone.py:corvidae_cognition"})
    monkeypatch.setattr(
        sys.modules[__name__], "_UNGUARDED_DEBT", _UNGUARDED_DEBT | stale
    )
    with pytest.warns(UserWarning, match="stale _UNGUARDED_DEBT"):
        test_no_new_unguarded_extra_imports()
