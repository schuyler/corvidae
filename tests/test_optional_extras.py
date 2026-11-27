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
import tomllib
from pathlib import Path

import pytest
from _optional_extras import (
    _REASONS,
    _reason,
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
    """A skip line should name the extra to install, not just 'skipped'.

    Iterating ``_REASONS`` rather than a copy of its keys (#1299): the hardcoded
    list silently stopped covering whatever was added to the dict later, which is
    the drift the reason lines exist to prevent.
    """
    assert _REASONS, "no extras named; the table is what this test reads"
    for module in _REASONS:
        assert module in _REASONS[module], _REASONS[module]
        assert requires_extra(module).mark.kwargs["reason"].find(module) != -1


def test_the_collection_time_guard_names_the_extra_in_its_skip():
    """``optional_extra`` must carry the same actionable reason as the marker.

    ``requires_extra`` covers imports inside functions; ``optional_extra`` is the
    ``with`` form used where the import runs at collection, and a green skip from
    it has to name the extra just the same (#1299).
    """
    with pytest.raises(pytest.skip.Exception) as excinfo:
        with optional_extra(ABSENT):
            raise AssertionError("the guarded import must not run")

    assert str(excinfo.value) == _reason(ABSENT)


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


def _importorskip_name(node: ast.stmt) -> str | None:
    """The package a module-scope ``pytest.importorskip("…")`` vouches for.

    ``importorskip`` at module scope stands the whole module down when the
    package is missing, which is the same outcome ``optional_extra`` gives, so
    an import below it is guarded in fact even though it is not in the house
    spelling (#1317). Only a literal argument can be read statically; anything
    computed is left as debt rather than assumed safe.
    """
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    func = node.value.func
    name = getattr(func, "attr", None) or getattr(func, "id", None)
    if name != "importorskip" or not node.value.args:
        return None
    arg = node.value.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value.split(".")[0]
    return None


def _import_time_imports(tree: ast.Module):
    """Yield ``(lineno, top_level, guarded_names, target)`` per import-time import.

    ``target`` is the dotted module *and the name imported from it* —
    ``corvidae_cognition.fork_log.ForkLog`` for ``from corvidae_cognition
    .fork_log import ForkLog``, and ``corvidae_cognition.fork_log`` for ``import
    corvidae_cognition.fork_log``. That is what the ledger records, so a grant
    for one name does not pre-approve another name, or another module, of the
    same package (#1321).

    Descent is generic on purpose: whatever runs when a module is imported counts
    — module scope, class bodies, and compound statements such as ``with``,
    ``try``, ``if``, ``for``, ``while`` and ``match``. It stops at function and
    lambda bodies, whose imports run on call rather than at collection.
    ``if TYPE_CHECKING:`` is skipped because it never executes, and a relative
    import is skipped because an extra installs as a top-level package, so
    ``from .x import y`` cannot name one.
    """

    def walk(nodes, guarded):
        # Guards accumulate in statement order: a module-scope importorskip
        # vouches for the imports below it, not for the ones above it.
        scope = set(guarded)
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.If) and _is_type_checking(node.test):
                continue
            vouched = _importorskip_name(node)
            if vouched is not None:
                scope.add(vouched)
                continue
            if isinstance(node, ast.With):
                yield from walk(node.body, frozenset(scope) | _guarded_names(node))
                continue
            if isinstance(node, ast.Try):
                yield from walk(node.body, frozenset(scope) | _re_raise_guard_names(node))
                for branch in [*[h.body for h in node.handlers], node.orelse, node.finalbody]:
                    yield from walk(branch, frozenset(scope))
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level:
                    continue
                for alias in node.names:
                    if isinstance(node, ast.Import):
                        target = alias.name
                    else:
                        # ``from pkg import sub`` may be importing a submodule,
                        # so the name inside is recorded: that spelling and
                        # ``import pkg.sub`` are one grant (#1321).
                        target = f"{node.module or ''}.{alias.name}".lstrip(".")
                    yield node.lineno, target.split(".")[0], frozenset(scope), target
                continue
            yield from walk(list(ast.iter_child_nodes(node)), frozenset(scope))

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
    "packages/rss-reader/tests/test_rss_reader.py:rss_reader.RSSPlugin",
    "packages/rss-reader/tests/test_rss_reader.py:rss_reader._bad",
    "packages/rss-reader/tests/test_rss_reader.py:rss_reader._localname",
    "packages/rss-reader/tests/test_rss_reader.py:rss_reader._parse_feed",
    "packages/rss-reader/tests/test_rss_reader.py:rss_reader._parse_item_date",
    "tests/cognition/fork_log_store.py:corvidae_cognition.fork_log.ForkLog",
    "tests/cognition/test_appraisal_fork.py:corvidae_cognition.appraisal.DEFAULT_STAGE2_FORK_PROMPT",
    "tests/cognition/test_appraisal_fork.py:corvidae_cognition.appraisal.STAGE2_EXTRA_BODY",
    "tests/cognition/test_appraisal_no_vector.py:corvidae_cognition.outcome_log.OutcomeLogPlugin",
    "tests/cognition/test_appraisal_stage2.py:corvidae_cognition.outcome_log.OutcomeLogPlugin",
    "tests/cognition/test_catchup_late_results.py:corvidae_cognition.memory.CatchupSegment",
    "tests/cognition/test_catchup_late_results.py:corvidae_cognition.memory.build_catchup_segments",
    "tests/cognition/test_catchup_late_results.py:corvidae_cognition.memory.logger",
    "tests/cognition/test_catchup_oversized_tool_call.py:corvidae_cognition.memory.DEFAULT_CONSOLIDATION_FORK_PROMPT",
    "tests/cognition/test_catchup_oversized_tool_call.py:corvidae_cognition.memory._measured_size",
    "tests/cognition/test_catchup_oversized_tool_call.py:corvidae_cognition.memory.build_catchup_segments",
    "tests/cognition/test_catchup_oversized_tool_call.py:corvidae_cognition.memory.logger",
    "tests/cognition/test_critique.py:corvidae_cognition.outcome_log.OutcomeLogPlugin",
    "tests/cognition/test_critique_fork.py:corvidae_cognition.critique.DEFAULT_FORK_PROMPTS",
    "tests/cognition/test_critique_lenses.py:corvidae_cognition.critique.FORKS_OFF_REASON",
    "tests/cognition/test_critique_one_fork.py:corvidae_cognition.critique",
    "tests/cognition/test_critique_one_fork.py:corvidae_cognition.critique.DEFAULT_FORK_PROMPTS",
    "tests/cognition/test_critique_proactivity.py:corvidae_cognition.critique.BUILTIN_LENSES",
    "tests/cognition/test_critique_proactivity.py:corvidae_cognition.outcome_log._scrub_outcomes_detail",
    "tests/cognition/test_critique_vector.py:corvidae_cognition.critique.NO_REPLY_MARKER",
    "tests/cognition/test_critique_vector.py:corvidae_cognition.critique.PROVENANCE_NOTE",
    "tests/cognition/test_critique_vector.py:corvidae_cognition.critique.RETENTION_WAKE_NOTE",
    "tests/cognition/test_default_prompts.py:corvidae_cognition.critique.DEFAULT_FORK_PROMPTS",
    "tests/cognition/test_default_prompts.py:corvidae_cognition.memory.DEFAULT_CONSOLIDATION_FORK_PROMPT",
    "tests/cognition/test_driver_memory_role.py:corvidae_cognition.funnel.FunnelPlugin",
    "tests/cognition/test_driver_memory_role.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_embed_width.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_embed_width.py:corvidae_cognition.retention.run_retention_job",
    "tests/cognition/test_entry_size_target.py:corvidae_cognition.memory.DEFAULT_CONSOLIDATION_FORK_PROMPT",
    "tests/cognition/test_entry_size_target.py:corvidae_cognition.memory.DEFAULT_ENTRY_TARGET_TOKENS",
    "tests/cognition/test_entry_size_target.py:corvidae_cognition.memory.RecordStamp",
    "tests/cognition/test_entry_size_target.py:corvidae_cognition.memory.consolidation_instruction",
    "tests/cognition/test_entry_size_target.py:corvidae_cognition.memory.entry_size_sentence",
    "tests/cognition/test_entry_size_target.py:corvidae_cognition.memory.resolve_entry_target_tokens",
    "tests/cognition/test_fork_log.py:corvidae_cognition.fork_log.ForkLog",
    "tests/cognition/test_fork_log.py:corvidae_cognition.fork_log.LABELS",
    "tests/cognition/test_fork_log.py:corvidae_cognition.fork_log.prompt_sha256",
    "tests/cognition/test_fork_log.py:corvidae_cognition.fork_log.question_sha256",
    "tests/cognition/test_fork_log_digest_granularity.py:corvidae_cognition.fork_log.ForkLog",
    "tests/cognition/test_fork_log_digest_granularity.py:corvidae_cognition.fork_log.digest_of",
    "tests/cognition/test_fork_log_digest_granularity.py:corvidae_cognition.fork_log.prompt_sha256",
    "tests/cognition/test_fork_log_read_guard.py:corvidae_cognition.fork_log.ForkLog",
    "tests/cognition/test_fork_log_read_guard.py:corvidae_cognition.fork_log.digest_of",
    "tests/cognition/test_fork_log_snapshot_capture.py:corvidae_cognition.fork_log.digest_of",
    "tests/cognition/test_funnel_policy.py:corvidae_cognition.funnel.FunnelPlugin",
    "tests/cognition/test_funnel_ponder_and_reload.py:corvidae_cognition.funnel",
    "tests/cognition/test_funnel_ponder_and_reload.py:corvidae_cognition.funnel.FunnelPlugin",
    "tests/cognition/test_goals.py:corvidae_cognition.goals.GoalsPlugin",
    "tests/cognition/test_goals.py:corvidae_cognition.goals.goals_command",
    "tests/cognition/test_goals.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_goals_repair.py:corvidae_cognition.goals.GOALS_DDL",
    "tests/cognition/test_goals_repair.py:corvidae_cognition.goals.goals_command",
    "tests/cognition/test_heartbeat.py:corvidae_cognition.heartbeat.DEFAULT_HEARTBEAT_PROMPT",
    "tests/cognition/test_heartbeat.py:corvidae_cognition.heartbeat.HEARTBEAT_CHANNEL",
    "tests/cognition/test_heartbeat.py:corvidae_cognition.heartbeat.HeartbeatPlugin",
    "tests/cognition/test_heartbeat.py:corvidae_cognition.memory",
    "tests/cognition/test_heartbeat.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_heartbeat_key_aliases.py:corvidae_cognition.goals.GoalsPlugin",
    "tests/cognition/test_heartbeat_key_aliases.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_heartbeat_window.py:corvidae_cognition.heartbeat.HEARTBEAT_CHANNEL",
    "tests/cognition/test_heartbeat_window.py:corvidae_cognition.heartbeat.HeartbeatPlugin",
    "tests/cognition/test_heartbeat_window.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_heartbeat_window.py:corvidae_cognition.memory.fetch_range_rows",
    "tests/cognition/test_lease_scopes.py:corvidae_cognition.appraisal.AppraisalPlugin",
    "tests/cognition/test_lease_scopes.py:corvidae_cognition.appraisal.DEFAULT_STAGE2_FORK_PROMPT",
    "tests/cognition/test_lease_scopes.py:corvidae_cognition.critique.CritiquePlugin",
    "tests/cognition/test_lease_scopes.py:corvidae_cognition.critique.DEFAULT_FORK_PROMPTS",
    "tests/cognition/test_lease_scopes.py:corvidae_cognition.memory.DEFAULT_CONSOLIDATION_FORK_PROMPT",
    "tests/cognition/test_lease_scopes.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_lease_scopes.py:corvidae_cognition.outcome_log.OutcomeLogPlugin",
    "tests/cognition/test_markdown_recall_chunking.py:corvidae_cognition.markdown_recall.chunking.chunk_page",
    "tests/cognition/test_markdown_recall_frontmatter.py:corvidae_cognition.markdown_recall.frontmatter.parse_frontmatter",
    "tests/cognition/test_markdown_recall_index.py:corvidae_cognition.markdown_recall.index.Hit",
    "tests/cognition/test_markdown_recall_index.py:corvidae_cognition.markdown_recall.index.MarkdownIndex",
    "tests/cognition/test_markdown_recall_index.py:corvidae_cognition.markdown_recall.index._EncoderIdentity",
    "tests/cognition/test_markdown_recall_plugin.py:corvidae_cognition.markdown_recall.MarkdownRecallPlugin",
    "tests/cognition/test_markdown_recall_plugin.py:corvidae_cognition.markdown_recall.index.Hit",
    "tests/cognition/test_markdown_recall_search_hook.py:corvidae_cognition.markdown_recall.MarkdownRecallPlugin",
    "tests/cognition/test_memory_catchup.py:corvidae_cognition.memory.CONSOLIDATION_EXTRA_BODY",
    "tests/cognition/test_memory_catchup.py:corvidae_cognition.memory.DEFAULT_CONSOLIDATION_FORK_PROMPT",
    "tests/cognition/test_memory_catchup.py:corvidae_cognition.memory.DEFAULT_ENTRY_TARGET_TOKENS",
    "tests/cognition/test_memory_catchup.py:corvidae_cognition.memory._fork_range_sentence",
    "tests/cognition/test_memory_catchup.py:corvidae_cognition.memory.consolidation_instruction",
    "tests/cognition/test_memory_consolidation.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_consolidation.py:corvidae_cognition.memory.RubricPrior",
    "tests/cognition/test_memory_contribution.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_contribution.py:corvidae_cognition.memory.WORST_MEMORY_LINE_PREFIX",
    "tests/cognition/test_memory_dedup.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_fork.py:corvidae_cognition.memory.DEFAULT_CONSOLIDATION_FORK_PROMPT",
    "tests/cognition/test_memory_hybrid_retrieval.py:corvidae_cognition.memory.CANDIDATES_CHOICES",
    "tests/cognition/test_memory_hybrid_retrieval.py:corvidae_cognition.memory.FTS_SIMILARITY_PROXY",
    "tests/cognition/test_memory_hybrid_retrieval.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_idle_clock.py:corvidae_cognition.memory",
    "tests/cognition/test_memory_idle_clock.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_idle_clock.py:corvidae_cognition.retention",
    "tests/cognition/test_memory_integration.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_line_ids.py:corvidae_cognition.memory.WORST_MEMORY_LINE_PREFIX",
    "tests/cognition/test_memory_line_ids.py:corvidae_cognition.memory._memory_line",
    "tests/cognition/test_memory_query_context.py:corvidae_cognition.memory.QUERY_CONTEXT_CHOICES",
    "tests/cognition/test_memory_query_context.py:corvidae_cognition.memory.retrieval_query",
    "tests/cognition/test_memory_raw_fallback.py:corvidae_cognition.memory.WORST_MEMORY_LINE_PREFIX",
    "tests/cognition/test_memory_raw_fallback.py:corvidae_cognition.memory.prompt_hash",
    "tests/cognition/test_memory_recall_identity.py:corvidae_cognition.memory.FTS_SIMILARITY_PROXY",
    "tests/cognition/test_memory_relative_gate.py:corvidae_cognition.memory.DEFAULT_BAND_MODERATE_Z",
    "tests/cognition/test_memory_relative_gate.py:corvidae_cognition.memory.DEFAULT_BAND_STRONG_Z",
    "tests/cognition/test_memory_relative_gate.py:corvidae_cognition.memory.RELATIVE_MIN_RECORDS",
    "tests/cognition/test_memory_retention.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_retention.py:corvidae_cognition.memory_tools.MemoryToolsPlugin",
    "tests/cognition/test_memory_retention.py:corvidae_cognition.retention.RETRIEVAL_BOOST",
    "tests/cognition/test_memory_retention.py:corvidae_cognition.retention.retention_score",
    "tests/cognition/test_memory_retention.py:corvidae_cognition.retention.run_retention_job",
    "tests/cognition/test_memory_retrieval.py:corvidae_cognition.funnel.FunnelPlugin",
    "tests/cognition/test_memory_retrieval.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_retrieval.py:corvidae_cognition.memory.SQLITE_VEC_KNN_MAX",
    "tests/cognition/test_memory_schema.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_stamps.py:corvidae_cognition.memory.DEFAULT_ENTRY_TARGET_TOKENS",
    "tests/cognition/test_memory_stamps.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_stamps.py:corvidae_cognition.memory.consolidation_instruction",
    "tests/cognition/test_memory_tools.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_tools.py:corvidae_cognition.memory_tools.MemoryToolsPlugin",
    "tests/cognition/test_memory_turn_backoff.py:corvidae_cognition.memory.TURN_FORK_FAILURE_LIMIT",
    "tests/cognition/test_memory_turn_trigger.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_memory_window_exclusion.py:corvidae_cognition.memory.VEC_OVERFETCH",
    "tests/cognition/test_plugin_shutdown_wired.py:corvidae_cognition.critique.CritiquePlugin",
    "tests/cognition/test_plugin_shutdown_wired.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.appraisal.AppraisalPlugin",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.critique.CritiquePlugin",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.critique.DEFAULT_FORK_PROMPTS",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.goals.GOALS_INSTRUCTION",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.goals.GoalsPlugin",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.heartbeat.DEFAULT_HEARTBEAT_PROMPT",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.heartbeat.HeartbeatPlugin",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.memory.DEFAULT_CONSOLIDATION_FORK_PROMPT",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.memory.RUBRIC_PROMPT",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.memory.RubricPrior",
    "tests/cognition/test_prompt_routing.py:corvidae_cognition.memory.resolve_consolidation_fork_prompt",
    "tests/cognition/test_recall_contribution.py:corvidae_cognition.markdown_recall.MarkdownRecallPlugin",
    "tests/cognition/test_recall_contribution.py:corvidae_cognition.recall.RecallSourceMixin",
    "tests/cognition/test_reconsolidate.py:corvidae_cognition.memory.MEMORY_DDL",
    "tests/cognition/test_reconsolidate.py:corvidae_cognition.memory.MEMORY_FTS_DDL",
    "tests/cognition/test_reconsolidate.py:corvidae_cognition.memory.MEMORY_FTS_INSERT_TRIGGER",
    "tests/cognition/test_reconsolidate.py:corvidae_cognition.memory.MEMORY_FTS_UPDATE_TRIGGER",
    "tests/cognition/test_reconsolidate.py:corvidae_cognition.memory.MEMORY_PROMPT_DDL",
    "tests/cognition/test_reconsolidate.py:corvidae_cognition.memory.prompt_hash",
    "tests/cognition/test_reconsolidate.py:corvidae_cognition.reconsolidate",
    "tests/cognition/test_redact_cascade.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_redact_exchange_log.py:corvidae_cognition.outcome_log.redact_exchange_log",
    "tests/cognition/test_retention_write_txn.py:corvidae_cognition.retention.run_retention_job",
    "tests/cognition/test_retrieval_benchmark.py:corvidae_cognition.memory.MemoryPlugin",
    "tests/cognition/test_retrieval_candidates.py:corvidae_cognition.memory.FTS_SIMILARITY_PROXY",
    "tests/cognition/test_retrieval_log_prune.py:corvidae_cognition.retention.run_retention_job",
    "tests/cognition/test_skills.py:corvidae_cognition.skills.SkillsPlugin",
    "tests/corvidae_restart/test_restart.py:corvidae_restart.RestartPlugin",
    "tests/corvidae_restart/test_restart.py:corvidae_restart._is_number",
    "tests/corvidae_restart/test_restart.py:corvidae_restart.read_boot_status",
    "tests/corvidae_restart/test_restart.py:corvidae_restart.read_jsonl",
    "tests/corvidae_restart/test_restart.py:corvidae_restart.truncate_stderr",
    "tests/preflight_cleanup_child.py:corvidae_restart.preflight",
    "tests/preflight_failed_load_child.py:corvidae_restart.preflight",
    "tests/preflight_leak_child.py:corvidae_restart.preflight",
    "tests/preflight_report_child.py:corvidae_restart.preflight",
    "tests/test_alignment/test_alignment_lens.py:corvidae_alignment.AlignmentLensPlugin",
    "tests/test_goals_notify/test_goals_notify.py:corvidae_cognition.goals.GoalsSpec",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.COUNT_PROGRESS_KEY",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.MAX_MESSAGES_KEY",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.OutboundCapPlugin",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.WINDOW_LIMIT_KEY",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.WINDOW_SECONDS_KEY",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.parse_count_progress",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.parse_max_messages",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.parse_window_limit",
    "tests/test_outbound_cap/test_outbound_cap.py:outbound_cap.parse_window_seconds",
    "tests/test_planning/test_planning_lens.py:corvidae_cognition.goals.GOALS_DDL",
    "tests/test_planning/test_planning_lens.py:corvidae_cognition.goals.GoalsPlugin",
    "tests/test_planning/test_planning_lens.py:corvidae_planning.PlanningLensPlugin",
    "tests/test_timepiece/test_timepiece.py:corvidae_timepiece.TimepiecePlugin",

})


def _tree_unguarded_sites() -> dict[str, int]:
    """``relative/path.py:dotted.module`` → how many sites import it unguarded.

    The count is what a repeated key needs: one ledger line per grant, and an
    extra import in a file that already lists that module changes the count
    without adding a key (#1321).
    """
    extra_names = _extra_import_names()
    found: dict[str, int] = {}
    for path in _scanned_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = path.relative_to(_REPO_ROOT).as_posix()
        for _lineno, top, guarded, target in _import_time_imports(tree):
            if top in extra_names and top not in guarded:
                key = f"{relative}:{target}"
                found[key] = found.get(key, 0) + 1
    return found


def _ledger_grants() -> dict[str, int]:
    """Parse ``_UNGUARDED_DEBT`` into ``"path.py:dotted.module" -> count``.

    An entry with no ``#N`` grants one site, which keeps the common line short;
    ``#N`` states how many sites the ledger tolerates for that file and module,
    so a repeated key cannot quietly pre-approve a new import (#1321).
    """
    grants: dict[str, int] = {}
    for entry in _UNGUARDED_DEBT:
        key, _, count = entry.partition("#")
        grants[key] = grants.get(key, 0) + int(count or 1)
    return grants


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
    sites = _tree_unguarded_sites()
    granted = _ledger_grants()
    new_debt = sorted(
        f"{key} (x{count})" for key, count in sites.items()
        if count > granted.get(key, 0)
    )
    assert not new_debt, (
        "new unguarded extra imports (wrap them in "
        '"with optional_extra("<name>"):" before the import):\n  '
        + "\n  ".join(new_debt)
    )


def test_ledger_has_no_stale_entries():
    """A ledger line is a standing grant, so it must not outlive its import.

    Dropping these to a warning (#1320) let a stale ``path:package`` line keep
    pre-approving its file: the pair stayed in the ledger, so a *new* unguarded
    import of that package in that file compared as already-known and passed
    (#1321). Split out as its own test rather than folded back into the scan so
    the failure names whose fault it is: ledger drift, not the diff that
    happened to run first. Delete the lines it names.
    """
    sites = _tree_unguarded_sites()
    stale = sorted(
        f"{key} (granted x{grant}, found x{sites.get(key, 0)})"
        for key, grant in _ledger_grants().items()
        if sites.get(key, 0) < grant
    )
    assert not stale, (
        "stale _UNGUARDED_DEBT entries — ledger drift, not your change. The "
        "tree guards these imports or no longer has them, and a line left "
        "behind pre-approves its file; delete these from _UNGUARDED_DEBT:\n  "
        + "\n  ".join(stale)
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


def test_a_stale_ledger_entry_names_the_lines_and_fails(monkeypatch):
    """#1321: it must fail, and the message must say it is ledger drift."""
    stale = "tests/cognition/gone.py:corvidae_cognition"
    monkeypatch.setattr(
        sys.modules[__name__], "_UNGUARDED_DEBT", _UNGUARDED_DEBT | {stale}
    )
    with pytest.raises(AssertionError) as excinfo:
        test_ledger_has_no_stale_entries()
    message = str(excinfo.value)
    assert stale in message
    assert "ledger drift, not your change" in message


def _imports_of(source: str):
    return list(_import_time_imports(ast.parse(source)))


def test_module_scope_importorskip_vouches_for_its_package():
    """#1317: ``pytest.importorskip("<pkg>")`` at module scope is a guard.

    It stands the whole module down when the package is missing, which is the
    outcome ``optional_extra`` gives; only the spelling differs.
    """
    entries = _imports_of(
        "import pytest\n"
        "pytest.importorskip('corvidae_vision')\n"
        "from corvidae_vision import Vision\n"
    )
    first = next(e for e in entries if e[1] == "corvidae_vision")
    lineno, guarded = first[0], first[2]
    assert lineno == 3
    assert "corvidae_vision" in guarded


def test_importorskip_for_a_different_package_vouches_for_nothing():
    """A skip on one package must not launder an import of another."""
    entries = _imports_of(
        "import pytest\n"
        "pytest.importorskip('playwright.async_api')\n"
        "from corvidae_vision import Vision\n"
    )
    guarded = next(e for e in entries if e[1] == "corvidae_vision")[2]
    assert "corvidae_vision" not in guarded


def test_importorskip_vouches_only_for_the_imports_below_it():
    """Order matters: a skip after the import cannot have guarded it."""
    entries = _imports_of(
        "from corvidae_vision import Vision\n"
        "import pytest\n"
        "pytest.importorskip('corvidae_vision')\n"
    )
    guarded = next(e for e in entries if e[1] == "corvidae_vision")[2]
    assert "corvidae_vision" not in guarded


def test_the_ledger_literal_has_no_duplicate_lines():
    """A duplicate entry is invisible in the frozenset, so check the source.

    #1321 measured the literal holding two lines its set had already collapsed.
    """
    source = pathlib.Path(__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "_UNGUARDED_DEBT" for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            literal = node.value.args[0]
            entries = [e.value for e in literal.elts]
            break
    else:  # pragma: no cover - the assignment moved
        raise AssertionError("_UNGUARDED_DEBT literal not found")
    duplicates = sorted({e for e in entries if entries.count(e) > 1})
    assert not duplicates, f"duplicate _UNGUARDED_DEBT lines: {duplicates}"


def test_a_second_site_of_a_granted_key_needs_its_count(monkeypatch):
    """The ``#N`` count is what a repeated key relies on (#1321).

    Without ``#N`` a ledger line grants exactly one site, so a second unguarded
    import of the same dotted module in the same file has to show up as new
    debt. This drives the arithmetic directly so the branch is pinned by a test
    rather than by a manual run in a dirty worktree.
    """
    key = "tests/cognition/example.py:corvidae_cognition.goals"
    monkeypatch.setattr(sys.modules[__name__], "_tree_unguarded_sites", lambda: {key: 2})
    monkeypatch.setattr(sys.modules[__name__], "_UNGUARDED_DEBT", frozenset({key}))
    with pytest.raises(AssertionError) as excinfo:
        test_no_new_unguarded_extra_imports()
    assert f"{key} (x2)" in str(excinfo.value)

    monkeypatch.setattr(sys.modules[__name__], "_UNGUARDED_DEBT", frozenset({f"{key}#2"}))
    test_no_new_unguarded_extra_imports()  # granted at that count: no failure

    monkeypatch.setattr(sys.modules[__name__], "_tree_unguarded_sites", lambda: {key: 1})
    with pytest.raises(AssertionError) as excinfo:
        test_ledger_has_no_stale_entries()
    assert "granted x2, found x1" in str(excinfo.value)
