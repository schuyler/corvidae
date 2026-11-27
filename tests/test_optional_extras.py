"""The optional-extra guard must tell an absent extra from a broken package (#1287).

What these tests pin is the failure mode that makes the guard safe to have: a
skip line is only honest when it fires for "not installed" and never for
"installed but broken or shadowed". A guard that treats any import error as an
environment gap turns a real regression into a green line, which is a worse
outcome than the collection error it replaced.
"""

from __future__ import annotations

import sys

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
