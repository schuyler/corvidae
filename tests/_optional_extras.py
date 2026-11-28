"""Skip markers for tests that need an optional extra (#1287).

The acp case (#1281, `tests/_acp_extra.py`) established the discipline; this
module names it once so the extras whose *own plugin package* is the optional
import get the same care instead of a marker that swallows whatever import error
happens to arrive.

Why the shape differs from `tests/_acp_extra.py`. The acp tests import the
third-party `acp` inside functions, so a module-level `pytestmark =
requires_acp` is enough to stand them down. The modules here import the plugin
package at module scope (`from context_panel import ContextPanelPlugin`), which
runs during collection, before any mark on the module can be consulted. A skip
therefore has to happen *before that import line executes*, hence the guard:

    from _optional_extras import optional_extra

    with optional_extra("context_panel"):
        from context_panel import ContextPanelPlugin

`with` is not decoration: a `with` (or `try`) block is what keeps the guarded
import legal under ruff's E402 while a bare statement before it would not be.
The name binds at module scope as usual, so the rest of the file is unchanged.

The distinction the whole module exists to keep: an extra that is **not
installed** versus a package that is **present but broken or shadowed**. Skipping
the first is correct — an environment gap is not a code regression. Skipping the
second hides a real failure behind a green line, which is worse than the
collection error it replaced. So the probe acts only on a `ModuleNotFoundError`
naming the probed module; a missing transitive dependency propagates with its own
`name`, and so does `ImportError: cannot import name … (unknown location)`, the
signature of a directory on `pythonpath` shadowing the real package as a
namespace package.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Iterator

import pytest

#: Skip lines name the extra to install, so a green skip stays actionable.
# The table is module name -> installable extra, and it is the guard's only source
# for the skip line. Both halves are tied to the manifest by
# tests/test_optional_extras.py: the extra must be declared in pyproject, and it
# must be the extra that provides the module being probed — either named after the
# dash-normalized module, or depending on it.
#
# The wording is an instruction, and it is phrased as an addition rather than as a
# whole command because `uv sync` is exact. Measured on this checkout at 08:00Z, a
# bare `uv sync --locked --extra inotify` would have removed `agent-client-protocol`
# from an environment that had it, so a skip line must not read as "run this and
# nothing else" while it hands the reader a working command.
_EXTRAS = {
    "cross_channel_send": "cross-channel-send",
    "corvidae_whatsapp": "whatsapp",
    "context_panel": "context-panel",
    "webhook_notify": "webhook-notify",
    "github_webhook": "github-webhook",
    "goals_notify": "goals-notify",
    "outbound_cap": "outbound-cap",
    # Package-shaped guards: the probe names the import, and the extra that
    # provides it is named differently. The rounds on #1341 and #1342 both found
    # the fallback printing the import name where a reader needed the extra name.
    "corvidae_cognition": "cognition",
    "corvidae_inotify": "inotify",
}
_REASONS = {
    module: (
        f"optional {module} extra not installed; add --extra {extra} to "
        "uv sync --locked (exact: unlisted extras are dropped)"
    )
    for module, extra in _EXTRAS.items()
}


def _reason(module: str) -> str:
    """The skip line for an absent extra.

    One generator — the `_REASONS` comprehension above — owns the whole sentence, and
    a table row supplies only its module-to-extra pair. That is what lets
    `test_every_reason_names_the_extra_that_provides_its_module` require exactly one
    `--extra` token in every line and require that token to name a declared extra which
    provides the module. Of the row's prose, the only part that check inspects is the
    token: the capture is `[A-Za-z0-9][A-Za-z0-9_-]*`, so a token ends at the first
    character outside that class, and the assertion counts how many such tokens appear.
    Appending text after the clause leaves that test green, while adding an additional
    `--extra` token fails on `assert len(hinted) == 1` — both measured, with the count
    visible in the failure output. A row may therefore carry prose the check never reads,
    but it cannot go silent about its extra or mention two.

    The generic line below is the fallback for an unlisted module, and the only place in
    this module that emits the phrase "the plugin cannot be imported"; every table row is
    the generator's template, whose text names `uv sync --locked` and ends in the
    parenthetical "(exact: unlisted extras are dropped)". Outside this module the same
    opening — "optional <module> extra not installed;" — survives in a literal the
    generator never sees, and its tail differs: `tests/_acp_extra.py` marks its skips with
    "optional acp extra not installed; the ACP transport cannot be imported".

    It is what the guard's own tests drive with a name deliberately absent from the table.
    """
    known = _REASONS.get(module)
    if known:
        return known
    return f"optional {module} extra not installed; the plugin cannot be imported"


def extra_importable(module: str) -> bool:
    """True when ``module`` imports, False only when it is genuinely absent.

    Raises whatever the import raises for every other failure, because a
    half-installed or shadowed package must not be reported as "not installed".
    """
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # Only the module named in the probe means "extra not installed". A
        # present-but-broken package raises ModuleNotFoundError for one of *its*
        # imports, and swallowing that would skip these tests for a false reason.
        if exc.name != module:
            raise
        return False
    return True


@contextmanager
def optional_extra(module: str) -> Iterator[None]:
    """Guard the module-scope import of ``module``; skip the module if absent.

    The skip is raised before the guarded import runs, which is the only moment
    that matters for a module-scope import. Any import failure other than a
    missing ``module`` propagates from the import itself, untouched.
    """
    if not extra_importable(module):
        pytest.skip(_reason(module), allow_module_level=True)
    yield


def requires_extra(module: str):
    """A ``skipif`` marker for tests whose extra is imported inside a function.

    Use `optional_extra` instead when the plugin package is imported at module
    scope; a mark cannot stand down an import that already failed.
    """
    return pytest.mark.skipif(not extra_importable(module), reason=_reason(module))
