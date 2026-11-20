"""Plugin classes for tests/test_plugin_loader.py (#1107); not collected.

Each class stands in for a bundled plugin or a replacement target. Entry
points in the tests point at these by ``tests.plugin_loader_fixtures:<Name>``.
A sibling module, ``tests/plugin_loader_boom.py``, raises on import.
"""

from __future__ import annotations

from collections import Counter
from importlib.metadata import EntryPoint

from corvidae.hooks import CorvidaePlugin, hookimpl

# Module path the fake entry points resolve through.
FIX = "tests.plugin_loader_fixtures"

# Constructor calls per class name, so a test can prove a class was (or was
# never) instantiated. Tests reset it through an autouse fixture.
CALLS: Counter[str] = Counter()


def ep(name: str, target: str) -> EntryPoint:
    """A fake ``corvidae`` entry point; ``target`` is a class name or module path."""
    value = target if ("." in target or ":" in target) else f"{FIX}:{target}"
    return EntryPoint(name=name, value=value, group="corvidae")


def entry_points_for(*eps: EntryPoint):
    """An ``entry_points`` stand-in that fails unless asked for group "corvidae"."""

    def fake(**kw):
        # The loader must scan the corvidae group; any other group is a bug.
        assert kw.get("group") == "corvidae", (
            f"entry_points called with {kw!r}; Runtime must ask for group='corvidae'"
        )
        return list(eps)

    return fake


# What InitCapture saw under config["_plugin_load"] during on_init (R11 feed).
CAPTURED: dict[str, object] = {}


class InitCapture(CorvidaePlugin):
    """Records ``config["_plugin_load"]`` as seen inside its own on_init."""

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        # Capture at hook time; a key added after on_init would read None here.
        CAPTURED["report"] = config.get("_plugin_load")


class Bundled(CorvidaePlugin):
    """Stands in for the bundled plugin that owns a slot."""

    def __init__(self) -> None:
        # Record the instantiation under the concrete class name.
        CALLS[type(self).__name__] += 1


class OtherBundled(CorvidaePlugin):
    """A second, unrelated bundled plugin, for the two-slots case."""

    def __init__(self) -> None:
        CALLS[type(self).__name__] += 1


class GoodTarget(Bundled):
    """A valid replacement: subclasses the bundled class."""


class NotASub(CorvidaePlugin):
    """A plugin class that does not subclass Bundled."""

    def __init__(self) -> None:
        CALLS[type(self).__name__] += 1


class CtorBoom(Bundled):
    """A target whose constructor raises."""

    def __init__(self) -> None:
        CALLS[type(self).__name__] += 1
        raise RuntimeError("boom-ctor")


class BundledCtorBoom(CorvidaePlugin):
    """A bundled plugin whose constructor raises (both-failing case)."""

    def __init__(self) -> None:
        CALLS[type(self).__name__] += 1
        raise RuntimeError("boom-bundled-ctor")


class CtorBoomSub(BundledCtorBoom):
    """A target that subclasses the failing bundled class and also raises."""

    def __init__(self) -> None:
        CALLS[type(self).__name__] += 1
        raise RuntimeError("boom-target-ctor")


class Plain(CorvidaePlugin):
    """An ordinary plugin with no replacement involved."""

    def __init__(self) -> None:
        CALLS[type(self).__name__] += 1


class Plain2(Plain):
    """A second ordinary plugin."""
