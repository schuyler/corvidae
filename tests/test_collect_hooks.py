"""The collect_ hook class and its gather helper (#1083, PRD §3.1.10).

A collect_ hook gathers one contribution per implementation for the caller
to combine. ``collect_results`` calls every implementation
concurrently and returns the non-None contributions in hook order
(tryfirst, the rest, trylast); one implementation's Exception is logged and
skipped, and a BaseException propagates.

Every test runs through a real plugin manager with the hook installed the
way startup installs it (``install_plugin_hookspecs``).
"""
from __future__ import annotations

import asyncio
import logging

import apluggy as pluggy
import pytest

from corvidae.hooks import (
    CorvidaePlugin,
    collect_results,
    create_plugin_manager,
    hookimpl,
    hookspec,
    install_plugin_hookspecs,
)

_HOOKS_LOGGER = "corvidae.hooks"


# ---------------------------------------------------------------------------
# Fixtures: an owner ("web") declaring one collect_ hook, and contributors
# ---------------------------------------------------------------------------


class WebSpec:
    """The collect_ hook owned by the test plugin registered as ``web``."""

    @hookspec
    async def collect_web_parts(self, value: int, extra: str) -> object | None:
        """Each implementer contributes one part, or None for nothing."""


class WebOwner(CorvidaePlugin):
    """Owner plugin: declares WebSpec."""

    hookspecs = WebSpec


def _contributor(result: object, **impl_opts: bool) -> object:
    """A plugin whose collect_web_parts returns ``result`` (async)."""

    class Contributor:
        @hookimpl(**impl_opts)
        async def collect_web_parts(self, value: int) -> object:
            return result

    return Contributor()


def _raiser(exc: BaseException) -> object:
    """A plugin whose collect_web_parts raises ``exc``."""

    class Raiser:
        @hookimpl
        async def collect_web_parts(self, value: int) -> object:
            raise exc

    return Raiser()


def _pm(*plugins: tuple[str, object]) -> pluggy.PluginManager:
    """A plugin manager with ``web`` and the given plugins, specs installed."""
    pm = create_plugin_manager()
    pm.register(WebOwner(), name="web")
    for name, plugin in plugins:
        pm.register(plugin, name=name)
    install_plugin_hookspecs(pm)
    return pm


# ---------------------------------------------------------------------------
# R2: hook order, None dropped
# ---------------------------------------------------------------------------


async def test_contributions_come_back_in_hook_order() -> None:
    """tryfirst, then the rest, then trylast, whatever the registration order."""
    pm = _pm(
        ("last", _contributor("last", trylast=True)),
        ("middle", _contributor("middle")),
        ("first", _contributor("first", tryfirst=True)),
    )

    parts = await collect_results(pm.hook.collect_web_parts, value=1, extra="x")

    assert parts == ["first", "middle", "last"]


async def test_none_contributes_nothing() -> None:
    """None is dropped; a falsy contribution such as 0 is still kept."""
    pm = _pm(("zero", _contributor(0)), ("abstains", _contributor(None)))

    parts = await collect_results(pm.hook.collect_web_parts, value=1, extra="x")

    assert parts == [0]


# ---------------------------------------------------------------------------
# R3: one implementation's Exception is logged and skipped
# ---------------------------------------------------------------------------


async def test_raising_implementation_is_logged_and_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pm = _pm(
        ("first", _contributor("first", tryfirst=True)),
        ("broken", _raiser(ValueError("boom"))),
        ("last", _contributor("last", trylast=True)),
    )

    with caplog.at_level(logging.WARNING, logger=_HOOKS_LOGGER):
        parts = await collect_results(pm.hook.collect_web_parts, value=1, extra="x")

    # The siblings' contributions survive.
    assert parts == ["first", "last"]
    # One WARNING names the hook and the plugin and carries the traceback.
    records = [r for r in caplog.records if r.name == _HOOKS_LOGGER]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert "collect_web_parts" in record.getMessage()
    assert "broken" in record.getMessage()
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], ValueError)


# ---------------------------------------------------------------------------
# R4: a BaseException propagates
# ---------------------------------------------------------------------------


async def test_cancelled_error_propagates() -> None:
    pm = _pm(
        ("a", _contributor("a")),
        ("cancels", _raiser(asyncio.CancelledError())),
    )

    with pytest.raises(asyncio.CancelledError):
        await collect_results(pm.hook.collect_web_parts, value=1, extra="x")


# ---------------------------------------------------------------------------
# R5: keyword-only arguments; each implementation gets what it declares
# ---------------------------------------------------------------------------


async def test_positional_hook_argument_is_refused() -> None:
    pm = _pm(("a", _contributor("a")))

    with pytest.raises(TypeError):
        await collect_results(pm.hook.collect_web_parts, 1, extra="x")  # type: ignore[misc]


async def test_implementation_receives_only_the_arguments_it_declares() -> None:
    """The caller passes value and extra; implementations declare a subset."""

    class ValueOnly:
        @hookimpl
        async def collect_web_parts(self, value: int) -> object:
            return ("value", value)

    class ExtraOnly:
        @hookimpl
        async def collect_web_parts(self, extra: str) -> object:
            return ("extra", extra)

    pm = _pm(("v", ValueOnly()), ("e", ExtraOnly()))

    parts = await collect_results(pm.hook.collect_web_parts, value=7, extra="x")

    assert sorted(parts) == [("extra", "x"), ("value", 7)]


# ---------------------------------------------------------------------------
# R6: sync implementations contribute too
# ---------------------------------------------------------------------------


async def test_sync_implementation_contributes() -> None:
    class SyncContributor:
        @hookimpl
        def collect_web_parts(self, value: int) -> object:
            return value + 1

    pm = _pm(("sync", SyncContributor()), ("a", _contributor("a", tryfirst=True)))

    parts = await collect_results(pm.hook.collect_web_parts, value=1, extra="x")

    assert parts == ["a", 2]


# ---------------------------------------------------------------------------
# R8: implementations run concurrently
# ---------------------------------------------------------------------------


async def test_implementations_run_concurrently() -> None:
    """Each implementation waits for the other to start; sequential would hang."""
    started_a, started_b = asyncio.Event(), asyncio.Event()

    class A:
        @hookimpl(tryfirst=True)
        async def collect_web_parts(self, value: int) -> object:
            started_a.set()
            await started_b.wait()
            return "a"

    class B:
        @hookimpl
        async def collect_web_parts(self, value: int) -> object:
            started_b.set()
            await started_a.wait()
            return "b"

    pm = _pm(("a", A()), ("b", B()))

    parts = await asyncio.wait_for(
        collect_results(pm.hook.collect_web_parts, value=1, extra="x"),
        timeout=2,
    )

    assert parts == ["a", "b"]
