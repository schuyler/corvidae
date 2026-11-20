"""Observe hooks fail soft by prefix; the rest fail hard (#1094).

Requirements 1-3 of the design: ``pm.ahook.<on_/before_/after_ hook>`` runs
every implementation, logs each ``Exception`` as a WARNING naming the hook and
the plugin, and returns the other values; ``on_init``, ``on_start`` and the
non-observe hooks stay hard-fail; ``pm.ahook_raise`` logs the same way but
re-raises the first failure once every sibling has finished; startup refuses a
wrapper on a fail-soft hook. Every test builds a real plugin manager.
"""

from __future__ import annotations

import asyncio
import logging

import apluggy as pluggy
import pytest

import corvidae.hooks as hooks

from corvidae.hooks import (
    CorvidaePlugin,
    create_plugin_manager,
    hookimpl,
    hookspec,
    install_plugin_hookspecs,
)

# on_notify's keyword arguments; the dispatcher only checks the names.
_NOTIFY = dict(channel=None, source="t", text="x", tool_call_id=None, annotations=None)


class _Good:
    """Returns a value from on_notify."""

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, annotations):
        return "good"


class _Bad:
    """Raises from on_notify."""

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, annotations):
        raise ValueError("boom")


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The WARNING records captured so far."""
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def _assert_logged_failure(record: logging.LogRecord, hook: str, plugin: str) -> None:
    """The record names the hook and the plugin and carries the traceback."""
    message = record.getMessage()
    assert hook in message
    assert plugin in message
    assert record.exc_info is not None or record.exc_text is not None


# --- Req 1: observe hooks fail soft -------------------------------------------


async def test_kernel_observe_hook_skips_failing_impl_and_logs(caplog) -> None:
    """A raising on_notify impl is logged; the other impl's value comes back."""
    pm = create_plugin_manager()
    pm.register(_Good(), name="good")
    pm.register(_Bad(), name="bad")

    with caplog.at_level(logging.WARNING):
        results = await pm.ahook.on_notify(**_NOTIFY)

    assert results == ["good"]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    _assert_logged_failure(warnings[0], "on_notify", "bad")


class _WebSpec:
    """Hooks owned by the plugin registered as ``web``."""

    @hookspec
    async def on_web_thing(self, value: int) -> str | None:
        """Observe hook."""


class _WebOwner(CorvidaePlugin):
    hookspecs = _WebSpec


class _WebGood:
    @hookimpl
    async def on_web_thing(self, value: int):
        return "ok"


class _WebBad:
    @hookimpl
    async def on_web_thing(self, value: int):
        raise RuntimeError("web boom")


def _web_pm() -> pluggy.PluginManager:
    """A manager with a plugin-owned observe hook and two implementers."""
    pm = create_plugin_manager()
    pm.register(_WebOwner(), name="web")
    pm.register(_WebGood(), name="webgood")
    pm.register(_WebBad(), name="webbad")
    install_plugin_hookspecs(pm)
    return pm


async def test_plugin_owned_observe_hook_fails_soft(caplog) -> None:
    """The same rule holds for an on_<plugin>_<x> hook a plugin declares."""
    pm = _web_pm()

    with caplog.at_level(logging.WARNING):
        results = await pm.ahook.on_web_thing(value=1)

    assert results == ["ok"]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    _assert_logged_failure(warnings[0], "on_web_thing", "webbad")


async def test_cancelled_error_propagates_after_sibling_finishes() -> None:
    """A BaseException propagates, but only after a slower sibling completed."""
    finished: list[str] = []

    class Slow:
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            await asyncio.sleep(0.05)
            finished.append("slow")

    class Cancelling:
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            raise asyncio.CancelledError

    pm = create_plugin_manager()
    pm.register(Slow(), name="slow")
    pm.register(Cancelling(), name="cancelling")

    with pytest.raises(asyncio.CancelledError):
        await pm.ahook.on_notify(**_NOTIFY)

    assert finished == ["slow"]


# --- pm.ahook_raise: log every failure, re-raise the first --------------------


async def test_ahook_raise_logs_each_failure_and_reraises_first(caplog) -> None:
    """ahook_raise logs every failure, finishes siblings, raises the first."""
    finished: list[str] = []

    class First:
        @hookimpl(tryfirst=True)
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            raise ValueError("first")

    class Second:
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            raise KeyError("second")

    class Slow:
        @hookimpl(trylast=True)
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            await asyncio.sleep(0.05)
            finished.append("slow")

    pm = create_plugin_manager()
    pm.register(First(), name="first")
    pm.register(Second(), name="second")
    pm.register(Slow(), name="slow")

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="first"):
        await pm.ahook_raise.on_notify(**_NOTIFY)

    assert finished == ["slow"]
    warnings = _warnings(caplog)
    assert len(warnings) == 2
    _assert_logged_failure(warnings[0], "on_notify", "first")
    _assert_logged_failure(warnings[1], "on_notify", "second")


async def test_ahook_raise_returns_values_when_nothing_fails() -> None:
    """With no failure ahook_raise behaves like ahook."""
    pm = create_plugin_manager()
    pm.register(_Good(), name="good")

    assert await pm.ahook_raise.on_notify(**_NOTIFY) == ["good"]


# --- Req 2: on_start and non-observe hooks stay hard-fail ---------------------


async def test_raising_on_start_propagates() -> None:
    """on_start is exempt from the prefix rule: a raise reaches the caller."""

    class BadStart:
        @hookimpl
        async def on_start(self, config):
            raise RuntimeError("start boom")

    pm = create_plugin_manager()
    pm.register(BadStart(), name="badstart")

    with pytest.raises(RuntimeError, match="start boom"):
        await pm.ahook.on_start(config={})


async def test_raising_on_init_propagates() -> None:
    """on_init is hard-fail too."""

    class BadInit:
        @hookimpl
        async def on_init(self, pm, config):
            raise RuntimeError("init boom")

    pm = create_plugin_manager()
    pm.register(BadInit(), name="badinit")

    with pytest.raises(RuntimeError, match="init boom"):
        await pm.ahook.on_init(pm=pm, config={})


async def test_raising_should_hook_propagates() -> None:
    """A should_ impl that raises propagates; veto hooks are not observe."""

    class BadGate:
        @hookimpl
        async def should_process_message(self, channel, sender, text, trace_id):
            raise RuntimeError("gate boom")

    pm = create_plugin_manager()
    pm.register(BadGate(), name="badgate")

    with pytest.raises(RuntimeError, match="gate boom"):
        await pm.ahook.should_process_message(
            channel=None, sender="s", text="t", trace_id="x"
        )


# --- Req 3: startup refuses a wrapper on a fail-soft hook ---------------------


@pytest.mark.parametrize("option", ["wrapper", "hookwrapper"])
def test_wrapper_on_kernel_observe_hook_is_refused(option: str) -> None:
    """The message names the hook, the plugin and the fix."""

    class Wrapping:
        @hookimpl(**{option: True})
        def on_notify(self, channel, source, text, tool_call_id, annotations):
            yield

    pm = create_plugin_manager()
    pm.register(Wrapping(), name="wrapping")

    with pytest.raises(pluggy.PluginValidationError) as excinfo:
        hooks.check_fail_soft_wrappers(pm)

    message = str(excinfo.value)
    assert "on_notify" in message
    assert "'wrapping'" in message
    assert f"{option}=True" in message
    assert "Remove" in message  # the remedy
    assert "failed implementation" in message  # the reason (#1104)


def test_wrapper_on_plugin_owned_observe_hook_is_refused() -> None:
    """Plugin-declared on_<plugin>_<x> hooks are covered as well."""

    class Wrapping:
        @hookimpl(wrapper=True)
        def on_web_thing(self, value: int):
            yield

    pm = _web_pm()
    pm.register(Wrapping(), name="wrapping")

    with pytest.raises(pluggy.PluginValidationError) as excinfo:
        hooks.check_fail_soft_wrappers(pm)

    message = str(excinfo.value)
    assert "on_web_thing" in message
    assert "'wrapping'" in message


def test_wrapper_on_on_start_is_not_refused() -> None:
    """on_start is hard-fail, so a wrapper there is legal."""

    class Wrapping:
        @hookimpl(wrapper=True)
        def on_start(self, config):
            yield

    pm = create_plugin_manager()
    pm.register(Wrapping(), name="wrapping")

    hooks.check_fail_soft_wrappers(pm)  # must not raise


# --- #1104: fail soft without apluggy's ahook_isolated -------------------------


class _Forbidden:
    """Stands in for pm.ahook_isolated; any use of it fails the test (#1104 R2)."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"pm.ahook_isolated.{name} was used")


def _pm_without_isolated(*plugins: tuple[object, str]) -> pluggy.PluginManager:
    """A real manager whose ahook_isolated can't be used, with plugins registered."""
    pm = create_plugin_manager()
    pm.ahook_isolated = _Forbidden()  # type: ignore[attr-defined]
    for plugin, name in plugins:
        pm.register(plugin, name=name)
    return pm


async def test_fail_soft_dispatch_does_not_need_ahook_isolated(caplog) -> None:
    """R2: ahook skips and logs a failing impl, ahook_raise re-raises it."""
    pm = _pm_without_isolated((_Good(), "good"), (_Bad(), "bad"))

    with caplog.at_level(logging.WARNING):
        results = await pm.ahook.on_notify(**_NOTIFY)
    assert results == ["good"]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    _assert_logged_failure(warnings[0], "on_notify", "bad")

    with pytest.raises(ValueError, match="boom"):
        await pm.ahook_raise.on_notify(**_NOTIFY)


async def test_base_exception_waits_for_siblings_without_ahook_isolated() -> None:
    """R1/R2: a CancelledError from one impl propagates after a slow sibling."""
    finished: list[str] = []

    class Slow:
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            await asyncio.sleep(0.05)
            finished.append("slow")

    class Cancelling:
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            raise asyncio.CancelledError

    pm = _pm_without_isolated((Slow(), "slow"), (Cancelling(), "cancelling"))

    with pytest.raises(asyncio.CancelledError):
        await pm.ahook.on_notify(**_NOTIFY)
    assert finished == ["slow"]


async def test_cancelling_the_caller_propagates_cancelled_error() -> None:
    """R4: cancelling the task awaiting a fail-soft call cancels the call."""
    started = asyncio.Event()

    class Waiting:
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            started.set()
            await asyncio.Event().wait()  # never set: only cancellation ends it

    pm = create_plugin_manager()
    pm.register(Waiting(), name="waiting")

    task = asyncio.create_task(pm.ahook.on_notify(**_NOTIFY))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_direct_call_to_a_registered_impl_still_raises() -> None:
    """R5: calling a registered plugin's hook method directly raises as before."""
    bad = _Bad()
    pm = create_plugin_manager()
    pm.register(bad, name="bad")

    with pytest.raises(ValueError, match="boom"):
        await bad.on_notify(**_NOTIFY)


async def test_direct_call_made_during_dispatch_still_raises() -> None:
    """R5: an impl that calls another plugin's hook method directly sees its raise."""
    bad = _Bad()
    seen: list[str] = []

    class Caller:
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            try:
                await bad.on_notify(**_NOTIFY)
            except ValueError:
                seen.append("raised")

    pm = create_plugin_manager()
    pm.register(bad, name="bad")
    pm.register(Caller(), name="caller")

    await pm.ahook.on_notify(**_NOTIFY)
    assert seen == ["raised"]


@pytest.mark.filterwarnings("ignore:Argument.*cannot be found:UserWarning")
async def test_direct_call_after_a_failed_dispatch_still_raises() -> None:
    """R5: a dispatch that raises leaves later direct calls in the task raising."""
    bad = _Bad()
    pm = create_plugin_manager()
    pm.register(bad, name="bad")

    with pytest.raises(pluggy.HookCallError):
        await pm.ahook.on_notify(channel=None)  # missing arguments

    with pytest.raises(ValueError, match="boom"):
        await bad.on_notify(**_NOTIFY)
