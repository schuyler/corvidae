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
import gc
import inspect
import logging
import warnings
from unittest.mock import AsyncMock

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


class _ChannelOnly:
    """Declares only on_notify's channel, so a call that omits the rest can reach it."""

    @hookimpl
    async def on_notify(self, channel):
        return "channel-only"


async def test_missing_argument_leaves_no_unawaited_coroutine() -> None:
    """#1116: a broadcast that omits an argument raises with no coroutine built.

    apluggy calls implementations newest first, so _ChannelOnly's coroutine
    would exist before _Good's missing ``source`` raised and was never awaited.
    """
    pm = create_plugin_manager()
    pm.register(_Good(), name="good")
    pm.register(_ChannelOnly(), name="channel_only")

    # Record every warning; a leaked coroutine shows up as a RuntimeWarning
    # once it is garbage-collected.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(pluggy.HookCallError, match="'source'"):
            await pm.ahook.on_notify(channel=None)
        gc.collect()

    leaked = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert leaked == []


async def test_registered_impl_is_still_a_coroutine_function() -> None:
    """R5: a wrapped async impl still reports as one, so AsyncMock(wraps=...) awaits it."""
    bad = _Bad()
    pm = create_plugin_manager()
    pm.register(bad, name="bad")

    assert inspect.iscoroutinefunction(bad.on_notify)
    with pytest.raises(ValueError, match="boom"):
        await AsyncMock(wraps=bad.on_notify)(**_NOTIFY)


# --- #1114: a wrapper that slipped past the startup check ---------------------


async def test_dispatch_refuses_a_wrapper_the_startup_check_never_saw() -> None:
    """A manager that never ran `check_fail_soft_wrappers` can still dispatch.

    The name snapshot includes the wrapper, apluggy returns one result per
    non-wrapper, and `zip(strict=True)` then raised a bare ValueError naming
    neither the hook nor the plugin. Dispatch now refuses with the same
    PluginValidationError the startup check raises.
    """

    class Wrapping:
        @hookimpl(wrapper=True)
        def on_web_thing(self, value: int):
            yield

    pm = _web_pm()  # no check_fail_soft_wrappers() here, by construction
    pm.register(Wrapping(), name="wrapping")

    with pytest.raises(pluggy.PluginValidationError) as excinfo:
        await pm.ahook.on_web_thing(value=1)

    message = str(excinfo.value)
    assert "on_web_thing" in message
    assert "'wrapping'" in message
    assert "wrapper=True" in message
    assert "Remove" in message


async def test_a_plain_fail_soft_hook_still_dispatches_after_the_refusal() -> None:
    """The refusal is per call, so nothing about the manager is left broken."""

    class Wrapping:
        @hookimpl(wrapper=True)
        def on_web_thing(self, value: int):
            yield

    class Plain:
        @hookimpl
        async def on_web_other(self, value: int) -> int:
            return value + 1

    pm = _web_pm()
    pm.register(Wrapping(), name="wrapping")
    pm.register(Plain(), name="plain")

    with pytest.raises(pluggy.PluginValidationError):
        await pm.ahook.on_web_thing(value=1)
    assert await pm.ahook.on_web_other(value=1) == [2]


# --- #1113: a static hookimpl keeps its marker through wrapping --------------


async def test_a_static_hookimpl_survives_re_registration() -> None:
    """`@hookimpl` above `@staticmethod` keeps its marker when the impl is wrapped.

    The marker sits on the staticmethod object, which `getattr` unwraps, so the
    wrapper used to carry a `__dict__` without it: registering the same plugin
    with a second manager found no marker and pluggy dropped the impl silently.
    """

    class Static:
        @hookimpl
        @staticmethod
        async def on_notify(channel, source, text, tool_call_id, annotations):
            return "static"

    plugin = Static()
    pm1 = create_plugin_manager()
    pm1.register(plugin, name="static")
    assert len(pm1.hook.on_notify.get_hookimpls()) == 1

    pm2 = create_plugin_manager()
    pm2.register(plugin, name="static")
    assert len(pm2.hook.on_notify.get_hookimpls()) == 1
    # And the impl is the wrapped one, not the bare staticmethod.
    [impl] = pm2.hook.on_notify.get_hookimpls()
    assert getattr(impl.function, "_corvidae_fail_soft", False)

    assert await pm2.ahook.on_notify(
        channel="irc:#test", source="t", text="x", tool_call_id=None, annotations={},
    ) == ["static"]


async def test_a_static_hookimpl_survives_unregister_and_register() -> None:
    """The same marker survives an unregister/register cycle on one manager."""

    class Static:
        @hookimpl
        @staticmethod
        async def on_notify(channel, source, text, tool_call_id, annotations):
            return "static"

    plugin = Static()
    pm = create_plugin_manager()
    pm.register(plugin, name="static")
    pm.unregister(plugin)
    pm.register(plugin, name="static")

    assert len(pm.hook.on_notify.get_hookimpls()) == 1


async def test_a_slots_plugin_with_a_hookimpl_slot_registers() -> None:
    """An impl held in a `__slots__` *slot* registers, and re-registers.

    This is the shape pluggy supports and the one a hand-written marker lookup
    broke with `TypeError: vars() argument must have __dict__ attribute`
    (#1142 review). A `__slots__` plugin that keeps the impl as a class-body
    method is a different shape: it is refused at `setattr` by design, because a
    read-only slot cannot be wrapped and running it unwrapped is worse (#1104).
    """

    async def on_notify(channel, source, text, tool_call_id, annotations):
        return "slot"

    class SlotPlugin:
        __slots__ = ("on_notify",)

    plugin = SlotPlugin()
    plugin.on_notify = hookimpl(on_notify)

    pm1 = create_plugin_manager()
    pm1.register(plugin, name="slot")
    assert len(pm1.hook.on_notify.get_hookimpls()) == 1

    pm2 = create_plugin_manager()
    pm2.register(plugin, name="slot")
    assert len(pm2.hook.on_notify.get_hookimpls()) == 1


async def test_an_inherited_static_hookimpl_survives_re_registration() -> None:
    """The impl lives on a base class; pluggy's lookup walks the MRO, so ours must."""

    class Base:
        @hookimpl
        @staticmethod
        async def on_notify(channel, source, text, tool_call_id, annotations):
            return "inherited"

    class Sub(Base):
        pass

    plugin = Sub()
    pm1 = create_plugin_manager()
    pm1.register(plugin, name="sub")
    assert len(pm1.hook.on_notify.get_hookimpls()) == 1

    pm2 = create_plugin_manager()
    pm2.register(plugin, name="sub")
    assert len(pm2.hook.on_notify.get_hookimpls()) == 1


async def test_an_instance_attribute_keeps_pluggy_precedence() -> None:
    """An instance value shadows a marked class attribute for pluggy; it must here too.

    Checking the class `__dict__` first gave the wrapper the *class's* options for
    an impl pluggy took from the instance, which only shows on the next
    registration: the class marker says `trylast`, so the order the second
    manager dispatches in is the assertion (#1142 review).
    """

    class Marked:
        @hookimpl(trylast=True)
        @staticmethod
        async def on_notify(channel, source, text, tool_call_id, annotations):
            return "class"

    async def from_instance(channel, source, text, tool_call_id, annotations):
        return "instance"

    class Other:
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            return "other"

    plugin = Marked()
    plugin.on_notify = hookimpl(from_instance)

    pm1 = create_plugin_manager()
    pm1.register(plugin, name="shadow")
    pm2 = create_plugin_manager()
    # Other first, so plain (non-trylast) order is shadow then other: a leaked
    # class marker with trylast would put shadow last and flip this.
    pm2.register(Other(), name="other")
    pm2.register(plugin, name="shadow")

    [shadow_impl] = [
        impl for impl in pm2.hook.on_notify.get_hookimpls()
        if impl.plugin_name == "shadow"
    ]
    assert not shadow_impl.trylast  # the instance marker, not the class one
    assert await pm2.ahook.on_notify(
        channel="irc:#test", source="t", text="x", tool_call_id=None, annotations={},
    ) == ["instance", "other"]


# --- #1096: a sync implementation of an async hook ---------------------------


def test_a_sync_implementation_of_an_async_hook_is_refused() -> None:
    """Refused at startup, naming the hook, the plugin and the fix (#1096).

    Left to run, its return value fails asyncio.gather and every sibling's
    result is lost, so the check is cheaper and louder than the symptom.
    """

    class Sync:
        @hookimpl
        def on_notify(self, channel, source, text, tool_call_id, annotations):
            return None

    pm = create_plugin_manager()
    pm.register(Sync(), name="sync")

    with pytest.raises(pluggy.PluginValidationError) as excinfo:
        hooks.check_async_hookimpls(pm)

    message = str(excinfo.value)
    assert "on_notify" in message
    assert "'sync'" in message
    assert "async def" in message
    assert "remove the implementation" in message


def test_a_sync_implementation_of_a_sync_hook_is_accepted() -> None:
    """Sync-spec hooks go through pm.hook and are untouched by the check."""

    class Sync:
        @hookimpl
        def register_tools(self, tool_registry):
            return None

    pm = create_plugin_manager()
    pm.register(Sync(), name="sync")

    hooks.check_async_hookimpls(pm)  # does not raise


# --- #1096: the check runs at startup ----------------------------------------


async def test_startup_refuses_a_sync_implementation_of_an_async_hook(
    monkeypatch,
) -> None:
    """The refusal happens in load_plugins, not only in the check itself (#1096).

    A check nothing calls is a comment. This pins the wiring: the plugin is
    already registered on the manager `create_plugin_manager` hands back, so the
    only thing that can refuse it is the startup step.
    """

    from corvidae import runtime as runtime_module

    class Sync:
        @hookimpl
        def on_notify(self, channel, source, text, tool_call_id, annotations):
            return None

    def manager_with_the_sync_plugin():
        pm = create_plugin_manager()
        pm.register(Sync(), name="sync")
        return pm

    monkeypatch.setattr(
        runtime_module, "create_plugin_manager", manager_with_the_sync_plugin
    )

    rt = runtime_module.Runtime()
    config = {
        "llm": {
            "providers": {"main": {"base_url": "http://localhost:8080"}},
            "profiles": {"main": {"model": "test-model"}},
        },
    }
    with pytest.raises(pluggy.PluginValidationError) as excinfo:
        await rt.load_plugins(config)

    message = str(excinfo.value)
    assert "on_notify" in message
    assert "'sync'" in message


# --- #1097: a raising implementation never stops the others -------------------


async def test_one_raising_on_stop_does_not_stop_the_others(caplog) -> None:
    """A plugin whose on_stop raises must not prevent the rest (#1097).

    `on_stop` is an observe hook, so the kernel's dispatch logs the failure and
    keeps going; runtime._run_shutdown calls it unguarded for exactly that
    reason.
    """
    import logging

    from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl

    ran: list[str] = []

    class Raising(CorvidaePlugin):
        @hookimpl
        async def on_stop(self) -> None:
            raise RuntimeError("shutdown boom")

    class Recording(CorvidaePlugin):
        @hookimpl
        async def on_stop(self) -> None:
            ran.append("recording")

    pm = create_plugin_manager()
    pm.register(Raising(), name="raising")
    pm.register(Recording(), name="recording")

    with caplog.at_level(logging.WARNING, logger="corvidae"):
        await pm.ahook.on_stop()

    assert ran == ["recording"]
    assert "raising" in caplog.text  # logged against the plugin, not a bare hook name


async def test_a_raising_metrics_subscriber_does_not_reach_the_emitter(caplog) -> None:
    """The kernel's own metric emitter needs no guard of its own (#1097)."""
    import logging

    from corvidae.hooks import CorvidaePlugin, hookimpl
    from corvidae.metrics import MetricsPlugin

    class Raising(CorvidaePlugin):
        @hookimpl
        async def on_metrics(self, name, value, tags):
            raise RuntimeError("subscriber boom")

    pm = create_plugin_manager()
    pm.register(Raising(), name="raising")
    emitter = MetricsPlugin(pm=pm)

    with caplog.at_level(logging.WARNING, logger="corvidae"):
        await emitter._emit("some.metric", 1.0, {"k": "v"})  # does not raise

    assert "raising" in caplog.text
