"""Registration-time guard against the pluggy silent-drop bug class.

pluggy only forwards a caller's value to a hookimpl param when that param
has NO default in the hookimpl's own signature. A hookimpl that re-declares
a spec-REQUIRED param WITH a default (e.g. ``config=None``) silently
receives its own default instead of the caller's value — pluggy raises no
error, and pluggy's own ``_verify_hook`` does not catch it either.

These tests exercise ``_check_hook_arg_binding``, a guard that walks every
registered hookcaller after registration and raises PluginValidationError
when an impl defaults a param the spec declares as required.
"""
from __future__ import annotations

import apluggy as pluggy
import pytest

from corvidae.hooks import create_plugin_manager, hookimpl, hookspec
from corvidae.hooks import _check_hook_arg_binding


class _Spec:
    """Minimal hookspec with one required and one optional param."""

    @hookspec
    async def do_thing(self, required_param, optional_param=None):
        """required_param has no default (spec-required).

        optional_param has a default (spec-optional) — impls are free to
        default it too without dropping caller-supplied values.
        """


def _make_pm() -> pluggy.PluginManager:
    # Project name must match the "corvidae" HookspecMarker/HookimplMarker
    # namespace used by the hookspec/hookimpl decorators imported above.
    # This is a fresh, isolated PluginManager instance distinct from the
    # real corvidae.hooks.create_plugin_manager()'s pm.
    pm = pluggy.PluginManager("corvidae")
    pm.add_hookspecs(_Spec)
    return pm


class TestSilentDropDetected:
    """(a) POSITIVE: impl defaults a spec-REQUIRED param -> raise."""

    def test_impl_defaulting_required_param_raises(self):
        pm = _make_pm()

        class BadPlugin:
            @hookimpl
            async def do_thing(self, required_param=None, optional_param=None):
                pass

        pm.register(BadPlugin(), name="bad_plugin")

        with pytest.raises(pluggy.PluginValidationError) as excinfo:
            _check_hook_arg_binding(pm)

        message = str(excinfo.value)
        assert "bad_plugin" in message
        assert "do_thing" in message
        assert "required_param" in message


class TestNoFalsePositives:
    """(b) and (c): legitimate defaulting patterns must not raise."""

    def test_impl_defaulting_spec_optional_param_does_not_raise(self):
        """(b) Defaulting a param the spec ALSO defaults is legitimate."""
        pm = _make_pm()

        class GoodPlugin:
            @hookimpl
            async def do_thing(self, required_param, optional_param=None):
                pass

        pm.register(GoodPlugin(), name="good_plugin")

        # Should not raise.
        _check_hook_arg_binding(pm)

    def test_impl_omitting_param_entirely_does_not_raise(self):
        """(c) pluggy tolerates an impl that omits a param entirely."""
        pm = _make_pm()

        class MinimalPlugin:
            @hookimpl
            async def do_thing(self, required_param):
                pass

        pm.register(MinimalPlugin(), name="minimal_plugin")

        # Should not raise.
        _check_hook_arg_binding(pm)

    def test_impl_with_no_params_does_not_raise(self):
        """An impl that declares none of the hook's params at all is fine."""
        pm = _make_pm()

        class NoOpPlugin:
            @hookimpl
            async def do_thing(self):
                pass

        pm.register(NoOpPlugin(), name="noop_plugin")

        # Should not raise.
        _check_hook_arg_binding(pm)


class TestRealPluginManagerIntegration:
    """(d) The real, fully-populated plugin manager must pass the guard.

    create_plugin_manager() itself only registers the internal
    _seed_hooks plugin; the real corvidae plugins are loaded afterwards
    by runtime.py via pm.load_setuptools_entrypoints("corvidae"). This
    test reproduces that same sequence so the guard is actually exercised
    against the full real plugin set, not just the seed plugin.
    """

    def test_real_plugin_manager_passes_guard(self):
        pm = create_plugin_manager()
        pm.load_setuptools_entrypoints("corvidae")

        # Should not raise. If it does, STOP: this indicates a live
        # instance of the silent-drop bug class in a real, registered
        # plugin — do not paper over it here.
        _check_hook_arg_binding(pm)


class TestRuntimeStartupCoverage:
    """Prove the guard actually runs at real startup via the runtime path.

    runtime.Runtime.start() calls create_plugin_manager(), then
    pm.load_setuptools_entrypoints("corvidae"), then
    _check_hook_arg_binding(pm) before validate_dependencies. Without that
    second guard call, the real entry-point plugins would never be checked
    at startup — create_plugin_manager() only registers _seed_hooks. These
    tests reproduce that exact sequence and confirm both that the real set
    passes and that a genuine violation is caught at that point.
    """

    def test_runtime_imports_and_calls_the_guard(self):
        """The guard is imported into runtime and invoked after entry-point
        loading — not just defined in hooks.py. Guards against the placement
        regression where the runtime path silently skips the check."""
        import inspect

        import corvidae.runtime as runtime

        # The function is imported into runtime's namespace.
        assert runtime._check_hook_arg_binding is _check_hook_arg_binding

        # It is actually invoked inside Runtime.start(), after
        # load_setuptools_entrypoints (so the real plugins are registered
        # by the time it runs).
        src = inspect.getsource(runtime.Runtime.start)
        assert "_check_hook_arg_binding(self.pm)" in src
        load_idx = src.index("load_setuptools_entrypoints")
        guard_idx = src.index("_check_hook_arg_binding(self.pm)")
        assert guard_idx > load_idx, (
            "guard must run AFTER entry-point plugins are registered"
        )

    def test_startup_sequence_passes_for_real_plugins(self):
        """Reproduce the runtime registration order and confirm the full
        real plugin set passes the guard at that point."""
        pm = create_plugin_manager()
        pm.load_setuptools_entrypoints("corvidae")

        # This is the line runtime.Runtime.start() runs at step 8b.
        _check_hook_arg_binding(pm)

    def test_bad_plugin_registered_at_startup_is_caught(self):
        """A silent-drop violation introduced among the real plugins (i.e.
        registered on a real spec after entry-point loading) is caught by
        the guard at the runtime call site — not silently tolerated."""
        pm = create_plugin_manager()
        pm.load_setuptools_entrypoints("corvidae")

        # on_message(self, channel, sender, text) declares `text` as a
        # spec-required param (no default). An impl that redeclares it WITH
        # a default silently drops the caller's value — exactly the bug
        # class. Register it as a real plugin would be, then run the
        # runtime-path guard.
        class SilentDropPlugin:
            @hookimpl
            async def on_message(self, channel, sender, text=None):
                pass

        pm.register(SilentDropPlugin(), name="silent_drop_plugin")

        with pytest.raises(pluggy.PluginValidationError) as excinfo:
            _check_hook_arg_binding(pm)

        message = str(excinfo.value)
        assert "silent_drop_plugin" in message
        assert "on_message" in message
        assert "text" in message


class TestEnrichedOnAgentResponseParams:
    """WP2.1 review-gate regression: logprobs/withheld are spec-REQUIRED.

    The enriched on_agent_response params were originally declared with
    defaults, which exempts them from this guard — an impl mirroring the
    defaulted signature would silently receive None forever (e.g. WP2.9's
    withheld=True on an enforce-on veto, seen as None by every consumer).
    The spec is now default-free; these tests pin both the guard behavior
    and the real end-to-end delivery.
    """

    def test_impl_defaulting_logprobs_or_withheld_is_flagged(self):
        pm = create_plugin_manager()

        class MirroringPlugin:
            @hookimpl
            async def on_agent_response(
                self, channel, request_text, response_text, exchange_key,
                origin, originating_text, logprobs=None, withheld=None,
            ):
                pass

        pm.register(MirroringPlugin(), name="mirroring_plugin")

        with pytest.raises(pluggy.PluginValidationError) as excinfo:
            _check_hook_arg_binding(pm)

        message = str(excinfo.value)
        assert "logprobs" in message
        assert "withheld" in message

    async def test_default_free_consumer_receives_logprobs_and_withheld(self):
        """A real default-free hookimpl receives the caller's values through
        actual pluggy dispatch — not an AsyncMock'd hookcaller."""
        pm = create_plugin_manager()
        received = {}

        class Consumer:
            @hookimpl
            async def on_agent_response(
                self, channel, request_text, response_text, exchange_key,
                origin, originating_text, logprobs, withheld,
            ):
                received.update(logprobs=logprobs, withheld=withheld)

        pm.register(Consumer(), name="consumer")
        _check_hook_arg_binding(pm)  # sanity: the consumer passes the guard

        await pm.ahook.on_agent_response(
            channel=None,
            request_text="req",
            response_text="resp",
            exchange_key="ek-1",
            origin="user",
            originating_text="req",
            logprobs={"content": [{"token": "resp", "logprob": -0.1}]},
            withheld=True,
        )

        assert received["logprobs"] == {"content": [{"token": "resp", "logprob": -0.1}]}
        assert received["withheld"] is True
