"""The ``trace`` protocol plugin (#1179, R1, R2, R3 dependencies, R6).

``trace`` declares ``on_trace_record`` (observe) and
``resolve_trace_origin`` (resolve) and has one behaviour: a WARNING at start
when outcome_log is not loaded, naming the loaded reporters whose records
are not written.
"""

import logging
from importlib.metadata import entry_points

import pytest

from corvidae.hooks import create_plugin_manager, install_plugin_hookspecs, validate_dependencies
from tests.cognition.trace_helpers import TraceRecorder


def _pm_with_trace():
    # A plugin manager with the protocol plugin registered and its specs
    # installed through the startup path.
    from corvidae_cognition.trace import TracePlugin

    pm = create_plugin_manager()
    trace = TracePlugin()
    pm.register(trace, name="trace")
    return pm, trace


class TestSpecs:
    def test_registered_under_the_trace_entry_point(self):
        # Without the entry point the plugin is never loaded, silently.
        found = {ep.name: ep for ep in entry_points(group="corvidae")}
        assert found["trace"].value == "corvidae_cognition.trace:TracePlugin"

    async def test_hooks_installed_and_unanswered_without_implementers(self):
        # Startup installs both specs; with no implementer the observe hook
        # returns [] and the resolve hook None, so a reporter never crashes.
        pm, _ = _pm_with_trace()
        install_plugin_hookspecs(pm)
        assert await pm.ahook.on_trace_record(
            trace_id="t1", channel_id="irc:#a", source="gates",
            origin=None, record={"send": {"vetoed": True}},
        ) == []
        assert await pm.ahook.resolve_trace_origin(trace_id="t1") is None


class TestDisabledOutcomeLog:
    async def _start(self, pm, trace):
        # Install specs and run the protocol plugin's startup.
        install_plugin_hookspecs(pm)
        await trace.on_init(pm=pm, config={})
        await trace.on_start(config={})

    async def test_warns_naming_loaded_reporters_and_the_remedy(self, caplog):
        # R2/R6: with outcome_log disabled the operator is told which
        # records are lost and how to keep them, instead of silence.
        from corvidae_cognition.gates import GatePlugin
        from corvidae_cognition.silence import SilencePlugin

        pm, trace = _pm_with_trace()
        pm.register(GatePlugin(), name="gates")
        pm.register(SilencePlugin(), name="silence")
        with caplog.at_level(logging.WARNING):
            await self._start(pm, trace)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        text = warnings[0].getMessage()
        assert "outcome_log" in text
        assert "gates" in text and "silence" in text
        # Only loaded reporters are named, not a fixed list.
        assert "appraisal" not in text
        assert "plugins.disabled" in text

    async def test_no_warning_when_outcome_log_is_loaded(self, caplog):
        from corvidae_cognition.gates import GatePlugin
        from corvidae_cognition.outcome_log import OutcomeLogPlugin

        pm, trace = _pm_with_trace()
        pm.register(GatePlugin(), name="gates")
        pm.register(OutcomeLogPlugin(), name="outcome_log")
        with caplog.at_level(logging.WARNING):
            await self._start(pm, trace)
        assert not [r for r in caplog.records if "outcome_log" in r.getMessage()]

    async def test_firing_raises_nothing_and_other_listeners_still_hear(self):
        # R6: no outcome_log, one other listener (critique's role in #1149).
        pm, trace = _pm_with_trace()
        recorder = TraceRecorder()
        pm.register(recorder, name="listener")
        await self._start(pm, trace)
        await pm.ahook.on_trace_record(
            trace_id="t1", channel_id="irc:#a", source="gates",
            origin=None, record={"send": {"vetoed": True}},
        )
        assert recorder.records == [{
            "trace_id": "t1", "channel_id": "irc:#a", "source": "gates",
            "origin": None, "record": {"send": {"vetoed": True}},
        }]


class TestDependents:
    @pytest.mark.parametrize("module, cls", [
        ("appraisal", "AppraisalPlugin"),
        ("critique", "CritiquePlugin"),
        ("gates", "GatePlugin"),
        ("memory", "MemoryPlugin"),
        ("silence", "SilencePlugin"),
        ("outcome_log", "OutcomeLogPlugin"),
    ])
    def test_reporters_and_sink_depend_on_trace(self, module, cls):
        # R3/R4: each caller names the hook's owner, so a missing owner is a
        # startup error rather than an AttributeError at the call site.
        import importlib

        plugin_cls = getattr(importlib.import_module(f"corvidae_cognition.{module}"), cls)
        assert "trace" in plugin_cls.depends_on

    def test_startup_fails_naming_the_dependent_when_trace_is_disabled(self):
        # R6: trace disabled, gates loaded -> startup refuses, naming both.
        from corvidae_cognition.gates import GatePlugin

        pm = create_plugin_manager()
        pm.register(GatePlugin(), name="gates")
        with pytest.raises(RuntimeError) as excinfo:
            validate_dependencies(pm)
        assert "GatePlugin" in str(excinfo.value)
        assert "'trace'" in str(excinfo.value)
