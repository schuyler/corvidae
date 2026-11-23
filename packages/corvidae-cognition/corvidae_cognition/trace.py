"""The ``trace`` protocol plugin: one event for reporting a fact about a trace.

Cognition plugins record what they decided about an exchange (a gate's
verdict, a critique outcome, a retrieval profile, appraisal's lens scores) by
firing ``on_trace_record``. Whoever listens hears it: outcome_log appends it
to its log, and other plugins may act on it. ``resolve_trace_origin`` asks
for a trace's origin (``user``, ``heartbeat``, ``task``, ...) when the asker
no longer holds it.

This plugin owns both specs and does nothing else, so a reporter depends on
it at no cost (docs/plugin-guide.md, "Firing a hook depends on its owner").
Its one behaviour is a startup WARNING when outcome_log is not loaded, so a
disabled log does not drop every record silently.

Shared types for record fragments that plugins other than the sink read live
in this module. None exists yet; the first is the held-draft fragment #1149
adds.
"""

import logging

from corvidae.hooks import CorvidaePlugin, hookimpl, hookspec

logger = logging.getLogger("corvidae.trace_plugin")


class TraceSpec:
    """Hooks the trace protocol plugin owns."""

    @hookspec
    async def on_trace_record(
        self,
        trace_id: str,
        channel_id: str,
        source: str,
        origin: str | None,
        record: dict,
    ) -> None:
        """Report one fact about a trace.

        ``source`` is the reporting plugin's name; listeners select records
        by it, so renaming a reporter breaks them. ``origin`` is the trace's
        origin when the reporter holds a propagated one, else None.
        ``record`` is a JSON-shaped dict whose shape belongs to the source.
        """

    @hookspec(firstresult=True)
    async def resolve_trace_origin(self, trace_id: str) -> str | None:
        """The origin recorded for ``trace_id``, or None when nobody knows it."""


class TracePlugin(CorvidaePlugin):
    """Declares the trace hooks; warns at start when outcome_log is absent."""

    hookspecs = TraceSpec
    # outcome_log implements the hooks; it is optional, and its absence is
    # what the startup warning reports.
    optional_depends_on = frozenset({"outcome_log"})

    def __init__(self, pm=None) -> None:
        # Accept pm at construction, like the other cognition plugins, so a
        # harness that broadcasts on_start without on_init still works.
        if pm is not None:
            self.pm = pm

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Name the loaded plugins that report through this hook: those that
        # list it in depends_on, other than the sink itself. With none
        # loaded (the minimal profile) there is nothing lost to report.
        if self.pm.has_plugin("outcome_log"):
            return
        reporters = sorted(
            name for name, plugin in self.pm.list_name_plugin()
            if plugin is not None and "trace" in getattr(plugin, "depends_on", ())
        )
        if not reporters:
            return
        logger.warning(
            "trace: outcome_log is not loaded, so trace records from %s are "
            "not written; remove outcome_log from plugins.disabled to keep them",
            ", ".join(reporters),
        )
