"""Shared test helpers for the ``trace`` protocol plugin (#1179).

``add_trace`` registers the protocol plugin and installs its hookspecs, so a
reporter under test can fire ``on_trace_record``. ``TraceRecorder`` is a
listener that keeps every record it hears. ``read_outcomes`` reads the JSONL
file outcome_log appends to.
"""

import json
from pathlib import Path

from corvidae.hooks import hookimpl


class TraceRecorder:
    """A listener on ``on_trace_record`` that keeps each call's arguments."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    @hookimpl
    async def on_trace_record(
        self, trace_id: str, channel_id: str, source: str,
        origin: str | None, record: dict,
    ) -> None:
        # Keep the call exactly as fired, one dict per record.
        self.records.append({
            "trace_id": trace_id, "channel_id": channel_id, "source": source,
            "origin": origin, "record": record,
        })

    def by_source(self, source: str) -> list[dict]:
        """The records one reporter fired, in order."""
        return [r for r in self.records if r["source"] == source]


def add_trace(pm, recorder: TraceRecorder | None = None) -> TraceRecorder | None:
    """Register the ``trace`` protocol plugin (and ``recorder``) on ``pm``
    and install its specs, as startup's install_plugin_hookspecs would."""
    from corvidae_cognition.trace import TracePlugin, TraceSpec

    pm.register(TracePlugin(), name="trace")
    pm.add_hookspecs(TraceSpec)
    if recorder is not None:
        pm.register(recorder, name="trace_recorder")
    return recorder


def outcomes_file(db_path) -> Path:
    """Where outcome_log writes for a DB at ``db_path``."""
    return Path(db_path).with_suffix(".outcomes.jsonl")


def read_outcomes(path) -> list[dict]:
    """Every line of an outcomes JSONL file, parsed; [] when it is absent."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
