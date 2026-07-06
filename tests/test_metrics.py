"""Tests for corvidae.metrics — MetricsPlugin, UsageLogPlugin, MetricsJsonlPlugin."""

import json
from pathlib import Path

import aiosqlite
import pytest

from corvidae.hooks import create_plugin_manager, hookimpl


class _MetricsRecorder:
    """Consumer plugin that records on_metrics events."""

    def __init__(self):
        self.events: list[tuple[str, float, dict]] = []

    @hookimpl
    async def on_metrics(self, name, value, tags):
        self.events.append((name, float(value), dict(tags)))


USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
ATTRIBUTION = {"stage": "turn", "channel_id": "irc:#general"}


async def _fire_response(pm, *, usage=USAGE, error=None, attribution=ATTRIBUTION,
                         role="main", model="m1", request_id="req1",
                         latency_ms=42.0):
    """Dispatch on_llm_response through the pm the way LLMPlugin's observer does."""
    await pm.ahook.on_llm_response(
        role=role,
        model=model,
        request_id=request_id,
        usage=usage,
        latency_ms=latency_ms,
        attribution=attribution,
        error=error,
    )


class TestMetricsPlugin:
    async def _setup(self):
        from corvidae.metrics import MetricsPlugin

        pm = create_plugin_manager()
        recorder = _MetricsRecorder()
        pm.register(recorder, name="recorder")
        plugin = MetricsPlugin()
        pm.register(plugin, name="metrics")
        await plugin.on_init(pm, {})
        return pm, recorder

    async def test_usage_dict_emits_token_and_latency_metrics(self):
        pm, recorder = await self._setup()
        await _fire_response(pm)

        by_name = {name: (value, tags) for name, value, tags in recorder.events}
        assert by_name["llm.tokens.prompt"][0] == 100.0
        assert by_name["llm.tokens.completion"][0] == 20.0
        assert by_name["llm.tokens.total"][0] == 120.0
        assert by_name["llm.latency_ms"][0] == 42.0
        # Tags carry role, model, and the attribution stage/channel.
        for value, tags in by_name.values():
            assert tags == {
                "role": "main",
                "model": "m1",
                "stage": "turn",
                "channel": "irc:#general",
            }
        assert "llm.errors" not in by_name

    async def test_missing_usage_emits_only_latency(self):
        pm, recorder = await self._setup()
        await _fire_response(pm, usage=None)

        names = [name for name, _, _ in recorder.events]
        assert names == ["llm.latency_ms"]

    async def test_partial_usage_skips_missing_fields(self):
        pm, recorder = await self._setup()
        await _fire_response(pm, usage={"prompt_tokens": 7})

        names = {name for name, _, _ in recorder.events}
        assert "llm.tokens.prompt" in names
        assert "llm.tokens.completion" not in names
        assert "llm.tokens.total" not in names

    async def test_error_emits_error_metric(self):
        pm, recorder = await self._setup()
        await _fire_response(pm, usage=None, error="boom")

        by_name = {name: value for name, value, _ in recorder.events}
        assert by_name["llm.errors"] == 1.0

    async def test_empty_attribution_yields_empty_tag_strings(self):
        pm, recorder = await self._setup()
        await _fire_response(pm, attribution={})

        _, _, tags = recorder.events[0]
        assert tags["stage"] == ""
        assert tags["channel"] == ""

    async def test_broadcast_reaches_two_consumers(self):
        from corvidae.metrics import MetricsPlugin

        pm = create_plugin_manager()
        recorder_a = _MetricsRecorder()
        recorder_b = _MetricsRecorder()
        pm.register(recorder_a, name="recorder_a")
        pm.register(recorder_b, name="recorder_b")
        plugin = MetricsPlugin()
        pm.register(plugin, name="metrics")
        await plugin.on_init(pm, {})

        await _fire_response(pm)

        assert recorder_a.events == recorder_b.events
        assert len(recorder_a.events) > 0


class TestUsageLogPlugin:
    async def _setup(self, db):
        from corvidae.metrics import UsageLogPlugin
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        plugin = UsageLogPlugin()
        pm.register(plugin, name="usage_log")
        await plugin.on_init(pm, {})
        await plugin.on_start({})
        return pm, plugin

    async def test_writes_row_readable_back_by_request_id(self, db):
        pm, plugin = await self._setup(db)
        await _fire_response(pm, request_id="req-abc")

        async with db.execute(
            "SELECT request_id, role, model, stage, channel_id, prompt_tokens, "
            "completion_tokens, total_tokens, latency_ms, error "
            "FROM usage_log WHERE request_id = ?",
            ("req-abc",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row[0] == "req-abc"
        assert row[1] == "main"
        assert row[2] == "m1"
        assert row[3] == "turn"
        assert row[4] == "irc:#general"
        assert row[5] == 100
        assert row[6] == 20
        assert row[7] == 120
        assert row[8] == 42.0
        assert row[9] is None

    async def test_error_row_has_null_usage_and_error_text(self, db):
        pm, plugin = await self._setup(db)
        await _fire_response(pm, request_id="req-err", usage=None, error="boom")

        async with db.execute(
            "SELECT prompt_tokens, total_tokens, error FROM usage_log "
            "WHERE request_id = ?",
            ("req-err",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row == (None, None, "boom")


class TestMetricsJsonlPlugin:
    async def _setup(self, tmp_path, configured=True):
        from corvidae.metrics import MetricsJsonlPlugin

        pm = create_plugin_manager()
        plugin = MetricsJsonlPlugin()
        pm.register(plugin, name="metrics_jsonl")
        config = {"_base_dir": tmp_path}
        if configured:
            config["daemon"] = {"metrics_jsonl": "metrics.jsonl"}
        await plugin.on_init(pm, config)
        await plugin.on_start(config)
        return pm, plugin

    async def test_writes_one_valid_json_line_per_event(self, tmp_path):
        pm, plugin = await self._setup(tmp_path)
        await pm.ahook.on_metrics(name="llm.tokens.total", value=120.0, tags={"role": "main"})
        await pm.ahook.on_metrics(name="llm.latency_ms", value=42.0, tags={})
        await plugin.on_stop()

        lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["name"] == "llm.tokens.total"
        assert first["value"] == 120.0
        assert first["tags"] == {"role": "main"}
        assert "ts" in first

    async def test_disabled_when_unconfigured(self, tmp_path):
        pm, plugin = await self._setup(tmp_path, configured=False)
        await pm.ahook.on_metrics(name="x", value=1.0, tags={})
        await plugin.on_stop()

        assert not (tmp_path / "metrics.jsonl").exists()
