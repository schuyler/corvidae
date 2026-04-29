"""Tests for the perf-mon plugin."""
import asyncio
import time

import pytest

from corvidae.channel import Channel
from corvidae.hooks import create_plugin_manager
from corvidae.tools.perf_mon import PerfMonPlugin


@pytest.fixture
def pm():
    return create_plugin_manager()


@pytest.fixture
async def perf_plugin(pm):
    plugin = PerfMonPlugin()
    await plugin.on_start(pm)
    yield plugin
    await plugin.on_stop(pm)


@pytest.fixture
async def channel(perf_plugin, tmp_path):
    db_path = tmp_path / "test.db"
    ch = Channel(
        transport="fake",
        scope="test-channel",
    )
    return ch


# ------------------------------------------------------------------
# Hook tests
# ------------------------------------------------------------------

class TestPerfMonHooks:
    async def test_before_agent_turn_records_start(
        self, perf_plugin, channel
    ):
        await perf_plugin.before_agent_turn(channel)
        assert channel.id in perf_plugin._turn_start_times

    async def test_after_agent_response_records_metrics(
        self, perf_plugin, channel
    ):
        await perf_plugin.before_agent_turn(channel)
        await asyncio.sleep(0.01)  # ensure time passes for latency

        # Simulate LLM response
        result = {"usage": {"completion_tokens": 42}}
        await perf_plugin.after_agent_response(channel, result)

        stats = perf_plugin._channel_stats.get(channel.id)
        assert stats is not None
        assert stats.total_turns == 1
        assert len(stats.turns) == 1
        assert stats.turns[0].tokens_out == 42
        assert stats.turns[0].latency_ms > 0

    async def test_after_agent_response_without_before_is_safe(
        self, perf_plugin, channel
    ):
        """Should not crash if before_agent_turn was missed."""
        result = {"usage": {"completion_tokens": 10}}
        await perf_plugin.after_agent_response(channel, result)
        assert channel.id not in perf_plugin._channel_stats

    async def test_multiple_turns_update_rollings(
        self, perf_plugin, channel
    ):
        for i in range(5):
            await perf_plugin.before_agent_turn(channel)
            await asyncio.sleep(0.01)  # ensure time passes
            result = {"usage": {"completion_tokens": 20 + i}}
            await perf_plugin.after_agent_response(channel, result)

        stats = perf_plugin._channel_stats[channel.id]
        assert stats.total_turns == 5
        assert stats.avg_latency_ms > 0
        assert stats.p50_latency_ms > 0
        assert stats.avg_tokens_out == 22  # mean of 20,21,22,23,24


# ------------------------------------------------------------------
# Tool tests
# ------------------------------------------------------------------

class TestPerfMonTool:
    async def test_perf_stats_empty(self, perf_plugin):
        result = await perf_plugin.perf_stats(None)
        assert "No performance data" in result

    async def test_perf_stats_shows_data(
        self, perf_plugin, channel
    ):
        await perf_plugin.before_agent_turn(channel)
        await asyncio.sleep(0.01)
        result = {"usage": {"completion_tokens": 30}}
        await perf_plugin.after_agent_response(channel, result)

        output = await perf_plugin.perf_stats(channel)
        assert channel.id in output
        assert "Latency:" in output
        assert "Tokens:" in output


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------

class TestPerfMonIntegration:
    async def test_full_turn_lifecycle(
        self, perf_plugin, channel
    ):
        """Simulate a complete user→agent turn cycle."""
        # 1. User sends message (tracked via turn_counter)
        channel.turn_counter = 0

        # 2. before_agent_turn fires
        await perf_plugin.before_agent_turn(channel)
        assert channel.id in perf_plugin._turn_start_times

        # 3. Small delay to ensure latency measurement is meaningful
        await asyncio.sleep(0.01)

        # 4. after_agent_response fires
        result = {"usage": {"completion_tokens": 50}}
        await perf_plugin.after_agent_response(channel, result)

        # 5. Verify metrics recorded
        stats = perf_plugin._channel_stats[channel.id]
        assert stats.total_turns == 1
        assert stats.turns[0].tokens_out == 50
        assert stats.turns[0].latency_ms > 0

        # 6. Tool returns formatted output
        output = await perf_plugin.perf_stats(channel)
        assert channel.id in output
        assert "Latency:" in output
