"""Tests for the transform_heartbeat_prompt hook and goals-plugin shaping."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from corvidae_cognition.heartbeat import frame_beat, HeartbeatPlugin
from corvidae_cognition.goals import GoalsPlugin


def _goal(id: int, status: str = "open", summary: str = "") -> dict:
    return {
        "id": id,
        "status": status,
        "summary": summary or f"goal {id}",
        "age_seconds": 3600,
    }


class TestGoalsTransformHeartbeatPrompt:
    """GoalsPlugin.transform_heartbeat_prompt shapes the beat."""

    @pytest.fixture
    def plugin(self) -> GoalsPlugin:
        pm = MagicMock()
        p = GoalsPlugin(pm)
        return p

    async def test_filters_out_blocked_and_waiting(self, plugin: GoalsPlugin) -> None:
        goals = [
            _goal(1, "open", "open goal"),
            _goal(2, "in-progress", "in-progress goal"),
            _goal(3, "blocked", "blocked goal"),
            _goal(4, "waiting", "waiting goal"),
        ]
        text = await plugin.transform_heartbeat_prompt(
            [{"id": 1, "age_seconds": 60, "channel_id": "test", "summary": "mem"}],
            [],
            "prompt",
            "goals instruction",
            diff=False,
            open_goals=goals,
        )
        assert "open goal" in text
        assert "in-progress goal" in text
        assert "blocked goal" not in text
        assert "waiting goal" not in text

    async def test_caps_at_five(self, plugin: GoalsPlugin) -> None:
        goals = [_goal(i, "open", f"goal {i}") for i in range(10)]
        text = await plugin.transform_heartbeat_prompt(
            [], [], "prompt", "goals instruction",
            diff=False, open_goals=goals,
        )
        for i in range(5):
            assert f"goal {i}" in text
        for i in range(5, 10):
            assert f"goal {i}" not in text

    async def test_strips_memories(self, plugin: GoalsPlugin) -> None:
        memories = [
            {"id": 1, "age_seconds": 60, "channel_id": "test", "summary": "recent memory"},
        ]
        text = await plugin.transform_heartbeat_prompt(
            memories, [], "prompt", "goals instruction",
            diff=False, open_goals=[_goal(1)],
        )
        assert "recent memory" not in text

    async def test_suppresses_retired_goals(self, plugin: GoalsPlugin) -> None:
        retired = [{"id": 99, "age_seconds": 7200, "summary": "retired goal summary"}]
        text = await plugin.transform_heartbeat_prompt(
            [], retired, "prompt", "goals instruction",
            diff=False, open_goals=[_goal(1)],
        )
        assert "retired goal summary" not in text


class TestHeartbeatSeed:
    """HeartbeatPlugin.transform_heartbeat_prompt is an identity seed."""

    @pytest.fixture
    def plugin(self) -> HeartbeatPlugin:
        pm = MagicMock()
        p = HeartbeatPlugin(pm)
        return p

    async def test_identity_returns_frame_beat_output(self, plugin: HeartbeatPlugin) -> None:
        goals = [_goal(1, "open", "open goal")]
        memories = [{"id": 1, "age_seconds": 60, "channel_id": "test", "summary": "mem"}]
        text = await plugin.transform_heartbeat_prompt(
            memories, [], "prompt", "goals instruction",
            diff=False, open_goals=goals,
        )
        expected = frame_beat(
            memories, [], "prompt", "goals instruction",
            diff=False, open_goals=goals,
        )
        assert text == expected
