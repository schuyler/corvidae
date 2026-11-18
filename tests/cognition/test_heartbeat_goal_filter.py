"""Tests for the on_heartbeat_section hook: goals plugin contributes its section."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from corvidae_cognition.goals import GoalsPlugin
from corvidae_cognition.heartbeat import HeartbeatPlugin, HeartbeatSpec


def _goal(id: int, status: str = "open", summary: str = "test goal") -> dict:
    return {"id": id, "status": status, "summary": summary, "age_seconds": 3600}


def _retired(id: int, summary: str = "retired summary") -> dict:
    return {"id": id, "age_seconds": 7200, "summary": summary}


class TestGoalsHeartbeatSection:
    """GoalsPlugin.on_heartbeat_section returns the goals section."""

    @pytest.fixture
    def plugin(self) -> GoalsPlugin:
        pm = MagicMock()
        p = GoalsPlugin(pm)
        p._goals_instruction = "GOALS INSTRUCTION"
        p._retired_goal_limit = 20
        return p

    @pytest.mark.asyncio
    async def test_returns_instruction_and_retired(self, plugin: GoalsPlugin) -> None:
        plugin.retired_goals = AsyncMock(return_value=[_retired(99, "old goal")])
        text = await plugin.on_heartbeat_section(prompt="prompt")
        assert text is not None
        assert "GOALS INSTRUCTION" in text
        assert "old goal" in text

    @pytest.mark.asyncio
    async def test_returns_instruction_only_when_no_retired(self, plugin: GoalsPlugin) -> None:
        plugin.retired_goals = AsyncMock(return_value=[])
        text = await plugin.on_heartbeat_section(prompt="prompt")
        assert text is not None
        assert "GOALS INSTRUCTION" in text
        assert "(none)" in text

    @pytest.mark.asyncio
    async def test_no_instruction_still_returns_retired(self, plugin: GoalsPlugin) -> None:
        plugin._goals_instruction = ""
        plugin.retired_goals = AsyncMock(return_value=[_retired(99, "old goal")])
        text = await plugin.on_heartbeat_section(prompt="prompt")
        assert text is not None
        assert "old goal" in text

    @pytest.mark.asyncio
    async def test_retired_goals_failure_returns_instruction_only(self, plugin: GoalsPlugin) -> None:
        plugin.retired_goals = AsyncMock(side_effect=Exception("db error"))
        text = await plugin.on_heartbeat_section(prompt="prompt")
        assert text is not None
        assert "GOALS INSTRUCTION" in text
        assert "(none)" in text


class TestHeartbeatSpec:
    """HeartbeatSpec.on_heartbeat_section is a plain async hook (not firstresult)."""

    def test_not_firstresult(self) -> None:
        spec = HeartbeatSpec.on_heartbeat_section
        # Plain hookspec: no firstresult attribute
        assert not hasattr(spec, "_hookspec_opts") or "firstresult" not in getattr(spec, "_hookspec_opts", {})


class TestHeartbeatPluginNoSeed:
    """HeartbeatPlugin no longer implements on_heartbeat_section (no identity seed)."""

    def test_no_on_heartbeat_section_impl(self) -> None:
        pm = MagicMock()
        p = HeartbeatPlugin(pm)
        # The plugin should NOT have a on_heartbeat_section method
        assert not hasattr(p, "on_heartbeat_section")
