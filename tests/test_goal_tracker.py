"""Tests for the goal-tracker plugin."""
import asyncio
import time

import pytest

from corvidae.channel import Channel
from corvidae.hooks import create_plugin_manager
from corvidae.tools.goal_tracker import (
    GoalStore,
    GoalTrackerPlugin,
    GoalStatus,
    TacticalGoal,
    default_strategy,
    strategy_context_preload,
    strategy_research_synthesis,
)


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "goals.db")


@pytest.fixture
async def store(tmp_db):
    s = GoalStore(type(tmp_db)(tmp_db))
    await s.open()
    yield s
    await s.close()


@pytest.fixture
async def pm():
    return create_plugin_manager()


@pytest.fixture
async def plugin(pm, tmp_db, tmp_path):
    p = GoalTrackerPlugin(
        db_path=tmp_db,
        idle_threshold_seconds=30.0,
        workspace_root=tmp_path,
    )
    await p.on_start(pm)
    yield p
    await p.on_stop(pm)


@pytest.fixture
async def channel(plugin):
    ch = Channel(transport="fake", scope="test-channel")
    return ch


# ------------------------------------------------------------------
# GoalStore tests
# ------------------------------------------------------------------

class TestGoalStore:
    async def test_save_and_load(self, store):
        goal = TacticalGoal(
            id="abc123", title="Test Goal", description="A test",
            priority=75,
        )
        await store.save_goal(goal)
        loaded = await store.load_all()
        assert len(loaded) == 1
        assert loaded[0].title == "Test Goal"
        assert loaded[0].priority == 75

    async def test_order_by_priority(self, store):
        for priority in [20, 80, 50]:
            await store.save_goal(TacticalGoal(
                id=f"id_{priority}", title=f"P{priority}",
                description="", priority=priority,
            ))
        loaded = await store.load_all()
        priorities = [g.priority for g in loaded]
        assert priorities == sorted(priorities, reverse=True)

    async def test_get_next_pending(self, store):
        low = TacticalGoal(id="low", title="Low", description="", priority=10)
        high = TacticalGoal(id="high", title="High", description="", priority=90)
        await store.save_goal(low)
        await store.save_goal(high)

        next_g = await store.get_next_pending()
        assert next_g is not None
        assert next_g.id == "high"

    async def test_done_goals_not_returned(self, store):
        done = TacticalGoal(
            id="done1", title="Done Goal", description="",
            status=GoalStatus.DONE, priority=99,
        )
        await store.save_goal(done)

        next_g = await store.get_next_pending()
        assert next_g is None


# ------------------------------------------------------------------
# Plugin lifecycle tests
# ------------------------------------------------------------------

class TestPluginLifecycle:
    async def test_on_start_registers(self, plugin):
        # After on_start, the plugin should be registered and store open
        assert plugin._store._conn is not None

    async def test_idle_since_reset_on_message(self, plugin, channel):
        await plugin.on_message_received(channel)
        assert plugin._idle_since is None


# ------------------------------------------------------------------
# Tool tests
# ------------------------------------------------------------------

class TestTools:
    async def test_add_goal(self, plugin):
        result = await plugin.add_goal("Test goal", "A description", priority=80)
        assert "Test goal" in result
        loaded = await plugin._store.load_all()
        assert len(loaded) == 1
        assert loaded[0].priority == 80

    async def test_list_goals_empty(self, plugin):
        result = await plugin.list_goals()
        assert "No tactical goals" in result

    async def test_list_goals_shows_status(self, plugin):
        await plugin.add_goal("Goal A", "Desc A", priority=50)
        result = await plugin.list_goals()
        assert "PENDING" in result
        assert "Goal A" in result


# ------------------------------------------------------------------
# Idle detection tests
# ------------------------------------------------------------------

class TestIdleDetection:
    async def test_no_pending_goals(self, plugin):
        result = await plugin.after_idle(60.0)
        assert "No pending goals" in result

    async def test_add_and_execute_goal(self, plugin):
        await plugin.add_goal("Quick task", "Just a test", priority=90)

        # Simulate idle — should pick up the goal and execute it
        result = await plugin.after_idle(60.0)
        assert "Quick task" in result


# ------------------------------------------------------------------
# Strategy tests
# ------------------------------------------------------------------

class TestStrategies:
    async def test_default_strategy(self, tmp_path):
        goal = TacticalGoal(id="x", title="X", description="")
        result = await default_strategy(goal, tmp_path)
        assert "not yet implemented" in result

    async def test_context_preload_clean_repo(
        self, plugin
    ):
        # Initialize git repo at workspace root
        import subprocess  # noqa: PLC0415
        subprocess.run(["git", "init"], cwd=str(plugin._workspace_root), capture_output=True)
        result = await strategy_context_preload(
            TacticalGoal(id="x", title="X", description=""),
            plugin._workspace_root,
        )
        assert "Context preload" in result

    async def test_research_synthesis_no_links_dir(self, tmp_path):
        result = await strategy_research_synthesis(
            TacticalGoal(id="x", title="X", description=""),
            tmp_path,
        )
        assert "Links/" in result
