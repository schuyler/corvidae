"""Tests for corvidae.attribution — the call-attribution contextvar module."""

import asyncio

from corvidae.attribution import (
    get_attribution,
    reset_attribution,
    set_attribution,
)


class TestSetAttribution:
    def test_set_merges_rather_than_replaces(self):
        # Setting one field, then another, yields both in the merged dict.
        token1 = set_attribution(stage="turn")
        token2 = set_attribution(channel_id="irc:#general")
        assert get_attribution() == {"stage": "turn", "channel_id": "irc:#general"}
        reset_attribution(token2)
        reset_attribution(token1)

    def test_set_overwrites_existing_field(self):
        # A later set of the same field shadows the earlier value.
        token1 = set_attribution(stage="turn")
        token2 = set_attribution(stage="compaction")
        assert get_attribution()["stage"] == "compaction"
        reset_attribution(token2)
        # Reset restores the shadowed value.
        assert get_attribution()["stage"] == "turn"
        reset_attribution(token1)

    def test_reset_restores_previous_state(self):
        # After resetting the outermost token, attribution is empty again.
        before = dict(get_attribution())
        token = set_attribution(stage="subagent")
        assert get_attribution()["stage"] == "subagent"
        reset_attribution(token)
        assert get_attribution() == before


class TestGetAttribution:
    def test_returns_empty_dict_not_none_when_unset(self):
        # The default must be an empty dict, never None.
        assert get_attribution() == {}
        assert get_attribution() is not None


class TestTaskIsolation:
    async def test_sibling_task_created_before_set_does_not_see_attribution(self):
        # Documents the propagation rule: contextvars snapshot at
        # asyncio.create_task time, so a task created BEFORE the set
        # never observes the attribution.
        started = asyncio.Event()
        release = asyncio.Event()
        observed: dict = {}

        async def sibling():
            started.set()
            await release.wait()
            observed.update(get_attribution())

        task = asyncio.create_task(sibling())
        await started.wait()

        token = set_attribution(stage="turn")
        release.set()
        await task
        reset_attribution(token)

        assert observed == {}

    async def test_child_task_created_after_set_sees_attribution(self):
        # Conversely, a task created AFTER the set inherits the snapshot.
        observed: dict = {}

        async def child():
            observed.update(get_attribution())

        token = set_attribution(stage="turn", channel_id="c1")
        await asyncio.create_task(child())
        reset_attribution(token)

        assert observed == {"stage": "turn", "channel_id": "c1"}


class TestNextAgentDepth:
    def test_main_turn_returns_one(self):
        from corvidae.attribution import next_agent_depth
        # Main turn has no agent_depth in attribution (defaults to 0)
        assert next_agent_depth(2) == 1

    def test_depth_one_returns_two(self):
        from corvidae.attribution import next_agent_depth, set_attribution, reset_attribution
        token = set_attribution(agent_depth=1)
        try:
            assert next_agent_depth(2) == 2
        finally:
            reset_attribution(token)

    def test_refuses_at_limit(self):
        from corvidae.attribution import next_agent_depth, set_attribution, reset_attribution
        token = set_attribution(agent_depth=2)
        try:
            try:
                next_agent_depth(2)
                assert False, "should have raised"
            except ValueError as exc:
                assert "nested agent depth 2 reached" in str(exc)
                assert "tools.max_agent_depth=2" in str(exc)
                assert "do this work in the current task instead" in str(exc)
        finally:
            reset_attribution(token)

    def test_refuses_above_limit(self):
        from corvidae.attribution import next_agent_depth, set_attribution, reset_attribution
        token = set_attribution(agent_depth=5)
        try:
            try:
                next_agent_depth(2)
                assert False, "should have raised"
            except ValueError:
                pass
        finally:
            reset_attribution(token)


class TestParseMaxAgentDepth:
    def test_default_is_two(self):
        from corvidae.attribution import parse_max_agent_depth
        assert parse_max_agent_depth({}) == 2

    def test_explicit_value(self):
        from corvidae.attribution import parse_max_agent_depth
        assert parse_max_agent_depth({"tools": {"max_agent_depth": 3}}) == 3

    def test_rejects_zero(self):
        from corvidae.attribution import parse_max_agent_depth
        try:
            parse_max_agent_depth({"tools": {"max_agent_depth": 0}})
            assert False, "should have raised"
        except ValueError as exc:
            assert "tools.max_agent_depth" in str(exc)

    def test_rejects_negative(self):
        from corvidae.attribution import parse_max_agent_depth
        try:
            parse_max_agent_depth({"tools": {"max_agent_depth": -1}})
            assert False, "should have raised"
        except ValueError:
            pass

    def test_rejects_string(self):
        from corvidae.attribution import parse_max_agent_depth
        try:
            parse_max_agent_depth({"tools": {"max_agent_depth": "2"}})
            assert False, "should have raised"
        except ValueError:
            pass

    def test_rejects_bool(self):
        from corvidae.attribution import parse_max_agent_depth
        try:
            parse_max_agent_depth({"tools": {"max_agent_depth": True}})
            assert False, "should have raised"
        except ValueError:
            pass

    def test_tools_not_a_dict(self):
        from corvidae.attribution import parse_max_agent_depth
        # tools is not a dict: falls back to default
        assert parse_max_agent_depth({"tools": "bad"}) == 2
