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
