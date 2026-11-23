"""The funnel's reloadable keys and its ``ponder`` tool, after #1176 (§5).

Ported from the retired ``test_funnel*.py`` files, whose admit/ingest/promise
tests the held store and ``test_funnel_policy.py`` now cover. What is kept is
what no Red test pins: the keys that reload without a restart, the tool's
schema and its bounds (a timeout that leaves the straggler running, a cancelled
call that leaves it running), and the missing-channel error.

Each test drives the real surface: ``on_config_reload`` through the plugin
manager's hook, ``transform_held_admission`` through ``pm.ahook`` and ``ponder``
as the tool registry hands it out, over futures shaped as the store's
``in_flight`` returns them.
"""

import asyncio
import logging
import time
from types import SimpleNamespace

import corvidae_cognition.funnel as funnel_mod
import pytest
from corvidae_cognition.funnel import FunnelPlugin

from .test_funnel_policy import (
    batch,
    ctx_for,
    make,
    offer,
    plan_of,
    ponder_tool,
    register_held_with,
    settle,
)


class TestReloadableKeys:
    """The four funnel.* keys apply on reload, and a bad reload falls back."""

    @pytest.mark.parametrize(
        "key",
        [
            "funnel.default_budget",
            "funnel.budgets",
            "funnel.window_budgets",
            "funnel.ingest_ttl",
        ],
    )
    def test_key_is_declared_reloadable(self, key):
        assert key in FunnelPlugin.reloadable_keys

    async def test_window_budgets_reload_applies_to_the_next_offer(self):
        env = await make()
        await env.pm.ahook.on_config_reload(config={"funnel": {"window_budgets": {"memory": 1}}})
        # A cap of one token cannot hold a three-word entry: dropped on a once batch.
        plans = await offer(env, [batch(env, "memory", ["one two three"])])
        assert plan_of(plans, 0).drop == [0]

    async def test_bad_ingest_ttl_reload_turns_expiry_off(self):
        env = await make({"funnel": {"ingest_ttl": 10}})
        old = lambda: batch(env, "critique", ["old entry"], age=20.0, once=False)  # noqa: E731
        assert plan_of(await offer(env, [old()]), 0).drop == [0]
        # An invalid reload leaves expiry off, so the same old batch is admitted.
        await env.pm.ahook.on_config_reload(config={"funnel": {"ingest_ttl": 0}})
        assert plan_of(await offer(env, [old()]), 0).admit == [0]


class TestNeverFits:
    """An entry past the whole budget or cap is dropped, even on a pushed batch."""

    async def test_entry_over_the_whole_budget_is_dropped_with_a_warning_naming_the_key(self, caplog):
        env = await make({"funnel": {"default_budget": 3}})
        big = " ".join(["word"] * 40)
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            plans = await offer(env, [batch(env, "critique", [big], once=False)])
        # Listed in drop, not left unlisted to be re-offered every turn forever.
        assert plan_of(plans, 0).drop == [0]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("funnel.budgets.critique" in m and "3-token" in m for m in warnings), warnings

    async def test_entry_over_the_window_cap_is_dropped_with_a_warning_naming_the_key(self, caplog):
        env = await make({"funnel": {"window_budgets": {"memory": 2}, "default_budget": 1000}})
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            plans = await offer(env, [batch(env, "memory", ["alpha beta gamma delta"], once=False)])
        assert plan_of(plans, 0).drop == [0]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("funnel.window_budgets.memory" in m and "2-token" in m for m in warnings), warnings


class TestPonderSchema:
    """The tool the model sees: no arguments, a one-line description with its bounds."""

    async def test_schema_takes_no_arguments_and_description_states_the_bounds(self):
        env = await make()
        registry: list = []
        env.pm.hook.register_tools(tool_registry=registry)
        tool = next(t for t in registry if t.name == "ponder")
        assert tool.schema["function"]["parameters"].get("properties", {}) == {}
        desc = tool.schema["function"]["description"].lower()
        assert "recollections catch up" in desc and "5 seconds" in desc and "at once" in desc
        assert "\n" not in desc

    def test_timeout_is_five_seconds(self):
        assert funnel_mod.PONDER_TIMEOUT_SECONDS == 5.0


class TestPonderBounds:
    """A timeout or a cancelled call never cancels the store's straggler."""

    async def test_another_channels_future_is_not_waited_on(self):
        env = await make()
        other: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        register_held_with(env, {"test:other": [other]})
        start = time.monotonic()
        out = await asyncio.wait_for(ponder_tool(env)(_ctx=ctx_for(env)), 1.5)
        assert time.monotonic() - start < 1.5
        assert "nothing is being recalled" in out.lower()

    async def test_already_landed_future_returns_at_once(self):
        env = await make()
        done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        done.set_result(True)
        register_held_with(env, {env.channel.id: [done]})
        out = await asyncio.wait_for(ponder_tool(env)(_ctx=ctx_for(env)), 1.0)
        assert "already landed" in out.lower() and "next turn" in out.lower()

    async def test_times_out_and_leaves_the_straggler_running(self, monkeypatch):
        env = await make()
        monkeypatch.setattr(funnel_mod, "PONDER_TIMEOUT_SECONDS", 0.2)
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        register_held_with(env, {env.channel.id: [future]})
        start = time.monotonic()
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert 0.18 <= time.monotonic() - start < 1.0
        # Names the wait and what happens to the recall; the future is untouched.
        assert "still waiting" in out.lower() and "0.2" in out and "later turn" in out.lower()
        assert not future.done()

    async def test_cancelling_the_call_leaves_the_straggler_running(self):
        env = await make()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        register_held_with(env, {env.channel.id: [future]})
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not future.done()

    @pytest.mark.parametrize("ctx", [None, SimpleNamespace(channel=None)])
    async def test_missing_channel_context_is_an_error(self, ctx):
        env = await make()
        # None is a direct call; channel=None is what dispatch gives a sub-agent loop.
        out = await ponder_tool(env)(_ctx=ctx)
        assert out.startswith("Error:") and "channel" in out and "do not call ponder" in out
