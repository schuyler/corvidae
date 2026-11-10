"""Tests for the funnel's ``ponder`` tool: wait for recall already in flight.

Every ``on_funnel_ingest`` submission for a turn is made inside that turn's
``before_agent_turn``, before the model call, so when the agent can call
``ponder`` the channel's pending promises are the recall in flight. ``ponder`` waits for those
promises to land, bounded by one timeout so a slow source cannot stall the
agent. With none in flight there is nothing to wait for and it returns at once.
It admits nothing itself: the next turn's sweep does that.
"""

import asyncio
import time
from types import SimpleNamespace

import aiosqlite
import corvidae_cognition.funnel as funnel_mod
import pytest

from .test_funnel_ingest import make, pending, text_of, turn


@pytest.fixture
async def build():
    # Factory for make(); closes the databases it opened.
    opened: list[aiosqlite.Connection] = []

    async def _build(config: dict | None = None):
        env = await make(config)
        opened.append(env.db)
        return env

    yield _build
    for db in opened:
        await db.close()


def ponder_tool(env: SimpleNamespace):
    # The ponder callable as the tool registry receives it.
    registry: list = []
    env.pm.hook.register_tools(tool_registry=registry)
    by_name = {t.name: t.fn for t in registry}
    assert "ponder" in by_name, f"funnel registered {sorted(by_name)}, not ponder"
    return by_name["ponder"]


def ctx_for(env: SimpleNamespace) -> SimpleNamespace:
    # The slice of ToolContext ponder reads: the channel.
    return SimpleNamespace(channel=env.channel)


async def submit_promise(env: SimpleNamespace, source: str = "memory") -> asyncio.Future:
    # Run the turn that collects one new, unresolved promise for ``source``.
    future = pending()
    env.provider.offer(env.channel.id, source, future)
    await turn(env)
    return future


def entries(*lines: str) -> funnel_mod.Ingest:
    # An Ingest with ``lines`` to admit.
    return funnel_mod.Ingest(entries=list(lines))


async def settle() -> None:
    # Let a freshly created task run up to its first await.
    for _ in range(3):
        await asyncio.sleep(0)


class TestRegistration:

    async def test_registered_through_the_real_pm(self, build):
        env = await build()
        assert callable(ponder_tool(env))

    async def test_schema_takes_no_arguments(self, build):
        env = await build()
        registry: list = []
        env.pm.hook.register_tools(tool_registry=registry)
        tool = next(t for t in registry if t.name == "ponder")
        assert tool.schema["function"]["parameters"].get("properties", {}) == {}

    async def test_description_says_what_it_waits_for_and_the_limits(self, build):
        env = await build()
        registry: list = []
        env.pm.hook.register_tools(tool_registry=registry)
        tool = next(t for t in registry if t.name == "ponder")
        desc = tool.schema["function"]["description"].lower()
        # The point of the tool: letting the agent's recollections catch up.
        assert "pause for a moment" in desc and "recollections catch up" in desc
        # The bound, and that it does not pause when nothing is running.
        assert "5 seconds" in desc and "at once" in desc
        # One line, as the schema builder keeps only the first.
        assert "\n" not in desc

    def test_timeout_is_five_seconds(self):
        assert funnel_mod.PONDER_TIMEOUT_SECONDS == 5.0


class TestNothingInFlight:

    async def test_returns_at_once_without_a_promise(self, build):
        env = await build()
        start = time.monotonic()
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert time.monotonic() - start < 1.5
        assert "nothing is being recalled" in out.lower()

    async def test_another_channels_promise_is_not_waited_on(self, build):
        env = await build()
        other = SimpleNamespace(id="test:other", conversation=env.conv)
        env.provider.offer(other.id, "memory", pending())
        await env.funnel.before_agent_turn(channel=other, trace_id="t", annotations={})
        start = time.monotonic()
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert time.monotonic() - start < 1.5
        assert "nothing is being recalled" in out.lower()

    async def test_done_but_empty_unswept_promise_is_not_in_flight(self, build):
        env = await build()
        future = await submit_promise(env)
        future.set_result(None)
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert "nothing is being recalled" in out.lower()

    async def test_push_arrivals_alone_do_not_make_it_wait(self, build):
        env = await build()
        env.funnel.ingest(env.channel, "critique", ["x"])
        start = time.monotonic()
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert time.monotonic() - start < 1.5
        assert "nothing is being recalled" in out.lower()


class TestWaitingOnRecall:

    async def test_lands_during_the_wait_returns_early_and_is_admitted_next_turn(self, build):
        env = await build()
        future = await submit_promise(env)
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        start = time.monotonic()
        future.set_result(entries("late entry"))
        out = await asyncio.wait_for(task, 1.0)
        assert time.monotonic() - start < 1.5
        assert "landed" in out.lower() and "next turn" in out.lower()
        # The ponder admitted nothing; the next turn's sweep does.
        assert "late entry" not in text_of(env.conv)
        await turn(env, annotations={})
        assert "late entry" in text_of(env.conv)

    async def test_finishing_empty_returns_early_and_says_so(self, build):
        env = await build()
        future = await submit_promise(env)
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        future.set_result(None)
        out = await asyncio.wait_for(task, 1.0)
        assert "nothing to add" in out.lower()

    async def test_empty_ingest_counts_as_nothing_to_add(self, build):
        env = await build()
        future = await submit_promise(env)
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        future.set_result(entries())
        out = await asyncio.wait_for(task, 1.0)
        assert "nothing to add" in out.lower()

    async def test_keeps_waiting_while_another_recall_is_still_running(self, build):
        env = await build()
        first = await submit_promise(env, "memory")
        second = await submit_promise(env, "wiki")
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        first.set_result(None)
        await settle()
        # One recall settled empty; the other is still running, so we are still waiting.
        assert not task.done()
        second.set_result(entries("wiki hit"))
        out = await asyncio.wait_for(task, 1.0)
        assert "landed" in out.lower()

    async def test_all_recalls_empty_ends_the_wait(self, build):
        env = await build()
        first = await submit_promise(env, "memory")
        second = await submit_promise(env, "wiki")
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        first.set_result(None)
        second.set_result(None)
        out = await asyncio.wait_for(task, 1.0)
        assert "nothing to add" in out.lower()

    async def test_already_landed_unswept_promise_returns_at_once(self, build):
        env = await build()
        future = await submit_promise(env)
        future.set_result(entries("early entry"))
        start = time.monotonic()
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert time.monotonic() - start < 1.5
        assert "already landed" in out.lower()

    async def test_funnel_cancelled_promise_counts_as_settled(self, build):
        env = await build()
        future = await submit_promise(env)
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        # What a supersede or an expiry does to a pending promise.
        future.cancel()
        out = await asyncio.wait_for(task, 1.0)
        assert "nothing to add" in out.lower()

    async def test_push_arrivals_do_not_end_the_wait(self, build, monkeypatch):
        env = await build()
        monkeypatch.setattr(funnel_mod, "PONDER_TIMEOUT_SECONDS", 0.3)
        await submit_promise(env)
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        env.funnel.ingest(env.channel, "critique", ["x"])
        await env.funnel.ingest_and_wake(env.channel, "user", "critique", ["y"])
        start = time.monotonic()
        out = await task
        assert time.monotonic() - start >= 0.2
        assert "still waiting" in out.lower()


class TestWaitBounds:

    async def test_returns_at_the_first_recall_with_entries_while_another_still_runs(self, build):
        env = await build()
        first = await submit_promise(env, "memory")
        await submit_promise(env, "wiki")
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        start = time.monotonic()
        # One lands with entries while the other is still running: no waiting for it.
        first.set_result(entries("memory hit"))
        out = await asyncio.wait_for(task, 1.0)
        assert time.monotonic() - start < 1.5
        assert "landed" in out.lower()

    async def test_total_wait_stays_within_one_timeout_across_promises(self, build, monkeypatch):
        env = await build()
        monkeypatch.setattr(funnel_mod, "PONDER_TIMEOUT_SECONDS", 0.4)
        first = await submit_promise(env, "memory")
        await submit_promise(env, "wiki")
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        start = time.monotonic()
        await asyncio.sleep(0.25)
        # One recall settles empty partway; the other never does. The timeout is
        # for the whole call, not restarted by each settle.
        first.set_result(None)
        out = await task
        elapsed = time.monotonic() - start
        assert 0.35 <= elapsed < 1.0
        assert "still waiting" in out.lower()

    async def test_one_already_landed_and_one_running_returns_at_once(self, build):
        env = await build()
        landed = await submit_promise(env, "memory")
        await submit_promise(env, "wiki")
        landed.set_result(entries("memory hit"))
        start = time.monotonic()
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert time.monotonic() - start < 1.5
        assert "already landed" in out.lower()

    async def test_empty_and_entries_settling_together_still_lands(self, build):
        env = await build()
        empty = await submit_promise(env, "memory")
        hit = await submit_promise(env, "wiki")
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        # Both finish in the same loop tick: the one with entries decides the result.
        empty.set_result(None)
        hit.set_result(entries("wiki hit"))
        out = await asyncio.wait_for(task, 1.0)
        assert "landed" in out.lower() and "nothing to add" not in out.lower()

    async def test_a_failed_recall_counts_as_settled_mid_wait(self, build):
        env = await build()
        future = await submit_promise(env)
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        future.set_exception(RuntimeError("recall exploded"))
        out = await asyncio.wait_for(task, 1.0)
        assert "nothing to add" in out.lower()

    async def test_a_failed_recall_is_not_in_flight_before_the_call(self, build):
        env = await build()
        future = await submit_promise(env)
        future.set_exception(RuntimeError("recall exploded"))
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert "nothing is being recalled" in out.lower()


class TestTimeoutAndCancel:

    async def test_times_out_and_leaves_the_recall_running(self, build, monkeypatch):
        env = await build()
        monkeypatch.setattr(funnel_mod, "PONDER_TIMEOUT_SECONDS", 0.2)
        future = await submit_promise(env)
        start = time.monotonic()
        out = await ponder_tool(env)(_ctx=ctx_for(env))
        assert 0.18 <= time.monotonic() - start < 1.0
        # Names the wait and says what happens to the recall.
        assert "still waiting" in out.lower() and "0.2" in out and "later turn" in out.lower()
        # Timing out must not cancel the recall: it may still land for a later turn.
        assert not future.done()

    async def test_cancelling_the_ponder_leaves_the_recall_running(self, build):
        env = await build()
        future = await submit_promise(env)
        task = asyncio.create_task(ponder_tool(env)(_ctx=ctx_for(env)))
        await settle()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not future.done()

    async def test_the_late_result_is_still_admitted_after_a_timeout(self, build, monkeypatch):
        env = await build()
        monkeypatch.setattr(funnel_mod, "PONDER_TIMEOUT_SECONDS", 0.05)
        future = await submit_promise(env)
        await ponder_tool(env)(_ctx=ctx_for(env))
        future.set_result(entries("slow entry"))
        await turn(env, annotations={})
        assert "slow entry" in text_of(env.conv)


class TestMissingChannel:

    @pytest.mark.parametrize("ctx", [None, SimpleNamespace(channel=None)])
    async def test_missing_channel_context_is_an_error(self, build, ctx):
        env = await build()
        # None is a direct call; channel=None is what dispatch gives a sub-agent loop.
        out = await ponder_tool(env)(_ctx=ctx)
        # Names the cause and tells the model what to do instead.
        assert out.startswith("Error:") and "channel" in out
        assert "do not call ponder" in out
