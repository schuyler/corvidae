"""Tests for the funnel's ``ponder`` tool: a short, interruptible pause.

``ponder`` is the agent clearing its throat. The tool call stays outstanding
for a few seconds (the kernel holds the next assistant turn until it
returns) or until the funnel gets something for the channel, whichever is
first. It only waits; it admits nothing itself.
"""

import asyncio
import logging
import time
from types import SimpleNamespace

import corvidae_cognition.funnel as funnel_mod
import pytest

from .test_funnel_ingest import make, pending, turn


@pytest.fixture
async def build():
    # Factory for make(); closes the databases it opened.
    opened = []

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


async def settle() -> None:
    # Let a freshly created task run up to its first await.
    for _ in range(3):
        await asyncio.sleep(0)


class TestRegistration:

    async def test_registered_through_the_real_pm(self, build):
        env = await build()
        assert callable(ponder_tool(env))

    async def test_schema_hides_ctx_and_describes_the_arguments(self, build):
        env = await build()
        registry: list = []
        env.pm.hook.register_tools(tool_registry=registry)
        tool = next(t for t in registry if t.name == "ponder")
        props = tool.schema["function"]["parameters"]["properties"]
        assert set(props) == {"seconds", "until_context"}

    def test_cap_is_five_seconds(self):
        assert funnel_mod.MAX_PONDER_SECONDS == 5.0


class TestFixedWait:

    async def test_waits_the_requested_time_when_nothing_arrives(self, build):
        env = await build()
        ponder = ponder_tool(env)
        start = time.monotonic()
        result = await ponder(0.2, _ctx=ctx_for(env))
        assert 0.18 <= time.monotonic() - start < 1.0
        assert "0.2" in result and "woke" not in result.lower()

    async def test_until_context_false_ignores_arrivals(self, build):
        env = await build()
        ponder = ponder_tool(env)
        task = asyncio.create_task(ponder(0.2, until_context=False, _ctx=ctx_for(env)))
        await settle()
        env.funnel.ingest(env.channel, "critique", ["late"])
        start = time.monotonic()
        result = await task
        assert time.monotonic() - start >= 0.1
        assert "woke" not in result.lower()


class TestEarlyWake:

    async def _woken_by(self, env, arrive) -> str:
        # Start a long ponder, run ``arrive``, return the result and check it was quick.
        ponder = ponder_tool(env)
        task = asyncio.create_task(ponder(5.0, _ctx=ctx_for(env)))
        await settle()
        start = time.monotonic()
        await arrive()
        result = await asyncio.wait_for(task, 1.0)
        assert time.monotonic() - start < 0.5
        return result

    async def test_ingest_wakes_it(self, build):
        env = await build()
        result = await self._woken_by(
            env, lambda: _call(env.funnel.ingest, env.channel, "critique", ["x"])
        )
        assert "woke" in result.lower()

    async def test_ingest_and_wake_wakes_it(self, build):
        env = await build()
        result = await self._woken_by(
            env,
            lambda: env.funnel.ingest_and_wake(env.channel, "user", "critique", ["x"]),
        )
        assert "woke" in result.lower()

    async def test_resolving_pull_promise_wakes_it(self, build):
        env = await build()
        future = pending()
        env.provider.offer(env.channel.id, "memory", future)
        await turn(env)

        async def resolve():
            future.set_result(funnel_mod.Ingest(entries=["late"]))

        result = await self._woken_by(env, resolve)
        assert "woke" in result.lower()

    async def test_other_channels_arrivals_do_not_wake_it(self, build):
        env = await build()
        other = SimpleNamespace(id="test:other", conversation=None)
        ponder = ponder_tool(env)
        task = asyncio.create_task(ponder(0.3, _ctx=ctx_for(env)))
        await settle()
        env.funnel.ingest(other, "critique", ["x"])
        result = await task
        assert "woke" not in result.lower()

    async def test_items_already_waiting_return_at_once(self, build):
        env = await build()
        env.funnel.ingest(env.channel, "critique", ["x"])
        ponder = ponder_tool(env)
        start = time.monotonic()
        result = await ponder(5.0, _ctx=ctx_for(env))
        assert time.monotonic() - start < 0.5
        assert "woke" in result.lower()

    async def test_ponder_admits_nothing_itself(self, build):
        env = await build()
        env.funnel.ingest(env.channel, "critique", ["x"])
        await ponder_tool(env)(1.0, _ctx=ctx_for(env))
        # The item is still queued for the next turn's drain.
        assert env.funnel._next_turn[env.channel.id]


async def _call(fn, *args):
    # Adapt a sync producer call to an awaitable for the arrive hook.
    fn(*args)


class TestCapAndInput:

    async def test_over_cap_is_clamped_and_says_so(self, build, monkeypatch, caplog):
        env = await build()
        monkeypatch.setattr(funnel_mod, "MAX_PONDER_SECONDS", 0.1)
        with caplog.at_level(logging.WARNING, logger="corvidae.funnel"):
            start = time.monotonic()
            result = await ponder_tool(env)(30, _ctx=ctx_for(env))
        assert time.monotonic() - start < 1.0
        # Names the requested value and the limit it was cut to.
        assert "30" in result and "0.1" in result
        assert any("ponder" in m and "30" in m for m in (r.getMessage() for r in caplog.records))

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
    async def test_rejects_non_positive_or_non_finite(self, build, bad):
        env = await build()
        result = await ponder_tool(env)(bad, _ctx=ctx_for(env))
        # Names the argument, the rejected value and the accepted range.
        assert result.startswith("Error:")
        assert "seconds" in result and repr(bad) in result
        # Accepted range and the remedy.
        assert "above 0 and at most 5" in result and "call ponder again" in result

    async def test_missing_channel_context_is_an_error(self, build):
        env = await build()
        result = await ponder_tool(env)(1.0, _ctx=None)
        # Names the cause and tells the model what to do instead.
        assert result.startswith("Error:") and "channel" in result
        assert "do not call ponder" in result


class TestCleanup:

    async def test_cancelled_ponder_releases_its_waiter(self, build):
        env = await build()
        task = asyncio.create_task(ponder_tool(env)(5.0, _ctx=ctx_for(env)))
        await settle()
        assert env.funnel._ponderers.get(env.channel.id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert env.channel.id not in env.funnel._ponderers

    async def test_finished_ponder_releases_its_waiter(self, build):
        env = await build()
        await ponder_tool(env)(0.05, _ctx=ctx_for(env))
        assert env.channel.id not in env.funnel._ponderers

    async def test_on_stop_wakes_waiting_ponders(self, build):
        env = await build()
        task = asyncio.create_task(ponder_tool(env)(5.0, _ctx=ctx_for(env)))
        await settle()
        await env.funnel.on_stop()
        await asyncio.wait_for(task, 1.0)

    async def test_two_ponders_on_one_channel_both_wake(self, build):
        env = await build()
        a = asyncio.create_task(ponder_tool(env)(5.0, _ctx=ctx_for(env)))
        b = asyncio.create_task(ponder_tool(env)(5.0, _ctx=ctx_for(env)))
        await settle()
        env.funnel.ingest(env.channel, "critique", ["x"])
        await asyncio.wait_for(asyncio.gather(a, b), 1.0)
        assert env.channel.id not in env.funnel._ponderers


class TestEmptyPromiseResults:
    """A promise that admits nothing is not an arrival (reviewer finding 1)."""

    @pytest.mark.parametrize("result", [None, "empty"])
    async def test_nothing_to_admit_does_not_wake_it(self, build, result):
        env = await build()
        future = pending()
        env.provider.offer(env.channel.id, "memory", future)
        await turn(env)
        task = asyncio.create_task(ponder_tool(env)(0.3, _ctx=ctx_for(env)))
        await settle()
        future.set_result(None if result is None else funnel_mod.Ingest(entries=[]))
        out = await task
        assert "woke" not in out.lower()

    @pytest.mark.parametrize("result", [None, "empty"])
    async def test_done_but_empty_promise_is_not_already_waiting(self, build, result):
        env = await build()
        future = pending()
        env.provider.offer(env.channel.id, "memory", future)
        await turn(env)
        future.set_result(None if result is None else funnel_mod.Ingest(entries=[]))
        start = time.monotonic()
        out = await ponder_tool(env)(0.2, _ctx=ctx_for(env))
        assert time.monotonic() - start >= 0.15
        assert "woke" not in out.lower()

    async def test_done_promise_with_entries_is_already_waiting(self, build):
        env = await build()
        future = pending()
        env.provider.offer(env.channel.id, "memory", future)
        await turn(env)
        future.set_result(funnel_mod.Ingest(entries=["late"]))
        start = time.monotonic()
        out = await ponder_tool(env)(5.0, _ctx=ctx_for(env))
        assert time.monotonic() - start < 0.5 and "woke" in out.lower()


class TestAlreadyWaitingBranches:

    async def test_origin_keyed_payload_counts_as_waiting(self, build):
        env = await build()
        await env.funnel.ingest_and_wake(env.channel, "critique", "critique", ["x"])
        start = time.monotonic()
        out = await ponder_tool(env)(5.0, _ctx=ctx_for(env))
        assert time.monotonic() - start < 0.5 and "woke" in out.lower()

    async def test_other_channels_origin_payload_does_not_count(self, build):
        env = await build()
        other = SimpleNamespace(id="test:other", conversation=None)
        await env.funnel.ingest_and_wake(other, "critique", "critique", ["x"])
        out = await ponder_tool(env)(0.2, _ctx=ctx_for(env))
        assert "woke" not in out.lower()


class TestMixedFlags:

    async def test_stale_empty_list_does_not_orphan_a_waiter(self, build):
        env = await build()
        ponder = ponder_tool(env)
        # A (no early wake, long) and B (no early wake, short) leave the key unregistered;
        # C registers; A finishing must not unregister C.
        a = asyncio.create_task(ponder(0.4, until_context=False, _ctx=ctx_for(env)))
        b = asyncio.create_task(ponder(0.05, until_context=False, _ctx=ctx_for(env)))
        await settle()
        await b
        c = asyncio.create_task(ponder(5.0, _ctx=ctx_for(env)))
        await settle()
        await a
        env.funnel.ingest(env.channel, "critique", ["x"])
        out = await asyncio.wait_for(c, 1.0)
        assert "woke" in out.lower()


class TestInputValidation:

    @pytest.mark.parametrize("bad", ["3", True, None])
    async def test_non_numbers_are_refused_not_crashed(self, build, bad):
        env = await build()
        out = await ponder_tool(env)(bad, _ctx=ctx_for(env))
        assert out.startswith("Error:") and "seconds" in out and repr(bad) in out


class TestModelVisibleDescription:
    """The model sees only the first docstring line, so the guidance lives there."""

    async def test_description_says_when_to_use_it_and_the_limits(self, build):
        env = await build()
        registry: list = []
        env.pm.hook.register_tools(tool_registry=registry)
        tool = next(t for t in registry if t.name == "ponder")
        desc = tool.schema["function"]["description"].lower()
        # Waiting on background recall is the point of the tool.
        assert "memory" in desc and "recall" in desc
        # The cap and the early exit are what the model can rely on.
        assert "5 seconds" in desc and "ends early" in desc
        # One line, as the schema builder keeps only the first.
        assert "\n" not in desc


class TestWakeOrder:

    async def test_ponderers_wake_after_the_stub_is_queued(self, build):
        env = await build()
        task = asyncio.create_task(ponder_tool(env)(5.0, _ctx=ctx_for(env)))
        await settle()
        seen: list[bool] = []

        async def on_notify(**kwargs):
            # At stub-enqueue time no ponder may have been woken yet.
            seen.append(any(e.is_set() for e in env.funnel._ponderers[env.channel.id]))

        env.pm.ahook.on_notify = on_notify
        await env.funnel.ingest_and_wake(env.channel, "user", "critique", ["x"])
        await asyncio.wait_for(task, 1.0)
        assert seen == [False]
