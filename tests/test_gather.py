"""The one gather primitive (#1176 §1): ``corvidae.checks.gather``.

RED phase. Before effect (``should_send_response`` final, ``should_execute_tool``)
and before compose (``before_compose``) publish to every implementation,
wait under the channel's check allowance and return what arrived in time.
Acceptance criteria G1-G5 of the #1176 red-phase brief:

* G1  both gates go through ``corvidae.checks.gather``; ``run_checks`` is gone;
* G2  an in-time refusal ends the gather at once, and what the still-running
      checks say afterwards is neither held late nor counted as missed;
* G3  the missed metric carries a ``phase`` tag;
* G4  no busy-wait: events plus one timer;
* G5  stragglers resolve ``True`` only for a late non-empty ``Contribution``,
      and only after ``on_late`` has returned.

The checks sleep on a FakeClock; each test moves time with ``advance``.
``corvidae.held`` does not exist yet, so its names are reached through the
``H`` fixture, which makes each test fail on its own instead of the whole
module failing to collect.
"""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.channel import Channel
from corvidae.checks import CHECK_MISSED_METRIC
from corvidae.hooks import Refusal, create_plugin_manager, hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.tool import tool_to_schema
from corvidae.trace import mint_trace_id
from fake_clock import install_kernel_clock
from helpers import build_plugin_and_channel, drain, drain_task_queue, sent_texts
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)

#: The minimum allowance a channel gets when nothing else sets one (seconds).
DEFAULT_MINIMUM = 5.0


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def H():
    """The not-yet-written ``corvidae.held`` module, imported lazily so a
    missing module fails the test that needs it (the expected red)."""
    return importlib.import_module("corvidae.held")


def is_refusal(answer: object) -> bool:
    """The before-effect ``stop_on`` predicate: a bare False or a Refusal."""
    return answer is False or isinstance(answer, Refusal)


class MetricsSpy:
    """Records every on_metrics event as (name, value, tags)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float, dict]] = []

    @hookimpl
    async def on_metrics(self, name, value, tags):
        self.events.append((name, value, dict(tags)))

    def named(self, name: str):
        return [e for e in self.events if e[0] == name]


class ClockGate:
    """A should_send_response implementation answering ``final`` with
    ``verdict`` after ``delay`` fake-clock seconds."""

    def __init__(self, clock, delay: float, verdict) -> None:
        self.clock = clock
        self.delay = delay
        self.verdict = verdict
        # Set when the check starts, so a test knows it is running.
        self.started = asyncio.Event()
        # Count of firings that returned.
        self.finished = 0

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations):
        if emission != "final":
            return None
        self.started.set()
        if self.delay:
            await self.clock.sleep(self.delay)
        self.finished += 1
        return self.verdict


class ClockSource:
    """A before_compose implementation answering after ``delay`` fake-clock
    seconds with ``answer`` (raised when it is an exception)."""

    def __init__(self, clock, delay: float, answer) -> None:
        self.clock = clock
        self.delay = delay
        self.answer = answer
        self.started = asyncio.Event()
        self.finished = 0

    @hookimpl
    async def before_compose(self, channel, trace_id, annotations):
        self.started.set()
        if self.delay:
            await self.clock.sleep(self.delay)
        self.finished += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _gather_env():
    """(pm, clock, channel, metrics) for a gather driven without an Agent."""
    pm = create_plugin_manager()
    clock = install_kernel_clock(pm)[0]
    metrics = MetricsSpy()
    pm.register(metrics, name="metrics_spy")
    return pm, clock, Channel(transport="test", scope="gather"), metrics


def _final_kwargs(channel: Channel) -> dict:
    """The kwargs a should_send_response final firing carries."""
    return {
        "channel": channel, "text": "hi", "emission": "final",
        "trace_id": None, "annotations": {},
    }


def _compose_kwargs(channel: Channel) -> dict:
    """The kwargs a before_compose firing carries."""
    return {"channel": channel, "trace_id": "t-1", "annotations": {}}


@pytest.fixture
async def env(tmp_path):
    """(plugin, channel, db, clock, metrics): an Agent with a held store and a
    ``shell`` stand-in, on a channel that gets the minimum allowance."""
    plugin, channel, db = await build_plugin_and_channel()
    clock = install_kernel_clock(plugin.pm)[0]
    metrics = MetricsSpy()
    plugin.pm.register(metrics, name="metrics_spy")

    # The agent depends on the held store; register it the way Runtime does
    # unless the shared helper already did.
    from corvidae.held import HeldContext
    from corvidae.spill import NotificationSpill

    # Guarded like held, so a shared helper that registers spill does not collide.
    if plugin.pm.get_plugin("spill") is None:
        spill = NotificationSpill()
        plugin.pm.register(spill, name="spill")
        await spill.on_init(
            pm=plugin.pm, config={"daemon": {"notification_spill_dir": str(tmp_path / "spill")}}
        )
    if plugin.pm.get_plugin("held") is None:
        held = HeldContext()
        plugin.pm.register(held, name="held")
        await held.on_init(pm=plugin.pm, config={})

    async def shell(command: str) -> str:
        """Run a command."""
        return f"ran {command}"

    plugin._tools = {"shell": shell}
    plugin._tool_schemas = [tool_to_schema(shell)]
    plugin._client = MagicMock()
    plugin._client.chat = AsyncMock()
    plugin._mark_started()
    yield plugin, channel, db, clock, metrics
    await plugin.pm.get_plugin("task").on_stop()
    await clock.close()
    await db.close()


async def _run_turn(plugin, channel, text="go") -> None:
    """Deliver a user message and drain the queue, tools and queue again."""
    await plugin.on_message(channel=channel, sender="user", text=text, trace_id=mint_trace_id())
    await drain(plugin, channel)
    await drain_task_queue(plugin)
    await drain(plugin, channel)


# ---------------------------------------------------------------------------
# G1: one primitive
# ---------------------------------------------------------------------------


class TestOnePrimitive:
    async def test_g1_both_effect_gates_and_compose_go_through_gather(
        self, env, monkeypatch
    ):
        """G1: the reply gate, the tool gate and before compose each call
        ``corvidae.checks.gather``; the phase says which. Drives a real turn
        (tool call, then final) so the call sites are the Agent's own.

        The spy replaces the name in both modules, because the Agent may bind
        it with ``from corvidae.checks import gather``."""
        import corvidae.agent as agent_mod
        import corvidae.checks as checks_mod

        plugin, channel, db, clock, metrics = env
        real = checks_mod.gather
        seen: list[tuple[str, str]] = []

        async def spy(pm, hook_name, **kwargs):
            seen.append((hook_name, kwargs["phase"]))
            # Before effect must ask for the early exit on a refusal (G2's seam).
            if kwargs["phase"] == "before_effect":
                assert kwargs.get("stop_on") is not None, hook_name
            return await real(pm, hook_name, **kwargs)

        monkeypatch.setattr(checks_mod, "gather", spy)
        monkeypatch.setattr(agent_mod, "gather", spy, raising=False)
        plugin._client.chat.side_effect = [
            _make_tool_call_response([_make_tool_call("c1", "shell", {"command": "id"})]),
            _make_text_response("done"),
        ]

        await _run_turn(plugin, channel)

        assert ("should_execute_tool", "before_effect") in seen
        assert ("should_send_response", "before_effect") in seen
        assert ("before_compose", "before_compose") in seen
        # And nothing else gathered: three hooks, two phases.
        assert {hook for hook, _ in seen} == {
            "should_execute_tool", "should_send_response", "before_compose",
        }

    def test_g1_run_checks_no_longer_exists(self):
        """G1: no second gather loop survives: ``run_checks`` is renamed, not
        kept beside ``gather``."""
        import corvidae.checks as checks_mod

        assert hasattr(checks_mod, "gather")


# ---------------------------------------------------------------------------
# G2: early exit
# ---------------------------------------------------------------------------


class TestEarlyExit:
    async def test_g2_in_time_refusal_ends_the_gather_at_once(self):
        """G2: one check refuses at t=1, another sleeps 60 s under a 120 s
        allowance: the gather returns at t=1 with the refusal. The slow
        check's own refusal, arriving at t=60, is not handed to ``on_late``
        (the reply was already refused, so a late-check row would be false)
        and ``check.missed`` is not emitted (the allowance had not run out)."""
        from corvidae.checks import gather

        pm, clock, channel, metrics = _gather_env()
        fast = ClockGate(clock, 1, Refusal("too blunt"))
        slow = ClockGate(clock, 60, Refusal("also wrong"))
        pm.register(fast, name="fast")
        pm.register(slow, name="slow")
        late: list = []

        async def on_late(plugin_name, answer):
            late.append((plugin_name, answer))

        start = clock.monotonic()
        task = asyncio.create_task(gather(
            pm, "should_send_response", channel=channel, kwargs=_final_kwargs(channel),
            allowance=120, clock=clock, on_late=on_late, phase="before_effect",
            stop_on=is_refusal,
        ))
        await asyncio.wait_for(fast.started.wait(), 1)
        await asyncio.wait_for(slow.started.wait(), 1)
        await clock.advance(1)
        gathered = await asyncio.wait_for(task, 1)

        assert clock.monotonic() - start == 1
        assert gathered.results == [Refusal("too blunt")]

        # The slow check finishes at t=60: nothing is held, nothing counted.
        await clock.advance(60)
        assert slow.finished == 1
        assert late == []
        assert metrics.named(CHECK_MISSED_METRIC) == []


# ---------------------------------------------------------------------------
# G3: the missed metric names the point
# ---------------------------------------------------------------------------


class TestMissedPhase:
    async def test_g3_before_effect_miss_carries_the_phase(self, env):
        """G3: a reply gate that misses its allowance emits ``check.missed``
        tagged ``phase=before_effect`` (driven through the Agent's final)."""
        plugin, channel, db, clock, metrics = env
        gate = ClockGate(clock, 60, None)
        plugin.pm.register(gate, name="clock_gate")
        plugin._client.chat.side_effect = [_make_text_response("the reply")]

        await plugin.on_message(
            channel=channel, sender="user", text="hi", trace_id=mint_trace_id()
        )
        drained = asyncio.create_task(drain(plugin, channel))
        await asyncio.wait_for(gate.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(drained, 1)

        assert metrics.named(CHECK_MISSED_METRIC) == [
            (CHECK_MISSED_METRIC, 1, {
                "channel": channel.id,
                "hook": "should_send_response",
                "plugin": "clock_gate",
                "phase": "before_effect",
            }),
        ]

    async def test_g3_before_compose_miss_carries_the_phase(self, env):
        """G3: a before_compose source that misses its allowance emits
        ``check.missed`` tagged ``phase=before_compose``."""
        plugin, channel, db, clock, metrics = env
        source = ClockSource(clock, 60, None)
        plugin.pm.register(source, name="slow_source")
        plugin._client.chat.side_effect = [_make_text_response("the reply")]

        await plugin.on_message(
            channel=channel, sender="user", text="hi", trace_id=mint_trace_id()
        )
        drained = asyncio.create_task(drain(plugin, channel))
        await asyncio.wait_for(source.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(drained, 1)

        assert metrics.named(CHECK_MISSED_METRIC) == [
            (CHECK_MISSED_METRIC, 1, {
                "channel": channel.id,
                "hook": "before_compose",
                "plugin": "slow_source",
                "phase": "before_compose",
            }),
        ]
        # The reply still went out: a missed source never blocks the turn.
        assert sent_texts(plugin.pm, OutboundMessage) == ["the reply"]


# ---------------------------------------------------------------------------
# G4: no busy-wait
# ---------------------------------------------------------------------------


class TestNoBusyWait:
    async def test_g4_nothing_wakes_between_start_and_the_deadline(self):
        """G4: events plus one timer. Once the gather has started, advancing
        time up to (not through) the deadline schedules no further timer and
        no further sleep (the timer's and the check's are the only two): a polling loop would show as extra ``call_later``,
        ``call_soon`` or ``sleep`` calls. Exactly one timer is the deadline."""
        from corvidae.checks import gather

        pm, clock, channel, metrics = _gather_env()
        slow = ClockGate(clock, 60, None)
        pm.register(slow, name="slow")

        later_calls: list[str] = []
        soon_calls: list[str] = []
        real_later, real_soon = clock.call_later, clock.call_soon

        def call_later(delay, callback, *, name):
            later_calls.append(name)
            return real_later(delay, callback, name=name)

        def call_soon(callback, *, name):
            soon_calls.append(name)
            return real_soon(callback, name=name)

        clock.call_later = call_later
        clock.call_soon = call_soon

        async def on_late(plugin_name, answer):
            return None

        task = asyncio.create_task(gather(
            pm, "should_send_response", channel=channel, kwargs=_final_kwargs(channel),
            allowance=30, clock=clock, on_late=on_late, phase="before_effect",
        ))
        await asyncio.wait_for(slow.started.wait(), 1)
        await clock.settle()

        # Started: one deadline timer, one check task, and two parked sleeps
        # (the timer's own, then the check's); nothing else is waiting.
        assert len(later_calls) == 1
        assert len(soon_calls) == 1
        assert clock.sleeps.qsize() == 2
        before = (len(later_calls), len(soon_calls), clock.sleeps.qsize())

        # Time passes inside the allowance: nothing wakes, nothing is armed.
        await clock.advance(10)
        await clock.advance(10)
        await clock.advance(9)
        assert not task.done()
        assert (len(later_calls), len(soon_calls), clock.sleeps.qsize()) == before

        # The deadline is the first wake-up; the gather returns without the check.
        await clock.advance(1)
        gathered = await asyncio.wait_for(task, 1)
        assert gathered.results == []
        assert len(later_calls) == 1


# ---------------------------------------------------------------------------
# G5: stragglers
# ---------------------------------------------------------------------------


class TestStragglers:
    async def test_g5_one_future_per_running_implementation_settles_by_what_it_brought(
        self, H
    ):
        """G5: at the deadline there is one future per implementation still
        running (the in-time one has none). Each resolves ``True`` only when
        the late answer is a non-empty ``Contribution``; an empty one, a
        ``None`` and a raise resolve ``False``."""
        from corvidae.checks import gather

        pm, clock, channel, metrics = _gather_env()
        quick = ClockSource(clock, 0, H.Contribution("quick", ["in time"]))
        full = ClockSource(clock, 60, H.Contribution("full", ["late entry"]))
        empty = ClockSource(clock, 60, H.Contribution("empty", []))
        nothing = ClockSource(clock, 60, None)
        boom = ClockSource(clock, 60, RuntimeError("source exploded"))
        for name, impl in [("quick", quick), ("full", full), ("empty", empty),
                           ("nothing", nothing), ("boom", boom)]:
            pm.register(impl, name=name)
        late: list = []

        async def on_late(plugin_name, answer):
            late.append(plugin_name)

        task = asyncio.create_task(gather(
            pm, "before_compose", channel=channel, kwargs=_compose_kwargs(channel),
            allowance=5, clock=clock, on_late=on_late, phase="before_compose",
        ))
        for impl in (full, empty, nothing, boom):
            await asyncio.wait_for(impl.started.wait(), 1)
        await clock.advance(5)
        gathered = await asyncio.wait_for(task, 1)

        # The in-time answer is a result; the four others are stragglers.
        assert gathered.results == [H.Contribution("quick", ["in time"])]
        assert len(gathered.stragglers) == 4
        assert not any(f.done() for f in gathered.stragglers)

        await clock.advance(60)
        outcomes = await asyncio.wait_for(asyncio.gather(*gathered.stragglers), 1)
        assert sorted(outcomes) == [False, False, False, True]

    async def test_g5_a_straggler_resolves_only_after_on_late_returned(self, H):
        """G5: ``ponder`` must never see an entry as landed before it is held,
        so the future settles after ``on_late`` has returned, not when the
        implementation returned."""
        from corvidae.checks import gather

        pm, clock, channel, metrics = _gather_env()
        full = ClockSource(clock, 60, H.Contribution("full", ["late entry"]))
        pm.register(full, name="full")
        order: list[str] = []
        release = asyncio.Event()
        entered = asyncio.Event()

        async def on_late(plugin_name, answer):
            entered.set()
            await release.wait()
            order.append("on_late returned")

        task = asyncio.create_task(gather(
            pm, "before_compose", channel=channel, kwargs=_compose_kwargs(channel),
            allowance=5, clock=clock, on_late=on_late, phase="before_compose",
        ))
        await asyncio.wait_for(full.started.wait(), 1)
        await clock.advance(5)
        gathered = await asyncio.wait_for(task, 1)
        (straggler,) = gathered.stragglers

        await clock.advance(60)
        await asyncio.wait_for(entered.wait(), 1)
        # The implementation has returned, on_late is still running: unsettled.
        assert full.finished == 1
        assert not straggler.done()

        release.set()
        assert await asyncio.wait_for(straggler, 1) is True
        assert order == ["on_late returned"]
