"""Critique runs every selected lens in one fork (#1190 PR 2).

One test (or one parametrized test) per acceptance criterion of #1190
PR 2's design (requirements on issue #1190), named by its tag:

- S1      every offered lens above "none" is selected; "none" wins ties;
- R1      the selected lenses are one fork, each lens's wake setting and
          outcome survive, in one record;
- R5.1/2  the idle fork's tool cap is ``critique.max_tool_calls`` (3), the
          before-send fork's is 0;
- R5.3    no model call while a user message waits (before render and
          between rounds), a compaction during the fork is a miss;
- R5.4    findings name the reply (or draft) they judge;
- R5.5    ``critique.fork_timeout`` bounds the whole fork;
- I3      the idle fork is cut at the reply;
- config  both keys are validated at startup, reload and fork time;
- tail    one reply-format tail per fork, briefs carry none.

Critique is driven the way the daemon drives it (``on_agent_response``,
appraisal's ``on_appraisal_idle``, then the queued task). The fork agent is a
fake that behaves as ``Agent.fork`` does for the parameters under test; the
kernel half (``yield_to_caller``) is tested against the real ``Agent.fork`` in
tests/test_fork_read_only_tools.py.
"""

import asyncio
import json
import re
import logging
import time as time_module

import pytest

from corvidae_cognition import critique as critique_module
from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS

from helpers import FakeForkAgent, window_of
from tests.cognition.test_critique import (
    FakeAppraisal, FixedRng, _read_outcomes, _teardown, build_critique,
)
from tests.cognition.test_critique_hooks import _begin
from tests.cognition.test_critique_vector import (
    BRIEF, Contributor, _env, _judged_turn, _lens, _warnings, _watch_held,
)

# A keyed reply, as the combined fork's reply format asks for (one key per lens).
def _keyed(**objections: str) -> str:
    """A reply with one objection per named lens (lens=objection text)."""
    return json.dumps({
        lens: {"objections": [{"claim": f"claim of {lens}", "objection": text}]}
        for lens, text in objections.items()
    })


KEYED_PREDICTIVE = _keyed(predictive="predictive-objection-3k")
PREDICTIVE_ONLY = {"predictive": 1.0}
TWO = {"predictive": 0.5, "constrained": 0.4, "none": 0.1}


class RoundsForkAgent(FakeForkAgent):
    """A fork agent whose fork makes ``rounds`` model requests, handing control
    to ``yield_to_caller`` between them as the kernel's tool loop does.

    ``waiting`` is what ``waiting_messages`` reports; ``events`` logs each
    request; ``on_request(i, channel)`` runs inside request i (0-based).
    """

    def __init__(self, *args, rounds=1, on_request=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rounds = rounds
        self.on_request = on_request
        self.waiting: dict[str, int] = {}
        self.events: list[str] = []

    def waiting_messages(self):
        return dict(self.waiting)

    async def fork(self, channel, instruction, **kwargs):
        # Record the call (and fail like the base fake does) before any round.
        result = await super().fork(channel, instruction, **kwargs)
        for i in range(self.rounds):
            if i:
                await kwargs["yield_to_caller"]()
            self.events.append(f"request {i + 1}")
            if self.on_request is not None:
                await self.on_request(i, channel)
        return result


async def _env_with(tmp_path, agent, *lenses, config=None, with_held=False):
    """critique's wiring with ``agent`` as the Agent plugin and a window of
    rows 1 (user), 2 (the reply) and 50 (a later user message)."""
    env = await build_critique(
        tmp_path, FakeAppraisal(), chat_text=KEYED_PREDICTIVE, config=config,
        with_held=with_held, fork=False,
        extra_plugins={"kb": Contributor(*lenses)} if lenses else None)
    env.pm.register(agent, name="agent")
    env.channel.conversation = window_of(env.channel.id, [1, 2, 50])
    env.fork_agent = agent
    env.plugin._rng = FixedRng(value=0.99)
    return env


@pytest.fixture
def fast_poll(monkeypatch):
    """Poll the waiting-message count every 10 ms instead of every second."""
    monkeypatch.setattr(
        critique_module, "YIELD_POLL_SECONDS", 0.01, raising=False)


async def _trigger(env, point, vector, trace_id="kp"):
    """Run critique's ``point`` ("idle" or "before_send") on ``vector``."""
    if point == "idle":
        await _judged_turn(env, trace_id, vector)
        return None
    await _begin(env, trace_id)
    return await env.plugin.on_appraisal_before_send(
        channel=env.channel, trace_id=trace_id, text="short draft", annotations={},
        lenses=vector)


# ---------------------------------------------------------------------------
# S1 - every lens above "none" is selected
# ---------------------------------------------------------------------------


async def test_s1_every_lens_above_none_is_selected_and_recorded(tmp_path):
    """S1: {predictive .4, constrained .35, adversarial 0, none .25} selects
    predictive and constrained, and the selection record lists both."""
    env = await _env(tmp_path)
    try:
        await _judged_turn(env, "ks1", {
            "predictive": 0.4, "constrained": 0.35, "adversarial": 0.0, "none": 0.25})
        selection = (await _read_outcomes(env.recorder, "ks1"))["lens_selection"]
        assert selection["lenses"] == ["predictive", "constrained"]
        [call] = env.fork_agent.calls
        assert "[Self-critique: adversarial]" not in call["instruction"]
    finally:
        await _teardown(env)


async def test_s1_none_wins_ties(tmp_path):
    """S1: {predictive .3, constrained .2, adversarial .2, none .3} selects
    nothing: "none" ties the top lens and wins; no fork runs."""
    env = await _env(tmp_path)
    try:
        await _judged_turn(env, "ks1t", {
            "predictive": 0.3, "constrained": 0.2, "adversarial": 0.2, "none": 0.3})
        selection = (await _read_outcomes(env.recorder, "ks1t"))["lens_selection"]
        assert "lenses" not in selection
        assert env.fork_agent.calls == []
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R1 - one fork, per-lens wake setting and records
# ---------------------------------------------------------------------------


async def test_r1_two_lenses_one_fork_waking_split_from_it(tmp_path):
    """R1: predictive (waits) and proactivity (wakes) are exactly one fork;
    proactivity's objection wakes the channel and predictive's waits, both
    from that fork's reply, and the record counts both."""
    agent = FakeForkAgent({"critique"}, text=_keyed(
        predictive="WAITS-objection-7q", proactivity="WAKES-objection-7q"))
    env = await _env_with(tmp_path, agent, with_held=True)
    try:
        _watch_held(env)
        await _judged_turn(env, "kr1", {"predictive": 0.5, "proactivity": 0.4, "none": 0.1})
        assert len(agent.calls) == 1
        # The waking lens's finding fires the stub; the other one only waits.
        env.held.hold_and_wake.assert_awaited_once()
        woken = "\n".join(env.held.hold_and_wake.await_args.kwargs["entries"])
        assert "WAKES-objection-7q" in woken and "WAITS-objection-7q" not in woken
        env.held.hold.assert_awaited_once()
        waited = "\n".join(env.held.hold.await_args.kwargs["entries"])
        assert "WAITS-objection-7q" in waited and "WAKES-objection-7q" not in waited
        critique = (await _read_outcomes(env.recorder, "kr1"))["critique"]
        assert critique["lenses"] == ["predictive", "proactivity"]
        assert critique["objections"] == 2
    finally:
        await _teardown(env)


async def test_r1_built_in_and_added_lens_keep_their_own_records(tmp_path):
    """R1: a built-in lens and an added lens in one fork share ONE
    ``critique`` record: the built-ins' keys beside the added lens's entry
    (lens names never collide with them, RESERVED_LENS_NAMES), so a reader
    keeping the last record per key loses neither."""
    agent = FakeForkAgent({"critique"}, text=_keyed(
        predictive="built-in-objection-2m", mine="added-objection-2m"))
    env = await _env_with(tmp_path, agent, _lens("mine"))
    try:
        await _judged_turn(env, "kr1b", {"predictive": 0.5, "mine": 0.4, "none": 0.1})
        assert len(agent.calls) == 1
        records = [r["record"]["critique"] for r in env.recorder.by_source("critique")
                   if r["trace_id"] == "kr1b" and "critique" in r["record"]]
        assert len(records) == 1
        record = records[0]
        assert record["lenses"] == ["predictive"] and record["objections"] == 1
        assert record["mine"]["objections"] == 1 and record["mine"]["sampled"] is False
    finally:
        await _teardown(env)


async def test_r1_before_send_one_fork_and_records_carry_lenses(tmp_path):
    """R1 and the record shapes: two lenses before send are one fork; both
    ``lens_selection_before_send`` and ``critique_before_send`` carry
    ``lenses`` as a list."""
    env = await _env(tmp_path)
    try:
        await _trigger(env, "before_send", TWO, "krb")
        assert len(env.fork_agent.calls) == 1
        outcomes = await _read_outcomes(env.recorder, "krb")
        assert outcomes["lens_selection_before_send"]["lenses"] == ["predictive", "constrained"]
        assert outcomes["critique_before_send"]["lenses"] == ["predictive", "constrained"]
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R5.1, R5.2 - the tool cap; I3 - the cut
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config,expected", [
    (None, 3),                                      # unset: the default
    ({"critique": {"max_tool_calls": 5}}, 5),       # operator-set
    ({"critique": {"max_tool_calls": 0}}, 0),       # zero is valid: no tools
])
async def test_r5_1_idle_fork_gets_the_configured_cap(tmp_path, config, expected):
    """R5.1: the idle fork is called with ``max_tool_calls`` = the configured
    value, 3 when unset."""
    env = await _env(tmp_path, config=config)
    try:
        await _judged_turn(env, "kc", PREDICTIVE_ONLY)
        [call] = env.fork_agent.calls
        assert call["max_tool_calls"] == expected
    finally:
        await _teardown(env)


async def test_r5_2_before_send_fork_calls_no_tools_and_never_yields(tmp_path):
    """R5.2: the before-send fork is called with ``max_tool_calls=0``, even
    when the idle cap is set, and with no ``yield_to_caller`` (it never waits)."""
    env = await _env(tmp_path, config={"critique": {"max_tool_calls": 5}})
    try:
        await _trigger(env, "before_send", PREDICTIVE_ONLY)
        [call] = env.fork_agent.calls
        assert call["max_tool_calls"] == 0
        assert call["yield_to_caller"] is None
    finally:
        await _teardown(env)


async def test_i3_idle_fork_is_cut_at_the_reply(tmp_path):
    """I3: the idle fork's ``cut`` is the index after the reply's row: the
    window holds rows 1, 2 (the reply) and 50, so the cut is 2 and row 50 is
    out of the fork's view."""
    env = await _env(tmp_path)
    try:
        await _judged_turn(env, "ki3", PREDICTIVE_ONLY)
        [call] = env.fork_agent.calls
        assert call["cut"] == 2
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R5.3 - no model call while a user message waits; compaction is a miss
# ---------------------------------------------------------------------------


async def test_r5_3_fork_does_not_start_while_a_message_waits(tmp_path, fast_poll):
    """R5.3 / I3: with a user message waiting on the channel critique does not
    fork (so never renders mid-turn); it forks once the channel clears."""
    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE)
    env = await _env_with(tmp_path, agent)
    agent.waiting[env.channel.id] = 1
    run = asyncio.create_task(_judged_turn(env, "kw1", PREDICTIVE_ONLY))
    try:
        await asyncio.sleep(0.1)
        assert agent.calls == []
        agent.waiting.clear()
        await asyncio.wait_for(run, 5)
        assert len(agent.calls) == 1
    finally:
        run.cancel()
        await _teardown(env)


async def test_r5_3_second_request_waits_for_the_channel_to_clear(tmp_path, fast_poll):
    """R5.3: a message arrives during the first request; the second request
    is not sent until it clears, and the lens is then recorded."""
    async def message_arrives(i, channel):
        # A user message lands while the first request is in flight.
        if i == 0:
            agent.waiting[channel.id] = 1

    agent = RoundsForkAgent(
        {"critique"}, text=KEYED_PREDICTIVE, rounds=2, on_request=message_arrives)
    env = await _env_with(tmp_path, agent)
    run = asyncio.create_task(_judged_turn(env, "kw2", PREDICTIVE_ONLY))
    try:
        await asyncio.sleep(0.1)
        assert agent.events == ["request 1"]
        agent.waiting.clear()
        await asyncio.wait_for(run, 5)
        assert agent.events == ["request 1", "request 2"]
        assert (await _read_outcomes(env.recorder, "kw2"))["critique"]["lenses"] == ["predictive"]
    finally:
        run.cancel()
        await _teardown(env)


@pytest.mark.parametrize("rounds", [2, 1], ids=["between-rounds", "last-request"])
async def test_r5_3_compaction_during_the_fork_is_a_miss(tmp_path, caplog, rounds):
    """R5.3: the channel compacts while the fork runs (before its next
    request, or during its last one): the lens skips with a warning naming
    lens, trace and channel, and nothing is recorded or held."""
    async def compacts(i, channel):
        # The first request is in flight when the window is compacted.
        if i == 0:
            await env.plugin.before_compaction(channel=channel, compacted_ids=[], cut=0)

    agent = RoundsForkAgent(
        {"critique"}, text=KEYED_PREDICTIVE, rounds=rounds, on_request=compacts)
    env = await _env_with(tmp_path, agent, with_held=True)
    _watch_held(env)
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kw3", PREDICTIVE_ONLY)
        [text] = _warnings(caplog, "skipped")
        assert "predictive" in text and "trace kw3 " in text and env.channel.id in text
        assert "compacted while the fork ran" in text
        # A second request was never sent.
        assert agent.events == ["request 1"]
        assert "critique" not in (await _read_outcomes(env.recorder, "kw3"))
        env.held.hold.assert_not_awaited()
        env.held.hold_and_wake.assert_not_awaited()
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R5.4 - findings name the reply they judge
# ---------------------------------------------------------------------------


async def test_r5_4_idle_finding_quotes_the_reply(tmp_path):
    """R5.4: each held finding from an idle lens quotes the start of the reply."""
    env = await _env(tmp_path, with_held=True)
    try:
        _watch_held(env)
        await _judged_turn(env, "k54", PREDICTIVE_ONLY)
        entries = env.held.hold.call_args.kwargs["entries"]
        assert entries and all("on your reply «resp»" in e for e in entries), entries
    finally:
        await _teardown(env)


async def test_r5_4_before_send_finding_quotes_the_draft(tmp_path):
    """R5.4: a before-send finding says it is about the draft, quoting it."""
    env = await _env(tmp_path)
    try:
        result = await _trigger(env, "before_send", PREDICTIVE_ONLY)
        assert result and all("on your draft «short draft»" in t for t in result), result
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R5.5 - the deadline
# ---------------------------------------------------------------------------


async def test_r5_5_fork_past_the_deadline_is_cancelled_and_a_miss(tmp_path, caplog):
    """R5.5: a fork that runs past ``critique.fork_timeout`` is cancelled;
    the warning names the key, lens, trace and channel; no outcome is recorded."""
    cancelled = []

    async def hangs(i, channel):
        # The request outlasts the deadline (but not the test's own timeout,
        # so a missing deadline fails an assertion instead of hanging); record
        # the cancellation the deadline causes.
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE, on_request=hangs)
    env = await _env_with(tmp_path, agent, config={"critique": {"fork_timeout": 0.05}})
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kd1", PREDICTIVE_ONLY)
        assert cancelled == [True]
        [text] = _warnings(caplog, "exceeded critique.fork_timeout")
        assert "predictive" in text and "trace kd1 " in text and env.channel.id in text
        assert "critique" not in (await _read_outcomes(env.recorder, "kd1"))
    finally:
        await _teardown(env)


async def test_r5_5_a_channel_busy_past_the_deadline_is_a_miss(tmp_path, caplog, fast_poll):
    """R5.5 / I2: a channel that stays busy past the deadline never reaches
    the fork; the warning says the channel stayed busy, and names the key."""
    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE)
    env = await _env_with(tmp_path, agent, config={"critique": {"fork_timeout": 0.05}})
    agent.waiting[env.channel.id] = 1
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kd2", PREDICTIVE_ONLY)
        assert agent.calls == []
        [text] = _warnings(caplog, "stayed busy past critique.fork_timeout")
        assert "predictive" in text and "trace kd2 " in text and env.channel.id in text
        assert "critique" not in (await _read_outcomes(env.recorder, "kd2"))
    finally:
        await _teardown(env)


# --- how long the fork waited (#1272) -------------------------------------------------
#
# A timeout with no duration attached is a dead end: 208 skips in one 6-hour
# window on one container all said "exceeded critique.fork_timeout (120.0 s)" and
# nothing said whether the fork was slow or hung, or how long the successful ones
# took. These three tests hold the two numbers that turn the symptom into a
# diagnosis: the elapsed wall time on the deadline's own warning, and a warning
# for a fork that finished but spent most of its allowance getting there.

async def test_r5_5_the_deadline_warning_says_how_long_the_fork_waited(tmp_path, caplog):
    """The timeout warning carries the wall time the fork occupied, so a slow
    fork and a hung one are distinguishable from the log alone."""
    async def hangs(i, channel):
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            raise

    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE, on_request=hangs)
    env = await _env_with(tmp_path, agent, config={"critique": {"fork_timeout": 0.05}})
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kd7", PREDICTIVE_ONLY)
        [text] = _warnings(caplog, "exceeded critique.fork_timeout")
        m = re.search(r"after (\d+\.\d+)s", text)
        assert m, f"the warning must state its own elapsed time: {text!r}"
        assert 0.05 <= float(m.group(1)) < 3.0, text
    finally:
        await _teardown(env)


class _ScriptedTime:
    """`time` as the reporting code would see it after a runner stall: every
    `monotonic()` read moves by `step`, so a span measured across two reads
    collapses to `step` no matter how long the wait really was. Everything else
    on the module delegates to the real thing, which keeps the blast radius to
    critique's three `time.monotonic()` calls. The event loop's clock, which
    `asyncio.timeout` actually arms and compares against, is untouched."""

    def __init__(self, step: float, start: float = 1_000.0) -> None:
        self.step = step
        self.now = start

    def monotonic(self) -> float:
        self.now += self.step
        return self.now

    def __getattr__(self, name):
        return getattr(time_module, name)


async def test_r5_5_a_stall_cannot_make_a_full_wait_log_as_no_wait(tmp_path, caplog, monkeypatch):
    """The logged wait is measured on the clock the deadline armed from (#1304).

    `started` was a `time.monotonic()` read *after* entering `asyncio.timeout`,
    so a stall between arming and that read spends the allowance before the
    measurement begins: `deadline.expired()` is true while the reported span is
    ~0, and the line says "after 0.0s" for a fork that was never given a turn —
    the exact confusion (#1272) the line exists to rule out. The scripted clock
    stands in for that stall as the reporting code experiences it.
    """
    async def hangs(i, channel):
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            raise

    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE, on_request=hangs)
    env = await _env_with(tmp_path, agent, config={"critique": {"fork_timeout": 0.05}})
    monkeypatch.setattr(critique_module, "time", _ScriptedTime(step=0.0002), raising=False)
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kds", PREDICTIVE_ONLY)
        [text] = _warnings(caplog, "exceeded critique.fork_timeout")
        m = re.search(r"after (\d+\.\d+)s", text)
        assert m, f"the warning must state its own elapsed time: {text!r}"
        assert float(m.group(1)) >= 0.05, (
            "a full timeout must not be reported as a shorter wait; the report "
            f"comes from a clock that can start after the timer: {text!r}"
        )
    finally:
        await _teardown(env)


class _StalledDeadline:
    """The real deadline, entered normally and then left with the loop blocked.

    `asyncio.timeout` arms when the context is entered; the stall that follows
    stands in for a runner that does not get back to the coroutine until the
    allowance is gone — the #1304 pathology, injected instead of waited for.
    """

    def __init__(self, inner, stall: float) -> None:
        self.inner = inner
        self.stall = stall

    async def __aenter__(self):
        await self.inner.__aenter__()
        time_module.sleep(self.stall)
        return self.inner

    async def __aexit__(self, *exc):
        return await self.inner.__aexit__(*exc)

    def expired(self) -> bool:
        return self.inner.expired()


class _StallingAsyncio:
    """The `asyncio` name as critique sees it, with a stall in the first deadline.

    Only this module's attribute is replaced, so no other plugin's timers and no
    test helper's `asyncio` calls are affected. Everything except `timeout`
    delegates to the real module.
    """

    def __init__(self, stall: float) -> None:
        self.stall = stall
        self.armed = False

    def timeout(self, delay: float):
        inner = asyncio.timeout(delay)
        if self.armed:
            return inner
        self.armed = True
        return _StalledDeadline(inner, self.stall)

    def __getattr__(self, name):
        return getattr(asyncio, name)


async def test_r5_5_a_stall_between_arming_and_the_body_is_counted(tmp_path, caplog, monkeypatch):
    """The reported wait covers the whole span the deadline bounded (#1304).

    A stall between arming and the fork's first statement is injected here
    rather than hoped for. A stopwatch started inside the block — on either
    clock — reports a wait near zero for a fork that never ran: below the
    allowance and far below the 0.08 s of wall time the loop really lost. The
    0.05 s floor sits between the two readings, above the 0.02 s allowance and
    below the injected stall, so it separates them without needing the stall to
    land exactly.
    """
    async def hangs(i, channel):
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            raise

    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE, on_request=hangs)
    env = await _env_with(tmp_path, agent, config={"critique": {"fork_timeout": 0.02}})
    monkeypatch.setattr(critique_module, "asyncio", _StallingAsyncio(stall=0.08))
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kdp", PREDICTIVE_ONLY)
        [text] = _warnings(caplog, "exceeded critique.fork_timeout")
        m = re.search(r"after (\d+\.\d+)s", text)
        assert m, f"the warning must state its own elapsed time: {text!r}"
        # Pin the allowance in the same phrase, so the number being checked
        # cannot be the configured bound the line also prints.
        assert "critique.fork_timeout (0.02 s) after " in text, text
        assert float(m.group(1)) >= 0.05, (
            f"the printed wait is shorter than the stall the runner took: {text!r}"
        )
        assert re.fullmatch(r"\d+\.\d{3}", m.group(1)), (
            "the wait needs enough digits to separate a small allowance from a "
            f"stalled measurement: {text!r}"
        )
    finally:
        await _teardown(env)


async def test_r5_5_a_small_allowance_is_logged_with_digits_that_can_say_it(tmp_path, caplog):
    """`fork_timeout` below 0.05 s is real config, so the line must be able to
    print it: one decimal rounds any such allowance to "0.0s", which is the same
    text a stalled measurement produces and tells a reader nothing (#1304)."""
    async def hangs(i, channel):
        try:
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            raise

    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE, on_request=hangs)
    env = await _env_with(tmp_path, agent, config={"critique": {"fork_timeout": 0.002}})
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kdm", PREDICTIVE_ONLY)
        [text] = _warnings(caplog, "exceeded critique.fork_timeout")
        m = re.search(r"after (\d+\.\d+)s", text)
        assert m, f"the warning must state its own elapsed time: {text!r}"
        # Name the allowance in the same phrase, so the digits under test are
        # the elapsed field and not the configured bound the line also prints.
        assert "critique.fork_timeout (0.002 s) after " in text, text
        assert float(m.group(1)) >= 0.002, (
            f"the printed wait is below the allowance that expired: {text!r}"
        )
        assert re.fullmatch(r"\d+\.\d{3}", m.group(1)), (
            "one decimal cannot express a 0.002 s allowance at all: "
            f"{text!r}"
        )
    finally:
        await _teardown(env)


async def test_a_slow_but_finished_fork_warns_with_its_wall_time(tmp_path, caplog):
    """A fork that spends over half of `critique.fork_timeout` warns while it
    still succeeded: the early signal #1272 asks for, before the allowance runs
    out. The lens names travel with it, since one fork carries every selected
    lens (#1226)."""
    async def slow(i, channel):
        await asyncio.sleep(0.5)

    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE, on_request=slow)
    env = await _env_with(tmp_path, agent, config={"critique": {"fork_timeout": 0.8}}, with_held=True)
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kd8", PREDICTIVE_ONLY)
        [text] = _warnings(caplog, "slow")
        assert "predictive" in text and "trace kd8 " in text and env.channel.id in text, text
        assert re.search(r"\d+\.\d+s", text), text
        assert "fork_timeout" in text, "the allowance it is measured against travels with it"
    finally:
        await _teardown(env)


async def test_a_fork_that_keeps_its_pace_logs_no_slow_warning(tmp_path, caplog):
    """The warning is a threshold, not a log line per fork: an ordinary fork
    must not add noise to every exchange."""
    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE)
    env = await _env_with(tmp_path, agent, config={"critique": {"fork_timeout": 5.0}}, with_held=True)
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kd9", PREDICTIVE_ONLY)
        assert _warnings(caplog, "slow") == [], caplog.messages
    finally:
        await _teardown(env)


async def test_r5_5_a_provider_timeout_is_not_reported_as_the_deadline(tmp_path, caplog):
    """R5.5: a TimeoutError raised from inside the request is a fork failure
    (named by type), not the deadline, which stays at its default here."""
    agent = RoundsForkAgent({"critique"}, text=KEYED_PREDICTIVE, error=TimeoutError("provider"))
    env = await _env_with(tmp_path, agent)
    caplog.set_level(logging.WARNING)
    try:
        await _judged_turn(env, "kd3", PREDICTIVE_ONLY)
        [text] = _warnings(caplog, "skipped")
        assert "TimeoutError" in text
        assert "fork_timeout" not in text
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# The two config keys: startup, reload, fork time
# ---------------------------------------------------------------------------

BAD_CAP = [-1, True, "3", 2.5]
BAD_TIMEOUT = [0, -5, True, "fast", float("nan"), float("inf")]
BAD_VALUES = [("max_tool_calls", v) for v in BAD_CAP] + [("fork_timeout", v) for v in BAD_TIMEOUT]
ACCEPTED = {"max_tool_calls": "a whole number ≥ 0", "fork_timeout": "positive"}


@pytest.mark.parametrize("key,value", BAD_VALUES)
async def test_config_bad_value_fails_startup_naming_key_value_and_accepted(
    tmp_path, key, value,
):
    """Config: a bad ``critique.<key>`` fails on_init, naming the key, the
    rejected value and what is accepted."""
    with pytest.raises(ValueError) as raised:
        await _env(tmp_path, config={"critique": {key: value}})
    message = str(raised.value)
    assert f"critique.{key}" in message
    assert repr(value) in message
    assert ACCEPTED[key] in message


@pytest.mark.parametrize("key,value", [("max_tool_calls", -1), ("fork_timeout", 0)])
async def test_config_bad_value_at_reload_warns(tmp_path, caplog, key, value):
    """Config: a bad value arriving by reload warns naming key and value, and
    the daemon keeps running."""
    caplog.set_level(logging.WARNING)
    env = await _env(tmp_path)
    try:
        await env.plugin.on_config_reload({**env.plugin.config, "critique": {key: value}})
        assert _warnings(caplog, f"critique.{key}", repr(value)), caplog.text
    finally:
        await _teardown(env)


async def test_config_bad_cap_at_fork_time_falls_back_to_the_default(tmp_path, caplog):
    """Config: a bad cap that reached the plugin by reload never reaches
    ``fork``: the fork gets 3 and a warning names the key and value."""
    caplog.set_level(logging.WARNING)
    env = await _env(tmp_path)
    try:
        await env.plugin.on_config_reload(
            {**env.plugin.config, "critique": {"max_tool_calls": "many"}})
        caplog.clear()
        await _judged_turn(env, "kf1", PREDICTIVE_ONLY)
        [call] = env.fork_agent.calls
        assert call["max_tool_calls"] == 3
        assert _warnings(caplog, "critique.max_tool_calls", "'many'"), caplog.text
    finally:
        await _teardown(env)


async def test_config_bad_timeout_at_fork_time_falls_back_to_the_default(tmp_path, caplog):
    """Config: a bad deadline that reached the plugin by reload does not break
    the fork: it runs under the default and a warning names key and value."""
    caplog.set_level(logging.WARNING)
    env = await _env(tmp_path)
    try:
        await env.plugin.on_config_reload(
            {**env.plugin.config, "critique": {"fork_timeout": "soon"}})
        caplog.clear()
        await _judged_turn(env, "kf2", PREDICTIVE_ONLY)
        assert len(env.fork_agent.calls) == 1
        assert (await _read_outcomes(env.recorder, "kf2"))["critique"]["lenses"] == ["predictive"]
        assert _warnings(caplog, "critique.fork_timeout", "'soon'"), caplog.text
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# Brief / tail split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lens", ["predictive", "constrained", "adversarial",
                                  "proactivity", "retention"])
def test_tail_built_in_briefs_carry_no_reply_format(lens):
    """Tail: the built-in fork prompts are briefs only; the reply format and
    "Do not call tools" are critique's, appended once per fork."""
    brief = DEFAULT_FORK_PROMPTS[lens]
    assert "Do not call tools" not in brief
    assert "single JSON object" not in brief


async def test_tail_added_lens_brief_gets_no_tail_appended(tmp_path):
    """Tail: a plugin lens's brief is stored as written, with no tail added."""
    env = await _env(tmp_path, _lens("mine"))
    try:
        assert env.plugin._fork_prompts["mine"] == BRIEF.format("mine")
    finally:
        await _teardown(env)


async def test_tail_one_reply_format_per_fork_keyed_by_lens(tmp_path):
    """Tail: three lenses in one fork get exactly one reply-format tail, which
    asks for one key per lens."""
    env = await _env(tmp_path, _lens("mine"))
    try:
        await _judged_turn(env, "kt1", {"predictive": 0.4, "constrained": 0.3, "mine": 0.2})
        [call] = env.fork_agent.calls
        text = call["instruction"]
        assert text.lower().count("single json object") == 1
        tail = text[text.lower().index("single json object"):]
        assert all(f'"{lens}"' in tail for lens in ("predictive", "constrained", "mine"))
    finally:
        await _teardown(env)


@pytest.mark.parametrize("point,config,tools_allowed", [
    ("idle", None, True),                                      # cap 3: may search
    ("idle", {"critique": {"max_tool_calls": 0}}, False),      # cap 0 at idle
    ("before_send", None, False),                              # no tools before send
])
async def test_tail_forbids_tools_only_when_the_cap_is_zero(
    tmp_path, point, config, tools_allowed,
):
    """Tail: "Do not call tools" appears only when the fork's cap is 0; with a
    cap the tail names it; ``response_format`` is sent only when the cap is 0
    and then holds one property per lens in the fork."""
    env = await _env(tmp_path, config=config)
    try:
        await _trigger(env, point, TWO)
        [call] = env.fork_agent.calls
        text = call["instruction"].lower()
        assert ("do not call tools" in text) is (not tools_allowed)
        if tools_allowed:
            assert "at most 3" in text
            assert "response_format" not in (call["extra_body"] or {})
        else:
            schema = call["extra_body"]["response_format"]["json_schema"]["schema"]
            assert set(schema["properties"]) == {"predictive", "constrained"}
            assert set(schema["required"]) == {"predictive", "constrained"}
    finally:
        await _teardown(env)
