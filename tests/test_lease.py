"""Leases (#364, #367): the manager, its tables and the default policy.

Unit tests on ``corvidae.lease.LeaseManager`` / ``ProviderLeases`` with a
hand-moved monotonic clock and a list-collecting emitter, one test per
acceptance criterion in .claude/plans/364-sticky-leases.md: R2, R3m, R3p, S1,
S2, R8, R9m, R10, R11 and R12. The path through the real Agent, TaskQueue and
LLMPlugin (R1, R5, R6, R7, R13) is tests/test_lease_integration.py.

``corvidae.lease`` does not exist yet, so it is imported inside ``make`` and a
test fails on its own missing piece rather than at collection.

Ordering is asserted from recorded events (``Env.log``) and snapshots, never
from sleeps. ``spin`` yields to the loop a fixed number of times; no wall time
passes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

#: Attribution of the calls whose events are asserted (R12): the call's
#: stage, channel and trigger become tags next to provider and lease_key.
ATTR = {"stage": "turn", "channel_id": "irc:#a", "trigger": "user"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Ticker:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        """The current fake monotonic time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move time forward."""
        self.now += seconds


async def spin(times: int = 30) -> None:
    """Let every runnable task take its steps; no wall time passes."""
    for _ in range(times):
        await asyncio.sleep(0)


def make(heads: dict[str, int] | None = None) -> SimpleNamespace:
    """A LeaseManager over ``heads`` (default one provider "main", 1 head)
    with a Ticker clock and an emitter appending (name, value, tags) to
    ``events``. ``log`` collects the order calls entered, for assertions."""
    from corvidae.lease import LeaseManager

    heads = heads or {"main": 1}
    clock = Ticker()
    events: list[tuple[str, float, dict]] = []
    mgr = LeaseManager(
        heads,
        monotonic=clock.monotonic,
        emit=lambda name, value, tags: events.append((name, value, dict(tags))),
    )
    first = next(iter(heads))
    return SimpleNamespace(
        mgr=mgr, prov=mgr.provider(first), clock=clock, events=events, log=[]
    )


class Probe:
    """One chat call parked inside ``ProviderLeases.call`` once it has the head.

    ``entered`` is set when the call resumes with its lease; ``finish`` lets
    it leave (raising first when ``boom``); ``wait_s`` is the wait ``call()``
    reported; ``error`` is the exception that ended it, if any.
    """

    def __init__(
        self, env: SimpleNamespace, key: str, label: str | None = None, *,
        prov: Any = None, attribution: dict | None = None, boom: bool = False,
    ) -> None:
        self.key = key
        self.label = label or key
        self.entered = asyncio.Event()
        self.finish = asyncio.Event()
        self.wait_s: float | None = None
        self.error: BaseException | None = None
        self._env = env
        self._boom = boom
        self.task = asyncio.create_task(
            self._run(prov or env.prov, attribution if attribution is not None else {})
        )

    async def _run(self, prov: Any, attribution: dict) -> None:
        """Enter the lease, record it, park until told to leave."""
        try:
            async with prov.call(self.key, attribution) as wait:
                self.wait_s = wait
                self._env.log.append(self.label)
                self.entered.set()
                await self.finish.wait()
                if self._boom:
                    raise RuntimeError("model call blew up")
        except BaseException as exc:
            # Recorded for the assertions, then re-raised so the task ends
            # exactly as the caller's would.
            self.error = exc
            raise

    async def leave(self) -> None:
        """Let the call exit normally and wait for its task."""
        self.finish.set()
        await asyncio.gather(self.task, return_exceptions=True)


async def take(env: SimpleNamespace, key: str, label: str | None = None, **kw: Any) -> Probe:
    """Start a call for ``key`` and let it run as far as it can."""
    probe = Probe(env, key, label, **kw)
    await spin()
    return probe


def named(env: SimpleNamespace, name: str) -> list[tuple[str, float, dict]]:
    """The emitted events called ``name``, in order."""
    return [e for e in env.events if e[0] == name]


def tags_for(key: str, provider: str = "main") -> dict[str, str]:
    """The tags ATTR-attributed events for ``key`` must carry."""
    return {
        "provider": provider, "lease_key": key,
        "stage": "turn", "channel": "irc:#a", "trigger": "user",
    }


def has_tags(tags: dict, expected: dict) -> bool:
    """True when every expected tag is present with that value."""
    return all(tags.get(k) == v for k, v in expected.items())


def holder_keys(env: SimpleNamespace, provider: str = "main") -> list[str]:
    """Keys holding a lease on ``provider`` now, sorted."""
    return sorted(h.key for h in env.mgr.snapshot(provider).holders)


def waiter_keys(env: SimpleNamespace, provider: str = "main") -> list[str]:
    """Keys waiting on ``provider`` now, in FIFO order."""
    return [w.key for w in env.mgr.snapshot(provider).waiters]


# ---------------------------------------------------------------------------
# R2, R3m, R3p, S1, S2: granting and the default policy
# ---------------------------------------------------------------------------


async def test_r2_two_heads_serve_two_keys_without_waiting() -> None:
    """R2: on heads 2 two keys' calls are in flight together, neither waits,
    and a key arriving while the first holder is idle takes the free head
    rather than displacing it."""
    env = make({"main": 2})

    a = await take(env, "A")
    b = await take(env, "B")

    # Both calls in flight together, neither waited.
    assert a.entered.is_set() and b.entered.is_set()
    assert (a.wait_s, b.wait_s) == (0.0, 0.0)
    await b.leave()
    await a.leave()
    await spin()

    # A fresh table with A idle and one head free.
    env = make({"main": 2})
    a = await take(env, "A")
    await a.leave()
    await spin()
    b = await take(env, "B")

    # B took the free head; idle A was not displaced.
    assert b.entered.is_set() and b.wait_s == 0.0
    assert holder_keys(env) == ["A", "B"]
    assert named(env, "lease.yield") == []
    await b.leave()


async def test_r3m_holders_second_call_does_not_wait_and_second_key_waits() -> None:
    """R3m: on heads 1 a holder's second call does not wait, and a second
    key's call waits until the head is free."""
    env = make()

    a1 = await take(env, "A", "A1")
    await a1.leave()
    a2 = await take(env, "A", "A2")
    b = await take(env, "B")

    # A's second call went straight through; B stays behind A's call.
    assert a2.entered.is_set() and a2.wait_s == 0.0
    assert env.log == ["A1", "A2"]
    assert not b.entered.is_set()
    # The head frees up when A's call ends: B takes it.
    await a2.leave()
    await spin()
    assert env.log == ["A1", "A2", "B"]
    await b.leave()


async def test_r3p_waiting_keys_are_granted_in_arrival_order() -> None:
    """R3p: three keys on heads 1 are granted A, B, C, as they arrived."""
    env = make()

    a = await take(env, "A")
    b = await take(env, "B")
    c = await take(env, "C")
    assert waiter_keys(env) == ["B", "C"]
    # Check the order after each call ends, so a wrong order fails on the
    # assertion instead of hanging on a call that is never granted.
    expected = [["A", "B"], ["A", "B", "C"]]
    for probe, log in zip((a, b), expected):
        await probe.leave()
        await spin()
        assert env.log == log
    await c.leave()

    assert env.log == ["A", "B", "C"]
    assert [e[2]["lease_key"] for e in named(env, "lease.grant")] == ["A", "B", "C"]


async def test_s1_holder_keeps_its_head_between_calls_while_nobody_waits() -> None:
    """S1: heads 1: A's call ends with no waiter: A still holds, idle, and
    its next call does not wait and is not a second grant."""
    env = make()
    a1 = await take(env, "A", "A1")

    await a1.leave()
    await spin()
    view = env.mgr.snapshot("main")
    assert [(h.key, h.in_call) for h in view.holders] == [("A", False)]
    a2 = await take(env, "A", "A2")

    assert a2.entered.is_set() and a2.wait_s == 0.0
    assert len(named(env, "lease.grant")) == 1
    await a2.leave()


async def test_s2_least_recently_used_idle_holder_yields_to_a_new_waiter() -> None:
    """S2: heads 2, A and B idle and A's last call ended before B's (B was
    granted first, so this is not oldest-grant order): C arrives, A yields
    and B keeps its head."""
    env = make({"main": 2})
    b = await take(env, "B")
    # A is granted strictly later than B, so oldest-grant would pick B.
    env.clock.advance(1.0)
    a = await take(env, "A")
    env.clock.advance(1.0)
    await a.leave()
    env.clock.advance(1.0)
    await b.leave()
    assert holder_keys(env) == ["A", "B"]

    c = await take(env, "C")

    assert c.entered.is_set()
    assert holder_keys(env) == ["B", "C"]
    yields = named(env, "lease.yield")
    assert [(e[2]["lease_key"], e[2]["reason"]) for e in yields] == [("A", "contention")]
    await c.leave()


async def test_s2_holder_mid_call_yields_when_its_call_ends() -> None:
    """S2: a holder with a call in flight keeps the head while the call runs
    and gives it to the waiter when the call ends."""
    env = make()
    a = await take(env, "A")
    b = await take(env, "B")

    # Mid-call the head is still A's.
    assert not b.entered.is_set()
    assert [(h.key, h.in_call) for h in env.mgr.snapshot("main").holders] == [("A", True)]

    await a.leave()
    await spin()

    assert b.entered.is_set()
    assert holder_keys(env) == ["B"]
    await b.leave()


# ---------------------------------------------------------------------------
# R8: stop and resume
# ---------------------------------------------------------------------------


async def test_r8_stopped_key_yields_at_its_call_boundary_and_resumes_later() -> None:
    """R8: stop never cuts a call; when it ends the stopped key gives up its
    head even with nobody waiting, another key runs, and the stopped key's
    next call waits (not dropped) until resume."""
    env = make()
    a1 = await take(env, "A", "A1")

    env.mgr.stop("main", "A")
    await spin()
    # Mid-call: nothing was taken away.
    assert not a1.task.done() and a1.error is None
    await a1.leave()
    await spin()
    assert holder_keys(env) == []

    a2 = await take(env, "A", "A2")
    b = await take(env, "B")
    # The head was free, yet the stopped key's call waits and B's does not.
    assert b.entered.is_set() and not a2.entered.is_set()
    await b.leave()
    await spin()
    assert not a2.entered.is_set()

    env.mgr.resume("main", "A")
    await spin()

    assert env.log == ["A1", "B", "A2"]
    await a2.leave()


async def test_r8_stop_on_one_provider_keeps_the_other_provider() -> None:
    """R8: stopped on p1, the key's lease on p2 is untouched: p2 still shows
    it as a holder, emitted nothing, and its next p2 call goes straight
    through, while its next p1 call waits."""
    env = make({"p1": 1, "p2": 1})
    p1, p2 = env.mgr.provider("p1"), env.mgr.provider("p2")
    a_p2 = await take(env, "A", "A-p2", prov=p2)
    await a_p2.leave()
    a_p1 = await take(env, "A", "A-p1", prov=p1)

    env.mgr.stop("p1", "A")
    await a_p1.leave()
    await spin()

    assert holder_keys(env, "p1") == []
    assert holder_keys(env, "p2") == ["A"]
    assert not [
        e for e in named(env, "lease.yield") + named(env, "lease.release")
        if e[2].get("provider") == "p2"
    ]
    again = await take(env, "A", "A-p2-again", prov=p2)
    blocked = await take(env, "A", "A-p1-again", prov=p1)

    assert again.entered.is_set() and again.wait_s == 0.0
    assert not blocked.entered.is_set()
    await again.leave()
    env.mgr.resume("p1", "A")
    await spin()
    await blocked.leave()


# ---------------------------------------------------------------------------
# R9m: voluntary yield
# ---------------------------------------------------------------------------


async def test_r9m_yield_lets_the_waiter_run_first_and_requeues_the_yielder_last() -> None:
    """R9m: heads 1, A in a call with B and C waiting and A's own next call
    parked between them (B, A2, C): after yield_key(A) A2 is at the end of the
    line, and the grants go A1, B, C, A2 with no resume needed."""
    env = make()
    a1 = await take(env, "A", "A1")
    b = await take(env, "B")
    a2 = await take(env, "A", "A2")
    c = await take(env, "C")
    assert waiter_keys(env) == ["B", "A", "C"]

    env.mgr.yield_key("A")
    assert waiter_keys(env) == ["B", "C", "A"]
    await a1.leave()
    await spin()
    # The waiter ran before the holder's next call.
    assert env.log == ["A1", "B"]
    assert not a2.entered.is_set()
    await b.leave()
    await spin()
    await c.leave()
    await spin()

    assert env.log == ["A1", "B", "C", "A2"]
    await a2.leave()


# ---------------------------------------------------------------------------
# R10: snapshot
# ---------------------------------------------------------------------------


async def test_r10_snapshot_in_grant_wait_yield_and_stop_states() -> None:
    """R10: the snapshot shows declared heads, holders with their age and
    in-call flag, and waiters in FIFO order with their ages, in the grant,
    wait, yield and stop states."""
    env = make()
    view = env.mgr.snapshot("main")
    assert (view.name, view.heads, view.holders, view.waiters) == ("main", 1, (), ())

    # Grant: A holds and is in a call.
    a = await take(env, "A")
    env.clock.advance(2.0)
    view = env.mgr.snapshot("main")
    assert [(h.key, h.held_s, h.in_call) for h in view.holders] == [("A", 2.0, True)]

    # Wait: B then C queue behind it, in order, with their ages.
    b = await take(env, "B")
    env.clock.advance(3.0)
    c = await take(env, "C")
    env.clock.advance(1.0)
    view = env.mgr.snapshot("main")
    assert [(w.key, w.waited_s) for w in view.waiters] == [("B", 4.0), ("C", 1.0)]
    assert [h.held_s for h in view.holders] == [6.0]

    # Yield: A yields mid-call; when the call ends B holds, C still waits.
    env.mgr.yield_key("A")
    await a.leave()
    await spin()
    view = env.mgr.snapshot("main")
    assert [(h.key, h.in_call) for h in view.holders] == [("B", True)]
    assert [w.key for w in view.waiters] == ["C"]

    # Stop: B stopped mid-call is still listed; when the call ends C holds
    # and B's next call waits.
    env.mgr.stop("main", "B")
    assert holder_keys(env) == ["B"]
    await b.leave()
    await spin()
    b2 = await take(env, "B", "B2")
    view = env.mgr.snapshot("main")
    assert [h.key for h in view.holders] == ["C"]
    assert [w.key for w in view.waiters] == ["B"]

    await c.leave()
    env.mgr.resume("main", "B")
    await spin()
    await b2.leave()


# ---------------------------------------------------------------------------
# R11: release on raise, cancel, shutdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["raise", "cancel"])
async def test_r11_holder_that_raises_or_is_cancelled_frees_the_head(how: str) -> None:
    """R11: a holder whose call raises, or is cancelled mid-call, releases
    the lease: the waiter gets the head."""
    env = make()
    a = await take(env, "A", boom=(how == "raise"))
    b = await take(env, "B")
    assert not b.entered.is_set()

    if how == "raise":
        a.finish.set()
    else:
        a.task.cancel()
    await asyncio.gather(a.task, return_exceptions=True)
    await spin()

    assert b.entered.is_set()
    assert isinstance(a.error, RuntimeError if how == "raise" else asyncio.CancelledError)
    await b.leave()


async def test_r11_call_cancelled_between_grant_and_resume_gives_the_grant_back() -> None:
    """R11: a call cancelled after its grant but before it resumes gives the
    head back instead of leaking it: the next waiter is granted."""
    env = make()
    a_cm = env.prov.call("A", {})
    await a_cm.__aenter__()
    b = Probe(env, "B")
    c = Probe(env, "C")
    await spin()

    # A leaves by hand, so B is granted but has not had a turn to resume.
    await a_cm.__aexit__(None, None, None)
    assert not b.entered.is_set()
    b.task.cancel()
    await asyncio.gather(b.task, return_exceptions=True)
    await spin()

    assert c.entered.is_set()
    await c.leave()


async def test_r11_shutdown_releases_leases_and_fails_waiters() -> None:
    """R11: close() releases every lease and fails each waiting call with an
    error naming the provider, so nothing hangs."""
    env = make()
    a = await take(env, "A")
    b = await take(env, "B")

    env.mgr.close()
    await asyncio.gather(b.task, return_exceptions=True)

    assert isinstance(b.error, RuntimeError)
    assert str(b.error) == "LLM provider main is shutting down; the call was not sent"
    assert env.mgr.snapshot("main").holders == ()
    # The call that was in flight ends cleanly against the released lease.
    await a.leave()
    assert a.error is None


# ---------------------------------------------------------------------------
# R12: events and their attribution
# ---------------------------------------------------------------------------


async def test_r12_grant_and_wait_events_carry_value_and_tags() -> None:
    """R12: a grant emits lease.grant (1.0); a call that waited also emits
    lease.wait_ms with the wait in ms and one that did not emits none; all
    carry provider, lease_key, stage, channel and trigger of the call."""
    env = make()
    a = await take(env, "A", attribution=dict(ATTR))
    b = await take(env, "B", attribution=dict(ATTR))
    env.clock.advance(2.5)
    await a.leave()
    await spin()

    grants = named(env, "lease.grant")
    assert [(e[1], e[2]["lease_key"]) for e in grants] == [(1.0, "A"), (1.0, "B")]
    assert all(has_tags(e[2], tags_for(e[2]["lease_key"])) for e in grants)
    waits = named(env, "lease.wait_ms")
    assert [(e[1], e[2]["lease_key"]) for e in waits] == [(2500.0, "B")]
    assert has_tags(waits[0][2], tags_for("B"))
    assert b.wait_s == 2.5
    await b.leave()


@pytest.mark.parametrize("reason", ["contention", "cooperative", "stop"])
async def test_r12_yield_event_carries_reason_and_last_call_tags(reason: str) -> None:
    """R12: a holder giving up its head emits lease.yield (1.0) tagged
    reason contention / cooperative / stop plus the tags of the last call
    entered under the lease, and no lease.release."""
    env = make()
    a = await take(env, "A", attribution=dict(ATTR))
    b = await take(env, "B")
    if reason == "cooperative":
        env.mgr.yield_key("A")
    elif reason == "stop":
        env.mgr.stop("main", "A")

    await a.leave()
    await spin()

    yields = named(env, "lease.yield")
    assert len(yields) == 1 and yields[0][1] == 1.0
    assert yields[0][2]["reason"] == reason
    assert has_tags(yields[0][2], tags_for("A"))
    assert named(env, "lease.release") == []
    await b.leave()


async def test_r12_release_event_is_emitted_only_at_shutdown() -> None:
    """R12: lease.release (value held ms, reason shutdown, the lease's last
    call's tags) comes from close() and from nothing else: a contention yield
    emits lease.yield and no release."""
    env = make()
    a = await take(env, "A", attribution=dict(ATTR))
    b = await take(env, "B", attribution=dict(ATTR))
    await a.leave()
    await spin()
    assert named(env, "lease.release") == []
    await b.leave()
    env.clock.advance(4.0)

    env.mgr.close()
    await spin()

    releases = named(env, "lease.release")
    assert len(releases) == 1
    assert releases[0][1] == 4000.0
    assert releases[0][2]["reason"] == "shutdown"
    assert releases[0][2]["lease_key"] == "B"
    assert has_tags(releases[0][2], tags_for("B"))


# ---------------------------------------------------------------------------
# Paths the design adds beyond the numbered criteria
# ---------------------------------------------------------------------------


async def test_r11_call_on_a_closed_provider_raises_and_takes_no_lease() -> None:
    """R11 (call-path step 0): after close() a new call raises the shutdown
    error naming the provider before touching the table, so no lease is taken
    that shutdown would never release."""
    env = make()
    env.mgr.close()

    with pytest.raises(RuntimeError) as raised:
        async with env.prov.call("A", {}):
            pytest.fail("a call on a closed provider must not run")

    assert str(raised.value) == "LLM provider main is shutting down; the call was not sent"
    assert env.mgr.snapshot("main").holders == ()
    assert named(env, "lease.grant") == []


async def test_r9m_yield_with_no_eligible_waiter_keeps_the_lease_and_the_next_call_passes() -> None:
    """R9m/R6: a yield with nobody waiting changes nothing: the holder keeps
    its head and a second call of the key goes straight through, instead of
    being ordered behind the key's own call in flight."""
    env = make()
    a1 = await take(env, "A", "A1")

    env.mgr.yield_key("A")
    a2 = await take(env, "A", "A2")

    assert a2.entered.is_set() and a2.wait_s == 0.0
    assert holder_keys(env) == ["A"]
    assert named(env, "lease.yield") == []
    await a1.leave()
    await a2.leave()


@pytest.mark.parametrize("method", ["stop", "resume"])
def test_r8_stop_or_resume_on_an_unknown_provider_names_it_and_the_bounded_ones(method: str) -> None:
    """R8: stop/resume on a provider with no heads is a ValueError whose
    message names the rejected provider and the bounded providers."""
    env = make({"main": 1, "other": 2})

    with pytest.raises(ValueError) as raised:
        getattr(env.mgr, method)("nope", "A")

    message = str(raised.value)
    assert "nope" in message and "main" in message and "other" in message


async def test_r13_chat_reports_the_error_once_with_a_wait_when_the_lease_raises() -> None:
    """R13: LLMClient.chat on a provider whose manager has shut down: the
    response observer fires exactly once with the error and a non-None
    lease_wait_ms, then the error propagates, and no request was sent."""
    from test_llm_embed import _make_mock_response, _make_mock_session

    from corvidae.attribution import reset_attribution, set_attribution
    from corvidae.llm import LLMClient

    env = make()
    env.mgr.close()
    client = LLMClient(base_url="http://localhost:8080", model="m")
    client.session = _make_mock_session(_make_mock_response())
    client.leases = env.prov
    responses: list[dict] = []

    class Observer:
        """Records each callback."""

        async def request(self, **kwargs: Any) -> None:
            """Not under test."""

        async def response(self, **kwargs: Any) -> None:
            """Keep the response callback."""
            responses.append(kwargs)

    client.observer = Observer()
    token = set_attribution(lease_key="A")
    try:
        with pytest.raises(RuntimeError, match="LLM provider main is shutting down"):
            await client.chat([{"role": "user", "content": "hi"}])
    finally:
        reset_attribution(token)

    assert len(responses) == 1
    assert responses[0]["error"] is not None
    assert responses[0]["lease_wait_ms"] is not None
    client.session.post.assert_not_called()


async def test_r13_embed_reports_no_lease_wait() -> None:
    """R13: embed() never takes a lease, so its observer response carries
    lease_wait_ms None (NULL in usage_log)."""
    from unittest.mock import AsyncMock, MagicMock

    from test_llm_embed import _make_mock_response, _make_mock_session

    from corvidae.llm import LLMClient

    client = LLMClient(base_url="http://localhost:8080", model="m")
    client.session = _make_mock_session(_make_mock_response())
    observer = MagicMock()
    observer.request = AsyncMock()
    observer.response = AsyncMock()
    client.observer = observer

    await client.embed(["alpha"], kind="document")

    observer.response.assert_called_once()
    assert observer.response.call_args.kwargs["lease_wait_ms"] is None
