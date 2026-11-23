"""RED tests for #1149 (with #1169 part 2): how appraisal and critique are wired
to ``corvidae_cognition.presend.PreSendHold`` (docs/1149-presend-hold-design.md).

The state machine itself (R3 to R6 interleavings through a real Agent) is
proved in test_presend_real_path.py. These tests pin the wiring: what the
hookimpls of appraisal and critique return, hand to the held store and report
on ``on_trace_record``. Each test's docstring names its requirement.

Harnesses reused:

- appraisal: ``PeerSpy``, ``_send`` and ``_idle_exchange`` from
  test_appraisal_hooks.py, and ``build_env`` (with its TraceRecorder) from
  test_appraisal_stage2.py;
- critique: ``build_critique``, ``_gate``, ``_begin``, ``_drain`` and
  ``PresentAppraisal`` from test_critique.py and test_critique_hooks.py.

The held store is a ``HeldSpy``, a ``HeldContext`` subclass that records
``hold`` and ``hold_and_wake`` instead of firing a stub. Reason: appraisal
reaches the store with ``get_dependency(pm, "held", HeldContext)``, so the
stand-in must be a ``HeldContext``, and these tests assert on the arguments
of the call, not on the stub turn the real store would queue (the real store
has its own tests in tests/test_held.py).

The window state is set the way the agent leaves it: the agent appends the
final's assistant row before ``on_agent_response`` fires, tagged
``MessageType.UNSENT`` when a gate vetoed it in time (``final_was_unsent``).
"""

import logging

import pytest

from corvidae.context import MessageType, unsent_marker_text
from corvidae.held import STUB_ANNOTATION, HeldContext
from corvidae.hooks import Refusal, hookimpl

from helpers import add_kernel_registry
from tests.cognition.test_appraisal_hooks import (
    IDLE_REPLY,
    SCORED_VECTOR,
    TRACE,
    PeerSpy,
    _idle_exchange,
    _send,
)
from tests.cognition.test_appraisal_stage2 import (
    _originate,
    _teardown,
    build_env,
)
from tests.cognition.test_critique import FixedRng, build_critique
from tests.cognition.test_critique import _teardown as _teardown_critique
from tests.cognition.test_critique_hooks import (
    DRAFT,
    OBJECTION_TEXT,
    VERDICT,
    PresentAppraisal,
    _begin,
    _drain,
    _gate,
)

# What the design names for the held batch's source.
SOURCE = "pre-send check"
OBJECTION = "objection text"


class HeldSpy(HeldContext):
    """A held store that records what it is handed and queues nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.wakes: list[dict] = []
        self.holds: list[dict] = []

    async def hold_and_wake(
        self, channel, origin, source, entries, stub_text=None,
    ) -> None:
        # Record the call whatever the call style (positional or keyword).
        self.wakes.append({
            "channel": channel, "origin": origin, "source": source,
            "entries": list(entries), "stub_text": stub_text,
        })

    async def hold(self, channel, source, entries) -> None:
        # Record a plain hold, so "no hold from critique" can be asserted.
        self.holds.append({"channel": channel, "source": source, "entries": list(entries)})


class ObjectingPeer(PeerSpy):
    """A peer whose ``on_appraisal_before_send`` returns ``objections``."""

    def __init__(self, objections, offers=None) -> None:
        super().__init__(offers if offers is not None else {"before_send": [("alpha", "Did alpha?")]})
        self.objections = objections

    @hookimpl
    async def on_appraisal_before_send(self, channel, trace_id, text, annotations, lenses):
        # Record as PeerSpy does, then answer with the configured objections.
        await super().on_appraisal_before_send(channel, trace_id, text, annotations, lenses)
        return self.objections


async def _presend_env(tmp_path, peer=None, chat_text=IDLE_REPLY):
    """An appraisal env with a HeldSpy as ``held``, ``peer`` beside it, and an
    exchange whose originating row is known. ``IDLE_REPLY`` answers both the
    before-send fork (it reads ``lenses``) and the stage-2 fork."""
    env = await build_env(tmp_path, chat_text=chat_text)
    # outcome_log looks the registry up for every record it writes; without
    # one it logs a WARNING that would pollute the warning-count tests.
    add_kernel_registry(env.pm)
    held = HeldSpy()
    env.pm.register(held, name="held")
    if peer is not None:
        env.pm.register(peer, name="peer")
    await _originate(env, TRACE)
    return env, held


def _append_final(channel, text: str, *, unsent: bool) -> None:
    """Append the final's assistant row as the agent does before
    ``on_agent_response``: UNSENT when the gate vetoed it in time."""
    channel.conversation.append(
        {"role": "assistant", "content": text},
        MessageType.UNSENT if unsent else MessageType.MESSAGE,
    )
    # An unsent final is followed by the kernel's not-sent marker row, as
    # Agent._append_unsent_marker leaves it.
    if unsent:
        channel.conversation.append(
            {"role": "system", "content": unsent_marker_text("final", text)},
            MessageType.CONTEXT,
        )


async def _final(env, draft: str, *, unsent: bool, annotations=None, trace_id=TRACE) -> None:
    """Fire appraisal's ``on_agent_response`` for a final, after the window
    holds its row."""
    _append_final(env.channel, draft, unsent=unsent)
    await env.appraisal.on_agent_response(
        channel=env.channel, request_text="req", response_text=draft,
        trace_id=trace_id, annotations=annotations or {}, logprobs=None,
    )


async def _refuse_then_final(env, draft: str, *, unsent: bool, annotations=None) -> None:
    """The gate refuses ``draft``, then the final's outcome lands."""
    refusal = await _send(env, draft, annotations=annotations or {})
    assert isinstance(refusal, Refusal)
    await _final(env, draft, unsent=unsent, annotations=annotations)


def _held_records(env) -> list[dict]:
    """The ``held`` records appraisal reported on the trace."""
    return [r for r in env.recorder.by_source("appraisal") if "held" in r["record"]]


# ---------------------------------------------------------------------------
# R1: appraisal refuses on peers' objections
# ---------------------------------------------------------------------------


async def test_r1_a_peers_objection_makes_appraisal_refuse_with_it_as_the_reason(tmp_path):
    """R1: a peer returning ["objection text"] from on_appraisal_before_send
    makes appraisal's should_send_response return a Refusal whose reason
    contains the objection."""
    env, _held = await _presend_env(tmp_path, ObjectingPeer([OBJECTION]))
    try:
        result = await _send(env, "a real draft", annotations={"origin": "user"})
        assert isinstance(result, Refusal)
        assert OBJECTION in result.reason
    finally:
        await _teardown(env)


async def test_r1_a_peer_returning_none_leaves_the_answer_none(tmp_path):
    """R1: a draft drawing no objection is sent: a peer returning None leaves
    appraisal's answer None."""
    env, _held = await _presend_env(tmp_path, ObjectingPeer(None))
    try:
        assert await _send(env, "a real draft", annotations={"origin": "user"}) is None
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R4 / R7: an in-time refusal is held and woken, and recorded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("annotations, origin", [
    ({}, "user"),                      # no annotation origin, no resolved one: "user"
    ({"origin": "heartbeat"}, "heartbeat"),   # the annotation origin wins
], ids=["user-default", "annotation-origin"])
async def test_r4_in_time_refusal_holds_and_wakes_with_the_exchanges_origin(
    tmp_path, annotations, origin,
):
    """R4: after an in-time refusal, appraisal's on_agent_response for that
    trace (final unsent) calls held.hold_and_wake with source "pre-send
    check", the objections as entries, and the hold origin: "user" when the
    annotations carry none and resolve_trace_origin gives none, else the
    annotation origin."""
    env, held = await _presend_env(tmp_path, ObjectingPeer([OBJECTION]))
    try:
        await _refuse_then_final(env, "a real draft", unsent=True, annotations=annotations)
        [wake] = held.wakes
        assert wake["source"] == SOURCE
        assert wake["entries"] == [OBJECTION]
        assert wake["origin"] == origin
        assert wake["channel"] is env.channel
    finally:
        await _teardown(env)


async def test_r7_in_time_refusal_is_reported_as_a_held_record(tmp_path):
    """R7: the in-time hold fires on_trace_record(source="appraisal",
    record={"held": {"objections": [...]}}) for the held trace."""
    env, _held = await _presend_env(tmp_path, ObjectingPeer([OBJECTION]))
    try:
        await _refuse_then_final(env, "a real draft", unsent=True)
        [record] = _held_records(env)
        assert record["trace_id"] == TRACE
        assert record["channel_id"] == env.channel.id
        assert record["record"] == {"held": {"objections": [OBJECTION]}}
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R5: a late refusal holds nothing
# ---------------------------------------------------------------------------


async def test_r5_a_late_refusal_holds_nothing_and_reports_no_held_record(tmp_path):
    """R5: the same flow with the final SENT (the refusal came late) calls no
    hold_and_wake and fires no held record; the kernel's late path carries the
    objections."""
    env, held = await _presend_env(tmp_path, ObjectingPeer([OBJECTION]))
    try:
        await _refuse_then_final(env, "a real draft", unsent=False)
        assert held.wakes == []
        assert _held_records(env) == []
    finally:
        await _teardown(env)


async def test_r5_a_pending_refusal_is_not_consumed_by_a_different_draft(tmp_path):
    """R5: a pending refusal whose draft text differs from response_text is
    not consumed: an unsent final of the same trace with other text holds
    nothing."""
    env, held = await _presend_env(tmp_path, ObjectingPeer([OBJECTION]))
    try:
        refusal = await _send(env, "the refused draft", annotations={})
        assert isinstance(refusal, Refusal)
        # A different final of the same trace, unsent: not the refused draft.
        await _final(env, "some other text", unsent=True)
        assert held.wakes == []
        assert _held_records(env) == []
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R6: the owners forward the kernel's hooks to the hold
# ---------------------------------------------------------------------------


async def test_r6_appraisal_forwards_the_turn_and_the_offer_and_a_declined_revision_clears(
    tmp_path,
):
    """R6: appraisal forwards before_agent_turn and on_held_admitted to its
    hold, and calls on_final before its decline early return. A held draft,
    the revising turn announced to appraisal's own hooks, the batch offered,
    then a revision that DECLINES (NO_REPLY, unsent): the span still ends at
    that final, so the channel's next final is checked again and refused. No
    forwarding of on_held_admitted, or on_final after the decline return,
    leaves the channel suppressed and the next final unchecked."""
    env, held = await _presend_env(tmp_path, ObjectingPeer([OBJECTION]))
    try:
        await _refuse_then_final(env, "a real draft", unsent=True)
        [wake] = held.wakes
        # The revising turn starts: the stub turn, with the hold's origin.
        await env.appraisal.before_agent_turn(
            channel=env.channel, trace_id="rev",
            annotations={"origin": wake["origin"], STUB_ANNOTATION: True})
        await env.appraisal.on_held_admitted(
            channel=env.channel, source=SOURCE, trace_id=None,
            admitted_keys=["k1"], duplicate_keys=[])
        # While the hold is outstanding appraisal checks nothing: the revision's
        # draft is answered None without a fork, though the peer still objects.
        asked = len(env.pm.get_plugin("peer").before_send)
        assert await _send(
            env, "a revised draft", annotations={"origin": wake["origin"]},
            trace_id="rev") is None
        assert len(env.pm.get_plugin("peer").before_send) == asked
        # The revision declines: NO_REPLY, withheld, as the agent leaves it.
        await _final(env, "NO_REPLY", unsent=True, annotations={"origin": "user"}, trace_id="rev")
        # The span ended: the next final on the channel is checked and refused.
        await _originate(env, "next")
        result = await _send(
            env, "another draft", annotations={"origin": "user"}, trace_id="next")
        assert isinstance(result, Refusal)
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R8: no stage 2 for a held trace
# ---------------------------------------------------------------------------


async def test_r8_a_held_trace_gets_no_stage_two_and_a_later_unheld_trace_does(tmp_path):
    """R8: a held trace enqueues no stage-2 task, so no on_appraisal_idle
    fires for it; a later, unheld trace still gets stage 2."""
    peer = ObjectingPeer([OBJECTION], offers={
        "before_send": [("alpha", "Did alpha?")], "idle": [("alpha", "Did alpha?")]})
    env, _held = await _presend_env(tmp_path, peer)
    try:
        await _refuse_then_final(env, "a real draft", unsent=True)
        # Nothing was queued for the held trace.
        assert env.task.task_queue.queue.empty()
        assert peer.idle == []
        # The next exchange (a different trace, nothing refused) is judged.
        peer.objections = None
        await _idle_exchange(env, "ki")
        assert [call["trace_id"] for call in peer.idle] == ["ki"]
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R9: critique's fallback, with appraisal absent
# ---------------------------------------------------------------------------


async def _critique_fallback_env(tmp_path):
    """Critique with no appraisal, fallback_rate 1.0, an objecting lens, and a
    HeldSpy patched over the real held store (see the module docstring)."""
    env = await build_critique(
        tmp_path, None, chat_text=VERDICT, with_held=True,
        config={"critique": {"fallback_rate": 1.0}})
    env.plugin._rng = FixedRng(value=0.5, choice_value="predictive")
    spy = HeldSpy()
    env.held.hold = spy.hold
    env.held.hold_and_wake = spy.hold_and_wake
    return env, spy


async def _critique_hold_a_draft(env, trace_id: str):
    """Gate a draft, land its unsent final, as the agent does for an in-time
    refusal. Returns the gate's answer."""
    await _begin(env, trace_id)
    answer = await _gate(env, trace_id=trace_id)
    _append_final(env.channel, DRAFT, unsent=True)
    await env.plugin.on_agent_response(
        channel=env.channel, request_text="req", response_text=DRAFT,
        trace_id=trace_id, annotations={}, logprobs=None)
    return answer


async def test_r9_without_appraisal_critique_refuses_an_objectionable_draft(tmp_path):
    """R9: with appraisal absent and critique.fallback_rate 1.0, critique's
    should_send_response on an objectionable draft returns a Refusal carrying
    the objection."""
    env, _spy = await _critique_fallback_env(tmp_path)
    try:
        await _begin(env, "kr9")
        result = await _gate(env, trace_id="kr9")
        assert isinstance(result, Refusal)
        assert OBJECTION_TEXT in result.reason
    finally:
        await _teardown_critique(env)


async def test_r9_critiques_unsent_final_holds_and_wakes_once_and_runs_no_idle_lens(tmp_path):
    """R9: critique's on_agent_response on the unsent final holds and wakes
    once (source "pre-send check", the objection, origin "user"), and runs no
    idle fallback lens for that trace (R8: no second fork, nothing queued)."""
    env, spy = await _critique_fallback_env(tmp_path)
    try:
        await _critique_hold_a_draft(env, "kr9")
        [wake] = spy.wakes
        assert wake["source"] == SOURCE
        assert len(wake["entries"]) == 1 and OBJECTION_TEXT in wake["entries"][0]
        assert wake["origin"] == "user"
        # Only the before-send fork ran; the idle fallback (rate 1.0 would
        # draw it every time) was skipped for the held trace.
        await _drain(env)
        assert len(env.fork_agent.calls) == 1
    finally:
        await _teardown_critique(env)


async def test_r9_critique_does_not_check_while_its_hold_is_outstanding(tmp_path):
    """R9: suppression applies while that hold is outstanding: the next final
    on the channel is not sampled, so no fork runs and the answer is None."""
    env, _spy = await _critique_fallback_env(tmp_path)
    try:
        await _critique_hold_a_draft(env, "kr9")
        forks = len(env.fork_agent.calls)
        await _begin(env, "kr9b")
        assert await _gate(env, trace_id="kr9b") is None
        assert len(env.fork_agent.calls) == forks
    finally:
        await _teardown_critique(env)


async def test_r6_critique_forwards_the_turn_and_the_offer_and_a_declined_revision_clears(
    tmp_path,
):
    """R6: with appraisal absent, critique forwards before_agent_turn and
    on_held_admitted to its hold, and calls on_final before its eligibility
    and decline early returns. A held draft, the revising turn announced,
    the batch offered, then a revision that DECLINES (NO_REPLY, unsent): the
    span ends at that final, so the channel's next final is sampled and
    refused again. Without the forwarding, or with on_final after the
    decline return, the channel stays suppressed and nothing is checked."""
    env, spy = await _critique_fallback_env(tmp_path)
    try:
        await _critique_hold_a_draft(env, "kr6")
        [wake] = spy.wakes
        await env.plugin.before_agent_turn(
            channel=env.channel, trace_id="rev",
            annotations={"origin": wake["origin"], STUB_ANNOTATION: True})
        await env.plugin.on_held_admitted(
            channel=env.channel, source=SOURCE, trace_id=None,
            admitted_keys=["k1"], duplicate_keys=[])
        # The revision declines, as the agent leaves it: NO_REPLY, unsent.
        _append_final(env.channel, "NO_REPLY", unsent=True)
        await env.plugin.on_agent_response(
            channel=env.channel, request_text="req", response_text="NO_REPLY",
            trace_id="rev", annotations={"origin": "user"}, logprobs=None)
        # The span ended: the next final is checked, and refused.
        await _begin(env, "kr6b")
        assert isinstance(await _gate(env, trace_id="kr6b"), Refusal)
    finally:
        await _teardown_critique(env)


# ---------------------------------------------------------------------------
# R10: a missed check allows the reply, with one warning
# ---------------------------------------------------------------------------


async def _with_critique(env):
    """Register critique beside appraisal, as test_appraisal_hooks does, with
    its sampler pinned so a vector miss draws no lens."""
    from corvidae_cognition.critique import CritiquePlugin, CritiqueSpec

    env.agent.stages.add("critique")
    critique = CritiquePlugin()
    env.pm.register(critique, name="critique")
    env.pm.add_hookspecs(CritiqueSpec)
    cfg = {"daemon": {"session_db": env.db_path}}
    await critique.on_init(pm=env.pm, config=cfg)
    await critique.on_start(config=cfg)
    critique._rng = FixedRng(value=0.99)
    return critique


@pytest.mark.parametrize("miss, cause", [
    ("fork-fails", "RuntimeError"),
    ("unusable-reply", "unusable"),
], ids=["fork-fails", "unusable-reply"])
async def test_r10_a_missed_check_allows_the_reply_with_exactly_one_warning(
    tmp_path, caplog, miss, cause,
):
    """R10: appraisal's before-send fork failing or unusable, with critique
    registered: the reply is allowed (None) and exactly ONE warning is logged,
    naming the channel, the trace and the cause (critique's own selection
    does not warn again)."""
    env, _held = await _presend_env(
        tmp_path, None, chat_text="not json at all" if miss == "unusable-reply" else IDLE_REPLY)
    critique = await _with_critique(env)
    try:
        if miss == "fork-fails":
            env.agent.error = RuntimeError("boom")
        with caplog.at_level(logging.WARNING):
            assert await _send(env, "a real draft", annotations={"origin": "user"}) is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        [text] = warnings
        assert env.channel.id in text and TRACE in text and cause in text
    finally:
        await critique.on_stop()
        await _teardown(env)


async def test_r10_forks_off_for_appraisal_logs_no_warning_per_final(tmp_path, caplog):
    """R10: forks off for appraisal's stage is stated once at startup, so a
    final with forks off logs no WARNING (debug at most)."""
    env, _held = await _presend_env(tmp_path, ObjectingPeer(None))
    try:
        env.agent.stages = set()
        with caplog.at_level(logging.DEBUG):
            assert await _send(env, "a real draft", annotations={"origin": "user"}) is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# R11: lens scores reported in both phases
# ---------------------------------------------------------------------------


def _phase_records(env, phase: str) -> list[dict]:
    """Appraisal's records for the trace that name ``phase``."""
    return [
        r["record"] for r in env.recorder.by_source("appraisal")
        if r["record"].get("phase") == phase]


async def test_r11_before_send_reports_the_lens_scores_with_its_phase(tmp_path):
    """R11: before send, appraisal fires on_trace_record(source="appraisal")
    with a record holding "phase": "before_send" and the lens scores."""
    env, _held = await _presend_env(tmp_path, ObjectingPeer(None))
    try:
        await _send(env, "a real draft", annotations={"origin": "user"})
        [record] = _phase_records(env, "before_send")
        assert record["lenses"] == SCORED_VECTOR
    finally:
        await _teardown(env)


async def test_r11_a_before_send_miss_reports_none_scores(tmp_path):
    """R11: the before-send record holds the scores as None on a miss, so a
    miss is countable."""
    env, _held = await _presend_env(tmp_path, ObjectingPeer(None), chat_text="not json at all")
    try:
        await _send(env, "a real draft", annotations={"origin": "user"})
        [record] = _phase_records(env, "before_send")
        assert "lenses" in record and record["lenses"] is None
    finally:
        await _teardown(env)


async def test_r11_the_idle_record_names_its_phase_beside_stage2(tmp_path):
    """R11: the idle record holds "phase": "idle" beside "stage2"."""
    env, _held = await _presend_env(tmp_path, ObjectingPeer(
        None, offers={"idle": [("alpha", "Did alpha?")]}))
    try:
        await _idle_exchange(env, "ki")
        [record] = _phase_records(env, "idle")
        assert "stage2" in record
    finally:
        await _teardown(env)
