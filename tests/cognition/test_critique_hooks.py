"""RED tests for #1169 part 1, critique side: critique implements appraisal's
hooks and reaches appraisal only through them.

Each test names the item of the design's test list
(.claude/plans/1169-part1.md) it covers:

- item 8   the before-send offer: no retention, a plugin lens only with
           ``before_send=True``;
- item 9   the chosen before-send lens runs inline on the whole draft, its
           objections are queued (never waking), and both records are written;
- item 11  without appraisal ``critique.fallback_rate`` decides whether one
           lens runs at each point; the NO_REPLY veto holds; one INFO line;
- item 12  with appraisal present critique's own gate forks nothing: the
           before-send lens comes only through the hook;
- item 14  critique keeps the originating text itself and appraisal's old
           seams are gone, boundary ratchet included;
- item 15  a non-bool ``Lens.before_send`` is rejected with a warning.

Critique is built with the helpers of tests/cognition/test_critique.py; the
"appraisal present" case registers a bare plugin under that name, since
critique decides presence with ``pm.has_plugin("appraisal")`` and calls
nothing on it.
"""

import json
import logging
import re
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from corvidae.hooks import CorvidaePlugin, hookimpl

from tests.cognition.test_critique import (
    FixedRng,
    _fire,
    _read_outcomes,
    _run_next_task,
    _teardown,
    build_critique,
)

# Longer than corvidae.fork.QUOTE_CHARS (120), with a line break, so a
# truncated or unflattened quote of the draft fails item 9.
DRAFT = (
    "The meeting is on Tuesday, in the small room on the third floor.\n"
    "I will send the notes tomorrow, once the minutes are approved by everyone."
)
OBJECTION_TEXT = "before-send-objection-9x2"
VERDICT = json.dumps({"objections": [
    {"claim": "I will send the notes tomorrow", "objection": OBJECTION_TEXT,
     "severity": 0.6},
]})
EMPTY = json.dumps({"objections": []})

ABOUT_TO_SEND = "the reply you are about to send"


class PresentAppraisal(CorvidaePlugin):
    """Stands for a registered appraisal plugin: critique must act through
    hooks only, so this has no methods at all."""

    depends_on = frozenset()


class Contributor(CorvidaePlugin):
    """A plugin outside critique.py that contributes ``lenses``."""

    depends_on = frozenset()

    def __init__(self, *lenses):
        self.lenses = lenses

    @hookimpl
    async def on_critique_init(self, config, add_lens):
        # Hand every lens to the callback.
        for lens in self.lenses:
            add_lens(lens)


def _lens(name, **kwargs):
    """A plugin lens with a brief and a question unique to it."""
    from corvidae_cognition.critique import Lens

    return Lens(
        name=name, fork_prompt=f"BRIEF-{name}",
        appraisal_question=f"Did the reply do {name}?", **kwargs)


async def _drain(env):
    """Run every task critique queued, in order."""
    while not env.task.task_queue.queue.empty():
        await _run_next_task(env.task)


async def _begin(env, trace_id, text="the user asked"):
    """Record the exchange's originating message, as the
    agent does when it persists the user message."""
    await env.plugin.on_message_persisted(
        channel=env.channel, trace_id=trace_id, rowid=1, text=text, annotations={})


async def _gate(env, text=DRAFT, *, trace_id="kg", annotations=None):
    """Fire critique's should_send_response for a final draft."""
    return await env.plugin.should_send_response(
        channel=env.channel, text=text, emission="final", trace_id=trace_id,
        annotations=annotations or {})


# ---------------------------------------------------------------------------
# Item 8: the before-send offer
# ---------------------------------------------------------------------------


async def test_before_send_offer_has_no_retention_and_only_opted_in_plugin_lenses(
    tmp_path,
):
    """Item 8: before send critique offers the built-ins except retention
    (even with a root set) and a plugin lens only when it declares
    before_send=True; the idle offer is unchanged (retention and every
    plugin lens)."""
    env = await build_critique(
        tmp_path, PresentAppraisal(),
        config={"critique": {"retention": {"root": "values/"}}},
        extra_plugins={"kb": Contributor(
            _lens("early", before_send=True), _lens("late"))},
    )
    try:
        before = await env.plugin.on_appraisal_prompts(
            channel=env.channel, trigger="before_send")
        names = [name for name, _question in before]
        assert "retention" not in names
        assert "early" in names and "late" not in names
        assert {"predictive", "constrained", "adversarial"} <= set(names)
        # Each offer is a (name, one-line question) pair.
        assert all(isinstance(q, str) and q and "\n" not in q for _n, q in before)
        # Idle: the full list, retention and both plugin lenses included.
        idle = [n for n, _q in await env.plugin.on_appraisal_prompts(
            channel=env.channel, trigger="idle")]
        assert {"retention", "early", "late"} <= set(idle)
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# Item 9: the chosen before-send lens
# ---------------------------------------------------------------------------


async def test_chosen_before_send_lens_runs_inline_and_its_objections_wait(tmp_path):
    """Item 9: on_appraisal_before_send runs the winning lens inline (the
    fork has happened when the hook returns, nothing was queued) with the
    whole draft in the instruction; its objections, proactivity's included,
    go through funnel.ingest and never ingest_and_wake; the outcomes carry
    critique_before_send and lens_selection_before_send; the hook returns
    None."""
    env = await build_critique(
        tmp_path, PresentAppraisal(), chat_text=VERDICT, with_funnel=True)
    try:
        env.funnel.ingest = Mock(wraps=env.funnel.ingest)
        env.funnel.ingest_and_wake = AsyncMock()
        await _begin(env, "kb9")
        result = await env.plugin.on_appraisal_before_send(
            channel=env.channel, trace_id="kb9", text=DRAFT, annotations={},
            lenses={"proactivity": 0.8, "none": 0.2})
        assert result is None
        # Inline: one fork already ran, under critique's attribution, and
        # nothing waits on the task queue.
        [call] = env.fork_agent.calls
        assert env.task.task_queue.queue.empty()
        assert call["attribution"].get("stage") == "critique"
        assert "[Self-critique: proactivity]" in call["instruction"]
        assert ABOUT_TO_SEND in call["instruction"]
        assert " ".join(DRAFT.split()) in call["instruction"]
        # Queued, not waking: proactivity wakes after the exchange but not
        # before the send.
        env.funnel.ingest_and_wake.assert_not_awaited()
        env.funnel.ingest.assert_called_once()
        entries = env.funnel.ingest.call_args.kwargs["entries"]
        assert len(entries) == 1 and OBJECTION_TEXT in entries[0]
        # Both records, under their own keys.
        outcomes = await _read_outcomes(env.recorder, "kb9")
        before = outcomes["critique_before_send"]
        assert before["lens"] == "proactivity"
        assert before["objections"] == 1
        assert before["sampled"] is False
        selection = outcomes["lens_selection_before_send"]
        assert selection["lens"] == "proactivity"
        assert selection["sampled"] is False
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# Items 11 and 12: with and without appraisal
# ---------------------------------------------------------------------------


async def test_fallback_rate_one_runs_one_lens_at_each_point(tmp_path):
    """Item 11: with no appraisal plugin and critique.fallback_rate 1.0, the
    gate runs one before-send lens (the draft in the instruction, objections
    queued, gate returns None) and the finished exchange runs one idle lens.
    The drawn lens is proactivity, which wakes after the exchange: before
    the send its finding still only waits (#1192)."""
    env = await build_critique(
        tmp_path, None, chat_text=VERDICT, with_funnel=True,
        config={"critique": {"fallback_rate": 1.0}})
    try:
        env.plugin._rng = FixedRng(value=0.5, choice_value="proactivity")
        env.funnel.ingest = Mock(wraps=env.funnel.ingest)
        env.funnel.ingest_and_wake = AsyncMock()
        await _begin(env, "kf1")
        assert await _gate(env, trace_id="kf1") is None
        [call] = env.fork_agent.calls
        assert ABOUT_TO_SEND in call["instruction"]
        assert "[Self-critique: proactivity]" in call["instruction"]
        env.funnel.ingest.assert_called_once()
        env.funnel.ingest_and_wake.assert_not_awaited()
        # A fallback run is not a sample below threshold, at either point.
        outcomes = await _read_outcomes(env.recorder, "kf1")
        assert outcomes["critique_before_send"]["sampled"] is False
        # After the exchange: one more lens, not a before-send one.
        await _fire(env, "kf1")
        await _drain(env)
        assert len(env.fork_agent.calls) == 2
        assert ABOUT_TO_SEND not in env.fork_agent.calls[1]["instruction"]
    finally:
        await _teardown(env)


async def test_fallback_rate_zero_runs_no_lens_at_either_point(tmp_path):
    """Item 11: at critique.fallback_rate 0.0 neither the gate nor the
    finished exchange runs a lens."""
    env = await build_critique(
        tmp_path, None, chat_text=VERDICT,
        config={"critique": {"fallback_rate": 0.0}})
    try:
        env.plugin._rng = FixedRng(value=0.5)
        await _begin(env, "kf0")
        assert await _gate(env, trace_id="kf0") is None
        await _fire(env, "kf0")
        await _drain(env)
        assert env.fork_agent.calls == []
    finally:
        await _teardown(env)


async def test_fallback_rate_defaults_to_a_quarter(tmp_path):
    """Item 11 (owner ruling, #1169 7093658157): with critique.fallback_rate
    unset, a draw of 0.24 runs the before-send fallback lens and 0.26 does
    not, so the default is 0.25."""
    for label, draw, forks in (("below", 0.24, 1), ("above", 0.26, 0)):
        # Each case gets its own directory, so the two session DBs differ.
        (tmp_path / label).mkdir()
        env = await build_critique(tmp_path / label, None, chat_text=EMPTY)
        try:
            env.plugin._rng = FixedRng(value=draw)
            await _begin(env, "kd")
            assert await _gate(env, trace_id="kd") is None
            assert len(env.fork_agent.calls) == forks, label
        finally:
            await _teardown(env)


async def test_no_reply_veto_holds_with_and_without_appraisal(tmp_path):
    """Item 11: critique's NO_REPLY veto on a critique-origin final still
    returns False, with no fork, whether appraisal is present or absent."""
    from corvidae_cognition.critique import NO_REPLY_MARKER

    for label, appraisal in (("absent", None), ("present", PresentAppraisal())):
        # Each case gets its own directory, so the two session DBs differ.
        (tmp_path / label).mkdir()
        env = await build_critique(
            tmp_path / label, appraisal, chat_text=VERDICT,
            config={"critique": {"fallback_rate": 1.0}})
        try:
            env.plugin._rng = FixedRng(value=0.5)
            await _begin(env, "kv")
            vetoed = await _gate(
                env, NO_REPLY_MARKER, trace_id="kv",
                annotations={"origin": "critique"})
            assert vetoed is False
            assert env.fork_agent.calls == []
        finally:
            await _teardown(env)


async def test_absent_appraisal_logs_one_info_line_naming_the_rate_key(
    tmp_path, caplog,
):
    """Item 11: without appraisal, on_start logs one INFO line that names
    critique.fallback_rate (the operator's lever) and says the provenance
    check is off."""
    with caplog.at_level(logging.INFO, logger="corvidae.critique"):
        env = await build_critique(tmp_path, None)
    try:
        lines = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and "fallback_rate" in r.getMessage()]
        assert len(lines) == 1, lines
        assert "provenance" in lines[0]
    finally:
        await _teardown(env)


async def test_with_appraisal_the_gate_forks_nothing_and_the_hook_does(tmp_path):
    """Item 12: with appraisal registered, critique's own should_send_response
    forks nothing even at fallback_rate 1.0, and the before-send lens runs
    only when appraisal's hook delivers a vector."""
    env = await build_critique(
        tmp_path, PresentAppraisal(), chat_text=EMPTY,
        config={"critique": {"fallback_rate": 1.0}})
    try:
        env.plugin._rng = FixedRng(value=0.5)
        await _begin(env, "k12")
        assert await _gate(env, trace_id="k12") is None
        assert env.fork_agent.calls == []
        # The hook path is the one that runs the lens.
        await env.plugin.on_appraisal_before_send(
            channel=env.channel, trace_id="k12", text=DRAFT, annotations={},
            lenses={"predictive": 1.0})
        [call] = env.fork_agent.calls
        assert "[Self-critique: predictive]" in call["instruction"]
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# Item 14: no reach into appraisal
# ---------------------------------------------------------------------------


async def test_critique_keeps_the_originating_text_from_its_own_hook(tmp_path):
    """Item 14: critique records the originating text from its own
    on_message_persisted: an idle lens forked after on_appraisal_idle quotes
    it, with an appraisal that exposes no reader (PresentAppraisal has no
    get_originating_text)."""
    env = await build_critique(tmp_path, PresentAppraisal(), chat_text=EMPTY)
    try:
        await _begin(env, "k14", text="ORIGINATING-TEXT-14")
        await env.plugin.on_appraisal_idle(
            channel=env.channel, trace_id="k14", response_text="resp",
            annotations={}, lenses={"predictive": 1.0}, past_claims=None)
        await _drain(env)
        [call] = env.fork_agent.calls
        assert "«ORIGINATING-TEXT-14»" in call["instruction"]
    finally:
        await _teardown(env)


def test_appraisal_has_no_listener_or_offer_seam_left():
    """Item 14: the add_stage2_listener / set_lens_offer / get_originating_text
    seams are gone from AppraisalPlugin."""
    from corvidae_cognition.appraisal import AppraisalPlugin

    for name in ("add_stage2_listener", "set_lens_offer", "get_originating_text"):
        assert not hasattr(AppraisalPlugin, name), name


# ---------------------------------------------------------------------------
# Item 15: Lens.before_send is validated
# ---------------------------------------------------------------------------


async def test_non_bool_before_send_is_rejected_with_a_warning_naming_it(
    tmp_path, caplog,
):
    """Item 15: a Lens whose before_send is not a bool is not loaded, and the
    warning names the plugin, the lens, the field, the rejected value and the
    accepted ones; a valid lens from the same plugin still loads."""
    caplog.set_level(logging.WARNING, logger="corvidae.critique")
    env = await build_critique(
        tmp_path, PresentAppraisal(),
        extra_plugins={"kb": Contributor(
            _lens("odd", before_send="yes"), _lens("fine", before_send=True))})
    try:
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        [text] = [w for w in warnings if "odd" in w]
        assert "kb" in text and "before_send" in text
        assert "'yes'" in text and "True or False" in text
        before = [n for n, _q in await env.plugin.on_appraisal_prompts(
            channel=env.channel, trigger="before_send")]
        assert "fine" in before and "odd" not in before
    finally:
        await _teardown(env)
