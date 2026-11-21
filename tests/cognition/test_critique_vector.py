"""The critique vector (#1078): the appraisal fork scores the offered lenses
and critique runs the one that wins.

One test per acceptance criterion of the design's test plan (T1-T10 and the
revision additions), named by its ``T#`` in the docstring:

- T1/T2   selection is argmax over THIS exchange's vector; "none" (alone or
          tied) runs no lens, and the choice is recorded either way;
- T3/T3b  a lens without ``appraisal_question`` is rejected, naming plugin,
          lens and key; a removed kwarg's warning names the replacement;
- T4/T4b  lens settings live under ``critique.lens.<name>.``; the old keys
          are refused at startup, naming the new key;
- T5      ``Lens.wakes`` decides whether a plugin lens wakes; the wake stub
          depends on configuration only;
- T6      the stage-2 instruction carries the offer block and one fork call;
- T7/T7b  the vector is renormalized; bad entries are dropped with warnings;
- T8      the vector and the choice are persisted;
- T9      ``critique.sample_below_rate`` still samples when "none" wins;
- T10     every in-tree lens declares a one-line question (planning and
          alignment are checked in their own test files).

Critique is driven the way the daemon drives it: ``on_agent_response`` and
the appraisal stage-2 listener (``handle_stage2``) for the same trace, then
every task critique queued. Appraisal is driven through its real fork path.
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.fork import quote
from corvidae_cognition.critique import (
    NO_REPLY_MARKER,
    PROVENANCE_NOTE,
    RETENTION_WAKE_NOTE,
)

from helpers import FakeForkAgent, window_of
from tests.cognition.test_appraisal_stage2 import (
    _read_appraisal,
    _teardown as _teardown_appraisal,
    build_env as build_appraisal_env,
)
from tests.cognition.test_critique import (
    CLAIM,
    FakeAppraisal,
    FixedRng,
    RenderingForkAgent,
    UNSUPPORTED_VERDICT,
    _fire,
    _provenance_exchange,
    _read_outcomes,
    _run_next_task,
    _run_provenance,
    _teardown,
    build_critique,
)

# The words a plugin lens's objection carries, so a test can find it downstream.
LENS_OBJECTION = "vector-objection-5t8"
LENS_VERDICT = json.dumps({"objections": [
    {"claim": "a claim", "objection": LENS_OBJECTION, "severity": 0.6},
]})
RETENTION_VERDICT = json.dumps({"objections": [
    {"claim": "Yoz runs ARDC tech", "objection": "not filed",
     "file": "people/yoz.md", "content": "- runs ARDC tech", "severity": 0.5},
]})

# The brief marker of a plugin lens, and the question it offers.
BRIEF = "LENSBRIEF-{}: look for the thing this lens exists to find."
QUESTION = "Did the reply promise {} without saying how?"

# The sentence the stub gains once a plugin lens that wakes is loaded
# (design, Revision 3; final wording).
WAKE_SENTENCE = (
    "Findings from other self-check lenses may also be in the critique "
    "context: reply in the channel about one only if your earlier reply could "
    "lead the other party to take an incorrect action; otherwise send nothing, "
    "as above."
)

ROOT = "values/"


def _lens(name="mine", **kwargs):
    """A Lens named ``name`` with a brief marker and a question unique to it."""
    from corvidae_cognition.critique import Lens

    kwargs.setdefault("fork_prompt", BRIEF.format(name))
    kwargs.setdefault("appraisal_question", QUESTION.format(name))
    return Lens(name=name, **kwargs)


class OfferAppraisal(FakeAppraisal):
    """FakeAppraisal that also keeps the lens-offer provider critique hands
    it (``AppraisalPlugin.set_lens_offer``), so a test reads the offer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offer = None

    def set_lens_offer(self, provider):
        # Critique registers its offers here at start-up.
        self.offer = provider


class Contributor(CorvidaePlugin):
    """A test plugin, outside critique.py, that contributes ``lenses``."""

    depends_on = frozenset()

    def __init__(self, *lenses):
        self.lenses = lenses

    @hookimpl
    async def on_critique_init(self, config, add_lens):
        # Hand every lens to the callback.
        for lens in self.lenses:
            add_lens(lens)


class Misbuilder(CorvidaePlugin):
    """A test plugin whose lens still passes a selector removed in #1078."""

    depends_on = frozenset()

    @hookimpl
    async def on_critique_init(self, config, add_lens):
        # Built inside the hook, so a TypeError is the hook's to raise.
        from corvidae_cognition.critique import Lens

        add_lens(Lens(name="oldstyle", fork_prompt="brief", sample_rate=0.5))


async def _env(
    tmp_path, *lenses, contributors=None, chat_text=LENS_VERDICT, config=None,
    with_funnel=False, rng=0.99, appraisal=None,
):
    """critique's real wiring with an offer-recording appraisal and the
    lenses held by one ``kb`` contributor (or ``contributors``, name ->
    plugin). ``rng`` is every random draw."""
    if contributors is None:
        contributors = {"kb": Contributor(*lenses)}
    env = await build_critique(
        tmp_path, appraisal or OfferAppraisal(stage2={"past_claims": []}),
        chat_text, with_funnel=with_funnel, config=config,
        extra_plugins=contributors,
    )
    env.plugin._rng = FixedRng(value=rng)
    return env


async def _drain(env):
    """Run every task critique queued, in order."""
    while not env.task.task_queue.queue.empty():
        await _run_next_task(env.task)


async def _judged_turn(env, trace_id="kv", vector=None, origin="user", stage2="vector"):
    """One exchange the way the daemon runs it: on_agent_response, then the
    stage-2 listener for the same trace carrying ``vector`` (or ``stage2``
    verbatim when it is not "vector"), then every queued task."""
    await _fire(env, trace_id, origin=origin)
    if stage2 == "vector":
        stage2 = {"past_claims": [], "lenses": vector}
    [listener] = env.appraisal.stage2_listeners
    await listener(
        channel=env.channel, trace_id=trace_id, response_text="resp",
        annotations={} if origin == "user" else {"origin": origin},
        stage2=stage2,
    )
    await _drain(env)


def _marker_runs(env, marker):
    """How many forks carried ``marker`` in their instruction."""
    return sum(marker in c["instruction"] for c in env.fork_agent.calls)


def _warnings(caplog, *needles):
    """WARNING-or-worse texts that contain every needle."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno >= 30 and all(n in r.getMessage() for n in needles)
    ]


def _watch_funnel(env):
    """Spy on funnel.ingest and replace ingest_and_wake with a mock."""
    env.funnel.ingest = Mock(wraps=env.funnel.ingest)
    env.funnel.ingest_and_wake = AsyncMock()


# ---------------------------------------------------------------------------
# T1, T2 - selection is argmax over this exchange's vector
# ---------------------------------------------------------------------------


class TestSelection:
    @pytest.mark.parametrize("winner, marker", [
        ("mine", "LENSBRIEF-mine:"),                      # a plugin lens
        ("constrained", "[Self-critique: constrained]"),  # a built-in lens
    ])
    async def test_top_of_the_vector_runs_alone_and_the_last_stage2_is_ignored(
        self, tmp_path, winner, marker,
    ):
        """T1: the lens with the highest share runs, and only it, although
        get_last_stage2 holds a vector the old selectors would act on."""
        stale = OfferAppraisal(
            stage1={"commitment_density": 0.9, "disagreement": 0.9},
            stage2={"ambiguity": 0.95, "novelty": 0.9, "valence": 0.0, "past_claims": []},
        )
        env = await _env(tmp_path, _lens("mine"), appraisal=stale)
        try:
            await _judged_turn(env, "kt1", {winner: 0.6, "predictive": 0.3, "none": 0.1})
            # Exactly one lens forked and it is the winner; the stale
            # vector's predictive lens did not run.
            assert [marker in c["instruction"] for c in env.fork_agent.calls] == [True]
            assert _marker_runs(env, "[Self-critique: predictive]") == 0
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("vector", [
        {"mine": 0.2, "none": 0.8},   # none wins
        {"mine": 0.5, "none": 0.5},   # none ties a lens: none wins the tie
    ], ids=["none-top", "none-tied"])
    async def test_none_runs_no_lens_and_is_recorded_without_a_lens(
        self, tmp_path, vector,
    ):
        """T2: "none" winning (or tying) runs nothing; lens_selection is
        recorded with no ``lens`` key (RNG above sample_below_rate)."""
        env = await _env(tmp_path, _lens("mine"))
        try:
            await _judged_turn(env, "kt2", vector)
            assert env.fork_agent.calls == []
            selection = (await _read_outcomes(env.db, "kt2"))["lens_selection"]
            assert "lens" not in selection
            assert selection["sampled"] is False
            assert selection["basis"] == "none"
        finally:
            await _teardown(env)

    async def test_the_earliest_offered_lens_wins_a_tie_between_lenses(self, tmp_path):
        """T1 (tie among lenses): the earlier lens in offer order wins; the
        built-ins come before an added lens."""
        env = await _env(tmp_path, _lens("mine"))
        try:
            await _judged_turn(env, "kt1b", {"mine": 0.4, "predictive": 0.4, "none": 0.2})
            assert [
                "[Self-critique: predictive]" in c["instruction"]
                for c in env.fork_agent.calls] == [True]
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# T3, T3b - a lens must declare its question
# ---------------------------------------------------------------------------


class TestQuestionRequired:
    async def test_lens_without_a_question_is_rejected_naming_plugin_lens_and_key(
        self, tmp_path, caplog,
    ):
        """T3: a Lens with no appraisal_question is not loaded; the warning
        names the plugin, the lens and what to add. A valid lens still loads."""
        from corvidae_cognition.critique import Lens

        caplog.set_level("WARNING")
        bare = Lens(name="bare", fork_prompt="a brief")
        env = await _env(tmp_path, contributors={
            "contrib": Contributor(bare, _lens("good"))})
        try:
            assert _warnings(caplog, "contrib", "bare", "appraisal_question"), caplog.text
            offered = [name for name, _ in env.appraisal.offer()]
            assert "good" in offered and "bare" not in offered
        finally:
            await _teardown(env)

    async def test_removed_selector_kwarg_warning_names_appraisal_question(
        self, tmp_path, caplog,
    ):
        """T3b: a lens built with a removed kwarg (sample_rate=) fails in the
        plugin's hook; the warning names the plugin and the replacement."""
        caplog.set_level("WARNING")
        env = await _env(tmp_path, contributors={"oldplug": Misbuilder()})
        try:
            assert _warnings(caplog, "oldplug", "appraisal_question"), caplog.text
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# T4, T4b - settings live under critique.lens.<name>.
# ---------------------------------------------------------------------------


class TestConfigKeys:
    async def test_fork_prompt_override_reaches_the_fork(self, tmp_path):
        """T4: critique.lens.<name>.fork_prompt replaces the lens's brief."""
        env = await _env(
            tmp_path, _lens("mine"),
            config={"critique": {"lens": {"mine": {"fork_prompt": "OPERATOR-BRIEF-4k"}}}})
        try:
            await _judged_turn(env, "kt4a", {"mine": 1.0})
            [call] = env.fork_agent.calls
            assert "OPERATOR-BRIEF-4k" in call["instruction"]
            assert "LENSBRIEF-mine:" not in call["instruction"]
        finally:
            await _teardown(env)

    async def test_appraisal_question_override_reaches_the_offer(self, tmp_path):
        """T4: critique.lens.<name>.appraisal_question replaces the question
        in what the appraisal fork is offered, for an added lens and a built-in."""
        env = await _env(
            tmp_path, _lens("mine"),
            config={"critique": {"lens": {
                "mine": {"appraisal_question": "OPERATOR-QUESTION-mine"},
                "predictive": {"appraisal_question": "OPERATOR-QUESTION-predictive"},
            }}})
        try:
            offer = dict(env.appraisal.offer())
            assert offer["mine"] == "OPERATOR-QUESTION-mine"
            assert offer["predictive"] == "OPERATOR-QUESTION-predictive"
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("old_config, key, remedy", [
        ({"critique": {"fork_prompt": {"predictive": "p"}}},
         "critique.fork_prompt.predictive", "critique.lens.predictive.fork_prompt"),
        ({"critique": {"fork_prompt": {"provenance": "p"}}},
         "critique.fork_prompt.provenance", "critique.provenance.fork_prompt"),
        ({"critique": {"fork_prompt": {"proactivity": "p"}}},
         "critique.fork_prompt.proactivity", "critique.proactivity.fork_prompt"),
        ({"critique": {"lens": {"ambiguity": 0.5}}},
         "critique.lens.ambiguity", "appraisal_question"),
        ({"critique": {"lens": {"predictive": {"sample_rate": 0.5}}}},
         "critique.lens.predictive.sample_rate", "appraisal_question"),
    ], ids=["fork-prompt", "provenance-prompt", "proactivity-prompt",
            "threshold", "sample-rate"])
    async def test_removed_key_fails_startup_naming_key_and_remedy(
        self, tmp_path, old_config, key, remedy,
    ):
        """T4/T4b: a still-set old key fails on_init with a message naming
        the key and the one that replaces it."""
        with pytest.raises(ValueError) as raised:
            await _env(tmp_path, config=old_config)
        assert key in str(raised.value)
        assert remedy in str(raised.value)

    async def test_provenance_fork_prompt_key_is_read(self, tmp_path):
        """T4b: critique.provenance.fork_prompt is the provenance brief."""
        env = await build_critique(
            tmp_path, OfferAppraisal(), config={"critique": {
                "sample_below_rate": 0.0,
                "provenance": {"fork_prompt": "OPERATOR-PROVENANCE-7m"}}},
            fork=False)
        agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
        try:
            await _provenance_exchange(env, agent)
            await _run_provenance(env)
            [call] = agent.calls
            assert call["instruction"].startswith("OPERATOR-PROVENANCE-7m")
        finally:
            await _teardown(env)

    async def test_proactivity_fork_prompt_key_is_read(self, tmp_path):
        """T4b: critique.proactivity.fork_prompt is the proactivity brief."""
        env = await build_critique(
            tmp_path, OfferAppraisal(), config={"critique": {
                "sample_below_rate": 0.0,
                "proactivity": {"fork_prompt": "OPERATOR-PROACTIVITY-3v"}}},
            fork=False)
        agent = RenderingForkAgent({"critique"}, text=json.dumps({"objections": []}))
        try:
            ex = await _provenance_exchange(env, agent)
            await env.plugin._run_proactivity(
                env.channel, ex.trace_id, "when is it due?", "Want me to do it?", origin=None)
            [call] = agent.calls
            assert call["instruction"].startswith("OPERATOR-PROACTIVITY-3v")
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# T5 - Lens.wakes and the stub
# ---------------------------------------------------------------------------


class TestWake:
    async def test_a_waking_plugin_lens_wakes_the_channel(self, tmp_path):
        """T5: a wakes=True plugin lens's objection fires ingest_and_wake on
        the critique origin."""
        env = await _env(tmp_path, _lens("mine", wakes=True), with_funnel=True)
        try:
            _watch_funnel(env)
            await _judged_turn(env, "kt5a", {"mine": 1.0})
            env.funnel.ingest_and_wake.assert_awaited_once()
            kwargs = env.funnel.ingest_and_wake.await_args.kwargs
            assert kwargs["origin"] == "critique"
            assert LENS_OBJECTION in "\n".join(kwargs["entries"])
        finally:
            await _teardown(env)

    async def test_a_plugin_lens_without_wakes_waits(self, tmp_path):
        """T5: without wakes the objection waits for the next turn."""
        env = await _env(tmp_path, _lens("mine"), with_funnel=True)
        try:
            _watch_funnel(env)
            await _judged_turn(env, "kt5b", {"mine": 1.0})
            env.funnel.ingest.assert_called_once()
            assert LENS_OBJECTION in "\n".join(env.funnel.ingest.call_args.kwargs["entries"])
            env.funnel.ingest_and_wake.assert_not_awaited()
        finally:
            await _teardown(env)

    async def test_a_waking_plugin_lens_waits_on_a_critique_origin_turn(self, tmp_path):
        """T5: the critique-origin brake holds for a waking plugin lens."""
        env = await _env(tmp_path, _lens("mine", wakes=True), with_funnel=True)
        try:
            _watch_funnel(env)
            await _judged_turn(env, "kt5c", {"mine": 1.0}, origin="critique")
            env.funnel.ingest.assert_called_once()
            env.funnel.ingest_and_wake.assert_not_awaited()
        finally:
            await _teardown(env)

    async def test_with_a_root_a_plugin_lens_wake_gets_the_retention_note_and_sentence(
        self, tmp_path,
    ):
        """T5: with a root set, the stub is RETENTION_WAKE_NOTE plus the
        appended sentence, even when the plugin lens's wake fires first."""
        env = await _env(
            tmp_path, _lens("mine", wakes=True), with_funnel=True,
            config={"critique": {"retention": {"root": ROOT}}})
        try:
            _watch_funnel(env)
            await _judged_turn(env, "kt5d", {"mine": 1.0})
            stub = env.funnel.ingest_and_wake.await_args.kwargs["stub_text"]
            base = RETENTION_WAKE_NOTE.format(root=quote(ROOT), marker=NO_REPLY_MARKER)
            assert stub.startswith(base)
            assert stub.endswith(WAKE_SENTENCE)
        finally:
            await _teardown(env)

    async def test_without_a_root_a_plugin_lens_wake_gets_the_provenance_note_and_sentence(
        self, tmp_path,
    ):
        """T5 (Revision 3): no root and a waking plugin lens loaded: the
        stub is PROVENANCE_NOTE plus the appended sentence."""
        env = await _env(tmp_path, _lens("mine", wakes=True), with_funnel=True)
        try:
            _watch_funnel(env)
            await _judged_turn(env, "kt5e", {"mine": 1.0})
            stub = env.funnel.ingest_and_wake.await_args.kwargs["stub_text"]
            assert stub.startswith(PROVENANCE_NOTE)
            assert stub.endswith(WAKE_SENTENCE)
        finally:
            await _teardown(env)

    async def test_with_no_waking_plugin_lens_the_retention_stub_is_unchanged(
        self, tmp_path,
    ):
        """T5: with only a non-waking plugin lens loaded, retention (selected
        by the vector) wakes with today's note, byte for byte."""
        env = await _env(
            tmp_path, _lens("mine"), with_funnel=True, chat_text=RETENTION_VERDICT,
            config={"critique": {"retention": {"root": ROOT}}})
        try:
            _watch_funnel(env)
            await _judged_turn(env, "kt5f", {"retention": 1.0})
            stub = env.funnel.ingest_and_wake.await_args.kwargs["stub_text"]
            assert stub == RETENTION_WAKE_NOTE.format(
                root=quote(ROOT), marker=NO_REPLY_MARKER)
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# T6, T8 - the stage-2 fork carries the offer; the vector is persisted
# ---------------------------------------------------------------------------

DIMENSIONS_JSON = {
    "valence": 0.1, "stakes": 0.2, "ambiguity": 0.3,
    "commitment_density": 0.4, "novelty": 0.6, "correction": False,
    "past_claims": [{"claim": "we met Friday", "queries": ["Friday"]}],
}
OFFER = [("alpha", "Did the reply say alpha?"), ("beta", "Did the reply say beta?")]


async def _appraise(tmp_path, reply, offer=OFFER, fork_prompt="Score it."):
    """Run one stage-2 task through the real fork path with ``offer`` as the
    critique offer (None: no provider) and ``reply`` as the fork's answer.
    Returns (env, agent, stored stage2)."""
    env = await build_appraisal_env(tmp_path)
    agent = FakeForkAgent({"appraisal"}, text=json.dumps(reply))
    env.pm.unregister(name="agent")
    env.pm.register(agent, name="agent")
    env.channel.conversation = window_of(env.channel.id, [41, 42])
    # An overridden fork prompt: the block must come after it.
    new_config = {**env.appraisal.config, "appraisal": {"stage2_fork_prompt": fork_prompt}}
    env.appraisal.config = new_config
    await env.appraisal.on_config_reload(new_config)
    if offer is not None:
        env.appraisal.set_lens_offer(lambda: offer)
    await env.outcome.record_exchange("k", env.channel.id, origin="user")
    await env.appraisal.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=41, text="is it Friday?", annotations={})
    await env.appraisal._run_stage2(env.channel, "k", "is it Friday?", "yes, Friday", None)
    stored = ((await _read_appraisal(env.db, "k")) or {}).get("stage2")
    await _teardown_appraisal(env)
    return env, agent, stored


class TestStage2Offer:
    async def test_instruction_holds_the_block_after_the_overridden_prompt_in_one_call(
        self, tmp_path,
    ):
        """T6: the fork instruction lists each offered lens's name and
        question after the (overridden) prompt; one fork call, no
        standalone call; dimensions and past_claims still parse."""
        reply = {**DIMENSIONS_JSON, "lenses": {"alpha": 60, "beta": 10, "none": 30}}
        env, agent, stored = await _appraise(tmp_path, reply)
        assert len(agent.calls) == 1
        assert env.stub.chat_calls == []
        instruction = agent.calls[0]["instruction"]
        assert instruction.startswith("Score it.")
        for name, question in OFFER:
            line = f"- {name}: {question}"
            assert line in instruction
            assert instruction.index(line) > len("Score it.")
        assert '"lenses"' in instruction
        assert stored["novelty"] == 0.6
        assert stored["past_claims"][0]["claim"] == "we met Friday"

    async def test_extra_body_carries_the_lenses_schema_without_touching_the_shared_one(
        self, tmp_path,
    ):
        """T6: extra_body has a lenses object with every offered name and
        "none", required; the module's static schema is not modified."""
        from corvidae_cognition.appraisal import STAGE2_SCHEMA

        reply = {**DIMENSIONS_JSON, "lenses": {"alpha": 60, "beta": 10, "none": 30}}
        _, agent, _ = await _appraise(tmp_path, reply)
        schema = agent.calls[0]["extra_body"]["response_format"]["json_schema"]["schema"]
        lenses = schema["properties"]["lenses"]
        assert set(lenses["properties"]) == {"alpha", "beta", "none"}
        assert set(lenses["required"]) == {"alpha", "beta", "none"}
        assert "lenses" in schema["required"]
        assert "lenses" not in STAGE2_SCHEMA["properties"]

    async def test_the_vector_is_persisted_in_the_stage2_envelope_renormalized(
        self, tmp_path,
    ):
        """T8: stage2["lenses"] is stored on the exchange, as shares of 1."""
        reply = {**DIMENSIONS_JSON, "lenses": {"alpha": 60, "beta": 10, "none": 30}}
        _, _, stored = await _appraise(tmp_path, reply)
        assert stored["lenses"] == pytest.approx({"alpha": 0.6, "beta": 0.1, "none": 0.3})


class TestChoiceRecorded:
    async def test_the_chosen_lens_is_recorded(self, tmp_path):
        """T8: lens_selection holds the lens, sampled false and the basis."""
        env = await _env(tmp_path, _lens("mine"))
        try:
            await _judged_turn(env, "kt8", {"mine": 0.7, "none": 0.3})
            outcomes = await _read_outcomes(env.db, "kt8")
            assert outcomes["lens_selection"] == {
                "lens": "mine", "sampled": False, "basis": "vector"}
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# T7, T7b - parsing the vector
# ---------------------------------------------------------------------------


def _parse(lenses, caplog, trace_id="k7", offered=("alpha", "beta")):
    """_stage2_from_text over a reply whose ``lenses`` field is ``lenses``
    (the raw JSON text, so NaN can be written); a None leaves the field out."""
    from corvidae_cognition.appraisal import _stage2_from_text

    body = json.dumps({k: v for k, v in DIMENSIONS_JSON.items()})
    if lenses is not None:
        body = body[:-1] + ', "lenses": ' + lenses + "}"
    caplog.set_level("WARNING")
    return _stage2_from_text(body, list(offered), trace_id)


class TestLensVector:
    def test_renormalizes_to_shares_of_one(self, caplog):
        """T7: the percentages become shares that sum to 1."""
        stage2 = _parse('{"alpha": 30, "beta": 10, "none": 10}', caplog)
        assert stage2["lenses"] == pytest.approx({"alpha": 0.6, "beta": 0.2, "none": 0.2})
        assert stage2["novelty"] == 0.6

    def test_unknown_name_and_negative_value_are_dropped_with_warnings(self, caplog):
        """T7: each drop is a warning naming the entry and the trace."""
        stage2 = _parse('{"alpha": 30, "ghost": 50, "beta": -5, "none": 10}', caplog)
        assert stage2["lenses"] == pytest.approx({"alpha": 0.75, "none": 0.25})
        assert _warnings(caplog, "ghost", "k7"), caplog.text
        assert _warnings(caplog, "beta", "k7"), caplog.text

    def test_non_numeric_value_is_dropped_with_a_warning(self, caplog):
        """T7: a non-numeric share is dropped, naming it."""
        stage2 = _parse('{"alpha": "lots", "beta": 10, "none": 10}', caplog)
        assert stage2["lenses"] == pytest.approx({"beta": 0.5, "none": 0.5})
        assert _warnings(caplog, "alpha", "k7"), caplog.text

    def test_nan_is_dropped_with_a_warning(self, caplog):
        """T7b: NaN is not a usable share."""
        stage2 = _parse('{"alpha": NaN, "beta": 10, "none": 10}', caplog)
        assert stage2["lenses"] == pytest.approx({"beta": 0.5, "none": 0.5})
        assert _warnings(caplog, "alpha", "k7"), caplog.text

    @pytest.mark.parametrize("lenses", [None, '"alpha"', '{"alpha": 0, "none": 0}'],
                             ids=["missing", "not-a-dict", "all-zero"])
    def test_unusable_vector_leaves_none_and_warns_naming_the_trace(self, caplog, lenses):
        """T7: no usable vector warns with the trace and the reason; the
        dimensions still land and stage2 has no ``lenses`` key."""
        stage2 = _parse(lenses, caplog)
        assert "lenses" not in stage2
        assert stage2["novelty"] == 0.6
        assert _warnings(caplog, "k7", "lenses"), caplog.text

    async def test_critique_warns_naming_the_trace_when_stage2_is_none(
        self, tmp_path, caplog,
    ):
        """T7: with no stage 2 critique treats the exchange as "none" and
        warns naming the trace and the reason; the choice is recorded."""
        caplog.set_level("WARNING")
        env = await _env(tmp_path, _lens("mine"))
        try:
            await _judged_turn(env, "kt7", stage2=None)
            assert _warnings(caplog, "kt7", "vector"), caplog.text
            assert env.fork_agent.calls == []
            selection = (await _read_outcomes(env.db, "kt7"))["lens_selection"]
            assert selection == {"sampled": False, "basis": "no_vector"}
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# T9 - sample_below_rate
# ---------------------------------------------------------------------------


class TestSampling:
    async def test_none_below_the_rate_runs_a_random_offered_lens_sampled(self, tmp_path):
        """T9: "none" wins and the draw is below critique.sample_below_rate:
        a random offered lens runs and is marked sampled."""
        env = await _env(tmp_path, _lens("mine"))
        env.plugin._rng = FixedRng(value=0.01, choice_value="mine")
        try:
            await _judged_turn(env, "kt9", {"mine": 0.1, "none": 0.9})
            assert _marker_runs(env, "LENSBRIEF-mine:") == 1
            outcomes = await _read_outcomes(env.db, "kt9")
            assert outcomes["critique"]["mine"]["sampled"] is True
            selection = outcomes["lens_selection"]
            assert (selection["lens"], selection["sampled"]) == ("mine", True)
        finally:
            await _teardown(env)


# ---------------------------------------------------------------------------
# T10 - the in-tree lenses declare a question
# ---------------------------------------------------------------------------


class TestOffers:
    async def test_every_built_in_lens_offers_a_one_line_question(self, tmp_path):
        """T10: predictive, constrained, adversarial and (with a root)
        retention each offer a non-empty single-line question, in that order."""
        env = await _env(tmp_path, config={"critique": {"retention": {"root": ROOT}}})
        try:
            offer = env.appraisal.offer()
            assert [name for name, _ in offer] == [
                "predictive", "constrained", "adversarial", "retention"]
            for name, question in offer:
                assert question.strip() and "\n" not in question, name
        finally:
            await _teardown(env)

    async def test_retention_is_not_offered_without_a_root(self, tmp_path):
        """T10: no knowledge-base root, no retention offer."""
        env = await _env(tmp_path)
        try:
            assert "retention" not in [name for name, _ in env.appraisal.offer()]
        finally:
            await _teardown(env)
