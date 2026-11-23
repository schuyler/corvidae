"""Proactivity as an ordinary lens (#1126).

Proactivity used to run beside the lenses whenever the reply matched a
permission-question regex. It is now a built-in lens: the appraisal fork's
critique vector picks it like any other, and its findings wake the channel
(``wakes=True`` in ``BUILTIN_LENSES``). Its keys live under
``critique.lens.proactivity``, with a static ``enabled`` switch the agent
can't change.

One test per acceptance criterion (R1-R10 on the issue). Critique is
driven the way the daemon drives it: ``on_agent_response``, then the
appraisal stage-2 listener for the same trace, then every task critique
queued.
"""

import json

import pytest
from corvidae_cognition.critique import (
    BUILTIN_LENSES,
)
from corvidae_cognition.outcome_log import _scrub_outcomes_detail

from tests.cognition.test_critique import _fire, _read_outcomes, _teardown
from tests.cognition.test_critique_vector import (
    _drain,
    _env,
    _warnings,
    _watch_funnel,
)

# The marker every proactivity fork instruction opens with.
MARKER = "[Self-critique: proactivity]"
# A reply that asks permission, the shape the old regex fired on, and one
# that doesn't: the vector alone decides, so tests that expect a fork use it.
ASKS = "Should I restart the indexer?"
PLAIN = "The indexer is running again."
# A proactivity finding, carrying words a test can find downstream.
OBJECTION = "proactivity-objection-2w"
VERDICT = json.dumps({"objections": [
    {"claim": "restart the indexer", "objection": OBJECTION, "severity": 0.7},
]})
# The stub sentence the owner approved for proactivity findings (R6).
SENTENCE = (
    "For each [proactivity] finding, do the thing you asked permission for: "
    "it is yours to do. Then send nothing unless the other party would "
    "otherwise act on a wrong belief."
)
OFF = {"lens": {"proactivity": {"enabled": False}}}


def _proactivity_runs(env) -> int:
    """How many forks were proactivity forks."""
    return sum(MARKER in c["instruction"] for c in env.fork_agent.calls)


async def _turn(env, trace_id, vector, *, origin="user", response_text=ASKS):
    """One judged exchange whose stage 2 carries ``vector``, then every
    queued task."""
    await _fire(env, trace_id, origin=origin, response_text=response_text,
                vector=vector)
    await _drain(env)


class TestChosenByTheVector:
    async def test_proactivity_runs_when_it_tops_the_vector(self, tmp_path):
        """R1: proactivity on top of the vector runs, alone."""
        env = await _env(tmp_path, chat_text=VERDICT)
        try:
            await _turn(env, "kp1", {"proactivity": 0.6, "predictive": 0.3, "none": 0.1},
                        response_text=PLAIN)
            assert [MARKER in c["instruction"] for c in env.fork_agent.calls] == [True]
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("vector", [
        {"predictive": 0.7, "proactivity": 0.2, "none": 0.1},
        {"proactivity": 0.1, "none": 0.9},
    ], ids=["another-lens-wins", "none-wins"])
    async def test_a_permission_question_alone_runs_no_proactivity_fork(
        self, tmp_path, vector,
    ):
        """R1/R3: a reply that asks permission gets no proactivity fork
        unless the vector picks proactivity (RNG above the sample rate)."""
        env = await _env(tmp_path, chat_text=VERDICT)
        try:
            await _turn(env, "kp2", vector)
            assert _proactivity_runs(env) == 0
        finally:
            await _teardown(env)

    async def test_offered_with_its_own_question(self, tmp_path):
        """R1: proactivity is offered with its question (the offer order is
        TestOffers' in test_critique_vector.py)."""
        env = await _env(tmp_path)
        try:
            assert dict(env.appraisal.offer())["proactivity"] == (
                BUILTIN_LENSES["proactivity"][0])
        finally:
            await _teardown(env)

    async def test_the_question_can_be_overridden(self, tmp_path):
        """R8: critique.lens.proactivity.appraisal_question reaches the offer."""
        env = await _env(tmp_path, config={"critique": {"lens": {"proactivity": {
            "appraisal_question": "OPERATOR-PROACTIVITY-Q"}}}})
        try:
            assert dict(env.appraisal.offer())["proactivity"] == "OPERATOR-PROACTIVITY-Q"
        finally:
            await _teardown(env)


class TestWakes:
    async def test_a_finding_wakes_with_the_proactivity_sentence(self, tmp_path):
        """R2/R6: a proactivity finding fires one wake on the critique
        origin, and the stub carries the proactivity sentence."""
        env = await _env(tmp_path, chat_text=VERDICT, with_funnel=True)
        try:
            _watch_funnel(env)
            await _turn(env, "kp3", {"proactivity": 1.0})
            env.funnel.ingest_and_wake.assert_awaited_once()
            kwargs = env.funnel.ingest_and_wake.await_args.kwargs
            assert kwargs["origin"] == "critique"
            assert OBJECTION in "\n".join(kwargs["entries"])
            assert SENTENCE in kwargs["stub_text"]
        finally:
            await _teardown(env)

    async def test_a_finding_on_a_correction_waits(self, tmp_path):
        """R2: on a critique-origin turn the finding waits for the next turn."""
        env = await _env(tmp_path, chat_text=VERDICT, with_funnel=True)
        try:
            _watch_funnel(env)
            await _turn(env, "kp4", {"proactivity": 1.0}, origin="critique",
                        response_text=PLAIN)
            env.funnel.ingest_and_wake.assert_not_awaited()
            entries = [e for c in env.funnel.ingest.call_args_list
                       for e in c.kwargs["entries"]]
            assert any(OBJECTION in e for e in entries), entries
        finally:
            await _teardown(env)

    async def test_a_low_severity_finding_still_wakes(self, tmp_path):
        """R9: no severity filter, like every other lens."""
        low = json.dumps({"objections": [
            {"claim": "restart the indexer", "objection": OBJECTION, "severity": 0.2}]})
        env = await _env(tmp_path, chat_text=low, with_funnel=True)
        try:
            _watch_funnel(env)
            await _turn(env, "kp5", {"proactivity": 1.0})
            env.funnel.ingest_and_wake.assert_awaited_once()
        finally:
            await _teardown(env)


class TestRecorded:
    async def test_recorded_under_critique_where_redaction_scrubs_it(self, tmp_path):
        """R7: the outcome lands under ``critique`` (not ``proactivity``),
        and the outcome-log scrubber tombstones its text."""
        env = await _env(tmp_path, chat_text=VERDICT)
        try:
            await _turn(env, "kp7", {"proactivity": 1.0})
            outcomes = await _read_outcomes(env.recorder, "kp7")
            assert "proactivity" not in outcomes
            assert outcomes["critique"]["lenses"] == ["proactivity"]
            scrubbed = json.loads(_scrub_outcomes_detail(json.dumps(outcomes), "[X]"))
            assert OBJECTION not in json.dumps(scrubbed)
        finally:
            await _teardown(env)

    async def test_a_failed_fork_warns_naming_the_lens(self, tmp_path, caplog):
        """R10: a proactivity fork whose reply is unusable is a warned skip."""
        caplog.set_level("WARNING")
        env = await _env(tmp_path, chat_text="not json at all")
        try:
            await _turn(env, "kp8", {"proactivity": 1.0})
            assert _warnings(caplog, "proactivity", "kp8", "skipped"), caplog.text
        finally:
            await _teardown(env)


class TestEnabled:
    async def test_disabled_is_not_offered_and_does_not_wake(self, tmp_path):
        """R5: enabled: false takes proactivity out of the offer; a vector
        that still names it runs nothing."""
        env = await _env(tmp_path, chat_text=VERDICT, config={"critique": OFF})
        try:
            assert "proactivity" not in [n for n, _ in env.appraisal.offer()]
            await _turn(env, "kp9", {"proactivity": 1.0})
            assert _proactivity_runs(env) == 0
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("value", ["false", 0, None], ids=["string", "int", "null"])
    async def test_a_non_bool_enabled_fails_startup(self, tmp_path, value):
        """R5: enabled must be true or false; the error names key and value."""
        with pytest.raises(ValueError) as raised:
            await _env(tmp_path, config={"critique": {"lens": {"proactivity": {
                "enabled": value}}}})
        assert "critique.lens.proactivity.enabled" in str(raised.value)
        assert "true or false" in str(raised.value)

    async def test_a_non_bool_enabled_at_reload_warns_and_stays_on(
        self, tmp_path, caplog,
    ):
        """R5: at reload a bad value warns and the lens falls back to the
        default (on), not to the previous config (off)."""
        caplog.set_level("WARNING")
        env = await _env(tmp_path, config={"critique": OFF})
        try:
            config = {**env.plugin.config, "critique": {"lens": {"proactivity": {
                "enabled": "no"}}}}
            await env.plugin.on_config_reload(config)
            assert _warnings(caplog, "critique.lens.proactivity.enabled"), caplog.text
            assert "proactivity" in [n for n, _ in env.appraisal.offer()]
        finally:
            await _teardown(env)


class TestKeys:
    async def test_the_fork_prompt_key_reaches_the_fork(self, tmp_path):
        """R4: critique.lens.proactivity.fork_prompt replaces the brief."""
        env = await _env(tmp_path, chat_text=VERDICT, config={"critique": {"lens": {
            "proactivity": {"fork_prompt": "OPERATOR-PROACTIVITY-3v"}}}})
        try:
            await _turn(env, "kp10", {"proactivity": 1.0})
            [call] = env.fork_agent.calls
            assert call["instruction"].startswith("OPERATOR-PROACTIVITY-3v")
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("old, key, remedy", [
        ({"fork_prompt": "p"}, "critique.proactivity.fork_prompt",
         "critique.lens.proactivity.fork_prompt"),
        ({"enabled": False}, "critique.proactivity.enabled",
         "critique.lens.proactivity.enabled"),
        ({"other": 1}, "critique.proactivity.other", "removed"),
    ], ids=["fork-prompt", "enabled", "unknown"])
    async def test_an_old_key_fails_startup_naming_its_replacement(
        self, tmp_path, old, key, remedy,
    ):
        """R4/R5: a key under critique.proactivity fails on_init, naming it
        and where it went."""
        with pytest.raises(ValueError) as raised:
            await _env(tmp_path, config={"critique": {"proactivity": old}})
        assert key in str(raised.value)
        assert remedy in str(raised.value)

    async def test_an_old_key_at_reload_warns_and_keeps_the_prompts(
        self, tmp_path, caplog,
    ):
        """R4: at reload the old key warns, naming the new one, and the
        previous fork prompts stay."""
        caplog.set_level("WARNING")
        env = await _env(tmp_path)
        try:
            before = dict(env.plugin._fork_prompts)
            config = {**env.plugin.config, "critique": {"proactivity": {
                "fork_prompt": "IGNORED"}}}
            await env.plugin.on_config_reload(config)
            assert _warnings(
                caplog, "critique.proactivity.fork_prompt",
                "critique.lens.proactivity.fork_prompt"), caplog.text
            assert env.plugin._fork_prompts == before
        finally:
            await _teardown(env)

    async def test_an_empty_old_block_starts(self, tmp_path):
        """``critique.proactivity:`` with no keys (YAML null) sets nothing."""
        env = await _env(tmp_path, config={"critique": {"proactivity": None}})
        await _teardown(env)
