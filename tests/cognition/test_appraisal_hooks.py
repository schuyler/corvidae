"""RED tests for #1169 part 1, appraisal side: appraisal owns the lens contract.

Appraisal declares three hooks (``on_appraisal_prompts``,
``on_appraisal_before_send``, ``on_appraisal_idle``) in ``AppraisalSpec`` and
reaches its peers only through them. Each test names the item of the design's
test list (.claude/plans/1169-part1.md) it covers:

- item 1   the three hooks are declared, their params have no defaults;
- item 2   the idle task fires ``on_appraisal_idle`` once, on a miss too;
- item 3   the idle offer comes from ``on_appraisal_prompts(trigger="idle")``;
- item 5   a final draft is scored by one fork and the hook hears the vector;
- item 6   a decline, a non-final emission or a missing trace does nothing;
- item 7   a before-send fork miss fires the hook with ``lenses=None``;
- item 10  ``appraisal.before_send_fork_prompt`` replaces the default;
- item 13  forks off for appraisal warn once at ``on_start``, naming the keys.

The peers are a recording plugin registered on a real apluggy manager with
the hookspecs installed, the way the daemon wires it, so a hook that is
declared wrongly or never fired fails here.
"""

import inspect
import json
import logging

import pytest

from corvidae.fork import quote
from corvidae.hooks import CorvidaePlugin, hookimpl

from tests.cognition.test_appraisal_stage2 import (
    _originate,
    _run_next_task,
    _teardown,
    build_env,
)

TRACE = "kbs"

# What the fork answers when it scores one offered lens ("alpha") and "none".
# The shares are percentages; appraisal normalizes them to sum to 1.
SCORED_REPLY = json.dumps({"lenses": {"alpha": 75, "none": 25}})
SCORED_VECTOR = {"alpha": 0.75, "none": 0.25}

# A stage-2 (idle) reply: past claims plus the lens scores.
IDLE_REPLY = json.dumps({
    "past_claims": [{"claim": "we agreed on Tuesday", "queries": ["Tuesday"]}],
    "lenses": {"alpha": 75, "none": 25},
})
IDLE_CLAIMS = [{"claim": "we agreed on Tuesday", "queries": ["Tuesday"]}]


class PeerSpy(CorvidaePlugin):
    """A peer that implements the three appraisal hooks and records them.

    ``offers`` maps a trigger ("idle" / "before_send") to the (lens,
    question) pairs this peer offers for it; any other trigger gets None.
    """

    depends_on = frozenset()

    def __init__(self, offers: dict | None = None) -> None:
        self.offers = offers or {}
        self.prompt_triggers: list[str] = []
        self.before_send: list[dict] = []
        self.idle: list[dict] = []

    @hookimpl
    async def on_appraisal_prompts(self, channel, trigger):
        # Record the trigger and offer what was configured for it.
        self.prompt_triggers.append(trigger)
        return self.offers.get(trigger)

    @hookimpl
    async def on_appraisal_before_send(self, channel, trace_id, text, annotations, lenses):
        # Record what the peer was told before the send.
        self.before_send.append({
            "trace_id": trace_id, "text": text, "annotations": annotations,
            "lenses": lenses,
        })

    @hookimpl
    async def on_appraisal_idle(
        self, channel, trace_id, response_text, annotations, lenses, past_claims,
    ):
        # Record what the peer was told after the exchange.
        self.idle.append({
            "trace_id": trace_id, "response_text": response_text,
            "annotations": annotations, "lenses": lenses,
            "past_claims": past_claims,
        })


def _wire(env, spy: PeerSpy) -> None:
    """Register ``spy`` beside appraisal. build_env installed the declared
    hookspecs, as the daemon does; pluggy checks a later hookimpl against them."""
    env.pm.register(spy, name="peer")


async def _before_send_env(tmp_path, chat_text=SCORED_REPLY, offers=None):
    """An appraisal env with one before-send offer ("alpha") and an exchange
    whose originating row is known and in the window."""
    env = await build_env(tmp_path, chat_text=chat_text)
    spy = PeerSpy({"before_send": offers if offers is not None else [("alpha", "Did alpha?")]})
    _wire(env, spy)
    await _originate(env, TRACE)
    return env, spy


async def _send(env, text, *, emission="final", trace_id=TRACE, annotations=None):
    """Fire appraisal's should_send_response gate as the agent's final does."""
    return await env.appraisal.should_send_response(
        channel=env.channel, text=text, emission=emission, trace_id=trace_id,
        annotations=annotations or {},
    )


async def _idle_exchange(env, trace_id="ki"):
    """Run one finished exchange through appraisal's silent idle task."""
    await _originate(env, trace_id)
    await env.appraisal.on_agent_response(
        channel=env.channel, request_text="req", response_text="resp",
        trace_id=trace_id, annotations={"origin": "user"}, logprobs=None,
    )
    await _run_next_task(env.task)


# ---------------------------------------------------------------------------
# Item 1: the hooks are declared
# ---------------------------------------------------------------------------


def test_appraisal_declares_the_three_hooks_without_param_defaults():
    """Item 1: AppraisalPlugin.hookspecs is AppraisalSpec, which declares the
    three hooks with the parameters the design names and no defaults (pluggy
    silently drops a default; tests/test_hook_arg_binding.py is the guard)."""
    from corvidae_cognition.appraisal import AppraisalPlugin, AppraisalSpec

    # The plugin owns the spec class.
    assert AppraisalPlugin.hookspecs is AppraisalSpec
    expected = {
        "on_appraisal_prompts": ["channel", "trigger"],
        "on_appraisal_before_send": [
            "channel", "trace_id", "text", "annotations", "lenses"],
        "on_appraisal_idle": [
            "channel", "trace_id", "response_text", "annotations", "lenses",
            "past_claims"],
    }
    for name, params in expected.items():
        sig = inspect.signature(getattr(AppraisalSpec, name))
        names = [p for p in sig.parameters if p != "self"]
        assert names == params, name
        # No parameter may carry a default.
        assert all(
            p.default is inspect.Parameter.empty
            for p in sig.parameters.values()), f"{name} has a defaulted parameter"


# ---------------------------------------------------------------------------
# Items 2 and 3: the idle point
# ---------------------------------------------------------------------------


async def test_idle_fires_the_hook_once_with_the_vector_and_the_claims(tmp_path):
    """Item 2: appraisal's silent task fires on_appraisal_idle once, carrying
    the fork's normalized lens vector and its past claims."""
    env = await build_env(tmp_path, chat_text=IDLE_REPLY)
    try:
        spy = PeerSpy({"idle": [("alpha", "Did alpha?")]})
        _wire(env, spy)
        await _idle_exchange(env)
        [call] = spy.idle
        assert call["trace_id"] == "ki"
        assert call["response_text"] == "resp"
        assert call["lenses"] == SCORED_VECTOR
        assert call["past_claims"] == IDLE_CLAIMS
    finally:
        await _teardown(env)


async def test_idle_fires_the_hook_once_with_none_on_a_fork_miss(tmp_path):
    """Item 2: when the fork's reply is unusable the hook still fires once,
    with lenses and past_claims both None (the listener contract today)."""
    env = await build_env(tmp_path, chat_text="no json here")
    try:
        spy = PeerSpy({"idle": [("alpha", "Did alpha?")]})
        _wire(env, spy)
        await _idle_exchange(env)
        [call] = spy.idle
        assert call["lenses"] is None
        assert call["past_claims"] is None
    finally:
        await _teardown(env)


async def test_idle_offer_is_collected_with_trigger_idle(tmp_path):
    """Item 3: the idle fork's lens block and schema come from the peers'
    on_appraisal_prompts(trigger="idle"), and that is the only trigger an
    idle exchange asks for."""
    env = await build_env(tmp_path, chat_text=IDLE_REPLY)
    try:
        spy = PeerSpy({"idle": [("alpha", "Did alpha?")]})
        _wire(env, spy)
        await _idle_exchange(env)
        assert spy.prompt_triggers == ["idle"]
        [call] = env.agent.calls
        assert "- alpha: Did alpha?" in call["instruction"]
        schema = call["extra_body"]["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]["lenses"]["properties"]) == {"alpha", "none"}
    finally:
        await _teardown(env)


async def test_idle_offer_is_critiques_with_critique_loaded(tmp_path):
    """Item 3: with critique loaded the idle offer is critique's idle offer,
    retention included when a root is set. A recording peer proves the offer
    travelled through the hook (the old set_lens_offer seam would also list
    retention, so the trigger log is the claim)."""
    from corvidae_cognition.critique import CritiquePlugin, CritiqueSpec

    env = await build_env(tmp_path, chat_text=IDLE_REPLY)
    critique = CritiquePlugin()
    try:
        spy = PeerSpy()
        env.pm.register(critique, name="critique")
        # Critique declares its own hookspecs; add them as startup would.
        env.pm.add_hookspecs(CritiqueSpec)
        _wire(env, spy)
        cfg = {
            "daemon": {"session_db": env.db_path},
            "critique": {"retention": {"root": "values/"}},
        }
        await critique.on_init(pm=env.pm, config=cfg)
        await critique.on_start(config=cfg)
        await _idle_exchange(env)
        # The offer was asked for through the hook, at the idle trigger.
        assert spy.prompt_triggers == ["idle"]
        [call] = env.agent.calls
        assert "- retention:" in call["instruction"]
        assert "- predictive:" in call["instruction"]
    finally:
        await critique.on_stop()
        await _teardown(env)


# ---------------------------------------------------------------------------
# Items 5, 6, 7, 10: before send
# ---------------------------------------------------------------------------


async def test_before_send_reaches_critique_through_the_hooks(tmp_path):
    """R1 (#1169 7089831895), before send: with appraisal and critique on one
    plugin manager, appraisal's gate asks critique for its before-send offer,
    scores it in one fork, and critique's on_appraisal_before_send runs the
    winning lens on the draft in a second fork; the gate returns None."""
    from corvidae_cognition.critique import CritiquePlugin, CritiqueSpec

    # Both forks answer this: appraisal reads the lenses (predictive wins);
    # critique's lens reads an empty objections list.
    reply = json.dumps({"lenses": {"predictive": 90, "none": 10}, "objections": []})
    env = await build_env(tmp_path, chat_text=reply)
    env.agent.stages.add("critique")
    critique = CritiquePlugin()
    try:
        env.pm.register(critique, name="critique")
        # Critique declares its own hookspecs; add them as startup would.
        env.pm.add_hookspecs(CritiqueSpec)
        cfg = {"daemon": {"session_db": env.db_path}}
        await critique.on_init(pm=env.pm, config=cfg)
        await critique.on_start(config=cfg)
        await _originate(env, TRACE)
        await critique.on_message_persisted(
            channel=env.channel, trace_id=TRACE, rowid=1, text="the user asked",
            annotations={})
        draft = "Tuesday works for me; I will book the room."
        assert await _send(env, draft, annotations={"origin": "user"}) is None
        # First the appraisal fork with critique's before-send offer, then
        # critique's lens fork on the draft.
        scoring, lens = env.agent.calls
        assert "- predictive:" in scoring["instruction"]
        assert "- retention:" not in scoring["instruction"]
        assert "[Self-critique: predictive]" in lens["instruction"]
        assert draft in lens["instruction"]
    finally:
        await critique.on_stop()
        await _teardown(env)


async def test_final_draft_is_scored_by_one_fork_and_the_hook_hears_it(tmp_path):
    """Item 5: a final draft makes appraisal fork once under attribution
    "appraisal", the instruction holding the whole draft (flattened, not
    truncated) and asking for the lenses only; on_appraisal_before_send fires
    once with the parsed vector and the gate returns None."""
    env, spy = await _before_send_env(tmp_path)
    try:
        # Long enough to pass any default quote length, with a line break
        # that the quote must flatten to one line.
        draft = ("The meeting is on Tuesday.\nBring the notes. " * 40).strip()
        result = await _send(env, draft, annotations={"origin": "user"})
        assert result is None
        # One fork, on the appraisal stage, quoting the whole draft.
        [call] = env.agent.calls
        assert call["stage"] == "appraisal"
        assert call["attribution"].get("stage") == "appraisal"
        assert call["attribution"].get("trace_id") == TRACE
        assert f"«{quote(draft, len(draft))}»" in call["instruction"]
        assert "- alpha: Did alpha?" in call["instruction"]
        # Only the lenses are asked for: no past_claims in the schema.
        schema = call["extra_body"]["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]) == {"lenses"}
        # The peers are asked for the before-send offer, and told the vector.
        assert spy.prompt_triggers == ["before_send"]
        [told] = spy.before_send
        assert told["trace_id"] == TRACE
        assert told["text"] == draft
        assert told["annotations"] == {"origin": "user"}
        assert told["lenses"] == SCORED_VECTOR
    finally:
        await _teardown(env)


@pytest.mark.parametrize("text, emission, trace_id", [
    ("", "final", TRACE),                       # an empty draft is a decline
    ("   ", "final", TRACE),                    # so is whitespace only
    ("NO_REPLY", "final", TRACE),               # the no-reply marker
    ("hmm NO_REPLY", "final", TRACE),           # a draft carrying the marker
    ("a real draft", "error", TRACE),           # not a final emission
    ("a real draft", "final", None),            # no trace to attribute to
], ids=["empty", "blank", "marker", "marker-inside", "non-final", "no-trace"])
async def test_decline_non_final_or_untraced_draft_forks_and_fires_nothing(
    tmp_path, text, emission, trace_id,
):
    """Item 6: none of these is a scored draft, so no fork runs and no hook
    fires; the gate gives no opinion."""
    env, spy = await _before_send_env(tmp_path)
    try:
        assert await _send(env, text, emission=emission, trace_id=trace_id) is None
        assert env.agent.calls == []
        assert spy.before_send == []
    finally:
        await _teardown(env)


@pytest.mark.parametrize("reason", ["unusable reply"])
async def test_fork_miss_fires_the_hook_with_no_vector_and_warns(
    tmp_path, caplog, reason,
):
    """Item 7: a before-send miss (a reply that is not JSON) still fires the
    hook, once, with lenses=None, and logs a warning naming the trace and the
    point. The "forks off" case is gone (#1149 R10): forks off for the stage
    is stated once at startup and logs no WARNING per final
    (tests/cognition/test_presend_hooks.py)."""
    env, spy = await _before_send_env(tmp_path, chat_text="not json at all")
    try:
        if reason == "forks off":
            env.agent.stages = set()
        with caplog.at_level(logging.WARNING, logger="corvidae.appraisal"):
            assert await _send(env, "a real draft") is None
        [told] = spy.before_send
        assert told["lenses"] is None
        assert told["text"] == "a real draft"
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            TRACE in w and "before" in w.lower() and "send" in w.lower()
            for w in warnings), warnings
    finally:
        await _teardown(env)


async def test_before_send_fork_prompt_override_replaces_the_default(tmp_path):
    """Item 10: appraisal.before_send_fork_prompt replaces the default
    instruction; with the key unset the default is used."""
    from corvidae_cognition.appraisal import DEFAULT_BEFORE_SEND_FORK_PROMPT

    env, _spy = await _before_send_env(tmp_path)
    try:
        # Default first: the shipped text leads the instruction.
        await _send(env, "a real draft")
        assert DEFAULT_BEFORE_SEND_FORK_PROMPT in env.agent.calls[-1]["instruction"]
        # Now the override, applied the way on_init resolves static config.
        await env.appraisal.on_init(pm=env.pm, config={
            "daemon": {"session_db": env.db_path},
            "appraisal": {"before_send_fork_prompt": "OPERATOR-BS-PROMPT-1169"},
        })
        await _send(env, "a real draft")
        instruction = env.agent.calls[-1]["instruction"]
        assert "OPERATOR-BS-PROMPT-1169" in instruction
        assert DEFAULT_BEFORE_SEND_FORK_PROMPT not in instruction
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# Item 13: forks off at startup
# ---------------------------------------------------------------------------


async def test_on_start_warns_once_when_forks_are_off_for_appraisal(tmp_path, caplog):
    """Item 13: appraisal registered but with forks off for its stage warns
    once at on_start, naming agent.forks.enabled and agent.forks.stages."""
    env = await build_env(tmp_path)
    try:
        env.agent.stages = set()
        with caplog.at_level(logging.WARNING, logger="corvidae.appraisal"):
            await env.appraisal.on_start(config={})
        [text] = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "agent.forks" in r.getMessage()]
        assert "agent.forks.enabled" in text and "agent.forks.stages" in text
    finally:
        await _teardown(env)


async def test_on_start_is_quiet_when_forks_are_on_for_appraisal(tmp_path, caplog):
    """Item 13 (the other side): with forks on for appraisal there is no
    startup warning about them."""
    env = await build_env(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="corvidae.appraisal"):
            await env.appraisal.on_start(config={})
        assert not [r for r in caplog.records if "agent.forks" in r.getMessage()]
    finally:
        await _teardown(env)
