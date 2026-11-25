"""Critique lenses as forks of the window (GitHub issue #51, phase 5):
each lens forks while the exchange is in the window (checked per lens),
the provenance lens keeps its evidence inline, and a lens whose
fork misses skips alone with a warning and no standalone call (#320, PRD
§3.8.5); a run where every lens missed records no outcome."""

import json
import logging

import pytest
from unittest.mock import AsyncMock

from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS
from corvidae.fork import ForkRefused, quote

from corvidae.context import MessageType
from helpers import FakeForkAgent, window_of
from .test_critique import (
    CLAIM, CUE_FREE_REPLY, EMPTY_VERDICT, SUPPORTED_VERDICT, SYSTEM_PROMPT, TOOL_TEXT, UNSUPPORTED_VERDICT,
    FakeAppraisal, RenderingForkAgent, verdicts_text, _run_next_task,
    _provenance_env, _provenance_exchange, _provenance_instructions, _read_outcomes,
    _run_provenance, _teardown, build_critique,
)

FORK_VERDICT = json.dumps({"objections": [{"claim": "it was Tuesday", "objection": "you said Monday"}]})
QUESTION, RESPONSE = "when is the meeting?", "It was Tuesday, as you said."


async def _run(tmp_path, agent, lenses, window_rowids=(7, 8), rowid=7, config=None,
               overrides=None):
    """Run the idle critique for exchange "k" the way the daemon does (the
    reply, then appraisal's stage-2 hook with a vector selecting ``lenses``,
    then the task critique queued); returns (env, the built-in lenses'
    outcome record, or None when no lens ran).

    Driven through the hooks because the one fork per point replaces the
    private per-lens runners the old version of this helper called (#1190 PR 2).
    """
    env = await build_critique(
        tmp_path, FakeAppraisal(), chat_text=EMPTY_VERDICT, config=config, fork=False)
    env.pm.register(agent, name="agent")
    env.channel.conversation = window_of(env.channel.id, list(window_rowids))
    env.channel.runtime_overrides.update(overrides or {})
    await env.plugin.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=rowid, text=QUESTION, annotations={})
    await env.plugin.on_agent_response(
        channel=env.channel, request_text=QUESTION, response_text=RESPONSE,
        trace_id="k", annotations={}, logprobs=None)
    # Every named lens beats "none" (absent, share 0), so all are selected.
    await env.plugin.on_appraisal_idle(
        channel=env.channel, trace_id="k", response_text=RESPONSE, annotations={},
        lenses={name: 1.0 for name in lenses}, past_claims=[])
    while not env.task.task_queue.queue.empty():
        await _run_next_task(env.task)
    # The built-ins' record; None when no lens ran (nothing is recorded for an
    # all-miss run). Other records (the selection, an added lens's) are skipped.
    outcome = next(
        (r["record"]["critique"] for r in env.recorder.by_source("critique")
         if r["trace_id"] == "k" and "lenses" in r["record"].get("critique", {})),
        None)
    await _teardown(env)
    return env, outcome


async def test_one_fork_covers_every_selected_lens(tmp_path):
    """R1 (replaces test_each_lens_forks): two selected lenses are exactly one
    fork, whose instruction holds each lens's section, brief and the exchange;
    the objections of both land in the one record."""
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT)
    env, outcome = await _run(tmp_path, agent, ["predictive", "constrained"])
    assert env.stub.chat_calls == []
    [call] = agent.calls
    instruction = call["instruction"]
    # One section per lens, in selection order, each carrying its own brief.
    sections = [instruction.index(f"## Lens: {lens}\n") for lens in ("predictive", "constrained")]
    assert sections == sorted(sections)
    for lens in ("predictive", "constrained"):
        assert DEFAULT_FORK_PROMPTS[lens] in instruction
    assert f"«{QUESTION}»" in instruction and f"«{RESPONSE}»" in instruction
    assert (call["attribution"]["stage"], call["attribution"]["trace_id"]) == ("critique", "k")
    assert outcome["lenses"] == ["predictive", "constrained"]
    assert outcome["objections"] == 2


async def test_stylistic_outcome_carries_objection_detail(tmp_path):
    """The envelope stores the objection objects, not just their count."""
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT)
    env, outcome = await _run(tmp_path, agent, ["predictive"])
    assert outcome["objections"] == 1
    assert outcome["detail"] == [
        {"claim": "it was Tuesday", "objection": "you said Monday",
         "lens": "predictive"}]


async def test_empty_stylistic_outcome_has_no_detail(tmp_path):
    """The detail key exists only when there is something to store."""
    agent = FakeForkAgent({"critique"}, text='{"objections": []}')
    env, outcome = await _run(tmp_path, agent, ["predictive"])
    assert outcome["objections"] == 0
    assert "detail" not in outcome


def _warnings(caplog) -> list[str]:
    """Messages of the WARNING records the critique plugin logged."""
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "corvidae.critique"]


@pytest.mark.parametrize("agent,window_rowids,rowid,reason", [
    # The reply row has no id (never persisted): idle contains the fork at the
    # reply (I3), so there is nothing to cut at.
    (FakeForkAgent({"critique"}, text=FORK_VERDICT), (7, None), 7, "not recorded"),
    (FakeForkAgent(set(), text=FORK_VERDICT), (7, 8), 7, "forks are not enabled for critique"),
    (FakeForkAgent({"critique"}, text=""), (7, 8), 7, "ValueError"),  # empty reply
    (FakeForkAgent({"critique"}, text="I think it's fine."), (7, 8), 7, "ValueError"),
    # A plain backend error, not a fork refusal: the HTTP 400 in #360 (#377).
    (FakeForkAgent({"critique"}, error=RuntimeError("400 model is not loaded")),
     (7, 8), 7, "RuntimeError"),
])
async def test_fork_miss_skips(tmp_path, caplog, agent, window_rowids, rowid, reason):
    """#320: a miss makes no background-role call, records no outcome, and
    logs a warning naming the lens, trace, channel and what was missing."""
    with caplog.at_level(logging.WARNING, logger="corvidae.critique"):
        env, outcome = await _run(tmp_path, agent, ["constrained"], window_rowids, rowid)
    # No standalone call and nothing recorded (no lens ran).
    assert env.stub.chat_calls == []
    assert outcome is None
    # Exactly one skip warning with the identifiers and the reason.
    [text] = [w for w in _warnings(caplog) if "skipped" in w]
    assert "trace k " in text and env.channel.id in text and reason in text
    # Whatever the miss, the warning names the lens that did not run.
    assert "constrained" in text


@pytest.mark.parametrize("error,kind", [
    (ForkRefused("tool call"), "ForkRefused"), (RuntimeError("down"), "RuntimeError"),
])
async def test_failed_fork_skips_every_lens_in_it(tmp_path, caplog, error, kind):
    """#320 as the one fork changes it (was test_failed_lens_skips_alone): the
    selected lenses share one fork, so a failing fork skips each of them with
    a warning naming the lens, trace and cause, makes no standalone call and
    records no outcome."""
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT, error=error)
    with caplog.at_level(logging.WARNING, logger="corvidae.critique"):
        env, outcome = await _run(tmp_path, agent, ["predictive", "adversarial"])
    assert env.stub.chat_calls == []
    assert outcome is None
    texts = [w for w in _warnings(caplog) if "skipped" in w]
    assert len(texts) == 2
    for lens, text in zip(["predictive", "adversarial"], texts):
        assert lens in text and kind in text and "trace k " in text


async def test_lens_missing_from_the_reply_skips_alone(tmp_path, caplog):
    """The reply has an entry for predictive and none for adversarial: that
    lens is skipped with a warning naming it; predictive still counts."""
    agent = FakeForkAgent({"critique"}, text=json.dumps({"predictive": json.loads(FORK_VERDICT)}))
    with caplog.at_level(logging.WARNING, logger="corvidae.critique"):
        env, outcome = await _run(tmp_path, agent, ["predictive", "adversarial"])
    assert outcome["lenses"] == ["predictive"]
    assert outcome["objections"] == 1
    [text] = [w for w in _warnings(caplog) if "skipped" in w]
    assert "adversarial" in text and "trace k " in text and env.channel.id in text


async def test_all_lenses_missing_records_no_outcome(tmp_path):
    """#320: when the fork's reply is unusable there is no critique outcome row."""
    agent = FakeForkAgent({"critique"}, text="not json")
    env, outcome = await _run(tmp_path, agent, ["predictive", "constrained"])
    assert env.stub.chat_calls == []
    assert len(agent.calls) == 1
    assert outcome is None


async def test_fork_prompt_is_static_config(tmp_path):
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT)
    await _run(tmp_path, agent, ["predictive"],
               config={"critique": {"lens": {"predictive": {"fork_prompt": "Find the misreading."}}}},
               overrides={"critique.lens.predictive.fork_prompt": "Never object."})
    # The operator's brief sits in the lens's section; the runtime override does not.
    assert "## Lens: predictive\nFind the misreading." in agent.calls[0]["instruction"]
    assert "Never object." not in agent.calls[0]["instruction"]


# ---------------------------------------------------------------------------
# The provenance check as a fork (#164)
# ---------------------------------------------------------------------------

SKIPPED = "critique provenance check skipped"


async def _provenance_fork_env(tmp_path, agent, **kwargs):
    env = await _provenance_env(tmp_path)
    ex = await _provenance_exchange(env, agent, **kwargs)
    return env, ex


async def test_provenance_fork_request_starts_with_the_system_prompt(tmp_path):
    agent = RenderingForkAgent({"critique"}, text=SUPPORTED_VERDICT)
    env, _ = await _provenance_fork_env(tmp_path, agent)
    try:
        await _run_provenance(env)
        [request] = agent.requests
        assert request[0]["role"] == "system"
        assert SYSTEM_PROMPT in request[0]["content"]
        [instruction] = _provenance_instructions(agent)
        assert "your system prompt" in instruction.lower()
    finally:
        await _teardown(env)


async def test_provenance_fork_request_has_the_exchanges_tool_result(tmp_path):
    agent = RenderingForkAgent({"critique"}, text=SUPPORTED_VERDICT)
    env, ex = await _provenance_fork_env(tmp_path, agent)
    try:
        await _run_provenance(env)
        [request] = agent.requests
        assert any(m.get("role") == "tool" and m["content"] == TOOL_TEXT for m in request)
        [instruction] = _provenance_instructions(agent)
        assert "tool results from this exchange" in instruction.lower()
    finally:
        await _teardown(env)


async def test_provenance_fork_ends_at_the_reply(tmp_path):
    agent = RenderingForkAgent({"critique"}, text=SUPPORTED_VERDICT)
    env, ex = await _provenance_fork_env(tmp_path, agent)
    try:
        ex.conv.messages.append({
            "role": "system", "content": "UNSENT-MARKER-XYZ",
            "_message_type": MessageType.CONTEXT})
        ex.conv.messages.append({
            "role": "user", "content": "LATER-USER-MESSAGE", "_db_id": 10_000,
            "_message_type": MessageType.MESSAGE})
        await _run_provenance(env)
        [call] = agent.calls
        reply_index = next(i for i, m in enumerate(ex.conv.messages)
                           if m.get("_db_id") == ex.reply_row)
        assert call["cut"] == reply_index + 1
        [request] = agent.requests
        text = json.dumps(request)
        assert "UNSENT-MARKER-XYZ" not in text and "LATER-USER-MESSAGE" not in text
        assert request[-2]["content"] == _response_of(ex)
    finally:
        await _teardown(env)


def _response_of(ex):
    return next(m["content"] for m in ex.conv.messages if m.get("_db_id") == ex.reply_row)


async def _assert_skipped(env, agent, caplog, reason, fork_calls=0):
    records = [r for r in caplog.records if SKIPPED in r.getMessage()]
    assert len(records) == 1, f"expected one {SKIPPED!r} warning, got {len(records)}"
    [record] = records
    message = record.getMessage()
    assert record.levelname == "WARNING" and "kp" in message and reason in message
    assert len(agent.calls) == fork_calls
    assert env.stub.chat_calls == []
    assert "provenance" not in (await _read_outcomes(env.recorder, "kp") or {})


def _forks_off(env, ex, agent):
    """Nothing to change: the agent is built with no fork stages."""


def _no_window(env, ex, agent):
    env.channel.conversation = None


def _reply_not_in_window(env, ex, agent):
    ex.conv.messages = [m for m in ex.conv.messages if m.get("_db_id") != ex.reply_row]
    ex.conv.messages.append({"role": "assistant", "content": "x", "_db_id": 9_999,
                             "_message_type": MessageType.MESSAGE})


@pytest.mark.parametrize("mutate,agent_stages,reason", [
    (_forks_off, set(), "forks are not enabled for critique"),
    (_no_window, {"critique"}, "no live window"),
    (_reply_not_in_window, {"critique"}, "the reply is not in the window"),
])
async def test_provenance_miss_skips_with_a_warning(tmp_path, caplog, mutate, agent_stages, reason):
    agent = RenderingForkAgent(agent_stages, text=UNSUPPORTED_VERDICT)
    env, ex = await _provenance_fork_env(tmp_path, agent)
    try:
        mutate(env, ex, agent)
        with caplog.at_level("WARNING", logger="corvidae.critique"):
            await _run_provenance(env)
        await _assert_skipped(env, agent, caplog, reason)
    finally:
        await _teardown(env)


async def test_provenance_skips_when_the_reply_was_not_persisted(tmp_path, caplog):
    agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
    env, ex = await _provenance_fork_env(tmp_path, agent, persist_reply=False)
    try:
        with caplog.at_level("WARNING", logger="corvidae.critique"):
            await _run_provenance(env)
        await _assert_skipped(env, agent, caplog, "the reply was not recorded or not persisted")
    finally:
        await _teardown(env)


async def test_provenance_skips_when_the_window_was_compacted_after_the_reply(tmp_path, caplog):
    agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
    env, ex = await _provenance_fork_env(tmp_path, agent)
    try:
        # before_compaction fires ahead of replace_with_summary; on_compaction
        # has not run, and the reply is still in the window. Fired through the
        # hook relay, as CompactionPlugin does, so a missing hookimpl fails.
        await env.pm.ahook.before_compaction(
            channel=env.channel, compacted_ids=[ex.origin], cut=1)
        with caplog.at_level("WARNING", logger="corvidae.critique"):
            await _run_provenance(env)
        await _assert_skipped(env, agent, caplog, "the window was compacted after the reply")
    finally:
        await _teardown(env)


async def test_provenance_skips_when_compaction_lands_during_the_check(tmp_path, caplog):
    # The compaction begins while the evidence search is awaiting, after the
    # pre-check passed: the check must look again before it forks.
    agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT)
    env, ex = await _provenance_fork_env(tmp_path, agent)
    gather = env.plugin._gather_evidence

    async def gather_then_compact(*args, **kwargs):
        evidence = await gather(*args, **kwargs)
        await env.pm.ahook.before_compaction(
            channel=env.channel, compacted_ids=[ex.origin], cut=1)
        return evidence

    env.plugin._gather_evidence = gather_then_compact
    try:
        with caplog.at_level("WARNING", logger="corvidae.critique"):
            await _run_provenance(env)
        await _assert_skipped(env, agent, caplog, "the window was compacted after the reply")
    finally:
        await _teardown(env)


@pytest.mark.parametrize("error", [ForkRefused("tool call"), RuntimeError("down")])
async def test_provenance_fork_failure_skips_without_a_standalone_call(tmp_path, caplog, error):
    agent = RenderingForkAgent({"critique"}, text=UNSUPPORTED_VERDICT, error=error)
    env, ex = await _provenance_fork_env(tmp_path, agent)
    try:
        with caplog.at_level("WARNING", logger="corvidae.critique"):
            await _run_provenance(env)
        await _assert_skipped(env, agent, caplog, "the fork failed", fork_calls=1)
    finally:
        await _teardown(env)


# ---------------------------------------------------------------------------
# Per-claim provenance verdicts (#186)
# ---------------------------------------------------------------------------

def _claims(n):
    return [{"claim": f"claim number {i + 1} is true", "queries": [f"claim{i + 1}"]}
            for i in range(n)]


def _held_entries(env):
    from corvidae_cognition.critique import PROVENANCE_NOTE

    env.held.hold_and_wake.assert_awaited_once()
    # Other waking lenses' sentences (proactivity's, #1126) follow the note.
    assert env.held.hold_and_wake.await_args.kwargs["stub_text"].startswith(PROVENANCE_NOTE)
    return env.held.hold_and_wake.await_args.kwargs["entries"]


async def _verdict_run(tmp_path, text, n=3):
    agent = RenderingForkAgent({"critique"}, text=text)
    env = await _provenance_env(tmp_path, with_held=True)
    env.held.hold_and_wake = AsyncMock()
    await _provenance_exchange(env, agent)
    await _run_provenance(env, claims=_claims(n))
    return env, agent


def test_provenance_extra_body_is_a_fixed_length_tuple_schema():
    from corvidae_cognition.critique import provenance_extra_body

    body = provenance_extra_body(3)
    assert list(body) == ["response_format"]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "provenance_verdicts"
    schema = fmt["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["verdicts"]
    assert schema["additionalProperties"] is False
    array = schema["properties"]["verdicts"]
    assert array["type"] == "array"
    assert array["minItems"] == array["maxItems"] == 3
    assert "items" not in array
    assert len(array["prefixItems"]) == 3
    for i, item in enumerate(array["prefixItems"]):
        assert item["type"] == "object"
        # The verdict is generated after its written reason.
        assert list(item["properties"]) == ["claim", "reason", "supported"]
        assert item["properties"] == {
            "claim": {"const": i + 1},
            "reason": {"type": "string"},
            "supported": {"type": "boolean"},
        }
        assert sorted(item["required"]) == ["claim", "reason", "supported"]
        assert item["additionalProperties"] is False


def test_provenance_extra_body_is_sized_per_request():
    from corvidae_cognition.critique import provenance_extra_body

    array = provenance_extra_body(1)["response_format"]["json_schema"]["schema"][
        "properties"]["verdicts"]
    assert array["minItems"] == array["maxItems"] == 1
    assert len(array["prefixItems"]) == 1


@pytest.mark.parametrize("n", [1, 3])
async def test_provenance_fork_carries_the_schema_sized_to_the_claims(tmp_path, n):
    from corvidae_cognition.critique import provenance_extra_body

    env, agent = await _verdict_run(tmp_path, verdicts_text(*[True] * n), n=n)
    try:
        [call] = agent.calls
        assert call["extra_body"] == provenance_extra_body(n)
    finally:
        await _teardown(env)


async def test_one_unsupported_claim_of_three_is_one_objection(tmp_path):
    env, agent = await _verdict_run(
        tmp_path, json.dumps({"verdicts": [
            {"claim": 1, "reason": "seen in the tool result", "supported": True},
            {"claim": 2, "reason": "nothing says so", "supported": False},
            {"claim": 3, "reason": "the user said it", "supported": True}]}))
    try:
        # R5.4: the finding names the reply it judges, then reads as before.
        assert _held_entries(env) == [
            f"on your reply \u00ab{quote(CUE_FREE_REPLY)}\u00bb: "
            f"[provenance] {_claims(3)[1]['claim']} \u2014 nothing says so"]
        outcomes = await _read_outcomes(env.recorder, "kp")
        assert outcomes["provenance"] == {
            "claims": 3, "candidates": 0, "supported": 2, "objections": 1,
            "detail": [{"claim": _claims(3)[1]["claim"],
                        "objection": "nothing says so",
                        "lens": "provenance"}]}
    finally:
        await _teardown(env)


def test_objection_quotes_our_claim_not_a_model_echo():
    from corvidae_cognition.critique import CritiquePlugin

    objections, supported = CritiquePlugin._verdicts_from_text(
        verdicts_text(False), [{"claim": "the real claim text"}])
    assert supported == 0
    assert objections == [
        {"claim": "the real claim text", "objection": "reason 1", "lens": "provenance"}]


async def test_all_supported_records_the_outcome_and_registers_nothing(tmp_path):
    env, agent = await _verdict_run(tmp_path, verdicts_text(True, True, True))
    try:
        env.held.hold_and_wake.assert_not_awaited()
        outcomes = await _read_outcomes(env.recorder, "kp")
        assert outcomes["provenance"] == {
            "claims": 3, "candidates": 0, "supported": 3, "objections": 0}
    finally:
        await _teardown(env)


async def test_unsupported_provenance_outcome_carries_detail(tmp_path):
    """The stored objection quotes the recorded claim, not the model's echo."""
    env, agent = await _verdict_run(
        tmp_path, _verdicts(_v(1), _v(2, reason="nothing says so",
                              supported=False)), n=2)
    try:
        outcomes = await _read_outcomes(env.recorder, "kp")
        assert outcomes["provenance"]["objections"] == 1
        assert outcomes["provenance"]["detail"] == [
            {"claim": _claims(2)[1]["claim"], "objection": "nothing says so",
             "lens": "provenance"}]
    finally:
        await _teardown(env)


def _verdicts(*items):
    return json.dumps({"verdicts": list(items)})


def _v(claim, reason="r", supported=True):
    return {"claim": claim, "reason": reason, "supported": supported}


@pytest.mark.parametrize("text", [
    json.dumps({"objections": []}),
    json.dumps({"verdicts": []}),
    _verdicts(_v(1), _v(2)),
    _verdicts(_v(1), _v(2), _v(3), _v(4)),
    _verdicts(_v(2), _v(1), _v(3)),
    _verdicts(_v(1), _v(2), _v(3, supported="yes")),
    _verdicts(_v(1), _v(2), _v(3, supported=1)),
    _verdicts(_v(True), _v(2), _v(3)),
    _verdicts(_v(1), _v(2), _v(3, reason=None)),
], ids=["old-format", "empty", "two-for-three", "four-for-three", "wrong-order",
        "supported-not-bool", "supported-is-int", "claim-is-bool", "reason-not-str"])
async def test_malformed_verdicts_skip_the_check(tmp_path, caplog, text):
    agent = RenderingForkAgent({"critique"}, text=text)
    env = await _provenance_env(tmp_path, with_held=True)
    env.held.hold_and_wake = AsyncMock()
    try:
        await _provenance_exchange(env, agent)
        with caplog.at_level("WARNING", logger="corvidae.critique"):
            await _run_provenance(env, claims=_claims(3))
        await _assert_skipped(env, agent, caplog, "the fork failed", fork_calls=1)
        env.held.hold_and_wake.assert_not_awaited()
    finally:
        await _teardown(env)


def test_provenance_instruction_rules_out_tools_and_replying_to_the_user():
    prompt = DEFAULT_FORK_PROMPTS["provenance"]
    assert "can't call tools" in prompt
    assert "not a reply to the user" in prompt


def test_critique_registers_its_llm_stage_as_chat():
    """R1a: critique owns the ``critique`` stage, a chat stage."""
    from corvidae_cognition.critique import CritiquePlugin
    from helpers import collect_llm_stages

    assert collect_llm_stages(CritiquePlugin(), "critique") == {"critique": "chat"}
