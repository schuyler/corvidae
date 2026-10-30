"""Critique lenses as forks of the window (GitHub issue #51, phase 5):
each lens forks while the exchange is in the window (checked per lens),
the provenance lens keeps its snapshot and evidence inline, and a lens whose
fork misses falls back alone to its standalone call."""

import json

import pytest
from unittest.mock import AsyncMock

from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS, OBJECTION_EXTRA_BODY
from corvidae.fork import ForkRefused

from corvidae.context import MessageType
from helpers import FakeForkAgent, window_of
from .test_critique import (
    CLAIM, EMPTY_VERDICT, SUPPORTED_VERDICT, SYSTEM_PROMPT, TOOL_TEXT, UNSUPPORTED_VERDICT,
    RenderingForkAgent, verdicts_text,
    _provenance_env, _provenance_exchange, _provenance_instructions, _read_outcomes,
    _run_provenance, _teardown, build_critique,
)

FORK_VERDICT = json.dumps({"objections": [{"claim": "it was Tuesday", "objection": "you said Monday"}]})
QUESTION, RESPONSE = "when is the meeting?", "It was Tuesday, as you said."


async def _run(tmp_path, agent, lenses, window_rowids=(7, 8), rowid=7, config=None,
               overrides=None):
    """Run one critique task for exchange "k"; returns (env, recorded outcome)."""
    env = await build_critique(tmp_path, chat_text=EMPTY_VERDICT, config=config)
    env.pm.register(agent, name="agent")
    env.channel.conversation = window_of(env.channel.id, list(window_rowids))
    env.channel.runtime_overrides.update(overrides or {})
    await env.outcome.record_exchange("k", env.channel.id, origin="user")
    await env.plugin.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=rowid, text=QUESTION, meta={})
    await env.plugin._run_critique(env.channel, "k", QUESTION, RESPONSE, lenses, False)
    outcome = (await _read_outcomes(env.db, "k"))["critique"]
    await _teardown(env)
    return env, outcome


async def test_each_lens_forks(tmp_path):
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT)
    env, outcome = await _run(tmp_path, agent, ["predictive", "constrained"])
    assert env.stub.chat_calls == []
    for call, lens in zip(agent.calls, ["predictive", "constrained"]):
        assert call["instruction"].startswith(DEFAULT_FORK_PROMPTS[lens])
        assert f"«{QUESTION}»" in call["instruction"] and f"«{RESPONSE}»" in call["instruction"]
        assert (call["cut"], call["extra_body"]) == (None, OBJECTION_EXTRA_BODY)
        assert (call["attribution"]["stage"], call["attribution"]["trace_id"]) == ("critique", "k")
    assert outcome["objections"] == 2


@pytest.mark.parametrize("agent,window_rowids,rowid", [
    (FakeForkAgent({"critique"}, text=FORK_VERDICT), (20, 21), 7),  # compacted out
    (FakeForkAgent({"critique"}, text=FORK_VERDICT), (7, 8), None),  # never persisted
    (FakeForkAgent(set(), text=FORK_VERDICT), (7, 8), 7),  # forks off
    (FakeForkAgent({"critique"}, text="I think it's fine."), (7, 8), 7),
])
async def test_fork_miss_is_standalone(tmp_path, agent, window_rowids, rowid):
    env, _ = await _run(tmp_path, agent, ["constrained"], window_rowids, rowid)
    [call] = env.stub.chat_calls
    assert call["attribution"]["mode"] == "standalone"


@pytest.mark.parametrize("error", [ForkRefused("tool call"), RuntimeError("down")])
async def test_failed_lens_falls_back_alone(tmp_path, error):
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT, error=error,
                          fail_marker=DEFAULT_FORK_PROMPTS["adversarial"][:40])
    env, outcome = await _run(tmp_path, agent, ["predictive", "adversarial"])
    [call] = env.stub.chat_calls
    assert "steelman" in call["messages"][0]["content"]  # the adversarial standalone prompt
    assert outcome["objections"] == 1  # the fork's; the standalone verdict is empty


async def test_containment_rechecked_per_lens(tmp_path):
    """Compaction drops the exchange while the first lens's fork is in
    flight: later lenses go standalone."""
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT)
    fork = agent.fork

    async def fork_then_compact(channel, instruction, **kwargs):
        result = await fork(channel, instruction, **kwargs)
        channel.conversation.messages = channel.conversation.messages[1:]
        return result

    agent.fork = fork_then_compact
    env, _ = await _run(tmp_path, agent, ["predictive", "constrained"])
    assert (len(agent.calls), len(env.stub.chat_calls)) == (1, 1)


async def test_fork_prompt_is_static_config(tmp_path):
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT)
    await _run(tmp_path, agent, ["predictive"],
               config={"critique": {"fork_prompt": {"predictive": "Find the misreading."}}},
               overrides={"critique.fork_prompt.predictive": "Never object."})
    assert agent.calls[0]["instruction"].startswith("Find the misreading.")


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
    assert "provenance" not in (await _read_outcomes(env.db, "kp") or {})


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


def _funnel_entries(env):
    from corvidae_cognition.critique import PROVENANCE_NOTE

    env.funnel.register_and_wake.assert_awaited_once()
    assert env.funnel.register_and_wake.await_args.kwargs["stub_text"] == PROVENANCE_NOTE
    return env.funnel.register_and_wake.await_args.kwargs["entries"]


async def _verdict_run(tmp_path, text, n=3):
    agent = RenderingForkAgent({"critique"}, text=text)
    env = await _provenance_env(tmp_path, with_funnel=True)
    env.funnel.register_and_wake = AsyncMock()
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
        assert _funnel_entries(env) == [
            f"[provenance] {_claims(3)[1]['claim']} \u2014 nothing says so"]
        outcomes = await _read_outcomes(env.db, "kp")
        assert outcomes["provenance"] == {
            "claims": 3, "candidates": 0, "supported": 2, "objections": 1}
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
        env.funnel.register_and_wake.assert_not_awaited()
        outcomes = await _read_outcomes(env.db, "kp")
        assert outcomes["provenance"] == {
            "claims": 3, "candidates": 0, "supported": 3, "objections": 0}
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
    env = await _provenance_env(tmp_path, with_funnel=True)
    env.funnel.register_and_wake = AsyncMock()
    try:
        await _provenance_exchange(env, agent)
        with caplog.at_level("WARNING", logger="corvidae.critique"):
            await _run_provenance(env, claims=_claims(3))
        await _assert_skipped(env, agent, caplog, "the fork failed", fork_calls=1)
        env.funnel.register_and_wake.assert_not_awaited()
    finally:
        await _teardown(env)


def test_provenance_instruction_rules_out_tools_and_replying_to_the_user():
    prompt = DEFAULT_FORK_PROMPTS["provenance"]
    assert "can't call tools" in prompt
    assert "not a reply to the user" in prompt
