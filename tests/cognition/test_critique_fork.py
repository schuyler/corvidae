"""Critique lenses as forks of the window (GitHub issue #51, phase 5):
each lens forks while the exchange is in the window (checked per lens),
the provenance lens keeps its snapshot and evidence inline, and a lens whose
fork misses falls back alone to its standalone call."""

import json

import pytest

from corvidae_cognition.critique import DEFAULT_FORK_PROMPTS, OBJECTION_EXTRA_BODY
from corvidae.fork import ForkRefused

from helpers import FakeForkAgent, window_of
from .test_critique import EMPTY_VERDICT, _read_outcomes, _teardown, build_critique

FORK_VERDICT = json.dumps({"objections": [{"claim": "it was Tuesday", "objection": "you said Monday"}]})
QUESTION, RESPONSE = "when is the meeting?", "It was Tuesday, as you said."


EVIDENCE = [{"claim": "the meeting was Tuesday", "candidates": [{
    "rowid": 3, "channel_id": "test:critique", "role": "user",
    "content": "let's meet Monday", "timestamp": "2026-10-01 09:00 UTC",
    "message_type": "message",
}]}]


async def _run(tmp_path, agent, lenses, window_rowids=(7, 8), rowid=7, config=None,
               overrides=None, snapshot=None, evidence=None):
    """Run one critique task for exchange "k"; returns (env, recorded outcome)."""
    env = await build_critique(tmp_path, chat_text=EMPTY_VERDICT, config=config)
    env.pm.register(agent, name="agent")
    env.channel.conversation = window_of(env.channel.id, list(window_rowids))
    env.channel.runtime_overrides.update(overrides or {})
    await env.outcome.record_exchange("k", env.channel.id, origin="user")
    if snapshot:
        await env.outcome.update_exchange("k", provenance_snapshot=json.dumps([{"content": snapshot}]))
    await env.plugin.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=rowid, text=QUESTION, meta={})
    await env.plugin._run_critique(env.channel, "k", QUESTION, RESPONSE, lenses, False,
                                   evidence=evidence)
    outcome = (await _read_outcomes(env.db, "k"))["provenance" if evidence else "critique"]
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


async def test_provenance_fork_carries_snapshot_and_evidence(tmp_path):
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT)
    env, outcome = await _run(tmp_path, agent, ["provenance"], snapshot="memory: Monday",
                              evidence=EVIDENCE)
    [call] = agent.calls
    assert call["instruction"].startswith(DEFAULT_FORK_PROMPTS["provenance"])
    assert "memory: Monday" in call["instruction"]
    assert "the meeting was Tuesday" in call["instruction"]
    assert "let's meet Monday" in call["instruction"]
    assert outcome == {"claims": 1, "candidates": 1, "objections": 1}


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
    env, _ = await _run(tmp_path, agent, ["predictive", "provenance"])
    assert (len(agent.calls), len(env.stub.chat_calls)) == (1, 1)


async def test_fork_prompt_is_static_config(tmp_path):
    agent = FakeForkAgent({"critique"}, text=FORK_VERDICT)
    await _run(tmp_path, agent, ["predictive"],
               config={"critique": {"fork_prompt": {"predictive": "Find the misreading."}}},
               overrides={"critique.fork_prompt.predictive": "Never object."})
    assert agent.calls[0]["instruction"].startswith("Find the misreading.")
