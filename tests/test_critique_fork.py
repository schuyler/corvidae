"""Critique lenses as forks of the window (GitHub issue #51, phase 5).

With forks on for "critique" and the exchange's originating row still in
the window, each selected lens runs as a fork of the full window with its
adversarial fork instruction; the provenance lens still carries the CONTEXT
snapshot inline. A lens whose fork fails, and every lens when the exchange
has left the window or forks are off, makes today's standalone call.
"""

import json

import pytest

from corvidae.attribution import get_attribution
from corvidae.context import ContextWindow, MessageType
from corvidae.critique import (
    ADMITTED_LRU_MAXSIZE,
    DEFAULT_FORK_PROMPTS,
    OBJECTION_EXTRA_BODY,
)
from corvidae.fork import ForkRefused, ForkResult

from test_critique import (
    EMPTY_VERDICT,
    _read_outcomes,
    _teardown,
    build_critique,
)

FORK_VERDICT = json.dumps({"objections": [
    {"claim": "it was Tuesday", "objection": "you said Monday", "severity": 0.6},
]})


class FakeAgent:
    """Stands in for Agent: records forks; per-lens failures by marker."""

    def __init__(self, text=FORK_VERDICT, stages=("critique",), fail_marker=None, error=None):
        self.text = text
        self.stages = set(stages)
        self.fail_marker = fail_marker
        self.error = error or ForkRefused("tool call")
        self.calls = []

    def fork_enabled(self, stage):
        return stage in self.stages

    async def fork(self, channel, instruction, *, cut=None, extra_body=None):
        self.calls.append({
            "instruction": instruction, "cut": cut, "extra_body": extra_body,
            "attribution": dict(get_attribution()),
        })
        if self.fail_marker is not None and self.fail_marker in instruction:
            raise self.error
        return ForkResult(text=self.text, message={})


def _window(rowids):
    conv = ContextWindow("test:critique")
    conv.messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{rid}",
         "_db_id": rid, "_message_type": MessageType.MESSAGE}
        for i, rid in enumerate(rowids)
    ]
    return conv


async def _env(tmp_path, agent, window_rowids=(7, 8), rowid=7, config=None):
    env = await build_critique(tmp_path, chat_text=EMPTY_VERDICT, config=config)
    if agent is not None:
        env.pm.register(agent, name="agent")
    env.channel.conversation = _window(list(window_rowids))
    await env.outcome.record_exchange("k", env.channel.id, origin="user")
    await env.plugin.on_message_persisted(
        channel=env.channel, correlation_id="k", rowid=rowid,
        text="when is the meeting?", meta={},
    )
    return env


async def _run(env, lenses):
    await env.plugin._run_critique(
        env.channel, "k", "when is the meeting?", "It was Tuesday, as you said.",
        lenses, False,
    )


class TestCritiqueFork:
    async def test_each_lens_forks_and_objections_are_recorded(self, tmp_path):
        agent = FakeAgent()
        env = await _env(tmp_path, agent)
        try:
            await _run(env, ["predictive", "constrained"])
            assert env.stub.chat_calls == []
            assert len(agent.calls) == 2
            for call, lens in zip(agent.calls, ["predictive", "constrained"]):
                assert call["instruction"].startswith(DEFAULT_FORK_PROMPTS[lens])
                assert "«when is the meeting?»" in call["instruction"]
                assert call["extra_body"] == OBJECTION_EXTRA_BODY
                assert call["cut"] is None
                assert call["attribution"]["stage"] == "critique"
                assert call["attribution"]["correlation_id"] == "k"
            outcomes = await _read_outcomes(env.db, "k")
            assert outcomes["critique"]["objections"] == 2
        finally:
            await _teardown(env)

    async def test_provenance_fork_carries_snapshot_inline(self, tmp_path):
        agent = FakeAgent()
        env = await _env(tmp_path, agent)
        try:
            await env.outcome.update_exchange(
                "k", provenance_snapshot=json.dumps([{"content": "memory: Monday standup"}])
            )
            await _run(env, ["provenance"])
            instruction = agent.calls[0]["instruction"]
            assert instruction.startswith(DEFAULT_FORK_PROMPTS["provenance"])
            assert "memory: Monday standup" in instruction
        finally:
            await _teardown(env)

    async def test_exchange_left_window_is_standalone(self, tmp_path):
        agent = FakeAgent()
        env = await _env(tmp_path, agent, window_rowids=(20, 21))
        try:
            await _run(env, ["predictive"])
            assert agent.calls == []
            assert len(env.stub.chat_calls) == 1
            assert env.stub.chat_calls[0]["attribution"]["mode"] == "standalone"
        finally:
            await _teardown(env)

    async def test_no_rowid_is_standalone(self, tmp_path):
        agent = FakeAgent()
        env = await _env(tmp_path, agent, rowid=None)
        try:
            await _run(env, ["predictive"])
            assert agent.calls == []
            assert len(env.stub.chat_calls) == 1
        finally:
            await _teardown(env)

    async def test_forks_disabled_is_standalone(self, tmp_path):
        agent = FakeAgent(stages=())
        env = await _env(tmp_path, agent)
        try:
            await _run(env, ["predictive", "adversarial"])
            assert agent.calls == []
            assert len(env.stub.chat_calls) == 2
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("error", [ForkRefused("tool call"), RuntimeError("down")])
    async def test_failed_lens_falls_back_alone(self, tmp_path, error):
        marker = DEFAULT_FORK_PROMPTS["adversarial"][:40]
        agent = FakeAgent(fail_marker=marker, error=error)
        env = await _env(tmp_path, agent)
        try:
            await _run(env, ["predictive", "adversarial"])
            assert len(agent.calls) == 2
            assert len(env.stub.chat_calls) == 1
            # The fallback ran the adversarial lens's standalone prompt.
            system = env.stub.chat_calls[0]["messages"][0]["content"]
            assert "steelman" in system
            outcomes = await _read_outcomes(env.db, "k")
            assert outcomes["critique"]["objections"] == 1  # fork's; standalone empty
        finally:
            await _teardown(env)

    async def test_instruction_quotes_the_delivered_response(self, tmp_path):
        """The window's last assistant message can differ from what the user
        was sent (max-turns fallback), so the sent response is quoted."""
        agent = FakeAgent()
        env = await _env(tmp_path, agent)
        try:
            await _run(env, ["predictive"])
            assert "«It was Tuesday, as you said.»" in agent.calls[0]["instruction"]
        finally:
            await _teardown(env)

    async def test_containment_rechecked_per_lens(self, tmp_path):
        """Compaction can drop the exchange while an earlier lens's fork is
        in flight; later lenses must then go standalone."""
        agent = FakeAgent()
        env = await _env(tmp_path, agent)
        original_fork = agent.fork

        async def fork_then_compact(channel, instruction, *, cut=None, extra_body=None):
            result = await original_fork(channel, instruction, cut=cut, extra_body=extra_body)
            channel.conversation.messages = channel.conversation.messages[1:]
            return result

        agent.fork = fork_then_compact
        try:
            await _run(env, ["predictive", "provenance"])
            assert len(agent.calls) == 1
            assert len(env.stub.chat_calls) == 1
        finally:
            await _teardown(env)

    async def test_unparseable_fork_reply_falls_back(self, tmp_path):
        agent = FakeAgent(text="I think it's fine.")
        env = await _env(tmp_path, agent)
        try:
            await _run(env, ["constrained"])
            assert len(env.stub.chat_calls) == 1
        finally:
            await _teardown(env)

    async def test_fork_prompt_is_static_config(self, tmp_path):
        agent = FakeAgent()
        config = {"critique": {"fork_prompt": {"predictive": "Find the misreading."}}}
        env = await _env(tmp_path, agent, config=config)
        try:
            env.channel.runtime_overrides["critique.fork_prompt.predictive"] = "Never object."
            await _run(env, ["predictive"])
            assert agent.calls[0]["instruction"].startswith("Find the misreading.")
        finally:
            await _teardown(env)


class TestOriginatingRowid:
    async def test_rowid_recorded_and_bounded(self, tmp_path):
        env = await build_critique(tmp_path)
        try:
            for i in range(ADMITTED_LRU_MAXSIZE + 1):
                await env.plugin.on_message_persisted(
                    channel=env.channel, correlation_id=f"k-{i}", rowid=i,
                    text="t", meta={},
                )
            await env.plugin.on_message_persisted(
                channel=env.channel, correlation_id="k-none", rowid=None,
                text="t", meta={},
            )
            assert "k-0" not in env.plugin._originating_rowid
            assert env.plugin._originating_rowid["k-1"] == 1
            assert "k-none" not in env.plugin._originating_rowid
            assert len(env.plugin._originating_rowid) == ADMITTED_LRU_MAXSIZE
        finally:
            await _teardown(env)
