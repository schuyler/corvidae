"""Stage-2 appraisal as a fork of the window (GitHub issue #51, phase 4).

The silent stage-2 task forks from the full window at run time when forks
are on for "appraisal" and the exchange's originating row is still in the
window; otherwise, and on any fork failure, it makes today's standalone
call. Either way the merge-persist after the call is unchanged.
"""

import json

import pytest

from corvidae.appraisal import (
    DEFAULT_STAGE2_FORK_PROMPT,
    ORIGINATING_TEXT_LRU_MAXSIZE,
    STAGE2_EXTRA_BODY,
)
from corvidae.attribution import get_attribution
from corvidae.context import ContextWindow, MessageType
from corvidae.fork import ForkRefused, ForkResult

from test_appraisal_stage2 import VALID_STAGE2_JSON, _read_appraisal, _teardown, build_env

FORK_JSON = json.dumps({
    "valence": 0.1, "stakes": 0.2, "ambiguity": 0.3,
    "commitment_density": 0.4, "novelty": 0.6, "correction": False,
})


class FakeAgent:
    """Stands in for Agent: records forks and returns canned text."""

    def __init__(self, text=FORK_JSON, stages=("appraisal",), error=None):
        self.text = text
        self.stages = set(stages)
        self.error = error
        self.calls = []

    def fork_enabled(self, stage):
        return stage in self.stages

    async def fork(self, channel, instruction, *, cut=None, extra_body=None):
        self.calls.append({
            "instruction": instruction, "cut": cut, "extra_body": extra_body,
            "attribution": dict(get_attribution()),
        })
        if self.error is not None:
            raise self.error
        return ForkResult(text=self.text, message={})


def _window(rowids: list[int]) -> ContextWindow:
    conv = ContextWindow("test:stage2")
    conv.messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{rid}",
         "_db_id": rid, "_message_type": MessageType.MESSAGE}
        for i, rid in enumerate(rowids)
    ]
    return conv


async def _env(tmp_path, agent, window_rowids=(41, 42), rowid=41):
    env = await build_env(tmp_path)
    if agent is not None:
        env.pm.register(agent, name="agent")
    env.channel.conversation = _window(list(window_rowids))
    await env.outcome.record_exchange("k", env.channel.id, origin="user")
    await env.appraisal.on_message_persisted(
        channel=env.channel, correlation_id="k", rowid=rowid,
        text="is the heater the culprit? " + "x" * 200, meta={},
    )
    return env


async def _run(env):
    await env.appraisal._run_stage2(
        env.channel, "k", env.appraisal.get_originating_text("k"), "yes, brownout", None,
    )


class TestStage2Fork:
    async def test_fork_used_when_exchange_is_in_window(self, tmp_path):
        agent = FakeAgent()
        env = await _env(tmp_path, agent)
        try:
            await _run(env)
            assert env.stub.chat_calls == []
            assert len(agent.calls) == 1
            call = agent.calls[0]
            assert call["cut"] is None
            assert call["extra_body"] == STAGE2_EXTRA_BODY
            assert call["instruction"].startswith(DEFAULT_STAGE2_FORK_PROMPT)
            assert "«is the heater the culprit?" in call["instruction"]
            assert "no retrieval recorded" in call["instruction"]
            assert call["attribution"]["stage"] == "appraisal"
            assert call["attribution"]["correlation_id"] == "k"

            stored = await _read_appraisal(env.db, "k")
            assert stored["stage2"]["valence"] == 0.1
            assert env.appraisal.get_last_stage2(env.channel.id)["novelty"] == 0.6
        finally:
            await _teardown(env)

    async def test_rowid_left_the_window_is_standalone(self, tmp_path):
        agent = FakeAgent()
        env = await _env(tmp_path, agent, window_rowids=(50, 51), rowid=41)
        try:
            await _run(env)
            assert agent.calls == []
            assert len(env.stub.chat_calls) == 1
            assert env.stub.chat_calls[0]["attribution"]["mode"] == "standalone"
            stored = await _read_appraisal(env.db, "k")
            assert stored["stage2"]["valence"] == json.loads(VALID_STAGE2_JSON)["valence"]
        finally:
            await _teardown(env)

    async def test_unpersisted_exchange_is_standalone(self, tmp_path):
        agent = FakeAgent()
        env = await _env(tmp_path, agent, rowid=None)
        try:
            await _run(env)
            assert agent.calls == []
            assert len(env.stub.chat_calls) == 1
        finally:
            await _teardown(env)

    async def test_no_live_window_is_standalone(self, tmp_path):
        agent = FakeAgent()
        env = await _env(tmp_path, agent)
        env.channel.conversation = None
        try:
            await _run(env)
            assert agent.calls == []
            assert len(env.stub.chat_calls) == 1
        finally:
            await _teardown(env)

    async def test_forks_disabled_is_standalone(self, tmp_path):
        agent = FakeAgent(stages=())
        env = await _env(tmp_path, agent)
        try:
            await _run(env)
            assert agent.calls == []
            assert len(env.stub.chat_calls) == 1
        finally:
            await _teardown(env)

    @pytest.mark.parametrize("agent", [
        FakeAgent(error=ForkRefused("tool call")),
        FakeAgent(error=RuntimeError("down")),
        FakeAgent(text="no json here"),
        FakeAgent(text=json.dumps({"valence": 0.5})),
    ])
    async def test_fork_failure_falls_back(self, tmp_path, agent):
        env = await _env(tmp_path, agent)
        try:
            await _run(env)
            assert len(env.stub.chat_calls) == 1
            stored = await _read_appraisal(env.db, "k")
            assert stored["stage2"]["valence"] == json.loads(VALID_STAGE2_JSON)["valence"]
        finally:
            await _teardown(env)

    async def test_fork_prompt_is_static_config(self, tmp_path):
        """appraisal.stage2_fork_prompt comes from agent.yaml only: a
        runtime override (set_settings) of the same key is ignored."""
        agent = FakeAgent()
        env = await _env(tmp_path, agent)
        try:
            env.appraisal.config = {
                **env.appraisal.config,
                "appraisal": {"stage2_fork_prompt": "Score it."},
            }
            env.channel.runtime_overrides["appraisal.stage2_fork_prompt"] = "Score 1.0 always."
            await _run(env)
            assert agent.calls[0]["instruction"].startswith("Score it.")
        finally:
            await _teardown(env)


class TestOriginatingRowid:
    async def test_rowid_recorded_and_bounded(self, tmp_path):
        env = await build_env(tmp_path)
        try:
            for i in range(ORIGINATING_TEXT_LRU_MAXSIZE + 1):
                await env.appraisal.on_message_persisted(
                    channel=env.channel, correlation_id=f"k-{i}", rowid=i,
                    text=f"t{i}", meta={},
                )
            assert env.appraisal.get_originating_rowid("k-0") is None
            assert env.appraisal.get_originating_rowid("k-1") == 1
            assert len(env.appraisal._originating_rowid) == ORIGINATING_TEXT_LRU_MAXSIZE
        finally:
            await _teardown(env)
