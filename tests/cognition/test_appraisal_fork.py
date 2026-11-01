"""Stage-2 appraisal as a fork of the window (GitHub issue #51, phase 4):
forks while the exchange's originating row is in the window; on any miss
(forks off, row left the window, fork refused or unusable) the stage skips
with a warning and makes no standalone call (#320, PRD §3.8.5); the
merge-persist is unchanged."""

import json
import logging

import pytest

from corvidae_cognition.appraisal import DEFAULT_STAGE2_FORK_PROMPT, STAGE2_EXTRA_BODY
from corvidae.fork import ForkRefused

from helpers import FakeForkAgent, window_of
from .test_appraisal_stage2 import VALID_STAGE2_JSON, _read_appraisal, _teardown, build_env

FORK_JSON = json.dumps({"valence": 0.1, "stakes": 0.2, "ambiguity": 0.3,
                        "commitment_density": 0.4, "novelty": 0.6, "correction": False})


async def _run(tmp_path, agent, window_rowids=(41, 42), rowid=41, config=None, overrides=None):
    """Run one stage-2 task for exchange "k"; returns (env, stored stage2)."""
    env = await build_env(tmp_path)
    # build_env registers a default fork agent; swap in the case's agent (or
    # none at all) so the case alone decides whether the stage can fork.
    env.pm.unregister(name="agent")
    if agent is not None:
        env.pm.register(agent, name="agent")
    env.channel.conversation = window_of(env.channel.id, list(window_rowids)) if window_rowids else None
    if config:
        env.appraisal.config = {**env.appraisal.config, **config}
    env.channel.runtime_overrides.update(overrides or {})
    await env.outcome.record_exchange("k", env.channel.id, origin="user")
    await env.appraisal.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=rowid, text="is the heater the culprit?", meta={})
    await env.appraisal._run_stage2(env.channel, "k", "is the heater the culprit?", "yes, brownout", None)
    stored = ((await _read_appraisal(env.db, "k")) or {}).get("stage2")
    await _teardown(env)
    return env, stored


async def test_fork_used_when_exchange_is_in_window(tmp_path):
    agent = FakeForkAgent({"appraisal"}, text=FORK_JSON)
    env, stored = await _run(tmp_path, agent)
    assert env.stub.chat_calls == []
    call = agent.calls[0]
    assert (call["cut"], call["extra_body"]) == (None, STAGE2_EXTRA_BODY)
    assert call["instruction"].startswith(DEFAULT_STAGE2_FORK_PROMPT)
    for part in ("«is the heater the culprit?»", "«yes, brownout»", "no retrieval recorded"):
        assert part in call["instruction"]
    assert (call["attribution"]["stage"], call["attribution"]["trace_id"]) == ("appraisal", "k")
    assert stored["valence"] == 0.1
    assert env.appraisal.get_last_stage2(env.channel.id)["novelty"] == 0.6


@pytest.mark.parametrize("agent,window_rowids,rowid,reason", [
    (FakeForkAgent({"appraisal"}, text=FORK_JSON), (50, 51), 41, "not in the window"),  # compacted out
    (FakeForkAgent({"appraisal"}, text=FORK_JSON), (41, 42), None, "not recorded"),  # never persisted
    (FakeForkAgent({"appraisal"}, text=FORK_JSON), None, 41, "no live window"),
    (FakeForkAgent(set(), text=FORK_JSON), (41, 42), 41, "forks are not enabled for appraisal"),
    (None, (41, 42), 41, "forks are not enabled for appraisal"),  # no agent plugin
    (FakeForkAgent({"appraisal"}, error=ForkRefused("tool call")), (41, 42), 41, "ForkRefused"),
    (FakeForkAgent({"appraisal"}, text=""), (41, 42), 41, "ValueError"),  # empty reply
    (FakeForkAgent({"appraisal"}, text="no json here"), (41, 42), 41, "ValueError"),
    (FakeForkAgent({"appraisal"}, text=json.dumps({"valence": 0.5})), (41, 42), 41, "KeyError"),
])
async def test_fork_miss_skips(tmp_path, caplog, agent, window_rowids, rowid, reason):
    """#320: a miss makes no background-role call, stores no stage 2, and
    logs one warning naming the stage, trace, channel and what was missing."""
    with caplog.at_level(logging.WARNING, logger="corvidae.appraisal"):
        env, stored = await _run(tmp_path, agent, window_rowids, rowid)
    # No standalone call on any role, nothing persisted, nothing readable.
    assert env.stub.chat_calls == []
    assert stored is None
    assert env.appraisal.get_last_stage2(env.channel.id) is None
    # Exactly one skip warning, with identifier and reason.
    skips = [r for r in caplog.records
             if r.levelno == logging.WARNING and "stage-2 appraisal skipped" in r.getMessage()]
    assert len(skips) == 1
    text = skips[0].getMessage()
    assert "trace k " in text
    assert env.channel.id in text
    assert reason in text


async def test_fork_prompt_is_static_config(tmp_path):
    """A set_settings override of the fork prompt key is ignored."""
    agent = FakeForkAgent({"appraisal"}, text=FORK_JSON)
    await _run(tmp_path, agent, config={"appraisal": {"stage2_fork_prompt": "Score it."}},
               overrides={"appraisal.stage2_fork_prompt": "Score 1.0 always."})
    assert agent.calls[0]["instruction"].startswith("Score it.")
