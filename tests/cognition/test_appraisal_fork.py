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

FORK_JSON = json.dumps({"past_claims": [{"claim": "the heater failed before", "queries": ["heater"]}]})


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
        new_config = {**env.appraisal.config, **config}
        env.appraisal.config = new_config
        await env.appraisal.on_config_reload(new_config)
    env.channel.runtime_overrides.update(overrides or {})
    await env.appraisal.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=rowid, text="is the heater the culprit?", annotations={})
    # The reply row, as on_agent_response records it before its first await.
    env.appraisal._record_reply(env.channel, "k")
    await env.appraisal._run_stage2(env.channel, "k", "is the heater the culprit?", "yes, brownout", None)
    stored = ((await _read_appraisal(env.recorder, "k")) or {}).get("stage2")
    await _teardown(env)
    return env, stored


async def test_fork_used_when_exchange_is_in_window(tmp_path):
    agent = FakeForkAgent({"appraisal"}, text=FORK_JSON)
    env, stored = await _run(tmp_path, agent)
    assert env.stub.chat_calls == []
    call = agent.calls[0]
    assert call["stage"] == "appraisal"
    # Cut at the reply (#341): the window is [user 41, assistant 42], so the
    # fork's prefix ends one past the reply rather than at the live window end.
    assert (call["cut"], call["extra_body"]) == (2, STAGE2_EXTRA_BODY)
    assert call["instruction"].startswith(DEFAULT_STAGE2_FORK_PROMPT)
    for part in ("«is the heater the culprit?»", "«yes, brownout»"):
        assert part in call["instruction"]
    assert (call["attribution"]["stage"], call["attribution"]["trace_id"]) == ("appraisal", "k")
    assert stored == json.loads(FORK_JSON)


@pytest.mark.parametrize("agent,window_rowids,rowid,reason", [
    (FakeForkAgent({"appraisal"}, text=FORK_JSON), (50, 51), 41, "not in the window"),  # compacted out
    (FakeForkAgent({"appraisal"}, text=FORK_JSON), (41, 42), None, "not recorded"),  # never persisted
    (FakeForkAgent({"appraisal"}, text=FORK_JSON), None, 41, "no live window"),
    (FakeForkAgent(set(), text=FORK_JSON), (41, 42), 41, "forks are not enabled for appraisal"),
    (None, (41, 42), 41, "forks are not enabled for appraisal"),  # no agent plugin
    (FakeForkAgent({"appraisal"}, error=ForkRefused("tool call")), (41, 42), 41, "ForkRefused"),
    # A plain backend error, not a fork refusal: the HTTP 400 in #360 (#377).
    (FakeForkAgent({"appraisal"}, error=RuntimeError("400 model is not loaded")),
     (41, 42), 41, "RuntimeError"),
    (FakeForkAgent({"appraisal"}, text=""), (41, 42), 41, "ValueError"),  # empty reply
    (FakeForkAgent({"appraisal"}, text="no json here"), (41, 42), 41, "ValueError"),
])
async def test_fork_miss_skips(tmp_path, caplog, agent, window_rowids, rowid, reason):
    """#320: a miss makes no background-role call, stores no stage 2, and
    logs one warning naming the stage, trace, channel and what was missing."""
    with caplog.at_level(logging.WARNING, logger="corvidae.appraisal"):
        env, stored = await _run(tmp_path, agent, window_rowids, rowid)
    # No standalone call on any role, nothing persisted, nothing readable.
    assert env.stub.chat_calls == []
    assert stored is None
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


def test_appraisal_registers_its_llm_stage_as_chat():
    """R1a: appraisal owns the ``appraisal`` stage, a chat stage."""
    from corvidae_cognition.appraisal import AppraisalPlugin
    from helpers import collect_llm_stages

    assert collect_llm_stages(AppraisalPlugin(), "appraisal") == {"appraisal": "chat"}


async def test_the_fork_is_cut_at_the_reply_not_the_window_end(tmp_path):
    """#341: a row written after the reply must not reach the judge's prefix.

    The window is [user 41, assistant 42, user 43]: 43 is the user's next
    message, written while stage 2 was queued. The fork's cut is 2, so the
    judge sees the exchange and not what followed it.
    """
    agent = FakeForkAgent({"appraisal"}, text=FORK_JSON)
    env, stored = await _run(tmp_path, agent, window_rowids=(41, 42, 43))
    call = agent.calls[0]
    assert call["cut"] == 2, "the fork must end one past the reply row"
    assert stored == json.loads(FORK_JSON)


async def test_a_reply_with_no_recorded_row_skips_stage2(tmp_path, caplog):
    """A cut with no reply to cut at is a miss, not a fork of the whole window."""
    agent = FakeForkAgent({"appraisal"}, text=FORK_JSON)
    env = await build_env(tmp_path)
    env.pm.unregister(name="agent")
    env.pm.register(agent, name="agent")
    env.channel.conversation = window_of(env.channel.id, [41, 42])
    await env.appraisal.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=41,
        text="is the heater the culprit?", annotations={})
    with caplog.at_level(logging.WARNING, logger="corvidae.appraisal"):
        # No _record_reply: the reply row was never noted.
        await env.appraisal._run_stage2(
            env.channel, "k", "is the heater the culprit?", "yes, brownout", None)
    assert agent.calls == []
    skips = [r.getMessage() for r in caplog.records
             if "stage-2 appraisal skipped" in r.getMessage()]
    assert len(skips) == 1 and "the reply was not recorded" in skips[0]
    await _teardown(env)
