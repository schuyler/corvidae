"""#1189: the appraisal vector is removed.

The appraisal fork asks for and parses only the critique lens scores and the
reply's past claims; the five dimensions, the correction flag, the memory
retrieval note, the importance prior built from them and memory's valence tag
are gone, and the prior's weight keys are warned by name at startup and
reload.
"""

import json
import logging

import aiosqlite
import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.context import MessageType
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae_cognition.outcome_log import OutcomeLogPlugin

from fake_clock import install_kernel_clock
from helpers import add_kernel_registry, attach_store, FakeForkAgent, stub_llm_config, window_of, with_chat_llm
from .test_appraisal_hooks import PeerSpy, _wire
from .test_appraisal_stage2 import StubClient, _read_appraisal, _teardown, build_env
from tests.cognition.trace_helpers import add_trace

# The six fields the fork no longer asks for.
_DIMENSIONS = ("valence", "stakes", "ambiguity", "commitment_density", "novelty", "correction")

# A reply in the shape the fork now asks for, with one claim and the lenses.
_REPLY = json.dumps({
    "past_claims": [{"claim": "we agreed on Tuesday", "queries": ["Tuesday"]}],
    "lenses": {"predictive": 70, "none": 30},
})


async def _fork_once(tmp_path, offer=None, reply=_REPLY):
    """Run one stage-2 fork for exchange "k" with a retrieval profile reported
    on the trace; return (the fork call, the stored stage 2)."""
    env = await build_env(tmp_path)
    agent = FakeForkAgent({"appraisal"}, text=reply)
    env.pm.unregister(name="agent")
    env.pm.register(agent, name="agent")
    env.channel.conversation = window_of(env.channel.id, [41, 42])
    if offer is not None:
        # A peer offers the lenses at the idle trigger (on_appraisal_prompts).
        _wire(env, PeerSpy({"idle": offer}))
    # A retrieval profile the old prompt would have quoted.
    await env.pm.ahook.on_trace_record(
        trace_id="k", channel_id=env.channel.id, source="memory", origin=None,
        record={"retrieval_top_score": 0.87, "retrieval_hit_count": 3},
    )
    await env.appraisal.on_message_persisted(
        channel=env.channel, trace_id="k", rowid=41, text="what did we agree?", annotations={})
    try:
        await env.appraisal._run_stage2(
            env.channel, "k", "what did we agree?", "we agreed on Tuesday", None)
        stored = ((await _read_appraisal(env.recorder, "k")) or {}).get("stage2")
    finally:
        await _teardown(env)
    return agent.calls[0], stored


# ---------------------------------------------------------------------------
# R1, R3: the fork's prompt, schema and parser
# ---------------------------------------------------------------------------


async def test_fork_instruction_names_no_dimension_and_no_retrieval_note(tmp_path):
    """The instruction sent to the fork asks for none of the six fields and
    carries no memory-retrieval line, even with a profile recorded."""
    call, _ = await _fork_once(tmp_path, offer=[("predictive", "Misread?")])
    instruction = call["instruction"]
    for name in _DIMENSIONS:
        assert name not in instruction, name
    assert "Memory retrieval" not in instruction
    assert "0.87" not in instruction
    # Still asks for the claims and the lens scores.
    assert "past_claims" in instruction
    assert '"lenses"' in instruction


@pytest.mark.parametrize("offer,expected", [
    (None, {"past_claims"}),
    ([("predictive", "Misread?")], {"past_claims", "lenses"}),
])
async def test_fork_schema_holds_only_claims_and_lenses(tmp_path, offer, expected):
    """The json_schema sent requires and describes only past_claims, plus
    lenses when lenses are offered."""
    call, _ = await _fork_once(tmp_path, offer=offer)
    schema = call["extra_body"]["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == expected
    assert set(schema["required"]) == expected


async def test_reply_with_only_claims_and_lenses_is_stored(tmp_path):
    """A reply in the new shape is a stage 2: claims and lens scores are
    persisted (they are critique's inputs and #141's labels)."""
    _, stored = await _fork_once(tmp_path, offer=[("predictive", "Misread?")])
    assert stored == {
        "past_claims": [{"claim": "we agreed on Tuesday", "queries": ["Tuesday"]}],
        "lenses": {"predictive": 0.7, "none": 0.3},
    }


def test_parser_drops_the_dimensions():
    """A reply that still carries the six fields parses without them."""
    from corvidae_cognition.appraisal import _stage2_from_text

    reply = json.dumps({
        "valence": 0.8, "stakes": 0.9, "ambiguity": 0.2,
        "commitment_density": 0.5, "novelty": 0.7, "correction": True,
        "past_claims": [],
    })
    assert _stage2_from_text(reply) == {"past_claims": []}


# ---------------------------------------------------------------------------
# R2: memory scores importance itself and writes no valence
# ---------------------------------------------------------------------------


async def test_consolidation_uses_memorys_own_rating_and_stores_no_valence(tmp_path):
    """With appraisal loaded and old stage-2 vectors covering the range, the
    record's importance is the consolidation reply's own rating and its
    valence is NULL."""
    from corvidae_cognition.appraisal import AppraisalPlugin
    from corvidae_cognition.memory import MemoryPlugin, RubricPrior

    consolidation_json = json.dumps({
        "entries": [{"narrative": "I discussed the plan with the user.",
                     "topic_tags": ["plan"]}],
        "participants": ["user"],
        "importance": 0.37,
    })

    # Wire persistence, outcome_log, a stub LLM, memory and appraisal.
    db_path = str(tmp_path / "sessions.db")
    db = await aiosqlite.connect(db_path, isolation_level=None)
    await init_db(db)
    try:
        pm = create_plugin_manager()
        # memory's on_start resolves the kernel clock and idle tracker (#1180).
        install_kernel_clock(pm)
        # memory's consolidation commit resolves the kernel registry (#1180).
        add_kernel_registry(pm)
        persistence = PersistencePlugin()
        persistence.db, persistence.writer = await attach_store(db)
        pm.register(persistence, name="persistence")
        # The trace protocol plugin first: outcome_log depends on it.
        add_trace(pm)
        outcome = OutcomeLogPlugin()
        pm.register(outcome, name="outcome_log")
        await outcome.on_init(pm=pm, config={})
        await outcome.on_start(config={})
        llm = LLMPlugin()
        pm.register(llm, name="llm")
        llm._clients["main"] = StubClient(consolidation_json)
        stub_llm_config(llm)
        cfg = with_chat_llm({"daemon": {"session_db": db_path}, "memory": {}})
        memory = MemoryPlugin()
        pm.register(FakeForkAgent(set()), name="agent")
        pm.register(memory, name="memory")
        await memory.on_init(pm=pm, config=cfg)
        await memory.on_start(config=cfg)
        appraisal = AppraisalPlugin()
        pm.register(appraisal, name="appraisal")
        await appraisal.on_init(pm=pm, config=cfg)
        # Run every on_start appraisal implements (none, today), as the
        # daemon's broadcast would, without restarting the other plugins.
        for impl in pm.hook.on_start.get_hookimpls():
            if impl.plugin is appraisal:
                await impl.function(config=cfg)

        # Appraisal leaves memory's own prior in place.
        assert isinstance(memory.importance_prior, RubricPrior)

        # Dialog to consolidate; nothing appraisal recorded can reach the prior.
        channel = Channel(transport="test", scope="conv", config=ChannelConfig())
        rowids = []
        for role, content in [("user", "here is the plan"),
                              ("assistant", "understood, I will do it")]:
            rowids.append(await persistence.persist_message(
                channel=channel, message={"role": role, "content": content},
                message_type=MessageType.MESSAGE,
            ))

        await memory._consolidate_range(channel.id, rowids[-1], "test")

        async with db.execute(
            "SELECT importance, valence FROM memory WHERE channel_id = ?", (channel.id,),
        ) as cursor:
            rows = await cursor.fetchall()
        assert rows, "consolidation wrote no record"
        for importance, valence in rows:
            assert importance == pytest.approx(0.37)
            assert valence is None
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# R4: the prior's weight keys are warned by name
# ---------------------------------------------------------------------------

_PRIOR_KEYS = ("appraisal.prior.w_stakes", "appraisal.prior.w_valence", "appraisal.prior.w_novelty")


@pytest.mark.parametrize("hook", ["on_init", "on_config_reload"])
async def test_each_removed_prior_key_is_warned_once_with_the_remedy(caplog, hook):
    """Each removed weight key logs exactly one WARNING that names the key
    and says to delete it from agent.yaml; the hook does not raise."""
    from corvidae_cognition.appraisal import AppraisalPlugin

    pm = create_plugin_manager()
    plugin = AppraisalPlugin()
    pm.register(plugin, name="appraisal")
    config = {"appraisal": {"prior": {"w_stakes": 0.4, "w_valence": 0.3, "w_novelty": 0.3}}}
    with caplog.at_level(logging.WARNING, logger="corvidae"):
        if hook == "on_init":
            await plugin.on_init(pm=pm, config=config)
        else:
            await plugin.on_init(pm=pm, config={})
            caplog.clear()
            await plugin.on_config_reload(config=config)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    for key in _PRIOR_KEYS:
        hits = [m for m in messages if key in m]
        assert len(hits) == 1, (key, messages)
        assert "delete it from agent.yaml" in hits[0], hits[0]
        assert "#1189" in hits[0], hits[0]
