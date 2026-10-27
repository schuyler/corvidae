"""Consolidation as a fork of the window (GitHub issue #51, phase 3).

Compaction trigger: before_compaction forks the record from the dropped
prefix and stashes it; the on_compaction task consumes it instead of the
standalone call. Idle trigger: fork the full window. Any miss is the
standalone call; everything after the LLM call is unchanged.
"""

import json

import pytest

from corvidae.fork import ForkRefused
from corvidae_cognition.memory import DEFAULT_CONSOLIDATION_FORK_PROMPT

from helpers import FakeForkAgent, window_of
from .test_memory_consolidation import build_consolidation_env, fetch_memories, seed_dialog

FORK_JSON = json.dumps({"summary": "I told kestrel it was the brownout detector.",
                        "topic_tags": ["wifi"], "participants": ["kestrel"]})
DIALOG = [
    ("user", "my weather station keeps dropping wifi"),
    ("assistant", "Does it happen when the heater kicks on?"),
    ("user", "yes, every time"),
    ("assistant", "That's the brownout detector resetting the radio."),
]


class _Registry:
    def __init__(self, channel):
        self.channel = channel

    def get(self, channel_id):
        return self.channel if channel_id == self.channel.id else None


@pytest.fixture
async def env():
    """(memory, stub client, channel, db, rowids) with DIALOG persisted and
    a live window holding it."""
    memory, persistence, stub, channel, db = await build_consolidation_env()
    memory.pm.register(_Registry(channel), name="registry")
    ids = await seed_dialog(persistence, channel, DIALOG)
    channel.conversation = window_of(channel.id, ids, contents=[c for _, c in DIALOG])
    yield memory, stub, channel, db, ids
    await db.close()


def _use(memory, agent):
    memory.pm.register(agent, name="agent")
    return agent


def _standalone(stub):
    return [c for c in stub.chat_calls if c["attribution"].get("stage") == "consolidation"]


async def _compaction(memory, channel, compacted, cut):
    await memory.before_compaction(channel=channel, compacted_ids=compacted, cut=cut)
    await memory.on_compaction(channel=channel, summary_msg={}, retain_count=1, compacted_ids=compacted)
    await memory.wait_for_background_tasks()


async def test_compaction_fork_record_is_stashed_and_consumed(env):
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    await _compaction(memory, channel, ids[:3], cut=3)

    call = agent.calls[0]
    assert call["cut"] == 3
    assert call["instruction"].startswith(DEFAULT_CONSOLIDATION_FORK_PROMPT)
    assert (call["attribution"]["stage"], call["attribution"]["trigger"]) == ("consolidation", "compaction")
    assert _standalone(stub) == []
    [record] = await fetch_memories(db)
    assert record["summary"] == json.loads(FORK_JSON)["summary"]
    assert (record["msg_id_start"], record["msg_id_end"]) == (ids[0], ids[2])
    assert memory._fork_stash == {}


@pytest.mark.parametrize("agent,drop_first_row", [
    (FakeForkAgent({"consolidation"}, error=ForkRefused("tool call")), False),
    (FakeForkAgent({"consolidation"}, error=RuntimeError("down")), False),
    (FakeForkAgent({"consolidation"}, text="not json"), False),
    (FakeForkAgent({"consolidation"}, text=json.dumps({"summary": " "})), False),
    (FakeForkAgent(set(), text=FORK_JSON), False),
    # The window lost a row of the range (an earlier compaction whose
    # consolidation failed): the fork can't see it all.
    (FakeForkAgent({"consolidation"}, text=FORK_JSON), True),
])
async def test_compaction_fork_miss_is_standalone(env, agent, drop_first_row):
    memory, stub, channel, db, ids = env
    _use(memory, agent)
    if drop_first_row:
        channel.conversation.messages.pop(0)
    await _compaction(memory, channel, ids[:3], cut=2 if drop_first_row else 3)
    assert memory._fork_stash == {}
    [call] = _standalone(stub)
    assert call["attribution"]["mode"] == "standalone"
    assert len(await fetch_memories(db)) == 1


async def test_stash_from_a_different_watermark_is_discarded(env):
    """The watermark moved between fork and task: the stashed record covers
    the wrong range, so the task makes the standalone call."""
    memory, stub, channel, db, ids = env
    _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    await memory.before_compaction(channel=channel, compacted_ids=ids[:3], cut=3)
    mem_db = await memory._ensure_schema()
    await memory._set_watermark(mem_db, channel.id, ids[0])
    await mem_db.commit()
    await memory.on_compaction(channel=channel, summary_msg={}, retain_count=1, compacted_ids=ids[:3])
    await memory.wait_for_background_tasks()
    assert len(_standalone(stub)) == 1
    assert (await fetch_memories(db))[0]["msg_id_start"] == ids[1]


async def test_summary_rows_in_range_do_not_block_the_fork(env):
    """A live summary has no _db_id but its message_log row can sit inside
    the range; only dialog rows count for containment."""
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    await db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, '{}', 0.0, 'summary')", (channel.id,))
    await db.commit()
    later = ids + await seed_dialog(memory.pm.get_plugin("persistence"), channel, [("user", "more")])
    channel.conversation = window_of(channel.id, later, summary=True)
    await memory.before_compaction(channel=channel, compacted_ids=later[:5], cut=6)
    assert len(agent.calls) == 1


@pytest.mark.parametrize("live_window,error,forked", [
    (True, None, True), (False, None, False), (True, RuntimeError("down"), False),
])
async def test_idle_trigger_forks_the_full_window(env, live_window, error, forked):
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON, error=error))
    if not live_window:
        channel.conversation = None
    await memory._consolidate_range(channel.id, ids[-1], "idle")
    if live_window:
        assert (agent.calls[0]["cut"], agent.calls[0]["attribution"]["trigger"]) == (None, "idle")
    assert len(_standalone(stub)) == (0 if forked else 1)
    assert len(await fetch_memories(db)) == 1


@pytest.mark.parametrize("summary,watermark_index,expected", [
    (False, None, "the whole conversation above"),
    (True, None, "follows the summary"),
    # Rows below the watermark are still in view: name the first new one.
    (True, 1, "«yes, every time»"),
])
async def test_range_boundary_sentence(env, summary, watermark_index, expected):
    memory, stub, channel, db, ids = env
    agent = _use(memory, FakeForkAgent({"consolidation"}, text=FORK_JSON))
    channel.conversation = window_of(channel.id, ids, summary=summary, contents=[c for _, c in DIALOG])
    if watermark_index is not None:
        mem_db = await memory._ensure_schema()
        await memory._set_watermark(mem_db, channel.id, ids[watermark_index])
        await mem_db.commit()
    await memory._consolidate_range(channel.id, ids[-1], "idle")
    assert expected in agent.calls[0]["instruction"]
