"""Compaction's consolidation forks a snapshot of the pre-compaction window
in the background (issue #628): the turn doesn't wait for the fork, and the
record is stamped with the compacted range."""

import asyncio

from corvidae.context import MessageType

from .test_memory_compaction_waits import ranges, view_requests
from .test_memory_consolidation import fetch_stamped_memories, fetch_watermark
from .test_memory_fork import DIALOG
from .test_memory_turn_trigger import (  # noqa: F401 (make_env fixture)
    REPLY,
    GatedForkAgent,
    make_env,
    tokens_of,
)


async def test_compaction_fork_runs_after_the_turn_on_a_snapshot(make_env):  # noqa: F811
    """R1/R2: before_compaction returns while the fork is still blocked; the
    live window is then compacted; the fork renders the snapshot (holding the
    compacted dialog) and the record covers the compacted range."""
    agent = GatedForkAgent({"consolidation"}, text=REPLY)
    # Threshold out of reach: no turn consolidation interferes.
    env = await make_env(tokens_of(DIALOG) * 10, agent=agent)
    memory, channel, ids = env.memory, env.channel, env.ids
    compacted = ids[:3]

    # The hook returns although the fork can't answer yet.
    await asyncio.wait_for(
        memory.before_compaction(channel=channel, compacted_ids=compacted, cut=3),
        timeout=1,
    )

    # Compaction replaces the dropped prefix with a summary.
    tail = channel.conversation.messages[3:]
    channel.conversation.messages[:] = [
        {"role": "assistant", "content": "[Summary] wifi",
         "_message_type": MessageType.SUMMARY},
        *tail,
    ]
    await memory.on_compaction(
        channel=channel, summary_msg={}, retain_count=1, compacted_ids=compacted
    )
    agent.gate.set()
    await memory.wait_for_background_tasks()

    # One fork, from the snapshot, which still holds the compacted dialog.
    [call] = agent.calls
    snapshot = call["snapshot"]
    assert snapshot is not None
    assert [m.get("_db_id") for m in snapshot.conv.messages[: call["cut"]]] == compacted
    assert view_requests(env.stub) == []
    # The record is stamped with the compacted range, as a compaction fork.
    rows = await fetch_stamped_memories(env.db)
    assert ranges(rows) == {(ids[0], ids[2], "compaction")}
    assert {r["mode"] for r in rows} == {"fork"}
    assert await fetch_watermark(env.db, channel.id) == ids[2]
