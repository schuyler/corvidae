"""Compaction waits for an in-flight turn or idle consolidation (issue #621):
it consolidates only what lies past the watermark that task leaves, so no
paid-for reply is discarded."""

import asyncio

from helpers import window_of

from .test_memory_consolidation import fetch_stamped_memories, fetch_watermark, seed_dialog
from .test_memory_fork import DIALOG
from .test_memory_turn_trigger import (  # noqa: F401 (make_env fixture)
    REPLY,
    GatedForkAgent,
    make_env,
    tokens_of,
)


def view_requests(stub) -> list[dict]:
    """Background view calls the consolidation stage sent (catch-up)."""
    return [c for c in stub.chat_calls if c["attribution"].get("stage") == "consolidation"]


async def start_gated_turn(make_env):  # noqa: F811 (fixture value)
    """An env whose window holds DIALOG[:2] and whose turn fork over them is
    in flight, blocked on the gate; then the window grows to all of DIALOG,
    as it would while the turn task waits."""
    agent = GatedForkAgent({"consolidation"}, text=REPLY)
    env = await make_env(tokens_of(DIALOG[:2]), agent=agent, window_ids=slice(0, 2))
    # Both replies embed to the stub's one vector; keep near-dup merge from
    # folding the two records together, so each keeps its own range.
    env.memory._dup_threshold = 1.01
    # The turn hook fixes range_end at ids[1] and spawns the blocked task.
    await env.memory.on_agent_response(channel=env.channel)
    assert env.channel.id in env.memory._idle_inflight
    # Two more messages arrive while the fork is pending.
    env.channel.conversation = window_of(
        env.channel.id, env.ids, contents=[c for _, c in DIALOG]
    )
    return env


def ranges(rows: list[dict]) -> set[tuple[int, int, str]]:
    """(msg_id_start, msg_id_end, trigger) of every stored record."""
    return {(r["msg_id_start"], r["msg_id_end"], r["trigger"]) for r in rows}


async def test_compaction_fork_waits_for_the_turn_in_the_background(make_env):  # noqa: F811
    """Acceptance (#621, as moved off the response path by #628): turn over
    1-2 in flight, compaction over 1-4. before_compaction returns at once;
    its background fork waits for the turn to commit, then forks only 3-4;
    no reply is discarded and no view call is made."""
    env = await start_gated_turn(make_env)
    memory, ids = env.memory, env.ids

    # Compaction fires while the turn fork is still blocked: no waiting.
    await asyncio.wait_for(
        memory.before_compaction(channel=env.channel, compacted_ids=ids, cut=len(ids)),
        timeout=1,
    )

    # The turn fork answers; compaction proceeds over what is left.
    env.agent.gate.set()
    await memory.on_compaction(
        channel=env.channel, summary_msg={}, retain_count=1, compacted_ids=ids
    )
    await memory.wait_for_background_tasks()

    # Two forks, each over its own range; nothing went to the view.
    assert len(env.agent.calls) == 2
    assert view_requests(env.stub) == []
    rows = await fetch_stamped_memories(env.db)
    assert ranges(rows) == {
        (ids[0], ids[1], "turn"),
        (ids[2], ids[3], "compaction"),
    }
    assert await fetch_watermark(env.db, env.channel.id) == ids[-1]


async def test_compaction_task_without_stash_waits_then_views_the_rest(make_env):  # noqa: F811
    """No stash (before_compaction didn't fork): the on_compaction task waits
    for the turn task, then its view call covers only the remainder."""
    env = await start_gated_turn(make_env)
    memory, ids = env.memory, env.ids

    # on_compaction alone: its task must wait for the turn task.
    await memory.on_compaction(
        channel=env.channel, summary_msg={}, retain_count=1, compacted_ids=ids
    )
    for _ in range(5):
        await asyncio.sleep(0)
    assert view_requests(env.stub) == []

    env.agent.gate.set()
    await memory.wait_for_background_tasks()

    # The turn record stands and the view covered only ids[2:].
    rows = await fetch_stamped_memories(env.db)
    assert ranges(rows) == {
        (ids[0], ids[1], "turn"),
        (ids[2], ids[3], "compaction"),
    }
    assert len(view_requests(env.stub)) == 1
    assert await fetch_watermark(env.db, env.channel.id) == ids[-1]


async def test_compaction_proceeds_when_the_inflight_task_fails(make_env):  # noqa: F811
    """Guard: a failing in-flight turn fork doesn't leave compaction hanging;
    it waits, then consolidates the whole range itself (compaction is never
    skipped). Passes without the wait too; it fails only if the wait hangs."""
    env = await start_gated_turn(make_env)
    memory, ids = env.memory, env.ids
    env.agent.error = RuntimeError("backend down")

    before = asyncio.create_task(
        memory.before_compaction(channel=env.channel, compacted_ids=ids, cut=len(ids))
    )
    for _ in range(5):
        await asyncio.sleep(0)
    env.agent.gate.set()
    await asyncio.wait_for(before, timeout=5)
    await memory.on_compaction(
        channel=env.channel, summary_msg={}, retain_count=1, compacted_ids=ids
    )
    await memory.wait_for_background_tasks()

    # The compaction fork failed too (same agent), so the view wrote it all.
    rows = await fetch_stamped_memories(env.db)
    assert ranges(rows) == {(ids[0], ids[3], "compaction")}
    assert await fetch_watermark(env.db, env.channel.id) == ids[-1]


async def test_cancelling_compaction_leaves_the_turn_task_running(make_env):  # noqa: F811
    """Cancelling compaction's background fork while it waits must not
    cancel the turn task it waits on: the turn still commits its record."""
    env = await start_gated_turn(make_env)
    memory, ids = env.memory, env.ids

    await memory.before_compaction(channel=env.channel, compacted_ids=ids, cut=len(ids))
    # The stashed background fork is waiting on the turn task: cancel it.
    [pending] = list(memory._fork_stash.values())
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)

    env.agent.gate.set()
    await memory.wait_for_background_tasks()
    rows = await fetch_stamped_memories(env.db)
    assert ranges(rows) == {(ids[0], ids[1], "turn")}


async def test_inflight_idle_is_waited_for_off_the_response_path(make_env):  # noqa: F811
    """An in-flight idle task (possibly a multi-call catch-up) is waited for
    in the background: before_compaction returns at once, and its snapshot
    fork takes only what lies past the idle task's watermark."""
    agent = GatedForkAgent({"consolidation"}, text=REPLY)
    env = await make_env(tokens_of(DIALOG), agent=agent)
    env.memory._dup_threshold = 1.01
    memory, channel = env.memory, env.channel
    # Idle over everything in message_log: an in-window fork, blocked.
    await memory._on_channel_idle(channel.id)
    assert channel.id in memory._idle_inflight
    # Two more messages arrive while the idle fork is pending.
    more = await seed_dialog(env.persistence, channel, DIALOG[:2])
    ids = env.ids + more
    channel.conversation = window_of(
        channel.id, ids, contents=[c for _, c in DIALOG + DIALOG[:2]]
    )

    # Compaction over all six: the hook returns at once.
    await asyncio.wait_for(
        memory.before_compaction(channel=channel, compacted_ids=ids, cut=len(ids)),
        timeout=1,
    )
    await memory.on_compaction(
        channel=channel, summary_msg={}, retain_count=1, compacted_ids=ids
    )

    # The idle fork answers; the snapshot fork then takes only the rest.
    agent.gate.set()
    await memory.wait_for_background_tasks()
    rows = await fetch_stamped_memories(env.db)
    assert ranges(rows) == {
        (ids[0], ids[3], "idle"),
        (ids[4], ids[5], "compaction"),
    }
    assert len(agent.calls) == 2
    assert view_requests(env.stub) == []
