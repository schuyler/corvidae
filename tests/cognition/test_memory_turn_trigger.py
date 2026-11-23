"""Turn trigger for consolidation (issue #636, D1-D10): at a turn boundary, a window
reaching ``memory.consolidate_after_tokens`` gets one fork stamped ``trigger='turn'``."""

import asyncio
import copy
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from corvidae.context import MessageType, message_tokens
from corvidae.hooks import create_plugin_manager
from corvidae_cognition.memory import MemoryPlugin

from helpers import FakeForkAgent, window_of, with_chat_llm
from .test_memory_consolidation import (
    EMBED_CONFIG, build_consolidation_env, entries_reply, fetch_stamped_memories,
    fetch_watermark, seed_dialog,
)
from .test_memory_fork import DIALOG, _Registry

#: A one-entry reply (the embed stub returns one fixed vector; see the
#: design's Traps list).
REPLY = entries_reply([("I told kestrel the wifi drops were the brownout detector.", ["wifi"])])


def tokens_of(rows: list[tuple[str, str]]) -> int:
    """Window-token sum of dialog rows, as the turn task counts them
    (``message_tokens`` ignores the ``_`` tags the window adds)."""
    return sum(message_tokens({"role": r, "content": c}) for r, c in rows)


class GatedForkAgent(FakeForkAgent):
    """A FakeForkAgent whose ``fork`` waits on an event before answering."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The gate starts closed; tests open it with gate.set().
        super().__init__(*args, **kwargs)
        self.gate = asyncio.Event()

    async def fork(self, *args: Any, **kwargs: Any) -> Any:
        # Block until the test opens the gate, then behave as the fake.
        await self.gate.wait()
        return await super().fork(*args, **kwargs)


async def seed_window_bounds(env: SimpleNamespace) -> None:
    """Commit a watermark at ids[1] and add one CONTEXT, one SUMMARY and one
    UNSENT row (persisted, and in the window with their ``_db_id``) above it.
    The sum a turn counts is then DIALOG[2:] exactly: rows at or below the
    watermark and the three non-MESSAGE types must all be left out."""
    # The watermark as a prior consolidation would have committed it.
    await env.db.execute(
        "INSERT OR REPLACE INTO consolidation_watermark (channel_id, last_message_id) "
        "VALUES (?, ?)",
        (env.channel.id, env.ids[1]),
    )
    await env.db.commit()
    # Big non-MESSAGE rows: counting any of them would overshoot N by a lot.
    for mtype in (MessageType.CONTEXT, MessageType.SUMMARY, MessageType.UNSENT):
        message = {"role": "assistant", "content": f"{mtype.value} filler " * 40}
        rowid = await env.persistence.persist_message(
            channel=env.channel, message=message, message_type=mtype,
        )
        env.channel.conversation.messages.append(
            {**message, "_db_id": rowid, "_message_type": mtype}
        )


@pytest.fixture
async def make_env() -> Any:
    """Factory for a consolidation env with DIALOG persisted, windowed over ``window_ids``, threshold ``n``."""
    dbs = []

    async def build(
        n: int, agent: FakeForkAgent | None = None, window_ids: slice = slice(None)
    ) -> SimpleNamespace:
        # Threshold goes in through config, as an operator sets it.
        config = copy.deepcopy(EMBED_CONFIG)
        config["memory"]["consolidate_after_tokens"] = n
        memory, persistence, stub, channel, db = await build_consolidation_env(config)
        dbs.append(db)
        stub.chat_text = REPLY
        # Swap in this test's fork agent (default: forks on for consolidation).
        agent = agent or FakeForkAgent({"consolidation"}, text=REPLY)
        memory.pm.unregister(name="agent")
        memory.pm.register(agent, name="agent")
        memory.pm.register(_Registry(channel), name="registry")
        # Persist the dialog and hold (part of) it in the live window.
        ids = await seed_dialog(persistence, channel, DIALOG)
        channel.conversation = window_of(
            channel.id, ids[window_ids], contents=[c for _, c in DIALOG][window_ids]
        )
        # on_start's backfill task must not count as a turn task.
        await memory.wait_for_background_tasks()
        return SimpleNamespace(
            memory=memory, stub=stub, channel=channel, db=db, ids=ids, agent=agent,
            persistence=persistence,
        )

    yield build
    for db in dbs:
        await db.close()


async def test_turn_at_threshold_forks_once_with_trigger_turn(make_env):
    """D1: N equal to the window sum forks once; every stored row and the
    attribution carry trigger='turn'; the watermark moves to range_end."""
    env = await make_env(tokens_of(DIALOG[2:]))
    # Window bounds and non-MESSAGE rows, as seed_window_bounds describes.
    await seed_window_bounds(env)
    range_end = max(m["_db_id"] for m in env.channel.conversation.messages)

    # The turn hook fires, then the spawned task runs to completion.
    await env.memory.on_agent_response(channel=env.channel)
    await env.memory.wait_for_background_tasks()

    # One fork, attributed to the turn trigger.
    assert len(env.agent.calls) == 1
    assert env.agent.calls[0]["attribution"]["trigger"] == "turn"
    # Every stored record is stamped turn and the watermark covers the range.
    rows = await fetch_stamped_memories(env.db)
    assert {r["trigger"] for r in rows} == {"turn"}
    # The range starts just above the old watermark and ends at the window's top.
    assert {r["msg_id_start"] for r in rows} == {env.ids[2]}
    assert await fetch_watermark(env.db, env.channel.id) == range_end


async def test_turn_below_threshold_does_not_fork(make_env):
    """D2: N one past the window sum forks nothing and stores nothing; the
    sum has the same bounds and type filter as D1."""
    env = await make_env(tokens_of(DIALOG[2:]) + 1)
    # Same bounds and type filter as D1.
    await seed_window_bounds(env)

    # Hook then task: the task sums, finds less than N, and ends.
    await env.memory.on_agent_response(channel=env.channel)
    await env.memory.wait_for_background_tasks()

    # No fork, no record, watermark untouched.
    assert env.agent.calls == []
    assert await fetch_stamped_memories(env.db) == []
    assert await fetch_watermark(env.db, env.channel.id) == env.ids[1]


async def test_turn_with_pending_tool_calls_does_not_fork(make_env):
    """D3: tool calls outstanding means the reply is not a turn boundary."""
    env = await make_env(tokens_of(DIALOG))
    env.channel.pending_tool_call_ids = {"call_1"}
    tasks_before = set(env.memory._tasks)

    await env.memory.on_agent_response(channel=env.channel)

    # No new task was spawned, and after draining there is still no fork.
    assert set(env.memory._tasks) - tasks_before == set()
    await env.memory.wait_for_background_tasks()
    assert env.agent.calls == []


async def test_turn_with_forks_off_is_silent(make_env, caplog):
    """D4: forks off for consolidation means no task, no fork, and no log
    record at WARNING or above from any logger."""
    env = await make_env(tokens_of(DIALOG), agent=FakeForkAgent(set(), text=REPLY))
    tasks_before = set(env.memory._tasks)
    # Open caplog after the env: building it logs unrelated warnings.
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    await env.memory.on_agent_response(channel=env.channel)
    spawned = set(env.memory._tasks) - tasks_before
    await env.memory.wait_for_background_tasks()

    assert spawned == set()
    assert env.agent.calls == []
    loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert loud == []


async def test_one_consolidation_in_flight_across_turn_and_idle(make_env):
    """D5: a channel with a turn task in flight gets no second turn or idle
    task, and two overlapping idle firings yield one task; the set is empty
    once the tasks end."""
    agent = GatedForkAgent({"consolidation"}, text=REPLY)
    env = await make_env(tokens_of(DIALOG), agent=agent)
    memory, cid = env.memory, env.channel.id

    # Part 1: the first hook spawns a task blocked on the gate, channel in flight.
    tasks_before = set(memory._tasks)
    await memory.on_agent_response(channel=env.channel)
    assert len(set(memory._tasks) - tasks_before) == 1
    assert cid in memory._idle_inflight

    # While in flight, neither another turn hook nor an idle firing spawns.
    tasks_mid = set(memory._tasks)
    await memory.on_agent_response(channel=env.channel)
    await memory._on_channel_idle(cid)
    assert set(memory._tasks) == tasks_mid

    # Open the gate: one fork in all, and the channel is released.
    agent.gate.set()
    await memory.wait_for_background_tasks()
    assert memory._idle_inflight == set()
    assert len(agent.calls) == 1

    # Part 2: count spawns of _consolidate_range; each blocks on a gate.
    gate = asyncio.Event()
    real = memory._consolidate_range
    spawned: list[tuple] = []

    def gated(*args: Any, **kwargs: Any) -> Any:
        # Count at spawn time; the coroutine blocks until the gate opens.
        spawned.append(args)

        async def run() -> None:
            await gate.wait()
            await real(*args, **kwargs)

        return run()

    memory._consolidate_range = gated
    # Persist one more row so idle has something past the watermark.
    new_id = await env.persistence.persist_message(
        channel=env.channel, message={"role": "user", "content": "and another thing"},
        message_type=MessageType.MESSAGE,
    )
    assert await fetch_watermark(env.db, cid) < new_id
    # Two idle firings overlap: both would pass a late membership check.
    await asyncio.gather(memory._on_channel_idle(cid), memory._on_channel_idle(cid))
    assert len(spawned) == 1

    # Let the one task finish; the in-flight set must release the channel.
    gate.set()
    await memory.wait_for_background_tasks()
    assert memory._idle_inflight == set()


async def test_turn_range_end_is_read_in_the_hook(make_env):
    """D6: a row persisted after the hook is not in the range."""
    agent = GatedForkAgent({"consolidation"}, text=REPLY)
    env = await make_env(tokens_of(DIALOG), agent=agent)
    tasks_before = set(env.memory._tasks)

    await env.memory.on_agent_response(channel=env.channel)
    # A turn task exists and is blocked on the fork gate.
    assert set(env.memory._tasks) - tasks_before

    # A late message enters the window with the next rowid, then is persisted;
    # the task's range was already fixed in the hook.
    late_id = env.ids[-1] + 1
    env.channel.conversation.messages.append({
        "role": "user", "content": "one more thing", "_db_id": late_id,
        "_message_type": MessageType.MESSAGE,
    })
    persisted = await env.persistence.persist_message(
        channel=env.channel, message={"role": "user", "content": "one more thing"},
        message_type=MessageType.MESSAGE,
    )
    assert persisted == late_id

    # Open the gate: the record covers only what the hook saw.
    agent.gate.set()
    await env.memory.wait_for_background_tasks()
    rows = await fetch_stamped_memories(env.db)
    assert {r["msg_id_end"] for r in rows} == {env.ids[-1]}
    assert await fetch_watermark(env.db, env.channel.id) == env.ids[-1]


async def test_late_message_consolidated_exactly_once(make_env):
    """A message persisted during a turn-trigger fork is consolidated exactly
    once, by the next consolidation (the #664 outcome that was never tested)."""
    agent = GatedForkAgent({"consolidation"}, text=REPLY)
    env = await make_env(tokens_of(DIALOG), agent=agent)
    # Disable the near-dup merge: the embed stub returns a fixed vector, so
    # every record is a near-duplicate of the first, and the merge would widen
    # the second record's range to overlap the first — hiding the mutation the
    # ticket names (the late id skipped, the fillers consolidated). With the
    # merge off, the second record's range is exactly (watermark, range_end].
    env.memory._dup_threshold = 2.0

    # First turn: the hook fixes the range at the hook-time window end.
    await env.memory.on_agent_response(channel=env.channel)

    # A late message enters the window and is persisted after the hook; it is
    # not in the first record's range.
    late_id = env.ids[-1] + 1
    env.channel.conversation.messages.append({
        "role": "user", "content": "one more thing", "_db_id": late_id,
        "_message_type": MessageType.MESSAGE,
    })
    persisted = await env.persistence.persist_message(
        channel=env.channel, message={"role": "user", "content": "one more thing"},
        message_type=MessageType.MESSAGE,
    )
    assert persisted == late_id

    # Open the gate: the first record covers only what the hook saw.
    agent.gate.set()
    await env.memory.wait_for_background_tasks()
    rows = await fetch_stamped_memories(env.db)
    assert {r["msg_id_end"] for r in rows} == {env.ids[-1]}
    assert await fetch_watermark(env.db, env.channel.id) == env.ids[-1]

    # Push the window past the threshold again and fire the turn hook: the
    # second consolidation must pick up the late message, exactly once. The
    # watermark is at env.ids[-1], so enough new rows (past it) are needed for
    # their token sum to reach the threshold.
    for i in range(4):
        rowid = await env.persistence.persist_message(
            channel=env.channel,
            message={"role": "user", "content": f"filler {i} to push past the threshold"},
            message_type=MessageType.MESSAGE,
        )
        env.channel.conversation.messages.append({
            "role": "user", "content": f"filler {i} to push past the threshold",
            "_db_id": rowid, "_message_type": MessageType.MESSAGE,
        })
    agent.gate.clear()  # close the gate for the second fork
    await env.memory.on_agent_response(channel=env.channel)
    agent.gate.set()
    await env.memory.wait_for_background_tasks()

    # The late message falls in exactly one record's range, and that record
    # starts at the late message (past the first record's end) — not
    # overlapping the first record, which is the mutation the ticket names.
    # (The near-dup merge is disabled above, so the second record's range is
    # exactly (watermark, range_end].)
    rows = await fetch_stamped_memories(env.db)
    containing = [
        r for r in rows
        if r["msg_id_start"] <= late_id <= r["msg_id_end"]
    ]
    assert len(containing) == 1, (
        f"late message {late_id} in {len(containing)} records: "
        f"{[(r['msg_id_start'], r['msg_id_end']) for r in rows]}"
    )
    assert containing[0]["msg_id_start"] == late_id, (
        f"containing record starts at {containing[0]['msg_id_start']}, "
        f"expected {late_id} (past the first record's end {env.ids[-1]})"
    )
    # The watermark moves past the late message (through the filler window).
    assert await fetch_watermark(env.db, env.channel.id) > late_id


@pytest.mark.parametrize("bad", [0, -1, 1.5, "4000", True, None])
async def test_consolidate_after_tokens_must_be_a_positive_int(bad):
    """D7: anything but a positive integer fails at startup, naming the key,
    the rejected value, the accepted form and the file to fix."""
    with pytest.raises(ValueError) as exc:
        await MemoryPlugin().on_init(
            pm=create_plugin_manager(),
            config=with_chat_llm({"memory": {"consolidate_after_tokens": bad}}),
        )

    # The error names the key, the rejected value, the accepted form and the file.
    msg = str(exc.value)
    assert "memory.consolidate_after_tokens" in msg
    assert repr(bad) in msg
    assert "positive integer" in msg
    assert "agent.yaml" in msg

    # The default is 4000, and an absent key leaves it in place. Imported here
    # so a missing constant fails this test, not collection of the others.
    from corvidae_cognition.memory import DEFAULT_CONSOLIDATE_AFTER_TOKENS

    assert DEFAULT_CONSOLIDATE_AFTER_TOKENS == 4000
    memory = MemoryPlugin()
    await memory.on_init(pm=create_plugin_manager(), config=with_chat_llm({"memory": {}}))
    assert memory._consolidate_after_tokens == 4000


async def test_turn_range_outside_window_skips_at_debug(make_env, caplog):
    """D8: a range with rows missing from the window is a DEBUG skip: no
    fork, no view call, no warning, watermark unchanged."""
    # The window holds rows 3-4 only; N is their sum so the task gets past
    # the threshold and reaches the containment check.
    env = await make_env(tokens_of(DIALOG[2:]), window_ids=slice(2, None))
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    await env.memory.on_agent_response(channel=env.channel)
    await env.memory.wait_for_background_tasks()

    # Neither a fork nor the catch-up view ran.
    assert env.agent.calls == []
    assert env.stub.chat_calls == []
    # Nothing at WARNING or above; exactly one DEBUG line naming the channel.
    loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert loud == []
    skips = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and env.channel.id in r.getMessage()
        and "turn consolidation skipped" in r.getMessage()
    ]
    assert len(skips) == 1, [r.getMessage() for r in caplog.records if r.name.startswith("corvidae")]
    # The watermark did not move.
    assert await fetch_watermark(env.db, env.channel.id) is None


@pytest.mark.parametrize(
    ("trigger", "tail"),
    [
        ("turn", "the next turn trigger retries"),
        # An idle skip is not retried by the next idle firing (#342), so its
        # line names what does retry instead (#1002).
        ("idle", "a new message, the turn trigger or compaction retries"),
    ],
)
async def test_skip_names_its_trigger(make_env, caplog, trigger, tail):
    """D9: a failing fork is one WARNING skip that names what will retry; the
    watermark stays."""
    agent = FakeForkAgent({"consolidation"}, error=RuntimeError("down"))
    env = await make_env(tokens_of(DIALOG), agent=agent)
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    await env.memory._consolidate_range(env.channel.id, env.ids[-1], trigger)

    skips = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and "consolidation skipped" in r.getMessage()
    ]
    assert len(skips) == 1, [r.getMessage() for r in caplog.records if r.name.startswith("corvidae")]
    assert tail in skips[0]
    assert await fetch_watermark(env.db, env.channel.id) is None


async def test_turn_hook_returns_while_fork_is_blocked(make_env):
    """D10: the hook awaits nothing: it completes on its first step while the
    spawned task is still waiting on the blocked fork."""
    agent = GatedForkAgent({"consolidation"}, text=REPLY)
    env = await make_env(tokens_of(DIALOG), agent=agent)
    tasks_before = set(env.memory._tasks)

    # One send(None): StopIteration means it finished without suspending.
    hook = env.memory.on_agent_response(channel=env.channel)
    try:
        with pytest.raises(StopIteration):
            hook.send(None)

        # The turn task is pending, not finished.
        [task] = set(env.memory._tasks) - tasks_before
        assert not task.done()
    finally:
        # Always close the driven coroutine, and release the fork and drain.
        hook.close()
        agent.gate.set()
        await env.memory.wait_for_background_tasks()


async def test_on_stop_refuses_a_late_turn_trigger(make_env):
    """#663: a reply that arrives while or after on_stop gathers spawns nothing.

    Such a task would not be in the gathered set, so it would run past the
    gather and could write on the connection on_stop closes.
    """
    env = await make_env(1)  # threshold 1: any un-consolidated range forks

    await env.memory.on_stop()
    await env.memory.on_agent_response(channel=env.channel)

    assert not env.memory._tasks, "no task may be spawned after on_stop began"
    assert not env.stub.chat_calls, "no consolidation call may be made"


async def test_on_stop_refuses_a_late_compaction_trigger(make_env):
    """#663: the compaction trigger is refused the same way."""
    env = await make_env(1)

    await env.memory.on_stop()
    await env.memory.on_compaction(
        channel=env.channel, summary_msg={}, retain_count=0, compacted_ids=env.ids
    )

    assert not env.memory._tasks, "no task may be spawned after on_stop began"


async def test_a_spawn_refused_after_on_stop_closes_its_coroutine(make_env):
    """#663: refusing closes the coroutine, so Python logs no never-awaited warning."""
    env = await make_env(1)
    await env.memory.on_stop()

    async def work():
        raise AssertionError("a refused coroutine must never run")

    coro = work()
    assert env.memory._spawn(coro) is None
    assert coro.cr_frame is None, "the coroutine was closed, not left un-awaited"


async def test_on_stop_refuses_a_late_idle_scan(make_env):
    """#663: an idle timer that fired before on_stop and resumes after it.

    ``on_stop`` cancels the watch, but a callback already suspended in a read
    is past cancelling: when it resumes, ``_spawn`` refuses. That refusal must
    be handled where it happens, because the idle path goes on to track the
    task and attach a done-callback to it.
    """
    env = await make_env(1)

    await env.memory.on_stop()
    # Exactly what the timer callback does when it resumes.
    await env.memory._on_channel_idle(env.channel.id)

    assert not env.memory._tasks, "no task may be spawned after on_stop began"
    assert env.channel.id not in env.memory._idle_inflight, (
        "the channel stays marked idle, so no later scan can run"
    )


async def test_on_stop_refuses_a_late_compaction_fork(make_env):
    """#663: the fork path's own refusal, which the stashes would trip over.

    ``before_compaction`` stashes the spawned task and its eviction calls
    ``cancel()`` on it: a ``None`` reaching the stash raises there instead.
    """
    env = await make_env(1)

    await env.memory.on_stop()
    await env.memory.before_compaction(
        env.channel, compacted_ids=env.ids, cut=env.ids[-1]
    )

    assert not env.memory._tasks, "no task may be spawned after on_stop began"
    assert not getattr(env.memory, "_fork_stash", {}), (
        "a refused fork left a None in the stash"
    )
