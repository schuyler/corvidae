"""Turn-trigger backoff (issue #620): after N consecutive failed turn forks on
a channel the turn trigger stops forking, with one warning, until the
watermark moves or the backoff interval passes. Idle and compaction are
unaffected."""

import logging

from corvidae_cognition.memory import TURN_FORK_FAILURE_LIMIT
from helpers import FakeForkAgent, window_of

from .test_memory_consolidation import fetch_watermark, seed_dialog
from .test_memory_fork import DIALOG
from .test_memory_turn_trigger import REPLY, make_env, tokens_of  # noqa: F401 (fixture)


async def failing_env(make_env):  # noqa: F811 (make_env fixture)
    """An env whose consolidation fork always raises."""
    agent = FakeForkAgent({"consolidation"}, error=RuntimeError("backend down"))
    # Threshold met by DIALOG[1:] too, so a watermark at ids[0] still forks.
    return await make_env(tokens_of(DIALOG[1:]), agent=agent)


async def run_turn(env) -> None:
    """Fire the turn hook and let the spawned task finish."""
    await env.memory.on_agent_response(channel=env.channel)
    await env.memory.wait_for_background_tasks()


def backoff_warnings(caplog) -> list[str]:
    """The WARNING records that announce a turn-trigger backoff."""
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and "backing off" in r.getMessage()
    ]


async def test_turns_stop_forking_after_n_failures_with_one_warning(make_env, caplog):  # noqa: F811 (make_env fixture)
    """N failing turns fork N times; later turns don't fork; one warning names
    the channel, the message-id range, the count and the remedy."""
    env = await failing_env(make_env)
    caplog.set_level(logging.DEBUG)
    caplog.clear()

    # N failing turns, then two more that must not fork.
    for _ in range(TURN_FORK_FAILURE_LIMIT + 2):
        await run_turn(env)

    assert len(env.agent.calls) == TURN_FORK_FAILURE_LIMIT
    warnings = backoff_warnings(caplog)
    assert len(warnings) == 1, warnings
    msg = warnings[0]
    # Identifier: channel and message-id range; count; remedy.
    assert env.channel.id in msg
    assert f"{env.ids[0]}-{env.ids[-1]}" in msg
    assert f"{TURN_FORK_FAILURE_LIMIT} consecutive" in msg
    assert "llm.stages.consolidation" in msg
    assert "idle and compaction" in msg
    # The skip that started the backoff doesn't promise a turn retry.
    skips = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and "consolidation skipped" in r.getMessage()
    ]
    assert len(skips) == TURN_FORK_FAILURE_LIMIT
    assert all("the next turn trigger retries" in m for m in skips[:-1])
    assert "the next idle or compaction (or, after the backoff, turn) trigger retries" in skips[-1]
    # The watermark never moved.
    assert await fetch_watermark(env.db, env.channel.id) is None


async def test_watermark_move_resets_the_backoff(make_env):  # noqa: F811 (make_env fixture)
    """A consolidation by another trigger (here: the watermark committed as an
    idle run would) ends the backoff; the next turn forks again."""
    env = await failing_env(make_env)
    for _ in range(TURN_FORK_FAILURE_LIMIT + 1):
        await run_turn(env)
    assert len(env.agent.calls) == TURN_FORK_FAILURE_LIMIT

    # Another trigger consolidated part of the range.
    await env.db.execute(
        "INSERT OR REPLACE INTO consolidation_watermark (channel_id, last_message_id) "
        "VALUES (?, ?)",
        (env.channel.id, env.ids[0]),
    )
    await env.db.commit()

    await run_turn(env)
    assert len(env.agent.calls) == TURN_FORK_FAILURE_LIMIT + 1


async def test_backoff_interval_expiry_allows_one_retry(make_env, caplog, monkeypatch):  # noqa: F811 (make_env fixture)
    """After the interval one turn forks again; failing, it backs off again
    with one more warning, and the turn after that doesn't fork."""
    env = await failing_env(make_env)
    now = [1000.0]
    monkeypatch.setattr(env.memory, "_now", lambda: now[0])
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    for _ in range(TURN_FORK_FAILURE_LIMIT + 1):
        await run_turn(env)
    assert len(env.agent.calls) == TURN_FORK_FAILURE_LIMIT

    # Past the interval: one retry, which fails and backs off again.
    now[0] += env.memory._idle_consolidate_after + 1
    await run_turn(env)
    await run_turn(env)

    assert len(env.agent.calls) == TURN_FORK_FAILURE_LIMIT + 1
    assert len(backoff_warnings(caplog)) == 2


async def test_idle_still_forks_during_backoff(make_env):  # noqa: F811 (make_env fixture)
    """Backoff is the turn trigger's only: an idle consolidation still forks."""
    env = await failing_env(make_env)
    for _ in range(TURN_FORK_FAILURE_LIMIT + 1):
        await run_turn(env)
    assert len(env.agent.calls) == TURN_FORK_FAILURE_LIMIT

    await env.memory._consolidate_range(env.channel.id, env.ids[-1], "idle")
    assert len(env.agent.calls) == TURN_FORK_FAILURE_LIMIT + 1


async def test_a_success_resets_the_count(make_env):  # noqa: F811 (make_env fixture)
    """N-1 failures, then a success, then fresh dialog and N more failing
    turns: all N fork, because the success moved the watermark and restarted
    the count (without the reset the first of them would start the backoff)."""
    env = await failing_env(make_env)
    for _ in range(TURN_FORK_FAILURE_LIMIT - 1):
        await run_turn(env)
    # The backend recovers for one turn: it commits and moves the watermark.
    env.agent.error = None
    env.agent.text = REPLY
    await run_turn(env)
    assert await fetch_watermark(env.db, env.channel.id) == env.ids[-1]
    forks_so_far = len(env.agent.calls)

    # New conversation above the watermark, enough to meet the threshold.
    new_ids = await seed_dialog(env.persistence, env.channel, DIALOG)
    env.channel.conversation = window_of(
        env.channel.id, env.ids + new_ids, contents=[c for _, c in DIALOG] * 2
    )
    env.agent.error = RuntimeError("backend down again")
    for _ in range(TURN_FORK_FAILURE_LIMIT):
        await run_turn(env)

    assert len(env.agent.calls) == forks_so_far + TURN_FORK_FAILURE_LIMIT


async def test_failure_after_a_watermark_move_restarts_the_count(make_env):  # noqa: F811 (make_env fixture)
    """The race the turn task's own reset can't see: the watermark moves
    after the task read it but before the next failure is recorded (a
    compaction committed meanwhile). The failure is counted against the new
    watermark, as the first of a fresh run, not the Nth of the old one."""
    env = await failing_env(make_env)
    rows = [(env.ids[0],), (env.ids[-1],)]
    # N-1 failures against watermark 0.
    for _ in range(TURN_FORK_FAILURE_LIMIT - 1):
        env.memory._note_turn_fork_failure(env.channel.id, rows, 0)
    # The next failure is read against a moved watermark.
    env.memory._note_turn_fork_failure(env.channel.id, rows, env.ids[0])

    failures = env.memory._turn_failures[env.channel.id]
    assert (failures.count, failures.watermark, failures.backoff_until) == (
        1, env.ids[0], None,
    )
