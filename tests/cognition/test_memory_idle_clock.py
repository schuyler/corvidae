"""Memory's idle consolidation and retention run on the kernel clock (#165).

Covers R1, R2, R4, R8 and R9a for the memory plugin. Every test drives a
FakeClock and never sleeps real time (R7). After each ``advance`` the tests
await ``memory.wait_for_background_tasks()`` before asserting, because
``advance`` only waits for clock-recorded tasks and not for the tasks memory
spawns through ``_spawn``. All times are whole seconds so sums are exact.
"""

import json
import logging
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.attribution import get_attribution
from corvidae.channel import Channel
from corvidae.context import MessageType
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae_cognition import retention as retention_module
from corvidae_cognition import memory as memory_module
from corvidae_cognition.memory import MemoryPlugin
from fake_clock import FakeClock, install_kernel_clock

CONSOLIDATION_JSON = json.dumps({
    "summary": "I helped kestrel debug the weather station's wifi dropouts.",
    "topic_tags": ["esp32", "wifi"],
    "participants": ["kestrel"],
})

PERIOD = 60
INTERVAL = 3600
MAX_DELAY = 600

CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "background": {"base_url": "http://localhost:8080", "model": "bg"},
        "embedding": {
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": 4,
        },
    },
    "memory": {
        "idle_consolidate_after": PERIOD,
        "retention": {"interval": INTERVAL, "max_delay": MAX_DELAY},
    },
}


class StubClient:
    """Minimal LLMClient stand-in that answers chat and embed calls."""

    def __init__(self):
        self.chat_calls: list[dict] = []

    async def chat(self, messages, tools=None, extra_body=None):
        self.chat_calls.append({"attribution": dict(get_attribution())})
        return {"choices": [{"message": {
            "role": "assistant", "content": CONSOLIDATION_JSON,
        }}]}

    async def embed(self, texts, kind=None):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class Env:
    """Handles to the wired plugins a test needs."""

    def __init__(self, memory, persistence, stub, clock, tracker, db, channel):
        self.memory = memory
        self.persistence = persistence
        self.stub = stub
        self.clock = clock
        self.tracker = tracker
        self.db = db
        self.channel = channel
        self.t0 = clock.now()

    async def seed(self, channel=None) -> list[int]:
        """Persist a short dialog on the channel; returns the rowids."""
        channel = channel or self.channel
        rowids = []
        for role, text in [("user", "old talk"), ("assistant", "old reply")]:
            rowids.append(await self.persistence.persist_message(
                channel=channel,
                message={"role": role, "content": text},
                message_type=MessageType.MESSAGE,
            ))
        return rowids

    async def advance(self, seconds: float) -> None:
        """Advance the fake clock, then let memory's spawned tasks finish."""
        await self.clock.advance(seconds)
        await self.memory.wait_for_background_tasks()

    async def memories(self) -> list[tuple[str, int]]:
        """Return (channel_id, msg_id_end) for every memory row."""
        async with self.db.execute(
            "SELECT channel_id, msg_id_end FROM memory ORDER BY id"
        ) as cursor:
            return [tuple(r) for r in await cursor.fetchall()]


@pytest.fixture
def retention_runs(monkeypatch) -> list[float]:
    """Replace the retention job with a recorder of its ``now`` argument."""
    runs: list[float] = []

    async def fake_job(memory, now=None):
        runs.append(now)

    monkeypatch.setattr(retention_module, "run_retention_job", fake_job)
    return runs


@pytest.fixture
async def make_env(retention_runs):
    """Factory wiring persistence, llm stub, a fake kernel clock and memory."""
    cleanups = []

    async def make(
        config=None, *, last_run=None, with_idle=True, pm=None, db=None,
        persistence=None, channel=None,
    ) -> Env:
        config = config or CONFIG
        if db is None:
            # A standalone plugin manager with an in-memory session DB.
            db = await aiosqlite.connect(":memory:")
            await init_db(db)
            pm = create_plugin_manager()
            persistence = PersistencePlugin()
            persistence.db = db
            pm.register(persistence, name="persistence")
            channel = Channel(transport="irc", scope="#test")
        cleanups.append(db.close)

        llm = LLMPlugin()
        pm.register(llm, name="llm")
        await llm.on_init(pm=pm, config=config)
        stub = StubClient()
        for role in ("main", "background", "embedding"):
            llm._clients[role] = stub
        llm.embedding_dimensions = 4

        if with_idle:
            clock, tracker = install_kernel_clock(pm)
        else:
            # The clock alone: memory must treat a missing tracker as "no kernel idle".
            clock, tracker = FakeClock(), None
            pm.register(clock, name="clock")
        cleanups.append(clock.close)

        memory = MemoryPlugin()
        pm.register(memory, name="memory")
        await memory.on_init(pm=pm, config=config)
        if last_run is not None:
            # Seed retention_meta before on_start so the startup check sees it.
            await memory._ensure_schema()
            await db.execute(
                "INSERT INTO retention_meta (last_run) VALUES (?)",
                (last_run(clock),),
            )
            await db.commit()
        await memory.on_start(config=config)
        cleanups.append(memory.on_stop)
        await memory.wait_for_background_tasks()
        prior = MagicMock()
        prior.score = AsyncMock(return_value=0.7)
        memory.importance_prior = prior
        return Env(memory, persistence, stub, clock, tracker, db, channel)

    yield make
    for cleanup in reversed(cleanups):
        await cleanup()


# ---------------------------------------------------------------------------
# Consolidation on per-channel idle (R1, R2)
# ---------------------------------------------------------------------------


async def test_consolidates_tail_period_after_channel_settles(make_env):
    """R1: a settled channel's tail is consolidated one period later, no traffic."""
    env = await make_env()
    cid = env.channel.id
    env.tracker.note_activity(cid)
    rowids = await env.seed()
    env.tracker.note_settled(cid)

    await env.advance(PERIOD)

    assert await env.memories() == [(cid, rowids[-1])]


async def test_no_consolidation_before_period(make_env):
    """R1: nothing is consolidated until the full period has passed."""
    env = await make_env()
    cid = env.channel.id
    env.tracker.note_activity(cid)
    await env.seed()
    env.tracker.note_settled(cid)

    await env.advance(PERIOD - 1)
    assert await env.memories() == []

    # One more second reaches the period, so the same setup does consolidate.
    await env.advance(1)
    assert len(await env.memories()) == 1


async def test_traffic_elsewhere_does_not_delay_consolidation(make_env):
    """R1: consolidation is keyed per channel; other channels' traffic is irrelevant."""
    env = await make_env()
    cid = env.channel.id
    env.tracker.note_activity(cid)
    rowids = await env.seed()
    env.tracker.note_settled(cid)

    # Another channel keeps the system busy right up to the period.
    await env.advance(PERIOD // 2)
    env.tracker.note_activity("irc:#other")
    await env.advance(PERIOD // 2 - 1)
    env.tracker.note_settled("irc:#other")
    await env.advance(1)

    assert await env.memories() == [(cid, rowids[-1])]


async def test_restored_tail_consolidated_after_restart_without_traffic(make_env):
    """R2: a tail seeded as quiet at boot is consolidated with no traffic at all."""
    env = await make_env()
    cid = env.channel.id
    rowids = await env.seed()
    # The last row is 30 s old; PersistencePlugin's start seeds the tracker from it.
    await env.db.execute(
        "UPDATE message_log SET timestamp = ?", (env.clock.now() - 30,)
    )
    await env.db.commit()
    await env.persistence.on_init(pm=env.memory.pm, config=CONFIG)
    await env.persistence.on_start(config=CONFIG)

    await env.advance(PERIOD - 30 - 1)
    assert await env.memories() == []

    await env.advance(1)
    assert await env.memories() == [(cid, rowids[-1])]


async def test_restored_channel_at_watermark_spawns_nothing(make_env):
    """R1: a quiet channel with nothing above the watermark spawns no consolidation."""
    env = await make_env()
    cid = env.channel.id
    rowids = await env.seed()
    # Consolidate everything first so the watermark covers the whole log.
    await env.memory._consolidate_range(cid, rowids[-1], "idle")
    assert len(await env.memories()) == 1
    chat_calls = len(env.stub.chat_calls)

    spawned = []
    env.memory._consolidate_range = lambda *a: spawned.append(a)
    env.tracker.note_restored(cid, env.clock.now() - PERIOD)
    await env.clock.advance(0)
    await env.advance(PERIOD)

    assert spawned == []
    assert len(env.stub.chat_calls) == chat_calls
    assert len(await env.memories()) == 1


async def test_repeat_does_not_double_spawn_inflight_consolidation(make_env):
    """R1: a repeat firing while a consolidation runs spawns no second one."""
    import asyncio

    env = await make_env()
    cid = env.channel.id
    await env.seed()
    gate = asyncio.Event()
    real = env.memory._consolidate_range
    spawned = []

    def gated(*args, **kwargs):
        # Count at spawn time; the coroutine blocks until the gate opens.
        spawned.append(args)

        async def run():
            await gate.wait()
            await real(*args, **kwargs)

        return run()

    env.memory._consolidate_range = gated
    env.tracker.note_restored(cid, env.clock.now() - PERIOD)
    await env.clock.advance(0)
    assert len(spawned) == 1

    # Two more repeats land while the first consolidation is still blocked.
    await env.clock.advance(PERIOD * 2)
    assert len(spawned) == 1

    gate.set()
    await env.memory.wait_for_background_tasks()


# ---------------------------------------------------------------------------
# Retention on the clock (R4, R9a)
# ---------------------------------------------------------------------------


async def test_retention_runs_each_interval_without_traffic_or_scheduler(
    make_env, retention_runs,
):
    """R4: retention runs every interval with no traffic and no scheduler loaded."""
    env = await make_env()
    assert env.memory.pm.get_plugin("scheduler") is None
    # The startup check already ran the job once.
    # Memory ran the check once; when it ran is not pinned.
    assert len(retention_runs) == 1

    for k in range(1, 4):
        await env.advance(INTERVAL)
        assert retention_runs == [env.t0 + j * INTERVAL for j in range(k + 1)]


async def test_retention_restart_within_interval_waits_for_due(make_env, retention_runs):
    """R4: a recent persisted last_run skips startup and the run waits for last_run + interval."""
    env = await make_env(last_run=lambda clock: clock.now() - 1000)
    assert retention_runs == []

    await env.advance(INTERVAL - 1000 - 1)
    assert retention_runs == []

    await env.advance(1)
    assert retention_runs == [env.t0 + INTERVAL - 1000]


async def test_retention_prefers_system_quiet(make_env, retention_runs):
    """R9a: when due, retention waits for the system to be quiet RETENTION_QUIET_SECONDS."""
    env = await make_env()
    retention_runs.clear()
    # A busy channel keeps the system from being quiet at the due time.
    env.tracker.note_activity("irc:#busy")
    await env.advance(INTERVAL + 100)
    assert retention_runs == []

    env.tracker.note_settled("irc:#busy")
    await env.advance(59)
    assert retention_runs == []

    await env.advance(1)
    assert retention_runs == [env.t0 + INTERVAL + 160]


async def test_retention_runs_by_max_delay_under_constant_traffic(
    make_env, retention_runs,
):
    """R9a: activity every 5 s never quiets the system, yet retention runs by due + max_delay."""
    env = await make_env()
    retention_runs.clear()
    due = env.t0 + INTERVAL

    # Activity and settle every 5 s keep the system quiet time under 60 s.
    for _ in range((INTERVAL + MAX_DELAY) // 5 + 4):
        env.tracker.note_activity("irc:#busy")
        env.tracker.note_settled("irc:#busy")
        await env.advance(5)
        if retention_runs:
            break

    assert len(retention_runs) == 1
    assert due + MAX_DELAY <= retention_runs[0] <= due + MAX_DELAY + 5


async def test_idle_upkeep_enqueues_no_turn(make_env, retention_runs):
    """R8: consolidation and retention fire without queueing work or notifying.

    A fork is allowed, so this does not assert the LLM was never called.
    """
    from helpers import build_plugin_and_channel

    agent, channel, db = await build_plugin_and_channel()
    pm = agent.pm
    persistence = pm.get_plugin("persistence")
    pm.ahook.on_notify = AsyncMock()
    env = await make_env(pm=pm, db=db, persistence=persistence, channel=channel)

    try:
        env.tracker.note_activity(channel.id)
        await env.seed()
        env.tracker.note_settled(channel.id)
        await env.advance(PERIOD)
        await env.advance(INTERVAL - PERIOD)

        # Both upkeep paths ran: one consolidation, and the startup plus due retention.
        assert len(await env.memories()) == 1
        assert len(retention_runs) == 2
        # Neither started a turn: nothing queued, nothing notified.
        assert all(q.is_empty for q in agent.queues.values())
        assert pm.get_plugin("task").task_queue.is_idle
        pm.ahook.on_notify.assert_not_called()
    finally:
        await pm.get_plugin("task").on_stop()


async def test_retention_write_failure_retries_at_bounded_rate(
    make_env, retention_runs, caplog,
):
    """R9a: a failing last_run write retries no faster than RETENTION_DB_RETRY_SECONDS.

    Guards the ``now + RETENTION_DB_RETRY_SECONDS`` floor: without it a
    ``last_run`` that never advances arms in the past and loops at zero delay.
    The tracker starts at t0, so the system is not quiet (60 s) at first and
    retention would not re-arm; the test first advances 60 s so the system is
    already quiet, resets the counters, then counts over a clean window.
    """
    attempts = []

    class FailingWrites:
        """Connection proxy: reads work, retention_meta writes raise."""

        def __init__(self, real):
            self._real = real

        def execute(self, sql, *args, **kwargs):
            if "retention_meta" in sql and sql.lstrip().upper().startswith(
                ("UPDATE", "INSERT")
            ):
                attempts.append(sql)
                raise sqlite3.OperationalError("disk I/O error")
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = FailingWrites(db)
    pm.register(persistence, name="persistence")
    caplog.set_level(logging.WARNING)
    env = await make_env(
        pm=pm, db=db, persistence=persistence,
        channel=Channel(transport="irc", scope="#test"),
    )

    # Get past the quiet period so retention is eligible to re-arm at once.
    await env.advance(60)
    # Count only attempts and warnings from the now-quiet window.
    attempts.clear()
    caplog.clear()

    # One second at a time, so a zero-delay retry loop would show as one attempt per step.
    for _ in range(60):
        await env.advance(1)

    limit = 60 // int(memory_module.RETENTION_DB_RETRY_SECONDS) + 1
    warnings = [
        r for r in caplog.records
        if "retention startup trigger failed" in r.getMessage()
    ]
    # Without the floor this is about 60 attempts; with it, at most 13.
    assert 1 <= len(attempts) <= limit
    assert len(warnings) == len(attempts)


async def test_clock_without_idle_tracker_warns_once_and_arms_nothing(
    make_env, retention_runs, caplog,
):
    """Memory uses the clock and tracker together or not at all.

    With a clock but no tracker it warns once, arms no timers, and still runs
    the startup retention check.
    """
    caplog.set_level(logging.WARNING)
    env = await make_env(with_idle=False)

    warnings = [
        r for r in caplog.records
        if "no kernel clock/idle tracker registered" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert env.clock.live_timer_names() == []
    # Memory ran the check once; when it ran is not pinned.
    assert len(retention_runs) == 1


# ---------------------------------------------------------------------------
# Startup deferral log (#175)
# ---------------------------------------------------------------------------


async def test_retention_arm_before_db_opens_logs_one_plain_warning(
    make_env, caplog
):
    """Memory's on_start runs before persistence opens the DB: the retention
    arm defers with one plain WARNING (no traceback), then arms once open."""
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    pm = create_plugin_manager()
    persistence = PersistencePlugin()  # DB not open yet, as at boot
    pm.register(persistence, name="persistence")
    channel = Channel(transport="irc", scope="#test")

    with caplog.at_level(logging.DEBUG, logger="corvidae.memory"):
        env = await make_env(pm=pm, db=db, persistence=persistence, channel=channel)
        # Persistence opens the DB; the re-arm then reads retention_meta.
        persistence.db = db
        await env.advance(memory_module.RETENTION_DB_RETRY_SECONDS)

    warnings = [r for r in caplog.records
                if r.name == "corvidae.memory" and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert warnings[0].levelno == logging.WARNING
    assert warnings[0].exc_info is None
    message = warnings[0].getMessage()
    assert "retention" in message and "not open" in message, message
    assert f"{memory_module.RETENTION_DB_RETRY_SECONDS}s" in message, message
    assert env.memory._retention_timer is not None
    assert env.memory._retention_timer.name == "memory.retention"


async def test_retention_meta_read_failure_keeps_traceback(make_env, caplog):
    """A real failure (DB open, reading retention_meta fails) still logs a
    WARNING with its traceback, and re-arms."""
    env = await make_env()
    real_execute = env.db.execute

    # Fail only the retention_meta read the arm performs.
    def failing_execute(sql, *args, **kwargs):
        if "FROM retention_meta" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return real_execute(sql, *args, **kwargs)

    env.db.execute = failing_execute
    try:
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await env.memory._arm_retention()
    finally:
        env.db.execute = real_execute

    warnings = [r for r in caplog.records
                if r.name == "corvidae.memory" and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert warnings[0].levelno == logging.WARNING
    assert warnings[0].exc_info is not None
    assert env.memory._retention_timer.name == "memory.retention.rearm"
