"""Tests for the goals plugin (#133): G1 tools and seed, G2 window entry, G3 retirement record.

Red phase: every test fails at collection with
    ModuleNotFoundError: No module named 'corvidae_cognition.goals'
because corvidae_cognition/goals.py does not exist yet.
"""

import asyncio
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corvidae.channel import Channel, ChannelRegistry
from corvidae.context import ContextWindow
from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import PersistencePlugin
from corvidae_cognition.funnel import FunnelPlugin
from corvidae_cognition.goals import GoalsPlugin, goals_command
from corvidae_cognition.memory import MemoryPlugin
from fake_clock import install_kernel_clock

DIMS = 32

CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "background": {"base_url": "http://localhost:8080", "model": "bg"},
        "embedding": {
            "kind": "embedding",
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": DIMS,
        },
    },
}


def bow_embed(text: str) -> list[float]:
    vec = [0.0] * DIMS
    for token in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class StubEncoder:
    async def embed(self, texts, kind=None):
        return [bow_embed(t) for t in texts]

    async def chat(self, messages, tools=None, extra_body=None):
        raise AssertionError("goals tests must not call chat()")

    async def start(self):
        pass

    async def stop(self):
        pass


def utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


@pytest.fixture
async def env(tmp_path, monkeypatch):
    """persistence (WAL file DB) + stub llm + clock + funnel + memory + goals."""
    db_path = str(tmp_path / "sessions.db")
    config = {**CONFIG, "_base_dir": str(tmp_path)}

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    pm.register(persistence, name="persistence")

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=CONFIG)
    encoder = StubEncoder()
    # LLMPlugin.on_start builds one client per profile; hand it the stub.
    monkeypatch.setattr(llm, "_create_client", lambda cfg, profile="main": encoder)

    clock, idle = install_kernel_clock(pm)

    funnel = FunnelPlugin()
    pm.register(funnel, name="funnel")
    await funnel.on_init(pm=pm, config=CONFIG)

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=CONFIG)

    goals = GoalsPlugin(pm)
    pm.register(goals, name="goals")
    await goals.on_init(pm=pm, config=CONFIG)

    # Production order: goals' on_start runs before persistence opens the DB
    # (on_start is broadcast LIFO), so goals must create its table lazily.
    await pm.ahook.on_start(config=config)
    db = persistence.db
    assert llm._config.profiles["embedding"]["dimensions"] == DIMS

    registry: list = []
    pm.hook.register_tools(tool_registry=registry)
    tools = {t.name: t.fn for t in registry}

    channel = Channel(transport="irc", scope="#test")
    channel.conversation = ContextWindow(channel.id)

    # Registry + a recording on_notify: the goals plugin notifies the origin
    # channel on state changes; the test captures the calls.
    channel_registry = ChannelRegistry()
    channel_registry.channels[channel.id] = channel
    pm.register(channel_registry, name="registry")
    notifications: list[dict] = []

    class _NotifyRecorder(CorvidaePlugin):
        @hookimpl
        async def on_notify(self, channel, source, text, tool_call_id, annotations):
            notifications.append(
                {"channel": channel.id, "source": source, "text": text}
            )

    pm.register(_NotifyRecorder())

    yield SimpleNamespace(
        pm=pm, db=db, db_path=db_path, clock=clock, memory=memory,
        goals=goals, tools=tools, channel=channel,
        registry=channel_registry, notifications=notifications,
    )

    await memory.on_stop()
    idle.stop()
    await clock.close()
    await db.close()


def ctx_for(channel):
    ctx = MagicMock()
    ctx.channel = channel
    return ctx


async def turn(env, channel=None):
    """Run before_agent_turn through the plugin manager."""
    channel = channel or env.channel
    await env.pm.ahook.before_agent_turn(channel=channel, trace_id=None, annotations={})


def goals_messages(channel) -> list[str]:
    return [
        m["content"]
        for m in channel.conversation.messages
        if m.get("role") == "system"
        and m.get("content", "").startswith("[CONTEXT from goals")
    ]


async def rows(env, sql: str, params=()) -> list[tuple]:
    async with env.db.execute(sql, params) as cursor:
        return await cursor.fetchall()


async def add_message(env, channel_id: str, content: str) -> int:
    cursor = await env.db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, ?, ?, ?)",
        (channel_id, json.dumps({"role": "user", "content": content}),
         env.clock.now(), "message"),
    )
    await env.db.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------


async def test_g1_tools_round_trip(env):
    """add_goal -> update_goal -> retire_goal; the goals row reflects each step."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "talking about the wiki backup")
    t0 = env.clock.now()

    await env.tools["add_goal"]("Get the wiki backup job running", _ctx=ctx)
    row = (await rows(
        env, "SELECT id, text, created_at, updated_at, retired_at, memory_id FROM goals"
    ))[0]
    assert row[1] == "Get the wiki backup job running"
    assert row[2] == row[3] == t0
    assert row[4] is None and row[5] is None
    gid = row[0]

    env.clock.jump(60)
    await env.tools["update_goal"](gid, "Get the wiki backup running nightly", _ctx=ctx)
    row = (await rows(
        env, "SELECT text, created_at, updated_at, retired_at FROM goals WHERE id = ?",
        (gid,),
    ))[0]
    assert row == ("Get the wiki backup running nightly", t0, t0 + 60, None)

    env.clock.jump(60)
    await env.tools["retire_goal"](gid, "I concluded it is done, because cron runs.", _ctx=ctx)
    row = (await rows(
        env, "SELECT retired_at, updated_at, memory_id FROM goals WHERE id = ?", (gid,)
    ))[0]
    assert row[0] == row[1] == t0 + 120
    assert row[2] is not None


async def test_g1_seed_cli_appears_on_any_channel(env):
    """`corvidae goals --db PATH seed X` in a second process; the next turn on any channel shows X."""
    from click.testing import CliRunner

    result = await asyncio.to_thread(
        CliRunner().invoke, goals_command,
        ["--db", env.db_path, "seed", "Watch the kestrel nest"],
    )
    assert result.exit_code == 0, (result.output, result.exception)

    other = Channel(transport="discord", scope="general")
    other.conversation = ContextWindow(other.id)
    await turn(env, other)

    entries = goals_messages(other)
    assert len(entries) == 1
    assert "Watch the kestrel nest" in entries[0]


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------


async def test_g2_supersession_and_empty_list(env):
    """One entry per change; the last entry is current; retiring the last goal appends 'No open goals'."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "context")

    t0 = env.clock.now()
    await env.tools["add_goal"]("Write the goals plugin", _ctx=ctx)
    await turn(env)
    first = goals_messages(env.channel)
    assert len(first) == 1
    assert f"Open goals as of {utc(t0)} (this list replaces any earlier goals list):" in first[0]
    assert "#1 Write the goals plugin" in first[0]

    env.clock.jump(60)
    await env.tools["update_goal"](1, "Write and test the goals plugin", _ctx=ctx)
    await turn(env)
    entries = goals_messages(env.channel)
    assert len(entries) == 2
    latest = entries[-1]
    assert f"as of {utc(t0 + 60)}" in latest
    assert "this list replaces any earlier goals list" in latest
    assert "#1 Write and test the goals plugin" in latest
    assert "#1 Write the goals plugin" not in latest

    env.clock.jump(60)
    await env.tools["retire_goal"](1, "I concluded it shipped, because tests pass.", _ctx=ctx)
    await turn(env)
    entries = goals_messages(env.channel)
    assert len(entries) == 3
    assert (
        f"No open goals as of {utc(t0 + 120)} (this replaces any earlier goals list)."
        in entries[-1]
    )
    assert "#1" not in entries[-1]


# ---------------------------------------------------------------------------
# G3 deterministic
# ---------------------------------------------------------------------------


async def test_g3_retirement_writes_memory_record(env):
    """retire_goal writes the summary-format memory row, importance 1.0, channel, and sets memory_id."""
    ctx = ctx_for(env.channel)
    msg_id = await add_message(env, env.channel.id, "kestrel nest evidence")
    t0 = env.clock.now()
    await env.tools["add_goal"]("Find the kestrel nest", _ctx=ctx)

    env.clock.jump(86400 * 2)
    conclusion = "I concluded the nest is on Brannan Street, because kestrels were seen carrying twigs."
    await env.tools["retire_goal"](1, conclusion, _ctx=ctx)

    (memory_id,) = (await rows(env, "SELECT memory_id FROM goals WHERE id = 1"))[0]
    assert memory_id is not None
    row = (await rows(
        env,
        "SELECT channel_id, created_at, summary, importance, msg_id_start, msg_id_end "
        "FROM memory WHERE id = ?",
        (memory_id,),
    ))[0]
    assert row[0] == env.channel.id
    assert row[1] == env.clock.now()
    assert row[2] == f'I retired my goal "Find the kestrel nest" (set {day(t0)}). {conclusion}'
    assert row[3] == 1.0
    assert row[4] == row[5] == msg_id


async def test_g3_retirement_record_is_retrievable(env):
    """Under a stub encoder, retrieve(evidence) returns the retirement record."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "kestrel nest evidence")
    await env.tools["add_goal"]("Find the kestrel nest", _ctx=ctx)
    await env.tools["retire_goal"](
        1, "I concluded the kestrel nest is on Brannan Street, because twigs.", _ctx=ctx
    )
    (memory_id,) = (await rows(env, "SELECT memory_id FROM goals WHERE id = 1"))[0]

    candidates, _degraded = await env.memory.retrieve(
        "kestrels carrying twigs to the nest on Brannan Street"
    )
    assert memory_id in [c["id"] for c in candidates]


# -- list_goals tests -------------------------------------------------------


async def test_list_goals_open(env):
    """list_goals returns open goals with ids and dates."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("Find the kestrel nest", _ctx=ctx)
    await env.tools["add_goal"]("Write the blog post", _ctx=ctx)
    result = await env.tools["list_goals"](_ctx=ctx)
    assert "#1" in result
    assert "#2" in result
    assert "Find the kestrel nest" in result
    assert "Write the blog post" in result
    assert "[open since" in result


async def test_list_goals_empty(env):
    """list_goals with no goals returns 'No open goals.'"""
    ctx = ctx_for(env.channel)
    result = await env.tools["list_goals"](_ctx=ctx)
    assert result == "No open goals."


async def test_list_goals_exclude_retired_by_default(env):
    """list_goals excludes retired goals by default."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "context")
    await env.tools["add_goal"]("Open goal", _ctx=ctx)
    await env.tools["add_goal"]("To be retired", _ctx=ctx)
    await env.tools["retire_goal"](2, "Done.", _ctx=ctx)
    result = await env.tools["list_goals"](_ctx=ctx)
    assert "#1" in result
    assert "#2" not in result
    assert "To be retired" not in result


async def test_list_goals_include_retired(env):
    """list_goals(include_retired=True) includes retired goals."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "context")
    await env.tools["add_goal"]("Open goal", _ctx=ctx)
    await env.tools["add_goal"]("To be retired", _ctx=ctx)
    await env.tools["retire_goal"](2, "Done.", _ctx=ctx)
    result = await env.tools["list_goals"](include_retired=True, _ctx=ctx)
    assert "#1" in result
    assert "#2" in result
    assert "[retired" in result


async def test_list_goals_truncates_long_text(env):
    """list_goals truncates goal text longer than 120 chars."""
    ctx = ctx_for(env.channel)
    long_text = "A" * 200
    await env.tools["add_goal"](long_text, _ctx=ctx)
    result = await env.tools["list_goals"](_ctx=ctx)
    # Truncated to 117 chars + "..."
    assert "..." in result
    # The full 200-char text should NOT appear
    assert long_text not in result


# ---------------------------------------------------------------------------
# G4: priority queue (prepend/append), in-progress, origin-channel, notify
# ---------------------------------------------------------------------------


async def test_g4_prepend_goes_to_front(env):
    """A prepended goal sorts before appended ones (lower position = front)."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("second", _ctx=ctx)
    await env.tools["add_goal"]("third", _ctx=ctx)
    await env.tools["add_goal"]("first", position="prepend", _ctx=ctx)

    data = await rows(
        env, "SELECT text, position FROM goals ORDER BY position ASC, id ASC"
    )
    assert [t for t, _ in data] == ["first", "second", "third"]


async def test_g4_append_is_default_and_ordered(env):
    """Append (the default) puts new goals at the back, in order."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("a", _ctx=ctx)
    await env.tools["add_goal"]("b", _ctx=ctx)
    data = await rows(
        env, "SELECT text, position FROM goals ORDER BY position ASC, id ASC"
    )
    assert [t for t, _ in data] == ["a", "b"]
    assert data[0][1] < data[1][1]


async def test_g4_position_validation(env):
    """An invalid position is rejected; no row is written."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("valid", _ctx=ctx)  # creates the table
    result = await env.tools["add_goal"]("x", position="sideways", _ctx=ctx)
    assert "position must be" in result
    assert (await rows(env, "SELECT COUNT(*) FROM goals"))[0][0] == 1


async def test_g4_in_progress_is_a_valid_status(env):
    """in-progress is accepted (the WIP state)."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("wip", _ctx=ctx)
    gid = (await rows(env, "SELECT id FROM goals"))[0][0]
    result = await env.tools["set_goal_status"](gid, "in-progress", _ctx=ctx)
    assert "status set to" in result
    status = (await rows(env, "SELECT status FROM goals WHERE id = ?", (gid,)))[0][0]
    assert status == "in-progress"


async def test_g4_origin_channel_captured(env):
    """The goal records the channel that created it."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("from signal", _ctx=ctx)
    origin = (await rows(env, "SELECT origin_channel FROM goals"))[0][0]
    assert origin == env.channel.id


async def test_g4_state_change_notifies_origin(env):
    """A status change notifies the origin channel (source='goals')."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("watch this", _ctx=ctx)
    gid = (await rows(env, "SELECT id FROM goals"))[0][0]
    env.notifications.clear()
    await env.tools["set_goal_status"](gid, "in-progress", _ctx=ctx)
    assert len(env.notifications) == 1
    n = env.notifications[0]
    assert n["channel"] == env.channel.id
    assert n["source"] == "goals"
    assert f"Goal #{gid}" in n["text"] and "in-progress" in n["text"]


async def test_g4_retire_notifies_origin(env):
    """Retiring a goal notifies the origin channel with the conclusion."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "working on it")
    await env.tools["add_goal"]("do the thing", _ctx=ctx)
    gid = (await rows(env, "SELECT id FROM goals"))[0][0]
    env.notifications.clear()
    await env.tools["retire_goal"](gid, "I concluded it is done.", _ctx=ctx)
    assert len(env.notifications) == 1
    n = env.notifications[0]
    assert n["channel"] == env.channel.id
    assert n["source"] == "goals"
    assert f"Goal #{gid} retired" in n["text"]


async def test_g4_notify_dropped_when_origin_not_registered(env):
    """A goal whose origin channel isn't registered: state recorded, notify
    dropped (no crash, nothing sent)."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("orphan", _ctx=ctx)
    gid = (await rows(env, "SELECT id FROM goals"))[0][0]
    await env.db.execute(
        "UPDATE goals SET origin_channel = ? WHERE id = ?",
        ("signal:does-not-exist", gid),
    )
    await env.db.commit()
    env.notifications.clear()
    result = await env.tools["set_goal_status"](gid, "blocked", _ctx=ctx)
    assert "status set to" in result
    assert env.notifications == []
    # The state was recorded even though the notify was dropped.
    assert (await rows(env, "SELECT status FROM goals WHERE id = ?", (gid,)))[0][0] == "blocked"


async def test_g4_open_goals_ordered_by_position(env):
    """open_goals() returns the queue front-first (position ASC)."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("low", _ctx=ctx)
    await env.tools["add_goal"]("high", position="prepend", _ctx=ctx)
    goals = await env.goals.open_goals()
    assert [g["summary"] for g in goals] == ["high", "low"]


# ---------------------------------------------------------------------------
# G5: review-loop test gaps (migration, notify edges, multi-prepend, no-op)
# ---------------------------------------------------------------------------


async def test_g5_migration_adds_columns_and_preserves_order(env):
    """A pre-migration table (no position/origin_channel) gains the columns;
    position = id (id order preserved), origin_channel is NULL."""
    # Create the table first (lazy), then rebuild it without the new columns
    # to simulate the old schema.
    await env.goals._require_db()
    await env.db.execute("ALTER TABLE goals RENAME TO goals_old")
    await env.db.execute(
        "CREATE TABLE goals (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
        "created_at REAL NOT NULL, updated_at REAL NOT NULL, retired_at REAL, "
        "memory_id INTEGER, status TEXT NOT NULL DEFAULT 'open')"
    )
    now = env.clock.now()
    for i, text in enumerate(["first", "second", "third"]):
        await env.db.execute(
            "INSERT INTO goals (id, text, created_at, updated_at, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            (i + 1, text, now, now),
        )
    await env.db.commit()
    env.goals._schema_ready = False  # force the migration to run
    db = await env.goals._require_db()
    async with db.execute("PRAGMA table_info(goals)") as cur:
        cols = {row[1] for row in await cur.fetchall()}
    assert "position" in cols and "origin_channel" in cols
    data = await rows(
        env, "SELECT id, text, position, origin_channel FROM goals ORDER BY id"
    )
    assert [d[1] for d in data] == ["first", "second", "third"]
    assert [d[2] for d in data] == [1, 2, 3]  # position = id
    assert all(d[3] is None for d in data)  # origin_channel NULL
    goals = await env.goals.open_goals()
    assert [g["summary"] for g in goals] == ["first", "second", "third"]


async def test_g5_add_goal_no_ctx_no_origin_no_notify(env):
    """add_goal with no channel context: origin_channel is NULL, and a later
    state change does not notify (no origin to notify)."""
    await env.tools["add_goal"]("no origin", _ctx=None)
    assert (await rows(env, "SELECT origin_channel FROM goals"))[0][0] is None
    gid = (await rows(env, "SELECT id FROM goals"))[0][0]
    env.notifications.clear()
    await env.tools["set_goal_status"](gid, "blocked", _ctx=None)
    assert env.notifications == []


async def test_g5_retire_already_retired_does_not_notify(env):
    """Retiring an already-retired goal errors and does not notify."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "work")
    await env.tools["add_goal"]("do it", _ctx=ctx)
    gid = (await rows(env, "SELECT id FROM goals"))[0][0]
    await env.tools["retire_goal"](gid, "done.", _ctx=ctx)
    env.notifications.clear()
    result = await env.tools["retire_goal"](gid, "again.", _ctx=ctx)
    assert "already retired" in result
    assert env.notifications == []


async def test_g5_multiple_prepends_preserve_order(env):
    """Multiple prepends: each new prepend goes to the very front."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("a", _ctx=ctx)
    await env.tools["add_goal"]("c", position="prepend", _ctx=ctx)
    await env.tools["add_goal"]("b", position="prepend", _ctx=ctx)
    data = await rows(env, "SELECT text FROM goals ORDER BY position ASC, id ASC")
    assert [d[0] for d in data] == ["b", "c", "a"]


async def test_g5_noop_status_change_does_not_notify(env):
    """Setting the same status does not fire a notification."""
    ctx = ctx_for(env.channel)
    await env.tools["add_goal"]("watch", _ctx=ctx)
    gid = (await rows(env, "SELECT id FROM goals"))[0][0]
    await env.tools["set_goal_status"](gid, "blocked", _ctx=ctx)
    env.notifications.clear()
    await env.tools["set_goal_status"](gid, "blocked", _ctx=ctx)  # no-op
    assert env.notifications == []
