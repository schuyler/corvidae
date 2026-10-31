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

from corvidae.channel import Channel
from corvidae.context import ContextWindow
from corvidae.hooks import create_plugin_manager
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
    # LLMPlugin.on_start builds one client per role; hand it the stub.
    monkeypatch.setattr(llm, "_create_client", lambda cfg, role="main": encoder)

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
    assert llm.embedding_dimensions == DIMS

    registry: list = []
    pm.hook.register_tools(tool_registry=registry)
    tools = {t.name: t.fn for t in registry}

    channel = Channel(transport="irc", scope="#test")
    channel.conversation = ContextWindow(channel.id)

    yield SimpleNamespace(
        pm=pm, db=db, db_path=db_path, clock=clock, memory=memory,
        goals=goals, tools=tools, channel=channel,
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
    await env.pm.ahook.before_agent_turn(channel=channel, trace_id=None, meta={})


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
    """Under a stub encoder, retrieve(channel, evidence) returns the retirement record."""
    ctx = ctx_for(env.channel)
    await add_message(env, env.channel.id, "kestrel nest evidence")
    await env.tools["add_goal"]("Find the kestrel nest", _ctx=ctx)
    await env.tools["retire_goal"](
        1, "I concluded the kestrel nest is on Brannan Street, because twigs.", _ctx=ctx
    )
    (memory_id,) = (await rows(env, "SELECT memory_id FROM goals WHERE id = 1"))[0]

    candidates, _degraded = await env.memory.retrieve(
        env.channel.id, "kestrels carrying twigs to the nest on Brannan Street"
    )
    assert memory_id in [c["id"] for c in candidates]
