"""Tests for the heartbeat's goal review (#133 addendum, H2 goals half).

Red phase: tests 1 and 3 fail with
    AttributeError: module 'corvidae_cognition.heartbeat' has no attribute
    'GOALS_INSTRUCTION'
because the constants and the goals wiring in _beat do not exist yet.
"""

import hashlib
import json
import math
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corvidae.channel import ChannelRegistry
from corvidae.context import ContextWindow
from corvidae.hooks import create_plugin_manager, hookimpl, validate_dependencies
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import PersistencePlugin
from corvidae_cognition import heartbeat
from corvidae_cognition.funnel import FunnelPlugin
from corvidae_cognition.goals import GoalsPlugin
from corvidae_cognition.heartbeat import HEARTBEAT_CHANNEL, HeartbeatPlugin
from corvidae_cognition.memory import MemoryPlugin
from fake_clock import install_kernel_clock

AFTER = 600
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
    "heartbeat": {"after": AFTER},
}


def bow_embed(text: str) -> list[float]:
    vec = [0.0] * DIMS
    for token in re.findall(r"\w+", text.lower()):
        vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class StubEncoder:
    async def embed(self, texts, kind=None):
        return [bow_embed(t) for t in texts]

    async def chat(self, messages, tools=None, extra_body=None):
        raise AssertionError("heartbeat goals tests must not call chat()")

    async def start(self):
        pass

    async def stop(self):
        pass


class AgentStub:
    """Records on_notify calls and, like the Agent (agent.py:896), runs
    before_agent_turn on the channel through the plugin manager."""

    def __init__(self, pm, tracker) -> None:
        self.pm = pm
        self.tracker = tracker
        self.calls: list[dict] = []

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, meta):
        self.calls.append({"channel": channel, "text": text})
        self.tracker.note_activity(channel.id)
        if channel.conversation is None:
            channel.conversation = ContextWindow(channel.id)
        await self.pm.ahook.before_agent_turn(channel=channel, trace_id=None, meta=meta)
        self.tracker.note_settled(channel.id)


@pytest.fixture
async def make_env(tmp_path, monkeypatch):
    cleanups = []

    async def make(*, with_goals: bool) -> SimpleNamespace:
        config = {**CONFIG, "_base_dir": str(tmp_path)}
        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        pm.register(persistence, name="persistence")
        pm.register(ChannelRegistry(), name="registry")

        llm = LLMPlugin()
        pm.register(llm, name="llm")
        await llm.on_init(pm=pm, config=config)
        monkeypatch.setattr(llm, "_create_client", lambda cfg, role="main": StubEncoder())

        clock, tracker = install_kernel_clock(pm)
        cleanups.append(clock.close)
        cleanups.append(tracker.stop)

        agent = AgentStub(pm, tracker)
        pm.register(agent, name="agent")

        funnel = FunnelPlugin()
        pm.register(funnel, name="funnel")
        await funnel.on_init(pm=pm, config=config)

        memory = MemoryPlugin()
        pm.register(memory, name="memory")
        await memory.on_init(pm=pm, config=config)

        goals = None
        if with_goals:
            goals = GoalsPlugin(pm)
            pm.register(goals, name="goals")
            await goals.on_init(pm=pm, config=config)

        hb = HeartbeatPlugin()
        pm.register(hb, name="heartbeat")
        validate_dependencies(pm)
        await hb.on_init(pm=pm, config=config)

        await pm.ahook.on_start(config=config)
        cleanups.append(persistence.db.close)
        cleanups.append(memory.on_stop)
        cleanups.append(hb.on_stop)

        registry: list = []
        pm.hook.register_tools(tool_registry=registry)
        tools = {t.name: t.fn for t in registry}
        return SimpleNamespace(
            pm=pm, clock=clock, agent=agent, tools=tools, db=persistence.db,
        )

    yield make
    for cleanup in reversed(cleanups):
        result = cleanup()
        if hasattr(result, "__await__"):
            await result


async def add_message(env, channel_id: str, content: str) -> None:

    await env.db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, ?, ?, ?)",
        (channel_id, json.dumps({"role": "user", "content": content}),
         env.clock.now(), "message"),
    )
    await env.db.commit()


def ctx_for(channel_id: str):
    ctx = MagicMock()
    ctx.channel.id = channel_id
    return ctx


async def test_beat_with_a_retired_goal_carries_instruction_header_and_record(make_env):
    """AC 1: with goals loaded and one retired goal, the beat's text carries
    GOALS_INSTRUCTION, RETIRED_GOALS_HEADER and the retirement record."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Get the wiki backup job running", _ctx=ctx)
    await env.tools["retire_goal"](1, "I concluded it is done, because cron runs.", _ctx=ctx)

    await env.clock.advance(AFTER)
    assert len(env.agent.calls) == 1
    text = env.agent.calls[0]["text"]
    assert heartbeat.GOALS_INSTRUCTION in text
    assert heartbeat.RETIRED_GOALS_HEADER in text
    assert "I concluded it is done, because cron runs." in text


async def test_beat_turn_carries_the_open_goals_entry(make_env):
    """AC 2 (regression pin; may already pass): a beat's notification turn on
    internal:heartbeat, through the plugin manager, gets the goals CONTEXT
    entry in that channel's window."""
    env = await make_env(with_goals=True)
    await env.tools["add_goal"]("Watch the kestrel nest", _ctx=ctx_for("irc:#test"))

    await env.clock.advance(AFTER)
    assert len(env.agent.calls) == 1
    channel = env.agent.calls[0]["channel"]
    assert channel.id == HEARTBEAT_CHANNEL
    entries = [
        m["content"]
        for m in channel.conversation.messages
        if m.get("role") == "system"
        and m.get("content", "").startswith("[CONTEXT from goals")
    ]
    assert len(entries) == 1
    assert "Watch the kestrel nest" in entries[0]


async def test_beat_without_goals_plugin_carries_no_goals_text(make_env):
    """AC 3: goals not registered; a beat is still sent, with no goals text."""
    env = await make_env(with_goals=False)

    await env.clock.advance(AFTER)
    assert len(env.agent.calls) == 1
    text = env.agent.calls[0]["text"]
    assert heartbeat.DEFAULT_HEARTBEAT_PROMPT in text
    assert heartbeat.GOALS_INSTRUCTION not in text
    assert heartbeat.RETIRED_GOALS_HEADER not in text
