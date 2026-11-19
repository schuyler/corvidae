"""Tests for the heartbeat's goal review (#133 addendum, H2 goals half)."""

import hashlib
import json
import math
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from corvidae.channel import ChannelRegistry
from corvidae.context import ContextWindow
from corvidae.hooks import (
    create_plugin_manager,
    hookimpl,
    install_plugin_hookspecs,
    validate_dependencies,
)
from corvidae.llm_plugin import LLMPlugin
from corvidae.persistence import PersistencePlugin
from corvidae_cognition import heartbeat
from corvidae_cognition.funnel import FunnelPlugin
from corvidae_cognition import goals as goals_mod
from corvidae_cognition.goals import GoalsPlugin
from corvidae_cognition.heartbeat import HEARTBEAT_CHANNEL, HeartbeatPlugin
from corvidae_cognition import memory as memory_mod
from corvidae_cognition.memory import MemoryPlugin
from fake_clock import install_kernel_clock

AFTER = 600
DIMS = 32

CONFIG = {
    "llm": {
        "providers": {
            "main": {"base_url": "http://localhost:8080"},
            "embedding": {"base_url": "http://localhost:8081"},
        },
        "profiles": {
            "main": {"model": "chat"},
            "background": {"model": "bg"},
            "embedding": {
                "provider": "embedding",
                "kind": "embedding",
                "model": "test-embedder",
                "dimensions": DIMS,
            },
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
    async def on_notify(self, channel, source, text, tool_call_id, annotations):
        self.calls.append({"channel": channel, "text": text})
        self.tracker.note_activity(channel.id)
        if channel.conversation is None:
            channel.conversation = ContextWindow(channel.id)
        await self.pm.ahook.before_agent_turn(channel=channel, trace_id=None, annotations=annotations)
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
        monkeypatch.setattr(llm, "_create_client", lambda cfg, profile="main": StubEncoder())

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
        # Plugin-owned hookspecs (collect_heartbeat_prompts) are installed at startup.
        install_plugin_hookspecs(pm)
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
    assert goals_mod.GOALS_INSTRUCTION in text
    assert goals_mod.RETIRED_GOALS_HEADER in text
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
    assert goals_mod.GOALS_INSTRUCTION not in text
    assert goals_mod.RETIRED_GOALS_HEADER not in text


async def test_beat_obeys_retired_goal_limit(make_env, monkeypatch):
    """The beat reviews only the newest ``goals.heartbeat_retired_limit``
    retired goals; older conclusions stay out of the notification."""
    monkeypatch.setitem(CONFIG, "goals", {"heartbeat_retired_limit": 1})
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the two jobs")
    await env.tools["add_goal"]("The old job", _ctx=ctx)
    await env.tools["retire_goal"](1, "Old conclusion: it ran every night.", _ctx=ctx)
    await env.tools["add_goal"]("The new job", _ctx=ctx)
    await env.tools["retire_goal"](2, "New conclusion: the schedule holds.", _ctx=ctx)

    await env.clock.advance(AFTER)
    assert len(env.agent.calls) == 1
    text = env.agent.calls[0]["text"]
    # The conclusion can also surface in the memory section (a beat reviews
    # the whole memory, #168); the limit governs the retired-goals list, so
    # assert on that block only: from its header to the end of the text (goals'
    # section is last).
    retired_section = text[text.index(goals_mod.RETIRED_GOALS_HEADER):]
    assert "New conclusion: the schedule holds." in retired_section
    assert "Old conclusion: it ran every night." not in retired_section


# --- #808: the pick rule ------------------------------------------------------


def test_goals_instruction_states_the_pick_rule():
    """#808: resume an in-progress goal, else take the first goal that is
    neither blocked nor waiting; and the sentence must read as a sentence."""
    text = goals_mod.GOALS_INSTRUCTION
    assert "in-progress" in text
    assert "blocked" in text and "waiting" in text
    # A leftover fragment from an earlier wording ("... next. do it.").
    assert ". do it" not in text


# --- #1084: the beat is a bare prompt that plugins fill ------------------------


class RaisingContributor:
    """A plugin whose beat contribution fails; the beat must survive it."""

    @hookimpl
    async def collect_heartbeat_prompts(self, channel):
        raise RuntimeError("contributor blew up")


async def test_beat_is_prompt_then_memory_then_goals_section(make_env):
    """R7, #1079 AC1: through a real plugin manager with heartbeat, memory and
    goals, the delivered text is the prompt, memory's section (its own hook,
    tryfirst), then goals' retired block and instruction at the tail (the
    call to action is the last thing in the window)."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Get the wiki backup job running", _ctx=ctx)
    await env.tools["retire_goal"](1, "I concluded it is done, because cron runs.", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert text.startswith(heartbeat.DEFAULT_HEARTBEAT_PROMPT)
    positions = [
        text.index(memory_mod.MEMORY_HEADER),
        text.index(goals_mod.RETIRED_GOALS_HEADER),
        text.index(goals_mod.GOALS_INSTRUCTION),
    ]
    assert positions == sorted(positions)
    assert text.rstrip().endswith(goals_mod.GOALS_INSTRUCTION)


async def test_beat_does_not_render_the_open_goals_list(make_env):
    """R3: open goals reach the beat through the window's goals entry only;
    the beat's own text no longer lists them."""
    env = await make_env(with_goals=True)
    await env.tools["add_goal"]("Watch the kestrel nest", _ctx=ctx_for("irc:#test"))

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert "Watch the kestrel nest" not in text
    assert "Open goals" not in text


async def test_a_raising_contributor_is_skipped_and_the_others_kept(make_env, caplog):
    """R7: one plugin's failing collect_heartbeat_prompts is logged, naming
    the hook and the plugin, and the beat still carries the other sections."""
    env = await make_env(with_goals=True)
    env.pm.register(RaisingContributor(), name="flaky")

    with caplog.at_level("WARNING"):
        await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert memory_mod.MEMORY_HEADER in text
    assert goals_mod.GOALS_INSTRUCTION in text
    assert any(
        "collect_heartbeat_prompts" in r.getMessage() and "flaky" in r.getMessage()
        for r in caplog.records
    )


async def test_second_beat_with_nothing_new_omits_the_retired_block(make_env):
    """R7/R4: goals keeps its own only-what's-new state. Beat two repeats the
    instruction but not the conclusions already in the window."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Get the wiki backup job running", _ctx=ctx)
    await env.tools["retire_goal"](1, "I concluded it is done, because cron runs.", _ctx=ctx)

    await env.clock.advance(AFTER)
    await env.clock.advance(AFTER)
    first, second = (c["text"] for c in env.agent.calls)
    assert goals_mod.RETIRED_GOALS_HEADER in first
    assert goals_mod.RETIRED_GOALS_HEADER not in second
    assert "I concluded it is done" not in second
    assert memory_mod.MEMORY_HEADER not in second
    assert goals_mod.GOALS_INSTRUCTION in second


async def test_compaction_of_the_beat_channel_resets_goals_diff_state(make_env):
    """R4: after the beat channel compacts, goals sends its retired block
    in full again, because the summary may not have kept structured data."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Get the wiki backup job running", _ctx=ctx)
    await env.tools["retire_goal"](1, "I concluded it is done, because cron runs.", _ctx=ctx)

    await env.clock.advance(AFTER)
    channel = env.agent.calls[0]["channel"]
    await env.pm.ahook.on_compaction(
        channel=channel, summary_msg={}, retain_count=0, compacted_ids=[]
    )
    await env.clock.advance(AFTER)
    assert goals_mod.RETIRED_GOALS_HEADER in env.agent.calls[1]["text"]
    # Another channel's compaction doesn't reset it: a third beat is diffed.
    await env.pm.ahook.on_compaction(
        channel=SimpleNamespace(id="irc:#test"), summary_msg={}, retain_count=0, compacted_ids=[]
    )
    await env.clock.advance(AFTER)
    assert goals_mod.RETIRED_GOALS_HEADER not in env.agent.calls[2]["text"]


async def test_beat_with_no_sections_is_the_bare_prompt(make_env):
    """R2: no goals, and nothing new in memory on beat two: no sections come
    back, so the beat is exactly heartbeat.prompt (no 'nothing new' text)."""
    env = await make_env(with_goals=False)

    await env.clock.advance(AFTER)
    await env.clock.advance(AFTER)
    assert env.agent.calls[1]["text"] == heartbeat.DEFAULT_HEARTBEAT_PROMPT


async def _goals_limit_seen(config: dict) -> int:
    """Init goals on ``config`` and return the limit its beat section asks
    for, read off the call to retired_goals."""
    plugin = GoalsPlugin()
    await plugin.on_init(MagicMock(), config)
    seen: list[int] = []

    async def retired(limit):
        seen.append(limit)
        return []

    plugin.retired_goals = retired
    await plugin.collect_heartbeat_prompts(channel=None)
    return seen[0]


@pytest.mark.parametrize("bad", [-1, 2.5, True, "20"])
async def test_goals_rejects_a_bad_heartbeat_retired_limit(bad):
    """#1079 AC6: a negative, fractional, boolean or string
    goals.heartbeat_retired_limit fails goals' startup, naming the key, the
    value, what is accepted and where to fix it."""
    with pytest.raises(ValueError) as exc:
        await GoalsPlugin().on_init(
            MagicMock(), {"goals": {"heartbeat_retired_limit": bad}}
        )
    message = str(exc.value)
    assert "goals.heartbeat_retired_limit" in message
    assert f"got {bad!r}" in message
    assert "0 or more" in message
    assert "agent.yaml" in message


async def test_heartbeat_retired_limit_defaults_to_five_and_accepts_zero():
    """#1079 AC5: absent or null is 5; 0 is a valid count (reviews none)."""
    assert await _goals_limit_seen({}) == 5
    assert await _goals_limit_seen({"goals": {"heartbeat_retired_limit": None}}) == 5
    assert await _goals_limit_seen({"goals": {"heartbeat_retired_limit": 0}}) == 0
    assert await _goals_limit_seen({"goals": {"heartbeat_retired_limit": 7}}) == 7


async def test_goals_heartbeat_instruction_override_is_what_the_hook_returns():
    """#1079 AC5: goals.heartbeat_instruction sets the instruction goals
    contributes."""
    plugin = GoalsPlugin()
    await plugin.on_init(MagicMock(), {"goals": {"heartbeat_instruction": "PICK ONE."}})

    async def none_retired(limit):
        return []

    plugin.retired_goals = none_retired
    section = await plugin.collect_heartbeat_prompts(channel=None)
    assert section.endswith("PICK ONE.")
    assert goals_mod.GOALS_INSTRUCTION not in section


async def test_later_beat_sends_only_the_new_memory_and_retired_goal(make_env):
    """R4/R5: between beats one memory arrives and one goal is retired; beat
    two carries those and leaves out what beat one already sent."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Old job", _ctx=ctx)
    await env.tools["retire_goal"](1, "OLDCONCLUSION it ran nightly.", _ctx=ctx)
    await env.clock.advance(AFTER)
    await env.tools["add_goal"]("New job", _ctx=ctx)
    await env.tools["retire_goal"](2, "NEWCONCLUSION the schedule holds.", _ctx=ctx)

    await env.clock.advance(AFTER)
    first, second = (c["text"] for c in env.agent.calls)
    assert "OLDCONCLUSION" in first
    # Beat two: the new conclusion (retired block and its memory record) is
    # present once each; the old one is not repeated anywhere.
    # The new memory record reaches beat two through the heartbeat's own
    # memory section (before goals' block), not only through goals' retired
    # block: a memory section dropped after beat one would fail here.
    assert "NEWCONCLUSION" in second[
        second.index(memory_mod.MEMORY_HEADER) : second.index(goals_mod.RETIRED_GOALS_HEADER)
    ]
    assert "OLDCONCLUSION" not in second
    # Joined by blank lines: pins the separator on the real path.
    assert f"\n\n{goals_mod.RETIRED_GOALS_HEADER}" in second
