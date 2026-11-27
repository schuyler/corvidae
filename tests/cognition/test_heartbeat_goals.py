"""Tests for the heartbeat's goal review (#133 addendum, H2 goals half).

The 2026-11-19 beat-size spec (Schuyler): a beat carries the frontmost
open or in-progress goals and the pick rule, and nothing else about goals —
no retired conclusions, no blocked or waiting goals - and no mention of them
either (owner ruling 2026-11-19). They stay on the board
(list_goals) but off the beat.
"""

import hashlib
import json
import math
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiosqlite
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


        # Memory reports on the trace protocol plugin's hook; Runtime's
        # install_plugin_hookspecs below installs its specs.
        from corvidae_cognition.trace import TracePlugin
        pm.register(TracePlugin(pm), name="trace")
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
        # Plugin-owned hookspecs (on_heartbeat_collect_prompts) are installed at startup.
        install_plugin_hookspecs(pm)
        validate_dependencies(pm)
        await hb.on_init(pm=pm, config=config)

        await pm.ahook.on_start(config=config)
        # A read-write autocommit seeding connection on the same file: the
        # plugins read on persistence.db (read-only) and write through the
        # route, so seeds and raw writes go through this one.
        seed = await aiosqlite.connect(
            str(tmp_path / "sessions.db"), isolation_level=None
        )
        cleanups.append(persistence.writer.close)
        cleanups.append(persistence.db.close)
        cleanups.append(seed.close)
        cleanups.append(memory.on_stop)
        cleanups.append(hb.on_stop)

        registry: list = []
        pm.hook.register_tools(tool_registry=registry)
        tools = {t.name: t.fn for t in registry}
        return SimpleNamespace(
            pm=pm, clock=clock, agent=agent, tools=tools, db=seed,
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


async def test_beat_with_a_retired_goal_carries_the_instruction_but_not_the_record(make_env):
    """The 2026-11-19 spec: retired goals never reach a beat. The beat still
    carries the pick rule, and the goals section is exactly the list followed
    by the instruction - no block between them."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Get the wiki backup job running", _ctx=ctx)
    await env.tools["add_goal"]("Water the ferns", _ctx=ctx)
    await env.tools["retire_goal"](1, "I concluded it is done, because cron runs.", _ctx=ctx)

    await env.clock.advance(AFTER)
    assert len(env.agent.calls) == 1
    text = env.agent.calls[0]["text"]
    assert goals_mod.GOALS_INSTRUCTION in text
    assert text.count(goals_mod.GOALS_INSTRUCTION) == 1
    # The goals section: list block, one blank line, instruction. Nothing else.
    idx_open = text.index("Open or in-progress goals as of")
    idx_inst = text.index(goals_mod.GOALS_INSTRUCTION)
    assert text.count("\n\n", idx_open, idx_inst) == 1
    # The retired goal is off the list too (#2 shows, #1 doesn't).
    assert "#2 Water the ferns" in text
    assert "Get the wiki backup job running" not in text[idx_open:]


async def test_beat_turn_admits_no_goals_context_entry(make_env):
    """#1091: the beat's turn on internal:heartbeat, through the plugin
    manager, gets the goals in the beat text and no goals CONTEXT entry."""
    env = await make_env(with_goals=True)
    await env.tools["add_goal"]("Watch the kestrel nest", _ctx=ctx_for("irc:#test"))

    await env.clock.advance(AFTER)
    assert len(env.agent.calls) == 1
    channel = env.agent.calls[0]["channel"]
    assert channel.id == HEARTBEAT_CHANNEL
    assert not [
        m for m in channel.conversation.messages
        if m.get("content", "").startswith("[CONTEXT from goals")
    ]
    assert "Watch the kestrel nest" in env.agent.calls[0]["text"]



async def test_beat_without_goals_plugin_carries_no_goals_text(make_env):
    """AC 3: goals not registered; a beat is still sent, with no goals text."""
    env = await make_env(with_goals=False)

    await env.clock.advance(AFTER)
    assert len(env.agent.calls) == 1
    text = env.agent.calls[0]["text"]
    assert heartbeat.DEFAULT_HEARTBEAT_PROMPT in text
    assert goals_mod.GOALS_INSTRUCTION not in text
    assert "Open or in-progress goals as of" not in text


# --- the 2026-11-19 beat-size spec: status filter and the five-goal cap -------


async def test_beat_shows_only_pursuable_goals(make_env):
    """Blocked and waiting goals never reach the beat; open and in-progress
    do, and in-progress keeps its tag for the pick rule."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await env.tools["add_goal"]("Alpha task", _ctx=ctx)
    await env.tools["add_goal"]("Beta task", _ctx=ctx)
    await env.tools["add_goal"]("Gamma task", _ctx=ctx)
    await env.tools["add_goal"]("Delta task", _ctx=ctx)
    await env.tools["set_goal_status"](2, "in-progress", _ctx=ctx)
    await env.tools["set_goal_status"](3, "blocked", _ctx=ctx)
    await env.tools["set_goal_status"](4, "waiting", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert "#1 Alpha task" in text
    assert "#2 [in-progress] Beta task" in text
    assert "Gamma task" not in text
    assert "Delta task" not in text
    assert "(+" not in text  # nothing hidden beyond the cap


async def test_beat_lists_only_the_five_frontmost_pursuable(make_env):
    """Seven open goals, default cap 5: the beat carries the frontmost in queue order,
    and names how many it left out so the board's size stays visible."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    for name in ("one", "two", "three", "four", "five", "six", "seven"):
        await env.tools["add_goal"](f"Task {name}", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    prev = -1
    for name in ("one", "two", "three", "four", "five"):
        assert f"Task {name}" in text, f"Task {name} missing from the beat"
        idx = text.index(f"Task {name}")
        assert idx > prev, f"Task {name} out of queue order"  # frontmost = order, not just membership
        prev = idx
    assert "Task six" not in text
    assert "Task seven" not in text
    assert "(+2 more" in text


async def test_blocked_and_retired_goals_do_not_inflate_the_hidden_count(make_env):
    """Seven pursuable plus a blocked and a retired goal: hidden counts only
    pursuable (7-5=2), and the off-beat goals' texts stay off the beat."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "tidying the board")
    for name in ("one", "two", "three", "four", "five", "six", "seven"):
        await env.tools["add_goal"](f"Task {name}", _ctx=ctx)
    await env.tools["add_goal"]("Task blocked", _ctx=ctx)
    await env.tools["add_goal"]("Task retired", _ctx=ctx)
    # (ids: seven=7, blocked=8, retired=9 - the off-by-one here once made
    # this test pass for the wrong reasons; reviewer gate caught it.)
    await env.tools["set_goal_status"](8, "blocked", _ctx=ctx)
    await env.tools["retire_goal"](9, "Done elsewhere.", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    idx = text.index("Open or in-progress goals as of")
    goals_block = text[idx:text.index(goals_mod.GOALS_INSTRUCTION)]
    assert "Task blocked" not in goals_block
    assert "Task retired" not in goals_block
    # 7 pursuable, cap 5: exactly 2 hidden - neither extra goal counts.
    assert "(+2 more pursuable" in goals_block
    assert "(+3 more" not in text and "(+4 more" not in text


async def test_beat_honors_a_smaller_heartbeat_limit(make_env, monkeypatch):
    """The cap is configurable: goals.heartbeat_limit=2 shows exactly two and
    names the other two."""
    monkeypatch.setitem(CONFIG, "goals", {"heartbeat_limit": 2})
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    for name in ("one", "two", "three", "four"):
        await env.tools["add_goal"](f"Task {name}", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert "Task one" in text and "Task two" in text
    assert "Task three" not in text and "Task four" not in text
    assert "(+2 more" in text


async def test_beat_with_limit_zero_shows_no_goals_but_names_them(make_env, monkeypatch):
    """goals.heartbeat_limit=0: the list header survives, no goals show, and
    the count line carries the board's size."""
    monkeypatch.setitem(CONFIG, "goals", {"heartbeat_limit": 0})
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await env.tools["add_goal"]("Task one", _ctx=ctx)
    await env.tools["add_goal"]("Task two", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert "Task one" not in text and "Task two" not in text
    idx = text.index("Open or in-progress goals as of")
    assert "(+2 more pursuable - call list_goals)" in text[idx:]
    assert goals_mod.GOALS_INSTRUCTION in text


@pytest.mark.parametrize("dead_key", [("goals", "heartbeat_retired_limit"),
                                      ("heartbeat", "retired_goal_limit")])
async def test_removed_retired_limit_keys_fail_startup(make_env, dead_key):
    """The beat's retired-goals review is gone; its keys fail loudly rather
    than configure nothing (removed-key precedent, channel.py)."""
    block, key = dead_key
    CONFIG.setdefault(block, {})[key] = 3
    try:
        with pytest.raises(ValueError, match=re.escape(f"{block}.{key} was removed")):
            await make_env(with_goals=True)
    finally:
        CONFIG[block].pop(key, None)


async def test_heartbeat_goal_limit_is_not_a_cap_alias():
    """heartbeat.goal_limit never shipped, so it is NOT a deprecated alias:
    it configures nothing, silently - the cap stays at its default."""
    assert await _beat_limit_seen({"heartbeat": {"goal_limit": 2}}) == 5


async def test_blocked_goals_become_visible_as_the_board_frees_slots(make_env):
    """The cap counts pursuable goals only: with the front goal blocked, the
    sixth goal moves into the five shown slots."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    for name in ("one", "two", "three", "four", "five", "six"):
        await env.tools["add_goal"](f"Task {name}", _ctx=ctx)
    await env.tools["set_goal_status"](1, "blocked", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert "Task one" not in text
    for name in ("two", "three", "four", "five", "six"):
        assert f"Task {name}" in text
    assert "(+" not in text


async def test_beat_with_only_blocked_goals_says_so_without_naming_them(make_env):
    """When nothing pursuable exists, the beat line replaces the list and
    counts the hidden goals without listing them."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await env.tools["add_goal"]("Stuck task", _ctx=ctx)
    await env.tools["add_goal"]("Patient task", _ctx=ctx)
    await env.tools["set_goal_status"](1, "blocked", _ctx=ctx)
    await env.tools["set_goal_status"](2, "waiting", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    # The beat does not count or name them either (owner ruling).
    assert "No open goals as of" in text
    line = text.split(goals_mod.GOALS_INSTRUCTION)[0].split("Open goals")[-1]
    assert "blocked" not in line and "waiting" not in line
    assert "Stuck task" not in text
    assert "Patient task" not in text
    assert goals_mod.GOALS_INSTRUCTION in text


# --- #808: the pick rule ------------------------------------------------------


def test_goals_instruction_states_the_pick_rule():
    """#808 + the 2026-11-19 spec: resume an in-progress goal, else take the
    first listed goal; nothing off the beat is mentioned (owner ruling); and the
    sentence must read as a sentence."""
    text = goals_mod.GOALS_INSTRUCTION
    assert "in-progress" in text
    # Owner ruling 2026-11-19: the pick rule must not point the beat at
    # anything off the beat.
    assert "list_goals" not in text
    assert "front of the queue" in text
    # The count belongs to the cap (goals.heartbeat_limit), not the rule:
    # the instruction stays true whatever the limit is set to.
    assert "five" not in text
    # A leftover fragment from an earlier wording ("... next. do it.").
    assert ". do it" not in text


# --- #1084: the beat is a bare prompt that plugins fill ------------------------


class RaisingContributor:
    """A plugin whose beat contribution fails; the beat must survive it."""

    @hookimpl
    async def on_heartbeat_collect_prompts(self, channel):
        raise RuntimeError("contributor blew up")


async def test_beat_is_memory_then_goals_sections_then_prompt(make_env):
    """R7, #1079 AC1, #1094 req 5: through a real plugin manager with
    heartbeat, memory and goals, the delivered text is memory's section (its
    own hook, tryfirst), then goals' list and instruction, then the heartbeat
    prompt last (the call to action is the last thing in the window,
    and says \"above\")."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Get the wiki backup job running", _ctx=ctx)

    await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert text.endswith(heartbeat.DEFAULT_HEARTBEAT_PROMPT)
    positions = [
        text.index(memory_mod.MEMORY_HEADER),
        text.index("Open or in-progress goals as of"),
        text.index(goals_mod.GOALS_INSTRUCTION),
        text.index(heartbeat.DEFAULT_HEARTBEAT_PROMPT),
    ]
    assert positions == sorted(positions)
    # Sections are separated from the prompt by one blank line.
    assert text.endswith(f"\n\n{heartbeat.DEFAULT_HEARTBEAT_PROMPT}")


async def test_beat_lists_the_front_of_the_board_in_full_every_time(make_env):
    """#1091 + the 2026-11-19 spec: the beat is the one place goals reach the
    agent, and each beat carries the (capped, filtered) list in full, ahead
    of the pick rule - a stale copy can't outlive a change."""
    env = await make_env(with_goals=True)
    await env.tools["add_goal"]("Watch the kestrel nest", _ctx=ctx_for("irc:#test"))

    await env.clock.advance(AFTER)
    await env.clock.advance(AFTER)
    for call in env.agent.calls:
        text = call["text"]
        assert "#1 Watch the kestrel nest" in text
        assert text.index("Open or in-progress goals as of") < text.index(goals_mod.GOALS_INSTRUCTION)


async def test_a_raising_contributor_is_skipped_and_the_others_kept(make_env, caplog):
    """R7: one plugin's failing on_heartbeat_collect_prompts is logged, naming
    the hook and the plugin, and the beat still carries the other sections."""
    env = await make_env(with_goals=True)
    env.pm.register(RaisingContributor(), name="flaky")

    with caplog.at_level("WARNING"):
        await env.clock.advance(AFTER)
    text = env.agent.calls[0]["text"]
    assert memory_mod.MEMORY_HEADER in text
    assert goals_mod.GOALS_INSTRUCTION in text
    assert text.endswith(heartbeat.DEFAULT_HEARTBEAT_PROMPT)
    assert any(
        "on_heartbeat_collect_prompts" in r.getMessage() and "flaky" in r.getMessage()
        for r in caplog.records
    )


async def test_second_beat_repeats_the_list_and_never_a_retired_block(make_env):
    """The list is full on every beat (no diff state), and the retired block
    that the beat used to carry is gone from every beat. Conclusions stay
    reachable as memories, not as a goals block."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Durable task", _ctx=ctx)
    await env.tools["add_goal"]("Get the wiki backup job running", _ctx=ctx)
    await env.tools["retire_goal"](2, "I concluded it is done, because cron runs.", _ctx=ctx)

    await env.clock.advance(AFTER)
    await env.clock.advance(AFTER)
    first, second = (c["text"] for c in env.agent.calls)
    for text in (first, second):
        assert "#1 Durable task" in text
        assert goals_mod.GOALS_INSTRUCTION in text
        # The retired goal is off the goals list; its conclusion may surface
        # in memory's section (a retirement writes a memory record) - that's
        # memory's business, not the beat's goals block.
        assert "Get the wiki backup job running" not in text[text.index("Open or in-progress goals as of"):]
    # Beat two repeats the list; the beat-one conclusion is not repeated.
    assert "Durable task" in second
    assert "I concluded it is done" not in second


async def test_beat_with_no_sections_is_the_bare_prompt(make_env):
    """R2: no goals, and nothing new in memory on beat two: no sections come
    back, so the beat is exactly heartbeat.prompt (no 'nothing new' text)."""
    env = await make_env(with_goals=False)

    await env.clock.advance(AFTER)
    await env.clock.advance(AFTER)
    assert env.agent.calls[1]["text"] == heartbeat.DEFAULT_HEARTBEAT_PROMPT


# --- beat-size config: goals.heartbeat_limit ----------------------------------


async def _beat_limit_seen(config: dict) -> int:
    """Init goals on ``config`` and return the beat-list cap it read."""
    plugin = GoalsPlugin()
    await plugin.on_init(MagicMock(), config)
    return plugin._beat_goal_limit


@pytest.mark.parametrize("bad", [-1, 2.5, True, "20"])
async def test_goals_rejects_a_bad_heartbeat_limit(bad):
    """A negative, fractional, boolean or string goals.heartbeat_limit fails
    goals' startup, naming the key, the value, what is accepted and where to
    fix it."""
    with pytest.raises(ValueError) as exc:
        await GoalsPlugin().on_init(
            MagicMock(), {"goals": {"heartbeat_limit": bad}}
        )
    message = str(exc.value)
    assert "goals.heartbeat_limit" in message
    assert f"got {bad!r}" in message
    assert "0 or more" in message
    assert "agent.yaml" in message


async def test_heartbeat_limit_defaults_to_five_and_accepts_zero():
    """Absent or null is 5; 0 is a valid count (shows none)."""
    assert await _beat_limit_seen({}) == 5
    assert await _beat_limit_seen({"goals": {"heartbeat_limit": None}}) == 5
    assert await _beat_limit_seen({"goals": {"heartbeat_limit": 0}}) == 0
    assert await _beat_limit_seen({"goals": {"heartbeat_limit": 7}}) == 7


async def test_goals_heartbeat_instruction_override_is_what_the_hook_returns():
    """#1079 AC5: goals.heartbeat_instruction sets the instruction goals
    contributes."""
    plugin = GoalsPlugin()
    await plugin.on_init(MagicMock(), {"goals": {"heartbeat_instruction": "PICK ONE."}})

    async def no_list():
        return None

    plugin._goals_list_text = no_list  # no DB behind this plugin
    section = await plugin.on_heartbeat_collect_prompts(channel=None)
    assert section.endswith("PICK ONE.")
    assert goals_mod.GOALS_INSTRUCTION not in section


async def test_later_beat_sends_only_the_new_memory_not_a_retired_block(make_env):
    """R4/R5: between beats one memory arrives and one goal is retired; beat
    two carries the new memory - through memory's own section - and there is
    no goals retired block anywhere."""
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("Old job", _ctx=ctx)
    await env.tools["retire_goal"](1, "OLDCONCLUSION it ran nightly.", _ctx=ctx)
    await env.clock.advance(AFTER)
    await env.tools["add_goal"]("Live job", _ctx=ctx)
    await env.tools["retire_goal"](2, "NEWCONCLUSION the schedule holds.", _ctx=ctx)

    await env.clock.advance(AFTER)
    first, second = (c["text"] for c in env.agent.calls)
    assert "OLDCONCLUSION" in first
    # Beat two: the new conclusion appears once, in memory's section, which
    # runs from its header to the goals list. A memory section dropped after
    # beat one would fail here.
    assert second.count("NEWCONCLUSION") == 1
    # Bounded by the pick rule: everything before it is memory's section plus
    # the (conclusion-free) goals block, and only memory carries conclusions.
    assert "NEWCONCLUSION" in second[
        second.index(memory_mod.MEMORY_HEADER) : second.index(goals_mod.GOALS_INSTRUCTION)
    ]
    assert "OLDCONCLUSION" not in second
    # Joined by blank lines: pins the separator on the real path. Both goals
    # are retired by beat two, so the replacement line is the none-open one.
    assert "\n\nNo open goals as of" in second


async def test_beat_omits_a_goal_retired_before_the_status_fix(make_env):
    """#1119: the beat keys on `retired_at`, so an unrepaired row stays off it.

    A row retired before the fix has `retired_at` set and `status='open'`. If the
    beat's filter read `status` instead, that row would come back as live work
    and the repair command would be load-bearing for correctness.
    """
    env = await make_env(with_goals=True)
    ctx = ctx_for("irc:#test")
    await add_message(env, "irc:#test", "talking about the wiki backup")
    await env.tools["add_goal"]("An old goal retired before the fix", _ctx=ctx)
    await env.db.execute(
        "UPDATE goals SET retired_at = ?, status = 'open' WHERE id = 1",
        (env.clock.now(),),
    )
    await env.db.commit()

    await env.clock.advance(AFTER)
    assert len(env.agent.calls) == 1
    text = env.agent.calls[0]["text"]
    assert "An old goal retired before the fix" not in text
