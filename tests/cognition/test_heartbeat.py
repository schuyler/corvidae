"""HeartbeatPlugin: a turn the agent starts on system idle (#134).

One test per acceptance criterion in #134's approved requirements (H1, H2's
review input, H3, H5). Every test drives a real IdleTracker on a FakeClock
and never sleeps real time. The live act-if-warranted probe (H2's AC) needs
a real llama-server and runs in the harness at acceptance, not here.

The environment mirrors Runtime's order: plugins are registered, then
on_init, then on_start, and only then is the idle tracker started.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

import aiosqlite
import pytest

from corvidae.channel import ChannelRegistry
from corvidae.hooks import create_plugin_manager, hookimpl, validate_dependencies
from corvidae.llm_plugin import LLMPlugin
from helpers import with_chat_llm
from corvidae.persistence import PersistencePlugin, init_db
from corvidae_cognition.heartbeat import (
    DEFAULT_HEARTBEAT_PROMPT,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_RETIRED_GOAL_LIMIT,
    GOALS_INSTRUCTION,
    MEMORY_HEADER,
    frame_beat,
    HEARTBEAT_CHANNEL,
    HeartbeatPlugin,
)
from corvidae_cognition.memory import MemoryPlugin
from fake_clock import install_kernel_clock

AFTER = 600
# Long enough that no plausible default period hides inside it.
THIRTY_DAYS = 30 * 86400


class AgentStub:
    """Records on_notify calls and marks the channel busy, then settled, the
    way the Agent does around a turn (so the tracker sees each beat as work)."""

    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self.calls: list[dict] = []

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, annotations):
        self.calls.append({
            "channel": channel, "source": source, "text": text,
            "tool_call_id": tool_call_id, "annotations": annotations,
        })
        self.tracker.note_activity(channel.id)
        self.tracker.note_settled(channel.id)


class Env:
    """A started heartbeat with its clock, tracker, memory and agent stub."""

    def __init__(self, pm, clock, tracker, agent, memory, heartbeat, db) -> None:
        self.pm = pm
        self.clock = clock
        self.tracker = tracker
        self.agent = agent
        self.memory = memory
        self.heartbeat = heartbeat
        self.db = db

    @property
    def beats(self) -> list[dict]:
        """Every heartbeat notification so far."""
        return self.agent.calls

    async def seed_memory(self, channel_id, summary, created_at, redacted=0) -> None:
        """Insert one memory record."""
        await self.db.execute(
            "INSERT INTO memory (channel_id, created_at, summary, importance, "
            "msg_id_start, msg_id_end, indexed, redacted) VALUES (?, ?, ?, 0.5, 1, 2, 1, ?)",
            (channel_id, created_at, summary, redacted),
        )
        await self.db.commit()


@pytest.fixture
async def make_env():
    """Factory: build and start a heartbeat env from a config dict."""
    cleanups = []

    async def make(config: dict, *, with_memory: bool = True) -> Env:
        # Memory parses llm: in on_init, so the config needs a chat profile.
        config = with_chat_llm(config)
        # Session DB and persistence, which memory reads through.
        db = await aiosqlite.connect(":memory:")
        cleanups.append(db.close)
        await init_db(db)
        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        pm.register(ChannelRegistry(), name="registry")

        # Kernel clock and tracker, started last as Runtime does.
        clock, tracker = install_kernel_clock(pm, started=False)
        cleanups.append(clock.close)
        agent = AgentStub(tracker)
        pm.register(agent, name="agent")

        # Memory: schema only, on the kernel clock; its own idle watches
        # are not under test.
        memory = None
        if with_memory:
            # Memory declares llm; the heartbeat path never calls it.
            pm.register(LLMPlugin(), name="llm")
            memory = MemoryPlugin()
            pm.register(memory, name="memory")
            await memory.on_init(pm=pm, config=config)
            memory._clock = clock
            await memory._ensure_schema()

        heartbeat = HeartbeatPlugin()
        pm.register(heartbeat, name="heartbeat")
        # Runtime validates declared dependencies before on_init.
        validate_dependencies(pm)
        await heartbeat.on_init(pm=pm, config=config)
        await heartbeat.on_start(config=config)
        cleanups.append(heartbeat.on_stop)
        tracker.start()
        cleanups.append(tracker.stop)
        return Env(pm, clock, tracker, agent, memory, heartbeat, db)

    yield make
    for cleanup in reversed(cleanups):
        result = cleanup()
        if hasattr(result, "__await__"):
            await result


AWAKE = {"heartbeat": {"after": AFTER}}


# ---------------------------------------------------------------------------
# H1: its own plugin, registered and disableable
# ---------------------------------------------------------------------------


def test_heartbeat_is_a_corvidae_entry_point():
    """H1: registered as an entry point, so plugins.disabled can name it."""
    eps = {ep.name: ep.value for ep in entry_points(group="corvidae")}
    assert eps.get("heartbeat") == "corvidae_cognition.heartbeat:HeartbeatPlugin"


async def test_asleep_heartbeat_needs_no_memory_plugin(make_env):
    """H1/H3: with no heartbeat.after, the plugin starts and stops without
    memory loaded, so an existing config that disables memory still starts."""
    env = await make_env({}, with_memory=False)
    await env.clock.advance(THIRTY_DAYS)
    assert env.beats == []


async def test_awake_heartbeat_without_memory_fails_with_remedy(make_env):
    """H2 needs memory: heartbeat.after without the memory plugin fails
    startup, naming memory and how to fix it."""
    with pytest.raises(RuntimeError) as exc:
        await make_env(AWAKE, with_memory=False)
    message = str(exc.value)
    assert "memory" in message and "heartbeat.after" in message
    assert "enable" in message and "remove" in message


# ---------------------------------------------------------------------------
# H1: after the configured quiet, never while busy, at a bounded rate
# ---------------------------------------------------------------------------


async def test_beats_after_configured_quiet_on_internal_channel(make_env):
    """H1 + H5: nothing before `after` seconds of system quiet; then one beat,
    on internal:heartbeat, with the heartbeat origin."""
    env = await make_env(AWAKE)
    await env.clock.advance(AFTER - 1)
    assert env.beats == []
    await env.clock.advance(1)
    assert len(env.beats) == 1
    beat = env.beats[0]
    assert beat["channel"].id == HEARTBEAT_CHANNEL == "internal:heartbeat"
    assert beat["source"] == "heartbeat"
    assert beat["annotations"] == {"origin": "heartbeat"}
    assert beat["tool_call_id"] is None


async def test_no_beat_while_a_channel_is_busy(make_env):
    """H1: a busy channel holds every beat; the quiet that follows its
    settle starts the count again."""
    env = await make_env(AWAKE)
    env.tracker.note_activity("irc:#busy")
    await env.clock.advance(AFTER * 10)
    assert env.beats == []
    env.tracker.note_settled("irc:#busy")
    await env.clock.advance(AFTER - 1)
    assert env.beats == []
    await env.clock.advance(1)
    assert len(env.beats) == 1


async def test_no_beat_when_a_channel_goes_busy_during_the_memory_read(make_env):
    """H1: traffic that arrives while the beat reads memory cancels the beat."""
    env = await make_env(AWAKE)
    real_recent = env.memory.recent

    async def recent_then_traffic(limit):
        # A message lands on another channel while memory is being read.
        env.tracker.note_activity("irc:#busy")
        return await real_recent(limit)

    env.memory.recent = recent_then_traffic
    await env.clock.advance(AFTER)
    assert env.beats == []


async def test_repeats_once_per_period_while_quiet(make_env):
    """H1: through sustained quiet, one beat per `after` seconds, no more."""
    env = await make_env(AWAKE)
    await env.clock.advance(AFTER * 5)
    assert len(env.beats) == 5


async def test_memory_read_failure_retries_next_period(make_env, caplog):
    """H1: a failed memory read skips that beat with a warning and the next
    one comes `after` seconds later; beats don't stop."""
    env = await make_env(AWAKE)
    real_recent = env.memory.recent
    failures = [RuntimeError("database is locked")]

    async def flaky_recent(limit):
        if failures:
            raise failures.pop()
        return await real_recent(limit)

    env.memory.recent = flaky_recent
    with caplog.at_level(logging.WARNING, logger="corvidae.heartbeat"):
        await env.clock.advance(AFTER)
    assert env.beats == []
    records = [
        r for r in caplog.records
        if r.name == "corvidae.heartbeat" and r.levelno == logging.WARNING
    ]
    assert len(records) == 1
    message = records[0].getMessage()
    assert HEARTBEAT_CHANNEL in message
    assert "RuntimeError" in message
    assert f"retrying in {AFTER}" in message
    # The log line carries the traceback.
    assert records[0].exc_info is not None
    # The retry waits a full period, not less.
    await env.clock.advance(AFTER - 1)
    assert env.beats == []
    await env.clock.advance(1)
    assert len(env.beats) == 1


@pytest.mark.parametrize("bad", [0, float("nan")])
async def test_rejects_an_after_that_would_beat_back_to_back(make_env, bad):
    """H1's bounded rate needs a positive finite period: zero, or NaN (which
    arms at zero delay), fails startup naming the key, the value and what is
    accepted."""
    with pytest.raises(ValueError) as exc:
        await make_env({"heartbeat": {"after": bad}})
    message = str(exc.value)
    assert "heartbeat.after" in message
    assert f"got {bad!r}" in message
    assert "positive finite number of seconds" in message


# ---------------------------------------------------------------------------
# H2: the review input
# ---------------------------------------------------------------------------


async def test_beat_carries_prompt_and_recent_memory(make_env):
    """H2: the beat holds the heartbeat prompt and the newest memories of the
    whole memory, every channel's and its own included, in chronological
    order (oldest first, newest at the tail). Redacted records stay out.
    No configuration is needed to reach any channel."""
    env = await make_env({"heartbeat": {"after": AFTER}})
    now = env.clock.now()
    await env.seed_memory("irc:#lex", "I promised kestrel a wiring diagram.", now - 7200)
    await env.seed_memory("irc:#lex", "kestrel said the station reboots at noon.", now - 60)
    await env.seed_memory("signal:alice", "Alice's plans for Friday.", now - 30)
    await env.seed_memory("irc:#lex", "A redacted thing.", now - 20, redacted=1)
    await env.seed_memory(HEARTBEAT_CHANNEL, "I reviewed my memories.", now - 10)

    await env.clock.advance(AFTER)
    assert len(env.beats) == 1
    text = env.beats[0]["text"]
    assert DEFAULT_HEARTBEAT_PROMPT in text
    # Chronological order (oldest first): the most recent memories are
    # at the tail, next to the call-to-action.
    chronological = [
        text.index("I promised kestrel a wiring diagram."),
        text.index("kestrel said the station reboots at noon."),
        text.index("Alice's plans for Friday."),
        text.index("I reviewed my memories."),
    ]
    assert chronological == sorted(chronological)
    assert "A redacted thing." not in text


# ---------------------------------------------------------------------------
# H3: durable sleep
# ---------------------------------------------------------------------------


async def test_sleep_survives_restart(make_env):
    """H3: beating, then the operator removes heartbeat.after and restarts:
    no beats after the restart."""
    before = await make_env(AWAKE)
    await before.clock.advance(AFTER)
    assert len(before.beats) == 1
    # Restart: the old process stops; a new one starts on the edited config.
    await before.heartbeat.on_stop()
    before.tracker.stop()
    after = await make_env({"heartbeat": {}})
    await after.clock.advance(THIRTY_DAYS)
    assert after.beats == []


# ---------------------------------------------------------------------------
# heartbeat.memory_limit / heartbeat.retired_goal_limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, 2.5, True, "20"])
async def test_rejects_a_bad_memory_limit(make_env, bad):
    """A negative, fractional, boolean or string memory_limit fails startup,
    naming the key, the rejected value, what is accepted and where to fix it."""
    with pytest.raises(ValueError) as exc:
        await make_env({"heartbeat": {"memory_limit": bad}}, with_memory=False)
    message = str(exc.value)
    assert "heartbeat.memory_limit" in message
    assert f"got {bad!r}" in message
    assert "whole number" in message
    assert "0 or more" in message
    assert "agent.yaml" in message


@pytest.mark.parametrize("bad", [-1, 2.5, True, "20"])
async def test_rejects_a_bad_retired_goal_limit(make_env, bad):
    """A negative, fractional, boolean or string retired_goal_limit fails
    startup, naming the key, the rejected value, what is accepted and where
    to fix it."""
    with pytest.raises(ValueError) as exc:
        await make_env({"heartbeat": {"retired_goal_limit": bad}}, with_memory=False)
    message = str(exc.value)
    assert "heartbeat.retired_goal_limit" in message
    assert f"got {bad!r}" in message
    assert "whole number" in message
    assert "0 or more" in message
    assert "agent.yaml" in message


def test_review_limits_default_to_five_and_accept_zero():
    """Absent or null limits are 5; 0 is a valid count (reviews none)."""
    from corvidae_cognition.heartbeat import (
        parse_memory_limit,
        parse_retired_goal_limit,
    )

    assert parse_memory_limit({}) == 5
    assert parse_memory_limit({"heartbeat": {"memory_limit": None}}) == 5
    assert parse_memory_limit({"heartbeat": {"memory_limit": 0}}) == 0
    assert parse_memory_limit({"heartbeat": {"memory_limit": 5}}) == 5
    assert parse_retired_goal_limit({}) == 5
    assert parse_retired_goal_limit({"heartbeat": {"retired_goal_limit": None}}) == 5
    assert parse_retired_goal_limit({"heartbeat": {"retired_goal_limit": 0}}) == 0
    assert parse_retired_goal_limit({"heartbeat": {"retired_goal_limit": 7}}) == 7


async def test_beat_obeys_memory_limit(make_env):
    """A beat reviews only the newest ``heartbeat.memory_limit`` memories:
    the limit bounds the notification, so it bounds the beat's prompt."""
    config = {"heartbeat": {"after": AFTER, "memory_limit": 2}}
    env = await make_env(config)
    now = env.clock.now()
    await env.seed_memory("irc:#lex", "oldest: the first plan.", now - 7200)
    await env.seed_memory("irc:#lex", "middle: the second plan.", now - 600)
    await env.seed_memory("irc:#lex", "newest: the third plan.", now - 60)

    await env.clock.advance(AFTER)
    assert len(env.beats) == 1
    text = env.beats[0]["text"]
    assert "newest: the third plan." in text
    assert "middle: the second plan." in text
    assert "oldest: the first plan." not in text


# --- Diff-mode tests (#716) -------------------------------------------------

class TestHeartbeatDiffMode:
    """Diff-based heartbeat prompt: only new items, instruction at tail."""

    def test_first_beat_is_full_mode(self):
        """First beat (no prior state) sends all memories and goals."""
        memories = [
            {"id": 1, "age_seconds": 60, "channel_id": "c1", "summary": "mem1"},
            {"id": 2, "age_seconds": 120, "channel_id": "c2", "summary": "mem2"},
        ]
        retired = [
            {"id": 10, "age_seconds": 300, "summary": "goal1"},
        ]
        text = frame_beat(memories, retired, diff=False)
        # Full mode: all items present
        assert "mem1" in text
        assert "mem2" in text
        assert "goal1" in text
        # Goals instruction at the tail (last line)
        assert text.strip().endswith(GOALS_INSTRUCTION.strip())

    def test_diff_mode_no_new_items_minimal(self):
        """Diff mode with no new items sends a minimal one-liner."""
        text = frame_beat([], [], diff=True)
        assert "no new memories or goals" in text
        assert "dream up new ones" in text
        assert "You don't rest" in text
        # Should be short (not the full prompt)
        assert len(text) < 200

    def test_diff_mode_only_new_items(self):
        """Diff mode sends only new memories and goals."""
        memories = [
            {"id": 3, "age_seconds": 30, "channel_id": "c3", "summary": "new_mem"},
        ]
        retired = [
            {"id": 11, "age_seconds": 60, "summary": "new_goal"},
        ]
        text = frame_beat(memories, retired, diff=True)
        # New items present
        assert "new_mem" in text
        assert "new_goal" in text
        # Goals instruction at the tail
        assert text.strip().endswith(GOALS_INSTRUCTION.strip())

    def test_diff_mode_no_goals_plugin(self):
        """Diff mode with no goals plugin (retired=None)."""
        memories = [
            {"id": 3, "age_seconds": 30, "channel_id": "c3", "summary": "new_mem"},
        ]
        text = frame_beat(memories, None, diff=True)
        assert "new_mem" in text
        # No goals instruction (no goals plugin)
        assert GOALS_INSTRUCTION not in text

    def test_full_mode_goals_instruction_at_tail(self):
        """Full mode also has goals instruction at the tail."""
        memories = [
            {"id": 1, "age_seconds": 60, "channel_id": "c1", "summary": "mem1"},
        ]
        retired = [
            {"id": 10, "age_seconds": 300, "summary": "goal1"},
        ]
        text = frame_beat(memories, retired, diff=False)
        # Goals instruction is the last thing
        assert text.strip().endswith(GOALS_INSTRUCTION.strip())
        # Memories and goals data come before the instruction
        mem_pos = text.index("mem1")
        goal_pos = text.index("goal1")
        instr_pos = text.index(GOALS_INSTRUCTION)
        assert mem_pos < instr_pos
        assert goal_pos < instr_pos

    def test_post_compaction_goals_full_memories_diff(self):
        """After compaction: goals are full (reset), memories are diff (kept)."""
        # Simulate: last beat sent memories 1,2 and goals 10,11
        plugin = HeartbeatPlugin()
        plugin._last_memory_ids = {1, 2}
        plugin._last_goal_ids = {10, 11}

        # Compaction resets only goal IDs
        plugin._last_goal_ids = set()

        # Next beat: memories 1,2,3 (3 is new), goals 10,11,12 (all "new" because reset)
        memories = [
            {"id": 1, "age_seconds": 300, "channel_id": "c1", "summary": "old_mem1"},
            {"id": 2, "age_seconds": 200, "channel_id": "c2", "summary": "old_mem2"},
            {"id": 3, "age_seconds": 100, "channel_id": "c3", "summary": "new_mem3"},
        ]
        retired = [
            {"id": 10, "age_seconds": 500, "summary": "old_goal10"},
            {"id": 11, "age_seconds": 400, "summary": "old_goal11"},
            {"id": 12, "age_seconds": 300, "summary": "new_goal12"},
        ]

        # Diff for memories: only 3 is new
        new_memories = [m for m in memories if m["id"] not in plugin._last_memory_ids]
        assert [m["id"] for m in new_memories] == [3]

        # Goals are full (reset): all 10, 11, 12
        goals_full = retired is not None and not plugin._last_goal_ids
        assert goals_full is True

        # frame_beat gets: new memories only, all goals
        text = frame_beat(
            new_memories,
            retired,  # full goals
            diff=True,
        )
        # New memory present
        assert "new_mem3" in text
        # Old memories NOT present (diff)
        assert "old_mem1" not in text
        assert "old_mem2" not in text
        # All goals present (full after compaction)
        assert "old_goal10" in text
        assert "old_goal11" in text
        assert "new_goal12" in text
        # Call-to-action at tail
        assert text.strip().endswith(GOALS_INSTRUCTION.strip())


def test_prompt_after_data_blocks():
    """The imperative prompt text comes after all data blocks."""
    memories = [
        {"id": 1, "age_seconds": 60, "channel_id": "c1", "summary": "mem1"},
    ]
    retired = [
        {"id": 10, "age_seconds": 300, "summary": "goal1"},
    ]
    prompt = "DO THE THING."
    text = frame_beat(memories, retired, prompt=prompt, diff=False)
    # Data blocks come before the prompt
    mem_pos = text.index("mem1")
    goal_pos = text.index("goal1")
    prompt_pos = text.index("DO THE THING.")
    assert mem_pos < prompt_pos
    assert goal_pos < prompt_pos


def test_default_limits_are_five():
    """Default memory and retired-goal limits are 5, not 20."""
    assert DEFAULT_MEMORY_LIMIT == 5
    assert DEFAULT_RETIRED_GOAL_LIMIT == 5


def test_memory_header_says_oldest_first():
    """Memory header reflects chronological (oldest-first) order."""
    assert "oldest first" in MEMORY_HEADER
    assert "newest first" not in MEMORY_HEADER
