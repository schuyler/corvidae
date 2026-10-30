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
import time
from importlib.metadata import entry_points

import aiosqlite
import pytest

from corvidae.channel import ChannelRegistry
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.persistence import PersistencePlugin, init_db
from corvidae_cognition.heartbeat import (
    DEFAULT_HEARTBEAT_PROMPT,
    HEARTBEAT_CHANNEL,
    HeartbeatPlugin,
)
from corvidae_cognition.memory import MemoryPlugin
from fake_clock import install_kernel_clock

AFTER = 600


class AgentStub:
    """Records on_notify calls and marks the channel busy, then settled, the
    way the Agent does around a turn (so the tracker sees each beat as work)."""

    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self.calls: list[dict] = []

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, meta):
        self.calls.append({
            "channel": channel, "source": source, "text": text,
            "tool_call_id": tool_call_id, "meta": meta,
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

        # Memory: schema only; its own idle watches are not under test.
        memory = None
        if with_memory:
            memory = MemoryPlugin()
            pm.register(memory, name="memory")
            await memory.on_init(pm=pm, config=config)
            await memory._ensure_schema()

        heartbeat = HeartbeatPlugin()
        pm.register(heartbeat, name="heartbeat")
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
    await env.clock.advance(AFTER * 10)
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
    assert beat["meta"] == {"origin": "heartbeat"}
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

    async def recent_then_traffic(channel_id, limit):
        # A message lands on another channel while memory is being read.
        env.tracker.note_activity("irc:#busy")
        return await real_recent(channel_id, limit)

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

    async def flaky_recent(channel_id, limit):
        if failures:
            raise failures.pop()
        return await real_recent(channel_id, limit)

    env.memory.recent = flaky_recent
    with caplog.at_level(logging.WARNING):
        await env.clock.advance(AFTER)
    assert env.beats == []
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert HEARTBEAT_CHANNEL in warning.getMessage()
    assert "RuntimeError" in warning.getMessage()
    await env.clock.advance(AFTER)
    assert len(env.beats) == 1


@pytest.mark.parametrize("bad", [0, -60, True, "hourly"])
async def test_rejects_an_after_that_is_not_a_positive_number(make_env, bad):
    """H1's bounded rate needs a positive period: anything else fails startup
    naming the key, the value and what is accepted."""
    with pytest.raises(ValueError) as exc:
        await make_env({"heartbeat": {"after": bad}})
    message = str(exc.value)
    assert "heartbeat.after" in message
    assert repr(bad) in message
    assert "positive number of seconds" in message


# ---------------------------------------------------------------------------
# H2: the review input
# ---------------------------------------------------------------------------


async def test_beat_carries_prompt_and_recent_memory_in_scope(make_env):
    """H2: the beat holds the heartbeat prompt and the memories of the
    channels in its scope, newest first. Out-of-scope, redacted and the
    heartbeat's own records stay out."""
    config = {
        "heartbeat": {"after": AFTER},
        "memory": {"channel_groups": {"review": [HEARTBEAT_CHANNEL, "irc:#lex"]}},
    }
    env = await make_env(config)
    now = time.time()
    await env.seed_memory("irc:#lex", "I promised kestrel a wiring diagram.", now - 7200)
    await env.seed_memory("irc:#lex", "kestrel said the station reboots at noon.", now - 60)
    await env.seed_memory("signal:alice", "Alice's private plans.", now - 30)
    await env.seed_memory("irc:#lex", "A redacted thing.", now - 20, redacted=1)
    await env.seed_memory(HEARTBEAT_CHANNEL, "I reviewed my memories.", now - 10)

    await env.clock.advance(AFTER)
    assert len(env.beats) == 1
    text = env.beats[0]["text"]
    assert DEFAULT_HEARTBEAT_PROMPT in text
    newer = text.index("kestrel said the station reboots at noon.")
    older = text.index("I promised kestrel a wiring diagram.")
    assert newer < older
    assert "Alice's private plans." not in text
    assert "A redacted thing." not in text
    assert "I reviewed my memories." not in text


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
    await after.clock.advance(AFTER * 10)
    assert after.beats == []
