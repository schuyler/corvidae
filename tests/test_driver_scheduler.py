"""Tests for the harness scheduler probes (harness/driver.py, #59).

The live probes need llama-server on the Buster host; these cover what
doesn't: the profile templates turn on what the probes need, the
sessions.db helpers read rows written by the real SchedulerPlugin and
PersistencePlugin, the verdict names each failure, and the reminder
probe's sequence is wired correctly against a faked IRC client.
"""
from __future__ import annotations

import time
import types
from pathlib import Path

import aiosqlite
import pytest
import yaml

from corvidae.channel import Channel, ChannelRegistry
from corvidae.clock import Clock
from corvidae.context import MessageType
from corvidae.hooks import create_plugin_manager
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.scheduler import SchedulerPlugin, frame_firing, parse_job

from harness import driver
from harness.driver import (
    HEARTBEAT_CHANNEL,
    HEARTBEAT_JOB_CHANNEL,
    PROFILES,
    REMIND_CHANNEL,
    REMIND_RESTART_CHANNEL,
    REMIND_TOKEN,
    heartbeat_exchange_count,
    reminder_firing_row,
    reminder_set_since,
    reminder_verdict,
)

REPO = Path(__file__).resolve().parent.parent


def _render(name: str) -> dict:
    """Load harness/<name>.yaml.in as run.sh would render it."""
    text = (REPO / "harness" / f"{name}.yaml.in").read_text()
    return yaml.safe_load(text.replace("@STATE_DIR@", "/state"))


class TestProfiles:
    def test_default_profile_runs_the_reminder_probes(self):
        probes, channels = PROFILES["buster"]
        assert driver.probe_reminder in probes and driver.probe_reminder_restart in probes
        cfg = _render("buster")
        for channel in (REMIND_CHANNEL, REMIND_RESTART_CHANNEL):
            assert channel in channels
            assert channel in cfg["irc"]["channels"]
        assert "scheduler" not in cfg["plugins"]["disabled"]

    def test_heartbeat_profile(self):
        probes, channels = PROFILES["buster-heartbeat"]
        assert probes == [driver.probe_heartbeat_smoke] and channels == [HEARTBEAT_CHANNEL]
        cfg = _render("buster-heartbeat")
        disabled = set(cfg["plugins"]["disabled"])
        assert {"scheduler", "outcome_log"}.isdisjoint(disabled)
        assert HEARTBEAT_CHANNEL in cfg["irc"]["channels"]
        (job,) = cfg["scheduler"]["jobs"]
        spec = parse_job(job, REPO)
        assert spec.channel_id == HEARTBEAT_JOB_CHANNEL and spec.origin == "heartbeat"


async def _scheduler_on(db_path: Path):
    """A SchedulerPlugin and PersistencePlugin on a real sessions.db file."""
    conn = await aiosqlite.connect(db_path)
    await init_db(conn)
    pm = create_plugin_manager()
    registry = ChannelRegistry({})
    pm.register(registry, name="registry")
    # The scheduler requires the kernel clock; these tests run on real time.
    pm.register(Clock(), name="clock")
    persistence = PersistencePlugin()
    persistence.db = conn
    pm.register(persistence, name="persistence")
    sched = SchedulerPlugin(pm)
    pm.register(sched, name="scheduler")
    await sched.on_init(pm=pm, config={})
    sched._registry = registry
    await sched._startup_if_ready()
    return conn, persistence, sched


class TestDbHelpers:
    async def test_helpers_read_real_rows(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        channel_id = f"irc:{REMIND_CHANNEL}"
        conn, persistence, sched = await _scheduler_on(db_path)
        try:
            t0 = time.time() - 1
            assert reminder_set_since(db_path, channel_id, t0) is None
            row_id = await sched.add_reminder(channel_id, REMIND_TOKEN, time.time() + 20)
            got = reminder_set_since(db_path, channel_id, t0)
            assert got is not None and got[0] == row_id

            # The firing row as Agent persists it: a system message with the
            # [source] prefix _build_conversation_message adds.
            (row,) = await sched._fetch(conn, "id = ?", (row_id,))
            content = f"[reminder]\n\n{frame_firing(row, sched._tz, None)}"
            channel = Channel(transport="irc", scope=REMIND_CHANNEL)
            await persistence.persist_message(
                channel=channel, message={"role": "system", "content": content},
                message_type=MessageType.MESSAGE,
            )
            fired = reminder_firing_row(db_path, channel_id, REMIND_TOKEN, t0)
            assert fired is not None and REMIND_TOKEN in fired[1]
            assert reminder_firing_row(db_path, channel_id, "OTHER-TOKEN", t0) is None
        finally:
            await conn.close()

    def test_missing_tables_read_as_nothing(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        assert reminder_set_since(db_path, "irc:#x", 0) is None
        assert heartbeat_exchange_count(db_path, HEARTBEAT_JOB_CHANNEL) == 0


class TestVerdict:
    def test_pass(self):
        assert reminder_verdict(True, True, True)[0]

    @pytest.mark.parametrize("args,needle", [
        ((False, False, False), "did not call remind_me"),
        ((True, False, False), "never fired"),
        ((True, True, False), "no unprompted reply"),
    ])
    def test_each_failure_is_named(self, args, needle):
        passed, details = reminder_verdict(*args)
        assert not passed and needle in details


class TestReminderProbeWiring:
    async def test_passes_only_on_token_after_the_firing_row(self, tmp_path, monkeypatch):
        """The ack may echo the token; only a reply after the firing row counts."""
        db_path = tmp_path / "sessions.db"
        channel_id = f"irc:{REMIND_CHANNEL}"
        conn, persistence, sched = await _scheduler_on(db_path)
        messages: dict[str, list] = {REMIND_CHANNEL: []}

        async def fake_ack(ctx, channel, text, sent, timeout):
            # The model sets the reminder and acks, echoing the token.
            await sched.add_reminder(channel_id, REMIND_TOKEN, time.time())
            messages[channel].append(driver.IRCMessage(time.time(), "buster", f"SCHEDULED {REMIND_TOKEN}"))
            # The scheduler fires: the [reminder] row lands, then the reply.
            await persistence.persist_message(
                channel=Channel(transport="irc", scope=REMIND_CHANNEL),
                message={"role": "system", "content": f"[reminder]\n\n... «{REMIND_TOKEN}» ..."},
                message_type=MessageType.MESSAGE,
            )
            if reply:
                messages[channel].append(driver.IRCMessage(time.time() + 1, "buster", f"Reminder: {REMIND_TOKEN}"))
            return True

        async def wait_until(channel, predicate, timeout, poll_interval=0.5):
            return predicate(messages.get(channel, []))

        monkeypatch.setattr(driver, "send_and_await_any_reply", fake_ack)
        monkeypatch.setattr(driver, "REMIND_FIRE_GRACE_S", 0)
        ctx = driver.RunContext(
            client=types.SimpleNamespace(messages=messages, wait_until=wait_until),
            bot_nick="buster", state_dir=tmp_path, run_dir=tmp_path, repo_dir=REPO,
            session_db=db_path, channel_ready={REMIND_CHANNEL: True}, httpd=None,
        )
        try:
            reply = False
            result = await driver.probe_reminder(ctx)
            assert result.passed is False and "no unprompted reply" in result.details

            reply = True
            messages[REMIND_CHANNEL].clear()
            result = await driver.probe_reminder(ctx)
            assert result.passed is True, result.details
        finally:
            await conn.close()
