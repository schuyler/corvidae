"""Tests for corvidae.persistence.PersistencePlugin."""

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from corvidae.trace import mint_trace_id
from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.persistence import init_db
from corvidae.hooks import create_plugin_manager


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "reasoning_history": "drop",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(agent_defaults=None):
    return ChannelRegistry(agent_defaults or AGENT_DEFAULTS)


def _make_channel(registry=None, transport="test", scope="scope1"):
    if registry is None:
        registry = _make_registry()
    return registry.get_or_create(transport, scope, config=ChannelConfig())


async def _make_plugin_with_db(db=None):
    """Build a PersistencePlugin with an injected in-memory DB.

    Returns (plugin, registry, db). Callers are responsible for closing db
    if they passed None (i.e., a fresh db was created here).
    """
    from corvidae.persistence import PersistencePlugin

    pm = create_plugin_manager()
    registry = _make_registry()
    pm.register(registry, name="registry")

    plugin = PersistencePlugin(pm)

    if db is None:
        db = await aiosqlite.connect(":memory:")
        await init_db(db)

    # Inject DB so on_start skips the real open
    plugin.db = db

    return plugin, registry, db


# ---------------------------------------------------------------------------
# TestOnStart
# ---------------------------------------------------------------------------


class TestOnStart:
    async def test_on_start_opens_db_and_creates_tables(self, tmp_path):
        """on_start opens an aiosqlite connection and runs init_db when db is None."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        assert plugin.db is None

        db_path = str(tmp_path / "test.db")
        config = {"daemon": {"session_db": db_path}}

        await plugin.on_start(config=config)

        assert plugin.db is not None
        async with plugin.db.execute("SELECT 1 FROM message_log LIMIT 1"):
            pass

        await plugin.db.close()

    async def test_on_start_skips_db_open_when_pre_injected(self, tmp_path):
        """on_start must not replace a pre-injected db."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)

        pre_injected = await aiosqlite.connect(":memory:")
        await init_db(pre_injected)
        plugin.db = pre_injected

        config = {"daemon": {"session_db": str(tmp_path / "should_not_open.db")}}
        await plugin.on_start(config=config)

        assert plugin.db is pre_injected

        await pre_injected.close()

    async def test_relative_session_db_resolves_against_base_dir(
        self, tmp_path, monkeypatch
    ):
        """A relative daemon.session_db path resolves against config['_base_dir'],
        not the process cwd -- the same mechanism MetricsJsonlPlugin already
        uses for daemon.metrics_jsonl (mirrored in
        tests/test_metrics.py::TestMetricsJsonlPlugin).

        Regression test: previously the raw relative path was passed
        straight to aiosqlite.connect(), so the DB landed next to the
        process cwd instead of next to agent.yaml.
        """
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)

        config_dir = tmp_path / "config_dir"
        config_dir.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        config = {"_base_dir": config_dir, "daemon": {"session_db": "sessions.db"}}
        try:
            await plugin.on_start(config=config)

            assert (config_dir / "sessions.db").exists(), (
                f"expected sessions.db in {config_dir}, cwd was {elsewhere}"
            )
            assert not (elsewhere / "sessions.db").exists()
        finally:
            if plugin.db is not None:
                await plugin.db.close()

    # base_dir tests removed: system prompt resolution moves to Agent


# ---------------------------------------------------------------------------
# TestOnStop
# ---------------------------------------------------------------------------


class TestOnStop:
    async def test_on_stop_closes_db(self):
        """on_stop must close the db connection."""
        plugin, _registry, db = await _make_plugin_with_db()

        closed = []
        original_close = db.close

        async def spy_close():
            closed.append(True)
            await original_close()

        db.close = spy_close

        await plugin.on_stop()

        assert closed, "db.close() was not called"

    async def test_on_stop_with_no_db_does_not_crash(self):
        """on_stop must be a no-op when db is None."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        assert plugin.db is None

        await plugin.on_stop()


# ---------------------------------------------------------------------------
# TestResolveConversation (NEW — RED phase)
# ---------------------------------------------------------------------------


class TestResolveConversation:
    """Tests for PersistencePlugin.resolve_conversation hookimpl.

    RED phase: these fail because resolve_conversation does not exist yet on
    PersistencePlugin.
    """

    async def test_resolve_conversation_returns_none_when_no_rows(self, db):
        """resolve_conversation must return None when no rows exist for the channel."""
        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        result = await plugin.resolve_conversation(channel=channel)

        assert result is None

    async def test_resolve_conversation_returns_tagged_message_dicts(self, db):
        """resolve_conversation returns list of dicts tagged with _message_type."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        base_ts = time.time()
        msg = {"role": "user", "content": "hello"}
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            (channel.id, json.dumps(msg), base_ts),
        )
        await db.commit()

        result = await plugin.resolve_conversation(channel=channel)

        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "hello"
        assert result[0]["_message_type"] == MessageType.MESSAGE

    async def test_resolve_conversation_excludes_other_channels(self, db):
        """resolve_conversation only returns rows for the given channel."""
        plugin, registry, _ = await _make_plugin_with_db(db)
        channel1 = registry.get_or_create("test", "scope1", config=ChannelConfig())
        channel2 = registry.get_or_create("test", "scope2", config=ChannelConfig())

        base_ts = time.time()
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            (channel1.id, json.dumps({"role": "user", "content": "ch1"}), base_ts),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            (channel2.id, json.dumps({"role": "user", "content": "ch2"}), base_ts + 1),
        )
        await db.commit()

        result = await plugin.resolve_conversation(channel=channel1)

        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "ch1"

    async def test_resolve_conversation_with_summary_filters_old_messages(self, db):
        """resolve_conversation respects summary boundary — returns summary + newer rows only."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        base_ts = time.time() - 10

        # Insert 2 old message rows
        for i in range(2):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                (channel.id, json.dumps({"role": "user", "content": f"old {i}"}), base_ts + i),
            )

        # Insert summary row with timestamp between old and new messages
        summary_ts = base_ts + 5
        summary_msg = {"role": "assistant", "content": "[Summary]"}
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'summary')",
            (channel.id, json.dumps(summary_msg), summary_ts),
        )

        # Insert 1 new message row after the summary boundary
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            (channel.id, json.dumps({"role": "user", "content": "new"}), base_ts + 10),
        )
        await db.commit()

        result = await plugin.resolve_conversation(channel=channel)

        assert result is not None
        # summary + 1 new message (2 old excluded by boundary)
        assert len(result) == 2
        types = {m["_message_type"] for m in result}
        assert MessageType.SUMMARY in types
        assert MessageType.MESSAGE in types

    async def test_resolve_conversation_orders_messages_by_timestamp(self, db):
        """resolve_conversation returns messages in chronological order."""
        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        base_ts = time.time()
        # Insert in reverse order
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            (channel.id, json.dumps({"role": "assistant", "content": "second"}), base_ts + 1),
        )
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            (channel.id, json.dumps({"role": "user", "content": "first"}), base_ts),
        )
        await db.commit()

        result = await plugin.resolve_conversation(channel=channel)

        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"


# ---------------------------------------------------------------------------
# TestPersistMessage (NEW — RED phase)
# ---------------------------------------------------------------------------


class TestPersistMessage:
    """Tests for PersistencePlugin.persist_message hookimpl.

    RED phase: these fail because persist_message does not exist yet.
    """

    async def test_persist_message_inserts_row(self, db):
        """persist_message must insert a row into message_log."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)
        msg = {"role": "user", "content": "hello"}

        await plugin.persist_message(
            channel=channel, message=msg, message_type=MessageType.MESSAGE
        )

        async with db.execute(
            "SELECT channel_id, message, message_type FROM message_log WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row[0] == channel.id
        assert json.loads(row[1]) == msg
        assert row[2] == "message"

    async def test_persist_message_persists_context_type(self, db):
        """persist_message with CONTEXT type must write message_type='context'."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)
        msg = {"role": "system", "content": "ctx"}

        await plugin.persist_message(
            channel=channel, message=msg, message_type=MessageType.CONTEXT
        )

        async with db.execute(
            "SELECT message_type FROM message_log WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "context"

    async def test_persist_message_strips_message_type_tag_from_db(self, db):
        """persist_message must not write _message_type into the JSON column."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)
        # Simulate a tagged message that might arrive
        msg = {"role": "user", "content": "hello", "_message_type": MessageType.MESSAGE}

        await plugin.persist_message(
            channel=channel, message=msg, message_type=MessageType.MESSAGE
        )

        async with db.execute(
            "SELECT message FROM message_log WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()

        stored = json.loads(row[0])
        assert "_message_type" not in stored, (
            "persist_message must strip _message_type before writing to DB"
        )

    async def test_persist_message_commits(self, db):
        """persist_message must commit so the row is visible to a fresh query."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        await plugin.persist_message(
            channel=channel,
            message={"role": "user", "content": "committed?"},
            message_type=MessageType.MESSAGE,
        )

        # Fresh query without relying on plugin's internal state
        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1


# ---------------------------------------------------------------------------
# TestOnCompaction (NEW — RED phase)
# ---------------------------------------------------------------------------


class TestOnCompaction:
    """Tests for PersistencePlugin.on_compaction hookimpl.

    RED phase: these fail because on_compaction does not exist yet.
    """

    async def test_on_compaction_inserts_summary_row(self, db):
        """on_compaction must insert a summary row into message_log."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)
        summary_msg = {"role": "assistant", "content": "[Summary]"}

        await plugin.on_compaction(
            channel=channel, summary_msg=summary_msg, retain_count=0, compacted_ids=[]
        )

        async with db.execute(
            "SELECT message_type, message FROM message_log WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row[0] == "summary"
        assert json.loads(row[1]) == summary_msg

    async def test_on_compaction_timestamp_boundary_with_retain(self, db):
        """on_compaction with retain_count > 0 sets summary_ts = oldest_retained - 1e-6.

        Verifies the exact arithmetic AND that resolve_conversation returns the
        correct working set afterward.
        """
        import pytest

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        base_ts = time.time() - 100  # well in the past

        # Insert 3 message rows at base_ts+0, base_ts+1, base_ts+2
        for i in range(3):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, 'message')",
                (channel.id, json.dumps({"role": "user", "content": f"msg {i}"}), base_ts + i),
            )
        await db.commit()

        summary_msg = {"role": "assistant", "content": "[Summary]"}
        # retain_count=1: keep the last 1 row (timestamp base_ts+2)
        # summary_ts must be exactly (base_ts+2) - 1e-6
        await plugin.on_compaction(
            channel=channel, summary_msg=summary_msg, retain_count=1, compacted_ids=[]
        )

        # Verify exact timestamp arithmetic
        async with db.execute(
            "SELECT timestamp FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        summary_ts = row[0]
        expected_ts = base_ts + 2 - 1e-6
        assert summary_ts == pytest.approx(expected_ts, abs=1e-9), (
            f"Summary timestamp {summary_ts} must be oldest_retained - 1e-6 = {expected_ts}"
        )

        # End-to-end: resolve_conversation returns summary + 1 retained message
        loaded = await plugin.resolve_conversation(channel=channel)
        assert loaded is not None
        assert len(loaded) == 2, (
            f"Expected 2 messages (summary + 1 retained), got {len(loaded)}"
        )

    async def test_on_compaction_zero_retain_uses_current_time(self, db):
        """on_compaction with retain_count=0 sets summary_ts to approximately now."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        before = time.time()
        summary_msg = {"role": "assistant", "content": "[Summary]"}
        await plugin.on_compaction(
            channel=channel, summary_msg=summary_msg, retain_count=0, compacted_ids=[]
        )
        after = time.time()

        async with db.execute(
            "SELECT timestamp FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert before <= row[0] <= after

    async def test_on_compaction_commits(self, db):
        """on_compaction must commit the summary row."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        await plugin.on_compaction(
            channel=channel,
            summary_msg={"role": "assistant", "content": "[Summary]"},
            retain_count=0,
            compacted_ids=[],
        )

        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND message_type = 'summary'",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1


# ---------------------------------------------------------------------------
# TestHookIntegration (UPDATED)
# ---------------------------------------------------------------------------


class TestHookIntegration:
    async def test_resolve_conversation_callable_via_broadcast_dispatch(self, db):
        """PersistencePlugin.resolve_conversation must be reachable via ahook dispatch.

        RED phase: fails because resolve_conversation hookimpl does not exist yet.
        """
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        plugin.db = db
        plugin._registry = registry
        pm.register(plugin, name="persistence")

        channel = _make_channel(registry)

        # resolve_conversation hookspec must exist and dispatch correctly
        result = await pm.ahook.resolve_conversation(channel=channel)

        # No rows — should resolve to None
        assert result is None

    async def test_persist_message_callable_via_broadcast_dispatch(self, db):
        """PersistencePlugin.persist_message must be reachable via ahook dispatch.

        RED phase: fails because persist_message hookimpl does not exist yet.
        """
        from corvidae.context import MessageType
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        plugin.db = db
        pm.register(plugin, name="persistence")

        channel = _make_channel(registry)
        msg = {"role": "user", "content": "broadcast test"}

        # Must not raise
        await pm.ahook.persist_message(
            channel=channel, message=msg, message_type=MessageType.MESSAGE
        )

        async with db.execute(
            "SELECT COUNT(*) FROM message_log WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1

    async def test_no_persistence_plugin_resolve_conversation_returns_none(self):
        """When no PersistencePlugin is registered, resolve_conversation resolves to None."""
        pm = create_plugin_manager()
        channel = _make_channel()

        result = await pm.ahook.resolve_conversation(channel=channel)

        assert result is None


# ---------------------------------------------------------------------------
# TestGracefulDegradation (UPDATED)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    async def test_no_persistence_plugin_agent_creates_context_window(self, caplog):
        """Agent must create a ContextWindow directly when resolve_conversation
        returns no results, and must log an appropriate error/warning.

        Agent creates a ContextWindow directly and calls resolve_conversation
        to restore history from persistence.
        """
        from corvidae.agent import Agent
        from corvidae.context import ContextWindow
        from corvidae.task import TaskPlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        pm.ahook.send = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        task_plugin = TaskPlugin(pm)
        pm.register(task_plugin, name="task")
        await task_plugin.on_start(config={})

        plugin = Agent(pm)
        pm.register(plugin, name="agent")
        plugin._registry = registry

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        })
        plugin._client = mock_client

        channel = registry.get_or_create("test", "scope1", config=ChannelConfig())

        # With no PersistencePlugin, Agent should still create a ContextWindow
        await plugin.on_message(channel=channel, sender="user", text="hello", trace_id=mint_trace_id())
        if channel.id in plugin.queues:
            await plugin.queues[channel.id].drain()

        # In the new design, channel.conversation must be a ContextWindow
        assert channel.conversation is not None
        assert isinstance(channel.conversation, ContextWindow), (
            f"Expected ContextWindow, got {type(channel.conversation)}"
        )


# ---------------------------------------------------------------------------
# TestWALMode
# ---------------------------------------------------------------------------


class TestWALMode:
    async def test_on_start_sets_wal_journal_mode_by_default(self, tmp_path):
        """After on_start, the SQLite connection must use WAL journal mode."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        db_path = str(tmp_path / "wal_default.db")
        config = {"daemon": {"session_db": db_path}}

        await plugin.on_start(config=config)

        async with plugin.db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        actual_mode = row[0]

        await plugin.db.close()

        assert actual_mode == "wal"

    async def test_on_start_respects_sqlite_journal_mode_config_override(self, tmp_path):
        """daemon.sqlite_journal_mode override is applied by on_start."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        db_path = str(tmp_path / "delete_mode.db")
        config = {
            "daemon": {
                "session_db": db_path,
                "sqlite_journal_mode": "delete",
            }
        }

        await plugin.on_start(config=config)

        async with plugin.db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        actual_mode = row[0]

        await plugin.db.close()

        assert actual_mode == "delete"

    async def test_on_start_logs_journal_mode_at_info_level(self, tmp_path, caplog):
        """on_start must emit an INFO-level log record mentioning the journal mode."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        db_path = str(tmp_path / "wal_log.db")
        config = {"daemon": {"session_db": db_path}}

        with caplog.at_level(logging.INFO, logger="corvidae.persistence"):
            await plugin.on_start(config=config)

        await plugin.db.close()

        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and r.name == "corvidae.persistence"
            and "wal" in r.getMessage().lower()
        ]
        assert info_records


async def test_on_start_invalid_journal_mode_lists_allowed_values(tmp_path):
    """An invalid sqlite_journal_mode names the accepted values."""
    from corvidae.persistence import PersistencePlugin

    pm = create_plugin_manager()
    registry = _make_registry()
    pm.register(registry, name="registry")

    plugin = PersistencePlugin(pm)
    config = {
        "daemon": {
            "session_db": str(tmp_path / "invalid_mode.db"),
            "sqlite_journal_mode": "bogus",
        }
    }

    with pytest.raises(ValueError) as excinfo:
        await plugin.on_start(config=config)

    message = str(excinfo.value)
    assert "bogus" in message
    for mode in ("delete", "truncate", "persist", "memory", "wal", "off"):
        assert mode in message

    await plugin.db.close()

# ---------------------------------------------------------------------------
# Degradation: corrupt rows and failed writes must not abort the turn
# ---------------------------------------------------------------------------


class TestPersistenceDegradation:
    async def test_resolve_conversation_skips_corrupt_row(self, db, caplog):
        """One unparseable row is skipped; the rest of the history loads."""
        from corvidae.context import MessageType  # noqa: F401  (row types)
        from corvidae.persistence import PersistencePlugin

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        valid_one = json.dumps({"role": "user", "content": "one"})
        corrupt = "{not valid json"
        valid_three = json.dumps({"role": "assistant", "content": "three"})

        async def insert(ts: float, payload: str) -> int:
            cursor = await db.execute(
                "INSERT INTO message_log "
                "(channel_id, message, timestamp, message_type) VALUES (?, ?, ?, ?)",
                (channel.id, payload, ts, "message"),
            )
            return cursor.lastrowid

        await insert(1.0, valid_one)
        corrupt_id = await insert(2.0, corrupt)
        await insert(3.0, valid_three)
        await db.commit()

        with caplog.at_level(logging.WARNING, logger="corvidae.persistence"):
            result = await plugin.resolve_conversation(channel=channel)

        assert [m["content"] for m in result] == ["one", "three"]
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(str(corrupt_id) in message for message in warnings), warnings

    # Summary-row degradation (#108): a corrupt summary must not abort the
    # load. Loading falls back to the newest summary that decodes, or to the
    # full non-summary history when none does.

    @staticmethod
    async def _insert_row(
        db, channel_id: str, ts: float, message_type: str,
        value_sql: str = "?", params: tuple = (),
    ) -> int:
        """Insert a message_log row whose message is the SQL expression value_sql.

        value_sql lets a test store bytes no Python str can carry, such as
        invalid UTF-8 cast to TEXT.
        """
        cursor = await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            f"VALUES (?, {value_sql}, ?, ?)",
            (channel_id, *params, ts, message_type),
        )
        return cursor.lastrowid

    @staticmethod
    def _warnings(caplog) -> list[str]:
        """Return the text of every WARNING-or-higher record captured."""
        return [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]

    async def test_resolve_conversation_corrupt_summary_falls_back_to_older_summary(
        self, db, caplog
    ):
        """A corrupt newest summary is skipped for the newest summary that decodes."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        def msg(role: str, content: str) -> tuple:
            return ("?", (json.dumps({"role": role, "content": content}),))

        # Timeline: m1 | older summary @1.5 | m2 | corrupt summary @2.5 | m3
        await self._insert_row(db, channel.id, 1.0, "message", *msg("user", "m1"))
        await self._insert_row(
            db, channel.id, 1.5, "summary", *msg("assistant", "[older summary]")
        )
        await self._insert_row(db, channel.id, 2.0, "message", *msg("assistant", "m2"))
        corrupt_id = await self._insert_row(
            db, channel.id, 2.5, "summary", "?", ("{not valid json",)
        )
        await self._insert_row(db, channel.id, 3.0, "message", *msg("user", "m3"))
        await db.commit()

        with caplog.at_level(logging.WARNING, logger="corvidae.persistence"):
            result = await plugin.resolve_conversation(channel=channel)

        # The older summary's boundary applies: m1 is excluded, m2 and m3 load.
        assert [m["content"] for m in result] == ["[older summary]", "m2", "m3"]
        assert result[0]["_message_type"] == MessageType.SUMMARY
        warnings = self._warnings(caplog)
        assert any(
            re.search(rf"\b{corrupt_id}\b", w) and channel.id in w
            for w in warnings
        ), warnings

    @pytest.mark.parametrize(
        ("value_sql", "params"),
        [
            pytest.param("?", ("{not valid json",), id="invalid-json"),
            pytest.param("CAST(X'FF7B' AS TEXT)", (), id="invalid-utf8-text"),
            pytest.param("X'FF7B'", (), id="blob"),
            pytest.param("?", ("null",), id="json-null"),
            pytest.param("?", ("[1, 2]",), id="json-array"),
            pytest.param("?", ("1" * 5000,), id="int-digit-limit"),
            pytest.param("?", ("[" * 100000,), id="deep-nesting"),
        ],
    )
    async def test_resolve_conversation_corrupt_summary_falls_back_to_full_history(
        self, db, caplog, value_sql, params
    ):
        """With no summary that decodes, the full non-summary history loads."""
        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        # Timeline: m1 | corrupt summary @1.5 | m2
        await self._insert_row(
            db, channel.id, 1.0, "message", "?",
            (json.dumps({"role": "user", "content": "m1"}),),
        )
        corrupt_id = await self._insert_row(
            db, channel.id, 1.5, "summary", value_sql, params
        )
        await self._insert_row(
            db, channel.id, 2.0, "message", "?",
            (json.dumps({"role": "assistant", "content": "m2"}),),
        )
        await db.commit()

        with caplog.at_level(logging.WARNING, logger="corvidae.persistence"):
            result = await plugin.resolve_conversation(channel=channel)

        # Rows on both sides of the corrupt summary's boundary load.
        assert [m["content"] for m in result] == ["m1", "m2"]
        warnings = self._warnings(caplog)
        assert any(
            re.search(rf"\b{corrupt_id}\b", w) and channel.id in w
            for w in warnings
        ), warnings

    async def test_persist_message_db_error_returns_none(
        self, db, monkeypatch, caplog
    ):
        """A locked DB is logged and swallowed so the turn keeps running."""
        from corvidae.context import MessageType

        plugin, registry, _ = await _make_plugin_with_db(db)
        channel = _make_channel(registry)

        async def locked(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(db, "execute", locked)

        with caplog.at_level(logging.WARNING, logger="corvidae.persistence"):
            result = await plugin.persist_message(
                channel=channel,
                message={"role": "user", "content": "hi"},
                message_type=MessageType.MESSAGE,
            )

        assert result is None
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(channel.id in message for message in warnings), warnings


# ---------------------------------------------------------------------------
# Idle-tracker restart seed (plans/kernel-clock.md, R2)
# ---------------------------------------------------------------------------


async def test_on_start_restores_channels_into_idle_tracker():
    """R2: on_start seeds the IdleTracker with every channel that has history,
    quiet since its LAST message_log row, so a channel watch fires for it after
    a restart with no traffic. Uses MAX(timestamp), not the first row."""
    from fake_clock import install_kernel_clock

    plugin, registry, db = await _make_plugin_with_db()
    pm = plugin.pm
    clock, tracker = install_kernel_clock(pm, started=False)
    now = clock.now()
    # Channel A: old first row, recent last row. Channel B: long quiet.
    for cid, ts in [("a", now - 1000), ("a", now - 100), ("b", now - 400)]:
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, 'message')",
            (cid, json.dumps({"role": "user", "content": "x"}), ts),
        )
    await db.commit()

    fired: list[str] = []

    async def on_idle(cid):
        fired.append(cid)

    tracker.watch_channels(300, on_idle, name="seed-test")
    try:
        await plugin.on_start(config={})
        tracker.start()

        # B has been quiet past the period: due at once.
        await clock.advance(0)
        assert fired == ["b"]
        # A went quiet 100 s ago: due 200 s after start (not at start, which
        # is what its first row would give).
        await clock.advance(199)
        assert fired == ["b"]
        await clock.advance(1)
        assert fired == ["b", "a"]
    finally:
        tracker.stop()
        await clock.close()
        await db.close()
