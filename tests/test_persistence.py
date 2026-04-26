"""Tests for corvidae.persistence.PersistencePlugin.

RED phase: these tests fail because corvidae/persistence.py does not exist yet
and the ensure_conversation hookspec has not been added to corvidae/hooks.py.
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.conversation import ConversationLog, init_db
from corvidae.hooks import call_firstresult_hook, create_plugin_manager


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
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

    Returns (plugin, registry, db).  Callers are responsible for closing db
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
        # db is None — on_start must open it
        assert plugin.db is None

        db_path = str(tmp_path / "test.db")
        config = {"daemon": {"session_db": db_path}}

        await plugin.on_start(config=config)

        assert plugin.db is not None
        # Tables must exist — querying message_log must not raise
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

        # Pre-inject an in-memory DB
        pre_injected = await aiosqlite.connect(":memory:")
        await init_db(pre_injected)
        plugin.db = pre_injected

        config = {"daemon": {"session_db": str(tmp_path / "should_not_open.db")}}
        await plugin.on_start(config=config)

        # The pre-injected connection must still be the one on the plugin
        assert plugin.db is pre_injected

        await pre_injected.close()

    async def test_on_start_stores_base_dir_from_config(self, tmp_path):
        """on_start must store _base_dir from config as plugin.base_dir."""
        plugin, _registry, db = await _make_plugin_with_db()
        config = {"_base_dir": tmp_path}
        await plugin.on_start(config=config)

        assert plugin.base_dir == tmp_path
        await db.close()

    async def test_on_start_defaults_base_dir_to_cwd(self):
        """on_start must default base_dir to Path('.') when _base_dir not in config."""
        plugin, _registry, db = await _make_plugin_with_db()
        config = {}
        await plugin.on_start(config=config)

        assert plugin.base_dir == Path(".")
        await db.close()


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

        # Must not raise
        await plugin.on_stop()


# ---------------------------------------------------------------------------
# TestEnsureConversation
# ---------------------------------------------------------------------------


class TestEnsureConversation:
    async def test_ensure_conversation_creates_conversation_log(self):
        """ensure_conversation must create and attach a ConversationLog when
        channel.conversation is None."""
        plugin, registry, db = await _make_plugin_with_db()
        plugin._registry = registry
        channel = _make_channel(registry)

        assert channel.conversation is None

        result = await plugin.ensure_conversation(channel=channel)

        assert result is True
        assert channel.conversation is not None
        assert isinstance(channel.conversation, ConversationLog)

        await db.close()

    async def test_ensure_conversation_returns_true_without_replacing_existing(self):
        """When channel.conversation is already set, ensure_conversation must
        return True immediately without replacing it."""
        plugin, registry, db = await _make_plugin_with_db()
        plugin._registry = registry
        channel = _make_channel(registry)

        # Pre-attach a conversation
        existing = ConversationLog(db, channel.id)
        channel.conversation = existing

        result = await plugin.ensure_conversation(channel=channel)

        assert result is True
        assert channel.conversation is existing  # unchanged

        await db.close()

    async def test_ensure_conversation_resolves_string_system_prompt(self):
        """ensure_conversation must pass through a literal string system_prompt."""
        agent_defaults = dict(AGENT_DEFAULTS)
        agent_defaults["system_prompt"] = "Literal prompt."
        registry = _make_registry(agent_defaults)

        plugin, _registry, db = await _make_plugin_with_db()
        plugin._registry = registry

        channel = registry.get_or_create("test", "s1", config=ChannelConfig())

        await plugin.ensure_conversation(channel=channel)

        assert channel.conversation.system_prompt == "Literal prompt."
        await db.close()

    async def test_ensure_conversation_resolves_file_list_system_prompt(self, tmp_path):
        """ensure_conversation must read file-list system_prompt entries and
        join them with double newlines."""
        prompt_a = tmp_path / "a.txt"
        prompt_b = tmp_path / "b.txt"
        prompt_a.write_text("Part A")
        prompt_b.write_text("Part B")

        agent_defaults = dict(AGENT_DEFAULTS)
        agent_defaults["system_prompt"] = [str(prompt_a), str(prompt_b)]
        registry = _make_registry(agent_defaults)

        plugin, _registry, db = await _make_plugin_with_db()
        plugin._registry = registry
        plugin.base_dir = tmp_path

        channel = registry.get_or_create("test", "s2", config=ChannelConfig())

        await plugin.ensure_conversation(channel=channel)

        assert channel.conversation.system_prompt == "Part A\n\nPart B"
        await db.close()


# ---------------------------------------------------------------------------
# TestHookIntegration
# ---------------------------------------------------------------------------


class TestHookIntegration:
    async def test_ensure_conversation_callable_via_call_firstresult_hook(self):
        """PersistencePlugin's ensure_conversation must be reachable via
        call_firstresult_hook — confirming hookspec wiring is correct."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        plugin.db = db
        plugin._registry = registry
        pm.register(plugin, name="persistence")

        channel = _make_channel(registry)

        result = await call_firstresult_hook(
            pm, "ensure_conversation", channel=channel
        )

        assert result is True
        assert channel.conversation is not None

        await db.close()

    async def test_no_persistence_plugin_returns_none_from_hook(self):
        """When no PersistencePlugin is registered, call_firstresult_hook must
        return None — graceful degradation path."""
        pm = create_plugin_manager()
        # PersistencePlugin intentionally NOT registered

        channel = _make_channel()

        result = await call_firstresult_hook(
            pm, "ensure_conversation", channel=channel
        )

        assert result is None
        assert channel.conversation is None


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    async def test_no_persistence_plugin_logs_error_in_agent(self, caplog):
        """AgentPlugin must log an ERROR when no persistence plugin handles
        ensure_conversation, and must not call the LLM."""
        from corvidae.agent import AgentPlugin
        from corvidae.task import TaskPlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        # No PersistencePlugin registered — graceful degradation path

        task_plugin = TaskPlugin(pm)
        pm.register(task_plugin, name="task")
        await task_plugin.on_start(config={})

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")
        plugin._registry = registry

        mock_client = MagicMock()
        mock_client.chat = AsyncMock()
        plugin.client = mock_client

        channel = registry.get_or_create("test", "scope1", config=ChannelConfig())

        with caplog.at_level(logging.ERROR, logger="corvidae.agent"):
            await plugin.on_message(channel=channel, sender="user", text="hello")
            # Drain the queue so _process_queue_item runs
            if channel.id in plugin.queues:
                await plugin.queues[channel.id].drain()

        error_records = [
            r
            for r in caplog.records
            if r.levelno >= logging.ERROR
            and "persistence" in r.getMessage().lower()
        ]
        assert error_records, (
            "AgentPlugin must log ERROR when no persistence plugin is registered"
        )

        # LLM must NOT have been called
        mock_client.chat.assert_not_called()


# ---------------------------------------------------------------------------
# TestWALMode
# ---------------------------------------------------------------------------


class TestWALMode:
    """RED phase: WAL journal_mode tests.

    These tests fail against the current code because on_start does not set
    the journal_mode after init_db.  They will pass once the feature is
    implemented.
    """

    async def test_on_start_sets_wal_journal_mode_by_default(self, tmp_path):
        """After on_start, the SQLite connection must use WAL journal mode when
        no daemon.sqlite_journal_mode config key is present."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        assert plugin.db is None

        db_path = str(tmp_path / "wal_default.db")
        config = {"daemon": {"session_db": db_path}}

        await plugin.on_start(config=config)

        async with plugin.db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        actual_mode = row[0]

        await plugin.db.close()

        assert actual_mode == "wal", (
            f"Expected journal_mode 'wal' by default, got '{actual_mode}'"
        )

    async def test_on_start_respects_sqlite_journal_mode_config_override(
        self, tmp_path
    ):
        """When daemon.sqlite_journal_mode is set in config, on_start must
        apply that mode instead of the default WAL."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        assert plugin.db is None

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

        assert actual_mode == "delete", (
            f"Expected journal_mode 'delete' from config override, got '{actual_mode}'"
        )

    async def test_on_start_logs_journal_mode_at_info_level(
        self, tmp_path, caplog
    ):
        """on_start must emit an INFO-level log record that includes the
        resulting journal_mode string."""
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        registry = _make_registry()
        pm.register(registry, name="registry")

        plugin = PersistencePlugin(pm)
        assert plugin.db is None

        db_path = str(tmp_path / "wal_log.db")
        config = {"daemon": {"session_db": db_path}}

        with caplog.at_level(logging.INFO, logger="corvidae.persistence"):
            await plugin.on_start(config=config)

        await plugin.db.close()

        info_records = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO
            and r.name == "corvidae.persistence"
            and "wal" in r.getMessage().lower()
        ]
        assert info_records, (
            "Expected an INFO log record from corvidae.persistence mentioning 'wal'"
        )
