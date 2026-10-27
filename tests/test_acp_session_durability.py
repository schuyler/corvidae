"""ACP Phase 2 session durability (WP-A2.1+): load / resume / close / list."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.channel import ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.outbound import OutboundMessage
from corvidae.persistence import PersistencePlugin, init_db


AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
}


async def _init_acp_with_persistence():
    """AcpPlugin + PersistencePlugin with in-memory sessions.db."""
    from corvidae.channels.acp import AcpPlugin, CorvidaeAcpAgent

    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")

    persistence = PersistencePlugin(pm)
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    persistence.db = db
    pm.register(persistence, name="persistence")

    plugin = AcpPlugin()
    pm.register(plugin, name="acp")
    await plugin.on_init(pm=pm, config={"_acp_mode": True})

    agent = CorvidaeAcpAgent(
        agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"},
        plugin=plugin,
    )
    conn = MagicMock()
    conn.session_update = AsyncMock()
    plugin._conn = conn
    agent.on_connect(conn)
    return plugin, registry, agent, persistence, db


@pytest.fixture
async def acp_durable(tmp_path):
    """Yield (plugin, registry, agent, persistence, db, cwd); close db after."""
    plugin, registry, agent, persistence, db = await _init_acp_with_persistence()
    try:
        yield plugin, registry, agent, persistence, db, str(tmp_path)
    finally:
        await db.close()


async def _seed_message(
    db: aiosqlite.Connection,
    channel_id: str,
    role: str,
    text: str,
    *,
    ts: float,
) -> None:
    """Insert one message_log row for an ACP channel."""
    await db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, ?, ?, ?)",
        (channel_id, json.dumps({"role": role, "content": text}), ts, "message"),
    )
    await db.commit()


class TestLoadSessionCapability:
    async def test_initialize_advertises_load_session(self):
        """initialize must set agentCapabilities.loadSession true (A16)."""
        from acp import PROTOCOL_VERSION
        from corvidae.channels.acp import CorvidaeAcpAgent

        agent = CorvidaeAcpAgent(
            agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"}
        )
        result = await agent.initialize(protocol_version=PROTOCOL_VERSION)

        caps = result.agent_capabilities
        assert caps is not None
        assert caps.load_session is True


class TestLoadSessionReplay:
    async def test_load_session_replays_user_and_assistant_from_db(self, acp_durable):
        """session/load replays DB history via session/update without forking id."""
        _plugin, registry, agent, _persistence, db, cwd = acp_durable
        session_id = "durability001"
        channel_id = f"acp:{session_id}"
        await _seed_message(db, channel_id, "user", "hello", ts=1.0)
        await _seed_message(db, channel_id, "assistant", "hi there", ts=2.0)

        result = await agent.load_session(cwd=cwd, session_id=session_id)

        assert result is not None
        channel = registry.get(channel_id)
        assert channel is not None
        assert channel.scope == session_id
        assert channel.runtime_overrides.get("cwd") == cwd
        # Same id — not a freshly minted uuid4.
        assert registry.get(channel_id) is channel

        updates = [
            call.kwargs["update"]
            for call in agent._conn.session_update.await_args_list
        ]
        assert len(updates) == 2
        assert updates[0].session_update == "user_message_chunk"
        assert updates[0].content.text == "hello"
        assert updates[1].session_update == "agent_message_chunk"
        assert updates[1].content.text == "hi there"
        # Conversation stays unloaded so Agent lazy-loads the same DB rows.
        assert channel.conversation is None

    async def test_load_session_unknown_id_raises_resource_not_found(self, acp_durable):
        """Unknown session id with no history must raise resource_not_found."""
        from acp import RequestError

        _plugin, _registry, agent, _persistence, _db, cwd = acp_durable

        with pytest.raises(RequestError) as exc_info:
            await agent.load_session(cwd=cwd, session_id="missing999")

        assert exc_info.value.code == -32002

    async def test_load_session_same_process_empty_new_session_ok(self, acp_durable):
        """session/new then session/load with no rows succeeds with zero replay."""
        _plugin, registry, agent, _persistence, _db, cwd = acp_durable
        created = await agent.new_session(cwd=cwd)
        session_id = created.session_id
        agent._conn.session_update.reset_mock()

        result = await agent.load_session(cwd=cwd, session_id=session_id)

        assert result is not None
        assert registry.get(f"acp:{session_id}") is not None
        agent._conn.session_update.assert_not_called()

    async def test_load_session_does_not_fork_channel_id(self, acp_durable):
        """After load, registry still keys history on the loaded session id."""
        from acp import text_block

        plugin, registry, agent, _persistence, db, cwd = acp_durable
        session_id = "nofork42"
        await _seed_message(db, f"acp:{session_id}", "user", "ping", ts=1.0)
        await _seed_message(db, f"acp:{session_id}", "assistant", "pong", ts=2.0)

        await agent.load_session(cwd=cwd, session_id=session_id)
        channel = registry.get(f"acp:{session_id}")
        assert channel is not None

        # Prompt against the loaded id must not mint a second channel.
        async def _reply(*, channel, sender: str, text: str, trace_id: str) -> None:
            await plugin.send(channel=channel, event=OutboundMessage(text="ok"))

        plugin.pm.ahook.on_message = _reply
        await agent.prompt(session_id=session_id, prompt=[text_block("again")])

        assert registry.get(f"acp:{session_id}") is channel
        acp_channels = [c for c in registry.all() if c.transport == "acp"]
        assert len(acp_channels) == 1
        assert acp_channels[0].scope == session_id


class TestResumeAndClose:
    async def test_initialize_advertises_resume_and_close(self):
        """initialize must advertise sessionCapabilities.resume and close."""
        from acp import PROTOCOL_VERSION
        from corvidae.channels.acp import CorvidaeAcpAgent

        agent = CorvidaeAcpAgent(
            agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"}
        )
        result = await agent.initialize(protocol_version=PROTOCOL_VERSION)
        caps = result.agent_capabilities
        assert caps is not None
        assert caps.load_session is True
        sc = caps.session_capabilities
        assert sc is not None
        assert sc.resume is not None
        assert sc.close is not None

    async def test_resume_session_reattaches_without_replay(self, acp_durable):
        """session/resume restores channel + cwd but must not replay history."""
        _plugin, registry, agent, _persistence, db, cwd = acp_durable
        session_id = "resume01"
        await _seed_message(db, f"acp:{session_id}", "user", "old", ts=1.0)
        await _seed_message(db, f"acp:{session_id}", "assistant", "reply", ts=2.0)

        result = await agent.resume_session(session_id=session_id, cwd=cwd)

        assert result is not None
        channel = registry.get(f"acp:{session_id}")
        assert channel is not None
        assert channel.runtime_overrides.get("cwd") == cwd
        assert channel.conversation is None
        agent._conn.session_update.assert_not_called()

    async def test_resume_unknown_raises_resource_not_found(self, acp_durable):
        """Unknown session id on resume raises resource_not_found."""
        from acp import RequestError

        _plugin, _registry, agent, _persistence, _db, cwd = acp_durable
        with pytest.raises(RequestError) as exc_info:
            await agent.resume_session(session_id="nosuch", cwd=cwd)
        assert exc_info.value.code == -32002

    async def test_close_session_cancels_prompt_and_drops_registry_entry(
        self, acp_durable
    ):
        """session/close cancels in-flight prompt and frees the registry entry."""
        from acp import text_block
        import asyncio

        plugin, registry, agent, _persistence, _db, cwd = acp_durable
        created = await agent.new_session(cwd=cwd)
        session_id = created.session_id
        channel = registry.get(f"acp:{session_id}")
        assert channel is not None

        async def _block(*, channel, sender: str, text: str, trace_id: str) -> None:
            channel.pending_tool_call_ids.add("call_close")

        plugin.pm.ahook.on_message = _block
        pending = asyncio.create_task(
            agent.prompt(session_id=session_id, prompt=[text_block("busy")])
        )
        await asyncio.sleep(0)
        assert not pending.done()

        result = await agent.close_session(session_id=session_id)
        assert result is not None
        cancelled = await pending
        assert cancelled.stop_reason == "cancelled"
        assert registry.get(f"acp:{session_id}") is None

    async def test_load_after_close_restores_from_db(self, acp_durable):
        """After close, session/load still restores history from message_log."""
        _plugin, registry, agent, _persistence, db, cwd = acp_durable
        session_id = "closerestore"
        await _seed_message(db, f"acp:{session_id}", "user", "kept", ts=1.0)
        await _seed_message(db, f"acp:{session_id}", "assistant", "still here", ts=2.0)

        await agent.resume_session(session_id=session_id, cwd=cwd)
        await agent.close_session(session_id=session_id)
        assert registry.get(f"acp:{session_id}") is None

        agent._conn.session_update.reset_mock()
        await agent.load_session(cwd=cwd, session_id=session_id)

        channel = registry.get(f"acp:{session_id}")
        assert channel is not None
        updates = [
            call.kwargs["update"]
            for call in agent._conn.session_update.await_args_list
        ]
        assert len(updates) == 2
        assert updates[0].content.text == "kept"
        assert updates[1].content.text == "still here"


class TestListSessions:
    async def test_initialize_advertises_list(self):
        """initialize must advertise sessionCapabilities.list."""
        from acp import PROTOCOL_VERSION
        from corvidae.channels.acp import CorvidaeAcpAgent

        agent = CorvidaeAcpAgent(
            agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"}
        )
        result = await agent.initialize(protocol_version=PROTOCOL_VERSION)
        sc = result.agent_capabilities.session_capabilities
        assert sc is not None
        assert sc.list is not None
        # delete remains unadvertised (deferred).
        assert sc.delete is None

    async def test_list_sessions_includes_indexed_cwd(self, acp_durable):
        """new_session upserts the index; list_sessions returns sessionId + cwd."""
        _plugin, _registry, agent, _persistence, _db, cwd = acp_durable
        created = await agent.new_session(cwd=cwd)

        result = await agent.list_sessions()

        assert result.next_cursor is None
        ids = {s.session_id for s in result.sessions}
        assert created.session_id in ids
        match = next(s for s in result.sessions if s.session_id == created.session_id)
        assert match.cwd == cwd

    async def test_list_sessions_filters_by_cwd(self, acp_durable, tmp_path):
        """Optional cwd filter returns only matching indexed sessions."""
        _plugin, _registry, agent, _persistence, _db, cwd_a = acp_durable
        # session/new now rejects a workspace that is not an existing directory.
        other = tmp_path / "other"
        other.mkdir()
        cwd_b = str(other)
        a = await agent.new_session(cwd=cwd_a)
        b = await agent.new_session(cwd=cwd_b)

        filtered = await agent.list_sessions(cwd=cwd_a)
        ids = {s.session_id for s in filtered.sessions}
        assert a.session_id in ids
        assert b.session_id not in ids

    async def test_list_sessions_survives_registry_clear(self, acp_durable):
        """Index survives dropping the in-memory registry entry."""
        _plugin, registry, agent, _persistence, _db, cwd = acp_durable
        created = await agent.new_session(cwd=cwd)
        registry.channels.pop(f"acp:{created.session_id}", None)
        assert registry.get(f"acp:{created.session_id}") is None

        result = await agent.list_sessions()
        ids = {s.session_id for s in result.sessions}
        assert created.session_id in ids
