"""Tests for AcpPlugin inertness, initialize, and sessions/prompt (WP-A0.2 / A1.1)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.channel import ChannelRegistry
from corvidae.hooks import create_plugin_manager


AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
}


def _make_pm_with_registry():
    """Plugin manager with a ChannelRegistry registered as ``registry``."""
    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")
    return pm, registry


async def _init_acp_plugin(*, acp_mode: bool = True):
    """Construct AcpPlugin with registry; return (plugin, registry, agent)."""
    from corvidae.channels.acp import AcpPlugin, CorvidaeAcpAgent

    pm, registry = _make_pm_with_registry()
    plugin = AcpPlugin()
    pm.register(plugin, name="acp")
    config = {"_acp_mode": True} if acp_mode else {}
    await plugin.on_init(pm=pm, config=config)
    agent = CorvidaeAcpAgent(
        agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"},
        plugin=plugin,
    )
    plugin._conn = MagicMock()
    plugin._conn.session_update = AsyncMock()
    agent._conn = plugin._conn
    return plugin, registry, agent


class TestAcpPluginInert:
    async def test_on_start_without_acp_mode_starts_no_stdio_task(self):
        """Without ``_acp_mode``, on_start must not own stdio / start an ACP task."""
        from corvidae.channels.acp import AcpPlugin

        pm, _registry = _make_pm_with_registry()
        plugin = AcpPlugin()
        pm.register(plugin, name="acp")
        await plugin.on_init(pm=pm, config={})

        await plugin.on_start(config={})

        assert getattr(plugin, "_task", None) is None


class TestCorvidaeAcpAgentInitialize:
    async def test_initialize_handshake_in_process(self):
        """initialize returns protocol 1, agent_info, and empty auth methods."""
        from acp import PROTOCOL_VERSION
        from corvidae.channels.acp import CorvidaeAcpAgent

        agent = CorvidaeAcpAgent(
            agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"}
        )
        result = await agent.initialize(protocol_version=PROTOCOL_VERSION)

        assert result.protocol_version == PROTOCOL_VERSION
        assert result.agent_info is not None
        assert result.agent_info.name == "corvidae"
        assert result.agent_info.title == "Corvidae"
        auth = result.auth_methods
        assert auth is None or auth == []


class TestAcpSendMessageFilter:
    async def test_send_message_ignores_non_acp_channel(self):
        """send_message on a non-acp channel must not emit ACP session updates."""
        from corvidae.channels.acp import AcpPlugin

        pm, registry = _make_pm_with_registry()
        plugin = AcpPlugin()
        pm.register(plugin, name="acp")
        await plugin.on_init(pm=pm, config={"_acp_mode": True})

        plugin._conn = MagicMock()
        plugin._conn.session_update = AsyncMock()

        cli_channel = registry.get_or_create("cli", "local")
        await plugin.send_message(channel=cli_channel, text="hello")

        plugin._conn.session_update.assert_not_called()


class TestAcpSessionsAndPrompt:
    async def test_new_session_creates_acp_channel_with_cwd(self, tmp_path):
        """session/new creates Channel(transport=acp) and stores cwd."""
        _plugin, registry, agent = await _init_acp_plugin()
        cwd = str(tmp_path)
        result = await agent.new_session(cwd=cwd)

        assert result.session_id
        channel = registry.get(f"acp:{result.session_id}")
        assert channel is not None
        assert channel.transport == "acp"
        assert channel.runtime_overrides.get("cwd") == cwd

    async def test_prompt_text_streams_agent_message_and_end_turn(self, tmp_path):
        """prompt flattens text, streams an agent message, and returns end_turn."""
        from acp import text_block

        plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))
        session_id = created.session_id

        async def _fake_on_message(channel, sender: str, text: str) -> None:
            # Simulate the agent loop finishing a text-only turn.
            await plugin.send_message(channel=channel, text=f"reply:{text}")

        plugin.pm.ahook.on_message = _fake_on_message

        response = await agent.prompt(
            session_id=session_id,
            prompt=[text_block("hello")],
        )

        assert response.stop_reason == "end_turn"
        assert plugin._conn.session_update.await_count >= 1

    async def test_send_thinking_and_tool_status_map_to_session_updates(self, tmp_path):
        """Thinking and tool status on an acp channel emit session_update calls."""
        plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))
        channel = plugin._registry.get(f"acp:{created.session_id}")

        await plugin.send_thinking(channel=channel, text="hmm")
        await plugin.send_tool_status(
            channel=channel, tool_name="shell", status="dispatched", args_summary="ls"
        )
        await plugin.send_tool_status(
            channel=channel, tool_name="shell", status="completed", result_summary="ok"
        )
        await plugin.send_progress(channel=channel, text="working…")

        assert plugin._conn.session_update.await_count >= 4

    async def test_prompt_waits_for_tool_drain_before_end_turn(self, tmp_path):
        """prompt must not resolve while pending_tool_call_ids is non-empty."""
        from acp import text_block

        plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))
        session_id = created.session_id

        gate = asyncio.Event()

        async def _fake_on_message(channel, sender: str, text: str) -> None:
            channel.pending_tool_call_ids.add("call_1")
            await gate.wait()
            channel.pending_tool_call_ids.discard("call_1")
            await plugin.send_message(channel=channel, text="done")

        plugin.pm.ahook.on_message = _fake_on_message

        prompt_task = asyncio.create_task(
            agent.prompt(session_id=session_id, prompt=[text_block("go")])
        )
        await asyncio.sleep(0)  # let prompt reach the gate
        assert not prompt_task.done()

        gate.set()
        response = await prompt_task
        assert response.stop_reason == "end_turn"


class TestAcpCancel:
    async def test_cancel_during_prompt_yields_cancelled_stop_reason(self, tmp_path):
        """cancel resolves an in-flight prompt with stop_reason cancelled."""
        from acp import text_block

        plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))
        session_id = created.session_id

        async def _fake_on_message(channel, sender: str, text: str) -> None:
            # Enqueue-style: leave the prompt future pending (tools in flight).
            channel.pending_tool_call_ids.add("call_1")

        plugin.pm.ahook.on_message = _fake_on_message

        prompt_task = asyncio.create_task(
            agent.prompt(session_id=session_id, prompt=[text_block("go")])
        )
        await asyncio.sleep(0)
        assert not prompt_task.done()

        await agent.cancel(session_id=session_id)
        response = await asyncio.wait_for(prompt_task, timeout=2)
        assert response.stop_reason == "cancelled"

    async def test_cancel_unknown_session_is_safe(self, tmp_path):
        """cancel on an unknown session_id must not raise."""
        _plugin, _registry, agent = await _init_acp_plugin()
        await agent.new_session(cwd=str(tmp_path))
        await agent.cancel(session_id="does-not-exist")
