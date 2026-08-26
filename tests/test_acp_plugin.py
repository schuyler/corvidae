"""Tests for AcpPlugin inertness and initialize (WP-A0.2: A4, A5, A6, A11)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
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

        # Fake ACP client connection — must not be called for cli channels.
        plugin._conn = MagicMock()
        plugin._conn.session_update = AsyncMock()

        cli_channel = registry.get_or_create("cli", "local")
        await plugin.send_message(channel=cli_channel, text="hello")

        plugin._conn.session_update.assert_not_called()
