"""Gate B ACP conformance: initialize → session/new → prompt → cancel (WP-A1.4)."""

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


@pytest.mark.asyncio
async def test_gate_b_initialize_session_prompt_cancel(tmp_path) -> None:
    """Homegrown Gate B: handshake, new session, prompt end_turn, then cancel.

    Stays in-process (under the global 15s timeout). Covers A14 baseline:
    initialize → session/new → prompt→updates→end_turn, plus cancel on a
    second in-flight prompt.
    """
    from acp import PROTOCOL_VERSION, text_block
    from corvidae.channels.acp import AcpPlugin, CorvidaeAcpAgent

    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")
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

    # 1. initialize
    init = await agent.initialize(protocol_version=PROTOCOL_VERSION)
    assert init.protocol_version == PROTOCOL_VERSION
    assert init.agent_info is not None
    assert init.agent_info.name == "corvidae"
    auth = init.auth_methods
    assert auth is None or auth == []

    # 2. session/new
    cwd = str(tmp_path)
    created = await agent.new_session(cwd=cwd)
    session_id = created.session_id
    channel = registry.get(f"acp:{session_id}")
    assert channel is not None
    assert channel.runtime_overrides.get("cwd") == cwd

    # 3. prompt → agent_message_chunk → end_turn
    async def _reply_on_message(*, channel, sender: str, text: str) -> None:
        await plugin.send_message(channel=channel, text=f"echo:{text}")

    plugin.pm.ahook.on_message = _reply_on_message
    prompt_resp = await agent.prompt(
        session_id=session_id,
        prompt=[text_block("gate-b")],
    )
    assert prompt_resp.stop_reason == "end_turn"
    assert conn.session_update.await_count >= 1

    # 4. cancel during a second prompt with tools still pending
    async def _block_on_message(*, channel, sender: str, text: str) -> None:
        channel.pending_tool_call_ids.add("call_gate_b")

    plugin.pm.ahook.on_message = _block_on_message
    pending = asyncio.create_task(
        agent.prompt(session_id=session_id, prompt=[text_block("again")])
    )
    await asyncio.sleep(0)
    assert not pending.done()

    await agent.cancel(session_id=session_id)
    cancelled = await pending
    assert cancelled.stop_reason == "cancelled"
    assert "call_gate_b" not in channel.pending_tool_call_ids
