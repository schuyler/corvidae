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


# ---------------------------------------------------------------------------
# Misconfiguration and unsupported input must be visible, not silently
# ignored: a bad cwd, mcpServers the transport cannot mount, and non-text
# prompt blocks all need an error or a warning.
# ---------------------------------------------------------------------------


class TestAcpCwdValidation:
    async def test_new_session_rejects_missing_cwd(self):
        """A cwd that is not an existing directory is invalid params."""
        from acp import RequestError

        _plugin, _registry, agent = await _init_acp_plugin()

        with pytest.raises(RequestError) as excinfo:
            await agent.new_session(cwd="/definitely/not/a/directory")

        assert excinfo.value.code == -32602  # invalid params

    async def test_new_session_rejects_cwd_that_is_a_file(self, tmp_path):
        """A cwd pointing at a regular file is invalid params."""
        from acp import RequestError

        _plugin, _registry, agent = await _init_acp_plugin()
        a_file = tmp_path / "not-a-dir"
        a_file.write_text("x")

        with pytest.raises(RequestError) as excinfo:
            await agent.new_session(cwd=str(a_file))

        assert excinfo.value.code == -32602

    async def test_new_session_warns_when_cwd_differs_from_process_cwd(
        self, tmp_path, caplog
    ):
        """A valid but different cwd is recorded; say that tools use the process cwd."""
        _plugin, _registry, agent = await _init_acp_plugin()

        with caplog.at_level("WARNING"):
            await agent.new_session(cwd=str(tmp_path))

        assert any(
            "working directory" in record.message.lower() for record in caplog.records
        )


class TestAcpIgnoredClientInput:
    async def test_new_session_warns_about_mcp_servers(self, tmp_path, caplog):
        """mcpServers are not mounted per session; the drop must be logged."""
        _plugin, _registry, agent = await _init_acp_plugin()

        with caplog.at_level("WARNING"):
            await agent.new_session(
                cwd=str(tmp_path), mcp_servers=[{"name": "fs", "command": "x"}]
            )

        assert any("mcpservers" in record.message.lower() for record in caplog.records)

    async def test_prompt_with_only_non_text_blocks_is_invalid(self, tmp_path):
        """An image-only prompt cannot become an empty user message."""
        from acp import RequestError

        _plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))

        # wait_for so a regression that leaves the prompt future pending fails
        # fast instead of hanging the suite.
        with pytest.raises(RequestError) as excinfo:
            await asyncio.wait_for(
                agent.prompt(
                    session_id=created.session_id,
                    prompt=[{"type": "image", "data": "abc", "mimeType": "image/png"}],
                ),
                timeout=2,
            )

        assert excinfo.value.code == -32602

    async def test_prompt_logs_dropped_non_text_blocks(self, tmp_path, caplog):
        """Text plus an image keeps the text and logs the dropped block type."""
        from acp import text_block

        plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))

        async def _fake_on_message(channel, sender: str, text: str) -> None:
            await plugin.send_message(channel=channel, text=f"reply:{text}")

        plugin.pm.ahook.on_message = _fake_on_message

        with caplog.at_level("WARNING"):
            response = await asyncio.wait_for(
                agent.prompt(
                    session_id=created.session_id,
                    prompt=[
                        text_block("hello"),
                        {"type": "image", "data": "abc", "mimeType": "image/png"},
                    ],
                ),
                timeout=2,
            )

        assert response.stop_reason == "end_turn"
        assert any("image" in record.message.lower() for record in caplog.records)


class TestAcpUnknownSession:
    async def test_prompt_unknown_session_is_resource_not_found(self):
        """prompt matches load/resume/close: not-found, not internal error."""
        from acp import RequestError, text_block

        _plugin, _registry, agent = await _init_acp_plugin()

        with pytest.raises(RequestError) as excinfo:
            await asyncio.wait_for(
                agent.prompt(session_id="nosuchsession", prompt=[text_block("hi")]),
                timeout=2,
            )

        assert excinfo.value.code == -32002
        assert excinfo.value.data == {"uri": "acp:nosuchsession"}


# ---------------------------------------------------------------------------
# A prompt must always end: either the turn completes, or the failure that
# stopped it resolves/clears the future. A pending future would hang the ACP
# request forever.
# ---------------------------------------------------------------------------


class TestPromptAlwaysCompletes:
    async def test_delivery_failure_still_completes_prompt(self, tmp_path):
        """A failing session_update must not leave the prompt future pending."""
        from acp import text_block

        plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))
        plugin._conn.session_update = AsyncMock(side_effect=RuntimeError("client gone"))

        async def _fake_on_message(channel, sender: str, text: str) -> None:
            # Mimic the agent loop: delivery happens on its own task, and the
            # Agent logs (does not re-raise) a failed send_message hook.
            async def _deliver() -> None:
                try:
                    await plugin.send_message(channel=channel, text="reply")
                except Exception:
                    pass

            asyncio.create_task(_deliver())

        plugin.pm.ahook.on_message = _fake_on_message

        response = await asyncio.wait_for(
            agent.prompt(session_id=created.session_id, prompt=[text_block("hi")]),
            timeout=2,
        )

        assert response.stop_reason == "end_turn"

    async def test_enqueue_failure_drops_the_future(self, tmp_path):
        """If the turn never starts, the future is cleared, not inherited."""
        from acp import text_block

        plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))

        async def _boom(channel, sender: str, text: str) -> None:
            raise RuntimeError("queue full")

        plugin.pm.ahook.on_message = _boom

        with pytest.raises(RuntimeError):
            await asyncio.wait_for(
                agent.prompt(session_id=created.session_id, prompt=[text_block("hi")]),
                timeout=2,
            )

        assert plugin._active_prompts.get(created.session_id) is None


class TestConcurrentPrompt:
    async def test_second_prompt_while_one_is_active_is_invalid_request(self, tmp_path):
        """Overwriting a live prompt future would orphan the first request."""
        from acp import RequestError, text_block

        plugin, _registry, agent = await _init_acp_plugin()
        created = await agent.new_session(cwd=str(tmp_path))
        enqueued = asyncio.Event()

        async def _enqueue(channel, sender: str, text: str) -> None:
            # A tool is in flight, so the turn's future stays pending.
            channel.pending_tool_call_ids.add("call_1")
            enqueued.set()

        plugin.pm.ahook.on_message = _enqueue

        first = asyncio.create_task(
            agent.prompt(session_id=created.session_id, prompt=[text_block("first")])
        )
        await asyncio.wait_for(enqueued.wait(), timeout=2)

        with pytest.raises(RequestError) as excinfo:
            await asyncio.wait_for(
                agent.prompt(session_id=created.session_id, prompt=[text_block("second")]),
                timeout=2,
            )

        assert excinfo.value.code == -32600  # invalid request
        # The first prompt is untouched and still cancellable.
        await agent.cancel(session_id=created.session_id)
        response = await asyncio.wait_for(first, timeout=2)
        assert response.stop_reason == "cancelled"
