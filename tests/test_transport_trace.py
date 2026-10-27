"""Transports mint the trace id (issue #95).

Pins the Agent-side contract (a caller-supplied id is used verbatim; a
missing/falsy id fails loudly), the new ``corvidae.trace`` leaf
module, and that cli / irc / discord each pass a minted id to
``pm.ahook.on_message``.  Outcome labels (O1..O6) are issue #95's
requirements.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pluggy
import pytest

from corvidae.hooks import hookimpl
from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response


# ---------------------------------------------------------------------------
# Recorder plugin and Agent fixture
# ---------------------------------------------------------------------------


class _Recorder:
    """Records the trace id each message-lifecycle hook receives."""

    def __init__(self, veto: bool = False) -> None:
        self.veto = veto
        # (hook name, trace id) in firing order.
        self.events: list[tuple[str, str]] = []

    def ids(self, hook: str) -> list[str]:
        """Trace ids seen by one hook, in order."""
        return [tid for name, tid in self.events if name == hook]

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        # Record the gate call; veto when configured.
        self.events.append(("gate", trace_id))
        return False if self.veto else None

    @hookimpl
    async def on_message_admitted(self, channel, trace_id, sender, text):
        self.events.append(("admitted", trace_id))

    @hookimpl
    async def on_message_rejected(self, channel, trace_id, sender, text):
        self.events.append(("rejected", trace_id))

    @hookimpl
    async def on_message_persisted(self, channel, trace_id, rowid, text, meta):
        self.events.append(("persisted", trace_id))


@pytest.fixture
async def agent_env():
    """A real Agent with a stub LLM client, plus a registered recorder."""
    plugin, channel, db = await build_plugin_and_channel()
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("hi"))
    plugin._client = client
    recorder = _Recorder()
    plugin.pm.register(recorder, name="recorder")
    yield plugin, channel, recorder
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


# ---------------------------------------------------------------------------
# O1: the caller-supplied id is the id every lifecycle hook sees
# ---------------------------------------------------------------------------


class TestAgentUsesCallerId:
    async def test_admit_path_uses_supplied_id(self, agent_env):
        """O1: a caller-supplied trace_id reaches should_process_message,
        on_message_admitted and on_message_persisted unchanged."""
        plugin, channel, rec = agent_env

        # Call through the plugin manager, as a transport does.
        await plugin.pm.ahook.on_message(
            channel=channel, sender="user", text="hello",
            trace_id="caller-id-1",
        )
        await drain(plugin, channel)

        # Same id everywhere, in the unchanged gate -> admitted -> persisted order (O5).
        assert rec.events == [
            ("gate", "caller-id-1"), ("admitted", "caller-id-1"), ("persisted", "caller-id-1"),
        ]

    async def test_veto_path_uses_supplied_id(self, agent_env):
        """O1: with a vetoing gate, should_process_message and
        on_message_rejected both receive the caller-supplied id."""
        plugin, channel, rec = agent_env
        rec.veto = True

        await plugin.pm.ahook.on_message(
            channel=channel, sender="user", text="hello",
            trace_id="caller-id-2",
        )

        # Same id on both, gate first, and no admission (O5).
        assert rec.events == [("gate", "caller-id-2"), ("rejected", "caller-id-2")]


# ---------------------------------------------------------------------------
# O2: a mis-wired transport fails loudly
# ---------------------------------------------------------------------------


class TestMissingIdFailsLoudly:
    async def test_omitted_trace_id_raises_hook_call_error(self, agent_env):
        """O2: a caller that omits trace_id gets HookCallError, not a
        silently-unlinked message."""
        plugin, channel, rec = agent_env

        # pluggy warns about the missing spec argument, then the Agent's
        # impl (which requires it) raises.
        with pytest.warns(UserWarning, match="trace_id"), \
                pytest.raises(pluggy.HookCallError):
            await plugin.pm.ahook.on_message(
                channel=channel, sender="user", text="hello",
            )
        # Nothing may have been gated or admitted.
        assert rec.events == []

    @pytest.mark.parametrize("bad", [None, ""])
    async def test_falsy_trace_id_raises_value_error(self, agent_env, bad):
        """O2: a None or empty trace_id raises ValueError before any
        gate hook fires (no silent fallback mint)."""
        plugin, channel, rec = agent_env

        with pytest.raises(ValueError):
            await plugin.pm.ahook.on_message(
                channel=channel, sender="user", text="hello",
                trace_id=bad,
            )
        assert rec.events == []

    async def test_falsy_id_raises_even_before_agent_is_ready(self, agent_env):
        """O2/D1: the id check runs before the readiness check, so a
        mis-wired transport fails loudly even while the agent is starting."""
        plugin, channel, rec = agent_env
        # Model the agent-not-ready window.
        plugin._client = None

        with pytest.raises(ValueError):
            await plugin.pm.ahook.on_message(
                channel=channel, sender="user", text="hello", trace_id=None,
            )
        assert rec.events == []


# ---------------------------------------------------------------------------
# D2: mint_trace_id lives in a leaf module
# ---------------------------------------------------------------------------


class TestMintTraceIdModule:
    def test_importable_and_unique(self):
        """O1: mint_trace_id is importable from corvidae.trace
        and returns distinct non-empty strings."""
        from corvidae.trace import mint_trace_id

        ids = {mint_trace_id() for _ in range(50)}
        assert len(ids) == 50
        assert all(isinstance(i, str) and i for i in ids)

    def test_agent_reexports_same_function(self):
        """O1: corvidae.agent keeps exposing the very same function so
        existing imports keep working."""
        import corvidae.agent
        import corvidae.trace

        assert corvidae.agent.mint_trace_id is corvidae.trace.mint_trace_id


# ---------------------------------------------------------------------------
# D3: transports pass a minted id
# ---------------------------------------------------------------------------


def _assert_minted(kwargs: dict) -> None:
    """The on_message call carried a non-empty string trace_id."""
    tid = kwargs.get("trace_id")
    assert isinstance(tid, str) and tid, kwargs


class TestTransportCallSites:
    async def test_cli_passes_trace_id(self):
        """O1: the CLI read loop passes a minted trace_id."""
        from corvidae.channels.cli import CLIPlugin
        from test_cli_plugin import _make_pm_with_registry

        pm, registry = _make_pm_with_registry(transport="cli")
        plugin = CLIPlugin(pm)
        pm.register(plugin, name="cli")
        plugin._registry = registry  # bypass on_start

        lines = iter(["hello world\n", ""])
        loop = asyncio.get_running_loop()

        async def fake_executor(executor, fn, *args):
            return fn(*args)

        # Drive one line then EOF, as test_read_loop_dispatches_on_message does.
        with patch.object(loop, "run_in_executor", side_effect=fake_executor), \
             patch("sys.stdin") as mock_stdin, \
             patch("sys.stdout"), \
             patch("corvidae.channels.cli.os.kill"):
            mock_stdin.readline = lambda: next(lines)
            await plugin._read_loop()

        _assert_minted(pm.ahook.on_message.await_args.kwargs)

    async def test_irc_channel_message_passes_trace_id(self):
        """O1: an IRC channel message passes a minted trace_id."""
        from corvidae.channels.irc import IRCPlugin
        from test_irc_plugin import _make_pm_with_registry

        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")
        plugin._registry = registry  # bypass on_start
        plugin.client = MagicMock()
        plugin.client.nickname = "TestBot"

        await plugin.on_channel_message(target="#test", by="user", message="hello bot")

        _assert_minted(pm.ahook.on_message.await_args.kwargs)

    async def test_irc_private_message_passes_trace_id(self):
        """O1: an IRC private message passes a minted trace_id."""
        from corvidae.channels.irc import IRCPlugin
        from test_irc_plugin import _make_pm_with_registry

        pm, registry = _make_pm_with_registry()
        plugin = IRCPlugin(pm)
        pm.register(plugin, name="irc")
        plugin._registry = registry  # bypass on_start
        plugin.client = MagicMock()
        plugin.client.nickname = "TestBot"

        await plugin.on_private_message(target="TestBot", by="user", message="psst")

        _assert_minted(pm.ahook.on_message.await_args.kwargs)

    async def test_discord_passes_distinct_trace_ids(self):
        """O1: Discord passes a minted trace_id, distinct per message."""
        from test_discord_plugin import _deliver, _start_plugin, make_dm

        h = await _start_plugin()
        try:
            await _deliver(h, make_dm("one"), make_dm("two"))
            calls = [c.kwargs for c in h.on_message.await_args_list]
            for kw in calls:
                _assert_minted(kw)
            ids = [kw["trace_id"] for kw in calls]
            assert len(set(ids)) == len(ids), ids
        finally:
            await h.plugin.on_stop()
