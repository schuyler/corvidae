"""A running Agent picks up registry changes on its next turn (issue #109).

A lazily reconnected MCP server rebuilds the tool registry; the Agent must
advertise and dispatch the new tools without a restart, rebuild nothing
when the registry is unchanged, and keep each conversation's per-channel
narrowing (#149).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.agent import mint_trace_id
from corvidae.channel import ChannelConfig
from corvidae.hooks import hookimpl
from corvidae.tool_collection import ToolCollectionPlugin

from helpers import build_plugin_and_channel, drain, drain_task_queue
from llm_response_fixtures import _make_text_response, _make_tool_call, _make_tool_call_response


def _fn(name: str, calls: list[str], result: str = "ok"):
    """An async tool named ``name`` that records each call."""

    async def tool(x: str = "") -> str:
        calls.append(name)
        return result

    tool.__name__ = name
    tool.__doc__ = f"The {name} tool."
    return tool


async def _setup(disabled: frozenset[str] = frozenset()):
    """(agent, channel, db, tools_plugin, fns, calls): an Agent snapshotting a
    live ToolCollectionPlugin whose tools are the mutable list ``fns``."""
    plugin, channel, db = await build_plugin_and_channel(
        channel_config=ChannelConfig(disabled_tools=disabled),
    )
    calls: list[str] = []
    fns = [_fn("shell", calls), _fn("read_file", calls)]

    class Source:
        @hookimpl
        def register_tools(self, tool_registry):
            tool_registry.extend(fns)

    plugin.pm.register(Source(), name="fake_source")
    tools_plugin = ToolCollectionPlugin(plugin.pm)
    plugin.pm.register(tools_plugin, name="tools")
    await tools_plugin.on_init(pm=plugin.pm, config={})
    await tools_plugin.on_start(config={})
    # What _start_plugin does with the tools plugin.
    plugin._snapshot_tools(tools_plugin)
    return plugin, channel, db, tools_plugin, fns, calls


def _names(schemas) -> list[str]:
    return [s["function"]["name"] for s in schemas or []]


async def _turn(plugin, channel, *responses, text: str = "hi") -> MagicMock:
    """Run one user message through the Agent against canned responses."""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=list(responses))
    plugin._client = client
    plugin._mark_started()
    await plugin.on_message(channel=channel, sender="u", text=text, trace_id=mint_trace_id())
    await drain(plugin, channel)
    await drain_task_queue(plugin)
    await drain(plugin, channel)
    return client


def _tools_sent(client: MagicMock, call: int = 0) -> list[str]:
    return _names(client.chat.call_args_list[call].kwargs.get("tools"))


async def _teardown(plugin, db) -> None:
    await plugin.pm.get_plugin("task").on_stop()
    await db.close()


class TestRefresh:
    async def test_reconnected_tool_is_advertised_and_dispatched(self):
        """R2: a new tool reaches the next turn's `tools` and dispatches."""
        plugin, channel, db, tp, fns, calls = await _setup()
        try:
            await _turn(plugin, channel, _make_text_response("one"))
            fns.append(_fn("srv__echo", calls))
            await tp.rebuild_registry()
            client = await _turn(
                plugin, channel,
                _make_tool_call_response([_make_tool_call("c1", "srv__echo", {"x": "a"})]),
                _make_text_response("two"),
            )
            assert "srv__echo" in _tools_sent(client, 0)
            assert calls == ["srv__echo"]
        finally:
            await _teardown(plugin, db)

    async def test_unchanged_registry_rebuilds_nothing(self):
        """R3: same schema objects across turns, no build_tool_view call."""
        plugin, channel, db, tp, fns, calls = await _setup()
        try:
            await _turn(plugin, channel, _make_text_response("one"))
            agent_schemas = plugin._tool_schemas
            view_schemas = channel.conversation.tool_view.schemas
            with patch("corvidae.agent.build_tool_view") as build:
                await _turn(plugin, channel, _make_text_response("two"))
            build.assert_not_called()
            assert plugin._tool_schemas is agent_schemas
            assert channel.conversation.tool_view.schemas is view_schemas
        finally:
            await _teardown(plugin, db)

    async def test_identical_rebuild_is_not_a_change(self):
        """R3: a rebuild yielding the same tools leaves the schemas alone."""
        plugin, channel, db, tp, fns, calls = await _setup()
        try:
            await _turn(plugin, channel, _make_text_response("one"))
            view_schemas = channel.conversation.tool_view.schemas
            await tp.rebuild_registry()
            await _turn(plugin, channel, _make_text_response("two"))
            assert channel.conversation.tool_view.schemas is view_schemas
        finally:
            await _teardown(plugin, db)

    async def test_removed_tool_is_unadvertised_and_refused(self):
        """R4: a tool that leaves the registry is gone and its call refused."""
        plugin, channel, db, tp, fns, calls = await _setup()
        try:
            fns.append(_fn("srv__echo", calls))
            await tp.rebuild_registry()
            plugin._snapshot_tools(tp)
            await _turn(plugin, channel, _make_text_response("one"))
            fns.pop()
            await tp.rebuild_registry()
            client = await _turn(
                plugin, channel,
                _make_tool_call_response([_make_tool_call("c1", "srv__echo", {"x": "a"})]),
                _make_text_response("two"),
            )
            assert "srv__echo" not in _tools_sent(client, 0)
            messages = client.chat.call_args_list[1].args[0]
            (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
            assert "srv__echo" in tool_msg["content"]
            assert calls == []
        finally:
            await _teardown(plugin, db)

    async def test_refresh_keeps_conversation_disabled_set(self):
        """R5: the refresh keeps the view's disabled set, even if config moved."""
        plugin, channel, db, tp, fns, calls = await _setup(frozenset({"shell"}))
        try:
            await _turn(plugin, channel, _make_text_response("one"))
            previous = channel.conversation.tool_view.disabled
            assert "shell" in previous
            # A config reload that re-enables shell must wait for compaction.
            channel.config = ChannelConfig(disabled_tools=frozenset())
            fns.append(_fn("srv__echo", calls))
            await tp.rebuild_registry()
            client = await _turn(plugin, channel, _make_text_response("two"))
            sent = _tools_sent(client, 0)
            assert "srv__echo" in sent
            assert "shell" not in sent
            assert channel.conversation.tool_view.disabled == previous
        finally:
            await _teardown(plugin, db)

    async def test_refresh_is_logged_once_with_diff(self, caplog):
        """R6: one INFO record naming added, removed and the generation."""
        plugin, channel, db, tp, fns, calls = await _setup()
        try:
            await _turn(plugin, channel, _make_text_response("one"))
            fns.append(_fn("srv__echo", calls))
            fns.pop(0)  # drops shell
            await tp.rebuild_registry()
            with caplog.at_level(logging.INFO, logger="corvidae.agent"):
                await _turn(plugin, channel, _make_text_response("two"))
                await _turn(plugin, channel, _make_text_response("three"))
            records = [r for r in caplog.records if "tool registry changed" in r.getMessage()]
            assert len(records) == 1
            text = records[0].getMessage()
            assert records[0].levelno == logging.INFO
            assert "srv__echo" in text and "shell" in text
            assert f"generation {tp.generation}" in text
        finally:
            await _teardown(plugin, db)
