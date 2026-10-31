"""Per-channel tool sets (GitHub issue #149, option 1).

``channels.<id>.tools.disabled`` narrows a channel's tool set on top of the
global ``tools.disabled``. The narrowed schemas are fixed for the
conversation (principle 2), shared with forks (`docs/design.md`, Forks), and enforced at
dispatch, not only hidden: a model can call a tool it saw earlier or one
named in injected text.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.agent import mint_trace_id
from corvidae.channel import Channel, ChannelConfig, ChannelRegistry, load_channel_config
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.tool import Tool, ToolContext, ToolRegistry, tool_to_schema
from corvidae.tool_collection import ToolCollectionPlugin
from corvidae.tool_policy import disabled_on, expand_disabled

from helpers import build_plugin_and_channel, drain, drain_task_queue
from llm_response_fixtures import _make_text_response, _make_tool_call, _make_tool_call_response


# ---------------------------------------------------------------------------
# Fixtures: three fake tools standing in for the built-ins
# ---------------------------------------------------------------------------


def _fake_tools(calls: list[str]) -> dict:
    """shell, task_pipeline and read_file stand-ins that record each call."""

    async def shell(command: str) -> str:
        """Run a command."""
        calls.append("shell")
        return "ran"

    async def task_pipeline(definition: str) -> str:
        """Run a pipeline."""
        calls.append("task_pipeline")
        return "piped"

    async def read_file(path: str) -> str:
        """Read a file."""
        calls.append("read_file")
        return "contents"

    return {"shell": shell, "task_pipeline": task_pipeline, "read_file": read_file}


async def _agent_with_tools(disabled: frozenset[str] = frozenset()):
    """(agent, channel, db, calls): an Agent whose channel disables ``disabled``."""
    plugin, channel, db = await build_plugin_and_channel(
        channel_config=ChannelConfig(disabled_tools=disabled),
    )
    calls: list[str] = []
    plugin._tools = _fake_tools(calls)
    plugin._tool_schemas = [tool_to_schema(fn) for fn in plugin._tools.values()]
    return plugin, channel, db, calls


def _schema_names(schemas) -> list[str]:
    return [s["function"]["name"] for s in schemas or []]


async def _teardown(plugin, db) -> None:
    await plugin.pm.get_plugin("task").on_stop()
    await db.close()


# Arguments each fake tool's schema accepts, so a refusal is never an
# argument-validation error in disguise.
_VALID_ARGS = {
    "shell": {"command": "id"},
    "task_pipeline": {"definition": "steps: []"},
    "read_file": {"path": "notes.txt"},
}


async def _run_tool_cycle(plugin, channel, tool_name: str, final: str = "done") -> MagicMock:
    """Drive one user message whose reply calls ``tool_name``, then a final
    text turn after the result. Returns the mock client."""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=[
        _make_tool_call_response([_make_tool_call("call_1", tool_name, _VALID_ARGS[tool_name])]),
        _make_text_response(final),
    ])
    plugin._client = client
    plugin._mark_started()
    await plugin.on_message(channel=channel, sender="u", text="go", trace_id=mint_trace_id())
    await drain(plugin, channel)
    await drain_task_queue(plugin)
    await drain(plugin, channel)
    return client


def _tool_result_sent(client: MagicMock) -> str:
    """The tool-role message content the second LLM call carried."""
    messages = client.chat.call_args_list[1].args[0]
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    return tool_msg["content"]


# ---------------------------------------------------------------------------
# Config parsing and validation
# ---------------------------------------------------------------------------


class TestChannelConfigParsing:
    def test_channel_tools_disabled_lands_on_channel_config(self):
        registry = ChannelRegistry({})
        load_channel_config(
            {"channels": {"irc:#lex": {"tools": {"disabled": ["shell", "web_fetch"]}}}},
            registry,
        )
        assert registry.get("irc:#lex").config.disabled_tools == frozenset({"shell", "web_fetch"})

    def test_absent_tools_block_disables_nothing(self):
        registry = ChannelRegistry({})
        load_channel_config({"channels": {"irc:#lex": {"max_turns": 3}}}, registry)
        assert registry.get("irc:#lex").config.disabled_tools == frozenset()

    @pytest.mark.parametrize("tools_value, fragments", [
        ("shell", ["channels.irc:#lex.tools", "'shell'", "mapping", "disabled"]),
        ({"disabled": "shell"}, ["channels.irc:#lex.tools.disabled", "'shell'", "list of tool names"]),
        ({"disabled": ["shell", 3]}, ["channels.irc:#lex.tools.disabled", "3", "list of tool names"]),
        ({"enabled": ["shell"]}, ["channels.irc:#lex.tools", "'enabled'", "disabled"]),
    ])
    def test_rejected_config_names_key_value_and_accepted(self, tools_value, fragments):
        registry = ChannelRegistry({})
        with pytest.raises(ValueError) as excinfo:
            load_channel_config({"channels": {"irc:#lex": {"tools": tools_value}}}, registry)
        for fragment in fragments:
            assert fragment in str(excinfo.value)

    def test_one_bad_entry_registers_nothing(self):
        """A malformed reload changes nothing (PRD §3.11.3): every entry is
        validated before any is applied."""
        registry = ChannelRegistry({})
        with pytest.raises(ValueError):
            load_channel_config({"channels": {
                "irc:#ok": {"tools": {"disabled": ["shell"]}},
                "irc:#bad": {"tools": {"disabled": "shell"}},
            }}, registry)
        assert registry.all() == []


class TestShellCoupling:
    @pytest.mark.parametrize("named", ["shell", "task_pipeline"])
    def test_either_name_disables_both(self, named):
        assert expand_disabled([named]) == frozenset({"shell", "task_pipeline"})

    def test_other_names_pass_through(self):
        assert expand_disabled(["web_fetch"]) == frozenset({"web_fetch"})

    async def test_global_disabled_shell_also_drops_task_pipeline(self):
        pm = create_plugin_manager()

        class Tools:
            @hookimpl
            def register_tools(self, tool_registry):
                for fn in _fake_tools([]).values():
                    tool_registry.append(fn)

        pm.register(Tools(), name="fake_tools")
        plugin = ToolCollectionPlugin(pm)
        pm.register(plugin, name="tools")
        await plugin.on_init(pm=pm, config={"tools": {"disabled": ["shell"]}})
        await plugin.on_start(config={})
        assert sorted(plugin.get_registry().as_dict()) == ["read_file"]

    async def test_global_disabled_must_be_a_list(self):
        pm = create_plugin_manager()
        plugin = ToolCollectionPlugin(pm)
        with pytest.raises(ValueError) as excinfo:
            await plugin.on_init(pm=pm, config={"tools": {"disabled": "shell"}})
        message = str(excinfo.value)
        assert "tools.disabled" in message and "'shell'" in message
        assert "list of tool names" in message


# ---------------------------------------------------------------------------
# Schemas: per channel, fixed for the conversation, shared with forks
# ---------------------------------------------------------------------------


class TestChannelSchemas:
    async def test_disabled_tool_is_absent_from_that_channel_only(self):
        plugin, channel, db, _ = await _agent_with_tools(frozenset({"shell"}))
        try:
            other = plugin._registry.get_or_create("test", "other")
            client = MagicMock()
            client.chat = AsyncMock(return_value=_make_text_response("hi"))
            plugin._client = client
            plugin._mark_started()

            await plugin.on_message(channel=channel, sender="u", text="a", trace_id=mint_trace_id())
            await drain(plugin, channel)
            await plugin.on_message(channel=other, sender="u", text="b", trace_id=mint_trace_id())
            await drain(plugin, other)

            restricted, unrestricted = (c.kwargs["tools"] for c in client.chat.call_args_list)
            # task_pipeline goes with shell (PRD §3.5.3).
            assert _schema_names(restricted) == ["read_file"]
            assert _schema_names(unrestricted) == ["shell", "task_pipeline", "read_file"]
        finally:
            await _teardown(plugin, db)

    async def test_schemas_are_byte_identical_across_turns(self):
        """Principle 2: the tool schemas head the prompt, so every turn of a
        conversation must send exactly the same bytes."""
        plugin, channel, db, _ = await _agent_with_tools(frozenset({"shell"}))
        try:
            client = MagicMock()
            client.chat = AsyncMock(return_value=_make_text_response("ok"))
            plugin._client = client
            plugin._mark_started()
            for text in ("one", "two", "three"):
                await plugin.on_message(channel=channel, sender="u", text=text, trace_id=mint_trace_id())
                await drain(plugin, channel)

            sent = [json.dumps(c.kwargs["tools"], sort_keys=False) for c in client.chat.call_args_list]
            assert len(sent) == 3
            assert sent[0] == sent[1] == sent[2]
            assert "\"shell\"" not in sent[0]
        finally:
            await _teardown(plugin, db)

    async def test_fork_sends_the_channels_schemas(self):
        """Prefix parity (`docs/design.md`, Forks): a fork of the channel uses the same
        per-channel schemas the turn did."""
        plugin, channel, db, _ = await _agent_with_tools(frozenset({"shell"}))
        try:
            client = MagicMock()
            client.chat = AsyncMock(return_value=_make_text_response("ok"))
            plugin._client = client
            plugin._mark_started()
            await plugin.on_message(channel=channel, sender="u", text="a", trace_id=mint_trace_id())
            await drain(plugin, channel)
            turn_tools = client.chat.call_args_list[0].kwargs["tools"]

            await plugin.fork(channel, "summarize")
            fork_tools = client.chat.call_args_list[1].kwargs["tools"]
            assert json.dumps(fork_tools) == json.dumps(turn_tools)
        finally:
            await _teardown(plugin, db)


# ---------------------------------------------------------------------------
# Enforcement at dispatch
# ---------------------------------------------------------------------------


class TestDispatchEnforcement:
    async def test_disabled_tool_call_is_refused_with_a_named_error(self, caplog):
        plugin, channel, db, calls = await _agent_with_tools(frozenset({"shell"}))
        try:
            with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
                client = await _run_tool_cycle(plugin, channel, "shell")
            assert calls == []
            result = _tool_result_sent(client)
            assert "'shell'" in result and channel.id in result and "disabled" in result
            assert any(
                "shell" in r.getMessage() and channel.id in r.getMessage()
                for r in caplog.records if r.levelno == logging.WARNING
            )
        finally:
            await _teardown(plugin, db)

    async def test_coupled_tool_is_refused_too(self):
        plugin, channel, db, calls = await _agent_with_tools(frozenset({"shell"}))
        try:
            client = await _run_tool_cycle(plugin, channel, "task_pipeline")
            assert calls == []
            assert "'task_pipeline'" in _tool_result_sent(client)
        finally:
            await _teardown(plugin, db)

    async def test_enabled_tool_still_runs(self):
        """Discriminating control: the same cycle with an allowed tool runs it."""
        plugin, channel, db, calls = await _agent_with_tools(frozenset({"shell"}))
        try:
            await _run_tool_cycle(plugin, channel, "read_file")
            assert calls == ["read_file"]
        finally:
            await _teardown(plugin, db)


# ---------------------------------------------------------------------------
# Hot reload: dispatch narrows at once, schemas at the next boundary
# ---------------------------------------------------------------------------


class TestReload:
    async def test_reload_updates_an_existing_channels_list(self):
        registry = ChannelRegistry({})
        load_channel_config({"channels": {"irc:#lex": {"max_turns": 3}}}, registry)
        channel = registry.get("irc:#lex")
        load_channel_config(
            {"channels": {"irc:#lex": {"max_turns": 9, "tools": {"disabled": ["shell"]}}}},
            registry,
        )
        assert registry.get("irc:#lex") is channel
        assert channel.config.disabled_tools == frozenset({"shell"})
        # Only the tool list is live; other keys still need a restart.
        assert channel.config.max_turns == 3

    async def test_narrowing_reload_refuses_at_once_and_reschemas_at_compaction(self):
        plugin, channel, db, calls = await _agent_with_tools()
        try:
            client = MagicMock()
            client.chat = AsyncMock(return_value=_make_text_response("ok"))
            plugin._client = client
            plugin._mark_started()
            await plugin.on_message(channel=channel, sender="u", text="a", trace_id=mint_trace_id())
            await drain(plugin, channel)
            before = client.chat.call_args_list[0].kwargs["tools"]
            assert "shell" in _schema_names(before)

            # The operator narrows the channel mid-conversation.
            channel.config.disabled_tools = frozenset({"shell"})
            assert "shell" in disabled_on(channel)

            # Dispatch refuses at once...
            cycle_client = await _run_tool_cycle(plugin, channel, "shell")
            assert calls == []
            # ...but the schemas this conversation has been sending do not move.
            assert len(cycle_client.chat.call_args_list) == 2
            for c in cycle_client.chat.call_args_list:
                assert json.dumps(c.kwargs["tools"]) == json.dumps(before)

            # At the compaction boundary the schemas pick up the new list.
            await plugin.on_compaction(
                channel=channel, summary_msg={"role": "assistant", "content": "s"},
                retain_count=1, compacted_ids=[],
            )
            client = MagicMock()
            client.chat = AsyncMock(return_value=_make_text_response("ok"))
            plugin._client = client
            plugin._mark_started()
            await plugin.on_message(channel=channel, sender="u", text="b", trace_id=mint_trace_id())
            await drain(plugin, channel)
            assert _schema_names(client.chat.call_args.kwargs["tools"]) == ["read_file"]
        finally:
            await _teardown(plugin, db)

    async def test_widening_reload_waits_for_the_boundary(self):
        """Re-enabling is a schema change too: until the boundary the tool
        stays hidden and refused, so the head of the prompt never moves."""
        plugin, channel, db, calls = await _agent_with_tools(frozenset({"shell"}))
        try:
            client = MagicMock()
            client.chat = AsyncMock(return_value=_make_text_response("ok"))
            plugin._client = client
            plugin._mark_started()
            await plugin.on_message(channel=channel, sender="u", text="a", trace_id=mint_trace_id())
            await drain(plugin, channel)

            channel.config.disabled_tools = frozenset()
            await _run_tool_cycle(plugin, channel, "shell")
            assert calls == []
        finally:
            await _teardown(plugin, db)


# ---------------------------------------------------------------------------
# Subagents inherit the calling channel's restriction
# ---------------------------------------------------------------------------


class TestSubagentInherits:
    async def test_subagent_cannot_reach_a_tool_disabled_on_its_channel(self):
        from corvidae.llm_plugin import LLMPlugin
        from corvidae.task import TaskQueue
        from corvidae.tools.subagent import SubagentPlugin

        pm = create_plugin_manager()
        pm.register(ChannelRegistry({}), name="registry")
        llm = LLMPlugin(pm)
        llm.main_client = MagicMock()
        llm.background_client = None
        pm.register(llm, name="llm")
        registry = ToolRegistry()
        for name, fn in _fake_tools([]).items():
            registry.add(Tool(name=name, fn=fn, schema={}))
        tools_plugin = ToolCollectionPlugin(pm)
        tools_plugin.registry = registry
        pm.register(tools_plugin, name="tools")
        plugin = SubagentPlugin(pm)
        await plugin.on_start(config={"llm": {"main": {"base_url": "x", "model": "m"}}})

        channel = Channel("irc", "#lex", config=ChannelConfig(disabled_tools=frozenset({"shell"})))
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=channel, tool_call_id="c1", task_queue=task_queue)

        captured: dict = {}

        async def capture(client, messages, tools, tool_schemas, **kwargs):
            captured.update(tools)
            return "done"

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=capture), \
             patch("corvidae.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("do it", "desc", ctx)
            await task_queue.enqueue.call_args[0][0].work()

        assert sorted(captured) == ["read_file"]
