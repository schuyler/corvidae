"""Tests for corvidae.tools.fork_tool — ForkToolPlugin.

The fork tool is the subagent pattern with one load-bearing difference: the
system prompt is the agent's own identity (``Agent.channel_system_prompt``),
not a generic subagent prompt. These tests verify that, plus the shared
subagent machinery (registration, launch, tool exclusion, LLM stage).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import create_plugin_manager
from corvidae.task import Task, TaskQueue
from corvidae.tool import Tool, ToolContext, ToolRegistry

from corvidae.agent import Agent
from corvidae.llm_plugin import LLMPlugin
from corvidae.tools.fork_tool import ForkToolPlugin
from helpers import KERNEL_LLM_STAGES, collect_llm_stages, stub_llm_config


MY_PROMPT = "I am Lex. Researcher, software engineer, and collaborator on both."

BASE_CONFIG = {
    "llm": {
        "main": {
            "base_url": "http://localhost:8080",
            "model": "test-model",
        },
    },
}


def _make_channel(transport="test", scope="scope1") -> Channel:
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


def _make_pm_with_agent(*tool_names: str):
    """Build a plugin manager with a ToolCollectionPlugin, a mock LLMPlugin,
    and a mock Agent whose channel_system_prompt returns MY_PROMPT."""
    from corvidae.tool_collection import ToolCollectionPlugin

    pm = create_plugin_manager()

    from corvidae.channel import ChannelRegistry

    pm.register(ChannelRegistry({}), name="registry")

    mock_llm = LLMPlugin(pm)
    mock_llm._stages = dict(KERNEL_LLM_STAGES)
    mock_llm._stages["fork"] = "chat"
    mock_llm._clients["main"] = MagicMock(name="main")
    stub_llm_config(mock_llm)
    pm.register(mock_llm, name="llm")

    # The mock Agent: its channel_system_prompt is my identity.
    mock_agent = Agent.__new__(Agent)
    mock_agent.channel_system_prompt = AsyncMock(return_value=MY_PROMPT)
    pm.register(mock_agent, name="agent")

    tool_registry = ToolRegistry()
    for name in tool_names:
        async def _placeholder() -> str:
            """Placeholder tool."""
            return "ok"

        _placeholder.__name__ = name
        tool_registry.add(Tool(name=name, fn=_placeholder, schema={}))

    tools_plugin = ToolCollectionPlugin(pm)
    tools_plugin.registry = tool_registry
    pm.register(tools_plugin, name="tools")

    return pm, tool_registry


async def _make_started_plugin(config=None) -> ForkToolPlugin:
    if config is None:
        config = BASE_CONFIG
    pm, _ = _make_pm_with_agent("shell", "fork")
    plugin = ForkToolPlugin(pm)
    await plugin.on_start(config=config)
    return plugin


class TestToolRegistration:
    async def test_register_tools_adds_fork_tool(self):
        pm = create_plugin_manager()
        plugin = ForkToolPlugin(pm)
        tool_list = []
        plugin.register_tools(tool_registry=tool_list)
        tool_names = [t.name for t in tool_list]
        assert "fork" in tool_names

    async def test_register_tools_appends_tool_instances(self):
        pm = create_plugin_manager()
        plugin = ForkToolPlugin(pm)
        tool_list = []
        plugin.register_tools(tool_registry=tool_list)
        assert len(tool_list) >= 1
        for item in tool_list:
            assert isinstance(item, Tool)


class TestLLMStage:
    async def test_fork_registers_its_stage_as_chat(self):
        plugin = ForkToolPlugin(create_plugin_manager())
        assert collect_llm_stages(plugin, "fork") == {"fork": "chat"}


class TestSystemPrompt:
    """The load-bearing property: the fork runs with MY prompt, fresh."""

    async def test_launch_uses_agent_system_prompt_not_generic(self):
        """The fork's messages list is headed by the agent's identity prompt,
        not a generic subagent prompt."""
        plugin = await _make_started_plugin()
        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=channel, tool_call_id="call-sp-01", task_queue=task_queue)

        captured = {}

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured["messages"] = list(messages)
            return "done"

        with patch("corvidae.tools.fork_tool.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("corvidae.tools.fork_tool.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("review this draft", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        msgs = captured["messages"]
        assert msgs[0] == {"role": "system", "content": MY_PROMPT}, (
            "the fork's system prompt must be the agent's identity, "
            "not a generic subagent prompt"
        )
        assert msgs[1] == {"role": "user", "content": "review this draft"}

    async def test_launch_builds_a_fresh_window_no_history(self):
        """The fork's messages list is exactly [my prompt, the task] — no
        channel history. That's the freshness that gives it fresh eyes."""
        plugin = await _make_started_plugin()
        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=channel, tool_call_id="call-fresh-01", task_queue=task_queue)

        captured = {}

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured["messages"] = list(messages)
            return "done"

        with patch("corvidae.tools.fork_tool.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("corvidae.tools.fork_tool.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("fact-check these claims", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        msgs = captured["messages"]
        assert len(msgs) == 2, (
            "the fork starts from a fresh window (my prompt + the task), "
            "with no channel history"
        )

    async def test_launch_asks_agent_for_the_system_prompt(self):
        """The fork calls Agent.channel_system_prompt for the invoking channel."""
        plugin = await _make_started_plugin()
        agent = plugin.pm.get_plugin("agent")
        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=channel, tool_call_id="call-agent-01", task_queue=task_queue)

        with patch("corvidae.tools.fork_tool.run_agent_loop",
                   side_effect=lambda *a, **k: _async_done()):
            await plugin._launch("task", ctx)

        agent.channel_system_prompt.assert_awaited_once_with(channel)


class TestToolExclusion:
    async def test_work_excludes_fork_from_tool_registry(self):
        """The tool registry passed to run_agent_loop does not contain 'fork'."""
        plugin = await _make_started_plugin()
        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=channel, tool_call_id="call-excl-01", task_queue=task_queue)

        captured_tools = {}

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured_tools.update(tools)
            return "done"

        with patch("corvidae.tools.fork_tool.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("corvidae.tools.fork_tool.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("task", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        assert "fork" not in captured_tools, (
            "fork must not appear in the tool registry passed to run_agent_loop "
            "(no recursion)"
        )
        assert "shell" in captured_tools, (
            "non-excluded tools (e.g. 'shell') must appear in the registry"
        )


class TestLaunch:
    async def test_launch_no_task_queue_returns_error(self):
        plugin = await _make_started_plugin()
        ctx = ToolContext(channel=_make_channel(), tool_call_id="call-01", task_queue=None)
        result = await plugin._launch("task", ctx)
        assert "task queue not available" in result

    async def test_launch_no_channel_returns_error(self):
        plugin = await _make_started_plugin()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=None, tool_call_id="call-02", task_queue=task_queue)
        result = await plugin._launch("task", ctx)
        assert "no channel context" in result
        task_queue.enqueue.assert_not_awaited()

    async def test_launch_enqueues_exactly_one_task(self):
        plugin = await _make_started_plugin()
        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=channel, tool_call_id="call-03", task_queue=task_queue)
        await plugin._launch("task", ctx)
        task_queue.enqueue.assert_awaited_once()
        enqueued_task = task_queue.enqueue.call_args[0][0]
        assert isinstance(enqueued_task, Task)

    async def test_launch_return_contains_task_id(self):
        plugin = await _make_started_plugin()
        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=channel, tool_call_id="call-04", task_queue=task_queue)
        result = await plugin._launch("task", ctx)
        enqueued_task = task_queue.enqueue.call_args[0][0]
        assert enqueued_task.task_id in result

    async def test_launch_asks_for_the_fork_stage(self):
        """The fork call site requests the 'fork' stage client."""
        plugin = await _make_started_plugin()
        llm_plugin = plugin.pm.get_plugin("llm")
        llm_plugin.get_client = MagicMock(side_effect=llm_plugin.get_client)
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()
        ctx = ToolContext(channel=_make_channel(), tool_call_id="call-stage-01",
                          task_queue=task_queue)
        await plugin._launch("task", ctx)
        llm_plugin.get_client.assert_called_once_with("fork")


async def _async_done() -> str:
    return "done"
