"""Tests for sherman.tools.subagent — SubagentPlugin.

These are Red TDD tests. They will fail with ImportError until
sherman/tools/subagent.py is implemented.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sherman.channel import Channel, ChannelConfig
from sherman.hooks import create_plugin_manager
from sherman.task import Task, TaskQueue
from sherman.tool import Tool, ToolContext, ToolRegistry

from sherman.tools.subagent import SubagentPlugin  # ImportError until implemented


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


BASE_CONFIG = {
    "llm": {
        "main": {
            "base_url": "http://localhost:8080",
            "model": "test-model",
        },
    },
}

BACKGROUND_CONFIG = {
    "llm": {
        "main": {
            "base_url": "http://localhost:8080",
            "model": "test-model",
        },
        "background": {
            "base_url": "http://localhost:8081",
            "model": "bg-model",
        },
    },
}


def _make_channel(transport="test", scope="scope1") -> Channel:
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


def _make_pm_with_tool_registry(*tool_names: str):
    """Build a minimal plugin manager with a mock agent_plugin that has a
    ToolRegistry containing named placeholder tools."""
    pm = create_plugin_manager()

    registry = ToolRegistry()
    for name in tool_names:
        async def _placeholder() -> str:
            """Placeholder tool."""
            return "ok"
        _placeholder.__name__ = name
        registry.add(Tool(name=name, fn=_placeholder, schema={}))

    mock_agent_plugin = MagicMock()
    mock_agent_plugin.tool_registry = registry
    pm.agent_plugin = mock_agent_plugin

    return pm, registry


async def _make_started_plugin(config=None) -> SubagentPlugin:
    """Create a SubagentPlugin and call on_start with the given config."""
    if config is None:
        config = BASE_CONFIG
    pm, _ = _make_pm_with_tool_registry("shell", "subagent")
    plugin = SubagentPlugin(pm)
    await plugin.on_start(config=config)
    return plugin


# ---------------------------------------------------------------------------
# TestToolRegistration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    async def test_register_tools_adds_subagent_tool(self):
        """After register_tools is called, the tool registry contains 'subagent'."""
        pm = create_plugin_manager()
        plugin = SubagentPlugin(pm)
        # Simulate the register_tools hook with a plain list (as hookspec specifies)
        tool_list = []
        plugin.register_tools(tool_registry=tool_list)

        tool_names = [t.name for t in tool_list]
        assert "subagent" in tool_names

    async def test_register_tools_schema_includes_instructions_and_description(self):
        """The subagent tool schema includes 'instructions' and 'description'
        parameters and omits '_ctx'."""
        pm = create_plugin_manager()
        plugin = SubagentPlugin(pm)
        tool_list = []
        plugin.register_tools(tool_registry=tool_list)

        subagent_tool = next(t for t in tool_list if t.name == "subagent")
        schema = subagent_tool.schema
        properties = schema["function"]["parameters"].get("properties", {})

        assert "instructions" in properties, "schema must include 'instructions'"
        assert "description" in properties, "schema must include 'description'"
        assert "_ctx" not in properties, "schema must omit '_ctx'"

    async def test_register_tools_appends_tool_instances(self):
        """register_tools appends Tool instances (not bare callables) to the list."""
        pm = create_plugin_manager()
        plugin = SubagentPlugin(pm)
        tool_list = []
        plugin.register_tools(tool_registry=tool_list)

        assert len(tool_list) >= 1
        for item in tool_list:
            assert isinstance(item, Tool), f"Expected Tool, got {type(item)}"


# ---------------------------------------------------------------------------
# TestLaunchWithMockedDependencies
# ---------------------------------------------------------------------------


class TestLaunchWithMockedDependencies:
    async def test_launch_no_task_queue_returns_error(self):
        """When ctx.task_queue is None, _launch returns the task queue error string."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-001",
            task_queue=None,
        )
        result = await plugin._launch("do something", "test task", ctx)
        assert result == "Error: task queue not available"

    async def test_launch_no_channel_returns_error(self):
        """When ctx.channel is None, _launch returns the channel error string."""
        plugin = await _make_started_plugin()

        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=None,
            tool_call_id="call-002",
            task_queue=task_queue,
        )
        result = await plugin._launch("do something", "test task", ctx)
        assert result == "Error: no channel context available for subagent"

    async def test_launch_enqueues_exactly_one_task(self):
        """When both channel and task_queue are present, exactly one Task is enqueued."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-003",
            task_queue=task_queue,
        )
        await plugin._launch("do something", "my description", ctx)

        task_queue.enqueue.assert_awaited_once()
        enqueued_task = task_queue.enqueue.call_args[0][0]
        assert isinstance(enqueued_task, Task)

    async def test_launch_return_contains_task_id(self):
        """The return value contains the task id."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-004",
            task_queue=task_queue,
        )
        result = await plugin._launch("do something", "my description", ctx)

        enqueued_task = task_queue.enqueue.call_args[0][0]
        assert enqueued_task.task_id in result

    async def test_launch_return_contains_description(self):
        """The return value contains the description."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-005",
            task_queue=task_queue,
        )
        description = "summarize the docs"
        result = await plugin._launch("do something", description, ctx)

        assert description in result

    async def test_launch_enqueued_task_has_correct_channel(self):
        """The enqueued Task has the channel from ctx."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-006",
            task_queue=task_queue,
        )
        await plugin._launch("do something", "desc", ctx)

        enqueued_task = task_queue.enqueue.call_args[0][0]
        assert enqueued_task.channel is channel

    async def test_launch_enqueued_task_has_correct_tool_call_id(self):
        """The enqueued Task carries the tool_call_id from ctx."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-007",
            task_queue=task_queue,
        )
        await plugin._launch("do something", "desc", ctx)

        enqueued_task = task_queue.enqueue.call_args[0][0]
        assert enqueued_task.tool_call_id == "call-007"


# ---------------------------------------------------------------------------
# TestLLMClientLifecycle
# ---------------------------------------------------------------------------


class TestLLMClientLifecycle:
    async def test_work_calls_client_start_before_run_agent_loop(self):
        """work() calls client.start() before run_agent_loop."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-lc-01",
            task_queue=task_queue,
        )

        call_order = []

        mock_client = MagicMock()

        async def mock_start():
            call_order.append("start")

        async def mock_stop():
            call_order.append("stop")

        mock_client.start = mock_start
        mock_client.stop = mock_stop

        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            call_order.append("run_agent_loop")
            return "done"

        with patch("sherman.tools.subagent.LLMClient", return_value=mock_client), \
             patch("sherman.tools.subagent.run_agent_loop", side_effect=fake_run_agent_loop), \
             patch("sherman.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("instructions", "desc", ctx)

            # Extract the work() coroutine from the enqueued Task and run it
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        assert call_order.index("start") < call_order.index("run_agent_loop"), \
            "client.start() must be called before run_agent_loop"

    async def test_work_calls_client_stop_after_run_agent_loop(self):
        """work() calls client.stop() after run_agent_loop, even on success."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-lc-02",
            task_queue=task_queue,
        )

        call_order = []

        mock_client = MagicMock()

        async def mock_start():
            call_order.append("start")

        async def mock_stop():
            call_order.append("stop")

        mock_client.start = mock_start
        mock_client.stop = mock_stop

        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            call_order.append("run_agent_loop")
            return "done"

        with patch("sherman.tools.subagent.LLMClient", return_value=mock_client), \
             patch("sherman.tools.subagent.run_agent_loop", side_effect=fake_run_agent_loop), \
             patch("sherman.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        assert call_order.index("stop") > call_order.index("run_agent_loop"), \
            "client.stop() must be called after run_agent_loop"

    async def test_work_calls_client_stop_even_when_run_agent_loop_raises(self):
        """work() calls client.stop() in the finally block even when run_agent_loop raises."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-lc-03",
            task_queue=task_queue,
        )

        stop_called = []

        mock_client = MagicMock()

        async def mock_start():
            pass

        async def mock_stop():
            stop_called.append(True)

        mock_client.start = mock_start
        mock_client.stop = mock_stop

        async def raising_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            raise RuntimeError("LLM connection failed")

        # strip_thinking is not patched here because it is never reached
        # when run_agent_loop raises — the exception exits work() before
        # strip_thinking is called.
        with patch("sherman.tools.subagent.LLMClient", return_value=mock_client), \
             patch("sherman.tools.subagent.run_agent_loop", side_effect=raising_run_agent_loop):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]

            with pytest.raises(RuntimeError, match="LLM connection failed"):
                await enqueued_task.work()

        assert stop_called, "client.stop() must be called even when run_agent_loop raises"


# ---------------------------------------------------------------------------
# TestToolExclusionVerification
# ---------------------------------------------------------------------------


class TestToolExclusionVerification:
    async def test_work_excludes_subagent_from_tool_registry(self):
        """The tool registry passed to run_agent_loop does not contain 'subagent'."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-excl-01",
            task_queue=task_queue,
        )

        captured_tools = {}

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured_tools.update(tools)
            return "done"

        with patch("sherman.tools.subagent.LLMClient", return_value=mock_client), \
             patch("sherman.tools.subagent.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("sherman.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        assert "subagent" not in captured_tools, \
            "subagent must not appear in the tool registry passed to run_agent_loop"

    async def test_work_passes_channel_and_task_queue_to_run_agent_loop(self):
        """work() passes channel and task_queue kwargs to run_agent_loop."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-excl-kwargs",
            task_queue=task_queue,
        )

        captured_kwargs = {}

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured_kwargs.update(kwargs)
            return "done"

        with patch("sherman.tools.subagent.LLMClient", return_value=mock_client), \
             patch("sherman.tools.subagent.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("sherman.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        assert captured_kwargs.get("channel") is channel, \
            "work() must pass channel to run_agent_loop"
        assert captured_kwargs.get("task_queue") is task_queue, \
            "work() must pass task_queue to run_agent_loop"

    async def test_work_includes_other_tools_in_registry(self):
        """Tools other than subagent are passed to run_agent_loop."""
        # Register shell in addition to the excluded tools
        pm, registry = _make_pm_with_tool_registry("shell", "subagent")
        plugin = SubagentPlugin(pm)
        await plugin.on_start(config=BASE_CONFIG)

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-excl-03",
            task_queue=task_queue,
        )

        captured_tools = {}

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured_tools.update(tools)
            return "done"

        with patch("sherman.tools.subagent.LLMClient", return_value=mock_client), \
             patch("sherman.tools.subagent.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("sherman.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        assert "shell" in captured_tools, \
            "Non-excluded tools (e.g. 'shell') must appear in the tool registry"


# ---------------------------------------------------------------------------
# TestErrorPaths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    async def test_work_propagates_run_agent_loop_exception(self):
        """When run_agent_loop raises, the exception propagates out of work()."""
        plugin = await _make_started_plugin()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-err-01",
            task_queue=task_queue,
        )

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        async def raising_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            raise ConnectionError("cannot reach LLM")

        with patch("sherman.tools.subagent.LLMClient", return_value=mock_client), \
             patch("sherman.tools.subagent.run_agent_loop", side_effect=raising_run_agent_loop):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]

            # The exception must propagate so TaskQueue.run_worker can catch it
            with pytest.raises(ConnectionError, match="cannot reach LLM"):
                await enqueued_task.work()

    async def test_task_queue_none_does_not_enqueue(self):
        """When task_queue is None, no task is enqueued (error returned directly)."""
        plugin = await _make_started_plugin()

        ctx = ToolContext(
            channel=_make_channel(),
            tool_call_id="call-err-02",
            task_queue=None,
        )
        result = await plugin._launch("instructions", "desc", ctx)

        # Returns error string — does not raise, does not enqueue
        assert "Error" in result

    async def test_channel_none_does_not_enqueue(self):
        """When channel is None, no task is enqueued (error returned directly)."""
        plugin = await _make_started_plugin()

        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=None,
            tool_call_id="call-err-03",
            task_queue=task_queue,
        )
        result = await plugin._launch("instructions", "desc", ctx)

        assert "Error" in result
        task_queue.enqueue.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestOnStart
# ---------------------------------------------------------------------------


class TestOnStart:
    async def test_on_start_captures_main_llm_config(self):
        """on_start captures the main LLM config when no background config exists."""
        pm, _ = _make_pm_with_tool_registry()
        plugin = SubagentPlugin(pm)
        await plugin.on_start(config=BASE_CONFIG)

        assert plugin._llm_config is not None
        assert plugin._llm_config["model"] == "test-model"

    async def test_on_start_prefers_background_llm_config(self):
        """on_start uses llm.background config when present."""
        pm, _ = _make_pm_with_tool_registry()
        plugin = SubagentPlugin(pm)
        await plugin.on_start(config=BACKGROUND_CONFIG)

        assert plugin._llm_config is not None
        assert plugin._llm_config["model"] == "bg-model"
        assert plugin._llm_config["base_url"] == "http://localhost:8081"
