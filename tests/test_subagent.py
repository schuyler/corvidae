"""Tests for corvidae.tools.subagent — SubagentPlugin.

These are Red TDD tests. They will fail with ImportError until
corvidae/tools/subagent.py is implemented.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.task import Task, TaskQueue
from corvidae.tool import Tool, ToolContext, ToolRegistry

from corvidae.llm_plugin import LLMPlugin
from corvidae.tools.subagent import SubagentPlugin  # ImportError until implemented


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
    """Build a minimal plugin manager with a ToolCollectionPlugin that has a
    ToolRegistry containing named placeholder tools."""
    from corvidae.tool_collection import ToolCollectionPlugin

    pm = create_plugin_manager()

    # Register a minimal ChannelRegistry.
    pm.register(ChannelRegistry({}), name="registry")

    # Register a mock LLMPlugin so SubagentPlugin._launch can get a client.
    mock_llm = LLMPlugin(pm)
    mock_llm.main_client = MagicMock()
    mock_llm.background_client = None
    pm.register(mock_llm, name="llm")

    tool_registry = ToolRegistry()
    for name in tool_names:
        async def _placeholder() -> str:
            """Placeholder tool."""
            return "ok"
        _placeholder.__name__ = name
        tool_registry.add(Tool(name=name, fn=_placeholder, schema={}))

    # Use a ToolCollectionPlugin with registry set directly to avoid
    # calling on_start (which would rebuild the registry from hooks).
    tools_plugin = ToolCollectionPlugin(pm)
    tools_plugin.registry = tool_registry
    pm.register(tools_plugin, name="tools")

    return pm, tool_registry


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
    """After Part 3, SubagentPlugin uses the shared background client from LLMPlugin.
    No ephemeral client is created — start/stop are not called on the shared client.
    """

    async def test_work_uses_shared_client_from_llm_plugin(self):
        """work() uses the shared background client from LLMPlugin, not an ephemeral one."""
        plugin = await _make_started_plugin()
        llm_plugin = plugin.pm.get_plugin("llm")
        shared_client = llm_plugin.get_client("background")

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-lc-01",
            task_queue=task_queue,
        )

        captured_client = {}

        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured_client["client"] = client
            return "done"

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=fake_run_agent_loop), \
             patch("corvidae.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        assert captured_client.get("client") is shared_client, \
            "work() must pass the shared LLMPlugin background client to run_agent_loop"

    async def test_work_does_not_call_start_or_stop_on_shared_client(self):
        """work() must NOT call start() or stop() on the shared background client."""
        plugin = await _make_started_plugin()
        llm_plugin = plugin.pm.get_plugin("llm")
        shared_client = llm_plugin.get_client("background")
        shared_client.start = AsyncMock()
        shared_client.stop = AsyncMock()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-lc-02",
            task_queue=task_queue,
        )

        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            return "done"

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=fake_run_agent_loop), \
             patch("corvidae.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        shared_client.start.assert_not_awaited()
        shared_client.stop.assert_not_awaited()

    async def test_work_propagates_run_agent_loop_exception_without_stopping_client(self):
        """work() propagates exceptions from run_agent_loop without stopping shared client."""
        plugin = await _make_started_plugin()
        llm_plugin = plugin.pm.get_plugin("llm")
        shared_client = llm_plugin.get_client("background")
        shared_client.stop = AsyncMock()

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-lc-03",
            task_queue=task_queue,
        )

        async def raising_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            raise RuntimeError("LLM connection failed")

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=raising_run_agent_loop):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]

            with pytest.raises(RuntimeError, match="LLM connection failed"):
                await enqueued_task.work()

        # Shared client must NOT be stopped — it's owned by LLMPlugin
        shared_client.stop.assert_not_awaited()


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

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured_tools.update(tools)
            return "done"

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("corvidae.tools.subagent.strip_thinking", side_effect=lambda x: x):
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

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured_kwargs.update(kwargs)
            return "done"

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("corvidae.tools.subagent.strip_thinking", side_effect=lambda x: x):
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

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured_tools.update(tools)
            return "done"

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("corvidae.tools.subagent.strip_thinking", side_effect=lambda x: x):
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

        async def raising_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            raise ConnectionError("cannot reach LLM")

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=raising_run_agent_loop):
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
    async def test_on_start_completes_without_error(self):
        """on_start must complete without raising, even without LLM config.

        After Part 3, SubagentPlugin.on_start does not parse LLM config
        directly — that is now LLMPlugin's responsibility.
        """
        pm, _ = _make_pm_with_tool_registry()
        plugin = SubagentPlugin(pm)
        # on_start should not raise
        await plugin.on_start(config=BASE_CONFIG)

    async def test_on_start_does_not_set_llm_config(self):
        """on_start must NOT set _llm_config — that attribute no longer exists.

        After Part 3, LLM client config is owned by LLMPlugin. SubagentPlugin
        reads the client from LLMPlugin at _launch time.
        """
        pm, _ = _make_pm_with_tool_registry()
        plugin = SubagentPlugin(pm)
        await plugin.on_start(config=BASE_CONFIG)

        assert not hasattr(plugin, "_llm_config"), (
            "SubagentPlugin must not have _llm_config after Part 3 "
            "(LLM config is owned by LLMPlugin)"
        )


# ---------------------------------------------------------------------------
# TestMaxToolResultCharsConsolidation
# ---------------------------------------------------------------------------


class TestMaxToolResultCharsConsolidation:
    """Verify that _launch reads max_result_chars from ToolCollectionPlugin, not from
    SubagentPlugin's own attribute or from config.

    After Part 4, SubagentPlugin reads max_result_chars from ToolCollectionPlugin
    directly rather than from AgentPlugin.
    """

    async def _launch_and_capture_max_result_chars(
        self,
        tools_max_result_chars: int,
    ) -> int:
        """Helper: configure ToolCollectionPlugin with a custom max_result_chars,
        run _launch, and return the max_result_chars value seen by run_agent_loop.
        """
        pm, _ = _make_pm_with_tool_registry("shell", "subagent")
        plugin = SubagentPlugin(pm)
        await plugin.on_start(config=BASE_CONFIG)

        # Override ToolCollectionPlugin's authoritative value directly.
        from corvidae.tool_collection import ToolCollectionPlugin
        tools_plugin = pm.get_plugin("tools")
        tools_plugin.max_result_chars = tools_max_result_chars

        channel = _make_channel()
        task_queue = MagicMock(spec=TaskQueue)
        task_queue.enqueue = AsyncMock()

        ctx = ToolContext(
            channel=channel,
            tool_call_id="call-consolidation",
            task_queue=task_queue,
        )

        captured = {}

        async def capture_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            captured["max_result_chars"] = kwargs.get("max_result_chars")
            return "done"

        with patch("corvidae.tools.subagent.run_agent_loop", side_effect=capture_run_agent_loop), \
             patch("corvidae.tools.subagent.strip_thinking", side_effect=lambda x: x):
            await plugin._launch("instructions", "desc", ctx)
            enqueued_task = task_queue.enqueue.call_args[0][0]
            await enqueued_task.work()

        return captured["max_result_chars"]

    async def test_launch_passes_tools_plugin_max_result_chars_to_run_agent_loop(self):
        """_launch must pass ToolCollectionPlugin.max_result_chars to run_agent_loop.

        After Part 4, SubagentPlugin reads max_result_chars from ToolCollectionPlugin
        rather than from its own attribute or from config.
        """
        custom_value = 42_000
        observed = await self._launch_and_capture_max_result_chars(custom_value)

        assert observed == custom_value, (
            f"run_agent_loop received max_result_chars={observed}, "
            f"expected ToolCollectionPlugin's value {custom_value}. "
            "SubagentPlugin._launch must read from ToolCollectionPlugin.max_result_chars."
        )

    async def test_launch_uses_tools_plugin_value_not_independent_config_read(self):
        """ToolCollectionPlugin's value must win over any independent config read.

        Sets ToolCollectionPlugin.max_result_chars to a value that differs from
        the config default (100_000).
        If SubagentPlugin reads config independently it will see 100_000;
        if it reads ToolCollectionPlugin it will see the custom value.
        """
        # A value that is neither 100_000 (config default) nor the __init__
        # sentinel — unambiguously identifies ToolCollectionPlugin as the source.
        authoritative_value = 75_555
        observed = await self._launch_and_capture_max_result_chars(authoritative_value)

        assert observed == authoritative_value, (
            f"run_agent_loop received max_result_chars={observed}, "
            f"expected {authoritative_value} from ToolCollectionPlugin. "
            "SubagentPlugin._launch must not read config independently."
        )

    async def test_subagent_does_not_have_independent_max_tool_result_chars_after_on_start(self):
        """After on_start, SubagentPlugin must not carry _max_tool_result_chars
        as an independently-read attribute.

        Before the fix, on_start sets self._max_tool_result_chars from config.
        After the fix, on_start does NOT set it — the attribute either does not
        exist on the instance or retains only the __init__ default (100_000)
        without any config-sourced update.

        The definitive check: if AgentPlugin._max_tool_result_chars differs
        from what config would produce, and _launch passes the AgentPlugin
        value, then SubagentPlugin is not doing an independent read. This
        structural test verifies the attribute is not set by on_start by
        checking the attribute was not updated from config.
        """
        pm, _ = _make_pm_with_tool_registry("shell", "subagent")
        plugin = SubagentPlugin(pm)

        # Config carries a custom value; if on_start reads this, the attribute
        # on SubagentPlugin will be 55_555 after on_start.
        config_with_custom = {
            **BASE_CONFIG,
            "agent": {"max_tool_result_chars": 55_555},
        }
        await plugin.on_start(config=config_with_custom)

        # After the fix, on_start must NOT have written 55_555 onto the
        # SubagentPlugin instance. The attribute should not hold a config-read
        # value (it should be absent or remain at the __init__ default).
        # Before the fix this is 55_555, causing the assertion to fail.
        assert getattr(plugin, "_max_tool_result_chars", None) != 55_555, (
            "on_start must not independently read max_tool_result_chars from config; "
            "found config-sourced value 55_555 on SubagentPlugin._max_tool_result_chars"
        )
