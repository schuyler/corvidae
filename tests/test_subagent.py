"""Tests for corvidae.tools.subagent — SubagentPlugin and run_agent_loop."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.task import Task, TaskQueue
from corvidae.tool import Tool, ToolContext, ToolRegistry

from corvidae.llm_plugin import LLMPlugin
from corvidae.tools.subagent import SubagentPlugin, run_agent_loop


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
    directly rather than from Agent.
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

        The definitive check: if Agent._max_tool_result_chars differs
        from what config would produce, and _launch passes the Agent
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


# ===========================================================================
# run_agent_loop tests (moved from tests/test_agent_loop.py)
# ===========================================================================


def _make_text_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _make_tool_call_response(calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": calls,
                }
            }
        ]
    }


def _make_tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def _make_tool_call_malformed_args(call_id: str, name: str, raw_args: str) -> dict:
    """Build a tool call dict with raw (potentially invalid) JSON arguments."""
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": raw_args,
        },
    }


# ---------------------------------------------------------------------------
# Basic run_agent_loop behavior
# ---------------------------------------------------------------------------


async def test_simple_response_no_tools():
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("Hello, world!"))

    messages = [{"role": "user", "content": "Hi"}]
    result = await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    assert result == "Hello, world!"
    assert messages[-1] == {"role": "assistant", "content": "Hello, world!"}


async def test_single_tool_call():
    tool_fn = AsyncMock(return_value="tool result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_1", "my_tool", {"x": "value"})]
            ),
            _make_text_response("final answer"),
        ]
    )

    messages = [{"role": "user", "content": "do thing"}]
    result = await run_agent_loop(
        client,
        messages,
        tools={"my_tool": tool_fn},
        tool_schemas=[],
    )

    # Tool should be called with the correct args
    tool_fn.assert_awaited_once_with(x="value")

    assert result == "final answer"

    # messages should contain: user, assistant w/ tool_call, tool result, assistant final
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]

    # assistant message with tool call
    assert messages[1].get("tool_calls") is not None

    # tool result message
    assert messages[2]["role"] == "tool"
    assert messages[2]["content"] == "tool result"

    # final assistant message
    assert messages[3] == {"role": "assistant", "content": "final answer"}


async def test_unknown_tool_returns_error():
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_x", "nonexistent_tool", {})]
            ),
            _make_text_response("ok"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    await run_agent_loop(
        client,
        messages,
        tools={},
        tool_schemas=[],
    )

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert "Error: unknown tool" in tool_result["content"]


async def test_tool_exception_returns_error():
    async def bad_tool(**kwargs):
        raise ValueError("something went wrong")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_2", "bad_tool", {})]
            ),
            _make_text_response("recovered"),
        ]
    )

    messages = [{"role": "user", "content": "break it"}]
    await run_agent_loop(
        client,
        messages,
        tools={"bad_tool": bad_tool},
        tool_schemas=[],
    )

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert tool_result["content"].startswith("Error:")


async def test_max_turns_exceeded():
    client = MagicMock()
    # Always returns a tool call, never terminates naturally
    client.chat = AsyncMock(
        return_value=_make_tool_call_response(
            [_make_tool_call("call_loop", "noop", {})]
        )
    )

    noop = AsyncMock(return_value="noop result")
    messages = [{"role": "user", "content": "spin"}]
    result = await run_agent_loop(
        client,
        messages,
        tools={"noop": noop},
        tool_schemas=[],
        max_turns=3,
    )

    assert result == "(max tool-calling rounds reached)"


async def test_multiple_tool_calls_in_one_turn():
    tool_a = AsyncMock(return_value="result_a")
    tool_b = AsyncMock(return_value="result_b")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [
                    _make_tool_call("call_a", "tool_a", {"n": 1}),
                    _make_tool_call("call_b", "tool_b", {"n": 2}),
                ]
            ),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "run both"}]
    result = await run_agent_loop(
        client,
        messages,
        tools={"tool_a": tool_a, "tool_b": tool_b},
        tool_schemas=[],
    )

    tool_a.assert_awaited_once_with(n=1)
    tool_b.assert_awaited_once_with(n=2)
    assert result == "done"

    # Both tool result messages should be present
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 2


async def test_run_agent_loop_does_not_pass_extra_body():
    """run_agent_loop no longer accepts extra_body — LLMClient handles it internally.
    Verify chat() is called without extra_body kwarg."""
    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value=_make_text_response("test response"))

    messages = [{"role": "user", "content": "hello"}]
    tools = {}
    tool_schemas = []

    result = await run_agent_loop(
        client=mock_client,
        messages=messages,
        tools=tools,
        tool_schemas=tool_schemas,
        max_turns=1,
    )

    # run_agent_loop passes no extra_body to client.chat() — LLMClient handles it
    mock_client.chat.assert_called_once()
    call_args = mock_client.chat.call_args
    assert "extra_body" not in call_args.kwargs


# ---------------------------------------------------------------------------
# Logging tests — INFO records with latency_ms
# ---------------------------------------------------------------------------


async def test_llm_response_logs_info_with_latency_ms(caplog):
    """After a successful LLM call, an INFO record with message 'LLM response
    received' and a numeric latency_ms attribute must be emitted by run_agent_turn."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("hello"))

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.INFO, logger="corvidae.agent_turn"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.agent_turn"]
    matching = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "LLM response received"]
    assert matching, "Expected INFO record with message 'LLM response received'"
    assert hasattr(matching[0], "latency_ms"), "'LLM response received' log must have latency_ms attribute"
    assert isinstance(matching[0].latency_ms, (int, float)), "latency_ms must be numeric"


async def test_tool_call_result_logs_info_with_latency_ms(caplog):
    """After a successful tool execution, an INFO record with message 'tool call
    result' and a numeric latency_ms attribute must be emitted."""
    tool_fn = AsyncMock(return_value="tool result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "my_tool", {"x": "v"})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.INFO, logger="corvidae.tool"):
        await run_agent_loop(client, messages, tools={"my_tool": tool_fn}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "tool call result"]
    assert matching, "Expected INFO record with message 'tool call result'"
    assert hasattr(matching[0], "latency_ms"), "'tool call result' log must have latency_ms attribute"
    assert isinstance(matching[0].latency_ms, (int, float)), "latency_ms must be numeric"


async def test_tool_call_dispatched_logs_info(caplog):
    """After dispatching a tool call, an INFO record with message 'tool call
    dispatched' must be emitted (latency_ms not required)."""
    tool_fn = AsyncMock(return_value="result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "my_tool", {})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.INFO, logger="corvidae.tool"):
        await run_agent_loop(client, messages, tools={"my_tool": tool_fn}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "tool call dispatched"]
    assert matching, "Expected INFO record with message 'tool call dispatched'"


async def test_tool_call_result_not_logged_for_unknown_tool(caplog):
    """When an unknown tool is called, 'tool call result' INFO must NOT be emitted."""
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "ghost_tool", {})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.INFO, logger="corvidae.tool"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    result_records = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "tool call result"]
    assert not result_records, "'tool call result' INFO must NOT be emitted for unknown tool"


async def test_tool_call_result_not_logged_on_exception(caplog):
    """When a tool raises an exception, 'tool call result' INFO must NOT be emitted."""
    async def bad_tool(**kwargs):
        raise ValueError("boom")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "bad_tool", {})]),
            _make_text_response("recovered"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.INFO, logger="corvidae.tool"):
        await run_agent_loop(client, messages, tools={"bad_tool": bad_tool}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    result_records = [r for r in records if r.levelno == logging.INFO and r.getMessage() == "tool call result"]
    assert not result_records, "'tool call result' INFO must NOT be emitted when tool raises"


# ---------------------------------------------------------------------------
# Logging tests — DEBUG records for content visibility
# ---------------------------------------------------------------------------


async def test_llm_response_content_debug_log(caplog):
    """After a successful LLM call, a DEBUG record with message 'LLM response
    content' must be emitted by run_agent_turn with has_reasoning_content and
    reasoning_content_length attributes."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("hello"))

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.DEBUG, logger="corvidae.agent_turn"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.agent_turn"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "LLM response content"
    ]
    assert matching, "Expected DEBUG record with message 'LLM response content'"
    rec = matching[0]
    assert hasattr(rec, "has_reasoning_content"), (
        "'LLM response content' DEBUG log must have has_reasoning_content attribute"
    )
    assert isinstance(rec.has_reasoning_content, bool), (
        "has_reasoning_content must be a bool"
    )
    assert hasattr(rec, "reasoning_content_length"), (
        "'LLM response content' DEBUG log must have reasoning_content_length attribute"
    )
    assert hasattr(rec, "content"), "'LLM response content' DEBUG log must have content attribute"


async def test_llm_response_content_debug_log_with_reasoning(caplog):
    """When response has reasoning_content, DEBUG log must have
    has_reasoning_content=True and reasoning_content_length > 0."""
    reasoning_text = "<reasoning>think step by step</reasoning>"
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "final answer",
                    "reasoning_content": reasoning_text,
                }
            }
        ]
    }
    client = MagicMock()
    client.chat = AsyncMock(return_value=response)

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.DEBUG, logger="corvidae.agent_turn"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.agent_turn"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "LLM response content"
    ]
    assert matching, "Expected DEBUG record with message 'LLM response content'"
    rec = matching[0]
    assert rec.has_reasoning_content is True, (
        "has_reasoning_content must be True when reasoning_content is present"
    )
    assert isinstance(rec.reasoning_content_length, int), (
        "reasoning_content_length must be an int when reasoning_content is present"
    )
    assert rec.reasoning_content_length > 0, (
        "reasoning_content_length must be > 0 for non-empty reasoning_content"
    )
    assert hasattr(rec, "content"), "'LLM response content' DEBUG log must have content attribute"


async def test_llm_response_content_truncated(caplog):
    """Content longer than 200 chars must be truncated in the DEBUG log."""
    long_content = "x" * 300
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response(long_content))

    messages = [{"role": "user", "content": "hi"}]
    with caplog.at_level(logging.DEBUG, logger="corvidae.agent_turn"):
        await run_agent_loop(client, messages, tools={}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.agent_turn"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "LLM response content"
    ]
    assert matching, "Expected DEBUG record with message 'LLM response content'"
    rec = matching[0]
    assert hasattr(rec, "content"), (
        "'LLM response content' DEBUG log must have content attribute"
    )
    assert len(rec.content) <= 203, (
        "Truncated content must be at most 203 chars (200 + '...')"
    )
    assert rec.content.endswith("..."), (
        "Truncated content must end with '...'"
    )


async def test_tool_call_arguments_debug_log(caplog):
    """After dispatching a tool call, a DEBUG record with message 'tool call
    arguments' must be emitted with tool and arguments attributes."""
    tool_fn = AsyncMock(return_value="result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "my_tool", {"x": "v"})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.DEBUG, logger="corvidae.tool"):
        await run_agent_loop(client, messages, tools={"my_tool": tool_fn}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "tool call arguments"
    ]
    assert matching, "Expected DEBUG record with message 'tool call arguments'"
    rec = matching[0]
    assert hasattr(rec, "tool"), (
        "'tool call arguments' DEBUG log must have tool attribute"
    )
    assert rec.tool == "my_tool", "tool attribute must match the called tool name"
    assert hasattr(rec, "arguments"), (
        "'tool call arguments' DEBUG log must have arguments attribute"
    )
    assert isinstance(rec.arguments, str), "arguments must be a string (JSON)"


async def test_tool_call_result_content_debug_log(caplog):
    """After a successful tool execution, a DEBUG record with message 'tool call
    result content' must be emitted with tool and content attributes."""
    tool_fn = AsyncMock(return_value="tool result")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "my_tool", {"x": "v"})]),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.DEBUG, logger="corvidae.tool"):
        await run_agent_loop(client, messages, tools={"my_tool": tool_fn}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    matching = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "tool call result content"
    ]
    assert matching, "Expected DEBUG record with message 'tool call result content'"
    rec = matching[0]
    assert hasattr(rec, "tool"), (
        "'tool call result content' DEBUG log must have tool attribute"
    )
    assert rec.tool == "my_tool", "tool attribute must match the called tool name"
    assert hasattr(rec, "content"), (
        "'tool call result content' DEBUG log must have content attribute"
    )


async def test_tool_call_result_content_not_logged_on_exception(caplog):
    """When a tool raises an exception, 'tool call result content' DEBUG must NOT
    be emitted."""
    async def bad_tool(**kwargs):
        raise ValueError("boom")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response([_make_tool_call("c1", "bad_tool", {})]),
            _make_text_response("recovered"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.DEBUG, logger="corvidae.tool"):
        await run_agent_loop(client, messages, tools={"bad_tool": bad_tool}, tool_schemas=[])

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    result_content_records = [
        r for r in records
        if r.levelno == logging.DEBUG and r.getMessage() == "tool call result content"
    ]
    assert not result_content_records, (
        "'tool call result content' DEBUG must NOT be emitted when tool raises"
    )


# ---------------------------------------------------------------------------
# Tool exception error message includes fn_name
# ---------------------------------------------------------------------------


async def test_tool_exception_error_message_includes_tool_name():
    """When a tool raises an exception in run_agent_loop, the tool result message
    must include the tool name."""
    async def bad_tool(**kwargs):
        raise RuntimeError("deliberate failure")

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call("call_err", "bad_tool", {})]
            ),
            _make_text_response("recovered"),
        ]
    )

    messages = [{"role": "user", "content": "trigger error"}]
    await run_agent_loop(
        client,
        messages,
        tools={"bad_tool": bad_tool},
        tool_schemas=[],
    )

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert "bad_tool" in tool_result["content"], (
        f"Error message must include the tool name 'bad_tool', got: {tool_result['content']!r}"
    )


# ---------------------------------------------------------------------------
# Malformed JSON tool call arguments
# ---------------------------------------------------------------------------


async def test_malformed_json_tool_args_returns_error():
    """When the LLM returns a tool call with malformed JSON arguments:
    - The tool function must NOT be called.
    - An error tool result message must be appended with content containing
      'Error: malformed arguments'.
    - The loop must continue and return the final text response.
    """
    tool_fn = AsyncMock(return_value="should not be called")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call_malformed_args("call_bad", "my_tool", "{not valid json")]
            ),
            _make_text_response("recovered after error"),
        ]
    )

    messages = [{"role": "user", "content": "do thing"}]
    result = await run_agent_loop(
        client,
        messages,
        tools={"my_tool": tool_fn},
        tool_schemas=[],
    )

    # Tool must not be called when arguments are malformed
    tool_fn.assert_not_awaited()

    # An error tool result must be appended
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1, f"Expected 1 tool result message, got {len(tool_messages)}"
    assert "Error: malformed arguments" in tool_messages[0]["content"], (
        f"Tool result must contain 'Error: malformed arguments', got: {tool_messages[0]['content']!r}"
    )

    # Loop must recover and return the final text response
    assert result == "recovered after error", (
        f"Expected 'recovered after error', got: {result!r}"
    )


async def test_malformed_json_does_not_skip_subsequent_calls():
    """When two tool calls are in one turn and the first has malformed JSON:
    - The first tool must NOT be called.
    - The second tool MUST be called.
    - Both tool result messages must be appended.
    """
    tool_a = AsyncMock(return_value="should not be called")
    tool_b = AsyncMock(return_value="result_b")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [
                    _make_tool_call_malformed_args("call_bad", "tool_a", "{not valid json"),
                    _make_tool_call("call_good", "tool_b", {"n": 2}),
                ]
            ),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "run both"}]
    result = await run_agent_loop(
        client,
        messages,
        tools={"tool_a": tool_a, "tool_b": tool_b},
        tool_schemas=[],
    )

    # First tool must not be called
    tool_a.assert_not_awaited()

    # Second tool must be called
    tool_b.assert_awaited_once_with(n=2)

    # Both tool result messages must be present
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 2, (
        f"Expected 2 tool result messages, got {len(tool_messages)}"
    )

    # First result is the error for the malformed call
    assert "Error: malformed arguments" in tool_messages[0]["content"], (
        f"First tool result must contain 'Error: malformed arguments', got: {tool_messages[0]['content']!r}"
    )

    # Second result is the successful tool output
    assert tool_messages[1]["content"] == "result_b", (
        f"Second tool result must be 'result_b', got: {tool_messages[1]['content']!r}"
    )

    assert result == "done"


async def test_malformed_json_logs_warning(caplog):
    """When malformed JSON tool call arguments are encountered, a WARNING log
    record with message 'malformed tool call arguments' must be emitted."""
    tool_fn = AsyncMock(return_value="irrelevant")
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[
            _make_tool_call_response(
                [_make_tool_call_malformed_args("call_bad", "my_tool", "{not valid json")]
            ),
            _make_text_response("done"),
        ]
    )

    messages = [{"role": "user", "content": "go"}]
    with caplog.at_level(logging.WARNING, logger="corvidae.tool"):
        await run_agent_loop(
            client,
            messages,
            tools={"my_tool": tool_fn},
            tool_schemas=[],
        )

    records = [r for r in caplog.records if r.name == "corvidae.tool"]
    warning_records = [
        r for r in records
        if r.levelno == logging.WARNING and r.getMessage() == "malformed tool call arguments"
    ]
    assert warning_records, (
        "Expected WARNING record with message 'malformed tool call arguments'"
    )
