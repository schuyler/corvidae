"""Tests for batched tool result handling (Bug 1 fix).

When the LLM requests multiple tool calls, each result arrives separately
via on_notify → serial queue → _process_queue_item. The fix ensures that:

1. Only the LAST tool result triggers an LLM call
2. Intermediate results are buffered in the conversation without an LLM call
3. The final LLM call sees ALL tool results in the prompt
4. No duplicate LLM calls or duplicate assistant responses occur
"""

import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call_response,
    _make_tool_call,
)


class TestBatchedToolResults:
    """Tests that multiple parallel tool calls produce exactly one LLM call
    after all results are collected, not one per result."""

    async def test_multiple_tools_exactly_one_final_llm_call(self):
        """When the LLM requests 2 tools, the LLM is called exactly twice:
        once to dispatch tools, and once after both results arrive."""
        plugin, channel, db = await build_plugin_and_channel()

        tool_results = {}

        async def tool_a(query: str) -> str:
            tool_results["a"] = query
            return "result_a"

        async def tool_b(query: str) -> str:
            tool_results["b"] = query
            return "result_b"

        from corvidae.tool import tool_to_schema
        plugin._tools = {"tool_a": tool_a, "tool_b": tool_b}
        plugin._tool_schemas = [tool_to_schema(tool_a), tool_to_schema(tool_b)]

        llm_call_count = 0

        async def counting_chat(messages, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count == 1:
                return _make_tool_call_response([
                    _make_tool_call("call_a", "tool_a", {"query": "hello_a"}),
                    _make_tool_call("call_b", "tool_b", {"query": "hello_b"}),
                ])
            else:
                return _make_text_response("done with both tools")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=counting_chat)
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="use tools")
        await drain(plugin, channel)
        assert llm_call_count == 1

        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        assert llm_call_count == 2, (
            f"Expected exactly 2 LLM calls, got {llm_call_count}. "
            "Before the fix, this would be 3+ (one per tool result)."
        )
        plugin.pm.ahook.send_message.assert_awaited_once_with(
            channel=channel, text="done with both tools", latency_ms=ANY,
        )
        assert tool_results == {"a": "hello_a", "b": "hello_b"}

        await task_plugin.on_stop()
        await db.close()

    async def test_three_tools_exactly_one_final_llm_call(self):
        """With 3 tool calls, total LLM calls = 2 (dispatch + one final)."""
        plugin, channel, db = await build_plugin_and_channel()

        async def tool_x(x: str) -> str:
            return f"x:{x}"

        async def tool_y(y: str) -> str:
            return f"y:{y}"

        async def tool_z(z: str) -> str:
            return f"z:{z}"

        from corvidae.tool import tool_to_schema
        plugin._tools = {"tool_x": tool_x, "tool_y": tool_y, "tool_z": tool_z}
        plugin._tool_schemas = [
            tool_to_schema(tool_x), tool_to_schema(tool_y), tool_to_schema(tool_z),
        ]

        llm_call_count = 0

        async def counting_chat(messages, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count == 1:
                return _make_tool_call_response([
                    _make_tool_call("call_x", "tool_x", {"x": "1"}),
                    _make_tool_call("call_y", "tool_y", {"y": "2"}),
                    _make_tool_call("call_z", "tool_z", {"z": "3"}),
                ])
            else:
                return _make_text_response("all three done")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=counting_chat)
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="go")
        await drain(plugin, channel)
        assert llm_call_count == 1

        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        assert llm_call_count == 2, f"Expected 2 LLM calls, got {llm_call_count}"
        plugin.pm.ahook.send_message.assert_awaited_once_with(
            channel=channel, text="all three done", latency_ms=ANY,
        )

        await task_plugin.on_stop()
        await db.close()

    async def test_user_message_does_not_clear_pending_ids(self):
        """User messages must NOT clear pending_tool_call_ids — the agent
        should remain responsive while tools run in the background."""
        plugin, channel, db = await build_plugin_and_channel()
        channel.pending_tool_call_ids = {"call_a", "call_b"}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("got it"))
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="quick question")
        await drain(plugin, channel)

        assert channel.pending_tool_call_ids == {"call_a", "call_b"}, (
            "pending_tool_call_ids must survive user messages so background "
            "tool results can still be batched and reported"
        )
        await db.close()

    async def test_single_tool_still_works(self):
        """Regression: single tool call round-trip still works."""
        plugin, channel, db = await build_plugin_and_channel()

        async def my_tool(query: str) -> str:
            return f"result: {query}"

        from corvidae.tool import tool_to_schema
        plugin._tools = {"my_tool": my_tool}
        plugin._tool_schemas = [tool_to_schema(my_tool)]

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response([
                    _make_tool_call("call_1", "my_tool", {"query": "test"})
                ]),
                _make_text_response("final answer"),
            ]
        )
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="go")
        await drain(plugin, channel)

        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        plugin.pm.ahook.send_message.assert_awaited_once_with(
            channel=channel, text="final answer", latency_ms=ANY,
        )
        await task_plugin.on_stop()
        await db.close()

    async def test_conversation_contains_all_tool_results_in_final_prompt(self):
        """The final LLM call's prompt must contain ALL tool results."""
        plugin, channel, db = await build_plugin_and_channel()

        async def tool_a(x: str) -> str:
            return "alpha"

        async def tool_b(x: str) -> str:
            return "beta"

        from corvidae.tool import tool_to_schema
        plugin._tools = {"tool_a": tool_a, "tool_b": tool_b}
        plugin._tool_schemas = [tool_to_schema(tool_a), tool_to_schema(tool_b)]

        captured_prompts = []

        async def capture_chat(messages, **kwargs):
            captured_prompts.append(messages)
            if len(captured_prompts) == 1:
                return _make_tool_call_response([
                    _make_tool_call("call_a", "tool_a", {"x": "1"}),
                    _make_tool_call("call_b", "tool_b", {"x": "2"}),
                ])
            else:
                return _make_text_response("final")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=capture_chat)
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="go")
        await drain(plugin, channel)

        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        assert len(captured_prompts) == 2
        final_prompt = captured_prompts[1]
        tool_contents = [m.get("content", "") for m in final_prompt if m.get("role") == "tool"]
        assert any("alpha" in c for c in tool_contents), f"Missing 'alpha': {tool_contents}"
        assert any("beta" in c for c in tool_contents), f"Missing 'beta': {tool_contents}"

        await task_plugin.on_stop()
        await db.close()

    async def test_no_duplicate_send_message_with_multi_tool(self):
        """send_message must be called exactly once, not multiple times."""
        plugin, channel, db = await build_plugin_and_channel()

        async def tool_a(x: str) -> str:
            return "a"

        async def tool_b(x: str) -> str:
            return "b"

        from corvidae.tool import tool_to_schema
        plugin._tools = {"tool_a": tool_a, "tool_b": tool_b}
        plugin._tool_schemas = [tool_to_schema(tool_a), tool_to_schema(tool_b)]

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response([
                    _make_tool_call("call_a", "tool_a", {"x": "1"}),
                    _make_tool_call("call_b", "tool_b", {"x": "2"}),
                ]),
                _make_text_response("final response"),
            ]
        )
        plugin._client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="go")
        await drain(plugin, channel)

        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        assert plugin.pm.ahook.send_message.await_count == 1, (
            f"send_message called {plugin.pm.ahook.send_message.await_count} times, expected 1"
        )
        await task_plugin.on_stop()
        await db.close()

    async def test_fresh_results_still_trigger_llm_after_user_message(self):
        """After a user message clears pending IDs, new tool dispatches from
        the user's response use the new generation and work normally."""
        plugin, channel, db = await build_plugin_and_channel()

        async def my_tool(x: str) -> str:
            return "tool_output"

        from corvidae.tool import tool_to_schema
        plugin._tools = {"my_tool": my_tool}
        plugin._tool_schemas = [tool_to_schema(my_tool)]

        llm_call_count = 0

        async def counting_chat(messages, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count == 1:
                # First turn: simple response (no tools)
                return _make_text_response("hello")
            elif llm_call_count == 2:
                # Second turn: dispatch tool
                return _make_tool_call_response([
                    _make_tool_call("new_call", "my_tool", {"x": "1"}),
                ])
            else:
                return _make_text_response("done")

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=counting_chat)
        plugin._client = mock_client

        # First message: simple response
        await plugin.on_message(channel=channel, sender="user", text="hi")
        await drain(plugin, channel)
        assert llm_call_count == 1

        # Second message triggers tool dispatch
        await plugin.on_message(channel=channel, sender="user", text="use tool")
        await drain(plugin, channel)
        assert llm_call_count == 2

        # Tool completes
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
        await drain(plugin, channel)

        # Tool result triggers final LLM call
        assert llm_call_count == 3
        plugin.pm.ahook.send_message.assert_awaited()

        await task_plugin.on_stop()
        await db.close()
