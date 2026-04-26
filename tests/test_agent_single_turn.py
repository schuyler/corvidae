"""Tests for Phase 3 AgentPlugin behavior: single-turn dispatch via run_agent_turn.

These tests specify behavior that doesn't exist yet (Red TDD). They fail because:
- AgentPlugin still calls run_agent_loop instead of run_agent_turn
- AgentPlugin doesn't dispatch tool calls as Tasks
- Channel doesn't have turn_counter
- ChannelConfig doesn't have max_turns
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from sherman.agent import AgentPlugin
from sherman.agent_loop import AgentTurnResult, run_agent_turn  # noqa: F401 (used in comments/type checking)
from sherman.channel import Channel, ChannelConfig, ChannelRegistry
from sherman.conversation import init_db
from sherman.hooks import create_plugin_manager
from sherman.task import Task, TaskPlugin, TaskQueue
from sherman.tool import ToolContext


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
}


# ---------------------------------------------------------------------------
# Response builder helpers
# ---------------------------------------------------------------------------


def _make_text_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _make_tool_call_response(calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
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


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


async def _build_plugin_and_channel(
    agent_defaults=None,
    channel_config=None,
):
    """Create plugin manager, registry, in-memory DB, TaskPlugin, AgentPlugin, and channel.

    Callers must call task_plugin.on_stop() and db.close() in teardown.
    Use the plugin_and_channel fixture instead when possible.
    """
    if agent_defaults is None:
        agent_defaults = AGENT_DEFAULTS

    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    registry = ChannelRegistry(agent_defaults)
    pm.register(registry, name="registry")

    pm.ahook.send_message = AsyncMock()
    pm.ahook.on_agent_response = AsyncMock()
    # NOTE: on_notify is NOT mocked — both AgentPlugin and TaskPlugin use it.

    # Register TaskPlugin
    task_plugin = TaskPlugin(pm)
    pm.register(task_plugin, name="task")
    await task_plugin.on_start(config={})

    plugin = AgentPlugin(pm)
    pm.register(plugin, name="agent_loop")

    plugin.db = db
    # Inject the registry directly (tests bypass on_start where _registry is resolved).
    plugin._registry = registry

    channel = registry.get_or_create(
        "test",
        "scope1",
        config=channel_config or ChannelConfig(),
    )

    return plugin, channel, db


@pytest.fixture
async def plugin_and_channel(request):
    """Pytest fixture yielding (plugin, channel, db) with TaskPlugin teardown."""
    params = getattr(request, "param", {}) or {}
    plugin, channel, db = await _build_plugin_and_channel(
        agent_defaults=params.get("agent_defaults"),
        channel_config=params.get("channel_config"),
    )
    yield plugin, channel, db
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


async def _drain(plugin, channel):
    """Drain the channel's SerialQueue. Safe when queue was never created."""
    if channel.id in plugin._queues:
        await plugin._queues[channel.id].drain()


async def _drain_task_queue(plugin):
    """Wait for all pending tasks in the TaskQueue to complete, including on_complete callbacks."""
    import asyncio
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin and task_plugin.task_queue:
        await task_plugin.task_queue.queue.join()
        # queue.join() unblocks after task_done() but before on_complete runs.
        # Yield to let the worker coroutine continue and fire on_complete -> on_notify.
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Test 1: tool call dispatches a Task
# ---------------------------------------------------------------------------


class TestToolCallDispatchesTask:
    async def test_tool_call_dispatches_task(self, plugin_and_channel):
        """User message -> LLM returns tool call -> Task enqueued with correct
        tool_call_id, description, and channel.
        """
        plugin, channel, db = plugin_and_channel

        mock_client = MagicMock()
        # First LLM call returns a tool call; we capture the enqueued task
        # before it executes by replacing the task_queue with a spy.
        mock_client.chat = AsyncMock(
            return_value=_make_tool_call_response(
                [_make_tool_call("call-001", "my_tool", {"x": "value"})]
            )
        )
        plugin.client = mock_client

        # Register a simple tool so the tool call doesn't fail on execution
        async def my_tool(x: str) -> str:
            """A simple test tool."""
            return f"result:{x}"

        plugin.tools = {"my_tool": my_tool}

        # Capture tasks enqueued by replacing enqueue with a spy
        enqueued_tasks: list[Task] = []
        task_plugin = plugin.pm.get_plugin("task")
        original_enqueue = task_plugin.task_queue.enqueue

        async def spy_enqueue(task: Task) -> None:
            enqueued_tasks.append(task)
            # Also enqueue for real so the worker doesn't get confused
            await original_enqueue(task)

        task_plugin.task_queue.enqueue = spy_enqueue

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await _drain(plugin, channel)

        # Phase 3: AgentPlugin._dispatch_tool_calls enqueues exactly one Task
        assert len(enqueued_tasks) == 1
        task = enqueued_tasks[0]
        assert task.tool_call_id == "call-001"
        assert task.description == "tool:my_tool"
        assert task.channel is channel


# ---------------------------------------------------------------------------
# Test 2: full tool call round-trip triggers continuation
# ---------------------------------------------------------------------------


class TestToolCallResultTriggersContinuation:
    async def test_tool_call_result_triggers_continuation(self, plugin_and_channel):
        """Full cycle: user -> tool call -> task executes -> notification ->
        second LLM call -> text response -> send_message called.
        """
        plugin, channel, db = plugin_and_channel

        async def my_tool(x: str) -> str:
            """A test tool."""
            return f"tool_result:{x}"

        plugin.tools = {"my_tool": my_tool}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                # First call: tool call
                _make_tool_call_response(
                    [_make_tool_call("call-abc", "my_tool", {"x": "data"})]
                ),
                # Second call: text response after tool result arrives
                _make_text_response("Here is the answer."),
            ]
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="do the thing")
        await _drain(plugin, channel)

        # Wait for TaskQueue to execute the tool task
        await _drain_task_queue(plugin)

        # The task completion triggers on_notify -> new QueueItem -> second LLM turn
        await _drain(plugin, channel)

        # After full cycle, send_message is called with the final text
        plugin.pm.ahook.send_message.assert_called_once()
        call_kwargs = plugin.pm.ahook.send_message.call_args.kwargs
        assert call_kwargs["channel"] is channel
        assert "answer" in call_kwargs["text"].lower()


# ---------------------------------------------------------------------------
# Test 3: multiple tool calls dispatched as separate Tasks
# ---------------------------------------------------------------------------


class TestMultipleToolCallsDispatched:
    async def test_multiple_tool_calls_dispatched(self, plugin_and_channel):
        """LLM returns 3 tool calls -> 3 Tasks enqueued."""
        plugin, channel, db = plugin_and_channel

        async def tool_a(x: str) -> str:
            """Tool A."""
            return f"a:{x}"

        async def tool_b(x: str) -> str:
            """Tool B."""
            return f"b:{x}"

        async def tool_c(x: str) -> str:
            """Tool C."""
            return f"c:{x}"

        plugin.tools = {"tool_a": tool_a, "tool_b": tool_b, "tool_c": tool_c}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=_make_tool_call_response(
                [
                    _make_tool_call("call-1", "tool_a", {"x": "1"}),
                    _make_tool_call("call-2", "tool_b", {"x": "2"}),
                    _make_tool_call("call-3", "tool_c", {"x": "3"}),
                ]
            )
        )
        plugin.client = mock_client

        enqueued_tasks: list[Task] = []
        task_plugin = plugin.pm.get_plugin("task")
        original_enqueue = task_plugin.task_queue.enqueue

        async def spy_enqueue(task: Task) -> None:
            enqueued_tasks.append(task)
            await original_enqueue(task)

        task_plugin.task_queue.enqueue = spy_enqueue

        await plugin.on_message(channel=channel, sender="user", text="do three things")
        await _drain(plugin, channel)

        assert len(enqueued_tasks) == 3
        call_ids = {t.tool_call_id for t in enqueued_tasks}
        assert call_ids == {"call-1", "call-2", "call-3"}
        descriptions = {t.description for t in enqueued_tasks}
        assert descriptions == {"tool:tool_a", "tool:tool_b", "tool:tool_c"}


# ---------------------------------------------------------------------------
# Test 4: user message resets turn_counter to 0
# ---------------------------------------------------------------------------


class TestMaxTurnsResetsOnUserMessage:
    async def test_max_turns_resets_on_user_message(self, plugin_and_channel):
        """User message resets channel.turn_counter to 0 before the LLM call."""
        plugin, channel, db = plugin_and_channel

        # Pre-set turn_counter to a non-zero value to confirm reset
        channel.turn_counter = 5

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hello"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="reset me")
        await _drain(plugin, channel)

        # After a user message with a text response, turn_counter was reset to 0
        # then incremented to 1 (one LLM turn taken, no tool calls).
        assert channel.turn_counter == 1


# ---------------------------------------------------------------------------
# Test 5: notification-triggered turn increments counter
# ---------------------------------------------------------------------------


class TestMaxTurnsIncrementsOnNotification:
    async def test_max_turns_increments_on_notification(self, plugin_and_channel):
        """Notification-triggered turn increments turn_counter."""
        plugin, channel, db = plugin_and_channel

        async def my_tool(x: str) -> str:
            """A tool."""
            return "done"

        plugin.tools = {"my_tool": my_tool}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                # User message -> tool call dispatched (turn_counter -> 1)
                _make_tool_call_response(
                    [_make_tool_call("call-x", "my_tool", {"x": "v"})]
                ),
                # Notification -> text response (turn_counter -> 2)
                _make_text_response("done with it"),
            ]
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="go")
        await _drain(plugin, channel)
        assert channel.turn_counter == 1

        await _drain_task_queue(plugin)
        await _drain(plugin, channel)

        # After notification turn: counter should be 2
        assert channel.turn_counter == 2


# ---------------------------------------------------------------------------
# Test 6: max_turns exceeded suppresses tool calls
# ---------------------------------------------------------------------------


class TestMaxTurnsExceededSuppressesToolCalls:
    async def test_max_turns_exceeded_suppresses_tool_calls(self):
        """With max_turns=1, first tool call is dispatched; second is suppressed."""
        # ChannelConfig(max_turns=1) constructed at runtime to avoid collection
        # errors before the field exists.
        plugin, channel, db = await _build_plugin_and_channel(
            channel_config=ChannelConfig(max_turns=1),
        )

        async def my_tool(x: str) -> str:
            """A tool."""
            return "tool result"

        plugin.tools = {"my_tool": my_tool}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                # User message -> tool call (turn 0 -> dispatched, counter -> 1)
                _make_tool_call_response(
                    [_make_tool_call("call-first", "my_tool", {"x": "a"})]
                ),
                # Notification -> LLM returns another tool call, but max_turns=1
                # so this should NOT be dispatched; fallback text sent instead.
                _make_tool_call_response(
                    [_make_tool_call("call-second", "my_tool", {"x": "b"})]
                ),
            ]
        )
        plugin.client = mock_client

        enqueued_tasks: list[Task] = []
        task_plugin = plugin.pm.get_plugin("task")
        original_enqueue = task_plugin.task_queue.enqueue

        async def spy_enqueue(task: Task) -> None:
            enqueued_tasks.append(task)
            await original_enqueue(task)

        task_plugin.task_queue.enqueue = spy_enqueue

        try:
            # First turn: user message -> first tool call dispatched
            await plugin.on_message(channel=channel, sender="user", text="go")
            await _drain(plugin, channel)
            assert len(enqueued_tasks) == 1

            # Task executes and triggers notification
            await _drain_task_queue(plugin)
            await _drain(plugin, channel)

            # Second LLM turn hit max_turns -> no second task, but send_message called
            assert len(enqueued_tasks) == 1  # still only 1
            assert plugin.pm.ahook.send_message.called
        finally:
            task_plugin = plugin.pm.get_plugin("task")
            if task_plugin:
                await task_plugin.on_stop()
            await db.close()


# ---------------------------------------------------------------------------
# Test 7: max_turns exceeded sends fallback text
# ---------------------------------------------------------------------------


class TestMaxTurnsExceededSendsFallbackText:
    async def test_max_turns_exceeded_sends_fallback_text(self):
        """When max_turns hit and LLM produced only tool calls, fallback text sent."""
        # ChannelConfig(max_turns=1) constructed at runtime to avoid collection
        # errors before the field exists.
        plugin, channel, db = await _build_plugin_and_channel(
            channel_config=ChannelConfig(max_turns=1),
        )

        async def my_tool(x: str) -> str:
            """A tool."""
            return "tool result"

        plugin.tools = {"my_tool": my_tool}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                # User message -> tool call (dispatched, counter -> 1)
                _make_tool_call_response(
                    [_make_tool_call("call-a", "my_tool", {"x": "a"})]
                ),
                # Notification -> tool call with empty text (max_turns hit -> fallback)
                _make_tool_call_response(
                    [_make_tool_call("call-b", "my_tool", {"x": "b"})]
                ),
            ]
        )
        plugin.client = mock_client

        try:
            await plugin.on_message(channel=channel, sender="user", text="go")
            await _drain(plugin, channel)
            await _drain_task_queue(plugin)
            await _drain(plugin, channel)

            plugin.pm.ahook.send_message.assert_called_once()
            sent_text = plugin.pm.ahook.send_message.call_args.kwargs["text"]
            assert "max tool-calling rounds reached" in sent_text
        finally:
            task_plugin = plugin.pm.get_plugin("task")
            if task_plugin:
                await task_plugin.on_stop()
            await db.close()


# ---------------------------------------------------------------------------
# Test 8: tool dispatch with ToolContext injection
# ---------------------------------------------------------------------------


class TestToolDispatchWithToolContext:
    async def test_tool_dispatch_with_tool_context(self, plugin_and_channel):
        """Tool with _ctx parameter receives a ToolContext with correct fields."""
        plugin, channel, db = plugin_and_channel

        received_ctx: list[ToolContext] = []

        async def ctx_tool(_ctx: ToolContext) -> str:
            """A tool that receives context."""
            received_ctx.append(_ctx)
            return "context received"

        plugin.tools = {"ctx_tool": ctx_tool}
        plugin.tool_schemas = []  # schemas not needed for this test

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [_make_tool_call("call-ctx", "ctx_tool", {})]
                ),
                _make_text_response("done"),
            ]
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="use ctx tool")
        await _drain(plugin, channel)
        await _drain_task_queue(plugin)
        await _drain(plugin, channel)

        assert len(received_ctx) == 1
        ctx = received_ctx[0]
        assert isinstance(ctx, ToolContext)
        assert ctx.channel is channel
        assert ctx.tool_call_id == "call-ctx"
        assert ctx.task_queue is plugin.pm.get_plugin("task").task_queue


# ---------------------------------------------------------------------------
# Test 9: unknown tool produces error message
# ---------------------------------------------------------------------------


class TestToolDispatchUnknownTool:
    async def test_tool_dispatch_unknown_tool(self, plugin_and_channel):
        """LLM calls a tool not in self.tools -> task result is error message."""
        plugin, channel, db = plugin_and_channel

        plugin.tools = {}  # No tools registered

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [_make_tool_call("call-unk", "nonexistent_tool", {"x": "v"})]
                ),
                _make_text_response("I tried."),
            ]
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="use missing tool")
        await _drain(plugin, channel)
        await _drain_task_queue(plugin)
        await _drain(plugin, channel)

        # The task should have completed with an error message, triggering a
        # second LLM call. The conversation should contain a tool result with error.
        conv = channel.conversation
        tool_messages = [m for m in conv.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "unknown tool" in tool_messages[0]["content"].lower() or \
               "error" in tool_messages[0]["content"].lower()


# ---------------------------------------------------------------------------
# Test 10: tool raises exception -> error message, no crash
# ---------------------------------------------------------------------------


class TestToolDispatchToolRaises:
    async def test_tool_dispatch_tool_raises(self, plugin_and_channel):
        """Tool raises exception -> task result is error message, no crash."""
        plugin, channel, db = plugin_and_channel

        async def failing_tool(x: str) -> str:
            """A tool that always fails."""
            raise RuntimeError("tool explosion")

        plugin.tools = {"failing_tool": failing_tool}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [_make_tool_call("call-fail", "failing_tool", {"x": "boom"})]
                ),
                _make_text_response("I see the tool failed."),
            ]
        )
        plugin.client = mock_client

        # Should not raise
        await plugin.on_message(channel=channel, sender="user", text="use failing tool")
        await _drain(plugin, channel)
        await _drain_task_queue(plugin)
        await _drain(plugin, channel)

        # Conversation should have a tool result with an error message
        conv = channel.conversation
        tool_messages = [m for m in conv.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "error" in tool_messages[0]["content"].lower()


# ---------------------------------------------------------------------------
# Test 11: no TaskPlugin registered -> log error, no crash
# ---------------------------------------------------------------------------


class TestNoTaskQueueLogsError:
    async def test_no_task_queue_logs_error(self, caplog):
        """No TaskPlugin registered -> log error and don't crash."""
        import logging

        db = await aiosqlite.connect(":memory:")
        await init_db(db)

        pm = create_plugin_manager()
        registry = ChannelRegistry(AGENT_DEFAULTS)
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()
        # NOTE: No TaskPlugin registered — pm.get_plugin("task") will return None

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")
        plugin.db = db
        plugin._registry = registry

        async def my_tool(x: str) -> str:
            """A tool."""
            return "result"

        plugin.tools = {"my_tool": my_tool}

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=_make_tool_call_response(
                [_make_tool_call("call-notq", "my_tool", {"x": "v"})]
            )
        )
        plugin.client = mock_client

        channel = registry.get_or_create("test", "scope_notq")

        # Should not raise even though no TaskPlugin is registered
        with caplog.at_level(logging.ERROR, logger="sherman.agent"):
            await plugin.on_message(channel=channel, sender="user", text="hi")
            await _drain(plugin, channel)

        # Must log an error about the missing TaskQueue
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            "task" in r.message.lower() or "queue" in r.message.lower()
            for r in error_records
        ), f"Expected error log about missing task queue, got: {[r.message for r in error_records]}"

        await db.close()


# ---------------------------------------------------------------------------
# Test 12: assistant message with tool_calls persisted in conversation
# ---------------------------------------------------------------------------


class TestAssistantMessageWithToolCallsPersisted:
    async def test_assistant_message_with_tool_calls_persisted(self, plugin_and_channel):
        """When LLM returns tool_calls, the full assistant message
        (including tool_calls array) is persisted in the conversation log.
        """
        plugin, channel, db = plugin_and_channel

        async def my_tool(x: str) -> str:
            """A tool."""
            return "done"

        plugin.tools = {"my_tool": my_tool}

        tool_call = _make_tool_call("call-persist", "my_tool", {"x": "q"})
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response([tool_call]),
                _make_text_response("finished"),
            ]
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="persist test")
        await _drain(plugin, channel)

        conv = channel.conversation
        assistant_messages = [m for m in conv.messages if m.get("role") == "assistant"]
        assert len(assistant_messages) >= 1

        # The first assistant message must include tool_calls
        first_assistant = assistant_messages[0]
        assert "tool_calls" in first_assistant
        assert first_assistant["tool_calls"][0]["id"] == "call-persist"


# ---------------------------------------------------------------------------
# Test 13: send_message NOT called when tool calls are dispatched
# ---------------------------------------------------------------------------


class TestNoSendMessageWhenToolCallsDispatched:
    async def test_no_send_message_when_tool_calls_dispatched(self, plugin_and_channel):
        """When tool calls are dispatched, send_message is NOT called immediately.

        Under single-turn dispatch (Phase 3), _process_queue_item dispatches
        tool calls as Tasks and returns WITHOUT calling send_message. The
        response is deferred until tool results arrive via notification.

        Under the old run_agent_loop, send_message IS called after the full
        loop completes, so this test should FAIL (Red TDD).
        """
        plugin, channel, db = plugin_and_channel

        tool_called = False

        async def my_tool(x: str) -> str:
            """A quick tool."""
            nonlocal tool_called
            tool_called = True
            return "tool result"

        plugin.tools = {"my_tool": my_tool}

        mock_client = MagicMock()
        # First call: tool call. Second call: text response (used by run_agent_loop).
        mock_client.chat = AsyncMock(
            side_effect=[
                _make_tool_call_response(
                    [_make_tool_call("call-1", "my_tool", {"x": "v"})]
                ),
                _make_text_response("done"),
            ]
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="use tool")
        await _drain(plugin, channel)

        # Under Phase 3 single-turn dispatch: after the first drain,
        # _process_queue_item dispatches the tool Task and returns.
        # send_message should NOT have been called yet.
        # Under old run_agent_loop: send_message IS called -> test fails.
        plugin.pm.ahook.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# Test 14: compaction failure resilience
# ---------------------------------------------------------------------------


class TestCompactionFailureResilience:
    async def test_compaction_failure_does_not_prevent_response(
        self, plugin_and_channel, caplog
    ):
        """A compaction failure does not prevent run_agent_turn or send_message.

        Patches call_firstresult_hook in sherman.agent to raise RuntimeError
        when invoked for compact_conversation, simulating a plugin crash.
        The agent's try/except must catch it, log a WARNING, and continue.
        """
        import logging
        from unittest.mock import patch

        plugin, channel, db = plugin_and_channel

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("response after failed compaction"))
        plugin.client = mock_client

        original_hook = __import__("sherman.hooks", fromlist=["call_firstresult_hook"]).call_firstresult_hook

        async def exploding_hook(pm, hook_name, **kwargs):
            if hook_name == "compact_conversation":
                raise RuntimeError("compaction boom")
            return await original_hook(pm, hook_name, **kwargs)

        with caplog.at_level(logging.WARNING, logger="sherman.agent"):
            with patch("sherman.agent.call_firstresult_hook", side_effect=exploding_hook):
                await plugin.on_message(channel=channel, sender="user", text="hello")
                await _drain(plugin, channel)

        # Despite compaction failure, run_agent_turn must have been called
        # (evidenced by mock_client.chat being called) and send_message must have fired.
        mock_client.chat.assert_called_once()
        plugin.pm.ahook.send_message.assert_called_once()
        sent_text = plugin.pm.ahook.send_message.call_args.kwargs["text"]
        assert "response after failed compaction" in sent_text

        # The failure must be logged as a warning, not silently swallowed.
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "compaction" in r.getMessage().lower()
        ]
        assert warning_records, (
            "Expected a WARNING log about compaction failure; "
            "exceptions must not be silently swallowed"
        )
