"""Tests for sherman.background — BackgroundTask, TaskQueue, task tools, and integration.

All tests are expected to FAIL until sherman/background.py is implemented
and sherman/agent_loop_plugin.py is updated for Phase 3.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from sherman.agent_loop import tool_to_schema
from sherman.channel import Channel, ChannelConfig, ChannelRegistry
from sherman.conversation import ConversationLog, init_db
from sherman.plugin_manager import create_plugin_manager

# The imports under test — will raise ImportError until the files exist.
from sherman.background import BackgroundTask, TaskQueue
from sherman.agent_loop_plugin import AgentLoopPlugin


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
}

BASE_CONFIG = {
    "llm": {
        "main": {
            "base_url": "http://localhost:8080",
            "model": "test-model",
        },
    },
    "daemon": {
        "session_db": ":memory:",
    },
}


def _make_channel(transport="test", scope="scope1") -> Channel:
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


async def _build_plugin_with_mocks():
    """Create an AgentLoopPlugin with mocked LLMClient and in-memory DB.

    Calls on_start so that TaskQueue and task tool closures are initialized.
    Returns (plugin, channel, db).
    """
    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.registry = registry
    pm.ahook.send_message = AsyncMock()
    pm.ahook.on_agent_response = AsyncMock()
    pm.ahook.on_task_complete = AsyncMock()

    plugin = AgentLoopPlugin(pm)
    pm.register(plugin, name="agent_loop")
    plugin.db = db

    mock_client = MagicMock()
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()

    with patch("sherman.agent_loop_plugin.LLMClient", return_value=mock_client), \
         patch("sherman.agent_loop_plugin.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
         patch("sherman.agent_loop_plugin.init_db", new_callable=AsyncMock):
        mock_connect.return_value = MagicMock()
        await plugin.on_start(config=BASE_CONFIG)

    # Restore the real db after on_start (on_start may have replaced it)
    plugin.db = db

    channel = registry.get_or_create("test", "scope1")
    return plugin, channel, db


# ---------------------------------------------------------------------------
# TestToolToSchema — zero-parameter function
# ---------------------------------------------------------------------------


class TestToolToSchema:
    def test_tool_to_schema_zero_params(self):
        """tool_to_schema on a zero-parameter async function should produce
        a schema with empty properties and no 'required' key."""

        async def noop() -> str:
            """Do nothing."""
            ...

        schema = tool_to_schema(noop)
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert params.get("properties") == {} or "properties" not in params or params["properties"] == {}
        assert "required" not in params


# ---------------------------------------------------------------------------
# TestBackgroundTask
# ---------------------------------------------------------------------------


class TestBackgroundTask:
    def test_task_id_auto_generated(self):
        channel = _make_channel()
        task = BackgroundTask(
            channel=channel,
            description="do something",
            instructions="step by step",
        )
        assert isinstance(task.task_id, str)
        assert len(task.task_id) == 12
        # Must be a valid hex string
        int(task.task_id, 16)

    def test_task_fields(self):
        channel = _make_channel()
        task = BackgroundTask(
            channel=channel,
            description="fetch data",
            instructions="go get it",
            task_id="abc123def456",
        )
        assert task.channel is channel
        assert task.description == "fetch data"
        assert task.instructions == "go get it"
        assert task.task_id == "abc123def456"

    def test_created_at_auto_set(self):
        channel = _make_channel()
        before = time.time()
        task = BackgroundTask(
            channel=channel,
            description="test",
            instructions="test instructions",
        )
        after = time.time()
        assert before <= task.created_at <= after


# ---------------------------------------------------------------------------
# TestTaskQueue
# ---------------------------------------------------------------------------


class TestTaskQueue:
    async def test_enqueue_and_dequeue(self):
        """Worker receives the enqueued task via execute_fn."""
        queue = TaskQueue()
        channel = _make_channel()
        task = BackgroundTask(channel=channel, description="t1", instructions="do t1")

        received = []
        done = asyncio.Event()

        async def execute_fn(t):
            received.append(t)
            done.set()
            return "result"

        async def on_complete(t, result):
            pass

        worker = asyncio.create_task(queue.run_worker(execute_fn, on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        assert len(received) == 1
        assert received[0] is task

    async def test_fifo_ordering(self):
        """Tasks are processed in the order they were enqueued."""
        queue = TaskQueue()
        channel = _make_channel()

        tasks = [
            BackgroundTask(channel=channel, description=f"t{i}", instructions=f"do t{i}")
            for i in range(3)
        ]

        order = []
        all_done = asyncio.Event()

        async def execute_fn(t):
            order.append(t.description)
            if len(order) == 3:
                all_done.set()
            return "ok"

        async def on_complete(t, result):
            pass

        worker = asyncio.create_task(queue.run_worker(execute_fn, on_complete))
        for t in tasks:
            await queue.enqueue(t)

        await asyncio.wait_for(all_done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        assert order == ["t0", "t1", "t2"]

    async def test_active_task_tracking(self):
        """active_task is set during execution and None before/after."""
        queue = TaskQueue()
        channel = _make_channel()
        task = BackgroundTask(channel=channel, description="active", instructions="run")

        executing = asyncio.Event()
        proceed = asyncio.Event()
        done = asyncio.Event()

        active_during = []

        async def execute_fn(t):
            active_during.append(queue.active_task)
            executing.set()
            await proceed.wait()
            return "done"

        async def on_complete(t, result):
            done.set()

        assert queue.active_task is None

        worker = asyncio.create_task(queue.run_worker(execute_fn, on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(executing.wait(), timeout=2.0)

        # active_task should be set while execute_fn is running
        assert queue.active_task is task

        proceed.set()
        await asyncio.wait_for(done.wait(), timeout=2.0)

        await asyncio.sleep(0)  # yield to event loop
        assert queue.active_task is None

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        # active_task captured inside execute_fn
        assert active_during[0] is task

    async def test_completed_dict(self):
        """Result is stored in completed dict keyed by task_id."""
        queue = TaskQueue()
        channel = _make_channel()
        task = BackgroundTask(channel=channel, description="store", instructions="work")
        done = asyncio.Event()

        async def execute_fn(t):
            return "stored result"

        async def on_complete(t, result):
            done.set()

        worker = asyncio.create_task(queue.run_worker(execute_fn, on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        assert task.task_id in queue.completed
        assert queue.completed[task.task_id] == "stored result"

    async def test_execute_fn_error(self):
        """When execute_fn raises, error is stored in completed and on_complete is called; worker continues."""
        queue = TaskQueue()
        channel = _make_channel()
        task_fail = BackgroundTask(channel=channel, description="fail", instructions="explode")
        task_ok = BackgroundTask(channel=channel, description="ok", instructions="succeed")

        completions = []
        all_done = asyncio.Event()

        async def execute_fn(t):
            if t.description == "fail":
                raise RuntimeError("boom")
            return "success"

        async def on_complete(t, result):
            completions.append((t.task_id, result))
            if len(completions) == 2:
                all_done.set()

        worker = asyncio.create_task(queue.run_worker(execute_fn, on_complete))
        await queue.enqueue(task_fail)
        await queue.enqueue(task_ok)

        await asyncio.wait_for(all_done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        # Error result stored
        assert task_fail.task_id in queue.completed
        assert "failed" in queue.completed[task_fail.task_id].lower() or "boom" in queue.completed[task_fail.task_id].lower()

        # on_complete called for both
        completed_ids = [c[0] for c in completions]
        assert task_fail.task_id in completed_ids
        assert task_ok.task_id in completed_ids

        # Worker continued and processed second task
        assert queue.completed[task_ok.task_id] == "success"

    async def test_worker_cancellation(self):
        """Cancelling the worker task raises CancelledError cleanly."""
        queue = TaskQueue()

        async def execute_fn(t):
            return "ok"

        async def on_complete(t, result):
            pass

        worker = asyncio.create_task(queue.run_worker(execute_fn, on_complete))
        # Give event loop a chance to start the worker
        await asyncio.sleep(0)
        worker.cancel()

        with pytest.raises(asyncio.CancelledError):
            await worker

    def test_status_no_tasks(self):
        """Empty queue returns a 'no tasks' message."""
        queue = TaskQueue()
        status = queue.status()
        assert "no tasks" in status.lower()

    async def test_status_with_active(self):
        """status() shows the active task during execution."""
        queue = TaskQueue()
        channel = _make_channel()
        task = BackgroundTask(channel=channel, description="active task desc", instructions="go")

        executing = asyncio.Event()
        proceed = asyncio.Event()

        async def execute_fn(t):
            executing.set()
            await proceed.wait()
            return "done"

        async def on_complete(t, result):
            pass

        worker = asyncio.create_task(queue.run_worker(execute_fn, on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(executing.wait(), timeout=2.0)

        status = queue.status()
        # Status must mention the active task in some form
        assert task.task_id in status or task.description in status

        proceed.set()
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    async def test_status_with_completed(self):
        """status() lists completed task results."""
        queue = TaskQueue()
        channel = _make_channel()
        task = BackgroundTask(channel=channel, description="done task", instructions="finish")
        done = asyncio.Event()

        async def execute_fn(t):
            return "task output"

        async def on_complete(t, result):
            done.set()

        worker = asyncio.create_task(queue.run_worker(execute_fn, on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        status = queue.status()
        assert task.task_id in status or "task output" in status


# ---------------------------------------------------------------------------
# TestBackgroundTaskTool (integration — closure behavior)
# ---------------------------------------------------------------------------


class TestBackgroundTaskTool:
    async def test_background_task_enqueues(self):
        """Calling the per-message background_task closure enqueues a task."""
        plugin, channel, db = await _build_plugin_with_mocks()

        # Simulate on_message creating the per-call closure by calling on_message
        # with a mocked run_agent_loop that invokes the background_task tool.
        # Instead, directly exercise the closure created in on_message by
        # inspecting what happens when on_message calls run_agent_loop.

        enqueued_tasks = []
        original_enqueue = plugin.task_queue.enqueue

        async def tracking_enqueue(task):
            enqueued_tasks.append(task)
            await original_enqueue(task)

        plugin.task_queue.enqueue = tracking_enqueue

        # Patch run_agent_loop to call background_task via local_tools
        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            # Call the per-call background_task closure from local_tools
            result = await tools["background_task"](
                description="my bg task",
                instructions="do the thing",
            )
            return result

        with patch("sherman.agent_loop_plugin.run_agent_loop", side_effect=fake_run_agent_loop):
            await plugin.on_message(channel=channel, sender="user", text="launch task")
            # Drain so the consumer runs fake_run_agent_loop (which calls background_task)
            if channel.id in plugin._queues:
                await plugin._queues[channel.id].drain()

        assert len(enqueued_tasks) == 1
        assert enqueued_tasks[0].description == "my bg task"
        assert enqueued_tasks[0].instructions == "do the thing"

        await plugin.on_stop()
        await db.close()

    async def test_background_task_uses_current_channel(self):
        """The per-call closure captures the correct channel."""
        plugin, channel, db = await _build_plugin_with_mocks()

        captured_channels = []
        original_enqueue = plugin.task_queue.enqueue

        async def tracking_enqueue(task):
            captured_channels.append(task.channel)
            await original_enqueue(task)

        plugin.task_queue.enqueue = tracking_enqueue

        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            await tools["background_task"](description="desc", instructions="instr")
            return "done"

        with patch("sherman.agent_loop_plugin.run_agent_loop", side_effect=fake_run_agent_loop):
            await plugin.on_message(channel=channel, sender="user", text="go")
            # Drain so the consumer runs fake_run_agent_loop (which calls background_task)
            if channel.id in plugin._queues:
                await plugin._queues[channel.id].drain()

        assert len(captured_channels) == 1
        assert captured_channels[0] is channel

        await plugin.on_stop()
        await db.close()

    async def test_task_status_reports_correctly(self):
        """task_status closure returns the queue status string."""
        plugin, channel, db = await _build_plugin_with_mocks()

        # task_status is in self.tools (registered during on_start)
        assert "task_status" in plugin.tools
        task_status_fn = plugin.tools["task_status"]

        result = await task_status_fn()
        assert isinstance(result, str)
        assert len(result) > 0

        await db.close()


# ---------------------------------------------------------------------------
# TestBackgroundWorkerIntegration
# ---------------------------------------------------------------------------


class TestBackgroundWorkerIntegration:
    async def test_worker_executes_with_agent_loop(self):
        """Worker calls run_agent_loop for a queued task and then send_message."""
        plugin, channel, db = await _build_plugin_with_mocks()

        loop_done = asyncio.Event()
        complete_done = asyncio.Event()

        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            loop_done.set()
            return "background result"

        original_on_task_complete = plugin._on_task_complete

        async def tracking_on_complete(task, result):
            await original_on_task_complete(task, result)
            complete_done.set()

        plugin._on_task_complete = tracking_on_complete

        task = BackgroundTask(
            channel=channel,
            description="bg work",
            instructions="do background work",
        )

        with patch("sherman.agent_loop_plugin.run_agent_loop", side_effect=fake_run_agent_loop):
            await plugin.task_queue.enqueue(task)
            await asyncio.wait_for(loop_done.wait(), timeout=2.0)
            await asyncio.wait_for(complete_done.wait(), timeout=2.0)
            # _on_task_complete calls on_notify which enqueues to the channel queue.
            # Drain so the notification consumer runs (and calls send_message) while
            # run_agent_loop is still patched.
            if channel.id in plugin._queues:
                await asyncio.wait_for(plugin._queues[channel.id].drain(), timeout=2.0)

        plugin.pm.ahook.send_message.assert_awaited()

        await plugin.on_stop()
        await db.close()

    async def test_worker_posts_to_correct_channel(self):
        """Two tasks on different channels are routed to their respective channels."""
        plugin, channel_a, db = await _build_plugin_with_mocks()

        pm = plugin.pm
        registry = pm.registry
        channel_b = registry.get_or_create("test", "scope2")

        task_a_done = asyncio.Event()
        task_b_done = asyncio.Event()

        call_count = [0]

        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            call_count[0] += 1
            return f"result {call_count[0]}"

        original_on_task_complete = plugin._on_task_complete

        async def tracking_on_complete(task, result):
            await original_on_task_complete(task, result)
            if task.channel is channel_a:
                task_a_done.set()
            elif task.channel is channel_b:
                task_b_done.set()

        plugin._on_task_complete = tracking_on_complete

        task_a = BackgroundTask(channel=channel_a, description="a", instructions="do a")
        task_b = BackgroundTask(channel=channel_b, description="b", instructions="do b")

        with patch("sherman.agent_loop_plugin.run_agent_loop", side_effect=fake_run_agent_loop):
            await plugin.task_queue.enqueue(task_a)
            await plugin.task_queue.enqueue(task_b)
            await asyncio.wait_for(task_a_done.wait(), timeout=3.0)
            await asyncio.wait_for(task_b_done.wait(), timeout=3.0)
            # _on_task_complete calls on_notify which enqueues to each channel queue.
            # Drain both so the notification consumers run while run_agent_loop is patched.
            for ch in [channel_a, channel_b]:
                if ch.id in plugin._queues:
                    await asyncio.wait_for(plugin._queues[ch.id].drain(), timeout=2.0)

        # send_message should have been called at least once per task
        send_calls = plugin.pm.ahook.send_message.call_args_list
        channels_called = [c.kwargs["channel"] for c in send_calls]
        assert channel_a in channels_called
        assert channel_b in channels_called

        await plugin.on_stop()
        await db.close()

    async def test_on_task_complete_hook_fired(self):
        """on_task_complete hook is called with correct task_id and result."""
        plugin, channel, db = await _build_plugin_with_mocks()

        done = asyncio.Event()

        async def fake_run_agent_loop(client, messages, tools, tool_schemas, **kwargs):
            return "hook test result"

        original_on_task_complete = plugin._on_task_complete

        async def tracking_on_complete(task, result):
            await original_on_task_complete(task, result)
            done.set()

        plugin._on_task_complete = tracking_on_complete

        task = BackgroundTask(channel=channel, description="hook test", instructions="fire hook")

        with patch("sherman.agent_loop_plugin.run_agent_loop", side_effect=fake_run_agent_loop):
            await plugin.task_queue.enqueue(task)
            await asyncio.wait_for(done.wait(), timeout=2.0)
            # _on_task_complete calls on_notify which enqueues to the channel queue.
            # Drain so the notification consumer runs while run_agent_loop is patched.
            if channel.id in plugin._queues:
                await asyncio.wait_for(plugin._queues[channel.id].drain(), timeout=2.0)

        plugin.pm.ahook.on_task_complete.assert_awaited()
        call_kwargs = plugin.pm.ahook.on_task_complete.call_args.kwargs
        assert call_kwargs["channel"] is channel
        assert call_kwargs["task_id"] == task.task_id
        assert call_kwargs["result"] == "hook test result"

        await plugin.on_stop()
        await db.close()

    async def test_on_stop_cancels_worker(self):
        """on_stop cancels the background worker task cleanly."""
        plugin, channel, db = await _build_plugin_with_mocks()

        # Worker should have been started during on_start
        assert plugin._worker_task is not None
        assert not plugin._worker_task.done()

        await plugin.on_stop()

        # Worker task should be done (cancelled) after on_stop
        assert plugin._worker_task.done()

        await db.close()
