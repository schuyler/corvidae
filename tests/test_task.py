"""Tests for sherman.task — Task, TaskQueue, TaskPlugin."""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from sherman.channel import Channel, ChannelConfig
from sherman.hooks import AgentSpec, create_plugin_manager

from sherman.task import Task, TaskPlugin, TaskQueue


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_sherman_logger():
    """Ensure the sherman logger propagates to root so caplog captures records.

    Other test modules (test_logging.py) may apply dictConfig with
    propagate=False on the sherman logger. This fixture resets it.
    """
    sherman_logger = logging.getLogger("sherman")
    original_propagate = sherman_logger.propagate
    original_handlers = sherman_logger.handlers[:]
    sherman_logger.propagate = True
    yield
    sherman_logger.propagate = original_propagate
    sherman_logger.handlers = original_handlers


def _make_channel(transport="test", scope="scope1") -> Channel:
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


# ---------------------------------------------------------------------------
# TestTask
# ---------------------------------------------------------------------------


class TestTask:
    def test_task_id_auto_generated(self):
        """task_id is a 12-character hex string when not supplied."""
        channel = _make_channel()

        async def work():
            return "done"

        task = Task(work=work, channel=channel)
        assert isinstance(task.task_id, str)
        assert len(task.task_id) == 12
        # Must be a valid hex string
        int(task.task_id, 16)

    def test_task_fields(self):
        """All fields are stored correctly when provided explicitly."""
        channel = _make_channel()

        async def work():
            return "result"

        task = Task(
            work=work,
            channel=channel,
            task_id="abc123def456",
            tool_call_id="call_99",
            description="fetch data",
        )
        assert task.work is work
        assert task.channel is channel
        assert task.task_id == "abc123def456"
        assert task.tool_call_id == "call_99"
        assert task.description == "fetch data"

    def test_created_at_auto_set(self):
        """created_at is set to a timestamp between before/after construction."""
        channel = _make_channel()

        async def work():
            return "done"

        before = time.time()
        task = Task(work=work, channel=channel)
        after = time.time()
        assert before <= task.created_at <= after

    def test_tool_call_id_defaults_none(self):
        """tool_call_id defaults to None."""
        channel = _make_channel()

        async def work():
            return "done"

        task = Task(work=work, channel=channel)
        assert task.tool_call_id is None

    def test_description_defaults_empty(self):
        """description defaults to empty string."""
        channel = _make_channel()

        async def work():
            return "done"

        task = Task(work=work, channel=channel)
        assert task.description == ""


# ---------------------------------------------------------------------------
# TestTaskQueue
# ---------------------------------------------------------------------------


class TestTaskQueue:
    async def test_enqueue_and_dequeue(self):
        """Worker receives the enqueued task and calls task.work()."""
        queue = TaskQueue()
        channel = _make_channel()
        done = asyncio.Event()
        work_called = []

        async def work():
            work_called.append(True)
            done.set()
            return "result"

        task = Task(work=work, channel=channel, description="t1")

        async def on_complete(t, result):
            pass

        worker = asyncio.create_task(queue.run_worker(on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        assert work_called == [True]

    async def test_fifo_ordering(self):
        """Tasks are processed in the order they were enqueued."""
        queue = TaskQueue()
        channel = _make_channel()
        order = []
        all_done = asyncio.Event()

        def make_work(label):
            async def work():
                order.append(label)
                if len(order) == 3:
                    all_done.set()
                return "ok"
            return work

        tasks = [
            Task(work=make_work(f"t{i}"), channel=channel, description=f"t{i}")
            for i in range(3)
        ]

        async def on_complete(t, result):
            pass

        worker = asyncio.create_task(queue.run_worker(on_complete))
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
        executing = asyncio.Event()
        proceed = asyncio.Event()
        done = asyncio.Event()
        active_during = []

        async def work():
            active_during.append(queue.active_task)
            executing.set()
            await proceed.wait()
            return "done"

        task = Task(work=work, channel=channel, description="active")

        async def on_complete(t, result):
            done.set()

        assert queue.active_task is None

        worker = asyncio.create_task(queue.run_worker(on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(executing.wait(), timeout=2.0)

        assert queue.active_task is task

        proceed.set()
        await asyncio.wait_for(done.wait(), timeout=2.0)
        await asyncio.sleep(0)
        assert queue.active_task is None

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        assert active_during[0] is task

    async def test_completed_dict(self):
        """Result is stored in completed dict keyed by task_id."""
        queue = TaskQueue()
        channel = _make_channel()
        done = asyncio.Event()

        async def work():
            return "stored result"

        task = Task(work=work, channel=channel, description="store")

        async def on_complete(t, result):
            done.set()

        worker = asyncio.create_task(queue.run_worker(on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        assert task.task_id in queue.completed
        assert queue.completed[task.task_id] == "stored result"

    async def test_work_error(self):
        """When work() raises, error is stored in completed and on_complete is called; worker continues."""
        queue = TaskQueue()
        channel = _make_channel()
        completions = []
        all_done = asyncio.Event()

        async def failing_work():
            raise RuntimeError("boom")

        async def succeeding_work():
            return "success"

        task_fail = Task(work=failing_work, channel=channel, description="fail")
        task_ok = Task(work=succeeding_work, channel=channel, description="ok")

        async def on_complete(t, result):
            completions.append((t.task_id, result))
            if len(completions) == 2:
                all_done.set()

        worker = asyncio.create_task(queue.run_worker(on_complete))
        await queue.enqueue(task_fail)
        await queue.enqueue(task_ok)

        await asyncio.wait_for(all_done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        assert task_fail.task_id in queue.completed
        fail_result = queue.completed[task_fail.task_id]
        assert "failed" in fail_result.lower() or "boom" in fail_result.lower()

        completed_ids = [c[0] for c in completions]
        assert task_fail.task_id in completed_ids
        assert task_ok.task_id in completed_ids

        assert queue.completed[task_ok.task_id] == "success"

    async def test_worker_cancellation(self):
        """Cancelling the worker task raises CancelledError cleanly."""
        queue = TaskQueue()

        async def on_complete(t, result):
            pass

        worker = asyncio.create_task(queue.run_worker(on_complete))
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
        executing = asyncio.Event()
        proceed = asyncio.Event()

        async def work():
            executing.set()
            await proceed.wait()
            return "done"

        task = Task(work=work, channel=channel, description="active task desc")

        async def on_complete(t, result):
            pass

        worker = asyncio.create_task(queue.run_worker(on_complete))
        await queue.enqueue(task)
        await asyncio.wait_for(executing.wait(), timeout=2.0)

        status = queue.status()
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
        done = asyncio.Event()

        async def work():
            return "task output"

        task = Task(work=work, channel=channel, description="done task")

        async def on_complete(t, result):
            done.set()

        worker = asyncio.create_task(queue.run_worker(on_complete))
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
# TestTaskQueueLogging
# ---------------------------------------------------------------------------


class TestTaskQueueLogging:
    async def test_enqueue_logs_debug(self, caplog):
        """enqueue() must emit a DEBUG record 'task enqueued' with task_id,
        channel, and description attributes."""
        queue = TaskQueue()
        channel = _make_channel()

        async def work():
            return "done"

        task = Task(work=work, channel=channel, description="log test task")

        with caplog.at_level(logging.DEBUG, logger="sherman.task"):
            await queue.enqueue(task)

        records = [r for r in caplog.records if r.name == "sherman.task"]
        matching = [
            r for r in records
            if r.levelno == logging.DEBUG and r.getMessage() == "task enqueued"
        ]
        assert matching, "Expected DEBUG record with message 'task enqueued'"
        rec = matching[0]
        assert hasattr(rec, "task_id"), "'task enqueued' log must have task_id attribute"
        assert rec.task_id == task.task_id
        assert hasattr(rec, "channel"), "'task enqueued' log must have channel attribute"
        assert rec.channel == channel.id
        assert hasattr(rec, "description"), "'task enqueued' log must have description attribute"
        assert rec.description == task.description

    async def test_run_worker_logs_task_started(self, caplog):
        """Worker must emit a DEBUG record 'task started' after dequeuing,
        with task_id and description attributes."""
        queue = TaskQueue()
        channel = _make_channel()
        done = asyncio.Event()

        async def work():
            done.set()
            return "result"

        task = Task(work=work, channel=channel, description="started log task")

        async def on_complete(t, result):
            pass

        with caplog.at_level(logging.DEBUG, logger="sherman.task"):
            worker = asyncio.create_task(queue.run_worker(on_complete))
            await queue.enqueue(task)
            await asyncio.wait_for(done.wait(), timeout=2.0)
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        records = [r for r in caplog.records if r.name == "sherman.task"]
        matching = [
            r for r in records
            if r.levelno == logging.DEBUG and r.getMessage() == "task started"
        ]
        assert matching, "Expected DEBUG record with message 'task started'"
        rec = matching[0]
        assert hasattr(rec, "task_id"), "'task started' log must have task_id attribute"
        assert rec.task_id == task.task_id
        assert hasattr(rec, "description"), "'task started' log must have description attribute"
        assert rec.description == task.description

    async def test_run_worker_logs_task_completed(self, caplog):
        """Worker must emit a DEBUG record 'task completed' after successful
        work(), with task_id and result_length attributes."""
        queue = TaskQueue()
        channel = _make_channel()
        done = asyncio.Event()

        async def work():
            return "completed output"

        task = Task(work=work, channel=channel, description="completed log task")

        async def on_complete(t, result):
            done.set()

        with caplog.at_level(logging.DEBUG, logger="sherman.task"):
            worker = asyncio.create_task(queue.run_worker(on_complete))
            await queue.enqueue(task)
            await asyncio.wait_for(done.wait(), timeout=2.0)
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        records = [r for r in caplog.records if r.name == "sherman.task"]
        matching = [
            r for r in records
            if r.levelno == logging.DEBUG and r.getMessage() == "task completed"
        ]
        assert matching, "Expected DEBUG record with message 'task completed'"
        rec = matching[0]
        assert hasattr(rec, "task_id"), "'task completed' log must have task_id attribute"
        assert rec.task_id == task.task_id
        assert hasattr(rec, "result_length"), "'task completed' log must have result_length attribute"
        assert isinstance(rec.result_length, int), "result_length must be an int"

    async def test_run_worker_logs_task_failed_warning(self, caplog):
        """Worker must emit a WARNING record 'task failed' when work() raises,
        with task_id attribute and exc_info attached."""
        queue = TaskQueue()
        channel = _make_channel()
        done = asyncio.Event()

        async def work():
            raise RuntimeError("boom")

        task = Task(work=work, channel=channel, description="fail task")

        async def on_complete(t, result):
            done.set()

        with caplog.at_level(logging.DEBUG, logger="sherman.task"):
            worker = asyncio.create_task(queue.run_worker(on_complete))
            await queue.enqueue(task)
            await asyncio.wait_for(done.wait(), timeout=2.0)
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        records = [r for r in caplog.records if r.name == "sherman.task"]
        matching = [
            r for r in records
            if r.levelno == logging.WARNING and r.getMessage() == "task failed"
        ]
        assert matching, "Expected WARNING record with message 'task failed'"
        rec = matching[0]
        assert hasattr(rec, "task_id"), "'task failed' log must have task_id attribute"
        assert rec.task_id == task.task_id
        assert rec.exc_info is not None, "'task failed' log must have exc_info attached"
        assert rec.exc_info != (None, None, None), (
            "exc_info must contain actual exception information"
        )


# ---------------------------------------------------------------------------
# TestTaskPlugin
# ---------------------------------------------------------------------------


class TestTaskPlugin:
    async def test_on_start_creates_queue_and_worker(self):
        """on_start initializes task_queue and starts the worker task."""
        pm = create_plugin_manager()
        pm.ahook.on_notify = AsyncMock()
        plugin = TaskPlugin(pm)

        assert plugin.task_queue is None
        assert plugin._worker_task is None

        await plugin.on_start(config={})

        assert plugin.task_queue is not None
        assert isinstance(plugin.task_queue, TaskQueue)
        assert plugin._worker_task is not None
        assert not plugin._worker_task.done()

        await plugin.on_stop()

    async def test_on_stop_cancels_worker(self):
        """on_stop cancels the worker task cleanly."""
        pm = create_plugin_manager()
        pm.ahook.on_notify = AsyncMock()
        plugin = TaskPlugin(pm)

        await plugin.on_start(config={})
        worker = plugin._worker_task
        assert not worker.done()

        await plugin.on_stop()

        assert worker.done()

    async def test_on_task_complete_fires_on_notify(self):
        """_on_task_complete fires on_notify with correct arguments."""
        pm = create_plugin_manager()
        pm.ahook.on_notify = AsyncMock()
        plugin = TaskPlugin(pm)
        await plugin.on_start(config={})

        channel = _make_channel()

        async def work():
            return "work output"

        task = Task(
            work=work,
            channel=channel,
            tool_call_id="call_42",
            description="notify test",
        )

        await plugin._on_task_complete(task, "work output")

        pm.ahook.on_notify.assert_awaited_once()
        call_kwargs = pm.ahook.on_notify.call_args.kwargs
        assert call_kwargs["channel"] is channel
        assert call_kwargs["source"] == "task"
        assert task.task_id in call_kwargs["text"]
        assert "work output" in call_kwargs["text"]
        assert call_kwargs["tool_call_id"] == "call_42"
        assert call_kwargs["meta"]["task_id"] == task.task_id

        await plugin.on_stop()

    def test_register_tools_registers_task_status(self):
        """TaskPlugin.register_tools appends a task_status tool to the registry.

        Phase 5: task_status moves from BackgroundPlugin to TaskPlugin.

        Note: register_tools is called before on_start intentionally — this test
        checks registration only, not invocation of the tool.
        """
        pm = create_plugin_manager()
        plugin = TaskPlugin(pm)

        tool_registry = []
        plugin.register_tools(tool_registry=tool_registry)

        tool_names = [
            item.name if hasattr(item, "name") else item.__name__
            for item in tool_registry
        ]
        assert "task_status" in tool_names, (
            f"Expected 'task_status' in registered tools, got: {tool_names}"
        )

    async def test_task_status_tool_returns_queue_info(self):
        """The task_status tool registered by TaskPlugin returns correct queue information.

        Verifies active task, pending queue size, and completed count are reflected
        in the output from the tool callable.
        """
        from sherman.tool import Tool

        pm = create_plugin_manager()
        pm.ahook.on_notify = AsyncMock()
        plugin = TaskPlugin(pm)
        await plugin.on_start(config={})

        # Confirm task_status is registered
        tool_registry = []
        plugin.register_tools(tool_registry=tool_registry)
        task_status_item = next(
            (item for item in tool_registry
             if (item.name if hasattr(item, "name") else item.__name__) == "task_status"),
            None,
        )
        assert task_status_item is not None, "task_status tool must be registered"

        # Resolve the callable
        task_status_fn = (
            task_status_item.fn if isinstance(task_status_item, Tool) else task_status_item
        )

        # With an empty queue, should report no tasks
        result = await task_status_fn()
        assert isinstance(result, str)
        assert "no tasks" in result.lower(), (
            f"Expected 'no tasks' in empty-queue status, got: {result!r}"
        )

        # Enqueue a task so there is a completed entry, then check status again
        channel = _make_channel()
        done = asyncio.Event()

        async def work():
            return "status tool test output"

        task = Task(work=work, channel=channel, description="status tool test")

        async def on_complete(t, res):
            done.set()

        # Stop the plugin's internal worker to avoid race with our manual worker
        plugin._worker_task.cancel()
        try:
            await plugin._worker_task
        except asyncio.CancelledError:
            pass

        worker = asyncio.create_task(plugin.task_queue.run_worker(on_complete))
        await plugin.task_queue.enqueue(task)
        await asyncio.wait_for(done.wait(), timeout=2.0)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        result = await task_status_fn()
        assert task.task_id in result or "status tool test output" in result, (
            f"Expected completed task info in status, got: {result!r}"
        )

        await plugin.on_stop()


# ---------------------------------------------------------------------------
# Item 7 red-phase: on_task_complete must NOT be in AgentSpec (dead hook)
# ---------------------------------------------------------------------------


class TestOnTaskCompleteRemoved:
    def test_agent_spec_does_not_define_on_task_complete(self):
        """AgentSpec must not define on_task_complete after the dead hook is removed.

        RED phase: this test fails while on_task_complete still exists in AgentSpec.
        It passes once the hookspec and call site are deleted.
        """
        assert not hasattr(AgentSpec, "on_task_complete"), (
            "on_task_complete is a dead hook with no implementations; "
            "it should be removed from AgentSpec"
        )
