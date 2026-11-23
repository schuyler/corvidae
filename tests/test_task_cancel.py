"""Tests for #504 direction 2 — cancelled tasks must leave a record.

Gap: in _run_one_worker, the asyncio.CancelledError handler removes the
task from _active_tasks and re-raises without appending to .completed,
without _persist_result, and without on_complete. The task then appears
in neither task_status nor task_result — invisible loss. Additionally,
the bare re-raise kills the worker coroutine forever (run_worker spawns
exactly max_workers and never respawns), so every observed cancel is
also permanent queue-capacity decay.

Required behavior (this issue's "done when"):
  (a) a cancelled task gets a completion record: a .completed entry and
      a JSONL line, both naming cancellation (not the failure template);
  (b) on_complete is invoked exactly once with the cancelled result —
      for work-level cancellation, awaited normally; for worker-level
      (shutdown) cancellation, best-effort via a detached dispatch,
      since awaiting inside an externally-cancelled task is unreliable;
  (c) shutdown semantics preserved: when the WORKER task is cancelled,
      the CancelledError still propagates out of _run_one_worker.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.task import Task, TaskQueue


def _make_channel(transport="test", scope="scope1") -> Channel:
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestTaskCancellationRecord:
    """#504-2: cancellation must be as inspectable as success and failure."""

    async def test_shutdown_cancel_records_task(self, tmp_path: Path):
        """Worker cancelled mid-work (daemon shutdown): the task gets a
        .completed entry and a JSONL record naming cancellation."""
        results_path = tmp_path / "results.jsonl"
        queue = TaskQueue(max_workers=1, results_path=results_path)
        channel = _make_channel()

        started = asyncio.Event()

        async def long_work():
            started.set()
            await asyncio.sleep(3600)
            return "never"

        task = Task(work=long_work, channel=channel, description="doomed")
        await queue.queue.put(task)

        notified = asyncio.Event()

        async def on_complete(t, r):
            assert t is task
            assert "cancel" in r.lower()
            notified.set()

        worker = asyncio.create_task(queue.run_worker(on_complete))
        await asyncio.wait_for(started.wait(), timeout=5)
        # Give the worker a beat to register the task as active.
        for _ in range(50):
            if task in queue._active_tasks:
                break
            await asyncio.sleep(0.02)
        assert task in queue._active_tasks

        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        # Detached shutdown notify is best-effort but must arrive while
        # the loop is still running (the daemon's loop outlives workers).
        await asyncio.wait_for(notified.wait(), timeout=2)

        recorded = list(queue.completed)
        assert len(recorded) == 1, (
            "cancelled task must appear in .completed exactly once "
            f"(#504-2), got {recorded}"
        )
        task_id, result = recorded[0]
        assert task_id == task.task_id
        assert "cancel" in result.lower(), (
            "cancelled record must name cancellation, not look like a "
            f"success or failure template: {result!r}"
        )

        lines = _read_jsonl(results_path)
        assert len(lines) == 1, "cancelled task must get one JSONL record"
        assert lines[0]["task_id"] == task.task_id
        assert "cancel" in lines[0]["result"].lower()

        # queue bookkeeping: task_done ran for the cancelled task, and it
        # is out of _active_tasks — no phantom active work after shutdown.
        assert task not in queue._active_tasks
        await asyncio.wait_for(queue.queue.join(), timeout=2)

    async def test_worker_survives_work_cancellation(self, tmp_path: Path):
        """Work-level cancellation (the awaited work raises CancelledError
        without the worker itself being cancelled) must NOT kill the
        worker: the cancelled task is recorded, delivered once via
        on_complete, and the worker keeps draining the queue."""
        results_path = tmp_path / "results.jsonl"
        queue = TaskQueue(max_workers=1, results_path=results_path)
        channel = _make_channel()
        completions = AsyncMock()

        started = asyncio.Event()
        self_cancel_done = asyncio.Event()

        async def self_cancel_work():
            # Simulate external cancel of the work task from the inside:
            # this is the shape of a timeout controller or an explicit
            # task-level cancel API calling work_task.cancel().
            started.set()
            current = asyncio.current_task()
            current.cancel()
            try:
                await asyncio.sleep(3600)
            finally:
                self_cancel_done.set()
            return "never"

        async def normal_work():
            return "second ok"

        t1 = Task(work=self_cancel_work, channel=channel, description="cancel me")
        t2 = Task(work=normal_work, channel=channel, description="runner-up")
        await queue.queue.put(t1)
        await queue.queue.put(t2)

        worker = asyncio.create_task(queue.run_worker(completions))
        await asyncio.wait_for(started.wait(), timeout=5)
        # t1 must survive cancellation AND t2 must still get processed —
        # only possible if the worker loop continued.
        result2 = await asyncio.wait_for(queue.queue.join(), timeout=5)
        assert result2 is None  # queue.join() returns only when all tasks done

        recorded = list(queue.completed)
        assert len(recorded) == 2, (
            "both the cancelled task and the follow-up task must be "
            f"recorded; got {recorded}"
        )
        ids = [tid for tid, _ in recorded]
        assert ids == [t1.task_id, t2.task_id]
        _, result1 = recorded[0]
        assert "cancel" in result1.lower()
        assert recorded[1][1] == "second ok"

        # Delivery: on_complete fired once per task, with the cancelled text.
        assert completions.await_count == 2, (
            "cancelled task must be delivered via on_complete exactly once "
            f"(#504-2), got {completions.await_count} calls"
        )
        first_call = completions.await_args_list[0]
        assert first_call.args[0] is t1
        assert "cancel" in first_call.args[1].lower()

        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

        lines = _read_jsonl(results_path)
        assert [ln["task_id"] for ln in lines] == [t1.task_id, t2.task_id]
        assert "cancel" in lines[0]["result"].lower()

    async def test_shutdown_notify_failure_does_not_mask_or_warn(self, caplog):
        """A raising detached on_complete on the shutdown path: the
        CancelledError still propagates, and the async exception is
        consumed and logged (no 'never retrieved' warning path)."""
        import logging

        caplog.set_level(logging.WARNING, logger="corvidae.task")
        queue = TaskQueue(max_workers=1)
        channel = _make_channel()
        started = asyncio.Event()
        done = asyncio.Event()

        async def long_work():
            started.set()
            await asyncio.sleep(3600)
            return "never"

        async def raising_notify(t, r):
            done.set()
            raise RuntimeError("notify exploded")

        await queue.queue.put(Task(work=long_work, channel=channel))
        worker = asyncio.create_task(queue.run_worker(raising_notify))
        await asyncio.wait_for(started.wait(), timeout=5)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        await asyncio.wait_for(done.wait(), timeout=2)
        for _ in range(20):
            if any("detached task-notify failed" in r.message for r in caplog.records):
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("raising detached notify must be logged by _notice_done")
        assert not queue._detached_notices or all(
            f.done() for f in queue._detached_notices
        )

    async def test_shutdown_cancel_still_propagates(self, tmp_path: Path):
        """Fix must not swallow shutdown: CancelledError still escapes
        _run_one_worker when the worker task itself is cancelled."""
        queue = TaskQueue(max_workers=1)
        channel = _make_channel()
        started = asyncio.Event()

        async def long_work():
            started.set()
            await asyncio.sleep(3600)
            return "never"

        await queue.queue.put(Task(work=long_work, channel=channel))
        worker = asyncio.create_task(queue.run_worker(AsyncMock()))
        await asyncio.wait_for(started.wait(), timeout=5)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
