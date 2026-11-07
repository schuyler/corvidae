"""Tests for SerialQueue.is_empty property (red phase)."""

import asyncio
import pytest

from corvidae.queue import SerialQueue


# ---------------------------------------------------------------------------
# SerialQueue.is_empty (red phase)
# ---------------------------------------------------------------------------


async def test_serial_queue_is_empty_when_new():
    """A freshly created SerialQueue has no items and is_empty returns True."""
    q = SerialQueue()
    assert q.is_empty is True


async def test_serial_queue_is_empty_returns_false_after_enqueue():
    """is_empty returns False once an item is enqueued but not yet consumed."""
    async def never_finish(item):
        # Simulate a long-running consumer so the item stays in queue.
        # We cancel the task immediately after checking, so just sleep briefly.
        await asyncio.sleep(10)

    q = SerialQueue()

    # Enqueue without a running consumer so item stays in the queue.
    # (put_nowait adds directly to the internal queue)
    q._queue.put_nowait("pending_item")

    assert q.is_empty is False


async def test_serial_queue_is_empty_true_after_item_consumed():
    """is_empty returns True after all enqueued items have been processed."""
    processed = asyncio.Event()

    async def process(item):
        processed.set()

    q = SerialQueue()
    q.start(process)

    await q.enqueue("item")
    await q.drain()

    assert q.is_empty is True

    await q.stop()


async def test_serial_queue_is_empty_false_with_multiple_items():
    """is_empty returns False when multiple items are pending."""
    q = SerialQueue()

    q._queue.put_nowait("item1")
    q._queue.put_nowait("item2")
    q._queue.put_nowait("item3")

    assert q.is_empty is False


# ---------------------------------------------------------------------------
# SerialQueue.take_while (#398 coalescing)
# ---------------------------------------------------------------------------


async def test_take_while_takes_head_run_and_leaves_rest_in_order():
    """R4 (#398): take_while removes the head items while the predicate holds,
    stops at the first failure, and leaves that item and everything after it
    queued in their original order."""
    q = SerialQueue()
    # No consumer: items stay queued so the test controls the queue.
    for item in ["u1", "u2", "n1", "u3"]:
        q._queue.put_nowait(item)

    taken = q.take_while(lambda i: i.startswith("u"))

    # Only the leading run is taken; "u3" sits behind "n1" and stays put.
    assert taken == ["u1", "u2"]
    assert [q._queue.get_nowait() for _ in range(q._queue.qsize())] == ["n1", "u3"]


async def test_take_while_returns_empty_when_head_fails_predicate():
    """R4 (#398): a head item failing the predicate takes nothing and stays."""
    q = SerialQueue()
    q._queue.put_nowait("n1")
    q._queue.put_nowait("u1")

    assert q.take_while(lambda i: i.startswith("u")) == []
    assert q._queue.qsize() == 2


async def test_take_while_on_empty_queue_returns_empty():
    """R1 (#398): an empty queue yields an empty list (no burst)."""
    q = SerialQueue()
    assert q.take_while(lambda i: True) == []


async def test_take_while_keeps_drain_waiting_for_the_current_item():
    """R8 (#398): items taken inside the consumer's current item are marked
    done at once, yet drain() still waits for the current item to finish."""
    in_process = asyncio.Event()
    release = asyncio.Event()
    taken_holder: list[list[str]] = []

    async def process(item):
        # The consumer takes the rest of the queue mid-item, then blocks.
        taken_holder.append(q.take_while(lambda i: True))
        in_process.set()
        await release.wait()

    q = SerialQueue()
    q.start(process)
    for item in ["a", "b", "c"]:
        await q.enqueue(item)
    # Bounded wait: a missing take_while must fail the test, not hang it.
    await asyncio.wait_for(in_process.wait(), timeout=2)
    assert taken_holder == [["b", "c"]]

    # drain() must not complete while the consumer still holds "a".
    drain_task = asyncio.create_task(q.drain())
    # A few deterministic yields give drain() every chance to finish early.
    for _ in range(5):
        await asyncio.sleep(0)
    assert not drain_task.done()

    release.set()
    await asyncio.wait_for(drain_task, timeout=2)
    await q.stop()


async def test_consumer_runs_with_an_empty_attribution_whoever_started_it():
    """A queue first fed inside another channel's turn doesn't inherit its attribution (#548)."""
    from corvidae.attribution import get_attribution, reset_attribution, set_attribution

    seen = []
    async def process(item):
        seen.append(dict(get_attribution()))

    q = SerialQueue()
    token = set_attribution(channel_id="signal:a", trace_id="t-9")
    try:
        q.start(process)
    finally:
        reset_attribution(token)
    await q.enqueue("item")
    await q.drain()
    await q.stop()
    assert seen == [{}]
