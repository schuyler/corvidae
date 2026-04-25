"""Tests for SerialQueue.is_empty property (red phase)."""

import asyncio
import pytest

from sherman.queue import SerialQueue


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
