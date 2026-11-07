"""Per-channel serial queue for agent loop invocations.

Each channel gets its own SerialQueue instance, owned by Agent
as queues: dict[str, SerialQueue]. The queue serializes all processing
(user messages and notifications) for a channel so concurrent arrivals
never race through the agent loop.

Logging:
    - ERROR: exceptions from process_fn (consumer continues after logging)

Note: QueueItem is defined in corvidae.agent and re-exported from here
for backward compatibility.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.agent import QueueItem

logger = logging.getLogger(__name__)


class SerialQueue:
    """Serial processing queue for a single channel.

    One instance per channel. Ensures that messages and notifications are
    processed one at a time, in enqueue order, preventing race conditions
    in the agent loop.

    Usage:
        q = SerialQueue()
        q.start(process_fn)   # launch consumer; process_fn: (item) -> Awaitable[None]
        await q.enqueue(item) # add item to queue
        await q.drain()       # wait for all queued items to finish
        await q.stop()        # cancel consumer task
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None

    @property
    def is_empty(self) -> bool:
        """True when the queue has no pending items.

        Note: This does not account for an item currently being processed.
        A queue with is_empty=True may still have a consumer actively
        processing the last dequeued item.
        """
        return self._queue.empty()

    async def enqueue(self, item) -> None:
        """Add an item to the queue.

        Uses put_nowait internally (queue is unbounded), but exposed as async
        so callers have a consistent awaitable interface.
        """
        self._queue.put_nowait(item)

    def take_while(self, predicate: Callable[[QueueItem], bool]) -> list[QueueItem]:
        """Remove and return the head items for as long as ``predicate`` holds.

        Stops at the first item that fails the predicate; that item stays
        queued, unmoved. Synchronous (no awaits), so nothing can interleave
        between a peek and its take. Each taken item is marked done at once:
        the consumer's current item is not, so drain() still waits for the
        turn that owns the taken items to finish.

        Args:
            predicate: Called with the item at the head of the queue.

        Returns:
            The taken items, in queue order (empty when the head fails).
        """
        taken: list[QueueItem] = []
        # Peek at the head without removing it; stop at the first failure.
        # shortcut: relies on a CPython-private attribute (asyncio.Queue._queue,
        # the deque its LifoQueue/PriorityQueue subclasses override); replace
        # the asyncio.Queue with an owned deque + Event if it ever moves.
        while self._queue._queue and predicate(self._queue._queue[0]):
            # get_nowait keeps asyncio's own bookkeeping consistent.
            taken.append(self._queue.get_nowait())
            self._queue.task_done()
        return taken

    def start(self, process_fn: Callable) -> None:
        """Launch the consumer task.

        Args:
            process_fn: Async callable invoked for each dequeued item.
        """
        # An empty context: a queue is first fed by whichever code enqueues
        # first, which can be a tool Task inside another channel's turn, and
        # its items must not inherit that turn's attribution.
        self._consumer_task = asyncio.create_task(
            self._run(process_fn), context=contextvars.Context()
        )

    async def _run(self, process_fn: Callable) -> None:
        """Consumer loop — runs until cancelled.

        Calls task_done() in a finally block to ensure drain() never deadlocks,
        even if process_fn raises.
        """
        while True:
            item = await self._queue.get()
            try:
                await process_fn(item)
            except Exception:
                logger.exception(
                    "SerialQueue consumer: process_fn raised for channel %s",
                    item.channel.id,
                )
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        """Wait until all currently queued items have been processed."""
        await self._queue.join()

    async def stop(self) -> None:
        """Cancel the consumer task and wait for it to finish."""
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        self._consumer_task = None


# Backward-compat re-export: QueueItem is now defined in corvidae.agent.
# Lazy import via __getattr__ avoids circular import (agent.py imports SerialQueue
# from this module). Type checkers may not resolve this — the TYPE_CHECKING import
# above handles static analysis.
def __getattr__(name: str):
    if name == "QueueItem":
        from corvidae.agent import QueueItem
        return QueueItem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
