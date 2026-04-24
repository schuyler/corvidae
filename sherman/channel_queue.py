"""Per-channel serial queue for agent loop invocations.

Each channel gets its own ChannelQueue instance, owned by AgentLoopPlugin
as _queues: dict[str, ChannelQueue]. The queue serializes all processing
(user messages and notifications) for a channel so concurrent arrivals
never race through the agent loop.

Logging:
    - ERROR: exceptions from process_fn (consumer continues after logging)
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from sherman.channel import Channel

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    """A single item in the per-channel processing queue.

    Attributes:
        role: "user" for inbound messages, "notification" for injected events.
        content: The text content to process.
        channel: The Channel this item belongs to (design review C2).
        sender: For user messages, the sender identity; None for notifications.
        source: For notifications, the origin (e.g. "background_task"); None for user messages.
        tool_call_id: For deferred tool results (background task completions).
        meta: Extensible metadata (task_id, etc.).
    """

    role: str
    content: str
    channel: Channel
    sender: str | None = None
    source: str | None = None
    tool_call_id: str | None = None
    meta: dict = field(default_factory=dict)  # Co1: no mutable default


class ChannelQueue:
    """Serial processing queue for a single channel.

    One instance per channel. Ensures that messages and notifications are
    processed one at a time, in enqueue order, preventing race conditions
    in the agent loop.

    Usage:
        q = ChannelQueue()
        q.start(process_fn)   # launch consumer; process_fn: (QueueItem) -> Awaitable[None]
        q.enqueue(item)       # add item to queue
        await q.drain()       # wait for all queued items to finish
        await q.stop()        # cancel consumer task
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None

    async def enqueue(self, item: QueueItem) -> None:
        """Add an item to the queue.

        Uses put_nowait internally (queue is unbounded), but exposed as async
        so callers have a consistent awaitable interface.
        """
        self._queue.put_nowait(item)

    def start(self, process_fn: Callable[[QueueItem], Awaitable[None]]) -> None:
        """Launch the consumer task.

        Args:
            process_fn: Async callable invoked for each dequeued item.
                        Signature: async def process_fn(item: QueueItem) -> None
        """
        self._consumer_task = asyncio.create_task(self._run(process_fn))

    async def _run(self, process_fn: Callable[[QueueItem], Awaitable[None]]) -> None:
        """Consumer loop — runs until cancelled.

        Calls task_done() in a finally block (design review C1) to ensure
        drain() never deadlocks, even if process_fn raises.
        """
        while True:
            item = await self._queue.get()
            try:
                await process_fn(item)
            except Exception:
                logger.exception(
                    "ChannelQueue consumer: process_fn raised for channel %s",
                    item.channel.id,
                )
            finally:
                self._queue.task_done()  # C1: always called, even on exception

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
