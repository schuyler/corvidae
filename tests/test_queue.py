"""Tests for corvidae.queue.SerialQueue and QueueItem.

Design requirements covered:
- C1: drain() must not deadlock when process_fn raises (task_done in finally)
- C2: QueueItem must carry a channel: Channel field
- C3: (covered in test_agent_loop_plugin.py)
"""

import asyncio
import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.queue import SerialQueue, QueueItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(scope: str = "test-scope") -> Channel:
    return Channel(transport="test", scope=scope, config=ChannelConfig())


# ---------------------------------------------------------------------------
# QueueItem structure
# ---------------------------------------------------------------------------


class TestQueueItem:
    def test_queue_item_has_channel_field(self):
        """QueueItem must carry a channel: Channel field (design review C2)."""
        ch = _make_channel()
        item = QueueItem(
            role="user",
            content="hello",
            channel=ch,
            sender="alice",
            source=None,
            tool_call_id=None,
        )
        assert item.channel is ch

    def test_queue_item_meta_default_is_empty_dict(self):
        """meta field must default to an empty dict, not a bare mutable default."""
        ch = _make_channel()
        item1 = QueueItem(role="user", content="a", channel=ch)
        item2 = QueueItem(role="user", content="b", channel=ch)
        # Each instance must get its own dict (no shared mutable default)
        assert item1.meta == {}
        assert item2.meta == {}
        assert item1.meta is not item2.meta

    def test_queue_item_meta_explicit(self):
        """meta can be set at construction time."""
        ch = _make_channel()
        item = QueueItem(role="user", content="x", channel=ch, meta={"task_id": "abc"})
        assert item.meta["task_id"] == "abc"


# ---------------------------------------------------------------------------
# Serialization: items processed in enqueue order
# ---------------------------------------------------------------------------


class TestSerialization:
    async def test_items_processed_in_order(self):
        """Enqueue two items; assert they are processed in FIFO order."""
        processed: list[str] = []

        async def collect(item: QueueItem) -> None:
            processed.append(item.content)

        ch = _make_channel()
        q = SerialQueue()
        q.start(collect)

        await q.enqueue(QueueItem(role="user", content="first", channel=ch))
        await q.enqueue(QueueItem(role="user", content="second", channel=ch))
        await q.drain()

        assert processed == ["first", "second"]

        await q.stop()

    async def test_multiple_items_all_processed(self):
        """All enqueued items are eventually processed."""
        processed: list[str] = []

        async def collect(item: QueueItem) -> None:
            processed.append(item.content)

        ch = _make_channel()
        q = SerialQueue()
        q.start(collect)

        for i in range(5):
            await q.enqueue(QueueItem(role="user", content=str(i), channel=ch))
        await q.drain()

        assert processed == ["0", "1", "2", "3", "4"]

        await q.stop()


# ---------------------------------------------------------------------------
# drain() — blocks until processing complete
# ---------------------------------------------------------------------------


class TestDrain:
    async def test_drain_blocks_until_processed(self):
        """drain() returns only after all enqueued items are processed."""
        barrier = asyncio.Event()
        processed: list[str] = []

        async def slow_collect(item: QueueItem) -> None:
            await barrier.wait()
            processed.append(item.content)

        ch = _make_channel()
        q = SerialQueue()
        q.start(slow_collect)

        await q.enqueue(QueueItem(role="user", content="blocked", channel=ch))

        # drain should not complete while barrier is unset
        drain_task = asyncio.create_task(q.drain())
        await asyncio.sleep(0)  # yield so consumer can start processing

        assert not drain_task.done(), "drain() completed before barrier was set"

        barrier.set()
        await drain_task

        assert processed == ["blocked"]

        await q.stop()


# ---------------------------------------------------------------------------
# drain() with error — must not deadlock (C1)
# ---------------------------------------------------------------------------


class TestDrainWithError:
    async def test_drain_returns_even_when_process_fn_raises(self):
        """If process_fn raises, drain() must still return.

        This verifies that the consumer calls task_done() in a finally block
        (design review C1). If it doesn't, queue.join() deadlocks and drain()
        hangs forever.
        """

        async def raising_fn(item: QueueItem) -> None:
            raise RuntimeError("deliberate failure")

        ch = _make_channel()
        q = SerialQueue()
        q.start(raising_fn)

        await q.enqueue(QueueItem(role="user", content="boom", channel=ch))

        # Must complete without hanging — if task_done() is missing from
        # finally, this will time out with asyncio.TimeoutError.
        try:
            await asyncio.wait_for(q.drain(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("drain() deadlocked after process_fn raised (task_done() missing from finally)")

        await q.stop()

    async def test_consumer_continues_after_error(self):
        """After an item raises, the consumer continues processing the next item."""
        processed: list[str] = []

        async def sometimes_raise(item: QueueItem) -> None:
            if item.content == "fail":
                raise RuntimeError("deliberate failure")
            processed.append(item.content)

        ch = _make_channel()
        q = SerialQueue()
        q.start(sometimes_raise)

        await q.enqueue(QueueItem(role="user", content="fail", channel=ch))
        await q.enqueue(QueueItem(role="user", content="ok", channel=ch))
        await q.drain()

        assert processed == ["ok"], "Consumer must continue after process_fn raises"

        await q.stop()


# ---------------------------------------------------------------------------
# stop() — cancels consumer
# ---------------------------------------------------------------------------


class TestStop:
    async def test_stop_cancels_consumer(self):
        """After stop(), the consumer task is cancelled."""
        processed: list[str] = []

        async def collect(item: QueueItem) -> None:
            processed.append(item.content)

        ch = _make_channel()
        q = SerialQueue()
        q.start(collect)
        await q.stop()

        # Consumer should no longer be running
        assert q._consumer_task is None or q._consumer_task.done()

    async def test_new_enqueues_not_processed_after_stop(self):
        """After stop(), newly enqueued items are not processed."""
        processed: list[str] = []

        async def collect(item: QueueItem) -> None:
            processed.append(item.content)

        ch = _make_channel()
        q = SerialQueue()
        q.start(collect)
        await q.stop()

        # Enqueue after stop — should not be processed
        await q.enqueue(QueueItem(role="user", content="after-stop", channel=ch))
        await asyncio.sleep(0.05)  # give any lingering consumer a chance to run

        assert "after-stop" not in processed


# ---------------------------------------------------------------------------
# RED PHASE: QueueItemRole enum (architecture critique)
# ---------------------------------------------------------------------------


class TestQueueItemRole:
    def test_queue_item_role_enum_exists(self):
        """QueueItemRole enum must be importable from corvidae.agent."""
        from corvidae.agent import QueueItemRole
        assert QueueItemRole is not None

    def test_queue_item_role_has_user_member(self):
        """QueueItemRole must have a USER member."""
        from corvidae.agent import QueueItemRole
        assert hasattr(QueueItemRole, "USER")

    def test_queue_item_role_has_notification_member(self):
        """QueueItemRole must have a NOTIFICATION member."""
        from corvidae.agent import QueueItemRole
        assert hasattr(QueueItemRole, "NOTIFICATION")

    def test_queue_item_role_is_enum(self):
        """QueueItemRole must be an enum (not just a class with attributes)."""
        import enum
        from corvidae.agent import QueueItemRole
        assert issubclass(QueueItemRole, enum.Enum)

    async def test_on_message_creates_item_with_user_role(self):
        """QueueItem created via on_message must use QueueItemRole.USER."""
        from corvidae.agent import AgentPlugin, QueueItemRole
        from corvidae.hooks import create_plugin_manager
        from corvidae.channel import ChannelRegistry, Channel, ChannelConfig
        from corvidae.queue import SerialQueue

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "", "max_context_tokens": 8000, "keep_thinking_in_history": False})
        pm.register(registry, name="registry")
        plugin = AgentPlugin.__new__(AgentPlugin)
        plugin.pm = pm
        plugin.client = object()  # non-None so on_message proceeds
        plugin.queues = {}
        plugin._registry = registry

        captured_items = []

        async def fake_process(item):
            captured_items.append(item)

        ch = Channel(transport="test", scope="s1", config=ChannelConfig())

        q = SerialQueue()
        q.start(fake_process)
        plugin.queues[ch.id] = q

        await plugin.on_message(channel=ch, sender="alice", text="hello")
        await q.drain()

        assert len(captured_items) == 1
        assert captured_items[0].role == QueueItemRole.USER

    async def test_on_notify_creates_item_with_notification_role(self):
        """QueueItem created via on_notify must use QueueItemRole.NOTIFICATION."""
        from corvidae.agent import AgentPlugin, QueueItemRole
        from corvidae.hooks import create_plugin_manager
        from corvidae.channel import ChannelRegistry, Channel, ChannelConfig
        from corvidae.queue import SerialQueue

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "", "max_context_tokens": 8000, "keep_thinking_in_history": False})
        pm.register(registry, name="registry")
        plugin = AgentPlugin.__new__(AgentPlugin)
        plugin.pm = pm
        plugin.client = object()
        plugin.queues = {}
        plugin._registry = registry

        captured_items = []

        async def fake_process(item):
            captured_items.append(item)

        ch = Channel(transport="test", scope="s2", config=ChannelConfig())

        q = SerialQueue()
        q.start(fake_process)
        plugin.queues[ch.id] = q

        await plugin.on_notify(channel=ch, source="task", text="done", tool_call_id=None, meta=None)
        await q.drain()

        assert len(captured_items) == 1
        assert captured_items[0].role == QueueItemRole.NOTIFICATION
