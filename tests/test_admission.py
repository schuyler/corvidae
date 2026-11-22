"""Tests for the admission seam and waiting-work depth (issue #1050, phase 1).

Kernel tests: nothing here imports corvidae_core_periphery.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from helpers import (
    KERNEL_LLM_STAGES,
    build_plugin_and_channel,
    drain,
    drain_task_queue,
    stub_llm_config,
)

from corvidae.agent import QueueItem, QueueItemRole
from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.llm_plugin import LLMPlugin
from corvidae.queue import SerialQueue
from corvidae.task import Task, TaskQueue
from corvidae.tool import ToolContext, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _channel(transport="test", scope="scope1") -> Channel:
    """A bare channel for offers that never touch a registry."""
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


async def _offer(pm, channel, kind="task", source_channel="test:src"):
    """Call offer_work with fixed trace/annotations; return its result."""
    from corvidae.admission import offer_work

    return await offer_work(
        pm,
        channel=channel,
        kind=kind,
        source_channel=source_channel,
        trace_id="trace-1",
        annotations={},
    )


class _Gate:
    """Plugin whose should_start_work records each offer and answers ``refusal``."""

    def __init__(self, refusal=None):
        self.refusal = refusal
        self.offers: list[dict] = []

    @hookimpl
    async def should_start_work(
        self, channel, kind, source_channel, trace_id, annotations
    ):
        # Record what the kernel offered, then vote.
        self.offers.append(
            {
                "channel": channel,
                "kind": kind,
                "source_channel": source_channel,
                "trace_id": trace_id,
                "annotations": annotations,
            }
        )
        return self.refusal


class _Raiser:
    """Plugin whose should_start_work always raises."""

    @hookimpl
    async def should_start_work(
        self, channel, kind, source_channel, trace_id, annotations
    ):
        raise RuntimeError("gate exploded")


class _SyncGate:
    """Plugin with a non-async should_start_work that refuses.

    Covers the *dispatcher's* tolerance only: startup now refuses a sync
    implementation of an async hook (#1096), so this plugin cannot be loaded by
    the daemon. The test that uses it registers it on a manager directly to show
    the dispatch path still handles one it is handed.
    """

    @hookimpl
    def should_start_work(self, channel, kind, source_channel, trace_id, annotations):
        from corvidae.admission import WorkRefusal

        return WorkRefusal("limits.max_waiting", "wait")


def _refusal():
    """The refusal every refusing gate in these tests returns."""
    from corvidae.admission import WorkRefusal

    return WorkRefusal("limits.max_waiting", "wait")


async def _until(predicate, timeout=5.0):
    """Yield to the loop until ``predicate()`` holds, bounded by ``timeout``."""

    async def _spin():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(_spin(), timeout)


def _user_item(channel, text="m") -> QueueItem:
    """A USER queue item for ``channel``."""
    return QueueItem(role=QueueItemRole.USER, content=text, channel=channel, sender="u")


def _note_item(channel, text="n") -> QueueItem:
    """A NOTIFICATION queue item for ``channel``."""
    return QueueItem(
        role=QueueItemRole.NOTIFICATION, content=text, channel=channel, source="t"
    )


# ---------------------------------------------------------------------------
# offer_work and the hookspec
# ---------------------------------------------------------------------------


class TestOfferWork:
    async def test_hookspec_exists(self):
        """should_start_work is declared in the AgentSpec (K1 seam)."""
        pm = create_plugin_manager()
        assert pm.hook.should_start_work.spec is not None

    async def test_no_impls_proceeds(self):
        """With no implementation registered, offer_work returns None."""
        pm = create_plugin_manager()
        assert await _offer(pm, _channel()) is None

    async def test_refusal_returns_text_and_warns_with_fields(self, caplog):
        """A refusal returns the exact tool text and logs one WARNING with
        kind, channel, source_channel and limit as record attributes."""
        pm = create_plugin_manager()
        pm.register(_Gate(_refusal()), name="gate")
        target = _channel(scope="target")
        with caplog.at_level(logging.WARNING):
            text = await _offer(pm, target, kind="note", source_channel="test:src")
        assert text == "Not started: limits.max_waiting reached — wait"
        records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and hasattr(r, "limit")
        ]
        assert len(records) == 1
        rec = records[0]
        assert rec.kind == "note"
        assert rec.channel == target.id
        assert rec.source_channel == "test:src"
        assert rec.limit == "limits.max_waiting"

    async def test_raising_impl_does_not_hide_sibling_refusal(self, caplog):
        """An implementation that raises is logged (ERROR with traceback) and
        a sibling's refusal is still returned."""
        pm = create_plugin_manager()
        # Pluggy calls the last-registered impl first: the raiser runs
        # before the gate, so the gate's vote must survive the raise.
        pm.register(_Gate(_refusal()), name="gate")
        pm.register(_Raiser(), name="raiser")
        with caplog.at_level(logging.WARNING):
            text = await _offer(pm, _channel())
        assert text == "Not started: limits.max_waiting reached — wait"
        errors = [
            r for r in caplog.records if r.levelno == logging.ERROR and r.exc_info
        ]
        assert errors, "the raising implementation must be logged with exc_info"

    async def test_raising_impl_alone_proceeds(self, caplog):
        """A raising implementation with no refusing sibling is no opinion."""
        pm = create_plugin_manager()
        pm.register(_Raiser(), name="raiser")
        with caplog.at_level(logging.WARNING):
            assert await _offer(pm, _channel()) is None

    async def test_sync_impl_refusal_is_honoured(self):
        """A non-async implementation's WorkRefusal counts."""
        pm = create_plugin_manager()
        pm.register(_SyncGate(), name="sync_gate")
        text = await _offer(pm, _channel())
        assert text == "Not started: limits.max_waiting reached — wait"


# ---------------------------------------------------------------------------
# Depth: waiting_work / waiting_messages / pending_items / pending_by_channel
# ---------------------------------------------------------------------------


class TestWaitingWork:
    async def _blocked_agent(self, monkeypatch):
        """Agent whose item processing blocks on ``release``.

        Returns (plugin, channel, db, release, started) where ``started``
        is set once an item is in flight.
        """
        plugin, channel, db = await build_plugin_and_channel()
        release = asyncio.Event()
        started = asyncio.Event()

        async def blocked(item):
            # Hold the item in flight until the test releases it.
            started.set()
            await release.wait()

        monkeypatch.setattr(plugin, "_process_queue_item", blocked)
        return plugin, channel, db, release, started

    async def _teardown(self, plugin, db, release):
        """Release blocked work, stop queues and the task plugin, close the db."""
        release.set()
        for q in plugin.queues.values():
            await q.stop()
        await plugin.pm.get_plugin("task").on_stop()
        await db.close()

    async def test_empty_at_rest(self):
        """With nothing queued, waiting_work() is {}."""
        plugin, channel, db = await build_plugin_and_channel()
        try:
            assert plugin.waiting_work() == {}
        finally:
            await plugin.pm.get_plugin("task").on_stop()
            await db.close()

    async def test_counts_in_flight_and_queued_user_items(self, monkeypatch):
        """The in-flight item and a USER item queued behind it both count,
        and the count returns to {} once both finish."""
        plugin, channel, db, release, started = await self._blocked_agent(monkeypatch)
        try:
            q = plugin._get_or_create_queue(channel)
            await q.enqueue(_user_item(channel, "one"))
            await asyncio.wait_for(started.wait(), 5)
            assert plugin.waiting_work() == {channel.id: 1}
            await q.enqueue(_user_item(channel, "two"))
            assert plugin.waiting_work() == {channel.id: 2}
            release.set()
            await asyncio.wait_for(drain(plugin, channel), 5)
            assert plugin.waiting_work() == {}
        finally:
            await self._teardown(plugin, db, release)

    async def test_counts_tasks_queued_and_running(self, monkeypatch):
        """TaskQueue tasks, running (4 workers) and queued behind them, are
        counted against task.channel.id and vanish when they finish."""
        plugin, channel, db = await build_plugin_and_channel()
        release = asyncio.Event()
        running = []

        async def instant(item):
            # Completion notifications are not the subject here.
            return None

        monkeypatch.setattr(plugin, "_process_queue_item", instant)
        task_queue = plugin.pm.get_plugin("task").task_queue

        async def blocked_work():
            # Park the worker until released.
            running.append(1)
            await release.wait()
            return "done"

        try:
            for i in range(6):
                await task_queue.enqueue(
                    Task(work=blocked_work, channel=channel, description=f"t{i}")
                )
            # 4 workers running, 2 still queued.
            await _until(lambda: len(running) == 4)
            assert plugin.waiting_work() == {channel.id: 6}
            release.set()
            await asyncio.wait_for(drain_task_queue(plugin), 5)
            await asyncio.wait_for(drain(plugin, channel), 5)
            assert plugin.waiting_work() == {}
        finally:
            release.set()
            for q in plugin.queues.values():
                await q.stop()
            await plugin.pm.get_plugin("task").on_stop()
            await db.close()

    async def test_waiting_messages_counts_user_items_only(self, monkeypatch):
        """A NOTIFICATION item is in waiting_work but not in waiting_messages."""
        plugin, channel, db, release, started = await self._blocked_agent(monkeypatch)
        try:
            q = plugin._get_or_create_queue(channel)
            await q.enqueue(_user_item(channel, "one"))
            await asyncio.wait_for(started.wait(), 5)
            await q.enqueue(_note_item(channel))
            assert plugin.waiting_work() == {channel.id: 2}
            assert plugin.waiting_messages() == {channel.id: 1}
        finally:
            await self._teardown(plugin, db, release)

    async def test_coalesced_items_still_count_until_turn_ends(self):
        """Items a turn absorbed with take_while stay in pending_items until
        the owning turn finishes."""
        channel = _channel()
        release = asyncio.Event()
        taken_ready = asyncio.Event()
        q = SerialQueue()

        async def process(item):
            # Absorb everything queued behind the owner, then park.
            q.take_while(lambda i: True)
            taken_ready.set()
            await release.wait()

        q.start(process)
        try:
            for n in range(3):
                await q.enqueue(_user_item(channel, f"m{n}"))
            await asyncio.wait_for(taken_ready.wait(), 5)
            assert len(q.pending_items()) == 3
            release.set()
            await asyncio.wait_for(q.drain(), 5)
            assert len(q.pending_items()) == 0
        finally:
            release.set()
            await q.stop()


# ---------------------------------------------------------------------------
# Callers: subagent and fork
# ---------------------------------------------------------------------------

_LLM_CONFIG = {
    "llm": {
        "providers": {"main": {"base_url": "http://localhost:8080"}},
        "profiles": {"main": {"model": "test-model"}},
    },
}


def _base_pm(stage: str):
    """A pm with registry, a stubbed LLMPlugin and a ToolCollectionPlugin."""
    from corvidae.tool_collection import ToolCollectionPlugin

    pm = create_plugin_manager()
    pm.register(ChannelRegistry({}), name="registry")
    llm = LLMPlugin(pm)
    llm._stages = dict(KERNEL_LLM_STAGES)
    llm._stages[stage] = "chat"
    llm._clients["main"] = MagicMock(name="main")
    stub_llm_config(llm)
    pm.register(llm, name="llm")
    tools = ToolCollectionPlugin(pm)
    tools.registry = ToolRegistry()
    pm.register(tools, name="tools")
    return pm


def _ctx_with_queue(channel):
    """A ToolContext over a mock TaskQueue; returns (ctx, task_queue)."""
    task_queue = MagicMock(spec=TaskQueue)
    task_queue.enqueue = AsyncMock()
    return (
        ToolContext(channel=channel, tool_call_id="call-1", task_queue=task_queue),
        task_queue,
    )


class TestCallersOffer:
    async def test_subagent_refused_returns_text_and_enqueues_nothing(self):
        """A refusal is subagent's result; nothing is enqueued; the offer is
        kind "task" on the calling channel."""
        from corvidae.tools.subagent import SubagentPlugin

        pm = _base_pm("subagent")
        plugin = SubagentPlugin(pm)
        await plugin.on_start(config=_LLM_CONFIG)
        gate = _Gate(_refusal())
        pm.register(gate, name="gate")
        channel = _channel()
        ctx, task_queue = _ctx_with_queue(channel)

        result = await plugin._launch("do it", "desc", ctx)

        assert result == "Not started: limits.max_waiting reached — wait"
        task_queue.enqueue.assert_not_awaited()
        assert [(o["kind"], o["channel"]) for o in gate.offers] == [("task", channel)]

    async def test_fork_refused_returns_text_and_enqueues_nothing(self):
        """A refusal is fork's result; nothing is enqueued; the offer is
        kind "fork" on the calling channel."""
        from corvidae.agent import Agent
        from corvidae.tools.fork_tool import ForkToolPlugin

        pm = _base_pm("fork")
        mock_agent = Agent.__new__(Agent)
        mock_agent.channel_system_prompt = AsyncMock(return_value="I am Lex.")
        pm.register(mock_agent, name="agent")
        plugin = ForkToolPlugin(pm)
        await plugin.on_start(config=_LLM_CONFIG)
        gate = _Gate(_refusal())
        pm.register(gate, name="gate")
        channel = _channel()
        ctx, task_queue = _ctx_with_queue(channel)

        result = await plugin._launch("review this", ctx)

        assert result == "Not started: limits.max_waiting reached — wait"
        task_queue.enqueue.assert_not_awaited()
        assert [(o["kind"], o["channel"]) for o in gate.offers] == [("fork", channel)]
