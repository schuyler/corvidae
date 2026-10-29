"""Tests for the Agent's push of activity and settle events to the kernel
IdleTracker (plans/kernel-clock.md, R3 and R8).

Most tests swap the tracker for a recorder so they pin exactly when the
Agent reports activity and settling. Two use the real IdleTracker on a
FakeClock to show what the signals do (and do not do) downstream. Nothing
sleeps real time (R7): the only waits are ``asyncio.sleep(0)`` yields and
``FakeClock.advance``.
"""

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

import pytest

from corvidae.hooks import hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id
from fake_clock import install_kernel_clock
from helpers import build_plugin_and_channel, drain, drain_task_queue, sent_texts
from llm_response_fixtures import _make_text_response

REPO = Path(__file__).resolve().parent.parent


class _Recorder:
    """Stands in for IdleTracker: records the Agent's calls, in order."""

    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def note_activity(self, channel_id):
        self.events.append(("activity", channel_id))

    def note_settled(self, channel_id):
        self.events.append(("settled", channel_id))


async def _yield(n: int = 20) -> None:
    """Let ready tasks run without moving any clock."""
    for _ in range(n):
        await asyncio.sleep(0)


@pytest.fixture
async def env():
    """(plugin, channel, recorder) with a mock LLM that answers "ok"."""
    plugin, channel, db = await build_plugin_and_channel()
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("ok"))
    plugin._client = client
    plugin._tools = {}
    recorder = _Recorder()
    plugin._idle = recorder
    yield plugin, channel, recorder
    for q in plugin.queues.values():
        await q.stop()
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


async def test_admitted_message_marks_busy_then_settles_after_turn(env):
    """R3: an admitted message reports activity at once and settles only
    after its turn finishes."""
    plugin, channel, rec = env
    gate = asyncio.Event()

    async def slow_chat(*args, **kwargs):
        # Hold the turn open until the test releases it.
        await gate.wait()
        return _make_text_response("ok")

    plugin._client.chat = slow_chat

    await plugin.on_message(channel=channel, sender="u", text="hi", trace_id=mint_trace_id())
    await _yield()
    # Mid-turn: busy, not yet settled.
    assert rec.events == [("activity", channel.id)]

    gate.set()
    await drain(plugin, channel)
    assert rec.events == [("activity", channel.id), ("settled", channel.id)]


async def test_rejected_message_is_not_activity(env):
    """R3: a message the inbound gate rejects never reaches a queue, so it
    reports neither activity nor settle."""
    plugin, channel, rec = env

    class Reject:
        @hookimpl
        async def should_process_message(self, channel, sender, text):
            return False

    plugin.pm.register(Reject(), name="reject")

    await plugin.on_message(channel=channel, sender="u", text="no", trace_id=mint_trace_id())
    await _yield()

    assert rec.events == []
    assert plugin.queues == {}


@pytest.mark.parametrize(
    "source,tool_call_id",
    [("scheduler", None), ("task", "call-1")],
)
async def test_notification_counts_as_activity(env, source, tool_call_id):
    """R3: on_notify (scheduler firing, task completion) reports activity
    before enqueueing, and the channel settles after the item is processed."""
    plugin, channel, rec = env
    if tool_call_id:
        # A real tool result: the call is outstanding until it arrives.
        channel.pending_tool_call_ids.add(tool_call_id)

    await plugin.pm.ahook.on_notify(
        channel=channel, source=source, text="event",
        tool_call_id=tool_call_id, meta=None,
    )
    # Activity is pushed synchronously from on_notify.
    assert rec.events[0] == ("activity", channel.id)

    await drain(plugin, channel)
    assert rec.events == [("activity", channel.id), ("settled", channel.id)]


async def test_pending_tool_calls_keep_channel_busy_until_last_result(env):
    """R3: outstanding tool calls are in-flight work. The channel does not
    settle until the last result has been processed."""
    plugin, channel, rec = env
    channel.pending_tool_call_ids |= {"a", "b"}

    await plugin.pm.ahook.on_notify(
        channel=channel, source="task", text="ra", tool_call_id="a", meta=None,
    )
    await drain(plugin, channel)
    # "b" is still outstanding: the first result buffered, no settle.
    assert ("settled", channel.id) not in rec.events

    await plugin.pm.ahook.on_notify(
        channel=channel, source="task", text="rb", tool_call_id="b", meta=None,
    )
    await drain(plugin, channel)
    assert rec.events.count(("settled", channel.id)) == 1
    assert rec.events[-1] == ("settled", channel.id)


async def test_second_queued_item_defers_settle(env):
    """R3: while another item waits in the channel's queue, finishing the
    current one does not settle. Only the last item does."""
    plugin, channel, rec = env
    gate = asyncio.Event()

    async def slow_chat(*args, **kwargs):
        # The first turn blocks so the second message queues behind it.
        await gate.wait()
        return _make_text_response("ok")

    plugin._client.chat = slow_chat

    await plugin.on_message(channel=channel, sender="u", text="one", trace_id=mint_trace_id())
    await _yield()
    await plugin.on_message(channel=channel, sender="u", text="two", trace_id=mint_trace_id())
    gate.set()
    await drain(plugin, channel)

    assert rec.events == [
        ("activity", channel.id),
        ("activity", channel.id),
        ("settled", channel.id),
    ]


async def test_agent_start_resolves_idle_tracker_from_plugin_manager():
    """R3: Agent._start_plugin finds the tracker via pm.get_plugin("idle").

    No test hand-assigns ``plugin._idle`` here: the kernel clock and tracker
    are installed in the plugin manager before the Agent starts, and a message
    must reach the tracker as activity, which fails if the Agent never
    resolves it.
    """
    from corvidae.llm_plugin import LLMPlugin
    from corvidae.tool_collection import ToolCollectionPlugin

    plugin, channel, db = await build_plugin_and_channel()
    clock, tracker = install_kernel_clock(plugin.pm)
    # Stand-ins for the two plugins _start_plugin borrows from. They must
    # not be callable: apluggy's register() calls a callable plugin and
    # registers the result instead.
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("ok"))
    llm = NonCallableMagicMock(spec=LLMPlugin)
    llm.main_client = client
    tools = NonCallableMagicMock(spec=ToolCollectionPlugin)
    tools.get_tools.return_value = ({}, [])
    tools.max_result_chars = 100_000
    plugin.pm.register(llm, name="llm")
    plugin.pm.register(tools, name="tools")
    # Spy on the real tracker to see what the Agent pushes to it.
    activity = []
    real_note_activity = tracker.note_activity
    tracker.note_activity = lambda cid: (activity.append(cid), real_note_activity(cid))[1]
    try:
        await plugin._start_plugin({})
        await plugin.on_message(channel=channel, sender="u", text="hi", trace_id=mint_trace_id())
        await drain(plugin, channel)
        assert activity == [channel.id]
    finally:
        tracker.stop()
        await clock.close()
        for q in plugin.queues.values():
            await q.stop()
        await plugin.pm.get_plugin("task").on_stop()
        await db.close()


async def test_idle_signals_enqueue_no_turn():
    """R8: after channel and system idle signals fire, no queue holds an
    item, the task queue is idle, and no extra turn ran."""
    plugin, channel, db = await build_plugin_and_channel()
    clock, tracker = install_kernel_clock(plugin.pm)
    plugin._idle = tracker
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("ok"))
    plugin._client = client
    plugin._tools = {}
    calls = {"channel": 0, "system": 0}

    async def on_channel(cid):
        calls["channel"] += 1

    async def on_system():
        calls["system"] += 1

    tracker.watch_channels(10, on_channel, name="t-channel")
    tracker.watch_system(10, on_system, name="t-system")
    try:
        await plugin.on_message(channel=channel, sender="u", text="hi", trace_id=mint_trace_id())
        await drain(plugin, channel)
        await drain_task_queue(plugin)
        replies = list(sent_texts(plugin.pm, OutboundMessage))
        chat_calls = client.chat.await_count

        await clock.advance(30)

        # Both signals fired.
        assert calls["channel"] >= 1 and calls["system"] >= 1
        # Nothing was enqueued or started by them.
        assert all(q.is_empty for q in plugin.queues.values())
        task_queue = plugin.pm.get_plugin("task").task_queue
        assert task_queue.is_idle
        assert client.chat.await_count == chat_calls
        assert sent_texts(plugin.pm, OutboundMessage) == replies
    finally:
        tracker.stop()
        await clock.close()
        for q in plugin.queues.values():
            await q.stop()
        await plugin.pm.get_plugin("task").on_stop()
        await db.close()


def test_idle_modules_do_not_read_agent_queues():
    """R3 last clause: idle detection never reads the Agent's queues or the
    task queue. A static scan of the kernel idle/clock modules and the
    cognition memory plugin finds no reference to either."""
    paths = [
        REPO / "corvidae" / "idle.py",
        REPO / "corvidae" / "clock.py",
        REPO / "packages" / "corvidae-cognition" / "corvidae_cognition" / "memory.py",
    ]
    offenders = []
    for path in paths:
        # A missing module is a failure too: the scan must cover all three.
        assert path.exists(), f"{path} does not exist"
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in ("queues", "task_queue"):
                offenders.append(f"{path.name}:{node.lineno} .{node.attr}")
            if isinstance(node, ast.Name) and node.id == "task_queue":
                offenders.append(f"{path.name}:{node.lineno} {node.id}")
    assert not offenders, offenders
