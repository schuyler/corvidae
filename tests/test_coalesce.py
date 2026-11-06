"""Tests for coalescing a burst of queued messages into one turn (#398, R1-R9).

Messages that queue up on a channel while a turn is in flight are answered by
one model call that sees all of them. The tests hold a first turn open on a
fake LLM client (its first call awaits an Event), queue the burst behind it,
release the gate, and then read what the model was asked and what went out.
Nothing sleeps real time; the retry test moves a FakeClock.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest
from fake_clock import install_kernel_clock
from helpers import build_plugin_and_channel, drain, drain_task_queue, sent_texts
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)

from corvidae.agent import QueueItem, QueueItemRole, _TurnRetry
from corvidae.attribution import get_attribution
from corvidae.channel import ChannelConfig
from corvidae.hooks import hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

#: Handler type: call index -> the response dict to return, or an exception to raise.
Handler = Callable[[int], Any]


class FakeLLM:
    """A fake main-model client whose first call blocks on ``gate``.

    Records every prompt (deep-copied), the attribution trace id seen at call
    time, and appends ``("chat", index)`` to ``log`` when given one so tests
    can order model calls against hook events.
    """

    def __init__(self, handler: Handler | None = None, log: list | None = None) -> None:
        """Start with no calls; ``handler`` defaults to ``reply<index>`` text."""
        self.handler: Handler = handler or (lambda i: _make_text_response(f"reply{i}"))
        self.calls: list[list[dict]] = []
        self.traces: list[str | None] = []
        self.log = log
        # Set when the first call has begun; the test releases it via ``gate``.
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        """Record the call; the first one waits for the gate, then answer."""
        index = len(self.calls)
        self.calls.append(copy.deepcopy(list(messages)))
        self.traces.append(get_attribution().get("trace_id"))
        if self.log is not None:
            self.log.append(("chat", index))
        if index == 0:
            self.started.set()
            await self.gate.wait()
        out = self.handler(index)
        if isinstance(out, Exception):
            raise out
        return out


def users(prompt: list[dict]) -> list[str]:
    """The user-message contents of one recorded prompt, in order."""
    return [m["content"] for m in prompt if m.get("role") == "user"]


def prompt_text(prompt: list[dict]) -> str:
    """All message contents of one recorded prompt joined, for substring checks."""
    return "\n".join(str(m.get("content")) for m in prompt)


@pytest.fixture
async def make_env():
    """Factory for (plugin, channel, db, llm); tears every environment down."""
    made: list[tuple[Any, Any]] = []

    async def factory(
        handler: Handler | None = None,
        *,
        config: ChannelConfig | None = None,
        log: list | None = None,
    ):
        """Build an Agent on an in-memory DB wired to a gated FakeLLM."""
        plugin, channel, db = await build_plugin_and_channel(channel_config=config)
        llm = FakeLLM(handler, log)
        plugin._client = MagicMock()
        plugin._client.chat = llm.chat
        plugin._mark_started()
        plugin._tools = {}
        made.append((plugin, db))
        return plugin, channel, db, llm

    yield factory
    for plugin, db in made:
        for q in plugin.queues.values():
            await q.stop()
        task_plugin = plugin.pm.get_plugin("task")
        if task_plugin:
            await task_plugin.on_stop()
        await db.close()


async def send(plugin: Any, channel: Any, text: str) -> str:
    """Deliver one inbound user message and return the trace id it carries."""
    trace_id = mint_trace_id()
    await plugin.on_message(channel=channel, sender="user", text=text, trace_id=trace_id)
    return trace_id


async def start_inflight(plugin: Any, channel: Any, llm: FakeLLM) -> str:
    """Send "m0" and wait until its model call is in flight (blocked on the gate)."""
    trace_id = await send(plugin, channel, "m0")
    await asyncio.wait_for(llm.started.wait(), timeout=5)
    return trace_id


async def queue_burst(plugin: Any, channel: Any, texts: list[str]) -> list[str]:
    """Send each text while a turn is in flight; return their trace ids in order."""
    return [await send(plugin, channel, t) for t in texts]


async def settle(plugin: Any, channel: Any, *, tasks: bool = True) -> None:
    """Drain the channel queue (and the task queue unless ``tasks`` is False,
    for tests whose tools are deliberately still running) until quiet."""
    for _ in range(6):
        await drain(plugin, channel)
        if tasks:
            await drain_task_queue(plugin)
    await drain(plugin, channel)


async def run_burst(
    plugin: Any, channel: Any, llm: FakeLLM, texts: list[str], *, tasks: bool = True
) -> list[str]:
    """Hold m0 in flight, queue ``texts`` behind it, release, settle; return trace ids."""
    await start_inflight(plugin, channel, llm)
    traces = await queue_burst(plugin, channel, texts)
    llm.gate.set()
    await settle(plugin, channel, tasks=tasks)
    return traces


# ---------------------------------------------------------------------------
# R1, R3: one call, one reply
# ---------------------------------------------------------------------------


async def test_burst_gets_one_call_with_all_messages_in_order(make_env):
    """R1 (#398): three messages sent during an in-flight turn produce exactly
    one more model call, its prompt holding all three in arrival order."""
    plugin, channel, db, llm = await make_env()

    await run_burst(plugin, channel, llm, ["m1", "m2", "m3"])

    # In-flight call plus exactly one for the burst.
    assert len(llm.calls) == 2
    assert users(llm.calls[0]) == ["m0"]
    assert users(llm.calls[1]) == ["m0", "m1", "m2", "m3"]


async def test_inflight_reply_unchanged_then_one_burst_reply(make_env):
    """R3 (#398): the in-flight turn's reply goes out unchanged, then exactly
    one reply answers the burst."""
    plugin, channel, db, llm = await make_env()

    await run_burst(plugin, channel, llm, ["m1", "m2", "m3"])

    assert sent_texts(plugin.pm, OutboundMessage) == ["reply0", "reply1"]


# ---------------------------------------------------------------------------
# R2: history rows and persisted events
# ---------------------------------------------------------------------------


async def test_burst_rows_in_order_and_persisted_events_before_call(make_env):
    """R2 (#398): three history rows in arrival order; three on_message_persisted
    events in order, each with its own trace id and rowid, all before the
    burst's model call."""
    events: list[tuple] = []
    plugin, channel, db, llm = await make_env(log=events)
    persisted: list[tuple[str, int | None]] = []

    class Spy:
        """Records every persistence event against the model calls."""

        @hookimpl
        async def on_message_persisted(self, channel, trace_id, rowid, text, meta):
            persisted.append((trace_id, rowid))
            events.append(("persisted", text))

    plugin.pm.register(Spy(), name="spy")

    await start_inflight(plugin, channel, llm)
    traces = await queue_burst(plugin, channel, ["m1", "m2", "m3"])
    llm.gate.set()
    await settle(plugin, channel)

    # Persisted events for the burst all precede the burst's model call.
    assert events == [
        ("persisted", "m0"), ("chat", 0),
        ("persisted", "m1"), ("persisted", "m2"), ("persisted", "m3"),
        ("chat", 1),
    ]
    # Each event carries its own message's trace id and a distinct rowid.
    assert [t for t, _ in persisted[1:]] == traces
    rowids = [r for _, r in persisted[1:]]
    assert all(isinstance(r, int) for r in rowids)
    assert len(set(rowids)) == 3 and rowids == sorted(rowids)
    # Window rows and log rows both hold the messages in arrival order.
    window = [m["content"] for m in channel.conversation.messages if m.get("role") == "user"]
    assert window == ["m0", "m1", "m2", "m3"]
    async with db.execute(
        "SELECT message FROM message_log WHERE channel_id = ? ORDER BY id", (channel.id,)
    ) as cur:
        logged = [json.loads(r[0]) async for r in cur]
    assert [m["content"] for m in logged if m.get("role") == "user"] == [
        "m0", "m1", "m2", "m3",
    ]


# ---------------------------------------------------------------------------
# R4: only adjacent messages merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["notification", "tool_result"])
async def test_only_adjacent_messages_merge_around_a_notification(make_env, kind):
    """R4 (#398): m1, m2, <notification or tool result>, m3 queued during a turn
    make exactly three more calls, in order: [m1+m2], the notification, m3."""
    plugin, channel, db, llm = await make_env()
    await start_inflight(plugin, channel, llm)

    # Queue m1, m2, the non-user item, then m3, in that order.
    await send(plugin, channel, "m1")
    await send(plugin, channel, "m2")
    if kind == "notification":
        await plugin.on_notify(
            channel=channel, source="scheduler", text="NOTE-TEXT",
            tool_call_id=None, meta=None,
        )
    else:
        # An outstanding call whose result arrives behind the burst.
        channel.pending_tool_call_ids.add("call-x")
        await plugin.on_notify(
            channel=channel, source="task", text="NOTE-TEXT",
            tool_call_id="call-x", meta=None,
        )
    await send(plugin, channel, "m3")
    llm.gate.set()
    await settle(plugin, channel)

    # In-flight call + three more.
    assert len(llm.calls) == 4
    first, second, third = llm.calls[1], llm.calls[2], llm.calls[3]
    assert users(first) == ["m0", "m1", "m2"] and "NOTE-TEXT" not in prompt_text(first)
    assert "NOTE-TEXT" in prompt_text(second) and "m3" not in prompt_text(second)
    assert users(third)[-1] == "m3"


async def test_retry_item_is_never_merged_into_a_burst(make_env):
    """R4 (#398), retry variant: m1, m2, a retry item, m3 queued during a turn;
    [m1+m2] is one call, the retry is never merged into it and makes no call
    (a successful turn before it is dequeued cancels it), and m3 is answered
    in its own call: two calls after the in-flight one."""
    plugin, channel, _, llm = await make_env()
    owner = await start_inflight(plugin, channel, llm)

    await send(plugin, channel, "m1")
    await send(plugin, channel, "m2")
    # A retry copy of a failed user turn sits in the queue, registered as the
    # channel's pending retry, between the burst and m3.
    retry_item = QueueItem(
        role=QueueItemRole.USER, content="m0", channel=channel, sender="user",
        trace_id=owner, originating=False, retry=True,
    )
    plugin._turn_retries[channel.id] = _TurnRetry(retry_item, 0, MagicMock())
    await plugin._get_or_create_queue(channel).enqueue(retry_item)
    await send(plugin, channel, "m3")
    llm.gate.set()
    await settle(plugin, channel)

    # Two calls after the in-flight one: [m1+m2], then m3 on its own.
    assert len(llm.calls) == 3
    assert users(llm.calls[1]) == ["m0", "m1", "m2"]
    assert users(llm.calls[2]) == ["m0", "m1", "m2", "m3"]


# ---------------------------------------------------------------------------
# R5, R6: budgets
# ---------------------------------------------------------------------------


async def test_burst_gets_one_max_turns_budget(make_env):
    """R5 (#398): with max_turns 2 and a model that always returns a tool call,
    a three-message burst queued with no tool cycle in flight makes exactly
    three model calls in all (burst call, two tool-result calls, the last ending
    in the max-turns fallback), not a budget per message."""

    def handler(i: int) -> Any:
        """m0 answers in text; every later call asks for the tool again."""
        if i == 0:
            return _make_text_response("reply0")
        return _make_tool_call_response([_make_tool_call(f"c{i}", "echo", {"x": "v"})])

    plugin, channel, _, llm = await make_env(handler, config=ChannelConfig(max_turns=2))

    async def echo(x: str) -> str:
        """Return the argument."""
        return f"echo:{x}"

    plugin._tools = {"echo": echo}

    await run_burst(plugin, channel, llm, ["m1", "m2", "m3"])

    # Exclude the in-flight m0 call: the burst's whole exchange is three calls.
    assert len(llm.calls) - 1 == 3


async def test_burst_during_tool_cycle_is_charged_to_the_cycle(make_env):
    """R6 (#398): with two tool calls pending and turn_counter at k, a
    three-message burst makes one model call (not three), does not reset
    turn_counter, leaves the pending set unchanged, and the two results reach
    the model in one later call."""
    release_tools = asyncio.Event()

    def handler(i: int) -> Any:
        """m0 asks for two tools; later calls answer in text."""
        if i == 0:
            return _make_tool_call_response([
                _make_tool_call("c1", "slow_a", {"x": "1"}),
                _make_tool_call("c2", "slow_b", {"x": "2"}),
            ])
        return _make_text_response(f"reply{i}")

    plugin, channel, _, llm = await make_env(handler)

    async def slow_a(x: str) -> str:
        """Wait for the test, then return."""
        await release_tools.wait()
        return f"A-RESULT:{x}"

    async def slow_b(x: str) -> str:
        """Wait for the test, then return."""
        await release_tools.wait()
        return f"B-RESULT:{x}"

    plugin._tools = {"slow_a": slow_a, "slow_b": slow_b}

    # Tools stay blocked, so only the channel queue is drained here.
    await run_burst(plugin, channel, llm, ["m1", "m2", "m3"], tasks=False)

    # One call for the whole burst while both tools are still pending.
    assert len(llm.calls) == 2
    assert channel.pending_tool_call_ids == {"c1", "c2"}
    # k was 1 after the tool dispatch; the burst call is charged, never reset.
    assert channel.turn_counter == 2

    release_tools.set()
    await settle(plugin, channel)

    # Both results arrive together in a single later call.
    assert len(llm.calls) == 3
    # Background results carry task.py's "[Task <id>] " prefix; compare the rest.
    results = [
        m["content"].split("] ", 1)[1] for m in llm.calls[2] if m.get("role") == "tool"
    ]
    assert sorted(results) == ["A-RESULT:1", "B-RESULT:2"]


# ---------------------------------------------------------------------------
# R7: the last message's trace owns the turn
# ---------------------------------------------------------------------------


async def test_burst_turn_carries_the_last_trace_everywhere(make_env):
    """R7 (#398): every turn hook, the model call's attribution (what usage rows
    carry) and the response see the last message's trace id; none carries an
    earlier burst message's."""
    plugin, channel, _, llm = await make_env()
    before: list[str] = []
    gates: list[str] = []

    class Spy:
        """Records the trace id each before_agent_turn and send-gate call sees."""

        @hookimpl
        async def before_agent_turn(self, channel, trace_id, meta):
            before.append(trace_id)

        @hookimpl
        async def should_send_response(self, channel, text, emission, trace_id, meta):
            gates.append(trace_id)

    plugin.pm.register(Spy(), name="spy")

    traces = await run_burst(plugin, channel, llm, ["m1", "m2", "m3"])
    last = traces[-1]

    # Skip m0's own turn (index 0); everything after belongs to the burst.
    responses = [c.kwargs["trace_id"] for c in plugin.pm.ahook.on_agent_response.await_args_list]
    assert before[1:] == [last]
    assert llm.traces[1:] == [last]
    assert responses[1:] == [last]
    # The send gate sees the burst's turn under the last trace only.
    assert last in gates
    assert not set(gates) & set(traces[:-1])


# ---------------------------------------------------------------------------
# R8: idle accounting
# ---------------------------------------------------------------------------


async def test_drain_waits_for_burst_turn_and_channel_never_idle_early(make_env):
    """R8 (#398): with three messages queued behind an in-flight turn,
    SerialQueue.drain() returns after exactly two model calls, and no idle
    signal fires for the channel between the burst's dequeue and the end of
    its turn."""
    events: list[tuple] = []
    plugin, channel, _, llm = await make_env(log=events)

    class Recorder:
        """Stands in for IdleTracker: logs activity and settle signals."""

        def note_activity(self, channel_id: str) -> None:
            """Record that the channel had an inbound item."""
            events.append(("activity", channel_id))

        def note_settled(self, channel_id: str) -> None:
            """Record that the channel reported itself settled."""
            events.append(("settled", channel_id))

    plugin._idle = Recorder()

    await run_burst(plugin, channel, llm, ["m1", "m2", "m3"])

    # drain() returned only after the in-flight call and one burst call.
    assert len(llm.calls) == 2
    # The channel settles once, after the burst's model call, never before it.
    settled = [i for i, e in enumerate(events) if e[0] == "settled"]
    burst_call = events.index(("chat", 1))
    assert len(settled) == 1 and settled[0] > burst_call


# ---------------------------------------------------------------------------
# R9: retry of a burst turn
# ---------------------------------------------------------------------------


async def test_failed_burst_turn_retries_once_for_the_whole_burst(make_env):
    """R9 (#398): a burst turn that cannot reach the backend and is retried
    answers the whole burst once: three message rows, one assistant reply for
    the burst, no row persisted twice."""

    def handler(i: int) -> Any:
        """m0 answers; the burst call fails once, then the retry succeeds."""
        if i == 1:
            return aiohttp.ClientConnectionError("refused")
        if i == 2:
            return _make_text_response("burst reply")
        return _make_text_response(f"reply{i}")

    plugin, channel, db, llm = await make_env(handler)
    clock, _idle = install_kernel_clock(plugin.pm)

    await start_inflight(plugin, channel, llm)
    await queue_burst(plugin, channel, ["m1", "m2", "m3"])
    llm.gate.set()
    await settle(plugin, channel)
    # The failure armed a retry timer; fire it and let the retry run.
    await clock.advance(5)
    await settle(plugin, channel)

    # In-flight call, the failed burst call, and one successful retry.
    assert len(llm.calls) == 3
    assert users(llm.calls[2]) == ["m0", "m1", "m2", "m3"]
    async with db.execute(
        "SELECT message FROM message_log WHERE channel_id = ?", (channel.id,)
    ) as cur:
        rows = [json.loads(r[0]) async for r in cur]
    user_rows = [m["content"] for m in rows if m.get("role") == "user"]
    assert sorted(user_rows) == ["m0", "m1", "m2", "m3"]  # none persisted twice
    assert [m["content"] for m in rows if m.get("role") == "assistant"].count("burst reply") == 1
    assert sent_texts(plugin.pm, OutboundMessage)[-1] == "burst reply"


# ---------------------------------------------------------------------------
# #513: attribution once per sender, not once per message in the burst
# ---------------------------------------------------------------------------


async def test_a_burst_from_one_sender_carries_one_attribution_row(make_env):
    """A coalesced burst is one turn (#398), so the sender and channel are named
    once before it rather than repeated before every message (#513)."""
    plugin, channel, db, llm = await make_env()
    channel.name = "#home"
    channel.sender_names["user"] = "schuyler"

    await run_burst(plugin, channel, llm, ["m1", "m2", "m3"])

    burst_prompt = llm.calls[1]
    kinds = [
        "attribution"
        if m.get("content", "").startswith("[sender:")
        else "other"
        for m in burst_prompt
    ]
    attributions = [m for m, kind in zip(burst_prompt, kinds) if kind == "attribution"]

    # One row per turn: the in-flight turn's, and the burst's. What must not
    # happen is a row per message, or two rows stacked with nothing between.
    assert len(attributions) == 2, attributions
    assert all("schuyler" in row["content"] for row in attributions)
    assert not any(a == b == "attribution" for a, b in zip(kinds, kinds[1:]))

    # And the messages themselves are still the person's own words.
    assert users(burst_prompt) == ["m0", "m1", "m2", "m3"]
