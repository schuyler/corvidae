"""Leases (#364, #367) through the real path.

A real ``Agent`` with its ``SerialQueue``s, ``TaskPlugin``'s ``TaskQueue`` and a
real ``LLMPlugin`` whose provider declares ``heads`` in
``llm.providers.main.heads``. Only the aiohttp session is faked: every POST is
recorded with the lease key its call carried and parked on an
``asyncio.Event`` the test sets, so the test controls the order model calls
leave in. Ordering is asserted from the recorded start order and from
``LeaseManager.snapshot``, never from sleeps. Covers R1, R5, R6, R7 and R13 of
.claude/plans/364-sticky-leases.md; the manager's own rules are unit tested in
tests/test_lease.py.

Kernel tests must not import corvidae_cognition: the "silent task" of the R6
case is enqueued by a small test plugin from ``on_agent_response``, as
critique would.

Names that do not exist yet (``LLMPlugin.leases``, ``usage_log.lease_wait_ms``)
are reached inside each test, so a test fails on its own missing piece.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import aiosqlite
import pytest
from fake_clock import FakeClock
from helpers import build_plugin_and_channel, sent_texts, start_llm_plugin
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)
from test_lease_scopes import STAGES, _Response, install_spawn_tools

from corvidae.attribution import get_attribution
from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import hookimpl
from corvidae.metrics import MetricsPlugin, UsageLogPlugin
from corvidae.outbound import OutboundMessage
from corvidae.task import Task
from corvidae.tool import Tool
from corvidae.trace import mint_trace_id


# ---------------------------------------------------------------------------
# The fake server: every POST is a parked Call
# ---------------------------------------------------------------------------

_ids = itertools.count(1)


@dataclass
class Call:
    """One chat POST: who sent it (the lease key its context carried), what
    it said last, and the event that lets it return."""

    seq: int
    key: str | None
    role: str
    text: str
    release: asyncio.Event = field(default_factory=asyncio.Event)


class _GatedPost:
    """The async context manager ``session.post`` returns: parks until the
    call is released (when the session parks it), then answers."""

    def __init__(self, session: "GatedSession", call: Call) -> None:
        self._session = session
        self._call = call

    async def __aenter__(self) -> _Response:
        # In flight from the moment the POST starts until it has answered.
        self._session.in_flight += 1
        self._session.max_in_flight = max(self._session.max_in_flight, self._session.in_flight)
        if self._session.park(self._call):
            await self._call.release.wait()
        return _Response(self._session.respond(self._call))

    async def __aexit__(self, *exc: object) -> bool:
        self._session.in_flight -= 1
        return False


class GatedSession:
    """aiohttp session stand-in. ``calls`` lists every POST in start order;
    ``park(call)`` says whether it waits for ``call.release``;
    ``respond(call)`` is the payload it answers with."""

    def __init__(self) -> None:
        self.headers: dict = {}
        self.calls: list[Call] = []
        self.park: Callable[[Call], bool] = lambda call: True
        self.respond: Callable[[Call], dict] = lambda call: _make_text_response("ok")
        # Set whenever a new call starts, so waiters re-check their predicate.
        self._changed = asyncio.Event()
        # POSTs in flight now, and the most there ever were at once.
        self.in_flight = 0
        self.max_in_flight = 0

    def post(self, url: str, json: dict | None = None, timeout: Any = None, **kw: Any) -> _GatedPost:
        """Record the POST with the lease key of the context it was sent from."""
        messages = (json or {}).get("messages") or [{}]
        last = messages[-1]
        text = last.get("content") if isinstance(last.get("content"), str) else ""
        call = Call(
            seq=len(self.calls), key=get_attribution().get("lease_key"),
            role=last.get("role", ""), text=text,
        )
        self.calls.append(call)
        self._changed.set()
        return _GatedPost(self, call)

    async def close(self) -> None:
        """Nothing to release."""

    async def started(self, pred: Callable[[Call], bool], what: str) -> Call:
        """The first call matching ``pred``, waiting (bounded) until one has started."""
        try:
            async with asyncio.timeout(5.0):
                while True:
                    hit = next((c for c in self.calls if pred(c)), None)
                    if hit is not None:
                        return hit
                    self._changed.clear()
                    await self._changed.wait()
        except TimeoutError:
            raise AssertionError(
                f"no model call matching {what!r} started; calls so far: "
                f"{[(c.key, c.text) for c in self.calls]}"
            ) from None

    def texts(self) -> list[str]:
        """The last-message text of every call, in start order."""
        return [c.text for c in self.calls]


async def until(pred: Callable[[], bool], what: str, timeout: float = 6.0) -> None:
    """Yield to the loop until ``pred()`` holds; no fixed delays."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0)


async def queued(stack: IStack, *keys: str | None) -> None:
    """Wait until exactly ``keys`` are queued for the main provider's head, in order."""
    await until(
        lambda: [w.key for w in stack.llm.leases.snapshot("main").waiters] == list(keys),
        f"{list(keys)} queued for the head",
    )


# ---------------------------------------------------------------------------
# The stack
# ---------------------------------------------------------------------------


@dataclass
class Gates:
    """Control of the ``block`` tool: per label, an event set when it is
    entered, one that lets it return, and the labels that returned."""

    entered: dict[str, asyncio.Event] = field(default_factory=lambda: defaultdict(asyncio.Event))
    release: dict[str, asyncio.Event] = field(default_factory=lambda: defaultdict(asyncio.Event))
    finished: list[str] = field(default_factory=list)


@dataclass
class IStack:
    """Agent + TaskQueue + real LLMPlugin on a gated session, two channels."""

    plugin: Any
    a: Channel
    b: Channel
    db: aiosqlite.Connection
    llm: Any
    session: GatedSession
    gates: Gates
    clock: FakeClock | None


async def build(monkeypatch: pytest.MonkeyPatch, heads: int | None, *, with_clock: bool = False) -> IStack:
    """Assemble the stack; ``heads`` goes in ``llm.providers.main.heads``
    (None leaves it undeclared)."""
    plugin, a, db = await build_plugin_and_channel(mock_on_agent_response=False)
    pm = plugin.pm
    await plugin.on_init(pm, {"agent": {"forks": {"enabled": True}}})
    clock = None
    if with_clock:
        # Registered before the LLMPlugin starts, so the lease manager reads it.
        clock = FakeClock()
        pm.register(clock, name="clock")
    provider: dict = {"base_url": "http://main.test/v1"}
    if heads is not None:
        provider["heads"] = heads
    section = {"providers": {"main": provider}, "profiles": {"main": {"model": "main-m"}}}
    llm, _posts = await start_llm_plugin(monkeypatch, section, STAGES, pm=pm)
    # No network: every client posts through the gated session.
    session = GatedSession()
    for client in llm._clients.values():
        client.session = session
    usage_log = UsageLogPlugin()
    pm.register(usage_log, name="usage_log")
    await usage_log.on_init(pm, {})
    await usage_log.on_start({})
    plugin._client = llm.get_client()
    plugin._mark_started()
    b = plugin._registry.get_or_create("test", "scope2", config=ChannelConfig())
    stack = IStack(plugin, a, b, db, llm, session, Gates(), clock)
    await install_tools(stack)
    return stack


async def install_tools(stack: IStack) -> None:
    """The real subagent and fork tools, plus ``block`` (parks on a Gates
    label) and ``echo`` (returns at once), as the Agent's and the loops' tools."""
    gates = stack.gates

    async def block(label: str) -> str:
        """Block until the test releases this label.

        Args:
            label: Which gate to wait on.
        """
        gates.entered[label].set()
        await gates.release[label].wait()
        gates.finished.append(label)
        return f"unblocked {label}"

    async def echo(text: str) -> str:
        """Return the text.

        Args:
            text: What to return.
        """
        return f"echoed {text}"

    await install_spawn_tools(stack)  # type: ignore[arg-type]
    registry = stack.plugin.pm.get_plugin("tools").get_registry()
    registry.add(Tool.from_function(block))
    registry.add(Tool.from_function(echo))
    stack.plugin._tools = registry.as_dict()
    stack.plugin._tool_schemas = registry.schemas()


@pytest.fixture
async def make_stack(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Callable[..., Any]]:
    """A factory for stacks; every stack it built is torn down afterwards
    (parked calls and tools released first so nothing hangs the teardown)."""
    built: list[IStack] = []

    async def factory(heads: int | None, **kw: Any) -> IStack:
        stack = await build(monkeypatch, heads, **kw)
        built.append(stack)
        return stack

    yield factory
    for stack in built:
        # Let anything still parked finish, then stop the plugins.
        stack.session.park = lambda call: False
        for call in stack.session.calls:
            call.release.set()
        for event in stack.gates.release.values():
            event.set()
        await stack.plugin.pm.get_plugin("task").on_stop()
        for queue in stack.plugin.queues.values():
            await queue.stop()
        await stack.llm.on_stop()
        await stack.db.close()


def tool_call(name: str, **args: Any) -> dict:
    """A model reply that calls one tool."""
    return _make_tool_call_response([_make_tool_call(f"c{next(_ids)}", name, args)])


def script(stack: IStack, rules: dict[str, dict], parked: set[str] | None = None) -> None:
    """Answer a user-role call whose text contains a rule's token with that
    reply (anything else "ok"); park exactly the calls whose text is in ``parked``."""
    def respond(call: Call) -> dict:
        if call.role == "user":
            for token, reply in rules.items():
                if token in call.text:
                    return reply
        return _make_text_response("ok")

    stack.session.respond = respond
    stack.session.park = lambda call: call.text in (parked or set())


async def send(stack: IStack, channel: Channel, text: str) -> None:
    """A user message arriving on ``channel`` through the Agent's front door."""
    await stack.plugin.on_message(
        channel=channel, sender="alice", text=text, trace_id=mint_trace_id()
    )


class SilentFork:
    """Test plugin standing in for critique: once, for one channel, enqueue a
    silent (deliver=False) task whose work makes one model call. The task
    captures the enqueuing context, so its call carries the channel's key."""

    def __init__(self, stack: IStack, channel: Channel) -> None:
        self.stack = stack
        self.channel = channel
        self.fired = False

    @hookimpl
    async def on_agent_response(self, channel: Channel) -> None:
        """Enqueue after the turn's final reply, as critique does (own channel, once)."""
        if self.fired or channel.id != self.channel.id:
            return
        self.fired = True
        stack = self.stack

        async def work() -> str:
            """One model call under the channel's key."""
            await stack.llm.get_client().chat(
                [{"role": "user", "content": f"SILENT:{channel.id}"}]
            )
            return "ok"

        task_queue = stack.plugin.pm.get_plugin("task").task_queue
        await task_queue.enqueue(
            Task(work=work, channel=channel, deliver=False, description="silent fork")
        )


# ---------------------------------------------------------------------------
# R1: no heads, no leases
# ---------------------------------------------------------------------------


async def test_r1_heads_undeclared_has_no_manager_and_calls_do_not_wait(make_stack) -> None:
    """R1: no heads declared: LLMPlugin has no lease manager and two
    channels' calls are in flight at the same time, each under its key."""
    stack = await make_stack(None)
    script(stack, {}, parked={"A1", "B1"})

    await send(stack, stack.a, "A1")
    await send(stack, stack.b, "B1")
    a = await stack.session.started(lambda c: c.text == "A1", "A1")
    b = await stack.session.started(lambda c: c.text == "B1", "B1")

    assert stack.llm.leases is None
    assert (a.key, b.key) == (stack.a.id, stack.b.id)


# ---------------------------------------------------------------------------
# R5: B gets the head while A's tool runs
# ---------------------------------------------------------------------------


async def test_r5_other_channel_gets_the_head_while_a_tool_runs(make_stack) -> None:
    """R5: heads 1, A's turn returns a tool call whose tool blocks: B's call
    goes out while the tool runs, and A's next call after the tool goes out
    after B's."""
    stack = await make_stack(1)
    script(stack, {"GO_TOOL": tool_call("block", label="t1")}, parked={"B1"})

    await send(stack, stack.a, "GO_TOOL")
    await asyncio.wait_for(stack.gates.entered["t1"].wait(), timeout=5.0)
    await send(stack, stack.b, "B1")
    b1 = await stack.session.started(lambda c: c.text == "B1", "B1")
    assert "t1" not in stack.gates.finished

    # The tool returns while B's call holds the head: A's next call queues.
    stack.gates.release["t1"].set()
    await queued(stack, stack.a.id)
    assert [c.key for c in stack.session.calls] == [stack.a.id, stack.b.id]

    b1.release.set()
    await stack.session.started(lambda c: c.seq >= 2, "A's call after the tool")

    assert [c.key for c in stack.session.calls] == [stack.a.id, stack.b.id, stack.a.id]


# ---------------------------------------------------------------------------
# R6, R7
# ---------------------------------------------------------------------------


async def test_r6_message_is_not_held_for_the_channels_own_fork(make_stack) -> None:
    """R6: heads 1: a message to A while A's silent fork's call is in flight
    goes out at once; the lease never orders A behind its own fork."""
    stack = await make_stack(1)
    script(stack, {}, parked={"A1", "A2", f"SILENT:{stack.a.id}"})
    stack.plugin.pm.register(SilentFork(stack, stack.a), name="silent_fork")

    await send(stack, stack.a, "A1")
    a1 = await stack.session.started(lambda c: c.text == "A1", "A1")
    a1.release.set()
    silent = await stack.session.started(lambda c: c.text.startswith("SILENT"), "silent fork")

    await send(stack, stack.a, "A2")
    a2 = await stack.session.started(lambda c: c.text == "A2", "A2")

    assert not silent.release.is_set()
    assert (a2.key, silent.key) == (stack.a.id, stack.a.id)
    # One lease, A's, covers both calls in flight; nobody waits.
    view = stack.llm.leases.snapshot("main")
    assert [h.key for h in view.holders] == [stack.a.id] and view.waiters == ()
    assert stack.session.texts() == ["A1", f"SILENT:{stack.a.id}", "A2"]


async def test_r7_real_subagent_on_one_head_finishes_with_its_parent(make_stack) -> None:
    """R7: heads 1, the real subagent tool: the parent channel and its
    subagent do not deadlock; the subagent's run is keyed by its task id and
    its result reaches the channel."""
    stack = await make_stack(1)
    script(stack, {"GO_SUB": tool_call("subagent", instructions="SUBWORK", description="sub")})

    await send(stack, stack.a, "GO_SUB")
    # Reply after the tool was enqueued, then reply after the subagent's result.
    await until(lambda: len(sent_texts(stack.plugin.pm, OutboundMessage)) >= 2, "parent and subagent both finished")

    keys = [c.key for c in stack.session.calls]
    assert any(k is not None and k.startswith("subagent:") for k in keys)
    assert keys[0] == stack.a.id and keys[-1] == stack.a.id
    # The one head was never exceeded, and a lease manager was in charge.
    assert stack.session.max_in_flight == 1
    assert stack.llm.leases is not None


# ---------------------------------------------------------------------------
# R13: the per-call lease wait
# ---------------------------------------------------------------------------


class _Metrics:
    """Consumer plugin recording every on_metrics event."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float, dict]] = []

    @hookimpl
    async def on_metrics(self, name: str, value: float, tags: dict) -> None:
        """Keep each event."""
        self.events.append((name, value, dict(tags)))


async def lease_waits(stack: IStack, rows: int) -> list[tuple[str | None, float | None]]:
    """(lease_key, lease_wait_ms) of each usage_log row, once ``rows`` exist."""
    async def read() -> list[tuple]:
        async with stack.db.execute(
            "SELECT lease_key, lease_wait_ms FROM usage_log ORDER BY id"
        ) as cursor:
            return await cursor.fetchall()

    result: list[tuple] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 6.0
    while len(result) < rows:
        result = await read()
        if loop.time() > deadline:
            raise AssertionError(f"usage_log has {len(result)} rows, wanted {rows}")
        await asyncio.sleep(0)
    return result


async def test_r13_call_that_waited_records_its_lease_wait(make_stack) -> None:
    """R13: heads 1: B's call waits 7 s of fake time for A's, so its
    usage_log row has lease_wait_ms 7000.0 and on_metrics sees
    llm.lease_wait_ms 7000.0 for B; A's call, which did not wait, records 0.0."""
    stack = await make_stack(1, with_clock=True)
    pm = stack.plugin.pm
    metrics = _Metrics()
    pm.register(MetricsPlugin(pm), name="metrics")
    pm.register(metrics, name="metrics_recorder")
    script(stack, {}, parked={"A1", "B1"})

    await send(stack, stack.a, "A1")
    a1 = await stack.session.started(lambda c: c.text == "A1", "A1")
    await send(stack, stack.b, "B1")
    await queued(stack, stack.b.id)
    await stack.clock.advance(7.0)
    a1.release.set()
    b1 = await stack.session.started(lambda c: c.text == "B1", "B1")
    b1.release.set()

    rows = await lease_waits(stack, 2)
    await until(
        lambda: any(n == "llm.lease_wait_ms" and t.get("lease_key") == stack.b.id
                    for n, _v, t in metrics.events),
        "llm.lease_wait_ms for B",
    )
    assert rows == [(stack.a.id, 0.0), (stack.b.id, 7000.0)]
    waited = [v for n, v, t in metrics.events
              if n == "llm.lease_wait_ms" and t.get("lease_key") == stack.b.id]
    assert waited == [7000.0]


async def test_r13_unbounded_provider_records_null_lease_wait(make_stack) -> None:
    """R13: with no heads declared the call never touched a lease: NULL in
    usage_log (not 0.0)."""
    stack = await make_stack(None)
    script(stack, {})

    await send(stack, stack.a, "A1")

    assert await lease_waits(stack, 1) == [(stack.a.id, None)]
