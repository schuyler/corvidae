"""The pre-send hold-and-revise state machine through a real Agent loop (#1149, #1169 part 2).

RED phase. Everything here drives a real ``Agent`` through its queue, in the
order the kernel runs it: a message or notification is queued,
``before_agent_turn`` fires, the held store admits what is due, the scripted
LLM answers, the ``should_send_response`` "final" gate runs under the check
allowance, ``on_agent_response`` fires, and the held stub turns that follow
run the same way. A hand-call of ``PreSendHold`` would skip that order, which
is what the criteria are about. (The class's own contract, and appraisal's and
critique's hookimpls, are in ``tests/cognition/test_presend_hooks.py``.)

Why ``Checker`` stands in for appraisal. ``Checker`` owns a
``PreSendHold(pm, reporter="checker")`` and wires it the way
docs/cognition.md ("Held for a revision") says appraisal does: the same four hooks,
in the same order. ``should_send_response`` on a "final" skips when
``hold.suppressed(channel.id)`` and otherwise returns
``await hold.refuse(...)`` for a draft the test scripted as objectionable;
``before_agent_turn`` forwards to ``hold.on_turn``; ``on_held_admitted``
forwards to ``hold.on_admitted``; ``on_agent_response`` awaits
``hold.on_final`` before anything else. The hold sees the same calls from the
kernel whichever plugin owns it, so what these tests prove about it carries to
appraisal; the cognition-level file covers appraisal's own wiring (its scoring,
its fork, its objection collection). ``Checker`` replaces only the scoring: the
test says which drafts draw objections.

Signatures follow the design's class sketch as written:
``refuse(channel, trace_id, text, annotations, objections)``,
``on_turn(channel, trace_id, annotations)``,
``on_admitted(channel, source, admitted_keys, duplicate_keys)`` (``on_held_admitted``
also receives a ``trace_id``, which the design says the hold does not use, so
``Checker`` does not forward it) and ``on_final(channel, trace_id, response_text, annotations)``.
The hold's logger name is not specified; warning assertions match on the
channel id and the word "objection", never on a logger name.

Criteria covered: R3, R4, R2/R5 (in time), R5 (late), R6 (stub text, unchecked
revision, span end, inputs a to e, tool-cycle span).

``corvidae_cognition.presend`` does not exist yet; its names are reached
through the ``P`` fixture so each test fails on its own.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from corvidae.context import MessageType
from corvidae.hooks import final_was_unsent, hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.queue import SerialQueue
from corvidae.tool import tool_to_schema
from corvidae.trace import mint_trace_id
from fake_clock import install_kernel_clock
from helpers import build_plugin_and_channel, drain, drain_task_queue, sent_texts
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)

#: The minimum allowance a channel gets when nothing else sets one (seconds).
DEFAULT_MINIMUM = 5.0

#: How the pre-send frame opens (R4); the rest of the header is the kernel's.
PRESEND_FRAME = "[CONTEXT from pre-send check"

#: How a framed late-check row opens (R5).
LATE_HEADER = "[CONTEXT from late check"

#: The objections a refused draft carries unless a test says otherwise.
OBJECTIONS = ["the date in that reply is wrong"]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def P():
    """The not-yet-written ``corvidae_cognition.presend`` module, imported lazily."""
    return importlib.import_module("corvidae_cognition.presend")


class Checker:
    """A test stand-in for appraisal's wiring of ``PreSendHold`` (module docstring).

    ``objectionable`` maps a draft's text to the objections it draws; a text not
    in it passes. ``checked`` lists every final the gate actually checked, in
    order; a final skipped because the hold was ``suppressed`` is not in it.
    ``delay`` makes the first check answer that many fake-clock seconds late.
    """

    def __init__(self, pm, P, objectionable: dict[str, list[str]], clock=None, delay: float = 0.0):
        self.hold = P.PreSendHold(pm, reporter="checker")
        self.objectionable = objectionable
        self.clock = clock
        self.delay = delay
        self.checked: list[str] = []
        self.started = asyncio.Event()

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations):
        # Only finals are checked; every other emission is no opinion.
        if emission != "final":
            return None
        # A hold is outstanding: do not check (the unchecked span).
        if self.hold.suppressed(channel.id):
            return None
        self.checked.append(text)
        self.started.set()
        # A slow check answers after the allowance, on the fake clock; only the
        # first check is slow, so a later turn's check does not wait.
        delay, self.delay = self.delay, 0.0
        if delay:
            await self.clock.sleep(delay)
        objections = self.objectionable.get(text)
        if objections:
            return await self.hold.refuse(channel, trace_id, text, annotations, objections)
        return None

    @hookimpl
    async def before_agent_turn(self, channel, trace_id, annotations):
        self.hold.on_turn(channel, trace_id, annotations)

    @hookimpl
    async def on_held_admitted(self, channel, source, admitted_keys, duplicate_keys):
        await self.hold.on_admitted(channel, source, admitted_keys, duplicate_keys)

    @hookimpl
    async def on_agent_response(
        self, channel, request_text, response_text, trace_id, annotations, logprobs
    ):
        # The hold's outcome check runs first, as appraisal's hookimpl does.
        await self.hold.on_final(channel, trace_id, response_text, annotations)


class NotifySpy:
    """Records every on_notify firing (alongside the Agent's own handler)."""

    def __init__(self):
        self.calls: list[dict] = []

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, annotations):
        self.calls.append({
            "channel": channel, "source": source, "text": text,
            "tool_call_id": tool_call_id, "annotations": dict(annotations or {}),
        })

    def stubs(self, origin: str) -> list[dict]:
        """The stub-style firings (no tool_call_id) carrying ``origin``."""
        return [
            c for c in self.calls
            if c["tool_call_id"] is None and c["annotations"].get("origin") == origin
        ]


class TraceSpy:
    """Records every ``on_trace_record`` and the unsent state of every final."""

    def __init__(self):
        self.records: list[dict] = []
        self.finals: list[tuple[str, bool]] = []

    @hookimpl
    async def on_trace_record(self, trace_id, channel_id, source, origin, record):
        self.records.append({"trace_id": trace_id, "source": source, "record": record})

    @hookimpl
    async def on_agent_response(
        self, channel, request_text, response_text, trace_id, annotations, logprobs
    ):
        # What the transport would have seen of this final, read inside the hook.
        self.finals.append((response_text, final_was_unsent(channel)))

    def held(self) -> list[dict]:
        """The records that report a held draft."""
        return [r for r in self.records if "held" in r["record"]]


class OfferSpy:
    """Records every batch the held store announced (``on_held_admitted``)."""

    def __init__(self):
        self.events: list[tuple[str, list[str]]] = []

    @hookimpl
    async def on_held_admitted(self, channel, source, trace_id, admitted_keys, duplicate_keys):
        self.events.append((source, list(admitted_keys)))

    def of(self, source: str) -> list[list[str]]:
        """The admitted keys of each batch announced for ``source``."""
        return [keys for s, keys in self.events if s == source]


class Policy:
    """A ``transform_held_admission`` wrapper acting on the pre-send batches.

    ``mode`` is "all" (leave the seed's plan), "drop" (admit none of the
    objections, listing them as policy drops) or "keep" (admit none and list
    nothing, so a pushed batch stays held for a later turn).
    """

    def __init__(self, P, mode: str = "all"):
        self.source = P.SOURCE
        self.mode = mode

    @hookimpl(wrapper=True)
    def transform_held_admission(self, channel, conversation, batches):
        plans = yield
        out = []
        for plan in plans:
            batch = batches[plan.batch]
            if batch.source != self.source or self.mode == "all":
                out.append(plan)
            elif self.mode == "drop":
                out.append(type(plan)(
                    batch=plan.batch, admit=[], duplicate=[],
                    drop=list(range(len(batch.entries))),
                ))
            else:
                out.append(type(plan)(batch=plan.batch, admit=[], duplicate=[], drop=[]))
        return out


@dataclass
class Then:
    """A scripted LLM answer that first runs ``act`` (to queue something
    behind the turn that is being answered)."""

    response: Any
    act: Any = None


@dataclass
class Env:
    plugin: Any
    channel: Any
    db: Any
    clock: Any
    checker: Checker
    spy: NotifySpy
    trace: TraceSpy
    offers: OfferSpy
    policy: Policy
    extras: dict = field(default_factory=dict)


def script(plugin, *steps) -> None:
    """Make the LLM answer ``steps`` in call order.

    A step is a response, an exception to raise, or a ``Then`` whose ``act``
    runs (and is awaited, if it returns an awaitable) before the response.
    """
    queue = list(steps)

    async def chat(*args, **kwargs):
        step = queue.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, Then):
            if step.act is not None:
                result = step.act()
                if inspect.isawaitable(result):
                    await result
            return step.response
        return step

    plugin._client.chat = AsyncMock(side_effect=chat)


@pytest.fixture
async def make_env(P, tmp_path):
    """A factory for ``Env``: an Agent with the held store, the trace protocol
    plugin, a ``Checker`` and a scripted LLM, on a channel with the minimum
    allowance. ``objectionable`` and ``delay`` configure the Checker."""
    built: list[Env] = []

    async def make(objectionable: dict[str, list[str]], delay: float = 0.0, mode: str = "all"):
        # mock_on_agent_response=False: the real hook must fire, or no plugin hears a final.
        plugin, channel, db = await build_plugin_and_channel(mock_on_agent_response=False)
        clock = install_kernel_clock(plugin.pm)[0]

        # The trace protocol plugin owns on_trace_record, which the hold fires.
        from corvidae.hooks import install_plugin_hookspecs
        from corvidae_cognition.trace import TracePlugin

        plugin.pm.register(TracePlugin(plugin.pm), name="trace")
        trace, spy, offers = TraceSpy(), NotifySpy(), OfferSpy()
        plugin.pm.register(trace, name="trace_spy")
        plugin.pm.register(spy, name="notify_spy")
        plugin.pm.register(offers, name="offer_spy")
        policy = Policy(P, mode)
        plugin.pm.register(policy, name="policy")
        install_plugin_hookspecs(plugin.pm)

        checker = Checker(plugin.pm, P, objectionable, clock=clock, delay=delay)
        plugin.pm.register(checker, name="checker")

        async def shell(command: str) -> str:
            """Run a command."""
            return f"ran {command}"

        plugin._tools = {"shell": shell}
        plugin._tool_schemas = [tool_to_schema(shell)]
        plugin._client = MagicMock()
        plugin._client.chat = AsyncMock()
        plugin._mark_started()
        env = Env(plugin, channel, db, clock, checker, spy, trace, offers, policy)
        built.append(env)
        return env

    yield make
    for env in built:
        await env.plugin.pm.get_plugin("task").on_stop()
        await env.clock.close()
        await env.db.close()


async def say(env: Env, text: str = "hello") -> None:
    """Deliver a user message and drain the channel queue."""
    await env.plugin.on_message(
        channel=env.channel, sender="user", text=text, trace_id=mint_trace_id()
    )
    await asyncio.wait_for(drain(env.plugin, env.channel), 2)


async def notify(env: Env, source: str, origin: str, text: str = "ping") -> None:
    """Queue a notification (no tool call) carrying ``origin``; does not drain."""
    await env.plugin.on_notify(
        channel=env.channel, source=source, text=text, tool_call_id=None,
        annotations={"origin": origin},
    )


async def settle_all(env: Env, rounds: int = 6) -> None:
    """Drain the queue, the tool tasks and the queue again, until quiet."""
    for _ in range(rounds):
        await asyncio.wait_for(drain(env.plugin, env.channel), 2)
        await asyncio.wait_for(drain_task_queue(env.plugin), 2)
    await asyncio.wait_for(drain(env.plugin, env.channel), 2)


def prompt_text(plugin, call_index: int) -> str:
    """Everything the LLM call at ``call_index`` was shown, as one string."""
    messages = plugin._client.chat.await_args_list[call_index].args[0]
    return "\n".join(str(m.get("content")) for m in messages)


def context_rows(channel) -> list[dict]:
    """The window's CONTEXT rows, in order."""
    return [
        m for m in channel.conversation.messages
        if m.get("_message_type") == MessageType.CONTEXT
    ]


def sent(env: Env) -> list[str]:
    """Texts the transport was handed as a reply, in order."""
    return sent_texts(env.plugin.pm, OutboundMessage)


def slow_tool_setup(plugin) -> asyncio.Event:
    """Give the agent a tool that blocks until the returned event is set."""
    release = asyncio.Event()

    async def slow_tool(x: str) -> str:
        """A tool that runs until the test releases it."""
        await release.wait()
        return f"result:{x}"

    plugin._tools = {"slow_tool": slow_tool}
    plugin._tool_schemas = [tool_to_schema(slow_tool)]
    return release


def slow_call(call_id: str = "call-1"):
    """A model response that calls slow_tool once."""
    return _make_tool_call_response([_make_tool_call(call_id, "slow_tool", {"x": "a"})])


# ---------------------------------------------------------------------------
# R3, R4, R2/R5 in time: the plain hold
# ---------------------------------------------------------------------------


class TestInTimeHold:
    async def test_r3_in_time_refusal_stores_the_draft_unsent_and_the_transport_never_sees_it(
        self, make_env
    ):
        """R3: a refusal inside the allowance leaves the draft in the window as
        an UNSENT row (and in message_log), ``final_was_unsent`` is true inside
        ``on_agent_response``, and ``send`` never receives the draft."""
        env = await make_env({"the draft": OBJECTIONS})
        script(env.plugin, _make_text_response("the draft"), _make_text_response("revised"))

        await say(env)

        unsent = [
            m for m in env.channel.conversation.messages
            if m.get("_message_type") == MessageType.UNSENT
        ]
        assert [m["content"] for m in unsent] == ["the draft"]
        assert ("the draft", True) in env.trace.finals
        async with env.db.execute(
            "SELECT message_type FROM message_log WHERE channel_id = ?", (env.channel.id,)
        ) as cursor:
            types = [row[0] async for row in cursor]
        assert MessageType.UNSENT.value in types
        # The transport saw no draft, in any outbound event.
        for call in env.plugin.pm.ahook.send.await_args_list:
            assert "the draft" not in str(getattr(call.kwargs["event"], "text", ""))

    async def test_r4_one_revising_turn_has_the_objections_framed_before_its_llm_call(
        self, make_env
    ):
        """R4: exactly one revising turn follows the hold. Its prompt, at its
        LLM call, holds the objections inside the pre-send frame; the draft
        turn's prompt does not. Its origin is "user", the held exchange's,
        through the design's fallback (a user message carries none)."""
        env = await make_env({"the draft": OBJECTIONS})
        script(env.plugin, _make_text_response("the draft"), _make_text_response("revised"))

        await say(env)

        # One draft call and one revising call, no more.
        assert env.plugin._client.chat.await_count == 2
        assert PRESEND_FRAME not in prompt_text(env.plugin, 0)
        assert PRESEND_FRAME in prompt_text(env.plugin, 1)
        assert OBJECTIONS[0] in prompt_text(env.plugin, 1)
        # Its origin is the user's: the stub that woke it carries "user".
        assert len(env.spy.stubs("user")) == 1
        assert sent(env) == ["revised"]

    async def test_r4_objections_are_offered_at_exactly_one_turn(self, make_env, P):
        """R4: the batch is offered once, at the revising turn; a later user
        message does not offer it again."""
        env = await make_env({"the draft": OBJECTIONS})
        script(
            env.plugin, _make_text_response("the draft"), _make_text_response("revised"),
            _make_text_response("later reply"),
        )

        await say(env)
        await say(env, "and another thing")

        assert len(env.offers.of(P.SOURCE)) == 1
        framed = [m for m in context_rows(env.channel) if m["content"].startswith(PRESEND_FRAME)]
        assert len(framed) == 1

    async def test_r2_r5_in_time_refusal_fires_exactly_one_stub_with_origin_user(
        self, make_env, P
    ):
        """R2 / R5 (in time): the refusal is held for revision, not withheld:
        exactly one stub with origin "user" fires, carrying the revision stub
        text, and one held record is reported."""
        env = await make_env({"the draft": OBJECTIONS})
        script(env.plugin, _make_text_response("the draft"), _make_text_response("revised"))

        await say(env)

        stubs = env.spy.stubs("user")
        assert len(stubs) == 1
        assert stubs[0]["text"] == P.REVISION_STUB
        assert len(env.trace.held()) == 1


# ---------------------------------------------------------------------------
# R5 late: a refusal after the allowance holds nothing and wakes nothing
# ---------------------------------------------------------------------------


class TestLateRefusal:
    async def test_r5_late_refusal_sends_the_reply_frames_it_next_turn_and_revises_nothing(
        self, make_env, monkeypatch
    ):
        """R5 (late): a refusal that arrives at t=60, after the allowance, is
        sent as written; its objections are framed as a late-check row at the
        next turn; no pre-send stub fires, no revising turn runs, and no held
        record is reported."""
        env = await make_env({"the reply": OBJECTIONS}, delay=60)
        script(env.plugin, _make_text_response("the reply"), _make_text_response("second"))

        await env.plugin.on_message(
            channel=env.channel, sender="user", text="hi", trace_id=mint_trace_id()
        )
        drained = asyncio.create_task(drain(env.plugin, env.channel))
        await asyncio.wait_for(env.checker.started.wait(), 1)
        await env.clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(drained, 1)
        assert sent(env) == ["the reply"]

        # The check returns late: nothing is queued and no turn runs.
        enqueued: list = []
        real = SerialQueue.enqueue

        async def spy_enqueue(self, item):
            enqueued.append(item)
            await real(self, item)

        monkeypatch.setattr(SerialQueue, "enqueue", spy_enqueue)
        await env.clock.advance(60)
        await drain(env.plugin, env.channel)
        assert enqueued == []
        assert env.plugin._client.chat.await_count == 1
        assert env.spy.stubs("user") == []

        # The next turn carries it, as a late-check row and not a pre-send one.
        await say(env, "next message")
        late = [m for m in context_rows(env.channel) if m["content"].startswith(LATE_HEADER)]
        assert len(late) == 1
        assert OBJECTIONS[0] in late[0]["content"]
        assert PRESEND_FRAME not in prompt_text(env.plugin, 1)
        assert env.plugin._client.chat.await_count == 2
        assert env.trace.held() == []


# ---------------------------------------------------------------------------
# R6: a revision is never held
# ---------------------------------------------------------------------------


class TestRevisionIsNeverHeld:
    async def test_r6_stub_text_has_no_no_reply_instruction(self, make_env, P):
        """R6: the stub that wakes the revising turn, and the constant it comes
        from, carry no ``NO_REPLY`` instruction."""
        env = await make_env({"the draft": OBJECTIONS})
        script(env.plugin, _make_text_response("the draft"), _make_text_response("revised"))

        await say(env)

        (stub,) = env.spy.stubs("user")
        assert "NO_REPLY" not in stub["text"]
        assert "NO_REPLY" not in P.REVISION_STUB

    async def test_r6_stub_text_offers_silence_as_well_as_revision(self, make_env, P):
        """R6 with the hold's other outcome named: the stub that wakes the
        revising turn offers a revision *and* sending nothing. A hold is a
        defect report, not an order to speak, and the decline it offers is one
        the kernel can honour — the revision span suppresses the pre-send check
        only (``suppressed()`` is read by critique and appraisal), so the
        silence veto still fires on that turn's final."""
        env = await make_env({"the draft": OBJECTIONS})
        script(env.plugin, _make_text_response("the draft"), _make_text_response("revised"))

        await say(env)

        (stub,) = env.spy.stubs("user")
        text = stub["text"].lower()
        # Pinned as an offered disjunction: two bare substrings would also pass
        # on wording that orders one outcome ("send nothing rather than revise").
        assert "either write a revised reply now, or decline this turn" in text, (
            "the stub no longer offers a choice between the two outcomes"
        )
        assert "NO_REPLY" not in stub["text"], "the token stays in the silence instruction"

    async def test_r6_a_decline_on_the_revising_turn_clears_the_span(self, make_env):
        """R6 with the silence branch taken: offering decline is honest only if
        taking it cannot re-arm the hold. The revision's final clears the span
        whatever it says, so the decline made no second hold and the channel's
        next final is checked again."""
        env = await make_env({
            "the draft": OBJECTIONS, "NO_REPLY": OBJECTIONS, "second reply": ["still wrong"],
        })
        script(
            env.plugin, _make_text_response("the draft"), _make_text_response("NO_REPLY"),
            _make_text_response("second reply"), _make_text_response("second revision"),
        )

        await say(env)
        # The decline went unchecked — the span was still open — and drew no hold.
        assert env.checker.checked == ["the draft"]
        assert len(env.trace.held()) == 1

        # Span cleared: this turn's final is checked again, and it makes the new hold.
        await say(env, "another question")
        assert env.checker.checked == ["the draft", "second reply"]
        assert len(env.trace.held()) == 2

    async def test_r6_revision_is_sent_unchecked_and_the_next_final_is_checked(
        self, make_env, caplog
    ):
        """R6: the revision's final is not checked (even though its text would
        draw objections) and is sent. After the span clears, the channel's next
        final is checked."""
        env = await make_env({
            "the draft": OBJECTIONS, "revised": OBJECTIONS, "second reply": ["still wrong"],
        })
        script(
            env.plugin, _make_text_response("the draft"), _make_text_response("revised"),
            _make_text_response("second reply"), _make_text_response("second revision"),
        )

        await say(env)
        assert env.checker.checked == ["the draft"]
        assert sent(env) == ["revised"]
        assert len(env.trace.held()) == 1
        # The objections were admitted, so the not-admitted warning stays quiet.
        assert not [r for r in caplog.records
                    if r.levelno >= logging.WARNING and "objection" in r.getMessage().lower()]

        # The span ended at the revision's final: the next final is checked.
        await say(env, "another question")
        assert env.checker.checked == ["the draft", "second reply"]
        assert len(env.trace.held()) == 2

    async def test_r6_a_stub_turn_deferred_by_an_outstanding_call_makes_no_second_hold(
        self, make_env
    ):
        """R6 input (a): the stub turn arrives while a tool call is outstanding,
        so nothing is offered and it is deferred; the re-fired stub's turn is
        the revision. The finals in between and the revision's are unchecked, so
        there is one hold."""
        env = await make_env({
            "the draft": OBJECTIONS, "noted": OBJECTIONS, "tool done": OBJECTIONS,
            "revised": OBJECTIONS,
        })
        release = slow_tool_setup(env.plugin)
        script(
            env.plugin, slow_call(), _make_text_response("the draft"),
            _make_text_response("noted"), _make_text_response("tool done"),
            _make_text_response("revised"),
        )

        await say(env, "use the tool")
        await say(env, "are you there")
        assert env.channel.pending_tool_call_ids
        release.set()
        await settle_all(env)

        assert env.checker.checked == ["the draft"]
        assert len(env.trace.held()) == 1
        assert env.plugin._client.chat.await_count == 5
        # Objections reached the re-fired turn only, which is the last call.
        assert PRESEND_FRAME in prompt_text(env.plugin, 4)
        for index in (2, 3):
            assert PRESEND_FRAME not in prompt_text(env.plugin, index)
        assert sent(env)[-1] == "revised"
        assert "the draft" not in sent(env)

    async def test_r6_an_item_with_another_origin_ahead_of_the_stub_makes_no_second_hold(
        self, make_env
    ):
        """R6 input (b), different origin: a notification of origin "other"
        queued ahead of the stub pops nothing and its final is unchecked; the
        stub turn is the revision, unchecked too. One hold."""
        env = await make_env({
            "the draft": OBJECTIONS, "other reply": OBJECTIONS, "revised": OBJECTIONS,
        })
        script(
            env.plugin,
            Then(
                _make_text_response("the draft"),
                lambda: notify(env, "other", "other"),
            ),
            _make_text_response("other reply"),
            _make_text_response("revised"),
        )

        await say(env)

        assert env.checker.checked == ["the draft"]
        assert len(env.trace.held()) == 1
        assert env.plugin._client.chat.await_count == 3
        assert PRESEND_FRAME not in prompt_text(env.plugin, 1)
        assert PRESEND_FRAME in prompt_text(env.plugin, 2)
        assert sent(env) == ["other reply", "revised"]

    async def test_r6_an_item_with_the_same_origin_ahead_of_the_stub_makes_no_second_hold(
        self, make_env
    ):
        """R6 input (b), same origin: the held exchange is a heartbeat-origin
        notification, and another heartbeat notification queued ahead of the
        stub pops the batch and becomes the revision. Its final is unchecked.
        The stub turn that follows is a new exchange and is checked normally."""
        env = await make_env({
            "the draft": OBJECTIONS, "second heartbeat": OBJECTIONS,
        })
        script(
            env.plugin,
            Then(
                _make_text_response("the draft"),
                lambda: notify(env, "heartbeat", "heartbeat", "tick two"),
            ),
            _make_text_response("second heartbeat"),
            _make_text_response("stub turn reply"),
        )

        await notify(env, "heartbeat", "heartbeat", "tick one")
        await asyncio.wait_for(drain(env.plugin, env.channel), 2)

        assert env.checker.checked == ["the draft", "stub turn reply"]
        assert len(env.trace.held()) == 1
        # The second heartbeat turn was offered the objections; the stub's was not.
        assert PRESEND_FRAME in prompt_text(env.plugin, 1)
        assert len(env.spy.stubs("heartbeat")) == 1  # the one stub; the ticks bypass the hook
        assert "second heartbeat" in sent(env)

    async def test_r6_a_retried_revising_turn_makes_no_second_hold(self, make_env):
        """R6 input (c): the revising turn's LLM call fails with the backend
        unreachable and is retried. The retry's final (same trace) goes out
        unchecked; one hold."""
        env = await make_env({"the draft": OBJECTIONS, "revised": OBJECTIONS})
        script(
            env.plugin, _make_text_response("the draft"),
            aiohttp.ClientConnectionError("refused"), _make_text_response("revised"),
        )

        await say(env)
        assert "revised" not in sent(env)
        await env.clock.advance(5)
        await asyncio.wait_for(drain(env.plugin, env.channel), 2)

        assert env.checker.checked == ["the draft"]
        assert len(env.trace.held()) == 1
        assert sent(env)[-1] == "revised"
        assert "the draft" not in sent(env)

    async def test_r6_objections_the_policy_admits_none_of_send_the_revision_unchecked_with_a_warning(
        self, make_env, caplog
    ):
        """R6 input (d): admission policy drops every pre-send entry. The
        revision is sent unchecked, with one WARNING naming the channel and the
        objections that did not reach it. A policy drop is equivalent to going
        over the budget: both reach ``on_held_admitted`` with
        ``admitted_keys == []``, which is all the hold reads."""
        env = await make_env({"the draft": OBJECTIONS, "revised": OBJECTIONS}, mode="drop")
        script(env.plugin, _make_text_response("the draft"), _make_text_response("revised"))

        with caplog.at_level(logging.WARNING):
            await say(env)

        assert env.checker.checked == ["the draft"]
        assert len(env.trace.held()) == 1
        assert sent(env) == ["revised"]
        assert PRESEND_FRAME not in prompt_text(env.plugin, 1)
        warnings = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING
            and env.channel.id in r.getMessage()
            and "objection" in r.getMessage().lower()
        ]
        assert len(warnings) == 1, [r.getMessage() for r in caplog.records]

    async def test_r6_a_stale_batch_offered_at_another_origins_turn_makes_no_second_hold(
        self, make_env
    ):
        """R6 input (e): a heartbeat-origin batch left held by an earlier hold
        (policy kept it) is offered at a heartbeat turn that runs ahead of the
        user hold's stub. That turn is not the revision: its final is
        unchecked, and so is the stub turn's, so the later hold is not held
        twice."""
        env = await make_env(
            {"hb draft": OBJECTIONS, "the draft": OBJECTIONS, "revised": OBJECTIONS},
            mode="keep",
        )
        script(
            env.plugin, _make_text_response("hb draft"), _make_text_response("hb revised"),
            Then(
                _make_text_response("the draft"),
                lambda: notify(env, "heartbeat", "heartbeat", "tick two"),
            ),
            _make_text_response("second heartbeat"),
            _make_text_response("revised"),
        )

        # Hold one, on a heartbeat exchange; policy keeps its batch held.
        await notify(env, "heartbeat", "heartbeat", "tick one")
        await asyncio.wait_for(drain(env.plugin, env.channel), 2)
        assert len(env.trace.held()) == 1
        env.policy.mode = "all"

        # Hold two, on a user exchange, with a heartbeat turn ahead of its stub.
        await say(env)

        assert env.checker.checked == ["hb draft", "the draft"]
        assert len(env.trace.held()) == 2
        assert sent(env)[-1] == "revised"
        assert PRESEND_FRAME in prompt_text(env.plugin, 4)

    async def test_r6_a_revising_turn_that_ends_with_no_final_does_not_leave_the_channel_unchecked(
        self, make_env
    ):
        """R6 span end: the revising turn fails with an error that is not
        retried (a ValueError is not an unreachable backend, see
        test_turn_retry.py), so it ends with no final. The batch was offered at
        that turn, so the hold is outstanding; the channel's next final with
        nothing outstanding ends the span. A message after that has its final
        checked, so a hold cannot leave the channel unchecked for good. A hold
        that cleared only on the revision's own trace would never clear."""
        env = await make_env({"the draft": OBJECTIONS, "later reply": ["still wrong"]})
        script(
            env.plugin, _make_text_response("the draft"), ValueError("bad request"),
            _make_text_response("in the span"), _make_text_response("later reply"),
            _make_text_response("later revision"),
        )

        await say(env)
        # The revising turn died: only the kernel's error notice went out (an
        # "error" emission, not a final), never the draft or a revision.
        assert "the draft" not in sent(env)
        assert len(sent(env)) == 1
        assert env.checker.checked == ["the draft"]
        assert len(env.trace.held()) == 1

        # The span ends at the channel's next final; the one after it is checked.
        await say(env, "anyone there")
        await say(env, "one more try")
        assert "later reply" in env.checker.checked
        assert len(env.trace.held()) == 2

    async def test_r6_an_interleaved_final_while_the_revisions_tools_run_does_not_end_the_span(
        self, make_env
    ):
        """R6 (tool-cycle span): the revising turn dispatches a tool; a user
        message that arrives meanwhile gets a final while the call is
        outstanding. That final does not end the span, so the revision's own
        final, after the result, is still unchecked. One hold."""
        env = await make_env({
            "the draft": OBJECTIONS, "interleaved reply": OBJECTIONS, "revised": OBJECTIONS,
        })
        release = slow_tool_setup(env.plugin)
        script(
            env.plugin, _make_text_response("the draft"), slow_call(),
            _make_text_response("interleaved reply"), _make_text_response("revised"),
        )

        await say(env)
        assert env.channel.pending_tool_call_ids
        await say(env, "are you there")
        release.set()
        await settle_all(env)

        assert env.checker.checked == ["the draft"]
        assert len(env.trace.held()) == 1
        assert sent(env)[-2:] == ["interleaved reply", "revised"]


# ---------------------------------------------------------------------------
# #1222: a revision queued behind another pending stub keeps its own text
# ---------------------------------------------------------------------------

#: Stands in for critique's wake note: text that must not stand in for the revision's.
CRITIQUE_NOTE = "CRITIQUE WAKE NOTE: reply NO_REPLY unless the findings change something"

#: A finding critique holds for its own origin's turn.
FINDING = "critique finding: the plan skips a step"


class TestRevisionBehindPendingStub:
    async def test_r1_r2_revision_row_reads_the_revision_stub_and_both_batches_are_admitted(
        self, make_env, P
    ):
        """#1222 R1/R2: a draft of a ``critique``-origin turn is refused while
        critique's wake stub is pending for the same channel. The revising turn
        (one turn) has the window's notification row read ``REVISION_STUB``,
        not critique's note (R1), and both critique's finding and the pre-send
        objections are admitted at it (R2)."""
        from corvidae.held import HeldContext
        from corvidae.hooks import get_dependency

        env = await make_env({"the draft": OBJECTIONS})
        held = get_dependency(env.plugin.pm, "held", HeldContext)

        async def critique_wakes() -> None:
            # Critique's wake lands while the draft is being written: its stub is pending.
            # Equivalent to critique's own call (critique.py, the waking-findings
            # hold_and_wake): same origin, source and a non-None stub text, no flag.
            await held.hold_and_wake(
                env.channel, "critique", "critique", [FINDING], stub_text=CRITIQUE_NOTE
            )

        script(
            env.plugin,
            Then(_make_text_response("the draft"), critique_wakes),
            _make_text_response("revised"),
        )

        # The held exchange is a critique-origin notification (no stub annotation).
        # Equivalent to a stub turn here: admit_for_turn keys on origin alone and
        # spends the pair either way, so critique's later stub is a fresh one.
        await notify(env, "critique", "critique", "tick")
        await asyncio.wait_for(drain(env.plugin, env.channel), 2)

        # One draft call and one revising call.
        assert env.plugin._client.chat.await_count == 2
        # R1: the stub row in the window and the revising prompt carry the revision's text.
        window = "\n".join(str(m.get("content")) for m in env.channel.conversation.messages)
        assert P.REVISION_STUB in window
        assert CRITIQUE_NOTE not in window
        assert P.REVISION_STUB in prompt_text(env.plugin, 1)
        assert CRITIQUE_NOTE not in prompt_text(env.plugin, 1)
        # R2: both batches reach the revising turn's prompt, and both were announced.
        assert FINDING in prompt_text(env.plugin, 1)
        assert OBJECTIONS[0] in prompt_text(env.plugin, 1)
        assert len(env.offers.of(P.SOURCE)) == 1
        assert len(env.offers.of("critique")) == 1
