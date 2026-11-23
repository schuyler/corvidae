"""Veto checks under a per-channel check allowance (#1162).

PRD principle 3, §3.7 Output gate 7, §3.13.5. The ``final`` firing of
``should_send_response`` runs every implementation concurrently under the
channel's check allowance. A verdict inside it decides as before; past it the
reply is sent, the miss is recorded (a debug line and an ``on_metrics``
counter), and a refusal that arrives late reaches the channel's window as a
framed CONTEXT row at its next turn without waking one.

The checks below sleep on a FakeClock, and each test moves time with
``advance``; the one deadline-instant test needs the real clock (50 ms).
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.channel import (
    ChannelConfig,
    ChannelRegistry,
    apply_channel_config,
    parse_channel_config,
    validate_agent_config,
)
from corvidae.checks import CHECK_MISSED_METRIC, resolve_check_allowance
from corvidae.context import LATE_CHECK_LABEL, NO_REASON_GIVEN, MessageType
from corvidae.hooks import Refusal, TransportPlugin, create_plugin_manager, hookimpl
from corvidae.outbound import OutboundMessage
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


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class ClockGate:
    """A should_send_response implementation that answers ``final`` after a
    fake-clock delay. ``plan`` is a list of (delay, verdict), one per final
    firing, in order; firings past the end answer None at once."""

    def __init__(self, clock, plan):
        self.clock = clock
        self.plan = list(plan)
        # Set when a final firing begins, so a test knows the check is running.
        self.started = asyncio.Event()
        # Count of firings that returned (in time or late).
        self.finished = 0

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations):
        if emission != "final":
            return None
        delay, verdict = self.plan.pop(0) if self.plan else (0, None)
        self.started.set()
        if delay:
            await self.clock.sleep(delay)
        self.finished += 1
        return verdict


class RaisingGate:
    """Raises on every final firing."""

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, annotations):
        if emission == "final":
            raise RuntimeError("gate exploded")
        return None


class MetricsSpy:
    """Records every on_metrics event as (name, value, tags)."""

    def __init__(self):
        self.events: list[tuple[str, float, dict]] = []

    @hookimpl
    async def on_metrics(self, name, value, tags):
        self.events.append((name, value, dict(tags)))

    def named(self, name):
        return [e for e in self.events if e[0] == name]


@pytest.fixture
async def env():
    """(plugin, channel, db, clock, metrics): an Agent on a channel whose
    transport declares no allowance, so it gets the minimum."""
    plugin, channel, db = await build_plugin_and_channel()
    clock = install_kernel_clock(plugin.pm)[0]
    metrics = MetricsSpy()
    plugin.pm.register(metrics, name="metrics_spy")
    plugin._client = MagicMock()
    plugin._client.chat = AsyncMock()
    plugin._mark_started()
    yield plugin, channel, db, clock, metrics
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await clock.close()
    await db.close()


def _script(plugin, *responses):
    """Make the LLM answer ``responses`` in call order."""
    plugin._client.chat.side_effect = list(responses)


async def _start_turn(plugin, channel, text="hello") -> asyncio.Task:
    """Deliver a user message and return a task that drains the queue."""
    await plugin.on_message(channel=channel, sender="user", text=text, trace_id=mint_trace_id())
    return asyncio.create_task(drain(plugin, channel))


async def _rows(db, channel_id):
    """(message dict, message_type) for every message_log row, in order."""
    async with db.execute(
        "SELECT message, message_type FROM message_log WHERE channel_id = ? ORDER BY id",
        (channel_id,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [(json.loads(m), t) for m, t in rows]


def _late_rows(messages):
    """Window rows that are late-check CONTEXT rows."""
    return [
        m for m in messages
        if m.get("_message_type") == MessageType.CONTEXT
        and isinstance(m.get("content"), str)
        and m["content"].startswith(f"[CONTEXT from {LATE_CHECK_LABEL}")
    ]


# ---------------------------------------------------------------------------
# The final firing under the allowance
# ---------------------------------------------------------------------------


class TestFinalUnderAllowance:
    async def test_verdict_in_time_vetoes(self, env):
        """A refusal that arrives inside the allowance withholds the reply."""
        plugin, channel, db, clock, metrics = env
        gate = ClockGate(clock, [(3, Refusal("too blunt"))])
        plugin.pm.register(gate, name="clock_gate")
        _script(plugin, _make_text_response("the reply"))

        drained = await _start_turn(plugin, channel)
        await asyncio.wait_for(gate.started.wait(), 1)
        await clock.advance(3)
        await asyncio.wait_for(drained, 1)

        assert sent_texts(plugin.pm, OutboundMessage) == []
        rows = await _rows(db, channel.id)
        assert [t for m, t in rows if m.get("role") == "assistant"] == ["unsent"]
        assert metrics.named(CHECK_MISSED_METRIC) == []

    async def test_slow_check_does_not_hold_the_reply_past_the_allowance(self, env):
        """The reply waits for the allowance and no longer; the check is left
        running, not cancelled."""
        plugin, channel, db, clock, metrics = env
        gate = ClockGate(clock, [(60, Refusal("too blunt"))])
        plugin.pm.register(gate, name="clock_gate")
        _script(plugin, _make_text_response("the reply"))

        drained = await _start_turn(plugin, channel)
        await asyncio.wait_for(gate.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM - 0.5)
        # Not before: given real time to finish, the turn still waits.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(drained), 0.2)
        assert sent_texts(plugin.pm, OutboundMessage) == []

        await clock.advance(0.5)
        await asyncio.wait_for(drained, 1)
        assert sent_texts(plugin.pm, OutboundMessage) == ["the reply"]
        rows = await _rows(db, channel.id)
        assert [t for m, t in rows if m.get("role") == "assistant"] == ["message"]

        # Still running; it finishes when its own time comes.
        assert gate.finished == 0
        await clock.advance(60)
        assert gate.finished == 1

    async def test_miss_emits_the_metric(self, env):
        """The miss is counted at the deadline, tagged channel, hook, plugin."""
        plugin, channel, db, clock, metrics = env
        gate = ClockGate(clock, [(60, None)])
        plugin.pm.register(gate, name="clock_gate")
        _script(plugin, _make_text_response("the reply"))

        drained = await _start_turn(plugin, channel)
        await asyncio.wait_for(gate.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(drained, 1)

        assert metrics.named(CHECK_MISSED_METRIC) == [
            (CHECK_MISSED_METRIC, 1, {
                "channel": channel.id,
                "hook": "should_send_response",
                "plugin": "clock_gate",
                "phase": "before_effect",
            }),
        ]

    async def test_wrapper_sees_the_in_time_verdicts(self, env):
        """A wrapper=True implementation still wraps the final firing, as
        outbound-cap's counter does: it sees the verdicts that arrived in
        time, and what it returns is what the Agent resolves."""
        plugin, channel, db, clock, metrics = env
        seen: list[list] = []

        class Wrapper:
            @hookimpl(wrapper=True, specname="should_send_response")
            def wrap(self, emission):
                results = yield
                seen.append(list(results))
                # Turn a pass into a veto, so the Agent visibly uses the
                # wrapper's list.
                return [*results, False]

        plugin.pm.register(Wrapper(), name="wrapper")
        plugin.pm.register(ClockGate(clock, [(0, True)]), name="accepting_gate")
        _script(plugin, _make_text_response("the reply"))

        drained = await _start_turn(plugin, channel)
        await asyncio.wait_for(drained, 1)

        assert seen == [[True]]
        assert sent_texts(plugin.pm, OutboundMessage) == []

    async def test_raising_implementation_leaves_siblings_intact(self, env, caplog):
        """One implementation raising does not discard another's veto."""
        plugin, channel, db, clock, metrics = env
        plugin.pm.register(RaisingGate(), name="raising_gate")
        plugin.pm.register(ClockGate(clock, [(0, False)]), name="vetoing_gate")
        _script(plugin, _make_text_response("the reply"))

        with caplog.at_level(logging.WARNING):
            drained = await _start_turn(plugin, channel)
            await asyncio.wait_for(drained, 1)

        assert sent_texts(plugin.pm, OutboundMessage) == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("raising_gate" in r.getMessage() for r in warnings)


class TestDeadlineInstant:
    async def test_check_finishing_with_the_deadline_is_a_counted_miss(self):
        """Regression: a check that finishes in the same loop step as the
        deadline, after the deadline fires, decides nothing and is still
        counted as a miss. The fake clock separates two releases with a
        settle, so this drives gather on the real clock with a releaser
        timer due at the deadline's own instant (50 ms)."""
        from types import SimpleNamespace

        from corvidae.checks import gather
        from corvidae.clock import Clock

        release = asyncio.Event()

        class Slow:
            @hookimpl
            async def should_send_response(self, channel, text, emission, trace_id, annotations):
                await release.wait()
                return Refusal("on the line")

        pm = create_plugin_manager()
        metrics = MetricsSpy()
        pm.register(Slow(), name="slow")
        pm.register(metrics, name="metrics_spy")
        clock = Clock()
        late: list = []

        async def on_late(plugin_name, verdict):
            late.append((plugin_name, verdict))

        async def releaser():
            release.set()

        # Schedule the releaser right after the deadline timer, same delay.
        real_call_later = clock.call_later

        def call_later(delay, callback, *, name):
            timer = real_call_later(delay, callback, name=name)
            real_call_later(delay, releaser, name="releaser")
            return timer

        clock.call_later = call_later
        channel = SimpleNamespace(id="x:y")
        try:
            gathered = await gather(
                pm, "should_send_response", channel=channel,
                kwargs={"channel": channel, "text": "hi", "emission": "final",
                        "trace_id": None, "annotations": {}},
                allowance=0.05, clock=clock, on_late=on_late, phase="before_effect",
            )
            results = gathered.results
            await asyncio.sleep(0.05)
        finally:
            await clock.close()

        assert results == []
        assert late == [("slow", Refusal("on the line"))]
        assert len(metrics.named(CHECK_MISSED_METRIC)) == 1


# ---------------------------------------------------------------------------
# A late refusal reaches the window at the next turn
# ---------------------------------------------------------------------------


class TestLateRefusal:
    async def test_late_refusal_lands_as_context_and_wakes_no_turn(self, env):
        plugin, channel, db, clock, metrics = env
        gate = ClockGate(clock, [(60, Refusal("the date in that reply is wrong"))])
        plugin.pm.register(gate, name="clock_gate")
        _script(plugin, _make_text_response("the reply"), _make_text_response("second reply"))

        drained = await _start_turn(plugin, channel)
        await asyncio.wait_for(gate.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(drained, 1)
        assert sent_texts(plugin.pm, OutboundMessage) == ["the reply"]

        # The check returns late: no turn runs, nothing is queued, and the
        # window does not change until the channel's next turn.
        window_before = len(channel.conversation.messages)
        await clock.advance(60)
        await drain(plugin, channel)
        assert gate.finished == 1
        assert plugin._client.chat.await_count == 1
        assert len(channel.conversation.messages) == window_before
        assert _late_rows(channel.conversation.messages) == []

        # The next turn carries it, after that turn's incoming message.
        drained = await _start_turn(plugin, channel, text="next message")
        await asyncio.wait_for(drained, 1)
        late = _late_rows(channel.conversation.messages)
        assert len(late) == 1
        assert "the date in that reply is wrong" in late[0]["content"]
        assert "clock_gate" in late[0]["content"]
        messages = channel.conversation.messages
        idx = messages.index(late[0])
        assert messages[idx - 1]["content"] == "next message"
        # The model saw it on that turn.
        prompt = plugin._client.chat.await_args_list[1].args[0]
        assert any("the date in that reply is wrong" in str(m.get("content")) for m in prompt)
        # Persisted as CONTEXT, so it reloads with the window.
        rows = await _rows(db, channel.id)
        assert any(
            t == "context" and "the date in that reply is wrong" in m["content"]
            for m, t in rows
        )

    async def test_late_bare_false_reaches_the_next_turn(self, env):
        """A late veto with no text still reaches the channel (#1201): one
        late row naming the plugin and saying no reason was given, as a late
        tool veto's does (PRD §3.1.21)."""
        plugin, channel, db, clock, metrics = env
        gate = ClockGate(clock, [(60, False)])
        plugin.pm.register(gate, name="clock_gate")
        _script(plugin, _make_text_response("the reply"), _make_text_response("second reply"))

        drained = await _start_turn(plugin, channel)
        await asyncio.wait_for(gate.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(drained, 1)
        await clock.advance(60)
        drained = await _start_turn(plugin, channel, text="next message")
        await asyncio.wait_for(drained, 1)

        late = _late_rows(channel.conversation.messages)
        assert len(late) == 1
        assert "clock_gate" in late[0]["content"]
        assert NO_REASON_GIVEN in late[0]["content"]
        assert len(metrics.named(CHECK_MISSED_METRIC)) == 1

    @pytest.mark.parametrize("cancelled", [False, True], ids=["pending", "pending-cleared"])
    async def test_late_row_never_lands_between_a_call_and_its_result(self, env, cancelled):
        """A late refusal that returns while a tool runs waits out the tool:
        an interleaved user turn does not deliver it, whether the call is
        still pending or the pending set was cleared (ACP's cancel_prompt);
        the turn after the result does."""
        plugin, channel, db, clock, metrics = env
        gate = ClockGate(clock, [(60, Refusal("late finding"))])
        plugin.pm.register(gate, name="clock_gate")
        release = asyncio.Event()

        async def slow_tool(x: str) -> str:
            """A tool that runs until the test releases it."""
            await release.wait()
            return f"result:{x}"

        plugin._tools = {"slow_tool": slow_tool}
        _script(
            plugin,
            _make_text_response("the reply"),
            _make_tool_call_response([_make_tool_call("call-1", "slow_tool", {"x": "a"})]),
            _make_text_response("answer while the tool runs"),
            _make_text_response("after the tool"),
        )

        # Turn 1: the reply goes out; the check misses.
        drained = await _start_turn(plugin, channel)
        await asyncio.wait_for(gate.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(drained, 1)

        # Turn 2 dispatches the slow tool; the check returns late meanwhile.
        drained = await _start_turn(plugin, channel, text="use the tool")
        await asyncio.wait_for(drained, 1)
        await clock.advance(60)
        assert gate.finished == 1

        # Turn 3 interleaves while the call is outstanding.
        if cancelled:
            channel.pending_tool_call_ids.clear()
        drained = await _start_turn(plugin, channel, text="are you there")
        await asyncio.wait_for(drained, 1)
        assert _late_rows(channel.conversation.messages) == []

        # The result arrives; its turn delivers the late row after it.
        release.set()
        await drain_task_queue(plugin)
        await drain(plugin, channel)
        messages = channel.conversation.messages
        late = _late_rows(messages)
        assert len(late) == 1
        result_idx = next(i for i, m in enumerate(messages) if m.get("tool_call_id") == "call-1")
        call_idx = next(i for i, m in enumerate(messages) if m.get("tool_calls"))
        late_idx = messages.index(late[0])
        assert call_idx < result_idx < late_idx


# ---------------------------------------------------------------------------
# Allowance lookup: operator, then transport, then the minimum
# ---------------------------------------------------------------------------


class _DeclaringTransport(TransportPlugin):
    """A transport that declares a 90-second default for its channels."""

    transport = "declaring"
    check_allowance = 90


class _SilentTransport(TransportPlugin):
    """A transport that declares no default."""

    transport = "silent"


def _lookup_env(channels: dict | None = None, agent: dict | None = None):
    """(pm, registry) with both test transports loaded and ``channels`` /
    ``agent`` applied as agent.yaml would apply them."""
    pm = create_plugin_manager()
    registry = ChannelRegistry()
    registry.agent_defaults = dict(agent or {})
    pm.register(registry, name="registry")
    pm.register(_DeclaringTransport(), name="declaring")
    pm.register(_SilentTransport(), name="silent")
    apply_channel_config(parse_channel_config({"channels": channels or {}}), registry)
    return pm, registry


class TestAllowanceLookup:
    def test_operator_value_wins(self):
        pm, registry = _lookup_env({"declaring:room": {"check_allowance": 30}})
        channel = registry.get("declaring:room")
        assert resolve_check_allowance(pm, registry, channel) == 30

    def test_operator_value_from_a_transport_wide_entry(self):
        pm, registry = _lookup_env({"declaring:*": {"check_allowance": 40}})
        channel = registry.get_or_create("declaring", "room", config=ChannelConfig())
        assert resolve_check_allowance(pm, registry, channel) == 40

    def test_transport_default_wins_over_the_minimum(self):
        pm, registry = _lookup_env()
        channel = registry.get_or_create("declaring", "room", config=ChannelConfig())
        assert resolve_check_allowance(pm, registry, channel) == 90

    def test_minimum_when_nothing_is_declared(self):
        pm, registry = _lookup_env()
        channel = registry.get_or_create("silent", "room", config=ChannelConfig())
        assert resolve_check_allowance(pm, registry, channel) == DEFAULT_MINIMUM

    def test_operator_sets_the_minimum(self):
        pm, registry = _lookup_env(agent={"check_allowance_min": 7})
        channel = registry.get_or_create("silent", "room", config=ChannelConfig())
        assert resolve_check_allowance(pm, registry, channel) == 7

    def test_never_below_the_minimum(self):
        pm, registry = _lookup_env({"declaring:room": {"check_allowance": 2}})
        channel = registry.get("declaring:room")
        assert resolve_check_allowance(pm, registry, channel) == DEFAULT_MINIMUM


class TestAllowanceValidation:
    @pytest.mark.parametrize("bad", ["fast", 0, -3, True])
    def test_channel_value_rejected_names_key_value_and_form(self, bad):
        with pytest.raises(ValueError) as exc:
            parse_channel_config({"channels": {"irc:#lex": {"check_allowance": bad}}})
        message = str(exc.value)
        assert "channels.irc:#lex.check_allowance" in message
        assert repr(bad) in message
        assert "positive number of seconds" in message

    @pytest.mark.parametrize("bad", ["soon", 0, False])
    def test_minimum_rejected_names_key_value_and_form(self, bad):
        with pytest.raises(ValueError) as exc:
            validate_agent_config({"check_allowance_min": bad})
        message = str(exc.value)
        assert "agent.check_allowance_min" in message
        assert repr(bad) in message
        assert "positive number of seconds" in message


class TestTransportDefaults:
    """The owner's table: messaging transports one to two minutes, interactive
    ones about ten seconds."""

    @pytest.mark.parametrize(
        "module, name",
        [("corvidae.channels.irc", "IRCPlugin"),
         ("corvidae.channels.signal", "SignalPlugin"),
         ("corvidae.channels.discord", "DiscordPlugin"),
         ("corvidae.channels.zulip", "ZulipPlugin")],
    )
    def test_messaging_transports(self, module, name):
        cls = getattr(pytest.importorskip(module), name)
        assert 60 <= cls.check_allowance <= 120

    @pytest.mark.parametrize(
        "module, name",
        [("corvidae.channels.web", "WebPlugin"),
         ("corvidae.channels.acp", "AcpPlugin"),
         ("corvidae.channels.cli", "CLIPlugin")],
    )
    def test_interactive_transports(self, module, name):
        cls = getattr(pytest.importorskip(module), name)
        assert cls.check_allowance == 10
