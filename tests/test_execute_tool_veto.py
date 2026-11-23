"""should_execute_tool: a per-call tool veto (#148).

PRD §3.5.12, §3.13.5. A plugin can refuse a tool call before it runs. Every
implementation runs under the channel's check allowance through
``corvidae.checks.gather``; any refusal wins and becomes the call's
result, so the turn continues. An implementation that raises counts as no
refusal. When the allowance runs out the call runs unchecked and the miss is
recorded; a refusal that arrives later reaches the channel's window at its
next turn as framed CONTEXT and wakes no turn. With no implementation loaded,
nothing is refused.

The checks sleep on a FakeClock; each test moves time with ``advance``.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fake_clock import install_kernel_clock
from helpers import build_plugin_and_channel, drain, drain_task_queue
from llm_response_fixtures import (
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.checks import CHECK_MISSED_METRIC
from corvidae.context import LATE_CHECK_LABEL, MessageType
from corvidae.hooks import Refusal, hookimpl
from corvidae.tool import tool_to_schema
from corvidae.tools.subagent import run_agent_loop
from corvidae.trace import mint_trace_id

#: The minimum allowance a channel gets when nothing else sets one (seconds).
DEFAULT_MINIMUM = 5.0


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class Veto:
    """A should_execute_tool implementation that records each offer and
    answers ``verdict`` after ``delay`` seconds of fake-clock time."""

    def __init__(self, verdict, *, clock=None, delay: float = 0):
        self.verdict = verdict
        self.clock = clock
        self.delay = delay
        # (channel, tool_name, arguments, trace_id) per offer.
        self.offers: list[tuple] = []
        # Set when an offer begins, so a test knows the check is running.
        self.started = asyncio.Event()
        # Count of offers that returned (in time or late).
        self.finished = 0

    @hookimpl
    async def should_execute_tool(self, channel, tool_name, arguments, trace_id):
        self.offers.append((channel, tool_name, arguments, trace_id))
        self.started.set()
        if self.delay:
            await self.clock.sleep(self.delay)
        self.finished += 1
        return self.verdict


class RaisingVeto:
    """Raises on every offer."""

    @hookimpl
    async def should_execute_tool(self, channel, tool_name, arguments, trace_id):
        raise RuntimeError("veto exploded")


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
    """(plugin, channel, clock, metrics, calls): an Agent with a ``shell``
    stand-in that records each run, on a channel that gets the minimum
    allowance."""
    plugin, channel, db = await build_plugin_and_channel()
    clock = install_kernel_clock(plugin.pm)[0]
    metrics = MetricsSpy()
    plugin.pm.register(metrics, name="metrics_spy")
    calls: list[str] = []

    async def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return f"ran {command}"

    plugin._tools = {"shell": shell}
    plugin._tool_schemas = [tool_to_schema(shell)]
    plugin._client = MagicMock()
    plugin._client.chat = AsyncMock()
    plugin._mark_started()
    yield plugin, channel, clock, metrics, calls
    await plugin.pm.get_plugin("task").on_stop()
    await clock.close()
    await db.close()


def _script(plugin, *responses):
    """Make the LLM answer ``responses`` in call order."""
    plugin._client.chat.side_effect = list(responses)


def _shell_call(args=None):
    """A model response that calls shell once."""
    return _make_tool_call_response(
        [_make_tool_call("call_1", "shell", args if args is not None else {"command": "id"})]
    )


async def _start_turn(plugin, channel, text="go", trace_id=None) -> asyncio.Task:
    """Deliver a user message; return a task that drains queue, tools, queue."""
    await plugin.on_message(
        channel=channel, sender="user", text=text, trace_id=trace_id or mint_trace_id()
    )

    async def run():
        await drain(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)

    return asyncio.create_task(run())


def _tool_result(plugin, call_index: int = 1) -> str:
    """The tool-role content the LLM call at ``call_index`` carried."""
    messages = plugin._client.chat.call_args_list[call_index].args[0]
    (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
    return tool_msg["content"]


def _late_rows(messages):
    """Window rows that are late-check CONTEXT rows."""
    return [
        m for m in messages
        if m.get("_message_type") == MessageType.CONTEXT
        and isinstance(m.get("content"), str)
        and m["content"].startswith(f"[CONTEXT from {LATE_CHECK_LABEL}")
    ]


# ---------------------------------------------------------------------------
# Refusal in time
# ---------------------------------------------------------------------------


class TestRefusal:
    async def test_refused_call_does_not_run_and_its_result_says_why(self, env, caplog):
        """The tool does not run; the model gets a result naming the tool,
        the reason and the next step; the turn continues; the log names the
        tool, the channel and the reason."""
        plugin, channel, clock, metrics, calls = env
        plugin.pm.register(Veto(Refusal("a guest is present")), name="guest_veto")
        _script(plugin, _shell_call(), _make_text_response("done"))

        with caplog.at_level(logging.INFO, logger="corvidae"):
            await asyncio.wait_for(await _start_turn(plugin, channel), 2)

        assert calls == []
        result = _tool_result(plugin)
        assert "'shell'" in result
        assert "a guest is present" in result
        assert "did not run" in result
        assert "Don't retry it unchanged" in result
        # The turn continued to its final text.
        assert plugin._client.chat.await_count == 2
        logged = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            "shell" in m and channel.id in m and "a guest is present" in m for m in logged
        ), logged

    async def test_any_refusal_wins(self, env):
        """One refusal beats an allow and a no-opinion. Registered first, so
        pluggy offers it last and its verdict is not the list's head."""
        plugin, channel, clock, metrics, calls = env
        plugin.pm.register(Veto(Refusal("not on this channel")), name="refuse")
        plugin.pm.register(Veto(None), name="no_opinion")
        plugin.pm.register(Veto(True), name="allow")
        _script(plugin, _shell_call(), _make_text_response("done"))

        await asyncio.wait_for(await _start_turn(plugin, channel), 2)

        assert calls == []
        assert "not on this channel" in _tool_result(plugin)

    async def test_refusal_skips_transform_tool_result(self, env):
        """A refusal is a pre-dispatch error: no tool output exists to transform."""
        plugin, channel, clock, metrics, calls = env
        transformed: list[str] = []

        class Transformer:
            @hookimpl
            async def transform_tool_result(self, tool_name, result, channel):
                transformed.append(tool_name)
                return None

        plugin.pm.register(Transformer(), name="transformer")
        plugin.pm.register(Veto(Refusal("no")), name="refuse")
        _script(plugin, _shell_call(), _make_text_response("done"))

        await asyncio.wait_for(await _start_turn(plugin, channel), 2)

        assert calls == []
        assert transformed == []

    async def test_bare_false_refuses_and_says_no_reason_was_given(self, env):
        plugin, channel, clock, metrics, calls = env
        plugin.pm.register(Veto(False), name="mute")
        _script(plugin, _shell_call(), _make_text_response("done"))

        await asyncio.wait_for(await _start_turn(plugin, channel), 2)

        assert calls == []
        result = _tool_result(plugin)
        assert "'shell'" in result
        assert "no reason given" in result


# ---------------------------------------------------------------------------
# What is offered
# ---------------------------------------------------------------------------


class TestOffer:
    async def test_offered_the_channel_tool_validated_arguments_and_trace(self, env):
        plugin, channel, clock, metrics, calls = env
        veto = Veto(None)
        plugin.pm.register(veto, name="recorder")
        trace = mint_trace_id()
        _script(plugin, _shell_call({"command": "ls"}), _make_text_response("done"))

        await asyncio.wait_for(await _start_turn(plugin, channel, trace_id=trace), 2)

        assert veto.offers == [(channel, "shell", {"command": "ls"}, trace)]
        assert calls == ["ls"]

    async def test_arguments_that_fail_validation_are_not_offered(self, env):
        """The hook sees validated arguments only (§3.5.1): a call whose
        arguments miss its schema is rejected before the offer."""
        plugin, channel, clock, metrics, calls = env
        veto = Veto(Refusal("never asked"))
        plugin.pm.register(veto, name="recorder")
        _script(plugin, _shell_call({"cmd": "ls"}), _make_text_response("done"))

        await asyncio.wait_for(await _start_turn(plugin, channel), 2)

        assert veto.offers == []
        assert "never asked" not in _tool_result(plugin)

    async def test_a_subagent_call_is_offered_and_refused(self, env):
        """run_agent_loop's dispatch goes through the Agent's check, with the
        calling channel and the parent turn's trace. (The subagent and fork
        tools hand it the check: test_subagent.py, test_fork_tool.py.)"""
        plugin, channel, clock, metrics, calls = env
        veto = Veto(Refusal("no shell for subagents here"))
        plugin.pm.register(veto, name="sub_veto")
        client = MagicMock()
        client.chat = AsyncMock(side_effect=[_shell_call(), _make_text_response("sub done")])
        messages: list[dict] = [{"role": "user", "content": "do it"}]
        trace = mint_trace_id()

        token = set_attribution(channel_id=channel.id, trace_id=trace)
        try:
            final = await asyncio.wait_for(
                run_agent_loop(
                    client, messages, plugin._tools, plugin._tool_schemas,
                    channel=channel, check=plugin.check_tool_call,
                ),
                2,
            )
        finally:
            reset_attribution(token)

        assert final == "sub done"
        assert calls == []
        assert veto.offers == [(channel, "shell", {"command": "id"}, trace)]
        (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
        assert "'shell'" in tool_msg["content"]
        assert "no shell for subagents here" in tool_msg["content"]


# ---------------------------------------------------------------------------
# Failure and absence
# ---------------------------------------------------------------------------


class TestFailOpen:
    async def test_raising_implementation_is_logged_and_counts_as_no_refusal(self, env, caplog):
        plugin, channel, clock, metrics, calls = env
        plugin.pm.register(RaisingVeto(), name="raising_veto")
        _script(plugin, _shell_call(), _make_text_response("done"))

        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(await _start_turn(plugin, channel), 2)

        assert calls == ["id"]
        assert _tool_result(plugin).endswith("] ran id")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "should_execute_tool" in r.getMessage()
            and "raising_veto" in r.getMessage()
            and channel.id in r.getMessage()
            and "no opinion" in r.getMessage()
            and r.exc_info
            for r in warnings
        ), [r.getMessage() for r in warnings]

    async def test_no_plugin_refuses_nothing(self, env):
        plugin, channel, clock, metrics, calls = env
        _script(plugin, _shell_call(), _make_text_response("done"))

        await asyncio.wait_for(await _start_turn(plugin, channel), 2)

        assert calls == ["id"]
        assert _tool_result(plugin).endswith("] ran id")
        assert metrics.named(CHECK_MISSED_METRIC) == []


# ---------------------------------------------------------------------------
# The allowance
# ---------------------------------------------------------------------------


class TestAllowance:
    async def test_call_runs_unchecked_when_the_allowance_runs_out(self, env, caplog):
        """The call waits for the allowance and no longer, then runs; the
        miss is a debug line and a check.missed metric naming the hook."""
        plugin, channel, clock, metrics, calls = env
        veto = Veto(Refusal("too slow to matter"), clock=clock, delay=60)
        plugin.pm.register(veto, name="slow_veto")
        _script(plugin, _shell_call(), _make_text_response("done"))

        with caplog.at_level(logging.DEBUG, logger="corvidae.checks"):
            turn = await _start_turn(plugin, channel)
            await asyncio.wait_for(veto.started.wait(), 1)
            await clock.advance(DEFAULT_MINIMUM - 0.5)
            await asyncio.sleep(0.05)
            assert calls == []
            await clock.advance(0.5)
            await asyncio.wait_for(turn, 2)

        assert calls == ["id"]
        assert _tool_result(plugin).endswith("] ran id")
        assert metrics.named(CHECK_MISSED_METRIC) == [
            (CHECK_MISSED_METRIC, 1, {
                "channel": channel.id,
                "hook": "should_execute_tool",
                "plugin": "slow_veto",
                "phase": "before_effect",
            }),
        ]
        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(
            "should_execute_tool" in m and "slow_veto" in m and "missed" in m for m in debug
        ), debug
        # Left running, not cancelled.
        assert veto.finished == 0

    async def test_late_refusal_reaches_the_next_turn_and_wakes_none(self, env):
        plugin, channel, clock, metrics, calls = env
        veto = Veto(Refusal("that command touched the prod host"), clock=clock, delay=60)
        plugin.pm.register(veto, name="slow_veto")
        _script(
            plugin, _shell_call(), _make_text_response("done"),
            _make_text_response("second reply"),
        )

        turn = await _start_turn(plugin, channel)
        await asyncio.wait_for(veto.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(turn, 2)
        assert calls == ["id"]

        # The refusal arrives late: nothing runs, nothing changes yet.
        window_before = len(channel.conversation.messages)
        await clock.advance(60)
        await drain(plugin, channel)
        assert veto.finished == 1
        assert plugin._client.chat.await_count == 2
        assert len(channel.conversation.messages) == window_before

        # The next turn carries it as framed CONTEXT.
        await asyncio.wait_for(await _start_turn(plugin, channel, text="next"), 2)
        (late,) = _late_rows(channel.conversation.messages)
        content = late["content"]
        assert f"[{LATE_CHECK_LABEL}: slow_veto]" in content
        assert "'shell'" in content
        assert "ran unchecked" in content
        assert "that command touched the prod host" in content
        prompt = plugin._client.chat.await_args_list[2].args[0]
        assert any("that command touched the prod host" in str(m.get("content")) for m in prompt)

    async def test_late_bare_false_reaches_the_next_turn_with_no_reason(self, env):
        """A bare False refuses (it is a refusal with no reason), so a late one
        reaches the next turn too, saying no reason was given."""
        plugin, channel, clock, metrics, calls = env
        veto = Veto(False, clock=clock, delay=60)
        plugin.pm.register(veto, name="slow_veto")
        _script(
            plugin, _shell_call(), _make_text_response("done"),
            _make_text_response("second reply"),
        )

        turn = await _start_turn(plugin, channel)
        await asyncio.wait_for(veto.started.wait(), 1)
        await clock.advance(DEFAULT_MINIMUM)
        await asyncio.wait_for(turn, 2)
        await clock.advance(60)
        await asyncio.wait_for(await _start_turn(plugin, channel, text="next"), 2)

        (late,) = _late_rows(channel.conversation.messages)
        content = late["content"]
        assert f"[{LATE_CHECK_LABEL}: slow_veto]" in content
        assert "'shell'" in content
        assert "ran unchecked" in content
        assert "no reason given" in content
