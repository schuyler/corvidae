"""Tests for the outbound_cap plugin (#791).

The hooks are driven directly: a real plugin manager with a real
OutboundCapPlugin, a real ChannelRegistry, a stub transport that owns chosen
channel ids, and a recording hookimpl that calls the gate the way the kernel
does (``resolve_reject_wins``), so a "veto" asserted here is the same
resolution the kernel performs.
"""

from __future__ import annotations

import logging

import pytest
from outbound_cap import (
    COUNT_PROGRESS_KEY,
    MAX_MESSAGES_KEY,
    OutboundCapPlugin,
    parse_count_progress,
    parse_max_messages,
)

from corvidae.channel import ChannelRegistry
from corvidae.hooks import (
    TransportPlugin,
    create_plugin_manager,
    resolve_reject_wins,
)

AGENT_DEFAULTS = {"max_turns": 20}


class FakeTransport(TransportPlugin):
    """A transport that owns a fixed set of channel ids."""

    def __init__(self, name: str, ids: set[str]) -> None:
        super().__init__()
        self.transport = name
        self._ids = ids

    def owns(self, channel) -> bool:
        return channel.id in self._ids


class Gate:
    """Calls should_send_response and resolves it as the kernel does."""

    def __init__(self, pm) -> None:
        self._pm = pm

    async def send(self, channel, text="hi", emission="final", trace_id="t1") -> bool:
        results = await self._pm.ahook.should_send_response(
            channel=channel,
            text=text,
            emission=emission,
            trace_id=trace_id,
            annotations={},
        )
        # The kernel's rule: any False vetoes, anything else sends.
        return resolve_reject_wins(results) is not False


class Harness:
    def __init__(self) -> None:
        self.pm = create_plugin_manager()
        self.registry = ChannelRegistry(AGENT_DEFAULTS)
        self.pm.register(self.registry, name="registry")
        self.plugin = OutboundCapPlugin()
        self.pm.register(self.plugin, name="outbound_cap")
        self.pm.register(FakeTransport("signal", {"signal:aci1", "signal:aci2"}), name="signal")
        self.gate = Gate(self.pm)

    async def init(self, config: dict | None = None) -> None:
        await self.plugin.on_init(self.pm, config or {})

    def channel(self, channel_id: str):
        return self.registry.get_or_create(channel_id.split(":")[0], channel_id.split(":", 1)[1])

    async def admitted(self, channel) -> None:
        await self.plugin.on_message_admitted(
            channel=channel, trace_id="t0", sender="u", text="hello"
        )


CONFIG = {"outbound_cap": {"max_messages": 2}}


@pytest.fixture
async def h():
    harness = Harness()
    await harness.init(CONFIG)
    return harness


# ---------------------------------------------------------------------------
# The acceptance cases from #791 requirement 6
# ---------------------------------------------------------------------------


async def test_messages_under_the_cap_are_sent(h):
    """Two messages with a limit of two both reach the transport."""
    channel = h.channel("signal:aci1")
    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is True


async def test_cap_reached_then_the_next_emission_is_vetoed(h):
    """The third message with no human message in between is withheld."""
    channel = h.channel("signal:aci1")
    for _ in range(2):
        assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is False


async def test_vetoed_emissions_stay_vetoed_until_a_reset(h):
    """Once capped, every later emission is withheld, not just the next one."""
    channel = h.channel("signal:aci1")
    for _ in range(4):
        await h.gate.send(channel)
    assert await h.gate.send(channel) is False
    assert await h.gate.send(channel, emission="progress") is False


async def test_a_human_message_resets_the_count(h):
    """on_message_admitted returns the channel to a clean counter."""
    channel = h.channel("signal:aci1")
    for _ in range(3):
        await h.gate.send(channel)
    assert await h.gate.send(channel) is False

    await h.admitted(channel)
    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is True


async def test_a_coalesced_burst_is_one_reset(h):
    """Several admitted messages before the turn runs leave the counter at 0."""
    channel = h.channel("signal:aci1")
    await h.gate.send(channel)
    await h.gate.send(channel)
    assert await h.gate.send(channel) is False

    # The burst: three admissions, one turn, one reply.
    for _ in range(3):
        await h.admitted(channel)
    assert await h.gate.send(channel) is True


async def test_counting_is_per_channel(h):
    """One channel reaching the cap does not withhold another's messages."""
    first = h.channel("signal:aci1")
    second = h.channel("signal:aci2")
    for _ in range(3):
        await h.gate.send(first)
    assert await h.gate.send(first) is False
    assert await h.gate.send(second) is True


async def test_an_uncapped_channel_is_never_capped():
    """No outbound_cap key means no counting and no veto."""
    harness = Harness()
    await harness.init({})
    channel = harness.channel("signal:aci1")
    for _ in range(20):
        assert await harness.gate.send(channel) is True


async def test_an_internal_channel_is_never_capped(h):
    """A channel no transport owns — internal:heartbeat — is the agent
    talking to itself, so it is exempt even with a limit configured."""
    channel = h.channel("internal:heartbeat")
    for _ in range(5):
        assert await h.gate.send(channel) is True


async def test_progress_does_not_count_by_default(h):
    """A progress emission is sent and not counted, so the cap is still two
    finals; the tool-using reply is not penalised."""
    channel = h.channel("signal:aci1")
    assert await h.gate.send(channel, emission="progress") is True
    assert await h.gate.send(channel, emission="progress") is True
    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is False


async def test_progress_counts_when_configured():
    """With count_progress true, a progress emission spends the budget."""
    harness = Harness()
    await harness.init({"outbound_cap": {"max_messages": 2, "count_progress": True}})
    channel = harness.channel("signal:aci1")
    assert await harness.gate.send(channel, emission="progress") is True
    assert await harness.gate.send(channel, emission="progress") is True
    assert await harness.gate.send(channel) is False


async def test_thinking_and_error_emissions_never_count(h):
    """Only finals (and progress, if configured) are messages."""
    channel = h.channel("signal:aci1")
    for _ in range(5):
        assert await h.gate.send(channel, emission="thinking") is True
    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is False


# ---------------------------------------------------------------------------
# Requirement 4: fail soft but loud
# ---------------------------------------------------------------------------


async def test_first_veto_warns_with_the_key_and_the_count(h, caplog):
    """The first withheld emission names the channel, emission, trace, count
    and the config key that set the limit."""
    channel = h.channel("signal:aci1")
    for _ in range(2):  # fill the cap; nothing is withheld yet
        await h.gate.send(channel)

    with caplog.at_level(logging.DEBUG, logger="corvidae.outbound_cap"):
        await h.gate.send(channel, trace_id="trace-9")

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "signal:aci1" in warnings[0]
    assert "final" in warnings[0]
    assert "trace-9" in warnings[0]
    assert MAX_MESSAGES_KEY in warnings[0]
    assert "limit 2" in warnings[0]


async def test_later_vetoes_log_at_info(h, caplog):
    """Only the first veto in a run warns; the rest are INFO."""
    channel = h.channel("signal:aci1")
    for _ in range(2):  # fill the cap
        await h.gate.send(channel)
    await h.gate.send(channel)  # the first veto: its WARNING is spent here

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="corvidae.outbound_cap"):
        for _ in range(3):
            await h.gate.send(channel)

    levels = [r.levelno for r in caplog.records if "outbound_cap" in r.name]
    assert levels.count(logging.WARNING) == 0, levels
    assert levels.count(logging.INFO) == 3, levels
    # Each line still names the channel, and the running count withheld.
    assert all("signal:aci1" in r.getMessage() for r in caplog.records), caplog.records
    assert "4 withheld" in caplog.records[-1].getMessage()


async def test_reset_warns_how_many_were_withheld(h, caplog):
    """The summary on reset says how many messages were held."""
    channel = h.channel("signal:aci1")
    for _ in range(5):
        await h.gate.send(channel)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="corvidae.outbound_cap"):
        await h.admitted(channel)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "signal:aci1" in warnings[0]
    assert "3 messages were withheld" in warnings[0]


async def test_a_reset_with_nothing_withheld_is_quiet(h, caplog):
    """Ordinary conversation never fills the log with cap lines."""
    channel = h.channel("signal:aci1")
    await h.gate.send(channel)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="corvidae.outbound_cap"):
        await h.admitted(channel)

    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []


# ---------------------------------------------------------------------------
# Requirement 5: config, and the errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, 2.0, "3", [], {}])
def test_bad_limits_are_rejected_naming_key_and_range(value):
    """0, negatives, bools, floats (even whole), strings and containers fail."""
    with pytest.raises(ValueError) as excinfo:
        parse_max_messages(value)

    message = str(excinfo.value)
    assert MAX_MESSAGES_KEY in message, message
    assert repr(value) in message, message
    assert "at least 1" in message, message


@pytest.mark.parametrize("value", [1, 5, 1000])
def test_good_limits_are_accepted(value):
    assert parse_max_messages(value) == value


def test_absent_limit_means_no_cap():
    assert parse_max_messages(None) is None


@pytest.mark.parametrize("value", ["true", 1, 0, [], {}])
def test_bad_count_progress_is_rejected(value):
    with pytest.raises(ValueError) as excinfo:
        parse_count_progress(value)

    assert COUNT_PROGRESS_KEY in str(excinfo.value)


async def test_a_bad_config_value_fails_at_startup():
    """The plugin refuses a bad value on init, naming the key and the value."""
    harness = Harness()
    with pytest.raises(ValueError) as excinfo:
        await harness.init({"outbound_cap": {"max_messages": 0}})

    assert MAX_MESSAGES_KEY in str(excinfo.value)
    assert "0" in str(excinfo.value)


async def test_a_bad_reload_is_logged_and_the_old_limit_kept(h, caplog):
    """A reload carrying a bad value keeps running rather than refusing
    every send, and says so at ERROR."""
    channel = h.channel("signal:aci1")

    with caplog.at_level(logging.DEBUG, logger="corvidae.outbound_cap"):
        await h.plugin.on_config_reload({"outbound_cap": {"max_messages": 0}})

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, errors
    assert MAX_MESSAGES_KEY in errors[0]

    # The limit from init is still in force: two messages, then a veto.
    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is False


async def test_a_good_reload_applies(h, caplog):
    """A reloaded limit takes effect on the next decision."""
    channel = h.channel("signal:aci1")
    await h.plugin.on_config_reload({"outbound_cap": {"max_messages": 1}})

    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is False


async def test_a_per_channel_override_beats_the_global_limit(h):
    """set_settings writes channel.runtime_overrides, which wins."""
    channel = h.channel("signal:aci1")
    other = h.channel("signal:aci2")
    channel.runtime_overrides[MAX_MESSAGES_KEY] = 1

    assert await h.gate.send(channel) is True
    assert await h.gate.send(channel) is False
    # The other channel still has the global limit of two.
    assert await h.gate.send(other) is True
    assert await h.gate.send(other) is True
    assert await h.gate.send(other) is False


async def test_a_broken_channel_object_fails_open(caplog):
    """A counter that raises must never silence the agent."""

    class Broken:
        id = "signal:aci1"

        @property
        def runtime_overrides(self):
            raise RuntimeError("boom")

    harness = Harness()
    await harness.init(CONFIG)
    with caplog.at_level(logging.DEBUG, logger="corvidae.outbound_cap"):
        assert await harness.gate.send(Broken()) is True

    assert any("failed" in r.getMessage() for r in caplog.records)
