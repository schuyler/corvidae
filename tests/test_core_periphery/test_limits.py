"""Tests for corvidae_core_periphery.limits.LimitsPlugin (issue #1050).

The agent is a real Agent (get_dependency type-checks) whose two depth reads
are replaced by counts the test controls; the clock is the kernel FakeClock.
"""

import inspect
import logging

import pytest

from corvidae.agent import Agent
from corvidae.channel import Channel, ChannelConfig


def _channel(scope="a") -> Channel:
    """A bare channel; its id is ``test:<scope>``."""
    return Channel(transport="test", scope=scope, config=ChannelConfig())


class Counts:
    """The depth numbers the fake reads report; tests mutate the dicts."""

    def __init__(self, work=None, messages=None):
        self.work = work or {}
        self.messages = messages or {}


async def _make(pm, monkeypatch, config, counts=None):
    """A started LimitsPlugin on ``pm`` (needs the kernel_clock fixture).

    Registers a real Agent as "agent" whose waiting_work/waiting_messages
    return ``counts``. Returns (plugin, counts).
    """
    from corvidae_core_periphery.limits import LimitsPlugin

    counts = counts or Counts()
    agent = Agent()
    monkeypatch.setattr(agent, "waiting_work", lambda: counts.work, raising=False)
    monkeypatch.setattr(
        agent, "waiting_messages", lambda: counts.messages, raising=False
    )
    pm.register(agent, name="agent")
    plugin = LimitsPlugin()
    pm.register(plugin, name="limits")
    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)
    return plugin, counts


async def _start_work(plugin, channel=None):
    """Call the plugin's should_start_work, awaiting it if it is async."""
    result = plugin.should_start_work(
        channel=channel or _channel(),
        kind="task",
        source_channel="test:src",
        trace_id=None,
        annotations={},
    )
    return await result if inspect.isawaitable(result) else result


async def _process(plugin, channel):
    """Call the plugin's should_process_message, awaiting it if async."""
    result = plugin.should_process_message(
        channel=channel, sender="u", text="hi", trace_id="t"
    )
    return await result if inspect.isawaitable(result) else result


async def _init_only(pm, config):
    """Run only on_init, the step that validates config."""
    from corvidae_core_periphery.limits import LimitsPlugin

    plugin = LimitsPlugin()
    pm.register(plugin, name="limits")
    await plugin.on_init(pm=pm, config=config)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    async def test_non_mapping_block_names_key_and_file(self, pm):
        """`limits: 5` is a startup error naming limits and agent.yaml."""
        with pytest.raises(ValueError) as exc:
            await _init_only(pm, {"limits": 5})
        assert "limits" in str(exc.value)
        assert "agent.yaml" in str(exc.value)

    async def test_unknown_key_names_it_and_accepted_keys(self, pm):
        """An unknown key is rejected, naming it and the accepted keys."""
        with pytest.raises(ValueError) as exc:
            await _init_only(pm, {"limits": {"max_hops": 4}})
        msg = str(exc.value)
        assert "max_hops" in msg
        for key in ("inbound_per_channel", "notice_interval", "max_waiting"):
            assert key in msg

    @pytest.mark.parametrize("key", ["inbound_per_channel", "notice_interval", "max_waiting"])
    @pytest.mark.parametrize("bad", [0, -1, True, "5", 1.5])
    async def test_bad_value_names_key_value_and_remedy(self, pm, key, bad):
        """A non-positive-int value names the dotted key, the rejected value
        and "at least 1"."""
        with pytest.raises(ValueError) as exc:
            await _init_only(pm, {"limits": {key: bad}})
        msg = str(exc.value)
        assert f"limits.{key}" in msg
        assert repr(bad) in msg
        assert "at least 1" in msg


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


class TestNoBlock:
    async def test_no_limits_block_gives_no_verdicts(self, pm, kernel_clock, monkeypatch):
        """Without a limits block both hooks return None whatever the counts."""
        counts = Counts({"test:a": 10_000}, {"test:a": 10_000})
        plugin, _ = await _make(pm, monkeypatch, {}, counts)
        assert await _start_work(plugin) is None
        assert await _process(plugin, _channel()) is None


class TestDefaults:
    async def test_ceiling_default_is_200(self, pm, kernel_clock, monkeypatch):
        """Absent keys default max_waiting to 200: refuse at the sum 200, not 199."""
        plugin, counts = await _make(pm, monkeypatch, {"limits": {}})
        counts.work = {"test:a": 99, "test:b": 100}
        assert await _start_work(plugin) is None
        counts.work = {"test:a": 100, "test:b": 100}
        assert await _start_work(plugin) is not None

    async def test_inbound_default_is_50(self, pm, kernel_clock, monkeypatch):
        """Absent keys default inbound_per_channel to 50: reject at 50, not 49."""
        plugin, counts = await _make(pm, monkeypatch, {"limits": {}})
        ch = _channel()
        counts.messages = {ch.id: 49}
        assert await _process(plugin, ch) is None
        counts.messages = {ch.id: 50}
        assert await _process(plugin, ch) is False


class TestCeiling:
    async def test_refuses_at_max_waiting_across_channels(self, pm, kernel_clock, monkeypatch):
        """The sum over channels at max_waiting yields a WorkRefusal naming
        limits.max_waiting in both limit and remedy; below it, None."""
        from corvidae.admission import WorkRefusal

        plugin, counts = await _make(pm, monkeypatch, {"limits": {"max_waiting": 5}})
        counts.work = {"test:a": 3, "test:b": 1}
        assert await _start_work(plugin) is None
        counts.work = {"test:a": 3, "test:b": 2}
        refusal = await _start_work(plugin)
        assert isinstance(refusal, WorkRefusal)
        assert refusal.limit == "limits.max_waiting"
        assert "limits.max_waiting" in refusal.remedy


class TestInbound:
    async def test_rejects_at_cap_for_that_channel_only(self, pm, kernel_clock, monkeypatch):
        """The channel's own count at the cap rejects; below it or only other
        channels over it does not."""
        plugin, counts = await _make(
            pm, monkeypatch, {"limits": {"inbound_per_channel": 3}}
        )
        ch = _channel("a")
        counts.messages = {ch.id: 2, "test:other": 99}
        assert await _process(plugin, ch) is None
        counts.messages = {ch.id: 3}
        assert await _process(plugin, ch) is False

    async def test_warning_once_per_notice_interval(
        self, pm, kernel_clock, monkeypatch, caplog
    ):
        """First rejection warns naming channel and key; a second within
        notice_interval does not; after the interval it warns again."""
        clock, _idle = kernel_clock
        plugin, counts = await _make(
            pm,
            monkeypatch,
            {"limits": {"inbound_per_channel": 1, "notice_interval": 60}},
        )
        ch = _channel("a")
        counts.messages = {ch.id: 1}

        def warnings():
            return [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING and ch.id in r.getMessage()
            ]

        with caplog.at_level(logging.WARNING):
            assert await _process(plugin, ch) is False
            assert len(warnings()) == 1
            assert "limits.inbound_per_channel" in warnings()[0].getMessage()
            await clock.advance(10)
            assert await _process(plugin, ch) is False
            assert len(warnings()) == 1
            await clock.advance(61)
            assert await _process(plugin, ch) is False
            assert len(warnings()) == 2
