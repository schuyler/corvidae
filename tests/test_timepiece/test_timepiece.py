"""Tests for timepiece: ambient wall-clock CONTEXT stamps per channel.

Goes through the real plugin manager so hook-binding and registration
bugs are caught (checklist step 3 shape).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import create_plugin_manager

from corvidae_timepiece import TimepiecePlugin

STAMP_RE = re.compile(r"^The current date and time are now [A-Za-z]+ \d{4}-\d{2}-\d{2} at \d{2}:\d{2} [A-Z]+$")
FIXED_TZ = ZoneInfo("UTC")
T0 = datetime(2026, 11, 7, 19, 15, tzinfo=FIXED_TZ)


class FakeChannel:
    """Minimal channel stand-in: the hook only touches .id and .conversation."""

    def __init__(self, channel_id: str = "test:chan"):
        self.id = channel_id
        self.conversation = ContextWindow(channel_id)


async def make_plugin(config: dict, now: datetime = T0) -> TimepiecePlugin:
    """Register the plugin through the real pm and fix its clock."""
    pm = create_plugin_manager()
    plugin = TimepiecePlugin()
    pm.register(plugin, name="timepiece")
    await pm.ahook.on_init(pm=pm, config=config)
    plugin._now = lambda: now
    return plugin


async def turn(plugin: TimepiecePlugin, channel: FakeChannel) -> None:
    """Fire before_agent_turn the way the agent does (agent.py:1375)."""
    await plugin.before_agent_turn(channel, trace_id=None, annotations={})


def context_entries(channel: FakeChannel) -> list[dict]:
    return [
        m for m in channel.conversation.messages
        if m.get("_message_type") == MessageType.CONTEXT
    ]


class TestStampInjection:
    async def test_first_turn_injects_stamp(self):
        plugin = await make_plugin({})
        channel = FakeChannel()
        await turn(plugin, channel)
        entries = context_entries(channel)
        assert len(entries) == 1
        assert entries[0]["role"] == "system"
        assert STAMP_RE.match(entries[0]["content"]), entries[0]["content"]
        # The fixed clock reads 19:15 UTC on a Saturday.
        assert "Saturday 2026-11-07 at 19:15 UTC" in entries[0]["content"]

    async def test_no_stamp_within_interval(self):
        plugin = await make_plugin({})
        channel = FakeChannel()
        await turn(plugin, channel)
        plugin._now = lambda: T0 + timedelta(minutes=9)
        await turn(plugin, channel)
        assert len(context_entries(channel)) == 1

    async def test_stamp_at_and_after_interval(self):
        plugin = await make_plugin({})
        channel = FakeChannel()
        await turn(plugin, channel)
        plugin._now = lambda: T0 + timedelta(minutes=10)  # exactly the interval
        await turn(plugin, channel)
        assert len(context_entries(channel)) == 2
        plugin._now = lambda: T0 + timedelta(minutes=11)
        await turn(plugin, channel)
        assert len(context_entries(channel)) == 2  # 11 - 10 = 1 min < interval

    async def test_per_channel_independence(self):
        plugin = await make_plugin({})
        a, b = FakeChannel("test:a"), FakeChannel("test:b")
        await turn(plugin, a)
        await turn(plugin, b)
        assert len(context_entries(a)) == 1
        assert len(context_entries(b)) == 1
        # One channel stamping must not move the other's clock.
        plugin._now = lambda: T0 + timedelta(minutes=10)
        await turn(plugin, a)
        assert len(context_entries(a)) == 2
        assert len(context_entries(b)) == 1

    async def test_none_conversation_is_ignored(self):
        plugin = await make_plugin({})
        channel = FakeChannel()
        channel.conversation = None
        await turn(plugin, channel)  # must not raise
        assert plugin._last.get(channel.id) is None


class TestConfig:
    async def test_custom_interval_respected(self):
        plugin = await make_plugin({"timepiece": {"interval_minutes": 30}})
        channel = FakeChannel()
        await turn(plugin, channel)
        plugin._now = lambda: T0 + timedelta(minutes=11)
        await turn(plugin, channel)
        assert len(context_entries(channel)) == 1
        plugin._now = lambda: T0 + timedelta(minutes=31)
        await turn(plugin, channel)
        assert len(context_entries(channel)) == 2

    async def test_float_interval_accepted(self):
        plugin = await make_plugin({"timepiece": {"interval_minutes": 0.5}})
        channel = FakeChannel()
        await turn(plugin, channel)
        # 1 min > 0.5 min interval, so a second stamp is expected.
        # (31 min would also satisfy the default 10-min interval,
        # making the test pass even if the config were ignored.)
        plugin._now = lambda: T0 + timedelta(minutes=1)
        await turn(plugin, channel)
        assert len(context_entries(channel)) == 2

    async def test_top_level_key_ignored(self):
        # Trap 2: config is the whole dict; a top-level same-name key
        # must not be read as the plugin's setting.
        plugin = await make_plugin({"interval_minutes": 1})
        channel = FakeChannel()
        await turn(plugin, channel)
        plugin._now = lambda: T0 + timedelta(minutes=2)
        await turn(plugin, channel)
        assert len(context_entries(channel)) == 1  # default 10 min, not 1

    async def test_configured_timezone_in_stamp(self):
        plugin = await make_plugin({"timepiece": {"timezone": "America/Los_Angeles"}})
        # The seam returns wall time in the plugin's configured zone:
        # 11:15 PST on 2026-11-07 (which is 19:15 UTC).
        plugin._now = lambda: datetime(
            2026, 11, 7, 11, 15, tzinfo=ZoneInfo("America/Los_Angeles")
        )
        channel = FakeChannel()
        await turn(plugin, channel)
        content = context_entries(channel)[0]["content"]
        assert "Saturday 2026-11-07 at 11:15 PST" in content, content

    async def test_malformed_interval_raises_naming_key_and_value(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest

        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"interval_minutes": "soon"}}
            )
        assert "timepiece.interval_minutes" in str(exc.value)
        assert "'soon'" in str(exc.value)

    async def test_malformed_timezone_raises_naming_key_and_value(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest

        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"timezone": "Mars/Olympus"}}
            )
        assert "timepiece.timezone" in str(exc.value)
        assert "Mars/Olympus" in str(exc.value)

    async def test_malformed_format_raises_naming_key_and_value(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest

        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"format": ""}}
            )
        assert "timepiece.format" in str(exc.value)


    async def test_zero_interval_rejected(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest
        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"interval_minutes": 0}}
            )
        assert "timepiece.interval_minutes" in str(exc.value)

    async def test_negative_interval_rejected(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest
        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"interval_minutes": -5}}
            )
        assert "timepiece.interval_minutes" in str(exc.value)

    async def test_boolean_interval_rejected(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest
        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"interval_minutes": True}}
            )
        assert "timepiece.interval_minutes" in str(exc.value)

    async def test_non_string_timezone_rejected(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest
        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"timezone": 123}}
            )
        assert "timepiece.timezone" in str(exc.value)

    async def test_empty_timezone_rejected(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest
        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"timezone": ""}}
            )
        assert "timepiece.timezone" in str(exc.value)


    async def test_oversized_interval_rejected(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest
        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"interval_minutes": 10**30}}
            )
        assert "timepiece.interval_minutes" in str(exc.value)

    async def test_non_string_format_rejected(self):
        pm = create_plugin_manager()
        plugin = TimepiecePlugin()
        pm.register(plugin, name="timepiece")
        import pytest
        with pytest.raises(ValueError) as exc:
            await pm.ahook.on_init(
                pm=pm, config={"timepiece": {"format": 123}}
            )
        assert "timepiece.format" in str(exc.value)


class TestCustomFormat:
    async def test_custom_strftime_format(self):
        plugin = await make_plugin(
            {"timepiece": {"format": "%B %d, %Y %I:%M %p"}}
        )
        channel = FakeChannel()
        await turn(plugin, channel)
        content = context_entries(channel)[0]["content"]
        # T0 = 2026-11-07 19:15 UTC -> "November 07, 2026 07:15 PM"
        assert "November 07, 2026 07:15 PM" in content, content

    async def test_default_format_unchanged(self):
        plugin = await make_plugin({})
        channel = FakeChannel()
        await turn(plugin, channel)
        content = context_entries(channel)[0]["content"]
        assert "The current date and time are now" in content
        assert "Saturday 2026-11-07 at 19:15 UTC" in content, content
