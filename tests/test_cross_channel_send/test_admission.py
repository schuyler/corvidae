"""notify_other_channel offers its work to the admission seam (issue #1050)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from corvidae.channel import ChannelConfig
from corvidae.hooks import hookimpl

from .test_cross_channel_send import _build_harness, _init_plugin


@pytest.fixture(autouse=True)
def _reset_attribution():
    """Keep attribution clean between tests."""
    from corvidae.attribution import _attribution

    token = _attribution.set({})
    yield
    _attribution.reset(token)


class _RefusingGate:
    """Plugin that records each should_start_work offer and refuses it."""

    def __init__(self):
        self.offers: list[dict] = []

    @hookimpl
    async def should_start_work(
        self, channel, kind, source_channel, trace_id, annotations
    ):
        # Record the offer, then refuse it.
        from corvidae.admission import WorkRefusal

        self.offers.append(
            {"channel": channel, "kind": kind, "source_channel": source_channel}
        )
        return WorkRefusal("limits.max_waiting", "wait")


async def test_refused_note_returns_text_and_does_not_notify():
    """A refusal is the tool's result, on_notify is not called, and the offer
    is kind "note" on the target channel attributed to the sending channel."""
    plugin, registry, db, agent = await _build_harness()
    pm = agent.pm
    try:
        await _init_plugin(plugin, pm)
        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {"channel_id": ch_a.id, "trace_id": "trace-1", "annotations": {}}
        )
        gate = _RefusingGate()
        pm.register(gate, name="gate")
        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.notify_other_channel(ch_b.id, "hello bob")

        assert result == "Not started: limits.max_waiting reached — wait"
        notify.assert_not_called()
        assert len(gate.offers) == 1
        offer = gate.offers[0]
        assert offer["kind"] == "note"
        assert offer["channel"] is ch_b
        assert offer["source_channel"] == ch_a.id
    finally:
        await pm.get_plugin("task").on_stop()
        await db.close()
