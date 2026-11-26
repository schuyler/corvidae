"""A turn that ends with no final releases the transport's typing slot (#424).

An admitted message whose turn dies before a response used to leave the
transport's owed-reply count above zero with no owner, so the typing indicator
refreshed forever. The kernel now reports that a turn ended without a final;
Discord and Signal release their slots on it.
"""

from __future__ import annotations

import pytest

from unittest.mock import AsyncMock, MagicMock

from helpers import build_plugin_and_channel
from llm_response_fixtures import _make_text_response

from corvidae.agent import QueueItem, QueueItemRole
from corvidae.hooks import hookimpl


class _Recorder:
    """Records turn-end observations (#424)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    @hookimpl
    async def on_turn_ended_without_final(self, channel, trace_id):
        self.calls.append((channel.id, trace_id))


async def _run_one_turn(plugin, channel, *, fail: bool):
    """Run one queued user item through _process_queue_item."""
    recorder = _Recorder()
    plugin.pm.register(recorder)
    item = QueueItem(
        role=QueueItemRole.USER, content="hello", channel=channel, sender="schuyler",
    )
    if fail:
        original = plugin._process_queue_item_attributed

        async def _boom(item, channel_, absorbed):
            raise RuntimeError("turn died before a final")

        plugin._process_queue_item_attributed = _boom
    try:
        try:
            await plugin._process_queue_item(item)
        except RuntimeError:
            # The queue's caller logs it; the turn-end observation still fires.
            pass
    finally:
        if fail:
            plugin._process_queue_item_attributed = original
    return recorder


async def test_a_turn_that_dies_before_a_final_is_reported():
    """The observation fires, so a transport can stop its indicator."""
    plugin, channel, db = await build_plugin_and_channel()
    try:
        recorder = await _run_one_turn(plugin, channel, fail=True)
        assert len(recorder.calls) == 1, recorder.calls
        assert recorder.calls[0][0] == channel.id
        assert recorder.calls[0][1], "the turn's trace id is carried"
    finally:
        await db.close()


async def test_a_turn_with_a_final_is_not_reported():
    """A final, sent or unsent, already releases through the response hooks."""
    plugin, channel, db = await build_plugin_and_channel()
    plugin._client = MagicMock()
    plugin._client.chat = AsyncMock(return_value=_make_text_response("hi"))
    try:
        recorder = await _run_one_turn(plugin, channel, fail=False)
        assert recorder.calls == [], recorder.calls
    finally:
        await db.close()
