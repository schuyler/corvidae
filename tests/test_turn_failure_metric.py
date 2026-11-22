"""A failed turn emits exactly one metric, and a good turn emits none (#1060)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

from helpers import build_plugin_and_channel, drain, sent_texts
from llm_response_fixtures import _make_text_response

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.trace import mint_trace_id


class Recorder(CorvidaePlugin):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    @hookimpl
    async def on_metrics(self, name, value, tags):
        self.events.append({"name": name, "value": value, "tags": tags})

    def failed(self) -> list[dict]:
        return [e for e in self.events if e["name"] == "agent.turn.failed"]


class Raising(CorvidaePlugin):
    @hookimpl
    async def on_metrics(self, name, value, tags):
        raise RuntimeError("consumer boom")


async def _turn(agent, channel):
    await agent.on_message(
        channel=channel, sender="user", text="hi", trace_id=mint_trace_id()
    )
    await drain(agent, channel)


async def test_a_failed_turn_emits_one_event():
    agent, channel, db = await build_plugin_and_channel()
    try:
        recorder = Recorder()
        agent.pm.register(recorder, name="recorder")

        client = AsyncMock()
        client.chat = AsyncMock(side_effect=RuntimeError("llm exploded"))
        agent._client = client
        agent._mark_started()

        await _turn(agent, channel)

        failed = recorder.failed()
        assert len(failed) == 1, recorder.events
        assert failed[0]["tags"]["channel"] == channel.id
        assert failed[0]["tags"]["exception"] == "RuntimeError"
        assert failed[0]["value"] == 1.0
    finally:
        await db.close()


async def test_a_failed_retry_emits_too():
    """The issue names this path: a failed retry is a failed turn (#1060)."""
    from corvidae.agent import QueueItem, QueueItemRole

    agent, channel, db = await build_plugin_and_channel()
    try:
        recorder = Recorder()
        agent.pm.register(recorder, name="recorder")

        client = AsyncMock()
        client.chat = AsyncMock(side_effect=RuntimeError("llm exploded"))
        agent._client = client
        agent._mark_started()

        item = QueueItem(
            role=QueueItemRole.USER, content="hi", channel=channel,
            sender="user", trace_id=mint_trace_id(), originating=False, retry=True,
        )
        from corvidae.agent import _TurnRetry

        agent._turn_retries[channel.id] = _TurnRetry(
            item=item, attempt=1, timer=MagicMock()
        )
        await agent._get_or_create_queue(channel).enqueue(item)
        await drain(agent, channel)

        assert len(recorder.failed()) == 1, recorder.events
    finally:
        await db.close()


async def test_a_successful_turn_emits_none():
    agent, channel, db = await build_plugin_and_channel()
    try:
        recorder = Recorder()
        agent.pm.register(recorder, name="recorder")

        client = AsyncMock()
        client.chat = AsyncMock(return_value=_make_text_response("hello"))
        agent._client = client
        agent._mark_started()

        await _turn(agent, channel)

        assert recorder.failed() == []
        from corvidae.outbound import OutboundMessage

        assert sent_texts(agent.pm, OutboundMessage) == ["hello"]
    finally:
        await db.close()


async def test_the_emit_comes_before_the_error_gate():
    """Ordering is the requirement: a raise in the error path cannot suppress it."""
    order: list[str] = []

    class Ordering(Recorder):
        @hookimpl
        async def on_metrics(self, name, value, tags):
            order.append("metric")
            await super().on_metrics(name, value, tags)

        @hookimpl
        async def resolve_llm_error(self, channel, error):
            order.append("error-gate")
            return None

    agent, channel, db = await build_plugin_and_channel()
    try:
        agent.pm.register(Ordering(), name="ordering")

        client = AsyncMock()
        client.chat = AsyncMock(side_effect=RuntimeError("llm exploded"))
        agent._client = client
        agent._mark_started()

        await _turn(agent, channel)

        assert order[:2] == ["metric", "error-gate"], order
    finally:
        await db.close()


async def test_a_raising_consumer_does_not_stop_the_error_message(caplog):
    agent, channel, db = await build_plugin_and_channel()
    try:
        agent.pm.register(Raising(), name="raising")

        client = AsyncMock()
        client.chat = AsyncMock(side_effect=RuntimeError("llm exploded"))
        agent._client = client
        agent._mark_started()

        with caplog.at_level(logging.WARNING, logger="corvidae"):
            await _turn(agent, channel)

        assert "raising" in caplog.text
        # The acceptance criterion: the user is still told the turn failed.
        from corvidae.outbound import OutboundMessage

        texts = sent_texts(agent.pm, OutboundMessage)
        assert texts, "the user got no error message"
    finally:
        await db.close()
