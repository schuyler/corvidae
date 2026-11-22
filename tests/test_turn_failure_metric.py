"""A failed turn emits exactly one metric, and a good turn emits none (#1060)."""

from __future__ import annotations

import logging

from corvidae.hooks import CorvidaePlugin, hookimpl, create_plugin_manager

from corvidae.trace import mint_trace_id

from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response


class Recorder(CorvidaePlugin):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    @hookimpl
    async def on_metrics(self, name, value, tags):
        self.events.append({"name": name, "value": value, "tags": tags})


class Raising(CorvidaePlugin):
    @hookimpl
    async def on_metrics(self, name, value, tags):
        raise RuntimeError("consumer boom")


async def test_a_failed_turn_emits_one_event():
    agent, channel, db = await build_plugin_and_channel()
    try:
        recorder = Recorder()
        agent.pm.register(recorder, name="recorder")

        async def boom(*args, **kwargs):
            raise RuntimeError("llm exploded")

        agent._client = type("C", (), {"chat": boom})()
        agent._mark_started()

        await agent.on_message(
            channel=channel, sender="user", text="hi", trace_id=mint_trace_id()
        )
        await drain(agent, channel)

        failed = [e for e in recorder.events if e["name"] == "agent.turn_failed"]
        assert len(failed) == 1, recorder.events
        assert failed[0]["tags"]["channel"] == channel.id
        assert failed[0]["tags"]["exception"] == "RuntimeError"
        assert failed[0]["value"] == 1.0
    finally:
        await db.close()


async def test_a_successful_turn_emits_none():
    agent, channel, db = await build_plugin_and_channel()
    try:
        recorder = Recorder()
        agent.pm.register(recorder, name="recorder")

        from unittest.mock import AsyncMock

        client = AsyncMock()
        client.chat = AsyncMock(return_value=_make_text_response("hello"))
        agent._client = client
        agent._mark_started()

        await agent.on_message(
            channel=channel, sender="user", text="hi", trace_id=mint_trace_id()
        )
        await drain(agent, channel)

        assert not [e for e in recorder.events if e["name"] == "agent.turn_failed"]
    finally:
        await db.close()


async def test_a_raising_consumer_does_not_stop_the_error_message(caplog):
    agent, channel, db = await build_plugin_and_channel()
    try:
        agent.pm.register(Raising(), name="raising")
        sent: list[str] = []

        async def boom(*args, **kwargs):
            raise RuntimeError("llm exploded")

        agent._client = type("C", (), {"chat": boom})()
        agent._mark_started()

        with caplog.at_level(logging.WARNING, logger="corvidae"):
            await agent.on_message(
            channel=channel, sender="user", text="hi", trace_id=mint_trace_id()
        )
            await drain(agent, channel)

        assert "raising" in caplog.text
    finally:
        await db.close()
