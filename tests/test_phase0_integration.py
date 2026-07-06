"""Phase 0 acceptance checks: metering at the LLMClient chokepoint.

Approximates the definition-of-done checks from
plans/implementation/phase-0.md with a stubbed HTTP session in place of a
live llama-server:
  - a turn produces a usage_log row tagged stage="turn" with the channel id;
  - a forced compaction produces a row tagged stage="compaction" (proving
    the metering site catches calls that bypass the turn loop);
  - work enqueued on the TaskQueue during an attributed operation carries
    the operation's attribution into its own LLM calls.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.agent import Agent, QueueItem, QueueItemRole
from corvidae.channel import ChannelConfig
from corvidae.context import MessageType

MOCK_COMPLETION = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "summary text or reply"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
}


def _make_always_ok_session():
    """Mock aiohttp session whose post() always returns a 200 completion."""
    def make_cm(*args, **kwargs):
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value=MOCK_COMPLETION)
        response.raise_for_status = MagicMock()
        response.headers = {}
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=response)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    session = AsyncMock()
    session.post = MagicMock(side_effect=make_cm)
    session.close = AsyncMock()
    return session


async def _build_stack():
    """Assemble Agent + LLMPlugin (stubbed session) + usage sinks on one pm."""
    from helpers import build_plugin_and_channel

    from corvidae.llm_plugin import LLMPlugin
    from corvidae.metrics import UsageLogPlugin

    plugin, channel, db = await build_plugin_and_channel()
    pm = plugin.pm

    config = {"llm": {"main": {"base_url": "http://test", "model": "test-model"}}}
    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm, config)
    await llm.on_start(config)
    # Replace the real aiohttp session with the stub — no network.
    llm.main_client.session = _make_always_ok_session()

    usage_log = UsageLogPlugin()
    pm.register(usage_log, name="usage_log")
    await usage_log.on_init(pm, {})
    await usage_log.on_start({})

    plugin._client = llm.main_client
    return plugin, channel, db, llm


async def _fetch_usage_rows(db):
    async with db.execute(
        "SELECT stage, channel_id, role, model, total_tokens FROM usage_log ORDER BY id"
    ) as cursor:
        return await cursor.fetchall()


class TestPhase0Acceptance:
    async def test_turn_llm_call_writes_usage_row_with_turn_stage(self):
        plugin, channel, db, llm = await _build_stack()
        try:
            item = QueueItem(
                role=QueueItemRole.USER,
                content="hello",
                channel=channel,
                sender="alice",
            )
            await plugin._process_queue_item(item)

            rows = await _fetch_usage_rows(db)
            assert len(rows) == 1
            stage, channel_id, role, model, total = rows[0]
            assert stage == "turn"
            assert channel_id == channel.id
            assert role == "main"
            assert model == "test-model"
            assert total == 14
        finally:
            await plugin.pm.get_plugin("task").on_stop()
            await llm.on_stop()
            await db.close()

    async def test_forced_compaction_writes_usage_row_with_compaction_stage(self):
        from corvidae.compaction import CompactionPlugin

        plugin, channel, db, llm = await _build_stack()
        try:
            compaction = CompactionPlugin()
            plugin.pm.register(compaction, name="compaction")
            await compaction.on_init(plugin.pm, {})

            # Build a conversation big enough to trip the 80% threshold of a
            # tiny max_tokens budget, then invoke the hook the way the agent
            # loop does (step 5).
            from corvidae.context import ContextWindow

            conv = ContextWindow(channel.id)
            for i in range(12):
                conv.append({"role": "user", "content": f"message {i} " + "x " * 40})
            channel.conversation = conv

            result = await compaction.compact_conversation(
                channel=channel, conversation=conv, max_tokens=120
            )
            assert result is True

            rows = await _fetch_usage_rows(db)
            assert len(rows) == 1
            stage, channel_id, role, model, total = rows[0]
            assert stage == "compaction"
            assert channel_id == channel.id
        finally:
            await plugin.pm.get_plugin("task").on_stop()
            await llm.on_stop()
            await db.close()

    async def test_compaction_inside_turn_restores_turn_attribution(self):
        # Compaction shadows the turn attribution around its LLM call only;
        # the turn's own LLM call afterwards is still tagged stage="turn".
        from corvidae.compaction import CompactionPlugin

        plugin, channel, db, llm = await _build_stack()
        try:
            compaction = CompactionPlugin()
            plugin.pm.register(compaction, name="compaction")
            await compaction.on_init(plugin.pm, {})

            # Pre-seed a conversation that will trip compaction (threshold is
            # evaluated against the channel's max_context_tokens).
            from corvidae.context import ContextWindow

            conv = ContextWindow(channel.id)
            conv.system_prompt = "You are a test assistant."
            for i in range(12):
                conv.append({"role": "user", "content": f"message {i} " + "x " * 40})
            channel.conversation = conv
            channel.config = ChannelConfig(max_context_tokens=140)

            item = QueueItem(
                role=QueueItemRole.USER,
                content="hello again",
                channel=channel,
                sender="alice",
            )
            await plugin._process_queue_item(item)

            rows = await _fetch_usage_rows(db)
            stages = [row[0] for row in rows]
            # One compaction call, then the turn's own LLM call.
            assert stages == ["compaction", "turn"]
            assert all(row[1] == channel.id for row in rows)
        finally:
            await plugin.pm.get_plugin("task").on_stop()
            await llm.on_stop()
            await db.close()

    async def test_task_enqueued_during_attributed_operation_carries_attribution(self):
        # The contextvars fix demonstrably works across the queue: LLM calls
        # made inside a Task are attributed to the enqueuer's stage/channel.
        from corvidae.attribution import reset_attribution, set_attribution
        from corvidae.task import Task

        plugin, channel, db, llm = await _build_stack()
        task_plugin = plugin.pm.get_plugin("task")
        try:
            done = asyncio.Event()

            async def work():
                # This LLM call happens in the worker, long after the
                # enqueuer's attribution was reset.
                await llm.main_client.chat([{"role": "user", "content": "bg"}])
                done.set()
                return "ok"

            token = set_attribution(stage="turn", channel_id=channel.id)
            try:
                task = Task(work=work, channel=channel)
            finally:
                reset_attribution(token)

            await task_plugin.task_queue.enqueue(task)
            await asyncio.wait_for(done.wait(), timeout=2.0)
            await task_plugin.task_queue.queue.join()

            rows = await _fetch_usage_rows(db)
            assert len(rows) == 1
            assert rows[0][0] == "turn"
            assert rows[0][1] == channel.id
        finally:
            await task_plugin.on_stop()
            await llm.on_stop()
            await db.close()
