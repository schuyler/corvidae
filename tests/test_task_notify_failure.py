"""A raising on_notify must not leave a tool call pending forever (#173).

TaskPlugin._on_task_complete delivers a finished tool task's result
through the on_notify broadcast. If that broadcast raises, the tool call
id must still clear and the model must still get exactly one result for
the call: an error result when the Agent never queued the real one, and
the real one alone when another plugin's hookimpl was what raised.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.trace import mint_trace_id

from helpers import build_plugin_and_channel, drain, drain_task_queue
from llm_response_fixtures import _make_text_response


def _tool_call_response(call_id: str = "call-001") -> dict:
    """An LLM response asking for one my_tool call."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "function": {"name": "my_tool", "arguments": '{"x": "val"}'},
                }],
            }
        }]
    }


class RaisingNotify(CorvidaePlugin):
    """A third-party plugin whose on_notify always raises."""

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, meta):
        raise RuntimeError("third-party notify explosion")


async def _run_tool_cycle(plugin, channel, on_tool_run=None) -> MagicMock:
    """Drive one user message through a tool call; return the mock LLM client.

    ``on_tool_run`` is called inside the tool, i.e. just before the task
    completes and its result goes out through on_notify.
    """

    # The tool the model calls.
    async def my_tool(x: str) -> str:
        """A simple test tool."""
        if on_tool_run is not None:
            on_tool_run()
        return f"result:{x}"

    plugin._tools = {"my_tool": my_tool}
    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[_tool_call_response(), _make_text_response("tool done")]
    )
    plugin._client = client

    await plugin.on_message(
        channel=channel, sender="user", text="use tool", trace_id=mint_trace_id()
    )
    # Let the turn dispatch, the task run and complete, and the result turn run.
    for _ in range(3):
        await drain(plugin, channel)
        await drain_task_queue(plugin)
        await asyncio.sleep(0)
    await drain(plugin, channel)
    return client


def _tool_messages(client: MagicMock, call_id: str = "call-001") -> list[dict]:
    """The role=tool messages for call_id in the second (post-tool) LLM call."""
    assert client.chat.await_count == 2, (
        f"expected a second LLM call after the tool result; got {client.chat.await_count}"
    )
    call = client.chat.await_args_list[1]
    messages = call.kwargs.get("messages") or call.args[0]
    return [
        m for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    ]


async def _teardown(plugin, db) -> None:
    """Stop the task worker and close the DB."""
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


class TestRaisingNotifyDeliversError:
    async def test_agent_enqueue_failure_delivers_error_result(self, caplog):
        # The Agent's own on_notify raises before queueing the result: the
        # model still gets an error result for the call, and the id clears.
        plugin, channel, db = await build_plugin_and_channel()
        try:
            original = plugin._note_activity
            failures = {"left": 0}

            # Fail the next _note_activity call made from on_notify.
            def flaky_note_activity(channel_id):
                if failures["left"]:
                    failures["left"] -= 1
                    raise RuntimeError("notify enqueue explosion")
                return original(channel_id)

            plugin._note_activity = flaky_note_activity

            with caplog.at_level(logging.WARNING, logger="corvidae"):
                # Arm the failure from inside the tool, so it hits on_notify.
                client = await _run_tool_cycle(
                    plugin, channel, on_tool_run=lambda: failures.update(left=1)
                )

            tool_messages = _tool_messages(client)
            assert len(tool_messages) == 1, tool_messages
            assert "error" in tool_messages[0]["content"].lower(), tool_messages
            assert channel.pending_tool_call_ids == set()

            # The failure is logged with channel id, task id and tool call id.
            records = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING
                and getattr(r, "tool_call_id", None) == "call-001"
            ]
            assert records, [r.getMessage() for r in caplog.records]
            record = records[0]
            assert "on_notify" in record.getMessage()
            assert getattr(record, "channel", None) == channel.id
            assert getattr(record, "task_id", None)
        finally:
            await _teardown(plugin, db)

    async def test_third_party_raise_yields_exactly_one_real_result(self):
        # Another plugin's on_notify raises after the Agent already queued
        # the real result: no duplicate error result is added.
        plugin, channel, db = await build_plugin_and_channel()
        try:
            plugin.pm.register(RaisingNotify(), name="raising_notify")
            client = await _run_tool_cycle(plugin, channel)

            tool_messages = _tool_messages(client)
            assert len(tool_messages) == 1, tool_messages
            assert "result:val" in tool_messages[0]["content"], tool_messages
            assert channel.pending_tool_call_ids == set()
        finally:
            await _teardown(plugin, db)
