"""Tests for hook call safety in AgentPlugin and TaskPlugin.

These are RED TDD tests — they fail because the hooks listed below do not
currently wrap exceptions in try/except blocks:

  corvidae/agent.py _process_queue_item:
    - Line ~285: send_message on the error path (after run_agent_turn fails)
    - Line ~295: after_persist_assistant
    - Line ~329: on_agent_response
    - Line ~334: send_message on the normal response path

  corvidae/task.py TaskPlugin._on_task_complete:
    - Line ~244: on_notify

All five tests should produce a FAIL (not an error) because the unguarded
hook calls propagate exceptions rather than catching and logging them.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.hooks import hookimpl
from corvidae.task import Task

from helpers import build_plugin_and_channel, drain, drain_task_queue


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_text_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


# ---------------------------------------------------------------------------
# Helper plugin: raises on a named hook
# ---------------------------------------------------------------------------


class RaisingPlugin:
    """A plugin that raises RuntimeError from a specified hook.

    Registered by tests to inject a failure into a specific hook call.
    Each hookimpl is marked with ``trylast=True`` so it runs after the
    normal implementation (e.g. after FakeTransport's send_message) and
    the failure is always visible to the caller.
    """

    def __init__(self, hook_name: str, error_msg: str = "hook explosion") -> None:
        self._hook_name = hook_name
        self._error_msg = error_msg
        self._call_count = 0

    @hookimpl(trylast=True)
    async def after_persist_assistant(self, channel, message):
        if self._hook_name == "after_persist_assistant":
            self._call_count += 1
            raise RuntimeError(self._error_msg)

    @hookimpl(trylast=True)
    async def on_agent_response(self, channel, request_text, response_text):
        if self._hook_name == "on_agent_response":
            self._call_count += 1
            raise RuntimeError(self._error_msg)

    @hookimpl(trylast=True)
    async def send_message(self, channel, text, latency_ms=None):
        if self._hook_name == "send_message":
            self._call_count += 1
            raise RuntimeError(self._error_msg)

    @hookimpl(trylast=True)
    async def on_notify(self, channel, source, text, tool_call_id, meta):
        if self._hook_name == "on_notify":
            self._call_count += 1
            raise RuntimeError(self._error_msg)


# ---------------------------------------------------------------------------
# Test 1: after_persist_assistant raises — processing continues, WARNING logged
# ---------------------------------------------------------------------------


class TestAfterPersistAssistantHookSafety:
    """after_persist_assistant raising must not crash the queue consumer.

    Expected behavior (not yet implemented):
      - Processing continues: send_message IS still called
      - A WARNING is logged

    Current behavior (causing test to FAIL):
      - The exception propagates out of _process_queue_item and crashes
        the SerialQueue consumer, so send_message is never reached.
    """

    async def test_after_persist_assistant_raise_continues_processing(
        self, caplog
    ):
        plugin, channel, db = await build_plugin_and_channel(mock_send_message=False, mock_on_agent_response=False)
        try:
            pm = plugin.pm

            # Track send_message calls via a real hookimpl
            sent: list[str] = []

            class CapturingSendPlugin:
                @hookimpl
                async def send_message(self, channel, text, latency_ms=None):
                    sent.append(text)

            pm.register(CapturingSendPlugin(), name="capturing_send")

            # Register the raising plugin for after_persist_assistant
            raising = RaisingPlugin("after_persist_assistant")
            pm.register(raising, name="raising")

            mock_client = MagicMock()
            mock_client.chat = AsyncMock(
                return_value=_make_text_response("hello from agent")
            )
            plugin.client = mock_client

            with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
                await plugin.on_message(channel=channel, sender="user", text="hi")
                await drain(plugin, channel)

            # The raising plugin must have been called
            assert raising._call_count == 1, (
                "Expected after_persist_assistant to be called once"
            )

            # Processing must continue: send_message must still fire
            assert len(sent) == 1, (
                f"Expected send_message to be called after after_persist_assistant raised, "
                f"but got {len(sent)} calls"
            )
            assert "hello from agent" in sent[0]

            # A WARNING must be logged
            warning_records = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING
                and "after_persist_assistant" in r.getMessage().lower()
            ]
            assert warning_records, (
                "Expected a WARNING log about after_persist_assistant failure"
            )
        finally:
            task_plugin = plugin.pm.get_plugin("task")
            if task_plugin:
                await task_plugin.on_stop()
            await db.close()


# ---------------------------------------------------------------------------
# Test 2: on_agent_response raises — send_message still called, WARNING logged
# ---------------------------------------------------------------------------


class TestOnAgentResponseHookSafety:
    """on_agent_response raising must not prevent send_message from firing.

    Expected behavior (not yet implemented):
      - send_message IS called after on_agent_response raises
      - A WARNING is logged

    Current behavior (causing test to FAIL):
      - The exception propagates and send_message is never reached.
    """

    async def test_on_agent_response_raise_does_not_skip_send_message(
        self, caplog
    ):
        plugin, channel, db = await build_plugin_and_channel(mock_send_message=False, mock_on_agent_response=False)
        try:
            pm = plugin.pm

            sent: list[str] = []

            class CapturingSendPlugin:
                @hookimpl
                async def send_message(self, channel, text, latency_ms=None):
                    sent.append(text)

            pm.register(CapturingSendPlugin(), name="capturing_send")

            raising = RaisingPlugin("on_agent_response")
            pm.register(raising, name="raising")

            mock_client = MagicMock()
            mock_client.chat = AsyncMock(
                return_value=_make_text_response("response text")
            )
            plugin.client = mock_client

            with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
                await plugin.on_message(channel=channel, sender="user", text="hello")
                await drain(plugin, channel)

            # on_agent_response must have been called
            assert raising._call_count == 1, (
                "Expected on_agent_response to be called once"
            )

            # send_message must still fire despite the exception
            assert len(sent) == 1, (
                f"Expected send_message to be called after on_agent_response raised, "
                f"but got {len(sent)} calls"
            )
            assert "response text" in sent[0]

            # A WARNING must be logged
            warning_records = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING
                and "on_agent_response" in r.getMessage().lower()
            ]
            assert warning_records, (
                "Expected a WARNING log about on_agent_response failure"
            )
        finally:
            task_plugin = plugin.pm.get_plugin("task")
            if task_plugin:
                await task_plugin.on_stop()
            await db.close()


# ---------------------------------------------------------------------------
# Test 3: send_message (normal path) raises — ERROR logged, no crash
# ---------------------------------------------------------------------------


class TestSendMessageNormalPathHookSafety:
    """send_message raising on the normal response path must be caught.

    The user won't receive the response, but the queue consumer must keep running.

    Expected behavior (not yet implemented):
      - An ERROR is logged (user lost their response)
      - The queue consumer does not crash

    Current behavior (causing test to FAIL):
      - The exception propagates and crashes the SerialQueue consumer.
    """

    async def test_send_message_raise_is_caught_with_error_log(self, caplog):
        plugin, channel, db = await build_plugin_and_channel(mock_send_message=False, mock_on_agent_response=False)
        try:
            pm = plugin.pm

            # Register a send_message plugin that always raises
            raising = RaisingPlugin("send_message", error_msg="transport down")
            pm.register(raising, name="raising")

            mock_client = MagicMock()
            mock_client.chat = AsyncMock(
                return_value=_make_text_response("a response")
            )
            plugin.client = mock_client

            with caplog.at_level(logging.ERROR, logger="corvidae.agent"):
                await plugin.on_message(channel=channel, sender="user", text="hi")
                await drain(plugin, channel)

            # send_message must have been attempted
            assert raising._call_count >= 1, (
                "Expected send_message to be called at least once"
            )

            # An ERROR must be logged (not just a warning — the user lost their response)
            error_records = [
                r for r in caplog.records
                if r.levelno >= logging.ERROR
            ]
            assert error_records, (
                "Expected an ERROR log when send_message raises on the normal response path"
            )

            # The queue consumer must still be alive: a second message must be processed
            mock_client.chat = AsyncMock(
                return_value=_make_text_response("second response")
            )
            second_call_count_before = raising._call_count
            await plugin.on_message(channel=channel, sender="user", text="second message")
            await drain(plugin, channel)

            assert raising._call_count > second_call_count_before, (
                "Queue consumer crashed after first send_message failure — "
                "second message was not processed"
            )
        finally:
            task_plugin = plugin.pm.get_plugin("task")
            if task_plugin:
                await task_plugin.on_stop()
            await db.close()


# ---------------------------------------------------------------------------
# Test 4: send_message (error path) raises — WARNING logged, no crash
# ---------------------------------------------------------------------------


class TestSendMessageErrorPathHookSafety:
    """send_message on the LLM error path raising must be caught with a WARNING.

    This is the path taken when run_agent_turn raises (line ~285 in agent.py).
    The agent is already in an error state, so a WARNING (not ERROR) is appropriate.

    Expected behavior (not yet implemented):
      - A WARNING is logged
      - The queue consumer does not crash

    Current behavior (causing test to FAIL):
      - The exception propagates out of the except block and crashes the consumer.
    """

    async def test_send_message_on_error_path_raise_is_caught(self, caplog):
        plugin, channel, db = await build_plugin_and_channel(mock_send_message=False, mock_on_agent_response=False)
        try:
            pm = plugin.pm

            # Register a send_message plugin that always raises
            raising = RaisingPlugin("send_message", error_msg="send failed during error handling")
            pm.register(raising, name="raising")

            mock_client = MagicMock()
            # Make run_agent_turn itself fail to trigger the error path
            mock_client.chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
            plugin.client = mock_client

            with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
                await plugin.on_message(channel=channel, sender="user", text="hi")
                await drain(plugin, channel)

            # send_message must have been attempted on the error path
            assert raising._call_count >= 1, (
                "Expected send_message to be called on the error path"
            )

            # A WARNING (or higher) must be logged about the send_message failure
            warning_records = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING
            ]
            assert warning_records, (
                "Expected a WARNING log when send_message raises on the error path"
            )

            # Queue consumer must still be alive
            mock_client.chat = AsyncMock(
                return_value=_make_text_response("recovery response")
            )
            second_call_count_before = raising._call_count
            await plugin.on_message(channel=channel, sender="user", text="try again")
            await drain(plugin, channel)

            assert raising._call_count > second_call_count_before, (
                "Queue consumer crashed after error-path send_message failure — "
                "second message was not processed"
            )
        finally:
            task_plugin = plugin.pm.get_plugin("task")
            if task_plugin:
                await task_plugin.on_stop()
            await db.close()


# ---------------------------------------------------------------------------
# Test 5: on_notify in task.py raises — task worker doesn't crash, WARNING logged
# ---------------------------------------------------------------------------


class TestOnNotifyInTaskCompleteSafety:
    """on_notify raising in TaskPlugin._on_task_complete must not crash the worker.

    TaskPlugin._on_task_complete calls self.pm.ahook.on_notify without a try/except.
    If a plugin implementing on_notify raises, the TaskQueue worker task is killed,
    and no subsequent tasks are processed.

    Expected behavior (not yet implemented):
      - The task worker continues running after the exception
      - A WARNING is logged

    Current behavior (causing test to FAIL):
      - The exception propagates out of _on_task_complete and through
        _run_one_worker, crashing the worker loop.
    """

    async def test_on_notify_raise_does_not_crash_task_worker(self, caplog):
        plugin, channel, db = await build_plugin_and_channel(mock_send_message=False, mock_on_agent_response=False)
        try:
            pm = plugin.pm

            # We need send_message to not fail so on_agent_response / send_message
            # don't interfere. Set both as no-op AsyncMocks.
            pm.ahook.send_message = AsyncMock()
            pm.ahook.on_agent_response = AsyncMock()

            raising = RaisingPlugin("on_notify")
            pm.register(raising, name="raising")

            # Tool-based setup: user message -> tool call -> task executes ->
            # _on_task_complete calls on_notify -> RaisingPlugin.on_notify raises
            async def my_tool(x: str) -> str:
                """A simple test tool."""
                return f"result:{x}"

            plugin.tools = {"my_tool": my_tool}

            mock_client = MagicMock()
            mock_client.chat = AsyncMock(
                side_effect=[
                    # First call: returns tool call
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call-001",
                                            "function": {
                                                "name": "my_tool",
                                                "arguments": '{"x": "val"}',
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    },
                    # Second call (only reached if on_notify survives and
                    # re-enqueues the notification): text response
                    _make_text_response("tool done"),
                ]
            )
            plugin.client = mock_client

            with caplog.at_level(logging.WARNING, logger="corvidae"):
                await plugin.on_message(channel=channel, sender="user", text="use tool")
                await drain(plugin, channel)
                await drain_task_queue(plugin)
                # Give the event loop a few cycles to let the exception surface
                for _ in range(5):
                    await asyncio.sleep(0)

            # on_notify must have been called (proving the task completed)
            assert raising._call_count >= 1, (
                "Expected on_notify to be called at least once (task completion)"
            )

            # The task worker must still be alive: enqueue and run a second task
            completed: list[str] = []

            async def second_work():
                completed.append("second")
                return "second result"

            second_task = Task(
                work=second_work,
                channel=channel,
                task_id="task-second",
                description="second task",
            )
            task_plugin = plugin.pm.get_plugin("task")
            await task_plugin.task_queue.enqueue(second_task)
            await drain_task_queue(plugin)
            for _ in range(3):
                await asyncio.sleep(0)

            assert completed == ["second"], (
                "Task worker crashed after on_notify raised — second task was not executed"
            )

            # A WARNING must be logged about the on_notify failure
            warning_records = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING
                and "on_notify" in r.getMessage().lower()
            ]
            assert warning_records, (
                "Expected a WARNING log when on_notify raises in _on_task_complete"
            )
        finally:
            task_plugin = plugin.pm.get_plugin("task")
            if task_plugin:
                await task_plugin.on_stop()
            await db.close()
