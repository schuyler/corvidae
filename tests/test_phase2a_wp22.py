"""RED tests for WP2.2 — Silent tasks + logprob passthrough (Phase 2A).

Scope (per plans/implementation/phase-2.md, "Phase 2A — Design" section,
WP2.2, as corrected by Design Review finding 1 / Fixes Applied / Re-review #1):

1. Task.deliver: bool = True + __post_init__ raising ValueError when
   deliver=False and tool_call_id is not None (trap #5).
   TaskPlugin._on_task_complete returns immediately for deliver=False tasks
   -- no send_tool_status, no on_notify, no main-model turn.
2. run_agent_turn extracts response["choices"][0].get("logprobs") before
   discarding the choice envelope (turn.py:75 currently keeps only
   ["message"]); AgentTurnResult gains logprobs: dict | None = None.
3. Request-body logprobs: Agent.on_init captures
   self._request_logprobs = agent_config.get("request_logprobs", False)
   (a new Agent scalar -- NOT self.config, which Agent does not retain).
   The step-7 extra_body merge of {"logprobs": True} when
   self._request_logprobs is true is WP2.3's edit (not WP2.2's); this test
   file only exercises the read/capture site and the observable outcome at
   the request boundary, per the design's corrected red-test #3.

This file is DEDICATED to WP2.2 per the parallel-authoring instructions: no
other test file or conftest is modified. Fixtures are defined locally or
reused from existing importable helpers (tests/helpers.py,
tests/llm_response_fixtures.py) only.

All tests in this file are expected to FAIL (red) against HEAD
bb03fa5bbc46e31ba757e133b38912b62fc65783 -- the WP2.2 implementation does not
exist yet.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import create_plugin_manager
from corvidae.task import Task, TaskPlugin
from corvidae.turn import AgentTurnResult, run_agent_turn

from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response


def _make_channel(transport="test", scope="scope1") -> Channel:
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


# ---------------------------------------------------------------------------
# 1a. Task.deliver field + __post_init__ guard (trap #5)
# ---------------------------------------------------------------------------


class TestTaskDeliverField:
    def test_deliver_defaults_true(self):
        """Task.deliver defaults to True (existing callers unaffected)."""
        channel = _make_channel()

        async def work():
            return "done"

        task = Task(work=work, channel=channel)
        assert task.deliver is True

    def test_deliver_false_without_tool_call_id_is_allowed(self):
        """deliver=False with no tool_call_id constructs cleanly."""
        channel = _make_channel()

        async def work():
            return "done"

        task = Task(work=work, channel=channel, deliver=False)
        assert task.deliver is False
        assert task.tool_call_id is None

    def test_deliver_false_with_tool_call_id_raises_value_error(self):
        """deliver=False AND tool_call_id set is invalid (trap #5): a silent
        task holding a tool-call id would leave pending_tool_call_ids never
        clearing, stalling the channel's tool batch forever."""
        channel = _make_channel()

        async def work():
            return "done"

        with pytest.raises(ValueError):
            Task(work=work, channel=channel, deliver=False, tool_call_id="call_1")

    def test_deliver_true_with_tool_call_id_is_allowed(self):
        """deliver=True (the default/existing behavior) with a tool_call_id
        is unaffected by the new guard."""
        channel = _make_channel()

        async def work():
            return "done"

        task = Task(work=work, channel=channel, deliver=True, tool_call_id="call_2")
        assert task.deliver is True
        assert task.tool_call_id == "call_2"


# ---------------------------------------------------------------------------
# 1b. TaskPlugin._on_task_complete early-return for deliver=False
# ---------------------------------------------------------------------------


class TestSilentTaskCompletion:
    async def test_deliver_false_task_completion_does_not_fire_on_notify(self):
        """deliver=False task completes without firing on_notify (spy) --
        no send_tool_status, no on_notify, no downstream main-model turn."""
        pm = create_plugin_manager()
        pm.ahook.on_notify = AsyncMock()
        pm.ahook.send_tool_status = AsyncMock()
        plugin = TaskPlugin(pm)
        await plugin.on_start(config={})

        channel = _make_channel()

        async def work():
            return "silent work output"

        task = Task(
            work=work,
            channel=channel,
            deliver=False,
            description="silent test",
        )

        await plugin._on_task_complete(task, "silent work output")

        pm.ahook.on_notify.assert_not_awaited()
        pm.ahook.send_tool_status.assert_not_awaited()

        await plugin.on_stop()

    async def test_deliver_true_task_completion_still_fires_on_notify(self):
        """deliver=True (default/existing) behavior is unchanged -- on_notify
        still fires."""
        pm = create_plugin_manager()
        pm.ahook.on_notify = AsyncMock()
        plugin = TaskPlugin(pm)
        await plugin.on_start(config={})

        channel = _make_channel()

        async def work():
            return "normal work output"

        task = Task(
            work=work,
            channel=channel,
            deliver=True,
            description="normal test",
        )

        await plugin._on_task_complete(task, "normal work output")

        pm.ahook.on_notify.assert_awaited_once()

        await plugin.on_stop()

    async def test_deliver_false_task_completion_logs_but_returns_immediately(self):
        """deliver=False suppresses completion delivery (_on_task_complete's
        hook firing) even when the task result looks like a failure string --
        the deliver flag, not the result content, gates delivery. Note: this
        test calls _on_task_complete directly (not through _run_one_worker),
        so it does not exercise or assert on the pre-existing
        `logger.warning("task failed", ...)` call in _run_one_worker --
        that worker-level failure logging is an unmodified code path outside
        WP2.2's scope and is unaffected by the deliver flag either way."""
        pm = create_plugin_manager()
        pm.ahook.on_notify = AsyncMock()
        pm.ahook.send_tool_status = AsyncMock()
        plugin = TaskPlugin(pm)
        await plugin.on_start(config={})

        channel = _make_channel()

        async def work():
            raise RuntimeError("boom")

        task = Task(
            work=work,
            channel=channel,
            deliver=False,
            description="silent failing test",
        )

        # Simulate what _run_one_worker would pass as `result` on failure.
        failure_result = "Task deadbeef failed: boom"
        await plugin._on_task_complete(task, failure_result)

        pm.ahook.on_notify.assert_not_awaited()
        pm.ahook.send_tool_status.assert_not_awaited()

        await plugin.on_stop()


# ---------------------------------------------------------------------------
# 2. run_agent_turn logprobs extraction + AgentTurnResult.logprobs
# ---------------------------------------------------------------------------


class TestAgentTurnResultLogprobs:
    def test_agent_turn_result_has_logprobs_field_defaulting_none(self):
        """AgentTurnResult gains logprobs: dict | None = None."""
        result = AgentTurnResult(
            message={"role": "assistant", "content": "hi"},
            tool_calls=[],
            text="hi",
            latency_ms=1.0,
        )
        assert result.logprobs is None

    async def test_run_agent_turn_extracts_logprobs_when_present(self):
        """A response whose choice envelope carries a logprobs key populates
        AgentTurnResult.logprobs -- turn.py:75 currently discards everything
        but response["choices"][0]["message"]."""
        logprobs_envelope = {
            "content": [
                {"token": "Hello", "logprob": -0.1, "top_logprobs": []},
            ]
        }
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello"},
                    "logprobs": logprobs_envelope,
                }
            ]
        }
        client = MagicMock()
        client.chat = AsyncMock(return_value=response)

        messages = [{"role": "user", "content": "hi"}]
        result = await run_agent_turn(client, messages, tool_schemas=[])

        assert result.logprobs == logprobs_envelope

    async def test_run_agent_turn_logprobs_none_when_absent(self):
        """A response with no logprobs key on the choice envelope (e.g.
        Anthropic-style providers) surfaces AgentTurnResult.logprobs as None
        -- never a faked substitute."""
        client = MagicMock()
        client.chat = AsyncMock(return_value=_make_text_response("no logprobs here"))

        messages = [{"role": "user", "content": "hi"}]
        result = await run_agent_turn(client, messages, tool_schemas=[])

        assert result.logprobs is None

    async def test_run_agent_turn_logprobs_none_when_explicitly_null(self):
        """A response with an explicit logprobs: null on the choice envelope
        surfaces AgentTurnResult.logprobs as None."""
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "logprobs": None,
                }
            ]
        }
        client = MagicMock()
        client.chat = AsyncMock(return_value=response)

        messages = [{"role": "user", "content": "hi"}]
        result = await run_agent_turn(client, messages, tool_schemas=[])

        assert result.logprobs is None


# ---------------------------------------------------------------------------
# 2b. logprobs threaded into the enriched on_agent_response call
#     (the one agent.py line WP2.2 owns, downstream of step 7)
# ---------------------------------------------------------------------------


class TestLogprobsThreadedIntoOnAgentResponse:
    async def test_on_agent_response_receives_logprobs_from_turn_result(self):
        """result.logprobs from run_agent_turn is threaded into the
        on_agent_response hook call. WP2.1 enriches the on_agent_response
        hookspec with a logprobs param; WP2.2's sole agent.py responsibility
        is passing result.logprobs into that call."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._client = MagicMock()

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "answer"},
            tool_calls=[],
            text="answer",
            latency_ms=1.0,
            logprobs={"content": [{"token": "answer", "logprob": -0.05}]},
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="query")
            await drain(plugin, channel)

        plugin.pm.ahook.on_agent_response.assert_awaited_once()
        call_kwargs = plugin.pm.ahook.on_agent_response.call_args.kwargs
        assert call_kwargs.get("logprobs") == {
            "content": [{"token": "answer", "logprob": -0.05}]
        }

        await db.close()

    async def test_on_agent_response_receives_none_logprobs_when_absent(self):
        """When the turn result carries no logprobs, on_agent_response is
        called with logprobs=None (not omitted, not a faked substitute)."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._client = MagicMock()

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "answer"},
            tool_calls=[],
            text="answer",
            latency_ms=1.0,
            logprobs=None,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="query")
            await drain(plugin, channel)

        plugin.pm.ahook.on_agent_response.assert_awaited_once()
        call_kwargs = plugin.pm.ahook.on_agent_response.call_args.kwargs
        assert call_kwargs.get("logprobs") is None

        await db.close()


# ---------------------------------------------------------------------------
# 3. Request-body logprobs: Agent.on_init captures self._request_logprobs
#    (the corrected read site -- NOT self.config, which Agent does not
#    retain; see Design Review finding 1 / Fixes Applied / Re-review #1).
# ---------------------------------------------------------------------------


class TestRequestLogprobsScalarCapturedAtInit:
    async def test_on_init_captures_request_logprobs_true(self):
        """agent.request_logprobs: true in config -> Agent captures it as
        self._request_logprobs = True at on_init, matching the existing
        _chars_per_token / _idle_cooldown sibling-scalar pattern."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        plugin = Agent()
        plugin.pm = pm

        config = {"agent": {"request_logprobs": True}}
        await plugin.on_init(pm=pm, config=config)

        assert plugin._request_logprobs is True

    async def test_on_init_captures_request_logprobs_false_explicit(self):
        """agent.request_logprobs: false in config -> self._request_logprobs
        is False."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        plugin = Agent()
        plugin.pm = pm

        config = {"agent": {"request_logprobs": False}}
        await plugin.on_init(pm=pm, config=config)

        assert plugin._request_logprobs is False

    async def test_on_init_defaults_request_logprobs_false_when_absent(self):
        """No agent.request_logprobs key in config -> self._request_logprobs
        defaults to False (llama-server supports the flag; Anthropic-style
        providers return nothing regardless)."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        plugin = Agent()
        plugin.pm = pm

        config = {"agent": {}}
        await plugin.on_init(pm=pm, config=config)

        assert plugin._request_logprobs is False

    async def test_on_init_defaults_request_logprobs_false_with_no_agent_key(self):
        """No 'agent' key at all in config -> self._request_logprobs is
        still False (matches the existing agent_config = config.get("agent",
        {}) fallback pattern used for _chars_per_token / _idle_cooldown)."""
        from corvidae.agent import Agent

        pm = create_plugin_manager()
        plugin = Agent()
        plugin.pm = pm

        await plugin.on_init(pm=pm, config={})

        assert plugin._request_logprobs is False


class TestRequestLogprobsInRequestBody:
    """Verifies the request-body outcome end to end (integration-style, via
    on_message -> _process_queue_item -> step 7 -> run_agent_turn), matching
    the design's corrected red-test #3: 'agent.request_logprobs: true in
    config puts "logprobs": true in the request body; false/absent does not.'

    This is WP2.2's own red-test obligation per the plan (it verifies WP2.2's
    logprobs behavior against WP2.3's merged step-7 filter); the step-7 merge
    implementation itself is WP2.3's edit, not WP2.2's. Per the 2A design's
    residual cosmetic C3, this bullet is only green once WP2.3's step-7 edit
    has landed -- until then it fails for the intended reason (no
    self._request_logprobs / no logprobs key merged into extra_body).
    """

    async def test_request_logprobs_true_adds_logprobs_key_to_extra_body(self):
        """agent.request_logprobs: true -> "logprobs": true appears in the
        extra_body passed to run_agent_turn (and therefore the request
        body)."""
        plugin, channel, db = await build_plugin_and_channel(
            agent_defaults={
                "system_prompt": "You are a test assistant.",
                "max_context_tokens": 8000,
                "keep_thinking_in_history": False,
            },
        )
        plugin._client = MagicMock()
        await plugin.on_init(pm=plugin.pm, config={"agent": {"request_logprobs": True}})

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "answer"},
            tool_calls=[],
            text="answer",
            latency_ms=1.0,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="query")
            await drain(plugin, channel)

        mock_turn.assert_awaited_once()
        call_args = mock_turn.call_args
        extra_body = call_args.kwargs.get("extra_body")
        if extra_body is None and len(call_args.args) >= 4:
            extra_body = call_args.args[3]
        assert extra_body is not None, "expected extra_body to carry the logprobs flag"
        assert extra_body.get("logprobs") is True

        await db.close()

    async def test_request_logprobs_false_omits_logprobs_key(self):
        """agent.request_logprobs: false (or absent) -> no "logprobs" key is
        merged into extra_body."""
        plugin, channel, db = await build_plugin_and_channel(
            agent_defaults={
                "system_prompt": "You are a test assistant.",
                "max_context_tokens": 8000,
                "keep_thinking_in_history": False,
            },
        )
        plugin._client = MagicMock()
        await plugin.on_init(pm=plugin.pm, config={"agent": {"request_logprobs": False}})

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "answer"},
            tool_calls=[],
            text="answer",
            latency_ms=1.0,
        )

        with patch("corvidae.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="query")
            await drain(plugin, channel)

        mock_turn.assert_awaited_once()
        call_args = mock_turn.call_args
        extra_body = call_args.kwargs.get("extra_body")
        if extra_body is None and len(call_args.args) >= 4:
            extra_body = call_args.args[3]
        if extra_body is not None:
            assert "logprobs" not in extra_body

        await db.close()
