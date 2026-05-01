"""Tests for workflow.stage — Stage dataclass, StageRunner."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.tool import Tool as CorvidaeTool
from workflow.types import StageOutcome, StageResult
from workflow.context import WorkflowContext
from workflow.stage import Stage, StageRunner


# -- Helpers for building mock LLM responses --


def _text_response(text: str) -> dict:
    """LLM response with text only, no tool calls."""
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _tool_call_response(calls: list[dict]) -> dict:
    """LLM response with tool calls."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": calls,
            }
        }]
    }


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    """Build a single tool call dict."""
    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# -- Stage dataclass tests --


def test_stage_defaults():
    """Stage has sensible defaults for optional fields."""
    stage = Stage(name="test", instruction="Do something")
    assert stage.tools == []
    assert stage.escape_targets == {}
    assert stage.exit_criteria is None
    assert stage.should_run is None
    assert stage.human_gate is False
    assert stage.is_mechanical is False
    assert stage.mechanical_fn is None


# -- Mechanical stage tests --


async def test_mechanical_stage_runs_fn():
    """Mechanical stages call mechanical_fn without touching the LLM."""
    client = MagicMock()
    client.chat = AsyncMock()
    runner = StageRunner(client)

    async def my_fn(ctx: WorkflowContext) -> StageResult:
        return StageResult(
            stage_name="mech", outcome=StageOutcome.PROCEED, summary="done"
        )

    stage = Stage(
        name="mech", instruction="N/A",
        is_mechanical=True, mechanical_fn=my_fn,
    )
    result = await runner.run(stage, WorkflowContext())

    assert result.outcome == StageOutcome.PROCEED
    assert result.summary == "done"
    client.chat.assert_not_called()


async def test_mechanical_stage_without_fn_raises():
    """Mechanical stage with no mechanical_fn raises ValueError."""
    client = MagicMock()
    runner = StageRunner(client)
    stage = Stage(name="bad", instruction="N/A", is_mechanical=True)

    with pytest.raises(ValueError, match="no mechanical_fn"):
        await runner.run(stage, WorkflowContext())


# -- LLM stage tests: immediate stage_complete --


async def test_llm_stage_immediate_complete():
    """LLM calls stage_complete on the first turn."""
    response = _tool_call_response([
        _tool_call("c1", "stage_complete", {
            "outcome": "proceed", "summary": "Task done",
        })
    ])
    client = MagicMock()
    client.chat = AsyncMock(return_value=response)

    runner = StageRunner(client)
    stage = Stage(name="test", instruction="Do something")
    result = await runner.run(stage, WorkflowContext())

    assert result.outcome == StageOutcome.PROCEED
    assert result.summary == "Task done"
    assert result.stage_name == "test"
    assert result.tool_calls[0].tool_name == "stage_complete"


# -- Multi-turn LLM stage tests --


async def test_llm_stage_multi_turn_with_tools():
    """LLM calls a tool, then calls stage_complete on the next turn."""
    # Turn 1: LLM calls a custom tool
    tool_resp = _tool_call_response([
        _tool_call("c1", "echo", {"message": "hello"})
    ])
    # Turn 2: LLM calls stage_complete
    complete_resp = _tool_call_response([
        _tool_call("c2", "stage_complete", {
            "outcome": "proceed", "summary": "Echoed successfully",
        })
    ])

    client = MagicMock()
    client.chat = AsyncMock(
        side_effect=[tool_resp, complete_resp]
    )

    async def echo(message: str) -> str:
        """Echo a message."""
        return f"Echo: {message}"

    runner = StageRunner(client)
    stage = Stage(
        name="test", instruction="Echo hello",
        tools=[CorvidaeTool.from_function(echo)],
    )
    result = await runner.run(stage, WorkflowContext())

    assert result.outcome == StageOutcome.PROCEED
    assert result.summary == "Echoed successfully"
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "echo"
    assert result.tool_calls[0].arguments == {"message": "hello"}
    assert result.tool_calls[0].result == "Echo: hello"
    assert result.tool_calls[1].tool_name == "stage_complete"
    assert client.chat.await_count == 2


# -- Max turns exhausted tests --


async def test_llm_stage_max_turns_exhausted():
    """If max_turns reached without stage_complete, return BLOCKED."""
    # LLM always returns text, never calls tools
    client = MagicMock()
    client.chat = AsyncMock(
        return_value=_text_response("I'm thinking about it...")
    )

    runner = StageRunner(client, max_turns=3)
    stage = Stage(name="test", instruction="Do something")
    result = await runner.run(stage, WorkflowContext())

    assert result.outcome == StageOutcome.BLOCKED
    assert "stage_complete" in result.summary.lower()


# -- Exit criteria tests --


async def test_exit_criteria_pass():
    """Stage completes when exit criteria pass."""
    response = _tool_call_response([
        _tool_call("c1", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])
    client = MagicMock()
    client.chat = AsyncMock(return_value=response)

    async def always_pass(result, ctx):
        return True

    runner = StageRunner(client)
    stage = Stage(
        name="test", instruction="Do something",
        exit_criteria=always_pass,
    )
    result = await runner.run(stage, WorkflowContext())
    assert result.outcome == StageOutcome.PROCEED


async def test_exit_criteria_fail_returns_blocked():
    """Stage returns BLOCKED when exit criteria fail after retries."""
    response = _tool_call_response([
        _tool_call("c1", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])
    client = MagicMock()
    client.chat = AsyncMock(return_value=response)

    async def always_fail(result, ctx):
        return False

    runner = StageRunner(client, max_turns=10)
    stage = Stage(
        name="test", instruction="Do something",
        exit_criteria=always_fail,
    )
    result = await runner.run(stage, WorkflowContext())
    assert result.outcome == StageOutcome.BLOCKED
    assert "exit criteria" in result.summary.lower()
