"""Tests for workflow.stage — Stage dataclass, StageRunner."""

import json
import logging
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


# -- Logging enrichment tests --


def _make_stage_runner_with_responses(responses):
    """Build a StageRunner with a mock client returning the given responses."""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=responses)
    return StageRunner(client)


async def _collect_workflow_logs(runner, stage, context, caplog):
    """Run a stage and collect logs from the workflow.stage logger.

    Ensures the workflow logger propagates so caplog can capture records.
    Restores the original propagation value on cleanup.
    """
    wf_logger = logging.getLogger("workflow")
    original_propagate = wf_logger.propagate
    wf_logger.propagate = True
    try:
        with caplog.at_level(logging.DEBUG, logger="workflow.stage"):
            result = await runner.run(stage, context)
    finally:
        wf_logger.propagate = original_propagate
    return result, caplog.records


def _find_log_records(records, message_substring):
    """Filter log records to those containing the given substring."""
    return [
        r for r in records if message_substring in r.message
    ]


async def test_log_tool_call_dispatched_with_name_and_args(caplog):
    """Tool calls are logged at INFO with tool name and truncated arguments."""
    echo_resp = _tool_call_response([
        _tool_call("c1", "echo", {"message": "hello world"})
    ])
    complete_resp = _tool_call_response([
        _tool_call("c2", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])

    async def echo(message: str) -> str:
        """Echo a message."""
        return f"Echo: {message}"

    runner = _make_stage_runner_with_responses([echo_resp, complete_resp])
    stage = Stage(
        name="test", instruction="Echo hello",
        tools=[CorvidaeTool.from_function(echo)],
    )

    result, records = await _collect_workflow_logs(
        runner, stage, WorkflowContext(), caplog
    )

    # Should log tool dispatch with tool name
    dispatch_logs = _find_log_records(records, "Tool dispatched")
    assert len(dispatch_logs) == 2  # echo + stage_complete
    # First dispatch should be the echo tool
    assert "echo" in dispatch_logs[0].message


def _extract_value_from_log(record, key):
    """Extract a key=value pair from a log message string."""
    msg = record.message
    # Look for key='value' or key="value" or key=value
    import re
    match = re.search(rf"{key}='([^']*)'", msg)
    if match:
        return match.group(1)
    return None


async def test_log_tool_call_result_with_name_and_truncated_result(caplog):
    """Tool call results are logged at INFO with tool name and truncated result."""
    echo_resp = _tool_call_response([
        _tool_call("c1", "echo", {"message": "hello"})
    ])
    complete_resp = _tool_call_response([
        _tool_call("c2", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])

    async def echo(message: str) -> str:
        """Echo a message."""
        return f"Echo: {message}"

    runner = _make_stage_runner_with_responses([echo_resp, complete_resp])
    stage = Stage(
        name="test", instruction="Echo hello",
        tools=[CorvidaeTool.from_function(echo)],
    )

    result, records = await _collect_workflow_logs(
        runner, stage, WorkflowContext(), caplog
    )

    # Should log tool result with tool name
    result_logs = _find_log_records(records, "Tool result")
    assert len(result_logs) >= 1
    assert "echo" in result_logs[0].message
    assert "Echo: hello" in result_logs[0].message


async def test_log_tool_call_result_truncates_long_content(caplog):
    """Tool call results are truncated when they exceed 500 chars."""
    long_response = "x" * 1000

    async def long_tool(message: str) -> str:
        """Return a long string."""
        return long_response

    tool_resp = _tool_call_response([
        _tool_call("c1", "long_tool", {"message": "hi"})
    ])
    complete_resp = _tool_call_response([
        _tool_call("c2", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])

    runner = _make_stage_runner_with_responses([tool_resp, complete_resp])
    stage = Stage(
        name="test", instruction="Do something",
        tools=[CorvidaeTool.from_function(long_tool)],
    )

    result, records = await _collect_workflow_logs(
        runner, stage, WorkflowContext(), caplog
    )

    result_logs = _find_log_records(records, "Tool result")
    assert len(result_logs) >= 1
    # The log message ends with truncation indicator
    assert result_logs[0].message.rstrip().endswith("...")
    # Should be much shorter than the full 1000-char result
    # (the full message is "long_tool " + 1000 x's = ~1010 chars)
    assert len(result_logs[0].message) < 600


async def test_log_llm_response_text(caplog):
    """LLM response text is logged at INFO (truncated)."""
    # First response: text + tool call
    text_tool_resp = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "I will echo hello now",
                "tool_calls": [
                    _tool_call("c1", "stage_complete", {
                        "outcome": "proceed", "summary": "All done",
                    })
                ],
            }
        }]
    }

    runner = _make_stage_runner_with_responses([text_tool_resp])
    stage = Stage(name="test", instruction="Do something")

    result, records = await _collect_workflow_logs(
        runner, stage, WorkflowContext(), caplog
    )

    # Should log LLM response text
    llm_logs = _find_log_records(records, "LLM response text")
    assert len(llm_logs) == 1
    assert "I will echo hello now" in llm_logs[0].message


async def test_log_stage_summary_with_tool_counts(caplog):
    """After stage completion, a compact summary is logged with tool call counts."""
    echo_resp = _tool_call_response([
        _tool_call("c1", "echo", {"message": "hello"}),
        _tool_call("c2", "echo", {"message": "world"}),
    ])
    complete_resp = _tool_call_response([
        _tool_call("c3", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])

    async def echo(message: str) -> str:
        """Echo a message."""
        return f"Echo: {message}"

    runner = _make_stage_runner_with_responses([echo_resp, complete_resp])
    stage = Stage(
        name="implement", instruction="Echo things",
        tools=[CorvidaeTool.from_function(echo)],
    )

    result, records = await _collect_workflow_logs(
        runner, stage, WorkflowContext(), caplog
    )

    # Should log a stage summary with tool call counts
    summary_logs = _find_log_records(records, "Stage summary")
    assert len(summary_logs) == 1
    assert "implement" in summary_logs[0].message
    # Should mention 3 tool calls (2 echo + 1 stage_complete)
    msg = summary_logs[0].message
    assert "3 tool calls" in msg
    # Should break down by tool name
    assert "echo" in msg
    assert "stage_complete" in msg


async def test_log_tool_args_truncated(caplog):
    """Tool call arguments are truncated in log output when long."""
    long_args = {"message": "x" * 1000}

    tool_resp = _tool_call_response([
        _tool_call("c1", "echo", long_args)
    ])
    complete_resp = _tool_call_response([
        _tool_call("c2", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])

    async def echo(message: str) -> str:
        """Echo a message."""
        return f"Echo: {message}"

    runner = _make_stage_runner_with_responses([tool_resp, complete_resp])
    stage = Stage(
        name="test", instruction="Echo",
        tools=[CorvidaeTool.from_function(echo)],
    )

    result, records = await _collect_workflow_logs(
        runner, stage, WorkflowContext(), caplog
    )

    dispatch_logs = _find_log_records(records, "Tool dispatched")
    assert len(dispatch_logs) >= 1
    # The log message ends with truncation indicator
    assert dispatch_logs[0].message.rstrip().endswith("...")
    # Should be much shorter than the full 1000-char args
    assert len(dispatch_logs[0].message) < 700


# -- Plan inclusion in LLM prompt tests --


async def test_plan_included_in_llm_prompt():
    """The plan artifact from context.facts is included in the LLM prompt."""
    response = _tool_call_response([
        _tool_call("c1", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])
    client = MagicMock()
    # Capture the messages sent to the LLM
    chat_calls = []
    async def capture_chat(messages, **kwargs):
        chat_calls.append(messages)
        return response
    client.chat = AsyncMock(side_effect=capture_chat)

    runner = StageRunner(client)
    stage = Stage(name="create_branch", instruction="Create a branch")

    # Create context with a plan artifact
    ctx = WorkflowContext()
    ctx.facts["plan"] = {
        "branch_name": "feat/issue-42",
        "task_type": "feature",
        "needs_tests": True,
    }

    await runner.run(stage, ctx)

    # Verify the LLM received the plan in its prompt
    assert len(chat_calls) == 1
    messages = chat_calls[0]
    user_msg = messages[-1]["content"] if isinstance(messages[-1], dict) else str(messages[-1])
    # Find the user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    assert len(user_messages) == 1
    content = user_messages[0]["content"]
    assert "Implementation Plan" in content
    assert "feat/issue-42" in content
    assert "feature" in content


async def test_no_plan_in_prompt_when_absent():
    """When no plan exists, the LLM prompt doesn't mention it."""
    response = _tool_call_response([
        _tool_call("c1", "stage_complete", {
            "outcome": "proceed", "summary": "Done",
        })
    ])
    client = MagicMock()
    chat_calls = []
    async def capture_chat(messages, **kwargs):
        chat_calls.append(messages)
        return response
    client.chat = AsyncMock(side_effect=capture_chat)

    runner = StageRunner(client)
    stage = Stage(name="test", instruction="Do something")

    ctx = WorkflowContext()  # No plan
    await runner.run(stage, ctx)

    user_messages = [
        m for m in chat_calls[0] if m.get("role") == "user"
    ]
    content = user_messages[0]["content"]
    assert "Implementation Plan" not in content
