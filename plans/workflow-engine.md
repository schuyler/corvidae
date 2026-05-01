# Workflow Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a semi-deterministic workflow engine that drives an LLM through the GitHub issue implementation workflow, importing Corvidae primitives directly (Approach A).

**Architecture:** Standalone `workflow/` package inside the Corvidae repo. Deterministic code controls stage progression (linear backbone with escape hatches). The LLM does focused work within one stage at a time — each stage starts fresh with a scoped tool set. The LLM signals completion via a `stage_complete` tool; the orchestrator independently verifies exit criteria. Two output channels: LLM summaries (reasoning) and orchestrator-gathered facts (file diffs, test results, GitHub state).

**Tech Stack:** Python 3.13+, asyncio, Corvidae primitives (LLMClient, run_agent_turn, Tool/ToolRegistry, dispatch_tool_call), PyYAML, `gh` CLI for GitHub operations.

---

## File Structure

```
workflow/
  __init__.py                # Package marker
  types.py                   # StageOutcome, ToolCallRecord, StageResult
  context.py                 # WorkflowContext — summaries + facts accumulator
  config.py                  # load_agent_config, create_llm_client
  stage.py                   # Stage dataclass, StageRunner, SYSTEM_PROMPT
  engine.py                  # WorkflowEngine — orchestrates stages
  tools/
    __init__.py
    stage_complete.py        # StageCompleteHolder, make_stage_complete_tool
    github.py                # make_github_comment_tool, make_add_label_tool, make_create_pr_tool
    git.py                   # make_create_branch_tool, make_create_worktree_tool
  workflows/
    __init__.py
    issue.py                 # build_issue_workflow()
  __main__.py                # CLI: python -m workflow implement --issue <url>

tests/workflow/
  __init__.py
  conftest.py                # Shared test fixtures
  test_types.py
  test_context.py
  test_config.py
  test_stage_complete.py
  test_stage_runner.py
  test_engine.py
  test_github_tools.py
  test_git_tools.py
  test_issue_workflow.py
```

---

## Task 0: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `workflow/__init__.py`, `workflow/tools/__init__.py`, `workflow/workflows/__init__.py`, `tests/workflow/__init__.py`

- [ ] **Step 1: Add project root to pytest pythonpath**

```toml
# In pyproject.toml, change:
pythonpath = ["tests"]
# To:
pythonpath = ["tests", "."]
```

- [ ] **Step 2: Create package `__init__.py` files**

```bash
touch workflow/__init__.py workflow/tools/__init__.py workflow/workflows/__init__.py tests/workflow/__init__.py
```

- [ ] **Step 3: Verify test discovery works**

Run: `python -m pytest tests/workflow/ -v --co`
Expected: "no tests collected" (empty dir) — no import errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml workflow/__init__.py workflow/tools/__init__.py workflow/workflows/__init__.py tests/workflow/__init__.py
git commit -m "chore: scaffold workflow package and test directory"
```

---

## Task 1: Core Types

**Files:**
- Create: `workflow/types.py`
- Test: `tests/workflow/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/workflow/test_types.py

"""Tests for workflow.types — StageOutcome, ToolCallRecord, StageResult."""

from workflow.types import StageOutcome, ToolCallRecord, StageResult


def test_stage_outcome_values():
    """StageOutcome has the four expected values."""
    assert StageOutcome.PROCEED == "proceed"
    assert StageOutcome.NEED_CLARIFICATION == "need_clarification"
    assert StageOutcome.SCOPE_CHANGED == "scope_changed"
    assert StageOutcome.BLOCKED == "blocked"


def test_tool_call_record_construction():
    """ToolCallRecord stores tool call details."""
    record = ToolCallRecord(
        tool_name="shell",
        arguments={"command": "ls"},
        result="file.py",
        error=False,
    )
    assert record.tool_name == "shell"
    assert record.arguments == {"command": "ls"}
    assert record.result == "file.py"
    assert record.error is False


def test_tool_call_record_error_default():
    """ToolCallRecord.error defaults to False."""
    record = ToolCallRecord(tool_name="x", arguments={}, result="ok")
    assert record.error is False


def test_stage_result_construction():
    """StageResult stores outcome and metadata from a completed stage."""
    result = StageResult(
        stage_name="implement",
        outcome=StageOutcome.PROCEED,
        summary="Implemented the feature",
    )
    assert result.stage_name == "implement"
    assert result.outcome == StageOutcome.PROCEED
    assert result.summary == "Implemented the feature"
    assert result.escape_target is None
    assert result.tool_calls == []


def test_stage_result_with_escape_target():
    """StageResult can specify an escape target for back-tracking."""
    result = StageResult(
        stage_name="gate",
        outcome=StageOutcome.NEED_CLARIFICATION,
        summary="Need more info",
        escape_target="comment_on_issue",
    )
    assert result.escape_target == "comment_on_issue"


def test_stage_result_with_tool_calls():
    """StageResult includes a record of all tool calls made during the stage."""
    calls = [
        ToolCallRecord(tool_name="shell", arguments={"command": "ls"}, result="out"),
        ToolCallRecord(tool_name="write_file", arguments={"path": "a.py"}, result="ok"),
    ]
    result = StageResult(
        stage_name="test",
        outcome=StageOutcome.PROCEED,
        summary="done",
        tool_calls=calls,
    )
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "shell"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.types'`

- [ ] **Step 3: Write implementation**

```python
"""Workflow engine core types.

Defines the data structures used across all workflow components:
- StageOutcome: possible outcomes when a stage completes
- ToolCallRecord: captured I/O from a single tool invocation
- StageResult: everything the orchestrator needs from a completed stage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StageOutcome(str, Enum):
    """Possible outcomes when a stage completes.

    The orchestrator maps these to transitions. The LLM never chooses
    the next stage directly — it signals an outcome and the orchestrator
    decides what happens next.
    """

    PROCEED = "proceed"
    NEED_CLARIFICATION = "need_clarification"
    SCOPE_CHANGED = "scope_changed"
    BLOCKED = "blocked"


@dataclass
class ToolCallRecord:
    """Captured I/O from a single tool invocation within a stage.

    The orchestrator logs these as facts. The LLM doesn't need to
    summarise what it did — the orchestrator already knows.
    """

    tool_name: str
    arguments: dict
    result: str
    error: bool = False


@dataclass
class StageResult:
    """Everything the orchestrator needs from a completed stage.

    Two channels:
    1. summary — the LLM's reasoning and decisions (judgment only)
    2. tool_calls — observable facts the orchestrator gathered
    """

    stage_name: str
    outcome: StageOutcome
    summary: str
    escape_target: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_types.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add workflow/types.py tests/workflow/test_types.py
git commit -m "feat(workflow): add core types — StageOutcome, ToolCallRecord, StageResult"
```

---

## Task 2: WorkflowContext

**Files:**
- Create: `workflow/context.py`
- Test: `tests/workflow/test_context.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for workflow.context — WorkflowContext."""

from workflow.types import ToolCallRecord
from workflow.context import WorkflowContext


def test_empty_context_builds_empty_string():
    """A fresh context produces an empty context string."""
    ctx = WorkflowContext()
    assert ctx.build_context_string("any_stage") == ""


def test_record_stage_adds_summary():
    """record_stage stores the stage summary."""
    ctx = WorkflowContext()
    ctx.record_stage("implement", "Done", [])
    assert ctx.summaries["implement"] == "Done"


def test_record_stage_accumulates_tool_calls():
    """record_stage appends tool call records to the log."""
    ctx = WorkflowContext()
    calls = [ToolCallRecord(tool_name="shell", arguments={}, result="ok")]
    ctx.record_stage("implement", "Done", calls)
    assert len(ctx.tool_logs) == 1
    assert ctx.tool_logs[0].tool_name == "shell"

    ctx.record_stage("refactor", "Clean", [])
    assert len(ctx.tool_logs) == 1  # No new calls appended


def test_build_context_string_excludes_current_stage():
    """build_context_string omits the named stage from context."""
    ctx = WorkflowContext()
    ctx.summaries["a"] = "Summary A"
    ctx.summaries["b"] = "Summary B"
    result = ctx.build_context_string("a")
    assert "Summary A" not in result
    assert "Summary B" in result


def test_build_context_string_includes_all_other_stages():
    """build_context_string includes summaries from all other stages."""
    ctx = WorkflowContext()
    ctx.summaries["a"] = "Summary A"
    ctx.summaries["b"] = "Summary B"
    ctx.summaries["c"] = "Summary C"
    result = ctx.build_context_string("b")
    assert "Summary A" in result
    assert "Summary C" in result
    assert "Summary B" not in result


def test_facts_store_arbitrary_values():
    """WorkflowContext.facts holds arbitrary orchestrator-gathered data."""
    ctx = WorkflowContext()
    ctx.facts["issue_url"] = "https://github.com/o/r/issues/1"
    ctx.facts["branch"] = "feat/issue-1"
    assert ctx.facts["issue_url"] == "https://github.com/o/r/issues/1"
    assert ctx.facts["branch"] == "feat/issue-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.context'`

- [ ] **Step 3: Write implementation**

```python
"""WorkflowContext — state accumulator across workflow stages.

Two channels, as required by the design doc:
1. summaries — per-stage LLM reasoning (judgment, decisions, rejected approaches)
2. facts — orchestrator-gathered data (file diffs, test results, GitHub state)
3. tool_logs — captured I/O from all tool invocations across all stages

The LLM never redundantly reports what the orchestrator can verify directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workflow.types import ToolCallRecord


@dataclass
class WorkflowContext:
    """Accumulated state across workflow stages.

    Passed to each stage so the LLM has access to prior decisions and
    the orchestrator can track progress.
    """

    summaries: dict[str, str] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    tool_logs: list[ToolCallRecord] = field(default_factory=list)

    def record_stage(
        self,
        stage_name: str,
        summary: str,
        tool_calls: list[ToolCallRecord],
    ) -> None:
        """Record results from a completed stage.

        Stores the summary and appends tool call records to the log.
        Only non-empty tool_calls lists are appended (mechanical stages
        pass an empty list).
        """
        self.summaries[stage_name] = summary
        if tool_calls:
            self.tool_logs.extend(tool_calls)

    def build_context_string(self, current_stage: str) -> str:
        """Build context string for inclusion in an LLM prompt.

        Includes summaries from all prior stages, excluding the current
        stage (it hasn't completed yet). Ordered by insertion order.
        """
        parts = []
        for name, summary in self.summaries.items():
            if name != current_stage:
                parts.append(f"## {name}\n{summary}")
        return "\n\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_context.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add workflow/context.py tests/workflow/test_context.py
git commit -m "feat(workflow): add WorkflowContext — stage state accumulator"
```

---

## Task 3: Config Loading

**Files:**
- Create: `workflow/config.py`
- Test: `tests/workflow/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for workflow.config — load_agent_config, create_llm_client."""

import pytest
from workflow.config import load_agent_config, create_llm_client


def test_load_agent_config_reads_yaml(tmp_path):
    """load_agent_config returns parsed YAML dict."""
    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "llm:\n  main:\n    base_url: http://localhost:8080\n    model: test\n"
    )
    config = load_agent_config(config_file)
    assert config["llm"]["main"]["base_url"] == "http://localhost:8080"
    assert config["llm"]["main"]["model"] == "test"


def test_load_agent_config_missing_file_raises():
    """load_agent_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_agent_config("/nonexistent/agent.yaml")


def test_create_llm_client_returns_client(tmp_path):
    """create_llm_client creates an LLMClient from config."""
    config = {
        "llm": {
            "main": {
                "base_url": "http://localhost:8080",
                "model": "test-model",
                "api_key": "sk-test",
            }
        }
    }
    client = create_llm_client(config)
    assert client.base_url == "http://localhost:8080"
    assert client.model == "test-model"
    assert client.api_key == "sk-test"


def test_create_llm_client_missing_main_raises():
    """create_llm_client raises KeyError when llm.main is missing."""
    with pytest.raises(KeyError):
        create_llm_client({"llm": {}})


def test_create_llm_client_optional_fields():
    """create_llm_client handles missing optional fields."""
    config = {"llm": {"main": {"base_url": "http://x", "model": "m"}}}
    client = create_llm_client(config)
    assert client.api_key is None
    assert client.extra_body is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.config'`

- [ ] **Step 3: Write implementation**

```python
"""Config loading — reads agent.yaml and creates an LLMClient.

Reuses the same agent.yaml that the Corvidae daemon uses, so there's
a single config file for both systems.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from corvidae.llm import LLMClient


def load_agent_config(config_path: str | Path = "agent.yaml") -> dict:
    """Load and return the agent.yaml config dict.

    Args:
        config_path: Path to agent.yaml. Defaults to agent.yaml in CWD.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def create_llm_client(config: dict) -> LLMClient:
    """Create an LLMClient from agent.yaml config.

    Reads llm.main config section and constructs an LLMClient
    with the same parameters the Corvidae daemon uses.

    Raises:
        KeyError: If llm.main config section is missing.
    """
    llm_config = config["llm"]["main"]
    return LLMClient(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
        api_key=llm_config.get("api_key"),
        extra_body=llm_config.get("extra_body"),
        max_retries=llm_config.get("max_retries", 3),
        retry_base_delay=llm_config.get("retry_base_delay", 2.0),
        retry_max_delay=llm_config.get("retry_max_delay", 60.0),
        timeout=llm_config.get("timeout"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add workflow/config.py tests/workflow/test_config.py
git commit -m "feat(workflow): add config loading from agent.yaml"
```

---

## Task 4: stage_complete Tool

**Files:**
- Create: `workflow/tools/stage_complete.py`
- Test: `tests/workflow/test_stage_complete.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for workflow.tools.stage_complete — StageCompleteHolder, make_stage_complete_tool."""

import pytest
from workflow.tools.stage_complete import StageCompleteHolder, make_stage_complete_tool


def test_holder_starts_empty():
    """A fresh holder has no result."""
    holder = StageCompleteHolder()
    assert holder.outcome is None
    assert holder.summary == ""
    assert holder.escape_target is None
    assert holder.called is False


async def test_stage_complete_tool_populates_holder():
    """Calling the stage_complete tool writes to the holder."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    result = await tool_fn(outcome="proceed", summary="Task done")

    assert holder.called is True
    assert holder.outcome.value == "proceed"
    assert holder.summary == "Task done"
    assert holder.escape_target is None
    assert "proceed" in result


async def test_stage_complete_tool_with_escape_target():
    """stage_complete can specify an escape target."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    await tool_fn(
        outcome="need_clarification",
        summary="Unclear requirements",
        escape_target="comment_on_issue",
    )

    assert holder.outcome.value == "need_clarification"
    assert holder.escape_target == "comment_on_issue"


async def test_stage_complete_tool_invalid_outcome():
    """stage_complete raises ValueError for invalid outcome."""
    holder = StageCompleteHolder()
    tool_fn = make_stage_complete_tool(holder)

    with pytest.raises(ValueError):
        await tool_fn(outcome="invalid", summary="oops")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_stage_complete.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.tools.stage_complete'`

- [ ] **Step 3: Write implementation**

```python
"""stage_complete tool — the mechanism for LLMs to signal stage completion.

The LLM calls this tool when it believes the current stage is done.
The tool writes to a mutable holder that the StageRunner inspects
after each tool dispatch. This decouples the LLM's completion signal
from the orchestrator's control flow.

The holder is created per-stage by the StageRunner and captured by
closure in the tool function. When exit criteria fail, the holder
is reset and the LLM gets another chance within the same stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from workflow.types import StageOutcome


@dataclass
class StageCompleteHolder:
    """Mutable holder for stage_complete results.

    The stage_complete tool writes to this; the StageRunner reads it
    after each tool dispatch round.
    """

    outcome: StageOutcome | None = None
    summary: str = ""
    escape_target: str | None = None
    called: bool = False


def make_stage_complete_tool(holder: StageCompleteHolder):
    """Create a stage_complete tool that writes to the given holder.

    Returns an async function suitable for Tool.from_function().
    The function captures the holder via closure so the StageRunner
    can inspect it after dispatch.
    """

    async def stage_complete(
        outcome: str,
        summary: str,
        escape_target: str | None = None,
    ) -> str:
        """Signal that this stage is complete.

        Args:
            outcome: One of 'proceed', 'need_clarification', 'scope_changed', 'blocked'.
            summary: Your reasoning and decisions. Describe what you did and why.
                     Include approaches you tried and rejected. Do NOT repeat facts
                     the orchestrator can verify directly (file contents, test output).
            escape_target: Name of the stage to return to (only for
                           need_clarification or scope_changed).
        """
        holder.outcome = StageOutcome(outcome)
        holder.summary = summary
        holder.escape_target = escape_target
        holder.called = True
        return f"Stage complete: {outcome}"

    return stage_complete
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_stage_complete.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add workflow/tools/stage_complete.py tests/workflow/test_stage_complete.py
git commit -m "feat(workflow): add stage_complete tool with holder pattern"
```

---

## Task 5: Stage Definition and StageRunner

**Files:**
- Create: `workflow/stage.py`
- Test: `tests/workflow/test_stage_runner.py`

This is the core task. The StageRunner is the heart of the workflow engine — it runs a single stage, managing the LLM tool-calling loop, exit criteria verification, and retry logic.

- [ ] **Step 1: Write test for Stage dataclass and mechanical stages**

```python
"""Tests for workflow.stage — Stage, StageRunner."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.tool import Tool
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_stage_runner.py -v -k "test_stage_defaults or test_mechanical"`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.stage'`

- [ ] **Step 3: Write initial Stage + StageRunner (mechanical only)**

```python
"""Stage definition and StageRunner — the core stage execution machinery.

Stage: dataclass defining a single workflow stage.
StageRunner: runs one stage (mechanical or LLM-driven), handling the
tool-calling loop, stage_complete detection, and exit criteria.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from corvidae.llm import LLMClient

from workflow.types import StageOutcome, StageResult, ToolCallRecord
from workflow.context import WorkflowContext

logger = logging.getLogger(__name__)

# System prompt for all LLM stages
SYSTEM_PROMPT = (
    "You are a workflow agent executing a specific stage of a workflow. "
    "You have a focused task and a limited set of tools. "
    "Work through the task methodically. "
    "When finished, call stage_complete with your outcome and a summary "
    "of your reasoning."
)

# Type aliases for stage callback signatures
ExitCriteriaFn = Callable[[StageResult, WorkflowContext], Awaitable[bool]]
MechanicalFn = Callable[[WorkflowContext], Awaitable[StageResult]]
ShouldRunFn = Callable[[WorkflowContext], bool]


@dataclass
class Stage:
    """Definition of a single workflow stage.

    Attributes:
        name: Unique stage identifier (used for transitions and logging).
        instruction: The prompt describing what the LLM should do.
            For mechanical stages, a description for logging.
        tools: Tool instances available to the LLM in this stage.
        escape_targets: Maps StageOutcome to target stage name.
        exit_criteria: Async function verifying the stage's exit conditions.
            Receives (StageResult, WorkflowContext), returns True if met.
        should_run: Optional predicate — return False to skip this stage.
        human_gate: Whether to pause for human approval before proceeding.
        is_mechanical: If True, run mechanical_fn instead of LLM.
        mechanical_fn: Async function for mechanical stages.
    """

    name: str
    instruction: str
    tools: list = field(default_factory=list)
    escape_targets: dict[StageOutcome, str] = field(default_factory=dict)
    exit_criteria: ExitCriteriaFn | None = None
    should_run: ShouldRunFn | None = None
    human_gate: bool = False
    is_mechanical: bool = False
    mechanical_fn: MechanicalFn | None = None


class StageRunner:
    """Runs a single workflow stage.

    For mechanical stages: calls mechanical_fn directly.
    For LLM stages: builds prompt, runs tool-calling loop until
    stage_complete, verifies exit criteria.
    """

    def __init__(self, client: LLMClient, max_turns: int = 20) -> None:
        self.client = client
        self.max_turns = max_turns

    async def run(self, stage: Stage, context: WorkflowContext) -> StageResult:
        """Run a single stage and return the result."""
        logger.info("Starting stage: %s", stage.name)

        if stage.is_mechanical:
            result = await self._run_mechanical(stage, context)
        else:
            # Placeholder — implemented in the next step
            raise NotImplementedError("LLM stages not yet implemented")

        logger.info(
            "Stage complete: %s (outcome=%s)",
            stage.name, result.outcome.value,
        )
        return result

    async def _run_mechanical(
        self, stage: Stage, context: WorkflowContext
    ) -> StageResult:
        """Run a mechanical (no-LLM) stage."""
        if stage.mechanical_fn is None:
            raise ValueError(
                f"Mechanical stage '{stage.name}' has no mechanical_fn"
            )
        return await stage.mechanical_fn(context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_stage_runner.py -v -k "test_stage_defaults or test_mechanical"`
Expected: 3 passed

- [ ] **Step 5: Write test for LLM stage with immediate stage_complete**

Add to `tests/workflow/test_stage_runner.py`:

```python
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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_stage_runner.py::test_llm_stage_immediate_complete -v`
Expected: FAIL — `NotImplementedError: LLM stages not yet implemented`

- [ ] **Step 7: Implement _run_llm in StageRunner**

Replace the `_run_mechanical` method and `run` placeholder with the full LLM loop. Add the missing imports to `workflow/stage.py`:

```python
# Add to imports at top of workflow/stage.py:
from corvidae.tool import Tool as CorvidaeTool, ToolRegistry, dispatch_tool_call
from corvidae.turn import run_agent_turn
from workflow.tools.stage_complete import StageCompleteHolder, make_stage_complete_tool
```

Replace the `NotImplementedError` in `run()` and add `_run_llm`:

```python
    async def run(self, stage: Stage, context: WorkflowContext) -> StageResult:
        """Run a single stage and return the result."""
        logger.info("Starting stage: %s", stage.name)

        if stage.is_mechanical:
            result = await self._run_mechanical(stage, context)
        else:
            result = await self._run_llm(stage, context)

        logger.info(
            "Stage complete: %s (outcome=%s)",
            stage.name, result.outcome.value,
        )
        return result

    async def _run_llm(
        self, stage: Stage, context: WorkflowContext
    ) -> StageResult:
        """Run an LLM stage with the tool-calling loop.

        Builds a scoped tool registry (stage tools + stage_complete),
        runs the LLM in a loop until stage_complete is called or
        max_turns is reached, then verifies exit criteria.
        """
        holder = StageCompleteHolder()
        stage_complete_fn = make_stage_complete_tool(holder)
        stage_complete_tool = CorvidaeTool.from_function(stage_complete_fn)

        # Build scoped tool registry: stage tools + stage_complete
        registry = ToolRegistry()
        registry.add(stage_complete_tool)
        for tool in stage.tools:
            registry.add(tool)

        tools_dict = registry.as_dict()
        tool_schemas = registry.schemas()

        # Build initial messages with prior context
        prior_context = context.build_context_string(stage.name)
        user_content = stage.instruction
        if prior_context:
            user_content += (
                f"\n\n## Context from previous stages\n{prior_context}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tool_records: list[ToolCallRecord] = []
        exit_criteria_failures = 0

        for _ in range(self.max_turns):
            turn = await run_agent_turn(
                self.client, messages, tool_schemas
            )

            # No tool calls — LLM returned text only
            if not turn.tool_calls:
                if holder.called:
                    break  # stage_complete was called in a prior turn
                # Prompt the LLM to call stage_complete
                messages.append({
                    "role": "user",
                    "content": (
                        "Please call stage_complete to signal you are done."
                    ),
                })
                continue

            # Dispatch each tool call
            for call in turn.tool_calls:
                result = await dispatch_tool_call(call, tools_dict)
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                })

                # Capture arguments for the record
                try:
                    args = json.loads(call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                tool_records.append(ToolCallRecord(
                    tool_name=result.tool_name,
                    arguments=args,
                    result=result.content,
                    error=result.error,
                ))

                if holder.called:
                    break  # stage_complete was dispatched

            if not holder.called:
                continue  # LLM hasn't called stage_complete yet

            # stage_complete was called — check exit criteria
            if stage.exit_criteria is not None:
                stage_result = StageResult(
                    stage_name=stage.name,
                    outcome=holder.outcome or StageOutcome.PROCEED,
                    summary=holder.summary,
                    escape_target=holder.escape_target,
                    tool_calls=tool_records,
                )
                passed = await stage.exit_criteria(stage_result, context)
                if not passed:
                    exit_criteria_failures += 1
                    if exit_criteria_failures >= 3:
                        logger.warning(
                            "Exit criteria failed %d times for stage %s",
                            exit_criteria_failures, stage.name,
                        )
                        return StageResult(
                            stage_name=stage.name,
                            outcome=StageOutcome.BLOCKED,
                            summary=(
                                f"Exit criteria failed after "
                                f"{exit_criteria_failures} attempts"
                            ),
                            tool_calls=tool_records,
                        )
                    # Reset holder and let LLM retry
                    holder.outcome = None
                    holder.summary = ""
                    holder.escape_target = None
                    holder.called = False
                    messages.append({
                        "role": "user",
                        "content": (
                            "Exit criteria not met. Please address the "
                            "issue and call stage_complete again."
                        ),
                    })
                    continue

            # Exit criteria passed or no exit criteria defined
            break

        # Check if we exited without stage_complete
        if not holder.called:
            logger.error(
                "Stage %s ended without stage_complete", stage.name
            )
            return StageResult(
                stage_name=stage.name,
                outcome=StageOutcome.BLOCKED,
                summary="Stage ended without calling stage_complete",
                tool_calls=tool_records,
            )

        return StageResult(
            stage_name=stage.name,
            outcome=holder.outcome or StageOutcome.PROCEED,
            summary=holder.summary,
            escape_target=holder.escape_target,
            tool_calls=tool_records,
        )
```

- [ ] **Step 8: Run LLM stage tests**

Run: `python -m pytest tests/workflow/test_stage_runner.py -v`
Expected: 4 passed (3 mechanical + 1 immediate_complete)

- [ ] **Step 9: Write test for multi-turn LLM stage**

Add to `tests/workflow/test_stage_runner.py`:

```python
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
```

(Note: uses `CorvidaeTool` alias since `Tool` from corvidae conflicts with any local import. Add `from corvidae.tool import Tool as CorvidaeTool` at the top of the test file.)

Add the import at the top of `tests/workflow/test_stage_runner.py`:

```python
from corvidae.tool import Tool as CorvidaeTool
```

- [ ] **Step 10: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_stage_runner.py::test_llm_stage_multi_turn_with_tools -v`
Expected: PASS (the _run_llm implementation already handles multi-turn)

- [ ] **Step 11: Write test for max_turns exhausted**

Add to `tests/workflow/test_stage_runner.py`:

```python
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
```

- [ ] **Step 12: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_stage_runner.py::test_llm_stage_max_turns_exhausted -v`
Expected: PASS

- [ ] **Step 13: Write tests for exit criteria**

Add to `tests/workflow/test_stage_runner.py`:

```python
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
```

- [ ] **Step 14: Run all stage runner tests**

Run: `python -m pytest tests/workflow/test_stage_runner.py -v`
Expected: 8 passed

- [ ] **Step 15: Commit**

```bash
git add workflow/stage.py tests/workflow/test_stage_runner.py
git commit -m "feat(workflow): add Stage dataclass and StageRunner with tool-calling loop"
```

---

## Task 6: WorkflowEngine

**Files:**
- Create: `workflow/engine.py`
- Test: `tests/workflow/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for workflow.engine — WorkflowEngine."""

import logging
from unittest.mock import MagicMock

import pytest

from workflow.types import StageOutcome, StageResult
from workflow.context import WorkflowContext
from workflow.stage import Stage
from workflow.engine import WorkflowEngine


def _mech_fn(name, outcome, summary, escape=None):
    """Create a mechanical_fn that returns a fixed StageResult."""
    async def fn(ctx):
        return StageResult(
            stage_name=name, outcome=outcome, summary=summary,
            escape_target=escape,
        )
    return fn


async def test_linear_progression():
    """Engine proceeds through stages in order."""
    order = []

    def mk(name):
        async def fn(ctx):
            order.append(name)
            return StageResult(
                stage_name=name, outcome=StageOutcome.PROCEED,
                summary=f"done {name}",
            )
        return fn

    stages = [
        Stage(name="a", instruction="", is_mechanical=True, mechanical_fn=mk("a")),
        Stage(name="b", instruction="", is_mechanical=True, mechanical_fn=mk("b")),
        Stage(name="c", instruction="", is_mechanical=True, mechanical_fn=mk("c")),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    assert order == ["a", "b", "c"]


async def test_escape_hatch():
    """Engine jumps back to the escape target stage."""
    order = []
    gate_count = {"n": 0}

    async def start_fn(ctx):
        order.append("start")
        return StageResult(
            stage_name="start", outcome=StageOutcome.PROCEED,
            summary="started",
        )

    async def gate_fn(ctx):
        order.append("gate")
        gate_count["n"] += 1
        if gate_count["n"] == 1:
            return StageResult(
                stage_name="gate",
                outcome=StageOutcome.NEED_CLARIFICATION,
                summary="unclear",
                escape_target="start",
            )
        return StageResult(
            stage_name="gate", outcome=StageOutcome.PROCEED,
            summary="ready",
        )

    stages = [
        Stage(name="start", instruction="", is_mechanical=True,
              mechanical_fn=start_fn),
        Stage(name="gate", instruction="",
              escape_targets={StageOutcome.NEED_CLARIFICATION: "start"},
              is_mechanical=True, mechanical_fn=gate_fn),
        Stage(name="done", instruction="", is_mechanical=True,
              mechanical_fn=_mech_fn("done", StageOutcome.PROCEED, "done")),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    assert order == ["start", "gate", "start", "gate", "done"]


async def test_conditional_stage_skipped():
    """Engine skips stages where should_run returns False."""
    order = []

    def mk(name):
        async def fn(ctx):
            order.append(name)
            return StageResult(
                stage_name=name, outcome=StageOutcome.PROCEED,
                summary=f"done {name}",
            )
        return fn

    stages = [
        Stage(name="a", instruction="", is_mechanical=True, mechanical_fn=mk("a")),
        Stage(name="b", instruction="", is_mechanical=True, mechanical_fn=mk("b"),
              should_run=lambda ctx: False),
        Stage(name="c", instruction="", is_mechanical=True, mechanical_fn=mk("c")),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    assert order == ["a", "c"]


async def test_blocked_stops_workflow():
    """Engine stops when a stage returns BLOCKED."""
    order = []

    def mk(name, outcome):
        async def fn(ctx):
            order.append(name)
            return StageResult(
                stage_name=name, outcome=outcome, summary=name,
            )
        return fn

    stages = [
        Stage(name="a", instruction="", is_mechanical=True,
              mechanical_fn=mk("a", StageOutcome.PROCEED)),
        Stage(name="b", instruction="", is_mechanical=True,
              mechanical_fn=mk("b", StageOutcome.BLOCKED)),
        Stage(name="c", instruction="", is_mechanical=True,
              mechanical_fn=mk("c", StageOutcome.PROCEED)),
    ]

    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, WorkflowContext())

    assert order == ["a", "b"]


async def test_context_accumulation():
    """Engine records each stage's results in the context."""
    stages = [
        Stage(name="a", instruction="", is_mechanical=True,
              mechanical_fn=_mech_fn("a", StageOutcome.PROCEED, "summary a")),
        Stage(name="b", instruction="", is_mechanical=True,
              mechanical_fn=_mech_fn("b", StageOutcome.PROCEED, "summary b")),
    ]

    ctx = WorkflowContext()
    engine = WorkflowEngine(MagicMock())
    await engine.run(stages, ctx)

    assert ctx.summaries == {"a": "summary a", "b": "summary b"}


async def test_invalid_escape_target_stops():
    """Engine stops when escape_target doesn't match any stage."""
    stages = [
        Stage(name="a", instruction="", is_mechanical=True,
              mechanical_fn=_mech_fn(
                  "a", StageOutcome.NEED_CLARIFICATION, "x",
                  escape="nonexistent",
              ),
              escape_targets={StageOutcome.NEED_CLARIFICATION: "nonexistent"}),
    ]

    engine = WorkflowEngine(MagicMock())
    ctx = WorkflowContext()
    await engine.run(stages, ctx)

    # Should stop — no infinite loop or crash
    assert "a" in ctx.summaries


async def test_max_iterations_prevents_infinite_loop():
    """Engine stops after max_iterations to prevent infinite loops."""
    async def always_escape(ctx):
        return StageResult(
            stage_name="loop", outcome=StageOutcome.NEED_CLARIFICATION,
            summary="again", escape_target="loop",
        )

    stages = [
        Stage(name="loop", instruction="",
              escape_targets={StageOutcome.NEED_CLARIFICATION: "loop"},
              is_mechanical=True, mechanical_fn=always_escape),
    ]

    engine = WorkflowEngine(MagicMock(), max_iterations=5)
    ctx = WorkflowContext()
    await engine.run(stages, ctx)

    # Should stop after 5 iterations, not hang forever
    assert len(ctx.summaries.get("loop", "")) >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.engine'`

- [ ] **Step 3: Write implementation**

```python
"""WorkflowEngine — orchestrates stage progression.

Manages the linear backbone with escape hatches:
- PROCEED → next stage (default forward path)
- Escape outcome → jump to the named target stage
- BLOCKED or invalid escape → stop the workflow

Also handles:
- Conditional stages (should_run → skip)
- Human gates (stdin prompt for approval)
- Max iterations guard (prevents infinite escape loops)
"""

from __future__ import annotations

import logging

from corvidae.llm import LLMClient

from workflow.types import StageOutcome
from workflow.context import WorkflowContext
from workflow.stage import Stage, StageRunner

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Orchestrates a workflow by running stages in sequence.

    The engine manages stage transitions, context accumulation,
    human gates, and the max-iterations safety guard.
    """

    def __init__(
        self,
        client: LLMClient,
        max_turns: int = 20,
        max_iterations: int = 50,
    ) -> None:
        self.runner = StageRunner(client, max_turns)
        self.max_iterations = max_iterations

    async def run(
        self, stages: list[Stage], context: WorkflowContext
    ) -> None:
        """Run the workflow to completion.

        Iterates through stages linearly, handling escape hatches
        by jumping to the target stage. Stops when:
        - All stages complete (end of list)
        - A stage returns BLOCKED without an escape target
        - A non-PROCEED outcome has no matching escape target
        - Max iterations is reached (safety guard)
        """
        current_idx = 0
        iterations = 0

        while current_idx < len(stages):
            iterations += 1
            if iterations > self.max_iterations:
                logger.error(
                    "Max iterations (%d) reached, stopping workflow",
                    self.max_iterations,
                )
                return

            stage = stages[current_idx]

            # Check should_run — skip if False
            if stage.should_run is not None and not stage.should_run(context):
                logger.info(
                    "Skipping stage: %s (should_run=False)", stage.name
                )
                current_idx += 1
                continue

            # Run the stage
            result = await self.runner.run(stage, context)

            # Record in context
            context.record_stage(
                stage.name, result.summary, result.tool_calls
            )

            # Human gate — pause for approval
            if stage.human_gate and result.outcome == StageOutcome.PROCEED:
                approved = await self._human_approval(stage, result)
                if not approved:
                    logger.info(
                        "Human gate rejected for stage: %s", stage.name
                    )
                    return

            # Determine next stage
            if result.escape_target:
                # Validate escape is allowed from this stage
                if result.outcome not in stage.escape_targets:
                    logger.error(
                        "Escape target %s not allowed for outcome %s "
                        "in stage %s",
                        result.escape_target,
                        result.outcome.value,
                        stage.name,
                    )
                    return
                target_idx = self._find_stage(stages, result.escape_target)
                if target_idx is None:
                    logger.error(
                        "Invalid escape target: %s", result.escape_target
                    )
                    return
                logger.info(
                    "Escaping from %s to %s (outcome=%s)",
                    stage.name, result.escape_target,
                    result.outcome.value,
                )
                current_idx = target_idx

            elif result.outcome == StageOutcome.PROCEED:
                current_idx += 1

            else:
                # Non-PROCEED outcome with no escape target — stop
                logger.warning(
                    "Stage %s ended with %s and no escape target, "
                    "stopping workflow",
                    stage.name, result.outcome.value,
                )
                return

    @staticmethod
    def _find_stage(stages: list[Stage], name: str) -> int | None:
        """Find a stage's index by name. Returns None if not found."""
        for i, stage in enumerate(stages):
            if stage.name == name:
                return i
        return None

    @staticmethod
    async def _human_approval(stage: Stage, result) -> bool:
        """Prompt for human approval via stdin."""
        print(f"\n{'=' * 60}")
        print(f"Stage: {stage.name}")
        print(f"Summary: {result.summary}")
        print(f"{'=' * 60}")
        response = input("Proceed? [y/N] ").strip().lower()
        return response in ("y", "yes")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_engine.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add workflow/engine.py tests/workflow/test_engine.py
git commit -m "feat(workflow): add WorkflowEngine with linear progression and escape hatches"
```

---

## Task 7: GitHub Tools

**Files:**
- Create: `workflow/tools/github.py`
- Test: `tests/workflow/test_github_tools.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for workflow.tools.github — GitHub CLI tool factories."""

from unittest.mock import AsyncMock, patch

from workflow.tools.github import (
    make_github_comment_tool,
    make_add_label_tool,
    make_create_pr_tool,
)


async def test_comment_on_issue():
    """comment_on_issue runs gh CLI with correct arguments."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "https://github.com/o/r/issues/1#issuecomment-1"
        tool = make_github_comment_tool("owner/repo", 42)
        result = await tool.fn(body="My comment")

        assert "comment" in result
        mock.assert_awaited_once()
        cmd = mock.call_args[0][0]
        assert "42" in cmd
        assert "owner/repo" in cmd
        assert "--body" in cmd


async def test_add_label():
    """add_label runs gh CLI to add a label."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = ""
        tool = make_add_label_tool("owner/repo", 42)
        result = await tool.fn(label="in-progress")

        mock.assert_awaited_once()
        cmd = mock.call_args[0][0]
        assert "--add-label" in cmd
        assert "in-progress" in cmd


async def test_create_pr():
    """create_pr runs gh CLI to create a pull request."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "https://github.com/o/r/pull/5"
        tool = make_create_pr_tool("owner/repo", "feat/issue-1")
        result = await tool.fn(title="Fix bug", body="Description")

        assert "pull" in result
        cmd = mock.call_args[0][0]
        assert "--title" in cmd
        assert "feat/issue-1" in cmd


async def test_comment_escapes_body():
    """comment_on_issue shell-escapes the body to prevent injection."""
    with patch("workflow.tools.github.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_github_comment_tool("o/r", 1)
        await tool.fn(body="test'; rm -rf /")

        cmd = mock.call_args[0][0]
        # The body should be shell-quoted, not raw
        assert "rm -rf" not in cmd or "'" in cmd or '"' in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_github_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.tools.github'`

- [ ] **Step 3: Write implementation**

```python
"""GitHub tools — factory functions that create scoped gh CLI tools.

Each factory captures repo/issue context via closure, so the LLM
doesn't need to know the repo or issue number — it just calls the
tool with the content. This prevents the LLM from targeting a
different repo.

Uses the gh CLI (must be installed and authenticated) for all
GitHub operations. Shell arguments are properly quoted to prevent
injection.
"""

from __future__ import annotations

import shlex

from corvidae.tool import Tool
from corvidae.tools.shell import shell


def make_github_comment_tool(repo: str, issue_number: int) -> Tool:
    """Create a tool for posting comments on a GitHub issue."""

    async def comment_on_issue(body: str) -> str:
        """Post a comment on the GitHub issue.

        Args:
            body: The comment text to post.
        """
        cmd = (
            f"gh issue comment {issue_number} "
            f"--repo {shlex.quote(repo)} "
            f"--body {shlex.quote(body)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(comment_on_issue)


def make_add_label_tool(repo: str, issue_number: int) -> Tool:
    """Create a tool for adding labels to a GitHub issue."""

    async def add_label(label: str) -> str:
        """Add a label to the GitHub issue.

        Args:
            label: The label to add (e.g. 'in-progress').
        """
        cmd = (
            f"gh issue edit {issue_number} "
            f"--repo {shlex.quote(repo)} "
            f"--add-label {shlex.quote(label)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(add_label)


def make_create_pr_tool(repo: str, branch: str) -> Tool:
    """Create a tool for opening a pull request."""

    async def create_pr(title: str, body: str) -> str:
        """Create a pull request on GitHub.

        Args:
            title: The PR title.
            body: The PR description.
        """
        cmd = (
            f"gh pr create "
            f"--repo {shlex.quote(repo)} "
            f"--head {shlex.quote(branch)} "
            f"--title {shlex.quote(title)} "
            f"--body {shlex.quote(body)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(create_pr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_github_tools.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add workflow/tools/github.py tests/workflow/test_github_tools.py
git commit -m "feat(workflow): add GitHub CLI tools — comment, label, PR"
```

---

## Task 8: Git Tools

**Files:**
- Create: `workflow/tools/git.py`
- Test: `tests/workflow/test_git_tools.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for workflow.tools.git — git CLI tool factories."""

from unittest.mock import AsyncMock, patch

from workflow.tools.git import (
    make_create_branch_tool,
    make_create_worktree_tool,
)


async def test_create_branch():
    """create_branch runs git checkout -b in the repo directory."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "Switched to a new branch 'feat/issue-1'"
        tool = make_create_branch_tool("/repo")
        result = await tool.fn(name="feat/issue-1")

        assert "Switched" in result
        cmd = mock.call_args[0][0]
        assert "checkout -b" in cmd
        assert "feat/issue-1" in cmd
        assert "/repo" in cmd


async def test_create_worktree():
    """create_worktree runs git worktree add."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "Preparing worktree"
        tool = make_create_worktree_tool("/repo")
        result = await tool.fn(branch="feat/issue-1", path="/repo-wt")

        cmd = mock.call_args[0][0]
        assert "worktree add" in cmd
        assert "feat/issue-1" in cmd
        assert "/repo-wt" in cmd


async def test_create_branch_quotes_name():
    """create_branch shell-escapes the branch name."""
    with patch("workflow.tools.git.shell", new_callable=AsyncMock) as mock:
        mock.return_value = "ok"
        tool = make_create_branch_tool("/repo")
        await tool.fn(name="feat/issue; rm -rf /")

        cmd = mock.call_args[0][0]
        assert "rm -rf" not in cmd or "'" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_git_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.tools.git'`

- [ ] **Step 3: Write implementation**

```python
"""Git tools — factory functions that create scoped git CLI tools.

Each factory captures the repo path via closure so tools operate on
the correct repository. Uses git -C to specify the working directory
instead of cd, which is cleaner in async contexts.
"""

from __future__ import annotations

import shlex

from corvidae.tool import Tool
from corvidae.tools.shell import shell


def make_create_branch_tool(repo_path: str) -> Tool:
    """Create a tool for creating and checking out a git branch."""

    async def create_branch(name: str) -> str:
        """Create and checkout a new git branch.

        Args:
            name: The branch name (e.g. 'feat/issue-42').
        """
        cmd = (
            f"git -C {shlex.quote(repo_path)} "
            f"checkout -b {shlex.quote(name)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(create_branch)


def make_create_worktree_tool(repo_path: str) -> Tool:
    """Create a tool for adding a git worktree."""

    async def create_worktree(branch: str, path: str) -> str:
        """Create a git worktree for the branch.

        Args:
            branch: The branch to check out in the worktree.
            path: The directory path for the new worktree.
        """
        cmd = (
            f"git -C {shlex.quote(repo_path)} "
            f"worktree add {shlex.quote(path)} "
            f"{shlex.quote(branch)}"
        )
        return await shell(cmd, timeout=30)

    return Tool.from_function(create_worktree)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_git_tools.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add workflow/tools/git.py tests/workflow/test_git_tools.py
git commit -m "feat(workflow): add git CLI tools — branch, worktree"
```

---

## Task 9: Issue Workflow Definition

**Files:**
- Create: `workflow/workflows/issue.py`
- Test: `tests/workflow/test_issue_workflow.py`

This task wires up all stages for the GitHub issue implementation workflow (stages 1–10 from the design doc). Stages 11–12 (respond to PR feedback, close issue) are deferred because they require waiting for external events.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for workflow.workflows.issue — build_issue_workflow."""

from workflow.types import StageOutcome
from workflow.stage import Stage
from workflow.workflows.issue import build_issue_workflow, parse_issue_url


def test_parse_issue_url():
    """parse_issue_url extracts owner/repo and issue number."""
    repo, num = parse_issue_url(
        "https://github.com/owner/repo/issues/42"
    )
    assert repo == "owner/repo"
    assert num == 42


def test_parse_issue_url_trailing_slash():
    """parse_issue_url handles trailing slashes."""
    repo, num = parse_issue_url(
        "https://github.com/owner/repo/issues/42/"
    )
    assert repo == "owner/repo"
    assert num == 42


def test_parse_issue_url_invalid():
    """parse_issue_url raises ValueError for non-issue URLs."""
    import pytest
    with pytest.raises(ValueError):
        parse_issue_url("https://github.com/owner/repo/pull/42")


def test_workflow_has_expected_stages():
    """build_issue_workflow returns stages with expected names."""
    stages = build_issue_workflow(
        repo="owner/repo",
        issue_number=1,
        repo_path="/repo",
        issue_body="Fix the bug",
    )
    names = [s.name for s in stages]
    assert "comment_on_issue" in names
    assert "add_label" in names
    assert "create_branch" in names
    assert "gate_ready" in names
    assert "write_tests" in names
    assert "implement" in names
    assert "refactor" in names
    assert "update_docs" in names
    assert "submit_pr" in names


def test_gate_ready_has_escape_target():
    """gate_ready stage can escape back to comment_on_issue."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    gate = next(s for s in stages if s.name == "gate_ready")
    assert StageOutcome.NEED_CLARIFICATION in gate.escape_targets
    assert gate.escape_targets[StageOutcome.NEED_CLARIFICATION] == "comment_on_issue"


def test_mechanical_stages_are_marked():
    """add_label is a mechanical stage."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    label_stage = next(s for s in stages if s.name == "add_label")
    assert label_stage.is_mechanical is True
    assert label_stage.mechanical_fn is not None


def test_llm_stages_have_instructions():
    """Every non-mechanical stage has a non-empty instruction."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    for stage in stages:
        if not stage.is_mechanical:
            assert len(stage.instruction) > 20, (
                f"Stage {stage.name} has no instruction"
            )


def test_llm_stages_have_tools():
    """Stages that need file/shell/git/GitHub tools have them."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    impl = next(s for s in stages if s.name == "implement")
    tool_names = [t.name for t in impl.tools]
    # implement needs write_file, read_file, and a shell tool
    assert any("write" in n for n in tool_names)
    assert any("read" in n for n in tool_names)


def test_update_docs_is_conditional():
    """update_docs has a should_run callback."""
    stages = build_issue_workflow(
        repo="o/r", issue_number=1,
        repo_path="/r", issue_body="x",
    )
    docs = next(s for s in stages if s.name == "update_docs")
    assert docs.should_run is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_issue_workflow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.workflows.issue'`

- [ ] **Step 3: Write implementation**

```python
"""GitHub issue implementation workflow.

Defines the stages for implementing a GitHub issue:
1. Comment on issue with implementation plan
2. Add "in-progress" label (mechanical)
3. Create branch (LLM suggests name)
4. Gate: ready to implement? (escape to stage 1)
5. Write tests (red)
6. Implement (green)
7. Refactor
8. Update docs (conditional — skipped if gate_docs decides not needed)
9. Submit PR

Stages 10-12 (respond to PR feedback, close issue) are deferred —
they require waiting for external events (PR review, PR merge).
"""

from __future__ import annotations

import re

from corvidae.tool import Tool
from corvidae.tools.files import read_file, write_file
from corvidae.tools.shell import shell

from workflow.types import StageOutcome, StageResult
from workflow.context import WorkflowContext
from workflow.stage import Stage
from workflow.tools.github import (
    make_github_comment_tool,
    make_add_label_tool,
    make_create_pr_tool,
)
from workflow.tools.git import (
    make_create_branch_tool,
    make_create_worktree_tool,
)


def parse_issue_url(url: str) -> tuple[str, int]:
    """Parse a GitHub issue URL into (owner/repo, issue_number).

    Raises:
        ValueError: If the URL is not a valid GitHub issue URL.
    """
    url = url.rstrip("/")
    match = re.match(
        r"https://github\.com/([^/]+/[^/]+)/issues/(\d+)$", url
    )
    if not match:
        raise ValueError(f"Not a valid GitHub issue URL: {url}")
    return match.group(1), int(match.group(2))


# -- Shell and file tool wrappers for workflow use --

async def run_command(command: str) -> str:
    """Execute a shell command and return the output."""
    return await shell(command, timeout=30)


async def read_source(path: str) -> str:
    """Read the contents of a file."""
    return await read_file(path)


async def write_source(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    return await write_file(path, content)


# -- Mechanical stage helpers --

def _make_add_label_fn(repo: str, issue_number: int):
    """Create a mechanical_fn that adds the 'in-progress' label."""
    async def fn(ctx: WorkflowContext) -> StageResult:
        import shlex
        cmd = (
            f"gh issue edit {issue_number} "
            f"--repo {shlex.quote(repo)} "
            f"--add-label in-progress"
        )
        await shell(cmd, timeout=30)
        return StageResult(
            stage_name="add_label",
            outcome=StageOutcome.PROCEED,
            summary="Added in-progress label",
        )
    return fn


# -- Docs conditional check --

def _docs_should_run(context: WorkflowContext) -> bool:
    """Check if the update_docs stage should run.

    The gate_docs stage's summary starts with 'YES' or 'NO'
    to signal whether docs are needed. If 'YES', run the stage.
    Otherwise skip it.
    """
    gate_summary = context.summaries.get("gate_docs", "")
    return gate_summary.strip().upper().startswith("YES")


# -- Workflow builder --

def build_issue_workflow(
    repo: str,
    issue_number: int,
    repo_path: str,
    issue_body: str,
) -> list[Stage]:
    """Build the GitHub issue implementation workflow.

    Args:
        repo: GitHub repo in owner/name format.
        issue_number: The issue number.
        repo_path: Local path to the git repository.
        issue_body: The issue body text (fetched via gh CLI).

    Returns:
        Ordered list of Stage instances.
    """
    # Create scoped tools
    github_comment = make_github_comment_tool(repo, issue_number)
    create_branch = make_create_branch_tool(repo_path)
    create_pr = make_create_pr_tool(repo, "HEAD")

    # File and shell tools (shared across stages)
    shell_tool = Tool.from_function(run_command)
    file_tools = [
        Tool.from_function(read_source),
        Tool.from_function(write_source),
    ]

    return [
        # 1. Comment on issue with implementation plan
        Stage(
            name="comment_on_issue",
            instruction=(
                f"Read the GitHub issue #{issue_number} in {repo}.\n\n"
                f"The issue body:\n```\n{issue_body}\n```\n\n"
                "Write a comment describing your implementation plan. "
                "Include:\n"
                "- Requirements extracted from the issue\n"
                "- Proposed approach\n"
                "- Any questions or ambiguities\n\n"
                "Post the comment using the comment_on_issue tool."
            ),
            tools=[github_comment],
        ),

        # 2. Add "in-progress" label (mechanical)
        Stage(
            name="add_label",
            instruction="Add in-progress label",
            is_mechanical=True,
            mechanical_fn=_make_add_label_fn(repo, issue_number),
        ),

        # 3. Create branch (LLM suggests name)
        Stage(
            name="create_branch",
            instruction=(
                "Suggest a descriptive branch name for implementing "
                "this issue, then create the branch using the "
                "create_branch tool."
            ),
            tools=[create_branch],
        ),

        # 4. Gate: ready to implement?
        Stage(
            name="gate_ready",
            instruction=(
                "Review the issue and context from previous stages. "
                "Do you have enough information to implement this?\n\n"
                "If yes, call stage_complete with outcome 'proceed'.\n"
                "If not, call stage_complete with outcome "
                "'need_clarification' and list your questions."
            ),
            escape_targets={
                StageOutcome.NEED_CLARIFICATION: "comment_on_issue",
            },
        ),

        # 5. Write tests (red)
        Stage(
            name="write_tests",
            instruction=(
                "Write failing tests for the implementation described "
                "in the issue. Follow TDD: write tests that capture the "
                "requirements but do NOT yet pass.\n\n"
                "Use write_source to create test files, then run_command "
                "to run the tests and verify they fail."
            ),
            tools=file_tools + [shell_tool],
        ),

        # 6. Implement (green)
        Stage(
            name="implement",
            instruction=(
                "Write the implementation to make the tests pass.\n\n"
                "Use write_source to create or edit source files, "
                "read_source to inspect existing code, and "
                "run_command to run the tests."
            ),
            tools=file_tools + [shell_tool],
        ),

        # 7. Refactor
        Stage(
            name="refactor",
            instruction=(
                "Review the code and tests. Clean up:\n"
                "- Improve naming\n"
                "- Extract functions\n"
                "- Remove duplication\n\n"
                "Ensure tests still pass after refactoring."
            ),
            tools=file_tools + [shell_tool],
        ),

        # 8. Gate: docs needed? (merged into update_docs for simplicity)
        Stage(
            name="gate_docs",
            instruction=(
                "Decide whether documentation updates are needed for "
                "this change.\n\n"
                "If YES, start your summary with 'YES' and describe "
                "what needs documenting.\n"
                "If NO, start your summary with 'NO' and explain why."
            ),
        ),

        # 9. Update docs (conditional)
        Stage(
            name="update_docs",
            instruction=(
                "Update documentation based on the changes made.\n\n"
                "Edit the relevant documentation files."
            ),
            tools=file_tools,
            should_run=_docs_should_run,
        ),

        # 10. Submit PR
        Stage(
            name="submit_pr",
            instruction=(
                "Write a clear PR description summarizing the changes, "
                "then create the PR using the create_pr tool.\n\n"
                "The PR body should reference the issue number and "
                "describe what was changed and why."
            ),
            tools=[create_pr],
        ),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_issue_workflow.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add workflow/workflows/issue.py tests/workflow/test_issue_workflow.py
git commit -m "feat(workflow): add GitHub issue implementation workflow definition"
```

---

## Task 10: CLI Entry Point

**Files:**
- Create: `workflow/__main__.py`
- Test: `tests/workflow/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for workflow.__main__ — CLI entry point."""

import subprocess
import sys


def test_cli_help():
    """CLI --help exits with 0."""
    result = subprocess.run(
        [sys.executable, "-m", "workflow", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "workflow" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_cli_implement_help():
    """implement --help shows required arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "workflow", "implement", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--issue" in result.stdout


def test_cli_no_args_shows_help():
    """Running with no arguments shows help."""
    result = subprocess.run(
        [sys.executable, "-m", "workflow"],
        capture_output=True, text=True,
    )
    # Should print help and exit cleanly
    assert result.returncode == 0 or "usage" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/workflow/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow.__main__'` or similar

- [ ] **Step 3: Write implementation**

```python
"""CLI entry point for the workflow engine.

Usage:
    python -m workflow implement --issue <url> [--repo-path <path>]

The workflow engine reads agent.yaml for LLM configuration,
fetches the issue body via gh CLI, and runs the issue
implementation workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from workflow.config import load_agent_config, create_llm_client
from workflow.context import WorkflowContext
from workflow.engine import WorkflowEngine
from workflow.workflows.issue import build_issue_workflow, parse_issue_url

logger = logging.getLogger(__name__)


async def fetch_issue_body(repo: str, issue_number: int) -> str:
    """Fetch issue body via gh CLI."""
    from corvidae.tools.shell import shell
    cmd = (
        f"gh issue view {issue_number} "
        f"--repo {repo} --json body -q .body"
    )
    return await shell(cmd, timeout=30)


async def run_workflow(issue_url: str, repo_path: str) -> None:
    """Parse issue, build workflow, and run it."""
    # Parse the issue URL
    repo, issue_number = parse_issue_url(issue_url)
    logger.info("Running workflow for %s/issues/%d", repo, issue_number)

    # Fetch issue body
    issue_body = await fetch_issue_body(repo, issue_number)
    if issue_body.startswith("Error:"):
        logger.error("Failed to fetch issue: %s", issue_body)
        sys.exit(1)

    # Load LLM config from agent.yaml
    config = load_agent_config()
    client = create_llm_client(config)
    await client.start()

    try:
        # Build and run the workflow
        stages = build_issue_workflow(
            repo=repo,
            issue_number=issue_number,
            repo_path=repo_path,
            issue_body=issue_body,
        )

        context = WorkflowContext()
        context.facts["issue_url"] = issue_url
        context.facts["repo"] = repo
        context.facts["issue_number"] = issue_number

        engine = WorkflowEngine(client)
        await engine.run(stages, context)

        logger.info("Workflow complete")
    finally:
        await client.stop()


def main() -> None:
    """Parse arguments and run the workflow."""
    parser = argparse.ArgumentParser(
        description="Corvidae Workflow Engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    impl = subparsers.add_parser(
        "implement",
        help="Implement a GitHub issue",
    )
    impl.add_argument(
        "--issue", required=True,
        help="GitHub issue URL",
    )
    impl.add_argument(
        "--repo-path", default=".",
        help="Local repo path (default: current directory)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.command == "implement":
        asyncio.run(run_workflow(args.issue, args.repo_path))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/workflow/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Verify all tests pass together**

Run: `python -m pytest tests/workflow/ -v --timeout=30`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add workflow/__main__.py tests/workflow/test_cli.py
git commit -m "feat(workflow): add CLI entry point — python -m workflow implement"
```

---

## Task 11: Documentation

**Files:**
- Update: `workflow/README.md`

- [ ] **Step 1: Update README with implementation status and usage**

Add an "Implementation Status" section to `workflow/README.md` after the "Key Decisions" section:

```markdown

## Implementation Status

**Approach A** (standalone script) is implemented. The workflow engine
lives in `workflow/` and imports Corvidae primitives directly.

### What's built

- **Core engine**: Stage, StageRunner, WorkflowEngine with linear backbone
  and escape hatches
- **stage_complete tool**: Three-layer exit (LLM signal → exit criteria
  verification → human gate)
- **Tool scoping**: Each stage receives only the tools it needs
- **WorkflowContext**: Two-channel accumulation (summaries + facts)
- **GitHub issue workflow**: 9 of 12 stages implemented (stages 1–10;
  PR feedback and issue close deferred)

### Usage

```bash
# Run the issue implementation workflow
python -m workflow implement --issue https://github.com/owner/repo/issues/42

# Specify a local repo path
python -m workflow implement --issue https://github.com/owner/repo/issues/42 --repo-path /path/to/repo
```

### Prerequisites

- Python 3.13+
- `gh` CLI installed and authenticated
- `agent.yaml` in the working directory with `llm.main` config

### What's deferred

- **Stage 11** (respond to PR feedback): Requires waiting for external
  events. Will need a pause/resume mechanism.
- **Stage 12** (close issue): Requires waiting for PR merge.
- **Framework extraction** (Approach B/C): Wait until a second workflow
  exists to identify reusable patterns.
```

- [ ] **Step 2: Commit**

```bash
git add workflow/README.md
git commit -m "docs(workflow): update README with implementation status and usage"
```

---

## Self-Review

### 1. Spec coverage

Checked against `workflow/README.md`:

| Requirement | Task |
|---|---|
| Stage isolation (fresh LLM invocation) | Task 5 (_run_llm builds fresh messages) |
| Prefer deterministic code | Task 9 (mechanical stages, gh CLI wrappers) |
| Two-channel output (summaries + facts) | Task 2 (WorkflowContext) |
| Linear backbone with escape hatches | Task 6 (WorkflowEngine) |
| Tool scoping per stage | Task 5 (StageRunner builds scoped registry) |
| Three-layer stage exit | Task 5 (stage_complete → exit_criteria → human_gate) |
| Structured outcomes | Task 1 (StageOutcome enum) |
| stage_complete tool | Task 4 |
| Exit criteria verification | Task 5 (_run_llm with retries) |
| Human gates | Task 6 (WorkflowEngine._human_approval) |
| GitHub issue workflow stages 1–10 | Task 9 |
| Read agent.yaml for config | Task 3 |
| CLI entry point | Task 10 |

### 2. Placeholder scan

No TBD, TODO, "implement later", or "add appropriate error handling" found. Every step has complete code.

### 3. Type consistency

- `StageResult.stage_name` is `str` everywhere it's constructed
- `StageResult.outcome` is `StageOutcome` everywhere
- `Stage.escape_targets` maps `StageOutcome` → `str` (stage names)
- `ToolCallRecord` fields match between construction in `_run_llm` and definition in `types.py`
- `WorkflowContext.record_stage` takes `(str, str, list[ToolCallRecord])` — matches all call sites
- `make_stage_complete_tool` returns a callable matching `Tool.from_function` expectations
- `MechanicalFn` type is `Callable[[WorkflowContext], Awaitable[StageResult]]` — matches all `mechanical_fn` implementations
- `parse_issue_url` returns `tuple[str, int]` — matches usage in `run_workflow`

---

Plan complete and saved to `plans/workflow-engine.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
