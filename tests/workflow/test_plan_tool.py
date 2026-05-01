"""Tests for workflow.tools.plan — PlanHolder and record_plan tool."""

import pytest

from corvidae.tool import Tool
from workflow.tools.plan import (
    PlanHolder,
    make_record_plan_tool,
    VALID_TASK_TYPES,
)


# -- PlanHolder tests --


def test_plan_holder_starts_empty():
    """PlanHolder.data is None by default."""
    holder = PlanHolder()
    assert holder.data is None


def test_plan_holder_stores_dict():
    """PlanHolder.data can be set to a dict."""
    holder = PlanHolder()
    holder.data = {"branch_name": "feat/test"}
    assert holder.data == {"branch_name": "feat/test"}


# -- record_plan tool factory tests --


def test_make_record_plan_tool_returns_tool():
    """make_record_plan_tool returns a Tool instance."""
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)
    assert isinstance(tool, Tool)
    assert tool.name == "record_plan"


def test_record_plan_tool_has_schema():
    """The record_plan tool has a valid schema with parameters."""
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)
    assert "function" in tool.schema
    params = tool.schema["function"]["parameters"]["properties"]
    assert "branch_name" in params
    assert "task_type" in params
    assert "needs_tests" in params
    assert "needs_docs" in params
    assert "files_to_change" in params
    assert "pr_title" in params
    assert "summary" in params


@pytest.mark.timeout(5)
async def test_record_plan_stores_in_holder():
    """record_plan writes structured data to the PlanHolder."""
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)
    result = await tool.fn(
        branch_name="feat/issue-42-add-logging",
        task_type="feature",
        needs_tests=True,
        needs_docs=False,
        files_to_change=["src/main.py", "src/utils.py"],
        pr_title="Add logging to utils",
        summary="Add structured logging to the utils module",
    )

    # Holder should have the plan data
    assert holder.data is not None
    assert holder.data["branch_name"] == "feat/issue-42-add-logging"
    assert holder.data["task_type"] == "feature"
    assert holder.data["needs_tests"] is True
    assert holder.data["needs_docs"] is False
    assert holder.data["files_to_change"] == ["src/main.py", "src/utils.py"]
    assert holder.data["pr_title"] == "Add logging to utils"
    assert holder.data["summary"] == "Add structured logging to the utils module"


@pytest.mark.timeout(5)
async def test_record_plan_returns_confirmation():
    """record_plan returns a readable confirmation message."""
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)
    result = await tool.fn(
        branch_name="fix/bug-99",
        task_type="bug_fix",
        needs_tests=True,
        files_to_change=["src/bug.py"],
    )
    assert "Plan recorded" in result
    assert "bug_fix" in result
    assert "fix/bug-99" in result
    assert "needs_tests=True" in result


@pytest.mark.timeout(5)
async def test_record_plan_defaults_optional_fields():
    """record_plan defaults needs_docs to True and files_to_change to []."""
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)
    await tool.fn(
        branch_name="feat/test",
        task_type="feature",
        needs_tests=False,
    )

    assert holder.data is not None
    assert holder.data["needs_docs"] is True
    assert holder.data["files_to_change"] == []
    assert holder.data["pr_title"] == ""
    assert holder.data["summary"] == ""


@pytest.mark.timeout(5)
async def test_record_plan_rejects_invalid_task_type():
    """record_plan returns an error for invalid task_type."""
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)
    result = await tool.fn(
        branch_name="feat/test",
        task_type="invalid_type",
        needs_tests=True,
    )

    # Should return an error message, not store data
    assert "Error" in result
    assert "invalid_type" in result
    assert holder.data is None


@pytest.mark.timeout(5)
async def test_record_plan_valid_task_types():
    """record_plan accepts all valid task types."""
    for task_type in VALID_TASK_TYPES:
        holder = PlanHolder()
        tool = make_record_plan_tool(holder)
        result = await tool.fn(
            branch_name="test-branch",
            task_type=task_type,
            needs_tests=True,
        )
        assert "Plan recorded" in result
        assert holder.data is not None
        assert holder.data["task_type"] == task_type


@pytest.mark.timeout(5)
async def test_record_plan_coerces_needs_values_to_bool():
    """record_plan ensures needs_tests and needs_docs are bools."""
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)
    await tool.fn(
        branch_name="test-branch",
        task_type="feature",
        needs_tests=1,  # truthy non-bool
        needs_docs=0,   # falsy non-bool
    )

    assert holder.data is not None
    assert holder.data["needs_tests"] is True
    assert holder.data["needs_docs"] is False


@pytest.mark.timeout(5)
async def test_record_plan_overwrites_previous():
    """Calling record_plan again overwrites the previous plan."""
    holder = PlanHolder()
    tool = make_record_plan_tool(holder)

    await tool.fn(
        branch_name="first-branch",
        task_type="feature",
        needs_tests=True,
    )
    assert holder.data["branch_name"] == "first-branch"

    await tool.fn(
        branch_name="second-branch",
        task_type="bug_fix",
        needs_tests=False,
    )
    assert holder.data["branch_name"] == "second-branch"
    assert holder.data["task_type"] == "bug_fix"
