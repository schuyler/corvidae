"""
Tests for planned changes in plans/declare-deps-clean-reexports.md.

All tests in this file are expected to FAIL against the current codebase
(red phase). They will pass after the implementation is applied.
"""

import importlib

import pytest


# ---------------------------------------------------------------------------
# 1. AgentPlugin.depends_on must include "task"
# ---------------------------------------------------------------------------

def test_agent_plugin_depends_on_includes_task():
    from corvidae.agent import AgentPlugin
    assert "task" in AgentPlugin.depends_on, (
        "AgentPlugin.depends_on should include 'task' — see plans/declare-deps-clean-reexports.md §2.1"
    )


# ---------------------------------------------------------------------------
# 2. IdleMonitorPlugin.depends_on must include "task"
# ---------------------------------------------------------------------------

def test_idle_monitor_plugin_depends_on_is_empty():
    from corvidae.idle import IdleMonitorPlugin
    assert "task" not in IdleMonitorPlugin.depends_on, (
        "IdleMonitorPlugin.depends_on should not include 'task' — Part 2 made it a pure on_idle consumer"
    )
    assert "agent_loop" not in IdleMonitorPlugin.depends_on, (
        "IdleMonitorPlugin.depends_on should not include 'agent_loop' — Part 2 made it a pure on_idle consumer"
    )


# ---------------------------------------------------------------------------
# 3. ThinkingPlugin must have depends_on including "registry"
# ---------------------------------------------------------------------------

def test_thinking_plugin_has_depends_on():
    from corvidae.thinking import ThinkingPlugin
    assert hasattr(ThinkingPlugin, "depends_on"), (
        "ThinkingPlugin should have a 'depends_on' class attribute — see plans/declare-deps-clean-reexports.md §2.3"
    )


def test_thinking_plugin_depends_on_includes_registry():
    from corvidae.thinking import ThinkingPlugin
    assert "registry" in ThinkingPlugin.depends_on, (
        "ThinkingPlugin.depends_on should include 'registry' — see plans/declare-deps-clean-reexports.md §2.3"
    )


# ---------------------------------------------------------------------------
# 4. agent_loop must NOT re-export ToolContext, execute_tool_call,
#    or tool_to_schema; but MUST still expose MAX_TOOL_RESULT_CHARS
#    and dispatch_tool_call.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agent_loop_module():
    return importlib.import_module("corvidae.agent_loop")


def test_agent_loop_does_not_export_ToolContext(agent_loop_module):
    assert not hasattr(agent_loop_module, "ToolContext"), (
        "corvidae.agent_loop should not re-export ToolContext — see plans/declare-deps-clean-reexports.md §2.7"
    )


def test_agent_loop_does_not_export_execute_tool_call(agent_loop_module):
    assert not hasattr(agent_loop_module, "execute_tool_call"), (
        "corvidae.agent_loop should not re-export execute_tool_call — see plans/declare-deps-clean-reexports.md §2.7"
    )


def test_agent_loop_does_not_export_tool_to_schema(agent_loop_module):
    assert not hasattr(agent_loop_module, "tool_to_schema"), (
        "corvidae.agent_loop should not re-export tool_to_schema — see plans/declare-deps-clean-reexports.md §2.7"
    )


def test_agent_loop_still_exports_MAX_TOOL_RESULT_CHARS(agent_loop_module):
    assert hasattr(agent_loop_module, "MAX_TOOL_RESULT_CHARS"), (
        "corvidae.agent_loop should still expose MAX_TOOL_RESULT_CHARS (used as default arg at line 132)"
    )


def test_agent_loop_still_exports_dispatch_tool_call(agent_loop_module):
    assert hasattr(agent_loop_module, "dispatch_tool_call"), (
        "corvidae.agent_loop should still expose dispatch_tool_call (called at line 168)"
    )
