"""
Tests for planned changes in plans/declare-deps-clean-reexports.md and
plans/agent-decomposition-parts-3-4.md.

Tests marked with a comment referencing the parts-3-4 plan are expected to
FAIL until Part 3 of the agent-decomposition refactor is implemented. Other
tests here FAIL until the declare-deps plan is applied.
"""

import importlib
import inspect

import pytest


# ---------------------------------------------------------------------------
# 1. Agent.depends_on must include "task"
# ---------------------------------------------------------------------------

def test_agent_plugin_depends_on_includes_task():
    from corvidae.agent import Agent
    assert "task" in Agent.depends_on, (
        "Agent.depends_on should include 'task' — see plans/declare-deps-clean-reexports.md §2.1"
    )


# ---------------------------------------------------------------------------
# 2. IdleMonitorPlugin.depends_on must include "task"
# ---------------------------------------------------------------------------

def test_idle_monitor_plugin_depends_on_is_empty():
    from corvidae.idle import IdleMonitorPlugin
    assert "task" not in IdleMonitorPlugin.depends_on, (
        "IdleMonitorPlugin.depends_on should not include 'task' — Part 2 made it a pure on_idle consumer"
    )
    assert "agent" not in IdleMonitorPlugin.depends_on, (
        "IdleMonitorPlugin.depends_on should not include 'agent' — Part 2 made it a pure on_idle consumer"
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
# 4. turn must NOT re-export tool-related symbols (single-LLM-turn module).
#    tools.subagent owns run_agent_loop and MUST expose MAX_TOOL_RESULT_CHARS
#    and dispatch_tool_call (used by run_agent_loop).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def turn_module():
    return importlib.import_module("corvidae.turn")


@pytest.fixture(scope="module")
def subagent_module():
    return importlib.import_module("corvidae.tools.subagent")


def test_turn_does_not_export_ToolContext(turn_module):
    assert not hasattr(turn_module, "ToolContext"), (
        "corvidae.turn should not re-export ToolContext"
    )


def test_turn_does_not_export_execute_tool_call(turn_module):
    assert not hasattr(turn_module, "execute_tool_call"), (
        "corvidae.turn should not re-export execute_tool_call"
    )


def test_turn_does_not_export_tool_to_schema(turn_module):
    assert not hasattr(turn_module, "tool_to_schema"), (
        "corvidae.turn should not re-export tool_to_schema"
    )


def test_subagent_still_exports_MAX_TOOL_RESULT_CHARS(subagent_module):
    assert hasattr(subagent_module, "MAX_TOOL_RESULT_CHARS"), (
        "corvidae.tools.subagent should expose MAX_TOOL_RESULT_CHARS (used as default arg in run_agent_loop)"
    )


def test_subagent_still_exports_dispatch_tool_call(subagent_module):
    assert hasattr(subagent_module, "dispatch_tool_call"), (
        "corvidae.tools.subagent should expose dispatch_tool_call (called inside run_agent_loop)"
    )


# ---------------------------------------------------------------------------
# 5. Agent.depends_on must include "llm" (Part 3 red phase)
# ---------------------------------------------------------------------------


def test_agent_plugin_depends_on_includes_llm():
    """Agent.depends_on must include 'llm' after Part 3 of the
    agent-decomposition refactor.

    See plans/agent-decomposition-parts-3-4.md §Part 3.
    """
    from corvidae.agent import Agent

    assert "llm" in Agent.depends_on, (
        "Agent.depends_on should include 'llm' — "
        "see plans/agent-decomposition-parts-3-4.md §Part 3"
    )


# ---------------------------------------------------------------------------
# 6. CompactionPlugin.depends_on must include "llm" (Part 3 red phase)
# ---------------------------------------------------------------------------


def test_compaction_plugin_depends_on_includes_llm():
    """CompactionPlugin.depends_on must include 'llm' after Part 3.

    Currently CompactionPlugin has no depends_on attribute at all.
    This test fails until depends_on = {'llm'} is added.

    See plans/agent-decomposition-parts-3-4.md §Part 3.
    """
    from corvidae.compaction import CompactionPlugin

    assert hasattr(CompactionPlugin, "depends_on"), (
        "CompactionPlugin must have a 'depends_on' class attribute"
    )
    assert "llm" in CompactionPlugin.depends_on, (
        "CompactionPlugin.depends_on should include 'llm' — "
        "see plans/agent-decomposition-parts-3-4.md §Part 3"
    )


# ---------------------------------------------------------------------------
# 7. SubagentPlugin.depends_on must include "llm" (Part 3 red phase)
# ---------------------------------------------------------------------------


def test_subagent_plugin_depends_on_includes_llm():
    """SubagentPlugin.depends_on must include 'llm' after Part 3.

    Currently SubagentPlugin.depends_on = {'agent_loop'} — does not
    include 'llm'. This test fails until 'llm' is added.

    See plans/agent-decomposition-parts-3-4.md §Part 3.
    """
    from corvidae.tools.subagent import SubagentPlugin

    assert "llm" in SubagentPlugin.depends_on, (
        "SubagentPlugin.depends_on should include 'llm' — "
        "see plans/agent-decomposition-parts-3-4.md §Part 3"
    )


# ---------------------------------------------------------------------------
# 8. SubagentPlugin.depends_on is {"llm", "tools"} with no "agent_loop" (Part 4 red phase)
# ---------------------------------------------------------------------------


def test_subagent_plugin_depends_on_is_llm_and_tools():
    """SubagentPlugin.depends_on must be exactly {"llm", "tools"} after Part 4.

    After Part 4, SubagentPlugin accesses the tool registry directly via
    ToolCollectionPlugin, removing the "agent_loop" dependency entirely.

    See plans/agent-decomposition-parts-3-4.md §Part 4.
    """
    from corvidae.tools.subagent import SubagentPlugin

    assert SubagentPlugin.depends_on == {"llm", "tools"}, (
        f"SubagentPlugin.depends_on should be {{'llm', 'tools'}}, got {SubagentPlugin.depends_on!r} — "
        "see plans/agent-decomposition-parts-3-4.md §Part 4"
    )


def test_subagent_plugin_depends_on_excludes_agent_loop():
    """SubagentPlugin.depends_on must NOT include 'agent_loop' after Part 4.

    The 'agent_loop' dependency is removed when SubagentPlugin starts
    reading tool config from ToolCollectionPlugin directly.

    See plans/agent-decomposition-parts-3-4.md §Part 4.
    """
    from corvidae.tools.subagent import SubagentPlugin

    assert "agent" not in SubagentPlugin.depends_on, (
        "SubagentPlugin.depends_on should not include 'agent' after Part 4 — "
        "see plans/agent-decomposition-parts-3-4.md §Part 4"
    )


def test_subagent_plugin_depends_on_includes_tools():
    """SubagentPlugin.depends_on must include 'tools' after Part 4.

    SubagentPlugin reads tool registry and max_result_chars from
    ToolCollectionPlugin, so it must declare this dependency.

    See plans/agent-decomposition-parts-3-4.md §Part 4.
    """
    from corvidae.tools.subagent import SubagentPlugin

    assert "tools" in SubagentPlugin.depends_on, (
        "SubagentPlugin.depends_on should include 'tools' — "
        "see plans/agent-decomposition-parts-3-4.md §Part 4"
    )


# ---------------------------------------------------------------------------
# 9. All registered plugins have depends_on and accept pm as first positional arg
# ---------------------------------------------------------------------------

_REGISTERED_PLUGIN_CLASSES = [
    "corvidae.persistence.PersistencePlugin",
    "corvidae.jsonl_log.JsonlLogPlugin",
    "corvidae.tools.CoreToolsPlugin",
    "corvidae.channels.cli.CLIPlugin",
    "corvidae.channels.irc.IRCPlugin",
    "corvidae.task.TaskPlugin",
    "corvidae.tools.subagent.SubagentPlugin",
    "corvidae.mcp_client.McpClientPlugin",
    "corvidae.llm_plugin.LLMPlugin",
    "corvidae.compaction.CompactionPlugin",
    "corvidae.thinking.ThinkingPlugin",
    "corvidae.context_compact.ContextCompactPlugin",
    "corvidae.tools.settings.RuntimeSettingsPlugin",
    "corvidae.tools.index.WorkspaceIndexerPlugin",
    "corvidae.tool_collection.ToolCollectionPlugin",
    "corvidae.agent.Agent",
    "corvidae.idle.IdleMonitorPlugin",
]


@pytest.mark.parametrize("dotted_name", _REGISTERED_PLUGIN_CLASSES)
def test_plugin_has_depends_on(dotted_name):
    """Every plugin registered in main.py must have a depends_on class attribute."""
    module_path, class_name = dotted_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert hasattr(cls, "depends_on"), (
        f"{dotted_name} must have a 'depends_on' class attribute — "
        "see plans/four-cleanups-design.md §2"
    )


@pytest.mark.parametrize("dotted_name", _REGISTERED_PLUGIN_CLASSES)
def test_plugin_init_accepts_pm_as_first_positional_arg(dotted_name):
    """Every plugin registered in main.py must accept pm as the first positional arg."""
    module_path, class_name = dotted_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.values())
    # params[0] is always 'self'; params[1] should be 'pm'
    assert len(params) >= 2, (
        f"{dotted_name}.__init__ must have at least one argument besides self"
    )
    assert params[1].name == "pm", (
        f"{dotted_name}.__init__ first non-self parameter must be 'pm', "
        f"got {params[1].name!r} — see plans/four-cleanups-design.md §2"
    )
    # pm must be positional (POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD), not keyword-only
    assert params[1].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), (
        f"{dotted_name}.__init__ 'pm' must be a positional parameter, "
        f"got kind={params[1].kind.name!r}"
    )
