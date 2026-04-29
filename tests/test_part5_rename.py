"""Failing tests for Part 5: AgentPlugin → Agent rename.

These tests are written to FAIL with the current code (where the class is
named AgentPlugin and registered as "agent_loop") and PASS after the rename
is implemented.
"""

import inspect
import pytest


# ---------------------------------------------------------------------------
# Test 1: Agent is importable from corvidae.agent
# ---------------------------------------------------------------------------

def test_agent_class_importable():
    """Agent (not AgentPlugin) must be the primary export from corvidae.agent."""
    from corvidae.agent import Agent  # noqa: F401 — import is the assertion


# ---------------------------------------------------------------------------
# Test 2: Agent has expected attributes
# ---------------------------------------------------------------------------

def test_agent_has_expected_attributes():
    """Agent class must expose depends_on, queues, and _client attributes."""
    from corvidae.agent import Agent

    # depends_on is a class-level attribute
    assert hasattr(Agent, "depends_on"), "Agent.depends_on not found"
    assert "registry" in Agent.depends_on
    assert "llm" in Agent.depends_on

    # Instance attributes set in __init__
    from unittest.mock import MagicMock
    pm = MagicMock()
    instance = Agent(pm)
    assert hasattr(instance, "queues"), "Agent instance missing 'queues'"
    assert hasattr(instance, "_client"), "Agent instance missing '_client'"
    assert instance._client is None, "_client should be None before on_start"
    assert isinstance(instance.queues, dict), "queues should be a dict"


# ---------------------------------------------------------------------------
# Test 3: corvidae/main.py registers the plugin as name="agent"
# ---------------------------------------------------------------------------

def test_main_registers_as_agent_not_agent_loop():
    """Agent must be registered under name 'agent' (via entry point), not 'agent_loop'."""
    from importlib.metadata import entry_points
    from corvidae.agent import Agent

    # Verify the entry point registers Agent under name 'agent'
    eps = {ep.name: ep for ep in entry_points(group="corvidae")}
    assert "agent" in eps, (
        "Entry point group 'corvidae' must have an 'agent' entry"
    )
    assert "agent_loop" not in eps, (
        "Entry point group 'corvidae' must not have an 'agent_loop' entry"
    )
    loaded = eps["agent"].load()
    assert loaded is Agent, (
        f"Entry point 'agent' must load corvidae.agent.Agent, got {loaded!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: Backward-compat alias AgentPlugin still works and is Agent
# ---------------------------------------------------------------------------

def test_agentplugin_alias_is_agent():
    """AgentPlugin must be an alias for Agent (same object)."""
    from corvidae.agent import Agent, AgentPlugin

    assert AgentPlugin is Agent, (
        "AgentPlugin must be the same object as Agent (backward-compat alias)"
    )
