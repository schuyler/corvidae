"""
Tests for Part 4 of agent decomposition: ToolCollectionPlugin extraction.

All tests in this file are expected to FAIL against the current codebase
(red phase). They will pass after the implementation in corvidae/tool_collection.py
is applied.
"""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. ToolCollectionPlugin class exists and can be imported
# ---------------------------------------------------------------------------

def test_tool_collection_plugin_importable():
    from corvidae.tool_collection import ToolCollectionPlugin  # noqa: F401


# ---------------------------------------------------------------------------
# 2. ToolCollectionPlugin.depends_on is set() (no dependencies)
# ---------------------------------------------------------------------------

def test_tool_collection_plugin_depends_on_is_empty():
    from corvidae.tool_collection import ToolCollectionPlugin
    assert ToolCollectionPlugin.depends_on == set(), (
        "ToolCollectionPlugin.depends_on must be set() — it has no dependencies"
    )


# ---------------------------------------------------------------------------
# 3. __init__ accepts pm, initializes registry as None and
#    max_result_chars as 100_000
# ---------------------------------------------------------------------------

def test_tool_collection_plugin_init():
    from corvidae.tool_collection import ToolCollectionPlugin
    pm = MagicMock()
    plugin = ToolCollectionPlugin(pm)
    assert plugin.pm is pm
    assert plugin.registry is None
    assert plugin.max_result_chars == 100_000


# ---------------------------------------------------------------------------
# 4. get_registry() returns the registry
# ---------------------------------------------------------------------------

def test_tool_collection_plugin_get_registry_returns_registry():
    from corvidae.tool_collection import ToolCollectionPlugin
    from corvidae.tool import ToolRegistry
    pm = MagicMock()
    plugin = ToolCollectionPlugin(pm)
    registry = ToolRegistry()
    plugin.registry = registry
    assert plugin.get_registry() is registry


# ---------------------------------------------------------------------------
# 5. Config: reads tools.max_result_chars as canonical key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_collection_plugin_reads_tools_max_result_chars():
    from corvidae.tool_collection import ToolCollectionPlugin
    pm = MagicMock()
    pm.hook = MagicMock()
    pm.hook.register_tools = MagicMock(return_value=[])
    plugin = ToolCollectionPlugin(pm)
    config = {"tools": {"max_result_chars": 50_000}}
    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)
    assert plugin.max_result_chars == 50_000


# ---------------------------------------------------------------------------
# 6. Config: falls back to agent.max_tool_result_chars if new key absent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_collection_plugin_falls_back_to_agent_max_tool_result_chars():
    from corvidae.tool_collection import ToolCollectionPlugin
    pm = MagicMock()
    pm.hook = MagicMock()
    pm.hook.register_tools = MagicMock(return_value=[])
    plugin = ToolCollectionPlugin(pm)
    config = {"agent": {"max_tool_result_chars": 75_000}}
    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)
    assert plugin.max_result_chars == 75_000


# ---------------------------------------------------------------------------
# 7. Config: defaults to 100_000 if neither key present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_collection_plugin_defaults_max_result_chars():
    from corvidae.tool_collection import ToolCollectionPlugin
    pm = MagicMock()
    pm.hook = MagicMock()
    pm.hook.register_tools = MagicMock(return_value=[])
    plugin = ToolCollectionPlugin(pm)
    config = {}
    await plugin.on_start(config=config)
    assert plugin.max_result_chars == 100_000


# ---------------------------------------------------------------------------
# 8. on_start calls pm.hook.register_tools and builds a ToolRegistry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_collection_plugin_on_start_builds_registry():
    from corvidae.tool_collection import ToolCollectionPlugin
    from corvidae.tool import Tool, ToolRegistry

    async def my_test_tool(x: str) -> str:
        """A test tool."""
        return x

    tool = Tool.from_function(my_test_tool)

    pm = MagicMock()
    pm.hook = MagicMock()
    # Simulate pluggy collecting tools: register_tools fills the list passed as tool_registry
    def fake_register_tools(tool_registry):
        tool_registry.append(tool)

    pm.hook.register_tools = MagicMock(side_effect=fake_register_tools)

    plugin = ToolCollectionPlugin(pm)
    await plugin.on_start(config={})

    pm.hook.register_tools.assert_called_once()
    assert plugin.registry is not None
    assert isinstance(plugin.registry, ToolRegistry)
    assert "my_test_tool" in plugin.registry.as_dict()


# ---------------------------------------------------------------------------
# 9. tools.disabled: drops named tools from the built registry (red phase,
#    disentangle-buster.md Design §2 "new small feature riding this stream")
# ---------------------------------------------------------------------------


def _fake_register_two_tools(pm):
    """Wire pm.hook.register_tools to collect tool_a and tool_b."""
    from corvidae.tool import Tool

    async def tool_a(x: str) -> str:
        """Tool A."""
        return x

    async def tool_b(x: str) -> str:
        """Tool B."""
        return x

    def fake_register_tools(tool_registry):
        tool_registry.append(Tool.from_function(tool_a))
        tool_registry.append(Tool.from_function(tool_b))

    pm.hook.register_tools = MagicMock(side_effect=fake_register_tools)


@pytest.mark.asyncio
async def test_tools_disabled_filters_named_tools():
    """A tool named in tools.disabled is dropped from the built registry;
    unnamed tools survive. This is the single chokepoint every registered
    tool passes through -- no per-plugin opt-outs."""
    from corvidae.tool_collection import ToolCollectionPlugin

    pm = MagicMock()
    pm.hook = MagicMock()
    _fake_register_two_tools(pm)

    plugin = ToolCollectionPlugin(pm)
    config = {"tools": {"disabled": ["tool_b"]}}
    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)

    tools = plugin.registry.as_dict()
    assert "tool_a" in tools
    assert "tool_b" not in tools


@pytest.mark.asyncio
async def test_tools_disabled_logs_dropped_count_at_info(caplog):
    """Dropping disabled tools logs at INFO (design: "log the count at INFO")."""
    import logging

    from corvidae.tool_collection import ToolCollectionPlugin

    pm = MagicMock()
    pm.hook = MagicMock()
    _fake_register_two_tools(pm)

    plugin = ToolCollectionPlugin(pm)
    config = {"tools": {"disabled": ["tool_b"]}}
    await plugin.on_init(pm=pm, config=config)
    with caplog.at_level(logging.INFO, logger="corvidae.tool_collection"):
        await plugin.on_start(config=config)

    assert any(
        record.levelno == logging.INFO and "disabled" in record.getMessage().lower()
        for record in caplog.records
    ), f"expected an INFO log mentioning disabled tools; got: {[r.getMessage() for r in caplog.records]}"


@pytest.mark.asyncio
async def test_tools_disabled_absent_key_is_noop():
    """No tools.disabled key at all -> every collected tool survives
    (the filter step is skipped, not applied with an empty allowlist that
    happens to keep everything)."""
    from corvidae.tool_collection import ToolCollectionPlugin

    pm = MagicMock()
    pm.hook = MagicMock()
    _fake_register_two_tools(pm)

    plugin = ToolCollectionPlugin(pm)
    config = {}  # no "tools" key at all
    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)

    tools = plugin.registry.as_dict()
    assert "tool_a" in tools
    assert "tool_b" in tools
