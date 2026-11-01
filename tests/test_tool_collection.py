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


# ---------------------------------------------------------------------------
# Registry generation counter (issue #109, R1)
# ---------------------------------------------------------------------------


async def _generation_plugin(tool_fns: list):
    """A started ToolCollectionPlugin whose register_tools yields the live
    contents of ``tool_fns``; mutate the list, then ``rebuild_registry``."""
    from corvidae.hooks import create_plugin_manager, hookimpl
    from corvidae.tool_collection import ToolCollectionPlugin

    pm = create_plugin_manager()

    class Source:
        @hookimpl
        def register_tools(self, tool_registry):
            # Offer whatever the test currently has in the list.
            tool_registry.extend(tool_fns)

    pm.register(Source(), name="source")
    plugin = ToolCollectionPlugin(pm)
    pm.register(plugin, name="tools")
    await plugin.on_init(pm=pm, config={})
    await plugin.on_start(config={})
    return plugin


def _make_fn(name: str, doc: str = "Do a thing."):
    async def fn(x: str) -> str:
        return x

    fn.__name__ = name
    fn.__doc__ = doc
    return fn


class TestRegistryGeneration:
    async def test_added_tool_bumps_generation_by_one(self):
        fns = [_make_fn("alpha")]
        plugin = await _generation_plugin(fns)
        before = plugin.generation
        fns.append(_make_fn("beta"))
        await plugin.rebuild_registry()
        assert plugin.generation == before + 1

    async def test_removed_tool_bumps_generation_by_one(self):
        fns = [_make_fn("alpha"), _make_fn("beta")]
        plugin = await _generation_plugin(fns)
        before = plugin.generation
        fns.pop()
        await plugin.rebuild_registry()
        assert plugin.generation == before + 1

    async def test_changed_schema_bumps_generation_by_one(self):
        fns = [_make_fn("alpha", "Old description.")]
        plugin = await _generation_plugin(fns)
        before = plugin.generation
        fns[0] = _make_fn("alpha", "New description.")
        await plugin.rebuild_registry()
        assert plugin.generation == before + 1

    async def test_identical_rebuild_keeps_generation(self):
        fns = [_make_fn("alpha")]
        plugin = await _generation_plugin(fns)
        before = plugin.generation
        # A rebuild from fresh-but-equal tool objects is not a change.
        fns[0] = _make_fn("alpha")
        await plugin.rebuild_registry()
        assert plugin.generation == before


# ---------------------------------------------------------------------------
# Issue #94: tools.call_timeout / tools.timeouts parsing (R6) and wiring
# ---------------------------------------------------------------------------


async def test_agent_start_copies_call_timeouts():
    """R2: Agent._start_plugin copies call_timeouts from the tools plugin."""
    from unittest.mock import NonCallableMagicMock

    from helpers import build_plugin_and_channel
    from corvidae.llm_plugin import LLMPlugin
    from corvidae.tool import ToolTimeouts
    from corvidae.tool_collection import ToolCollectionPlugin

    agent, _channel, db = await build_plugin_and_channel()
    llm = NonCallableMagicMock(spec=LLMPlugin)
    llm.main_client = MagicMock()
    tools = NonCallableMagicMock(spec=ToolCollectionPlugin)
    tools.get_tools.return_value = ({}, [])
    tools.max_result_chars = 100_000
    tools.generation = 0
    tools.call_timeouts = ToolTimeouts(0.1)
    agent.pm.register(llm, name="llm")
    agent.pm.register(tools, name="tools")
    try:
        await agent._start_plugin({})
        assert agent._tool_call_timeouts is tools.call_timeouts
    finally:
        await agent.pm.get_plugin("task").on_stop()
        await db.close()


def test_parse_accepts_configured_values():
    """R1/R3: a configured positive call_timeout and per-tool overrides are
    used as given, ints and floats alike; an absent tools block gives 600."""
    from corvidae.tool_collection import parse_tool_timeouts

    assert parse_tool_timeouts({"call_timeout": 30}).default == 30
    assert parse_tool_timeouts({"call_timeout": 0.5}).default == 0.5
    assert parse_tool_timeouts(
        {"timeouts": {"slow_tool": 0.2}}
    ).per_tool == {"slow_tool": 0.2}
    assert parse_tool_timeouts(None).default == 600.0


@pytest.mark.parametrize("value", ["abc", -1, True, float("nan"), float("inf"), [1]])
def test_parse_rejects_bad_call_timeout(value):
    """A bad call_timeout names the key, the value and the accepted form."""
    from corvidae.tool_collection import parse_tool_timeouts

    with pytest.raises(ValueError) as excinfo:
        parse_tool_timeouts({"call_timeout": value})
    msg = str(excinfo.value)
    assert "tools.call_timeout" in msg
    assert repr(value) in msg
    assert "number of seconds >= 0" in msg
    assert "0 or null disables" in msg


def test_parse_rejects_non_mapping_timeouts():
    """A non-mapping tools.timeouts is rejected with an example."""
    from corvidae.tool_collection import parse_tool_timeouts

    with pytest.raises(ValueError) as excinfo:
        parse_tool_timeouts({"timeouts": [1]})
    msg = str(excinfo.value)
    assert "tools.timeouts must be a mapping" in msg
    assert "[1]" in msg
    # Accepted values: a mapping of tool names to seconds.
    assert "tool names to seconds" in msg


@pytest.mark.parametrize("value", [-2, "x"])
def test_parse_rejects_bad_entry(value):
    """A bad per-tool entry names tools.timeouts.<name> and the value."""
    from corvidae.tool_collection import parse_tool_timeouts

    with pytest.raises(ValueError) as excinfo:
        parse_tool_timeouts({"timeouts": {"slow": value}})
    msg = str(excinfo.value)
    assert "tools.timeouts.slow" in msg
    assert repr(value) in msg
    assert "number of seconds >= 0" in msg
    assert "0 or null disables" in msg


def test_parse_rejects_non_string_key():
    """A non-string key is rejected; the message says keys are tool names (strings)."""
    from corvidae.tool_collection import parse_tool_timeouts

    with pytest.raises(ValueError) as excinfo:
        parse_tool_timeouts({"timeouts": {1: 5}})
    msg = str(excinfo.value)
    assert "tools.timeouts" in msg
    assert "non-string tool name" in msg
    assert "1" in msg and "int" in msg
    # Remedy: keys must be tool names, which are strings.
    assert "tool names" in msg
    assert "strings" in msg


@pytest.mark.asyncio
async def test_unknown_timeouts_entry_warns_and_is_kept(caplog):
    """An entry naming no registered tool warns once and is kept."""
    import logging

    from corvidae.tool import Tool
    from corvidae.tool_collection import ToolCollectionPlugin

    async def echo(x: str) -> str:
        """Echo."""
        return x

    pm = MagicMock()
    pm.hook = MagicMock()
    pm.hook.register_tools = MagicMock(
        side_effect=lambda tool_registry: tool_registry.append(Tool.from_function(echo))
    )

    plugin = ToolCollectionPlugin(pm)
    # One unknown name and one registered name: only the unknown one warns,
    # and only once the registry is built (not at on_init).
    config = {"tools": {"timeouts": {"ghost": 5, "echo": 5}}}
    with caplog.at_level(logging.WARNING, logger="corvidae.tool_collection"):
        await plugin.on_init(pm=pm, config=config)
        await plugin.on_start(config=config)

    warns = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "tools.timeouts.ghost" in r.getMessage()
        and "keeping it" in r.getMessage()
    ]
    assert len(warns) == 1
    assert not any(
        "tools.timeouts.echo" in r.getMessage() for r in caplog.records
    )
    assert plugin.call_timeouts.per_tool["ghost"] == 5
    assert plugin.call_timeouts.per_tool["echo"] == 5


@pytest.mark.parametrize("value", [[1], "x", 5])
def test_parse_rejects_non_mapping_tools(value):
    """A non-mapping ``tools`` is rejected naming the key, value and accepted form."""
    from corvidae.tool_collection import parse_tool_timeouts

    with pytest.raises(ValueError) as excinfo:
        parse_tool_timeouts(value)
    msg = str(excinfo.value)
    # Key, accepted form and rejected value, per the design's message.
    assert "tools must be a mapping" in msg
    assert repr(value) in msg
