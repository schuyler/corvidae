"""Tests for the plugin registration refactor (entry points + on_init hook).

These tests cover:
1. on_init hookspec on AgentSpec
2. CorvidaePlugin base class in corvidae.hooks
3. No-arg constructors for all 16 entry-point plugins
4. CorvidaePlugin subclassing for each plugin
5. Entry points registered in the "corvidae" group
6. RuntimeSettingsPlugin.on_init extends blocklist from config
7. DreamPlugin.on_start and on_idle have @hookimpl decorators

All tests FAIL until the implementation is complete.
"""

import pytest

from corvidae.hooks import AgentSpec, create_plugin_manager


# ---------------------------------------------------------------------------
# 1. on_init hookspec on AgentSpec
# ---------------------------------------------------------------------------


def test_agentspec_has_on_init():
    """AgentSpec must have an on_init hookspec."""
    assert hasattr(AgentSpec, "on_init"), (
        "AgentSpec must have an on_init hookspec"
    )


async def test_on_init_hookspec_is_callable_via_pm():
    """on_init must be dispatchable via pm.ahook.on_init without error."""

    class _Listener:
        from corvidae.hooks import hookimpl

        @hookimpl
        async def on_init(self, pm, config):
            self.got_pm = pm
            self.got_config = config

    pm = create_plugin_manager()
    listener = _Listener()
    pm.register(listener)
    await pm.ahook.on_init(pm=pm, config={"key": "value"})

    assert listener.got_pm is pm
    assert listener.got_config == {"key": "value"}


# ---------------------------------------------------------------------------
# 2. CorvidaePlugin base class
# ---------------------------------------------------------------------------


def test_corvidae_plugin_exists_in_hooks():
    """CorvidaePlugin must be importable from corvidae.hooks."""
    from corvidae.hooks import CorvidaePlugin  # noqa: F401


def test_corvidae_plugin_depends_on_is_frozenset():
    """CorvidaePlugin.depends_on must be a frozenset."""
    from corvidae.hooks import CorvidaePlugin

    plugin = CorvidaePlugin()
    assert isinstance(plugin.depends_on, frozenset), (
        "CorvidaePlugin.depends_on must be a frozenset, not "
        f"{type(plugin.depends_on).__name__}"
    )


async def test_corvidae_plugin_on_init_stores_pm_and_config():
    """CorvidaePlugin.on_init must store pm and config as instance attributes."""
    from corvidae.hooks import CorvidaePlugin

    pm = create_plugin_manager()
    plugin = CorvidaePlugin()
    pm.register(plugin)
    await plugin.on_init(pm=pm, config={"section": "data"})

    assert plugin.pm is pm, "CorvidaePlugin.on_init must set self.pm"
    assert plugin.config == {"section": "data"}, (
        "CorvidaePlugin.on_init must set self.config"
    )


def test_corvidae_plugin_on_init_has_hookimpl():
    """CorvidaePlugin.on_init must be decorated with @hookimpl."""
    from corvidae.hooks import CorvidaePlugin, hookimpl
    import pluggy

    marker_attr = hookimpl.project_name + "_impl"  # "corvidae_impl"
    impl_opts = getattr(CorvidaePlugin.on_init, marker_attr, None)
    assert impl_opts is not None, (
        "CorvidaePlugin.on_init must be decorated with @hookimpl"
    )


# ---------------------------------------------------------------------------
# 3. No-arg constructors for entry-point plugins
#    (ChannelRegistry is intentionally excluded per the design)
# ---------------------------------------------------------------------------


def test_persistence_plugin_no_arg_constructor():
    """PersistencePlugin() must work with no arguments."""
    from corvidae.persistence import PersistencePlugin
    plugin = PersistencePlugin()
    assert plugin is not None


def test_jsonl_log_plugin_no_arg_constructor():
    """JsonlLogPlugin() must work with no arguments."""
    from corvidae.jsonl_log import JsonlLogPlugin
    plugin = JsonlLogPlugin()
    assert plugin is not None


def test_core_tools_plugin_no_arg_constructor():
    """CoreToolsPlugin() must work with no arguments."""
    from corvidae.tools import CoreToolsPlugin
    plugin = CoreToolsPlugin()
    assert plugin is not None


def test_cli_plugin_no_arg_constructor():
    """CLIPlugin() must work with no arguments."""
    from corvidae.channels.cli import CLIPlugin
    plugin = CLIPlugin()
    assert plugin is not None


def test_irc_plugin_no_arg_constructor():
    """IRCPlugin() must work with no arguments."""
    from corvidae.channels.irc import IRCPlugin
    plugin = IRCPlugin()
    assert plugin is not None


def test_task_plugin_no_arg_constructor():
    """TaskPlugin() must work with no arguments."""
    from corvidae.task import TaskPlugin
    plugin = TaskPlugin()
    assert plugin is not None


def test_subagent_plugin_no_arg_constructor():
    """SubagentPlugin() must work with no arguments."""
    from corvidae.tools.subagent import SubagentPlugin
    plugin = SubagentPlugin()
    assert plugin is not None


def test_mcp_client_plugin_no_arg_constructor():
    """McpClientPlugin() must work with no arguments."""
    from corvidae.mcp_client import McpClientPlugin
    plugin = McpClientPlugin()
    assert plugin is not None


def test_llm_plugin_no_arg_constructor():
    """LLMPlugin() must work with no arguments."""
    from corvidae.llm_plugin import LLMPlugin
    plugin = LLMPlugin()
    assert plugin is not None


def test_compaction_plugin_no_arg_constructor():
    """CompactionPlugin() must work with no arguments."""
    from corvidae.compaction import CompactionPlugin
    plugin = CompactionPlugin()
    assert plugin is not None


def test_thinking_plugin_no_arg_constructor():
    """ThinkingPlugin() must work with no arguments."""
    from corvidae.thinking import ThinkingPlugin
    plugin = ThinkingPlugin()
    assert plugin is not None


def test_runtime_settings_plugin_no_arg_constructor():
    """RuntimeSettingsPlugin() must work with no arguments."""
    from corvidae.tools.settings import RuntimeSettingsPlugin
    plugin = RuntimeSettingsPlugin()
    assert plugin is not None


def test_tool_collection_plugin_no_arg_constructor():
    """ToolCollectionPlugin() must work with no arguments."""
    from corvidae.tool_collection import ToolCollectionPlugin
    plugin = ToolCollectionPlugin()
    assert plugin is not None


def test_dream_plugin_no_arg_constructor():
    """DreamPlugin() must work with no arguments."""
    from corvidae.tools.dream import DreamPlugin
    plugin = DreamPlugin()
    assert plugin is not None


def test_agent_no_arg_constructor():
    """Agent() must work with no arguments."""
    from corvidae.agent import Agent
    plugin = Agent()
    assert plugin is not None


def test_idle_monitor_plugin_no_arg_constructor():
    """IdleMonitorPlugin() must work with no arguments."""
    from corvidae.idle import IdleMonitorPlugin
    plugin = IdleMonitorPlugin()
    assert plugin is not None


def test_channel_registry_no_arg_constructor():
    """ChannelRegistry() must work with no arguments (agent_defaults defaults to {})."""
    from corvidae.channel import ChannelRegistry
    registry = ChannelRegistry()
    assert registry.agent_defaults == {}


# ---------------------------------------------------------------------------
# 4. CorvidaePlugin subclassing
# ---------------------------------------------------------------------------


def test_persistence_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.persistence import PersistencePlugin
    assert issubclass(PersistencePlugin, CorvidaePlugin), (
        "PersistencePlugin must subclass CorvidaePlugin"
    )


def test_jsonl_log_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.jsonl_log import JsonlLogPlugin
    assert issubclass(JsonlLogPlugin, CorvidaePlugin), (
        "JsonlLogPlugin must subclass CorvidaePlugin"
    )


def test_core_tools_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.tools import CoreToolsPlugin
    assert issubclass(CoreToolsPlugin, CorvidaePlugin), (
        "CoreToolsPlugin must subclass CorvidaePlugin"
    )


def test_cli_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.channels.cli import CLIPlugin
    assert issubclass(CLIPlugin, CorvidaePlugin), (
        "CLIPlugin must subclass CorvidaePlugin"
    )


def test_irc_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.channels.irc import IRCPlugin
    assert issubclass(IRCPlugin, CorvidaePlugin), (
        "IRCPlugin must subclass CorvidaePlugin"
    )


def test_task_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.task import TaskPlugin
    assert issubclass(TaskPlugin, CorvidaePlugin), (
        "TaskPlugin must subclass CorvidaePlugin"
    )


def test_subagent_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.tools.subagent import SubagentPlugin
    assert issubclass(SubagentPlugin, CorvidaePlugin), (
        "SubagentPlugin must subclass CorvidaePlugin"
    )


def test_mcp_client_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.mcp_client import McpClientPlugin
    assert issubclass(McpClientPlugin, CorvidaePlugin), (
        "McpClientPlugin must subclass CorvidaePlugin"
    )


def test_llm_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.llm_plugin import LLMPlugin
    assert issubclass(LLMPlugin, CorvidaePlugin), (
        "LLMPlugin must subclass CorvidaePlugin"
    )


def test_compaction_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.compaction import CompactionPlugin
    assert issubclass(CompactionPlugin, CorvidaePlugin), (
        "CompactionPlugin must subclass CorvidaePlugin"
    )


def test_thinking_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.thinking import ThinkingPlugin
    assert issubclass(ThinkingPlugin, CorvidaePlugin), (
        "ThinkingPlugin must subclass CorvidaePlugin"
    )


def test_runtime_settings_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.tools.settings import RuntimeSettingsPlugin
    assert issubclass(RuntimeSettingsPlugin, CorvidaePlugin), (
        "RuntimeSettingsPlugin must subclass CorvidaePlugin"
    )


def test_tool_collection_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.tool_collection import ToolCollectionPlugin
    assert issubclass(ToolCollectionPlugin, CorvidaePlugin), (
        "ToolCollectionPlugin must subclass CorvidaePlugin"
    )


def test_dream_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.tools.dream import DreamPlugin
    assert issubclass(DreamPlugin, CorvidaePlugin), (
        "DreamPlugin must subclass CorvidaePlugin"
    )


def test_agent_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.agent import Agent
    assert issubclass(Agent, CorvidaePlugin), (
        "Agent must subclass CorvidaePlugin"
    )


def test_idle_monitor_plugin_subclasses_corvidae_plugin():
    from corvidae.hooks import CorvidaePlugin
    from corvidae.idle import IdleMonitorPlugin
    assert issubclass(IdleMonitorPlugin, CorvidaePlugin), (
        "IdleMonitorPlugin must subclass CorvidaePlugin"
    )


# ---------------------------------------------------------------------------
# 5. Entry points registered in the "corvidae" group
# ---------------------------------------------------------------------------


def test_corvidae_entry_point_group_exists():
    """The 'corvidae' entry point group must exist after uv sync."""
    from importlib.metadata import entry_points
    eps = entry_points(group="corvidae")
    assert len(eps) > 0, (
        "No entry points found in 'corvidae' group. "
        "Run 'uv sync' and ensure [project.entry-points.corvidae] is in pyproject.toml."
    )


def test_corvidae_entry_points_contain_expected_names():
    """The 'corvidae' entry point group must contain all 16 expected plugin names."""
    from importlib.metadata import entry_points
    eps = entry_points(group="corvidae")
    names = {ep.name for ep in eps}

    expected = {
        "persistence",
        "jsonl_log",
        "core_tools",
        "cli",
        "irc",
        "task",
        "subagent",
        "mcp",
        "llm",
        "compaction",
        "thinking",
        "runtime_settings",
        "tools",
        "dream",
        "agent",
        "idle_monitor",
    }
    missing = expected - names
    assert not missing, (
        f"Entry point group 'corvidae' is missing: {sorted(missing)}"
    )


def test_corvidae_entry_points_load_correct_classes():
    """Each entry point must load the correct plugin class."""
    from importlib.metadata import entry_points

    expected_map = {
        "persistence": "corvidae.persistence:PersistencePlugin",
        "jsonl_log": "corvidae.jsonl_log:JsonlLogPlugin",
        "core_tools": "corvidae.tools:CoreToolsPlugin",
        "cli": "corvidae.channels.cli:CLIPlugin",
        "irc": "corvidae.channels.irc:IRCPlugin",
        "task": "corvidae.task:TaskPlugin",
        "subagent": "corvidae.tools.subagent:SubagentPlugin",
        "mcp": "corvidae.mcp_client:McpClientPlugin",
        "llm": "corvidae.llm_plugin:LLMPlugin",
        "compaction": "corvidae.compaction:CompactionPlugin",
        "thinking": "corvidae.thinking:ThinkingPlugin",
        "runtime_settings": "corvidae.tools.settings:RuntimeSettingsPlugin",
        "tools": "corvidae.tool_collection:ToolCollectionPlugin",
        "dream": "corvidae.tools.dream:DreamPlugin",
        "agent": "corvidae.agent:Agent",
        "idle_monitor": "corvidae.idle:IdleMonitorPlugin",
    }

    eps = {ep.name: ep for ep in entry_points(group="corvidae")}
    errors = []
    for name, expected_value in expected_map.items():
        if name not in eps:
            errors.append(f"{name!r}: not found in entry points")
            continue
        ep = eps[name]
        actual_value = ep.value
        if actual_value != expected_value:
            errors.append(
                f"{name!r}: expected {expected_value!r}, got {actual_value!r}"
            )

    assert not errors, "Entry point value mismatches:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# 6. RuntimeSettingsPlugin.on_init extends blocklist from config
# ---------------------------------------------------------------------------


async def test_runtime_settings_plugin_default_blocklist():
    """RuntimeSettingsPlugin() must initialise blocklist with 'system_prompt'."""
    from corvidae.tools.settings import RuntimeSettingsPlugin

    plugin = RuntimeSettingsPlugin()
    assert "system_prompt" in plugin.blocklist, (
        "RuntimeSettingsPlugin.__init__ must set blocklist to include 'system_prompt'"
    )


async def test_runtime_settings_plugin_on_init_extends_blocklist():
    """RuntimeSettingsPlugin.on_init must extend blocklist from config immutable_settings."""
    from corvidae.tools.settings import RuntimeSettingsPlugin

    pm = create_plugin_manager()
    plugin = RuntimeSettingsPlugin()
    pm.register(plugin, name="runtime_settings")
    await plugin.on_init(
        pm=pm,
        config={"agent": {"immutable_settings": ["max_turns", "max_context_tokens"]}},
    )

    assert "system_prompt" in plugin.blocklist, (
        "system_prompt must remain in blocklist after on_init"
    )
    assert "max_turns" in plugin.blocklist, (
        "max_turns must be added to blocklist via on_init config"
    )
    assert "max_context_tokens" in plugin.blocklist, (
        "max_context_tokens must be added to blocklist via on_init config"
    )


async def test_runtime_settings_plugin_on_init_empty_immutable_settings():
    """RuntimeSettingsPlugin.on_init with empty immutable_settings leaves default blocklist."""
    from corvidae.tools.settings import RuntimeSettingsPlugin

    pm = create_plugin_manager()
    plugin = RuntimeSettingsPlugin()
    pm.register(plugin, name="runtime_settings")
    await plugin.on_init(pm=pm, config={})

    assert "system_prompt" in plugin.blocklist, (
        "system_prompt must remain in blocklist when immutable_settings absent from config"
    )


# ---------------------------------------------------------------------------
# 7. DreamPlugin @hookimpl decorators on on_start and on_idle
# ---------------------------------------------------------------------------


def test_dream_plugin_on_start_has_hookimpl():
    """DreamPlugin.on_start must be decorated with @hookimpl."""
    from corvidae.tools.dream import DreamPlugin
    from corvidae.hooks import hookimpl

    marker_attr = hookimpl.project_name + "_impl"  # "corvidae_impl"
    impl_opts = getattr(DreamPlugin.on_start, marker_attr, None)
    assert impl_opts is not None, (
        "DreamPlugin.on_start must be decorated with @hookimpl — "
        "currently missing, which means on_start is never dispatched by pluggy"
    )


def test_dream_plugin_on_idle_has_hookimpl():
    """DreamPlugin.on_idle must be decorated with @hookimpl."""
    from corvidae.tools.dream import DreamPlugin
    from corvidae.hooks import hookimpl

    marker_attr = hookimpl.project_name + "_impl"  # "corvidae_impl"
    impl_opts = getattr(DreamPlugin.on_idle, marker_attr, None)
    assert impl_opts is not None, (
        "DreamPlugin.on_idle must be decorated with @hookimpl — "
        "currently missing, which means on_idle is never dispatched by pluggy"
    )


async def test_dream_plugin_dispatched_via_pm_on_start():
    """DreamPlugin.on_start must be called when pm.ahook.on_start is broadcast."""
    from corvidae.tools.dream import DreamPlugin

    pm = create_plugin_manager()
    plugin = DreamPlugin()
    plugin.workspace_root = None  # avoid Path resolution in on_start
    pm.register(plugin)

    # on_start will fail without a proper config — we only care it's called,
    # not that it succeeds. Catch any error from missing config but verify
    # dispatch happened by checking on_start is recognised as a hookimpl.
    try:
        await pm.ahook.on_start(config={})
    except (TypeError, KeyError, AttributeError, ValueError):
        pass  # Expected — on_start needs config; we only test dispatch registration

    # If @hookimpl is missing, the hook simply never called on_start on this plugin.
    # We verify dispatch by checking the hookimpl is registered.
    hook_caller = pm.hook.on_start
    impl_plugins = [impl.plugin for impl in hook_caller.get_hookimpls()]
    assert plugin in impl_plugins, (
        "DreamPlugin must be registered as an on_start hookimpl. "
        "Ensure @hookimpl is applied to DreamPlugin.on_start."
    )


async def test_dream_plugin_dispatched_via_pm_on_idle():
    """DreamPlugin.on_idle must be called when pm.ahook.on_idle is broadcast."""
    from corvidae.tools.dream import DreamPlugin

    pm = create_plugin_manager()
    plugin = DreamPlugin()
    plugin.workspace_root = None
    plugin._db_path = None
    plugin._last_dream_time = 0.0
    pm.register(plugin)

    hook_caller = pm.hook.on_idle
    impl_plugins = [impl.plugin for impl in hook_caller.get_hookimpls()]
    assert plugin in impl_plugins, (
        "DreamPlugin must be registered as an on_idle hookimpl. "
        "Ensure @hookimpl is applied to DreamPlugin.on_idle."
    )


def test_agent_on_start_does_not_have_hookimpl():
    """Agent.on_start must NOT be decorated with @hookimpl.
    It is called explicitly after pm.ahook.on_start() broadcast to
    ensure ToolCollectionPlugin.on_start (trylast=True) completes first.
    Adding @hookimpl would cause double initialization.
    """
    from corvidae.agent import Agent
    from corvidae.hooks import hookimpl

    marker_attr = hookimpl.project_name + "_impl"
    impl_opts = getattr(Agent.on_start, marker_attr, None)
    assert impl_opts is None, (
        "Agent.on_start must NOT have @hookimpl — it is called explicitly "
        "after the on_start broadcast to prevent double initialization"
    )
