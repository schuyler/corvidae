"""
Tests for the hot-reload plugin system (Red phase of TDD).

All tests in this file are expected to FAIL against the current codebase
since corvidae/hot_reload.py does not exist yet. They will pass after the
implementation described in hot_reload_plan.md is applied.

Tests verify:
  - New hookspecs on AgentSpec (on_plugin_added, on_plugin_removed)
  - New methods on existing plugins (rebuild_registry, refresh_tools)
  - HotReloadPlugin behaviour: reload, remove, list, rollback, protection
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.hooks import AgentSpec, CorvidaePlugin, create_plugin_manager, hookimpl


# ---------------------------------------------------------------------------
# 1. New hookspecs exist on AgentSpec
# ---------------------------------------------------------------------------


def test_hookspecs_exist():
    """on_plugin_added and on_plugin_removed must be defined on AgentSpec."""
    assert hasattr(AgentSpec, "on_plugin_added"), (
        "AgentSpec must have an on_plugin_added hookspec"
    )
    assert hasattr(AgentSpec, "on_plugin_removed"), (
        "AgentSpec must have an on_plugin_removed hookspec"
    )


# ---------------------------------------------------------------------------
# 2. rebuild_registry exists on ToolCollectionPlugin
# ---------------------------------------------------------------------------


def test_tool_collection_rebuild_registry_exists():
    """ToolCollectionPlugin must have a rebuild_registry() method."""
    from corvidae.tool_collection import ToolCollectionPlugin

    assert hasattr(ToolCollectionPlugin, "rebuild_registry"), (
        "ToolCollectionPlugin must have a rebuild_registry() method"
    )
    assert callable(ToolCollectionPlugin.rebuild_registry), (
        "ToolCollectionPlugin.rebuild_registry must be callable"
    )


# ---------------------------------------------------------------------------
# 3. refresh_tools exists on Agent
# ---------------------------------------------------------------------------


def test_agent_refresh_tools_exists():
    """Agent must have a refresh_tools() method."""
    from corvidae.agent import Agent

    assert hasattr(Agent, "refresh_tools"), (
        "Agent must have a refresh_tools() method"
    )
    assert callable(Agent.refresh_tools), (
        "Agent.refresh_tools must be callable"
    )


# ---------------------------------------------------------------------------
# 4. HotReloadPlugin can be imported and has expected interface
# ---------------------------------------------------------------------------


def test_hot_reload_plugin_importable():
    """corvidae.hot_reload.HotReloadPlugin must be importable."""
    from corvidae.hot_reload import HotReloadPlugin  # noqa: F401


def test_hot_reload_plugin_has_manage_plugins_tool():
    """HotReloadPlugin must have a manage_plugins tool or method."""
    from corvidae.hot_reload import HotReloadPlugin

    assert hasattr(HotReloadPlugin, "manage_plugins") or hasattr(
        HotReloadPlugin, "_manage_plugins"
    ), "HotReloadPlugin must expose a manage_plugins tool/method"


def test_hot_reload_plugin_has_core_plugins_constant():
    """corvidae.hot_reload must define CORE_PLUGINS as a frozenset."""
    from corvidae.hot_reload import CORE_PLUGINS

    assert isinstance(CORE_PLUGINS, frozenset), "CORE_PLUGINS must be a frozenset"
    # Core plugin names the design specifies
    for name in ("agent", "llm", "tools", "registry", "task", "hot_reload"):
        assert name in CORE_PLUGINS, f"{name!r} must be in CORE_PLUGINS"


# ---------------------------------------------------------------------------
# Helper: build a minimal plugin manager with the new hookspecs loaded
# ---------------------------------------------------------------------------


def _make_pm():
    """Return a PluginManager with AgentSpec hookspecs registered."""
    return create_plugin_manager()


# ---------------------------------------------------------------------------
# 5. test_core_plugin_protection
# ---------------------------------------------------------------------------


async def test_core_plugin_protection():
    """Reloading a core plugin (e.g. 'agent') must be rejected with an error."""
    from corvidae.hot_reload import HotReloadPlugin

    pm = _make_pm()
    plugin = HotReloadPlugin(pm)
    pm.register(plugin, name="hot_reload")
    await plugin.on_init(pm=pm, config={})

    # Register a mock core plugin so pm.get_plugin("agent") returns something
    mock_agent = MagicMock()
    pm.register(mock_agent, name="agent")

    with pytest.raises((RuntimeError, ValueError)):
        await plugin.reload_plugin("agent")


# ---------------------------------------------------------------------------
# 6. test_dependency_protection
# ---------------------------------------------------------------------------


async def test_dependency_protection():
    """A plugin that another registered plugin depends on must not be reloadable."""
    from corvidae.hot_reload import HotReloadPlugin

    pm = _make_pm()
    plugin = HotReloadPlugin(pm)
    pm.register(plugin, name="hot_reload")
    await plugin.on_init(pm=pm, config={})

    # Plugin A is a leaf; plugin B declares it in depends_on
    class PluginA(CorvidaePlugin):
        depends_on = frozenset()

    class PluginB(CorvidaePlugin):
        depends_on = frozenset({"plugin_a"})

    pm.register(PluginA(), name="plugin_a")
    pm.register(PluginB(), name="plugin_b")

    with pytest.raises((RuntimeError, ValueError)):
        await plugin.reload_plugin("plugin_a")


# ---------------------------------------------------------------------------
# 7. test_reload_mock_plugin — real module on disk, reload picks up new code
# ---------------------------------------------------------------------------


async def test_reload_mock_plugin(tmp_path):
    """Reload a leaf plugin loaded from disk; verify new code takes effect."""
    from corvidae.hot_reload import HotReloadPlugin

    # Write version 1 of the plugin module
    mod_dir = tmp_path / "reload_test_pkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    plugin_file = mod_dir / "leaf_plugin.py"
    plugin_file.write_text(
        textwrap.dedent("""\
        class LeafPlugin:
            depends_on = frozenset()
            version = 1

            def __init__(self, pm=None):
                self.pm = pm

            async def on_init(self, pm, config):
                self.pm = pm

            async def on_start(self, config):
                pass

            async def on_stop(self):
                pass
        """)
    )

    # Add tmp_path to sys.path so we can import the module
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("reload_test_pkg.leaf_plugin")
        leaf_instance = mod.LeafPlugin()
        assert leaf_instance.version == 1

        pm = _make_pm()
        hot_reload = HotReloadPlugin(pm)
        pm.register(hot_reload, name="hot_reload")
        await hot_reload.on_init(pm=pm, config={})

        pm.register(leaf_instance, name="leaf")

        # Write version 2 of the plugin module
        plugin_file.write_text(
            textwrap.dedent("""\
            class LeafPlugin:
                depends_on = frozenset()
                version = 2

                def __init__(self, pm=None):
                    self.pm = pm

                async def on_init(self, pm, config):
                    self.pm = pm

                async def on_start(self, config):
                    pass

                async def on_stop(self):
                    pass
            """)
        )

        await hot_reload.reload_plugin("leaf")

        # After reload, the registered instance should have version == 2
        new_instance = pm.get_plugin("leaf")
        assert new_instance is not None
        assert new_instance.version == 2
    finally:
        sys.path.remove(str(tmp_path))
        # Clean up any cached modules from this test
        for key in list(sys.modules.keys()):
            if "reload_test_pkg" in key:
                del sys.modules[key]


# ---------------------------------------------------------------------------
# 8. test_rollback_on_import_error
# ---------------------------------------------------------------------------


async def test_rollback_on_import_error(tmp_path):
    """If reload fails at the import step, the old plugin must be restored."""
    from corvidae.hot_reload import HotReloadPlugin

    mod_dir = tmp_path / "rollback_pkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    plugin_file = mod_dir / "rollback_plugin.py"
    plugin_file.write_text(
        textwrap.dedent("""\
        class RollbackPlugin:
            depends_on = frozenset()
            version = "original"

            def __init__(self, pm=None):
                self.pm = pm

            async def on_init(self, pm, config):
                self.pm = pm

            async def on_start(self, config):
                pass

            async def on_stop(self):
                pass
        """)
    )

    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("rollback_pkg.rollback_plugin")
        original_instance = mod.RollbackPlugin()

        pm = _make_pm()
        hot_reload = HotReloadPlugin(pm)
        pm.register(hot_reload, name="hot_reload")
        await hot_reload.on_init(pm=pm, config={})
        pm.register(original_instance, name="rollback_leaf")

        # Corrupt the module file so the reload fails
        plugin_file.write_text("this is not valid python !!!###")

        with pytest.raises(Exception):
            await hot_reload.reload_plugin("rollback_leaf")

        # Old plugin must still be registered under the same name
        restored = pm.get_plugin("rollback_leaf")
        assert restored is original_instance, (
            "Original plugin must be restored after failed reload"
        )
    finally:
        sys.path.remove(str(tmp_path))
        for key in list(sys.modules.keys()):
            if "rollback_pkg" in key:
                del sys.modules[key]


# ---------------------------------------------------------------------------
# 9. test_rollback_on_init_error
# ---------------------------------------------------------------------------


async def test_rollback_on_init_error(tmp_path):
    """If new plugin's on_init raises, the old plugin must be restored."""
    from corvidae.hot_reload import HotReloadPlugin

    mod_dir = tmp_path / "init_err_pkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    plugin_file = mod_dir / "init_err_plugin.py"
    plugin_file.write_text(
        textwrap.dedent("""\
        class InitErrPlugin:
            depends_on = frozenset()
            version = "original"

            def __init__(self, pm=None):
                self.pm = pm

            async def on_init(self, pm, config):
                self.pm = pm

            async def on_start(self, config):
                pass

            async def on_stop(self):
                pass
        """)
    )

    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("init_err_pkg.init_err_plugin")
        original_instance = mod.InitErrPlugin()

        pm = _make_pm()
        hot_reload = HotReloadPlugin(pm)
        pm.register(hot_reload, name="hot_reload")
        await hot_reload.on_init(pm=pm, config={})
        pm.register(original_instance, name="init_err_leaf")

        # Write a new version whose on_init raises
        plugin_file.write_text(
            textwrap.dedent("""\
            class InitErrPlugin:
                depends_on = frozenset()
                version = "new"

                def __init__(self, pm=None):
                    self.pm = pm

                async def on_init(self, pm, config):
                    raise RuntimeError("on_init intentionally failed")

                async def on_start(self, config):
                    pass

                async def on_stop(self):
                    pass
            """)
        )

        with pytest.raises(Exception):
            await hot_reload.reload_plugin("init_err_leaf")

        restored = pm.get_plugin("init_err_leaf")
        assert restored is original_instance, (
            "Original plugin must be restored after on_init failure"
        )
    finally:
        sys.path.remove(str(tmp_path))
        for key in list(sys.modules.keys()):
            if "init_err_pkg" in key:
                del sys.modules[key]


# ---------------------------------------------------------------------------
# 10. test_hooks_fire_on_reload
# ---------------------------------------------------------------------------


async def test_hooks_fire_on_reload(tmp_path):
    """on_plugin_removed fires before re-register; on_plugin_added fires after."""
    from corvidae.hot_reload import HotReloadPlugin

    mod_dir = tmp_path / "hooks_fire_pkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    plugin_file = mod_dir / "hooks_fire_plugin.py"
    plugin_file.write_text(
        textwrap.dedent("""\
        class HooksFirePlugin:
            depends_on = frozenset()

            def __init__(self, pm=None):
                self.pm = pm

            async def on_init(self, pm, config):
                self.pm = pm

            async def on_start(self, config):
                pass

            async def on_stop(self):
                pass
        """)
    )

    events: list[str] = []

    class ObserverPlugin:
        @hookimpl
        async def on_plugin_removed(self, name: str) -> None:
            events.append(f"removed:{name}")

        @hookimpl
        async def on_plugin_added(self, name: str, plugin: object) -> None:
            events.append(f"added:{name}")

    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("hooks_fire_pkg.hooks_fire_plugin")
        instance = mod.HooksFirePlugin()

        pm = _make_pm()
        observer = ObserverPlugin()
        pm.register(observer, name="observer")

        hot_reload = HotReloadPlugin(pm)
        pm.register(hot_reload, name="hot_reload")
        await hot_reload.on_init(pm=pm, config={})
        pm.register(instance, name="hooks_fire_leaf")

        # Write version 2 of the module so reload has new code to pick up
        plugin_file.write_text(
            textwrap.dedent("""\
            class HooksFirePlugin:
                depends_on = frozenset()
                version = 2

                def __init__(self, pm=None):
                    self.pm = pm

                async def on_init(self, pm, config):
                    self.pm = pm

                async def on_start(self, config):
                    pass

                async def on_stop(self):
                    pass
            """)
        )

        await hot_reload.reload_plugin("hooks_fire_leaf")

        assert "removed:hooks_fire_leaf" in events, (
            "on_plugin_removed must fire during reload"
        )
        assert "added:hooks_fire_leaf" in events, (
            "on_plugin_added must fire after reload"
        )
        # removed must come before added
        removed_idx = events.index("removed:hooks_fire_leaf")
        added_idx = events.index("added:hooks_fire_leaf")
        assert removed_idx < added_idx, (
            "on_plugin_removed must fire before on_plugin_added"
        )
    finally:
        sys.path.remove(str(tmp_path))
        for key in list(sys.modules.keys()):
            if "hooks_fire_pkg" in key:
                del sys.modules[key]


# ---------------------------------------------------------------------------
# 11. test_tool_recollection_after_reload
# ---------------------------------------------------------------------------


async def test_tool_recollection_after_reload(tmp_path):
    """ToolCollectionPlugin.rebuild_registry is invoked when a plugin is reloaded."""
    from corvidae.tool_collection import ToolCollectionPlugin
    from corvidae.hot_reload import HotReloadPlugin

    # Write a real leaf plugin module on disk (version 1)
    mod_dir = tmp_path / "recollect_pkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    plugin_file = mod_dir / "leaf_plugin.py"
    plugin_file.write_text(
        textwrap.dedent("""\
        class LeafPlugin:
            depends_on = frozenset()
            version = 1

            def __init__(self, pm=None):
                self.pm = pm

            async def on_init(self, pm, config):
                self.pm = pm

            async def on_start(self, config):
                pass

            async def on_stop(self):
                pass
        """)
    )

    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("recollect_pkg.leaf_plugin")
        leaf_instance = mod.LeafPlugin()

        pm = _make_pm()

        # Set up ToolCollectionPlugin with a mocked rebuild_registry
        tools_plugin = ToolCollectionPlugin(pm)
        tools_plugin.rebuild_registry = AsyncMock()
        pm.register(tools_plugin, name="tools")

        hot_reload = HotReloadPlugin(pm)
        pm.register(hot_reload, name="hot_reload")
        await hot_reload.on_init(pm=pm, config={})

        pm.register(leaf_instance, name="leaf_name")

        # Write version 2 of the module so reload picks up new code
        plugin_file.write_text(
            textwrap.dedent("""\
            class LeafPlugin:
                depends_on = frozenset()
                version = 2

                def __init__(self, pm=None):
                    self.pm = pm

                async def on_init(self, pm, config):
                    self.pm = pm

                async def on_start(self, config):
                    pass

                async def on_stop(self):
                    pass
            """)
        )

        await hot_reload.reload_plugin("leaf_name")

        # ToolCollectionPlugin.rebuild_registry must have been called after reload
        tools_plugin.rebuild_registry.assert_awaited()
    finally:
        sys.path.remove(str(tmp_path))
        for key in list(sys.modules.keys()):
            if "recollect_pkg" in key:
                del sys.modules[key]


# ---------------------------------------------------------------------------
# 12. test_remove_plugin
# ---------------------------------------------------------------------------


async def test_remove_plugin():
    """remove_plugin unregisters the plugin and fires on_plugin_removed.

    Also verifies that ToolCollectionPlugin.rebuild_registry is awaited after
    removal — this relies on ToolCollectionPlugin having an on_plugin_removed
    hookimpl that calls rebuild_registry (added in Green phase).
    """
    from corvidae.hot_reload import HotReloadPlugin
    from corvidae.tool_collection import ToolCollectionPlugin

    events: list[str] = []

    class ObserverPlugin:
        @hookimpl
        async def on_plugin_removed(self, name: str) -> None:
            events.append(f"removed:{name}")

    pm = _make_pm()
    observer = ObserverPlugin()
    pm.register(observer, name="observer")

    # Register ToolCollectionPlugin with rebuild_registry mocked so we can
    # assert it is called when on_plugin_removed fires.
    tools_plugin = ToolCollectionPlugin(pm)
    tools_plugin.rebuild_registry = AsyncMock()
    pm.register(tools_plugin, name="tools")

    hot_reload = HotReloadPlugin(pm)
    pm.register(hot_reload, name="hot_reload")
    await hot_reload.on_init(pm=pm, config={})

    class LeafPlugin(CorvidaePlugin):
        depends_on = frozenset()

        async def on_stop(self):
            pass

    leaf = LeafPlugin()
    pm.register(leaf, name="removable_leaf")

    await hot_reload.remove_plugin("removable_leaf")

    assert pm.get_plugin("removable_leaf") is None, (
        "Plugin must be unregistered after remove_plugin"
    )
    assert "removed:removable_leaf" in events, (
        "on_plugin_removed must fire after remove_plugin"
    )
    # ToolCollectionPlugin.rebuild_registry must be awaited after removal so the
    # tool registry no longer includes tools from the removed plugin.
    tools_plugin.rebuild_registry.assert_awaited()


# ---------------------------------------------------------------------------
# 13. test_list_plugins
# ---------------------------------------------------------------------------


async def test_list_plugins():
    """list_plugins returns correct reloadability annotations."""
    from corvidae.hot_reload import HotReloadPlugin

    pm = _make_pm()
    hot_reload = HotReloadPlugin(pm)
    pm.register(hot_reload, name="hot_reload")
    await hot_reload.on_init(pm=pm, config={})

    # Register a mock core plugin and a leaf plugin
    class MockAgentPlugin(CorvidaePlugin):
        depends_on = frozenset()

    class LeafPlugin(CorvidaePlugin):
        depends_on = frozenset()

    pm.register(MockAgentPlugin(), name="agent")
    pm.register(LeafPlugin(), name="leaf_x")

    listing = await hot_reload.list_plugins()

    # Should return a list of dicts or similar with name and reloadable flag
    assert listing is not None, "list_plugins must return a non-None result"
    names = {entry["name"] for entry in listing}
    assert "hot_reload" in names
    assert "agent" in names
    assert "leaf_x" in names

    # hot_reload and agent are core — not reloadable
    by_name = {entry["name"]: entry for entry in listing}
    assert by_name["agent"]["reloadable"] is False, (
        "Core plugin 'agent' must be marked not reloadable"
    )
    assert by_name["hot_reload"]["reloadable"] is False, (
        "Core plugin 'hot_reload' must be marked not reloadable"
    )
    assert by_name["leaf_x"]["reloadable"] is True, (
        "Leaf plugin 'leaf_x' must be marked reloadable"
    )
