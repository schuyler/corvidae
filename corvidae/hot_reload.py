"""HotReloadPlugin — runtime plugin hot-reload via the manage_plugins tool.

Provides the ability to reload, remove, and list plugins at runtime without
restarting the daemon. Maintains integrity by:
  - Protecting core plugins from reload/removal
  - Protecting plugins that other registered plugins declare as dependencies
  - Rolling back to the previous instance if a reload fails at any step

Fires on_plugin_added / on_plugin_removed broadcast hooks so other plugins
(e.g. ToolCollectionPlugin, Agent) can refresh their state.

Config: none.
"""
from __future__ import annotations

import importlib
import logging
import os

from corvidae.hooks import CorvidaePlugin, hookimpl

logger = logging.getLogger(__name__)

CORE_PLUGINS = frozenset({
    "agent",
    "llm",
    "tools",
    "registry",
    "task",
    "persistence",
    "hot_reload",
})


class HotReloadPlugin(CorvidaePlugin):
    """Plugin that owns the hot-reload lifecycle for all other plugins."""

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.config: dict = {}

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Update stored config so plugins reloaded after a config change receive current values."""
        self.config = config

    # ------------------------------------------------------------------
    # Public API used by tests and the manage_plugins tool
    # ------------------------------------------------------------------

    def _is_reloadable(self, name: str) -> bool:
        """Return True if the named plugin may be reloaded or removed.

        A plugin is non-reloadable if:
          - It is in CORE_PLUGINS, OR
          - Any other registered plugin declares it in its depends_on.
        """
        if name in CORE_PLUGINS:
            return False
        for plugin_name, plugin in self.pm.list_name_plugin():
            if plugin_name == name:
                continue
            dep_on = getattr(plugin, "depends_on", None)
            if dep_on and name in dep_on:
                return False
        return True

    async def reload_plugin(self, name: str) -> None:
        """Reload a registered plugin by re-importing its module.

        Sequence:
          1. Validate name is registered and reloadable.
          2. Capture old_instance, module path, class name.
          3. Call await old_instance.on_stop() directly.
          4. pm.unregister(old_instance).
          5. Broadcast on_plugin_removed.
          6. importlib.reload the module.
          7. Instantiate new_class(), call on_init and on_start directly.
          8. pm.register(new_instance, name=name).
          9. Broadcast on_plugin_added.

        On any failure in steps 6–9, re-register old_instance under the
        same name (no lifecycle re-call — its state is intact).

        Raises:
            ValueError: If name is not registered.
            RuntimeError: If the plugin is not reloadable (core or depended-upon).
        """
        old_instance = self.pm.get_plugin(name)
        if old_instance is None:
            raise ValueError(f"No plugin registered under name {name!r}")
        if not self._is_reloadable(name):
            raise RuntimeError(
                f"Plugin {name!r} is not reloadable "
                f"(core plugin or required by another plugin)"
            )

        # Capture metadata before unregistering
        module_path = type(old_instance).__module__
        class_name = type(old_instance).__name__

        # Step 3: stop old instance directly (not broadcast)
        on_stop = getattr(old_instance, "on_stop", None)
        if on_stop is not None:
            await on_stop()

        # Step 4: unregister old instance
        self.pm.unregister(old_instance)

        # Step 5: broadcast removal
        await self.pm.ahook.on_plugin_removed(name=name)

        # Steps 6–9: load new code; roll back on any failure
        try:
            mod = importlib.import_module(module_path)
            # Remove cached bytecode so reload reads fresh source.
            # Without this, reload() may serve stale .pyc when the
            # source file's mtime hasn't changed (sub-second writes).
            cached = getattr(mod, "__cached__", None)
            if cached:
                try:
                    os.remove(cached)
                except OSError:
                    pass
            importlib.invalidate_caches()
            mod = importlib.reload(mod)
            new_class = getattr(mod, class_name)
            new_instance = new_class()
            pm = self.pm
            await new_instance.on_init(pm=pm, config=self.config)
            await new_instance.on_start(config=self.config)
            pm.register(new_instance, name=name)
        except Exception:
            logger.exception(
                "reload_plugin: failed to load new version of %r; rolling back", name
            )
            # Re-register old instance — state is intact, skip lifecycle calls
            self.pm.register(old_instance, name=name)
            raise

        # Step 11: broadcast addition.  If a hookimpl raises, the new plugin
        # is already registered but dependent plugins may have stale state.
        # Log clearly and re-raise so the caller knows.
        try:
            await self.pm.ahook.on_plugin_added(name=name, plugin=new_instance)
        except Exception:
            logger.exception(
                "reload_plugin: on_plugin_added broadcast failed for %r; "
                "plugin is registered but dependent plugins may have stale state",
                name,
            )
            raise

    async def remove_plugin(self, name: str) -> None:
        """Unregister a plugin and broadcast on_plugin_removed.

        Raises:
            ValueError: If name is not registered.
            RuntimeError: If the plugin is not reloadable (core or depended-upon).
        """
        instance = self.pm.get_plugin(name)
        if instance is None:
            raise ValueError(f"No plugin registered under name {name!r}")
        if not self._is_reloadable(name):
            raise RuntimeError(
                f"Plugin {name!r} cannot be removed "
                f"(core plugin or required by another plugin)"
            )

        on_stop = getattr(instance, "on_stop", None)
        if on_stop is not None:
            await on_stop()

        self.pm.unregister(instance)
        await self.pm.ahook.on_plugin_removed(name=name)

    async def list_plugins(self) -> list[dict]:
        """Return a list of dicts describing each registered plugin.

        Each dict has:
            name: str — the registered plugin name
            plugin: object — the plugin instance
            reloadable: bool — whether the plugin may be reloaded/removed
        """
        result = []
        for plugin_name, plugin in self.pm.list_name_plugin():
            result.append({
                "name": plugin_name,
                "plugin": plugin,
                "reloadable": self._is_reloadable(plugin_name),
            })
        return result

    async def manage_plugins(self, action: str, name: str | None = None) -> str:
        """Manage runtime plugins. Actions: reload, remove, list.

        Args:
            action: One of 'reload', 'remove', 'list'.
            name: Plugin name (required for reload/remove).

        Returns:
            A human-readable status string.
        """
        if action == "list":
            listing = await self.list_plugins()
            lines = []
            for entry in listing:
                flag = "reloadable" if entry["reloadable"] else "core/protected"
                lines.append(f"  {entry['name']}: {flag}")
            return "Registered plugins:\n" + "\n".join(lines)
        elif action == "reload":
            if not name:
                return "Error: 'name' is required for action 'reload'"
            await self.reload_plugin(name)
            return f"Plugin {name!r} reloaded successfully"
        elif action == "remove":
            if not name:
                return "Error: 'name' is required for action 'remove'"
            await self.remove_plugin(name)
            return f"Plugin {name!r} removed successfully"
        else:
            return f"Error: unknown action {action!r}. Valid: reload, remove, list"

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Register the manage_plugins tool."""
        plugin = self

        async def manage_plugins(action: str, name: str | None = None) -> str:
            """Manage runtime plugins. Actions: reload, remove, list.

            Args:
                action: One of 'reload', 'remove', 'list'.
                name: Plugin name (required for reload/remove).

            Returns:
                A human-readable status string.
            """
            return await plugin.manage_plugins(action=action, name=name)

        tool_registry.append(manage_plugins)
