"""ToolCollectionPlugin — collects tools from all plugins at startup.

Calls the register_tools hook during on_start (broadcast phase), builds
a ToolRegistry, and exposes it as an immutable collection. Plugins that
need tool access declare depends_on = {"tools"}.

Config:
    tools:
      max_result_chars: 100000  # optional, default 100_000

    # Legacy (deprecated):
    agent:
      max_tool_result_chars: 100000  # migrated to tools.max_result_chars
"""
import logging
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tool import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class ToolCollectionPlugin(CorvidaePlugin):
    """Plugin that collects and owns the tool registry."""

    depends_on = frozenset()

    def __init__(self, pm=None):
        if pm is not None:
            self.pm = pm
        self.registry: ToolRegistry | None = None
        self.max_result_chars: int = 100_000

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        tools_config = config.get("tools", {})
        fallback = config.get("agent", {}).get("max_tool_result_chars")
        if "max_result_chars" in tools_config:
            self.max_result_chars = tools_config["max_result_chars"]
        elif fallback is not None:
            logger.warning(
                "Config key agent.max_tool_result_chars is deprecated; "
                "use tools.max_result_chars instead"
            )
            self.max_result_chars = fallback
        else:
            self.max_result_chars = 100_000

    @hookimpl(trylast=True)
    async def on_start(self, config: dict) -> None:
        await self.rebuild_registry()

    async def rebuild_registry(self) -> None:
        """Collect tools from all plugins and rebuild the tool registry."""
        # Collect tools from all plugins via register_tools hook (sync).
        collected: list = []
        self.pm.hook.register_tools(tool_registry=collected)

        # Build registry
        tool_registry = ToolRegistry()
        for item in collected:
            if isinstance(item, Tool):
                tool_registry.add(item)
            else:
                tool_registry.add(Tool.from_function(item))

        self.registry = tool_registry
        logger.info("Tools collected: %d", len(tool_registry))

    @hookimpl
    async def on_plugin_added(self, name: str, plugin: object) -> None:
        """Rebuild the tool registry when a plugin is added at runtime."""
        await self.rebuild_registry()

    @hookimpl
    async def on_plugin_removed(self, name: str) -> None:
        """Rebuild the tool registry when a plugin is removed at runtime."""
        await self.rebuild_registry()

    def get_tools(self) -> tuple[dict, list[dict]]:
        """Return (tools_dict, tool_schemas) for the agent loop."""
        return self.registry.as_dict(), self.registry.schemas()

    def get_registry(self) -> ToolRegistry:
        """Return the full ToolRegistry for inspection/filtering."""
        return self.registry
