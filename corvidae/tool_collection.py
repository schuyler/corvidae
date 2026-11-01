"""ToolCollectionPlugin — collects tools from all plugins at startup.

Calls the register_tools hook during on_start (broadcast phase), builds
a ToolRegistry, and exposes it as an immutable collection. Plugins that
need tool access declare depends_on = {"tools"}.

Config:
    tools:
      max_result_chars: 100000  # optional, default 100_000
      disabled: [shell]         # optional; tool names dropped after collection
                                # (shell and task_pipeline disable together)
      call_timeout: 600         # optional; seconds bound on every tool call
                                # (0 or null disables)
      timeouts: {task_pipeline: 3600}  # optional; per-tool overrides

    # Legacy (deprecated):
    agent:
      max_tool_result_chars: 100000  # migrated to tools.max_result_chars
"""
import json
import logging
import math
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tool import DEFAULT_TOOL_TIMEOUTS, Tool, ToolRegistry, ToolTimeouts
from corvidae.tool_policy import expand_disabled, parse_disabled_list

logger = logging.getLogger(__name__)


# Accepted-values text shared by every call_timeout / timeouts.<name> rejection.
_SECONDS_RULE = "a number of seconds >= 0 (0 or null disables the bound)"


def _parse_seconds(key: str, value: object) -> float | None:
    """Validate one timeout value; 0 and None both mean no bound (None)."""
    # null disables the bound, like 0.
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{key} must be {_SECONDS_RULE}; got {value!r} ({type(value).__name__})"
        )
    return None if value == 0 else float(value)


def parse_tool_timeouts(tools_config: object) -> ToolTimeouts:
    """Parse ``tools.call_timeout`` and ``tools.timeouts`` into a ToolTimeouts.

    Raises ValueError naming the key, the rejected value and the accepted
    values. An absent ``call_timeout`` gives the 600 s default.
    """
    if tools_config is None:
        return DEFAULT_TOOL_TIMEOUTS
    if not isinstance(tools_config, dict):
        raise ValueError(
            f"tools must be a mapping of tool settings (e.g. {{call_timeout: 600}}); "
            f"got {tools_config!r} ({type(tools_config).__name__})"
        )
    # call_timeout: absent keeps the default; null/0 disables.
    if "call_timeout" in tools_config:
        default = _parse_seconds("tools.call_timeout", tools_config["call_timeout"])
    else:
        default = DEFAULT_TOOL_TIMEOUTS.default
    # timeouts: a mapping of tool name -> seconds.
    raw = tools_config.get("timeouts")
    per_tool: dict[str, float | None] = {}
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError(
                "tools.timeouts must be a mapping of tool names to seconds "
                f"(e.g. {{task_pipeline: 3600}}); got {raw!r} ({type(raw).__name__})"
            )
        for name, value in raw.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"tools.timeouts has a non-string tool name {name!r} "
                    f"({type(name).__name__}); keys must be tool names (strings)"
                )
            per_tool[name] = _parse_seconds(f"tools.timeouts.{name}", value)
    return ToolTimeouts(default, per_tool)


class ToolCollectionPlugin(CorvidaePlugin):
    """Plugin that collects and owns the tool registry."""

    depends_on = frozenset()

    def __init__(self, pm=None):
        if pm is not None:
            self.pm = pm
        self.registry: ToolRegistry | None = None
        self.max_result_chars: int = 100_000
        # Per-call time bounds (tools.call_timeout / tools.timeouts).
        self.call_timeouts: ToolTimeouts = DEFAULT_TOOL_TIMEOUTS
        self._disabled: list[str] = []
        # Bumped by rebuild_registry whenever the tool set (names or
        # schemas) differs from the previous build, so a consumer such as
        # the Agent can tell cheaply whether its snapshot is stale (#109).
        self.generation: int = 0
        self._fingerprint: str | None = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        tools_config = config.get("tools", {})
        # Validated (a bare string would otherwise split into characters)
        # and widened to coupled tools: shell and task_pipeline go together.
        self._disabled = sorted(expand_disabled(
            parse_disabled_list(tools_config.get("disabled"), "tools.disabled")
        ))
        # Bounds on tool calls; Runtime validates first, so this cannot fail there.
        self.call_timeouts = parse_tool_timeouts(tools_config)
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
        # Warn (once each) about overrides naming no registered tool; an MCP
        # tool registers later, so the entry is kept.
        known = self.registry.as_dict() if self.registry is not None else {}
        for name in self.call_timeouts.per_tool:
            if name not in known:
                logger.warning(
                    "tools.timeouts.%s names no registered tool; keeping it "
                    "(an MCP tool registers when its server connects) \u2014 "
                    "check the spelling against the tool list",
                    name,
                )

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

        # tools.disabled is the single chokepoint every registered tool
        # passes through -- drop named tools here, not per-plugin opt-outs.
        if self._disabled:
            before = len(tool_registry)
            tool_registry = tool_registry.exclude(*self._disabled)
            dropped = before - len(tool_registry)
            if dropped:
                logger.info("Tools disabled by config: %d", dropped)

        self.registry = tool_registry
        logger.info("Tools collected: %d", len(tool_registry))

        # Advance the generation only when the names or schemas changed.
        fingerprint = json.dumps(
            tool_registry.schemas(), sort_keys=True, default=str
        )
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self.generation += 1

    def get_tools(self) -> tuple[dict, list[dict]]:
        """Return (tools_dict, tool_schemas) for the agent loop."""
        return self.registry.as_dict(), self.registry.schemas()

    def get_registry(self) -> ToolRegistry:
        """Return the full ToolRegistry for inspection/filtering."""
        return self.registry
