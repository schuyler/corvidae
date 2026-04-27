"""Daemon entry point for corvidae.

This module configures logging from the YAML config file (or applies defaults)
before any other initialization. All corvidae modules use module-level loggers
named `corvidae.<module>` via `logging.getLogger(__name__)`. The `corvidae` root
logger controls the entire logging hierarchy.

Logging Configuration:
    The `logging` key in agent.yaml is passed to `logging.config.dictConfig()`.
    If omitted, built-in defaults apply: INFO level to stderr, standard format.
    See `_DEFAULT_LOGGING` for the default configuration schema.

Shutdown:
    SIGINT/SIGTERM trigger graceful shutdown via `stop_event`. The shutdown
    signal is logged before plugins are stopped.
"""

import asyncio
import logging
import logging.config
import signal
from pathlib import Path

import yaml

from corvidae.agent import AgentPlugin
from corvidae.channel import ChannelRegistry, load_channel_config
from corvidae.channels.cli import CLIPlugin
from corvidae.channels.irc import IRCPlugin
from corvidae.hooks import create_plugin_manager, validate_dependencies
from corvidae.logging import (  # noqa: F401 — re-exported for backward compat
    StructuredFormatter,
    _BUILTIN_LOG_ATTRS,
    _DEFAULT_LOGGING,
)
from corvidae.tools import CoreToolsPlugin

logger = logging.getLogger(__name__)


async def main(config_path: str = "agent.yaml") -> None:
    """Daemon entry point.

    1. Load YAML config from config_path. Raises FileNotFoundError if missing.
    2. Configure logging from config section or defaults.
    3. Create plugin manager via create_plugin_manager().
    4. Call await pm.ahook.on_start(config=config).
    5. Wait for SIGINT or SIGTERM.
    6. Call await pm.ahook.on_stop().
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Configure logging first — before any other work
    log_config = config.get("logging", _DEFAULT_LOGGING)
    logging.config.dictConfig(log_config)
    logger.info(
        "logging configured",
        extra={"source": "yaml" if "logging" in config else "defaults"},
    )

    config["_base_dir"] = Path(config_path).parent

    pm = create_plugin_manager()
    pm.load_setuptools_entrypoints("corvidae")

    # Extract agent-level defaults for channel config resolution
    agent_defaults = config.get("agent", {})
    registry = ChannelRegistry(agent_defaults)

    # Register registry as a named plugin so plugins access it
    # via pm.get_plugin("registry") / get_dependency()
    pm.register(registry, name="registry")

    # Pre-register channels from YAML config (must happen before on_start)
    load_channel_config(config, registry)

    # Register PersistencePlugin after registry, before other plugins
    from corvidae.persistence import PersistencePlugin
    persistence_plugin = PersistencePlugin(pm)
    pm.register(persistence_plugin, name="persistence")

    # Register CoreToolsPlugin before AgentPlugin so tools are collected during on_start
    core_tools = CoreToolsPlugin()
    pm.register(core_tools, name="core_tools")

    # Register CLIPlugin before AgentPlugin (transport plugins first)
    cli_plugin = CLIPlugin(pm)
    pm.register(cli_plugin, name="cli")

    # Register IRCPlugin before AgentPlugin (transport plugins first)
    irc_plugin = IRCPlugin(pm)
    pm.register(irc_plugin, name="irc")

    # Register TaskPlugin before AgentPlugin (provides task queue)
    from corvidae.task import TaskPlugin
    task_plugin = TaskPlugin(pm)
    pm.register(task_plugin, name="task")

    # Register SubagentPlugin after TaskPlugin, before AgentPlugin
    from corvidae.tools.subagent import SubagentPlugin
    subagent_plugin = SubagentPlugin(pm)
    pm.register(subagent_plugin, name="subagent")

    # Register McpClientPlugin before AgentPlugin (provides MCP server tools)
    from corvidae.mcp_client import McpClientPlugin
    mcp_plugin = McpClientPlugin()
    pm.register(mcp_plugin, name="mcp")

    # Register CompactionPlugin before AgentPlugin (provides default compaction strategy)
    from corvidae.compaction import CompactionPlugin
    compaction_plugin = CompactionPlugin()
    pm.register(compaction_plugin, name="compaction")

    # Register ThinkingPlugin before AgentPlugin (handles <think> block stripping)
    from corvidae.thinking import ThinkingPlugin
    thinking_plugin = ThinkingPlugin(pm)
    pm.register(thinking_plugin, name="thinking")

    # Register RuntimeSettingsPlugin before AgentPlugin (provides set_settings tool)
    from corvidae.tools.settings import RuntimeSettingsPlugin
    immutable_settings = set(agent_defaults.get("immutable_settings", []))
    runtime_settings_plugin = RuntimeSettingsPlugin(immutable_settings=immutable_settings)
    pm.register(runtime_settings_plugin, name="runtime_settings")

    # Register AgentPlugin after tool-providing and transport plugins.
    # AgentPlugin.on_start/on_stop are called explicitly (not via broadcast)
    # so that on_start runs after all plugins are ready and on_stop runs
    # before other plugins tear down resources.
    agent_loop = AgentPlugin(pm)
    pm.register(agent_loop, name="agent_loop")

    # Register IdleMonitorPlugin after AgentPlugin (depends on agent_loop)
    from corvidae.idle import IdleMonitorPlugin
    idle_monitor_plugin = IdleMonitorPlugin(pm)
    pm.register(idle_monitor_plugin, name="idle_monitor")

    validate_dependencies(pm)

    await pm.ahook.on_start(config=config)
    await agent_loop.on_start(config=config)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()

    logger.info("shutdown signal received, stopping")

    await agent_loop.on_stop()
    await pm.ahook.on_stop()


def cli() -> None:
    """Console script entry point."""
    asyncio.run(main())
