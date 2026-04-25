"""Daemon entry point for sherman.

This module configures logging from the YAML config file (or applies defaults)
before any other initialization. All sherman modules use module-level loggers
named `sherman.<module>` via `logging.getLogger(__name__)`. The `sherman` root
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

from sherman.agent import AgentPlugin
from sherman.channel import ChannelRegistry, load_channel_config
from sherman.channels.cli import CLIPlugin
from sherman.channels.irc import IRCPlugin
from sherman.hooks import create_plugin_manager
from sherman.logging import (  # noqa: F401 — re-exported for backward compat
    StructuredFormatter,
    _BUILTIN_LOG_ATTRS,
    _DEFAULT_LOGGING,
)
from sherman.tools import CoreToolsPlugin

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
    pm.load_setuptools_entrypoints("sherman")

    # Extract agent-level defaults for channel config resolution
    agent_defaults = config.get("agent", {})
    registry = ChannelRegistry(agent_defaults)

    # Option B injection: attach registry to PM so plugins access it
    # via self.pm.registry
    pm.registry = registry

    # Pre-register channels from YAML config (must happen before on_start)
    load_channel_config(config, registry)

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
    from sherman.task import TaskPlugin
    task_plugin = TaskPlugin(pm)
    pm.register(task_plugin, name="task")

    # Register SubagentPlugin after TaskPlugin, before AgentPlugin
    from sherman.tools.subagent import SubagentPlugin
    subagent_plugin = SubagentPlugin(pm)
    pm.register(subagent_plugin, name="subagent")

    # Register AgentPlugin after tool-providing and transport plugins
    agent_loop = AgentPlugin(pm)
    pm.register(agent_loop, name="agent_loop")
    pm.agent_plugin = agent_loop

    await pm.ahook.on_start(config=config)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()

    logger.info("shutdown signal received, stopping")

    await pm.ahook.on_stop()


def cli() -> None:
    """Console script entry point."""
    asyncio.run(main())
