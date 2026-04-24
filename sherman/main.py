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

from sherman.agent_loop_plugin import AgentLoopPlugin
from sherman.channel import ChannelRegistry, load_channel_config
from sherman.cli_plugin import CLIPlugin
from sherman.irc_plugin import IRCPlugin
from sherman.plugin_manager import create_plugin_manager
from sherman.tools import CoreToolsPlugin

logger = logging.getLogger(__name__)

# Default logging configuration applied when no `logging` section exists in
# agent.yaml. Passed directly to logging.config.dictConfig(). Key choices:
#   - Output to stderr (stdout reserved for structured output if needed)
#   - Sherman loggers at INFO (production operational level)
#   - Root logger at WARNING (suppresses noisy library debug output)
#   - disable_existing_loggers: False preserves loggers created at import time
#   - propagate: False on sherman logger prevents double-output through root
#
# Users can override by providing a `logging` section in agent.yaml that
# follows the same schema (log levels, file handlers, JSON formatters, etc.).
_DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "sherman": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}


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

    # Extract agent-level defaults for channel config resolution
    agent_defaults = config.get("agent", {})
    registry = ChannelRegistry(agent_defaults)

    # Option B injection: attach registry to PM so plugins access it
    # via self.pm.registry
    pm.registry = registry

    # Pre-register channels from YAML config (must happen before on_start)
    load_channel_config(config, registry)

    # Register CoreToolsPlugin before AgentLoopPlugin so tools are collected during on_start
    core_tools = CoreToolsPlugin()
    pm.register(core_tools, name="core_tools")

    # Register CLIPlugin before AgentLoopPlugin (transport plugins first)
    cli_plugin = CLIPlugin(pm)
    pm.register(cli_plugin, name="cli")

    # Register IRCPlugin before AgentLoopPlugin (transport plugins first)
    irc_plugin = IRCPlugin(pm)
    pm.register(irc_plugin, name="irc")

    # Register AgentLoopPlugin after tool-providing and transport plugins
    agent_loop = AgentLoopPlugin(pm)
    pm.register(agent_loop, name="agent_loop")

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
