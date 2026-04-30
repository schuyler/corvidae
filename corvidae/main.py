"""Daemon entry point for corvidae.

This module configures logging from the YAML config file (or applies defaults)
before any other initialization. All corvidae modules use module-level loggers
named `corvidae.<module>` via `logging.getLogger(__name__)`. The `corvidae` root
logger controls the entire logging hierarchy.

Logging Configuration:
    The `logging` key in agent.yaml supplies simplified options (level, file)
    forwarded to `configure_logging()`. If omitted, defaults apply: INFO level,
    stderr in programmatic mode, file logging (corvidae.log) in CLI mode.

Shutdown:
    SIGINT/SIGTERM trigger graceful shutdown via `stop_event`. The shutdown
    signal is logged before plugins are stopped.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pluggy

import yaml

from corvidae.channel import ChannelRegistry, load_channel_config
from corvidae.hooks import create_plugin_manager, validate_dependencies
from corvidae.logging import StructuredFormatter  # noqa: F401 — re-exported
from corvidae.logging import configure_logging

if TYPE_CHECKING:
    from corvidae.agent import Agent

logger = logging.getLogger(__name__)


async def main(config_path: str = "agent.yaml", *, cli_mode: bool = False) -> None:
    """Daemon entry point.

    1. Load YAML config from config_path. Raises FileNotFoundError if missing.
    2. Configure logging from config section or defaults.
    3. Create plugin manager via create_plugin_manager().
    4. Construct ChannelRegistry, load entry-point plugins.
    5. Call await pm.ahook.on_init(pm=pm, config=config).
    6. Call await pm.ahook.on_start(config=config).
    7. Call await agent.on_start(config=config) explicitly (not via broadcast).
    8. Wait for SIGINT or SIGTERM.
    9. Call await pm.ahook.on_stop().
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Configure logging first — before any other work
    log_section = config.get("logging", {})
    log_level = log_section.get("level", "INFO")
    log_file = log_section.get("file")
    if log_file is None and cli_mode:
        log_file = "corvidae.log"
    configure_logging(level=log_level, file=log_file)
    logger.info(
        "logging configured",
        extra={"level": log_level, "file": log_file or "stderr"},
    )

    config["_base_dir"] = Path(config_path).parent

    pm = create_plugin_manager()

    # ChannelRegistry is not an entry-point plugin — construct and register explicitly.
    registry = ChannelRegistry()
    pm.register(registry, name="registry")
    registry.agent_defaults = config.get("agent", {})
    load_channel_config(config, registry)

    # Load all entry-point plugins. These are instantiated with no arguments.
    pm.load_setuptools_entrypoints("corvidae")

    validate_dependencies(pm)

    await pm.ahook.on_init(pm=pm, config=config)
    await pm.ahook.on_start(config=config)

    agent = pm.get_plugin("agent")
    if agent is None:
        raise RuntimeError("Agent plugin not registered — check entry points")
    await agent.on_start(config=config)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _shutdown_requested = False

    def _handle_stop_signal() -> None:
        nonlocal _shutdown_requested
        if _shutdown_requested:
            sys.stderr.write("second interrupt received, force-exiting\n")
            sys.stderr.flush()
            os._exit(1)
        _shutdown_requested = True
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_stop_signal)

    await stop_event.wait()

    logger.info("shutdown signal received, stopping")

    try:
        await asyncio.wait_for(
            _run_shutdown(agent, pm),
            timeout=3.0,
        )
    except asyncio.TimeoutError:
        logger.warning("graceful shutdown timed out after 3s, force-exiting")
        logging.shutdown()
        os._exit(1)


async def _run_shutdown(agent: object, pm: pluggy.PluginManager) -> None:
    """Run agent and plugin shutdown in order. Called under a timeout in main().

    On timeout cancellation, pm.ahook.on_stop() may not execute if
    agent.on_stop() is the one hanging. This is intentional — the process
    is force-exiting anyway.
    """
    await agent.on_stop()
    await pm.ahook.on_stop()


def cli() -> None:
    """Console script entry point.

    Dispatches to sub-commands before starting the daemon:
      corvidae scaffold <name>   — generate a new tool plugin package
    """
    if len(sys.argv) > 1 and sys.argv[1] == "scaffold":
        from corvidae.scaffold import scaffold_cli
        scaffold_cli(sys.argv[2:])
    else:
        asyncio.run(main(cli_mode=True))
