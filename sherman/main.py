"""Daemon entry point for sherman."""

import asyncio
import signal
from pathlib import Path

import yaml

from sherman.agent_loop_plugin import AgentLoopPlugin
from sherman.channel import ChannelRegistry, load_channel_config
from sherman.cli_plugin import CLIPlugin
from sherman.plugin_manager import create_plugin_manager
from sherman.tools import CoreToolsPlugin


async def main(config_path: str = "agent.yaml") -> None:
    """Daemon entry point.

    1. Load YAML config from config_path. Raises FileNotFoundError if missing.
    2. Create plugin manager via create_plugin_manager().
    3. Call await pm.ahook.on_start(config=config).
    4. Wait for SIGINT or SIGTERM.
    5. Call await pm.ahook.on_stop().
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

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

    # Register AgentLoopPlugin after tool-providing and transport plugins
    agent_loop = AgentLoopPlugin(pm)
    pm.register(agent_loop, name="agent_loop")

    await pm.ahook.on_start(config=config)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()

    await pm.ahook.on_stop()


def cli() -> None:
    """Console script entry point."""
    asyncio.run(main())
