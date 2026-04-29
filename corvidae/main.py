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

from corvidae.agent import Agent
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

    # Register JsonlLogPlugin after PersistencePlugin (observes conversation events)
    from corvidae.jsonl_log import JsonlLogPlugin
    jsonl_log_plugin = JsonlLogPlugin(pm)
    pm.register(jsonl_log_plugin, name="jsonl_log")

    # Register CoreToolsPlugin before Agent so tools are collected during on_start
    core_tools = CoreToolsPlugin(pm)
    pm.register(core_tools, name="core_tools")

    # Register CLIPlugin before Agent (transport plugins first)
    cli_plugin = CLIPlugin(pm)
    pm.register(cli_plugin, name="cli")

    # Register IRCPlugin before Agent (transport plugins first)
    irc_plugin = IRCPlugin(pm)
    pm.register(irc_plugin, name="irc")

    # Register TaskPlugin before Agent (provides task queue)
    from corvidae.task import TaskPlugin
    task_plugin = TaskPlugin(pm)
    pm.register(task_plugin, name="task")

    # Register SubagentPlugin after TaskPlugin, before Agent
    from corvidae.tools.subagent import SubagentPlugin
    subagent_plugin = SubagentPlugin(pm)
    pm.register(subagent_plugin, name="subagent")

    # Register McpClientPlugin before Agent (provides MCP server tools)
    from corvidae.mcp_client import McpClientPlugin
    mcp_plugin = McpClientPlugin(pm)
    pm.register(mcp_plugin, name="mcp")

    # Register LLMPlugin before CompactionPlugin and Agent (owns LLM client lifecycle)
    from corvidae.llm_plugin import LLMPlugin
    llm_plugin = LLMPlugin(pm)
    pm.register(llm_plugin, name="llm")

    # Register CompactionPlugin before Agent (provides default compaction strategy)
    from corvidae.compaction import CompactionPlugin
    compaction_plugin = CompactionPlugin(pm)
    pm.register(compaction_plugin, name="compaction")

    # Register ThinkingPlugin before Agent (handles <think> block stripping)
    from corvidae.thinking import ThinkingPlugin
    thinking_plugin = ThinkingPlugin(pm)
    pm.register(thinking_plugin, name="thinking")

    # Disabled — ContextCompactPlugin fights with CompactionPlugin over the same
    # conversation. Re-enable when the two are coordinated or merged.
    # from corvidae.context_compact import ContextCompactPlugin
    # context_compact_plugin = ContextCompactPlugin(pm)
    # pm.register(context_compact_plugin, name="context_compact")

    # Register RuntimeSettingsPlugin before Agent (provides set_settings tool)
    from corvidae.tools.settings import RuntimeSettingsPlugin
    immutable_settings = set(agent_defaults.get("immutable_settings", []))
    runtime_settings_plugin = RuntimeSettingsPlugin(pm, immutable_settings=immutable_settings)
    pm.register(runtime_settings_plugin, name="runtime_settings")

    # Disabled per operator request — re-enable when ready.
    # from corvidae.tools.index import WorkspaceIndexerPlugin
    # local_indexer_plugin = WorkspaceIndexerPlugin(pm)
    # pm.register(local_indexer_plugin, name="local_indexer")

   # Register ToolCollectionPlugin after all tool-providing plugins (its on_start
    # uses trylast=True so it fires after all other on_start hooks have run).
    from corvidae.tool_collection import ToolCollectionPlugin
    tool_collection_plugin = ToolCollectionPlugin(pm)
    pm.register(tool_collection_plugin, name="tools")

    # Register DreamPlugin for background memory consolidation
    from corvidae.tools.dream import DreamPlugin
    workspace_root = Path(config_path).parent
    dream_plugin = DreamPlugin(workspace_root=workspace_root)
    pm.register(dream_plugin, name="dream")

    # Register Agent after tool-providing and transport plugins.
    # Agent.on_start/on_stop are called explicitly (not via broadcast)
    # so that on_start runs after all plugins are ready and on_stop runs
    # before other plugins tear down resources.
    agent = Agent(pm)
    pm.register(agent, name="agent")

    # Register IdleMonitorPlugin after Agent (depends on agent)
    from corvidae.idle import IdleMonitorPlugin
    idle_monitor_plugin = IdleMonitorPlugin(pm)
    pm.register(idle_monitor_plugin, name="idle_monitor")

    validate_dependencies(pm)

    await pm.ahook.on_start(config=config)
    await agent.on_start(config=config)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()

    logger.info("shutdown signal received, stopping")

    await agent.on_stop()
    await pm.ahook.on_stop()


def cli() -> None:
    """Console script entry point."""
    asyncio.run(main())
