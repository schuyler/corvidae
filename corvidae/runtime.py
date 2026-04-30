"""Agent lifecycle management for corvidae subcommands.

The Runtime class extracts the agent startup/shutdown lifecycle from the
legacy main() function into a reusable class that any subcommand can
import and use.

Shutdown:
    SIGINT/SIGTERM trigger graceful shutdown via stop_event. A second signal
    during shutdown triggers os._exit(1) immediately. Graceful shutdown runs
    under a 3-second timeout; os._exit(1) on timeout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import yaml

from corvidae.channel import ChannelRegistry, load_channel_config
from corvidae.hooks import create_plugin_manager, validate_dependencies
from corvidae.logging import configure_logging

logger = logging.getLogger(__name__)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides on top of base, returning a new dict.

    Merge rules:
    - Dicts merge recursively.
    - Non-dict values replace the base value.
    - None values in overrides are skipped (base value preserved).
    - Top-level keys absent from base are added.
    - The base argument is not mutated.
    """
    result = dict(base)
    for key, override_val in overrides.items():
        if override_val is None:
            # None is a no-op: preserve whatever base has (or its absence).
            continue
        base_val = result.get(key)
        if isinstance(base_val, dict) and isinstance(override_val, dict):
            result[key] = deep_merge(base_val, override_val)
        else:
            result[key] = override_val
    return result


class Runtime:
    """Manages the plugin manager lifecycle for corvidae subcommands."""

    def __init__(
        self,
        config_path: str = "agent.yaml",
        overrides: dict | None = None,
    ):
        self.config_path = config_path
        self.overrides = overrides or {}
        self.pm = None
        self.registry = None

    async def start(self) -> None:
        """Load config, merge overrides, create PM, load plugins, start.

        Steps:
        1.  Load YAML config from config_path. Raises FileNotFoundError if missing.
        2.  Deep-merge self.overrides on top of the YAML config.
        3.  Set config["_base_dir"] from config_path (after merge; cannot be overridden).
        4.  Configure logging (MUST be first operational step after config load).
        5.  Create plugin manager.
        6.  Create ChannelRegistry and register it with the plugin manager.
        7.  Load channel config from merged config.
        7b. Block disabled plugins via pm.set_blocked for each name in
            config["plugins"]["disabled"]. Must precede step 8 so blocked
            plugins are never instantiated by the entry-point loader.
        8.  Load entry-point plugins (corvidae group).
        9.  Validate plugin dependencies.
        10. await pm.ahook.on_init(pm=pm, config=config).
        11. await pm.ahook.on_start(config=config).
            NOTE: Agent.on_start has no @hookimpl — the broadcast skips it.
        12. Retrieve agent plugin; raise RuntimeError if absent.
            await agent.on_start(config=config) explicitly, after the broadcast
            so that plugins Agent depends on have already started.
        """
        # 1. Load YAML config
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # 2. Deep-merge overrides
        config = deep_merge(config, self.overrides)

        # 3. Set _base_dir after merge — cannot be overridden
        config["_base_dir"] = Path(self.config_path).parent

        # 4. Configure logging — must be first operational step
        log_section = config.get("logging", {})
        log_level = log_section.get("level", "INFO")
        log_file = log_section.get("file")
        configure_logging(level=log_level, file=log_file)
        logger.info(
            "logging configured",
            extra={"level": log_level, "file": log_file or "stderr"},
        )

        # 5. Create plugin manager
        self.pm = create_plugin_manager()

        # 6. Create ChannelRegistry and register with PM
        self.registry = ChannelRegistry()
        self.pm.register(self.registry, name="registry")
        self.registry.agent_defaults = config.get("agent", {})

        # 7. Load channel config
        load_channel_config(config, self.registry)

        # 7b. Block disabled plugins before loading entry points so they are
        #     never instantiated. Must precede load_setuptools_entrypoints.
        disabled_plugins = config.get("plugins", {}).get("disabled", [])
        for name in disabled_plugins:
            self.pm.set_blocked(name)

        # 8. Load entry-point plugins
        self.pm.load_setuptools_entrypoints("corvidae")

        # 9. Validate dependencies
        validate_dependencies(self.pm)

        # 10. on_init broadcast
        await self.pm.ahook.on_init(pm=self.pm, config=config)

        # 11. on_start broadcast (Agent.on_start has no @hookimpl — skipped here)
        await self.pm.ahook.on_start(config=config)

        # 12. Explicit agent.on_start — after broadcast so dependencies have started
        agent = self.pm.get_plugin("agent")
        if agent is None:
            raise RuntimeError("Agent plugin not registered — check entry points")
        await agent.on_start(config=config)

    async def stop(self) -> None:
        """Graceful shutdown: agent.on_stop, then pm.ahook.on_stop.

        Runs under a 3-second timeout; os._exit(1) on timeout.
        Raises RuntimeError if agent plugin is not registered.
        """
        agent = self.pm.get_plugin("agent")
        if agent is None:
            raise RuntimeError("Agent plugin not registered — cannot stop cleanly")

        try:
            await asyncio.wait_for(
                self._run_shutdown(agent),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            logger.warning("graceful shutdown timed out after 3s, force-exiting")
            logging.shutdown()
            os._exit(1)

    async def _run_shutdown(self, agent: object) -> None:
        """Run agent and plugin shutdown in order. Called under a timeout in stop()."""
        await agent.on_stop()
        await self.pm.ahook.on_stop()

    async def run(self) -> None:
        """start(), wait for SIGINT/SIGTERM, then stop()."""
        await self.start()

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

        await self.stop()
