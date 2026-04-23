"""Plugin manager creation and configuration.

This module creates the pluggy PluginManager that coordinates all plugins.
The plugin manager is the central orchestrator — plugins register with it,
and it dispatches hook calls to the appropriate implementations.

Logging:
    - DEBUG: plugin manager created
"""

import logging

import apluggy as pluggy

from sherman.hooks import AgentSpec

logger = logging.getLogger(__name__)


def create_plugin_manager() -> pluggy.PluginManager:
    """Create and configure the plugin manager with AgentSpec hooks.

    Returns:
        A pluggy.PluginManager instance with the Sherman AgentSpec loaded.

    Logs a DEBUG message when the manager is created.
    """
    pm = pluggy.PluginManager("sherman")
    pm.add_hookspecs(AgentSpec)

    logger.debug("plugin manager created")

    return pm
