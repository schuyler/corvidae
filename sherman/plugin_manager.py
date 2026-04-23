import apluggy as pluggy
from sherman.hooks import AgentSpec

def create_plugin_manager() -> pluggy.PluginManager:
    """Create and configure the plugin manager with AgentSpec hooks."""
    pm = pluggy.PluginManager("sherman")
    pm.add_hookspecs(AgentSpec)
    return pm
