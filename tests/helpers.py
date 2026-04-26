"""Shared async helper functions for integration tests."""

import asyncio

import aiosqlite


async def build_plugin_and_channel(
    agent_defaults=None,
    channel_config=None,
    *,
    mock_send_message: bool = True,
    mock_on_agent_response: bool = True,
):
    """Assemble plugin graph with in-memory DB.

    Returns (plugin, channel, db). Callers are responsible for teardown:
    task_plugin.on_stop() and db.close().

    Args:
        agent_defaults: Override AGENT_DEFAULTS for channel registry.
        channel_config: Override default ChannelConfig.
        mock_send_message: If True, replace send_message hook with AsyncMock.
        mock_on_agent_response: If True, replace on_agent_response hook with AsyncMock.
    """
    from unittest.mock import AsyncMock

    from corvidae.agent import AgentPlugin
    from corvidae.channel import ChannelConfig, ChannelRegistry
    from corvidae.conversation import init_db
    from corvidae.hooks import create_plugin_manager
    from corvidae.persistence import PersistencePlugin
    from corvidae.task import TaskPlugin
    from corvidae.thinking import ThinkingPlugin

    _AGENT_DEFAULTS = {
        "system_prompt": "You are a test assistant.",
        "max_context_tokens": 8000,
        "keep_thinking_in_history": False,
    }

    if agent_defaults is None:
        agent_defaults = _AGENT_DEFAULTS

    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    registry = ChannelRegistry(agent_defaults)
    pm.register(registry, name="registry")

    if mock_send_message:
        pm.ahook.send_message = AsyncMock()
    if mock_on_agent_response:
        pm.ahook.on_agent_response = AsyncMock()

    task_plugin = TaskPlugin(pm)
    pm.register(task_plugin, name="task")
    await task_plugin.on_start(config={})

    persistence = PersistencePlugin(pm)
    persistence.db = db
    persistence._registry = registry
    pm.register(persistence, name="persistence")

    thinking_plugin = ThinkingPlugin(pm)
    pm.register(thinking_plugin, name="thinking")

    plugin = AgentPlugin(pm)
    pm.register(plugin, name="agent_loop")
    plugin._registry = registry

    channel = registry.get_or_create(
        "test",
        "scope1",
        config=channel_config or ChannelConfig(),
    )

    return plugin, channel, db


async def drain(plugin, channel):
    """Drain the channel's SerialQueue."""
    if channel.id in plugin.queues:
        await plugin.queues[channel.id].drain()


async def drain_task_queue(plugin):
    """Wait for all TaskQueue tasks to complete including on_complete callbacks."""
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin and task_plugin.task_queue:
        await task_plugin.task_queue.queue.join()
        await asyncio.sleep(0)
