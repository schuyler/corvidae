"""Shared async helper functions for integration tests."""

import asyncio
import contextlib

import aiosqlite


def sent(pm, kind: type) -> list[tuple[object, object]]:
    """Return (channel, event) for every ``send`` of an event of ``kind``.

    Reads the AsyncMock that ``build_plugin_and_channel`` installs as
    ``pm.ahook.send``, in call order. ``kind`` is an outbound event class
    from corvidae.outbound, e.g. ``sent(pm, OutboundMessage)``.
    """
    return [
        (call.kwargs["channel"], call.kwargs["event"])
        for call in pm.ahook.send.await_args_list
        if isinstance(call.kwargs["event"], kind)
    ]


def sent_texts(pm, kind: type) -> list[str]:
    """Return the ``text`` of every sent event of ``kind``, in call order."""
    return [event.text for _channel, event in sent(pm, kind)]


async def build_plugin_and_channel(
    agent_defaults=None,
    channel_config=None,
    *,
    mock_send: bool = True,
    mock_on_agent_response: bool = True,
    db_path: str = ":memory:",
):
    """Assemble plugin graph with an in-memory (or ``db_path`` file) DB.

    Returns (plugin, channel, db). Callers are responsible for teardown:
    task_plugin.on_stop() and db.close().

    Args:
        agent_defaults: Override AGENT_DEFAULTS for channel registry.
        channel_config: Override default ChannelConfig.
        mock_send: If True, replace the outbound ``send`` hook with an
            AsyncMock; read what was sent with ``sent()``.
        mock_on_agent_response: If True, replace on_agent_response hook with AsyncMock.
        db_path: The session DB; a file path lets a second connection (e.g.
            a read-only probe) see what the turn persisted.
    """
    from unittest.mock import AsyncMock

    from corvidae.agent import Agent
    from corvidae.channel import ChannelConfig, ChannelRegistry
    from corvidae.persistence import PersistencePlugin, init_db
    from corvidae.hooks import create_plugin_manager
    from corvidae.task import TaskPlugin
    from corvidae.thinking import ThinkingPlugin

    _AGENT_DEFAULTS = {
        "system_prompt": "You are a test assistant.",
        "max_context_tokens": 8000,
        "reasoning_history": "drop",
    }

    if agent_defaults is None:
        agent_defaults = _AGENT_DEFAULTS

    db = await aiosqlite.connect(db_path)
    await init_db(db)

    pm = create_plugin_manager()
    registry = ChannelRegistry()
    registry.agent_defaults = agent_defaults
    pm.register(registry, name="registry")

    if mock_send:
        pm.ahook.send = AsyncMock()
    if mock_on_agent_response:
        pm.ahook.on_agent_response = AsyncMock()

    task_plugin = TaskPlugin()
    pm.register(task_plugin, name="task")
    await task_plugin.on_init(pm=pm, config={})
    await task_plugin.on_start(config={})

    persistence = PersistencePlugin()
    persistence.db = db
    persistence._registry = registry
    pm.register(persistence, name="persistence")

    thinking_plugin = ThinkingPlugin()
    pm.register(thinking_plugin, name="thinking")
    await thinking_plugin.on_init(pm=pm, config={})

    plugin = Agent()
    pm.register(plugin, name="agent")
    plugin.pm = pm
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


class FakeForkAgent:
    """Stands in for Agent in fork tests (GitHub issue #51).

    Forks are enabled for ``stages``; each fork is recorded with its
    attribution and answers ``text``, or raises ``error``. ``fail_marker``
    limits the error to instructions containing it.
    """

    def __init__(self, stages, text="", error=None, fail_marker=None):
        self.stages = set(stages)
        self.text = text
        self.error = error
        self.fail_marker = fail_marker
        self.calls = []

    def fork_enabled(self, stage):
        return stage in self.stages

    async def fork(self, channel, instruction, *, cut=None, extra_body=None):
        from corvidae.attribution import get_attribution
        from corvidae.fork import ForkResult

        self.calls.append({
            "instruction": instruction, "cut": cut, "extra_body": extra_body,
            "attribution": dict(get_attribution()),
        })
        if self.error is not None and (self.fail_marker is None or self.fail_marker in instruction):
            raise self.error
        return ForkResult(text=self.text, message={"role": "assistant", "content": self.text})


def window_of(channel_id, rowids, summary=False, contents=None):
    """A ContextWindow of alternating user/assistant MESSAGE rows tagged with
    ``rowids`` (contents default to "m<rowid>"), optionally after a summary."""
    from corvidae.context import ContextWindow, MessageType

    conv = ContextWindow(channel_id)
    conv.system_prompt = "You are Lex."
    if summary:
        conv.messages.append({"role": "assistant", "content": "[Summary of earlier conversation]\nold",
                              "_message_type": MessageType.SUMMARY})
    for i, rid in enumerate(rowids):
        conv.messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": contents[i] if contents else f"m{rid}",
            "_db_id": rid, "_message_type": MessageType.MESSAGE,
        })
    return conv


@contextlib.asynccontextmanager
async def acp_agent_env(agent_defaults: dict):
    """An AcpPlugin and CorvidaeAcpAgent over an in-memory sessions DB, with
    a mocked client connection (``conn.session_update`` is an AsyncMock).

    Yields a SimpleNamespace with pm, registry, db, plugin, agent and conn;
    closes the DB on exit. Callers still call ``agent.initialize``.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from corvidae.channel import ChannelRegistry
    from corvidae.channels.acp import AcpPlugin, CorvidaeAcpAgent
    from corvidae.hooks import create_plugin_manager
    from corvidae.persistence import PersistencePlugin, init_db

    pm = create_plugin_manager()
    registry = ChannelRegistry(agent_defaults)
    pm.register(registry, name="registry")
    persistence = PersistencePlugin(pm)
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    persistence.db = db
    pm.register(persistence, name="persistence")
    plugin = AcpPlugin()
    pm.register(plugin, name="acp")
    await plugin.on_init(pm=pm, config={"_acp_mode": True})
    agent = CorvidaeAcpAgent(
        agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"},
        plugin=plugin,
    )
    conn = MagicMock()
    conn.session_update = AsyncMock()
    plugin._conn = conn
    agent.on_connect(conn)
    try:
        yield SimpleNamespace(pm=pm, registry=registry, db=db, plugin=plugin, agent=agent, conn=conn)
    finally:
        await db.close()
