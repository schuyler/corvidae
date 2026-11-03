"""The agent picks up a hot-swapped llm.main client whichever order the
reload hook reaches the plugins in (#382).

`Agent.on_config_reload` re-borrows the client from `LLMPlugin`, and
`ConfigWatcherPlugin` dispatches `on_config_reload` hookimpl-by-hookimpl in
`reversed()` order. So the agent only saw a new client when the LLM
plugin's hookimpl happened to come first — which is entry-point
registration order, and CLAUDE.md says that is not deterministic.

The LLM plugin is real here, with `_create_client` stubbed, so the
`tryfirst` that orders the swap is the one under test rather than a
marker copied into a fake.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.config_watcher import ConfigWatcherPlugin
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.llm_plugin import LLMPlugin

pytestmark = pytest.mark.timeout(15)


class _FakeAgent:
    """Stands in for Agent: re-borrows the client when its hook runs."""

    def __init__(self, llm: LLMPlugin) -> None:
        self.llm = llm
        self.client = llm.get_client()

    @hookimpl
    async def on_config_reload(self, config):
        self.client = self.llm.get_client()


async def _dispatch(registration_order: list[str]) -> tuple[LLMPlugin, _FakeAgent]:
    """Register in ``registration_order`` and dispatch one reload through the
    real ConfigWatcherPlugin path."""
    llm = LLMPlugin()
    llm._role_configs["main"] = {"base_url": "http://old", "model": "old"}
    llm._clients["main"] = MagicMock(name="old-client")

    def _new_client(cfg, role):
        """A stand-in client whose lifecycle methods are awaitable."""
        client = MagicMock(name="new-client")
        client.start = AsyncMock()
        client.stop = AsyncMock()
        return client

    llm._create_client = MagicMock(side_effect=_new_client)
    agent = _FakeAgent(llm)

    pm = create_plugin_manager()
    watcher = ConfigWatcherPlugin()
    for name in registration_order:
        pm.register({"llm": llm, "agent": agent}[name], name=name)
    pm.register(watcher, name="watcher")

    # The watcher dispatches through the PM that on_init hands it.
    await watcher.on_init(pm=pm, config={})
    # The agent looks its dependency up by the registered name "llm".
    agent.pm = pm

    await watcher._dispatch_config_reload(
        {"llm": {"main": {"base_url": "http://new", "model": "new"}}}
    )
    return llm, agent


async def test_agent_sees_the_new_client_when_llm_runs_first():
    """Baseline: the order that works today keeps working."""
    llm, agent = await _dispatch(["agent", "llm"])
    assert agent.client is llm.get_client()


async def test_agent_sees_the_new_client_when_agent_runs_first():
    """The order that works today is not guaranteed; the other must work too."""
    llm, agent = await _dispatch(["llm", "agent"])
    assert agent.client is llm.get_client()
