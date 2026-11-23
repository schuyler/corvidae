"""Compaction's failure paths name the channel and do not treat an
unreachable backend as a summary failure (#400, PRD §3.3.2).

Two behaviours:
  - the skip warning in the agent's turn names the channel and the exception
    type, so a reader can tell which channel stopped compacting;
  - when the compaction fork fails because the backend is unreachable, the
    background view is not attempted. An outage already waits out
    ``model_not_loaded_wait`` once; spending it twice on the same dead
    backend delays the turn's own retry loop for nothing.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.hooks import create_plugin_manager

from helpers import FakeForkAgent
from test_compaction_fork import _compact, _straddling_window

pytestmark = pytest.mark.timeout(15)


def _unreachable() -> aiohttp.ClientConnectionError:
    """A connection-class failure: the server could not be reached."""
    return aiohttp.ClientConnectionError("connection refused")


# ---------------------------------------------------------------------------
# An unreachable backend defers compaction instead of burning the view call
# ---------------------------------------------------------------------------


async def test_unreachable_fork_does_not_attempt_the_view():
    """A connection-class fork failure skips the background view, which would
    spend the same wait again against a backend that is down."""
    conv = _straddling_window()
    agent = FakeForkAgent({"compaction"}, error=_unreachable())
    result, order, summarize = await _compact(conv, agent)

    summarize.assert_not_called()
    assert result is None
    assert order.calls == []
    assert len(conv.messages) == 10


async def test_unreachable_fork_records_the_cooldown():
    """Deferring counts as a failed attempt, so the next turn does not retry
    into the same outage immediately."""
    conv = _straddling_window()
    agent = FakeForkAgent({"compaction"}, error=_unreachable())
    pm = create_plugin_manager()
    pm.register(agent, name="agent")
    plugin = CompactionPlugin(pm=pm)
    channel = MagicMock()
    channel.id = "test:scope"
    summarize = AsyncMock(return_value="sum")

    from unittest.mock import patch
    with patch.object(plugin, "_view_summary", summarize):
        await plugin.resolve_compaction(
            channel=channel, conversation=conv, max_tokens=20000
        )

    assert "test:scope" in plugin._last_failed_compaction


async def test_non_connection_fork_failure_still_falls_back_to_the_view():
    """A 400 from a live backend is a summary miss, not an outage: the view
    is still tried, exactly as today."""
    conv = _straddling_window()
    agent = FakeForkAgent({"compaction"}, error=RuntimeError("400 bad request"))
    result, _, summarize = await _compact(conv, agent)

    summarize.assert_called_once()
    assert result is True


# ---------------------------------------------------------------------------
# The skip warning names the channel
# ---------------------------------------------------------------------------


async def test_skip_warning_names_the_channel(caplog):
    """PRD §3.3.2's warning and AGENTS.md both require the identifier."""
    from corvidae.agent import Agent

    agent_plugin = Agent.__new__(Agent)
    agent_plugin.pm = MagicMock()
    agent_plugin.queues = {}

    channel = MagicMock()
    channel.id = "signal:test"

    with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
        agent_plugin._warn_compaction_failed(channel, RuntimeError("boom"))

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "no warning logged"
    message = warnings[0]
    assert "signal:test" in message
    assert "RuntimeError" in message
    assert "compaction" in message.lower()
