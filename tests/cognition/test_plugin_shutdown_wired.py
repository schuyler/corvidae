"""Every cognition plugin's on_stop must stay a hookimpl (#1229 review).

A decorator lost in a rebase resolution is silent: the method still exists, so
nothing raises, and corvidae simply never calls it on shutdown.
"""

from __future__ import annotations

import pytest

from corvidae.hooks import create_plugin_manager
from corvidae_cognition.critique import CritiquePlugin
from corvidae_cognition.memory import MemoryPlugin


@pytest.mark.parametrize(
    ("plugin_class", "name"),
    [(CritiquePlugin, "critique"), (MemoryPlugin, "memory")],
)
def test_on_stop_is_registered_as_a_hookimpl(plugin_class, name):
    pm = create_plugin_manager()
    pm.register(plugin_class(), name=name)

    registered = [impl.function.__name__ for impl in pm.hook.on_stop.get_hookimpls()]

    assert "on_stop" in registered


async def test_memory_on_stop_closes_the_fork_log(tmp_path):
    """A fork-log connection left open keeps the interpreter alive (#1229 C2).

    ``ForkLog`` opens its aiosqlite connection lazily on the first record, so a
    process that ran a consolidation fork still has a worker thread at shutdown
    unless ``on_stop`` closes it. The connection is swapped for a sentinel here
    so the assertion is about the call, not about aiosqlite's internals.
    """
    plugin = MemoryPlugin()
    closed: list[bool] = []

    class _Log:
        async def close(self) -> None:
            closed.append(True)

    plugin._fork_log = _Log()

    await plugin.on_stop()

    assert closed == [True]
