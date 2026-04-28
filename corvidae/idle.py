"""IdleMonitorPlugin — pure consumer of the on_idle hook.

Implements idle behaviors that run when the agent system becomes idle.
Idle detection is now push-based: AgentPlugin calls _maybe_fire_idle()
after each queue item completes and broadcasts the on_idle hook when
all queues are empty and the cooldown has elapsed.
"""

import logging

from corvidae.hooks import hookimpl

logger = logging.getLogger(__name__)


class IdleMonitorPlugin:
    """Pure consumer of the on_idle hook. Implements idle behaviors."""

    depends_on = set()

    def __init__(self, pm) -> None:
        self.pm = pm

    @hookimpl
    async def on_idle(self) -> None:
        pass
