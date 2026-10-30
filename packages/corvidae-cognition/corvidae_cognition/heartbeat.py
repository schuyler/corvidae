"""HeartbeatPlugin — skeleton (red phase, #134)."""

from __future__ import annotations

from corvidae.hooks import CorvidaePlugin

HEARTBEAT_CHANNEL = "internal:heartbeat"
DEFAULT_HEARTBEAT_PROMPT = "heartbeat"


class HeartbeatPlugin(CorvidaePlugin):
    """Skeleton."""

    depends_on = frozenset({"idle", "registry"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm

    async def on_stop(self) -> None:
        """Skeleton."""

    async def on_start(self, config: dict) -> None:
        """Skeleton."""
