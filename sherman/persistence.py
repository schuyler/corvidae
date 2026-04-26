"""PersistencePlugin — SQLite database lifecycle and conversation initialization.

Manages the SQLite database connection and lazy conversation initialization.
Registered before AgentPlugin so ensure_conversation is available when the
agent loop processes its first message.

Config:
    daemon:
      session_db: sessions.db   # path to SQLite database
"""

import logging
from pathlib import Path

import aiosqlite

from sherman.channel import ChannelRegistry, resolve_system_prompt
from sherman.conversation import ConversationLog, init_db
from sherman.hooks import get_dependency, hookimpl

logger = logging.getLogger(__name__)


class PersistencePlugin:
    """Plugin that manages SQLite database lifecycle and conversation initialization.

    Attributes:
        pm: Plugin manager instance.
        db: aiosqlite.Connection, opened in on_start, closed in on_stop.
            Public for test injection (same pattern as former AgentPlugin.db).
        base_dir: Base path for resolving relative system prompt file paths.
    """

    depends_on = {"registry"}

    def __init__(self, pm) -> None:
        self.pm = pm
        self.db: aiosqlite.Connection | None = None
        self.base_dir: Path = Path(".")
        self._registry: ChannelRegistry | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        self.base_dir = config.get("_base_dir", Path("."))

        # Open SQLite database (only if not already injected for testing)
        if self.db is None:
            db_path = config.get("daemon", {}).get("session_db", "sessions.db")
            self.db = await aiosqlite.connect(db_path)
            await init_db(self.db)

    @hookimpl
    async def on_stop(self) -> None:
        if self.db:
            await self.db.close()

    @hookimpl
    async def ensure_conversation(self, channel) -> bool | None:
        """Create and load a ConversationLog for the channel if not present."""
        if channel.conversation is not None:
            return True

        conv = ConversationLog(self.db, channel.id)
        resolved = self._registry.resolve_config(channel)
        conv.system_prompt = resolve_system_prompt(
            resolved["system_prompt"], self.base_dir
        )
        await conv.load()
        channel.conversation = conv

        logger.info(
            "conversation initialized for channel",
            extra={"channel": channel.id},
        )
        return True
