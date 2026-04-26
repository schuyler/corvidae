"""PersistencePlugin — SQLite database lifecycle and conversation initialization.

Manages the SQLite database connection and lazy conversation initialization.
Registered before AgentPlugin so ensure_conversation is available when the
agent loop processes its first message.

Config:
    daemon:
      session_db: sessions.db       # path to SQLite database
      sqlite_journal_mode: wal      # SQLite journal mode (default "wal");
                                    # allowed values: delete, truncate, persist,
                                    # memory, wal, off
"""

import logging
from pathlib import Path

import aiosqlite

from corvidae.channel import ChannelRegistry, resolve_system_prompt
from corvidae.conversation import DEFAULT_CHARS_PER_TOKEN, ConversationLog, init_db
from corvidae.hooks import get_dependency, hookimpl

logger = logging.getLogger(__name__)

_ALLOWED_JOURNAL_MODES = {"delete", "truncate", "persist", "memory", "wal", "off"}


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
        self._chars_per_token: float = DEFAULT_CHARS_PER_TOKEN

    @hookimpl
    async def on_start(self, config: dict) -> None:
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        self.base_dir = config.get("_base_dir", Path("."))
        agent_config = config.get("agent", {})
        self._chars_per_token = agent_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)

        # Open SQLite database (only if not already injected for testing)
        if self.db is None:
            db_path = config.get("daemon", {}).get("session_db", "sessions.db")
            self.db = await aiosqlite.connect(db_path)
            await init_db(self.db)

        # Set journal mode (WAL by default for concurrent access)
        journal_mode = config.get("daemon", {}).get("sqlite_journal_mode", "wal").lower()
        if journal_mode not in _ALLOWED_JOURNAL_MODES:
            raise ValueError(f"Invalid sqlite_journal_mode: {journal_mode!r}")
        async with self.db.execute(f"PRAGMA journal_mode={journal_mode}") as cursor:
            row = await cursor.fetchone()
        actual_mode = row[0] if row else journal_mode
        logger.info(
            "SQLite journal mode set to %s", actual_mode,
            extra={"journal_mode": actual_mode},
        )

    @hookimpl
    async def on_stop(self) -> None:
        if self.db:
            await self.db.close()

    @hookimpl
    async def ensure_conversation(self, channel) -> bool | None:
        """Create and load a ConversationLog for the channel if not present."""
        if channel.conversation is not None:
            return True

        conv = ConversationLog(self.db, channel.id, chars_per_token=self._chars_per_token)
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
