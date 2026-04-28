"""JSONL conversation log plugin.

Writes an append-only JSONL log alongside the SQLite conversation store.
Each conversation event (message append, compaction summary) is written as
a single JSON line to a per-channel file.

Config:
    daemon:
      jsonl_log_dir: logs/   # directory for JSONL files; omit to disable
"""

import json
import logging
from pathlib import Path
from typing import IO

from corvidae.conversation import MessageType
from corvidae.hooks import hookimpl

logger = logging.getLogger(__name__)


class JsonlLogPlugin:
    """Plugin that writes conversation events to per-channel JSONL files.

    Registers an observer on each channel's ConversationLog after
    PersistencePlugin creates it. Each append and compaction summary
    produces one JSON line in the log file for that channel.

    If ``jsonl_log_dir`` is not configured, the plugin is a complete no-op.
    """

    depends_on = {"persistence"}

    def __init__(self):
        self._log_dir: Path | None = None
        self._handles: dict[str, IO] = {}  # channel_id -> open file handle

    @hookimpl
    async def on_start(self, config: dict) -> None:
        log_dir = config.get("daemon", {}).get("jsonl_log_dir")
        if log_dir is None:
            return
        base_dir = config.get("_base_dir", Path("."))
        self._log_dir = Path(base_dir) / log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        logger.info("JSONL log directory: %s", self._log_dir)

    @hookimpl(trylast=True)
    async def ensure_conversation(self, channel) -> bool | None:
        if self._log_dir is None:
            return None
        if channel.conversation is None:
            return None
        # Register observer if not already registered
        if self._observer not in channel.conversation.observers:
            channel.conversation.observers.append(self._observer)
        return None  # Don't claim we handled initialization

    async def _observer(self, channel_id: str, message: dict, message_type: MessageType, ts: float) -> None:
        record = {
            "ts": ts,
            "channel": channel_id,
            "type": message_type.value,
            "message": message,
        }
        fh = self._get_handle(channel_id)
        fh.write(json.dumps(record) + "\n")
        fh.flush()

    def _sanitize_channel_id(self, channel_id: str) -> str:
        return channel_id.replace("/", "_").replace(":", "_")

    def _get_handle(self, channel_id: str) -> IO:
        if channel_id not in self._handles:
            sanitized = self._sanitize_channel_id(channel_id)
            path = self._log_dir / f"{sanitized}.jsonl"
            self._handles[channel_id] = open(path, "a")
        return self._handles[channel_id]

    @hookimpl
    async def on_stop(self) -> None:
        for fh in self._handles.values():
            fh.flush()
            fh.close()
        self._handles.clear()
