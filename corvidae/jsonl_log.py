"""JSONL conversation log plugin.

Writes an append-only JSONL log alongside the SQLite conversation store.
Each conversation event (message append, compaction summary) is written as
a single JSON line to a per-channel file.

Config:
    daemon:
      jsonl_log_dir: logs/   # directory for JSONL files; omit to disable
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import IO

from corvidae.hooks import CorvidaePlugin, hookimpl

logger = logging.getLogger(__name__)


class JsonlLogPlugin(CorvidaePlugin):
    """Plugin that writes conversation events to per-channel JSONL files.

    Implements on_conversation_event and on_compaction hookimpls. Each
    append and compaction summary produces one JSON line in the log file
    for that channel.

    If ``jsonl_log_dir`` is not configured, the plugin is a complete no-op.
    """

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._log_dir: Path | None = None
        self._handles: dict[str, IO] = {}  # channel_id -> open file handle

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        log_dir = config.get("daemon", {}).get("jsonl_log_dir")
        if log_dir is None:
            return
        base_dir = config.get("_base_dir", Path("."))
        self._log_dir = Path(base_dir) / log_dir

    @hookimpl
    async def on_start(self, config: dict) -> None:
        if self._log_dir is None:
            return
        await asyncio.to_thread(self._log_dir.mkdir, parents=True, exist_ok=True)
        logger.info("JSONL log directory: %s", self._log_dir)

    @hookimpl
    async def on_conversation_event(self, channel, message: dict, message_type) -> None:
        """Write a JSONL record for a conversation message event."""
        if self._log_dir is None:
            return
        # Strip _message_type tag if present
        clean = {k: v for k, v in message.items() if k != "_message_type"}
        record = {
            "ts": time.time(),
            "channel": channel.id,
            "type": str(message_type.value) if hasattr(message_type, "value") else str(message_type),
            "message": clean,
        }
        await asyncio.to_thread(self._write_record, channel.id, record)

    @hookimpl
    async def on_compaction(self, channel, summary_msg: dict, retain_count: int) -> None:
        """Write a JSONL record for a compaction summary event."""
        if self._log_dir is None:
            return
        # Strip _message_type tag if present
        clean = {k: v for k, v in summary_msg.items() if k != "_message_type"}
        record = {
            "ts": time.time(),
            "channel": channel.id,
            "type": "summary",
            "message": clean,
        }
        await asyncio.to_thread(self._write_record, channel.id, record)

    def _write_record(self, channel_id: str, record: dict) -> None:
        """Write a single JSON record to the channel's log file (sync, for use with to_thread)."""
        fh = self._get_handle(channel_id)
        fh.write(json.dumps(record) + "\n")
        fh.flush()

    def _close_all_handles(self) -> None:
        """Flush and close all open file handles (sync, for use with to_thread)."""
        for fh in self._handles.values():
            fh.flush()
            fh.close()
        self._handles.clear()

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
        await asyncio.to_thread(self._close_all_handles)
