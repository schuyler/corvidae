"""Context panel: a live, read-only view of any loaded channel's context window, in a section of the web console's chat page (#687)."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from dataclasses import dataclass
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import Iterator

import aiosqlite
from aiohttp import web

from corvidae.channel import Channel, ChannelRegistry
from corvidae.channels.web import Panel
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.outbound import (
    Outbound,
    OutboundProgress,
    OutboundThinking,
    OutboundToolStatus,
)
from corvidae.persistence import PersistencePlugin

logger = logging.getLogger(__name__)

#: Rows an open returns unless the client asks for all of them.
WINDOW_ROWS = 200


@dataclass
class _Live:
    """Live events buffered for one channel."""

    mark: tuple[int, str]  # window mark when the first event arrived
    events: list[dict]  # wire-ready live events, arrival order


def _visible(row: dict) -> dict:
    """The row without its private ("_"-prefixed) bookkeeping keys."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _wire_row(row: object) -> dict:
    """A window row as the client sees it: its type tag and visible fields."""
    # A non-dict row (PRD 3.1.18.1) is shown as its repr.
    if not isinstance(row, dict):
        return {"type": None, "msg": {"role": None, "content": repr(row)}}
    kind = row.get("_message_type", MessageType.MESSAGE)
    return {"type": MessageType(kind).value, "msg": _visible(row)}


def _row_bytes(row: object) -> bytes:
    """The bytes a row contributes to digests; fails on the rows token counting fails on."""
    # Same serialization as the kernel's message_tokens, so no new failure mode.
    if not isinstance(row, dict):
        return json.dumps([None, repr(row)], default=str).encode()
    kind = MessageType(row.get("_message_type", MessageType.MESSAGE)).value
    return json.dumps([kind, _visible(row)], default=str).encode()


def _tools(conv: ContextWindow) -> list[dict] | None:
    """The window's tool schemas, or None while the tool view is unresolved."""
    return conv.tool_view.schemas if conv.tool_view is not None else None


def _digests(conv: ContextWindow, count: int) -> tuple[str, str]:
    """sha256 of head + rows[:count], and of head + all rows, in one pass.

    Requires count <= len(conv.messages).
    """
    # The head is the system prompt and tool view, which compaction replaces late.
    head = json.dumps([conv.system_prompt, _tools(conv)], default=str)
    digest = hashlib.sha256(head.encode())
    prefix = None
    for index, row in enumerate(conv.messages):
        # Snapshot the hash once the client's rows have been fed in.
        if index == count:
            prefix = digest.copy().hexdigest()
        digest.update(b"\n" + _row_bytes(row))
    full = digest.hexdigest()
    return (full if prefix is None else prefix), full


def _mark(conv: ContextWindow) -> tuple[int, str]:
    """Window length plus a digest of row 0: moves when rows land or the head moves."""
    rows = conv.messages
    first = hashlib.sha256(_row_bytes(rows[0])).hexdigest() if rows else ""
    return len(rows), first


async def _stamp(db: aiosqlite.Connection, channel: Channel, rows: list, wire: list[dict]) -> None:
    """Attach each wire row's message_log time as `ts`; rows without a logged id get none."""
    by_id = {
        r["_db_id"]: w
        for r, w in zip(rows, wire)
        if isinstance(r, dict) and isinstance(r.get("_db_id"), int)
    }
    if not by_id:
        return
    # One range query rather than an id list, which could exceed SQLite's parameter limit.
    async with db.execute(
        "SELECT id, timestamp FROM message_log WHERE channel_id = ? AND id BETWEEN ? AND ?",
        (channel.id, min(by_id), max(by_id)),
    ) as cursor:
        for row_id, ts in await cursor.fetchall():
            if row_id in by_id:
                by_id[row_id]["ts"] = ts


def _json(data: dict) -> web.Response:
    """A JSON response; serialized here, so the caller's window read has no await before it."""
    return web.json_response(data, dumps=partial(json.dumps, default=str))


class ContextPanelPlugin(CorvidaePlugin):
    """Web console sidebar section with one tab per loaded channel, each showing that channel's context window."""

    depends_on = frozenset({"registry", "persistence"})
    optional_depends_on = frozenset({"web"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._configured = False
        self._serving = False
        self._live: dict[str, _Live] = {}
        self._seq: Iterator[int] = itertools.count(1)

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        # No block: off. A bare `context_panel:` enables it, like `web:`.
        if "context_panel" not in config:
            logger.debug("context_panel: no block in config; panel off")
            return
        block = config["context_panel"]
        if block is not None and not isinstance(block, dict):
            raise ValueError(
                f"Config error: context_panel must be a mapping (e.g. "
                f"context_panel: {{}}), got {type(block).__name__} {block!r} "
                f"— check agent.yaml"
            )
        self._configured = True

    @hookimpl
    async def on_web_console_init(self, panels: list[Panel]) -> None:
        # Register only when configured; _serving then turns on live buffering.
        if not self._configured:
            return
        self._serving = True
        panels.append(
            Panel(
                plugin=self,
                title="Context",
                static_dir=Path(str(files("context_panel") / "static")),
                routes=[
                    web.get("/channels", self.channels),
                    web.get("/window", self.window),
                    web.get("/poll", self.poll),
                ],
            )
        )

    @hookimpl
    async def send(self, channel: Channel, event: Outbound) -> None:
        """Observe every channel's outbound events; buffer the live ones."""
        if not self._serving:
            return
        # Wire form of the three event kinds; the final reply is already a row.
        if isinstance(event, OutboundThinking):
            wire = {"kind": "thinking", "text": event.text}
        elif isinstance(event, OutboundProgress):
            wire = {"kind": "progress", "text": event.text}
        elif isinstance(event, OutboundToolStatus):
            wire = {
                "kind": "tool_status",
                "tool": event.tool_name,
                "status": event.status,
                "summary": event.args_summary or event.result_summary or "",
                "tool_call_id": event.tool_call_id,
            }
        else:
            return
        conv = channel.conversation
        if conv is None:
            return
        # A different mark means rows landed since: start a fresh buffer.
        mark = _mark(conv)
        live = self._live.get(channel.id)
        if live is None or live.mark != mark:
            live = self._live[channel.id] = _Live(mark=mark, events=[])
        live.events.append({"seq": next(self._seq), **wire})

    def _db(self) -> aiosqlite.Connection:
        """The persistence plugin's database connection."""
        return get_dependency(self.pm, "persistence", PersistencePlugin).db

    def _channel(self, request: web.Request) -> Channel | None:
        """The registry channel named by ?channel=, or None if it isn't registered."""
        registry = get_dependency(self.pm, "registry", ChannelRegistry)
        return registry.get(request.query.get("channel", ""))

    async def channels(self, request: web.Request) -> web.Response:
        """List every channel whose window is loaded."""
        registry = get_dependency(self.pm, "registry", ChannelRegistry)
        loaded = sorted(
            (c for c in registry.all() if c.conversation is not None), key=lambda c: c.id
        )
        return _json(
            {
                "channels": [
                    {
                        "id": c.id,
                        # Present only when the transport named the channel.
                        **({"name": c.name} if c.name else {}),
                        "messages": len(c.conversation.messages),
                        "tokens": c.conversation.token_estimate(),
                        "budget": registry.resolve_config(c)["max_context_tokens"],
                    }
                    for c in loaded
                ]
            }
        )

    async def window(self, request: web.Request) -> web.Response:
        """Open a channel's window: head, the newest rows (or all) and the cursor."""
        channel = self._channel(request)
        conv = channel.conversation if channel is not None else None
        if conv is None:
            return _json({"loaded": False})
        rows = conv.messages
        shown = rows if request.query.get("all") == "1" else rows[-WINDOW_ROWS:]
        wire = [_wire_row(r) for r in shown]
        body = {
            "loaded": True,
            "system_prompt": conv.system_prompt,
            "tools": _tools(conv),
            "total": len(rows),
            "first": len(rows) - len(shown),
            "rows": wire,
            "cursor": {"count": len(rows), "digest": _digests(conv, len(rows))[1]},
        }
        # The response is built before this await; only ts is attached after it.
        await _stamp(self._db(), channel, shown, wire)
        return _json(body)

    async def poll(self, request: web.Request) -> web.Response:
        """Tail a channel from the client's cursor, plus its live events."""
        channel = self._channel(request)
        conv = channel.conversation if channel is not None else None
        if conv is None:
            return _json({"loaded": False})
        # Live events count only while the window still has the mark they arrived at.
        entry = self._live.get(channel.id)
        live = entry.events if entry is not None and entry.mark == _mark(conv) else []
        count = request.query.get("count", "")
        digest = request.query.get("digest")
        # Resync unless the client's rows are still a prefix of the window.
        if digest is None or not count.isdecimal() or int(count) > len(conv.messages):
            return _json({"loaded": True, "resync": True, "live": live})
        prefix, full = _digests(conv, int(count))
        if prefix != digest:
            return _json({"loaded": True, "resync": True, "live": live})
        tail = conv.messages[int(count):]
        wire = [_wire_row(r) for r in tail]
        body = {
            "loaded": True,
            "resync": False,
            "rows": wire,
            "cursor": {"count": len(conv.messages), "digest": full},
            "live": live,
        }
        # The response is built before this await; only ts is attached after it.
        await _stamp(self._db(), channel, tail, wire)
        return _json(body)
