"""Context panel: a live, read-only view of any loaded channel's context window (#687)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterator

from aiohttp import web

from corvidae.channel import Channel
from corvidae.channels.web import Panel
from corvidae.context import ContextWindow
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.outbound import Outbound

#: Rows an open returns unless the client asks for all of them.
WINDOW_ROWS = 200


@dataclass
class _Live:
    """Live events buffered for one channel."""

    mark: tuple[int, str]  # window mark when the first event arrived
    events: list[dict]  # wire-ready live events, arrival order


def _visible(row: dict) -> dict:
    raise NotImplementedError


def _wire_row(row: object) -> dict:
    raise NotImplementedError


def _row_bytes(row: object) -> bytes:
    raise NotImplementedError


def _digests(conv: ContextWindow, count: int) -> tuple[str, str]:
    raise NotImplementedError


def _mark(conv: ContextWindow) -> tuple[int, str]:
    raise NotImplementedError


class ContextPanelPlugin(CorvidaePlugin):
    """Web console panel showing every loaded channel's context window."""

    depends_on = frozenset({"registry"})
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

    @hookimpl
    async def on_web_console_init(self, panels: list[Panel]) -> None:
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
        return None

    async def channels(self, request: web.Request) -> web.Response:
        return web.json_response({})

    async def window(self, request: web.Request) -> web.Response:
        return web.json_response({})

    async def poll(self, request: web.Request) -> web.Response:
        return web.json_response({})
