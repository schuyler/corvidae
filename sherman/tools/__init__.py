# sherman/tools/__init__.py

import aiohttp

from sherman.hooks import hookimpl
from sherman.tool import Tool

from sherman.tools.shell import shell
from sherman.tools.files import read_file, write_file
from sherman.tools.web import web_fetch_with_session, web_fetch


class CoreToolsPlugin:
    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    @hookimpl
    async def on_start(self, **_) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )

    @hookimpl
    async def on_stop(self) -> None:
        if self._session is not None:
            await self._session.close()

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self  # capture for closure

        async def web_fetch(url: str) -> str:
            """Fetch a URL and return its text content."""
            return await web_fetch_with_session(plugin._session, url)

        tool_registry.extend([
            Tool.from_function(shell),
            Tool.from_function(read_file),
            Tool.from_function(write_file),
            Tool.from_function(web_fetch),
        ])
