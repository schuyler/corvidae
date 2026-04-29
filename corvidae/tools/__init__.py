# corvidae/tools/__init__.py

import aiohttp

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tool import Tool

from corvidae.tools.shell import shell
from corvidae.tools.files import read_file, write_file
from corvidae.tools.web import web_fetch, web_fetch_with_session, web_search


class CoreToolsPlugin(CorvidaePlugin):
    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        self.pm = pm  # May be None; on_init will set it via CorvidaePlugin.on_init
        self._session: aiohttp.ClientSession | None = None
        self._shell_timeout: int = 30
        self._web_fetch_timeout: int = 15
        self._web_max_response_bytes: int = 50_000
        self._max_file_read_bytes: int = 1024 * 1024
        self._web_search_max_results: int = 8

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        tools_config = config.get("tools", {})
        self._shell_timeout = tools_config.get("shell_timeout", 30)
        self._web_fetch_timeout = tools_config.get("web_fetch_timeout", 15)
        self._web_max_response_bytes = tools_config.get("web_max_response_bytes", 50_000)
        self._max_file_read_bytes = tools_config.get("max_file_read_bytes", 1024 * 1024)
        self._web_search_max_results = tools_config.get("web_search_max_results", 8)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._web_fetch_timeout)
        )

    @hookimpl
    async def on_stop(self) -> None:
        if self._session is not None:
            await self._session.close()

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self

        async def shell(command: str) -> str:
            """Execute a shell command and return the output."""
            from corvidae.tools.shell import shell as _shell_impl
            return await _shell_impl(command, timeout=plugin._shell_timeout)

        async def read_file(path: str) -> str:
            """Read the contents of a file."""
            from corvidae.tools.files import read_file as _read_file_impl
            return await _read_file_impl(path, max_size=plugin._max_file_read_bytes)

        async def web_fetch(url: str) -> str:
            """Fetch a URL and return its text content."""
            return await web_fetch_with_session(
                plugin._session, url,
                max_response_bytes=plugin._web_max_response_bytes,
                timeout=plugin._web_fetch_timeout,
            )

        async def _web_search(query: str, max_results: int | None = None) -> str:
            """Search the web via DuckDuckGo."""
            return await web_search(query, max_results=max_results or plugin._web_search_max_results)
        _web_search.__name__ = "web_search"

        from corvidae.tools.task_pipeline import TaskPipelinePlugin
        from corvidae.tools.index import WorkspaceIndexerPlugin

        task_plugin = TaskPipelinePlugin(self.pm)
        task_plugin.register_tools(tool_registry)

        tool_registry.extend([
            Tool.from_function(shell),
            Tool.from_function(read_file),
            Tool.from_function(write_file),
            Tool.from_function(web_fetch),
            Tool.from_function(_web_search),
        ])
