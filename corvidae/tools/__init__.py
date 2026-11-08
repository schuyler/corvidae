# corvidae/tools/__init__.py

import aiohttp

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.spill import NotificationSpill
from corvidae.tool import Tool
from corvidae.tool_collection import ToolCollectionPlugin

from corvidae.tools.shell import shell
from corvidae.tools.files import read_file, write_file
from corvidae.tools.web import web_fetch, web_fetch_with_session, web_search


class CoreToolsPlugin(CorvidaePlugin):
    depends_on = frozenset({"tools", "spill"})

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

        # Declare the timeout shell really uses, read live at dispatch (R4).
        shell.own_timeout = lambda: plugin._shell_timeout  # type: ignore[attr-defined]
        # A shell command can read a notification spill file.
        shell.reads_spill_files = True  # type: ignore[attr-defined]

        async def read_file(
            path: str,
            offset: int | None = None,
            length: int | None = None,
            start_line: int | None = None,
            end_line: int | None = None,
        ) -> str:
            """Read a file. offset and length select a character range; start_line and end_line (1-based, inclusive) select a line range; use one kind per call. A ranged result's header names what it returned and where to continue."""
            from corvidae.tools.files import read_file as _read_file_impl
            # A spill file past its age is gone; say so rather than "not found".
            stale = get_dependency(plugin.pm, "spill", NotificationSpill).stale_pointer_error(path)
            if stale is not None:
                return stale
            # An unranged read is untouched; a ranged one is clamped to the
            # live result cap so it is never truncated after the fact.
            if offset is None and length is None and start_line is None and end_line is None:
                return await _read_file_impl(path, max_size=plugin._max_file_read_bytes)
            max_chars = get_dependency(plugin.pm, "tools", ToolCollectionPlugin).max_result_chars
            return await _read_file_impl(
                path, offset=offset, length=length, max_chars=max_chars,
                start_line=start_line, end_line=end_line,
            )

        # read_file reads notification spill files.
        read_file.reads_spill_files = True  # type: ignore[attr-defined]
        # File contents are outside text, spill files included.
        read_file.returns_untrusted = True  # type: ignore[attr-defined]

        async def web_fetch(url: str) -> str:
            """Fetch a URL and return its text content."""
            return await web_fetch_with_session(
                plugin._session, url,
                max_response_bytes=plugin._web_max_response_bytes,
                timeout=plugin._web_fetch_timeout,
            )

        # web_fetch's own timeout, read live at dispatch (R4).
        web_fetch.own_timeout = lambda: plugin._web_fetch_timeout  # type: ignore[attr-defined]
        # A fetched page is outside text.
        web_fetch.returns_untrusted = True  # type: ignore[attr-defined]

        async def _web_search(query: str, max_results: int | None = None) -> str:
            """Search the web via DuckDuckGo."""
            return await web_search(query, max_results=max_results or plugin._web_search_max_results)
        _web_search.__name__ = "web_search"
        # Search snippets are outside text.
        _web_search.returns_untrusted = True  # type: ignore[attr-defined]

        from corvidae.tools.task_pipeline import TaskPipelinePlugin

        task_plugin = TaskPipelinePlugin(self.pm)
        task_plugin.register_tools(tool_registry)

        tool_registry.extend([
            Tool.from_function(shell),
            Tool.from_function(read_file),
            Tool.from_function(write_file),
            Tool.from_function(web_fetch),
            Tool.from_function(_web_search),
        ])
