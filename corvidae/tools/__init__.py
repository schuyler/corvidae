# corvidae/tools/__init__.py
"""The built-in tool plugins.

Three plugins, each loaded from its own entry point and disableable by name
in ``plugins.disabled``:

- ``ShellToolPlugin`` (``shell_tools``): ``shell`` and ``task_pipeline``.
  ``task_pipeline`` runs each step through the shell, so it rides with
  ``shell``: disabling the plugin removes both (PRD §3.5.3).
- ``FileToolsPlugin`` (``file_tools``): ``read_file`` and ``write_file``.
- ``WebToolsPlugin`` (``web_tools``): ``web_fetch`` and ``web_search``; owns
  the ``aiohttp.ClientSession`` ``web_fetch`` reuses.

Each reads only its own keys from the ``tools`` config block. The tool
implementations live in ``shell.py``, ``files.py`` and ``web.py``; the raw
functions are re-exported here for callers and tests.
"""

import aiohttp

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.spill import NotificationSpill
from corvidae.tool import Tool, ToolContext
from corvidae.tool_collection import ToolCollectionPlugin

from corvidae.tools.shell import shell
from corvidae.tools.files import page_chars, read_file, write_file
from corvidae.tools.web import web_fetch, web_fetch_with_session, web_search


class ShellToolPlugin(CorvidaePlugin):
    """Registers ``shell`` and ``task_pipeline``; owns ``tools.shell_timeout``."""

    # Resolves no other plugin: the shell closure reads only its own timeout.
    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        self.pm = pm  # May be None; on_init will set it via CorvidaePlugin.on_init
        self._shell_timeout: int = 30

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        # Read this plugin's one key from the shared tools block.
        await super().on_init(pm, config)
        self._shell_timeout = config.get("tools", {}).get("shell_timeout", 30)

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        # The tool closures read this plugin's settings live, at call time.
        plugin = self

        async def shell(command: str) -> str:
            """Execute a shell command and return the output."""
            from corvidae.tools.shell import shell as _shell_impl
            return await _shell_impl(command, timeout=plugin._shell_timeout)

        # Declare the timeout shell really uses, read live at dispatch (R4).
        shell.own_timeout = lambda: plugin._shell_timeout  # type: ignore[attr-defined]
        # A shell command can read a notification spill file.
        shell.reads_spill_files = True  # type: ignore[attr-defined]

        # task_pipeline is shell by another name (PRD §3.5.3), so the shell
        # plugin registers it: disabling shell_tools removes both.
        from corvidae.tools.task_pipeline import TaskPipelinePlugin

        TaskPipelinePlugin(self.pm).register_tools(tool_registry)

        tool_registry.append(Tool.from_function(shell))


class FileToolsPlugin(CorvidaePlugin):
    """Registers ``read_file`` and ``write_file``; owns ``tools.max_file_read_bytes``."""

    # read_file resolves the spill service (stale pointers), the tool
    # collection (the live result cap and page fraction) and, optionally,
    # the channel registry (the calling channel's budget).
    depends_on = frozenset({"tools", "spill"})
    optional_depends_on = frozenset({"registry"})

    def __init__(self, pm=None) -> None:
        self.pm = pm  # May be None; on_init will set it via CorvidaePlugin.on_init
        self._max_file_read_bytes: int = 1024 * 1024

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        # Read this plugin's one key from the shared tools block.
        await super().on_init(pm, config)
        self._max_file_read_bytes = config.get("tools", {}).get(
            "max_file_read_bytes", 1024 * 1024
        )

    def _channel_budget(self, ctx: ToolContext | None) -> int | None:
        """The calling channel's resolved max_context_tokens, or None."""
        # No context or no channel means no budget to size a page by.
        if ctx is None or ctx.channel is None:
            return None
        # The registry is optional: without it the budget is unknown.
        registry = self.pm.get_plugin("registry")
        if registry is None:
            return None
        return registry.resolve_config(ctx.channel)["max_context_tokens"]

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        # The tool closures read this plugin's settings live, at call time.
        plugin = self

        async def read_file(
            path: str,
            offset: int | None = None,
            length: int | None = None,
            start_line: int | None = None,
            end_line: int | None = None,
            _ctx: ToolContext | None = None,
        ) -> str:
            """Read a file. offset and length select a character range; start_line and end_line (1-based, inclusive) select a line range; use one kind per call. A ranged result's header names what it returned and where to continue."""
            from corvidae.tools.files import read_file as _read_file_impl
            # A spill file past its age is gone; say so rather than "not found".
            spill = get_dependency(plugin.pm, "spill", NotificationSpill)
            stale = spill.stale_pointer_error(path)
            if stale is not None:
                return stale
            # Every read is one page, sized to the calling channel's budget and
            # never above the live result cap, so it is never cut after the fact.
            tools = get_dependency(plugin.pm, "tools", ToolCollectionPlugin)
            budget = plugin._channel_budget(_ctx)
            max_chars = page_chars(
                tools.max_result_chars,
                budget,
                tools.read_file_window_fraction,
            )
            # The share in tokens, which the page is counted against: the
            # character estimate only sizes the first try (#752).
            max_tokens = (
                None if budget is None
                else int(tools.read_file_window_fraction * budget)
            )
            # An unranged read of a spill file pages from the start, so reading
            # one back never yields a result that is cut and spilled again.
            unranged = offset is None and length is None and start_line is None and end_line is None
            if unranged and spill.contains(path):
                offset = 0
            elif unranged:
                return await _read_file_impl(
                    path, max_size=plugin._max_file_read_bytes, max_chars=max_chars,
                    max_tokens=max_tokens,
                )
            return await _read_file_impl(
                path, offset=offset, length=length, max_chars=max_chars,
                start_line=start_line, end_line=end_line, max_tokens=max_tokens,
            )

        # read_file reads notification spill files.
        read_file.reads_spill_files = True  # type: ignore[attr-defined]
        # File contents are outside text, spill files included.
        read_file.returns_untrusted = True  # type: ignore[attr-defined]

        tool_registry.extend([
            Tool.from_function(read_file),
            Tool.from_function(write_file),
        ])


class WebToolsPlugin(CorvidaePlugin):
    """Registers ``web_fetch`` and ``web_search``; owns the HTTP session and
    the ``tools.web_*`` keys."""

    # Resolves no other plugin: both closures read only this plugin's state.
    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        self.pm = pm  # May be None; on_init will set it via CorvidaePlugin.on_init
        self._session: aiohttp.ClientSession | None = None
        self._web_fetch_timeout: int = 15
        self._web_max_response_bytes: int = 50_000
        self._web_search_max_results: int = 8

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        # Read this plugin's keys from the shared tools block.
        await super().on_init(pm, config)
        tools_config = config.get("tools", {})
        self._web_fetch_timeout = tools_config.get("web_fetch_timeout", 15)
        self._web_max_response_bytes = tools_config.get("web_max_response_bytes", 50_000)
        self._web_search_max_results = tools_config.get("web_search_max_results", 8)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # One session for every web_fetch call, bounded by the fetch timeout.
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._web_fetch_timeout)
        )

    @hookimpl
    async def on_stop(self) -> None:
        # Close the session opened in on_start, if it was.
        if self._session is not None:
            await self._session.close()

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        # The tool closures read this plugin's settings live, at call time.
        plugin = self

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

        tool_registry.extend([
            Tool.from_function(web_fetch),
            Tool.from_function(_web_search),
        ])
