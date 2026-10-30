"""McpClientPlugin — MCP client that exposes MCP server tools to the agent loop.

Connects to configured MCP servers during on_start, caches the tool list for
synchronous delivery via register_tools, and keeps sessions alive via
AsyncExitStack for the daemon's lifetime.

Design rationale: plans/mcp-client-plugin.md

Config (top-level 'mcp:' key in agent.yaml):
    mcp:
      servers:
        <name>:
          transport: stdio | sse | streamable_http
          command: <executable>         # stdio only
          args: [...]                   # stdio only
          env: {...}                    # stdio only, optional
          url: <endpoint>              # sse and streamable_http
          headers: {...}               # sse and streamable_http, optional;
                                       # values may use ${VAR} from the environment
          tool_prefix: <prefix>        # optional, default: server name
          timeout_seconds: 30          # optional, default: 30 (per tool call)
          connect_timeout_seconds: 30  # optional, default: 30 (connect + initialize + list_tools)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tool import Tool

logger = logging.getLogger(__name__)

# ${VAR} references in header values, substituted from the environment.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class McpHeaderEnvError(ValueError):
    """A header value names an environment variable that is unset or empty.

    A config error rather than a connect failure: the daemon's environment
    cannot change under it, so retrying the server would fail the same way.
    """


def _resolve_headers(name: str, headers: dict | None) -> dict[str, str] | None:
    """Return ``headers`` with every ${VAR} replaced from the environment.

    Raises McpHeaderEnvError naming the server, header and variable when a
    referenced variable is unset, empty, or contains CR, LF or NUL. The
    message never includes a header value, so a secret cannot reach the log
    through it.
    """
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for header, value in headers.items():
        # Substitute each ${VAR}; an unset, empty or multi-line one is a
        # config error.
        def _lookup(match: re.Match, header: str = header) -> str:
            var = match.group(1)
            env_value = os.environ.get(var)
            if not env_value:
                raise McpHeaderEnvError(
                    f"MCP server {name!r}: header {header!r} references "
                    f"${{{var}}}, which is not set in the daemon's environment; "
                    f"set {var} in the environment corvidae runs in and restart. "
                    "Server skipped."
                )
            # CR, LF or NUL is illegal in a header, and httpx's error for it
            # quotes the value into a logged traceback; refuse it here.
            if any(ch in env_value for ch in "\r\n\0"):
                raise McpHeaderEnvError(
                    f"MCP server {name!r}: header {header!r} references "
                    f"${{{var}}}, whose value contains a newline or NUL "
                    f"(often a trailing newline); fix {var} in the daemon's "
                    "environment and restart. Server skipped."
                )
            return env_value

        resolved[str(header)] = _ENV_REF.sub(_lookup, str(value))
    return resolved


@dataclass
class _McpServerState:
    """Runtime state for one connected MCP server."""

    name: str
    session: object          # mcp.ClientSession
    prefix: str
    timeout_seconds: float
    mcp_tools: list          # list[mcp.types.Tool]


class McpClientPlugin(CorvidaePlugin):
    """Plugin that connects to MCP servers and exposes their tools to Corvidae.

    Lifecycle:
        on_start  — connects to MCP servers via AsyncExitStack, fetches tool
                    lists, builds cached Tool list.
        register_tools — extends tool_registry with cached Tool instances (sync)
        on_stop   — closes all sessions and transports via AsyncExitStack.aclose()

    Ordering: main.py calls Agent.on_start explicitly after the
    broadcast, so MCP connections are guaranteed to complete before
    register_tools fires.
    """

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._servers: list[_McpServerState] = []
        self._cached_tools: list[Tool] = []
        self._exit_stack: AsyncExitStack | None = None
        self._servers_config: dict = {}
        # Configs for servers that did not connect at startup, retried on the
        # next MCP tool call (name -> config).
        self._pending_servers: dict[str, dict] = {}

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self._servers_config = config.get("mcp", {}).get("servers", {})

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Connect to MCP servers and build tool list."""
        servers_config = self._servers_config
        if not servers_config:
            logger.debug("McpClientPlugin: no servers configured")
            return

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        self._pending_servers = {}

        for server_name, server_cfg in servers_config.items():
            try:
                await self._connect_server(server_name, server_cfg)
            except McpHeaderEnvError as exc:
                # A config error: report it once, with the remedy, and skip
                # the server for good (no lazy retry; see McpHeaderEnvError).
                logger.warning("%s", exc, extra={"server": server_name})
                continue
            except Exception:
                logger.warning(
                    "MCP server failed to connect, skipping",
                    extra={"server": server_name},
                    exc_info=True,
                )
            # A server that did not register (exception or timeout) is kept
            # for a lazy reconnect rather than skipped for the process lifetime.
            if not any(server.name == server_name for server in self._servers):
                self._pending_servers[server_name] = server_cfg

        self._cached_tools = self._build_tool_list()
        logger.info(
            "McpClientPlugin started",
            extra={"servers": len(self._servers), "tools": len(self._cached_tools)},
        )

    @hookimpl
    async def on_stop(self) -> None:
        """Close all MCP sessions and transports via AsyncExitStack."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            # AsyncExitStack.aclose() aggregates cleanup from multiple MCP
            # transports whose exception types are not under this codebase's
            # control, so narrowing the catch to specific types is not feasible.
            except Exception:
                logger.warning(
                    "McpClientPlugin: error during exit stack cleanup", exc_info=True
                )
            finally:
                self._exit_stack = None
        logger.debug("McpClientPlugin stopped")

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Append cached Tool instances to tool_registry (sync delivery)."""
        tool_registry.extend(self._cached_tools)

    async def _retry_pending_servers(self) -> None:
        """Reconnect servers that failed at startup, on demand.

        Called from every MCP tool call. Successful reconnects refresh the
        cached tool list and the shared registry.

        shortcut: retries need a working MCP tool to trigger them, so a
        daemon whose servers all failed at startup never retries; upgrade to
        a background retry loop if all-server outage recovery matters.
        """
        if not self._pending_servers:
            return

        reconnected = False
        for name, cfg in list(self._pending_servers.items()):
            try:
                await self._connect_server(name, cfg)
            except Exception:
                logger.warning(
                    "MCP server lazy reconnect failed",
                    extra={"server": name},
                    exc_info=True,
                )
                continue
            if any(server.name == name for server in self._servers):
                self._pending_servers.pop(name, None)
                reconnected = True
            # A timeout leaves the server pending for the next tool call.

        if reconnected:
            self._cached_tools = self._build_tool_list()
            # Refresh the shared registry so later tool consumers (e.g.
            # subagents) see the reconnected server's tools.
            tools_plugin = self.pm.get_plugin("tools") if self.pm is not None else None
            rebuild = getattr(tools_plugin, "rebuild_registry", None)
            if inspect.iscoroutinefunction(rebuild):
                await rebuild()

    async def _connect_server(self, name: str, cfg: dict) -> None:
        """Connect to one MCP server and register its tools.

        Enters the transport and session into a per-server exit stack, all
        within connect_timeout_seconds (default 30). On success that stack is
        handed to self._exit_stack, so the server stays alive for the
        daemon's lifetime and closes with the rest on on_stop; on failure or
        timeout it is closed at once, terminating a stdio subprocess.
        Appends a _McpServerState entry to self._servers on success.

        Args:
            name: Server name from config (used as default tool prefix).
            cfg: Server config dict with transport, command/args/env or url,
                 and optional headers (HTTP transports), tool_prefix,
                 timeout_seconds and connect_timeout_seconds keys.

        Note: The caller (on_start) catches all exceptions from
        this method, logs them as warnings, and skips the server. Exceptions
        that may arise include ValueError (unknown transport, or a
        non-positive connect_timeout_seconds) and KeyError (missing required
        config keys like 'command' or 'url'). McpHeaderEnvError (a ${VAR}
        in headers that is unset, empty, or holds a newline) is caught
        separately and not retried. A handshake that exceeds
        connect_timeout_seconds is logged here and returns without
        registering the server.
        """
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        transport = cfg["transport"]
        prefix = cfg.get("tool_prefix", name)
        timeout = float(cfg.get("timeout_seconds", 30))
        connect_timeout = float(cfg.get("connect_timeout_seconds", 30))
        # A zero or negative bound would fail every handshake; reject it as
        # a config error (on_start logs it and skips the server).
        if connect_timeout <= 0:
            raise ValueError(
                f"connect_timeout_seconds must be positive for server {name!r}"
            )

        # Resolve ${VAR} references before anything opens: a missing variable
        # raises McpHeaderEnvError, which on_start reports without retrying.
        headers = _resolve_headers(name, cfg.get("headers"))

        # Each server gets its own exit stack so a server that fails or hangs
        # mid-connect can be torn down (subprocess included) on the spot,
        # without touching servers that already connected.
        local_stack = AsyncExitStack()
        deadline = asyncio.timeout(connect_timeout)
        try:
            # Bound the whole handshake: a server that never answers
            # initialize (or list_tools) would otherwise block the on_start
            # broadcast, and with it daemon startup, forever (#72).
            async with deadline:
                if transport == "stdio":
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=cfg.get("env") or None,
                    )
                    read, write = await local_stack.enter_async_context(
                        stdio_client(params)
                    )
                elif transport == "sse":
                    read, write = await local_stack.enter_async_context(
                        sse_client(cfg["url"], headers=headers)
                    )
                elif transport == "streamable_http":
                    # The SDK closes only an HTTP client it created itself, so
                    # ours goes on the server's stack and closes with it.
                    http_client = await local_stack.enter_async_context(
                        create_mcp_http_client(headers=headers)
                    )
                    read, write = await local_stack.enter_async_context(
                        streamable_http_client(cfg["url"], http_client=http_client)
                    )
                else:
                    raise ValueError(
                        f"Unknown MCP transport {transport!r} for server {name!r}"
                    )

                session = await local_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                result = await session.list_tools()
        except BaseException as exc:
            # Runs outside the timeout block, so the cancel the timeout used
            # has already become TimeoutError and teardown is not cancelled.
            await self._close_failed_server(name, local_stack)
            # Our own bound expired: log it once, without the traceback of
            # the cancelled handshake, and skip the server. Any other
            # failure (including a TimeoutError raised inside the
            # handshake) propagates to on_start's logged skip.
            if isinstance(exc, TimeoutError) and deadline.expired():
                logger.warning(
                    "MCP server connect timed out after %ss; unavailable",
                    connect_timeout,
                    extra={"server": name},
                )
                return
            raise

        # Connected: hand the server's resources to the daemon-lifetime stack
        # so on_stop closes them with everything else.
        self._exit_stack.push_async_callback(local_stack.aclose)

        self._servers.append(_McpServerState(
            name=name,
            session=session,
            prefix=prefix,
            timeout_seconds=timeout,
            mcp_tools=result.tools,
        ))

        logger.info(
            "MCP server connected",
            extra={"server": name, "transport": transport, "tools": len(result.tools)},
        )

    async def _close_failed_server(self, name: str, stack: AsyncExitStack) -> None:
        """Tear down a server whose connect failed, logging (never raising)
        cleanup errors so they cannot mask the connect failure itself."""
        try:
            await stack.aclose()
        # Transport cleanup raises types outside this codebase's control
        # (anyio, mcp), as in on_stop.
        except Exception:
            logger.warning(
                "MCP server cleanup after failed connect raised",
                extra={"server": name},
                exc_info=True,
            )

    def _build_tool_list(self) -> list[Tool]:
        """Build the cached Tool list from all connected servers.

        Deduplicates by tool name: if two servers expose a tool with the same
        prefixed name, the second occurrence is skipped and a warning is logged.
        """
        tools = []
        seen: dict[str, str] = {}  # tool_name -> server_name
        for server in self._servers:
            for mcp_tool in server.mcp_tools:
                tool = _make_tool(server, mcp_tool, self)
                if tool.name in seen:
                    logger.warning(
                        "MCP tool name collision: %r from server %r "
                        "duplicates tool from server %r; skipping duplicate",
                        tool.name, server.name, seen[tool.name],
                    )
                    continue
                seen[tool.name] = server.name
                tools.append(tool)
        return tools


def _make_tool(server: _McpServerState, mcp_tool, plugin=None) -> Tool:
    """Build a Corvidae Tool from one MCP tool definition.

    Captures session, original tool name, and timeout in the closure.
    No _ctx parameter — MCP tools receive only LLM-provided arguments.
    """
    tool_name = (
        f"{server.prefix}__{mcp_tool.name}" if server.prefix else mcp_tool.name
    )
    schema = _mcp_tool_to_schema(tool_name, mcp_tool)

    session = server.session
    original_name = mcp_tool.name
    timeout = server.timeout_seconds
    server_name = server.name

    async def tool_fn(**kwargs) -> str:
        # Give startup-failed servers a chance to come back before the call.
        if plugin is not None:
            await plugin._retry_pending_servers()
        return await _call_mcp_tool(session, original_name, kwargs, timeout, server_name)

    tool_fn.__name__ = tool_name
    return Tool(name=tool_name, fn=tool_fn, schema=schema)


# Top-level JSON Schema keys that some MCP servers emit but the
# OpenAI function-calling API rejects.
_UNSAFE_SCHEMA_KEYS = frozenset({"$schema", "$id", "$comment", "$defs", "definitions"})


def _mcp_tool_to_schema(tool_name: str, mcp_tool) -> dict:
    """Translate MCP tool to Corvidae's OpenAI function-call schema.

    MCP's input_schema is already a JSON Schema dict — same format Corvidae
    uses for function parameters. Strips known-unsafe top-level keys before
    wrapping it in the OpenAI envelope.

    Design note: Sanitization is shallow (top-level only). Deeply nested
    $schema/$id in sub-schemas are unlikely to cause API rejections since
    they're inside 'properties'. If this assumption proves wrong, deep
    sanitization would be needed.
    """
    parameters = dict(mcp_tool.input_schema or {"type": "object", "properties": {}})
    stripped = _UNSAFE_SCHEMA_KEYS & parameters.keys()
    if stripped:
        for key in stripped:
            del parameters[key]
        logger.debug(
            "Stripped unsafe schema keys from MCP tool %r: %s",
            tool_name, ", ".join(sorted(stripped)),
        )
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": mcp_tool.description or "",
            "parameters": parameters,
        },
    }


async def _call_mcp_tool(
    session,
    original_name: str,
    kwargs: dict,
    timeout: float,
    server_name: str,
) -> str:
    """Execute one MCP tool call and return the result as a string."""
    try:
        result = await asyncio.wait_for(
            session.call_tool(original_name, arguments=kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return (
            f"Error: MCP tool '{original_name}' on server '{server_name}' "
            f"timed out after {timeout}s"
        )
    except Exception as exc:
        logger.warning(
            "MCP tool call raised exception",
            extra={"server": server_name, "tool": original_name},
            exc_info=True,
        )
        return (
            f"Error: MCP tool '{original_name}' on server '{server_name}' "
            f"failed: {exc}"
        )

    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        else:
            logger.debug(
                "MCP tool returned non-text content block, skipping",
                extra={
                    "server": server_name,
                    "tool": original_name,
                    "block_type": type(block).__name__,
                },
            )

    text = "\n".join(parts) if parts else "(no output)"
    if result.is_error:
        return "Error: " + text
    return text
