"""McpClientPlugin — MCP client that exposes MCP server tools to the agent loop.

Connects to configured MCP servers during on_start, caches the tool list for
synchronous delivery via register_tools, and keeps sessions alive via
AsyncExitStack for the daemon's lifetime.

Design rationale: plans/mcp-client-plugin.md

Config (top-level 'mcp:' key in agent.yaml):
    mcp:
      servers:
        <name>:
          transport: stdio | sse
          command: <executable>         # stdio only
          args: [...]                   # stdio only
          env: {...}                    # stdio only, optional
          url: <endpoint>              # sse only
          tool_prefix: <prefix>        # optional, default: server name
          timeout_seconds: 30          # optional, default: 30
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass

from corvidae.hooks import hookimpl
from corvidae.tool import Tool

logger = logging.getLogger(__name__)


@dataclass
class _McpServerState:
    """Runtime state for one connected MCP server."""

    name: str
    session: object          # mcp.ClientSession
    prefix: str
    timeout_seconds: float
    mcp_tools: list          # list[mcp.types.Tool]


class McpClientPlugin:
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

    depends_on = set()

    def __init__(self, pm) -> None:
        self.pm = pm
        self._servers: list[_McpServerState] = []
        self._cached_tools: list[Tool] = []
        self._exit_stack: AsyncExitStack | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Connect to MCP servers and build tool list."""
        servers_config = config.get("mcp", {}).get("servers", {})
        if not servers_config:
            logger.debug("McpClientPlugin: no servers configured")
            return

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for server_name, server_cfg in servers_config.items():
            try:
                await self._connect_server(server_name, server_cfg)
            except Exception:
                logger.warning(
                    "MCP server failed to connect, skipping",
                    extra={"server": server_name},
                    exc_info=True,
                )

        self._cached_tools = self._build_tool_list()
        logger.info(
            "McpClientPlugin started",
            extra={"servers": len(self._servers), "tools": len(self._cached_tools)},
        )

    @hookimpl
    async def on_stop(self) -> None:
        """Close all MCP sessions and transports via AsyncExitStack."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        logger.debug("McpClientPlugin stopped")

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Append cached Tool instances to tool_registry (sync delivery)."""
        tool_registry.extend(self._cached_tools)

    async def _connect_server(self, name: str, cfg: dict) -> None:
        """Connect to one MCP server and register its tools.

        Enters the transport and session into self._exit_stack so they are
        kept alive for the daemon's lifetime and closed together on on_stop.
        Appends a _McpServerState entry to self._servers on success.

        Args:
            name: Server name from config (used as default tool prefix).
            cfg: Server config dict with transport, command/args/env or url,
                 and optional tool_prefix and timeout_seconds keys.

        Note: The caller (on_start) catches all exceptions from
        this method, logs them as warnings, and skips the server. Exceptions
        that may arise include ValueError (unknown transport) and KeyError
        (missing required config keys like 'command' or 'url').
        """
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.sse import sse_client

        transport = cfg["transport"]
        prefix = cfg.get("tool_prefix", name)
        timeout = float(cfg.get("timeout_seconds", 30))

        if transport == "stdio":
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env") or None,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
        elif transport == "sse":
            read, write = await self._exit_stack.enter_async_context(
                sse_client(cfg["url"])
            )
        else:
            raise ValueError(
                f"Unknown MCP transport {transport!r} for server {name!r}"
            )

        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        result = await session.list_tools()

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

    def _build_tool_list(self) -> list[Tool]:
        """Build the cached Tool list from all connected servers.

        Deduplicates by tool name: if two servers expose a tool with the same
        prefixed name, the second occurrence is skipped and a warning is logged.
        """
        tools = []
        seen: dict[str, str] = {}  # tool_name -> server_name
        for server in self._servers:
            for mcp_tool in server.mcp_tools:
                tool = _make_tool(server, mcp_tool)
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


def _make_tool(server: _McpServerState, mcp_tool) -> Tool:
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
        return await _call_mcp_tool(session, original_name, kwargs, timeout, server_name)

    tool_fn.__name__ = tool_name
    return Tool(name=tool_name, fn=tool_fn, schema=schema)


# Top-level JSON Schema keys that some MCP servers emit but the
# OpenAI function-calling API rejects.
_UNSAFE_SCHEMA_KEYS = frozenset({"$schema", "$id", "$comment", "$defs", "definitions"})


def _mcp_tool_to_schema(tool_name: str, mcp_tool) -> dict:
    """Translate MCP tool to Corvidae's OpenAI function-call schema.

    MCP's inputSchema is already a JSON Schema dict — same format Corvidae
    uses for function parameters. Strips known-unsafe top-level keys before
    wrapping it in the OpenAI envelope.

    Design note: Sanitization is shallow (top-level only). Deeply nested
    $schema/$id in sub-schemas are unlikely to cause API rejections since
    they're inside 'properties'. If this assumption proves wrong, deep
    sanitization would be needed.
    """
    parameters = dict(mcp_tool.inputSchema or {"type": "object", "properties": {}})
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
        return f"Error: MCP tool '{original_name}' failed: {exc}"

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
    if result.isError:
        return "Error: " + text
    return text
