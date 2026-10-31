"""McpClientPlugin — MCP client that exposes MCP server tools to the agent loop.

Connects to configured MCP servers during on_start, caches the tool list for
synchronous delivery via register_tools, and keeps each server's session alive
in a long-lived owner task for the daemon's lifetime.

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
from dataclasses import dataclass, field

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


def _exc_type_names(exc: BaseException) -> str:
    """Name an exception's type, looking inside anyio/asyncio exception
    groups so the log says which error actually occurred."""
    if isinstance(exc, BaseExceptionGroup):
        return ", ".join(sorted({_exc_type_names(e) for e in exc.exceptions}))
    return type(exc).__name__


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


# Upper bound on how long on_stop waits for owner tasks to close their
# transports. Runtime gives the whole stop broadcast 3 s, so stay under it.
_STOP_BOUND_SECONDS = 2.5
# After cancelling a straggler, how long to wait for it to unwind.
_CANCEL_GRACE_SECONDS = 0.3


@dataclass
class _ServerOwner:
    """The long-lived task that owns one server's transport and session.

    anyio requires a cancel scope to exit in the task that entered it, and
    on_start and on_stop run in different tasks, so everything a server opens
    is entered and exited inside ``task``. ``stop`` tells it to close up.
    """

    name: str
    stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


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
        on_start  — connects to MCP servers, each in its own owner task,
                    fetches tool lists, builds cached Tool list.
        register_tools — extends tool_registry with cached Tool instances (sync)
        on_stop   — signals every owner task to close its session and
                    transport, and awaits them together within a bound

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
        # One owner task per connected server (name -> owner).
        self._owners: dict[str, _ServerOwner] = {}
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

        self._pending_servers = {}

        try:
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
                # for a lazy reconnect rather than skipped for the process
                # lifetime.
                if not any(server.name == server_name for server in self._servers):
                    self._pending_servers[server_name] = server_cfg
        except asyncio.CancelledError:
            # Interrupted mid-connect: the servers already connected have
            # owner tasks that nothing else will stop, so close them before
            # the cancellation continues.
            await self._stop_owners()
            raise

        self._cached_tools = self._build_tool_list()
        logger.info(
            "McpClientPlugin started",
            extra={"servers": len(self._servers), "tools": len(self._cached_tools)},
        )

    @hookimpl
    async def on_stop(self) -> None:
        """Close every MCP session and transport via its owner task."""
        await self._stop_owners()
        logger.debug("McpClientPlugin stopped")

    async def _stop_owners(self) -> None:
        """Signal all owner tasks to exit and await them together.

        Concurrent, not one after another: the SDK's stdio_client can take
        about 2 s per server to terminate a process, and Runtime allows the
        whole stop broadcast 3 s. An owner that has not finished within
        _STOP_BOUND_SECONDS is cancelled and logged. Never raises (a
        cancellation of this coroutine itself excepted).
        """
        owners = list(self._owners.values())
        tasks = [o.task for o in owners if o.task is not None]
        # Tell every owner to close its transport at the same moment.
        for owner in owners:
            owner.stop.set()
        try:
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=_STOP_BOUND_SECONDS)
            else:
                pending = set()
            # Cancel stragglers and give them a moment to unwind.
            for owner in owners:
                if owner.task in pending:
                    logger.warning(
                        "MCP server %r did not shut down within %ss and was "
                        "cancelled; the server process may need to be killed "
                        "by hand",
                        owner.name, _STOP_BOUND_SECONDS,
                        extra={"server": owner.name},
                    )
                    owner.task.cancel()
            if pending:
                await asyncio.wait(pending, timeout=_CANCEL_GRACE_SECONDS)
        except asyncio.CancelledError:
            # on_stop itself was cancelled (stop budget spent): cancel the
            # owners so none outlives the daemon, then let it propagate.
            for task in tasks:
                task.cancel()
            raise
        finally:
            self._servers.clear()
            self._owners.clear()
            self._cached_tools = []
        # Any owner that died with an exception its own task did not log.
        for owner in owners:
            task = owner.task
            if task is None or not task.done() or task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                logger.warning(
                    "MCP server %r owner task failed (%s); the server "
                    "process may need to be killed by hand",
                    owner.name, _exc_type_names(exc),
                    extra={"server": owner.name},
                    exc_info=exc,
                )

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

        Starts a per-server owner task that enters the transport and session
        and runs the handshake within connect_timeout_seconds (default 30),
        then waits for it to report. On success the owner stays alive for the
        daemon's lifetime and closes its own resources when on_stop signals
        it, so enter and exit happen in the same task; on failure or timeout
        the owner closes everything at once (terminating a stdio subprocess)
        and this method returns only after that teardown has finished.
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
        connect_timeout_seconds is logged by the owner and returns without
        registering the server.
        """
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

        # The owner task opens and (later) closes everything this server
        # needs; `ready` carries its handshake result back to this task.
        owner = _ServerOwner(name=name)
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        owner.task = asyncio.create_task(
            self._run_owner(owner, cfg, headers, connect_timeout, ready),
            name=f"mcp-owner-{name}",
        )
        try:
            state = await ready
        except BaseException:
            # Failed, or this task was cancelled mid-connect: make sure the
            # owner has cancelled and finished its teardown before leaving,
            # so no subprocess outlives the failure.
            owner.task.cancel()
            await asyncio.wait([owner.task])
            raise
        if state is None:
            # Timed out; the owner logged it and closed everything.
            await asyncio.wait([owner.task])
            return

        # Connected: the owner stays alive until on_stop signals it.
        self._owners[name] = owner
        self._servers.append(state)
        logger.info(
            "MCP server connected",
            extra={"server": name, "transport": cfg["transport"],
                   "tools": len(state.mcp_tools)},
        )

    async def _run_owner(
        self,
        owner: _ServerOwner,
        cfg: dict,
        headers: dict[str, str] | None,
        connect_timeout: float,
        ready: asyncio.Future,
    ) -> None:
        """Body of a server's owner task: connect, serve, close.

        Resolves ``ready`` with the server's _McpServerState (or None after
        a connect timeout, or the connect exception), then waits for
        ``owner.stop`` and closes the server's resources in this same task.
        Teardown errors are logged here, not raised, so they cannot break
        on_stop's other servers.
        """
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        name = owner.name
        stack = owner.stack
        transport = cfg["transport"]
        deadline = asyncio.timeout(connect_timeout)
        try:
            # Bound the whole handshake inside this task: a server that never
            # answers initialize (or list_tools) would otherwise block the
            # on_start broadcast, and with it daemon startup, forever (#72).
            async with deadline:
                if transport == "stdio":
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=cfg.get("env") or None,
                    )
                    read, write = await stack.enter_async_context(
                        stdio_client(params)
                    )
                elif transport == "sse":
                    read, write = await stack.enter_async_context(
                        sse_client(cfg["url"], headers=headers)
                    )
                elif transport == "streamable_http":
                    # The SDK closes only an HTTP client it created itself, so
                    # ours goes on the server's stack and closes with it.
                    http_client = await stack.enter_async_context(
                        create_mcp_http_client(headers=headers)
                    )
                    read, write = await stack.enter_async_context(
                        streamable_http_client(cfg["url"], http_client=http_client)
                    )
                else:
                    raise ValueError(
                        f"Unknown MCP transport {transport!r} for server {name!r}"
                    )

                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                result = await session.list_tools()
        except BaseException as exc:
            # Runs outside the timeout block, so the cancel the timeout used
            # has already become TimeoutError and teardown is not cancelled.
            await self._close_failed_server(name, stack)
            # Our own bound expired: log it once, without the traceback of
            # the cancelled handshake, and report "not registered". Any other
            # failure (including a TimeoutError raised inside the handshake)
            # goes to _connect_server and on to on_start's logged skip.
            if isinstance(exc, TimeoutError) and deadline.expired():
                logger.warning(
                    "MCP server connect timed out after %ss; unavailable",
                    connect_timeout,
                    extra={"server": name},
                )
                if not ready.done():
                    ready.set_result(None)
                return
            if not ready.done():
                if isinstance(exc, asyncio.CancelledError):
                    ready.cancel()
                else:
                    ready.set_exception(exc)
            if isinstance(exc, asyncio.CancelledError):
                raise
            return

        # Connected: report to _connect_server, then hold the transport open
        # until on_stop signals. The close runs in this task, the one that
        # entered the cancel scopes.
        ready.set_result(_McpServerState(
            name=name,
            session=session,
            prefix=cfg.get("tool_prefix", name),
            timeout_seconds=float(cfg.get("timeout_seconds", 30)),
            mcp_tools=result.tools,
        ))
        try:
            await owner.stop.wait()
        finally:
            try:
                await stack.aclose()
            # Transport cleanup raises types outside this codebase's control
            # (anyio, mcp); one server's failure must not break the others'
            # shutdown, so log it with the server and exception type.
            except Exception as exc:
                logger.warning(
                    "MCP server %r raised %s while shutting down; the server "
                    "process may need to be killed by hand",
                    name, _exc_type_names(exc),
                    extra={"server": name},
                    exc_info=True,
                )

    async def _close_failed_server(self, name: str, stack: AsyncExitStack) -> None:
        """Tear down a server whose connect failed, logging (never raising)
        cleanup errors so they cannot mask the connect failure itself."""
        try:
            await stack.aclose()
        # Transport cleanup raises types outside this codebase's control
        # (anyio, mcp), as in the owner's shutdown.
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
