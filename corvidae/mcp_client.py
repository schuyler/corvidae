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
import math
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from corvidae.clock import Clock
from corvidae.envref import EnvRefError, expand_env_refs
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.shutdown import parse_shutdown_timeouts
from corvidae.tool import Tool

logger = logging.getLogger(__name__)


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


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """The exceptions at the bottom of a (possibly nested) exception group."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for member in exc.exceptions for leaf in _leaf_exceptions(member)]
    return [exc]


def _is_connect_failure(exc: BaseException) -> bool:
    """True when ``exc`` from a connect is a failed connect to skip and retry.

    False means it must propagate. One rule serves ``on_start`` and
    ``_retry_round`` so the two cannot drift apart (#596):

    - An ``Exception`` is a failed connect.
    - A ``CancelledError`` is a failed connect only when nobody is cancelling
      this task; a cancel of the task itself is a shutdown (#385).
    - A ``BaseExceptionGroup`` follows its leaves: any leaf that is neither an
      ``Exception`` nor a ``CancelledError`` (``KeyboardInterrupt``,
      ``SystemExit``) propagates, and a group of cancels is judged as a bare
      cancel is. A group is not an ``Exception`` when a member is a
      ``BaseException``, which is how it slipped past the old handlers.
    """
    if isinstance(exc, Exception):
        return True
    leaves = _leaf_exceptions(exc)
    if any(not isinstance(leaf, (Exception, asyncio.CancelledError)) for leaf in leaves):
        return False
    task = asyncio.current_task()
    return not (task is not None and task.cancelling())


def _failure_log_parts(exc: BaseException, cfg: dict) -> tuple[str, bool]:
    """How a connect-failure warning reports ``exc``: (message suffix, exc_info).

    A server whose configured URL carries credentials (a userinfo or a query
    string) gets no traceback: the HTTP client quotes the URL in its errors --
    httpx2's ``HTTPStatusError`` reads ``Client error '401 …' for url
    '<url>'`` -- so the rendered record would carry the secret (#386). The
    warning instead ends with the exception's own text, every URL in it
    redacted like the endpoint, which keeps the status the operator needs.

    Any other server keeps its full traceback and no suffix. Never raises.
    """
    url = cfg.get("url")
    credentialed = (
        cfg.get("transport") != "stdio"
        and isinstance(url, str)
        and ("@" in url or "?" in url)
    )
    if not credentialed:
        return "", True
    return (
        f": {_redact_urls_in_text(_exc_text(exc))} "
        "(traceback omitted: the configured URL carries credentials)",
        False,
    )


def _exc_text(exc: BaseException) -> str:
    """An exception's message on one line, reading through exception groups
    to the errors inside them, since a group's own text names none.

    Never raises: it runs inside failure logging.
    """
    try:
        if isinstance(exc, BaseExceptionGroup):
            return "; ".join(_exc_text(e) for e in exc.exceptions)
        return " ".join(str(exc).split())
    except Exception:
        # A __str__ that raises, or a pathologically deep group. The type is
        # already in the warning, so a placeholder loses little.
        logger.warning(
            "could not read the text of %s for the MCP failure log",
            type(exc).__name__,
        )
        return "(message unavailable)"


def _redact_urls_in_text(text: str) -> str:
    """``text`` with every URL in it reduced as ``_describe_string`` would.

    This catches the configured URL in whatever form a library reports it --
    httpx2 percent-encodes the URL it puts in an error, so ``token=SEC RET1``
    arrives as ``token=SEC%20RET1`` and an exact-substring match would miss
    it (#386 review). A URL that cannot be read is replaced entirely. Every
    match contains "://", so ``_describe_string`` always redacts it rather
    than echoing it.
    """
    return _URL_IN_TEXT.sub(lambda m: _describe_string(m.group(0)), text)


# Every character ``str.splitlines`` treats as a line boundary, mapped to its
# percent-encoded UTF-8 form. A logged URL must not carry one: a log viewer or
# `grep` that splits on it would read one warning as two lines (#588).
_LINE_BOUNDARY_ENCODING = {
    ord(char): "".join(f"%{byte:02X}" for byte in char.encode("utf-8"))
    for char in "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
}


def _one_line(text: str) -> str:
    """``text`` with every line-boundary character percent-encoded.

    Encoded rather than dropped, so the operator can still see that the
    configured value held an odd character. Never raises.
    """
    return text.translate(_LINE_BOUNDARY_ENCODING)


def _server_endpoint(cfg: dict) -> str:
    """The server's command (stdio) or url (sse, streamable_http), for logs.

    Called from failure logging, so it must not raise: a retry-round config
    that fails validation may lack the key, and then the endpoint is "?".

    An HTTP endpoint is logged without its credentials (#386). A server may
    carry a token in its query string or a password in its userinfo, and this
    runs on every failed connect and every retry, so logging the URL whole
    would write that secret to the daemon log repeatedly. Scheme, host, port
    and path survive, which is what tells an operator which endpoint failed;
    a configured query is shown as "?…" so its presence is still visible.
    """
    key = "command" if cfg.get("transport") == "stdio" else "url"
    if key == "command":
        return _one_line(str(cfg.get(key, "?")))
    # A config that failed validation may lack the key entirely; keep the
    # historical "?" so the log still says the endpoint was unreadable.
    if key not in cfg:
        return "?"
    value = cfg[key]
    # A string with no credential marker (a schemeless `localhost:8080`) has
    # no position a credential could occupy, and is the typo the operator
    # must see, so it is shown as configured.
    if isinstance(value, str) and value and not any(
        marker in value for marker in _CREDENTIAL_MARKERS
    ):
        # Shown as configured, except for a line-boundary character (#588).
        return _one_line(value)
    return _redact_url(value)


def _redact_url(value: object) -> str:
    """A URL safe to log: scheme, host, port and path only.

    Never raises. Redaction runs inside failure logging, so a value it cannot
    parse becomes a constant rather than an exception that would replace the
    failure being reported. Both call sites evaluate this *inside* an except
    handler, so a raise here would escalate a failed connect into a startup
    failure (#386 review).

    Returns ``_UNPARSEABLE`` for anything that is not a URL the client could
    use, which is also how callers tell a URL from other config values.
    """
    try:
        parts = urlsplit(str(value))
        # `hostname` and `port` raise ValueError on a malformed authority
        # (`host:abc`, an out-of-range port), so they are read inside the try.
        host = parts.hostname
        port = parts.port
    except ValueError:
        return _UNPARSEABLE
    # A value with no scheme is not a URL the client can use; logging it whole
    # would put whatever it is into the log unredacted.
    if not parts.scheme or not parts.netloc or not host:
        return _UNPARSEABLE
    # Drop userinfo by rebuilding netloc from host and port alone. An IPv6
    # literal keeps its brackets, or the colons read as a port separator.
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    # A separator in the path or host must not put a second line in the log (#588).
    return _one_line(
        f"{parts.scheme}://{netloc}{parts.path}{'?…' if parts.query else ''}"
    )


def _describe_config_value(
    value: object, secret_bearing: bool = False, _depth: int = 0
) -> str:
    """A rejected config value, in a form safe to put in a log line.

    The invariant, rather than a list of shapes:

        No string containing ``@``, ``?`` or ``://`` is echoed into a message.

    Those three markers are userinfo, query and scheme -- the positions a
    credential can occupy. A string without them holds nothing secret, and is
    echoed because a wrong value is what tells the operator what to fix.

    An earlier version enumerated shapes instead, and four review rounds each
    found one it missed: a list, a bare string, a URL inside a container, and
    a URL ``urlsplit`` rejects. The marker test is total, so a new shape
    cannot defeat it (#386 review).

    Containers are described element-wise, so the invariant holds wherever a
    URL is nested, including as a dict key. ``secret_bearing`` marks the key
    whose whole value is a credential by definition -- ``headers`` -- which is
    summarised by type and size whatever shape it arrived in.

    Never raises and never recurses without bound: this runs inside
    config-error handling, so it must not replace the error it describes.
    """
    if _depth >= _MAX_DESCRIBE_DEPTH:
        return f"{type(value).__name__} (nested too deeply to show)"
    if secret_bearing and not isinstance(value, (int, float, bool, type(None))):
        if isinstance(value, str):
            return f"str of length {len(value)}"
        return f"{type(value).__name__}"
    if isinstance(value, str):
        return _describe_string(value)
    if isinstance(value, dict):
        # A nested `headers` value is a credential wherever it sits -- a
        # list-shaped server entry or `servers` block reaches it here rather
        # than through the caller's own secret_bearing flag (#386 review).
        inner = ", ".join(
            f"{_describe_config_value(k, _depth=_depth + 1)}: "
            f"{_describe_config_value(v, secret_bearing=k == 'headers', _depth=_depth + 1)}"
            for k, v in list(value.items())[:4]
        )
        more = ", …" if len(value) > 4 else ""
        return "{" + inner + more + "}"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(
            _describe_config_value(v, _depth=_depth + 1)
            for v in list(value)[:4]
        )
        more = ", …" if len(value) > 4 else ""
        return "[" + inner + more + "]"
    # Anything else is described by type: a repr can hold a credential, and
    # the key in the message is the actionable part.
    if isinstance(value, (int, float, bool, type(None))):
        return repr(value)
    return type(value).__name__


def _describe_string(value: str) -> str:
    """A string for a log line: redacted if it could hold a credential.

    See ``_describe_config_value`` for the invariant. A string with none of
    the credential markers is returned as its repr, which is the diagnostic.
    """
    if not any(marker in value for marker in _CREDENTIAL_MARKERS):
        return repr(value)
    # It could hold a credential. A URL we can read keeps the parts that
    # identify the endpoint; anything else is replaced entirely, because we
    # cannot tell which part was the secret.
    redacted = _redact_url(value)
    return redacted if redacted != _UNPARSEABLE else "(credentials removed)"


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
        try:
            resolved[str(header)] = expand_env_refs(str(value))
        except EnvRefError as exc:
            if exc.reason == "multiline":
                raise McpHeaderEnvError(
                    f"MCP server {name!r}: header {_describe_config_value(header)} references "
                    f"${{{exc.var}}}, whose value contains a newline or NUL "
                    f"(often a trailing newline); fix {exc.var} in the daemon's "
                    "environment and restart. Server skipped."
                ) from None
            raise McpHeaderEnvError(
                f"MCP server {name!r}: header {_describe_config_value(header)} references "
                f"${{{exc.var}}}, which is not set in the daemon's environment; "
                f"set {exc.var} in the environment corvidae runs in and restart. "
                "Server skipped."
            ) from None
    return resolved


# Upper bound on how long on_stop waits for owner tasks to close their
# transports. Runtime gives the whole stop broadcast daemon.shutdown_timeout
# (3 s by default), so this is the MCP client's share of it (#760): the
# graceful budget less a reserve for the cancel grace below and for the rest
# of the broadcast (other plugins' on_stop, idle.stop, clock.close). At the
# default 3 s that is the historical 2.5 s.
_STOP_RESERVE_SECONDS = 0.5
# Smallest bound worth handing to asyncio.wait: a value that rounds to zero
# turns the wait into a poll, and a config value can be arbitrarily small.
_MIN_STOP_BOUND_SECONDS = 0.05
# Fallback for when the config cannot be read (on_stop without on_init).
_DEFAULT_STOP_BOUND_SECONDS = 2.5
# After cancelling a straggler, how long to wait for it to unwind.
_CANCEL_GRACE_SECONDS = 0.3


def stop_bound_seconds(shutdown_timeout: float) -> float:
    """MCP's share of the graceful phase (``daemon.shutdown_timeout``).

    A reserve rather than a fraction, because the cancel grace is spent
    *after* the bound: a bound of 80% of a 1 s budget would cancel at 0.8 s
    and then wait up to 0.3 s more, which is the overrun this bound exists to
    prevent (#760). Never zero, so a tiny budget still gets a real timeout.
    """
    return max(_MIN_STOP_BOUND_SECONDS, shutdown_timeout - _STOP_RESERVE_SECONDS)


# Transports a server config may name.
_TRANSPORTS = ("stdio", "sse", "streamable_http")
# What _redact_url returns for a value it cannot read as a URL. Also how
# callers tell a URL-shaped config value from any other kind.
_UNPARSEABLE = "(unparseable endpoint)"
# How deep _describe_config_value will descend before summarising. Config
# nesting is shallow; the bound exists so a pathological value cannot raise
# RecursionError inside config-error handling.
_MAX_DESCRIBE_DEPTH = 8
# The positions a credential can occupy in a string: userinfo, query, scheme.
# A string with none of these is safe to echo.
_CREDENTIAL_MARKERS = ("@", "?", "://")
# A URL inside free text such as an exception message: a scheme, "://", and
# everything up to whitespace or a quote/bracket that commonly delimits it.
# The lookbehind starts a match only at the start of a scheme-character run;
# without it a long run with no "://" backtracks quadratically (100k chars
# took ~7 s on the event loop).
_URL_IN_TEXT = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+")
# Reconnect backoff ladder for servers that failed at startup; the same
# 10 s doubling to 300 s as the IRC, Signal and Discord transports.
_INITIAL_BACKOFF = 10
_BACKOFF_MULTIPLIER = 2
_BACKOFF_CAP = 300


@dataclass
class _ServerSettings:
    """A server's validated config values, read before anything opens."""

    transport: str
    prefix: str
    timeout: float
    connect_timeout: float
    headers: dict[str, str] | None


def _as_seconds(name: str, cfg: dict, key: str) -> float:
    """Read a numeric seconds key (default 30), naming key and value on error.

    NaN and infinity are rejected too: NaN would read as a connect failure
    and be retried forever, and an infinite bound is no bound.
    """
    value = cfg.get(key, 30)
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"mcp.servers.{name}.{key} must be a number, got "
            f"{_describe_config_value(value)}"
        ) from None
    # A non-finite number would never time out or always fail.
    if not math.isfinite(seconds):
        raise ValueError(
            f"mcp.servers.{name}.{key} must be a finite number, got "
            f"{_describe_config_value(value)}"
        )
    return seconds


def validate_mcp_config(config: dict) -> dict:
    """Return the ``mcp.servers`` mapping of ``config``, validated.

    A missing or null ``mcp`` block, or a null ``servers``, means no servers.
    Raises ValueError naming the key and the rejected value when ``mcp`` is
    not a mapping or ``mcp.servers`` is neither null nor a mapping, so
    ``Runtime.start()`` can report it as a ``Config error`` like any other
    malformed block instead of an AttributeError traceback.
    """
    # mcp: null (or absent) configures nothing; otherwise it must be a mapping.
    block = config.get("mcp")
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(
            "mcp must be a mapping with a servers key, got "
            f"{_describe_config_value(block)}"
        )
    # servers: null configures nothing; otherwise a mapping of name to entry.
    servers = block.get("servers")
    if servers is None:
        return {}
    if not isinstance(servers, dict):
        raise ValueError(
            f"mcp.servers must be a mapping of server names to settings, "
            f"got {_describe_config_value(servers)}"
        )
    return servers


def _parse_server_config(name: str, cfg: dict) -> _ServerSettings:
    """Validate one server's config and resolve its ${VAR} headers.

    Raises McpHeaderEnvError, ValueError or KeyError for a config error (any
    other exception is also treated as one by on_start):
    the daemon cannot fix these by retrying, so callers skip the server for
    good. Raised before any transport opens, which keeps a config error
    distinguishable from a connect failure (which is retried). Messages name
    the server and the key to fix.
    """
    # A non-mapping server entry (e.g. `a: null`) has no keys to read.
    if not isinstance(cfg, dict):
        raise ValueError(
            f"mcp.servers.{name} must be a mapping, got "
            f"{_describe_config_value(cfg)}; "
            f"give it transport and command/url keys"
        )
    transport = cfg["transport"]
    # An unknown transport can never connect.
    if transport not in _TRANSPORTS:
        raise ValueError(
            f"unknown transport {_describe_config_value(transport)} "
            f"for MCP server {name!r}; "
            f"set mcp.servers.{name}.transport to one of {', '.join(_TRANSPORTS)}"
        )
    # Each transport needs its endpoint key (KeyError names it), as a string.
    endpoint_key = "command" if transport == "stdio" else "url"
    endpoint = cfg[endpoint_key]
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError(
            f"mcp.servers.{name}.{endpoint_key} must be a non-empty string, "
            f"got {_describe_config_value(endpoint)}"
        )
    # args is a list of strings and env a mapping; the SDK rejects anything
    # else at connect time, which would be retried forever.
    args = cfg.get("args")
    if args is not None and (
        not isinstance(args, list) or not all(isinstance(a, str) for a in args)
    ):
        raise ValueError(
            f"mcp.servers.{name}.args must be a list of strings, got "
            f"{_describe_config_value(args)}"
        )
    env = cfg.get("env")
    if env is not None and not isinstance(env, dict):
        raise ValueError(
            f"mcp.servers.{name}.env must be a mapping of variable names "
            f"to values, got {_describe_config_value(env)}"
        )
    prefix = cfg.get("tool_prefix", name)
    timeout = _as_seconds(name, cfg, "timeout_seconds")
    connect_timeout = _as_seconds(name, cfg, "connect_timeout_seconds")
    # A zero or negative bound would fail every handshake.
    if connect_timeout <= 0:
        raise ValueError(
            f"connect_timeout_seconds must be positive for server {name!r}; "
            f"got {connect_timeout:g}, fix mcp.servers.{name}.connect_timeout_seconds"
        )
    # headers must be a mapping of names to values.
    raw_headers = cfg.get("headers")
    if raw_headers is not None and not isinstance(raw_headers, dict):
        raise ValueError(
            f"mcp.servers.{name}.headers must be a mapping of header names "
            f"to values, got {_describe_config_value(raw_headers, secret_bearing=True)}"
        )
    # Resolve ${VAR} references: a missing variable raises McpHeaderEnvError.
    headers = _resolve_headers(name, cfg.get("headers"))
    return _ServerSettings(transport, prefix, timeout, connect_timeout, headers)


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

    Ordering: the on_start broadcast runs every implementation
    concurrently, so ToolCollectionPlugin can collect register_tools
    before the connects finish. on_start therefore rebuilds the shared
    registry once its servers are connected (#405), as the retry path
    does for a server that reconnects later. Runtime calls Agent.on_start
    explicitly after the broadcast, so the Agent's first read of the
    registry already includes the startup servers' tools.
    """

    depends_on = frozenset({"clock"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._servers: list[_McpServerState] = []
        self._cached_tools: list[Tool] = []
        # One owner task per connected server (name -> owner).
        self._owners: dict[str, _ServerOwner] = {}
        self._servers_config: dict = {}
        # Configs for servers that did not connect at startup, retried by the
        # background loop and on MCP tool calls (name -> config).
        self._pending_servers: dict[str, dict] = {}
        # Kernel clock (resolved in on_start) and the background retry task.
        self._clock: Clock | None = None
        self._retry_task: asyncio.Task | None = None
        # Serialises retry rounds so the loop and a tool call never connect
        # the same server twice.
        self._retry_lock = asyncio.Lock()
        # Set by on_stop: wakes the loop, and makes a connect that finishes
        # afterwards tear itself down instead of registering an owner.
        self._stop_event = asyncio.Event()
        self._stopping = False
        # Delay the next failed round will be followed by (for the log line).
        self._next_delay: float = _INITIAL_BACKOFF

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        # Raises ValueError for a malformed mcp block (Runtime reports it).
        self._servers_config = validate_mcp_config(config)
        # Derive the stop bound from daemon.shutdown_timeout so an operator
        # who lowers that value doesn't turn every shutdown with a slow stdio
        # MCP server into a forced os._exit(1) mid-broadcast.
        self._stop_bound = stop_bound_seconds(
            parse_shutdown_timeouts(config.get("daemon")).graceful
        )

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Connect to MCP servers and build tool list."""
        servers_config = self._servers_config
        if not servers_config:
            logger.debug("McpClientPlugin: no servers configured")
            return

        self._pending_servers = {}
        self._stopping = False
        self._stop_event = asyncio.Event()
        self._clock = get_dependency(self.pm, "clock", Clock)

        try:
            for server_name, server_cfg in servers_config.items():
                # A config error is reported once, with the remedy, and the
                # server is skipped for good: retrying cannot fix it.
                try:
                    _parse_server_config(server_name, server_cfg)
                except McpHeaderEnvError as exc:
                    logger.warning("%s", exc, extra={"server": server_name})
                    continue
                except (ValueError, KeyError) as exc:
                    detail = (
                        f"missing key {exc.args[0]!r}" if isinstance(exc, KeyError)
                        else str(exc)
                    )
                    logger.warning(
                        "MCP server %r skipped: config error (%s); fix "
                        "mcp.servers.%s in agent.yaml and restart",
                        server_name, detail, server_name,
                        extra={"server": server_name},
                    )
                    continue
                except Exception:
                    # Anything else is still a bad config: skip this server,
                    # never abort the broadcast for the ones after it.
                    logger.warning(
                        "MCP server %r skipped: unexpected error reading "
                        "its config; check mcp.servers.%s in agent.yaml "
                        "and restart",
                        server_name, server_name,
                        extra={"server": server_name},
                        exc_info=True,
                    )
                    continue
                try:
                    await self._connect_server(server_name, server_cfg)
                except BaseException as exc:
                    # A real cancel (shutdown), bare or wrapped in a group,
                    # goes to the outer handler. A cancel with no one
                    # cancelling this task is a failed connect that surfaced
                    # as CancelledError (#385), alone or in a group (#596).
                    if not _is_connect_failure(exc):
                        raise
                    detail, with_traceback = _failure_log_parts(exc, server_cfg)
                    logger.warning(
                        "MCP server %r (%s) failed to connect (%s), skipping%s; "
                        "the background retry will try again",
                        server_name, _server_endpoint(server_cfg),
                        _exc_type_names(exc), detail,
                        extra={"server": server_name},
                        exc_info=with_traceback,
                    )
                # A server that did not register (exception or timeout) is
                # kept for the retry loop rather than skipped for the
                # process lifetime.
                if not any(server.name == server_name for server in self._servers):
                    self._pending_servers[server_name] = server_cfg
        except (asyncio.CancelledError, BaseExceptionGroup):
            # Interrupted mid-connect: the servers already connected have
            # owner tasks that nothing else will stop, so close them before
            # the cancellation continues. A group reaches here only when the
            # inner handler decided it must propagate (#596).
            await self._stop_owners()
            raise

        self._cached_tools = self._build_tool_list()
        logger.info(
            "McpClientPlugin started",
            extra={"servers": len(self._servers), "tools": len(self._cached_tools)},
        )
        # ToolCollectionPlugin.on_start may already have collected while these
        # connects were in flight (the broadcast runs implementations
        # concurrently; trylast orders starts, not finishes), so push the
        # connected servers' tools into the shared registry now (#405).
        if self._servers:
            await self._rebuild_shared_registry(
                [server.name for server in self._servers], "connected"
            )
        # Retry what failed in the background, so a daemon whose servers all
        # failed recovers without waiting for an MCP tool call.
        if self._pending_servers:
            self._retry_task = asyncio.create_task(
                self._retry_loop(), name="mcp-retry-loop"
            )

    @hookimpl
    async def on_stop(self) -> None:
        """Stop the retry loop, then close every MCP session and transport."""
        # From here a connect that finishes late tears itself down.
        self._stopping = True
        self._stop_event.set()
        task, self._retry_task = self._retry_task, None
        if task is not None:
            # Cancel the loop (it may be mid-connect) and wait for it to unwind.
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Expected: we just cancelled it. Re-raise only if on_stop
                # itself is being cancelled.
                if asyncio.current_task().cancelling():
                    raise
            except Exception:
                logger.warning(
                    "MCP retry loop ended with an error during shutdown",
                    exc_info=True,
                )
        await self._stop_owners()
        logger.debug("McpClientPlugin stopped")

    async def _stop_owners(self) -> None:
        """Signal all owner tasks to exit and await them together.

        Concurrent, not one after another: the SDK's stdio_client can take
        about 2 s per server to terminate a process, and Runtime allows the
        whole stop broadcast daemon.shutdown_timeout (3 s by default). An owner that has not finished within
        the derived stop bound (80% of daemon.shutdown_timeout) is cancelled
        and logged. Never raises (a cancellation of this coroutine itself
        excepted).
        """
        owners = list(self._owners.values())
        tasks = [o.task for o in owners if o.task is not None]
        stop_bound = getattr(self, '_stop_bound', _DEFAULT_STOP_BOUND_SECONDS)
        # Tell every owner to close its transport at the same moment.
        for owner in owners:
            owner.stop.set()
        try:
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=stop_bound)
            else:
                pending = set()
            # Cancel stragglers and give them a moment to unwind.
            for owner in owners:
                if owner.task in pending:
                    logger.warning(
                        "MCP server %r did not shut down within %ss and was "
                        "cancelled; the server process may need to be killed "
                        "by hand",
                        owner.name, stop_bound,
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

    async def _retry_loop(self) -> None:
        """Reconnect pending servers on the 10 s doubling, 300 s capped ladder.

        Sleeps on the kernel clock, so a fake clock drives it in tests and
        on_stop's stop event wakes it. The delay doubles after a round that
        reconnects nothing and resets after one that does. Exits once
        nothing is pending.
        """
        assert self._clock is not None
        delay: float = _INITIAL_BACKOFF
        while self._pending_servers and not self._stopping:
            # What the log line should promise if this round fails.
            self._next_delay = min(delay * _BACKOFF_MULTIPLIER, _BACKOFF_CAP)
            await self._clock.sleep(delay, wake=self._stop_event)
            if self._stopping:
                return
            async with self._retry_lock:
                reconnected = await self._retry_round(from_loop=True)
            if reconnected:
                delay = _INITIAL_BACKOFF
            else:
                delay = min(delay * _BACKOFF_MULTIPLIER, _BACKOFF_CAP)

    async def _retry_pending_servers(self) -> None:
        """Fast-path retry, called from every MCP tool call.

        Skips when nothing is pending, or when the loop (or another tool
        call) already holds a retry round: the tool call must not wait out
        a connect handshake of up to 30 s.
        """
        if not self._pending_servers or self._stopping or self._retry_lock.locked():
            return
        async with self._retry_lock:
            await self._retry_round()

    async def _retry_round(self, from_loop: bool = False) -> bool:
        """Try each pending server once; the caller holds the retry lock.

        Successful reconnects refresh the cached tool list and the shared
        registry. Returns True when at least one server reconnected.
        """
        reconnected = False
        connected_names: list[str] = []
        for name, cfg in list(self._pending_servers.items()):
            # The other caller may have reconnected it, or on_stop begun.
            if name not in self._pending_servers or self._stopping:
                continue
            try:
                await self._connect_server(name, cfg)
            except BaseException as exc:
                # A real cancel propagates; a cancel with no one cancelling
                # this task is a failed connect surfacing as CancelledError
                # (#385), alone or in a group (#596), and is handled like any
                # other connect failure.
                if not _is_connect_failure(exc):
                    raise
                # The loop knows its next delay; a tool-call retry does not.
                when = (
                    f"retrying in {self._next_delay:g}s" if from_loop
                    else "the background retry will try again"
                )
                detail, with_traceback = _failure_log_parts(exc, cfg)
                logger.warning(
                    "MCP server %r (%s) reconnect failed (%s)%s; %s",
                    name, _server_endpoint(cfg), _exc_type_names(exc), detail,
                    when,
                    extra={"server": name},
                    exc_info=with_traceback,
                )
                continue
            if any(server.name == name for server in self._servers):
                self._pending_servers.pop(name, None)
                reconnected = True
                connected_names.append(name)
            # A timeout leaves the server pending for the next round.

        if reconnected:
            self._cached_tools = self._build_tool_list()
            # Refresh the shared registry so the Agent and later tool
            # consumers (e.g. subagents) see the reconnected server's tools.
            await self._rebuild_shared_registry(connected_names, "reconnected")
        return reconnected

    async def _rebuild_shared_registry(self, names: list[str], event: str) -> None:
        """Ask ToolCollectionPlugin to re-collect tools after servers connect.

        The tools plugin may have built the registry before these servers
        finished connecting (the on_start broadcast runs concurrently, #405),
        so their tools reach it only through this rebuild. The Agent notices
        the generation change on its next turn (#109/#271). A failed rebuild
        is logged, never raised: the servers are connected, only the shared
        registry is stale.

        Args:
            names: The servers that just connected, for the warning.
            event: "connected" or "reconnected", for the warning.
        """
        tools_plugin = self.pm.get_plugin("tools") if self.pm is not None else None
        rebuild = getattr(tools_plugin, "rebuild_registry", None)
        if not inspect.iscoroutinefunction(rebuild):
            return
        try:
            await rebuild()
        except Exception:
            logger.warning(
                "Tool registry rebuild failed after MCP server(s) %s %s; "
                "their tools reach the registry on the next successful rebuild",
                ", ".join(repr(n) for n in names), event,
                exc_info=True,
            )

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

        Note: config errors (ValueError, KeyError, McpHeaderEnvError) are
        raised by _parse_server_config before anything opens; on_start
        reports them and never queues the server for retry. Any other
        exception is a connect failure, logged by the caller and retried; so
        is a CancelledError that nobody requested, which the callers check
        with ``cancelling()``. A handshake that exceeds connect_timeout_seconds is logged by the
        owner and returns without registering the server.
        """
        # Validate config before anything opens, so a config error raises
        # here, in this task, rather than killing the owner with ``ready``
        # unresolved. McpHeaderEnvError/ValueError/KeyError mean "config".
        settings = _parse_server_config(name, cfg)
        transport = settings.transport
        prefix = settings.prefix
        timeout = settings.timeout
        connect_timeout = settings.connect_timeout
        headers = settings.headers

        # The owner task opens and (later) closes everything this server
        # needs; `ready` carries its handshake result back to this task.
        owner = _ServerOwner(name=name)
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        owner.task = asyncio.create_task(
            self._run_owner(
                owner, cfg, headers, connect_timeout, ready,
                transport=transport, prefix=prefix, timeout=timeout,
            ),
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

        if self._stopping:
            # on_stop ran while this connect was in flight and has already
            # cleared _owners; registering now would leak the owner.
            logger.warning(
                "MCP server %r connected after shutdown began; closing it",
                name, extra={"server": name},
            )
            owner.stop.set()
            await asyncio.wait([owner.task])
            return

        if name in self._owners:
            # A concurrent retry connected this server first. Keep that
            # owner and close this duplicate, so no owner is orphaned.
            logger.warning(
                "MCP server %r connected twice (concurrent retry); closing "
                "the duplicate connection",
                name, extra={"server": name},
            )
            owner.stop.set()
            await asyncio.wait([owner.task])
            return

        # Connected: the owner stays alive until on_stop signals it.
        self._owners[name] = owner
        self._servers.append(state)
        logger.info(
            "MCP server connected",
            extra={"server": name, "transport": transport,
                   "tools": len(state.mcp_tools)},
        )

    async def _run_owner(
        self,
        owner: _ServerOwner,
        cfg: dict,
        headers: dict[str, str] | None,
        connect_timeout: float,
        ready: asyncio.Future,
        *,
        transport: str,
        prefix: str,
        timeout: float,
    ) -> None:
        """Body of a server's owner task: connect, serve, close.

        Resolves ``ready`` with the server's _McpServerState (or None after
        a connect timeout, or the connect exception), then waits for
        ``owner.stop`` and closes the server's resources in this same task.
        Teardown errors are logged here, not raised, so they cannot break
        on_stop's other servers. Anything raised before ``ready`` resolves
        (including the SDK imports) resolves it with that exception and
        closes the stack, so _connect_server never waits forever. A
        transport that cancels this task from inside on failure (anyio) is
        not a real cancel: when the stack's close then raises, ``ready``
        resolves with that error and the task returns instead of raising
        CancelledError.
        """
        name = owner.name
        stack = owner.stack
        deadline = asyncio.timeout(connect_timeout)
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
            from mcp.client.sse import sse_client
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client

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
            cleanup_exc = await self._close_failed_server(stack)
            # A transport that fails (anyio task group on ConnectError)
            # cancels this task from inside; once the stack has closed,
            # cancelling() is 0 for that and stays nonzero for a real cancel.
            # The cleanup exception is then the real failure (#385).
            internal_cancel = (
                isinstance(exc, asyncio.CancelledError)
                and asyncio.current_task().cancelling() == 0
                and cleanup_exc is not None
            )
            if internal_cancel:
                if not ready.done():
                    ready.set_exception(cleanup_exc)
                return
            if cleanup_exc is not None:
                logger.warning(
                    "MCP server cleanup after failed connect raised",
                    extra={"server": name},
                    exc_info=cleanup_exc,
                )
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
            prefix=prefix,
            timeout_seconds=timeout,
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

    async def _close_failed_server(
        self, stack: AsyncExitStack
    ) -> Exception | None:
        """Tear down a server whose connect failed. Returns the exception
        the teardown raised, or None; never raises it. The caller decides
        whether it is a cleanup error to log or the connect failure itself
        (a transport that cancels the owner re-raises its error from here)."""
        try:
            await stack.aclose()
        # Transport cleanup raises types outside this codebase's control
        # (anyio, mcp), as in the owner's shutdown.
        except Exception as exc:
            return exc
        return None

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

    Captures session and original tool name in the closure; the timeout is
    read from ``server`` at call time and declared as ``own_timeout``.
    No _ctx parameter — MCP tools receive only LLM-provided arguments.
    """
    tool_name = (
        f"{server.prefix}__{mcp_tool.name}" if server.prefix else mcp_tool.name
    )
    schema = _mcp_tool_to_schema(tool_name, mcp_tool)

    session = server.session
    original_name = mcp_tool.name
    server_name = server.name

    async def tool_fn(**kwargs) -> str:
        # Give startup-failed servers a chance to come back before the call.
        if plugin is not None:
            await plugin._retry_pending_servers()
        # timeout_seconds is read at call time, not captured at registration.
        return await _call_mcp_tool(
            session, original_name, kwargs, server.timeout_seconds, server_name
        )

    tool_fn.__name__ = tool_name
    # Declare the server's own timeout so dispatch skips a longer outer bound.
    tool_fn.own_timeout = lambda: server.timeout_seconds  # type: ignore[attr-defined]
    # Every MCP tool returns outside content.
    tool_fn.returns_untrusted = True  # type: ignore[attr-defined]
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
        # Name the tool and the server, as the timeout and exception branches
        # do: an error payload alone cannot be traced to the call that made it.
        return (
            f"Error: MCP tool '{original_name}' on server '{server_name}' "
            f"returned an error: {text}"
        )
    return text
