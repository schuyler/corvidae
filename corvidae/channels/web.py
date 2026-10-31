"""Web transport plugin: a browser chat UI served over aiohttp (issue #87).

Serves a single-page chat UI on a local port. A browser logs in once with
a token (``/?token=…``), which sets a session cookie; the page then talks
to the agent over one WebSocket carrying JSON frames.

Channels are ``web:<slug>`` — one per named conversation. Every message
on a web channel originates here, so the plugin can show a user message
from the moment it's sent until its row reaches ``message_log`` (the
"unsent" entries), and replay persisted history to new tabs.

Each WebSocket connection is owned by its handler task: ``_handlers`` holds
every live handler, and each handler's outermost ``finally`` releases
everything its connection holds. ``_clients`` is only the fan-out set.
``on_stop`` closes sockets, then cancels whatever handlers remain, so a
handler stuck in a gate (aiohttp drops its own reference once the
connection is lost) can't outlive shutdown.

Logging:
    - WARNING: non-loopback bind; generated-token login URL; slow client
      dropped; history unavailable; config changes needing a restart
    - INFO: listening (provided token); invalid login attempt; token change
    - DEBUG: inert (no ``web`` block); close/abort details
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from corvidae.channel import Channel, ChannelRegistry
from corvidae.context import MessageType, unsent_marker_emission
from corvidae.persistence import decode_message_row, parse_message_type
from corvidae.trace import mint_trace_id
from corvidae.hooks import TransportPlugin, get_dependency, hookimpl
from corvidae.outbound import (
    OutboundMessage,
    OutboundProgress,
    OutboundThinking,
    OutboundToolStatus,
)
from corvidae.thinking import strip_thinking

logger = logging.getLogger(__name__)

# Conversation names: lowercase slug, up to 64 characters (R13).
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
# Provided tokens: RFC 3986 unreserved characters only, since the token is
# pasted into a URL by hand.
_TOKEN_RE = re.compile(r"[A-Za-z0-9._~-]{16,}")
# task.py frames tool results as "[Task <id>] <result>".
_TASK_PREFIX = re.compile(r"^\[Task [^\]]*\] ")

_COOKIE = "corvidae_web"
_COOKIE_MAX_AGE = 34_560_000  # 400 days, Chrome's cap
_QUEUE_MAX = 1024
_BUFFER_MAX = 512
_HISTORY_ROWS = 200
# The stored message_type values a history row may carry (anything else is corrupt).
_MESSAGE_TYPES = frozenset(t.value for t in MessageType)
_SUMMARY_CHARS = 200
_CLOSE_TIMEOUT = 1.0
_HANDLER_WAIT = 0.5
_HEARTBEAT = 30.0  # seconds; module-level so tests can shorten it
_TOKEN_HINT = 'generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'

_STATIC_DIR = Path(str(resources.files("corvidae.channels") / "web_static"))

_LOGIN_PAGE = (
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta http-equiv="refresh" content="0;url=/"><title>corvidae</title>'
    '</head><body><p><a href="/">Continue</a></p></body></html>'
)


# ---------------------------------------------------------------------------
# Pure helpers: comparison, hosts and origins, tokens, config
# ---------------------------------------------------------------------------

def _ascii_equal(a: str, b: str) -> bool:
    """Constant-time equality for ASCII strings; non-ASCII is simply unequal."""
    return a.isascii() and b.isascii() and hmac.compare_digest(
        a.encode("ascii"), b.encode("ascii")
    )


def _strip_brackets(host: str) -> str:
    """Remove one pair of surrounding brackets (``[::1]`` → ``::1``)."""
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _canonical_host(host: str) -> str:
    """Canonical form: compressed IP literal, else lowercased name."""
    bare = _strip_brackets(host)
    try:
        return ipaddress.ip_address(bare).compressed
    except ValueError:
        return bare.lower()


def _is_wildcard(host: str) -> bool:
    """True for unspecified addresses (``0.0.0.0``, ``::``)."""
    try:
        return ipaddress.ip_address(_strip_brackets(host)).is_unspecified
    except ValueError:
        return False


def _is_loopback(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals."""
    canonical = _canonical_host(host)
    if canonical == "localhost":
        return True
    try:
        return ipaddress.ip_address(canonical).is_loopback
    except ValueError:
        return False


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _netloc(host: str, port: int, scheme: str) -> str:
    """Canonical host (bracketed if IPv6) plus ``:port`` unless default."""
    canonical = _canonical_host(host)
    if ":" in canonical:
        canonical = f"[{canonical}]"
    if port == _default_port(scheme):
        return canonical
    return f"{canonical}:{port}"


def _normalise_origin(text: str) -> str:
    """Canonical ``scheme://netloc`` for an origin; raises ValueError."""
    if not isinstance(text, str):
        raise ValueError("origin must be a string")
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    # Only bare http(s) origins: no userinfo, query, fragment or path.
    if scheme not in ("http", "https"):
        raise ValueError(f"origin scheme must be http or https: {text!r}")
    if not parts.hostname:
        raise ValueError(f"origin has no host: {text!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"origin must not carry credentials: {text!r}")
    if "?" in text or "#" in text:
        raise ValueError(f"origin must not have a query or fragment: {text!r}")
    if parts.path not in ("", "/"):
        raise ValueError(f"origin must not have a path: {text!r}")
    port = parts.port  # raises ValueError for a malformed port
    return f"{scheme}://{_netloc(parts.hostname, port or _default_port(scheme), scheme)}"


def _host_values(origin: str) -> set[str]:
    """Host header values that address an origin (``netloc``, plus the
    explicit default port for a default-port origin)."""
    scheme, netloc = origin.split("://", 1)
    values = {netloc}
    has_port = not netloc.endswith("]") and ":" in netloc
    if not has_port:
        values.add(f"{netloc}:{_default_port(scheme)}")
    return values


def _compute_origins(
    host: str, port: int, allowed_origins: list[str] | None
) -> tuple[list[str], set[str], str]:
    """Allowed origins, allowed Host values and login URL base for a bind."""
    origins: list[str] = []

    # The bind address itself is an origin unless it's a wildcard; a
    # loopback bind also accepts the other loopback spellings.
    if not _is_wildcard(host):
        origins.append(_normalise_origin("http://" + _netloc(host, port, "http")))
        if _canonical_host(host) in ("127.0.0.1", "::1", "localhost"):
            for name in ("127.0.0.1", "localhost", "::1"):
                origin = "http://" + _netloc(name, port, "http")
                if origin not in origins:
                    origins.append(origin)

    # Configured origins (reverse proxy, Tailscale name) come next.
    configured = [_normalise_origin(o) for o in allowed_origins or []]
    for origin in configured:
        if origin not in origins:
            origins.append(origin)

    hosts: set[str] = set()
    for origin in origins:
        hosts |= _host_values(origin)
    base = configured[0] if configured else origins[0]
    return origins, hosts, base


def _session_for(token: str) -> str:
    """Stateless session cookie value: same token → same cookie."""
    return hmac.new(token.encode("ascii"), b"corvidae-web-session", hashlib.sha256).hexdigest()


def _resolve_token(block: dict) -> tuple[str, bool]:
    """``web.token`` → ``CORVIDAE_WEB_TOKEN`` → generated. Returns
    ``(token, generated)``; raises ValueError for a bad provided token."""
    given = block.get("token") or os.environ.get("CORVIDAE_WEB_TOKEN")
    if not given:
        return secrets.token_urlsafe(32), True
    # Provided tokens must be URL-safe by hand and long enough to guess.
    if not isinstance(given, str) or not _TOKEN_RE.fullmatch(given):
        raise ValueError(
            "web: token must be at least 16 characters of A-Z a-z 0-9 . _ ~ -; "
            + _TOKEN_HINT
        )
    return given, False


@dataclass
class _Settings:
    """Validated listener settings from a ``web`` block."""

    host: str
    port: int
    allowed_origins: list[str] | None  # normalised


def _parse_block(raw: object) -> _Settings:
    """Validate a ``web`` block; raises ValueError naming the field."""
    # `web:` with no value means "enabled with defaults".
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("web: the web block must be a mapping")

    host = raw.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        raise ValueError("web: host must be a non-empty string")

    port = raw.get("port", 8765)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("web: port must be an integer from 0 to 65535")

    token = raw.get("token")
    if token is not None and not isinstance(token, str):
        raise ValueError("web: token must be a string")

    # allowed_origins: null or a list of strings, each a valid origin.
    origins = raw.get("allowed_origins")
    normalised: list[str] | None = None
    if origins is not None:
        if not isinstance(origins, list) or not all(isinstance(o, str) for o in origins):
            raise ValueError("web: allowed_origins must be a list of strings")
        normalised = []
        for origin in origins:
            try:
                normalised.append(_normalise_origin(origin))
            except ValueError as exc:
                raise ValueError(f"web: allowed_origins entry is invalid: {exc}") from None

    # An ephemeral port with a hostname would bind one port per address.
    if port == 0:
        try:
            ipaddress.ip_address(_strip_brackets(host))
        except ValueError:
            raise ValueError("web: port 0 requires host to be an IP literal") from None

    if _is_wildcard(host) and not normalised:
        raise ValueError(
            f"web: allowed_origins must be set when host is a wildcard ({host})"
        )
    return _Settings(host=host, port=port, allowed_origins=normalised)


# ---------------------------------------------------------------------------
# Per-connection and per-message state
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class _Unsent:
    """A user message shown in the pane whose row isn't persisted yet."""

    text: str
    # Trace id minted by _on_send and passed to on_message; the
    # admitted/rejected/persisted hooks match their entry by it.
    trace_id: str
    # Set by on_message_admitted: the agent accepted it, so a reply is owed.
    admitted: bool = False
    id: str = field(default_factory=lambda: secrets.token_hex(8))


class _Client:
    """One WebSocket connection's fan-out state."""

    def __init__(self, ws: web.WebSocketResponse, transport) -> None:
        self.ws = ws
        self.transport = transport
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self.writer: asyncio.Task | None = None
        self.conversation: str | None = None  # channel id, e.g. "web:main"
        self.buffer: list[dict] | None = None  # set while history is fetched
        self.dropped = False


class _BadRequest(Exception):
    """A client frame failed validation."""


class _NotReady(Exception):
    """The agent hasn't started yet."""


def _abort_if_stuck(transport, force: bool = False) -> None:
    """Abort a transport that can't finish closing on its own.

    ``transport.close()`` waits to flush its write buffer, which never
    happens with a peer that stopped reading; ``abort()`` drops the buffer
    and fires ``connection_lost`` at once. Aborting with an empty buffer
    would gain nothing and could cut a close handshake short, so only a
    forced abort or unsent bytes trigger it.
    """
    if transport is None:
        return
    try:
        if force or transport.get_write_buffer_size() > 0:
            transport.abort()
    except Exception:
        logger.debug("web: abort failed", exc_info=True)


def _text(value: object) -> str:
    """A row field as text; anything that isn't a str is empty."""
    return value if isinstance(value, str) else ""


def _slug(value: object) -> str:
    """Validate a conversation name from a client frame."""
    if not isinstance(value, str) or not _SLUG.fullmatch(value):
        raise _BadRequest("conversation must match [a-z0-9][a-z0-9_-]{0,63}")
    return value


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class WebPlugin(TransportPlugin):
    """Browser chat transport. Inert unless agent.yaml has a ``web`` block."""

    transport = "web"

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # Configuration (on_init / on_config_reload).
        self._enabled = False
        self._host = "127.0.0.1"
        self._port = 8765
        self._allowed_origins: list[str] | None = None
        self._token = ""
        self._session = ""
        self._token_generated = False
        # Listener (on_start).
        self.port: int | None = None
        self._runner: web.AppRunner | None = None
        self._registry: ChannelRegistry | None = None
        self._origins: list[str] = []
        self._hosts: set[str] = set()
        self._secure_hosts: set[str] = set()
        self._login_base = ""
        self._csp = ""
        # Connections.
        self._clients: set[_Client] = set()
        self._handlers: set[asyncio.Task] = set()
        self._close_tasks: set[asyncio.Task] = set()
        self._stopping = False
        # Conversation state.
        self._unsent: dict[str, list[_Unsent]] = {}
        self._working: set[str] = set()
        self._warned_no_db = False
        # The idle tracker watch that clears "working"; None without one.
        self._idle_watch = None

    # -- lifecycle ----------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        # No block → inert, and nothing above DEBUG (R2).
        if "web" not in config:
            logger.debug("web: no web block; transport disabled")
            return
        raw = config["web"]
        settings = _parse_block(raw)
        token, generated = _resolve_token(raw or {})
        self._host = settings.host
        self._port = settings.port
        self._allowed_origins = settings.allowed_origins
        self._set_token(token, generated)
        self._enabled = True

    @hookimpl
    async def on_start(self, config: dict) -> None:
        if not self._enabled:
            return
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        self._stopping = False

        # Follow the idle tracker's settle signal for the working indicator.
        # shortcut: without a tracker (hand-built plugin managers) "working"
        # clears on the final reply only; Runtime always registers one.
        idle = self.pm.get_plugin("idle")
        if idle is None:
            logger.warning(
                "web: no idle tracker registered; the working indicator will "
                "clear on reply only, so it may clear before queued work ends"
            )
        else:
            # Period 0 fires at settle; the long repeat keeps a quiet channel
            # from re-firing (the callback is idempotent anyway).
            self._idle_watch = idle.watch_channels(
                0, self._on_channel_settled, name="web.working", repeat=3600,
            )

        # Bind; a failure raises out of startup. access_log=None keeps
        # `?token=` out of the log; the short shutdown timeout keeps
        # runner.cleanup() inside Runtime's stop budget.
        runner = web.AppRunner(self._build_app(), access_log=None, shutdown_timeout=0.5)
        await runner.setup()
        site = web.TCPSite(runner, _strip_brackets(self._host), self._port)
        try:
            await site.start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner
        self.port = runner.addresses[0][1]
        self._apply_origins()

        # Startup notices: non-loopback warning, then how to log in.
        if not _is_loopback(self._host):
            logger.warning(
                "web: listening on %s, which is not a loopback address; anyone "
                "who can reach it and holds the token can run shell commands as "
                "this user", self._login_base,
            )
        if self._token_generated:
            self._log_login_url()
        else:
            logger.info("web: listening on %s/", self._login_base)

    @hookimpl
    async def on_stop(self) -> None:
        if not self._enabled:
            return
        self._stopping = True
        # Stop following settle events.
        if self._idle_watch is not None:
            self._idle_watch.cancel()
            self._idle_watch = None

        # 1. Close every fan-out socket and finish pending drop-path closes;
        #    each is bounded by _close_client, which logs and absorbs its own
        #    errors, so only a cancellation can come back here.
        closes = [self._close_client(c, 1001, b"server shutting down")
                  for c in list(self._clients)]
        await asyncio.gather(*closes, *list(self._close_tasks), return_exceptions=True)

        # 2. Cancel every handler still running — including those of clients
        #    the drop path already removed from _clients.
        handlers = [t for t in self._handlers if not t.done()]
        for task in handlers:
            task.cancel()
        if handlers:
            _, pending = await asyncio.wait(handlers, timeout=_HANDLER_WAIT)
            if pending:
                logger.warning("web: %d socket handler(s) did not finish", len(pending))

        # 3. Stop the listener.
        if self._runner is not None:
            runner, self._runner = self._runner, None
            await runner.cleanup()

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        # Inert at startup: a new block needs a restart.
        if not self._enabled:
            if "web" in config:
                logger.warning("web: a web block was added; restart to apply")
            return
        if "web" not in config:
            logger.warning("web: block removed; restart to apply")
            return

        # Validate everything first; a bad field raises and changes nothing.
        raw = config["web"]
        settings = _parse_block(raw)
        token, generated = _resolve_token(raw or {})

        # Listener settings only apply at startup.
        if _canonical_host(settings.host) != _canonical_host(self._host):
            logger.warning("web: host changed; restart to apply")
        if settings.port != self._port:
            logger.warning("web: port changed; restart to apply")
        if settings.allowed_origins != self._allowed_origins:
            logger.warning("web: allowed_origins changed; restart to apply")

        # Token: keep a generated one while none is provided.
        if generated and self._token_generated:
            return
        if token == self._token:
            return
        self._set_token(token, generated)
        if generated:
            self._log_login_url()
        logger.info("web: token changed; existing sessions are invalidated")
        for client in list(self._clients):
            self._clients.discard(client)
            self._spawn_close(client, 4001, b"token changed")

    def _set_token(self, token: str, generated: bool) -> None:
        self._token = token
        self._session = _session_for(token)
        self._token_generated = generated

    def _log_login_url(self) -> None:
        # Pre-formatted so the token is in the message only: no args, no extra.
        logger.warning(
            f"web: log in at {self._login_base}/?token={self._token} "
            "(set web.token or CORVIDAE_WEB_TOKEN to keep logins across restarts)"
        )

    def _apply_origins(self) -> None:
        """Derive origins, Host values, Secure hosts and the CSP for the bound port."""
        origins, hosts, base = _compute_origins(self._host, self.port, self._allowed_origins)
        self._origins, self._hosts, self._login_base = origins, hosts, base
        self._secure_hosts = set()
        ws_sources = []
        for origin in origins:
            scheme, netloc = origin.split("://", 1)
            if scheme == "https":
                self._secure_hosts |= _host_values(origin)
            # CSP host-sources can't express IPv6 literals; 'self' covers those.
            if not netloc.startswith("["):
                ws_sources.append(f"{'wss' if scheme == 'https' else 'ws'}://{netloc}")
        self._csp = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            f"connect-src 'self' {' '.join(ws_sources)}; img-src 'self' data:; "
            "object-src 'none'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'"
        )

    def _ready(self) -> bool:
        """True once Agent has started (which implies the DB is open)."""
        agent = self.pm.get_plugin("agent")
        return agent is not None and bool(getattr(agent, "started", False))

    def _db(self):
        """The persistence plugin's connection, or None."""
        persistence = self.pm.get_plugin("persistence")
        return getattr(persistence, "db", None) if persistence is not None else None

    # -- HTTP ---------------------------------------------------------------

    def _build_app(self) -> web.Application:
        """Routes and middlewares (security headers → Host check → auth)."""

        @web.middleware
        async def security_headers(request: web.Request, handler):
            # Headers on every response, including errors; an unexpected
            # exception becomes a logged 500 (no URL: it may carry ?token=).
            try:
                response = await handler(request)
            except web.HTTPException as exc:
                self._apply_headers(exc.headers)
                raise
            except Exception:
                logger.exception("web: unhandled error in %s handler", request.method)
                response = web.Response(status=500, text="internal error")
            if isinstance(response, web.WebSocketResponse):
                return response
            self._apply_headers(response.headers)
            return response

        @web.middleware
        async def host_check(request: web.Request, handler):
            # DNS-rebinding guard: only Host values we derived.
            if request.host.lower() not in self._hosts:
                raise web.HTTPForbidden(text="bad host")
            return await handler(request)

        @web.middleware
        async def auth(request: web.Request, handler):
            # Login: a token on `/` trades for the session cookie.
            if request.path == "/" and "token" in request.query:
                if _ascii_equal(request.query.get("token", ""), self._token):
                    return self._login_response(request)
                logger.info("web: login with an invalid token from %s", request.remote)
                raise web.HTTPUnauthorized(text="invalid token")
            # Everything else needs the session cookie.
            cookie = request.cookies.get(_COOKIE)
            if cookie is None or not _ascii_equal(cookie, self._session):
                raise web.HTTPUnauthorized(
                    text="not logged in: open the login URL from the corvidae log"
                )
            return await handler(request)

        app = web.Application(middlewares=[security_headers, host_check, auth])
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws_handler)
        app.router.add_static("/static", _STATIC_DIR, follow_symlinks=False, show_index=False)
        return app

    def _apply_headers(self, headers) -> None:
        headers["Content-Security-Policy"] = self._csp
        headers["X-Content-Type-Options"] = "nosniff"
        headers["Referrer-Policy"] = "no-referrer"
        headers["Cache-Control"] = "no-store"

    def _login_response(self, request: web.Request) -> web.Response:
        """A 200 refresh page (not a redirect, so the Strict cookie applies)."""
        response = web.Response(text=_LOGIN_PAGE, content_type="text/html")
        secure = request.host.lower() in self._secure_hosts
        response.set_cookie(
            _COOKIE, self._session, max_age=_COOKIE_MAX_AGE, path="/",
            httponly=True, samesite="Strict", secure=True if secure else None,
        )
        return response

    async def _index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(_STATIC_DIR / "index.html")

    # -- WebSocket ----------------------------------------------------------

    async def _ws_handler(self, request: web.Request) -> web.StreamResponse:
        # Refusals before the connection exists: shutting down, bad Origin.
        if self._stopping:
            raise web.HTTPServiceUnavailable(text="shutting down")
        try:
            origin_ok = _normalise_origin(request.headers.get("Origin")) in self._origins
        except ValueError:
            origin_ok = False
        if not origin_ok:
            raise web.HTTPForbidden(text="bad origin")

        # From here this task owns the connection until its finally.
        task = asyncio.current_task()
        self._handlers.add(task)
        client: _Client | None = None
        try:
            ws = web.WebSocketResponse(max_msg_size=1 << 20, heartbeat=_HEARTBEAT)
            transport = request.transport
            await ws.prepare(request)
            client = _Client(ws, transport)
            client.writer = asyncio.create_task(self._write_loop(client))
            self._clients.add(client)
            # Frames are handled one at a time, in order.
            try:
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        await self._handle_frame(client, msg.data)
            except Exception:
                logger.exception("web: socket handler failed")
            return ws
        finally:
            self._release(client, task)

    def _release(self, client: _Client | None, task: asyncio.Task | None) -> None:
        """Release a connection's fan-out state (the handler's last act)."""
        # Stop fan-out, then the writer, then abort a socket whose bytes
        # are stuck (e.g. a heartbeat timeout, where aiohttp only calls
        # transport.close() and would wait forever on a peer that stopped
        # reading).
        if client is not None:
            self._clients.discard(client)
            if client.writer is not None:
                client.writer.cancel()
            _abort_if_stuck(client.transport)
        self._handlers.discard(task)

    async def _write_loop(self, client: _Client) -> None:
        """Send queued frames in order until the socket fails."""
        try:
            while True:
                frame = await client.queue.get()
                await client.ws.send_json(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("web: send failed; writer exiting", exc_info=True)

    async def _close_client(self, client, code: int, message: bytes) -> None:
        """Close a socket within ~1 s; abort the transport otherwise."""
        completed = False
        try:
            await asyncio.wait_for(client.ws.close(code=code, message=message), _CLOSE_TIMEOUT)
            completed = True
        except asyncio.TimeoutError:
            logger.debug("web: close timed out; aborting")
        except Exception:
            logger.debug("web: close failed; aborting", exc_info=True)
        finally:
            # Also runs on cancellation: an interrupted close still aborts.
            # A "completed" close can still leave unsent bytes behind (aiohttp
            # returns True after a write error having only called
            # transport.close()), so those abort too.
            _abort_if_stuck(client.transport, force=not completed)

    def _spawn_close(self, client: _Client, code: int, message: bytes) -> None:
        """Start a tracked close task (on_stop awaits any still pending)."""
        task = asyncio.create_task(self._close_client(client, code, message))
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)

    # -- fan-out ------------------------------------------------------------

    def _enqueue(self, client: _Client, frame: dict) -> None:
        """Queue a frame for one client; a full queue drops the client."""
        if client.dropped:
            return
        try:
            client.queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("web: a client is not keeping up; dropping its connection")
            self._drop(client)

    def _drop(self, client: _Client) -> None:
        """Remove a slow client from fan-out and close it, without awaiting."""
        client.dropped = True
        self._clients.discard(client)
        # During shutdown, abort at once rather than start a close task
        # nobody would await.
        if self._stopping:
            _abort_if_stuck(client.transport, force=True)
            return
        self._spawn_close(client, 1008, b"too slow")

    def _broadcast(self, channel_id: str, frame: dict) -> None:
        """Send a frame to every client subscribed to a channel."""
        for client in list(self._clients):
            if client.conversation != channel_id:
                continue
            # Mid-subscribe clients buffer until their history is queued.
            if client.buffer is not None:
                client.buffer.append(frame)
                if len(client.buffer) > _BUFFER_MAX:
                    self._drop(client)
            else:
                self._enqueue(client, frame)

    def _error(self, client: _Client, request: str | None, code: str, message: str) -> None:
        self._enqueue(client, {"type": "error", "request": request, "code": code,
                               "message": message})

    # -- inbound frames -----------------------------------------------------

    async def _handle_frame(self, client: _Client, data: str) -> None:
        """Validate and dispatch one client frame; failures become `error`."""
        request: str | None = None
        try:
            try:
                msg = json.loads(data)
            except (ValueError, RecursionError):
                raise _BadRequest("frame is not valid JSON") from None
            if not isinstance(msg, dict):
                raise _BadRequest("frame must be a JSON object")
            kind = msg.get("type")
            if not isinstance(kind, str):
                raise _BadRequest("type must be a string")
            request = kind
            if kind == "list":
                await self._on_list(client)
            elif kind == "subscribe":
                await self._on_subscribe(client, msg)
            elif kind == "send":
                await self._on_send(client, msg)
            else:
                raise _BadRequest("unknown frame type")
        except _NotReady:
            self._error(client, request, "starting", "the agent is still starting")
        except _BadRequest as exc:
            self._error(client, request, "bad_request", str(exc))
        except Exception:
            logger.exception("web: failed to handle a %s frame", request)
            self._error(client, request, "bad_request", "the message could not be processed")

    async def _on_list(self, client: _Client) -> None:
        if not self._ready():
            raise _NotReady
        items = await self._list_conversations()
        self._enqueue(client, {"type": "conversations", "items": items})

    async def _on_subscribe(self, client: _Client, msg: dict) -> None:
        name = _slug(msg.get("conversation"))
        if not self._ready():
            raise _NotReady
        channel_id = f"web:{name}"

        # Synchronously: switch, start buffering, snapshot unsent entries —
        # before any await, so no live frame or unsent message is missed.
        client.conversation = channel_id
        client.buffer = []
        snapshot = list(self._unsent.get(channel_id, []))
        try:
            rows = await self._fetch_history_rows(channel_id)
            items = self._rows_to_frames(rows, channel_id)
            items += [{"type": "user", "text": e.text, "id": e.id} for e in snapshot]
            self._enqueue(client, {
                "type": "history", "conversation": name, "items": items,
                "working": channel_id in self._working,
            })
        finally:
            # Flush what arrived meanwhile; never leave the client buffering.
            buffered, client.buffer = client.buffer or [], None
            for frame in buffered:
                self._enqueue(client, frame)

    async def _on_send(self, client: _Client, msg: dict) -> None:
        name = _slug(msg.get("conversation"))
        text = msg.get("text")
        if not isinstance(text, str):
            raise _BadRequest("text must be a string")
        # Lone surrogates would poison the stored conversation.
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise _BadRequest("text is not valid Unicode") from None
        if not text.strip():
            return
        if not self._ready():
            raise _NotReady

        # Track the message until it's persisted, rejected or dropped.
        channel = self._registry.get_or_create("web", name)
        entry = _Unsent(text, mint_trace_id())
        self._unsent.setdefault(channel.id, []).append(entry)
        try:
            self._broadcast(channel.id, {"type": "user", "conversation": name,
                                         "text": text, "id": entry.id})
            await self.pm.ahook.on_message(channel=channel, sender="web", text=text,
                                           trace_id=entry.trace_id)
        finally:
            # Admission fires before on_message returns, so an entry still
            # un-admitted here was dropped, rejected, raised or cancelled.
            # Removal is idempotent (rejection already removed its entry).
            if not entry.admitted:
                self._remove_entry(channel.id, entry)

    def _remove_entry(self, channel_id: str, entry: _Unsent) -> None:
        """Remove an unsent entry by identity; drop an emptied list."""
        entries = self._unsent.get(channel_id)
        if entries is None:
            return
        remaining = [e for e in entries if e is not entry]
        if remaining:
            self._unsent[channel_id] = remaining
        else:
            del self._unsent[channel_id]

    def _entry_for(self, channel_id: str, trace_id: str) -> _Unsent | None:
        """The tracked entry carrying this trace id, if any."""
        for e in self._unsent.get(channel_id, []):
            if e.trace_id == trace_id:
                return e
        return None

    # -- history and listing -------------------------------------------------

    def _warn_no_db(self) -> None:
        if not self._warned_no_db:
            self._warned_no_db = True
            logger.warning("web: history is unavailable without the persistence plugin")
        else:
            logger.debug("web: history is unavailable without the persistence plugin")

    async def _fetch_history_rows(self, channel_id: str) -> list[tuple]:
        """The newest rows for a channel, oldest first."""
        db = self._db()
        if db is None:
            self._warn_no_db()
            return []
        try:
            async with db.execute(
                "SELECT id, CAST(message AS BLOB), message_type FROM message_log "
                "WHERE channel_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
                (channel_id, _HISTORY_ROWS),
            ) as cursor:
                rows = await cursor.fetchall()
        except Exception:
            logger.warning("web: could not read history for %s", channel_id, exc_info=True)
            return []
        return list(reversed(rows))

    def _rows_to_frames(self, rows: list[tuple], channel_id: str) -> list[dict]:
        """Map message_log rows to the frames the live path would have sent.

        A corrupt row (or one with an unknown message_type) is skipped with a
        WARNING naming the row and channel; the other rows still replay.
        """
        frames: list[dict] = []
        tool_names: dict[str, str] = {}
        # Frames the most recent assistant row produced, so a not-sent
        # marker right after it can hold back what a gate suppressed live.
        last_assistant_frames: list[dict] = []
        for row in rows:
            # Corrupt rows are skipped with a warning, not fatal.
            rowid, raw, message_type = row
            msg = decode_message_row(
                rowid, raw, channel_id,
                consequence="the rest of the history still replays",
            )
            if msg is None:
                continue
            if message_type not in _MESSAGE_TYPES:
                parse_message_type(
                    rowid, message_type, channel_id,
                    consequence="the rest of the history still replays",
                )
                continue

            # Summaries become dividers; injected context is never shown,
            # and neither is a message a send gate vetoed (it was never sent).
            if message_type == "summary":
                frames.append({"type": "divider", "text": _text(msg.get("content"))})
                continue
            if message_type == "context":
                # A progress/thinking marker: drop that frame of the
                # assistant row it follows (the row itself stays MESSAGE).
                suppressed = unsent_marker_emission(msg.get("content"))
                if suppressed in ("progress", "thinking"):
                    held = [f for f in last_assistant_frames if f["type"] == suppressed]
                    frames = [f for f in frames if not any(f is h for h in held)]
                continue
            if message_type == "unsent":
                continue

            role = msg.get("role")
            last_assistant_frames = []
            if role == "user":
                frames.append({"type": "user", "text": _text(msg.get("content")), "id": None})
            elif role == "assistant":
                # Reasoning, then text (progress if it carried calls), then calls.
                reasoning = _text(msg.get("reasoning_content"))
                if reasoning:
                    frames.append({"type": "thinking", "text": reasoning})
                    last_assistant_frames.append(frames[-1])
                calls = msg.get("tool_calls")
                calls = calls if isinstance(calls, list) else []
                content = strip_thinking(_text(msg.get("content")))
                if content:
                    frames.append({"type": "progress" if calls else "message", "text": content})
                    last_assistant_frames.append(frames[-1])
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    function = function if isinstance(function, dict) else {}
                    name = _text(function.get("name")) or "unknown"
                    call_id = call.get("id")
                    if isinstance(call_id, str):
                        tool_names[call_id] = name
                    frames.append({
                        "type": "tool_status", "tool": name, "status": "dispatched",
                        "summary": _text(function.get("arguments"))[:_SUMMARY_CHARS],
                    })
            elif role == "tool":
                call_id = msg.get("tool_call_id")
                name = tool_names.get(call_id, "unknown") if isinstance(call_id, str) else "unknown"
                summary = _TASK_PREFIX.sub("", _text(msg.get("content")), count=1)
                frames.append({"type": "tool_status", "tool": name, "status": "completed",
                               "summary": summary[:_SUMMARY_CHARS]})
            # role "system" (notifications) is never shown live, so not replayed.
        return frames

    async def _list_conversations(self) -> list[dict]:
        """Web conversations from message_log merged with the registry."""
        latest: dict[str, float] = {}

        # Persisted conversations.
        db = self._db()
        if db is None:
            self._warn_no_db()
        else:
            try:
                async with db.execute(
                    "SELECT channel_id, MAX(timestamp) FROM message_log "
                    "WHERE channel_id LIKE 'web:%' GROUP BY channel_id"
                ) as cursor:
                    rows = await cursor.fetchall()
            except Exception:
                logger.warning("web: could not list conversations", exc_info=True)
                rows = []
            for channel_id, ts in rows:
                if channel_id.startswith("web:"):
                    latest[channel_id[4:]] = float(ts)

        # Live channels; on overlap the more recent time wins.
        for channel in self._registry.by_transport("web"):
            latest[channel.scope] = max(latest.get(channel.scope, channel.last_active),
                                        channel.last_active)

        items = [{"name": n, "last_active": t} for n, t in latest.items() if _SLUG.fullmatch(n)]
        items.sort(key=lambda i: i["last_active"], reverse=True)
        return items

    # -- outbound hooks -------------------------------------------------------

    def owns(self, channel: Channel) -> bool:
        """Broadcast filter, narrowed: enabled, and the channel is a web one.

        TransportPlugin.send uses it for every outbound event; the inbound
        lifecycle hooks below call it too.
        """
        return self._enabled and super().owns(channel)

    def _set_working(self, channel: Channel, value: bool) -> None:
        """Change the indicator, broadcasting only on a change."""
        if value == (channel.id in self._working):
            return
        if value:
            self._working.add(channel.id)
        else:
            self._working.discard(channel.id)
        self._broadcast(channel.id, {"type": "working", "conversation": channel.scope,
                                     "value": value})

    def _work_owed(self, channel: Channel) -> bool:
        """True while a reply is still owed on a channel."""
        if any(e.admitted for e in self._unsent.get(channel.id, [])):
            return True
        return bool(channel.pending_tool_call_ids)

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Push the reply to the conversation's browsers; clear "working"
        once no more work is owed."""
        self._broadcast(channel.id, {"type": "message", "conversation": channel.scope,
                                     "text": event.text})
        # With a tracker the settle callback clears "working" once the
        # channel has drained; without one, clear here.
        if self._idle_watch is None and not self._work_owed(channel):
            self._set_working(channel, False)

    async def _on_channel_settled(self, channel_id: str) -> None:
        """Idle-tracker callback: a channel drained; clear "working" if no reply is owed."""
        # The watch fires for every channel; only web's matter.
        if not channel_id.startswith("web:") or self._registry is None:
            return
        channel = self._registry.get(channel_id)
        if channel is None or not self.owns(channel):
            return
        # Tool calls still pending or unsent messages held: stay working;
        # the next settle re-checks.
        if not self._work_owed(channel):
            self._set_working(channel, False)

    async def send_thinking(self, channel: Channel, event: OutboundThinking) -> None:
        """Push reasoning content to the conversation's browsers."""
        self._broadcast(channel.id, {"type": "thinking", "conversation": channel.scope,
                                     "text": event.text})

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        """Push intermediate text, minus any inline thinking, if any is left."""
        visible = strip_thinking(event.text)
        if visible:
            self._broadcast(channel.id, {"type": "progress", "conversation": channel.scope,
                                         "text": visible})

    async def send_tool_status(self, channel: Channel, event: OutboundToolStatus) -> None:
        """Push a tool call's status line to the conversation's browsers."""
        self._broadcast(channel.id, {
            "type": "tool_status", "conversation": channel.scope, "tool": event.tool_name,
            "status": event.status,
            "summary": event.args_summary or event.result_summary or "",
        })

    # -- message lifecycle hooks ---------------------------------------------

    @hookimpl
    async def on_message_admitted(
        self, channel: Channel, trace_id: str, sender: str, text: str
    ) -> None:
        if not self.owns(channel):
            return
        # Only _on_send calls on_message on a web channel, and it registers
        # its entry under this id first, so an unmatched id has no real
        # source; ignore it rather than invent an entry.
        entry = self._entry_for(channel.id, trace_id)
        if entry is None:
            logger.debug("web: admission for unknown trace id %s", trace_id)
            return
        entry.admitted = True
        self._set_working(channel, True)

    @hookimpl
    async def on_message_rejected(
        self, channel: Channel, trace_id: str, sender: str, text: str
    ) -> None:
        if not self.owns(channel):
            return
        # Match this send's own entry by id; an unknown id removes nothing
        # and is reported to clients with a null id.
        entry = self._entry_for(channel.id, trace_id)
        if entry is not None:
            self._remove_entry(channel.id, entry)
        self._broadcast(channel.id, {"type": "rejected", "conversation": channel.scope,
                                     "text": text, "id": entry.id if entry else None})

    @hookimpl
    async def on_message_persisted(
        self, channel: Channel, trace_id: str, rowid: int | None, text: str, meta: dict
    ) -> None:
        if not self.owns(channel):
            return
        # Ids we never saw (notification-originated turns) are a no-op.
        entry = self._entry_for(channel.id, trace_id)
        if entry is not None:
            self._remove_entry(channel.id, entry)
