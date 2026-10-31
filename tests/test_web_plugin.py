"""Tests for corvidae.channels.web.WebPlugin (issue #87).

Drives a real WebPlugin listening on 127.0.0.1 with an ephemeral port,
through aiohttp's client (and raw sockets where the client would refuse
to send what a hostile or broken peer sends). The plugin runs inside a
real plugin manager with a real ChannelRegistry and PersistencePlugin;
the Agent is a stub whose on_message mirrors Agent.on_message's control
flow (gate → reject-wins → rejected or admitted), so gate plugins drive
the real hook interleavings.

Interface the tests pin (private names the design names):
  module corvidae.channels.web:
    WebPlugin, _Client, _Unsent,
    _ascii_equal, _canonical_host, _normalise_origin, _compute_origins,
    _resolve_token
  WebPlugin attributes: _enabled, _host, _port (configured), port (bound),
    _token, _session, _clients, _handlers, _unsent, _working, _runner
  WebPlugin methods: _build_app(), _broadcast(channel_id, frame) (sync),
    _close_client(client, code, message), _fetch_history_rows(channel_id),
    _rows_to_frames(rows), on_config_reload(config)
"""

from __future__ import annotations

import asyncio
import base64
import glob
import hashlib
import hmac
import json
import logging
import os
import socket
import sqlite3
import time
import tomllib
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from helpers import CORRUPT_MESSAGES, insert_raw_message, skip_warnings
from aiohttp import web

from corvidae.channel import Channel, ChannelRegistry
from corvidae.hooks import (
    _check_hook_arg_binding,
    create_plugin_manager,
    hookimpl,
    resolve_reject_wins,
)
from corvidae.outbound import OutboundMessage, OutboundProgress, OutboundThinking, OutboundToolStatus
from corvidae.persistence import PersistencePlugin

# The plugin under test does not exist until the green phase; this import
# failing is the red phase's expected failure.
from corvidae.channels import web as web_mod
from corvidae.channels.web import (
    WebPlugin,
    _compute_origins,
    _normalise_origin,
    _resolve_token,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN = "test-token-abcdefghijklmnop"
OTHER_TOKEN = "other-token-qrstuvwxyz0123"
REPO_ROOT = Path(__file__).resolve().parent.parent


def session_for(token: str) -> str:
    """The session cookie value the design derives from a token."""
    # Independent re-statement of the design's formula, so a change to the
    # derivation (which would break R5's restart survival) fails here.
    return hmac.new(
        token.encode("ascii"), b"corvidae-web-session", hashlib.sha256
    ).hexdigest()


def free_port() -> int:
    """Pick a currently free TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Agent stub and gate plugins
# ---------------------------------------------------------------------------

class AgentStub:
    """Registered as "agent". on_message mirrors Agent.on_message's flow."""

    def __init__(self, pm) -> None:
        self.pm = pm
        self.started = True
        self.queues: dict = {}
        # (channel, sender, text, trace_id) per admitted message.
        self.calls: list[tuple] = []

    @hookimpl
    async def on_message(self, channel, sender: str, text: str,
                         trace_id: str) -> None:
        # Use the transport-minted id for every hook, as Agent does.
        tid = trace_id
        results = await self.pm.ahook.should_process_message(
            channel=channel, sender=sender, text=text, trace_id=tid,
        )
        # Reject-wins resolution, then exactly one of rejected/admitted.
        if resolve_reject_wins(results) is False:
            try:
                await self.pm.ahook.on_message_rejected(
                    channel=channel, trace_id=tid, sender=sender, text=text,
                )
            except Exception:
                logging.getLogger(__name__).warning("rejected failed", exc_info=True)
            return
        try:
            await self.pm.ahook.on_message_admitted(
                channel=channel, trace_id=tid, sender=sender, text=text,
            )
        except Exception:
            logging.getLogger(__name__).warning("admitted failed", exc_info=True)
        self.calls.append((channel, sender, text, tid))


class HoldGate:
    """should_process_message blocks until released, then returns `result`."""

    def __init__(self, result: bool | None = None) -> None:
        self.result = result
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        self.entered.set()
        await self.release.wait()
        return self.result


class OrderedGate:
    """Per-call behaviour keyed on call order: call i awaits events[i]
    (if any) and returns results[i]."""

    def __init__(self, events: list, results: list) -> None:
        self.events = events
        self.results = results
        self.count = 0

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        i = self.count
        self.count += 1
        if self.events[i] is not None:
            await self.events[i].wait()
        return self.results[i]


class VetoGate:
    """Vetoes every message."""

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        return False


class RaiseGate:
    """Raises out of the gate, as a broken third-party gate would."""

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        raise RuntimeError("gate exploded")


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

class WSConn:
    """A WebSocket client that records every frame in arrival order.

    `wait(pred)` scans forward from a cursor, so successive waits observe
    frames in order; `frames` keeps the full record for ordering checks.
    """

    def __init__(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self.ws = ws
        self.frames: list[dict] = []
        self._pos = 0
        self._changed = asyncio.Event()
        self.closed = False
        self.close_code: int | None = None
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        # Record text frames until the socket closes.
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self.frames.append(json.loads(msg.data))
                self._changed.set()
        self.closed = True
        self.close_code = self.ws.close_code
        self._changed.set()

    async def send(self, **frame) -> None:
        await self.ws.send_json(frame)

    async def wait(self, pred, timeout: float = 3.0) -> dict:
        """Return the first frame at/after the cursor matching pred."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            for i in range(self._pos, len(self.frames)):
                if pred(self.frames[i]):
                    self._pos = i + 1
                    return self.frames[i]
            if self.closed:
                raise AssertionError(f"socket closed; frames={self.frames}")
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise AssertionError(f"no matching frame; frames={self.frames}")
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except asyncio.TimeoutError:
                pass

    async def wait_type(self, type_: str, timeout: float = 3.0, **fields) -> dict:
        """Wait for a frame of `type_` whose fields equal `fields`."""
        return await self.wait(
            lambda f: f.get("type") == type_
            and all(f.get(k) == v for k, v in fields.items()),
            timeout,
        )

    async def sync(self) -> None:
        """Round-trip a `list` so every frame enqueued before it has arrived."""
        await self.send(type="list")
        await self.wait(lambda f: f.get("type") == "conversations"
                        or (f.get("type") == "error" and f.get("request") == "list"))

    def types(self) -> list[str]:
        return [f.get("type") for f in self.frames]

    async def wait_closed(self, timeout: float = 3.0) -> int | None:
        await asyncio.wait_for(self._task, timeout)
        return self.close_code


async def wait_until(cond, timeout: float = 3.0) -> None:
    """Poll `cond()` until true or fail."""
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def ws_text_frame(text: str) -> bytes:
    """Encode a masked client→server WebSocket text frame."""
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    n = len(payload)
    header = bytes([0x81])
    if n < 126:
        header += bytes([0x80 | n])
    elif n < 65536:
        header += bytes([0x80 | 126]) + n.to_bytes(2, "big")
    else:
        header += bytes([0x80 | 127]) + n.to_bytes(8, "big")
    return header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


class RawWS:
    """A WebSocket peer on a raw socket that never reads after the handshake."""

    def __init__(self, reader, writer) -> None:
        self.reader = reader
        self.writer = writer

    @classmethod
    async def connect(cls, port: int, cookie: str, origin: str) -> "RawWS":
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        # Shrink the receive buffer so the server's send path backs up fast.
        sock = writer.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            f"Origin: {origin}\r\nCookie: corvidae_web={cookie}\r\n\r\n"
        )
        writer.write(request.encode("ascii"))
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        assert b" 101 " in head.split(b"\r\n", 1)[0], head
        return cls(reader, writer)

    async def send(self, **frame) -> None:
        self.writer.write(ws_text_frame(json.dumps(frame)))
        await self.writer.drain()

    def close(self) -> None:
        self.writer.transport.abort()


async def raw_http(port: int, request: bytes) -> bytes:
    """Send raw request bytes and return the status line."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(request)
        await writer.drain()
        return await reader.readline()
    finally:
        writer.transport.abort()


# ---------------------------------------------------------------------------
# Environment fixture
# ---------------------------------------------------------------------------

class Env:
    """One running WebPlugin plus its collaborators and an HTTP session."""

    def __init__(self) -> None:
        self.pm = None
        self.registry: ChannelRegistry | None = None
        self.agent: AgentStub | None = None
        self.persistence: PersistencePlugin | None = None
        self.plugin: WebPlugin | None = None
        self.http: aiohttp.ClientSession | None = None
        self.conns: list[WSConn] = []
        self.raws: list[RawWS] = []

    @property
    def port(self) -> int:
        return self.plugin.port

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def origin(self) -> str:
        return self.base

    def cookie(self, token: str = TOKEN) -> dict:
        return {"Cookie": f"corvidae_web={session_for(token)}"}

    async def connect(self, conversation: str | None = None, **headers) -> WSConn:
        """Open an authenticated WS; optionally subscribe and await history."""
        hdrs = {"Origin": self.origin, **self.cookie(), **headers}
        ws = await self.http.ws_connect(f"{self.base}/ws", headers=hdrs)
        conn = WSConn(ws)
        self.conns.append(conn)
        if conversation is not None:
            await conn.send(type="subscribe", conversation=conversation)
            await conn.wait_type("history", conversation=conversation)
        return conn

    async def raw(self) -> RawWS:
        r = await RawWS.connect(self.port, session_for(TOKEN), self.origin)
        self.raws.append(r)
        return r

    async def raw_subscribed(self, name: str = "main") -> RawWS:
        """A raw peer subscribed to `name`, past its history fetch (so
        broadcasts reach its queue, not the subscribe buffer)."""
        r = await self.raw()
        await r.send(type="subscribe", conversation=name)
        await wait_until(lambda: any(c.conversation == f"web:{name}" and c.buffer is None
                                     for c in self.plugin._clients))
        return r

    def channel(self, name: str = "main") -> Channel:
        return self.registry.get_or_create("web", name)

    def client_for(self, conversation_id: str):
        """The single _Client subscribed to conversation_id."""
        found = [c for c in self.plugin._clients if c.conversation == conversation_id]
        assert len(found) == 1, found
        return found[0]


@pytest.fixture(autouse=True)
def _no_env_token(monkeypatch):
    """Tests control CORVIDAE_WEB_TOKEN explicitly."""
    monkeypatch.delenv("CORVIDAE_WEB_TOKEN", raising=False)


@pytest.fixture
async def make_env(db):
    """Factory: build and start an Env; everything is torn down afterwards."""
    envs: list[Env] = []

    async def make(web_block=None, *, ready: bool = True, persistence: bool = True,
                   start: bool = True, token: str | None = TOKEN,
                   use_db: bool = True, idle: bool = False) -> Env:
        env = Env()
        env.pm = create_plugin_manager()
        env.registry = ChannelRegistry()
        env.pm.register(env.registry, name="registry")
        # Real persistence plugin with the test DB injected (not started).
        if persistence:
            env.persistence = PersistencePlugin(env.pm)
            env.persistence.db = db if use_db else None
            env.pm.register(env.persistence, name="persistence")
        env.agent = AgentStub(env.pm)
        env.agent.started = ready
        env.pm.register(env.agent, name="agent")
        env.plugin = WebPlugin(env.pm)
        env.pm.register(env.plugin, name="web")
        # Optional kernel clock + idle tracker, as Runtime registers them.
        env.clock = env.idle = None
        if idle:
            from fake_clock import install_kernel_clock

            env.clock, env.idle = install_kernel_clock(env.pm)
        block = {"host": "127.0.0.1", "port": 0} if web_block is None else web_block
        if token is not None and isinstance(block, dict) and "token" not in block:
            block = {**block, "token": token}
        config = {"web": block}
        env.config = config
        await env.plugin.on_init(pm=env.pm, config=config)
        if start:
            await env.plugin.on_start(config=config)
        env.http = aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar())
        envs.append(env)
        return env

    yield make

    # Teardown: stop the plugin first (bounded, and it releases handlers a
    # failed test left stuck), then close what's left client-side.
    for env in envs:
        await env.plugin.on_stop()
        if env.idle is not None:
            env.idle.stop()
            await env.clock.close()
        for r in env.raws:
            r.close()
        for c in env.conns:
            await c.ws.close()
        await env.http.close()


# ===========================================================================
# 1. Config
# ===========================================================================

class TestConfig:

    async def test_absent_block_is_inert_and_quiet(self, caplog):
        caplog.set_level(logging.DEBUG)
        pm = create_plugin_manager()
        registry = ChannelRegistry()
        pm.register(registry, name="registry")
        plugin = WebPlugin(pm)
        pm.register(plugin, name="web")
        await plugin.on_init(pm=pm, config={})
        await plugin.on_start(config={})
        assert plugin._enabled is False
        assert plugin._runner is None
        # Hooks on a web channel are no-ops when inert.
        ch = registry.get_or_create("web", "main")
        await plugin.send(channel=ch, event=OutboundMessage(text="x"))
        await plugin.on_message_admitted(channel=ch, trace_id="c", sender="web", text="x")
        await plugin.on_config_reload(config={})
        await plugin.on_stop()
        loud = [r for r in caplog.records if r.levelno > logging.DEBUG
                and r.name.startswith("corvidae")]
        assert loud == []

    async def test_bare_block_enables_with_defaults(self):
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(), name="registry")
        plugin = WebPlugin(pm)
        await plugin.on_init(pm=pm, config={"web": None})
        assert plugin._enabled is True
        assert plugin._host == "127.0.0.1"
        assert plugin._port == 8765

    @pytest.mark.parametrize("block, field", [
        ([], "web"),
        ({"host": ""}, "host"),
        ({"port": "x"}, "port"),
        ({"host": "localhost", "port": 0}, "port"),
        ({"token": 123}, "token"),
        ({"allowed_origins": ["http://box/path"]}, "allowed_origins"),
        ({"host": "0.0.0.0", "token": TOKEN}, "allowed_origins"),
    ])
    async def test_malformed_field_raises_naming_it(self, block, field):
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(), name="registry")
        plugin = WebPlugin(pm)
        with pytest.raises(ValueError, match=field) as exc:
            await plugin.on_init(pm=pm, config={"web": block})
        assert str(exc.value).startswith("web:")

    async def test_non_loopback_bind_warns_once(self, make_env, caplog):
        caplog.set_level(logging.DEBUG)
        await make_env({"host": "0.0.0.0", "port": 0,
                        "allowed_origins": ["http://box.example:8765"]})
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        # The missing idle tracker (no Runtime here) warns separately (#172).
        loopback = [r for r in warnings if "loopback" in r.getMessage()]
        assert len(loopback) == 1, [r.getMessage() for r in warnings]
        assert "not a loopback address" in loopback[0].getMessage()

    async def test_bind_failure_raises(self, make_env):
        env = await make_env()
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(), name="registry")
        other = WebPlugin(pm)
        cfg = {"web": {"host": "127.0.0.1", "port": env.port, "token": TOKEN}}
        await other.on_init(pm=pm, config=cfg)
        with pytest.raises(OSError):
            await other.on_start(config=cfg)
        await other.on_stop()


# ===========================================================================
# 2. Token
# ===========================================================================

class TestToken:

    def test_resolution_order(self, monkeypatch):
        # Neither → generated; env → env; config beats env; empty falls through.
        token, generated = _resolve_token({})
        assert generated is True and len(token) >= 43
        monkeypatch.setenv("CORVIDAE_WEB_TOKEN", OTHER_TOKEN)
        assert _resolve_token({"token": ""}) == (OTHER_TOKEN, False)
        assert _resolve_token({"token": TOKEN}) == (TOKEN, False)

    @pytest.mark.parametrize("bad", ["short-token", "a" * 20 + "+", "a" * 20 + "é"])
    async def test_bad_provided_token_rejected(self, bad):
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(), name="registry")
        plugin = WebPlugin(pm)
        with pytest.raises(ValueError, match="token"):
            await plugin.on_init(pm=pm, config={"web": {"token": bad}})

    async def test_generated_token_logged_once_as_url(self, make_env, caplog):
        caplog.set_level(logging.DEBUG)
        env = await make_env(token=None)
        token = env.plugin._token
        hits = [r for r in caplog.records if token in r.getMessage()]
        assert len(hits) == 1
        rec = hits[0]
        assert rec.levelno == logging.WARNING
        assert f"?token={token}" in rec.getMessage()
        assert not rec.args
        _assert_token_not_in_structured_fields(caplog.records, token)

    async def test_provided_token_never_logged(self, make_env, caplog):
        caplog.set_level(logging.DEBUG)
        env = await make_env()
        async with env.http.get(f"{env.base}/?token={TOKEN}") as resp:
            assert resp.status == 200
        async with env.http.get(f"{env.base}/?token=wrong-{TOKEN}") as resp:
            assert resp.status == 401
        await env.plugin.on_stop()
        for r in caplog.records:
            assert TOKEN not in r.getMessage()
        _assert_token_not_in_structured_fields(caplog.records, TOKEN)


def _parse_csp(header: str) -> dict[str, set[str]]:
    """CSP header → {directive: set of sources}."""
    out: dict[str, set[str]] = {}
    for part in header.split(";"):
        words = part.split()
        if words:
            out[words[0]] = set(words[1:])
    return out


def _expected_csp(ws_sources: set[str]) -> dict[str, set[str]]:
    """The design's policy, with the given explicit ws sources. IPv6
    literals have no CSP host-source form, so `ws://[::1]` is never listed."""
    return {
        "default-src": {"'self'"}, "script-src": {"'self'"}, "style-src": {"'self'"},
        "connect-src": {"'self'"} | ws_sources, "img-src": {"'self'", "data:"},
        "object-src": {"'none'"}, "base-uri": {"'none'"}, "form-action": {"'none'"},
        "frame-ancestors": {"'none'"},
    }


_STD_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {
    "message", "asctime", "taskName"}


def _assert_token_not_in_structured_fields(records, token: str) -> None:
    """No record carries the token in `args` or in an `extra` field."""
    for r in records:
        assert token not in repr(r.args)
        for k, v in vars(r).items():
            if k not in _STD_ATTRS:
                assert token not in repr(v), (k, r)


# ===========================================================================
# 3. Origins and hosts
# ===========================================================================

class TestOrigins:

    @pytest.mark.parametrize("given, expected", [
        ("HTTP://Example.COM/", "http://example.com"),
        ("http://[0:0:0:0:0:0:0:1]:8765", "http://[::1]:8765"),
        ("https://box.example:443", "https://box.example"),
    ])
    def test_normalise_origin(self, given, expected):
        assert _normalise_origin(given) == expected

    @staticmethod
    def _trio(port: int = 8000):
        return {f"http://127.0.0.1:{port}", f"http://localhost:{port}",
                f"http://[::1]:{port}"}

    @pytest.mark.parametrize("host, base", [
        ("127.0.0.1", "http://127.0.0.1:8000"),
        ("0:0:0:0:0:0:0:1", "http://[::1]:8000"),
    ])
    def test_loopback_hosts_get_the_trio(self, host, base):
        origins, hosts, login_base = _compute_origins(host, 8000, None)
        assert set(origins) == self._trio()
        assert login_base == base
        assert {"127.0.0.1:8000", "localhost:8000", "[::1]:8000"} <= hosts

    def test_configured_https_default_port(self):
        origins, hosts, login_base = _compute_origins(
            "127.0.0.1", 8000, ["https://box.example:443"])
        assert "https://box.example" in origins
        assert _normalise_origin("https://box.example") in origins
        assert {"box.example", "box.example:443"} <= hosts
        assert login_base == "https://box.example"

    async def test_end_to_end_configured_origin_and_host(self, make_env):
        port = free_port()
        env = await make_env({"host": "127.0.0.1", "port": port,
                              "allowed_origins": [f"http://mybox.local:{port}"]})
        hdrs = {"Origin": f"http://mybox.local:{port}",
                "Host": f"mybox.local:{port}", **env.cookie()}
        ws = await env.http.ws_connect(f"{env.base}/ws", headers=hdrs)
        conn = WSConn(ws)
        env.conns.append(conn)
        await conn.sync()


# ===========================================================================
# 4. Auth and HTTP surface
# ===========================================================================

class TestAuth:

    async def test_login_sets_cookie_and_refreshes(self, make_env):
        env = await make_env()
        async with env.http.get(f"{env.base}/?token={TOKEN}", allow_redirects=False) as r:
            assert r.status == 200
            body = await r.text()
            cookies = r.headers.getall("Set-Cookie")
        assert 'http-equiv="refresh"' in body and "url=/" in body
        assert len(cookies) == 1
        c = cookies[0]
        assert c.startswith(f"corvidae_web={session_for(TOKEN)};")
        lowered = c.lower()
        assert "httponly" in lowered
        assert "samesite=strict" in lowered
        assert "max-age=34560000" in lowered
        assert "path=/" in lowered
        assert "secure" not in lowered

    async def test_secure_cookie_for_https_origin_host(self, make_env):
        env = await make_env({"host": "127.0.0.1", "port": 0,
                              "allowed_origins": ["https://box.example"]})
        async with env.http.get(f"{env.base}/?token={TOKEN}",
                                headers={"Host": "box.example"}) as r:
            assert r.status == 200
            assert "secure" in r.headers["Set-Cookie"].lower()
            csp = _parse_csp(r.headers["Content-Security-Policy"])
            assert "wss://box.example" in csp["connect-src"]
        # Same server, plain-http loopback Host: no Secure.
        async with env.http.get(f"{env.base}/?token={TOKEN}") as r:
            assert r.status == 200
            assert "secure" not in r.headers["Set-Cookie"].lower()

    @pytest.mark.parametrize("query", [f"wrong-{TOKEN}", "%C3%A9" * 20])
    async def test_bad_token_401(self, make_env, query):
        env = await make_env()
        async with env.http.get(f"{env.base}/?token={query}") as r:
            assert r.status == 401
            assert "Set-Cookie" not in r.headers

    async def test_undecodable_cookie_bytes_401_not_500(self, make_env, caplog):
        caplog.set_level(logging.DEBUG)
        env = await make_env()
        req = (f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{env.port}\r\n"
               "Connection: close\r\n").encode("ascii") \
            + b"Cookie: corvidae_web=\xff\xfe\r\n\r\n"
        status = await raw_http(env.port, req)
        assert b" 401 " in status, status
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []

    @pytest.mark.parametrize("path", ["/", "/static/app.js", "/ws"])
    async def test_routes_need_cookie(self, make_env, path):
        env = await make_env()
        async with env.http.get(f"{env.base}{path}") as r:
            assert r.status == 401

    @pytest.mark.parametrize("origin", ["http://evil.example", None])
    async def test_ws_bad_origin_403(self, make_env, origin):
        env = await make_env()
        hdrs = dict(env.cookie())
        if origin is not None:
            hdrs["Origin"] = origin
        with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
            await env.http.ws_connect(f"{env.base}/ws", headers=hdrs)
        assert exc.value.status == 403

    async def test_bad_host_403(self, make_env):
        env = await make_env()
        async with env.http.get(f"{env.base}/", headers={"Host": "evil.example",
                                                         **env.cookie()}) as r:
            assert r.status == 403

    def _assert_security_headers(self, headers, port):
        assert _parse_csp(headers["Content-Security-Policy"]) == _expected_csp(
            {f"ws://127.0.0.1:{port}", f"ws://localhost:{port}"})
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["Cache-Control"] == "no-store"

    async def test_security_headers_on_200_401_403(self, make_env):
        env = await make_env()
        async with env.http.get(f"{env.base}/", headers=env.cookie()) as r:
            assert r.status == 200
            self._assert_security_headers(r.headers, env.port)
        async with env.http.get(f"{env.base}/") as r:
            assert r.status == 401
            self._assert_security_headers(r.headers, env.port)
        async with env.http.get(f"{env.base}/", headers={"Host": "evil.example"}) as r:
            assert r.status == 403
            self._assert_security_headers(r.headers, env.port)

    async def test_500_has_headers_and_logs_no_token(self, make_env, monkeypatch, caplog):
        caplog.set_level(logging.DEBUG)
        original = WebPlugin._build_app

        async def boom(request):
            raise RuntimeError("kaboom")

        def build(self):
            app = original(self)
            app.router.add_get("/boom", boom)
            return app

        monkeypatch.setattr(WebPlugin, "_build_app", build)
        env = await make_env()
        async with env.http.get(f"{env.base}/boom?token={TOKEN}", headers=env.cookie()) as r:
            assert r.status == 500
            self._assert_security_headers(r.headers, env.port)
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors
        for r in caplog.records:
            text = r.getMessage() + (r.exc_text or "")
            assert "token=" not in text
            assert TOKEN not in text

    async def test_cookie_survives_restart_with_same_token(self, make_env):
        env1 = await make_env()
        async with env1.http.get(f"{env1.base}/?token={TOKEN}") as r:
            cookie = r.headers["Set-Cookie"].split(";", 1)[0]
        await env1.plugin.on_stop()
        env2 = await make_env()
        async with env2.http.get(f"{env2.base}/", headers={"Cookie": cookie}) as r:
            assert r.status == 200

    async def test_cookie_rejected_after_token_change(self, make_env):
        env1 = await make_env()
        async with env1.http.get(f"{env1.base}/?token={TOKEN}") as r:
            cookie = r.headers["Set-Cookie"].split(";", 1)[0]
        env2 = await make_env(token=OTHER_TOKEN)
        async with env2.http.get(f"{env2.base}/", headers={"Cookie": cookie}) as r:
            assert r.status == 401

    async def test_no_access_log(self, make_env, caplog):
        caplog.set_level(logging.INFO, logger="aiohttp.access")
        env = await make_env()
        async with env.http.get(f"{env.base}/?token={TOKEN}") as r:
            assert r.status == 200
        await env.plugin.on_stop()
        assert [r for r in caplog.records if r.name == "aiohttp.access"] == []

    async def test_access_log_positive_control(self, caplog):
        # Proves the assertion above can fail: a default runner does log.
        caplog.set_level(logging.INFO, logger="aiohttp.access")
        app = web.Application()

        async def ok(request):
            return web.Response(text="ok")

        app.router.add_get("/", ok)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"http://127.0.0.1:{port}/?token=x") as r:
                    assert r.status == 200
        finally:
            await runner.cleanup()
        assert len([r for r in caplog.records if r.name == "aiohttp.access"]) == 1


# ===========================================================================
# 5. Inbound frames
# ===========================================================================

class TestInbound:

    async def test_send_fires_on_message_and_broadcasts_user(self, make_env):
        env = await make_env()
        conn = await env.connect("main")
        await conn.send(type="send", conversation="main", text="hello")
        user = await conn.wait_type("user", text="hello")
        assert user["conversation"] == "main"
        assert isinstance(user["id"], str) and user["id"]
        await wait_until(lambda: env.agent.calls)
        channel, sender, text, _ = env.agent.calls[0]
        assert channel.id == "web:main"
        assert sender == "web"
        assert text == "hello"

    async def test_send_not_ready(self, make_env):
        env = await make_env()
        conn = await env.connect("main")
        env.agent.started = False
        await conn.send(type="send", conversation="main", text="hello")
        err = await conn.wait_type("error")
        assert err["request"] == "send" and err["code"] == "starting"
        env.agent.started = True
        await conn.sync()
        assert "user" not in conn.types()
        assert env.agent.calls == []

    @pytest.mark.parametrize("request_type", ["send", "subscribe"])
    @pytest.mark.parametrize("name", ["a" * 65, "Main", "main\n"])
    async def test_bad_names(self, make_env, request_type, name):
        env = await make_env()
        conn = await env.connect()
        await conn.send(type=request_type, conversation=name, text="hi")
        err = await conn.wait_type("error")
        assert err["request"] == request_type and err["code"] == "bad_request"
        assert "history" not in conn.types()
        assert env.agent.calls == []

    @pytest.mark.parametrize("text", [5, "\ud800"])
    async def test_bad_text(self, make_env, text):
        env = await make_env()
        conn = await env.connect("main")
        await conn.send(type="send", conversation="main", text=text)
        err = await conn.wait_type("error")
        assert err["request"] == "send" and err["code"] == "bad_request"
        await conn.sync()
        assert env.agent.calls == []
        assert "user" not in conn.types()

    @pytest.mark.parametrize("raw, expected_request", [
        ("{", None),
        ("[" * 100_000 + "]" * 100_000, None),
        ('{"type": "nonsense"}', "nonsense"),
    ])
    async def test_malformed_frames_answered_and_socket_survives(self, make_env, raw,
                                                                    expected_request):
        env = await make_env()
        conn = await env.connect()
        await conn.ws.send_str(raw)
        err = await conn.wait_type("error")
        assert err["code"] == "bad_request"
        assert err["request"] == expected_request
        # The socket still works afterwards.
        await conn.send(type="list")
        await conn.wait_type("conversations")


# ===========================================================================
# 6. Outbound hooks
# ===========================================================================

class TestOutbound:

    async def test_send_hooks_ignore_non_web_channels(self, make_env):
        env = await make_env()
        calls = []
        env.plugin._broadcast = lambda *a, **k: calls.append(a)
        irc = env.registry.get_or_create("irc", "main")
        await env.plugin.send(channel=irc, event=OutboundMessage(text="x"))
        await env.plugin.send(channel=irc, event=OutboundThinking(text="x"))
        await env.plugin.send(channel=irc, event=OutboundProgress(text="x"))
        await env.plugin.send(channel=irc, event=OutboundToolStatus(tool_name="t", status="dispatched", args_summary="a", result_summary=None))
        await env.plugin.on_message_admitted(channel=irc, trace_id="c",
                                             sender="x", text="x")
        await env.plugin.on_message_rejected(channel=irc, trace_id="c",
                                             sender="x", text="x")
        await env.plugin.on_message_persisted(channel=irc, trace_id="c",
                                              rowid=1, text="x", meta={})
        assert calls == []
        assert env.plugin._unsent == {}
        assert env.plugin._working == set()

    async def test_fan_out_to_subscribers_only(self, make_env):
        env = await make_env()
        a = await env.connect("main")
        b = await env.connect("main")
        other = await env.connect("other")
        await env.pm.ahook.send(channel=env.channel("main"), event=OutboundMessage(text="reply"))
        for c in (a, b):
            f = await c.wait_type("message", text="reply")
            assert f["conversation"] == "main"
        await other.sync()
        assert "message" not in other.types()

    async def test_frame_types(self, make_env):
        env = await make_env()
        conn = await env.connect("main")
        ch = env.channel("main")
        await env.pm.ahook.send(channel=ch, event=OutboundThinking(text="pondering"))
        await env.pm.ahook.send(channel=ch, event=OutboundProgress(text="<think>secret</think>visible"))
        await env.pm.ahook.send(channel=ch, event=OutboundToolStatus(tool_name="read_file", status="dispatched", args_summary='{"p":1}'))
        await env.pm.ahook.send(channel=ch, event=OutboundToolStatus(tool_name="read_file", status="completed", result_summary="contents"))
        assert (await conn.wait_type("thinking"))["text"] == "pondering"
        assert (await conn.wait_type("progress"))["text"] == "visible"
        d = await conn.wait_type("tool_status", status="dispatched")
        assert d["tool"] == "read_file" and d["summary"] == '{"p":1}'
        c = await conn.wait_type("tool_status", status="completed")
        assert c["summary"] == "contents"
        for f in conn.frames:
            if f["type"] != "history":
                assert f["conversation"] == "main"


# ===========================================================================
# 7. Working indicator and unsent messages
# ===========================================================================

def _tid_of(env, text: str) -> str:
    return next(c[3] for c in env.agent.calls if c[2] == text)


async def _persist(env, tid: str, name: str = "main") -> None:
    await env.pm.ahook.on_message_persisted(
        channel=env.channel(name), trace_id=tid, rowid=1, text="", meta={})


class TestWorking:

    async def test_indicator_lifecycle(self, make_env):
        env = await make_env()
        conn = await env.connect("main")
        ch = env.channel("main")
        await conn.send(type="send", conversation="main", text="one")
        await conn.wait_type("working", value=True)
        await wait_until(lambda: env.agent.calls)
        # Admitted but not yet persisted: a reply doesn't turn it off.
        await env.pm.ahook.send(channel=ch, event=OutboundMessage(text="r1"))
        await conn.wait_type("message", text="r1")
        await conn.sync()
        assert "web:main" in env.plugin._working
        assert not any(f["type"] == "working" and f["value"] is False for f in conn.frames)
        # Persisted, queue empty, nothing pending → off.
        await _persist(env, _tid_of(env, "one"))
        await env.pm.ahook.send(channel=ch, event=OutboundMessage(text="r2"))
        await conn.wait_type("working", value=False)
        assert "web:main" not in env.plugin._working
    # -- #172: the indicator follows the idle tracker's settle signal --------

    async def _send(self, env, conn, text):
        """Send ``text`` from the browser and wait until it is admitted."""
        n = len(env.agent.calls)
        await conn.send(type="send", conversation="main", text=text)
        await wait_until(lambda: len(env.agent.calls) > n)

    async def test_no_queue_reference_in_module(self):
        # R1: web never reaches for another component's queues.
        src = Path(web_mod.__file__).read_text()
        assert "queues" not in src

    async def test_clears_on_settle_after_final_reply(self, make_env):
        # R2: working stays true through the reply, clears at settle, and
        # the message frame precedes the working=false frame.
        env = await make_env(idle=True)
        conn = await env.connect("main")
        ch = env.channel("main")
        await self._send(env, conn, "one")
        env.idle.note_activity(ch.id)
        await _persist(env, _tid_of(env, "one"))
        await env.pm.ahook.send(channel=ch, event=OutboundMessage(text="r1"))
        await conn.wait_type("message", text="r1")
        await conn.sync()
        assert "web:main" in env.plugin._working
        env.idle.note_settled(ch.id)
        await env.clock.advance(0)
        await conn.wait_type("working", value=False)
        kinds = [(f["type"], f.get("value", f.get("text"))) for f in conn.frames]
        assert kinds.index(("message", "r1")) < kinds.index(("working", False))
        assert "web:main" not in env.plugin._working

    async def test_stays_working_while_tool_calls_pending(self, make_env):
        # R2: pending tool calls hold the indicator until a later settle.
        env = await make_env(idle=True)
        conn = await env.connect("main")
        ch = env.channel("main")
        await self._send(env, conn, "one")
        env.idle.note_activity(ch.id)
        await _persist(env, _tid_of(env, "one"))
        ch.pending_tool_call_ids.add("call-1")
        env.idle.note_settled(ch.id)
        await env.clock.advance(0)
        await conn.sync()
        assert "web:main" in env.plugin._working
        ch.pending_tool_call_ids.clear()
        env.idle.note_activity(ch.id)
        env.idle.note_settled(ch.id)
        await env.clock.advance(0)
        await conn.wait_type("working", value=False)

    async def test_stays_working_between_two_queued_messages(self, make_env):
        # R3: an admitted-but-unsent second message keeps the indicator on
        # across the first turn's settle; it clears after the second.
        env = await make_env(idle=True)
        conn = await env.connect("main")
        ch = env.channel("main")
        await self._send(env, conn, "one")
        await self._send(env, conn, "two")
        env.idle.note_activity(ch.id)
        await _persist(env, _tid_of(env, "one"))
        await env.pm.ahook.send(channel=ch, event=OutboundMessage(text="r1"))
        env.idle.note_settled(ch.id)
        await env.clock.advance(0)
        await conn.sync()
        assert "web:main" in env.plugin._working
        assert not any(f["type"] == "working" and f["value"] is False for f in conn.frames)
        env.idle.note_activity(ch.id)
        await _persist(env, _tid_of(env, "two"))
        await env.pm.ahook.send(channel=ch, event=OutboundMessage(text="r2"))
        env.idle.note_settled(ch.id)
        await env.clock.advance(0)
        await conn.wait_type("working", value=False)

    async def test_other_transports_settle_ignored(self, make_env):
        # Risk: the watch fires for every channel; only web's count.
        env = await make_env(idle=True)
        conn = await env.connect("main")
        ch = env.channel("main")
        await self._send(env, conn, "one")
        await _persist(env, _tid_of(env, "one"))
        env.idle.note_activity("irc:other")
        env.idle.note_settled("irc:other")
        await env.clock.advance(0)
        await conn.sync()
        assert "web:main" in env.plugin._working
        assert ch.id == "web:main"

    async def test_no_idle_tracker_warns_and_clears_on_reply(self, make_env, caplog):
        # R5: no tracker -> one WARNING naming it, and the final reply clears.
        caplog.set_level(logging.DEBUG)
        env = await make_env()
        warns = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.WARNING and "idle" in r.getMessage()]
        assert len(warns) == 1, warns
        assert "idle tracker" in warns[0] and "clear on reply" in warns[0]
        conn = await env.connect("main")
        ch = env.channel("main")
        await self._send(env, conn, "one")
        await _persist(env, _tid_of(env, "one"))
        await env.pm.ahook.send(channel=ch, event=OutboundMessage(text="r1"))
        await conn.wait_type("working", value=False)

    async def test_rejected_frame(self, make_env):
        env = await make_env()
        env.pm.register(VetoGate(), name="veto")
        conn = await env.connect("main")
        await conn.send(type="send", conversation="main", text="nope")
        user = await conn.wait_type("user", text="nope")
        rej = await conn.wait_type("rejected")
        assert rej["text"] == "nope"
        assert rej["id"] == user["id"]
        assert rej["conversation"] == "main"
        assert "web:main" not in env.plugin._working


def _history_users(history: dict) -> list[dict]:
    return [i for i in history["items"] if i["type"] == "user"]


class TestUnsent:

    async def test_gate_window_visible_to_new_subscriber(self, make_env):
        env = await make_env()
        gate = HoldGate()
        env.pm.register(gate, name="gate")
        a = await env.connect()
        await a.send(type="send", conversation="main", text="hi")
        await asyncio.wait_for(gate.entered.wait(), 3)
        b = await env.connect()
        await b.send(type="subscribe", conversation="main")
        history = await b.wait_type("history")
        last = history["items"][-1]
        assert last["type"] == "user" and last["text"] == "hi"
        assert isinstance(last["id"], str)
        gate.release.set()
        await wait_until(lambda: env.agent.calls)

    async def test_identical_texts_staggered_gates(self, make_env):
        env = await make_env()
        ev_a, ev_b = asyncio.Event(), asyncio.Event()
        gate = OrderedGate([ev_a, ev_b], [None, None])
        env.pm.register(gate, name="gate")
        a = await env.connect()
        b = await env.connect()
        await a.send(type="send", conversation="main", text="hi")
        await wait_until(lambda: gate.count == 1)
        await b.send(type="send", conversation="main", text="hi")
        await wait_until(lambda: gate.count == 2)
        ev_b.set()
        await wait_until(lambda: len(env.agent.calls) == 1)
        await b.sync()  # B's send handler (and its finally) has finished
        c = await env.connect()
        await c.send(type="subscribe", conversation="main")
        history = await c.wait_type("history")
        users = _history_users(history)
        assert [u["text"] for u in users[-2:]] == ["hi", "hi"]
        ev_a.set()
        await wait_until(lambda: len(env.agent.calls) == 2)
        await a.sync()
        entries = env.plugin._unsent["web:main"]
        assert len(entries) == 2
        # A's entry (appended first) is bound to A's id (admitted second).
        assert entries[0].trace_id == env.agent.calls[1][3]
        assert entries[1].trace_id == env.agent.calls[0][3]

    async def test_snapshot_taken_before_fetch(self, make_env):
        env = await make_env()
        a = await env.connect()
        await a.send(type="send", conversation="main", text="hi")
        await wait_until(lambda: env.agent.calls)
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked_fetch(channel_id):
            entered.set()
            await release.wait()
            return []

        env.plugin._fetch_history_rows = blocked_fetch
        b = await env.connect()
        await b.send(type="subscribe", conversation="main")
        await asyncio.wait_for(entered.wait(), 3)
        await _persist(env, _tid_of(env, "hi"))
        release.set()
        history = await b.wait_type("history")
        assert history["items"][-1]["text"] == "hi"

    async def test_persisted_message_leaves_unsent(self, make_env):
        env = await make_env()
        a = await env.connect()
        await a.send(type="send", conversation="main", text="hi")
        await a.send(type="send", conversation="main", text="still waiting")
        await wait_until(lambda: len(env.agent.calls) == 2)
        await _persist(env, _tid_of(env, "hi"))
        # Only the persisted message leaves; the other stays visible.
        assert [e.text for e in env.plugin._unsent["web:main"]] == ["still waiting"]
        b = await env.connect()
        await b.send(type="subscribe", conversation="main")
        history = await b.wait_type("history")
        assert [u["text"] for u in _history_users(history)] == ["still waiting"]
        await _persist(env, _tid_of(env, "still waiting"))
        assert "web:main" not in env.plugin._unsent

    async def test_raising_gate(self, make_env, caplog):
        env = await make_env()
        env.pm.register(RaiseGate(), name="raise")
        a = await env.connect()
        await a.send(type="send", conversation="main", text="hi")
        err = await a.wait_type("error")
        assert err["request"] == "send" and err["code"] == "bad_request"
        assert "web:main" not in env.plugin._unsent
        await a.send(type="list")
        await a.wait_type("conversations")
        assert any(r.exc_info for r in caplog.records if r.levelno >= logging.ERROR)

    async def test_two_tabs_veto_marks_only_its_own(self, make_env):
        env = await make_env()
        ev_a = asyncio.Event()
        gate = OrderedGate([ev_a, None], [False, None])
        env.pm.register(gate, name="gate")
        a = await env.connect("main")
        b = await env.connect("main")
        await a.send(type="send", conversation="main", text="a-msg")
        await wait_until(lambda: gate.count == 1)
        await b.send(type="send", conversation="main", text="b-msg")
        await wait_until(lambda: len(env.agent.calls) == 1)
        id_a = (await a.wait_type("user", text="a-msg"))["id"]
        id_b = (await a.wait_type("user", text="b-msg"))["id"]
        assert id_a != id_b
        ev_a.set()
        rej = await b.wait_type("rejected")
        assert rej["id"] == id_a
        # Removing A's entry must leave B's (admitted, unpersisted) in place.
        await a.sync()
        remaining = env.plugin._unsent["web:main"]
        assert [e.text for e in remaining] == ["b-msg"]
        c = await env.connect()
        await c.send(type="subscribe", conversation="main")
        assert [u["text"] for u in _history_users(await c.wait_type("history"))] == ["b-msg"]


# ===========================================================================
# 8. Ordering
# ===========================================================================

class TestOrdering:

    async def test_live_frames_after_history(self, make_env):
        env = await make_env()
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked_fetch(channel_id):
            entered.set()
            await release.wait()
            return []

        env.plugin._fetch_history_rows = blocked_fetch
        conn = await env.connect()
        await conn.send(type="subscribe", conversation="main")
        await asyncio.wait_for(entered.wait(), 3)
        await env.pm.ahook.send(channel=env.channel("main"), event=OutboundMessage(text="live"))
        await asyncio.sleep(0.05)
        assert "message" not in conn.types()
        release.set()
        await conn.wait_type("history")
        await conn.wait_type("message", text="live")
        types = conn.types()
        assert types.index("history") < types.index("message")

    async def test_subscribe_survives_bad_rows(self, make_env):
        env = await make_env()

        def broken(rows, channel_id):
            raise ValueError("odd row")

        env.plugin._rows_to_frames = broken
        conn = await env.connect()
        await conn.send(type="subscribe", conversation="main")
        await conn.wait_type("error", request="subscribe")
        client = env.client_for("web:main")
        assert client.buffer is None
        await env.pm.ahook.send(channel=env.channel("main"), event=OutboundMessage(text="still here"))
        await conn.wait_type("message", text="still here")


# ===========================================================================
# 9. Backpressure and bounded shutdown
# ===========================================================================

class SpyTransport:
    """Wraps a real transport, recording abort() calls."""

    def __init__(self, real) -> None:
        self.real = real
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True
        if self.real is not None:
            self.real.abort()

    def is_closing(self) -> bool:
        return self.real.is_closing() if self.real is not None else True

    def __getattr__(self, name):
        return getattr(self.real, name)


def _fill(client) -> None:
    """Fill a client's queue to capacity without yielding."""
    while not client.queue.full():
        client.queue.put_nowait({"type": "message", "conversation": "main", "text": "x"})


BIG = "x" * 16384


async def _flood(env, channel_id: str = "web:main", n: int = 1500) -> None:
    """Broadcast large frames, yielding so writers can run until they block.

    Stops early if the client is dropped (its queue filled). Loopback
    kernel buffers hold several MB, so the writer blocks only once those
    and the transport buffer are full.
    """
    for i in range(n):
        if not any(c.conversation == channel_id for c in env.plugin._clients):
            return
        env.plugin._broadcast(channel_id, {"type": "message", "conversation": "main",
                                           "text": BIG})
        if i % 50 == 0:
            await asyncio.sleep(0.01)


def _writer_blocked(client) -> bool:
    """True once the transport holds unsent bytes (the peer isn't reading)."""
    return client.transport.get_write_buffer_size() > 0


async def _timed_stop(env) -> float:
    t0 = time.monotonic()
    await env.plugin.on_stop()
    return time.monotonic() - t0


class TestBackpressure:

    async def test_stop_with_peer_that_never_reads(self, make_env):
        env = await make_env()
        raw = await env.raw_subscribed()
        await _flood(env)
        assert await _timed_stop(env) < 2.5
        assert not env.plugin._handlers
        assert not env.plugin._clients

    async def test_stop_with_send_stuck_in_gate(self, make_env):
        env = await make_env()
        gate = HoldGate()
        env.pm.register(gate, name="gate")
        conn = await env.connect()
        await conn.send(type="send", conversation="main", text="hi")
        await asyncio.wait_for(gate.entered.wait(), 3)
        handlers = set(env.plugin._handlers)
        assert len(handlers) == 1
        assert await _timed_stop(env) < 2.5
        assert all(t.done() for t in handlers)
        assert not env.plugin._handlers and not env.plugin._clients
        assert "web:main" not in env.plugin._unsent

    async def test_dropped_client_stuck_in_gate(self, make_env):
        env = await make_env()
        gate = HoldGate()
        env.pm.register(gate, name="gate")
        raw = await env.raw_subscribed()
        await raw.send(type="send", conversation="main", text="hi")
        await asyncio.wait_for(gate.entered.wait(), 3)
        client = env.client_for("web:main")
        _fill(client)
        env.plugin._broadcast("web:main", {"type": "message", "conversation": "main",
                                           "text": "overflow"})
        assert client not in env.plugin._clients
        handlers = set(env.plugin._handlers)
        assert len(handlers) == 1
        assert await _timed_stop(env) < 2.5
        assert all(t.done() for t in handlers)
        assert not env.plugin._handlers
        assert "web:main" not in env.plugin._unsent

    async def test_pending_drop_close_completes_on_stop(self, make_env):
        env = await make_env()
        raw = await env.raw_subscribed()
        client = env.client_for("web:main")
        spy = SpyTransport(client.transport)
        client.transport = spy
        await _flood(env, n=400)  # enough to fill kernel buffers and block the writer
        assert _writer_blocked(client)
        _fill(client)
        env.plugin._broadcast("web:main", {"type": "message", "conversation": "main",
                                           "text": "overflow"})
        assert client not in env.plugin._clients
        assert await _timed_stop(env) < 2.5
        assert spy.aborted
        assert not env.plugin._handlers

    async def test_close_client_aborts_on_timeout(self, make_env):
        env = await make_env()

        class StuckWS:
            async def close(self, **kw):
                await asyncio.sleep(3600)

        transport = MagicMock()
        transport.is_closing.return_value = False
        client = SimpleNamespace(ws=StuckWS(), transport=transport)
        t0 = time.monotonic()
        await env.plugin._close_client(client, 1001, b"bye")
        assert time.monotonic() - t0 < 2.0
        transport.abort.assert_called_once()

    async def test_heartbeat_timeout_aborts_stuck_socket(self, make_env, monkeypatch):
        # A peer that stops reading and answering pings (a phone asleep):
        # aiohttp's heartbeat only calls transport.close(), which would
        # wait forever on the stuck buffer; the handler's release aborts.
        monkeypatch.setattr(web_mod, "_HEARTBEAT", 0.4)
        env = await make_env()
        raw = await env.raw_subscribed()
        client = env.client_for("web:main")
        transport = client.transport
        await _flood(env, n=400)
        assert _writer_blocked(client)
        await wait_until(lambda: not env.plugin._handlers, timeout=5)
        assert transport.is_closing()
        assert transport.get_write_buffer_size() == 0

    async def test_stop_with_nobody_connected_closes_listener(self, make_env):
        env = await make_env()
        port = env.port
        assert await _timed_stop(env) < 2.5
        assert env.plugin._runner is None
        await env.plugin.on_stop()  # idempotent
        with pytest.raises(OSError):
            _, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()

    async def test_disconnect_releases_everything(self, make_env):
        env = await make_env()
        conn = await env.connect("main")
        client = env.client_for("web:main")
        transport = client.transport
        await conn.ws.close()
        # The writer is cancelled synchronously but finishes on a later tick.
        await wait_until(lambda: not env.plugin._handlers and client.writer.done())
        assert not env.plugin._clients
        assert transport.is_closing()


# ===========================================================================
# 10. Replay and listing
# ===========================================================================

async def _seed(db, channel_id: str, rows: list[tuple]) -> None:
    """Insert (message_dict, message_type, timestamp) rows."""
    for msg, mtype, ts in rows:
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            (channel_id, json.dumps(msg), ts, mtype),
        )
    await db.commit()


async def _history(env, name: str = "main") -> list[dict]:
    conn = await env.connect()
    await conn.send(type="subscribe", conversation=name)
    return (await conn.wait_type("history"))["items"]


class TestReplay:

    async def test_row_mapping(self, make_env, db):
        env = await make_env()
        await _seed(db, "web:main", [
            ({"role": "user", "content": "hello"}, "message", 1.0),
            ({"role": "system", "content": "notification"}, "message", 2.0),
            ({"role": "user", "content": "memory blob"}, "context", 3.0),
            ({"role": "assistant", "reasoning_content": "hmm", "content": None,
              "tool_calls": [{"id": "c1", "type": "function", "function": {
                  "name": "read_file", "arguments": '{"path": "x"}'}}]}, "message", 4.0),
            ({"role": "tool", "tool_call_id": "c1",
              "content": "[Task abc123] file text"}, "message", 5.0),
            ({"role": "assistant", "content": "Let me check",
              "tool_calls": [{"id": "c2", "type": "function", "function": {
                  "name": "shell", "arguments": "ls"}}]}, "message", 6.0),
            ({"role": "tool", "tool_call_id": "zz", "content": "orphan"}, "message", 7.0),
            ({"role": "assistant", "content": "<think>private</think>Final answer"},
             "message", 8.0),
        ])
        items = await _history(env)
        simplified = [{k: v for k, v in i.items() if k not in ("conversation",)}
                      for i in items]
        assert simplified == [
            {"type": "user", "text": "hello", "id": None},
            {"type": "thinking", "text": "hmm"},
            {"type": "tool_status", "tool": "read_file", "status": "dispatched",
             "summary": '{"path": "x"}'},
            {"type": "tool_status", "tool": "read_file", "status": "completed",
             "summary": "file text"},
            {"type": "progress", "text": "Let me check"},
            {"type": "tool_status", "tool": "shell", "status": "dispatched",
             "summary": "ls"},
            {"type": "tool_status", "tool": "unknown", "status": "completed",
             "summary": "orphan"},
            {"type": "message", "text": "Final answer"},
        ]

    @pytest.mark.parametrize(("value_sql", "params"), CORRUPT_MESSAGES)
    async def test_corrupt_row_is_skipped_with_warning(
        self, make_env, db, caplog, value_sql, params
    ):
        """A corrupt row among the newest rows is skipped; the rest replay (#232)."""
        env = await make_env()
        await _seed(db, "web:main", [
            ({"role": "user", "content": "before"}, "message", 1.0),
        ])
        corrupt_id = await insert_raw_message(
            db, "web:main", value_sql, params, ts=2.0
        )
        await _seed(db, "web:main", [
            ({"role": "assistant", "content": "after"}, "message", 3.0),
        ])
        with caplog.at_level(logging.WARNING, logger="corvidae.persistence"):
            items = await _history(env)
        assert [i["text"] for i in items] == ["before", "after"]
        skips = skip_warnings(caplog, corrupt_id, "web:main")
        assert skips, [r.getMessage() for r in caplog.records]
        assert any("inspect or redact the row" in w for w in skips), skips

    async def test_unknown_message_type_is_skipped_with_warning(
        self, make_env, db, caplog
    ):
        """A row with a message_type outside MessageType is skipped, not shown."""
        env = await make_env()
        await _seed(db, "web:main", [
            ({"role": "user", "content": "before"}, "message", 1.0),
        ])
        bad_id = await insert_raw_message(
            db, "web:main", "?", (json.dumps({"role": "user", "content": "x"}),),
            message_type="bogus", ts=2.0,
        )
        with caplog.at_level(logging.WARNING, logger="corvidae.persistence"):
            items = await _history(env)
        assert [i["text"] for i in items] == ["before"]
        assert any("'bogus'" in w for w in skip_warnings(caplog, bad_id, "web:main"))

    async def test_summary_divider_placed_by_timestamp(self, make_env, db):
        env = await make_env()
        await _seed(db, "web:main", [
            ({"role": "user", "content": "old"}, "message", 1.0),
            ({"role": "user", "content": "new"}, "message", 3.0),
            ({"role": "system", "content": "the summary"}, "summary", 2.0),
        ])
        items = await _history(env)
        assert [(i["type"], i["text"]) for i in items] == [
            ("user", "old"), ("divider", "the summary"), ("user", "new")]

    async def test_row_cap(self, make_env, db):
        env = await make_env()
        await _seed(db, "web:main", [
            ({"role": "user", "content": f"m{i}"}, "message", float(i)) for i in range(250)
        ])
        items = await _history(env)
        assert [i["text"] for i in items] == [f"m{i}" for i in range(50, 250)]

    async def test_db_error_gives_empty_history_and_warning(self, make_env, caplog):
        caplog.set_level(logging.DEBUG)

        class BrokenDB:
            def execute(self, *a, **k):
                raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

        env = await make_env()
        env.persistence.db = BrokenDB()
        assert await _history(env) == []
        assert any(r.levelno == logging.WARNING and r.exc_info for r in caplog.records)

    async def test_listing(self, make_env, db):
        env = await make_env()
        await _seed(db, "web:a", [({"role": "user", "content": "x"}, "message", 100.0)])
        await _seed(db, "web:b", [({"role": "user", "content": "x"}, "message", 300.0)])
        await _seed(db, "web:Bad", [({"role": "user", "content": "x"}, "message", 400.0)])
        await _seed(db, "irc:#x", [({"role": "user", "content": "x"}, "message", 500.0)])
        env.channel("c").last_active = 200.0
        env.channel("d").last_active = 50.0
        env.registry.get_or_create("irc", "y").last_active = 600.0
        conn = await env.connect()
        await conn.send(type="list")
        items = (await conn.wait_type("conversations"))["items"]
        assert [i["name"] for i in items] == ["b", "c", "a", "d"]
        assert items[0]["last_active"] == 300.0


# ===========================================================================
# 11. Config reload
# ===========================================================================

class TestReload:

    async def test_token_change_closes_sockets_and_invalidates(self, make_env):
        env = await make_env()
        conn = await env.connect("main")
        await env.plugin.on_config_reload(config={"web": {
            "host": "127.0.0.1", "port": 0, "token": OTHER_TOKEN}})
        assert await conn.wait_closed() == 4001
        async with env.http.get(f"{env.base}/", headers=env.cookie(TOKEN)) as r:
            assert r.status == 401
        async with env.http.get(f"{env.base}/", headers=env.cookie(OTHER_TOKEN)) as r:
            assert r.status == 200

    async def test_generated_token_kept(self, make_env):
        env = await make_env(token=None)
        token = env.plugin._token
        await env.plugin.on_config_reload(config={"web": {"host": "127.0.0.1", "port": 0}})
        assert env.plugin._token == token

    async def test_malformed_reload_raises_and_changes_nothing(self, make_env):
        env = await make_env()
        with pytest.raises(ValueError, match="port"):
            await env.plugin.on_config_reload(config={"web": {"port": "x", "token": OTHER_TOKEN}})
        with pytest.raises(ValueError, match="token"):
            await env.plugin.on_config_reload(config={"web": {
                "host": "127.0.0.1", "port": 0, "token": "short"}})
        assert env.plugin._token == TOKEN


# ===========================================================================
# 12. Package data and registration
# ===========================================================================

class TestPackaging:

    def test_package_data_covers_every_static_file(self):
        base = Path(str(resources.files("corvidae.channels")))
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        patterns = pyproject["tool"]["setuptools"]["package-data"]["corvidae.channels"]
        covered = {
            p for p in set().union(*(glob.glob(pat, root_dir=base) for pat in patterns))
            if (base / p).is_file()
        }
        static = base / "web_static"
        actual = {str(p.relative_to(base)) for p in static.rglob("*") if p.is_file()}
        assert covered == actual
        for name in ["index.html", "app.js", "app.css", "vendor/marked.min.js",
                     "vendor/purify.min.js", "vendor/README.md"]:
            assert (static / name).is_file(), name

    def test_entry_point_registered(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        eps = pyproject["project"]["entry-points"]["corvidae"]
        assert eps["web"] == "corvidae.channels.web:WebPlugin"

    def test_hook_arg_binding(self):
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(), name="registry")
        pm.register(WebPlugin(pm), name="web")
        _check_hook_arg_binding(pm)


# ===========================================================================
# Issue #95: the transport mints the trace id
# ===========================================================================

class TestTransportMintedIds:

    def test_no_context_inheritance_machinery(self):
        """O3: the web transport no longer uses a ContextVar to link an
        entry to its admission/rejection."""
        import inspect
        import re

        assert not hasattr(web_mod, "_CURRENT_SEND")
        source = inspect.getsource(web_mod)
        assert not re.search(r"^\s*(import contextvars|from contextvars)", source, re.M)
        assert not hasattr(web_mod, "contextvars")

    async def test_unknown_id_leaves_entries_untouched(self, make_env):
        """O1/O3: an admission or rejection carrying an id no entry owns is
        a no-op: held entries are neither removed nor marked admitted, and
        the rejected frame carries id None."""
        env = await make_env()
        conn = await env.connect("main")
        # Hold one send inside the gate so its entry is unsent and unadmitted.
        gate = HoldGate()
        env.pm.register(gate, name="gate")
        await conn.send(type="send", conversation="main", text="t")
        await asyncio.wait_for(gate.entered.wait(), 3)
        before = list(env.plugin._unsent["web:main"])
        held = before[0]
        assert held.admitted is False
        try:
            ch = env.channel("main")
            # A rejection for an unknown id: broadcast with no entry id.
            await env.pm.ahook.on_message_rejected(
                channel=ch, trace_id="unknown-id", sender="web", text="t")
            rej = await conn.wait_type("rejected")
            assert rej["id"] is None
            assert env.plugin._unsent["web:main"] == before

            # An admission for an unknown id: no new entry, nothing marked.
            await env.pm.ahook.on_message_admitted(
                channel=ch, trace_id="unknown-id", sender="web", text="t2")
            assert env.plugin._unsent["web:main"] == before
            assert held.admitted is False
        finally:
            gate.release.set()

    async def test_later_send_vetoed_marks_only_itself(self, make_env):
        """O1: the LATER send is vetoed
        while the earlier is admitted and not yet persisted -- only the later
        is marked rejected and the earlier entry survives."""
        env = await make_env()
        # First call admitted; second call vetoed.
        gate = OrderedGate([None, None], [None, False])
        env.pm.register(gate, name="gate")
        a = await env.connect("main")
        await a.send(type="send", conversation="main", text="first")
        await wait_until(lambda: len(env.agent.calls) == 1)
        await a.send(type="send", conversation="main", text="second")
        id_first = (await a.wait_type("user", text="first"))["id"]
        id_second = (await a.wait_type("user", text="second"))["id"]
        assert id_first != id_second

        rej = await a.wait_type("rejected")
        assert rej["id"] == id_second
        # The earlier, admitted-and-unpersisted entry is still there.
        assert [e.text for e in env.plugin._unsent["web:main"]] == ["first"]
