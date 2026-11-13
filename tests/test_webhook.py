"""Tests for corvidae.channels.webhook (issue #488).

A real WebhookPlugin on a real plugin manager (install_plugin_hookspecs,
the real on_webhook_subscribe hook), listening on 127.0.0.1 and driven
through aiohttp's client. Subscribers are test-only plugins.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import re
import socket
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from corvidae.agent import Agent, QueueItemRole
from corvidae.channel import Channel
from corvidae.hooks import create_plugin_manager, hookimpl, install_plugin_hookspecs

# Absent until the green phase; this import failing is the red failure.
from corvidae.channels import webhook as webhook_mod
from corvidae.channels.webhook import WebhookPlugin

SECRET = "gh-secret"
TOKEN = "siri-token"
ENDPOINTS = {
    "github": {"auth": "hmac-sha256", "token": SECRET},
    "siri": {"token": TOKEN},
    "open": None,
    "lonely": {"token": TOKEN},
}
TOK = {"gh": {"token": "t"}}


def free_port() -> int:
    """A currently free TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sign(body: bytes, secret: str = SECRET) -> str:
    """The X-Hub-Signature-256 value GitHub sends for body."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def wait_until(cond, timeout: float = 3.0) -> None:
    """Poll cond() until true or fail."""
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.01)


class Subscriber:
    """Test-only plugin subscribing handlers to endpoints."""

    optional_depends_on = frozenset({"webhook"})

    def __init__(self, subs: dict[str, list]) -> None:
        self.subs = subs

    @hookimpl
    async def on_webhook_subscribe(self, subscriptions: dict) -> None:
        # Add every configured handler under its endpoint.
        for endpoint, handlers in self.subs.items():
            subscriptions.setdefault(endpoint, []).extend(handlers)


def recorder() -> tuple[list, Any]:
    """A handler that appends each delivery to a list."""
    got: list = []

    async def record(delivery) -> None:
        got.append(delivery)
    return got, record


async def exploding(delivery) -> None:
    raise RuntimeError("subscriber bug")


@pytest.fixture
async def start():
    """Factory: start a webhook plugin with subscribers; yields its base URL."""
    plugins, sessions = [], []

    async def make(subs=None, block=None, extra=(), port=None):
        pm = create_plugin_manager()
        plugin = WebhookPlugin()
        pm.register(plugin, name="webhook")
        pm.register(Subscriber(subs or {}), name="subscriber")
        for name, p in extra:
            # Stands in for the on_init these plugins would get at startup.
            p.pm = pm
            pm.register(p, name=name)
        install_plugin_hookspecs(pm)
        port = port or free_port()
        cfg = {"webhook": {"host": "127.0.0.1", "port": port, "endpoints": ENDPOINTS, **(block or {})}}
        await plugin.on_init(pm=pm, config=cfg)
        await plugin.on_start(config=cfg)
        plugins.append(plugin)
        sessions.append(http := aiohttp.ClientSession())
        return http, f"http://127.0.0.1:{port}/hook"

    yield make
    for plugin in plugins:
        await plugin.on_stop()
    for http in sessions:
        await http.close()


# AC1: inert configs bind nothing and log nothing above DEBUG.
@pytest.mark.parametrize("cfg", [{}, {"webhook": None}, {"webhook": {}}, {"webhook": {"host": "127.0.0.1"}}])
async def test_inert_config(cfg, caplog, monkeypatch):
    async def no_bind(self):
        raise AssertionError("bound a listener")
    monkeypatch.setattr(web.TCPSite, "start", no_bind)
    pm = create_plugin_manager()
    plugin = WebhookPlugin()
    pm.register(plugin, name="webhook")
    install_plugin_hookspecs(pm)
    caplog.set_level(logging.DEBUG)
    await plugin.on_init(pm=pm, config=cfg)
    await plugin.on_start(config=cfg)
    await plugin.on_stop()
    assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []


# AC3 (and the wildcard disposition: hostnames and "0" need tokens too).
@pytest.mark.parametrize(
    ("block", "fragments"),
    [
        ([1], ["webhook", "must be a mapping"]),
        ({"host": "", "endpoints": TOK}, ["webhook.host"]),
        ({"port": 70000, "endpoints": TOK}, ["webhook.port", "70000", "65535"]),
        ({"port": True, "endpoints": TOK}, ["webhook.port", "True"]),
        ({"host": "example.org", "port": 0, "endpoints": TOK}, ["webhook.port", "requires host to be an IP literal"]),
        ({"max_body_bytes": 0, "endpoints": TOK}, ["webhook.max_body_bytes", "0"]),
        ({"endpoints": [1]}, ["webhook.endpoints"]),
        ({"endpoints": {"Bad!": None}}, ["Bad!", "[a-z0-9][a-z0-9_-]{0,63}"]),
        ({"endpoints": {"gh": 5}}, ["webhook.endpoints.gh"]),
        ({"endpoints": {"gh": {"auth": "basic"}}}, ["webhook.endpoints.gh.auth", "basic", "bearer, hmac-sha256"]),
        ({"endpoints": {"gh": {"token": "tök"}}}, ["webhook.endpoints.gh.token"]),
        ({"endpoints": {"gh": {"token": "${WEBHOOK_TEST_UNSET}"}}}, ["webhook.endpoints.gh.token", "WEBHOOK_TEST_UNSET"]),
        ({"endpoints": {"gh": {"auth": "hmac-sha256"}}}, ["webhook.endpoints.gh.token", "required for hmac-sha256"]),
        ({"host": "0.0.0.0", "endpoints": {"gh": None}}, ["webhook.endpoints.gh.token", "0.0.0.0", "bind a specific IP address"]),
        ({"host": "::", "endpoints": {"gh": None}}, ["webhook.endpoints.gh.token", "::", "bind a specific IP address"]),
        ({"host": "localhost", "endpoints": {"gh": None}}, ["webhook.endpoints.gh.token", "localhost", "bind a specific IP address"]),
        ({"endpoints": {"siri": "TOPSECRET"}}, ["webhook.endpoints.siri", "got str", "mapping"]),
        ({"endpoints": ["TOPSECRET"]}, ["webhook.endpoints", "got list", "mapping"]),
        ({"endpoints": {"gh": ["TOPSECRET"]}}, ["webhook.endpoints.gh", "got list", "mapping"]),
        ({"host": "0", "endpoints": {"gh": None}}, ["webhook.endpoints.gh.token", "bind a specific IP address"]),
    ],
)
async def test_config_rejected(block, fragments, monkeypatch):
    monkeypatch.delenv("WEBHOOK_TEST_UNSET", raising=False)
    plugin = WebhookPlugin()
    with pytest.raises(ValueError) as exc:
        await plugin.on_init(pm=create_plugin_manager(), config={"webhook": block})
    msg = str(exc.value)
    assert msg.startswith("webhook:") and "check agent.yaml" in msg
    for fragment in fragments:
        assert fragment in msg
    assert "tök" not in msg and "TOPSECRET" not in msg


BODY = b'{"action": "opened"}'


# AC2 (and the ASCII signature-compare disposition: non-ASCII gives 401).
@pytest.mark.parametrize(
    ("endpoint", "headers", "status"),
    [
        ("siri", {"Authorization": f"Bearer {TOKEN}"}, 202),
        ("siri", {}, 401),
        ("siri", {"Authorization": "Bearer wrong"}, 401),
        ("github", {"X-Hub-Signature-256": sign(BODY)}, 202),
        ("github", {}, 401),
        ("github", {"X-Hub-Signature-256": "sha256=not-hex"}, 401),
        ("github", {"X-Hub-Signature-256": sign(BODY, "wrong")}, 401),
        ("github", {"X-Hub-Signature-256": "sha256=" + "é" * 64}, 401),
        ("open", {}, 202),
    ],
)
async def test_auth(start, endpoint, headers, status):
    http, base = await start({endpoint: [recorder()[1]]})
    async with http.post(f"{base}/{endpoint}", data=BODY, headers=headers) as resp:
        assert resp.status == status
        if status == 401:
            error = (await resp.json())["error"]
            remedy = "Authorization: Bearer" if endpoint == "siri" else "webhook.endpoints.github.token"
            assert f'"{endpoint}"' in error and remedy in error


# AC4 + AC12: every subscriber gets one delivery whose origin is minted by the ingress.
async def test_two_subscribers_both_receive(start):
    (got_a, a), (got_b, b) = recorder(), recorder()
    http, base = await start({"open": [a, b]})
    async with http.post(f"{base}/open", json={"origin": "evil"}) as resp:
        assert resp.status == 202
        assert await resp.json() == {"status": "accepted", "deliveries": 2}
    await wait_until(lambda: got_a and got_b)
    for got in (got_a, got_b):
        assert got[0].endpoint == "open"
        assert got[0].payload == {"origin": "evil"}
        assert got[0].origin == "webhook:open"


# AC5: 404, 400 name the endpoint and the remedy; 405 by status only.
@pytest.mark.parametrize(
    ("method", "endpoint", "data", "status", "remedy"),
    [
        ("POST", "nosuch", b"{}", 404, "check the URL"),
        ("POST", "lonely", b"{}", 404, "check the URL"),
        ("POST", "open", b"not json", 400, "application/json"),
        ("POST", "open", b"[1, 2]", 400, "application/json"),
        ("GET", "open", None, 405, None),
    ],
)
async def test_error_codes(start, method, endpoint, data, status, remedy):
    http, base = await start({"open": [recorder()[1]]})
    async with http.request(method, f"{base}/{endpoint}", data=data) as resp:
        assert resp.status == status
        if remedy:
            body = await resp.json()
            assert body["status"] == "error"
            assert f'"{endpoint}"' in body["error"] and remedy in body["error"]


# AC6: a streamed body over the cap gets 413 and reaches no subscriber.
async def test_oversized_body_413(start):
    got, rec = recorder()
    http, base = await start({"open": [rec]}, block={"max_body_bytes": 1024})

    async def stream():
        yield b'{"x": "'
        for _ in range(64):
            yield b"x" * 16384
        yield b'"}'
    async with http.post(f"{base}/open", data=stream()) as resp:
        assert resp.status == 413
    async with http.post(f"{base}/open", json={}) as resp:
        assert resp.status == 202
    await wait_until(lambda: got)
    assert len(got) == 1


# AC7 (and the handler-name disposition: a partial has no __qualname__).
@pytest.mark.parametrize(
    ("bad", "name"), [(exploding, "exploding"), (functools.partial(exploding), "functools.partial")]
)
async def test_raising_subscriber_isolated(start, caplog, bad, name):
    got, rec = recorder()
    http, base = await start({"open": [bad, rec]})
    async with http.post(f"{base}/open", json={}) as resp:
        assert resp.status == 202
    await wait_until(lambda: got and any(r.levelno == logging.ERROR for r in caplog.records))
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "open" in errors[0].getMessage() and name in errors[0].getMessage()
    assert errors[0].exc_info is not None


# AC8 + AC9: a blocked subscriber delays nobody, keeps the newest QUEUE_MAX in order.
async def test_slow_subscriber_drops_oldest(start, caplog, monkeypatch):
    monkeypatch.setattr(webhook_mod, "QUEUE_MAX", 3)
    release, slow_got = asyncio.Event(), []

    async def slow(delivery) -> None:
        slow_got.append(delivery.payload["n"])
        await release.wait()
    fast_got, fast = recorder()
    http, base = await start({"open": [slow, fast]})
    for n in range(11):
        async with asyncio.timeout(2):
            async with http.post(f"{base}/open", json={"n": n}) as resp:
                assert resp.status == 202
        if n == 0:
            await wait_until(lambda: slow_got == [0])
    await wait_until(lambda: len(fast_got) == 11)
    assert [d.payload["n"] for d in fast_got] == list(range(11))
    release.set()
    await wait_until(lambda: len(slow_got) == 4)
    assert slow_got == [0, 8, 9, 10]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("open" in m and "slow" in m and re.search(r"\b7\b", m) for m in warnings)


# AC10: a repeated delivery id is a 200 duplicate and is not dispatched.
@pytest.mark.parametrize("header", ["X-GitHub-Delivery", "Idempotency-Key"])
async def test_duplicate_delivery(start, header):
    got, rec = recorder()
    http, base = await start({"open": [rec]})
    for key, status in [("a", 202), ("a", 200), ("b", 202)]:
        async with http.post(f"{base}/open", json={"k": key}, headers={header: key}) as resp:
            assert resp.status == status
            if status == 200:
                assert await resp.json() == {"status": "duplicate"}
    await wait_until(lambda: len(got) == 2)
    assert [(d.payload["k"], d.delivery_id) for d in got] == [("a", "a"), ("b", "b")]


async def test_dedupe_window_evicts_oldest(start):
    http, base = await start({"open": [recorder()[1]]})
    for i in [*range(33), 0]:
        async with http.post(f"{base}/open", json={}, headers={"X-GitHub-Delivery": str(i)}) as resp:
            assert resp.status == 202


# AC11: a subscriber can reach a conversation through on_notify.
async def test_subscriber_notifies_agent(start):
    channel = Channel(transport="irc", scope="#lex")
    agent = Agent()
    items: list = []

    async def capture(item) -> None:
        items.append(item)
    agent._process_queue_item = capture

    async def forward(delivery) -> None:
        await agent.pm.ahook.on_notify(
            channel=channel, source=delivery.origin, text=json.dumps(delivery.payload),
            tool_call_id=None, annotations=None,
        )

    http, base = await start({"open": [forward]}, extra=[("agent", agent)])
    async with http.post(f"{base}/open", json={"event": "push"}) as resp:
        assert resp.status == 202
    await wait_until(lambda: items)
    assert items[0].role is QueueItemRole.NOTIFICATION
    assert items[0].channel is channel and '"event": "push"' in items[0].content
    for q in agent.queues.values():
        await q.stop()


# AC13: a failed bind logs one ERROR naming host:port and on_start returns.
async def test_bind_failure_degrades(start, caplog):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        await start({"open": [recorder()[1]]}, port=port)
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert f"127.0.0.1:{port}" in errors[0] and "webhook.port" in errors[0]


# AC: event type from provider header is passed through to subscribers.
async def test_event_type_header_passed_to_subscriber(start):
    deliveries = []

    async def handler(d):
        deliveries.append(d)

    http, base = await start({"open": [handler]})
    async with http.post(
        f"{base}/open",
        json={"action": "opened"},
        headers={"X-GitHub-Event": "pull_request"},
    ) as resp:
        assert resp.status == 202
    await wait_until(lambda: deliveries)
    assert deliveries[0].event_type == "pull_request"


# AC: no event type header means event_type is None.
async def test_no_event_type_header_is_none(start):
    deliveries = []

    async def handler(d):
        deliveries.append(d)

    http, base = await start({"open": [handler]})
    async with http.post(f"{base}/open", json={"action": "opened"}) as resp:
        assert resp.status == 202
    await wait_until(lambda: deliveries)
    assert deliveries[0].event_type is None
