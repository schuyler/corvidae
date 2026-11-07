"""Webhook ingress: route external HTTP events to subscribing plugins (#488).

Listens on one port and accepts ``POST /hook/<endpoint>``. A request that
passes the endpoint's auth is parsed as a JSON object and queued for every
plugin that subscribed to that endpoint through ``on_webhook_subscribe``.
The ingress never writes to a conversation and never interprets a payload;
a subscriber that wants the model to see an event calls ``on_notify``.

Each (endpoint, handler) subscription has its own bounded queue and worker
task, so a slow or failing handler delays no one else.

Logging:
    - ERROR: bind failure; a subscriber raised (with traceback)
    - WARNING: endpoints configured without subscribers or the reverse; a
      subscriber's queue overflowed (oldest delivery dropped)
    - INFO: listening; each accepted delivery; each rejected request
    - DEBUG: inert (no endpoints configured)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import re
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from corvidae.channels.web import _ascii_equal, _strip_brackets
from corvidae.envref import EnvRefError, expand_env_refs
from corvidae.hooks import CorvidaePlugin, hookimpl, hookspec

logger = logging.getLogger(__name__)

# Per-subscriber queue length, per-endpoint dedupe window, and the headers
# that carry a provider's delivery id (first match wins).
QUEUE_MAX = 100
DEDUPE_IDS = 32
DELIVERY_ID_HEADERS = ("X-GitHub-Delivery", "Idempotency-Key")

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_AUTH_MODES = ("bearer", "hmac-sha256")


@dataclass(frozen=True)
class WebhookDelivery:
    """One accepted request, as handed to each subscriber."""

    endpoint: str
    payload: dict[str, Any]  # treat as read-only; shared by all subscribers
    delivery_id: str | None

    @property
    def origin(self) -> str:
        """Identity minted by the ingress, never taken from the payload."""
        return f"webhook:{self.endpoint}"


WebhookHandler = Callable[[WebhookDelivery], Awaitable[None]]


class WebhookSpec:
    """Hooks the webhook plugin owns."""

    @hookspec
    async def on_webhook_subscribe(
        self, subscriptions: dict[str, list[WebhookHandler]]
    ) -> None:
        """Append handlers: ``subscriptions.setdefault("<endpoint>", []).append(h)``."""


@dataclass(frozen=True)
class _Endpoint:
    auth: str
    token: str | None


@dataclass(frozen=True)
class _Settings:
    host: str
    port: int
    max_body_bytes: int
    endpoints: dict[str, _Endpoint]


@dataclass(eq=False)
class _Subscriber:
    """One (endpoint, handler) registration with its queue and worker."""

    endpoint: str
    name: str
    handler: WebhookHandler
    queue: asyncio.Queue[WebhookDelivery] = field(init=False)
    dropped: int = 0
    task: asyncio.Task[None] | None = None

    def __post_init__(self) -> None:
        self.queue = asyncio.Queue(QUEUE_MAX)


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The host as an IP address, or None for a hostname."""
    try:
        return ipaddress.ip_address(_strip_brackets(host))
    except ValueError:
        return None


def _fail(detail: str) -> ValueError:
    """A config error in the house format."""
    return ValueError(f"webhook: {detail} — check agent.yaml")


def _parse_token(name: str, raw: Any) -> str | None:
    """A validated token (``${VAR}`` expanded), or None."""
    key = f"webhook.endpoints.{name}.token"
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = expand_env_refs(raw)
        except EnvRefError as exc:
            raise _fail(f"{key} references ${{{exc.var}}}, which is {exc.reason}; set the variable") from None
    if not isinstance(raw, str) or not raw or not raw.isascii():
        raise _fail(f"{key} must be a non-empty ASCII string or null; set it to the shared secret")
    return raw


def _parse_block(block: dict[str, Any]) -> _Settings | None:
    """Validate the webhook block; None when it configures no endpoints."""
    host = block.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        raise _fail(f"webhook.host is {host!r}; use an IP address such as 127.0.0.1")
    port = block.get("port", 8400)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise _fail(f"webhook.port is {port!r}; use an integer from 0 to 65535")
    if port == 0 and _ip(host) is None:
        raise _fail("webhook.port 0 requires host to be an IP literal; set webhook.host to an IP address")
    cap = block.get("max_body_bytes", 1048576)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise _fail(f"webhook.max_body_bytes is {cap!r}; use a positive integer")

    raw_endpoints = block.get("endpoints")
    if not raw_endpoints:
        return None
    if not isinstance(raw_endpoints, dict):
        raise _fail(f"webhook.endpoints is {raw_endpoints!r}; use a mapping of endpoint name to settings")

    # A null token is safe only where the network is the control: a specific
    # IP address. Wildcards, hostnames and unparseable hosts need a token.
    addr = _ip(host)
    open_host = addr is not None and not addr.is_unspecified
    endpoints: dict[str, _Endpoint] = {}
    for name, raw in raw_endpoints.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise _fail(f"webhook.endpoints name {name!r} must match {_NAME.pattern}; rename it")
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise _fail(f"webhook.endpoints.{name} is {raw!r}; use a mapping with auth and token")
        auth = raw.get("auth", "bearer")
        if auth not in _AUTH_MODES:
            raise _fail(f"webhook.endpoints.{name}.auth is {auth!r}; use one of: {', '.join(_AUTH_MODES)}")
        token = _parse_token(name, raw.get("token"))
        if token is None and auth == "hmac-sha256":
            raise _fail(f"webhook.endpoints.{name}.token is required for hmac-sha256; set it to the webhook's secret")
        if token is None and not open_host:
            raise _fail(f"webhook.endpoints.{name}.token is null but host is {host!r}; set a token or bind a specific IP address")
        endpoints[name] = _Endpoint(auth, token)
    return _Settings(host, port, cap, endpoints)


def _reply(status: int, body: dict[str, Any]) -> web.Response:
    return web.json_response(body, status=status)


class WebhookPlugin(CorvidaePlugin):
    """The webhook ingress."""

    hookspecs = WebhookSpec

    def __init__(self) -> None:
        self._settings: _Settings | None = None
        self._runner: web.AppRunner | None = None
        self._subscribers: dict[str, list[_Subscriber]] = {}
        self._seen: dict[str, deque[str]] = {}

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        block = config.get("webhook")
        if not block:
            logger.debug("webhook: no webhook block; ingress disabled")
            return
        if not isinstance(block, dict):
            raise _fail(f"webhook is {block!r}; it must be a mapping")
        self._settings = _parse_block(block)
        if self._settings is None:
            logger.debug("webhook: no endpoints configured; ingress disabled")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        settings = self._settings
        if settings is None:
            return

        # Collect each plugin's handlers, then give every one a queue and worker.
        subscriptions: dict[str, list[WebhookHandler]] = {}
        await self.pm.ahook.on_webhook_subscribe(subscriptions=subscriptions)
        for name in settings.endpoints:
            self._seen[name] = deque(maxlen=DEDUPE_IDS)
            for handler in subscriptions.get(name, []):
                sub = _Subscriber(name, getattr(handler, "__qualname__", repr(handler)), handler)
                sub.task = asyncio.create_task(self._work(sub), name=f"webhook:{name}:{sub.name}")
                self._subscribers.setdefault(name, []).append(sub)

        silent = sorted(set(settings.endpoints) - set(self._subscribers))
        unreachable = sorted(set(subscriptions) - set(settings.endpoints))
        if silent or unreachable:
            logger.warning(
                "webhook: configured endpoints with no subscriber (they answer 404): %s; "
                "subscribed endpoints with no webhook.endpoints entry (unreachable): %s",
                silent or "none", unreachable or "none",
            )

        # access_log=None keeps nothing request-derived in the log; the short
        # shutdown timeout keeps cleanup inside Runtime's stop budget.
        app = web.Application(client_max_size=settings.max_body_bytes)
        app.router.add_post("/hook/{endpoint}", self._handle)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.5)
        await runner.setup()
        try:
            await web.TCPSite(runner, _strip_brackets(settings.host), settings.port).start()
        except OSError as exc:
            # Not a fatal startup path (PRD 3.1.17): stay inert and say how to fix it.
            logger.error(
                "webhook: cannot bind %s:%s (%s: %s); free the port or change webhook.port, then restart",
                settings.host, settings.port, type(exc).__name__, exc,
            )
            await runner.cleanup()
            await self._stop_workers()
            return
        self._runner = runner
        logger.info(
            "webhook: listening on %s:%s for endpoints %s",
            settings.host, runner.addresses[0][1], sorted(self._subscribers),
        )

    @hookimpl
    async def on_stop(self) -> None:
        # Close the listener first so no request is accepted after its worker is gone.
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        await self._stop_workers()

    async def _stop_workers(self) -> None:
        """Cancel every subscriber worker; queued deliveries are discarded."""
        tasks = [s.task for subs in self._subscribers.values() for s in subs if s.task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._subscribers = {}

    async def _work(self, sub: _Subscriber) -> None:
        """Run one subscriber's handler over its queue, in order."""
        while True:
            delivery = await sub.queue.get()
            try:
                await sub.handler(delivery)
            except Exception:
                logger.exception(
                    'webhook: subscriber %s failed on endpoint "%s" (delivery %s)',
                    sub.name, sub.endpoint, delivery.delivery_id,
                )

    def _enqueue(self, sub: _Subscriber, delivery: WebhookDelivery) -> None:
        """Queue a delivery, dropping the subscriber's oldest when it is full."""
        if sub.queue.full():
            sub.queue.get_nowait()
            sub.dropped += 1
            logger.warning(
                'webhook: subscriber %s on endpoint "%s" is not keeping up; dropped its oldest '
                "delivery (%d dropped so far)", sub.name, sub.endpoint, sub.dropped,
            )
        sub.queue.put_nowait(delivery)

    def _reject(self, request: web.Request, endpoint: str, status: int, text: str) -> web.Response:
        """Log and answer a rejected request; never logs the token or body."""
        logger.info('webhook: rejected request to "%s" from %s: %d', endpoint, request.remote, status)
        return _reply(status, {"status": "error", "error": text})

    async def _handle(self, request: web.Request) -> web.Response:
        endpoint = request.match_info["endpoint"]
        subs = self._subscribers.get(endpoint)
        config = self._settings.endpoints.get(endpoint) if self._settings else None
        if not subs or config is None:
            return self._reject(
                request, endpoint, 404,
                f'no webhook endpoint "{endpoint}" has a subscriber; check the URL',
            )

        # Bearer needs no body, so check it before reading any.
        if config.auth == "bearer" and config.token is not None:
            scheme, _, value = request.headers.get("Authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not _ascii_equal(value.strip(), config.token):
                return self._reject(
                    request, endpoint, 401,
                    f'webhook "{endpoint}": missing or wrong bearer token; send Authorization: Bearer <token>',
                )

        # aiohttp raises 413 once the stream passes client_max_size.
        body = await request.read()

        if config.auth == "hmac-sha256":
            assert config.token is not None
            expected = "sha256=" + hmac.new(config.token.encode(), body, hashlib.sha256).hexdigest()
            if not _ascii_equal(request.headers.get("X-Hub-Signature-256", ""), expected):
                return self._reject(
                    request, endpoint, 401,
                    f'webhook "{endpoint}": X-Hub-Signature-256 is missing or does not match the body; '
                    f"check the webhook secret matches webhook.endpoints.{endpoint}.token",
                )

        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            return self._reject(
                request, endpoint, 400,
                f'webhook "{endpoint}": body is not a JSON object; send Content-Type application/json '
                "(GitHub: set Content type to application/json)",
            )

        # A repeated delivery id is a provider retry; remember ids only once dispatched.
        delivery_id = next((v for h in DELIVERY_ID_HEADERS if (v := request.headers.get(h))), None)
        seen = self._seen[endpoint]
        if delivery_id is not None:
            if delivery_id in seen:
                return _reply(200, {"status": "duplicate"})
            seen.append(delivery_id)

        delivery = WebhookDelivery(endpoint, payload, delivery_id)
        for sub in subs:
            self._enqueue(sub, delivery)
        logger.info('webhook: accepted delivery %s on "%s" for %d subscribers', delivery_id, endpoint, len(subs))
        return _reply(202, {"status": "accepted", "deliveries": len(subs)})
