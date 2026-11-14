"""Corvidae plugin: deliver webhook events to channels as notifications.

Subscribes to endpoints of the webhook ingress (``corvidae.channels.webhook``)
and notifies each delivery's payload, as compact JSON, to the channels configured
for its endpoint through the kernel's spill service. The kernel fences the
text, renders the ``origin`` annotation in the header, and, for a payload cut
at ``max_payload_chars``, renders the cut marker and spill file pointer.

Volume control (#793), per endpoint, in two layers, both on by default:

- **Batching.** A delivery on an endpoint with no open window opens one for
  ``batch_seconds``. Everything that arrives in the window is buffered and
  flushed together: one delivery notifies as before, more than one as a
  compact JSON array in arrival order with a ``deliveries=<n>`` annotation. A
  burst of forty CI events costs one turn instead of forty.
- **A rate cap.** At most ``max_notifications`` flushes per endpoint per
  ``per_seconds``. A flush over the cap is held, and its deliveries stay
  buffered up to ``max_buffered``; beyond that the oldest are dropped and
  counted. The next flush that gets through carries ``dropped=<n>``, and a
  window that dropped deliveries and is followed by no flush still sends one
  notification carrying only ``dropped=<n>`` — the count reaches the model
  rather than only the log. Every drop is a WARNING naming the endpoint and
  the running count.

Both counts travel as annotations the plugin declares in
``model_visible_annotations``, so the kernel builds the header from them and
they never sit in the fenced payload text where they would read as
sender-controlled data (PRD §3.3.11).

Origin ``webhook:<endpoint>`` in the cognition plugins:

- appraisal: runs stage 2 as for any non-critique exchange.
- outcome_log: records the origin verbatim in ``exchange_log.origin``.
- gates: non-``user``, so the decide gate may withhold a low-salience final
  when ``gate.send.enforce`` is set, as for heartbeat and task.
- critique: not eligible, as for heartbeat.
- funnel: keys stubs on its own origins only; unaffected.

Channels resolve at flush, not in ``on_start``, deliberately: the ingress may
deliver before this plugin starts, and the registry is resolved in ``on_init``
for the same reason.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field

from corvidae.channel import (
    ChannelNotReady,
    ChannelRegistry,
    resolve_persisted_channel,
)
from corvidae.channels.webhook import WebhookDelivery, WebhookHandler
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.spill import NotificationSpill
from corvidae.tuning import config_value

logger = logging.getLogger(__name__)

DEFAULT_MAX_PAYLOAD_CHARS = 8000

# The throttle's built-in values (#793). Batching on by default is the point
# of the change; an operator restoring the old behaviour sets
# ``batch_seconds: 0`` and a high ``max_notifications``.
THROTTLE_DEFAULTS: dict[str, float] = {
    "batch_seconds": 60.0,
    "max_notifications": 10,
    "per_seconds": 3600.0,
    "max_buffered": 500,
}
_THROTTLE_KEYS = tuple(THROTTLE_DEFAULTS)
_CHANNELS_KEY = "channels"


def _bad(key: str, value: object, want: str) -> ValueError:
    """The config error naming the key, the rejected value and the fix."""
    return ValueError(f"Config error: {key} is {value!r}; {want} — check agent.yaml")


def _compact(payload: object) -> str:
    """One payload as compact JSON, the text a single delivery has always sent."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class _Limit:
    """One endpoint's channels and its resolved throttle."""

    channels: tuple[str, ...]
    batch_seconds: float
    max_notifications: int
    per_seconds: float
    max_buffered: int


@dataclass
class _State:
    """One endpoint's window: what is buffered, when it last flushed, drops."""

    buffer: list[WebhookDelivery] = field(default_factory=list)
    window: asyncio.Task | None = None
    flushes: deque[float] = field(default_factory=deque)
    # Running total for the WARNING, and the part not yet reported to the
    # model by a flush.
    dropped: int = 0
    pending_dropped: int = 0


class WebhookNotifyPlugin(CorvidaePlugin):
    """Notifies configured channels of webhook deliveries, throttled per endpoint."""

    depends_on = frozenset({"registry", "spill"})
    optional_depends_on = frozenset({"webhook"})
    model_visible_annotations = frozenset({"origin", "deliveries", "dropped"})

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self._registry = get_dependency(pm, "registry", ChannelRegistry)
        self._spill = get_dependency(pm, "spill", NotificationSpill)
        self._max_chars = config_value(
            config, "webhook_notify.max_payload_chars", None
        )
        if self._max_chars is None:
            self._max_chars = DEFAULT_MAX_PAYLOAD_CHARS
        self._targets: dict[str, _Limit] = {}
        self._state: dict[str, _State] = {}
        # Validate the block; an absent or null one leaves the plugin inert.
        block = config.get("webhook_notify")
        if block is not None and not isinstance(block, dict):
            raise _bad("webhook_notify", block, "it must be a mapping")
        if type(self._max_chars) is not int or self._max_chars < 1:
            raise _bad("webhook_notify.max_payload_chars", self._max_chars,
                       "use a positive integer")
        defaults = self._parse_defaults((block or {}).get("defaults"))
        endpoints = (block or {}).get("endpoints")
        if endpoints is not None and not isinstance(endpoints, dict):
            raise _bad("webhook_notify.endpoints", endpoints,
                       "use a mapping of endpoint name to a list of channel ids")
        for name, spec in (endpoints or {}).items():
            self._targets[name] = self._parse_endpoint(name, spec, defaults)

    # -- config ------------------------------------------------------------

    def _number(self, key: str, name: str, value: object) -> float | int:
        """One throttle value, validated: the ranges differ per key."""
        if isinstance(value, bool):
            raise _bad(key, value, "use a number, not true/false")
        if name == "batch_seconds":
            if not isinstance(value, (int, float)) or value < 0:
                raise _bad(
                    key, value, "use a number of seconds >= 0 (0 turns batching off)"
                )
            return float(value)
        if name in ("max_notifications", "max_buffered"):
            if not isinstance(value, int) or value < 1:
                raise _bad(key, value, "use a positive integer")
            return value
        # per_seconds
        if not isinstance(value, (int, float)) or value <= 0:
            raise _bad(key, value, "use a number of seconds > 0")
        return float(value)

    def _parse_defaults(self, block: object) -> dict[str, float]:
        """`webhook_notify.defaults`: the throttle every endpoint inherits."""
        defaults = dict(THROTTLE_DEFAULTS)
        if block is None:
            return defaults
        if not isinstance(block, dict):
            raise _bad("webhook_notify.defaults", block,
                       "use a mapping of throttle keys")
        for name, value in block.items():
            if name not in _THROTTLE_KEYS:
                raise _bad(
                    f"webhook_notify.defaults.{name}", value,
                    f"use one of {', '.join(_THROTTLE_KEYS)}",
                )
            defaults[name] = self._number(f"webhook_notify.defaults.{name}", name, value)
        return defaults

    def _parse_endpoint(
        self, name: str, spec: object, defaults: dict[str, float]
    ) -> _Limit:
        """One endpoint: a bare channel list, or a mapping with `channels`."""
        key = f"webhook_notify.endpoints.{name}"
        if isinstance(spec, list):
            channels, overrides = spec, {}
        elif isinstance(spec, dict):
            channels = spec.get(_CHANNELS_KEY)
            overrides = {k: v for k, v in spec.items() if k != _CHANNELS_KEY}
            if channels is None:
                raise _bad(f"{key}.{_CHANNELS_KEY}", None,
                           "it is required, as a non-empty list of channel ids")
        else:
            raise _bad(
                key, spec,
                "use a non-empty list of channel ids, or a mapping with "
                "channels: and throttle keys",
            )
        if not isinstance(channels, list) or not channels:
            raise _bad(key, channels, "use a non-empty list of channel ids such as irc:#general")
        for cid in channels:
            transport, _, scope = cid.partition(":") if isinstance(cid, str) else ("", "", "")
            if not transport or not scope:
                raise _bad(key, cid, "use transport:scope, e.g. irc:#general")
        limits = dict(defaults)
        for throttle, value in overrides.items():
            if throttle not in _THROTTLE_KEYS:
                raise _bad(
                    f"{key}.{throttle}", value,
                    f"use {_CHANNELS_KEY} or one of {', '.join(_THROTTLE_KEYS)}",
                )
            limits[throttle] = self._number(f"{key}.{throttle}", throttle, value)
        # dict.fromkeys drops duplicate ids and keeps config order.
        return _Limit(
            channels=tuple(dict.fromkeys(channels)),
            batch_seconds=limits["batch_seconds"],
            max_notifications=limits["max_notifications"],
            per_seconds=limits["per_seconds"],
            max_buffered=limits["max_buffered"],
        )

    # -- delivery ----------------------------------------------------------

    @hookimpl
    async def on_webhook_subscribe(self, subscriptions: dict[str, list[WebhookHandler]]) -> None:
        # One bound method serves every endpoint; it reads delivery.endpoint.
        for endpoint in self._targets:
            subscriptions.setdefault(endpoint, []).append(self._deliver)

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the flush timers, and report what was still buffered."""
        for endpoint, state in self._state.items():
            if state.window is not None:
                state.window.cancel()
                state.window = None
            if state.buffer or state.pending_dropped:
                logger.warning(
                    "webhook_notify: endpoint %s shutting down with %d buffered "
                    "deliveries and %d dropped and unreported; they never reach "
                    "a channel",
                    endpoint, len(state.buffer), state.pending_dropped,
                )

    async def _deliver(self, delivery: WebhookDelivery) -> None:
        """Buffer a delivery for its endpoint, flushing when the window closes."""
        limit = self._targets[delivery.endpoint]
        state = self._state.setdefault(delivery.endpoint, _State())
        state.buffer.append(delivery)
        if len(state.buffer) > limit.max_buffered:
            # Drop-oldest: the count is what keeps this audible.
            state.buffer.pop(0)
            state.dropped += 1
            state.pending_dropped += 1
            logger.warning(
                "webhook_notify: endpoint %s buffer is full at %d deliveries; "
                "dropped the oldest, %d dropped so far",
                delivery.endpoint, limit.max_buffered, state.dropped,
            )
        if limit.batch_seconds == 0:
            # Batching off: deliver on arrival, as before #793.
            await self._flush(delivery.endpoint)
        elif state.window is None:
            self._arm(delivery.endpoint, limit.batch_seconds)

    def _arm(self, endpoint: str, delay: float) -> None:
        """Start the endpoint's flush timer, replacing any timer already set."""
        state = self._state[endpoint]
        if state.window is not None:
            state.window.cancel()
        state.window = asyncio.get_running_loop().create_task(
            self._after(endpoint, delay), name=f"webhook-notify-flush-{endpoint}"
        )

    async def _after(self, endpoint: str, delay: float) -> None:
        """Wait out the window (or the cap), then flush."""
        await asyncio.sleep(delay)
        self._state[endpoint].window = None
        await self._flush(endpoint)

    async def _flush(self, endpoint: str) -> None:
        """Send what the endpoint has buffered, if the rate cap allows it."""
        limit = self._targets[endpoint]
        state = self._state[endpoint]
        state.window = None
        now = asyncio.get_running_loop().time()
        while state.flushes and now - state.flushes[0] >= limit.per_seconds:
            state.flushes.popleft()
        if len(state.flushes) >= limit.max_notifications:
            # Over the cap: hold everything and retry when a slot frees, rather
            # than lose the buffer to a burst that happens to exceed the rate.
            self._arm(endpoint, max(state.flushes[0] + limit.per_seconds - now, 0.05))
            return
        if state.buffer:
            deliveries, state.buffer = state.buffer, []
        elif state.pending_dropped:
            deliveries = []  # a drop report: nothing survived this window
        else:
            return
        dropped, state.pending_dropped = state.pending_dropped, 0
        state.flushes.append(now)
        await self._notify(endpoint, deliveries, dropped)

    async def _notify(
        self, endpoint: str, deliveries: list[WebhookDelivery], dropped: int
    ) -> None:
        """Send one notification for the endpoint's window to each target."""
        limit = self._targets[endpoint]
        origin = f"webhook:{endpoint}"
        if len(deliveries) == 1:
            text = _compact(deliveries[0].payload)
        elif deliveries:
            # Arrival order: the newest payload is last, so a cut inline head
            # starts at the oldest and the newest survives in the spill file.
            text = "[" + ",".join(_compact(d.payload) for d in deliveries) + "]"
        else:
            # A drop report carries no payload and no count in its text: the
            # count is an annotation, so nothing sender-shaped reaches the fence.
            text = "{}"
        annotations = {"origin": origin}
        if len(deliveries) > 1:
            annotations["deliveries"] = str(len(deliveries))
        if dropped:
            annotations["dropped"] = str(dropped)
        channels = []
        for cid in limit.channels:
            try:
                channel = await resolve_persisted_channel(self.pm, self._registry, cid)
            except ChannelNotReady:
                logger.warning(
                    "webhook_notify: endpoint %s skipped channel %s, whose "
                    "transport cannot build it yet; the digest of %d deliveries "
                    "does not reach it",
                    endpoint, cid, max(len(deliveries), 1),
                )
                continue
            channels.append(channel)
        if not channels:
            return
        # The kernel cuts an over-long payload, spills the rest and marks the header.
        await self._spill.notify(
            channels, source=origin, text=text,
            annotations=annotations, limit=self._max_chars,
        )
