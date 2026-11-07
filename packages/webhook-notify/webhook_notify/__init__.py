"""Corvidae plugin: deliver webhook events to channels as notifications.

Subscribes to endpoints of the webhook ingress (``corvidae.channels.webhook``)
and notifies each delivery's payload, as compact JSON, to the channels configured
for its endpoint through the kernel's spill service. The kernel fences the
text, renders the ``origin`` annotation in the header, and, for a payload cut
at ``max_payload_chars``, renders the cut marker and spill file pointer.

Origin ``webhook:<endpoint>`` in the cognition plugins:

- appraisal: runs stage 2 as for any non-critique exchange.
- outcome_log: records the origin verbatim in ``exchange_log.origin``.
- gates: non-``user``, so the decide gate may withhold a low-salience final
  when ``gate.send.enforce`` is set, as for heartbeat and task.
- critique: not eligible, as for heartbeat.
- funnel: keys stubs on its own origins only; unaffected.

Channels resolve at delivery, not in ``on_start``, deliberately: the ingress
may deliver before this plugin starts, and the registry is resolved in
``on_init`` for the same reason.
"""

from __future__ import annotations

import json
import logging

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


def _bad(key: str, value: object, want: str) -> ValueError:
    """The config error naming the key, the rejected value and the fix."""
    return ValueError(f"Config error: {key} is {value!r}; {want} — check agent.yaml")


class WebhookNotifyPlugin(CorvidaePlugin):
    """Notifies configured channels of webhook deliveries."""

    depends_on = frozenset({"registry", "spill"})
    optional_depends_on = frozenset({"webhook"})
    model_visible_annotations = frozenset({"origin"})

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
        self._targets: dict[str, tuple[str, ...]] = {}
        # Validate the block; an absent or null one leaves the plugin inert.
        block = config.get("webhook_notify")
        if block is not None and not isinstance(block, dict):
            raise _bad("webhook_notify", block, "it must be a mapping")
        if type(self._max_chars) is not int or self._max_chars < 1:
            raise _bad("webhook_notify.max_payload_chars", self._max_chars,
                       "use a positive integer")
        endpoints = (block or {}).get("endpoints")
        if endpoints is not None and not isinstance(endpoints, dict):
            raise _bad("webhook_notify.endpoints", endpoints,
                       "use a mapping of endpoint name to a list of channel ids")
        for name, ids in (endpoints or {}).items():
            key = f"webhook_notify.endpoints.{name}"
            if not isinstance(ids, list) or not ids:
                raise _bad(key, ids, "use a non-empty list of channel ids such as irc:#general")
            for cid in ids:
                transport, _, scope = cid.partition(":") if isinstance(cid, str) else ("", "", "")
                if not transport or not scope:
                    raise _bad(key, cid, "use transport:scope, e.g. irc:#general")
            # dict.fromkeys drops duplicate ids and keeps config order.
            self._targets[name] = tuple(dict.fromkeys(ids))

    @hookimpl
    async def on_webhook_subscribe(self, subscriptions: dict[str, list[WebhookHandler]]) -> None:
        # One bound method serves every endpoint; it reads delivery.endpoint.
        for endpoint in self._targets:
            subscriptions.setdefault(endpoint, []).append(self._deliver)

    async def _deliver(self, delivery: WebhookDelivery) -> None:
        """Notify each target channel of the delivery's endpoint."""
        text = json.dumps(delivery.payload, separators=(",", ":"), ensure_ascii=False)
        channels = []
        for cid in self._targets[delivery.endpoint]:
            try:
                channel = await resolve_persisted_channel(self.pm, self._registry, cid)
            except ChannelNotReady:
                logger.warning(
                    "webhook_notify: delivery %s on endpoint %s skipped channel %s, "
                    "whose transport cannot build it yet; it reaches that channel "
                    "once the transport has started",
                    delivery.delivery_id, delivery.endpoint, cid,
                )
                continue
            channels.append(channel)
        # The kernel cuts an over-long payload, spills the rest and marks the header.
        await self._spill.notify(
            channels, source=delivery.origin, text=text,
            annotations={"origin": delivery.origin}, limit=self._max_chars,
        )
