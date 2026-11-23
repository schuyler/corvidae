"""WhatsApp transport: corvidae's side of the Baileys sidecar (#1220).

The sidecar links to WhatsApp as a second client and speaks newline-delimited
JSON-RPC 2.0 over a unix socket; this plugin is the corvidae half. It connects
to an already-running sidecar, turns its ``message`` notifications into channel
messages, and sends replies with ``send_text``.

The seam was agreed on #1220: the plugin never touches WhatsApp itself, and the
sidecar never knows what a channel is.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.hooks import TransportPlugin, get_dependency, hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

from .sidecar import NOT_CONNECTED, NOT_LINKED, SidecarClient, SidecarError

logger = logging.getLogger("corvidae.whatsapp")

#: Seconds between reconnect attempts. The sidecar may be started after the
#: daemon, or restarted under it, so a refusal is never fatal.
RECONNECT_DELAY = 5.0

#: Distinct WhatsApp message ids remembered for de-duplication. A reconnect
#: replays unacked messages, so the same id can arrive twice; the window only
#: has to cover a replay, not the channel's history.
SEEN_LIMIT = 512

#: Seconds between WARNINGs while the sidecar stays unreachable. Logging every
#: attempt at WARNING trains the operator to skip WARNING lines; the first
#: failure warns at once and the rest drop to DEBUG until this elapses.
RECONNECT_WARN_INTERVAL = 300.0


class WhatsAppPlugin(TransportPlugin):
    """The WhatsApp transport.

    Inert unless ``agent.yaml`` has a ``whatsapp:`` block, the same rule the
    other transports follow: absence means no connection and nothing logged
    above DEBUG, while a present block with a bad field raises at startup
    naming the key.
    """

    transport = "whatsapp"
    conversational = True

    def __init__(self) -> None:
        self._socket_path: str | None = None
        self._client: SidecarClient | None = None
        self._connect_task: asyncio.Task | None = None
        self._session_state: str = "unknown"
        self._registry: ChannelRegistry | None = None
        #: Message ids already handed to the agent, oldest first.
        self._seen: collections.OrderedDict[str, None] = collections.OrderedDict()
        #: Consecutive failed connect attempts, and when the last one warned.
        self._connect_failures: int = 0
        self._last_connect_warn: float = 0.0

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read the whatsapp: block; raise on a present-but-malformed one."""
        await super().on_init(pm, config)
        block = config.get("whatsapp")
        if block is None:
            return
        if not isinstance(block, dict):
            raise ValueError("whatsapp: config block must be a mapping")
        socket_path = block.get("socket")
        if not socket_path or not isinstance(socket_path, str):
            raise ValueError(
                "whatsapp: config field 'socket' must be a non-empty path, "
                f"got {socket_path!r}"
            )
        self._socket_path = socket_path

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Resolve the registry always; connect only when configured."""
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        if self._socket_path is None:
            return
        self._connect_task = asyncio.create_task(self._connection_loop())

    @hookimpl
    async def on_stop(self) -> None:
        """Stop the connection loop and close the socket."""
        task, self._connect_task = self._connect_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    async def _connection_loop(self) -> None:
        """Connect, serve until the socket drops, then retry — never give up.

        The sidecar is a separate process the operator starts, so its absence
        is normal. The first failed attempt warns, naming the socket, the
        attempt number and the start command; later ones log at DEBUG with the
        count, and a fresh WARNING comes every RECONNECT_WARN_INTERVAL so a long
        outage stays visible without burying the log.
        """
        assert self._socket_path is not None
        while True:
            client = SidecarClient(self._socket_path, on_notification=self._on_notification)
            try:
                await client.connect()
            except (FileNotFoundError, ConnectionError, OSError) as exc:
                self._connect_failures += 1
                now = time.monotonic()
                # First failure, and one every RECONNECT_WARN_INTERVAL after it,
                # at WARNING; everything between at DEBUG with the count, so a
                # sidecar that stays down does not bury the log.
                if self._last_connect_warn == 0.0 or now - self._last_connect_warn >= RECONNECT_WARN_INTERVAL:
                    self._last_connect_warn = now
                    logger.warning(
                        "whatsapp: cannot reach the sidecar socket %s (%s); retrying every %.0fs"
                        " (attempt %d). Start it with: node services/whatsapp-sidecar/index.js",
                        self._socket_path, type(exc).__name__, RECONNECT_DELAY,
                        self._connect_failures,
                    )
                else:
                    logger.debug(
                        "whatsapp: sidecar still unreachable at %s (attempt %d, %s)",
                        self._socket_path, self._connect_failures, type(exc).__name__,
                    )
                await asyncio.sleep(RECONNECT_DELAY)
                continue
            self._connect_failures = 0
            self._last_connect_warn = 0.0
            self._client = client
            logger.info("whatsapp: connected to the sidecar at %s", self._socket_path)
            try:
                await self._report_status()
                await client._read_task
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "whatsapp: the sidecar connection dropped; reconnecting in %.0fs",
                    RECONNECT_DELAY, exc_info=True,
                )
            finally:
                self._client = None
                await client.close()
            await asyncio.sleep(RECONNECT_DELAY)

    async def _report_status(self) -> None:
        """Ask the sidecar where the link stands and say so in the log."""
        if self._client is None:
            return
        try:
            status = await self._client.call("status")
        except SidecarError as exc:
            self._log_sidecar_error("status", exc)
            return
        state = (status or {}).get("state", "unknown")
        self._session_state = state
        logger.info("whatsapp: sidecar link state is %s", state)

    def _log_sidecar_error(self, what: str, exc: SidecarError) -> None:
        """One operator-facing line per sidecar error, naming the fix."""
        if exc.code == NOT_LINKED:
            logger.warning(
                "whatsapp: %s failed because the link is logged out; %s",
                what, exc.fix,
            )
        elif exc.code == NOT_CONNECTED:
            logger.info("whatsapp: %s skipped, the link is not ready: %s", what, exc.fix)
        else:
            logger.warning("whatsapp: %s failed: %s; %s", what, exc, exc.fix)

    async def _on_notification(self, method: str, params: dict) -> None:
        """Handle one unprompted message from the sidecar."""
        if method == "message":
            await self._on_incoming(params)
        elif method == "session":
            state = params.get("state", "unknown")
            if state != self._session_state:
                logger.info("whatsapp: link state changed to %s", state)
            self._session_state = state
        else:
            logger.debug("whatsapp: ignoring notification %s", method)

    async def _on_incoming(self, message: dict) -> None:
        """Turn one inbound WhatsApp message into a channel message."""
        if message.get("from_me"):
            return
        # A reconnect replays unacked messages, so the same wamid can arrive
        # twice; the second copy must not start a second turn.
        wamid = message.get("id")
        if wamid and wamid in self._seen:
            logger.info("whatsapp: duplicate message %s in %s; ignored", wamid,
                        message.get("chat_jid"))
            return
        chat_jid = message.get("chat_jid")
        text = message.get("text") or ""
        if not chat_jid:
            logger.warning("whatsapp: message without a chat_jid; dropping")
            return
        if not text.strip():
            # v1 carries text only; media-only messages are named rather than
            # silently dropped, so the gap is visible in the log.
            logger.info(
                "whatsapp: message in %s carries no text (media is not carried yet);"
                " nothing was sent to the agent", chat_jid,
            )
            return
        if self._registry is None:
            logger.warning("whatsapp: message arrived before the registry resolved; dropping")
            return
        channel = self._registry.get_or_create("whatsapp", chat_jid, config=ChannelConfig())
        sender = message.get("sender_jid") or chat_jid
        await self.pm.ahook.on_message(
            channel=channel, sender=sender, text=text, trace_id=mint_trace_id(),
        )
        # Remember the id only once the message has been handed over: a message
        # dropped above (no text, no registry) keeps its id, so a replay of it
        # is still delivered. Delivery is at-most-once: corvidae's hook guard
        # (hooks.py#guarded) swallows a raising implementation, so this line runs
        # even when the hand-over failed and a replay is then suppressed. That is
        # the deliberate direction -- a duplicate turn is worse than a lost one,
        # and an implementation that raised has usually seen the message.
        if wamid:
            self._seen[wamid] = None
            while len(self._seen) > SEEN_LIMIT:
                self._seen.popitem(last=False)

    # Outbound ---------------------------------------------------------------

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Send a final reply through the sidecar, logging why when it cannot."""
        if self._client is None:
            logger.warning(
                "whatsapp: cannot reply on %s: the sidecar is not connected;"
                " start it and the link will resume", channel.id,
            )
            return
        try:
            await self._client.call(
                "send_text",
                {"chat_jid": channel.scope, "text": event.text, "reply_to": None},
            )
        except SidecarError as exc:
            self._log_sidecar_error(f"send on {channel.id}", exc)
