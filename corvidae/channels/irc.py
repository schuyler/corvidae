"""IRC transport plugin using pydle library."""

import asyncio
import collections
import logging
from typing import Optional

import pydle

logger = logging.getLogger("corvidae.irc_plugin")

# Most messages held while disconnected; the oldest is dropped past this, so a
# long outage (up to the 300s backoff cap and beyond) cannot grow memory.
_OUTBOUND_QUEUE_MAX = 50

from corvidae.channel import Channel, ChannelRegistry
from corvidae.channels.split import split_message
from corvidae.trace import mint_trace_id
from corvidae.hooks import TransportPlugin, get_dependency, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress


class IRCClient(pydle.Client):
    """Thin pydle subclass. Holds reference to IRCPlugin."""

    # Connection management defaults (seconds)
    _INITIAL_BACKOFF = 10
    _BACKOFF_MULTIPLIER = 2
    _BACKOFF_CAP = 300
    _CONNECTION_POLL_INTERVAL = 60

    def __init__(self, plugin, nickname, server=None, port=None, tls=False, **kwargs):
        super().__init__(nickname, **kwargs)
        self.plugin = plugin
        self._server = server
        self._port = port
        self._tls = tls

    async def on_isupport_modes(self, value):
        """Handle MODES ISUPPORT token; pydle crashes when value is None."""
        if value is not None:
            self._mode_limit = int(value)

    async def on_unknown(self, message):
        """Demote pydle's WARNING for unhandled numerics to DEBUG."""
        logger.debug(
            "Unhandled IRC command: [%s] %s %s",
            message.source,
            message.command,
            message.params,
        )

    async def on_raw(self, message):
        """Derive each message sender's account from that message's own tags.

        pydle copies the `account` tag into its user record only when the
        nick is already known and never clears it on an untagged message, so
        a stale record would let a later unidentified user of the same nick
        inherit the account (#77). With account-tag negotiated, every
        PRIVMSG carries the tag iff the sender is logged in, so the record
        is overwritten from the tag (absent => not logged in) before pydle
        dispatches the message.
        """
        if message.command == "PRIVMSG" and self._capabilities.get("account-tag"):
            nick, _ = self._parse_user(message.source)
            account = message.tags.get("account") or None
            await self._sync_user(
                nick, {"account": account, "identified": account is not None}
            )
        await super().on_raw(message)

    async def on_connect(self):
        """Join configured channels after connecting."""
        await super().on_connect()
        for channel in self.plugin.channels:
            await self.join(channel)
        # Deliver output queued during the outage, after the joins.
        await self.plugin._flush_outbound()

    async def on_channel_message(self, target, by, message):
        """Forward channel messages to plugin."""
        await self.plugin.on_channel_message(target, by, message)

    async def on_private_message(self, target, by, message):
        """Forward private messages to plugin."""
        await self.plugin.on_private_message(target, by, message)

    async def connect_with_retry(self):
        """Connect with exponential backoff retry."""
        delay = self._INITIAL_BACKOFF
        task = self.plugin._connect_task  # capture for shutdown check
        try:
            while task is not None:
                try:
                    await self.connect(self._server, self._port, tls=self._tls)
                    delay = self._INITIAL_BACKOFF  # reset on success
                    # Hold connection open, polling for shutdown/disconnect
                    while self.plugin._connect_task is not None:
                        await asyncio.sleep(self._CONNECTION_POLL_INTERVAL)
                        if not self.connected:
                            break
                except Exception as exc:
                    # Any failure at all must re-enter backoff: an exception
                    # type outside (OSError, pydle.Error) otherwise ends the
                    # connect task and leaves the bot offline for good.
                    logger.warning(
                        "IRC connection error to %s:%s as %s, retrying: %s",
                        self._server,
                        self._port,
                        self.nickname,
                        exc,
                        exc_info=True,
                    )
                if self.plugin._connect_task is None:
                    return
                await asyncio.sleep(delay)
                delay = min(delay * self._BACKOFF_MULTIPLIER, self._BACKOFF_CAP)
        except asyncio.CancelledError:
            if self.plugin._connect_task is None:
                return
            raise


class IRCPlugin(TransportPlugin):
    """IRC transport plugin following CLIPlugin pattern."""

    transport = "irc"

    def __init__(self, pm=None):
        if pm is not None:
            self.pm = pm
        self.client: Optional[IRCClient] = None
        self._connect_task: Optional[asyncio.Task] = None
        self.channels: list[str] = []
        self._registry: Optional[ChannelRegistry] = None
        self._message_chunk_size: int = 400
        self._irc_server: str = "irc.libera.chat"
        self._irc_port: int = 6667
        self._irc_nick: str = "corvidae"
        self._irc_tls: bool = False
        # Nicks / services account names allowed to drive the agent, as
        # configured; compared under the server's casemapping at match time. Empty means nobody: IRC is default-deny (#77).
        self._allow: set[str] = set()
        # Output held while disconnected, oldest first (#106).
        # True while _flush_outbound is delivering _outbound[0]; that entry
        # is in flight and must not be dropped by overflow handling.
        self._flushing = False
        self._outbound: collections.deque[tuple[Channel, str]] = collections.deque()
        # Messages dropped from the queue in the current outage.
        self._dropped: int = 0

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        irc_config = config.get("irc")
        if irc_config is None:
            return
        self._irc_server = irc_config.get("host", "irc.libera.chat")
        self._irc_port = irc_config.get("port", 6667)
        self._irc_nick = irc_config.get("nick", "corvidae")
        self.channels = irc_config.get("channels", [])
        self._irc_tls = irc_config.get("tls", False)
        self._message_chunk_size = irc_config.get("message_chunk_size", 400)

        # Validate the allowlist at startup so a typo fails loudly, naming the
        # field, instead of silently locking everyone out (or in).
        if "allow" in irc_config:
            allow = irc_config["allow"]
            if not isinstance(allow, list) or not all(
                isinstance(entry, str) for entry in allow
            ):
                raise ValueError(
                    "irc: config field 'allow' must be a list of nicks or "
                    f"account names (got {allow!r}) — fix the irc.allow key "
                    "in agent.yaml"
                )
            # Stored raw; _fold applies the server's casemapping at match time.
            self._allow = set(allow)
        else:
            # Upgrade notice: IRC used to accept everyone.
            logger.warning(
                "irc: no 'allow' list in the irc config block; IRC is "
                "default-deny and will ignore every sender until irc.allow "
                "is set in agent.yaml"
            )

    @hookimpl
    async def on_start(self, config: dict) -> None:
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        irc_config = config.get("irc")
        if irc_config is None:
            return

        self.client = IRCClient(
            self, self._irc_nick,
            server=self._irc_server, port=self._irc_port, tls=self._irc_tls,
        )
        self._connect_task = asyncio.create_task(self.client.connect_with_retry())

    # Capabilities that make pydle track services accounts for us.
    _ACCOUNT_CAPS = ("account-notify", "account-tag", "extended-join")

    def _account_of(self, nick: str) -> tuple[bool, Optional[str]]:
        """(network reports accounts, the sender's account or None).

        The account comes from pydle's user state. Account support is
        inferred from the negotiated IRCv3 capabilities, or from the sender
        already having an account (e.g. learned from WHOIS).
        """
        users = getattr(self.client, "users", None) or {}
        info = users.get(nick) or {}
        account = info.get("account") or None
        caps = getattr(self.client, "_capabilities", None) or {}
        supported = bool(account) or any(caps.get(c) for c in self._ACCOUNT_CAPS)
        return supported, account

    def _fold(self, name: str) -> str:
        """Normalize a nick/account under the server's CASEMAPPING.

        Uses the client's normalizer (ascii / rfc1459 / strict-rfc1459), not
        str.casefold, which is broader than any server (``straße`` vs
        ``strasse``). Before a client exists, falls back to ASCII lowering.
        """
        normalize = getattr(self.client, "normalize", None)
        if normalize is None:
            return name.lower()
        return normalize(name)

    def _is_allowed(self, nick: str) -> bool:
        """Default-deny check. Where the network has accounts the account
        must match (an unidentified or unknown sender is refused); otherwise
        the nick does."""
        supported, account = self._account_of(nick)
        identity = account if supported else nick
        if identity is None:
            return False
        # Normalize both sides at match time.
        allowed = {self._fold(entry) for entry in self._allow}
        return self._fold(identity) in allowed

    def _gate_channel(self, scope: str, nick: str) -> Channel:
        """The channel to gate a message on.

        An allowed sender gets the registered channel. A refused one gets an
        existing registry entry if there is one, else an unregistered
        Channel, so nothing outlives the refusal (#77, as Signal #103).
        """
        if self._is_allowed(nick):
            return self._registry.get_or_create("irc", scope)
        existing = self._registry.get(f"irc:{scope}")
        if existing is not None:
            return existing
        return Channel(transport="irc", scope=scope)

    @hookimpl
    async def should_process_message(
        self, channel: Channel, sender: str, text: str, trace_id: str
    ) -> bool | None:
        """Default-deny gate: only an irc.allow sender may drive the agent.

        trace_id is bound without a default (tests/test_hook_arg_binding.py).
        """
        if not self.owns(channel):
            return None
        if self._is_allowed(sender):
            return True
        # Silent refusal: log once, say nothing to the sender.
        _supported, account = self._account_of(sender)
        logger.info(
            "irc: rejected message from unauthorized sender nick=%s%s",
            sender,
            f" account={account}" if account else "",
        )
        return False

    async def on_channel_message(self, target, by, message):
        """Forward channel messages to agent loop (called by IRCClient)."""
        if self.client is not None and by == self.client.nickname:
            return
        channel = self._gate_channel(target, by)
        await self.pm.ahook.on_message(
            channel=channel,
            sender=by,
            text=message,
            # Transport-minted id; the Agent uses it verbatim (#95).
            trace_id=mint_trace_id(),
        )

    async def on_private_message(self, target, by, message):
        """Forward private messages to agent loop (called by IRCClient)."""
        if self.client is not None and by == self.client.nickname:
            return
        channel = self._gate_channel(by, by)
        await self.pm.ahook.on_message(
            channel=channel,
            sender=by,
            text=message,
            # Transport-minted id; the Agent uses it verbatim (#95).
            trace_id=mint_trace_id(),
        )

    async def _deliver(self, channel: Channel, text: str) -> None:
        """Send text now: one PRIVMSG per non-blank line, split to fit."""
        # Split into chunks that fit an IRC line, then send line by line.
        chunks = split_message(text, max_len=self._message_chunk_size)
        for chunk in chunks:
            for line in chunk.split('\n'):
                line = line.rstrip('\r')
                if not line.strip():
                    continue
                await self.client.message(channel.scope, line)

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Deliver a reply, or queue it while disconnected or mid-flush."""
        text = event.text
        # Queue when offline, or when older output is still waiting, so a new
        # message cannot overtake queued ones.
        if (
            self.client is None
            or not self.client.connected
            or self._outbound
        ):
            # Bounded: drop the oldest entry rather than grow without limit.
            # While flushing, the head is in flight (the flush pops it after
            # delivery), so the oldest droppable entry is index 1.
            drop_index = 1 if self._flushing else 0
            if len(self._outbound) >= _OUTBOUND_QUEUE_MAX and len(
                self._outbound
            ) > drop_index:
                old_channel, old_text = self._outbound[drop_index]
                del self._outbound[drop_index]
                self._dropped += 1
                logger.warning(
                    "IRC outbound queue full, dropped oldest message "
                    "(channel=%s, chars=%d, dropped=%d this outage)",
                    old_channel.id,
                    len(old_text),
                    self._dropped,
                )
            self._outbound.append((channel, text))
            logger.warning(
                "IRC message queued until reconnect, not dropped "
                "(channel=%s, chars=%d, depth=%d)",
                channel.id,
                len(text),
                len(self._outbound),
            )
            return
        await self._deliver(channel, text)

    async def _flush_outbound(self) -> None:
        """Deliver queued output in order; keep any remainder on failure."""
        flushed = 0
        stopped_early = False
        # Mark the head as in flight so overflow drops never remove it.
        self._flushing = True
        try:
            # Pop only after delivery so a failure leaves the rest queued, and
            # concurrent sends keep queueing behind the head.
            while self._outbound:
                if self.client is None or not self.client.connected:
                    stopped_early = True
                    break
                channel, text = self._outbound[0]
                try:
                    await self._deliver(channel, text)
                except Exception:
                    logger.warning(
                        "IRC flush of queued output failed (channel=%s, "
                        "%d still queued); will retry on next connect",
                        channel.id,
                        len(self._outbound),
                        exc_info=True,
                    )
                    return
                self._outbound.popleft()
                flushed += 1
        finally:
            self._flushing = False
        # Report accurately: a disconnect mid-flush leaves a remainder.
        logger.info(
            "IRC outbound flush %s (sent=%d, dropped=%d, remaining=%d)",
            "stopped on disconnect" if stopped_early else "complete",
            flushed,
            self._dropped,
            len(self._outbound),
        )
        if not stopped_early:
            self._dropped = 0

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        """Show intermediate text as an ordinary message."""
        await self.send_message(channel, OutboundMessage(text=event.text))

    @hookimpl
    async def on_stop(self) -> None:
        if self._outbound:
            # Queue is memory-only; say so rather than lose it silently.
            logger.warning(
                "IRC shutting down with %d queued message(s) undelivered; "
                "they are discarded",
                len(self._outbound),
            )
        task = self._connect_task
        self._connect_task = None  # shutdown signal
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Exception awaiting connect_task during shutdown", exc_info=True)
        if self.client is not None:
            try:
                await self.client.quit("Shutting down")
            except Exception:
                logger.debug("Exception during client.quit", exc_info=True)
            self.client = None


# Re-exported here (imported above) so this module's send_message, and
# tests that patch `corvidae.channels.irc.split_message`, keep resolving it
# via this module's namespace.
