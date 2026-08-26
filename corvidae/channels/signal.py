"""Signal transport plugin, speaking JSON-RPC to a signal-cli daemon.

Connects to an already-running `signal-cli daemon` over a unix socket
(newline-delimited JSON-RPC 2.0). corvidae never spawns or supervises
signal-cli itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.channels.split import split_message
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger("corvidae.signal_plugin")

# How often the per-channel typing-indicator refresh loop re-sends "typing"
# while a turn is outstanding. Signal clients expire an indicator after
# roughly 15s with no refresh. Read by name (not bound as a default
# argument) so that tests patching this module attribute take effect --
# see tests/test_signal_plugin.py's _fast_typing_refresh.
TYPING_REFRESH_SECONDS = 10

# An inbound envelope older than this (wall-clock at decode time, vs. the
# envelope's own send timestamp) is treated as backlog and gets a
# "[sent ...]" prefix prepended to its text, so the agent can tell how old
# the message actually is rather than reading it as current.
STALE_MESSAGE_THRESHOLD_SECONDS = 5 * 60

# Connection retry backoff ladder -- the same one IRCClient uses.
_INITIAL_BACKOFF = 10
_BACKOFF_MULTIPLIER = 2
_BACKOFF_CAP = 300


def _format_stale_prefix(timestamp_ms: int) -> str:
    """Render an envelope timestamp as a "[sent YYYY-MM-DD HH:MM UTC]" prefix."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return f"[sent {dt.strftime('%Y-%m-%d %H:%M')} UTC]\n"


class SignalPlugin(CorvidaePlugin):
    """Transport plugin for a Signal DM conversation via signal-cli's daemon."""

    depends_on = frozenset({"registry"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._registry: Optional[ChannelRegistry] = None

        # Config, populated by on_init. _socket_path stays None when the
        # signal: block is absent -- that is the inert-plugin signal on_start
        # checks, since a present-but-empty-ish block would still set it.
        self._socket_path: Optional[str] = None
        self._account: Optional[str] = None
        self._allow: set[str] = set()
        self._message_chunk_size: int = 2000

        # Connection state.
        self._connect_task: Optional[asyncio.Task] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._next_request_id = 1
        self._pending_requests: dict[int, asyncio.Future] = {}

        # Inbound notifications are queued and processed by a separate task
        # (below) rather than handled inline in the read loop: handling one
        # (typing, receipts) sends its own JSON-RPC requests and awaits
        # their responses, and those responses only ever arrive by the read
        # loop reading the next line -- handling inline would deadlock the
        # transport against its own request the first time it tried to
        # await a response. A single consumer preserves the arrival order
        # of inbound messages while the read loop stays free to deliver
        # responses.
        self._notification_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None

        # Forward alias resolution, shared by allowlist matching and the
        # channels: config override: a configured E.164 allow entry is
        # resolved to an ACI via a single batched getUserStatus call after
        # each successful connect (never at on_init/on_start -- see
        # _maybe_start_allowlist_resolution). Keyed by the alias string, so
        # a channels: alias can be matched back to whichever sender it
        # resolved to. A resolved alias is cached for the process lifetime;
        # one that failed to resolve is retried after the next connect.
        self._resolved_aliases: dict[str, str] = {}
        self._allow_resolution_task: Optional[asyncio.Task] = None

        # Liveness bookkeeping, per channel id (transport:scope).
        self._outstanding: dict[str, int] = {}
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._pending_timestamps: dict[str, deque] = {}
        self._admitted_correlation_ids: dict[str, set[str]] = {}

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read the signal: config block.

        Absence means the transport stays inert -- no connection, no
        channels, nothing above DEBUG. A present block missing a required
        field, or with a malformed field, raises immediately with a
        message naming the problem: silence is for absence, not errors.
        """
        await super().on_init(pm, config)
        signal_config = config.get("signal")
        if signal_config is None:
            return
        if not isinstance(signal_config, dict):
            raise ValueError("signal: config block must be a mapping")

        socket_path = signal_config.get("socket")
        if not socket_path:
            raise ValueError("signal: config is missing required field 'socket'")
        account = signal_config.get("account")
        if not account:
            raise ValueError("signal: config is missing required field 'account'")
        allow = signal_config.get("allow", [])
        if not isinstance(allow, list):
            raise ValueError("signal: config field 'allow' must be a list")

        self._socket_path = socket_path
        self._account = account
        self._allow = set(allow)
        self._message_chunk_size = signal_config.get("message_chunk_size", 2000)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Resolve the channel registry, then open the socket under retry.

        The registry is resolved even when the plugin ends up inert, so
        validate_dependencies sees the same behavior regardless of config.
        """
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        if self._socket_path is None:
            return
        self._connect_task = asyncio.create_task(self._connection_loop())
        self._processor_task = asyncio.create_task(self._process_notifications())

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the connect/read task, any typing loops, and close the socket."""
        task = self._connect_task
        self._connect_task = None  # shutdown signal for the retry loop
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("signal: exception awaiting connect task during shutdown", exc_info=True)

        processor = self._processor_task
        self._processor_task = None
        if processor is not None:
            processor.cancel()
            try:
                await processor
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("signal: exception awaiting processor task during shutdown", exc_info=True)

        resolution = self._allow_resolution_task
        self._allow_resolution_task = None
        if resolution is not None:
            resolution.cancel()
            try:
                await resolution
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("signal: exception awaiting allowlist resolution task during shutdown", exc_info=True)

        for typing_task in list(self._typing_tasks.values()):
            typing_task.cancel()
        for typing_task in list(self._typing_tasks.values()):
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("signal: exception awaiting typing task during shutdown", exc_info=True)
        self._typing_tasks.clear()

        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.warning("signal: exception closing socket during shutdown", exc_info=True)
            self._writer = None
            self._reader = None

    # -----------------------------------------------------------------
    # Connection loop
    # -----------------------------------------------------------------

    async def _connection_loop(self) -> None:
        """Connect to the signal-cli socket, read frames, and retry on drop.

        Retries with the same backoff ladder IRC uses: 10s initial, doubling,
        capped at 300s. Delay resets to the initial value after any
        successful connection, so a drop right after a long-lived connection
        retries quickly rather than at whatever the ladder had climbed to
        before.
        """
        delay = _INITIAL_BACKOFF
        while self._connect_task is not None:
            try:
                reader, writer = await asyncio.open_unix_connection(path=self._socket_path)
            except OSError as exc:
                logger.warning("signal: connection error, retrying: %s", exc, exc_info=True)
                if self._connect_task is None:
                    return
                await asyncio.sleep(delay)
                delay = min(delay * _BACKOFF_MULTIPLIER, _BACKOFF_CAP)
                continue

            self._reader = reader
            self._writer = writer
            delay = _INITIAL_BACKOFF  # reset on successful connect
            self._maybe_start_allowlist_resolution()
            try:
                await self._read_loop(reader)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("signal: read loop raised unexpectedly", exc_info=True)
            finally:
                self._reject_pending_requests()
                try:
                    writer.close()
                except Exception:
                    logger.debug("signal: exception closing writer after disconnect", exc_info=True)
                self._reader = None
                self._writer = None

            if self._connect_task is None:
                return
            await asyncio.sleep(delay)
            delay = min(delay * _BACKOFF_MULTIPLIER, _BACKOFF_CAP)

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        """Read newline-delimited JSON-RPC frames until the connection drops.

        Never processes a notification inline -- see the comment on
        _notification_queue in __init__ for why. A malformed line is logged
        and skipped so one bad frame does not take the whole connection down.
        """
        while True:
            line = await reader.readline()
            if not line:
                return  # EOF: connection dropped
            stripped = line.strip()
            if not stripped:
                continue
            try:
                frame = json.loads(stripped.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("signal: malformed JSON-RPC frame, skipping: %s", exc, exc_info=True)
                continue
            self._dispatch_frame(frame)

    def _dispatch_frame(self, frame: dict) -> None:
        """Route a decoded frame: a notification (has "method") is queued
        for the processor task; a response to a request this client made
        (has "id", no "method") resolves its pending future immediately."""
        if "method" in frame:
            self._notification_queue.put_nowait((frame.get("method"), frame.get("params") or {}))
            return
        req_id = frame.get("id")
        if req_id is not None:
            self._resolve_pending_request(frame, req_id)

    async def _process_notifications(self) -> None:
        """Consume queued notifications one at a time, in the order they
        arrived, so rapid-succession messages are never processed out of
        order or interleaved. Runs for the plugin's lifetime, independent
        of individual connection cycles."""
        while True:
            method, params = await self._notification_queue.get()
            try:
                await self._handle_notification(method, params)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("signal: error handling inbound message, continuing", exc_info=True)

    def _resolve_pending_request(self, frame: dict, req_id: Any) -> None:
        """Deliver a JSON-RPC response to the future _send_rpc is awaiting."""
        future = self._pending_requests.pop(req_id, None)
        if future is None or future.done():
            return
        if "error" in frame:
            error = frame["error"]
            # Read as a sentence, not a dict repr -- but keep the numeric
            # code, since e.g. -32601 (wrong method name / version skew)
            # and -32602 (missing/wrong account) call for different
            # operator action.
            if isinstance(error, dict):
                message = error.get("message", str(error))
                code = error.get("code")
                text = f"{message} (code {code})" if code is not None else message
            else:
                text = str(error)
            future.set_exception(RuntimeError(text))
        else:
            future.set_result(frame.get("result"))

    def _reject_pending_requests(self) -> None:
        """On disconnect, fail any outstanding request rather than hang it."""
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(ConnectionError("signal-cli connection lost"))
        self._pending_requests.clear()

    async def _send_rpc(self, method: str, params: dict) -> Any:
        """Send a JSON-RPC request and await its response."""
        writer = self._writer
        if writer is None:
            raise ConnectionError("not connected to signal-cli")
        req_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future
        frame = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        writer.write((json.dumps(frame) + "\n").encode("utf-8"))
        await writer.drain()
        return await future

    # -----------------------------------------------------------------
    # Inbound decoding and filtering
    # -----------------------------------------------------------------

    async def _handle_notification(self, method: str, params: dict) -> None:
        """Handle an unsolicited JSON-RPC notification from signal-cli."""
        if method != "receive":
            logger.debug("signal: ignoring notification method %r", method)
            return
        envelope = params.get("envelope") or {}

        # Rule 1: only a dataMessage is input. A syncMessage is the account's
        # own traffic echoed from another device signed into it -- treating
        # it as input is the infinite-loop failure mode this transport must
        # avoid.
        data_message = envelope.get("dataMessage")
        if data_message is None:
            logger.debug("signal: envelope carries no dataMessage, ignoring")
            return

        # Rule 2: not from the bot's own account. The bot's own identity is
        # known to this transport only as its configured E.164 number, so
        # this matches on sourceNumber.
        source_number = envelope.get("sourceNumber")
        if source_number is not None and source_number == self._account:
            logger.debug("signal: ignoring message from the bot's own account")
            return

        # Rule 3: group conversations are out of scope.
        if data_message.get("groupInfo") is not None:
            logger.debug("signal: ignoring group message")
            return

        # Rule 4: text must be non-empty after attachments are dropped.
        # Reactions, quotes, edits, and remote deletes all fall out here
        # since none of them carry a "message" field.
        attachments = data_message.get("attachments")
        if attachments:
            logger.debug("signal: dropping %d attachment(s)", len(attachments))
        text = data_message.get("message")
        if not text:
            logger.debug("signal: ignoring envelope with no text content")
            return

        source_uuid = envelope.get("sourceUuid")
        if not source_uuid:
            logger.warning("signal: envelope has no sourceUuid, cannot scope a channel, dropping")
            return

        timestamp = envelope.get("timestamp", data_message.get("timestamp"))
        channel = await self._resolve_channel(source_uuid, source_number)
        text = self._maybe_prefix_stale(text, timestamp)

        # Typing and the read receipt only ever apply to an authorized
        # sender -- an unauthorized one gets no signal the account is live.
        # on_message still fires regardless so the real should_process_message
        # gate can veto and log the attempt.
        if await self._is_allowed(source_uuid) and timestamp is not None:
            self._push_pending_timestamp(channel, timestamp)
            await self._begin_typing(channel)

        await self.pm.ahook.on_message(channel=channel, sender=source_uuid, text=text)

    def _maybe_prefix_stale(self, text: str, timestamp_ms: int | None) -> str:
        """Prepend a "[sent ...]" prefix when an envelope is backlog."""
        if timestamp_ms is None:
            return text
        age_seconds = (time.time() * 1000 - timestamp_ms) / 1000
        if age_seconds <= STALE_MESSAGE_THRESHOLD_SECONDS:
            return text
        return _format_stale_prefix(timestamp_ms) + text

    # -----------------------------------------------------------------
    # Channel scope and identity
    # -----------------------------------------------------------------

    async def _resolve_channel(self, source_uuid: str, source_number: str | None) -> Channel:
        """Get or create the ACI-keyed channel for a sender, applying any
        E.164 alias's ChannelConfig on first creation.

        Scope is always the ACI -- phone numbers are never used as the
        stored scope since sourceNumber can be absent under phone-number
        privacy, and channel ids are the persistence key for sessions.db,
        the jsonl logs, and memory: keying on a field that can vanish
        would fork a sender's history.
        """
        channel_id = f"signal:{source_uuid}"
        existing = self._registry.get(channel_id)
        if existing is not None:
            return existing
        alias_config = await self._resolve_alias_config(source_uuid, source_number)
        return self._registry.get_or_create("signal", source_uuid, config=alias_config)

    async def _resolve_alias_config(
        self, source_uuid: str, source_number: str | None
    ) -> ChannelConfig | None:
        """Find a pre-registered E.164-keyed alias channel for this sender
        and return its ChannelConfig, or None if none matches.

        A direct sourceNumber match is free. Otherwise -- typically when
        phone-number privacy hides sourceNumber -- this reuses the
        allowlist's forward E.164->ACI resolution (see
        _resolve_allowlist_aliases) rather than a reverse ACI->number
        lookup: signal-cli's contact store only resolves a number for a
        sender it already treats as a contact (profile_sharing, set once
        the bot has replied at least once), and an unknown ACI silently
        returns a phantom entry with no number -- indistinguishable from
        "not found" -- so a reverse lookup on the very first message from
        a sender is order-dependent and can go wrong silently. Reuse is
        sound because any sender whose override could matter must also be
        allowlisted -- an unauthorized sender never gets a reply, so a
        channels: alias that resolves to nobody in allow: is simply inert,
        not broken.
        """
        alias_channels = {
            c.scope: c for c in self._registry.by_transport("signal") if c.scope.startswith("+")
        }
        if not alias_channels:
            return None
        if source_number is not None and source_number in alias_channels:
            return alias_channels[source_number].config

        await self._await_allow_resolution()
        for alias, aci in self._resolved_aliases.items():
            if aci == source_uuid and alias in alias_channels:
                return alias_channels[alias].config
        return None

    # -----------------------------------------------------------------
    # Authorization gate
    # -----------------------------------------------------------------

    def _configured_allow_aliases(self) -> list[str]:
        """The configured `allow` entries that are E.164 numbers rather
        than ACIs, and so need forward resolution."""
        return [a for a in self._allow if a.startswith("+")]

    def _maybe_start_allowlist_resolution(self) -> None:
        """Kick off forward resolution of any configured E.164 allowlist
        entries that have not resolved yet.

        Called after each successful connect, never from on_init/on_start,
        so an unreachable daemon at startup leaves the plugin inert rather
        than blocking it. Only entries still unresolved are retried, since
        a reconnect may succeed where a prior attempt hit a rate limit or
        an up-but-not-yet-usable daemon.
        """
        pending = [a for a in self._configured_allow_aliases() if a not in self._resolved_aliases]
        if not pending:
            return
        self._allow_resolution_task = asyncio.create_task(self._resolve_allowlist_aliases(pending))

    async def _resolve_allowlist_aliases(self, aliases: list[str]) -> None:
        """Forward-resolve configured E.164 allowlist entries to ACIs in a
        single batched getUserStatus call.

        Batched because this is a CDS lookup with a server-side rate
        limit -- resolving N aliases as N separate calls risks exhausting
        it. A failure of the whole batch (e.g. a rate limit, or a daemon
        that is connected but not yet usable) is logged at ERROR with the
        underlying error text; a per-alias entry that comes back with no
        ACI (the number simply isn't registered on Signal) is logged as
        its own, separate ERROR -- so the two failure modes read
        differently in the log instead of both looking like an ordinary
        unauthorized-sender refusal.
        """
        try:
            result = await self._send_rpc("getUserStatus", {"recipient": aliases})
        except Exception as exc:
            logger.error(
                "signal: failed to resolve configured allowlist alias(es) %s: %s",
                ", ".join(aliases), exc,
            )
            return
        entries = result if isinstance(result, list) else [result]
        aci_by_alias = {
            entry.get("recipient") or entry.get("number"): entry.get("uuid")
            for entry in entries if isinstance(entry, dict)
        }
        for alias in aliases:
            aci = aci_by_alias.get(alias)
            if not aci:
                logger.error("signal: configured allowlist alias %s did not resolve to an account", alias)
                continue
            self._resolved_aliases[alias] = aci

    async def _await_allow_resolution(self) -> None:
        """Block until any in-flight allowlist alias resolution has
        settled, without starting one itself -- resolution only ever
        starts after a connect (_maybe_start_allowlist_resolution)."""
        task = self._allow_resolution_task
        if task is not None and not task.done():
            await task

    async def _is_allowed(self, source_uuid: str) -> bool:
        """Default-deny allowlist check: the sender's ACI must be a
        configured allow entry directly, or be what a configured E.164
        allow entry resolved to. Awaits any resolution still in flight, so
        a message arriving right after connect is judged against the
        finished result rather than a partial one."""
        if source_uuid in self._allow:
            return True
        await self._await_allow_resolution()
        return source_uuid in self._resolved_aliases.values()

    @hookimpl
    async def should_process_message(
        self, channel: Channel, sender: str, text: str, correlation_id: str
    ) -> bool | None:
        """Default-deny gate: only an allowlisted ACI (or its known E.164
        alias) may drive the agent. correlation_id is bound without a
        default -- see tests/test_hook_arg_binding.py.
        """
        if not channel.matches_transport("signal"):
            return None
        if await self._is_allowed(sender):
            return True
        logger.info("signal: rejected message from unauthorized sender %s", sender)
        return False

    @hookimpl
    async def on_message_rejected(
        self, channel: Channel, correlation_id: str, sender: str, text: str
    ) -> None:
        """A second gate plugin may veto a message this transport already
        counted as outstanding; give the typing count back so the refresh
        loop still reaches zero.

        Deliberately does not pop _pending_timestamps here: the transport
        only pushes a timestamp for a sender its own allowlist admits, so
        a transport-side refusal never pushes one and there is nothing to
        pop for it. That leaves a gap -- if a *second* gate plugin ever
        vetoes a message this transport already admitted, that message's
        timestamp is never popped, and every later read receipt on the
        channel attaches to that stale timestamp instead of its own. Safe
        today only because the transport's own allowlist check is the
        only gate on signal channels; any future gate plugin on signal
        channels must add the pop here.
        """
        if not channel.matches_transport("signal"):
            return
        self._decrement_outstanding(channel)

    # -----------------------------------------------------------------
    # Liveness: typing indicator and read receipt
    # -----------------------------------------------------------------

    async def _begin_typing(self, channel: Channel) -> None:
        """Count this message as outstanding; on the first one, send typing
        immediately (covering queueing delay) and start the refresh loop."""
        count = self._outstanding.get(channel.id, 0) + 1
        self._outstanding[channel.id] = count
        if count == 1:
            await self._send_typing(channel)
            self._typing_tasks[channel.id] = asyncio.create_task(
                self._typing_refresh_loop(channel)
            )

    def _decrement_outstanding(self, channel: Channel) -> None:
        """Give back one outstanding slot: a final reply send, an
        error-path send, or a rejection all close out one owed message."""
        count = self._outstanding.get(channel.id, 0)
        if count > 0:
            self._outstanding[channel.id] = count - 1

    async def _typing_refresh_loop(self, channel: Channel) -> None:
        """Re-send typing every TYPING_REFRESH_SECONDS while the channel's
        outstanding count stays positive; stop once it reaches zero or a
        send fails."""
        try:
            while self._outstanding.get(channel.id, 0) > 0:
                await asyncio.sleep(TYPING_REFRESH_SECONDS)
                if self._outstanding.get(channel.id, 0) <= 0:
                    return
                if not await self._send_typing(channel):
                    return
        finally:
            self._typing_tasks.pop(channel.id, None)

    async def _send_typing(self, channel: Channel) -> bool:
        """Send one typing-indicator request. A failure is logged and never
        blocks the reply; returns whether the send succeeded, so the
        refresh loop can stop retrying against a channel that keeps
        failing."""
        try:
            # sendTyping is one method with a boolean "stop" -- not a
            # separate stop method or an action string. False starts (or
            # refreshes) the indicator.
            await self._send_rpc("sendTyping", {"recipient": channel.scope, "stop": False})
            return True
        except Exception as exc:
            logger.warning("signal: failed to send typing indicator: %s", exc, exc_info=True)
            return False

    def _push_pending_timestamp(self, channel: Channel, timestamp_ms: int) -> None:
        """Record an inbound envelope's send timestamp, FIFO per channel,
        for the read receipt to pop when that turn starts processing."""
        self._pending_timestamps.setdefault(channel.id, deque()).append(timestamp_ms)

    @hookimpl
    async def on_message_admitted(
        self, channel: Channel, correlation_id: str, sender: str, text: str
    ) -> None:
        """Record an admitted USER message's correlation id, so the matching
        on_message_persisted firing (processing start) can be recognized as
        this channel's next owed read receipt."""
        if not channel.matches_transport("signal"):
            return
        self._admitted_correlation_ids.setdefault(channel.id, set()).add(correlation_id)

    @hookimpl
    async def on_message_persisted(
        self, channel: Channel, correlation_id: str, rowid: int | None, text: str, meta: dict
    ) -> None:
        """Fire the read receipt at processing start, not arrival -- a
        message can sit queued behind contended local inference, and a
        "read" marker followed by silence is worse than no marker. Pops the
        channel's oldest pending envelope timestamp and sends a receipt for
        it, but only for a correlation id this transport itself admitted --
        this hook also fires for notification-originated turns, which never
        pushed a timestamp."""
        if not channel.matches_transport("signal"):
            return
        admitted = self._admitted_correlation_ids.get(channel.id)
        if not admitted or correlation_id not in admitted:
            return
        admitted.discard(correlation_id)
        pending = self._pending_timestamps.get(channel.id)
        if not pending:
            return
        timestamp_ms = pending.popleft()
        try:
            await self._send_rpc(
                "sendReceipt", {"recipient": channel.scope, "targetTimestamp": timestamp_ms}
            )
        except Exception as exc:
            logger.warning("signal: failed to send read receipt: %s", exc, exc_info=True)

    # -----------------------------------------------------------------
    # Outbound send
    # -----------------------------------------------------------------

    async def _deliver(self, channel: Channel, text: str) -> None:
        """Split text to the configured chunk size and deliver each chunk
        verbatim -- no newline stripping; Signal bodies are multi-line."""
        chunks = split_message(text, max_len=self._message_chunk_size)
        for chunk in chunks:
            try:
                await self._send_rpc("send", {"recipient": channel.scope, "message": chunk})
            except Exception as exc:
                logger.warning("signal: failed to deliver message: %s", exc, exc_info=True)

    @hookimpl
    async def send_message(self, channel: Channel, text: str, latency_ms: float | None = None) -> None:
        """Deliver a final reply, split to fit and never truncated, and
        close out one outstanding slot."""
        if not channel.matches_transport("signal"):
            return
        await self._deliver(channel, text)
        self._decrement_outstanding(channel)

    @hookimpl
    async def send_progress(self, channel: Channel, text: str) -> None:
        """Deliver intermediate assistant text through the same path as
        send_message, but without the reply bookkeeping -- it is real
        liveness (unlike raw tool traces), fires mid-turn, and does not
        close out the outstanding count."""
        if not channel.matches_transport("signal"):
            return
        await self._deliver(channel, text)
