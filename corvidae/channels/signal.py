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
from datetime import datetime, timezone
from typing import Any, Optional

from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.channels.split import split_message
from corvidae.trace import mint_trace_id
from corvidae.hooks import TransportPlugin, get_dependency, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress

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

# How long resolve_channel waits for an in-flight E.164 alias resolution
# before deferring (returning None so the caller retries). Bounded so a
# signal-cli that accepts the request but never answers cannot stall the
# scheduler's timer pass.
ALIAS_RESOLUTION_WAIT_SECONDS = 5.0

# Connection retry backoff ladder -- the same one IRCClient uses.
_INITIAL_BACKOFF = 10
_BACKOFF_MULTIPLIER = 2
_BACKOFF_CAP = 300


def _format_stale_prefix(timestamp_ms: int) -> str:
    """Render an envelope timestamp as a "[sent YYYY-MM-DD HH:MM UTC]" prefix."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return f"[sent {dt.strftime('%Y-%m-%d %H:%M')} UTC]\n"


class SignalPlugin(TransportPlugin):
    """Transport plugin for a Signal DM conversation via signal-cli's daemon."""

    transport = "signal"

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
        # Set once any resolution attempt has finished, successfully or
        # not. Until then an alias's ACI is simply unknown, and
        # resolve_channel defers rather than cache a channel without the
        # alias's config (issue #76).
        self._alias_resolution_attempted = False

        # Liveness bookkeeping, per channel id (transport:scope).
        self._outstanding: dict[str, int] = {}
        self._typing_tasks: dict[str, asyncio.Task] = {}
        # channel id -> {trace id -> envelope timestamp (ms)} for
        # messages that took a typing slot and are owed a read receipt.
        self._pending_receipts: dict[str, dict[str, int]] = {}
        # Trace ids admitted while their on_message call is still
        # running; _handle_notification's finally consumes them.
        self._admitted: set[str] = set()

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
        # The allowlist is checked before the channel is resolved, so a
        # refused sender never adds an entry to the registry (PRD §3.2.6,
        # issue #103): anyone who knows the number could otherwise grow it
        # by one channel per distinct ACI.
        allowed = await self._is_allowed(source_uuid)
        if allowed:
            channel = await self._resolve_channel(source_uuid, source_number)
        else:
            channel = self._transient_channel(source_uuid)
        text = self._maybe_prefix_stale(text, timestamp)

        # Typing and the read receipt only ever apply to an authorized
        # sender -- an unauthorized one gets no signal the account is live.
        # on_message still fires regardless so the real should_process_message
        # gate can veto and log the attempt.
        # The id is minted here and keys everything this message holds, so
        # the lifecycle hooks can be matched back to it without inference.
        trace_id = mint_trace_id()
        try:
            if allowed and timestamp is not None:
                # Remember the timestamp (for the read receipt) and take a
                # typing slot; both are released below if never admitted.
                self._pending_receipts.setdefault(channel.id, {})[trace_id] = timestamp
                await self._begin_typing(channel)

            await self.pm.ahook.on_message(
                channel=channel, sender=source_uuid, text=text, trace_id=trace_id,
            )
        finally:
            # Admission or rejection fires before on_message returns.
            # Anything not admitted by then (gate veto, agent not ready, an
            # exception, cancellation) gets no reply and no receipt: drop
            # its timestamp and give its typing slot back, but only if it
            # actually took one (a stranger or timestamp-less envelope did not).
            if trace_id not in self._admitted:
                if self._pop_pending(channel, trace_id) is not None:
                    self._decrement_outstanding(channel)
            self._admitted.discard(trace_id)

    def _pop_pending(self, channel: Channel, trace_id: str) -> int | None:
        """Remove and return a message's pending receipt timestamp, or None
        if it has none. Drops the channel's dict once it is empty."""
        pending = self._pending_receipts.get(channel.id)
        if pending is None:
            return None
        timestamp_ms = pending.pop(trace_id, None)
        # Don't let emptied per-channel dicts accumulate.
        if not pending:
            del self._pending_receipts[channel.id]
        return timestamp_ms

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

    def _transient_channel(self, source_uuid: str) -> Channel:
        """The channel a refused sender's message is gated on.

        A live registry entry is reused (a sender dropped from allow:
        keeps the channel it already had); otherwise the Channel is built
        without registering it, so the gate still sees the
        ``signal:<aci>`` id it logs, and nothing outlives the refusal.
        """
        existing = self._registry.get(f"signal:{source_uuid}")
        if existing is not None:
            return existing
        return Channel(transport="signal", scope=source_uuid)

    @hookimpl
    async def resolve_channel(self, channel_id: str) -> Channel | None:
        """Resolve a persisted ``signal:<ACI>`` id to its live channel.

        Routes through _resolve_channel with the ACI as scope and no
        source_number, so a channel first created here (e.g. a reminder
        due after a restart, before the sender has messaged again) gets
        the E.164 alias's ChannelConfig via forward resolution, the same
        as an inbound message under phone-number privacy would. Returns
        None for ids this transport does not own, and also for its own ids
        while that config cannot be applied yet: before on_start has
        resolved the registry, and before the first alias resolution has
        finished (PRD §3.2.7, issue #76). The caller retries; a channel
        created now would keep the default config for the process lifetime.
        """
        # Only ids of the form "signal:<non-empty scope>" belong here.
        transport, sep, scope = channel_id.partition(":")
        if transport != "signal" or not sep or not scope:
            return None
        if self._registry is None:
            return None
        existing = self._registry.get(channel_id)
        if existing is not None:
            return existing
        if not await self._alias_config_ready(channel_id):
            logger.debug(
                "signal: deferring %s until E.164 alias resolution has run", channel_id
            )
            return None
        return await self._resolve_channel(scope, None)

    async def _alias_config_ready(self, channel_id: str) -> bool:
        """Whether every E.164 alias that could configure a channel has
        had its chance to resolve to an ACI.

        Only aliases that key a channels: override AND appear in allow:
        matter -- an alias missing from allow: is never resolved and is
        inert. Waits a bounded time for an in-flight resolution. Once any
        attempt has finished, a failed alias stops blocking: a batch
        failure is retried only on the next connect, and deferring until
        then could hold a reminder for the life of the connection. That
        fallback is logged, since the channel then runs without its
        alias's config.
        """
        alias_channels = {
            c.scope for c in self._registry.by_transport("signal") if c.scope.startswith("+")
        }
        needed = alias_channels & set(self._configured_allow_aliases())

        # Nothing to resolve, or everything already resolved.
        def unresolved() -> list[str]:
            return sorted(a for a in needed if a not in self._resolved_aliases)

        if not unresolved():
            return True

        # A resolution is in flight (we are connected): give it a bounded
        # wait. asyncio.wait never cancels the task it watches.
        task = self._allow_resolution_task
        if not self._alias_resolution_attempted and task is not None and not task.done():
            await asyncio.wait({task}, timeout=ALIAS_RESOLUTION_WAIT_SECONDS)

        pending = unresolved()
        if not pending:
            return True
        if not self._alias_resolution_attempted:
            return False
        # shortcut: a failed alias is treated as unmatched once an attempt
        # has run, upgrade to deferring if reminders on an aliased channel
        # firing with the default config turn out to matter.
        logger.warning(
            "signal: creating %s without E.164 alias config: alias(es) %s did not "
            "resolve; check signal-cli and the allow: entries, then restart to apply",
            channel_id, ", ".join(pending),
        )
        return True

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
        finally:
            # Success or failure, an attempt has now run; see
            # _alias_config_ready for why that matters.
            self._alias_resolution_attempted = True

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
        self, channel: Channel, sender: str, text: str, trace_id: str
    ) -> bool | None:
        """Default-deny gate: only an allowlisted ACI (or its known E.164
        alias) may drive the agent. trace_id is bound without a
        default -- see tests/test_hook_arg_binding.py.
        """
        if not channel.matches_transport("signal"):
            return None
        if await self._is_allowed(sender):
            return True
        logger.info("signal: rejected message from unauthorized sender %s", sender)
        return False

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
            # A loop from an earlier turn may still be asleep; it will see the
            # count back above zero and carry on, so starting a second one
            # would double the refreshes and leave one untracked by on_stop.
            existing = self._typing_tasks.get(channel.id)
            if existing is None or existing.done():
                self._typing_tasks[channel.id] = asyncio.create_task(
                    self._typing_refresh_loop(channel)
                )

    def _decrement_outstanding(self, channel: Channel) -> None:
        """Give back one outstanding slot: a final reply send, an
        error-path send, or a message that was never admitted all close out one
        owed message."""
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
            # Forget this loop only if it is still the tracked one, never a
            # successor registered under the same channel id.
            if self._typing_tasks.get(channel.id) is asyncio.current_task():
                del self._typing_tasks[channel.id]

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

    @hookimpl
    async def on_message_admitted(
        self, channel: Channel, trace_id: str, sender: str, text: str
    ) -> None:
        """Note that this message was admitted, so the inbound handler's
        finally keeps its pending receipt and typing slot (they are released
        by the reply or by on_message_persisted instead). Admission fires
        before on_message returns."""
        if not channel.matches_transport("signal"):
            return
        self._admitted.add(trace_id)

    @hookimpl
    async def on_message_persisted(
        self, channel: Channel, trace_id: str, rowid: int | None, text: str, meta: dict
    ) -> None:
        """Fire the read receipt at processing start, not arrival -- a
        message can sit queued behind contended local inference, and a
        "read" marker followed by silence is worse than no marker. Pops this
        message's own pending timestamp by trace id and sends a receipt
        for it. This hook also fires for notification-originated turns,
        which never had a pending entry, so those are a no-op."""
        if not channel.matches_transport("signal"):
            return
        self._sweep_orphans_before(channel, trace_id)
        timestamp_ms = self._pop_pending(channel, trace_id)
        if timestamp_ms is None:
            return
        try:
            await self._send_rpc(
                "sendReceipt", {"recipient": channel.scope, "targetTimestamp": timestamp_ms}
            )
        except Exception as exc:
            logger.warning("signal: failed to send read receipt: %s", exc, exc_info=True)

    def _sweep_orphans_before(self, channel: Channel, trace_id: str) -> None:
        """Release pending entries admitted before ``trace_id`` that were
        never persisted (issue #104).

        Entries are inserted in the order their messages were enqueued
        (the notification processor is serial), and the channel's agent
        queue processes items one at a time in that order. So when
        ``trace_id`` is persisted, every message admitted before it has
        finished; one whose entry is still here died before persistence
        (conversation init raised, the item was dropped) and will never
        get a receipt or a reply. Its timestamp is dropped without a
        receipt and its typing slot given back, so the indicator does not
        refresh forever. An id this transport never recorded (a
        notification-born turn) sweeps nothing: its place in the order is
        unknown.
        """
        pending = self._pending_receipts.get(channel.id)
        if not pending or trace_id not in pending:
            return
        orphans: list[str] = []
        for pending_id in pending:
            if pending_id == trace_id:
                break
            orphans.append(pending_id)
        for orphan in orphans:
            del pending[orphan]
            self._decrement_outstanding(channel)
            logger.warning(
                "signal: message %s on %s was admitted but never persisted; "
                "dropping its read receipt and typing slot",
                orphan, channel.id,
            )

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

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Deliver a final reply, split to fit and never truncated, and
        close out one outstanding slot."""
        await self._deliver(channel, event.text)
        self._decrement_outstanding(channel)

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        """Deliver intermediate assistant text through the same path as
        send_message, but without the reply bookkeeping -- it is real
        liveness (unlike raw tool traces), fires mid-turn, and does not
        close out the outstanding count."""
        await self._deliver(channel, event.text)
