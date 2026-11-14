"""Signal transport plugin, speaking JSON-RPC to a signal-cli daemon.

Connects to an already-running `signal-cli daemon` over a unix socket
(newline-delimited JSON-RPC 2.0). corvidae never spawns or supervises
signal-cli itself.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from corvidae.attachments import save_attachment, safe_suffix
from corvidae.attribution import get_attribution
from corvidae.channel import Channel, ChannelConfig, ChannelRegistry
from corvidae.channels._registers import shipped_register
from corvidae.channels.split import split_message
from corvidae.context import MessageType
from corvidae.hooks import TransportPlugin, final_was_unsent, get_dependency, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress
from corvidae.trace import mint_trace_id

logger = logging.getLogger("corvidae.signal_plugin")


import re

# Signal attachment types (protobuf enum AttachmentType).
ATT_IMAGE = 1
ATT_AUDIO = 2
ATT_CONTACT = 3
ATT_VIDEO = 4
ATT_DOCUMENT = 5
ATT_GIF = 6
ATT_STICKER = 7

# MIME-based download gating. signal-cli does not reliably send the
# protobuf `type` enum; `contentType` is the primary signal.
# Prefixes cover whole families; exact types are the application/*
# allowlist. image/svg+xml and text/html are excluded: they can carry
# scripts and the agent has no safe renderer for them.
_DOWNLOADABLE_MIME_PREFIXES = ("image/", "text/", "audio/")
_DOWNLOADABLE_MIME_EXACT = frozenset({
    "application/pdf",
    "application/json",
    "application/xml",
})
# Excluded from image/* and text/* prefixes (script-capable).
_MIME_EXCLUDED = frozenset({
    "image/svg+xml",
    "text/html",
})

# MIME → file extension mapping (for the extension override).
_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/avif": ".avif",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "text/xml": ".xml",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "application/xml": ".xml",
}

# Sanitize a sender-controlled subtype for use in agent-visible notes.
# Restrict to [a-z0-9.+-], cap length, fallback to "attachment" if empty.
_SAFE_SUBTYPE = re.compile(r"[^a-z0-9.+-]")


def _file_data_uri(path: Path) -> str:
    # signal-cli needs a media type in the data URI; guess it from the name,
    # as it once did from the file, so images do not arrive as generic files.
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};filename={path.name};base64,{payload}"


def _mime_base(content_type: str) -> str:
    """Strip MIME parameters: 'application/json; charset=utf-8' -> 'application/json'."""
    return content_type.split(";", 1)[0].strip().lower()


def _is_downloadable(mime_base: str, att_type: int) -> bool:
    """MIME-based download gate.

    Primary: contentType (MIME type, parameters stripped).
    Fallback: legacy protobuf type enum (image/GIF only) for older
    signal-cli versions that do not send contentType.
    """
    if not mime_base:
        return att_type in (ATT_IMAGE, ATT_GIF)
    if mime_base in _MIME_EXCLUDED:
        return False
    if mime_base in _DOWNLOADABLE_MIME_EXACT:
        return True
    return any(mime_base.startswith(p) for p in _DOWNLOADABLE_MIME_PREFIXES)


def _sanitize_mime_part(type_name: str) -> str:
    """Sanitize a sender-controlled MIME name part (major or subtype)
    for agent-visible notes."""
    return _SAFE_SUBTYPE.sub("", type_name)[:32] or "attachment"


def _safe_id(raw_id: str) -> str | None:
    """Restrict an attachment id to [A-Za-z0-9._-] for filename use.

    Returns None if the id contains disallowed characters (refuse download).
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", raw_id):
        return None
    return raw_id

ATTACHMENT_TYPE_NAMES = {
    ATT_IMAGE: "image",
    ATT_AUDIO: "audio",
    ATT_CONTACT: "contact",
    ATT_VIDEO: "video",
    ATT_DOCUMENT: "document",
    ATT_GIF: "gif",
    ATT_STICKER: "sticker",
}

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

# How long a liveness RPC (typing, read receipt) waits for signal-cli's
# response. A signal-cli that accepts the frame but never answers would
# otherwise hold the serial inbound processor -- and, in send_message, the
# reply -- for as long as the daemon stays wedged. Deliberately not applied
# to `send`: a send that times out may have been delivered, and giving up on
# it would change delivery semantics rather than only liveness (#413).
LIVENESS_RPC_TIMEOUT = 10.0
# Sizing note: LIVENESS_RPC_TIMEOUT is 10s because it covers typing
# indicators and read receipts, which must be snappy. An attachment
# download is a different beast: a 10 MB image over a slow mobile link
# can legitimately exceed 10s, and the user gets [image: download failed]
# for a message that would have arrived. 60s is generous without
# imposing a meaningful head-of-line delay on messages queued behind it.
ATTACHMENT_DOWNLOAD_TIMEOUT = 60.0

# Connection retry backoff ladder -- the same one IRCClient uses.
_INITIAL_BACKOFF = 10
_BACKOFF_MULTIPLIER = 2
_BACKOFF_CAP = 300


def _format_stale_prefix(timestamp_ms: int) -> str:
    """Render an envelope timestamp as a "[sent YYYY-MM-DD HH:MM UTC]" prefix."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return f"[sent {dt.strftime('%Y-%m-%d %H:%M')} UTC]\n"


# Reactions: the trigger map holds this many recent messages, the oldest evicted.
TRIGGER_MAP_SIZE = 256
# A channel holds at most this many reaction lines waiting for its next turn.
MAX_PENDING_REACTIONS = 20
# The tool's local emoji check: signal-cli judges a single grapheme cluster,
# so this only bounds what is sent (code points, not bytes).
MAX_REACT_EMOJI_CODE_POINTS = 16
# A target timestamp must be below year 9999, in milliseconds.
_MAX_TARGET_TS_MS = 253402300800000

# A reaction's emoji is judged against a real emoji table, not a grammar
# (#780). Two hand-written grammars were each found to let something through:
# tag characters in a subdivision flag carry hidden ASCII, and ZWJ-joined
# enclosed letters or regional-indicator pairs spell words. The transport is
# in the kernel like Discord's, but the table is its own extra (`signal`), so
# the import is guarded and a configured transport refuses to start without
# it — the kernel job installs no extras and must still import this module.
try:  # pragma: no cover - the kernel job takes the ImportError branch
    import emoji as _emoji
except ImportError:  # pragma: no cover
    _emoji = None

_INSTALL_HINT = (
    "uv sync --extra signal, naming the other extras you use as well "
    "(e.g. --extra dev --extra acp --extra signal): uv sync removes extras "
    "you don't name"
)


def emoji_table_missing() -> bool:
    """Whether the emoji table the reaction check needs failed to import."""
    return _emoji is None


def _is_single_emoji(value: str) -> bool:
    """Whether ``value`` is exactly one emoji, per the RGI table (#780).

    A flag (``🇯🇵``), a keycap (``1️⃣``) and a ZWJ sequence (``❤️‍🔥``,
    ``🤷‍♀️``, ``🏳️‍🌈``, the subdivision flags) pass; a tag sequence with
    hidden ASCII, enclosed letters joined by ZWJ, a regional-indicator word, a
    non-RGI ZWJ join or a trailing newline do not.

    One normalisation precedes the lookup: a single trailing U+FE0F (VS16) is
    stripped when what precedes it is itself a table entry, so a base or a
    base plus skin tone followed by VS16 (``👍️``, ``👍🏽️``) is accepted as
    it was before the table. Nothing else is normalised: a string ending in
    two VS16, or a VS16 after text, still fails.

    ``emoji.is_emoji`` is what the 2.x series makes it: the *whole string* is
    one emoji, so this is a membership test rather than a scan.
    ``tests/test_signal_reaction_events.py::test_the_table_is_what_decides``
    pins that semantics, so a version bump that changed it fails there instead
    of quietly widening what reaches the prompt.
    """
    if _emoji is None:
        return False
    # Strip one trailing VS16 only when the remainder is itself a table entry.
    if value.endswith("\ufe0f") and _emoji.is_emoji(value[:-1]):
        return True
    return _emoji.is_emoji(value)


def _format_reaction_time(timestamp_ms: int) -> str:
    """Render a ms epoch as ``YYYY-MM-DD HH:MM UTC``."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


class SignalPlugin(TransportPlugin):
    """Transport plugin for a Signal DM conversation via signal-cli's daemon."""

    transport = "signal"
    # A chat transport: a person is on the other end who may end the exchange.
    conversational = True

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
        self._attachment_dir: Path = Path("state/signal-attachments")
        self._max_attachment_size: int = 10 * 1024 * 1024
        # The only directory tree send_signal_attachment may read from (#544).
        self._send_attachment_root: Path = Path("state")

        # Connection state.
        self._connect_task: Optional[asyncio.Task] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._next_request_id = 1
        self._rpc_timeout = 30.0
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

        # Trigger map: trace id -> (channel id, author ACI, envelope timestamp
        # ms) for messages from allowed senders, so a reaction tool can find the
        # message that started its turn. Bounded, oldest evicted; in memory
        # only, so a restart forgets it and the tool reports that.
        self._triggers: OrderedDict[str, tuple[str, str, int]] = OrderedDict()
        # channel id -> reaction lines waiting for that channel's next turn.
        # shortcut: lost on a crash, upgrade if acknowledgements prove worth a WAL.
        self._pending_reactions: dict[str, list[str]] = {}

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

        # A configured transport needs the emoji table for its reaction check
        # (#780). Refused here, at startup, rather than dropping every
        # reaction at run time: silence is for absence, not for a missing
        # install. Bound once at import, so installing the package needs a restart.
        if emoji_table_missing():
            raise RuntimeError(
                "SignalPlugin: agent.yaml has a signal: block, but the 'emoji' "
                f"package its reaction check needs is not installed. Install it "
                f"with {_INSTALL_HINT}."
            )

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
        attachment_dir = signal_config.get("attachment_dir", "state/signal-attachments")
        self._attachment_dir = Path(attachment_dir)
        max_size = signal_config.get("max_attachment_size", 10 * 1024 * 1024)
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
            raise ValueError(
                f"signal: max_attachment_size must be a positive integer, "
                f"got {max_size!r}"
            )
        self._max_attachment_size = max_size
        # The send root must be a non-empty string. An explicit null is a
        # stated value, not an omission, so it is rejected rather than read
        # as "use the default" (an absent key does mean the default).
        send_root = signal_config.get("send_attachment_root", "state")
        if not isinstance(send_root, str) or not send_root:
            raise ValueError(
                f"signal: send_attachment_root must be a non-empty string, "
                f"got {send_root!r}"
            )
        self._send_attachment_root = Path(send_root)

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
                # StreamReader limit sized to the max attachment: base64
                # inflates by ~4/3, plus 64 KiB headroom for JSON framing.
                # A frame that still exceeds it is dropped by readline(),
                # and _read_loop fails the in-flight RPCs (no 60 s hang).
                # Integer arithmetic (no float precision loss).
                stream_limit = (self._max_attachment_size * 4 + 2) // 3 + 64 * 1024
                reader, writer = await asyncio.open_unix_connection(
                    path=self._socket_path, limit=stream_limit,
                )
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
            try:
                line = await reader.readline()
            except ValueError as exc:
                # StreamReader buffer limit exceeded -- an oversized frame
                # (typically a base64-encoded attachment). The connection
                # stays alive; the in-flight RPCs are failed below rather
                # than left to time out (see the threat model for the
                # buffer-clearing/tail-frame mechanics).
                logger.warning(
                    "signal: frame exceeded the StreamReader limit (%s); "
                    "discarded and in-flight requests failed. If this was "
                    "a legitimate attachment, raise "
                    "signal.max_attachment_size.",
                    exc,
                )
                self._reject_pending_requests(
                    "signal-cli response exceeded the stream limit"
                )
                continue
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

    def _reject_pending_requests(
        self, message: str = "signal-cli connection lost"
    ) -> None:
        """Fail any outstanding request rather than hang it."""
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(ConnectionError(message))
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
        try:
            return await asyncio.wait_for(future, timeout=self._rpc_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"signal-cli RPC {method!r} timed out after {self._rpc_timeout:.0f} s; "
                "the daemon may be wedged"
            ) from None
        finally:
            # A caller that bounds this await (the liveness RPCs, #413) cancels
            # it on timeout. The response may never come, so drop the entry
            # here rather than leaking one per wedged call; the pop is
            # idempotent with the one the response path already did.
            self._pending_requests.pop(req_id, None)

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

        # Reaction events (R8) are handled and end here; a null or absent
        # reaction falls through to Rule 4.
        if data_message.get("reaction") is not None:
            await self._handle_reaction(envelope, data_message["reaction"])
            return

        # Rule 4: text must be non-empty, or there must be a downloadable
        # attachment. Quotes, edits, and remote deletes fall out here since
        # none of them carry a "message" field.
        attachments = data_message.get("attachments") or []
        text = data_message.get("message") or ""

        if not text and not attachments:
            logger.debug("signal: ignoring envelope with no content")
            return

        source_uuid = envelope.get("sourceUuid")
        if not source_uuid:
            logger.warning("signal: envelope has no sourceUuid, cannot scope a channel, dropping")
            return

        timestamp = envelope.get("timestamp", data_message.get("timestamp"))
        # The allowlist is checked before the channel is resolved AND before
        # any attachment download, so a refused sender never adds an entry
        # to the registry (PRD §3.2.6, issue #103) and never triggers a
        # disk write: anyone who knows the number could otherwise grow the
        # registry by one channel per distinct ACI and fill the disk with
        # unsolicited files.
        allowed = await self._is_allowed(source_uuid)
        if allowed:
            channel = await self._resolve_channel(source_uuid, source_number)
        else:
            channel = self._transient_channel(source_uuid)

        # Surface attachments: download images to disk for allowed senders,
        # note metadata only for everyone else.
        attachment_notes: list[str] = []
        for att in attachments:
            mime_base = _mime_base(att.get("contentType") or "")
            att_type = att.get("type", 0)
            downloadable = _is_downloadable(mime_base, att_type) and allowed
            # Label for agent-visible notes: downloaded attachments use
            # the major type (image, text, audio) or the subtype for
            # application/* (pdf, json, xml); not-downloaded ones use the
            # sanitized subtype (svg, html, zip, ...) so the agent can see
            # why nothing was fetched. Falls back to the protobuf type
            # name if no MIME type.
            if mime_base and "/" in mime_base:
                major, subtype = mime_base.split("/", 1)
                if downloadable and major != "application":
                    type_name = _sanitize_mime_part(major)
                else:
                    type_name = _sanitize_mime_part(subtype)
            else:
                type_name = ATTACHMENT_TYPE_NAMES.get(att_type, f"type{att_type}")
            if downloadable:
                path = await self._download_attachment(envelope, att, mime_base)
                if path is not None:
                    attachment_notes.append(f"[{type_name}: {path}]")
                else:
                    attachment_notes.append(f"[{type_name}: download failed]")
            else:
                attachment_notes.append(f"[{type_name} attached]")

        if attachment_notes:
            prefix = "\n".join(attachment_notes)
            text = f"{prefix}\n{text}" if text else prefix

        text = self._maybe_prefix_stale(text, timestamp)

        # Typing and the read receipt only ever apply to an authorized
        # sender -- an unauthorized one gets no signal the account is live.
        # on_message still fires regardless so the real should_process_message
        # gate can veto and log the attempt.
        # The id is minted here and keys everything this message holds, so
        # the lifecycle hooks can be matched back to it without inference.
        trace_id = mint_trace_id()
        # Remember which message started this turn, for the reaction tools:
        # only an allowed sender's, and only with an int timestamp (a float
        # or missing one cannot name a message).
        if allowed and type(timestamp) is int:
            self._triggers[trace_id] = (channel.id, source_uuid, timestamp)
            while len(self._triggers) > TRIGGER_MAP_SIZE:
                self._triggers.popitem(last=False)
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
                    await self._decrement_outstanding(channel)
            self._admitted.discard(trace_id)


    async def _handle_reaction(self, envelope: dict, reaction: Any) -> None:
        """Queue an allowed sender's emoji reaction as a context line for
        their channel's next turn. Starts no turn; every drop is logged."""
        # A reaction cannot be scoped to a channel without a sender.
        source_uuid = envelope.get("sourceUuid")
        if not source_uuid:
            logger.warning(
                "signal: reaction envelope has no sourceUuid, cannot scope a channel, dropping"
            )
            return

        # Same allowlist predicate as the text path, before any state is
        # touched: a refused sender gets one INFO line and no channel.
        if not await self._is_allowed(source_uuid):
            logger.info("signal: ignoring reaction from unauthorized sender %s", source_uuid)
            return

        # A value of the wrong type is dropped; show it cut to 32 characters
        # first, then repr'd, so an escape is never split.
        if not isinstance(reaction, dict):
            logger.warning(
                "signal: dropped reaction from %s: not an object, got %r",
                source_uuid, str(reaction)[:32],
            )
            return

        # A removal is noise by decision (design: Deferred).
        if reaction.get("isRemove") is True:
            logger.debug("signal: dropping reaction removal from %s", source_uuid)
            return

        # The emoji is sender-controlled and enters the prompt: validate it.
        emoji = reaction.get("emoji")
        if not isinstance(emoji, str) or not _is_single_emoji(emoji):
            logger.warning(
                "signal: dropped reaction from %s: %r is not a single emoji "
                "(a sequence carrying text or a non-emoji ZWJ join is not accepted)",
                source_uuid, str(emoji)[:32],
            )
            return

        # The target's time must be a plausible int (type check so True fails).
        target_ts = reaction.get("targetSentTimestamp")
        if type(target_ts) is not int or not 0 < target_ts < _MAX_TARGET_TS_MS:
            logger.warning(
                "signal: dropped reaction from %s: bad targetSentTimestamp %r",
                source_uuid, str(target_ts)[:32],
            )
            return

        # Whose message: "your" when the target author is known and is not
        # the sender; the legacy targetAuthor (maybe a number) is not used.
        author = reaction.get("targetAuthorUuid")
        if not author:
            whose = "a"
        elif author == source_uuid:
            whose = "their own"
        else:
            whose = "your"
        line = f"[reaction: {emoji} on {whose} message from {_format_reaction_time(target_ts)}]"

        # Resolve the channel as the text path does for an allowed sender.
        channel = await self._resolve_channel(source_uuid, envelope.get("sourceNumber"))
        pending = self._pending_reactions.setdefault(channel.id, [])
        if len(pending) >= MAX_PENDING_REACTIONS:
            pending.pop(0)
            logger.warning(
                "signal: %s has %d reactions waiting for its next turn; dropped the oldest",
                channel.id, MAX_PENDING_REACTIONS,
            )
        pending.append(line)

    @hookimpl
    async def before_agent_turn(
        self, channel: Channel, trace_id: str | None, annotations: dict
    ) -> None:
        """Append this channel's waiting reaction lines as CONTEXT rows."""
        # Another transport's channel is none of ours (checklist trap 11).
        if not self.owns(channel):
            return
        # A channel with no window yet has nowhere to put a row; keep the lines.
        conv = channel.conversation
        if conv is None:
            return
        # Nothing waiting for this channel: the common case, so exit early.
        pending = self._pending_reactions.get(channel.id)
        if not pending:
            return
        # Append each row before removing it, so a failure loses nothing.
        # shortcut: the row bypasses the cognition trust quarantine and has
        # no kernel provenance header (PRD 3.3.11); safe only because the
        # accepted emoji is one table entry and the rest of the row is built here.
        # The allowlist check is the transport's own, not should_process_message.
        # The Agent persists rows appended here that lack a _db_id.
        while pending:
            conv.append({"role": "system", "content": pending[0]}, MessageType.CONTEXT)
            pending.pop(0)
        del self._pending_reactions[channel.id]

    async def _download_attachment(
        self,
        envelope: dict,
        attachment: dict,
        mime_base: str,
    ) -> Path | None:
        """Download a signal-cli attachment to disk via getAttachment RPC.

        Returns the path on success, None on any failure. The caller
        (which has already passed the allowlist check) surfaces the
        result as a note in the message text.

        ``mime_base`` is the parameter-stripped, lowercased contentType,
        parsed once in _handle_notification and passed down.
        """
        raw_id = attachment.get("id")
        if not raw_id:
            logger.warning(
                "signal: attachment has no id, refusing download; "
                "attachment keys: %s", sorted(attachment.keys()),
            )
            return None

        # Sanitize the id for filename use. Refuse if it contains
        # disallowed characters (newline, bracket, slash, etc.).
        safe_id = _safe_id(str(raw_id))
        if safe_id is None:
            logger.warning(
                "signal: attachment id contains disallowed characters; "
                "refusing download: %r", str(raw_id)[:80],
            )
            return None

        params = {"id": str(raw_id)}
        try:
            result = await asyncio.wait_for(
                self._send_rpc("getAttachment", params),
                ATTACHMENT_DOWNLOAD_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("signal: getAttachment RPC timed out after %ss", ATTACHMENT_DOWNLOAD_TIMEOUT)
            return None
        except Exception:
            logger.exception("signal: getAttachment RPC failed")
            return None

        if not result or "data" not in result:
            logger.warning("signal: getAttachment response has no data")
            return None

        # Decode off the event loop: base64 on a line up to max_attachment_size * 4/3 would
        # block other channels.
        try:
            raw = await asyncio.to_thread(
                lambda: base64.b64decode(result["data"], validate=True),
            )
        except Exception:
            logger.exception("signal: attachment data is not valid base64")
            return None

        # Extension: MIME map first, else the filename's safe extension. An
        # id that already ends in an extension keeps it through the stem.
        # The MIME type is sender-controlled — a labelling convenience,
        # not a security boundary. No content sniffing is done.
        suffix = _MIME_TO_EXT.get(mime_base) or safe_suffix(attachment.get("filename"))

        ts = envelope.get("timestamp", 0)
        att_type = attachment.get("type", 0)
        filename = f"{ts}-{att_type}-{safe_id}"

        # save_attachment enforces the size cap and keeps the path inside
        # the attachment dir.
        path = await save_attachment(
            raw,
            self._attachment_dir,
            filename,
            suffix=suffix,
            max_bytes=self._max_attachment_size,
            transport="signal",
        )
        return path

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
    async def resolve_register_prompt(self, channel: Channel) -> str | None:
        """Default register for Signal channels (one-on-one chats): chat
        norms plus plain text, since Signal shows Markdown literally."""
        if not self.owns(channel):
            return None
        return shipped_register("chat", "signal_dm")

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

    async def _decrement_outstanding(self, channel: Channel) -> None:
        """Give back one outstanding slot: a message that was never admitted,
        or one the orphan sweep found dead, closes out one owed message (a
        final, sent or unsent, releases every slot instead; see
        _release_all). The slot that takes the count to zero also clears the
        indicator explicitly, rather than leaving it to expire on the client
        (issue #390). A count already at zero is left alone and sends
        nothing."""
        count = self._outstanding.get(channel.id, 0)
        if count > 0:
            # Decrement before awaiting the stop, so a message arriving while
            # the stop is in flight sees 0 -> 1 and its start goes out after it.
            self._outstanding[channel.id] = count - 1
            if count == 1:
                await self._send_typing(channel, stop=True)

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

    async def _send_typing(self, channel: Channel, stop: bool = False) -> bool:
        """Send one typing-indicator request. A failure is logged and never
        blocks the reply; returns whether the send succeeded, so the
        refresh loop can stop retrying against a channel that keeps
        failing."""
        try:
            # sendTyping is one method with a boolean "stop" -- not a
            # separate stop method or an action string. False starts (or
            # refreshes) the indicator; True clears it.
            # Bounded (#413): a wedged daemon must not hold the inbound
            # processor, which sends the start frame inline.
            await asyncio.wait_for(
                self._send_rpc("sendTyping", {"recipient": channel.scope, "stop": stop}),
                LIVENESS_RPC_TIMEOUT,
            )
            return True
        except Exception as exc:
            # Name the channel and what was being sent, and the remedy: the
            # indicator is cosmetic, so only a persistent failure needs action.
            action = "stop" if stop else "start or refresh"
            logger.warning(
                "signal: failed to %s typing indicator on %s (%s): %s; "
                "the reply is unaffected -- if this repeats, check that "
                "signal-cli is running and its socket is reachable",
                action, channel.id, type(exc).__name__, exc, exc_info=True,
            )
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
        self, channel: Channel, trace_id: str, rowid: int | None, text: str, annotations: dict
    ) -> None:
        """Fire the read receipt at processing start, not arrival -- a
        message can sit queued behind contended local inference, and a
        "read" marker followed by silence is worse than no marker. Pops this
        message's own pending timestamp by trace id and sends a receipt
        for it. This hook also fires for notification-originated turns,
        which never had a pending entry, so those are a no-op."""
        if not channel.matches_transport("signal"):
            return
        await self._sweep_orphans_before(channel, trace_id)
        timestamp_ms = self._pop_pending(channel, trace_id)
        if timestamp_ms is None:
            return
        try:
            # Bounded (#413): on_message_persisted runs on the serial inbound
            # processor, so an unanswered receipt would hold the next message.
            await asyncio.wait_for(
                self._send_rpc(
                    "sendReceipt", {"recipient": channel.scope, "targetTimestamp": timestamp_ms}
                ),
                LIVENESS_RPC_TIMEOUT,
            )
        except Exception as exc:
            # Names the channel and the remedy, as the typing rescue does: the
            # marker is cosmetic, only a persistent failure needs action.
            logger.warning(
                "signal: failed to send read receipt on %s (%s): %s; "
                "the reply is unaffected -- if this repeats, check that "
                "signal-cli is running and its socket is reachable",
                channel.id, type(exc).__name__, exc, exc_info=True,
            )

    async def _sweep_orphans_before(self, channel: Channel, trace_id: str) -> None:
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
            await self._decrement_outstanding(channel)
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

    @hookimpl
    async def on_agent_response(
        self,
        channel,
        request_text: str,
        response_text: str,
        trace_id: str | None,
        annotations: dict,
        logprobs: dict | None,
    ) -> None:
        """Release per-turn state for an unsent final (it never reaches
        send_message); a sent final is released there, so exactly once."""
        if not self.owns(channel):
            return
        # An unsent final never reaches send_message, so release here. The
        # slots are given back in _release_all's synchronous prefix, before
        # its first await (the typing stop), so the check and the release
        # still happen before this hookimpl yields.
        try:
            unsent = final_was_unsent(channel)
        except Exception:
            logger.warning(
                "%s: could not tell whether the final on %s was left unsent; "
                "releasing nothing (the turn may stay open until the next final)",
                type(self).__name__, channel.id, exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return
        if unsent:
            # An unsent final releases what a sent one would: every slot.
            await self._release_all(channel)

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Deliver a final reply, split to fit and never truncated, and
        close out every outstanding slot: one reply answers a coalesced burst
        (#398), so the indicator stops once, when the count was above zero."""
        await self._deliver(channel, event.text)
        await self._release_all(channel)

    async def _release_all(self, channel: Channel) -> None:
        """Close out every outstanding slot on a final, sent or unsent: one
        final answers a coalesced burst (#398). When the count was above zero
        the indicator is cleared explicitly, once (#390); at zero nothing is
        sent."""
        # Zero the count before awaiting the stop, as _decrement_outstanding
        # does, so a message arriving meanwhile sees 0 -> 1 and the release
        # happens in this coroutine's synchronous prefix.
        if self._outstanding.get(channel.id, 0) > 0:
            self._outstanding[channel.id] = 0
            await self._send_typing(channel, stop=True)

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        """Deliver intermediate assistant text through the same path as
        send_message, but without the reply bookkeeping -- it is real
        liveness (unlike raw tool traces), fires mid-turn, and does not
        close out the outstanding count."""
        await self._deliver(channel, event.text)

    # -----------------------------------------------------------------
    # Outbound files (#544)
    # -----------------------------------------------------------------

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Offer the agent one way to send a file to a Signal contact.

        Only when a ``signal:`` block configured the transport: an inert
        plugin has no daemon to send through. The tool is named
        ``send_signal_attachment`` because the tool registry keeps the last of
        any two tools sharing a name, and Discord's tool already holds
        ``send_attachment``.
        """
        if self._socket_path is not None:
            tool_registry.append(self.send_signal_attachment)
            tool_registry.append(self.signal_react)
            tool_registry.append(self.signal_unreact)

    async def send_signal_attachment(
        self, channel_id: str, path: str, caption: str = "",
        voice_note: bool = False,
    ) -> str:
        """Send a file under signal.send_attachment_root to a Signal contact, with an optional caption.

        Args:
            channel_id: The target contact in "signal:<id>" form. Only a
                contact who has messaged this account since it started is
                known.
            path: The file to send, relative to signal.send_attachment_root
                (or an absolute path inside it). Paths outside that root are
                refused.
            caption: Optional text to send with the file. Pass "" for none.
            voice_note: If True, send as a voice note bubble instead of a
                regular file attachment. Use for audio files (WAV, OGG, MP3).

        Returns:
            A confirmation naming the file and channel, or an error saying
            what was wrong and what to do next.
        """
        # Check the arguments first, as the Discord tool does: a model can
        # pass anything, and each refusal names the argument.
        if not isinstance(channel_id, str) or not channel_id:
            return "Could not send: channel_id must be a non-empty string."
        if not isinstance(path, str) or not path.strip():
            return "Could not send: path must be a non-empty string."
        if caption is None:
            caption = ""
        if not isinstance(caption, str):
            return "Could not send: caption must be a string."
        if not isinstance(voice_note, bool):
            return "Could not send: voice_note must be a boolean."
        if self._registry is None:
            return (
                f"Could not send: the channel registry is not available for "
                f"{channel_id!r}; retry once corvidae has finished starting."
            )

        # Refuse a foreign id by its transport prefix before any registry
        # lookup, so an id that is not in the registry still gets this error.
        if channel_id.partition(":")[0] != self.transport:
            return (
                f"Could not send: {channel_id!r} is not a Signal channel; "
                f"the target must be 'signal:<id>'."
            )
        channel = self._registry.get(channel_id)
        if channel is None:
            return (
                f"Could not send: {channel_id!r} is not a known Signal "
                f"channel; an allowlisted contact becomes known when they "
                f"message this account (see signal.allow)."
            )

        # Resolve the path under the root, jail it, and read its size. The
        # jail runs before any existence check so a missing path outside the
        # root gets the root error. A path that cannot be resolved or stat'ed
        # (a NUL byte, a permission error) is the same kind of failure.
        try:
            root = self._send_attachment_root.resolve()
            raw = Path(path)
            target = (raw if raw.is_absolute() else root / raw).resolve()
            if not target.is_relative_to(root):
                return (
                    f"Could not send: {path!r} is outside the allowed root "
                    f"{root}; put the file there or set "
                    f"signal.send_attachment_root."
                )
            if not target.is_file():
                return (
                    f"Could not send: {target} is not a file; check it "
                    f"exists under {root}."
                )
            size = target.stat().st_size
            # The inbound size cap bounds outbound files too (R3): one
            # setting, both directions. A file exactly at the cap is sent.
            if size > self._max_attachment_size:
                return (
                    f"Could not send: {target.name} is {size} bytes, over the "
                    f"{self._max_attachment_size} byte limit "
                    f"(signal.max_attachment_size)."
                )
            # The daemon gets the bytes as a data URI, not a path it would
            # have to read itself. Read and encode off the loop: the string
            # can be megabytes.
            uri = await asyncio.to_thread(_file_data_uri, target)
        except (OSError, ValueError) as exc:
            logger.warning(
                "signal: cannot read %r for send_signal_attachment (%s)",
                path, type(exc).__name__, exc_info=True,
            )
            return (
                f"Could not send: cannot read {path!r} ({type(exc).__name__}); "
                f"check the path and its permissions."
            )

        # The caption rides with the file; one longer than a message is split
        # so nothing is cut, and blank pieces are never sent.
        chunks = [
            c for c in split_message(caption, max_len=self._message_chunk_size)
            if c.strip()
        ]

        # The file frame carries the first chunk as its message. Later chunks
        # follow as text frames through _send_rpc, not _deliver, which would
        # swallow a failure and let this report success for a lost caption.
        params: dict[str, Any] = {
            "recipient": channel.scope, "attachments": [uri],
        }
        if voice_note:
            params["voiceNote"] = True
        if chunks:
            params["message"] = chunks[0]
        # file_sent tells the handler how far the send got; part counts the
        # caption chunk in flight, 1 being the one riding with the file (it is
        # bound here so the handler can always read it).
        file_sent = False
        part = 1
        # The reply's per-recipient "results" is not inspected (#608): every
        # send here names one recipient, and signal-cli (SendMessageResultUtils
        # .outputResult, v0.14.7) answers a send with no successful recipient
        # with a JSON-RPC error, which _send_rpc raises. A result whose
        # entries mix success and failure needs several recipients, i.e. a
        # group, which this transport does not send to.
        try:
            await self._send_rpc("send", params)
            file_sent = True
            for part, chunk in enumerate(chunks[1:], start=2):
                await self._send_rpc(
                    "send", {"recipient": channel.scope, "message": chunk},
                )
        except Exception as exc:
            logger.warning(
                "signal: send_signal_attachment failed for %s to %s (%s)",
                target.name, channel_id, type(exc).__name__, exc_info=True,
            )
            # Say how far the send got, so the remedy is the right one: a
            # failure after the file went must not invite sending it again.
            if file_sent:
                return (
                    f"Could not send: {target.name} reached {channel_id} but "
                    f"caption part {part} of {len(chunks)} failed "
                    f"({type(exc).__name__}); do not resend the file, send "
                    f"the rest of the caption as a message."
                )
            if isinstance(exc, ConnectionError):
                return (
                    f"Could not send: signal-cli is not connected "
                    f"({target.name} to {channel_id}); retry once it is."
                )
            return (
                f"Could not send: {target.name} to {channel_id} failed "
                f"({type(exc).__name__}); nothing was delivered, retry."
            )
        return f"Sent {target.name} to {channel_id}."

    # -----------------------------------------------------------------
    # Reactions (R2-R5)
    # -----------------------------------------------------------------

    async def signal_react(self, emoji: str) -> str:
        """React with an emoji to the message that started this turn.

        Args:
            emoji: One emoji, such as a thumbs-up.

        Returns:
            A confirmation, or an error saying what was wrong and what to do next.
        """
        return await self._send_reaction(emoji, remove=False)

    async def signal_unreact(self, emoji: str) -> str:
        """Remove your emoji reaction from the message that started this turn.

        Args:
            emoji: The emoji you reacted with.

        Returns:
            A confirmation, or an error saying what was wrong and what to do next.
        """
        return await self._send_reaction(emoji, remove=True)

    async def _send_reaction(self, emoji: str, remove: bool) -> str:
        """Send a sendReaction RPC aimed at the message that triggered the
        current turn; every failure is returned as text, never raised."""
        verb = "remove reaction" if remove else "react"
        # Local emoji check: the daemon judges the rest.
        if (
            not isinstance(emoji, str)
            or not emoji
            or len(emoji) > MAX_REACT_EMOJI_CODE_POINTS
        ):
            return (
                f"Could not {verb}: {emoji!r} is not a usable emoji (it must be "
                f"1 to {MAX_REACT_EMOJI_CODE_POINTS} characters); try a single "
                f"emoji, or reply in text."
            )

        # The target comes from the turn's provenance, never an argument.
        attribution = get_attribution()
        trace_id = attribution.get("trace_id")
        entry = self._triggers.get(trace_id) if trace_id else None
        if entry is None or entry[0] != attribution.get("channel_id"):
            return (
                "Could not {verb}: there is no message to react to. This turn "
                "either did not start from a message, or the transport has "
                "forgotten it; send a text reply instead.".format(verb=verb)
            )
        channel_id, author, timestamp = entry

        # The sendReaction frame; the recipient is the sender's ACI, the channel
        # scope, and the target is the message that started this turn.
        params: dict[str, Any] = {
            "recipient": channel_id.partition(":")[2],
            "emoji": emoji,
            "targetAuthor": author,
            "targetTimestamp": timestamp,
        }
        # Removing a reaction is the same call with remove set.
        if remove:
            params["remove"] = True
        # The daemon may be down or reject the call; report it, never raise.
        try:
            await self._send_rpc("sendReaction", params)
        except Exception as exc:
            logger.warning(
                "signal: sendReaction (%s) failed on %s (%s)",
                verb, channel_id, type(exc).__name__, exc_info=True,
            )
            return (
                f"Could not {verb} on {channel_id}: {exc}; retry, or reply in text."
            )
        return f"{'Removed' if remove else 'Added'} reaction {emoji} on the message that started this turn."
