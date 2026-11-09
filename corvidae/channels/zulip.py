"""Zulip transport plugin, built on Zulip's REST event-queue API (#640).

Connects to a Zulip organisation as a bot user and talks in direct messages,
group direct messages and channels. In channels (and group DMs) the bot by
default only reacts when it is mentioned; a 1:1 DM always reaches it.
Only allowlisted Zulip user ids may drive the agent.

There is no SDK: the transport speaks five endpoints over aiohttp, which the
kernel already depends on (``GET /users/me``, ``POST /register``,
``GET /events``, ``POST /messages``, ``POST /typing``), so there is no extra
to install.

Scopes are ids, never names (PRD principle 10): ``zulip:<stream_id>`` for a
channel, ``zulip:dm:<user id>`` for a 1:1 DM, ``zulip:dm:<id>,<id>`` (sorted,
the bot excluded) for a group DM. A Zulip *topic* is a name users can rename,
so it is deliberately not part of any scope: it is shown to the model as part
of the channel's label (``#general > plans``) and it is where the reply goes
(the topic of the newest admitted message, kept in memory only; with none
remembered, ``zulip.default_topic``).

Needs Zulip 9.0 (feature level 248) or newer, the first server that accepts the
modern ``channel`` and ``direct`` message types; an older server is refused at
connect time with a message saying so.

A lost event queue (the server garbage-collected it, or the process restarted)
is re-registered; messages sent while it was gone are *not* replayed, and the
WARNING says so (#640 D4).
"""

from __future__ import annotations

import asyncio
import base64
import collections
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

import aiohttp

from corvidae.channel import (
    Channel,
    ChannelRegistry,
    remember_sender_name,
)
from corvidae.channels._registers import shipped_register
from corvidae.channels.split import split_message
from corvidae.hooks import TransportPlugin, final_was_unsent, get_dependency, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress
from corvidae.trace import mint_trace_id

logger = logging.getLogger("corvidae.zulip_plugin")

# How often the per-channel typing refresh loop re-sends "start" while a reply
# is owed. A Zulip server expires a typing notice after about 15 s and its own
# web client re-sends every 10 s, so 8 s keeps the indicator continuous. Read by
# name (not bound as a default argument) so tests patching it take effect.
TYPING_REFRESH_SECONDS = 8

# Zulip's default message length limit, in Unicode code points. The server's own
# value (from /register) wins when it is smaller.
ZULIP_MESSAGE_LIMIT = 10000

# The oldest Zulip feature level this transport talks to: Zulip 9.0, which added
# "channel" as a message and typing type (the older value "stream" is on its way
# out and "direct" needs 7.0).
MIN_FEATURE_LEVEL = 248

# Where a reply goes when no topic is remembered for the channel (the first
# turn after a restart, or a channel no allowed message has reached yet). 60 is the longest topic every
# Zulip server accepts (newer ones allow 1024), so the default always fits.
DEFAULT_TOPIC = "corvidae"
_MAX_DEFAULT_TOPIC = 60

# Connection retry backoff ladder -- the same one IRC, Signal and Discord use.
_INITIAL_BACKOFF = 10
_BACKOFF_MULTIPLIER = 2
_BACKOFF_CAP = 300

# Replies held while the server is unreachable, oldest first, bounded (PRD 3.2
# item 8). Past the bound the oldest is dropped and the drop is logged.
_OUTBOUND_QUEUE_MAX = 50

# HTTP budgets. Ordinary calls give up after this long; the long poll uses the
# server's own recommended timeout, with this as the fallback.
REQUEST_TIMEOUT = 30.0
DEFAULT_LONGPOLL_TIMEOUT = 90.0

# A 429 carries retry-after; waits longer than this are cut so one rate-limit
# reply cannot park the transport for minutes. A send retries this many times.
_MAX_RETRY_AFTER = 60.0
_DEFAULT_RETRY_AFTER = 5.0
_SEND_RATE_LIMIT_RETRIES = 2

# Channel scopes this transport owns: a bare stream id, or "dm:" and the
# comma-separated user ids. Ascii digits only, no leading zero (a second spelling of one id would be a second channel).
_SCOPE_RE = re.compile(r"[1-9][0-9]*|dm:[1-9][0-9]*(,[1-9][0-9]*)*")

# Codes that mean the credentials (or the account, or the organisation) are
# unusable. Retrying cannot fix any of them.
_FATAL_CODES = frozenset({"UNAUTHORIZED", "USER_DEACTIVATED", "REALM_DEACTIVATED"})

# A mention sigil is "@" directly followed by "*": @**Name**, @**all**, @*group*.
# "@_**Name**" (a silent mention) notifies no one and is left alone.
_MENTION_SIGIL = re.compile(r"@(?=\*)")

# Hosts for which plain http is not worth a warning (a local test server).
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


# ---------------------------------------------------------------------------
# Errors: what a request can come back as, and what the loop does about each
# ---------------------------------------------------------------------------


class _ZulipError(Exception):
    """Base of the transport's own signals; never escapes the plugin."""


class _Fatal(_ZulipError):
    """The server will never accept us as configured. The text is for the
    operator: it names the site and says what to change."""


class _QueueLost(_ZulipError):
    """BAD_EVENT_QUEUE_ID: the server garbage-collected our event queue."""


class _RateLimited(_ZulipError):
    """HTTP 429. ``retry_after`` is how many seconds the server asked for."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after}s")
        self.retry_after = retry_after


class _Transient(_ZulipError):
    """A failure a retry may cure: network error, timeout, 5xx, non-JSON reply."""


class _Rejected(_ZulipError):
    """The server understood and refused (a 4xx): retrying the same request
    cannot help. Carries the server's own code and text."""

    def __init__(self, status: int, code: str, msg: str) -> None:
        super().__init__(f"HTTP {status} {code}: {msg}")
        self.status = status
        self.code = code
        self.msg = msg


@dataclass
class _EventQueue:
    """A registered event queue and our place in it."""

    queue_id: str
    last_event_id: int
    longpoll_timeout: float


@dataclass
class _Outbound:
    """One chunk of a reply waiting to be posted."""

    channel_id: str
    form: dict[str, str]
    text: str
    message_no: int  # chunks of one reply share this, so a refusal drops them all


def _defuse_mentions(text: str) -> str:
    """Break every notifying mention in generated text.

    Zulip has no allowed-mentions switch (Discord's ``AllowedMentions.none()``),
    so a reply that said ``@**all**`` would page the whole channel. A zero-width
    space between the ``@`` and the ``*`` stops Zulip parsing a mention; the text
    still reads the same. Silent mentions (``@_**Name**``) and ordinary ``@``
    signs are untouched. Applied before splitting, so the chunks concatenate back
    to this text.
    """
    return _MENTION_SIGIL.sub("@\u200b", text)


def _is_loopback(host: str) -> bool:
    """Whether a site host is this machine."""
    return host in _LOOPBACK_HOSTS


class ZulipPlugin(TransportPlugin):
    """Transport plugin for Zulip direct messages, group DMs and channels."""

    transport = "zulip"
    # A chat transport: a person is on the other end who may end the exchange.
    conversational = True

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._registry: Optional[ChannelRegistry] = None

        # Config, populated by on_init. _site stays None when the zulip: block
        # is absent -- that is the inert-plugin signal on_start checks.
        self._site: Optional[str] = None
        self._email: str = ""
        self._api_key: str = ""
        self._allow: set[str] = set()
        self._message_chunk_size: int = ZULIP_MESSAGE_LIMIT
        self._require_mention: bool = True
        #: Per-channel overrides of the transport default, keyed by stream id
        #: as a string.
        self._channel_require_mention: dict[str, bool] = {}
        self._default_topic: str = DEFAULT_TOPIC

        # What the server told us at register time.
        self._bot_id: Optional[int] = None
        self._mention_re: Optional[re.Pattern[str]] = None
        self._server_max_len: Optional[int] = None
        self._empty_topic_label: str = "general chat"

        # Connection state.
        self._session: Optional[aiohttp.ClientSession] = None
        self._connect_task: Optional[asyncio.Task] = None

        # The topic of the newest admitted message per channel id: where the
        # next reply goes. Memory only; a restart forgets it (default_topic).
        self._topics: dict[str, str] = {}

        # Replies held while the server is unreachable (PRD 3.2 item 8).
        self._outbound: collections.deque[_Outbound] = collections.deque()
        self._flush_lock = asyncio.Lock()
        # True while _flush_outbound is posting _outbound[0]; that head is in
        # flight and must not be dropped by overflow handling.
        self._flushing = False
        self._dropped = 0
        self._message_counter = 0

        # Liveness bookkeeping, per channel id (transport:scope).
        self._outstanding: dict[str, int] = {}
        self._typing_tasks: dict[str, asyncio.Task] = {}
        # Trace ids admitted while their on_message call is still running;
        # _handle_message's finally consumes them.
        self._admitted: set[str] = set()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read the zulip: config block.

        Absence means the transport stays inert. A present block with a missing
        or malformed field raises a ValueError naming the field, the value and
        what is accepted.
        """
        await super().on_init(pm, config)
        block = config.get("zulip")
        if block is None:
            return
        if not isinstance(block, dict):
            raise ValueError(
                "zulip: config field zulip must be a mapping of settings, got "
                f"{block!r}; write it as a zulip: block in agent.yaml"
            )

        # The site: a bare scheme://host[:port], no path, query or credentials.
        site = block.get("site")
        if site is None:
            raise ValueError(
                "zulip: no site; set zulip.site in agent.yaml to your "
                "organisation's URL, e.g. https://chat.example.org"
            )
        parts = urlsplit(site) if isinstance(site, str) else None
        if (
            parts is None
            or parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or parts.username
            or parts.password
        ):
            # A URL with a user:password or a query string (?api_key=...) may carry a
            # secret; never echo it.
            hidden = parts is not None and bool(parts.username or parts.password or parts.query)
            shown = (
                "a URL with credentials or a query string in it (not shown)"
                if hidden
                else repr(site)
            )
            raise ValueError(
                f"zulip: config field zulip.site must be a bare http(s) URL such as "
                f"https://chat.example.org, got {shown}; fix zulip.site in agent.yaml"
            )
        site = f"{parts.scheme}://{parts.netloc}"

        # The bot's login. The email is not secret but the key is, so both may
        # come from the environment; an empty value falls through to it.
        email = block.get("email") or os.environ.get("ZULIP_EMAIL")
        if email and not isinstance(email, str):
            raise ValueError(
                f"zulip: config field zulip.email must be a string, got {email!r}; "
                "fix zulip.email in agent.yaml"
            )
        if not email:
            raise ValueError(
                "zulip: no bot email; set zulip.email in agent.yaml or the "
                "ZULIP_EMAIL environment variable (the bot's email, from "
                "Settings → Bots in Zulip)"
            )
        api_key = block.get("api_key") or os.environ.get("ZULIP_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ValueError(
                "zulip: no bot API key; set zulip.api_key in agent.yaml or the "
                "ZULIP_API_KEY environment variable (the bot's key, from "
                "Settings → Bots in Zulip)"
            )

        # The allowlist: Zulip user ids, as YAML ints or quoted digit strings.
        # An email is refused rather than guessed at: it is not stable (and under
        # restricted visibility the API email is a synthetic one).
        allow = block.get("allow", [])
        if not isinstance(allow, list):
            raise ValueError(
                f"zulip: config field zulip.allow must be a list of Zulip user ids, "
                f"got {allow!r}; write `allow: [123, 456]` in the zulip: block of agent.yaml"
            )
        allowed: set[str] = set()
        for entry in allow:
            is_int = isinstance(entry, int) and not isinstance(entry, bool)
            if not (is_int or (isinstance(entry, str) and re.fullmatch(r"[0-9]+", entry))):
                raise ValueError(
                    f"zulip: config field zulip.allow has the entry {entry!r}, which is "
                    "not a Zulip user id; use the number from the user's profile "
                    "(an integer, or digits in quotes), not an email or a name — "
                    "fix zulip.allow in agent.yaml"
                )
            allowed.add(str(entry))

        # Outbound chunk size, in UTF-8 bytes. Bytes are never fewer than the
        # code points Zulip limits, so a byte limit is always safe.
        chunk_size = block.get("message_chunk_size", ZULIP_MESSAGE_LIMIT)
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or not 100 <= chunk_size <= ZULIP_MESSAGE_LIMIT
        ):
            raise ValueError(
                "zulip: config field 'message_chunk_size' must be an integer "
                f"between 100 and {ZULIP_MESSAGE_LIMIT}, got {chunk_size!r}; "
                "fix zulip.message_chunk_size in agent.yaml"
            )

        # require_mention is a plain bool; a string like "false" is truthy, so
        # anything else is rejected rather than coerced.
        require_mention = block.get("require_mention", True)
        if not isinstance(require_mention, bool):
            raise ValueError(
                "zulip: config field 'require_mention' must be true or false, got "
                f"{require_mention!r}; fix zulip.require_mention in agent.yaml"
            )

        # Where a reply with no remembered topic goes.
        default_topic = block.get("default_topic", DEFAULT_TOPIC)
        if (
            not isinstance(default_topic, str)
            or not default_topic.strip()
            or len(default_topic) > _MAX_DEFAULT_TOPIC
        ):
            raise ValueError(
                "zulip: config field 'default_topic' must be a non-blank string of "
                f"at most {_MAX_DEFAULT_TOPIC} characters, got {default_topic!r}; "
                "fix zulip.default_topic in agent.yaml"
            )

        overrides = self._read_channel_overrides(block.get("channels"))

        # Everything validated: commit it in one step so a failed init leaves
        # the plugin inert, never half-configured.
        self._site = site
        self._email = email
        self._api_key = api_key
        self._allow = allowed
        self._message_chunk_size = chunk_size
        self._require_mention = require_mention
        self._default_topic = default_topic
        self._channel_require_mention = overrides

        # Credentials and chat over plain http to another machine cross the
        # network in the clear. The operator's call, but not a silent one.
        if parts.scheme == "http" and not _is_loopback(parts.hostname):
            logger.warning(
                "zulip: %s is plain http, so the bot's API key and every message "
                "cross the network unencrypted; use an https:// site in "
                "zulip.site unless this is a trusted network",
                site,
            )

    @staticmethod
    def _read_channel_overrides(channels: Any) -> dict[str, bool]:
        """Validate ``zulip.channels`` and return {stream id: require_mention}.

        Keys are Zulip stream ids (digits), because a name or a ``zulip:`` prefix
        would match no channel and leave an override silently doing nothing.
        """
        # An empty YAML section parses as null: that is "no overrides", not an
        # error for an operator who commented out the only entry.
        if channels is None:
            return {}
        if not isinstance(channels, dict):
            raise ValueError(
                "zulip: config field zulip.channels must be a mapping of stream id to "
                f'settings, got {channels!r}; write `channels: {{"7": '
                "{require_mention: false}}` in the zulip: block of agent.yaml"
            )
        result: dict[str, bool] = {}
        # Every key seen, including entries that set nothing, so a quoted and an
        # unquoted spelling of one id cannot hide behind an empty entry.
        seen: set[str] = set()
        for stream_id, entry in channels.items():
            key = str(stream_id)
            if not re.fullmatch(r"[1-9][0-9]*", key):
                raise ValueError(
                    f"zulip: config entry zulip.channels.{stream_id} is not a Zulip "
                    "stream id; use the number alone, as it appears after `zulip:` "
                    "in the agent's logs — digits only, quoted or not, no leading "
                    "zeros and no prefix — in the zulip: block of agent.yaml"
                )
            if not isinstance(entry, dict):
                raise ValueError(
                    f"zulip: config entry zulip.channels.{stream_id} must be a mapping "
                    f'of settings, got {entry!r}; write `channels: {{"{key}": '
                    "{require_mention: false}}` in the zulip: block of agent.yaml"
                )
            unknown = sorted(set(entry) - {"require_mention"}, key=str)
            if unknown:
                written = ", ".join(f"{k}={entry[k]!r}" for k in unknown)
                raise ValueError(
                    f"zulip: config entry zulip.channels.{stream_id} has unknown "
                    f"key(s) {written}; accepted: require_mention — fix the key in "
                    "the zulip: block of agent.yaml"
                )
            if key in seen:
                raise ValueError(
                    f"zulip: config entry zulip.channels.{key} is listed twice — as a "
                    "quoted string and as a number — and only one would take "
                    "effect; keep one entry in the zulip: block of agent.yaml"
                )
            seen.add(key)
            if "require_mention" not in entry:
                continue
            value = entry["require_mention"]
            if not isinstance(value, bool):
                raise ValueError(
                    f"zulip: config field zulip.channels.{stream_id}.require_mention "
                    f"must be true or false, got {value!r}; fix it in the zulip: "
                    "block of agent.yaml"
                )
            result[key] = value
        return result

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Resolve the registry, then start the connection task."""
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        if self._site is None:
            return
        # One session for the life of the plugin. The header is built by hand
        # because aiohttp 3.14 deprecates BasicAuth, and a hand-built one works on
        # every version; Zulip reads the credentials as UTF-8.
        token = base64.b64encode(f"{self._email}:{self._api_key}".encode()).decode("ascii")
        self._session = aiohttp.ClientSession(headers={"Authorization": f"Basic {token}"})
        self._connect_task = asyncio.create_task(self._connection_loop())

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the connection and typing tasks and close the session."""
        task = self._connect_task
        self._connect_task = None  # shutdown signal for the loop

        # Cancel and await each task, logging anything but cancellation.
        typing_tasks = list(self._typing_tasks.values())
        for name, t in [("connection", task)] + [("typing", t) for t in typing_tasks]:
            if t is None:
                continue
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "zulip: exception awaiting %s task during shutdown", name, exc_info=True
                )
        self._typing_tasks.clear()

        # The queue is memory-only; say so rather than lose it silently.
        if self._outbound:
            logger.warning(
                "zulip: shutting down with %d queued message(s) undelivered to %s; "
                "they are discarded",
                len(self._outbound),
                self._site,
            )

        session = self._session
        self._session = None
        if session is not None:
            try:
                await session.close()
            except Exception:
                logger.warning("zulip: exception closing the HTTP session", exc_info=True)

    # -----------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[dict[str, str]] = None,
        form: Optional[dict[str, str]] = None,
        timeout: float = REQUEST_TIMEOUT,
    ) -> dict:
        """One API call; returns the parsed success body or raises a _ZulipError.

        Sorts every way a call can end into the five signals the loop and the
        sender act on: _Fatal (credentials), _QueueLost, _RateLimited,
        _Transient (retry may cure it) and _Rejected (it will not).
        """
        session = self._session
        if session is None:
            raise _Transient("the HTTP session is closed (the transport is stopping)")
        url = f"{self._site}/api/v1{path}"
        try:
            async with session.request(
                method,
                url,
                params=query,
                data=form,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                status = resp.status
                header_retry = resp.headers.get("Retry-After")
                try:
                    parsed = await resp.json(content_type=None)
                except ValueError:
                    # A proxy error page or a captive portal, not Zulip's JSON.
                    parsed = None
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise _Transient(f"{type(exc).__name__}: {exc}") from exc

        body: Optional[dict] = parsed if isinstance(parsed, dict) else None
        code = str(body.get("code", "")) if body else ""
        msg = str(body.get("msg", "")) if body else ""

        # Rate limit first: the server says exactly how long to wait.
        if status == 429 or code == "RATE_LIMIT_HIT":
            raise _RateLimited(self._retry_after(body, header_retry))
        # Credentials, deactivated account or organisation: nothing to retry.
        if status == 401 or code in _FATAL_CODES:
            raise _Fatal(
                f"zulip: {self._site} refused the login for {self._email} "
                f"({msg or code or 'HTTP ' + str(status)}); check zulip.email and "
                "zulip.api_key (or ZULIP_EMAIL / ZULIP_API_KEY) in agent.yaml, and "
                "that the bot is active — not retrying"
            )
        if code == "BAD_EVENT_QUEUE_ID":
            raise _QueueLost()
        # A 5xx, or a 2xx that is not Zulip's JSON, may be gone on the next try.
        if status >= 500 or body is None:
            raise _Transient(
                f"HTTP {status} from {path}" + ("" if body is not None else " (not JSON)")
            )
        if status >= 400 or body.get("result") == "error":
            raise _Rejected(status, code or "ERROR", msg or "no explanation given")
        return body

    @staticmethod
    def _retry_after(body: Optional[dict], header: Optional[str]) -> float:
        """Seconds the server asked us to wait, from the body or the header."""
        for raw in ((body or {}).get("retry-after"), header):
            try:
                if raw is not None:
                    return max(0.0, float(raw))
            except (TypeError, ValueError):
                logger.debug("zulip: unreadable retry-after %r, trying the next source", raw)
        return _DEFAULT_RETRY_AFTER

    # -----------------------------------------------------------------
    # Connection loop
    # -----------------------------------------------------------------

    async def _register(self) -> _EventQueue:
        """Learn who we are, register an event queue, and read the limits."""
        me = await self._request("GET", "/users/me")
        bot_id = int(me["user_id"])
        bot_name = str(me.get("full_name") or "")
        body = await self._request(
            "POST",
            "/register",
            form={
                # Messages only. Edits and deletes are separate event types and
                # are never delivered: the log is append-only (principle 4).
                "event_types": json.dumps(["message"]),
                # fetch_event_types defaults to event_types, so the realm data
                # that carries max_message_length and the empty-topic label
                # must be asked for by name.
                "fetch_event_types": json.dumps(["realm"]),
                # Raw Markdown, which is what the model should read.
                "apply_markdown": json.dumps(False),
                # Without this the server rewrites the empty topic to its
                # display label, and a reply to that label might land elsewhere.
                "client_capabilities": json.dumps({"empty_topic_name": True}),
            },
        )
        level = int(body.get("zulip_feature_level") or 0)
        if level < MIN_FEATURE_LEVEL:
            raise _Fatal(
                f"zulip: {self._site} is at Zulip feature level {level}, older than "
                f"{MIN_FEATURE_LEVEL} (Zulip 9.0), the first release this transport "
                "can post to; upgrade the server or point zulip.site at a newer "
                "one — not retrying"
            )
        # Everything learned, committed together.
        self._bot_id = bot_id
        self._mention_re = self._compile_mention_re(bot_id, bot_name)
        limit = body.get("max_message_length")
        self._server_max_len = limit if isinstance(limit, int) and limit > 0 else None
        label = body.get("realm_empty_topic_display_name")
        if isinstance(label, str) and label:
            self._empty_topic_label = label
        timeout = body.get("event_queue_longpoll_timeout_seconds")
        longpoll = (
            float(timeout)
            if isinstance(timeout, (int, float)) and timeout > 0
            else (DEFAULT_LONGPOLL_TIMEOUT)
        )
        logger.info(
            "zulip: connected to %s as %s (user id %d, feature level %d)",
            self._site,
            self._email,
            bot_id,
            level,
        )
        return _EventQueue(
            queue_id=str(body["queue_id"]),
            last_event_id=int(body.get("last_event_id", -1)),
            longpoll_timeout=longpoll,
        )

    @staticmethod
    def _compile_mention_re(bot_id: int, bot_name: str) -> re.Pattern[str]:
        """The pattern matching a mention of this bot in raw message text.

        ``@**Name**``, ``@**Name|id**`` and the silent ``@_**Name**`` forms, and
        any name with this bot's id. Other people's mentions never match.
        """
        alternatives = [rf"@_?\*\*[^*|]*\|{bot_id}\*\*"]
        if bot_name:
            alternatives.insert(0, rf"@_?\*\*{re.escape(bot_name)}(?:\|{bot_id})?\*\*")
        return re.compile("|".join(alternatives))

    async def _get_events(self, queue: _EventQueue) -> list[dict]:
        """One long poll. Acknowledges everything up to ``last_event_id``."""
        body = await self._request(
            "GET",
            "/events",
            query={"queue_id": queue.queue_id, "last_event_id": str(queue.last_event_id)},
            timeout=queue.longpoll_timeout,
        )
        events = body.get("events")
        return [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []

    async def _connection_loop(self) -> None:
        """Register, then long-poll for events, forever.

        A lost queue is re-registered at once; a transient failure retries the
        same queue on the shared backoff ladder (10 s, doubling, capped at 300 s,
        reset on success); a rate limit waits what the server asked; bad
        credentials or a too-old server end the loop with one ERROR.
        """
        backoff = _INITIAL_BACKOFF
        queue: Optional[_EventQueue] = None
        while self._connect_task is not None:
            try:
                if queue is None:
                    queue = await self._register()
                    backoff = _INITIAL_BACKOFF
                events = await self._get_events(queue)
                # Process in order; ack only after handling, so a crash replays.
                for event in sorted(events, key=lambda e: e.get("id", 0)):
                    await self._handle_event(event)
                    queue.last_event_id = max(queue.last_event_id, int(event.get("id", -1)))
                backoff = _INITIAL_BACKOFF
                # The server answered, so anything held while it was away can go.
                if self._outbound:
                    await self._flush_outbound()
            except asyncio.CancelledError:
                raise
            except _Fatal as exc:
                logger.error("%s", exc)
                return
            except _QueueLost:
                logger.warning(
                    "zulip: the event queue on %s for %s was garbage-collected; "
                    "registering a new one. Messages sent while it was gone are "
                    "not replayed",
                    self._site,
                    self._email,
                )
                queue = None
                continue
            except _RateLimited as exc:
                wait = min(exc.retry_after, _MAX_RETRY_AFTER)
                logger.warning(
                    "zulip: %s rate-limited %s; waiting %ss before the next request",
                    self._site,
                    self._email,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            except Exception as exc:
                # Anything else (network, 5xx, a bad reply) retries on the ladder.
                logger.warning(
                    "zulip: connection error to %s as %s, retrying in %ss: %s: %s",
                    self._site,
                    self._email,
                    backoff,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_MULTIPLIER, _BACKOFF_CAP)

    # -----------------------------------------------------------------
    # Inbound
    # -----------------------------------------------------------------

    async def _handle_event(self, event: dict) -> None:
        """Handle one event. A corrupt message is skipped with a WARNING, not
        allowed to stop the loop (PRD principle 5)."""
        if event.get("type") != "message":
            return
        try:
            await self._handle_message(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            message = event.get("message")
            logger.warning(
                "zulip: skipping a message event from %s that could not be handled (event id %s, "
                "message id %s) and continuing",
                self._site,
                event.get("id"),
                message.get("id") if isinstance(message, dict) else None,
                exc_info=True,
            )

    def _require_mention_for(self, scope: str) -> bool:
        """Whether this channel answers only when mentioned: its own entry, else
        the transport default."""
        return self._channel_require_mention.get(scope, self._require_mention)

    async def _handle_message(self, event: dict) -> None:
        """Filter one inbound message and, if it passes, fire on_message."""
        msg = event["message"]
        sender_id = int(msg["sender_id"])

        # The bot's own sends come back through its queue; never answer them.
        if sender_id == self._bot_id:
            logger.debug("zulip: ignoring our own message %s", msg.get("id"))
            return

        flags = event.get("flags") or msg.get("flags") or []
        kind = msg["type"]
        topic: Optional[str] = None
        stream_name = ""
        group_names: list[str] = []
        if kind in ("stream", "channel"):
            # A channel: keyed on the stable stream id; the topic is a label.
            scope = str(int(msg["stream_id"]))
            stream_name = str(msg.get("display_recipient") or "")
            raw_topic = msg.get("subject")
            topic = str(raw_topic if raw_topic is not None else msg.get("topic", ""))
            is_group = False
            require = self._require_mention_for(scope)
        elif kind in ("private", "direct"):
            # A DM: keyed on the other people's user ids, sorted, bot excluded.
            people = {int(r["id"]): str(r.get("full_name") or "") for r in msg["display_recipient"]}
            others = sorted(set(people) - {self._bot_id})
            if not others:
                logger.debug("zulip: ignoring a message with no one but us in it")
                return
            scope = "dm:" + ",".join(str(i) for i in others)
            is_group = len(others) > 1
            group_names = [people[i] or f"user {i}" for i in others]
            # A 1:1 DM is addressed to the bot by nature; a group DM is several
            # people talking, so it follows the transport default like a channel.
            require = self._require_mention if is_group else False
        else:
            raise ValueError(f"unknown message type {kind!r}")

        # The server's `mentioned` flag: a direct mention, or one through a user
        # group the bot is in. A wildcard (@all, @topic) pages everyone, not us.
        mentioned = "mentioned" in flags
        addressed: Optional[bool] = (
            None if (kind in ("private", "direct") and not is_group) else mentioned
        )
        if require and not mentioned:
            # DEBUG: this fires on every line people send each other here. The
            # channel and sender are in the text (a default formatter drops
            # `extra`), and also in `extra` for structured sinks.
            logger.debug(
                "zulip: message from %s in zulip:%s does not mention the bot, "
                "not admitted (zulip.require_mention)",
                sender_id,
                scope,
                extra={"channel": f"zulip:{scope}", "sender": str(sender_id)},
            )
            return

        # Take the bot's own mention token out, so the model reads the request.
        text = str(msg.get("content") or "")
        if self._mention_re is not None:
            text = self._mention_re.sub("", text)
        text = text.strip()
        if not text:
            logger.debug("zulip: ignoring message with no text content")
            return

        # The allowlist is checked before the channel is resolved, so a refused
        # sender never adds a registry entry (PRD 3.2 item 6).
        sender = str(sender_id)
        allowed = sender in self._allow
        channel = (
            self._require_registry().get_or_create("zulip", scope)
            if allowed
            else self._transient_channel(scope)
        )
        # Labels and reply routing only ever come from an allowed sender: a
        # stranger's topic must not steer where the agent answers or what the
        # channel is called to the model.
        if allowed:
            if topic is not None:
                self._topics[channel.id] = topic
                shown = topic if topic else self._empty_topic_label
                channel.name = f"#{stream_name} > {shown}"
            elif is_group:
                channel.name = (
                    "group DM with "
                    + ", ".join(group_names[:5])
                    + (f" and {len(group_names) - 5} more" if len(group_names) > 5 else "")
                )
            full_name = msg.get("sender_full_name")
            if full_name:
                remember_sender_name(channel.sender_names, sender, str(full_name))

        # Typing only ever applies to an allowlisted sender -- others get no sign
        # the bot is live. on_message still fires so the real gate can veto and
        # log the attempt. The trace id is minted here (#95).
        trace_id = mint_trace_id()
        took_slot = allowed
        try:
            if took_slot:
                await self._begin_typing(channel)
            channel.addressed = addressed
            await self.pm.ahook.on_message(
                channel=channel,
                sender=sender,
                text=text,
                trace_id=trace_id,
            )
        finally:
            # Admission fires before on_message returns. Anything not admitted
            # by then (gate veto, agent not ready, an exception, cancellation)
            # gets no reply, so give its typing slot back and stop the notice.
            if took_slot and trace_id not in self._admitted:
                if self._decrement_outstanding(channel):
                    await self._send_typing(channel, "stop")
            self._admitted.discard(trace_id)

    def _transient_channel(self, scope: str) -> Channel:
        """The channel a refused sender's message is gated on.

        A live registry entry is reused (a channel an allowed user already talks
        in); otherwise the Channel is built without registering it, so the gate
        still sees the ``zulip:<scope>`` id it logs, and nothing outlives the
        refusal.
        """
        existing = self._require_registry().get(f"zulip:{scope}")
        if existing is not None:
            return existing
        return Channel(transport="zulip", scope=scope)

    def _require_registry(self) -> ChannelRegistry:
        """The channel registry, resolved in on_start.

        A message can only arrive after on_start, so this never fires in
        practice; if it does, the error says what ran out of order.
        """
        if self._registry is None:
            raise RuntimeError(
                "zulip: the channel registry is not available yet (on_start has "
                "not run); a message arrived before the transport was started"
            )
        return self._registry

    # -----------------------------------------------------------------
    # Channel resolution, register, gate
    # -----------------------------------------------------------------

    @hookimpl
    async def resolve_channel(self, channel_id: str) -> Channel | None:
        """Resolve a persisted ``zulip:<scope>`` id to its live channel.

        Defers (None) for other transports, malformed or non-canonical scopes
        (a group DM's ids must be ascending), and before on_start has resolved
        the registry.
        """
        transport, sep, scope = channel_id.partition(":")
        if transport != "zulip" or not sep or not _SCOPE_RE.fullmatch(scope):
            return None
        if scope.startswith("dm:"):
            ids = [int(i) for i in scope[3:].split(",")]
            if ids != sorted(set(ids)):
                return None
        if self._registry is None:
            return None
        return self._registry.get_or_create("zulip", scope)

    @hookimpl
    async def resolve_register_prompt(self, channel: Channel) -> str | None:
        """Default register: chat norms for a 1:1 DM, plus group manners for
        channels and group DMs."""
        if not self.owns(channel):
            return None
        if channel.scope.startswith("dm:") and "," not in channel.scope:
            return shipped_register("chat")
        return shipped_register("chat", "zulip_group")

    @hookimpl
    async def should_process_message(
        self, channel: Channel, sender: str, text: str, trace_id: str
    ) -> bool | None:
        """Default-deny gate: only allowlisted user ids may drive the agent.
        trace_id is bound without a default -- see tests/test_hook_arg_binding.py."""
        if not self.owns(channel):
            return None
        if sender in self._allow:
            return True
        # Silent refusal: log once, say nothing to the sender.
        logger.info("zulip: rejected message from unauthorized sender %s", sender)
        return False

    @hookimpl
    async def on_message_admitted(
        self, channel: Channel, trace_id: str, sender: str, text: str
    ) -> None:
        """Note that this message was admitted, so the inbound handler's finally
        keeps its typing slot (send_message gives it back instead)."""
        # Inbound hooks reach every transport; ignore other transports'.
        if not self.owns(channel):
            return
        self._admitted.add(trace_id)

    # -----------------------------------------------------------------
    # Liveness: typing indicator
    # -----------------------------------------------------------------

    async def _begin_typing(self, channel: Channel) -> None:
        """Count this message as outstanding; on the first one, send typing
        immediately and start the refresh loop."""
        count = self._outstanding.get(channel.id, 0) + 1
        self._outstanding[channel.id] = count
        if count == 1:
            await self._send_typing(channel, "start")
            # A loop from an earlier turn may still be asleep; it will see the
            # count back above zero and carry on, so a second one would double
            # the refreshes and leave one untracked by on_stop.
            existing = self._typing_tasks.get(channel.id)
            if existing is None or existing.done():
                self._typing_tasks[channel.id] = asyncio.create_task(
                    self._typing_refresh_loop(channel)
                )

    def _decrement_outstanding(self, channel: Channel) -> bool:
        """Give back one slot for a message that was never admitted. True when
        that left nothing owed (a final releases every slot instead)."""
        count = self._outstanding.get(channel.id, 0)
        if count > 0:
            self._outstanding[channel.id] = count - 1
            return count - 1 == 0
        return False

    def _release_all(self, channel: Channel) -> bool:
        """Close out every outstanding slot on a final, sent or unsent: one final
        answers a coalesced burst. True when a slot was held, i.e. a typing
        notice is showing and needs a stop. Synchronous on purpose: callers run
        it before their first await."""
        held = self._outstanding.get(channel.id, 0) > 0
        if held:
            self._outstanding[channel.id] = 0
        return held

    async def _typing_refresh_loop(self, channel: Channel) -> None:
        """Re-send typing every TYPING_REFRESH_SECONDS while a reply is owed;
        stop once nothing is owed or a send fails."""
        try:
            while self._outstanding.get(channel.id, 0) > 0:
                await asyncio.sleep(TYPING_REFRESH_SECONDS)
                if self._outstanding.get(channel.id, 0) <= 0:
                    return
                if not await self._send_typing(channel, "start"):
                    return
        finally:
            # Forget this loop only if it is still the tracked one, never a
            # successor registered under the same channel id.
            if self._typing_tasks.get(channel.id) is asyncio.current_task():
                del self._typing_tasks[channel.id]

    def _typing_form(self, channel: Channel, op: str) -> Optional[dict[str, str]]:
        """The /typing form for a channel, or None when its scope is unusable."""
        if channel.scope.startswith("dm:"):
            ids = [int(i) for i in channel.scope[3:].split(",")]
            return {"type": "direct", "op": op, "to": json.dumps(ids)}
        if channel.scope.isdigit():
            return {
                "type": "channel",
                "op": op,
                "stream_id": channel.scope,
                "topic": self._topics.get(channel.id, self._default_topic),
            }
        return None

    async def _send_typing(self, channel: Channel, op: str) -> bool:
        """Send one typing notice. A failure is logged and never blocks the
        reply; returns whether it succeeded."""
        form = self._typing_form(channel, op)
        if form is None:
            logger.debug("zulip: no typing target for %s", channel.id)
            return False
        try:
            await self._request("POST", "/typing", form=form)
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "zulip: could not send a typing %s notice for %s on %s: %s: %s",
                op,
                channel.id,
                self._site,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return False

    # -----------------------------------------------------------------
    # Outbound
    # -----------------------------------------------------------------

    def _destination(self, channel: Channel) -> Optional[dict[str, str]]:
        """The ``/messages`` fields that address a channel, or None.

        A channel goes to the topic of the newest admitted message, else
        ``default_topic``; a DM to its user ids. An empty topic is a real topic
        and is sent back verbatim.
        """
        if channel.scope.startswith("dm:"):
            ids = [int(i) for i in channel.scope[3:].split(",")]
            return {"type": "direct", "to": json.dumps(ids)}
        if channel.scope.isdigit():
            topic = self._topics.get(channel.id)
            return {
                "type": "channel",
                "to": channel.scope,
                "topic": self._default_topic if topic is None else topic,
            }
        return None

    def _chunk_limit(self) -> int:
        """Bytes per chunk: the configured size, or the server's limit if lower.

        The server counts code points and this counts bytes, which are never
        fewer, so the limit is conservative (multi-byte text splits earlier than
        strictly needed). shortcut: a plain split, so a Markdown code fence can
        be cut across two messages at this size; upgrade to a fence-aware split
        (as Discord has) if that is seen.
        """
        limit = self._message_chunk_size
        if self._server_max_len is not None:
            limit = min(limit, self._server_max_len)
        # split_message refuses anything under six bytes.
        return max(6, limit)

    async def _deliver(self, channel: Channel, text: str) -> None:
        """Split text to the chunk limit and post each non-blank chunk in order.

        Mentions are defused first. A chunk the server refuses outright drops the
        rest of that reply, so a channel the bot cannot post in logs once, not
        once per chunk.
        """
        destination = self._destination(channel)
        if destination is None:
            logger.warning(
                "zulip: %s is not a Zulip channel or DM scope; dropping the reply "
                "(it was not delivered to %s)",
                channel.id,
                self._site,
            )
            return
        self._message_counter += 1
        message_no = self._message_counter
        for chunk in split_message(_defuse_mentions(text), max_len=self._chunk_limit()):
            # Zulip rejects empty content, so skip whitespace-only chunks.
            if not chunk.strip():
                continue
            self._enqueue_outbound(_Outbound(channel.id, dict(destination), chunk, message_no))
            if message_no in await self._flush_outbound():
                return

    def _enqueue_outbound(self, item: _Outbound) -> None:
        """Queue a chunk, dropping the oldest (and saying so) past the bound."""
        if len(self._outbound) >= _OUTBOUND_QUEUE_MAX:
            # While flushing, the head is in flight, so the oldest droppable
            # entry is the one behind it.
            drop_index = 1 if self._flushing else 0
            if len(self._outbound) > drop_index:
                old = self._outbound[drop_index]
                del self._outbound[drop_index]
                self._dropped += 1
                logger.warning(
                    "zulip: outbound queue full (%d); dropped the oldest queued "
                    "message for %s (dropped=%d this outage)",
                    _OUTBOUND_QUEUE_MAX,
                    old.channel_id,
                    self._dropped,
                )
        self._outbound.append(item)

    async def _post_message(self, item: _Outbound) -> None:
        """Post one chunk, waiting out a rate limit a couple of times."""
        form = {**item.form, "content": item.text}
        for attempt in range(_SEND_RATE_LIMIT_RETRIES + 1):
            try:
                await self._request("POST", "/messages", form=form)
                return
            except _RateLimited as exc:
                if attempt == _SEND_RATE_LIMIT_RETRIES:
                    # Still limited: hold it for the next flush, not lost.
                    raise _Transient(f"still rate limited after {attempt + 1} tries") from exc
                await asyncio.sleep(min(exc.retry_after, _MAX_RETRY_AFTER))

    async def _flush_outbound(self) -> set[int]:
        """Post queued chunks in order; returns the reply numbers refused for good.

        A transient failure leaves the rest queued (WARNING) for the next flush,
        which the connection loop triggers after every successful poll. A
        refusal drops every queued chunk of that reply.
        """
        refused: set[int] = set()
        async with self._flush_lock:
            self._flushing = True
            try:
                while self._outbound:
                    item = self._outbound[0]
                    try:
                        await self._post_message(item)
                    except asyncio.CancelledError:
                        raise
                    except (_Transient, _RateLimited) as exc:
                        logger.warning(
                            "zulip: could not post to %s on %s; message queued until "
                            "the server answers, not dropped (depth=%d): %s",
                            item.channel_id,
                            self._site,
                            len(self._outbound),
                            exc,
                        )
                        return refused
                    except (_Rejected, _Fatal) as exc:
                        # The server will never take this one: drop the whole reply.
                        reason = exc.msg if isinstance(exc, _Rejected) else str(exc)
                        logger.warning(
                            "zulip: could not post to %s (topic %r) on %s: %s; the "
                            "message was dropped — if the server says the bot may not "
                            "post, check it is subscribed to the channel and allowed to",
                            item.channel_id,
                            item.form.get("topic"),
                            self._site,
                            reason,
                        )
                        refused.add(item.message_no)
                        self._outbound = collections.deque(
                            i for i in self._outbound if i.message_no != item.message_no
                        )
                        continue
                    self._outbound.popleft()
                    self._dropped = 0
            finally:
                self._flushing = False
        return refused

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
        # An unsent final never reaches send_message, so release here, in the
        # same synchronous step as the check (before any await).
        try:
            unsent = final_was_unsent(channel)
        except Exception:
            logger.warning(
                "%s: could not tell whether the final on %s was left unsent; "
                "releasing nothing (the turn may stay open until the next final)",
                type(self).__name__,
                channel.id,
                exc_info=True,
                extra={"channel": channel.id, "trace_id": trace_id},
            )
            return
        if unsent and self._release_all(channel):
            await self._send_typing(channel, "stop")

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Deliver a final reply and close out every outstanding slot: one reply
        answers a coalesced burst."""
        await self._deliver(channel, event.text)
        if self._release_all(channel):
            await self._send_typing(channel, "stop")

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        """Deliver intermediate assistant text without closing out the
        outstanding slots (the final reply still owes them)."""
        await self._deliver(channel, event.text)
