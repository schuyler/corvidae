"""Discord transport plugin, built on discord.py's gateway client.

Connects to Discord as a bot account and talks in DMs, server channels, and
threads. In servers the bot by default only reacts when it is mentioned; DMs
always reach it. Only allowlisted Discord user ids may drive the agent.

By default the client does not request the privileged Message Content
intent: Discord still delivers message content for DMs and for messages that
mention the bot, which are the only messages this transport processes. With
``require_mention: false`` the bot answers every server message from an
allowlisted user, and the client requests that intent (it must also be
enabled in the Discord Developer Portal). A ``discord.channels.<channel id>``
entry may override ``require_mention`` for one channel, and its threads inherit
it (#512); the intent is requested once for the client, so a single
conversational channel is enough to need it.

The SDK (discord-py) is optional: add ``--extra discord`` to the ``uv sync``
line. This module
never imports it at load time, because pluggy loads every entry point
eagerly and an ImportError here would stop every install without the extra
from starting. It is imported on first use, and ``on_init`` fails with an
install hint when a ``discord:`` block is configured without it.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import aiohttp

from corvidae.attachments import save_attachment
from corvidae.channel import Channel, ChannelRegistry
from corvidae.channels._registers import shipped_register
from corvidae.channels.split import split_message
from corvidae.hooks import TransportPlugin, final_was_unsent, get_dependency, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress
from corvidae.trace import mint_trace_id

logger = logging.getLogger("corvidae.discord_plugin")

# How often the per-channel typing-indicator refresh loop re-sends "typing"
# while a turn is outstanding. Discord's indicator lasts about 10s, so 8s
# keeps it continuous. Read by name (not bound as a default argument) so that
# tests patching this module attribute take effect.
TYPING_REFRESH_SECONDS = 8

# Discord's hard limit on a single message's length, in characters.
DISCORD_MESSAGE_LIMIT = 2000

# Read granularity for an attachment body; also the unit the cap is checked in.
_DOWNLOAD_CHUNK_BYTES = 64 * 1024

# How long one attachment download may take before it is given up on. Images
# are small; this is a stuck-CDN guard, not a throughput target.
ATTACHMENT_DOWNLOAD_TIMEOUT = 30.0

# The extension we choose for a downloaded image, keyed by the content type
# Discord asserts. A remote filename is only a fallback: the suffix decides how
# the file is read later, so it should be ours.
_IMAGE_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

# Attachment kinds we fetch and store. Everything else is named in the note and
# left on Discord's CDN (video, audio, stickers): #521.
_IMAGE_CONTENT_TYPES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp",
})

# Default location for downloaded files, and the default cap for one file.
DEFAULT_ATTACHMENT_DIR = "state/discord-attachments"
DEFAULT_MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024

# Connection retry backoff ladder -- the same one IRC and Signal use.
_INITIAL_BACKOFF = 10
_BACKOFF_MULTIPLIER = 2
_BACKOFF_CAP = 300

# Longest code-fence info string (language tag) that is carried across a
# chunk boundary; longer ones are dropped so the reserve below stays valid.
_LANG_MAX_BYTES = 16

# Bytes reserved in every chunk for a reopening prefix ("```" + lang + "\n")
# and a closing suffix ("\n```"), so adding them never overflows the limit.
_FENCE_RESERVE = len("```") + _LANG_MAX_BYTES + len("\n") + len("\n```")

# A code-fence marker.
_FENCE = "```"

# Channel scopes this transport owns: a bare channel/thread id, or
# "dm:<user id>". Ascii digits only.
_SCOPE_RE = re.compile(r"(dm:)?[0-9]+")


def _nbytes(text: str) -> int:
    """UTF-8 byte length of text."""
    return len(text.encode("utf-8"))


def _fence_lang(body: str, match_start: int, match_end: int, line_start: bool) -> str:
    """Language tag of the opening fence found at body[match_start:match_end].

    Returns "" unless the fence sits at the start of a line, its info string
    is complete within the body (ends in a newline), contains no backtick,
    and is at most _LANG_MAX_BYTES bytes.
    """
    # The fence must start a line to be an opening code-block fence.
    if match_start == 0:
        if not line_start:
            return ""
    elif body[match_start - 1] != "\n":
        return ""
    # The info string runs to the end of the line; without a newline in this
    # body it may be cut, so it cannot be trusted.
    newline = body.find("\n", match_end)
    if newline == -1:
        return ""
    info = body[match_end:newline].strip()
    if "`" in info or _nbytes(info) > _LANG_MAX_BYTES:
        return ""
    return info


def _split_discord_segments(text: str, max_len: int) -> list[tuple[str, str, str]]:
    """Split text into (prefix, body, suffix) segments for Discord.

    Invariants (for max_len >= _FENCE_RESERVE + 6): the bodies concatenate to
    text exactly; prefix + body + suffix is at most max_len UTF-8 bytes; and
    each prefix + body + suffix holds an even number of ``` markers, so
    Discord renders each chunk's code as code. A fence left open at a chunk
    boundary is closed by the suffix and reopened (with the same language
    tag) by the next chunk's prefix.
    """
    # Whole text fits and its fences are already balanced: nothing to do.
    if _nbytes(text) <= max_len and text.count(_FENCE) % 2 == 0:
        return [("", text, "")]

    # Break the text into packable pieces: whole lines (newline kept), with
    # any line too big for a body cut by the shared word/codepoint splitter.
    # Each piece records whether it starts a line.
    budget = max_len - _FENCE_RESERVE
    pieces: list[tuple[str, bool]] = []
    for line in _lines_keeping_newline(text):
        if _nbytes(line) <= budget:
            pieces.append((line, True))
        else:
            for i, part in enumerate(split_message(line, budget)):
                pieces.append((part, i == 0))

    # Greedily pack pieces into bodies, remembering whether each body starts
    # at the beginning of a line.
    bodies: list[tuple[str, bool]] = []
    current = ""
    current_line_start = True
    for piece, starts_line in pieces:
        if current and _nbytes(current) + _nbytes(piece) > budget:
            bodies.append((current, current_line_start))
            current = ""
        if not current:
            current_line_start = starts_line
        current += piece
    if current:
        bodies.append((current, current_line_start))

    # Walk the bodies tracking fence state, adding the reopening prefix and
    # closing suffix. State is the parity of ``` markers seen in bodies so
    # far (counted per body, so a marker cut across two bodies counts in
    # neither and every chunk stays balanced).
    segments: list[tuple[str, str, str]] = []
    odd = False
    lang = ""
    for body, line_start in bodies:
        prefix = f"{_FENCE}{lang}\n" if odd else ""
        # Toggle state for each marker in this body, noting the language of
        # each opening one.
        for match in re.finditer(re.escape(_FENCE), body):
            odd = not odd
            if odd:
                lang = _fence_lang(body, match.start(), match.end(), line_start)
        # Still inside a fence at the end of the body: close it.
        suffix = ""
        if odd:
            suffix = _FENCE if body.endswith("\n") else "\n" + _FENCE
        segments.append((prefix, body, suffix))
    return segments


def _lines_keeping_newline(text: str) -> list[str]:
    """Split text on '\\n' only, keeping each newline on its line, so that
    joining the result reproduces text exactly (unlike str.splitlines)."""
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    # The last element has no trailing newline; drop it if empty.
    if lines[-1]:
        out.append(lines[-1])
    return out


def split_discord_message(text: str, max_len: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split text into Discord-sized chunks, keeping code blocks intact.

    Whitespace-only chunks are kept so that the reconstruction invariant
    stays exact; the sender skips them at send time.
    """
    return [p + b + s for p, b, s in _split_discord_segments(text, max_len)]


def _sdk() -> ModuleType:
    """Return the discord.py module, imported on first use (see the module
    docstring for why it is not imported at load time)."""
    import discord

    return discord


@functools.cache
def _client_class() -> type:
    """Build the gateway client class on first use.

    It subclasses ``discord.Client``, so it can only exist once the SDK is
    imported; it is built once and reused.
    """
    discord = _sdk()

    class _DiscordClient(discord.Client):
        """discord.Client wired to a DiscordPlugin.

        discord.py runs each gateway event as its own task, so on_message does
        nothing but hand the message to the plugin's synchronous inbound queue.
        """

        def __init__(self, plugin: "DiscordPlugin", *, intents: Any) -> None:
            super().__init__(intents=intents)
            self._plugin = plugin

        async def on_ready(self) -> None:
            """Log the connected account and tell the plugin the connection is good."""
            user_id = self.user.id if self.user is not None else None
            logger.info("discord: connected as user id %s", user_id)
            self._plugin._on_ready()

        async def on_message(self, message: Any) -> None:
            """Queue an inbound message for the plugin's single consumer task."""
            self._plugin._enqueue(message, self.user.id if self.user is not None else None)

    return _DiscordClient


#: How many sender display names one channel remembers (#513).
SENDER_NAME_CACHE_MAX = 256


def remember_sender_name(names: dict[str, str], sender: str, name: str) -> None:
    """Record a sender's display name on the channel, keeping the map bounded.

    A channel people talk in all day would otherwise keep a name for every
    sender it has ever seen. The cache is a convenience for the next turn's
    attribution, so the least recently seen entry goes when it is full (#513).
    """
    # Re-stating a name moves it to the newest end, so the person who talks
    # most is not the first evicted for having been seen first.
    if sender in names:
        del names[sender]
    elif len(names) >= SENDER_NAME_CACHE_MAX:
        names.pop(next(iter(names)))
    names[sender] = name


def _channel_name(channel: Any) -> str | None:
    """The name to show the model for a Discord channel or thread (#513).

    A guild channel is its own name. A thread is its title and its parent, so
    the model can tell "a thread in #home" from "#home". Anything without a
    name of its own gets None rather than its id dressed up as one.
    """
    name = getattr(channel, "name", None)
    parent = getattr(channel, "parent", None)
    parent_name = getattr(parent, "name", None) if parent is not None else None
    if name is None:
        return None
    if parent_name is not None:
        return f"thread {name} in #{parent_name}"
    return f"#{name}"


def _attachment_kind(att: Any) -> str:
    """A short word for an attachment we do not download: image, video, ..."""
    content_type = str(getattr(att, "content_type", "") or "").lower()
    for prefix, kind in (
        ("image/", "image"), ("video/", "video"), ("audio/", "audio"),
    ):
        if content_type.startswith(prefix):
            return kind
    name = str(getattr(att, "filename", "") or "").lower()
    if name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return "image"
    return "document"


class DiscordPlugin(TransportPlugin):
    """Transport plugin for Discord DMs, server channels, and threads."""

    transport = "discord"
    # A chat transport: a person is on the other end who may end the exchange.
    conversational = True

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._registry: Optional[ChannelRegistry] = None

        # Config, populated by on_init. _token stays None when the discord:
        # block is absent -- that is the inert-plugin signal on_start checks.
        self._token: Optional[str] = None
        self._allow: set[str] = set()
        self._message_chunk_size: int = DISCORD_MESSAGE_LIMIT
        self._attachment_dir: Path = Path(DEFAULT_ATTACHMENT_DIR)
        self._max_attachment_size: int = DEFAULT_MAX_ATTACHMENT_SIZE
        self._send_attachment_root: Path = Path("state")
        self._require_mention: bool = True
        #: Per-channel overrides of the transport default (#512), keyed by the
        #: Discord channel id as a string. A thread looks up its parent's entry.
        self._channel_require_mention: dict[str, bool] = {}

        # Connection state. _client is the live (or latest) gateway client.
        self._client: Any = None
        self._connect_task: Optional[asyncio.Task] = None
        self._backoff: float = _INITIAL_BACKOFF

        # Inbound messages are queued synchronously and consumed by one task,
        # because discord.py dispatches every event as its own task and
        # handling them there would let messages overtake each other.
        self._inbound: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None

        # Liveness bookkeeping, per channel id (transport:scope).
        self._outstanding: dict[str, int] = {}
        self._typing_tasks: dict[str, asyncio.Task] = {}
        # Trace ids admitted while their on_message call is still
        # running; _handle_message's finally consumes them (#102).
        self._admitted: set[str] = set()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read the discord: config block.

        Absence means the transport stays inert. A present block with a
        missing or malformed field raises a ValueError naming the field.
        """
        await super().on_init(pm, config)
        block = config.get("discord")
        if block is None:
            return
        if not isinstance(block, dict):
            raise ValueError("discord: config block must be a mapping")

        # A configured block needs the SDK. Checked here, at startup, rather
        # than when the connection loop first builds a client, so a missing
        # extra fails loudly instead of as a retried connection error. Not
        # cached: an install can gain or lose the SDK between runs.
        try:
            _sdk()
        except ImportError as exc:
            raise RuntimeError(
                "DiscordPlugin: agent.yaml has a discord: block, but the "
                "Discord SDK (discord-py) is not installed. Install it with "
                "uv sync --extra discord, adding the other extras you use "
                "(e.g. --extra cognition): uv sync removes extras you don't name."
            ) from exc

        # The token may live in the environment instead of agent.yaml.
        token = block.get("token") or os.environ.get("DISCORD_TOKEN")
        if not token:
            raise ValueError(
                "discord: no bot token; set 'token' in the discord: block "
                "or the DISCORD_TOKEN environment variable"
            )

        allow = block.get("allow", [])
        if not isinstance(allow, list):
            raise ValueError("discord: config field 'allow' must be a list")

        chunk_size = block.get("message_chunk_size", DISCORD_MESSAGE_LIMIT)
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or not 100 <= chunk_size <= DISCORD_MESSAGE_LIMIT
        ):
            raise ValueError(
                "discord: config field 'message_chunk_size' must be an "
                f"integer between 100 and {DISCORD_MESSAGE_LIMIT}"
            )

        # require_mention is a plain bool; a string like "false" is truthy,
        # so anything else is rejected rather than coerced.
        require_mention = block.get("require_mention", True)
        if not isinstance(require_mention, bool):
            raise ValueError(
                "discord: config field 'require_mention' must be true or false, "
                f"got {require_mention!r}; fix it in the discord: block of agent.yaml"
            )

        # Per-channel overrides (#512). Each entry may set its own
        # require_mention; an unknown key is named rather than ignored, because
        # a typo there would silently leave the channel on the default.
        channels = block.get("channels")
        # An empty YAML section parses as null, and an operator who comments out
        # the only entry should get the default, not a startup failure.
        if channels is None:
            channels = {}
        if not isinstance(channels, dict):
            raise ValueError(
                "discord: config field 'channels' must be a mapping of channel "
                f"id to settings, got {channels!r}; write `channels: {{\"<channel "
                "id>\": {require_mention: false}}}` in the discord: block of "
                "agent.yaml"
            )
        channel_overrides: dict[str, bool] = {}
        for channel_id, entry in channels.items():
            # The key is matched against the id Discord delivers, so it has to be
            # that id and nothing else. `discord:<id>` and `#general` are the two
            # shapes a reader of the *other* channels: block will try by habit,
            # and either one would match no channel and leave the override doing
            # nothing (#512).
            # A leading zero is never part of a real snowflake, and a quoted
            # "0012345" would match no channel — the silent no-op this check
            # exists to prevent.
            if not re.fullmatch(r"[1-9][0-9]*", str(channel_id)):
                raise ValueError(
                    "discord: config entry 'channels."
                    f"{channel_id}' is not a Discord channel id; use the number "
                    "alone, as it appears after `discord:` in the agent's logs "
                    "— 123456789012345678, digits only, quoted or not, no leading "
                    "zeros and no prefix — in the discord: block of agent.yaml"
                )
            if not isinstance(entry, dict):
                raise ValueError(
                    "discord: config entry 'channels."
                    f"{channel_id}' must be a mapping of settings, got {entry!r}; "
                    "write `channels: {\"<channel id>\": {require_mention: false}}` "
                    "in the discord: block of agent.yaml"
                )
            # YAML gives keys as whatever it parsed: an int typo key sorts
            # against a str one with TypeError, so the message would be lost on
            # the way out. Compare as text.
            unknown_keys = sorted(set(entry) - {"require_mention"}, key=str)
            if unknown_keys:
                written = ", ".join(f"{key}={entry[key]!r}" for key in unknown_keys)
                raise ValueError(
                    "discord: config entry 'channels."
                    f"{channel_id}' has unknown key(s) {written}; accepted: "
                    "require_mention — fix the key in the discord: block of "
                    "agent.yaml"
                )
            if "require_mention" not in entry:
                continue
            value = entry["require_mention"]
            if not isinstance(value, bool):
                raise ValueError(
                    "discord: config field 'channels."
                    f"{channel_id}.require_mention' must be true or false, got "
                    f"{value!r}; fix it in the discord: block of agent.yaml"
                )
            key = str(channel_id)
            if key in channel_overrides:
                raise ValueError(
                    "discord: config entry 'channels."
                    f"{key}' is listed twice — as a quoted string and as a "
                    "number — and only one would take effect; keep one entry in "
                    "the discord: block of agent.yaml"
                )
            channel_overrides[key] = value

        attachment_dir = block.get("attachment_dir", DEFAULT_ATTACHMENT_DIR)
        if not isinstance(attachment_dir, str) or not attachment_dir:
            raise ValueError(
                "discord: config field 'attachment_dir' must be a non-empty path"
            )

        max_size = block.get("max_attachment_size", DEFAULT_MAX_ATTACHMENT_SIZE)
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError(
                "discord: config field 'max_attachment_size' must be a positive "
                f"integer number of bytes, got {max_size!r}"
            )

        send_root = block.get("send_attachment_root", None)
        if send_root is not None and (not isinstance(send_root, str) or not send_root):
            raise ValueError(
                "discord: config field 'send_attachment_root' must be a non-empty "
                "path, or absent to use state/"
            )

        self._token = str(token)
        self._attachment_dir = Path(attachment_dir)
        self._max_attachment_size = max_size
        self._send_attachment_root = Path(send_root) if send_root else Path("state")
        # Ids are compared as strings against str(author.id); YAML parses
        # them as ints.
        self._allow = {str(entry) for entry in allow}
        self._message_chunk_size = chunk_size
        self._require_mention = require_mention
        self._channel_require_mention = channel_overrides

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Resolve the registry, then start the connection and inbound tasks."""
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        if self._token is None:
            return
        self._connect_task = asyncio.create_task(self._connection_loop())
        self._processor_task = asyncio.create_task(self._process_inbound())

    @hookimpl
    async def on_stop(self) -> None:
        """Close the client and cancel the connect, processor and typing tasks."""
        task = self._connect_task
        self._connect_task = None  # shutdown signal for the retry loop

        # Close the gateway client so a blocked start() returns.
        client = self._client
        if client is not None:
            await self._close_client(client)

        # Cancel and await each task, logging anything but cancellation.
        processor = self._processor_task
        self._processor_task = None
        typing_tasks = list(self._typing_tasks.values())
        for name, t in [("connect", task), ("processor", processor)] + [
            ("typing", t) for t in typing_tasks
        ]:
            if t is None:
                continue
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "discord: exception awaiting %s task during shutdown", name, exc_info=True
                )
        self._typing_tasks.clear()

    async def _close_client(self, client: Any) -> None:
        """Close a client, logging (not raising) any failure."""
        try:
            await client.close()
        except Exception:
            logger.warning("discord: exception closing client", exc_info=True)

    # -----------------------------------------------------------------
    # Connection loop
    # -----------------------------------------------------------------

    def _make_client(self) -> Any:
        """Build a gateway client. Default intents, plus the privileged
        Message Content intent only when require_mention is false (see the
        module docstring)."""
        # Start from the default intents.
        intents = _sdk().Intents.default()
        # Unmentioned server messages carry no text without the intent. The
        # intent is per client, not per channel, so one conversational channel
        # is enough to need it (#512).
        if not self._require_mention or False in self._channel_require_mention.values():
            intents.message_content = True
        return _client_class()(self, intents=intents)

    def _on_ready(self) -> None:
        """A connection fully came up: the next failure retries quickly."""
        self._backoff = _INITIAL_BACKOFF

    async def _connection_loop(self) -> None:
        """Run the gateway client, retrying on failure with IRC's backoff.

        10s initial, doubling, capped at 300s; reset to the initial delay
        when the client reports ready. A rejected token is not retried.
        """
        self._backoff = _INITIAL_BACKOFF
        while self._connect_task is not None:
            client = self._make_client()
            # Assigned before start() so sends and on_stop can reach it.
            self._client = client
            try:
                await client.start(self._token)
            except asyncio.CancelledError:
                raise
            except _sdk().LoginFailure as exc:
                # A bad token never gets better by retrying.
                logger.error(
                    "discord: login failed; check that discord.token (or "
                    "DISCORD_TOKEN) is a valid bot token: %s", exc,
                )
                return
            except Exception as exc:
                logger.warning("discord: connection error, retrying: %s", exc, exc_info=True)
            else:
                if self._connect_task is not None:
                    logger.warning("discord: client stopped unexpectedly, reconnecting")
            finally:
                # close() is idempotent; a failed login would otherwise leak
                # its aiohttp session.
                await self._close_client(client)

            if self._connect_task is None:
                return
            await asyncio.sleep(self._backoff)
            self._backoff = min(self._backoff * _BACKOFF_MULTIPLIER, _BACKOFF_CAP)

    # -----------------------------------------------------------------
    # Inbound
    # -----------------------------------------------------------------

    def _enqueue(self, message: Any, bot_user_id: Optional[int]) -> None:
        """Queue an inbound message (synchronous, so arrival order is kept)."""
        self._inbound.put_nowait((message, bot_user_id))

    async def _process_inbound(self) -> None:
        """Consume queued messages one at a time, in arrival order."""
        while True:
            message, bot_user_id = await self._inbound.get()
            try:
                await self._handle_message(message, bot_user_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("discord: error handling inbound message, continuing", exc_info=True)

    async def _handle_message(self, message: Any, bot_user_id: Optional[int]) -> None:
        """Filter one inbound message and, if it passes, fire on_message."""
        author = message.author

        # Never respond to ourselves or to other bots.
        if author.bot or (bot_user_id is not None and author.id == bot_user_id):
            logger.debug("discord: ignoring message from bot account %s", author.id)
            return

        is_dm = message.guild is None
        text = message.content or ""
        attachments = list(message.attachments or ())

        if not is_dm:
            # With require_mention (the default) the bot only reacts when it
            # is itself mentioned (not merely @everyone/@here, which
            # mentioned_in would accept). Without it, every guild message
            # continues; the allowlist gate still refuses other senders.
            if self._require_mention_for(message.channel):
                if not any(u.id == bot_user_id for u in message.mentions):
                    logger.debug("discord: ignoring guild message that does not mention the bot")
                    return
                if not text and not attachments:
                    # Discord should deliver content for mentions even without
                    # the privileged intent; flag it if that ever changes.
                    logger.warning(
                        "discord: mention in channel %s arrived with empty content; dropping",
                        message.channel.id,
                    )
                    return
            # Remove the bot's own mention token (both forms); a no-op when
            # the message does not mention the bot.
            text = re.sub(rf"<@!?{bot_user_id}>", "", text)

        # Scope: DMs key on the user (stable across DM channel ids), servers
        # and threads on their own channel id.
        sender = str(author.id)
        scope = f"dm:{sender}" if is_dm else str(message.channel.id)
        # The name to put beside the id (#513): the guild nickname a person is
        # reading in this server, then their global display name, then the
        # username they log in with. Any of them is a string somebody else
        # chose, so core frames it as data and never as the message itself.
        sender_name = (
            getattr(author, "nick", None)
            or getattr(author, "global_name", None)
            or getattr(author, "name", None)
        )
        # The allowlist is checked before the channel is resolved, so a
        # refused sender never adds a registry entry (PRD §3.2 default-deny;
        # #274, as #103 did for Signal and #77 for IRC).
        allowed = sender in self._allow
        if allowed:
            channel = self._registry.get_or_create("discord", scope)
        else:
            channel = self._transient_channel(scope)
        # A name for the model to read, never the key (#513). A thread carries
        # its own title and the channel it hangs off, or "this is a thread in
        # #home" would be unanswerable from inside the thread.
        if not is_dm:
            channel.name = _channel_name(message.channel)
        # The name for this sender, learned here and read back by core when the
        # turn is built (#513). A cache of what the transport has seen, so a
        # rename is picked up on the message that carries it.
        if sender_name:
            remember_sender_name(channel.sender_names, sender, sender_name)

        # Attachments last, once the gate above has decided who this is: a
        # refused sender's file is named, never fetched (#103, #520).
        if attachments:
            note = await self._attachment_note(message, attachments, allowed)
            text = f"{note}\n{text}".strip() if text.strip() else note

        # Nothing left to say (empty, whitespace, or a bare mention).
        text = text.strip()
        if not text:
            logger.debug("discord: ignoring message with no text content")
            return

        # Typing only ever applies to an allowlisted sender -- others get no
        # sign the bot is live. on_message still fires so the real
        # should_process_message gate can veto and log the attempt.
        # Mint the trace id here; the Agent uses it verbatim (#95), and it
        # keys whether this message was admitted.
        trace_id = mint_trace_id()
        took_slot = allowed
        try:
            if took_slot:
                await self._begin_typing(channel)

            await self.pm.ahook.on_message(
                channel=channel, sender=sender, text=text, trace_id=trace_id,
            )
        finally:
            # Admission fires before on_message returns. Anything not
            # admitted by then (gate veto, agent not ready, an exception,
            # cancellation) gets no reply, so give its typing slot back --
            # but only if it took one (a stranger did not).
            if took_slot and trace_id not in self._admitted:
                self._decrement_outstanding(channel)
            self._admitted.discard(trace_id)

    def _require_mention_for(self, channel: Any) -> bool:
        """Whether this Discord channel answers only on an @-mention (#512).

        The transport value is the default. A `discord.channels.<id>` entry
        overrides it, and a thread looks up its parent's entry before falling
        back to the default, so a thread whose parent is not listed stays
        mention-only rather than silently becoming conversational.
        """
        own = str(getattr(channel, "id", ""))
        if own in self._channel_require_mention:
            return self._channel_require_mention[own]
        parent = getattr(channel, "parent_id", None)
        if parent is not None:
            listed = self._channel_require_mention.get(str(parent))
            if listed is not None:
                return listed
        return self._require_mention

    async def _attachment_note(
        self, message: Any, attachments: list, allowed: bool,
    ) -> str:
        """The note that stands in for attachments, one line per file.

        Downloaded images become ``[image: <path>]``; anything refused or
        failed becomes ``[image: download failed]``; other kinds are named and
        left where they are. Only an allowlisted sender's files are fetched.
        """
        notes: list[str] = []
        for _ in getattr(message, "stickers", None) or ():
            # Discord puts stickers outside .attachments; name them, never fetch.
            notes.append("[sticker attached]")
        session = None
        try:
            for index, att in enumerate(attachments):
                content_type = str(getattr(att, "content_type", "") or "").lower()
                is_image = content_type in _IMAGE_CONTENT_TYPES or (
                    not content_type and _attachment_kind(att) == "image"
                )
                if not allowed or not is_image:
                    notes.append(f"[{_attachment_kind(att)} attached]")
                    continue
                if session is None:
                    # One session for the message: a burst of images is the
                    # common case, and a session per file pays TLS each time.
                    timeout = aiohttp.ClientTimeout(total=ATTACHMENT_DOWNLOAD_TIMEOUT)
                    session = aiohttp.ClientSession(timeout=timeout)
                try:
                    data = await self._download_attachment(session, att)
                    path = None
                    if data is not None:
                        path = await save_attachment(
                            data,
                            self._attachment_dir,
                            f"{message.id}-{index}",
                            filename=getattr(att, "filename", None),
                            suffix=_IMAGE_SUFFIX.get(content_type, ""),
                            max_bytes=self._max_attachment_size,
                            transport="discord",
                        )
                except Exception:
                    # One odd file must not cost the message its text.
                    logger.warning("discord: attachment handling failed", exc_info=True)
                    path = None
                notes.append(f"[image: {path}]" if path else "[image: download failed]")
        finally:
            if session is not None:
                try:
                    await session.close()
                except Exception:
                    logger.debug("discord: closing the attachment session failed", exc_info=True)
        return "\n".join(notes)

    async def _download_attachment(self, session: Any, att: Any) -> bytes | None:
        """Fetch one attachment's bytes, or None. Never raises."""
        url = getattr(att, "url", None)
        if not url:
            logger.warning("discord: attachment has no url, refusing download")
            return None
        # Discord states the size up front; a stated oversize needs no fetch.
        size = getattr(att, "size", None)
        if isinstance(size, int) and size > self._max_attachment_size:
            logger.warning(
                "discord: attachment is %d bytes, over the %d byte cap",
                size, self._max_attachment_size,
            )
            return None
        try:
            async with session.get(str(url)) as resp:
                if resp.status != 200:
                    logger.warning(
                        "discord: attachment download returned %s", resp.status
                    )
                    return None
                # read(n) returns whatever one socket read delivered, so the
                # body has to be accumulated and the cap checked as it grows.
                buf = bytearray()
                async for chunk in resp.content.iter_chunked(_DOWNLOAD_CHUNK_BYTES):
                    buf += chunk
                    if len(buf) > self._max_attachment_size:
                        logger.warning(
                            "discord: attachment body over the %d byte cap",
                            self._max_attachment_size,
                        )
                        return None
                return bytes(buf) if buf else None
        except Exception:
            logger.warning("discord: attachment download failed", exc_info=True)
            return None

    def _transient_channel(self, scope: str) -> Channel:
        """The channel a refused sender's message is gated on.

        A live registry entry is reused (a guild channel an allowed user
        already talks in, or a sender since dropped from ``allow``);
        otherwise the Channel is built without registering it, so the gate
        still sees the ``discord:<scope>`` id it logs, and nothing outlives
        the refusal.
        """
        existing = self._registry.get(f"discord:{scope}")
        if existing is not None:
            return existing
        return Channel(transport="discord", scope=scope)

    # -----------------------------------------------------------------
    # Channel resolution
    # -----------------------------------------------------------------

    @hookimpl
    async def resolve_channel(self, channel_id: str) -> Channel | None:
        """Resolve a persisted ``discord:<scope>`` id to its live channel.

        Defers (None) for other transports, malformed scopes, and before
        on_start has resolved the registry.
        """
        transport, sep, scope = channel_id.partition(":")
        if transport != "discord" or not sep or not _SCOPE_RE.fullmatch(scope):
            return None
        if self._registry is None:
            return None
        return self._registry.get_or_create("discord", scope)

    async def _messageable(self, channel: Channel) -> Any:
        """Find the discord object to send to for a channel, or None.

        DMs go through the user (cache, then fetch) and create_dm; server
        channels and threads through the channel cache, then fetch. Lookup
        failures are logged and yield None.
        """
        client = self._client
        if client is None:
            logger.debug("discord: no client yet, cannot reach %s", channel.id)
            return None
        try:
            if channel.scope.startswith("dm:"):
                user_id = int(channel.scope[3:])
                user = client.get_user(user_id) or await client.fetch_user(user_id)
                return await user.create_dm()
            channel_id = int(channel.scope)
            return client.get_channel(channel_id) or await client.fetch_channel(channel_id)
        except Exception as exc:
            logger.warning("discord: cannot reach %s: %s", channel.id, exc, exc_info=True)
            return None

    # -----------------------------------------------------------------
    # Authorization gate
    # -----------------------------------------------------------------

    @hookimpl
    async def resolve_register_prompt(self, channel: Channel) -> str | None:
        """Default register: chat norms for DMs, plus group manners for
        server channels and threads."""
        if not self.owns(channel):
            return None
        # DM scopes are "dm:<user id>"; everything else is a server channel.
        if channel.scope.startswith("dm:"):
            return shipped_register("chat")
        return shipped_register("chat", "discord_group")

    @hookimpl
    async def should_process_message(
        self, channel: Channel, sender: str, text: str, trace_id: str
    ) -> bool | None:
        """Default-deny gate: only allowlisted user ids may drive the agent.
        trace_id is bound without a default -- see
        tests/test_hook_arg_binding.py."""
        if not channel.matches_transport("discord"):
            return None
        if sender in self._allow:
            return True
        logger.info("discord: rejected message from unauthorized sender %s", sender)
        return False

    @hookimpl
    async def on_message_admitted(
        self, channel: Channel, trace_id: str, sender: str, text: str
    ) -> None:
        """Note that this message was admitted, so the inbound handler's
        finally keeps its typing slot (send_message gives it back instead).
        Admission fires before on_message returns."""
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
        """Give back one outstanding slot for a message that was never admitted
        (a final, sent or unsent, releases every slot instead; see
        _release_all)."""
        count = self._outstanding.get(channel.id, 0)
        if count > 0:
            self._outstanding[channel.id] = count - 1

    async def _typing_refresh_loop(self, channel: Channel) -> None:
        """Re-send typing every TYPING_REFRESH_SECONDS while a reply is owed;
        stop once the count reaches zero or a send fails."""
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
        """Send one typing indicator. A failure is logged and never blocks
        the reply; returns whether it succeeded."""
        try:
            target = await self._messageable(channel)
            if target is None:
                return False
            await target.typing()
            return True
        except Exception as exc:
            logger.warning("discord: failed to send typing indicator: %s", exc, exc_info=True)
            return False

    # -----------------------------------------------------------------
    # Outbound send
    # -----------------------------------------------------------------

    async def _deliver(self, channel: Channel, text: str) -> None:
        """Split text to the chunk size and send each non-blank chunk, never
        allowing it to ping anyone."""
        target = await self._messageable(channel)
        if target is None:
            logger.warning("discord: no destination for %s, dropping message", channel.id)
            return
        for chunk in split_discord_message(text, self._message_chunk_size):
            # Discord rejects empty messages, so skip whitespace-only chunks.
            if not chunk.strip():
                continue
            try:
                # Injected text must not be able to mention @everyone, roles,
                # or users, whatever the client's default says.
                await target.send(chunk, allowed_mentions=_sdk().AllowedMentions.none())
            except Exception as exc:
                logger.warning("discord: failed to deliver message: %s", exc, exc_info=True)

    @hookimpl
    async def on_agent_response(
        self,
        channel,
        request_text: str,
        response_text: str,
        trace_id: str | None,
        meta: dict,
        logprobs: dict | None,
    ) -> None:
        """Release per-turn state for an unsent final (it never reaches
        send_message); a sent final is released there, so exactly once."""
        if not self.owns(channel):
            return
        # An unsent final never reaches send_message, so release here, in
        # the same synchronous step as the check (before any await).
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
            self._release_all(channel)

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Deliver a final reply and close out every outstanding slot: one
        reply answers a coalesced burst (#398)."""
        await self._deliver(channel, event.text)
        self._release_all(channel)

    def _release_all(self, channel: Channel) -> None:
        """Close out every outstanding slot on a final, sent or unsent: one
        final answers a coalesced burst (#398). The refresh loop ends when it
        next sees no slot owed; Discord has no stop frame to send."""
        # Only a channel that took a slot has an entry to clear.
        if self._outstanding.get(channel.id, 0) > 0:
            self._outstanding[channel.id] = 0

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        """Deliver intermediate assistant text without closing out the
        outstanding slots (the final reply still owes them)."""
        await self._deliver(channel, event.text)

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Offer the agent one way to post a file (#521).

        Only when the transport is configured: without a token there is no
        Discord to send to, and the tool would only ever say so.
        """
        if self._token is not None:
            tool_registry.append(self.send_attachment)

    async def send_attachment(
        self, channel_id: str, path: str, caption: str = "",
    ) -> str:
        """Send a file from the workspace to a Discord channel.

        Args:
            channel_id: The target channel in "discord:<scope>" form, as
                list_channels reports it.
            path: The file to send, relative to the working directory or
                inside send_attachment_root. Paths outside that root are
                refused.
            caption: Optional text to send alongside the file.

        Returns:
            A confirmation naming the file and channel, or an error saying
            what was wrong and how to fix it.
        """
        if self._registry is None:
            return "Could not send: the channel registry is not available."
        if not isinstance(channel_id, str) or not channel_id:
            return "Could not send: channel_id must be a non-empty string."
        if not isinstance(path, str) or not path.strip():
            return "Could not send: path must be a non-empty string."
        if caption is None:
            caption = ""
        if not isinstance(caption, str):
            return "Could not send: caption must be a string."

        channel = self._registry.get(channel_id)
        if channel is None or not self.owns(channel):
            return (
                f"Could not send: {channel_id!r} is not a Discord channel; "
                "the target must start with 'discord:'."
            )

        root = self._send_attachment_root.resolve()
        raw = Path(path)
        # A relative path is relative to the root, which is what the docstring
        # promises and what a reader of an error message expects.
        target_path = (raw if raw.is_absolute() else root / raw).resolve()
        if not target_path.is_relative_to(root):
            return (
                f"Could not send: {path!r} is outside the allowed root "
                f"{root}; put the file there or set discord.send_attachment_root."
            )
        try:
            if not target_path.is_file():
                return f"Could not send: {target_path} is not a readable file."
            size = target_path.stat().st_size
        except OSError as exc:
            return f"Could not send: cannot read {target_path} ({exc})."
        if size > self._max_attachment_size:
            return (
                f"Could not send: {target_path.name} is {size} bytes, over the "
                f"{self._max_attachment_size} byte limit "
                "(discord.max_attachment_size)."
            )

        messageable = await self._messageable(channel)
        if messageable is None:
            return f"Could not send: no live destination for {channel_id}."
        # The caption shares the message with the file, so it splits like any
        # other text: first chunk rides with the file, the rest follow it.
        chunks = [c for c in split_discord_message(caption, self._message_chunk_size) if c.strip()]
        try:
            # discord.File opens the file synchronously; at the 10 MB cap that
            # is a few milliseconds, so it stays on the loop.
            await messageable.send(
                content=chunks[0] if chunks else None,
                file=_sdk().File(str(target_path)),
                allowed_mentions=_sdk().AllowedMentions.none(),
            )
            for chunk in chunks[1:]:
                await messageable.send(chunk, allowed_mentions=_sdk().AllowedMentions.none())
        except Exception as exc:
            logger.warning("discord: send_attachment failed: %s", exc, exc_info=True)
            return f"Could not send: Discord rejected the upload ({exc})."
        return f"Sent {target_path.name} to {channel_id}."
