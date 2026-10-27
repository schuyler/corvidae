"""Discord transport plugin, built on discord.py's gateway client.

Connects to Discord as a bot account and talks in DMs, server channels, and
threads. In servers the bot only reacts when it is mentioned; DMs always
reach it. Only allowlisted Discord user ids may drive the agent.

The client deliberately does not request the privileged Message Content
intent: Discord still delivers message content for DMs and for messages that
mention the bot, which are the only messages this transport processes.

The SDK (discord-py) is optional: ``uv sync --extra discord``. This module
never imports it at load time, because pluggy loads every entry point
eagerly and an ImportError here would stop every install without the extra
from starting. It is imported on first use, and ``on_init`` fails with an
install hint when a ``discord:`` block is configured without it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import functools
import re
from types import ModuleType
from typing import Any, Optional

from corvidae.channel import Channel, ChannelRegistry
from corvidae.channels.split import split_message
from corvidae.trace import mint_trace_id
from corvidae.hooks import TransportPlugin, get_dependency, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress

logger = logging.getLogger("corvidae.discord_plugin")

# How often the per-channel typing-indicator refresh loop re-sends "typing"
# while a turn is outstanding. Discord's indicator lasts about 10s, so 8s
# keeps it continuous. Read by name (not bound as a default argument) so that
# tests patching this module attribute take effect.
TYPING_REFRESH_SECONDS = 8

# Discord's hard limit on a single message's length, in characters.
DISCORD_MESSAGE_LIMIT = 2000

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


class DiscordPlugin(TransportPlugin):
    """Transport plugin for Discord DMs, server channels, and threads."""

    transport = "discord"

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._registry: Optional[ChannelRegistry] = None

        # Config, populated by on_init. _token stays None when the discord:
        # block is absent -- that is the inert-plugin signal on_start checks.
        self._token: Optional[str] = None
        self._allow: set[str] = set()
        self._message_chunk_size: int = DISCORD_MESSAGE_LIMIT

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
                "Discord SDK (discord-py) is not installed. Install it with: "
                "uv sync --extra discord"
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

        self._token = str(token)
        # Ids are compared as strings against str(author.id); YAML parses
        # them as ints.
        self._allow = {str(entry) for entry in allow}
        self._message_chunk_size = chunk_size

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
        """Build a gateway client. Default intents only: no privileged
        Message Content intent (see the module docstring)."""
        return _client_class()(self, intents=_sdk().Intents.default())

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

        # Attachments are not supported; note and discard them.
        if message.attachments:
            logger.debug("discord: dropping %d attachment(s)", len(message.attachments))

        is_dm = message.guild is None
        text = message.content or ""

        if not is_dm:
            # In servers the bot only reacts when it is itself mentioned
            # (not merely @everyone/@here, which mentioned_in would accept).
            if not any(u.id == bot_user_id for u in message.mentions):
                logger.debug("discord: ignoring guild message that does not mention the bot")
                return
            if not text:
                # Discord should deliver content for mentions even without
                # the privileged intent; flag it if that ever changes.
                logger.warning(
                    "discord: mention in channel %s arrived with empty content; dropping",
                    message.channel.id,
                )
                return
            # Remove the bot's own mention token (both forms).
            text = re.sub(rf"<@!?{bot_user_id}>", "", text)

        # Nothing left to say (empty, whitespace, attachment-only, or a bare mention).
        text = text.strip()
        if not text:
            logger.debug("discord: ignoring message with no text content")
            return

        # Scope: DMs key on the user (stable across DM channel ids), servers
        # and threads on their own channel id.
        sender = str(author.id)
        scope = f"dm:{sender}" if is_dm else str(message.channel.id)
        channel = self._registry.get_or_create("discord", scope)

        # Typing only ever applies to an allowlisted sender -- others get no
        # sign the bot is live. on_message still fires so the real
        # should_process_message gate can veto and log the attempt.
        if sender in self._allow:
            await self._begin_typing(channel)

        # Mint the trace id here; the Agent uses it verbatim (#95).
        await self.pm.ahook.on_message(
            channel=channel, sender=sender, text=text,
            trace_id=mint_trace_id(),
        )

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
    async def on_message_rejected(
        self, channel: Channel, trace_id: str, sender: str, text: str
    ) -> None:
        """Give back the outstanding slot of a message another gate vetoed,
        so the typing refresh loop still reaches zero.

        Only an allowlisted sender ever took a slot (_handle_message counts
        nobody else), so a refused stranger's rejection gives nothing back --
        otherwise it would stop typing for an allowlisted user's turn that
        is still running in the same channel."""
        if not channel.matches_transport("discord"):
            return
        if sender not in self._allow:
            return
        self._decrement_outstanding(channel)

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
        """Give back one outstanding slot (a reply sent or a rejection)."""
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

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        """Deliver a final reply and close out one outstanding slot."""
        await self._deliver(channel, event.text)
        self._decrement_outstanding(channel)

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        """Deliver intermediate assistant text without closing out the
        outstanding slot (the final reply still owes one)."""
        await self._deliver(channel, event.text)
