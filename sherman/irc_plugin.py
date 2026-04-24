"""IRC transport plugin using pydle library."""

import asyncio
import logging
import re
from typing import Optional

import pydle

logger = logging.getLogger(__name__)

from sherman.channel import Channel
from sherman.hooks import hookimpl


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

    async def on_connect(self):
        """Join configured channels after connecting."""
        await super().on_connect()
        for channel in self.plugin.channels:
            await self.join(channel)

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
                except (OSError, pydle.Error) as exc:
                    logger.warning("IRC connection error, retrying: %s", exc)
                if self.plugin._connect_task is None:
                    return
                await asyncio.sleep(delay)
                delay = min(delay * self._BACKOFF_MULTIPLIER, self._BACKOFF_CAP)
        except asyncio.CancelledError:
            if self.plugin._connect_task is None:
                return
            raise


class IRCPlugin:
    """IRC transport plugin following CLIPlugin pattern."""

    def __init__(self, pm):
        self.pm = pm
        self.client: Optional[IRCClient] = None
        self._connect_task: Optional[asyncio.Task] = None
        self.channels: list[str] = []

    @hookimpl
    async def on_start(self, config: dict) -> None:
        irc_config = config.get("irc")
        if irc_config is None:
            return

        server = irc_config.get("host", "irc.libera.chat")
        port = irc_config.get("port", 6667)
        nick = irc_config.get("nick", "sherman")
        self.channels = irc_config.get("channels", [])
        tls = irc_config.get("tls", False)

        self.client = IRCClient(self, nick, server=server, port=port, tls=tls)
        self._connect_task = asyncio.create_task(self.client.connect_with_retry())

    async def on_channel_message(self, target, by, message):
        """Forward channel messages to agent loop (called by IRCClient)."""
        if self.client is not None and by == self.client.nickname:
            return
        channel = self.pm.registry.get_or_create("irc", target)
        await self.pm.ahook.on_message(
            channel=channel,
            sender=by,
            text=message,
        )

    async def on_private_message(self, target, by, message):
        """Forward private messages to agent loop (called by IRCClient)."""
        if self.client is not None and by == self.client.nickname:
            return
        channel = self.pm.registry.get_or_create("irc", by)
        await self.pm.ahook.on_message(
            channel=channel,
            sender=by,
            text=message,
        )

    @hookimpl
    async def send_message(self, channel: Channel, text: str) -> None:
        if not channel.matches_transport("irc"):
            return
        if self.client is None or not self.client.connected:
            return
        chunks = split_message(text)
        for chunk in chunks:
            if not chunk.strip():
                continue
            await self.client.message(channel.scope, chunk)

    @hookimpl
    async def on_stop(self) -> None:
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


def split_message(text: str, max_len: int = 400) -> list[str]:
    """Split text into chunks that fit within max_len UTF-8 bytes.

    Three-tier splitting: paragraphs → sentences → words.
    Oversized words are split into multiple max_len chunks.
    Preserves all whitespace and separators when reassembling chunks.
    """
    if len(text.encode('utf-8')) <= max_len:
        return [text]

    chunks = []

    # Tier 1: Try splitting on paragraph boundaries (\n\n)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        current = ""
        for i, para in enumerate(paragraphs):
            # For the first paragraph, don't add separator
            # For subsequent paragraphs, add separator before the paragraph
            if i == 0:
                candidate = current + para
            else:
                candidate = current + '\n\n' + para
            if len(candidate.encode('utf-8')) <= max_len:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Recursively split the paragraph that doesn't fit
                # If this is not the first paragraph, the recursive result needs a leading separator
                sub_chunks = split_message(para, max_len)
                if i > 0 and sub_chunks:
                    sub_chunks[0] = '\n\n' + sub_chunks[0]
                chunks.extend(sub_chunks)
                current = ""
        if current:
            chunks.append(current)
        return chunks

    # Tier 2: Try splitting on sentence boundaries (.!? + whitespace)
    # Use lookahead to preserve the whitespace after sentence endings
    sentences = re.split(r'(?<=[.!?])(\s+)', text)
    if len(sentences) > 1:
        current = ""
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # Include the whitespace that follows the sentence
            if i + 1 < len(sentences) and re.match(r'^\s+$', sentences[i + 1]):
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1

            candidate = current + sentence
            if len(candidate.encode('utf-8')) <= max_len:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                chunks.extend(split_message(sentence, max_len))
                current = ""
        if current:
            chunks.append(current)
        return chunks

    # Tier 3: Split on word boundaries (spaces)
    words = text.split(' ')
    current = ""
    for i, word in enumerate(words):
        # Track if this word should have a leading space
        has_leading_space = i > 0

        # Check if this word itself exceeds max_len
        if len(word.encode('utf-8')) > max_len:
            # Output current chunk first
            if current:
                # If this is not the first word, add trailing space to preserve it
                if has_leading_space and not current.endswith(' '):
                    current = current + ' '
                chunks.append(current)
                current = ""

            # Split the oversized word into max_len chunks
            # Note: we don't include the leading space in the oversized word chunks
            # to avoid exceeding max_len. The space was added to the previous chunk.
            word_bytes = word.encode('utf-8')
            for start in range(0, len(word_bytes), max_len):
                chunk_bytes = word_bytes[start:start + max_len]
                chunk = chunk_bytes.decode('utf-8', errors='ignore')
                chunks.append(chunk)
            # Continue to next word (don't add space handling below)
            continue

        # Add space for non-first words
        if has_leading_space:
            word = ' ' + word
        candidate = current + word
        if len(candidate.encode('utf-8')) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)

    return chunks if chunks else [text]
