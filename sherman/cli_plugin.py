# sherman/cli_plugin.py

import asyncio
import logging
import sys

from sherman.hooks import hookimpl

logger = logging.getLogger(__name__)


class CLIPlugin:
    """Transport plugin for stdin/stdout interaction.

    Implements on_start, send_message, and on_stop hooks. Only active when
    at least one cli channel is configured.
    """

    def __init__(self, pm) -> None:
        self.pm = pm
        self._task: asyncio.Task | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Start the stdin read loop if any cli channels are configured."""
        if not self.pm.registry.by_transport("cli"):
            logger.debug("CLIPlugin: no cli channels configured, skipping read loop")
            return
        self._task = asyncio.create_task(self._read_loop(), name="cli-read-loop")

    async def _read_loop(self) -> None:
        """Read lines from stdin and dispatch as on_message events.

        Always routes to the cli:local channel. CLI is single-user/single-scope
        by design — only cli:local receives stdin input.
        """
        channel = self.pm.registry.get_or_create("cli", "local")
        print("Agent ready. Type a message, or Ctrl-D to quit.\n")

        loop = asyncio.get_running_loop()
        while True:
            sys.stdout.write("> ")
            sys.stdout.flush()
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except asyncio.CancelledError:
                raise
            if not line:
                # EOF (Ctrl-D or closed pipe)
                break
            text = line.strip()
            if not text:
                continue
            await self.pm.ahook.on_message(
                channel=channel,
                sender="user",
                text=text,
            )

    @hookimpl
    async def send_message(self, channel, text: str, latency_ms: float | None = None) -> None:
        """Print agent response to stdout if this is a cli channel.

        If latency_ms is provided, appends a dim ANSI-formatted timing line
        (e.g. `(32.5s)`).
        """
        if not channel.matches_transport("cli"):
            return
        if latency_ms is not None:
            print(f"\n{text}\n\033[2m({latency_ms/1000:.1f}s)\033[0m\n")
        else:
            print(f"\n{text}\n")

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the read loop and await its cancellation."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("CLIPlugin read loop raised unexpected exception")
        self._task = None
