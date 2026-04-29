# corvidae/channels/cli.py

import asyncio
import logging
import os
import signal
import sys

from corvidae.channel import ChannelRegistry
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger(__name__)


class CLIPlugin(CorvidaePlugin):
    """Transport plugin for stdin/stdout interaction.

    Implements on_start, send_message, and on_stop hooks. Only active when
    at least one cli channel is configured.
    """

    depends_on = frozenset({"registry"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self._task: asyncio.Task | None = None
        self._registry: ChannelRegistry | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Start the stdin read loop if any cli channels are configured."""
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        if not self._registry.by_transport("cli"):
            logger.debug("CLIPlugin: no cli channels configured, skipping read loop")
            return
        self._task = asyncio.create_task(self._read_loop(), name="cli-read-loop")

    async def _read_loop(self) -> None:
        """Read lines from stdin and dispatch as on_message events.

        Always routes to the cli:local channel. CLI is single-user/single-scope
        by design — only cli:local receives stdin input.
        """
        channel = self._registry.get_or_create("cli", "local")
        print("Agent ready. Type a message, or Ctrl-D to quit.\n")

        # Print initial prompt — subsequent prompts appear in send_message
        # after each response, since on_message is now fire-and-enqueue
        # and returns immediately before the agent loop runs.
        sys.stdout.write("> ")
        sys.stdout.flush()

        loop = asyncio.get_running_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except asyncio.CancelledError:
                raise
            if not line:
                # EOF (Ctrl-D or closed pipe) — trigger graceful shutdown
                # by sending SIGINT to ourselves, reusing the signal handler
                # in main().
                logger.info("EOF received, initiating shutdown")
                os.kill(os.getpid(), signal.SIGINT)
                return
            text = line.strip()
            if not text:
                sys.stdout.write("> ")
                sys.stdout.flush()
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
        (e.g. `(32.5s)`). Prints the next `>` prompt after the response so
        timing is correct now that on_message is fire-and-enqueue.
        """
        # Broadcast-filter: pluggy calls all transports; return early if this
        # channel does not belong to the CLI transport.
        if not channel.matches_transport("cli"):
            return
        if latency_ms is not None:
            print(f"\n{text}\n\033[2m({latency_ms/1000:.1f}s)\033[0m\n")
        else:
            print(f"\n{text}\n")
        sys.stdout.write("> ")
        sys.stdout.flush()

    @hookimpl
    async def send_thinking(self, channel, text: str) -> None:
        """Display reasoning/thinking content in bright blue."""
        if not channel.matches_transport("cli"):
            return
        print(f"[94m{text}[0m")
        sys.stdout.flush()

    @hookimpl
    async def send_tool_status(self, channel, tool_name: str, status: str, args_summary: str | None = None, result_summary: str | None = None) -> None:
        """Display tool call status in dim yellow."""
        if not channel.matches_transport("cli"):
            return
        if status == "dispatched":
            args_display = ""
            if args_summary:
                # Try to pretty-print JSON args, truncate
                try:
                    import json
                    parsed = json.loads(args_summary)
                    args_display = " " + json.dumps(parsed, ensure_ascii=False)[:120]
                except (json.JSONDecodeError, ValueError):
                    args_display = " " + args_summary[:120]
            print(f"\033[95m⚙ {tool_name}{args_display}\033[0m")
        elif status == "completed":
            result_display = ""
            if result_summary:
                result_display = f" → {result_summary[:80]}"
            print(f"\033[95m✓ {tool_name}{result_display}\033[0m")
        sys.stdout.flush()

    @hookimpl
    async def send_progress(self, channel, text: str) -> None:
        """Display intermediate assistant text in grey before tool dispatch."""
        if not channel.matches_transport("cli"):
            return
        print(f"\033[90m{text}\033[0m")
        sys.stdout.flush()

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
