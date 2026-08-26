"""ACP transport: ``corvidae acp`` command and ``AcpPlugin``.

The click command requires the optional ``agent-client-protocol`` extra.
``AcpPlugin`` loads with the rest of the daemon but stays inert unless
``config["_acp_mode"]`` is set by the command — so ``serve`` / ``cli`` never
steal stdin/stdout. The ``acp`` package is imported lazily so missing the
extra does not break other subcommands.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import sys
from typing import Any

import click

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.runtime import Runtime

logger = logging.getLogger(__name__)


def _package_version() -> str:
    """Best-effort corvidae version for ACP agentInfo."""
    try:
        return importlib.metadata.version("corvidae")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


class CorvidaeAcpAgent:
    """ACP Agent adapter bridging JSON-RPC sessions to Corvidae channels."""

    def __init__(
        self,
        agent_info: dict[str, str] | None = None,
        *,
        plugin: AcpPlugin | None = None,
    ) -> None:
        # agent_info keys: name, title, version (optional overrides).
        self._agent_info = dict(agent_info or {})
        self._plugin = plugin
        self._conn: Any = None

    def on_connect(self, conn: Any) -> None:
        """SDK calls this when the client connection is ready."""
        self._conn = conn
        if self._plugin is not None:
            self._plugin._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Negotiate ACP v1 and advertise empty authMethods (D4)."""
        # Lazy import so loading this module without the extra still works.
        from acp import PROTOCOL_VERSION, InitializeResponse
        from acp.schema import AgentCapabilities, Implementation

        info = self._agent_info
        agent_info = Implementation(
            name=info.get("name", "corvidae"),
            title=info.get("title", "Corvidae"),
            version=info.get("version", _package_version()),
        )
        # Echo a version we support; prefer PROTOCOL_VERSION when client asks higher.
        negotiated = (
            protocol_version
            if protocol_version <= PROTOCOL_VERSION
            else PROTOCOL_VERSION
        )
        return InitializeResponse(
            protocol_version=negotiated,
            agent_capabilities=AgentCapabilities(),
            agent_info=agent_info,
            auth_methods=[],
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create an ``acp:<sessionId>`` channel and record the workspace cwd."""
        from uuid import uuid4

        from acp import NewSessionResponse

        if self._plugin is None or self._plugin._registry is None:
            raise RuntimeError("CorvidaeAcpAgent.new_session requires an AcpPlugin with registry")

        session_id = uuid4().hex
        channel = self._plugin._registry.get_or_create("acp", session_id)
        channel.runtime_overrides["cwd"] = cwd
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Enqueue user text via on_message and wait for the turn to finish."""
        from acp import PromptResponse

        if self._plugin is None or self._plugin._registry is None:
            raise RuntimeError("CorvidaeAcpAgent.prompt requires an AcpPlugin with registry")

        channel = self._plugin._registry.get(f"acp:{session_id}")
        if channel is None:
            raise RuntimeError(f"unknown ACP session {session_id!r}")

        text = _flatten_prompt_text(prompt)
        future = self._plugin.begin_prompt(session_id)
        await self._plugin.pm.ahook.on_message(
            channel=channel,
            sender="user",
            text=text,
        )
        return await future

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel an in-flight prompt for this session (A10)."""
        if self._plugin is None:
            return
        self._plugin.cancel_prompt(session_id)


def _flatten_prompt_text(prompt: list[Any]) -> str:
    """Join text content blocks from an ACP prompt into one user message."""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, dict):
            text = block.get("text") or ""
        else:
            text = getattr(block, "text", None) or ""
        if text:
            parts.append(str(text))
    return "\n".join(parts)


class AcpPlugin(CorvidaePlugin):
    """Transport plugin for Agent Client Protocol sessions (``acp:<sessionId>``)."""

    depends_on = frozenset({"registry"})

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._conn: Any = None
        self._registry = None
        # session_id -> Future[PromptResponse] for the in-flight ACP prompt (A9).
        self._active_prompts: dict[str, asyncio.Future] = {}
        # (channel.id, tool_name) -> synthetic toolCallId for ACP updates.
        self._tool_call_ids: dict[tuple[str, str], str] = {}

    def begin_prompt(self, session_id: str) -> asyncio.Future:
        """Register a Future that send_message resolves when the turn is idle."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._active_prompts[session_id] = future
        return future

    def cancel_prompt(self, session_id: str) -> None:
        """Resolve an in-flight prompt as cancelled; no-op if none is active."""
        from acp import PromptResponse

        # Clear pending tools so a late send_message cannot race to end_turn.
        if self._registry is not None:
            channel = self._registry.get(f"acp:{session_id}")
            if channel is not None:
                channel.pending_tool_call_ids.clear()

        future = self._active_prompts.pop(session_id, None)
        if future is None or future.done():
            return
        future.set_result(PromptResponse(stop_reason="cancelled"))

    def _complete_prompt_if_idle(self, channel) -> None:
        """Resolve the active prompt when no tool calls remain for this session."""
        from acp import PromptResponse

        session_id = channel.scope
        future = self._active_prompts.get(session_id)
        if future is None or future.done():
            return
        if channel.pending_tool_call_ids:
            return
        future.set_result(PromptResponse(stop_reason="end_turn"))
        self._active_prompts.pop(session_id, None)

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Resolve registry; store pm/config via base class."""
        await super().on_init(pm, config)
        from corvidae.channel import ChannelRegistry

        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Start the ACP stdio server only when ``_acp_mode`` is set."""
        # Inert under serve/cli — never touch stdin/stdout without ACP mode (A4).
        if not config.get("_acp_mode"):
            logger.debug("AcpPlugin: not in _acp_mode, skipping ACP server")
            return

        acp_cfg = config.get("acp") or {}
        info = dict(acp_cfg.get("agent_info") or {})
        agent = CorvidaeAcpAgent(agent_info=info, plugin=self)
        self._task = asyncio.create_task(
            self._run_acp(agent),
            name="acp-stdio-server",
        )

    async def _run_acp(self, agent: CorvidaeAcpAgent) -> None:
        """Run the official SDK agent loop on stdio until the client disconnects."""
        from acp import run_agent

        try:
            await run_agent(agent)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ACP stdio server exited with error")
            raise
        else:
            # Client closed the connection — ask Runtime.run() to shut down
            # (same pattern as CLI EOF).
            logger.info("ACP client disconnected, initiating shutdown")
            try:
                import os
                import signal

                os.kill(os.getpid(), signal.SIGINT)
            except OSError:
                logger.exception("failed to signal shutdown after ACP disconnect")

    async def _session_update(self, channel, update: Any) -> None:
        """Send a session/update if we have a live ACP connection."""
        if self._conn is None:
            logger.debug("AcpPlugin: no ACP connection for session_update")
            return
        await self._conn.session_update(session_id=channel.scope, update=update)

    @hookimpl
    async def send_message(
        self, channel, text: str, latency_ms: float | None = None
    ) -> None:
        """Forward final assistant text and complete the ACP prompt when idle."""
        # Broadcast-filter: ignore other transports (A11).
        if not channel.matches_transport("acp"):
            return
        if self._conn is None:
            logger.debug("AcpPlugin.send_message: no ACP connection yet")
            # Still allow tests / prompt completion without a live client.
            self._complete_prompt_if_idle(channel)
            return
        from acp import text_block, update_agent_message

        await self._session_update(channel, update_agent_message(text_block(text)))
        self._complete_prompt_if_idle(channel)

    @hookimpl
    async def send_thinking(self, channel, text: str) -> None:
        """Map reasoning content to agent_thought_chunk updates."""
        if not channel.matches_transport("acp"):
            return
        from acp import update_agent_thought_text

        await self._session_update(channel, update_agent_thought_text(text))

    @hookimpl
    async def send_progress(self, channel, text: str) -> None:
        """Map intermediate assistant text to agent_message_chunk updates."""
        if not channel.matches_transport("acp"):
            return
        from acp import text_block, update_agent_message

        await self._session_update(channel, update_agent_message(text_block(text)))

    @hookimpl
    async def send_tool_status(
        self,
        channel,
        tool_name: str,
        status: str,
        args_summary: str | None = None,
        result_summary: str | None = None,
    ) -> None:
        """Map tool lifecycle events to tool_call / tool_call_update."""
        if not channel.matches_transport("acp"):
            return
        from uuid import uuid4

        from acp import start_tool_call, tool_content, text_block, update_tool_call

        key = (channel.id, tool_name)
        if status == "dispatched":
            tool_call_id = uuid4().hex
            self._tool_call_ids[key] = tool_call_id
            update = start_tool_call(
                tool_call_id,
                tool_name,
                kind="other",
                status="pending",
                raw_input=args_summary,
            )
            await self._session_update(channel, update)
        elif status == "completed":
            tool_call_id = self._tool_call_ids.pop(key, uuid4().hex)
            content = None
            if result_summary:
                content = [tool_content(text_block(result_summary))]
            update = update_tool_call(
                tool_call_id,
                status="completed",
                raw_output=result_summary,
                content=content,
            )
            await self._session_update(channel, update)

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the ACP server task if it was started."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("AcpPlugin ACP task raised on shutdown")
        self._task = None


@click.command("acp")
@click.option("--config", default="agent.yaml", help="Path to config file")
def acp_command(config: str) -> None:
    """Start corvidae as an ACP agent on stdio (editor / IDE clients)."""
    # Require the official ACP SDK before owning stdin/stdout for JSON-RPC.
    try:
        import acp  # noqa: F401
    except ImportError:
        click.echo(
            "The ACP SDK is not installed. Install the optional extra:\n"
            "  uv sync --extra acp\n"
            "  # or: uv sync --extra acp --extra dev",
            err=True,
        )
        sys.exit(1)

    # ACP mode: keep JSON-RPC on stdout; route logs to a file (A2).
    runtime = Runtime(
        config_path=config,
        overrides={
            "_acp_mode": True,
            "logging": {"file": "corvidae-acp.log"},
        },
    )
    asyncio.run(runtime.run())
