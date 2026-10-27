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
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import yaml

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.runtime import Runtime
from corvidae.trace import mint_trace_id

logger = logging.getLogger(__name__)

# Sidecar index for session/list (cwd + activity); lives in sessions.db.
# How often the stdio loop re-checks whether Agent has finished on_start.
AGENT_READY_POLL_SECONDS = 0.05

_ACP_SESSION_INDEX_DDL = """
CREATE TABLE IF NOT EXISTS acp_session_index (
    session_id TEXT PRIMARY KEY,
    cwd TEXT NOT NULL,
    updated_at REAL NOT NULL
)
"""


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
        from acp.schema import (
            AgentCapabilities,
            Implementation,
            SessionCapabilities,
            SessionCloseCapabilities,
            SessionListCapabilities,
            SessionResumeCapabilities,
        )

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
        # Phase 2 session durability capabilities (A16); delete stays off.
        return InitializeResponse(
            protocol_version=negotiated,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                    close=SessionCloseCapabilities(),
                ),
            ),
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

        # Reject a workspace we cannot use before creating any state.
        _validate_cwd(cwd)
        session_id = uuid4().hex
        _warn_ignored_session_input(session_id, mcp_servers, additional_directories)
        channel = self._plugin._registry.get_or_create("acp", session_id)
        channel.runtime_overrides["cwd"] = cwd
        await self._plugin.index_session(session_id, cwd)
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Enqueue user text via on_message and wait for the turn to finish."""
        from acp import PromptResponse, RequestError

        if self._plugin is None or self._plugin._registry is None:
            raise RuntimeError("CorvidaeAcpAgent.prompt requires an AcpPlugin with registry")

        # Unknown session is a protocol-level not-found, matching load/resume/close.
        channel = self._plugin._registry.get(f"acp:{session_id}")
        if channel is None:
            raise RequestError.resource_not_found(uri=f"acp:{session_id}")

        # Only text blocks become user text; make any dropped block visible,
        # and never enqueue an empty message.
        text, dropped = _flatten_prompt_text(prompt)
        if dropped:
            logger.warning(
                "dropped non-text ACP prompt blocks %s; only text blocks are supported",
                dropped,
                extra={"session_id": session_id, "block_types": dropped},
            )
        if not text.strip():
            raise RequestError.invalid_params(
                {
                    "prompt": "no text content; corvidae supports text prompt blocks only",
                    "block_types": dropped,
                }
            )
        # A second prompt would overwrite the live future and orphan the first.
        if self._plugin.prompt_in_flight(session_id):
            raise RequestError.invalid_request(
                {
                    "sessionId": session_id,
                    "reason": "a prompt is already in progress",
                }
            )
        future = self._plugin.begin_prompt(session_id)
        started = False
        try:
            await self._plugin.pm.ahook.on_message(
                channel=channel,
                sender="user",
                text=text,
                trace_id=mint_trace_id(),
            )
            started = True
        finally:
            # The turn never started: drop the future so a later prompt cannot
            # inherit it. The original exception reaches the client unchanged.
            if not started:
                self._plugin.abandon_prompt(session_id)
        return await future

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel an in-flight prompt for this session (A10)."""
        if self._plugin is None:
            return
        self._plugin.cancel_prompt(session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Restore ``acp:<sessionId>`` from sessions.db and replay history (A16)."""
        from acp import LoadSessionResponse
        from acp import update_agent_message_text, update_user_message_text
        from corvidae.context import MessageType

        channel, history = await self._reattach_session(
            session_id, cwd, mcp_servers, additional_directories
        )

        # Replay text turns to the client; leave channel.conversation None so
        # the agent loop lazy-loads the same DB rows on the next prompt.
        if history and self._conn is not None:
            for msg in history:
                text = _message_text(msg)
                if not text:
                    continue
                role = msg.get("role")
                msg_type = msg.get("_message_type")
                if role == "user":
                    update = update_user_message_text(text)
                elif role == "assistant" or msg_type == MessageType.SUMMARY:
                    update = update_agent_message_text(text)
                else:
                    # Skip tool/system rows — client sees text turns only.
                    continue
                await self._conn.session_update(session_id=session_id, update=update)

        return LoadSessionResponse()

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Reattach ``acp:<sessionId>`` without replaying history (A16)."""
        from acp.schema import ResumeSessionResponse

        await self._reattach_session(session_id, cwd, mcp_servers, additional_directories)
        return ResumeSessionResponse()

    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        """Cancel in-flight work and drop the in-memory channel (keep DB)."""
        from acp import RequestError
        from acp.schema import CloseSessionResponse

        if self._plugin is None or self._plugin._registry is None:
            raise RuntimeError("CorvidaeAcpAgent.close_session requires an AcpPlugin with registry")

        channel_id = f"acp:{session_id}"
        registry = self._plugin._registry
        if registry.get(channel_id) is None:
            raise RequestError.resource_not_found(uri=channel_id)

        # Same cancel semantics as session/cancel, then free registry resources.
        self._plugin.cancel_prompt(session_id)
        registry.channels.pop(channel_id, None)
        return CloseSessionResponse()

    async def _reattach_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
    ) -> tuple[Any, list | None]:
        """get_or_create the ACP channel, set cwd, load history; raise if unknown.

        Returns ``(channel, history)``. ``history`` may be ``None`` for a
        same-process empty session (e.g. just created via session/new).
        """
        from acp import RequestError

        if self._plugin is None or self._plugin._registry is None:
            raise RuntimeError("CorvidaeAcpAgent requires an AcpPlugin with registry")

        # Same workspace rules as session/new, before touching the registry.
        _validate_cwd(cwd)
        _warn_ignored_session_input(session_id, mcp_servers, additional_directories)
        registry = self._plugin._registry
        channel_id = f"acp:{session_id}"
        # Detect same-process live channels before get_or_create would mint one.
        existed = registry.get(channel_id) is not None
        channel = registry.get_or_create("acp", session_id)
        channel.runtime_overrides["cwd"] = cwd

        history = await self._plugin.pm.ahook.load_conversation(channel=channel)
        if history is None and not existed:
            # Brand-new id with no DB rows is not a loadable session — drop the
            # empty registry entry so a later call does not treat it as live.
            registry.channels.pop(channel_id, None)
            raise RequestError.resource_not_found(uri=channel_id)
        await self._plugin.index_session(session_id, cwd)
        return channel, history

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Discover indexed ACP sessions (full page; no pagination yet)."""
        from acp.schema import ListSessionsResponse, SessionInfo

        if self._plugin is None:
            raise RuntimeError("CorvidaeAcpAgent.list_sessions requires an AcpPlugin")

        # Phase 2: ignore cursor and return the full set (next_cursor=None).
        rows = await self._plugin.list_indexed_sessions(cwd=cwd)
        sessions = [
            SessionInfo(
                session_id=session_id,
                cwd=session_cwd,
                updated_at=_iso8601_utc(updated_at),
            )
            for session_id, session_cwd, updated_at in rows
        ]
        return ListSessionsResponse(sessions=sessions, next_cursor=None)


def _iso8601_utc(ts: float) -> str:
    """Format a unix timestamp as an ISO-8601 UTC string ending in Z."""
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _message_text(msg: dict) -> str:
    """Extract plain text from a persisted conversation message dict."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or ""
            else:
                text = getattr(block, "text", None) or ""
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return ""


def _flatten_prompt_text(prompt: list[Any]) -> tuple[str, list[str]]:
    """Join text blocks into one user message, and list the dropped block types."""
    parts: list[str] = []
    dropped: list[str] = []
    for block in prompt:
        if isinstance(block, dict):
            text = block.get("text") or ""
            block_type = block.get("type") or type(block).__name__
        else:
            text = getattr(block, "text", None) or ""
            block_type = getattr(block, "type", type(block).__name__)
        if text:
            parts.append(str(text))
        else:
            dropped.append(str(block_type))
    return "\n".join(parts), dropped


def _validate_cwd(cwd: str) -> None:
    """Reject a workspace cwd that is not an existing directory (invalid params).

    The cwd is recorded for ``session/list``; tools still run in the process
    working directory, so a differing cwd is logged rather than left silent.
    """
    from acp import RequestError

    path = Path(cwd)
    if not path.is_dir():
        raise RequestError.invalid_params(
            {"cwd": cwd, "reason": "not an existing directory"}
        )
    if path.resolve() != Path.cwd().resolve():
        logger.warning(
            "ACP session cwd differs from the process working directory; "
            "tools run in the process directory",
            extra={"cwd": cwd, "process_cwd": str(Path.cwd())},
        )


def _warn_ignored_session_input(
    session_id: str,
    mcp_servers: list[Any] | None,
    additional_directories: list[str] | None,
) -> None:
    """Log client fields this transport does not act on, so the gap is visible."""
    if mcp_servers:
        logger.warning(
            "ACP client sent mcpServers; corvidae does not mount per-session "
            "MCP servers — configure mcp: in agent.yaml instead",
            extra={"session_id": session_id, "mcp_server_count": len(mcp_servers)},
        )
    if additional_directories:
        logger.warning(
            "ACP client sent additionalDirectories; corvidae ignores them",
            extra={
                "session_id": session_id,
                "additional_directory_count": len(additional_directories),
            },
        )


class AcpPlugin(CorvidaePlugin):
    """Transport plugin for Agent Client Protocol sessions (``acp:<sessionId>``)."""

    depends_on = frozenset({"registry"})

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._conn: Any = None
        self._registry = None
        # ACP loop crashed (surfaced as a non-zero exit by the command).
        self._failed = False
        # on_stop is cancelling the loop: do not re-signal shutdown.
        self._stopping = False
        # session_id -> Future[PromptResponse] for the in-flight ACP prompt (A9).
        self._active_prompts: dict[str, asyncio.Future] = {}
        # (channel.id, tool_name) -> synthetic toolCallId for ACP updates.
        self._tool_call_ids: dict[tuple[str, str], str] = {}

    def _persistence_db(self):
        """Return PersistencePlugin.db when available, else None."""
        if self.pm is None:
            return None
        plugin = self.pm.get_plugin("persistence")
        if plugin is None:
            return None
        return getattr(plugin, "db", None)

    async def index_session(self, session_id: str, cwd: str) -> None:
        """Upsert cwd into acp_session_index (fail-soft if no DB)."""
        db = self._persistence_db()
        if db is None:
            return
        try:
            await db.execute(_ACP_SESSION_INDEX_DDL)
            await db.execute(
                "INSERT INTO acp_session_index (session_id, cwd, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "cwd = excluded.cwd, updated_at = excluded.updated_at",
                (session_id, cwd, time.time()),
            )
            await db.commit()
        except Exception:
            logger.warning(
                "failed to upsert acp_session_index",
                exc_info=True,
                extra={"session_id": session_id},
            )

    async def list_indexed_sessions(
        self, cwd: str | None = None
    ) -> list[tuple[str, str, float]]:
        """Return (session_id, cwd, updated_at) rows for session/list.

        Prefers the durable index. When persistence is absent, falls back to
        in-registry ACP channels so unit tests without a DB still work.
        """
        db = self._persistence_db()
        if db is not None:
            try:
                await db.execute(_ACP_SESSION_INDEX_DDL)
                if cwd is None:
                    sql = (
                        "SELECT session_id, cwd, updated_at FROM acp_session_index "
                        "ORDER BY updated_at DESC"
                    )
                    params: tuple = ()
                else:
                    sql = (
                        "SELECT session_id, cwd, updated_at FROM acp_session_index "
                        "WHERE cwd = ? ORDER BY updated_at DESC"
                    )
                    params = (cwd,)
                async with db.execute(sql, params) as cursor:
                    return list(await cursor.fetchall())
            except Exception:
                logger.warning("failed to read acp_session_index", exc_info=True)

        # No persistence: list live registry ACP channels only.
        if self._registry is None:
            return []
        rows: list[tuple[str, str, float]] = []
        for channel in self._registry.by_transport("acp"):
            channel_cwd = channel.runtime_overrides.get("cwd") or ""
            if cwd is not None and channel_cwd != cwd:
                continue
            rows.append((channel.scope, channel_cwd, channel.last_active))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows

    def prompt_in_flight(self, session_id: str) -> bool:
        """Whether this session already has an unresolved prompt future."""
        future = self._active_prompts.get(session_id)
        return future is not None and not future.done()

    def begin_prompt(self, session_id: str) -> asyncio.Future:
        """Register a Future that send_message resolves when the turn is idle."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._active_prompts[session_id] = future
        return future

    def abandon_prompt(self, session_id: str) -> None:
        """Drop an unstarted prompt future so a later prompt cannot inherit it."""
        future = self._active_prompts.pop(session_id, None)
        if future is not None and not future.done():
            future.cancel()

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

    def _agent_ready(self) -> bool:
        """True when no Agent is registered or it has finished on_start."""
        agent = self.pm.get_plugin("agent")
        return agent is None or bool(getattr(agent, "started", True))

    async def _wait_for_agent(self) -> None:
        """Poll until the Agent can accept on_message (see Agent.started)."""
        while not self._agent_ready():
            await asyncio.sleep(AGENT_READY_POLL_SECONDS)

    async def _run_acp(self, agent: CorvidaeAcpAgent) -> None:
        """Run the official SDK agent loop on stdio until the client disconnects."""
        from acp import run_agent

        try:
            # Runtime starts Agent only after the on_start broadcast this task
            # begins in. A prompt read before then is dropped by the Agent and
            # its future never resolves, so wait before reading stdio; the
            # client's requests sit in the pipe meanwhile.
            await self._wait_for_agent()
            await run_agent(agent)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._failed = True
            logger.exception("ACP stdio server exited with error")
        else:
            logger.info("ACP client disconnected, initiating shutdown")
        finally:
            # ACP owns stdin/stdout: once the loop ends the process cannot do
            # anything useful, so always ask Runtime.run() to stop. on_stop
            # sets _stopping before cancelling, so that exit must stay quiet.
            if not self._stopping:
                self._request_shutdown()

    def _request_shutdown(self) -> None:
        """Ask Runtime.run() to stop, reusing its SIGINT handler (CLI EOF does)."""
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except OSError:
            logger.exception("failed to signal shutdown after the ACP loop exited")

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

        try:
            await self._session_update(channel, update_agent_message(text_block(text)))
        finally:
            # The turn is over even if the client connection died mid-update;
            # leaving the future pending would hang the ACP prompt forever.
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
        # Runtime is already stopping; the loop's exit must not re-signal.
        self._stopping = True
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

    # Fail on an unusable config path before starting, with one clean line:
    # an editor surfaces stderr, not a Python traceback.
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Error: config file not found: {config}", err=True)
        sys.exit(1)
    if not config_path.is_file():
        click.echo(f"Error: config path is not a regular file: {config}", err=True)
        sys.exit(1)

    # ACP mode: keep JSON-RPC on stdout; route logs to a file (A2).
    runtime = Runtime(
        config_path=config,
        overrides={
            "_acp_mode": True,
            "logging": {"file": "corvidae-acp.log"},
        },
    )
    try:
        asyncio.run(runtime.run())
    except yaml.YAMLError as exc:
        click.echo(f"Error: invalid YAML in {config}: {exc}", err=True)
        sys.exit(1)
    except ValueError as exc:
        # Config validation raises ValueError (removed keys, bad values).
        click.echo(f"Error: invalid config {config}: {exc}", err=True)
        sys.exit(1)

    # A crashed ACP loop asked the runtime to stop; exit non-zero so the client
    # sees a failed provider instead of a process that still looks alive.
    plugin = runtime.pm.get_plugin("acp") if runtime.pm is not None else None
    if getattr(plugin, "_failed", False):
        click.echo(
            "Error: the ACP server exited unexpectedly; see corvidae-acp.log "
            "next to the config file.",
            err=True,
        )
        sys.exit(1)
