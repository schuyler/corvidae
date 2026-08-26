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
    """ACP Agent adapter (initialize only in WP-A0.2; sessions in WP-A1.x)."""

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


class AcpPlugin(CorvidaePlugin):
    """Transport plugin for Agent Client Protocol sessions (``acp:<sessionId>``)."""

    depends_on = frozenset({"registry"})

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._conn: Any = None
        self._registry = None

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

    @hookimpl
    async def send_message(
        self, channel, text: str, latency_ms: float | None = None
    ) -> None:
        """Forward assistant text to the ACP client for acp channels only."""
        # Broadcast-filter: ignore other transports (A11).
        if not channel.matches_transport("acp"):
            return
        # Session/prompt streaming lands in WP-A1.1; filter is enough for A0.2.
        if self._conn is None:
            logger.debug("AcpPlugin.send_message: no ACP connection yet")
            return
        from acp import text_block, update_agent_message

        session_id = channel.scope
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_message(text_block(text)),
        )

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
