"""ACP transport command (WP-A0.1) and plugin surface (later WPs).

The ``acp`` click command starts a Runtime in ACP mode. The optional
``agent-client-protocol`` extra must be installed; otherwise the command
exits with an install hint and never touches stdio as an ACP server.
"""

from __future__ import annotations

import asyncio
import sys

import click

from corvidae.runtime import Runtime


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
