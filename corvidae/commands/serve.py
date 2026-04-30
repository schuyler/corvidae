"""Headless daemon subcommand for corvidae."""

from __future__ import annotations

import asyncio

import click

from corvidae.runtime import Runtime


@click.command("serve")
@click.option("--config", default="agent.yaml", help="Path to config file")
def serve_command(config):
    """Start corvidae as a headless daemon."""
    runtime = Runtime(config_path=config)
    asyncio.run(runtime.run())
