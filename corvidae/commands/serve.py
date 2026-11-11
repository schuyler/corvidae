"""Headless daemon subcommand for corvidae."""

from __future__ import annotations

import click

from corvidae.runtime import Runtime, run_daemon


@click.command("serve")
@click.option("--config", default="agent.yaml", help="Path to config file")
def serve_command(config):
    """Start corvidae as a headless daemon."""
    runtime = Runtime(config_path=config)
    run_daemon(runtime)
