"""Corvidae command dispatcher.

Discovers subcommands from the 'corvidae.commands' entry point group at
import time and registers them with the corvidae click Group.

Third-party plugins register subcommands the same way:

    [project.entry-points."corvidae.commands"]
    scaffold = "corvidae_scaffold:scaffold_command"

Standing constraint: modules loaded via this entry point group are imported
before Runtime.start() configures logging or creates an event loop. These
modules must not have module-level side effects that depend on logging
configuration or a running event loop.
"""

from __future__ import annotations

import importlib.metadata
import logging

import click

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def corvidae(ctx):
    """Corvidae agent daemon."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def discover_commands():
    """Load subcommands from the 'corvidae.commands' entry point group."""
    for ep in importlib.metadata.entry_points(group="corvidae.commands"):
        try:
            corvidae.add_command(ep.load(), ep.name)
        except Exception:
            logger.warning(
                "failed to load command %r, skipping", ep.name, exc_info=True
            )


discover_commands()
