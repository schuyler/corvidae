"""Default register text shipped with the transport plugins (#316).

A channel's register is appended after the agent-wide identity in its
system prompt. Transports answer the ``resolve_register_prompt`` hook with
text built here from the fragments in ``corvidae/channels/registers/``;
operators override it with ``register_prompt`` in ``agent.yaml``.
"""

from __future__ import annotations

from importlib import resources

import apluggy as pluggy


def shipped_register(*names: str) -> str:
    """Join the named shipped fragments into one register.

    Each fragment is ``registers/<name>.md`` inside this package, stripped
    and joined with a blank line, the same way ``resolve_system_prompt``
    joins a file list.
    """
    base = resources.files("corvidae.channels").joinpath("registers")
    # Read through importlib.resources so it also works from a wheel.
    return "\n\n".join(base.joinpath(f"{name}.md").read_text().strip() for name in names)


def silence_fragment(pm: pluggy.PluginManager) -> str | None:
    """The silence plugin's register fragment, or None when it isn't loaded.

    Looked up by name so the kernel never imports ``corvidae_cognition``.
    """
    plugin = pm.get_plugin("silence")
    if plugin is None:
        return None
    return plugin.register_fragment()


def with_silence(pm: pluggy.PluginManager, register: str) -> str:
    """``register`` plus the silence fragment (after a blank line) when loaded."""
    fragment = silence_fragment(pm)
    return register if fragment is None else f"{register}\n\n{fragment}"
