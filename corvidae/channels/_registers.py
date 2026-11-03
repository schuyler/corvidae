"""Default register text shipped with the transport plugins (#316).

A channel's register is appended after the agent-wide identity in its
system prompt. Transports answer the ``resolve_register_prompt`` hook with
text built here from the fragments in ``corvidae/channels/registers/``;
operators override it with ``register_prompt`` in ``agent.yaml``.
"""

from __future__ import annotations

from importlib import resources


def shipped_register(*names: str) -> str:
    """Join the named shipped fragments into one register.

    Each fragment is ``registers/<name>.md`` inside this package, stripped
    and joined with a blank line, the same way ``resolve_system_prompt``
    joins a file list.
    """
    base = resources.files("corvidae.channels").joinpath("registers")
    # Read through importlib.resources so it also works from a wheel.
    return "\n\n".join(base.joinpath(f"{name}.md").read_text().strip() for name in names)
