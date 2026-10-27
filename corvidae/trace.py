"""Trace id minting.

A leaf module (stdlib imports only) so transports can mint the id for a
user message without importing ``corvidae.agent`` and its dependencies.
The transport mints the id, passes it to ``on_message``, and the Agent uses
it verbatim for the gate, lifecycle hooks and persistence (issue #95).
"""

import time
import uuid


def mint_trace_id() -> str:
    """Mint a time-sortable, globally-unique trace id.

    Shape: ``<hex-time>-<hex12>``. The hex-time prefix is
    monotone-nondecreasing across calls (wall-clock seconds); the uuid4
    suffix guarantees distinctness within the same second.
    """
    return f"{int(time.time()):x}-{uuid.uuid4().hex[:12]}"
