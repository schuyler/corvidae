"""Call attribution for observability.

A single ContextVar holds a small dict describing what the current code
path is doing on whose behalf. LLMPlugin's observer reads it when a call
fires; callers set it at the top of a logical operation and reset it in
a ``finally`` block using the returned token.

Recognized fields (the dict is open — plugins add their own stage values
and metadata):
    stage: str — what kind of operation is running. Built-in values:
        "turn", "compaction", "subagent".
    channel_id: str — the Channel.id the operation is running on behalf of.
    trace_id: str — the trace this call belongs to.
    annotations: dict — the opaque, plugin-owned annotations of the turn.

Propagation rule: contextvars snapshot at ``asyncio.create_task`` time.
A task created before ``set_attribution`` was called never sees the
attribution. TaskQueue workers are created once at startup, so Task
captures ``contextvars.copy_context()`` at creation and the worker runs
the work inside that captured context (see corvidae.task). A channel's
SerialQueue consumer is the opposite: it starts in an empty context, because
whichever code first enqueues to it may be running inside another turn.
"""

import contextvars

# The single attribution ContextVar. The default is an empty dict so
# get_attribution() never returns None.
_attribution: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "corvidae_attribution", default={}
)


def set_attribution(**fields) -> contextvars.Token:
    """Merge fields into the current attribution; returns a reset token."""
    merged = {**_attribution.get(), **fields}
    return _attribution.set(merged)


def get_attribution() -> dict:
    """Return the current attribution dict (possibly empty). Never None."""
    return _attribution.get()


def reset_attribution(token: contextvars.Token) -> None:
    """Restore the attribution state captured by a set_attribution token."""
    _attribution.reset(token)


def next_agent_depth(max_depth: int) -> int:
    """Return the depth for a newly spawned background agent.

    Reads the current ``agent_depth`` from attribution (0 for a main turn),
    refuses if it is at or above ``max_depth``, otherwise returns
    ``depth + 1``. The caller sets the returned value via
    ``set_attribution(agent_depth=...)`` in the spawned work.

    Args:
        max_depth: The configured ``tools.max_agent_depth`` limit.

    Returns:
        The depth value for the spawned agent.

    Raises:
        ValueError: If the current depth is at or above ``max_depth``.
    """
    current = get_attribution().get("agent_depth", 0)
    if current >= max_depth:
        raise ValueError(
            f"nested agent depth {current} reached "
            f"(tools.max_agent_depth={max_depth}); "
            "do this work in the current task instead."
        )
    return current + 1


DEFAULT_MAX_AGENT_DEPTH = 2


def parse_max_agent_depth(config: dict) -> int:
    """Parse ``tools.max_agent_depth`` from config.

    Args:
        config: The full agent.yaml config dict.

    Returns:
        The maximum nesting depth for background agents (default 2).

    Raises:
        ValueError: If the value is not a positive integer.
    """
    tools = config.get("tools", {})
    if not isinstance(tools, dict):
        tools = {}
    raw = tools.get("max_agent_depth", DEFAULT_MAX_AGENT_DEPTH)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise ValueError(
            f"tools.max_agent_depth must be a positive integer "
            f"(>= 1); got {raw!r}"
        )
    return raw
