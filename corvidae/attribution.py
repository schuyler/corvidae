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
    agent_depth: int — how many background agents deep this operation is
        (absent or 0 for a main turn); see next_agent_depth (#816).

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


# Kernel-enforced bound on nested background agents (#816). The depth is
# attribution, so it rides the contextvars snapshot into every Task a spawned
# loop enqueues: no tool signature carries it.
AGENT_DEPTH_FIELD = "agent_depth"


def agent_depth() -> int:
    """How many background agents deep the current code is (0 in a main turn)."""
    value = get_attribution().get(AGENT_DEPTH_FIELD, 0)
    # Attribution is a plain dict that plugins may write; a bad value is read
    # as 0 rather than crashing a spawn.
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def next_agent_depth(max_depth: int) -> int | None:
    """The depth a newly spawned background agent runs at, or None at the limit.

    Every spawning tool (subagent, fork, and anything added later) asks this
    before it builds a Task, so nesting is bounded by one kernel rule instead
    of a per-tool exclusion list.
    """
    depth = agent_depth()
    return None if depth >= max_depth else depth + 1


def agent_depth_refusal(max_depth: int) -> str:
    """The refusal a spawning tool returns at the limit: names key and remedy."""
    return (
        f"Error: nested agent depth {max_depth} reached (tools.max_agent_depth); "
        "do this work in the current task instead."
    )
