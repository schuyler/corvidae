"""Call attribution for observability.

A single ContextVar holds a small dict describing what the current code
path is doing on whose behalf. LLMPlugin's observer reads it when a call
fires; callers set it at the top of a logical operation and reset it in
a ``finally`` block using the returned token.

Recognized fields (the dict is open — plugins may add their own):
    stage: str — what kind of operation is running. Built-in values:
        "turn", "compaction", "subagent"; later phases add
        "consolidation", "appraisal", "critique".
    channel_id: str — the Channel.id the operation is running on behalf of.
    exchange_key: str — the exchange this call belongs to (Phase 2).

Propagation rule: contextvars snapshot at ``asyncio.create_task`` time.
A task created before ``set_attribution`` was called never sees the
attribution. TaskQueue workers are created once at startup, so Task
captures ``contextvars.copy_context()`` at creation and the worker runs
the work inside that captured context (see corvidae.task).
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
