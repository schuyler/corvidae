"""Admission seam for internal work (K1, issue #1050).

A tool that is about to start work on a queue (a subagent, a fork, a note to
another channel) offers it here first. Plugins vote through the
``should_start_work`` hook; the first ``WorkRefusal`` wins and the tool
returns its text instead of enqueuing anything. Results of work already
started are never offered (PRD §3.12 item 2).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.channel import Channel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkRefusal:
    """A plugin's refusal to let internal work start.

    Attributes:
        limit: The config key that was reached, e.g. ``limits.max_waiting``.
        remedy: What the caller can do next, in plain language.
    """

    limit: str
    remedy: str


async def offer_work(
    pm,
    *,
    channel: "Channel",
    kind: str,
    source_channel: str | None,
    trace_id: str | None,
    annotations: dict,
) -> str | None:
    """Offer work to every ``should_start_work`` implementation.

    Returns the refusal text the starting tool returns, or None to proceed.
    Implementations are called one by one (not through ``pm.ahook``, whose
    gather would lose every sibling's verdict when one raises), in pluggy's
    call order. An implementation that raises is logged and counts as no
    opinion; the first ``WorkRefusal`` wins.
    """
    hook = pm.hook.should_start_work
    caller_kwargs = {
        "channel": channel,
        "kind": kind,
        "source_channel": source_channel,
        "trace_id": trace_id,
        "annotations": annotations,
    }
    # pluggy calls hookimpls last-registered first; wrappers are not voters.
    for impl in reversed(hook.get_hookimpls()):
        if impl.wrapper or impl.hookwrapper:
            continue
        # Pass only the kwargs the implementation declared, as pluggy does.
        impl_kwargs = {k: caller_kwargs[k] for k in impl.argnames if k in caller_kwargs}
        try:
            verdict = impl.function(**impl_kwargs)
            # A sync implementation's verdict counts too.
            if inspect.isawaitable(verdict):
                verdict = await verdict
        except Exception:
            logger.exception(
                "should_start_work implementation %s raised for kind %s on %s; "
                "treated as no opinion",
                impl.plugin_name,
                kind,
                channel.id,
            )
            continue
        if not isinstance(verdict, WorkRefusal):
            continue
        # One warning names the kind, both channels and the limit.
        logger.warning(
            "work refused: %s on %s (from %s): %s reached",
            kind,
            channel.id,
            source_channel,
            verdict.limit,
            extra={
                "kind": kind,
                "channel": channel.id,
                "source_channel": source_channel,
                "limit": verdict.limit,
                "trace_id": trace_id,
            },
        )
        return f"Not started: {verdict.limit} reached — {verdict.remedy}"
    return None
