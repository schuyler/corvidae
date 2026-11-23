"""The one gather primitive, under a per-channel check allowance (#1162, #1176).

PRD principle 3, §3.5.12, §3.7 Output gate 7, §3.13.5. Code that publishes to
every implementation of a hook, waits for the answers for no longer than the
channel's check allowance, and returns what arrived in time is ``gather``. It
has two kinds of caller, told apart by ``phase``: the before-effect gates
(the final ``should_send_response``, ``should_execute_tool``), which ask for
verdicts, and before compose (``before_compose``), which asks for context
contributions. Each implementation is called separately and concurrently; the
caller resolves the in-time answers its own way. An implementation still
running at the deadline is left running: the miss is recorded (a debug line
and the ``check.missed`` counter, tagged with the phase), and what it returns
later goes to the caller's ``on_late``.

Nothing here names a hook, an emission or a tool.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from corvidae.held import Contribution
from corvidae.hooks import owning_transport

if TYPE_CHECKING:
    from corvidae.channel import Channel, ChannelRegistry
    from corvidae.clock import Clock

logger = logging.getLogger(__name__)

#: Counter emitted once per implementation still running at the deadline.
#: Tags: ``channel``, ``hook``, ``plugin``, ``phase``.
CHECK_MISSED_METRIC = "check.missed"

#: The minimum allowance, in seconds, when ``agent.check_allowance_min`` is
#: unset. It is also what a channel whose transport declares none gets.
DEFAULT_CHECK_ALLOWANCE_MIN = 5.0

#: Called with (plugin name, verdict) for a check that returns after the
#: deadline without raising.
LateHandler = Callable[[str, object], Awaitable[None]]

#: Late-answer deliveries in flight, held so the loop does not drop them.
_LATE_DELIVERIES: set[asyncio.Future[None]] = set()

#: The point a gather serves; tags the missed metric.
Phase = Literal["before_compose", "before_effect"]


@dataclass(frozen=True)
class Gathered:
    """What one gather returns."""

    #: In-time answers in pluggy's call order, after the wrappers ran.
    results: list[object]
    #: One future per implementation not answered in time; it resolves True
    #: when its late answer was a non-empty ``Contribution`` (after ``on_late``
    #: returned), False otherwise (a raise, an empty answer, a discarded one).
    stragglers: list["asyncio.Future[bool]"]


def resolve_check_allowance(pm, registry: "ChannelRegistry", channel: "Channel") -> float:
    """The check allowance for ``channel``, in seconds.

    The operator's value for the channel (its exact entry, else the most
    specific matching ``transport:*`` entry), else the default the owning
    transport declares, else the minimum. Never below the minimum
    (``agent.check_allowance_min``): a lower value resolves to it.
    """
    minimum = registry.agent_defaults.get("check_allowance_min", DEFAULT_CHECK_ALLOWANCE_MIN)
    # Operator: the exact entry, then wide entries, most specific (last) first.
    value = channel.config.check_allowance
    if value is None:
        for entry in reversed(channel.wide_entries):
            if entry.config.check_allowance is not None:
                value = entry.config.check_allowance
                break
    # Transport: the default of a loaded transport that owns the channel.
    if value is None:
        facts = owning_transport(pm, channel)
        if facts is not None:
            value = facts.check_allowance
    # Nothing declared: the minimum; otherwise never below it.
    if value is None:
        return minimum
    return max(value, minimum)


async def gather(
    pm,
    hook_name: str,
    *,
    channel: "Channel",
    kwargs: dict,
    allowance: float,
    clock: "Clock",
    on_late: LateHandler,
    phase: Phase,
    stop_on: Callable[[object], bool] | None = None,
) -> Gathered:
    """Publish ``hook_name`` and gather the answers that arrive in ``allowance`` seconds.

    Each non-wrapper implementation runs as its own kernel-clock task, so the
    implementations run concurrently and a straggler stays referenced (and is
    cancelled by ``Clock.close`` at shutdown). ``results`` are the in-time
    answers in pluggy's call order, after every ``wrapper=True`` implementation
    has seen and possibly replaced that list, as apluggy's own dispatch does.
    An implementation that raises is logged at warning and counts as no
    opinion.

    ``stop_on`` ends the gather at once: it is checked on each in-time answer,
    raw, before any wrapper has seen it, and a match returns without waiting
    for the rest. What the implementations still running then answer is logged
    at DEBUG and discarded (not given to ``on_late``, not counted as missed):
    the gather had already decided, and a late-check row would say the answer
    did not affect it.

    The allowance is a per-call argument. A per-implementation limit (a
    blocking check) would change the wait condition below, not this signature.
    """
    hook = getattr(pm.hook, hook_name)
    # pluggy calls non-wrappers last-registered first and stores wrappers after
    # them; keep both orders.
    impls = list(reversed(hook.get_hookimpls()))
    voters = [i for i in impls if not (i.wrapper or i.hookwrapper)]

    def _args(impl) -> dict:
        # Pass only the kwargs the implementation declared, as pluggy does.
        return {k: kwargs[k] for k in impl.argnames if k in kwargs}

    # Wrappers run their code up to `yield` before any voter starts.
    wrappers = []
    for impl in impls:
        if impl.hookwrapper:
            # apluggy's async dispatch refuses these too.
            logger.warning(
                "%s: %s is a hookwrapper=True implementation, which veto checks "
                "do not run; use wrapper=True", hook_name, impl.plugin_name,
            )
            continue
        if not impl.wrapper:
            continue
        try:
            gen = impl.function(**_args(impl))
            next(gen)
        except Exception:
            logger.warning(
                "%s wrapper %s failed before its yield on %s; left out",
                hook_name, impl.plugin_name, channel.id,
                exc_info=True, extra={"channel": channel.id},
            )
            continue
        wrappers.append((impl, gen))

    # Shared state between the deadline and the per-implementation tasks.
    verdicts: dict[int, object] = {}
    done: set[int] = set()
    # Implementations that finished (answered or raised) before the gather
    # ended; every other one is a straggler, however soon after it returns.
    in_time: set[int] = set()
    expired = False
    # True when a stop_on match ended the gather before the deadline.
    stopped_early = False
    all_done = asyncio.Event()
    deadline_hit = asyncio.Event()
    stop_hit = asyncio.Event()
    # One future per implementation; only those still running are returned.
    loop = asyncio.get_running_loop()
    settled: dict[int, asyncio.Future[bool]] = {
        index: loop.create_future() for index in range(len(voters))
    }

    async def _deliver(index: int, plugin_name: str, verdict: object) -> None:
        """Hand one late answer to ``on_late``, then settle its straggler future."""
        brought = False
        try:
            await on_late(plugin_name, verdict)
            brought = isinstance(verdict, Contribution) and bool(verdict.entries)
        except Exception:
            logger.warning(
                "%s: handling the late answer of %s on %s failed",
                hook_name, plugin_name, channel.id,
                exc_info=True, extra={"channel": channel.id},
            )
        finally:
            settled[index].set_result(brought)

    def _check_task(index: int, impl) -> Callable[[], Awaitable[None]]:
        """The clock callback that runs one implementation to completion."""

        async def run() -> None:
            # Set once a late answer has been handed to its own task, which
            # then settles the straggler future.
            handed_off = False
            try:
                try:
                    verdict = impl.function(**_args(impl))
                    # A sync implementation's verdict counts too.
                    if inspect.isawaitable(verdict):
                        verdict = await verdict
                except Exception:
                    logger.warning(
                        "%s implementation %s raised on %s%s; treated as no opinion",
                        hook_name, impl.plugin_name, channel.id,
                        " after its allowance ran out" if expired else "",
                        exc_info=True, extra={"channel": channel.id},
                    )
                    verdict = None
                    raised = True
                else:
                    raised = False
                late = expired and not raised
                # In time: record it for the caller (a raiser has no verdict),
                # and end the gather if it is the answer stop_on waits for.
                if not expired:
                    in_time.add(index)
                    if not raised:
                        verdicts[index] = verdict
                        if stop_on is not None and stop_on(verdict):
                            stop_hit.set()
                done.add(index)
                if len(done) == len(voters):
                    all_done.set()
                if late and stopped_early:
                    # The gather ended on a refusal; this answer came too late
                    # to matter and a late-check row for it would be false.
                    logger.debug(
                        "%s implementation %s answered on %s after the gather had "
                        "already refused",
                        hook_name, impl.plugin_name, channel.id,
                        extra={"channel": channel.id},
                    )
                elif late:
                    logger.debug(
                        "%s implementation %s returned on %s after its allowance ran out",
                        hook_name, impl.plugin_name, channel.id,
                        extra={"channel": channel.id},
                    )
                    # on_late runs in its own task, off the clock's: a slow
                    # handler must not hold the clock's task open (the fake
                    # clock's settle waits on every clock task), and the future
                    # settles only after the handler returned.
                    handed_off = True
                    delivery = asyncio.ensure_future(_deliver(index, impl.plugin_name, verdict))
                    _LATE_DELIVERIES.add(delivery)
                    delivery.add_done_callback(_LATE_DELIVERIES.discard)
            finally:
                # Every other path settles empty: in time, discarded or raised.
                if not handed_off and not settled[index].done():
                    settled[index].set_result(False)

        return run

    async def _expire() -> None:
        # From here on anything that finishes is late, even in this loop step.
        nonlocal expired
        expired = True
        deadline_hit.set()

    # No implementation, nothing to wait for and no deadline to arm.
    timer = (
        clock.call_later(allowance, _expire, name=f"check-deadline:{hook_name}:{channel.id}")
        if voters
        else None
    )
    try:
        if not voters:
            all_done.set()
        for index, impl in enumerate(voters):
            clock.call_soon(
                _check_task(index, impl),
                name=f"check:{hook_name}:{impl.plugin_name}:{channel.id}",
            )
        # Wait on plain events, never on the clock: this runs in a channel's
        # queue consumer, which must not be a task the clock tracks.
        waiters = [
            asyncio.ensure_future(all_done.wait()),
            asyncio.ensure_future(deadline_hit.wait()),
            asyncio.ensure_future(stop_hit.wait()),
        ]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
        # Stop the clock here: anything that finishes from now on is late. A
        # stop_on match with the deadline not yet hit is an early exit.
        stopped_early = stop_hit.is_set() and not deadline_hit.is_set()
        expired = True
    finally:
        if timer is not None:
            timer.cancel()

    # Record each implementation still running at the deadline. An early exit
    # records none: the allowance had not run out.
    for index, impl in enumerate(voters):
        if stopped_early or index in in_time:
            continue
        logger.debug(
            "%s implementation %s on %s missed its %ss allowance; going ahead without it",
            hook_name, impl.plugin_name, channel.id, allowance,
            extra={"channel": channel.id},
        )
        try:
            await pm.ahook.on_metrics(
                name=CHECK_MISSED_METRIC, value=1,
                tags={
                    "channel": channel.id, "hook": hook_name,
                    "plugin": impl.plugin_name, "phase": phase,
                },
            )
        except Exception:
            logger.warning(
                "on_metrics failed for %s on %s", CHECK_MISSED_METRIC, channel.id,
                exc_info=True, extra={"channel": channel.id},
            )

    # Every in-time verdict, None included, as apluggy hands wrappers.
    results: list[object] = [verdicts[i] for i in sorted(verdicts)]
    # Wrappers see the in-time list, innermost first, as apluggy tears down.
    for impl, gen in reversed(wrappers):
        try:
            gen.send(results)
        except StopIteration as stop:
            results = stop.value
        except Exception:
            logger.warning(
                "%s wrapper %s raised on %s; its verdict list passes on unchanged",
                hook_name, impl.plugin_name, channel.id,
                exc_info=True, extra={"channel": channel.id},
            )
    return Gathered(
        results=results,
        stragglers=[settled[i] for i in range(len(voters)) if i not in in_time],
    )
