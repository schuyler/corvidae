"""Veto checks under a per-channel check allowance (#1162).

PRD principle 3, §3.5.12, §3.7 Output gate 7, §3.13.5. A check that runs
before a reply is sent or a tool call runs may delay it for no longer than the
channel's check allowance. ``run_checks`` calls every implementation of a veto
hook separately and concurrently, waits up to the allowance, and returns the
verdicts that arrived in time; the caller resolves them its own way. A check
still running at the deadline is left running: the miss is recorded (a debug
line and the ``check.missed`` counter), and what it returns later goes to the
caller's ``on_late``.

Nothing here names a hook, an emission or a tool: the final firing of
``should_send_response`` is the first caller, and ``should_execute_tool``
(#148) is meant to be the second.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from corvidae.hooks import TransportPlugin

if TYPE_CHECKING:
    from corvidae.channel import Channel, ChannelRegistry
    from corvidae.clock import Clock

logger = logging.getLogger(__name__)

#: Counter emitted once per implementation still running at the deadline.
#: Tags: ``channel``, ``hook``, ``plugin``.
CHECK_MISSED_METRIC = "check.missed"

#: The minimum allowance, in seconds, when ``agent.check_allowance_min`` is
#: unset. It is also what a channel whose transport declares none gets.
DEFAULT_CHECK_ALLOWANCE_MIN = 5.0

#: Called with (plugin name, verdict) for a check that returns after the
#: deadline without raising.
LateHandler = Callable[[str, object], Awaitable[None]]


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
    # Transport: the first loaded transport that owns the channel.
    if value is None:
        for plugin in pm.get_plugins():
            if isinstance(plugin, TransportPlugin) and plugin.owns(channel):
                value = plugin.check_allowance
                break
    if value is None:
        return minimum
    return max(value, minimum)


async def run_checks(
    pm,
    hook_name: str,
    *,
    channel: "Channel",
    kwargs: dict,
    allowance: float,
    clock: "Clock",
    on_late: LateHandler,
) -> list[object]:
    """Run every implementation of ``hook_name`` under ``allowance`` seconds.

    Each non-wrapper implementation runs as its own kernel-clock task, so the
    checks run concurrently and a straggler stays referenced (and is cancelled
    by ``Clock.close`` at shutdown). Returns the verdicts that arrived in time,
    in pluggy's call order, after every ``wrapper=True`` implementation has
    seen and possibly replaced that list, as apluggy's own dispatch does. An
    implementation that raises is logged at warning and counts as no opinion.
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
    expired = False
    all_done = asyncio.Event()
    deadline_hit = asyncio.Event()

    def _check_task(index: int, impl) -> Callable[[], Awaitable[None]]:
        """The clock callback that runs one implementation to completion."""

        async def run() -> None:
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
                late = False
            else:
                late = expired
            # In time: record it for the caller. Late: hand it on.
            if not expired:
                verdicts[index] = verdict
            done.add(index)
            if len(done) == len(voters):
                all_done.set()
            if late:
                logger.debug(
                    "%s implementation %s returned on %s after its allowance ran out",
                    hook_name, impl.plugin_name, channel.id,
                    extra={"channel": channel.id},
                )
                try:
                    await on_late(impl.plugin_name, verdict)
                except Exception:
                    logger.warning(
                        "%s: handling the late verdict of %s on %s failed",
                        hook_name, impl.plugin_name, channel.id,
                        exc_info=True, extra={"channel": channel.id},
                    )

        return run

    async def _expire() -> None:
        # The deadline timer only wakes the waiter; the waiter does the rest.
        deadline_hit.set()

    timer = clock.call_later(
        allowance, _expire, name=f"check-deadline:{hook_name}:{channel.id}"
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
        ]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
        # Stop the clock here: anything that finishes from now on is late.
        expired = True
    finally:
        timer.cancel()

    # Record each implementation still running at the deadline.
    for index, impl in enumerate(voters):
        if index in done:
            continue
        logger.debug(
            "%s implementation %s on %s missed its %ss allowance; going ahead without it",
            hook_name, impl.plugin_name, channel.id, allowance,
            extra={"channel": channel.id},
        )
        try:
            await pm.ahook.on_metrics(
                name=CHECK_MISSED_METRIC, value=1,
                tags={"channel": channel.id, "hook": hook_name, "plugin": impl.plugin_name},
            )
        except Exception:
            logger.warning(
                "on_metrics failed for %s on %s", CHECK_MISSED_METRIC, channel.id,
                exc_info=True, extra={"channel": channel.id},
            )

    results: list[object] = [verdicts[i] for i in sorted(verdicts) if verdicts[i] is not None]
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
    return results
