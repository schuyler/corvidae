"""OutboundCapPlugin — bound the agent's messages between human messages (#791).

On 2026-11-12 the agent sent about 600 Signal messages in one night with no
human message in between. Nothing in the kernel bounded that: the output gate
can withhold a *particular* emission (the silence plugin's ``NO_REPLY``), but
nothing counted emissions per channel, so a looping turn, an injected
instruction or a notification storm could talk to a person indefinitely.

It carries two independent limits. The first counts, per channel, the messages
the gate lets through since the last admitted human message on that channel,
and withholds everything after ``outbound_cap.max_messages`` until a human
speaks again. That bounds a channel nobody is talking on, but not one where a
human keeps talking, or one whose counter keeps resetting, so a loop that sends
several messages per human message is still unbounded over time: the second
limit, ``outbound_cap.max_messages_per_window`` in
``outbound_cap.window_seconds``, bounds that. It is a sliding window of
emission timestamps per channel on the kernel clock, and a human message does
**not** reset it. Either limit may be set alone; with both, whichever is
reached first withholds. It is a plugin, not a
kernel change, because ``should_send_response`` is the seam every text
emission already passes through, and the kernel's existing veto path already
does the rest: a withheld final is stored as ``UNSENT``, is not sent, and
leaves a not-sent marker in the window, so the agent knows it was held and
history keeps it.

What is capped, and what is not
-------------------------------

* Counting is per channel id. ``on_message_admitted`` resets the counter to 0,
  so the reply to a human message is always within the cap (``max_messages``
  is at least 1) and only agent messages beyond M with no human message in
  between are withheld.
* A channel with no limit configured is never capped, and neither is a channel
  **no transport owns**, for example ``internal:heartbeat``. That is the agent
  talking to itself; capping it would break the heartbeat rather than protect a
  person. Scheduler reminders and task results notify the originating transport
  channel, so those finals are capped.
* A message counts only after the combined gate verdict: a wrapper on
  ``should_send_response`` sees every gate's result and counts the emission
  only if none vetoed it, so a ``NO_REPLY`` vetoed elsewhere spends no budget.
* By default only ``final`` emissions count, so a tool-using reply that sends
  "let me look that up" (a ``progress`` emission) is not penalised.
  ``outbound_cap.count_progress`` counts those too.
* Known gaps, accepted for the MVP (#791): sends that bypass the gate are not
  capped — ``send_signal_attachment``, ``signal_react``, tool-status events and
  the restart notice. The cap bounds messages, not turns (#793). An
  agent-started send that is later withheld on the target channel has already
  returned success to the sending turn, which is why every withheld emission is
  logged.

With the plugin disabled — or with no ``outbound_cap`` key — nothing is
counted and nothing is withheld.
"""

from __future__ import annotations

import logging
import time

from corvidae.hooks import CorvidaePlugin, TransportPlugin, hookimpl, resolve_reject_wins
from corvidae.tuning import resolve_tunable

logger = logging.getLogger("corvidae.outbound_cap")

# The plugin-wide defaults, resolved per decision through resolve_tunable so
# both tuning surfaces work (set_settings per channel, hot config reload).
MAX_MESSAGES_KEY = "outbound_cap.max_messages"
COUNT_PROGRESS_KEY = "outbound_cap.count_progress"
# The second limit (#821): at most this many agent emissions per channel in
# this many seconds, sliding. Both must be set for the limit to be on.
WINDOW_LIMIT_KEY = "outbound_cap.max_messages_per_window"
WINDOW_SECONDS_KEY = "outbound_cap.window_seconds"

# What counts as one message when count_progress is false.
COUNTED_EMISSIONS = frozenset({"final"})
COUNTED_EMISSIONS_WITH_PROGRESS = frozenset({"final", "progress"})


def _bad_int(key: str, value: object) -> ValueError:
    """The config error naming the key, the rejected value and the range.

    ``bool`` is checked explicitly because ``True`` is an ``int`` in Python and
    ``max_messages: true`` is a typo, not a limit of one.
    """
    return ValueError(
        f"Config error: {key} is {value!r}; expected a positive integer "
        f"(a whole number, at least 1) — check agent.yaml"
    )


def parse_max_messages(value: object, key: str = MAX_MESSAGES_KEY) -> int | None:
    """Validate a ``max_messages`` value, or ``None`` for "not configured".

    ``None`` (and an absent key) means no cap. Everything else must be an
    ``int`` of at least 1: a bool, a float — even a whole one — a string, 0 and
    negatives are rejected, because each of them would either cap a channel at
    zero messages or silently mean something other than what was written.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _bad_int(key, value)
    return value


def parse_count_progress(value: object, key: str = COUNT_PROGRESS_KEY) -> bool:
    """Validate a ``count_progress`` value; the default is ``False``."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(
            f"Config error: {key} is {value!r}; expected true or false — check agent.yaml"
        )
    return value


def parse_window_limit(value: object, key: str = WINDOW_LIMIT_KEY) -> int | None:
    """Validate a ``max_messages_per_window`` value, or ``None`` for "not set"."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _bad_int(key, value)
    return value


def parse_window_seconds(value: object, key: str = WINDOW_SECONDS_KEY) -> int | None:
    """Validate a ``window_seconds`` value, or ``None`` for "not set".

    A window is a whole number of seconds. ``window_seconds`` without a limit is
    ignored rather than refused: it caps nothing, which is the fail-open
    direction, and an operator may set one before the other.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _bad_int(key, value)
    return value


class OutboundCapPlugin(CorvidaePlugin):
    """Caps agent messages per channel between admitted human messages."""

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self._pm = pm
        self._config = config
        # channel id -> messages counted since the last admitted human message
        self._counts: dict[str, int] = {}
        # channel id -> messages withheld since the last reset
        self._withheld: dict[str, int] = {}
        # channel id -> emission times inside the window, oldest first (#821)
        self._window_hits: dict[str, list[float]] = {}
        # channel id -> messages the window withheld since it last reopened
        self._window_withheld: dict[str, int] = {}
        # channel ids that have already logged their first WARNING this run
        self._warned: set[str] = set()
        # Validate at startup so a bad value names itself here rather than
        # capping nothing at the first send.
        block = config.get("outbound_cap")
        if block is not None and not isinstance(block, dict):
            # `outbound_cap: 5` would otherwise silently mean "no cap".
            raise ValueError(
                f"Config error: outbound_cap is {block!r}; expected a mapping "
                f"with max_messages and count_progress keys — check agent.yaml"
            )
        if isinstance(block, dict):
            parse_max_messages(block.get("max_messages"), MAX_MESSAGES_KEY)
            parse_count_progress(block.get("count_progress"), COUNT_PROGRESS_KEY)
            self._check_window(block)

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Adopt the reloaded config; a bad value is logged and ignored.

        The kernel has already merged and validated the file, so a value that
        fails here is one only this plugin judges (a limit of 0, a bool). The
        running limit stays in force rather than the daemon refusing every
        send, which is the fail-soft half of "fail soft but loud".
        """
        block = config.get("outbound_cap")
        if isinstance(block, dict):
            try:
                parse_max_messages(block.get("max_messages"), MAX_MESSAGES_KEY)
                parse_count_progress(block.get("count_progress"), COUNT_PROGRESS_KEY)
                self._check_window(block)
            except ValueError as exc:
                logger.error(
                    "outbound_cap: keeping the previous limit; %s",
                    exc,
                )
                return
        self._config = config

    @hookimpl
    async def on_message_admitted(self, channel, trace_id: str, sender: str, text: str) -> None:
        """A human message resets the channel's count (#791 requirement 2).

        A coalesced burst resets once per admitted message, which is the same
        thing: the count is 0 either way before the turn runs.
        """
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return
        self._counts.pop(channel_id, None)
        withheld = self._withheld.pop(channel_id, 0)
        if withheld:
            logger.warning(
                "outbound_cap: a human message on %s reset the cap; %d message%s were withheld",
                channel_id,
                withheld,
                "" if withheld == 1 else "s",
                extra={"channel": channel_id},
            )
        # The between-human marker only: the window is not reset by a human,
        # so its own marker and count outlive this.
        self._warned.discard(f"{channel_id}\x00between-human")

    def _transport_owned(self, channel) -> bool:
        """Whether some transport plugin delivers to ``channel``.

        Internal channels (``internal:heartbeat``) have a transport name no
        plugin claims, and are never capped: they are the agent talking to
        itself.
        """
        try:
            plugins = self._pm.get_plugins()
        except Exception:  # pragma: no cover - a pm without get_plugins
            # Fail open: treat the channel as owned so the cap still applies.
            logger.warning(
                "outbound_cap: could not list plugins to find the owner of %s; "
                "treating it as transport-owned",
                getattr(channel, "id", None),
                exc_info=True,
            )
            return True
        for plugin in plugins:
            if isinstance(plugin, TransportPlugin):
                try:
                    if plugin.owns(channel):
                        return True
                except Exception:  # pragma: no cover - a broken transport
                    # Skip the broken transport and ask the next one.
                    logger.warning(
                        "outbound_cap: transport %s failed owns() for %s; skipping it",
                        type(plugin).__name__,
                        getattr(channel, "id", None),
                        exc_info=True,
                    )
                    continue
        return False

    @staticmethod
    def _check_window(block: dict) -> None:
        """Validate the window pair, refusing a limit with no window.

        A limit with no ``window_seconds`` would have to invent a window, and
        the operator's intent would be lost silently: say so instead.
        """
        limit = parse_window_limit(block.get("max_messages_per_window"))
        seconds = parse_window_seconds(block.get("window_seconds"))
        if limit is not None and seconds is None:
            raise ValueError(
                f"Config error: {WINDOW_SECONDS_KEY} is not set, but "
                f"{WINDOW_LIMIT_KEY} is {limit!r}; expected a positive whole "
                f"number of seconds for the window — check agent.yaml"
            )

    def _now(self) -> float:
        """The kernel clock's time, or a monotonic fallback without one."""
        clock = None
        try:
            clock = self._pm.get_plugin("clock")
        except Exception:  # pragma: no cover - a pm without get_plugin
            clock = None
        if clock is None:
            # No clock registered (a bare plugin manager): monotonic is the
            # right kind of time for a sliding window, just not a swappable one.
            return time.monotonic()
        return clock.now()

    def _window(self, channel) -> tuple[int | None, int | None, str]:
        """The channel's window limit, its length and the key that set it."""
        overrides = getattr(channel, "runtime_overrides", None) or {}
        key = WINDOW_LIMIT_KEY
        limit_raw = resolve_tunable(channel, self._config, WINDOW_LIMIT_KEY, None)
        if overrides and WINDOW_LIMIT_KEY in overrides:
            key = f"runtime override {WINDOW_LIMIT_KEY}"
        try:
            limit = parse_window_limit(limit_raw, key)
            seconds = parse_window_seconds(
                resolve_tunable(channel, self._config, WINDOW_SECONDS_KEY, None)
            )
        except ValueError as exc:
            # A bad override must not wedge the channel: log and cap nothing.
            logger.warning(
                "outbound_cap: ignoring the window limit; %s",
                exc,
                extra={"channel": getattr(channel, "id", None)},
            )
            return None, None, key
        if limit is not None and seconds is None:
            logger.warning(
                "outbound_cap: ignoring the window limit on %s; %s is %r but %s is "
                "not set, so there is no window to count in",
                getattr(channel, "id", None),
                WINDOW_LIMIT_KEY,
                limit,
                WINDOW_SECONDS_KEY,
                extra={"channel": getattr(channel, "id", None)},
            )
            return None, None, key
        return limit, seconds, key

    def _window_hits_now(self, channel_id: str, seconds: int) -> list[float]:
        """The channel's emission times still inside the window, oldest first.

        Slides as it reads: times older than the window are dropped, so the
        list never grows past what the window can hold.
        """
        now = self._now()
        cutoff = now - seconds
        hits = [t for t in self._window_hits.get(channel_id, []) if t > cutoff]
        if hits:
            self._window_hits[channel_id] = hits
        else:
            self._window_hits.pop(channel_id, None)
        return hits

    def _limit(self, channel) -> tuple[int | None, str]:
        """The channel's effective limit and the key that set it."""
        overrides = getattr(channel, "runtime_overrides", None) or {}
        key = MAX_MESSAGES_KEY
        raw = resolve_tunable(channel, self._config, MAX_MESSAGES_KEY, None)
        if overrides and MAX_MESSAGES_KEY in overrides:
            key = f"runtime override {MAX_MESSAGES_KEY}"
        try:
            return parse_max_messages(raw, key), key
        except ValueError as exc:
            # A bad per-channel override must not wedge the channel: log once
            # per call and cap nothing, which is the fail-open direction.
            logger.warning(
                "outbound_cap: ignoring the limit; %s",
                exc,
                extra={"channel": getattr(channel, "id", None)},
            )
            return None, key

    def _counted_kinds(self, channel) -> frozenset[str]:
        """The emission kinds that spend budget on ``channel``."""
        counted = parse_count_progress(
            resolve_tunable(channel, self._config, COUNT_PROGRESS_KEY, False)
        )
        return COUNTED_EMISSIONS_WITH_PROGRESS if counted else COUNTED_EMISSIONS

    @hookimpl
    async def should_send_response(
        self, channel, text: str, emission: str, trace_id, annotations
    ) -> bool | None:
        """Veto once either limit is reached; count nothing here.

        Counting happens in ``count_after_verdict``, after every gate has
        voted, so an emission another gate vetoes spends no budget. Each
        reached limit is named in the log, so an M veto, an N veto and the case
        where both hold are distinguishable. Fail-open: a broken check must
        never silence the agent, so an exception logs and returns no opinion.
        """
        try:
            limit, key = self._limit(channel)
            window_limit, window_seconds, window_key = self._window(channel)
            if limit is None and window_limit is None:
                return None
            channel_id = channel.id
            if not self._transport_owned(channel):
                return None

            reasons: list[tuple[str, str, int]] = []
            if window_limit is not None:
                hits = self._window_hits_now(channel_id, window_seconds)
                if len(hits) >= window_limit:
                    reasons.append((
                        "window",
                        f"window limit {window_limit} in {window_seconds}s set by {window_key}",
                        len(hits),
                    ))
            if limit is not None:
                count = self._counts.get(channel_id, 0)
                if count >= limit:
                    reasons.append((
                        "between-human",
                        f"between-human limit {limit} set by {key}",
                        count,
                    ))

            if reasons:
                return self._withhold(channel_id, emission, trace_id, reasons)

            # A window that was withholding and now lets a message through has
            # slid open: say how much it held, once.
            held = self._window_withheld.pop(channel_id, 0)
            if held:
                self._warned.discard(f"{channel_id}\x00window")
                logger.warning(
                    "outbound_cap: the window on %s reopened; %d message%s were "
                    "withheld by %s (%d in %ds)",
                    channel_id, held, "" if held == 1 else "s", window_key,
                    window_limit, window_seconds,
                    extra={"channel": channel_id},
                )
            return None
        except Exception:
            logger.warning(
                "outbound_cap: the cap check failed for the %s emission of %s; sending it",
                emission,
                trace_id,
                exc_info=True,
                extra={"channel": getattr(channel, "id", None), "trace_id": trace_id},
            )
            return None

    def _withhold(
        self, channel_id: str, emission: str, trace_id,
        reasons: list[tuple[str, str, int]],
    ) -> bool:
        """Count and log one withheld emission, and return the veto.

        ``reasons`` is one ``(which, words, count)`` per limit that is reached:
        the between-human cap, the window, or both. The first veto for a
        channel and limit warns; the rest are INFO, so a long cap does not fill
        the log.
        """
        held = 0
        for which, _words, _count in reasons:
            if which == "window":
                self._window_withheld[channel_id] = self._window_withheld.get(channel_id, 0) + 1
                held = self._window_withheld[channel_id]
            else:
                self._withheld[channel_id] = self._withheld.get(channel_id, 0) + 1
                held = self._withheld[channel_id]

        words = " and ".join(w for _which, w, _c in reasons)
        first = f"{channel_id}\x00{reasons[0][0]}"
        if first not in self._warned:
            self._warned.add(first)
            logger.warning(
                "outbound_cap: withholding the %s emission for %s on %s: "
                "%d withheld since the cap was reached, %s",
                emission, trace_id, channel_id, held, words,
                extra={"channel": channel_id, "trace_id": trace_id},
            )
        else:
            logger.info(
                "outbound_cap: withheld the %s emission for %s on %s "
                "(%d withheld since the cap was reached; %s)",
                emission, trace_id, channel_id, held, words,
                extra={"channel": channel_id, "trace_id": trace_id},
            )
        return False

    @hookimpl(wrapper=True, specname="should_send_response")
    def count_after_verdict(
        self, channel, emission: str, trace_id
    ):
        """Count an emission only once the combined verdict lets it through.

        A wrapper sees the result list of every gate after they ran (apluggy
        drives it from ``_async_multicall``), so a veto from any other gate
        keeps the emission out of the count. The results pass through
        unchanged; a failure here logs and never changes the verdict.
        """
        # Let every gate vote; an exception from a gate propagates untouched
        # so the kernel's own fail-open handling in _gate_send applies.
        results = yield
        try:
            if resolve_reject_wins(results) is not False:
                limit, _key = self._limit(channel)
                window_limit, window_seconds, _window_key = self._window(channel)
                if (
                    (limit is not None or window_limit is not None)
                    and self._transport_owned(channel)
                    and emission in self._counted_kinds(channel)
                ):
                    if limit is not None:
                        self._counts[channel.id] = self._counts.get(channel.id, 0) + 1
                    if window_limit is not None:
                        # Append after pruning, so the list is the window's
                        # contents rather than everything ever sent.
                        hits = self._window_hits_now(channel.id, window_seconds)
                        self._window_hits[channel.id] = [*hits, self._now()]
        except Exception:
            logger.warning(
                "outbound_cap: counting the %s emission of %s failed; it was sent uncounted",
                emission,
                trace_id,
                exc_info=True,
                extra={"channel": getattr(channel, "id", None), "trace_id": trace_id},
            )
        return results
