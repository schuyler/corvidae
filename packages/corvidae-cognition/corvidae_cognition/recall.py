"""Shared mixin for per-turn recall sources (#718, part of #711).

A *recall source* looks something up for the turn the agent is about to take
(a wiki, a folder of notes) and hands the result to the funnel as CONTEXT. The
funnel owns admission (dedupe, budget, framing, persistence, expiry) and the
pull hook ``on_funnel_ingest``; this module owns what every source repeats:

- the gate: search only when the window tail is a plain user message, with the
  query memory builds (``retrieval_query``);
- ``SearchGate``: one cooldown per channel and at most one search in flight per
  channel, on an injectable clock;
- ``parse_recall_settings``: the ``cooldown`` and ``timeout`` keys and their
  rejected-value messages;
- fail-soft wrapping: a search that raises or times out is logged with the
  source, the channel and the traceback and resolves to ``None``;
- the ``on_funnel_ingest`` implementation that submits the promise.

A source is a plugin that subclasses ``RecallSourceMixin`` and
``CorvidaePlugin``, calls ``init_recall(config)`` from its ``on_init``, and
supplies ``search(query, ctx)`` and ``render(hit)``. The mixin is a helper, not
a plugin: it has no entry point. Retrieved text reaches the window only through
the funnel's framing (PRD principle 9), so the mixin adds no route for
untrusted text of its own.

Without the funnel nothing calls ``on_funnel_ingest``, so a source does nothing;
the mixin declares the funnel as an optional dependency so that is logged at
DEBUG rather than as a stray hook implementation.

Same-turn admission (#726): a promise submitted already done is admitted
before that turn's model call, and a result that finishes after the funnel's
sweep but before the same turn's collect is carried into that turn's drain
rather than lost when the mixin submits again. A fast local source therefore
needs no timing cooperation from its provider.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from corvidae.context import MessageType
from corvidae.hooks import hookimpl
from corvidae.tuning import config_value
from corvidae_cognition.funnel import Ingest
from corvidae_cognition.memory import retrieval_query

logger = logging.getLogger("corvidae.recall")

# Seconds between two searches on one channel, and the bound on one search.
DEFAULT_COOLDOWN = 5.0
DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class RecallSettings:
    """The shared per-source settings, read from the source's own config block."""

    # Seconds that must pass between the start of one search and the next, per channel.
    cooldown: float
    # Seconds one search may run before it is cancelled and logged.
    timeout: float


@dataclass(frozen=True)
class RecallContext:
    """What a source's ``search`` is told about the turn it is searching for."""

    # The channel whose window the result is for.
    channel: Any
    # The turn's trace id.
    trace_id: str
    # The turn's annotations, or None.
    annotations: dict | None


def _read_seconds(block_key: str, field: str, block: dict, default: float, *, zero_ok: bool) -> float:
    """Read one seconds-valued key from a source's block, or raise naming it.

    A missing key gives ``default``. A bool, a non-number, a non-finite number
    or a number outside the accepted range raises ``ValueError`` naming the
    key, the rejected value, what is accepted and where to fix it.
    """
    key = f"{block_key}.{field}"
    # Missing: the default applies and nothing is logged (no config is silent).
    if field not in block:
        return default
    raw = block[field]
    # bool is an int subclass, but True is not a number of seconds.
    ok = isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw)
    if ok:
        ok = raw >= 0 if zero_ok else raw > 0
    if not ok:
        accepted = "a number of seconds, 0 or more" if zero_ok else "a number of seconds greater than 0"
        raise ValueError(
            f"Config error: {key} must be {accepted}, got {raw!r} — "
            f"set {key} in agent.yaml, or remove it to use the default of {default:g}"
        )
    return float(raw)


def parse_recall_settings(config: dict, key: str) -> RecallSettings:
    """Read ``cooldown`` and ``timeout`` from the config block named ``key``.

    ``config`` is the whole agent.yaml dict; only the ``key`` block is read, so
    a top-level ``cooldown`` is ignored. No block gives the defaults. A block
    that is not a mapping, or a malformed value in it, raises ``ValueError``
    (the host's ``on_init`` lets it propagate, which aborts startup). Keys the
    host defines in the same block are left alone.
    """
    block = config_value(config, key, None)
    # No block at all: defaults, silently.
    if block is None:
        return RecallSettings(DEFAULT_COOLDOWN, DEFAULT_TIMEOUT)
    # A block that is not a mapping cannot hold keys.
    if not isinstance(block, dict):
        raise ValueError(
            f"Config error: {key} must be a mapping, got {block!r} — "
            f"check {key} in agent.yaml"
        )
    return RecallSettings(
        cooldown=_read_seconds(key, "cooldown", block, DEFAULT_COOLDOWN, zero_ok=True),
        timeout=_read_seconds(key, "timeout", block, DEFAULT_TIMEOUT, zero_ok=False),
    )


class SearchGate:
    """Cooldown and single-flight, per channel, on an injectable clock.

    ``try_start`` says whether a search may begin now and, when it does,
    records the start; ``finish`` releases the channel's in-flight slot. The
    cooldown runs from the start of a search, so a search that failed or timed
    out counts as a search and a retry waits out the cooldown too.
    """

    def __init__(self, cooldown: float, now: Callable[[], float] = time.monotonic) -> None:
        # Seconds between the starts of two searches on one channel.
        self._cooldown = cooldown
        # The clock; tests pass a fake.
        self._now = now
        # Channels with a search in flight, and when each channel last started one.
        self._in_flight: set[str] = set()
        self._last_start: dict[str, float] = {}

    def try_start(self, channel_id: str) -> bool:
        """True, and the search is recorded, when ``channel_id`` may search now."""
        # One search in flight per channel: a second would supersede the first.
        if channel_id in self._in_flight:
            return False
        # Within the cooldown of the last start: wait.
        now = self._now()
        last = self._last_start.get(channel_id)
        if last is not None and now - last < self._cooldown:
            return False
        # Allowed: record the start and take the in-flight slot.
        self._last_start[channel_id] = now
        self._in_flight.add(channel_id)
        return True

    def finish(self, channel_id: str) -> None:
        """Release ``channel_id``'s in-flight slot; harmless if it holds none."""
        self._in_flight.discard(channel_id)


def _tail_is_plain_user_message(messages: list[dict]) -> bool:
    """True when the window ends in a user MESSAGE with non-blank text.

    The same test memory applies, except that CONTEXT rows at the tail are
    looked through: ``on_funnel_ingest`` runs last in the turn, after the
    funnel's drain (or another source) may have appended CONTEXT below the
    user's message, and that is still a user turn. Recall is skipped on turns
    that a tool result, a notification or an agent reply ends.
    """
    # Walk back past trailing CONTEXT rows to the last row that is not one.
    for tail in reversed(messages):
        if tail.get("_message_type") == MessageType.CONTEXT:
            continue
        return (
            tail.get("role") == "user"
            and tail.get("_message_type", MessageType.MESSAGE) == MessageType.MESSAGE
            and isinstance(tail.get("content"), str)
            and bool(tail["content"].strip())
        )
    # An empty window (or one of CONTEXT only) has no user message.
    return False


class RecallSourceMixin:
    """Mixin for a plugin that recalls from one source on every user turn.

    Subclass it together with ``CorvidaePlugin``. Set ``source_name`` (the
    funnel's frame label and the key of ``funnel.budgets``) and ``config_key``
    (the host's own config block), call ``init_recall(config)`` from ``on_init``,
    and override ``search`` and ``render``.
    """

    # Declared so a disabled funnel logs this plugin's hookimpl at DEBUG, not WARNING.
    optional_depends_on = frozenset({"funnel"})
    # The frame label for the funnel; the host sets it.
    source_name: str = ""
    # The host's agent.yaml block, holding cooldown and timeout; the host sets it.
    config_key: str = ""
    # What the query holds besides the unanswered user messages (see retrieval_query).
    query_context: str = "previous_reply"
    # Built by init_recall.
    _recall_settings: RecallSettings | None = None
    _recall_gate: SearchGate | None = None

    def init_recall(self, config: dict, now: Callable[[], float] = time.monotonic) -> None:
        """Read the settings and build the gate; call from the host's ``on_init``.

        Raises ``ValueError`` for a malformed block, which aborts startup.
        """
        self._recall_settings = parse_recall_settings(config, self.config_key)
        self._recall_gate = SearchGate(self._recall_settings.cooldown, now)

    async def search(self, query: str, ctx: RecallContext) -> Sequence[Any] | None:
        """Return the hits for ``query`` in rank order, or None/empty for none.

        Overridden by the source. It may raise: the mixin logs it and the turn
        goes on without the source's results.
        """
        raise NotImplementedError(
            f"recall source {self.source_name!r} does not implement search(); "
            f"override RecallSourceMixin.search in its plugin class"
        )

    def render(self, hit: Any) -> str:
        """Render one hit as the line the funnel admits.

        Overridden by the source. Render stably: the funnel dedupes on whole
        text for now (#725), so a line that changes between turns defeats it.
        """
        raise NotImplementedError(
            f"recall source {self.source_name!r} does not implement render(); "
            f"override RecallSourceMixin.render in its plugin class"
        )

    @hookimpl
    async def on_funnel_ingest(self, channel, trace_id, annotations, submit) -> None:
        """Start a background search for this turn and submit its promise.

        Returns promptly. Does nothing when the tail is not a plain user
        message, the query is blank, a search is still pending for the channel
        (submitting again would cancel it) or the cooldown has not elapsed.
        """
        gate = self._recall_gate
        # The host forgot init_recall: name the fix instead of failing on None.
        if gate is None:
            raise RuntimeError(
                f"recall source {self.source_name!r} was used before init_recall(); "
                f"call self.init_recall(config) from its on_init"
            )
        conv = getattr(channel, "conversation", None)
        # Only a window whose tail is a plain user message is searched for.
        if conv is None or not _tail_is_plain_user_message(conv.messages):
            return
        # The query memory builds; a blank one has nothing to search with.
        query = retrieval_query(conv.messages, self.query_context)
        if not query.strip():
            return
        # Single-flight and cooldown: skip quietly, the next turn asks again.
        if not gate.try_start(channel.id):
            logger.debug(
                "recall source %s skipped for %s: a search is pending or the cooldown has not elapsed",
                self.source_name, channel.id,
                extra={"channel_id": channel.id, "source": self.source_name},
            )
            return
        # Run the search in the background and release the slot however it ends.
        ctx = RecallContext(channel=channel, trace_id=trace_id, annotations=annotations)
        promise = asyncio.ensure_future(self._recall(query, ctx))
        promise.add_done_callback(lambda _done, cid=channel.id: gate.finish(cid))
        submit(self.source_name, promise)

    async def _recall(self, query: str, ctx: RecallContext) -> Ingest | None:
        """Run ``search`` under the timeout; resolve to an ``Ingest`` or None.

        Never raises, except ``CancelledError``: the funnel cancels a promise
        that expires or at shutdown, which is not a failure and is not logged.
        """
        settings = self._recall_settings
        assert settings is not None  # set by init_recall, checked before this runs
        channel_id = ctx.channel.id
        # Search under the bound; a slow or failing source costs this turn its results only.
        try:
            hits = await asyncio.wait_for(self.search(query, ctx), timeout=settings.timeout)
        except TimeoutError:
            logger.warning(
                "recall source %s timed out after %g s for channel %s; the turn went "
                "on without its results — raise %s.timeout in agent.yaml if the "
                "source is only slow",
                self.source_name, settings.timeout, channel_id, self.config_key,
                extra={"channel_id": channel_id, "source": self.source_name},
            )
            return None
        except Exception as exc:
            logger.warning(
                "recall source %s failed for channel %s (%s); the turn went on "
                "without its results — fix the source named here",
                self.source_name, channel_id, type(exc).__name__, exc_info=True,
                extra={"channel_id": channel_id, "source": self.source_name},
            )
            return None
        # Nothing found: admit nothing.
        if not hits:
            return None
        # Render each hit; one that cannot render is dropped and the rest stand.
        entries: list[str] = []
        for hit in hits:
            try:
                entries.append(self.render(hit))
            except Exception as exc:
                logger.warning(
                    "recall source %s could not render a hit for channel %s (%s); "
                    "dropped that hit and kept the rest — fix render() in the source named here",
                    self.source_name, channel_id, type(exc).__name__, exc_info=True,
                    extra={"channel_id": channel_id, "source": self.source_name},
                )
        return Ingest(entries=entries) if entries else None
