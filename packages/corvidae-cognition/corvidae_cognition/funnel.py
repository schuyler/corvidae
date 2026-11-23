"""FunnelPlugin — the held-text admission policy (#1176).

The kernel's ``held`` store holds text for a channel's next turn, frames it as
data and appends it. What it leaves to policy is *what* of the held text a turn
admits, and this plugin is that policy: it implements
``transform_held_admission``, which the store calls once per turn with every due
batch, and answers with one plan per batch. Without this plugin the store's seed
admits every entry as offered.

Behavior, per batch and in the order offered:
  1. TTL: when ``funnel.ingest_ttl`` is set, a batch with no origin (not a stub
     batch) older than the TTL on the kernel clock is dropped whole.
  2. Dedupe: an entry whose exact text already appears inside a CONTEXT row in
     the window is a duplicate. A batch may carry keys (#624): then an entry is
     a duplicate when its key appears inside a CONTEXT row framed for the same
     source, so a stable identity (memory's record id) dedupes even when the
     rendered line (its age, its band) reads differently now.
  3. Budget: token-count entries and admit them in order, skipping any entry
     that would overrun what is left of the batch's budget. When the source has
     a window cap (``funnel.window_budgets.<source>``), the budget is also
     limited to what is left under the cap after the source's CONTEXT already in
     the window. An entry larger than the source's whole budget or cap is
     dropped with a warning, since it can never fit. A skip on a ``once`` batch
     is a drop (the batch is offered at one turn only); on any other batch it is
     left unlisted, so the store keeps the entry for a later turn.

The entries the call admits from earlier batches of the same source count as
already in the window, for dedupe and the window cap alike, because the store
appends them in the order the plans come back.

Ponder: the funnel also gives the agent a ``ponder`` tool: a wait of at most
``PONDER_TIMEOUT_SECONDS`` for recall already in flight. The before-compose
gather runs before the turn's model call, so the stragglers the store holds for
the channel when the agent calls a tool are the recall still in flight for it.
The tool waits on those (it returns at the first that brought entries, or once
all have settled empty) and returns at once when none is running. A message that
arrives during the wait gets its own turn and its own gather; the tool does not
follow it. The call stays outstanding, so the kernel holds the next assistant
turn and counts the channel busy until it returns.

Config:
    funnel:
      default_budget: 512      # tokens per batch offered
      budgets:                 # optional per-source overrides
        memory: 512
      window_budgets:          # optional per-source window caps (unset: none)
        memory: 4096
      ingest_ttl: 600          # seconds before a held batch is dropped (unset: never)
"""

from __future__ import annotations

import asyncio
import logging
import time

from corvidae.clock import Clock
from corvidae.context import ContextWindow, MessageType, count_tokens
from corvidae.held import FRAME_HEADER, BatchPlan, HeldBatch, HeldContext, frame
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.tool import Tool, ToolContext, tool_to_schema

logger = logging.getLogger("corvidae.funnel")

DEFAULT_BUDGET_TOKENS = 512

# How long one ponder call may wait for recall to land, in seconds. It is a
# timeout so a slow source cannot stall the agent, not a pause to fill.
PONDER_TIMEOUT_SECONDS = 5.0


def _read_window_budgets(raw: object) -> dict[str, int]:
    """Validate ``funnel.window_budgets`` into ``{source: cap}``.

    A missing key means no caps. A non-mapping value, or a source whose cap
    is not a non-negative integer, logs a warning naming the key, the value
    and what is accepted, and is ignored (that source stays uncapped).
    """
    # Unset: no source is capped.
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "funnel.window_budgets must be a mapping of source name to a "
            "non-negative integer token cap, got %r; ignoring it, so no "
            "source is capped — fix funnel.window_budgets in agent.yaml",
            raw,
        )
        return {}
    caps: dict[str, int] = {}
    for source, cap in raw.items():
        # bool is an int subclass, but True is not a token count.
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            logger.warning(
                "funnel.window_budgets.%s must be a non-negative integer "
                "token cap, got %r; ignoring it, so %s is uncapped — fix "
                "funnel.window_budgets.%s in agent.yaml",
                source, cap, source, source,
            )
            continue
        caps[source] = cap
    return caps


def _read_ingest_ttl(raw: object) -> float | None:
    """Validate ``funnel.ingest_ttl`` into seconds, or ``None`` for no expiry.

    Unset means queued ingests never expire. A value that is not a number
    greater than 0 (a bool is not a number here) logs a warning naming the
    key, the value and what is accepted, and expiry stays off.
    """
    # Unset: nothing expires.
    if raw is None:
        return None
    # bool is an int subclass, but True is not a number of seconds.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        logger.warning(
            "funnel.ingest_ttl must be a number of seconds greater than 0, "
            "got %r; expiry stays off — set funnel.ingest_ttl to a positive "
            "number of seconds in agent.yaml, or remove it",
            raw,
        )
        return None
    return float(raw)


def _landed(straggler: asyncio.Future[bool]) -> bool:
    """True when ``straggler`` finished with ``True``: its late answer brought entries.

    A cancelled or failed future, and one that settled ``False``, all brought
    nothing, so none of them counts as context arriving.
    """
    # Unfinished or cancelled: nothing to read (result() would raise on a cancel).
    if not straggler.done() or straggler.cancelled():
        return False
    # Failed: nothing arrived.
    if straggler.exception() is not None:
        return False
    return straggler.result() is True


def _source_rows(rows: list[str], source: str) -> list[str]:
    """The CONTEXT ``rows`` framed for ``source``.

    The store opens every row it writes with the source's frame header, so the
    header identifies which source a row came from.
    """
    header = FRAME_HEADER.format(source=source)
    return [row for row in rows if row.startswith(header)]


class FunnelPlugin(CorvidaePlugin):
    """The admission policy for held text: dedupe, budgets, window caps, TTL."""

    reloadable_keys = (
        "funnel.default_budget",
        "funnel.budgets",
        "funnel.window_budgets",
        "funnel.ingest_ttl",
    )

    # Resolved when first needed (by type), never at on_init: the store and the
    # clock are kernel plugins that exist by then, but a test registers its own.
    depends_on = frozenset({"held", "clock"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # funnel.ingest_ttl in seconds; None means held batches never expire.
        self._ttl: float | None = None
        self._default_budget: int = DEFAULT_BUDGET_TOKENS
        self._budgets: dict[str, int] = {}
        # Per-source caps on what one window holds (funnel.window_budgets).
        self._window_budgets: dict[str, int] = {}

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read funnel.* config."""
        await super().on_init(pm, config)
        self._read_config(config)

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Re-read budgets, window caps and the ingest TTL on config reload."""
        self._read_config(config)

    def _read_config(self, config: dict) -> None:
        """Read and validate the four ``funnel.*`` keys."""
        funnel_cfg = config.get("funnel", {}) or {}
        self._default_budget = funnel_cfg.get("default_budget", DEFAULT_BUDGET_TOKENS)
        self._budgets = dict(funnel_cfg.get("budgets", {}) or {})
        self._window_budgets = _read_window_budgets(funnel_cfg.get("window_budgets"))
        self._ttl = _read_ingest_ttl(funnel_cfg.get("ingest_ttl"))

    @hookimpl
    async def transform_held_admission(
        self, channel, conversation: ContextWindow, batches: list[HeldBatch]
    ) -> list[BatchPlan]:
        """One plan per batch: TTL, dedupe and budget, in the order offered.

        Rows the call admits from earlier batches of a source are remembered as
        the store will frame them, so a later batch of that source dedupes and
        counts against the window cap as if those rows were in the window.
        """
        # The window's CONTEXT rows.
        window_rows = [
            msg.get("content") or ""
            for msg in conversation.messages
            if msg.get("_message_type") == MessageType.CONTEXT
        ]
        now = get_dependency(self.pm, "clock", Clock).monotonic()
        # Source -> rows this call has admitted for it so far.
        admitted_rows: dict[str, list[str]] = {}
        plans: list[BatchPlan] = []
        for index, batch in enumerate(batches):
            rows = window_rows + admitted_rows.get(batch.source, [])
            plan = self._plan(channel.id, index, batch, rows, now)
            plans.append(plan)
            # Later batches of this source see these entries as a window row.
            if plan.admit:
                admitted_rows.setdefault(batch.source, []).append(
                    frame(batch.source, [batch.entries[i] for i in plan.admit])
                )
        return plans

    def _plan(
        self, channel_id: str, index: int, batch: HeldBatch, rows: list[str], now: float
    ) -> BatchPlan:
        """Plan one batch against ``rows`` (the window's CONTEXT as it will stand)."""
        entries = batch.entries
        everything = list(range(len(entries)))
        # 1. TTL: a batch that is not a stub batch expires by kernel-clock age.
        if self._ttl is not None and batch.origin is None and now - batch.created > self._ttl:
            if entries:
                logger.info(
                    "funnel dropped %d held %s entr%s on channel %s: held %.0fs, past "
                    "funnel.ingest_ttl of %gs",
                    len(entries), batch.source, "y" if len(entries) == 1 else "ies",
                    channel_id, now - batch.created, self._ttl,
                    extra={"channel_id": channel_id, "source": batch.source},
                )
            return BatchPlan(index, [], [], everything)

        # 2. Dedupe against the window's CONTEXT. Entries live inside framed
        # rows, so substring containment is the match. Keyed entries match
        # their key against this source's rows only; keyless ones match their
        # whole text against every CONTEXT row.
        own_rows = _source_rows(rows, batch.source) if batch.keys is not None else []
        duplicate: list[int] = []
        fresh: list[int] = []
        for i in everything:
            if batch.keys is not None:
                seen = any(batch.keys[i] in row for row in own_rows)
            else:
                seen = any(entries[i] in row for row in rows)
            (duplicate if seen else fresh).append(i)
        if duplicate:
            logger.debug(
                "funnel deduped %d entr%s already in window",
                len(duplicate), "y" if len(duplicate) == 1 else "ies",
                extra={"channel_id": channel_id, "source": batch.source},
            )

        # 3. Budget, limited by what is left under the source's window cap
        # after its CONTEXT already in the window (frames included).
        # shortcut: new entries are charged without their frame, so the
        # window can pass the cap by one frame; charge it if caps under ~200
        # tokens matter.
        budget = self._budgets.get(batch.source, self._default_budget)
        cap = self._window_budgets.get(batch.source)
        room: int | None = None
        held = 0
        if cap is not None:
            held = sum(count_tokens(row) for row in _source_rows(rows, batch.source))
            room = max(cap - held, 0)
        limit = budget if room is None else min(budget, room)
        admit: list[int] = []
        drop: list[int] = []
        spent = 0
        skipped = 0
        for i in fresh:
            tokens = count_tokens(entries[i])
            # An entry larger than the whole cap or budget can never fit, on
            # any turn, so it is dropped whatever kind of batch it is in.
            if (cap is not None and tokens > cap) or tokens > budget:
                bound = f"{cap}-token window cap, so raise funnel.window_budgets.{batch.source}" \
                    if cap is not None and tokens > cap \
                    else f"{budget}-token budget, so raise funnel.budgets.{batch.source}"
                logger.warning(
                    "funnel dropped a %s entry on channel %s: %d tokens exceeds the "
                    "%s to admit entries this size",
                    batch.source, channel_id, tokens, bound,
                    extra={"channel_id": channel_id, "source": batch.source},
                )
                drop.append(i)
            elif spent + tokens > limit:
                # Skipped: dropped on a once batch, left unlisted (held) otherwise.
                skipped += 1
                if batch.once:
                    drop.append(i)
            else:
                spent += tokens
                admit.append(i)
        if skipped and room is not None and room < budget:
            # The window cap bound, not the per-batch budget.
            logger.info(
                "funnel skipped %d %s entr%s on channel %s: the window holds %d of "
                "its %d-token cap (per-batch budget %d); compaction frees room, or "
                "raise funnel.window_budgets.%s",
                skipped, batch.source, "y" if skipped == 1 else "ies", channel_id,
                held, cap, budget, batch.source,
                extra={"channel_id": channel_id, "source": batch.source},
            )
        elif skipped:
            logger.info(
                "funnel skipped %d %s entr%s on channel %s over the %d-token budget",
                skipped, batch.source, "y" if skipped == 1 else "ies", channel_id, budget,
                extra={"channel_id": channel_id, "source": batch.source},
            )
        return BatchPlan(index, admit, duplicate, sorted(drop))

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Give the agent ``ponder``, a bounded wait for recall already in flight."""
        # The tool is a closure so the schema builder sees a plain function; it
        # reaches the funnel's state through this reference.
        plugin = self

        async def ponder(_ctx: ToolContext | None = None) -> str:
            """Pause for a moment (up to 5 seconds) to let your recollections catch up; returns at once if none are still coming in."""
            return await plugin._ponder(_ctx)

        # Same wrapping goals uses, so the schema skips the injected _ctx.
        tool_registry.append(Tool(name=ponder.__name__, fn=ponder, schema=tool_to_schema(ponder)))

    async def _ponder(self, ctx: ToolContext | None) -> str:
        """Wait for this channel's before-compose stragglers, up to the timeout.

        The gather for the turn ran before its model call, so the store's
        stragglers for the channel are the recall in flight when this is
        called. Returns at once when one already landed with entries or none is
        still running; otherwise waits for the first that brings entries, or
        for all to settle empty. The futures belong to the store: waiting never
        cancels them, and nothing is admitted here, since the next turn does that.
        """
        # The channel decides whose stragglers to wait on.
        if ctx is None or ctx.channel is None:
            return (
                "Error: ponder has no channel context; it can only run inside a "
                "channel turn, so do not call ponder here"
            )
        futures = get_dependency(self.pm, "held", HeldContext).in_flight(ctx.channel.id)

        # A recall already landed with entries: the next turn admits it.
        if any(_landed(f) for f in futures):
            return "Recall already landed: it goes to the admission policy on the next turn (budget, dedupe and TTL still apply)."
        # Only an unfinished future can still bring something.
        running = [f for f in futures if not f.done()]
        if not running:
            return "Nothing is being recalled for this channel, so there is nothing to wait for."

        timeout = PONDER_TIMEOUT_SECONDS
        start = time.monotonic()
        # asyncio.wait, never wait_for: wait_for would cancel the recall on timeout.
        while running:
            # The timeout bounds the whole call, so it is not restarted by each settle.
            remaining = timeout - (time.monotonic() - start)
            done, still_running = await asyncio.wait(
                running, timeout=remaining, return_when=asyncio.FIRST_COMPLETED,
            )
            # Nothing finished within what was left of the timeout.
            if not done:
                break
            elapsed = time.monotonic() - start
            # A future that brought entries ends the wait; the others can follow.
            if any(_landed(f) for f in done):
                return f"Recall landed after {elapsed:.1f}s: it goes to the admission policy on the next turn (budget, dedupe and TTL still apply)."
            # Settled empty, failed or cancelled: keep waiting on the rest.
            running = list(still_running)

        # Every future settled without entries.
        if not running:
            return f"Recall finished after {time.monotonic() - start:.1f}s with nothing to add."
        # Timed out: the recalls keep running and are held when they land.
        return (
            f"Still waiting on recall after {timeout:g}s; carry on, whatever it finds "
            "is admitted on a later turn."
        )
