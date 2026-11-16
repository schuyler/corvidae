"""`corvidae reconsolidate`: re-consolidate a channel's history into a copy (#932).

An operator-only, offline command. It never runs inside the daemon, is not a
tool, and nothing calls it from a hook. It works on a *copy* of the daemon's
``sessions.db`` (``--snapshot-from`` makes one and marks it), never the live
file, so a prompt or segment-budget variant can be measured against real
history without touching Lex's memory (#583).

For each selected stretch of a channel it re-segments ``message_log`` at the
``--segment-budget`` flag (independent of ``max_context_tokens``), sends each
segment to the consolidation stage with the chosen instruction, and writes the
reply as fresh records stamped ``mode = standalone``, ``trigger =
reconsolidate``. The consolidation records (``fork`` / ``standalone``) whose
message ranges intersect the segment are *superseded*, not deleted: they get
``superseded_by`` and ``indexed = 0``, and their access statistics fold into
the new records the way a near-duplicate merge folds them (PRD §3.7 Memory 3).
Goal records, other channels' records, redacted records and records holding
another channel's text are left alone. ``corvidae redact`` reaches the stored
replies through the ``memory_reply`` scrubber (:func:`redact_memory_reply`),
but a *copy* is a whole duplicate of ``sessions.db``: redacting the live file
does not redact its copies, so delete a copy or redact it too.

What the copy gains, all added by this command and none read by the daemon:

- ``memory.segment_budget`` and ``memory.reply_id`` on every record it writes;
- ``memory_reply``: the model's whole reply for each segment (a later
  ``claims`` array can be indexed from it without another pass), with the
  instruction and system-prompt hashes (their texts live in ``memory_prompt``,
  PRD Memory 7);
- ``reconsolidation_copy`` (the marker that makes a file "a copy"),
  ``reconsolidation_run`` (the settings a resumed run must repeat) and
  ``reconsolidation_plan`` (the stretches chosen, fixed at the first start so
  a resume does not re-select from a database it has already changed).

Each segment commits in one transaction (its reply, its records, the
supersede and the vector rows), so a run that stops is resumable: the segments
with a ``memory_reply`` row are skipped on the next start.

shortcut: no near-duplicate merge against other records, no halving of an
embedding batch the server rejects (the entry is stored ``embedded = 0`` for
the daemon's backfill), and no runtime overrides (``set_settings``) or
``transform_system_prompt`` hooks, since there is no daemon; ``--system-prompt``
is the final prompt. Upgrade when a measurement needs one of them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import click
import yaml

from corvidae.channel import ChannelRegistry, apply_channel_config, parse_channel_config
from corvidae.context import message_tokens
from corvidae.fork import ViewInputs
from corvidae.llm import LLMClient
from corvidae.llm_plugin import parse_llm_config
from corvidae.persistence import add_column_if_missing
from corvidae.thinking import apply_reasoning_history
from corvidae.prompts import resolve_prompt
from corvidae_cognition.memory import (
    CONSOLIDATION_EXTRA_BODY,
    RUBRIC_PROMPT,
    RecordStamp,
    RubricPrior,
    _consolidation_from_text,
    _measured_size,
    build_catchup_segments,
    consolidation_instruction,
    dialog_from_rows,
    fetch_range_rows,
    prompt_hash,
    resolve_consolidation_fork_prompt,
    resolve_entry_target_tokens,
    source_timestamp,
)

# A "corvidae.*" name, so the daemon's logging configuration reaches it (the
# cognition modules never use __name__; tests/cognition/test_logger_names.py).
logger = logging.getLogger("corvidae.reconsolidate")

#: What the records this command writes carry as ``trigger`` (owner ruling,
#: #932): a rebuilt-view call is ``standalone`` by PRD Memory 2's definition.
RECONSOLIDATE_TRIGGER = "reconsolidate"
RECONSOLIDATE_MODE = "standalone"

#: Record modes the supersede rule may demote (consolidation output only).
_CONSOLIDATION_MODES = ("fork", "standalone")

#: Marks a file as a copy this command made; a live sessions.db has none.
COPY_MARKER_DDL = (
    "CREATE TABLE IF NOT EXISTS reconsolidation_copy "
    "(source TEXT NOT NULL, created_at REAL NOT NULL)"
)
#: The settings a resumed run must repeat (one row per key).
RUN_DDL = (
    "CREATE TABLE IF NOT EXISTS reconsolidation_run "
    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
)
#: The stretches chosen per channel, fixed at the first start. A channel with
#: nothing to do holds one (-1, -1) row so it is not re-planned.
PLAN_DDL = (
    "CREATE TABLE IF NOT EXISTS reconsolidation_plan "
    "(channel_id TEXT NOT NULL, after_id INTEGER NOT NULL, through_id INTEGER NOT NULL, "
    "PRIMARY KEY (channel_id, after_id))"
)
#: One row per finished segment: the model's whole reply and what made it.
REPLY_DDL = (
    "CREATE TABLE IF NOT EXISTS memory_reply ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "channel_id TEXT NOT NULL, "
    "msg_id_start INTEGER NOT NULL, "
    "msg_id_end INTEGER NOT NULL, "
    "segment_budget INTEGER NOT NULL, "
    "prompt_hash TEXT NOT NULL, "
    "system_prompt_hash TEXT NOT NULL, "
    "model TEXT, "
    "reply TEXT, "  # NULL: the segment held no dialog, so no call was made
    "created_at REAL NOT NULL, "
    "UNIQUE (channel_id, msg_id_start, msg_id_end))"
)

# The client keys LLMClient takes: derived, not hand-listed, so a constructor
# rename cannot silently drop a profile key (the same rule the replay uses).
_LLM_CLIENT_KEYS = frozenset(inspect.signature(LLMClient.__init__).parameters) - {"self"}

#: ``agent.mid_conversation_system_role`` values the daemon accepts.
_MID_ROLES = ("system", "user")


# ---------------------------------------------------------------------------
# Settings, clients and the report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSettings:
    """What one run does. The identity fields (everything but ``channels``)
    must match on a resume."""

    segment_budget: int
    instruction: str        # the full text sent: prompt plus the size sentence
    system_prompt: str      # heads every view; recorded by hash (PRD Memory 7)
    channels: tuple[str, ...] | None = None   # None: every channel in message_log
    since: float | None = None
    until: float | None = None
    min_source_tokens: int | None = None
    rubric_prompt: str = RUBRIC_PROMPT

    @property
    def scoped(self) -> bool:
        """Whether any flag narrows the selection below the whole channel."""
        return not (self.since is None and self.until is None and self.min_source_tokens is None)

    def identity(self) -> dict[str, str]:
        """The settings a resumed run must repeat, as stored strings."""
        return {
            "segment_budget": str(self.segment_budget),
            "prompt_hash": prompt_hash(self.instruction),
            "system_prompt_hash": prompt_hash(self.system_prompt),
            "since": repr(self.since),
            "until": repr(self.until),
            "min_source_tokens": repr(self.min_source_tokens),
        }


@dataclass
class Clients:
    """The model clients a run uses. ``embedding`` None stores every entry
    ``embedded = 0`` for the daemon's backfill."""

    consolidation: object
    rubric: object
    embedding: object | None = None
    #: (encoder, dimensions, document_prefix, query_prefix) the config embeds
    #: with; compared with ``embedding_meta`` before any vector is written.
    embedding_identity: tuple[str, int, str, str] | None = None


@dataclass
class RunReport:
    """What a run did (or, with ``dry_run``, would do)."""

    segments_total: int = 0
    segments_done: int = 0
    segments_skipped: int = 0
    records_written: int = 0
    records_superseded: int = 0
    estimated_tokens: int = 0
    #: (channel_id, first message id, last message id) of segments left undone.
    failed: list[tuple[str, int, int]] = field(default_factory=list)
    #: Records the supersede rule took that the selection did not name.
    pulled: list[int] = field(default_factory=list)


@dataclass
class ChannelPlan:
    """The stretches of one channel to re-consolidate."""

    channel_id: str
    intervals: list[tuple[int, int]]   # (after_id, through_id] each, ascending
    selected: list[int]                # record ids the flags named
    pulled: list[int]                  # record ids the widening took beyond them


# ---------------------------------------------------------------------------
# The copy: snapshot, marker and the live-db refusal
# ---------------------------------------------------------------------------


def _backup(source: Path, dest: Path) -> None:
    """Copy ``source`` into a new ``dest`` with sqlite's online-backup API
    (consistent while the daemon writes), then mark it as a copy."""
    src = sqlite3.connect(source)
    try:
        dst = sqlite3.connect(dest)
        try:
            # The backup API reads a WAL database as a transaction, so the
            # copy is whole even while the daemon is committing.
            src.backup(dst)
            dst.execute(COPY_MARKER_DDL)
            dst.execute(
                "INSERT INTO reconsolidation_copy (source, created_at) VALUES (?, ?)",
                (str(source), time.time()),
            )
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()


async def snapshot_database(source: Path, dest: Path) -> None:
    """Snapshot ``source`` into the new file ``dest`` and mark it a copy.

    Raises:
        click.ClickException: ``source`` is missing, or ``dest`` exists (it
            is never overwritten), or the backup fails (nothing is left at
            ``dest`` then).
    """
    # The source must be a real file: sqlite would silently create an empty one.
    if not source.is_file():
        raise click.ClickException(
            f"Cannot snapshot {source}: database not found — check --snapshot-from"
        )
    # Never overwrite: a copy may hold a run in progress.
    if dest.exists():
        raise click.ClickException(
            f"Cannot snapshot into {dest}: it already exists — choose a new "
            "--db path (one copy per variant), or drop --snapshot-from to "
            "resume the copy that is there"
        )
    try:
        await asyncio.to_thread(_backup, source, dest)
    except (sqlite3.Error, OSError) as exc:
        # A half-written copy would look resumable; remove it and say so.
        logger.warning("snapshot of %s into %s failed", source, dest, exc_info=True)
        for leftover in (dest, Path(f"{dest}-wal"), Path(f"{dest}-shm")):
            leftover.unlink(missing_ok=True)
        raise click.ClickException(
            f"Snapshot of {source} into {dest} failed ({type(exc).__name__}: {exc}); "
            "nothing was written — retry once the database is readable"
        ) from exc


def check_is_copy(path: Path, *, live_path: Path | None) -> None:
    """Refuse anything but a copy this command made.

    Two guards: the file is not the daemon's configured database
    (``daemon.session_db``), and it carries the marker table a snapshot adds.

    Raises:
        click.ClickException: ``path`` is missing, is the live database, or
            is not marked.
    """
    if not path.is_file():
        raise click.ClickException(f"Database not found: {path} — check --db")
    # The configured live database is refused even if something marked it.
    if live_path is not None and live_path.exists() and os.path.samefile(path, live_path):
        raise click.ClickException(
            f"{path} is the daemon's live database (daemon.session_db); refusing "
            "to rewrite it — pass --snapshot-from with it and a new --db path"
        )
    conn = sqlite3.connect(path)
    try:
        marked = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'reconsolidation_copy'"
        ).fetchone()
    finally:
        conn.close()
    if marked is None:
        raise click.ClickException(
            f"{path} is not a reconsolidation copy (it has no reconsolidation_copy "
            "marker); make one with --snapshot-from <the daemon's sessions.db> "
            "and a new --db path"
        )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


async def _table_exists(db: aiosqlite.Connection, name: str) -> bool:
    """Whether ``name`` is a table (or virtual table) in the database."""
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ) as cursor:
        return await cursor.fetchone() is not None


async def require_memory_table(db: aiosqlite.Connection) -> None:
    """Refuse a database the memory plugin never wrote to.

    Raises:
        click.ClickException: the copy has no ``memory`` table, so there is
            nothing to re-consolidate.
    """
    if not await _table_exists(db, "memory"):
        raise click.ClickException(
            "The database has no memory table, so there are no records to "
            "supersede; run the daemon with the memory plugin enabled first, "
            "then snapshot it again"
        )


async def ensure_schema(db: aiosqlite.Connection) -> None:
    """Add this command's tables and columns to the copy (idempotent)."""
    for ddl in (RUN_DDL, PLAN_DDL, REPLY_DDL):
        await db.execute(ddl)
    # Nullable columns, so old rows read NULL and the daemon ignores them.
    await add_column_if_missing(db, "memory", "segment_budget", "INTEGER")
    await add_column_if_missing(db, "memory", "reply_id", "INTEGER")
    # The instruction texts are kept once per hash, beside the daemon's own.
    await db.execute(
        "CREATE TABLE IF NOT EXISTS memory_prompt (hash TEXT PRIMARY KEY, text TEXT NOT NULL)"
    )
    await db.commit()


async def _load_vec(db: aiosqlite.Connection) -> bool:
    """Load sqlite-vec on ``db`` when the copy has a ``memory_vec`` table.

    Returns whether vectors are in play. Superseding must delete the old
    records' vector rows, so a vec table without the extension is refused
    before anything is written.

    Raises:
        click.ClickException: ``memory_vec`` exists and the extension cannot
            be loaded.
    """
    if not await _table_exists(db, "memory_vec"):
        return False
    try:
        import sqlite_vec

        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
    except (ImportError, AttributeError, sqlite3.Error) as exc:
        logger.warning("cannot load sqlite-vec for memory_vec", exc_info=True)
        raise click.ClickException(
            "This database has memory_vec, which needs the sqlite-vec extension, "
            f"and it could not be loaded ({type(exc).__name__}). Nothing was "
            "changed. Install the cognition extra (uv sync --extra cognition)."
        ) from exc
    return True


async def redact_memory_reply(
    db: aiosqlite.Connection,
    channel_id: str,
    msg_ids: list[int],
    dry_run: bool,
    tombstone: str,
) -> int | None:
    """Tombstone the stored replies whose segment covers a redacted message.

    Registered as the ``memory_reply`` scrubber in the ``corvidae.redact``
    entry-point group. A reply restates its segment's conversation in the
    model's words, so a redaction of a message in that segment must reach it
    as it reaches the segment's records. The row stays (a resume still sees
    the segment as finished); only ``reply`` is overwritten. The predicate
    runs once per id so a large range never exceeds SQLite's bound-variable
    limit. Returns the rows changed (or that would be, in dry-run mode), or
    None when the table is absent (a database no reconsolidation ever ran on).
    """
    # A database no reconsolidation touched has no table to scrub.
    if not await _table_exists(db, "memory_reply"):
        return None
    changed: set[int] = set()
    for msg_id in msg_ids:
        async with db.execute(
            "SELECT id FROM memory_reply WHERE channel_id = ? AND msg_id_start <= ? "
            "AND msg_id_end >= ? AND reply IS NOT NULL AND reply != ?",
            (channel_id, msg_id, msg_id, tombstone),
        ) as cursor:
            changed.update(row[0] for row in await cursor.fetchall())
    if not dry_run:
        for reply_id in sorted(changed):
            await db.execute("UPDATE memory_reply SET reply = ? WHERE id = ?", (tombstone, reply_id))
        await db.commit()
    return len(changed)


# ---------------------------------------------------------------------------
# Planning: which stretches of a channel to redo
# ---------------------------------------------------------------------------


def _intersects(start: int, end: int, after_id: int, through_id: int) -> bool:
    """Whether the record range [start, end] meets the stretch (after_id, through_id]."""
    return start <= through_id and end > after_id


async def _merge_intervals(
    db: aiosqlite.Connection, channel_id: str, intervals: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Join stretches that overlap or have no message of the channel between.

    ``message_log`` ids are global, so two stretches that look apart by id may
    be neighbours in the channel's own sequence; segmenting them together
    avoids a tiny segment at each seam.
    """
    merged: list[tuple[int, int]] = []
    for after_id, through_id in sorted(intervals):
        if merged:
            prev_after, prev_through = merged[-1]
            gap = 0
            if after_id > prev_through:
                # Any channel message in (prev_through, after_id] separates them.
                async with db.execute(
                    "SELECT COUNT(*) FROM message_log "
                    "WHERE channel_id = ? AND id > ? AND id <= ?",
                    (channel_id, prev_through, after_id),
                ) as cursor:
                    gap = (await cursor.fetchone())[0]
            if after_id <= prev_through or gap == 0:
                merged[-1] = (prev_after, max(prev_through, through_id))
                continue
        merged.append((after_id, through_id))
    return merged


async def _source_tokens(
    db: aiosqlite.Connection, channel_id: str, start: int, end: int
) -> int:
    """Tokens in a record's source range, measured the way the segmenter
    measures a message (dialog rows only; ids are the channel's own)."""
    source = await fetch_range_rows(db, channel_id, start - 1, end)
    return sum(message_tokens(m) for m in dialog_from_rows(source, channel_id))


async def plan_channel(
    db: aiosqlite.Connection, channel_id: str, settings: RunSettings
) -> ChannelPlan:
    """Choose the stretches of ``channel_id`` to re-consolidate.

    Unscoped: the whole channel from id 0. Scoped: the ranges of the active
    consolidation records the flags select, joined, then widened until no
    other active record reaches past a stretch it meets (superseding it would
    otherwise leave its far end with no indexed record). The records the
    widening took beyond the selection are returned as ``pulled``.
    """
    async with db.execute(
        "SELECT id, msg_id_start, msg_id_end, created_at FROM memory "
        "WHERE channel_id = ? AND mode IN (?, ?) AND superseded_by IS NULL "
        "AND redacted = 0 AND merged_foreign = 0 ORDER BY id",
        (channel_id, *_CONSOLIDATION_MODES),
    ) as cursor:
        active = await cursor.fetchall()

    # Unscoped: everything the channel has, no widening needed.
    if not settings.scoped:
        async with db.execute(
            "SELECT MAX(id) FROM message_log WHERE channel_id = ?", (channel_id,)
        ) as cursor:
            top = (await cursor.fetchone())[0]
        intervals = [(0, top)] if top is not None else []
        return ChannelPlan(channel_id, intervals, [r[0] for r in active], [])

    # Scoped: pick the records the flags name.
    selected: list[tuple[int, int, int, float]] = []
    for record in active:
        _id, start, end, created_at = record
        if settings.since is not None and created_at < settings.since:
            continue
        if settings.until is not None and created_at > settings.until:
            continue
        if (
            settings.min_source_tokens is not None
            and await _source_tokens(db, channel_id, start, end) < settings.min_source_tokens
        ):
            continue
        selected.append(record)
    if not selected:
        return ChannelPlan(channel_id, [], [], [])

    # Join the selected ranges, then widen over every active record they meet
    # until nothing reaches past a stretch (a fixed point).
    intervals = await _merge_intervals(
        db, channel_id, [(r[1] - 1, r[2]) for r in selected]
    )
    while True:
        grown = False
        for _id, start, end, _created in active:
            for i, (after_id, through_id) in enumerate(intervals):
                if _intersects(start, end, after_id, through_id):
                    widened = (min(after_id, start - 1), max(through_id, end))
                    if widened != (after_id, through_id):
                        intervals[i] = widened
                        grown = True
        intervals = await _merge_intervals(db, channel_id, intervals)
        if not grown:
            break
    chosen = {r[0] for r in selected}
    pulled = [
        r[0] for r in active
        if r[0] not in chosen and any(_intersects(r[1], r[2], a, t) for a, t in intervals)
    ]
    return ChannelPlan(channel_id, intervals, sorted(chosen), pulled)


async def _stored_plan(db: aiosqlite.Connection, channel_id: str) -> list[tuple[int, int]] | None:
    """The stretches fixed for ``channel_id`` at an earlier start, or None."""
    if not await _table_exists(db, "reconsolidation_plan"):
        return None
    async with db.execute(
        "SELECT after_id, through_id FROM reconsolidation_plan "
        "WHERE channel_id = ? ORDER BY after_id",
        (channel_id,),
    ) as cursor:
        found = await cursor.fetchall()
    if not found:
        return None
    # The (-1, -1) sentinel means "planned, nothing to do".
    return [(a, t) for a, t in found if a >= 0]


async def _done_segments(db: aiosqlite.Connection, channel_id: str) -> set[tuple[int, int]]:
    """(first id, last id) of the segments a ``memory_reply`` row already covers."""
    if not await _table_exists(db, "memory_reply"):
        return set()
    async with db.execute(
        "SELECT msg_id_start, msg_id_end FROM memory_reply WHERE channel_id = ?",
        (channel_id,),
    ) as cursor:
        return set(await cursor.fetchall())


async def _check_identity(db: aiosqlite.Connection, settings: RunSettings, where: str) -> None:
    """Record the run's settings on the first start; refuse a resume that differs.

    Raises:
        click.ClickException: a stored setting differs from this command's,
            naming each key, both values and the remedy.
    """
    wanted = settings.identity()
    async with db.execute("SELECT key, value FROM reconsolidation_run") as cursor:
        stored = dict(await cursor.fetchall())
    # First start: remember the settings.
    if not stored:
        await db.executemany(
            "INSERT INTO reconsolidation_run (key, value) VALUES (?, ?)", list(wanted.items())
        )
        await db.commit()
        return
    differing = [k for k, v in wanted.items() if stored.get(k) != v]
    if differing:
        detail = "; ".join(f"{k} is {stored.get(k)} there, {wanted[k]} here" for k in differing)
        raise click.ClickException(
            f"{where} already holds a run with other settings ({detail}); rerun "
            "with the same settings to resume it, or make a new --snapshot-from "
            "copy for a different variant"
        )


# ---------------------------------------------------------------------------
# Valence and embeddings
# ---------------------------------------------------------------------------


async def _mean_valence(
    db: aiosqlite.Connection, channel_id: str, start: int, end: int
) -> float | None:
    """Mean stage-2 appraisal valence over the range on this channel, or None.

    The same reading the appraisal plugin's ``mean_valence`` makes of
    ``exchange_log`` (the plugin is not running offline).
    """
    if not await _table_exists(db, "exchange_log"):
        return None
    async with db.execute(
        "SELECT trace_id, CAST(appraisal AS BLOB) FROM exchange_log "
        "WHERE message_rowid IS NOT NULL AND message_rowid >= ? AND message_rowid <= ? "
        "AND appraisal IS NOT NULL AND channel_id = ?",
        (start, end, channel_id),
    ) as cursor:
        found = await cursor.fetchall()
    valences: list[float] = []
    for trace_id, raw in found:
        try:
            stage2 = json.loads(raw).get("stage2")
        except (ValueError, AttributeError):
            # One corrupt appraisal loses only itself.
            logger.warning(
                "reconsolidate: appraisal for trace %s on %s is unreadable; skipping it "
                "in the range's valence", trace_id, channel_id, exc_info=True,
            )
            continue
        if isinstance(stage2, dict) and stage2.get("valence") is not None:
            valences.append(float(stage2["valence"]))
    return sum(valences) / len(valences) if valences else None


async def _embed(
    clients: Clients, texts: list[str], where: str, dimensions: int | None
) -> list[list[float] | None]:
    """Embed ``texts`` as documents; any failure stores them unembedded (None).

    ``dimensions`` None means vectors may not be written (no vec table, no
    client, or an encoder mismatch), so no call is made. The daemon backfills
    ``embedded = 0`` rows, so a failure degrades and is logged, never blocks
    the run.
    """
    if clients.embedding is None or dimensions is None:
        return [None] * len(texts)
    try:
        vectors = await clients.embedding.embed(texts, kind="document")
        if len(vectors) != len(texts):
            raise ValueError(f"embedding returned {len(vectors)} vectors for {len(texts)} texts")
    except Exception:
        logger.warning(
            "reconsolidate: embedding failed for %s (%d entries); storing them "
            "embedded=0 for the daemon's backfill", where, len(texts), exc_info=True,
        )
        return [None] * len(texts)
    # A vector of the wrong width would corrupt the index; keep only good ones.
    good = [v if len(v) == dimensions else None for v in vectors]
    if None in good:
        logger.warning(
            "reconsolidate: embedding for %s came back %s wide, not the index's %d; "
            "those entries are stored embedded=0", where, sorted({len(v) for v in vectors}),
            dimensions,
        )
    return good


async def _vector_dimensions(
    db: aiosqlite.Connection, clients: Clients, vec_in_play: bool
) -> int | None:
    """The width vectors may be written at, or None when none may be.

    Vectors need the vec table, an embedding client, and an ``embedding_meta``
    row equal to the configured encoder; any mismatch turns them off (with a
    warning), exactly as the daemon does, so encoders are never mixed.
    """
    if not vec_in_play or clients.embedding is None or clients.embedding_identity is None:
        return None
    async with db.execute(
        "SELECT encoder, dimensions, document_prefix, query_prefix FROM embedding_meta LIMIT 1"
    ) as cursor:
        row = await cursor.fetchone()
    stored = (row[0], row[1], row[2] or "", row[3] or "") if row else None
    if stored != clients.embedding_identity:
        logger.warning(
            "reconsolidate: embedding_meta records %s but the config embeds with %s; "
            "new records are stored embedded=0 — revert the llm config or let the "
            "daemon backfill", stored, clients.embedding_identity,
        )
        return None
    return clients.embedding_identity[1]


# ---------------------------------------------------------------------------
# One segment
# ---------------------------------------------------------------------------


async def _call_consolidation(
    client, stamp: RecordStamp, channel_id: str, messages: list[dict], span: tuple[int, int],
):
    """The segment's reply and its parse, or None when two attempts are unusable.

    Transport errors propagate (the run stops, finished segments are kept); an
    unusable reply is retried once, as the daemon does.
    """
    for attempt in range(2):
        response = await client.chat(messages, extra_body=CONSOLIDATION_EXTRA_BODY)
        content = response["choices"][0]["message"].get("content")
        text = content if isinstance(content, str) else ""
        try:
            return text, _consolidation_from_text(
                text, stamp, channel_id=channel_id, msg_id_range=span
            )
        except ValueError as exc:
            logger.warning(
                "reconsolidate: reply unusable for channel %s, message ids %d-%d "
                "(attempt %d/2): %s", channel_id, span[0], span[1], attempt + 1, exc,
            )
    return None


async def _commit_segment(
    db: aiosqlite.Connection, *, settings: RunSettings, channel_id: str,
    span: tuple[int, int], stamp: RecordStamp, reply_text: str | None,
    entries: list[tuple[str, list[str]]], participants: list[str], importance: float,
    valence: float | None, created_at: float, vectors: list[list[float] | None],
    vec_in_play: bool,
) -> tuple[int, int]:
    """Write one segment in one transaction; returns (records, superseded).

    The reply row, the new records, the supersede of every intersecting
    consolidation record, the folded statistics and the vector rows land
    together or not at all. With ``entries`` empty only the reply row is
    written (a segment with no dialog), so a resume skips it.
    """
    start, end = span
    try:
        # The two texts, once per hash (PRD Memory 7).
        for text in (settings.instruction, settings.system_prompt):
            await db.execute(
                "INSERT OR IGNORE INTO memory_prompt (hash, text) VALUES (?, ?)",
                (prompt_hash(text), text),
            )
        cursor = await db.execute(
            "INSERT INTO memory_reply (channel_id, msg_id_start, msg_id_end, segment_budget, "
            "prompt_hash, system_prompt_hash, model, reply, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (channel_id, start, end, settings.segment_budget,
             prompt_hash(settings.instruction), prompt_hash(settings.system_prompt),
             stamp.model, reply_text, time.time()),
        )
        reply_id = cursor.lastrowid
        if not entries:
            await db.commit()
            return 0, 0

        # The records this segment replaces: active consolidation records of
        # the channel whose range meets this one, plus those an earlier
        # segment of this run already superseded (so a record straddling a
        # seam is credited to each side, the owner's sum ruling).
        async with db.execute(
            "SELECT id, retrieval_count, last_retrieved_at, importance, superseded_by "
            "FROM memory WHERE channel_id = ? AND mode IN (?, ?) AND redacted = 0 "
            "AND merged_foreign = 0 AND msg_id_start <= ? AND msg_id_end >= ? "
            "AND (trigger IS NULL OR trigger != ?) "
            "AND (superseded_by IS NULL OR superseded_by IN "
            "     (SELECT id FROM memory WHERE trigger = ?))",
            (channel_id, *_CONSOLIDATION_MODES, end, start,
             RECONSOLIDATE_TRIGGER, RECONSOLIDATE_TRIGGER),
        ) as cursor:
            old = await cursor.fetchall()
        # Fold the statistics as a near-dup merge does: counts summed, the
        # latest access, the larger importance.
        folded_count = sum(r[1] or 0 for r in old)
        accessed = [r[2] for r in old if r[2] is not None]
        folded_last = max(accessed) if accessed else None
        folded_importance = max([importance] + [r[3] for r in old if r[3] is not None])

        new_ids: list[int] = []
        for (narrative, tags), vector in zip(entries, vectors):
            cursor = await db.execute(
                "INSERT INTO memory (channel_id, created_at, summary, importance, valence, "
                "topic_tags, participants, msg_id_start, msg_id_end, retrieval_count, "
                "last_retrieved_at, embedded, prompt_hash, mode, model, trigger, "
                "segment_budget, reply_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (channel_id, created_at, narrative, folded_importance, valence,
                 json.dumps(tags), json.dumps(participants), start, end, folded_count,
                 folded_last, 1 if (vec_in_play and vector is not None) else 0,
                 stamp.hash, stamp.mode, stamp.model, stamp.trigger,
                 settings.segment_budget, reply_id),
            )
            new_ids.append(cursor.lastrowid)
            if vec_in_play and vector is not None:
                import sqlite_vec

                await db.execute(
                    "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, sqlite_vec.serialize_float32(vector)),
                )

        # Demote the replaced records (first writer wins the pointer, so a
        # straddler keeps pointing at the first segment's first record) and
        # drop their vector rows.
        for old_id, *_rest in old:
            await db.execute(
                "UPDATE memory SET superseded_by = ?, indexed = 0, embedded = 0 "
                "WHERE id = ? AND superseded_by IS NULL",
                (new_ids[0], old_id),
            )
            if vec_in_play:
                await db.execute("DELETE FROM memory_vec WHERE memory_id = ?", (old_id,))
        await db.commit()
        return len(new_ids), len(old)
    except BaseException:
        # Nothing of the segment may land half-written.
        await db.rollback()
        raise


async def _run_segment(
    db: aiosqlite.Connection, *, settings: RunSettings, clients: Clients, channel_id: str,
    segment, dimensions: int | None, vec_in_play: bool, report: RunReport,
) -> None:
    """Consolidate one segment and commit it, or record that it was left."""
    span = (segment.rows[0][0], segment.end_id)
    dialog = dialog_from_rows(segment.rows, channel_id)
    model = clients.consolidation.request_model(CONSOLIDATION_EXTRA_BODY)
    stamp = RecordStamp(settings.instruction, RECONSOLIDATE_MODE, model, RECONSOLIDATE_TRIGGER)

    # No dialog: nothing to remember, but the segment is finished.
    if not dialog:
        await _commit_segment(
            db, settings=settings, channel_id=channel_id, span=span, stamp=stamp,
            reply_text=None, entries=[], participants=[], importance=0.0, valence=None,
            created_at=0.0, vectors=[], vec_in_play=vec_in_play,
        )
        report.segments_done += 1
        return

    result = await _call_consolidation(
        clients.consolidation, stamp, channel_id, segment.messages, span
    )
    if result is None:
        # Left undone: the old records stay, and a rerun retries it.
        report.failed.append((channel_id, span[0], span[1]))
        return
    reply_text, consolidation = result

    # Importance: the reply's own rating, else the rubric (as the daemon does).
    importance = await RubricPrior(
        lambda: clients.rubric, lambda: settings.rubric_prompt
    ).score(dialog, msg_id_range=span, hint=consolidation.importance)
    valence = await _mean_valence(db, channel_id, span[0], span[1])
    created_at = await source_timestamp(db, span[1])
    vectors = await _embed(
        clients, [e.narrative for e in consolidation.entries],
        f"{channel_id} {span[0]}-{span[1]}", dimensions,
    )
    written, superseded = await _commit_segment(
        db, settings=settings, channel_id=channel_id, span=span, stamp=stamp,
        reply_text=reply_text,
        entries=[(e.narrative, list(e.topic_tags)) for e in consolidation.entries],
        participants=list(consolidation.participants), importance=importance,
        valence=valence, created_at=created_at, vectors=vectors, vec_in_play=vec_in_play,
    )
    report.segments_done += 1
    report.records_written += written
    report.records_superseded += superseded


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


async def _channels(db: aiosqlite.Connection, settings: RunSettings) -> list[str]:
    """The channels to process: the ``--channel`` flags, else every channel
    with history."""
    async with db.execute(
        "SELECT DISTINCT channel_id FROM message_log ORDER BY channel_id"
    ) as cursor:
        present = [row[0] for row in await cursor.fetchall()]
    if settings.channels is None:
        return present
    missing = [c for c in settings.channels if c not in present]
    if missing:
        raise click.ClickException(
            f"Channel {missing[0]!r} has no message_log rows in the database; "
            f"channels present: {', '.join(present) or '(none)'} — check --channel"
        )
    return list(settings.channels)


async def run_reconsolidation(
    db: aiosqlite.Connection, *, settings: RunSettings,
    inputs_for: Callable[[str], ViewInputs], clients: Clients,
    dry_run: bool = False, echo: Callable[[str], None] = click.echo,
) -> RunReport:
    """Re-consolidate the selected stretches of each channel on the copy ``db``.

    With ``dry_run`` nothing is written and no model is called: the report
    lists segments, the estimated request tokens and the records the widening
    pulled in. Otherwise each segment commits on its own; a transport error
    propagates after the finished segments are kept, and an unusable reply
    leaves its segment in ``report.failed``.

    Raises:
        click.ClickException: the database cannot be used (no memory table, a
            vec table without the extension, a resume with other settings, an
            unknown channel).
    """
    report = RunReport()
    await require_memory_table(db)
    vec_in_play = await _load_vec(db)
    if not dry_run:
        await ensure_schema(db)
        await _check_identity(db, settings, "this copy")
    dimensions = (
        None if dry_run else await _vector_dimensions(db, clients, vec_in_play)
    )

    for channel_id in await _channels(db, settings):
        # The stretches were fixed at the first start; later starts reuse them.
        intervals = await _stored_plan(db, channel_id)
        if intervals is None:
            plan = await plan_channel(db, channel_id, settings)
            intervals = plan.intervals
            report.pulled.extend(plan.pulled)
            echo(
                f"{channel_id}: {len(plan.selected)} record(s) selected, "
                f"{len(plan.pulled)} pulled in by range"
                + (": " + ", ".join(f"memory {i}" for i in plan.pulled) if plan.pulled else "")
            )
            if not dry_run:
                await db.executemany(
                    "INSERT INTO reconsolidation_plan (channel_id, after_id, through_id) "
                    "VALUES (?, ?, ?)",
                    [(channel_id, a, t) for a, t in intervals] or [(channel_id, -1, -1)],
                )
                await db.commit()

        # The channel's own policies, with the segment budget in place of
        # max_context_tokens (the flag is independent of it).
        inputs = dataclasses.replace(inputs_for(channel_id), max_tokens=settings.segment_budget)
        done = await _done_segments(db, channel_id)
        for after_id, through_id in intervals:
            segments = await build_catchup_segments(
                db, channel_id, after_id, through_id, inputs, settings.instruction
            )
            for segment in segments:
                report.segments_total += 1
                span = (segment.rows[0][0], segment.end_id)
                if span in done:
                    report.segments_skipped += 1
                    continue
                if dry_run:
                    size = _measured_size(segment.messages)
                    report.estimated_tokens += size
                    echo(
                        f"  {channel_id} message ids {span[0]}-{span[1]}: "
                        f"{len(segment.rows)} row(s), ~{size} tokens"
                    )
                    continue
                await _run_segment(
                    db, settings=settings, clients=clients, channel_id=channel_id,
                    segment=segment, dimensions=dimensions, vec_in_play=vec_in_play,
                    report=report,
                )

    # The closing line: what a dry run would send, or what the run did.
    if dry_run:
        echo(
            f"would send {report.segments_total - report.segments_skipped} segment(s) "
            f"(~{report.estimated_tokens} tokens estimated), {report.segments_skipped} "
            "already done"
        )
    else:
        echo(
            f"{report.segments_done} segment(s) done, {report.segments_skipped} already "
            f"done, {len(report.failed)} left undone; {report.records_written} record(s) "
            f"written, {report.records_superseded} superseded"
        )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_time(flag: str, value: str | None) -> float | None:
    """A ``--since`` / ``--until`` value as epoch seconds.

    Accepts epoch seconds or an ISO-8601 date or datetime (UTC when it names
    no zone).

    Raises:
        click.ClickException: naming the flag, the value and the accepted forms.
    """
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise click.ClickException(
            f"{flag}: cannot read {value!r} as a time; use an ISO-8601 date or "
            "datetime (2026-11-01 or 2026-11-01T12:00, UTC unless it names a zone) "
            "or epoch seconds"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _load_config(config_path: Path) -> dict:
    """Read ``agent.yaml``.

    Raises:
        click.ClickException: the file cannot be read or is not YAML.
    """
    try:
        return yaml.safe_load(config_path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise click.ClickException(
            f"Cannot read config {config_path} ({type(exc).__name__}: {exc}) — "
            "check --config"
        ) from exc


def _configured_live_db(config_path: Path) -> Path | None:
    """The daemon's configured ``daemon.session_db``, or None when the config
    cannot say (the marker check still refuses an unmarked file)."""
    if not config_path.is_file():
        return None
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        logger.warning(
            "cannot read %s to find daemon.session_db; relying on the copy marker alone",
            config_path, exc_info=True,
        )
        return None
    name = (config.get("daemon") or {}).get("session_db", "sessions.db")
    return config_path.parent / name


def _build_client(llm_config, stage: str, kind: str) -> tuple[LLMClient, dict]:
    """The offline client for ``stage``'s profile, as the daemon resolves it.

    Raises:
        click.ClickException: no profile serves the stage.
    """
    profile = llm_config.profile_for(stage, kind)
    if profile is None:
        raise click.ClickException(
            f"No llm profile serves the {stage!r} stage (llm.stages.{stage} or "
            f"llm.defaults.{kind}); configure one in agent.yaml"
        )
    cfg = llm_config.profiles[profile]
    return LLMClient(**{k: cfg[k] for k in _LLM_CLIENT_KEYS if k in cfg}), cfg


def _inputs_resolver(config: dict, system_prompt: str) -> Callable[[str], ViewInputs]:
    """Per-channel ViewInputs from ``agent.yaml``, layered as the daemon layers
    them (exact entry, transport-wide entries, ``agent:``, built-ins).

    Only the reasoning policy varies per channel here; the budget is the
    ``--segment-budget`` flag and the system prompt is the operator's file.
    """
    agent_cfg = config.get("agent") or {}
    registry = ChannelRegistry(agent_defaults=agent_cfg)
    # Validated the way the daemon validates it, so a bad channels: block fails here.
    try:
        apply_channel_config(parse_channel_config(config), registry)
    except ValueError as exc:
        raise click.ClickException(f"{exc} — check the channels: section of agent.yaml") from exc
    mid_role = agent_cfg.get("mid_conversation_system_role", "system")
    if mid_role not in _MID_ROLES:
        logger.warning(
            "agent.mid_conversation_system_role %r is not one of %s; using 'system'",
            mid_role, ", ".join(_MID_ROLES),
        )
        mid_role = "system"
    # No daemon key file offline; prompts here are not compared across runs.
    fence_key = os.urandom(32)

    def resolve(channel_id: str) -> ViewInputs:
        transport, _, scope = channel_id.partition(":")
        resolved = registry.resolve_config(registry.get_or_create(transport, scope))
        policy = resolved["reasoning_history"]
        return ViewInputs(
            system_prompt=system_prompt,
            max_tokens=resolved["max_context_tokens"],
            mid_role=mid_role,
            apply_reasoning_history=lambda messages: apply_reasoning_history(messages, policy),
            fence_key=fence_key,
        )

    return resolve


@click.command("reconsolidate")
@click.option("--db", "db_path", required=True, type=click.Path(path_type=Path),
              help="The copy to rewrite (never the daemon's live sessions.db).")
@click.option("--snapshot-from", "snapshot_from", type=click.Path(path_type=Path), default=None,
              help="Snapshot this sessions.db into a new --db first and mark it a copy.")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default="agent.yaml",
              show_default=True, help="agent.yaml: llm profiles, channels, memory settings.")
@click.option("--system-prompt", "system_prompt_file", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="The identity + register that heads every view (recorded on each segment).")
@click.option("--prompt", "prompt_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="The consolidation instruction; default memory.consolidation_fork_prompt.")
@click.option("--segment-budget", type=click.IntRange(min=1), required=True,
              help="Tokens per segment, independent of max_context_tokens; stamped on each record.")
@click.option("--channel", "channels", multiple=True,
              help="Only this channel id (repeatable); default every channel.")
@click.option("--since", default=None,
              help="Only records written at or after this time (ISO-8601 UTC or epoch seconds).")
@click.option("--until", default=None,
              help="Only records written at or before this time (ISO-8601 UTC or epoch seconds).")
@click.option("--min-source-tokens", type=click.IntRange(min=1), default=None,
              help="Only records whose source range holds at least this many tokens.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Report segments, estimated tokens and pulled-in records; write nothing.")
def reconsolidate_command(
    db_path: Path, snapshot_from: Path | None, config_path: Path, system_prompt_file: Path,
    prompt_file: Path | None, segment_budget: int, channels: tuple[str, ...],
    since: str | None, until: str | None, min_source_tokens: int | None, dry_run: bool,
) -> None:
    """Re-consolidate a channel's history into a COPY of sessions.db.

    Operator-only and offline. Records the consolidation stage writes replace
    (supersede, never delete) the daemon's consolidation records whose message
    ranges they cover. Rerun the same command to resume.
    """
    # Cheap flag checks first, so a typo fails before any file is touched.
    since_ts = _parse_time("--since", since)
    until_ts = _parse_time("--until", until)

    async def _run() -> None:
        # The copy: made now, or already there and marked.
        if snapshot_from is not None:
            await snapshot_database(snapshot_from, db_path)
        check_is_copy(db_path, live_path=_configured_live_db(config_path))

        # Everything the model calls need, from the operator's files and config.
        config = _load_config(config_path)
        try:
            system_prompt = system_prompt_file.read_text()
            base = prompt_file.read_text() if prompt_file else resolve_consolidation_fork_prompt(config)
            instruction = consolidation_instruction(base, resolve_entry_target_tokens(config))
            rubric_prompt = resolve_prompt(config, "memory.rubric_prompt", RUBRIC_PROMPT)
            llm_config = parse_llm_config(config.get("llm"))
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                f"Cannot build the run's inputs ({type(exc).__name__}: {exc}) — check "
                f"--system-prompt, --prompt and {config_path}"
            ) from exc
        consolidation, _ = _build_client(llm_config, "consolidation", "chat")
        rubric, _ = _build_client(llm_config, "rubric", "chat")
        embedding = identity = None
        if llm_config.profile_for("embedding", "embedding") is not None:
            embedding, cfg = _build_client(llm_config, "embedding", "embedding")
            identity = (
                cfg.get("model"), cfg.get("dimensions"),
                cfg.get("document_prefix", "") or "", cfg.get("query_prefix", "") or "",
            )
        settings = RunSettings(
            segment_budget=segment_budget, instruction=instruction,
            system_prompt=system_prompt, channels=tuple(channels) or None, since=since_ts,
            until=until_ts, min_source_tokens=min_source_tokens, rubric_prompt=rubric_prompt,
        )
        inputs_for = _inputs_resolver(config, system_prompt)

        started = [consolidation] + ([rubric] if rubric is not consolidation else [])
        started += [embedding] if embedding is not None else []
        for client in started:
            await client.start()
        try:
            async with aiosqlite.connect(db_path) as db:
                await db.execute("PRAGMA busy_timeout = 5000")
                try:
                    report = await run_reconsolidation(
                        db, settings=settings, inputs_for=inputs_for,
                        clients=Clients(consolidation, rubric, embedding, identity),
                        dry_run=dry_run,
                    )
                except click.ClickException:
                    raise
                except Exception as exc:
                    # A transport failure: finished segments are kept.
                    logger.warning("reconsolidate stopped on %s", db_path, exc_info=True)
                    raise click.ClickException(
                        f"Stopped on {type(exc).__name__}: {exc}. Segments already "
                        f"committed in {db_path} are kept; if the model server is "
                        "unreachable, check llm.providers.<name>.base_url, then "
                        "rerun the same command to resume"
                    ) from exc
        finally:
            for client in started:
                await client.stop()
        if report.failed:
            spans = ", ".join(f"{c} {a}-{b}" for c, a, b in report.failed)
            raise click.ClickException(
                f"{len(report.failed)} segment(s) got no usable reply and were left "
                f"undone ({spans}); their old records are untouched — rerun the same "
                "command to retry them"
            )

    asyncio.run(_run())
