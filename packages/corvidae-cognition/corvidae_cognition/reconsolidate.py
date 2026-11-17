"""`corvidae reconsolidate`: re-consolidate a channel's history into a copy (#932).

An operator-only, offline command (never a tool; no hook calls it). It works
on a *copy* of ``sessions.db`` (``--snapshot-from`` makes and marks one), never
the live file, so a prompt or segment-budget variant can be measured on real
history (#583).

Per channel it re-segments ``message_log`` at ``--segment-budget`` tokens with
``build_catchup_segments`` (the daemon's catch-up builder), sends each segment
to the consolidation stage with the chosen instruction, and stores the reply
as records stamped ``mode = standalone``, ``trigger = reconsolidate``. Active
``fork`` / ``standalone`` records whose range meets a segment are superseded
(``superseded_by``, ``indexed = 0``, vector row dropped), with their
``retrieval_count`` summed, ``last_retrieved_at`` the latest and ``importance``
the larger, as a near-dup merge folds them. Goal records and every other kind
are left alone.

Added to the copy only: ``memory.segment_budget``, ``memory.reply_id``, and the
tables ``memory_reply`` (the model's whole reply per segment, with the
instruction and system-prompt hashes; texts go in ``memory_prompt``),
``reconsolidation_copy`` (the marker) and ``reconsolidation_plan`` (the
stretches, fixed at the first start). Each segment commits in one transaction,
so a rerun skips the segments that have a ``memory_reply`` row.

New entries are embedded as the daemon embeds them, so retrieval on the copy
ranks old and new records alike. With a ``memory_vec`` table and an embedding
profile, ``embedding_meta`` must match the profile (else the run refuses before
writing); an embedding call that fails or returns the wrong width leaves that
segment's entries ``embedded = 0``, with a warning, for the retention job's
backfill.

shortcut: no valence, no near-dup merge, no halving of a rejected embedding
batch, no widening of a scoped stretch over a record that reaches past it, and
a record straddling two segments is credited to the first only. Upgrade when a
measurement needs one.
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
from corvidae.prompts import resolve_prompt
from corvidae.thinking import apply_reasoning_history
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

# A "corvidae.*" name, so the daemon's logging configuration reaches it
# (tests/cognition/test_logger_names.py).
logger = logging.getLogger("corvidae.reconsolidate")

#: Owner ruling (#932): a rebuilt-view call is ``standalone`` by PRD Memory 2.
RECONSOLIDATE_TRIGGER = "reconsolidate"
RECONSOLIDATE_MODE = "standalone"
#: The record modes the supersede rule may demote.
_CONSOLIDATION_MODES = ("fork", "standalone")

#: Marks a file as a copy this command made; a live sessions.db has none.
COPY_MARKER_DDL = (
    "CREATE TABLE IF NOT EXISTS reconsolidation_copy "
    "(source TEXT NOT NULL, created_at REAL NOT NULL)"
)
#: The stretches chosen per channel; a (-1, -1) row means "nothing to do".
PLAN_DDL = (
    "CREATE TABLE IF NOT EXISTS reconsolidation_plan (channel_id TEXT NOT NULL, "
    "after_id INTEGER NOT NULL, through_id INTEGER NOT NULL, PRIMARY KEY (channel_id, after_id))"
)
#: One row per finished segment; ``reply`` is NULL when it held no dialog.
REPLY_DDL = (
    "CREATE TABLE IF NOT EXISTS memory_reply (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "channel_id TEXT NOT NULL, msg_id_start INTEGER NOT NULL, msg_id_end INTEGER NOT NULL, "
    "segment_budget INTEGER NOT NULL, prompt_hash TEXT NOT NULL, "
    "system_prompt_hash TEXT NOT NULL, model TEXT, reply TEXT, created_at REAL NOT NULL, "
    "UNIQUE (channel_id, msg_id_start, msg_id_end))"
)

# The client keys LLMClient takes, derived so a constructor rename cannot
# silently drop a profile key.
_LLM_CLIENT_KEYS = frozenset(inspect.signature(LLMClient.__init__).parameters) - {"self"}


@dataclass(frozen=True)
class RunSettings:
    """What one run does."""

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
        """Whether any flag narrows the run below whole channels."""
        return not (self.since is None and self.until is None and self.min_source_tokens is None)


@dataclass
class Clients:
    """The model clients a run uses."""

    consolidation: object
    rubric: object
    #: Embeds new entries; None leaves them ``embedded = 0``.
    embedding: object | None = None
    #: (encoder, dimensions, document_prefix, query_prefix) the profile embeds
    #: with, compared with ``embedding_meta`` before any vector is written.
    embedding_identity: tuple | None = None


@dataclass
class RunReport:
    """What a run did (or, with ``dry_run``, would do)."""

    segments_total: int = 0
    segments_done: int = 0
    segments_skipped: int = 0
    records_written: int = 0
    records_superseded: int = 0
    estimated_tokens: int = 0
    #: (channel_id, first id, last id) of segments left undone.
    failed: list[tuple[str, int, int]] = field(default_factory=list)
    #: Records the supersede rule takes that the scope flags did not select.
    pulled: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The copy
# ---------------------------------------------------------------------------


def _backup(source: Path, dest: Path) -> None:
    """Copy ``source`` into a new ``dest`` with sqlite's online-backup API
    (consistent while the daemon writes), then mark it as a copy."""
    src = sqlite3.connect(source)
    try:
        dst = sqlite3.connect(dest)
        try:
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
        click.ClickException: ``source`` is missing, ``dest`` exists (never
            overwritten), or the backup fails (nothing is left at ``dest``).
    """
    if not source.is_file():
        raise click.ClickException(
            f"Cannot snapshot {source}: database not found — check --snapshot-from"
        )
    if dest.exists():
        raise click.ClickException(
            f"Cannot snapshot into {dest}: it already exists — choose a new --db "
            "path (one copy per variant), or drop --snapshot-from to resume it"
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
    """Refuse the daemon's configured database and any file without the
    marker a snapshot adds.

    Raises:
        click.ClickException: ``path`` is missing, live, or not marked.
    """
    if not path.is_file():
        raise click.ClickException(f"Database not found: {path} — check --db")
    if live_path is not None and live_path.exists() and os.path.samefile(path, live_path):
        raise click.ClickException(
            f"{path} is the daemon's live database (daemon.session_db); refusing to "
            "rewrite it — pass --snapshot-from with it and a new --db path"
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
            f"{path} is not a reconsolidation copy (no reconsolidation_copy marker); "
            "make one with --snapshot-from <the daemon's sessions.db> and a new --db path"
        )


async def _table_exists(db: aiosqlite.Connection, name: str) -> bool:
    """Whether ``name`` is a table (or virtual table)."""
    async with db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ) as cursor:
        return await cursor.fetchone() is not None


async def _prepare(db: aiosqlite.Connection, dry_run: bool) -> bool:
    """Check the copy has memory, load sqlite-vec if it has a vec table, and
    (unless dry-run) add this command's schema. Returns whether vec is in play.

    Raises:
        click.ClickException: no memory table, or a vec table whose extension
            will not load (superseding must drop vector rows).
    """
    if not await _table_exists(db, "memory"):
        raise click.ClickException(
            "The database has no memory table, so there are no records to supersede; "
            "run the daemon with the memory plugin enabled, then snapshot it again"
        )
    vec = await _table_exists(db, "memory_vec")
    if vec:
        try:
            import sqlite_vec

            await db.enable_load_extension(True)
            await db.load_extension(sqlite_vec.loadable_path())
            await db.enable_load_extension(False)
        except (ImportError, AttributeError, sqlite3.Error) as exc:
            logger.warning("cannot load sqlite-vec for memory_vec", exc_info=True)
            raise click.ClickException(
                "This database has memory_vec, which needs the sqlite-vec extension, and "
                f"it could not be loaded ({type(exc).__name__}). Nothing was changed. "
                "Install the cognition extra (uv sync --extra cognition)."
            ) from exc
    if not dry_run:
        await db.execute(PLAN_DDL)
        await db.execute(REPLY_DDL)
        await db.execute(
            "CREATE TABLE IF NOT EXISTS memory_prompt (hash TEXT PRIMARY KEY, text TEXT NOT NULL)"
        )
        # Nullable columns: old rows read NULL and the daemon ignores them.
        await add_column_if_missing(db, "memory", "segment_budget", "INTEGER")
        await add_column_if_missing(db, "memory", "reply_id", "INTEGER")
        await db.commit()
    return vec


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _intersects(start: int, end: int, after_id: int, through_id: int) -> bool:
    """Whether the record range [start, end] meets the stretch (after_id, through_id]."""
    return start <= through_id and end > after_id


async def plan_channel(
    db: aiosqlite.Connection, channel_id: str, settings: RunSettings
) -> tuple[list[tuple[int, int]], int, list[int]]:
    """The stretches of ``channel_id`` to redo, how many records the flags
    selected, and the ids of other active records the supersede rule will
    take because their range meets a stretch.

    Unscoped: the whole channel from id 0. Scoped: the ranges of the active
    consolidation records inside the time window and over the size floor.
    """
    async with db.execute(
        "SELECT id, msg_id_start, msg_id_end, created_at FROM memory "
        "WHERE channel_id = ? AND mode IN (?, ?) AND superseded_by IS NULL "
        "AND redacted = 0 AND merged_foreign = 0 ORDER BY id",
        (channel_id, *_CONSOLIDATION_MODES),
    ) as cursor:
        active = await cursor.fetchall()

    # Unscoped: everything the channel has.
    if not settings.scoped:
        async with db.execute(
            "SELECT MAX(id) FROM message_log WHERE channel_id = ?", (channel_id,)
        ) as cursor:
            top = (await cursor.fetchone())[0]
        return ([(0, top)] if top is not None else []), len(active), []

    # Scoped: pick the records the flags name.
    selected = []
    for record in active:
        _id, start, end, created_at = record
        if settings.since is not None and created_at < settings.since:
            continue
        if settings.until is not None and created_at > settings.until:
            continue
        if settings.min_source_tokens is not None:
            # Dialog tokens in the record's range, as the segmenter measures a message.
            source = await fetch_range_rows(db, channel_id, start - 1, end)
            tokens = sum(message_tokens(m) for m in dialog_from_rows(source, channel_id))
            if tokens < settings.min_source_tokens:
                continue
        selected.append(record)

    # Join overlapping ranges into stretches.
    intervals: list[tuple[int, int]] = []
    for after_id, through_id in sorted((r[1] - 1, r[2]) for r in selected):
        if intervals and after_id < intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], through_id))
        else:
            intervals.append((after_id, through_id))
    chosen = {r[0] for r in selected}
    pulled = [
        r[0] for r in active
        if r[0] not in chosen and any(_intersects(r[1], r[2], a, t) for a, t in intervals)
    ]
    return intervals, len(selected), pulled


# ---------------------------------------------------------------------------
# One segment
# ---------------------------------------------------------------------------


async def _vector_width(db: aiosqlite.Connection, clients: Clients, vec: bool) -> int | None:
    """The width new vectors are written at, or None when none are.

    Raises:
        click.ClickException: the copy's ``embedding_meta`` names another
            encoder than the configured profile (mixing encoders corrupts
            similarity), before anything is written.
    """
    if not vec or clients.embedding is None or clients.embedding_identity is None:
        return None
    async with db.execute(
        "SELECT encoder, dimensions, document_prefix, query_prefix FROM embedding_meta LIMIT 1"
    ) as cursor:
        row = await cursor.fetchone()
    stored = (row[0], row[1], row[2] or "", row[3] or "") if row else None
    if stored != clients.embedding_identity:
        raise click.ClickException(
            f"embedding_meta records {stored} but the embedding profile in agent.yaml is "
            f"{clients.embedding_identity}; mixing encoders corrupts similarity — point "
            "--config at the profile that built this database"
        )
    return clients.embedding_identity[1]


async def _embed(
    clients: Clients, texts: list[str], where: str, width: int | None
) -> list[list[float] | None]:
    """Embed ``texts`` as documents; a failure or wrong width gives None for
    those entries (stored ``embedded = 0`` for the retention backfill)."""
    if width is None:
        return [None] * len(texts)
    try:
        vectors = await clients.embedding.embed(texts, kind="document")
        if len(vectors) != len(texts):
            raise ValueError(f"{len(vectors)} vectors for {len(texts)} texts")
    except Exception:
        logger.warning("reconsolidate: embedding failed for %s (%d entries); storing them "
                       "embedded=0 for the retention backfill", where, len(texts), exc_info=True)
        return [None] * len(texts)
    good = [v if len(v) == width else None for v in vectors]
    if None in good:
        logger.warning("reconsolidate: embedding for %s is not %d wide; those entries are "
                       "stored embedded=0 for the retention backfill", where, width)
    return good


async def _consolidate(client, stamp: RecordStamp, channel_id: str, segment):
    """The segment's reply text and its parse, or None after two unusable
    replies. Transport errors propagate (finished segments are kept)."""
    span = (segment.rows[0][0], segment.end_id)
    for attempt in range(2):
        response = await client.chat(segment.messages, extra_body=CONSOLIDATION_EXTRA_BODY)
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
    span: tuple[int, int], stamp: RecordStamp, reply: str | None,
    entries: list, participants: list[str], importance: float, created_at: float,
    vec: bool, vectors: list | None = None,
) -> tuple[int, int]:
    """Write one segment in one transaction; returns (records, superseded).

    ``entries`` empty writes only the reply row (a segment with no dialog), so
    a rerun skips it.
    """
    start, end = span
    try:
        # The two prompt texts, once per hash (PRD Memory 7).
        for text in (settings.instruction, settings.system_prompt):
            await db.execute(
                "INSERT OR IGNORE INTO memory_prompt (hash, text) VALUES (?, ?)",
                (prompt_hash(text), text),
            )
        cursor = await db.execute(
            "INSERT INTO memory_reply (channel_id, msg_id_start, msg_id_end, segment_budget, "
            "prompt_hash, system_prompt_hash, model, reply, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (channel_id, start, end, settings.segment_budget, prompt_hash(settings.instruction),
             prompt_hash(settings.system_prompt), stamp.model, reply, time.time()),
        )
        reply_id = cursor.lastrowid
        if not entries:
            await db.commit()
            return 0, 0

        # The records this segment replaces, and their folded statistics: the
        # sum, the latest access, the larger importance (a near-dup merge's rules).
        async with db.execute(
            "SELECT id, retrieval_count, last_retrieved_at, importance FROM memory "
            "WHERE channel_id = ? AND mode IN (?, ?) AND superseded_by IS NULL "
            "AND redacted = 0 AND merged_foreign = 0 AND msg_id_start <= ? AND msg_id_end >= ?",
            (channel_id, *_CONSOLIDATION_MODES, end, start),
        ) as cursor:
            old = await cursor.fetchall()
        count = sum(r[1] or 0 for r in old)
        accessed = [r[2] for r in old if r[2] is not None]
        last = max(accessed) if accessed else None
        importance = max([importance] + [r[3] for r in old if r[3] is not None])

        # The new records, sharing the segment's range (PRD Memory 2).
        new_ids: list[int] = []
        for entry, vector in zip(entries, vectors or [None] * len(entries)):
            cursor = await db.execute(
                "INSERT INTO memory (channel_id, created_at, summary, importance, topic_tags, "
                "participants, msg_id_start, msg_id_end, retrieval_count, last_retrieved_at, "
                "embedded, prompt_hash, mode, model, trigger, segment_budget, reply_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (channel_id, created_at, entry.narrative, importance, json.dumps(list(entry.topic_tags)),
                 json.dumps(participants), start, end, count, last, 1 if vector is not None else 0, stamp.hash, stamp.mode,
                 stamp.model, stamp.trigger, settings.segment_budget, reply_id),
            )
            new_ids.append(cursor.lastrowid)
            if vector is not None:
                import sqlite_vec

                await db.execute(
                    "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, sqlite_vec.serialize_float32(vector)),
                )

        # Demote the old records and drop their vector rows.
        for old_id, *_rest in old:
            await db.execute(
                "UPDATE memory SET superseded_by = ?, indexed = 0, embedded = 0 WHERE id = ?",
                (new_ids[0], old_id),
            )
            if vec:
                await db.execute("DELETE FROM memory_vec WHERE memory_id = ?", (old_id,))
        await db.commit()
        return len(new_ids), len(old)
    except BaseException:
        # Nothing of the segment may land half-written.
        await db.rollback()
        raise


async def _run_segment(
    db: aiosqlite.Connection, settings: RunSettings, clients: Clients, channel_id: str,
    segment, vec: bool, width: int | None, report: RunReport,
) -> None:
    """Consolidate one segment and commit it, or record that it was left."""
    span = (segment.rows[0][0], segment.end_id)
    dialog = dialog_from_rows(segment.rows, channel_id)
    stamp = RecordStamp(
        settings.instruction, RECONSOLIDATE_MODE,
        clients.consolidation.request_model(CONSOLIDATION_EXTRA_BODY), RECONSOLIDATE_TRIGGER,
    )
    common = dict(settings=settings, channel_id=channel_id, span=span, stamp=stamp, vec=vec)

    # No dialog: nothing to remember, but the segment is finished.
    if not dialog:
        await _commit_segment(
            db, reply=None, entries=[], participants=[], importance=0.0, created_at=0.0, **common
        )
        report.segments_done += 1
        return

    result = await _consolidate(clients.consolidation, stamp, channel_id, segment)
    if result is None:
        # Left undone: the old records stay, and a rerun retries it.
        report.failed.append((channel_id, *span))
        return
    reply, consolidation = result
    # Importance: the reply's own rating, else the rubric (as the daemon does).
    importance = await RubricPrior(
        lambda: clients.rubric, lambda: settings.rubric_prompt
    ).score(dialog, msg_id_range=span, hint=consolidation.importance)
    vectors = await _embed(
        clients, [e.narrative for e in consolidation.entries],
        f"{channel_id} {span[0]}-{span[1]}", width,
    )
    written, superseded = await _commit_segment(
        db, reply=reply, vectors=vectors, entries=list(consolidation.entries),
        participants=list(consolidation.participants), importance=importance,
        created_at=await source_timestamp(db, span[1]), **common,
    )
    report.segments_done += 1
    report.records_written += written
    report.records_superseded += superseded


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


async def run_reconsolidation(
    db: aiosqlite.Connection, *, settings: RunSettings,
    inputs_for: Callable[[str], ViewInputs], clients: Clients,
    dry_run: bool = False, echo: Callable[[str], None] = click.echo,
) -> RunReport:
    """Re-consolidate the selected stretches of each channel on the copy ``db``.

    With ``dry_run`` nothing is written and no model is called. Otherwise each
    segment commits on its own: a transport error propagates after finished
    segments are kept, and an unusable reply leaves its segment in
    ``report.failed``.

    Raises:
        click.ClickException: no memory table, a vec table without its
            extension, or an unknown ``--channel``.
    """
    report = RunReport()
    vec = await _prepare(db, dry_run)
    width = None if dry_run else await _vector_width(db, clients, vec)

    # Which channels: the flags, else every channel with history.
    async with db.execute("SELECT DISTINCT channel_id FROM message_log ORDER BY channel_id") as cur:
        present = [row[0] for row in await cur.fetchall()]
    channels = present if settings.channels is None else list(settings.channels)
    for channel_id in channels:
        if channel_id not in present:
            raise click.ClickException(
                f"Channel {channel_id!r} has no message_log rows in the database; channels "
                f"present: {', '.join(present) or '(none)'} — check --channel"
            )

    for channel_id in channels:
        # The stretches are fixed at the first start; reruns reuse them.
        stored = None
        if await _table_exists(db, "reconsolidation_plan"):
            async with db.execute(
                "SELECT after_id, through_id FROM reconsolidation_plan WHERE channel_id = ? "
                "ORDER BY after_id", (channel_id,),
            ) as cursor:
                found = await cursor.fetchall()
            # Rows exist once planned; (-1, -1) alone means nothing to do.
            stored = [(a, t) for a, t in found if a >= 0] if found else None
        if stored is None:
            intervals, selected, pulled = await plan_channel(db, channel_id, settings)
            report.pulled.extend(pulled)
            echo(
                f"{channel_id}: {selected} record(s) selected, {len(pulled)} more met by range"
                + (": " + ", ".join(f"memory {i}" for i in pulled) if pulled else "")
            )
            if not dry_run:
                await db.executemany(
                    "INSERT INTO reconsolidation_plan (channel_id, after_id, through_id) "
                    "VALUES (?, ?, ?)",
                    [(channel_id, a, t) for a, t in intervals] or [(channel_id, -1, -1)],
                )
                await db.commit()
        else:
            intervals = stored

        # The segment budget replaces the channel's max_context_tokens.
        inputs = dataclasses.replace(inputs_for(channel_id), max_tokens=settings.segment_budget)
        done: set[tuple[int, int]] = set()
        if await _table_exists(db, "memory_reply"):
            async with db.execute(
                "SELECT msg_id_start, msg_id_end FROM memory_reply WHERE channel_id = ?",
                (channel_id,),
            ) as cursor:
                done = set(await cursor.fetchall())
        for after_id, through_id in intervals:
            for segment in await build_catchup_segments(
                db, channel_id, after_id, through_id, inputs, settings.instruction
            ):
                report.segments_total += 1
                span = (segment.rows[0][0], segment.end_id)
                if span in done:
                    report.segments_skipped += 1
                elif dry_run:
                    size = _measured_size(segment.messages)
                    report.estimated_tokens += size
                    echo(f"  {channel_id} message ids {span[0]}-{span[1]}: "
                         f"{len(segment.rows)} row(s), ~{size} tokens")
                else:
                    await _run_segment(db, settings, clients, channel_id, segment, vec, width, report)

    echo(
        f"{'would send' if dry_run else 'done'}: {report.segments_total} segment(s), "
        f"{report.segments_skipped} already done, {len(report.failed)} left undone, "
        f"~{report.estimated_tokens} tokens estimated, {report.records_written} record(s) "
        f"written, {report.records_superseded} superseded"
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_time(flag: str, value: str | None) -> float | None:
    """A ``--since`` / ``--until`` value as epoch seconds (epoch or ISO-8601,
    UTC when no zone). Raises ClickException naming the flag and value."""
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
            f"{flag}: cannot read {value!r} as a time; use an ISO-8601 date or datetime "
            "(2026-11-01 or 2026-11-01T12:00, UTC unless it names a zone) or epoch seconds"
        ) from exc
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()


def _configured_live_db(config_path: Path) -> Path | None:
    """The daemon's ``daemon.session_db``, or None when the config cannot say
    (the marker check still refuses an unmarked file)."""
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("cannot read %s for daemon.session_db; relying on the copy marker",
                       config_path, exc_info=True)
        return None
    return config_path.parent / (config.get("daemon") or {}).get("session_db", "sessions.db")


def _build_client(llm_config, stage: str, kind: str = "chat") -> tuple[LLMClient, dict]:
    """The offline client for ``stage``'s profile, as the daemon resolves it,
    and the profile's settings."""
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
    them (exact ``channels:`` entry, transport-wide entries, ``agent:``)."""
    agent_cfg = config.get("agent") or {}
    registry = ChannelRegistry(agent_defaults=agent_cfg)
    try:
        apply_channel_config(parse_channel_config(config), registry)
    except ValueError as exc:
        raise click.ClickException(f"{exc} — check the channels: section of agent.yaml") from exc
    mid_role = agent_cfg.get("mid_conversation_system_role", "system")
    if mid_role not in ("system", "user"):
        logger.warning("agent.mid_conversation_system_role %r is not 'system' or 'user'; "
                       "using 'system'", mid_role)
        mid_role = "system"
    # No daemon key file offline; prompts here are not compared across runs.
    fence_key = os.urandom(32)

    def resolve(channel_id: str) -> ViewInputs:
        transport, _, scope = channel_id.partition(":")
        resolved = registry.resolve_config(registry.get_or_create(transport, scope))
        policy = resolved["reasoning_history"]
        return ViewInputs(
            system_prompt, resolved["max_context_tokens"], mid_role,
            lambda messages: apply_reasoning_history(messages, policy), fence_key,
        )

    return resolve


@click.command("reconsolidate")
@click.option("--db", "db_path", required=True, type=click.Path(path_type=Path),
              help="The copy to rewrite (never the daemon's live sessions.db).")
@click.option("--snapshot-from", type=click.Path(path_type=Path), default=None,
              help="Snapshot this sessions.db into a new --db first and mark it a copy.")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default="agent.yaml",
              show_default=True, help="agent.yaml: llm profiles, channels, memory settings.")
@click.option("--system-prompt", "system_prompt_file", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="The identity + register that heads every view (recorded per segment).")
@click.option("--prompt", "prompt_file", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="The consolidation instruction; default memory.consolidation_fork_prompt.")
@click.option("--segment-budget", type=click.IntRange(min=1), required=True,
              help="Tokens per segment, independent of max_context_tokens; stamped on each record.")
@click.option("--channel", "channels", multiple=True,
              help="Only this channel id (repeatable); default every channel.")
@click.option("--since", default=None, help="Only records created at or after this (ISO-8601 UTC or epoch).")
@click.option("--until", default=None, help="Only records created at or before this (ISO-8601 UTC or epoch).")
@click.option("--min-source-tokens", type=click.IntRange(min=1), default=None,
              help="Only records whose source range holds at least this many tokens.")
@click.option("--dry-run", is_flag=True, help="Report segments, estimated tokens and records met; write nothing.")
def reconsolidate_command(
    db_path: Path, snapshot_from: Path | None, config_path: Path, system_prompt_file: Path,
    prompt_file: Path | None, segment_budget: int, channels: tuple[str, ...],
    since: str | None, until: str | None, min_source_tokens: int | None, dry_run: bool,
) -> None:
    """Re-consolidate a channel's history into a COPY of sessions.db.

    Operator-only and offline. New records supersede (never delete) the
    consolidation records whose message ranges they cover. Rerun the same
    command to resume.
    """
    # Cheap flag checks first, so a typo fails before any file is touched.
    since_ts = _parse_time("--since", since)
    until_ts = _parse_time("--until", until)

    async def _run() -> None:
        if snapshot_from is not None:
            await snapshot_database(snapshot_from, db_path)
        check_is_copy(db_path, live_path=_configured_live_db(config_path))

        # The run's inputs, from the operator's files and config.
        try:
            config = yaml.safe_load(config_path.read_text()) or {}
            system_prompt = system_prompt_file.read_text()
            base = prompt_file.read_text() if prompt_file else resolve_consolidation_fork_prompt(config)
            instruction = consolidation_instruction(base, resolve_entry_target_tokens(config))
            rubric_prompt = resolve_prompt(config, "memory.rubric_prompt", RUBRIC_PROMPT)
            llm_config = parse_llm_config(config.get("llm"))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise click.ClickException(
                f"Cannot build the run's inputs ({type(exc).__name__}: {exc}) — check "
                f"--system-prompt, --prompt and {config_path}"
            ) from exc
        consolidation, _ = _build_client(llm_config, "consolidation")
        rubric, _ = _build_client(llm_config, "rubric")
        # An embedding profile is optional; without one new entries stay unembedded.
        embedding = identity = None
        if llm_config.profile_for("embedding", "embedding") is not None:
            embedding, cfg = _build_client(llm_config, "embedding", "embedding")
            identity = (cfg.get("model"), cfg.get("dimensions"),
                        cfg.get("document_prefix", "") or "", cfg.get("query_prefix", "") or "")
        settings = RunSettings(
            segment_budget, instruction, system_prompt, tuple(channels) or None,
            since_ts, until_ts, min_source_tokens, rubric_prompt,
        )
        inputs_for = _inputs_resolver(config, system_prompt)

        clients = [consolidation, rubric] + ([embedding] if embedding is not None else [])
        for client in clients:
            await client.start()
        try:
            async with aiosqlite.connect(db_path) as db:
                await db.execute("PRAGMA busy_timeout = 5000")
                try:
                    report = await run_reconsolidation(
                        db, settings=settings, inputs_for=inputs_for,
                        clients=Clients(consolidation, rubric, embedding, identity), dry_run=dry_run,
                    )
                except click.ClickException:
                    raise
                except Exception as exc:
                    # A transport failure: finished segments are kept.
                    logger.warning("reconsolidate stopped on %s", db_path, exc_info=True)
                    raise click.ClickException(
                        f"Stopped on {type(exc).__name__}: {exc}. Segments already committed "
                        f"in {db_path} are kept; if the model server is unreachable, check "
                        "llm.providers.<name>.base_url, then rerun the same command to resume"
                    ) from exc
        finally:
            for client in clients:
                await client.stop()
        if report.failed:
            spans = ", ".join(f"{c} {a}-{b}" for c, a, b in report.failed)
            raise click.ClickException(
                f"{len(report.failed)} segment(s) got no usable reply and were left undone "
                f"({spans}); their old records are untouched — rerun the same command to retry"
            )

    asyncio.run(_run())
