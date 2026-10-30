"""Operator-only redaction CLI for corvidae (WP1b.4).

Second-process discipline (bootstrap-mapping §4.12):
  - Own connection, PRAGMA busy_timeout = 5000
  - Short transactions
  - WAL mode asserted (abort with a clear message if not WAL)
  - Schema-presence probe: never creates schema, skips absent surfaces
    with printed notices (schema is daemon-owned; a partial CLI-created
    schema would diverge from the daemon's embedding_meta/vec setup)

Forms:
    corvidae redact --db sessions.db message <id> [<id2>...]
    corvidae redact --db sessions.db memory <memory_id>
    corvidae redact --db sessions.db range <start_id> <end_id>

exchange_log (owned by the cognition extra's outcome_log) copies conversation
text into two columns, and is scrubbed too when present (#160): the
redacted message's owning exchange loses its stage-2 past_claims text
(counts kept), and every exchange in the channel from that message on loses
its provenance_snapshot.

The message_fts update trigger (message_log_au, plain DELETE — not the FTS5
'delete' command) propagates tombstones to message_fts automatically when the
trigger is present. The memory_au trigger propagates tombstones to memory_fts
when the memory row's summary is updated via UPDATE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import click

logger = logging.getLogger("corvidae.redact")

_TOMBSTONE_TEXT = "[redacted by operator {date}]"
_MEMORY_TOMBSTONE = "[redacted by operator]"


def _iso_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _tombstone_content(iso_date: str) -> str:
    return _TOMBSTONE_TEXT.format(date=iso_date)


def _make_tombstone_message(msg_json: str, iso_date: str) -> str:
    """Replace content with tombstone; drop tool_calls and payload fields; keep role."""
    try:
        msg = json.loads(msg_json)
    except Exception:
        return json.dumps({"role": "unknown", "content": _tombstone_content(iso_date)})
    result: dict = {"role": msg.get("role", "unknown")}
    result["content"] = _tombstone_content(iso_date)
    # tool_calls and other payload fields are intentionally omitted
    return json.dumps(result)


def _extract_sample_token(text: str) -> str | None:
    """Extract the first non-trivial word from text for FTS verification.

    If text looks like JSON (message_log content), uses the content field.
    Returns None if no suitable token is found.
    """
    content = text
    try:
        msg = json.loads(text)
        content = msg.get("content", "") or ""
    except (json.JSONDecodeError, AttributeError) as exc:
        # Plain-text memory summaries are expected; log at DEBUG so a genuine
        # parse failure is visible when diagnosing redaction, then continue.
        logger.debug("sample token: message is not JSON, using raw text: %s", exc)
    # Find words of 4+ alpha characters; skip tombstone text
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", content)
             if w.lower() not in {"redacted", "operator", "that", "this", "with"}]
    return words[0] if words else None


async def _verify_fts_clean(
    db: aiosqlite.Connection, sample_token: str
) -> str:
    """Run FTS MATCH for sample_token against present FTS surfaces.

    Returns a single-line verification message suitable for click.echo.
    """
    tables = await _probe_tables(db)
    checked = []
    hits_found = []
    for table in ("message_fts", "memory_fts"):
        if table not in tables:
            continue
        try:
            async with db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?",
                (f'"{sample_token}"',),
            ) as c:
                count = (await c.fetchone())[0]
            checked.append(table)
            if count > 0:
                hits_found.append(f"{table}: {count} hits")
        except Exception:
            checked.append(f"{table}(error)")
    if not checked:
        return "verified: no FTS surfaces present"
    if hits_found:
        return "WARNING: FTS still contains hits for '{}': {}".format(
            sample_token, "; ".join(hits_found)
        )
    return "verified: 0 hits for '{}' in {}".format(
        sample_token, ", ".join(checked)
    )


async def _check_wal(db: aiosqlite.Connection) -> None:
    """Assert WAL journal mode; raise RuntimeError with WAL/journal mention if not WAL."""
    async with db.execute("PRAGMA journal_mode") as cursor:
        row = await cursor.fetchone()
    mode = (row[0] if row else "").lower()
    if mode not in ("wal", "memory"):
        raise RuntimeError(
            f"Database journal mode is '{mode}' — the redact CLI requires WAL mode. "
            "The corvidae daemon configures WAL on startup. "
            "Run: PRAGMA journal_mode = WAL; to convert the database."
        )


async def _probe_tables(db: aiosqlite.Connection) -> set[str]:
    """Return names present in sqlite_master (tables and triggers)."""
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger', 'shadow', 'view')"
    ) as cursor:
        rows = await cursor.fetchall()
    return {row[0] for row in rows}


def _scrub_past_claims(appraisal_json: str | None, tombstone: str) -> str | None:
    """Return appraisal JSON with stage2.past_claims text tombstoned, or None
    when there is nothing to change.

    Each claim and each of its queries becomes ``tombstone``, so the number
    of claims and queries survives for calibration while the words do not.
    No appraisal, no stage 2, or claims already tombstoned → None.
    """
    # NULL is the common case: most exchanges never get a stage-2 appraisal.
    # The column is written by outcome_log's JSON merge, so it parses, and
    # appraisal normalizes past_claims to a list of {claim, queries}.
    if appraisal_json is None:
        return None
    appraisal = json.loads(appraisal_json)
    stage2 = appraisal.get("stage2") or {}
    claims = stage2.get("past_claims")
    if not claims:
        return None
    scrubbed = [
        {"claim": tombstone, "queries": [tombstone] * len(claim["queries"])}
        for claim in claims
    ]
    # Already scrubbed by an earlier redaction: report no change.
    if scrubbed == claims:
        return None
    stage2["past_claims"] = scrubbed
    return json.dumps(appraisal)


async def _scrub_exchange_log(
    db: aiosqlite.Connection,
    channel_id: str,
    msg_ids: list[int],
    dry_run: bool,
    iso_date: str,
) -> set[str]:
    """Scrub exchange_log text derived from redacted messages in one channel.

    For each redacted id X: the exchange that owns X (in the channel, the
    largest message_rowid <= X) gets its stage-2 past_claims tombstoned.
    Every exchange in the channel with message_rowid >= the smallest X loses
    its provenance_snapshot, since any later window may have carried X's
    text as CONTEXT. Exchanges with a NULL message_rowid are not reached.

    Returns the trace ids changed (or that would be, in dry-run mode). The
    caller checks that the exchange_log table exists.
    """
    changed: set[str] = set()
    if not msg_ids:
        return changed
    tombstone = _tombstone_content(iso_date)

    # Find each redacted id's owning exchange, once per exchange.
    owners: set[str] = set()
    for msg_id in msg_ids:
        async with db.execute(
            "SELECT trace_id FROM exchange_log "
            "WHERE channel_id = ? AND message_rowid IS NOT NULL "
            "AND message_rowid <= ? ORDER BY message_rowid DESC LIMIT 1",
            (channel_id, msg_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            owners.add(row[0])

    # Tombstone the owners' past_claims text.
    for trace_id in owners:
        async with db.execute(
            "SELECT appraisal FROM exchange_log WHERE trace_id = ?", (trace_id,)
        ) as cursor:
            row = await cursor.fetchone()
        scrubbed = _scrub_past_claims(row[0], tombstone)
        if scrubbed is None:
            continue
        changed.add(trace_id)
        if not dry_run:
            await db.execute(
                "UPDATE exchange_log SET appraisal = ? WHERE trace_id = ?",
                (scrubbed, trace_id),
            )

    # Drop every snapshot in the channel from the first redacted id on.
    snapshot_where = (
        "WHERE channel_id = ? AND message_rowid >= ? "
        "AND provenance_snapshot IS NOT NULL"
    )
    snapshot_args = (channel_id, min(msg_ids))
    async with db.execute(
        f"SELECT trace_id FROM exchange_log {snapshot_where}", snapshot_args
    ) as cursor:
        changed.update(row[0] for row in await cursor.fetchall())
    if not dry_run:
        await db.execute(
            f"UPDATE exchange_log SET provenance_snapshot = NULL {snapshot_where}",
            snapshot_args,
        )
        await db.commit()
    return changed


async def redact_messages(
    db: aiosqlite.Connection,
    message_ids: list[int],
    dry_run: bool = False,
    notices: list | None = None,
) -> dict:
    """Tombstone message_log rows by id.

    Step 0: probe schema; skip absent surfaces with notices.
    Step 1: tombstone message rows (message_fts updated by trigger if present).
    Step 2: cascade to same-channel intersecting memory records.
    Step 3: scrub exchange_log text derived from the redacted rows.
    The caller prints verification; this function returns stats.

    Returns:
        ``affected_messages`` is the number of supplied ids that actually
        existed and were tombstoned (in dry-run mode, would have been),
        ``affected_memories`` is the cascade count, ``affected_exchanges``
        the number of exchange_log rows scrubbed, and ``requested_messages``
        is the number of ids supplied. The CLI prints ``found of requested``
        so a mistyped id is visible instead of reporting a confident hit.

    WAL mode is required and asserted.
    """
    await _check_wal(db)

    tables = await _probe_tables(db)
    has_memory = "memory" in tables
    has_memory_vec = "memory_vec" in tables
    has_message_fts = "message_fts" in tables
    has_exchange_log = "exchange_log" in tables

    if notices is not None:
        if not has_message_fts:
            notices.append(
                "SKIP: message_fts table not present (pre-1b DB); "
                "the daemon's backfill will index the tombstone text when it starts"
            )
        if not has_memory:
            notices.append("SKIP: memory table not present (pre-1b DB); memory cascade skipped")
        if not has_exchange_log:
            notices.append(
                "SKIP: exchange_log table not present (no cognition extra); "
                "exchange_log scrub skipped"
            )

    iso_date = _iso_date()

    if dry_run:
        # Count without writing, counting only ids that actually exist.
        affected_messages = 0
        affected_memories = 0
        dry_by_channel: dict[str, list[int]] = {}
        for msg_id in message_ids:
            async with db.execute(
                "SELECT channel_id FROM message_log WHERE id = ?", (msg_id,)
            ) as cursor:
                ch_row = await cursor.fetchone()
            if ch_row is None:
                continue
            affected_messages += 1
            dry_by_channel.setdefault(ch_row[0], []).append(msg_id)
            if not has_memory:
                continue
            ch = ch_row[0]
            async with db.execute(
                "SELECT COUNT(*) FROM memory "
                "WHERE channel_id = ? AND msg_id_start <= ? AND msg_id_end >= ? "
                "AND redacted = 0",
                (ch, msg_id, msg_id),
            ) as cursor:
                cnt_row = await cursor.fetchone()
            affected_memories += cnt_row[0] if cnt_row else 0
        # Count the exchange_log rows the scrub would change.
        dry_exchanges: set[str] = set()
        if has_exchange_log:
            for channel_id, ids in dry_by_channel.items():
                dry_exchanges |= await _scrub_exchange_log(
                    db, channel_id, ids, dry_run=True, iso_date=iso_date
                )
        return {
            "affected_messages": affected_messages,
            "affected_memories": affected_memories,
            "affected_exchanges": len(dry_exchanges),
            "requested_messages": len(message_ids),
        }

    # Step 1: tombstone message rows
    # The message_log_au trigger propagates the change to message_fts if present.
    affected_messages = 0
    by_channel: dict[str, list[int]] = {}
    for msg_id in message_ids:
        async with db.execute(
            "SELECT message, channel_id FROM message_log WHERE id = ?", (msg_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            continue
        by_channel.setdefault(row[1], []).append(msg_id)
        tombstone = _make_tombstone_message(row[0], iso_date)
        await db.execute(
            "UPDATE message_log SET message = ? WHERE id = ?", (tombstone, msg_id)
        )
        affected_messages += 1
    await db.commit()

    # Step 2: memory cascade (per-channel intersection)
    # A foreign-channel memory's numeric range can contain a redacted id from a
    # different channel (global AUTOINCREMENT id). Match on channel_id to prevent
    # over-redaction of unrelated channels' memory records.
    affected_memories = 0
    if has_memory:
        for msg_id in message_ids:
            async with db.execute(
                "SELECT channel_id FROM message_log WHERE id = ?", (msg_id,)
            ) as cursor:
                ch_row = await cursor.fetchone()
            if ch_row is None:
                continue
            ch = ch_row[0]

            async with db.execute(
                "SELECT id FROM memory "
                "WHERE channel_id = ? AND msg_id_start <= ? AND msg_id_end >= ? "
                "AND redacted = 0",
                (ch, msg_id, msg_id),
            ) as cursor:
                mem_rows = await cursor.fetchall()

            for (mem_id,) in mem_rows:
                # The memory_au trigger (AFTER UPDATE OF summary ON memory) keeps
                # memory_fts in sync — the tombstone is indexed, the original is not.
                await db.execute(
                    "UPDATE memory SET "
                    "summary = ?, redacted = 1, indexed = 0, embedded = 0 "
                    "WHERE id = ?",
                    (_MEMORY_TOMBSTONE, mem_id),
                )
                if has_memory_vec:
                    await db.execute(
                        "DELETE FROM memory_vec WHERE memory_id = ?", (mem_id,)
                    )
                affected_memories += 1
        await db.commit()

    # Step 3: scrub exchange_log text derived from the redacted messages.
    affected_exchanges: set[str] = set()
    if has_exchange_log:
        for channel_id, ids in by_channel.items():
            affected_exchanges |= await _scrub_exchange_log(
                db, channel_id, ids, dry_run=False, iso_date=iso_date
            )

    return {
        "affected_messages": affected_messages,
        "affected_memories": affected_memories,
        "affected_exchanges": len(affected_exchanges),
        "requested_messages": len(message_ids),
    }


async def redact_memory_id(
    db: aiosqlite.Connection,
    memory_id: int,
    dry_run: bool = False,
) -> dict:
    """Tombstone a memory record and its raw message range.

    Uses WHERE channel_id = record.channel_id AND id BETWEEN ... to prevent
    tombstoning interleaved foreign-channel rows (global AUTOINCREMENT id
    regression — message_log ids are shared across all channels).
    """
    await _check_wal(db)

    tables = await _probe_tables(db)
    has_memory = "memory" in tables
    has_memory_vec = "memory_vec" in tables
    has_exchange_log = "exchange_log" in tables

    if not has_memory:
        raise ValueError("memory table not present in this database")

    async with db.execute(
        "SELECT id, channel_id, msg_id_start, msg_id_end "
        "FROM memory WHERE id = ?",
        (memory_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        raise ValueError(f"memory record {memory_id} not found")

    mid, channel_id, msg_id_start, msg_id_end = row

    iso_date = _iso_date()

    if dry_run:
        async with db.execute(
            "SELECT id FROM message_log "
            "WHERE channel_id = ? AND id BETWEEN ? AND ?",
            (channel_id, msg_id_start, msg_id_end),
        ) as cursor:
            dry_ids = [row[0] for row in await cursor.fetchall()]
        # Count the exchange_log rows the scrub would change.
        dry_exchanges: set[str] = set()
        if has_exchange_log:
            dry_exchanges = await _scrub_exchange_log(
                db, channel_id, dry_ids, dry_run=True, iso_date=iso_date
            )
        return {
            "affected_messages": len(dry_ids),
            "affected_memories": 1,
            "affected_exchanges": len(dry_exchanges),
        }

    # Step 1: tombstone own-channel message rows only.
    # channel_id predicate prevents over-redaction of interleaved foreign-channel rows.
    async with db.execute(
        "SELECT id, message FROM message_log "
        "WHERE channel_id = ? AND id BETWEEN ? AND ? ORDER BY id",
        (channel_id, msg_id_start, msg_id_end),
    ) as cursor:
        msg_rows = await cursor.fetchall()

    for row_id, msg_json in msg_rows:
        tombstone = _make_tombstone_message(msg_json, iso_date)
        await db.execute(
            "UPDATE message_log SET message = ? WHERE id = ?", (tombstone, row_id)
        )
    await db.commit()

    # Step 2: tombstone the memory record itself
    await db.execute(
        "UPDATE memory SET "
        "summary = ?, redacted = 1, indexed = 0, embedded = 0 "
        "WHERE id = ?",
        (_MEMORY_TOMBSTONE, memory_id),
    )
    if has_memory_vec:
        await db.execute(
            "DELETE FROM memory_vec WHERE memory_id = ?", (memory_id,)
        )
    await db.commit()

    # Step 2b: cascade to other memory records intersecting (same channel)
    # The channel_id match prevents over-redaction of foreign-channel memories
    # whose numeric ranges happen to contain the tombstoned message ids.
    affected_memories = 1
    redacted_msg_ids = [row_id for row_id, _ in msg_rows]
    if redacted_msg_ids:
        for msg_id in redacted_msg_ids:
            async with db.execute(
                "SELECT id FROM memory "
                "WHERE channel_id = ? AND msg_id_start <= ? AND msg_id_end >= ? "
                "AND redacted = 0 AND id != ?",
                (channel_id, msg_id, msg_id, memory_id),
            ) as cursor:
                other_mems = await cursor.fetchall()
            for (other_id,) in other_mems:
                await db.execute(
                    "UPDATE memory SET "
                    "summary = ?, redacted = 1, indexed = 0, embedded = 0 "
                    "WHERE id = ?",
                    (_MEMORY_TOMBSTONE, other_id),
                )
                if has_memory_vec:
                    await db.execute(
                        "DELETE FROM memory_vec WHERE memory_id = ?", (other_id,)
                    )
                affected_memories += 1
        await db.commit()

    # Step 3: scrub exchange_log text derived from the redacted range.
    affected_exchanges: set[str] = set()
    if has_exchange_log:
        affected_exchanges = await _scrub_exchange_log(
            db, channel_id, redacted_msg_ids, dry_run=False, iso_date=iso_date
        )

    return {
        "affected_messages": len(msg_rows),
        "affected_memories": affected_memories,
        "affected_exchanges": len(affected_exchanges),
    }


async def redact_range(
    db: aiosqlite.Connection,
    start_id: int,
    end_id: int,
    dry_run: bool = False,
) -> dict:
    """Tombstone all message_log rows in the given id range (inclusive)."""
    await _check_wal(db)

    async with db.execute(
        "SELECT id FROM message_log WHERE id BETWEEN ? AND ? ORDER BY id",
        (start_id, end_id),
    ) as cursor:
        rows = await cursor.fetchall()

    message_ids = [row[0] for row in rows]
    if not message_ids:
        return {"affected_messages": 0, "affected_memories": 0, "affected_exchanges": 0}
    return await redact_messages(db, message_ids, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Click CLI entry point
# ---------------------------------------------------------------------------


@click.group("redact")
@click.option("--db", "db_path", required=True, help="Path to sessions.db")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print affected counts without writing")
@click.pass_context
def redact_command(ctx, db_path, dry_run):
    """Operator redaction tool for corvidae memory stores.

    Tombstones message_log rows, cascades to intersecting memory records,
    clears both FTS surfaces. Requires WAL journal mode.
    """
    # Check the path before connecting: aiosqlite silently creates an empty
    # database for a typo, which then fails later with "no such table".
    if not Path(db_path).is_file():
        raise click.ClickException(f"Database not found: {db_path}")
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    ctx.obj["dry_run"] = dry_run


@redact_command.command("message")
@click.argument("ids", nargs=-1, type=int, required=True)
@click.pass_context
def _message_cmd(ctx, ids):
    """Tombstone specific message_log rows by id."""
    db_path = ctx.obj["db_path"]
    dry_run = ctx.obj["dry_run"]

    async def _run():
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA busy_timeout = 5000")
        try:
            # Capture sample token before tombstoning for verification
            sample_token = None
            if not dry_run and ids:
                async with conn.execute(
                    "SELECT message FROM message_log WHERE id = ? LIMIT 1",
                    (ids[0],),
                ) as c:
                    msg_row = await c.fetchone()
                if msg_row:
                    sample_token = _extract_sample_token(msg_row[0])

            result = await redact_messages(conn, list(ids), dry_run=dry_run)
            # Report found-vs-requested so a mistyped id is visible.
            requested = result.get("requested_messages", len(ids))
            if dry_run:
                click.echo(
                    f"[dry-run] would redact {result['affected_messages']} of "
                    f"{requested} messages, "
                    f"{result.get('affected_memories', 0)} memory records, "
                    f"{result.get('affected_exchanges', 0)} exchange_log rows"
                )
            else:
                click.echo(
                    f"Redacted {result['affected_messages']} of {requested} "
                    f"messages, "
                    f"{result.get('affected_memories', 0)} memory records, "
                    f"{result.get('affected_exchanges', 0)} exchange_log rows"
                )
                if sample_token:
                    click.echo(await _verify_fts_clean(conn, sample_token))
                else:
                    click.echo("verified: no sample token extracted (short/empty content)")
        finally:
            await conn.close()

    asyncio.run(_run())


@redact_command.command("memory")
@click.argument("memory_id", type=int)
@click.pass_context
def _memory_cmd(ctx, memory_id):
    """Tombstone a memory record and its raw message range."""
    db_path = ctx.obj["db_path"]
    dry_run = ctx.obj["dry_run"]

    async def _run():
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA busy_timeout = 5000")
        try:
            # Capture sample token from memory summary before tombstoning.
            # Probe first: on a database without the memory schema the query
            # would fail with a raw SQLite error; redact_memory_id reports it.
            sample_token = None
            if not dry_run and "memory" in await _probe_tables(conn):
                async with conn.execute(
                    "SELECT summary FROM memory WHERE id = ?", (memory_id,)
                ) as c:
                    mem_row = await c.fetchone()
                if mem_row:
                    sample_token = _extract_sample_token(mem_row[0])

            try:
                result = await redact_memory_id(conn, memory_id, dry_run=dry_run)
            except ValueError as exc:
                # e.g. "memory table not present in this database": an
                # operator error, reported as one line rather than a traceback.
                raise click.ClickException(str(exc)) from exc
            if dry_run:
                click.echo(
                    f"[dry-run] would redact {result['affected_messages']} messages, "
                    f"{result['affected_memories']} memory records, "
                    f"{result['affected_exchanges']} exchange_log rows"
                )
            else:
                click.echo(
                    f"Redacted {result['affected_messages']} messages, "
                    f"{result['affected_memories']} memory records, "
                    f"{result['affected_exchanges']} exchange_log rows"
                )
                if sample_token:
                    click.echo(await _verify_fts_clean(conn, sample_token))
                else:
                    click.echo("verified: no sample token extracted (short/empty content)")
        finally:
            await conn.close()

    asyncio.run(_run())


@redact_command.command("range")
@click.argument("start_id", type=int)
@click.argument("end_id", type=int)
@click.pass_context
def _range_cmd(ctx, start_id, end_id):
    """Tombstone all message_log rows in the given id range."""
    db_path = ctx.obj["db_path"]
    dry_run = ctx.obj["dry_run"]

    async def _run():
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA busy_timeout = 5000")
        try:
            # Capture sample token from first message in range before tombstoning
            sample_token = None
            if not dry_run:
                async with conn.execute(
                    "SELECT message FROM message_log "
                    "WHERE id BETWEEN ? AND ? ORDER BY id LIMIT 1",
                    (start_id, end_id),
                ) as c:
                    range_row = await c.fetchone()
                if range_row:
                    sample_token = _extract_sample_token(range_row[0])

            result = await redact_range(conn, start_id, end_id, dry_run=dry_run)
            # The ids that exist in the range; redact_range omits the key
            # (and finds nothing) when the range is empty.
            requested = result.get("requested_messages", 0)
            if dry_run:
                click.echo(
                    f"[dry-run] would redact {result['affected_messages']} of "
                    f"{requested} messages, "
                    f"{result.get('affected_memories', 0)} memory records, "
                    f"{result.get('affected_exchanges', 0)} exchange_log rows"
                )
            else:
                click.echo(
                    f"Redacted {result['affected_messages']} of {requested} "
                    f"messages, "
                    f"{result.get('affected_memories', 0)} memory records, "
                    f"{result.get('affected_exchanges', 0)} exchange_log rows"
                )
                if sample_token:
                    click.echo(await _verify_fts_clean(conn, sample_token))
                else:
                    click.echo("verified: no sample token extracted (short/empty content)")
        finally:
            await conn.close()

    asyncio.run(_run())
