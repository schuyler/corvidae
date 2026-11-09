"""Retention scoring, demotion, and re-promotion for MemoryPlugin (WP1b.1).

bootstrap-mapping §3.1: usage-weighted retention, not culling.
Constants are §6-tunable; the job runs as a silent background task
(MemoryPlugin._spawn), never through the TaskQueue. MemoryPlugin starts it
from a kernel-clock timer every ``memory.retention.interval`` (preferring a
quiet system, at most ``memory.retention.max_delay`` late) and from a
startup check; ``retention_meta.last_run`` rate-limits both.
"""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from corvidae.attribution import reset_attribution, set_attribution

if TYPE_CHECKING:
    from corvidae_cognition.memory import MemoryPlugin

logger = logging.getLogger("corvidae.retention")

# §6-tunable: raise to weight recall traffic more heavily in the score.
RETRIEVAL_BOOST: float = 0.5


def retention_score(
    importance: float,
    retrieval_count: int,
    last_activity: float,
    now: float,
    half_life_days: float,
) -> float:
    """Usage-weighted retention (bootstrap-mapping §3.1).

    last_activity = max(created_at, last_retrieved_at or 0).
    """
    recency = math.exp(-((now - last_activity) / 86400.0) / half_life_days)
    return importance * (1.0 + RETRIEVAL_BOOST * math.log1p(retrieval_count)) * recency


async def run_retention_job(memory: "MemoryPlugin", now: float | None = None) -> None:
    """Retention job: demote → re-promote → prune the retrieval logs (when
    ``memory.retention.retrieval_log_max_age_days`` is set, #654) → backfill.

    Pass ordering is intentional: re-promoted records (indexed=1, embedded=0)
    are eligible for the same run's backfill pass, providing prompt vector
    restoration after recall_raw re-promotion.
    """
    if now is None:
        now = time.time()

    db = await memory._ensure_schema()
    if db is None:
        return

    retention_cfg = memory._memory_cfg.get("retention", {}) or {}
    grace_days = float(retention_cfg.get("grace_days", 14))
    importance_floor = float(retention_cfg.get("importance_floor", 0.8))
    demote_below = float(retention_cfg.get("demote_below", 0.15))
    half_life_days = float(retention_cfg.get("half_life_days", 90))
    backfill_batch = int(retention_cfg.get("backfill_batch", 32))
    grace_seconds = grace_days * 86400.0

    # ------------------------------------------------------------------
    # Pass 1: Demotion
    # indexed=1 AND redacted=0; past grace; below importance floor; score < threshold
    # ------------------------------------------------------------------
    async with db.execute(
        "SELECT id, importance, retrieval_count, last_retrieved_at, created_at "
        "FROM memory WHERE indexed = 1 AND redacted = 0"
    ) as cursor:
        rows = await cursor.fetchall()

    # (id, last_retrieved_at as selected): the write re-checks both (#272).
    demoted: list[tuple[int, float | None]] = []
    for row in rows:
        mid, importance, retrieval_count, last_retrieved_at, created_at = row
        if now - created_at < grace_seconds:
            continue
        if importance >= importance_floor:
            continue
        last_activity = max(created_at, last_retrieved_at or 0.0)
        score = retention_score(importance, retrieval_count, last_activity, now, half_life_days)
        if score < demote_below:
            demoted.append((mid, last_retrieved_at))

    if demoted:
        # One transaction on memory's write connection: the whole batch
        # demotes or none of it does (#257). A failure rolls back and
        # propagates; the job's caller logs it and the next run retries.
        # Each UPDATE re-checks the selection, so a row redacted, demoted
        # or retrieved since the SELECT above is left alone (#272).
        demoted_count = 0
        async with memory._db_lock, memory._write_txn() as wdb:
            for mid, selected_last_retrieved in demoted:
                cursor = await wdb.execute(
                    "UPDATE memory SET indexed = 0, embedded = 0 "
                    "WHERE id = ? AND indexed = 1 AND redacted = 0 "
                    "AND last_retrieved_at IS ?",
                    (mid, selected_last_retrieved),
                )
                if cursor.rowcount != 1:
                    continue
                demoted_count += 1
                if memory._vec_available:
                    try:
                        await wdb.execute(
                            "DELETE FROM memory_vec WHERE memory_id = ?", (mid,)
                        )
                    except Exception:
                        # The demotion stands; only its vec row is left behind.
                        # Nothing clears it later: the backfill selects
                        # indexed = 1 and this row is now indexed = 0, and the
                        # demotion pass only flips rows it finds at indexed = 1
                        # itself. Retrieval joins `m.indexed = 1` on both KNN
                        # queries, so the stale row is never returned; what it
                        # costs is the vec-rows <= indexed invariant and one
                        # dead over-fetch slot.
                        logger.warning(
                            "retention: vec row for memory %d was left behind "
                            "(delete failed) and no later pass clears it; the "
                            "row is demoted, so vector search never matches it "
                            "(both KNN queries join indexed = 1). The vec "
                            "table now holds one more row than there are "
                            "indexed memories. Clean it with DELETE FROM "
                            "memory_vec WHERE memory_id = %d, in a sqlite3 "
                            "that has the sqlite-vec extension loaded "
                            "(memory_vec is a vec0 virtual table)",
                            mid, mid, exc_info=True,
                        )
        if demoted_count:
            logger.info("retention: demoted %d records", demoted_count)

    # ------------------------------------------------------------------
    # Pass 2: Re-promotion
    # indexed=0 AND redacted=0 AND superseded_by IS NULL; score >= threshold
    # Re-promotion flips indexed=1 ONLY — no inline embed.
    # The backfill pass restores the vec row through the one embed code path.
    # ------------------------------------------------------------------
    async with db.execute(
        "SELECT id, importance, retrieval_count, last_retrieved_at, created_at "
        "FROM memory WHERE indexed = 0 AND redacted = 0 AND superseded_by IS NULL"
    ) as cursor:
        rows = await cursor.fetchall()

    repromoted_ids = []
    for row in rows:
        mid, importance, retrieval_count, last_retrieved_at, created_at = row
        last_activity = max(created_at, last_retrieved_at or 0.0)
        score = retention_score(importance, retrieval_count, last_activity, now, half_life_days)
        if score >= demote_below:
            repromoted_ids.append(mid)

    if repromoted_ids:
        # One transaction on memory's write connection (#257).
        # The UPDATE re-checks the selection, so a row superseded or
        # redacted since the SELECT is never re-indexed (#309, #272).
        repromoted_count = 0
        async with memory._db_lock, memory._write_txn() as wdb:
            for mid in repromoted_ids:
                cursor = await wdb.execute(
                    "UPDATE memory SET indexed = 1 WHERE id = ? AND indexed = 0 "
                    "AND redacted = 0 AND superseded_by IS NULL",
                    (mid,),
                )
                repromoted_count += cursor.rowcount == 1
        if repromoted_count:
            logger.info("retention: re-promoted %d records", repromoted_count)

    # ------------------------------------------------------------------
    # Pass 2b: prune the retrieval logs (#654), when an age is configured.
    # Runs before backfill, which returns early when embedding is off. A
    # failure rolls back, is logged, and leaves backfill to run.
    # ------------------------------------------------------------------
    max_age_days = memory._retrieval_log_max_age_days
    if max_age_days is not None:
        cutoff = now - max_age_days * 86400.0
        try:
            async with memory._db_lock, memory._write_txn() as wdb:
                # Children first, then the parents they belong to.
                cand_cursor = await wdb.execute(
                    "DELETE FROM retrieval_candidate WHERE retrieval_id IN "
                    "(SELECT id FROM retrieval_log WHERE ts < ?)",
                    (cutoff,),
                )
                log_cursor = await wdb.execute(
                    "DELETE FROM retrieval_log WHERE ts < ?", (cutoff,)
                )
            if log_cursor.rowcount:
                logger.info(
                    "retention: pruned %d retrieval_log rows and %d "
                    "retrieval_candidate rows older than %g days",
                    log_cursor.rowcount, cand_cursor.rowcount, max_age_days,
                )
        except Exception as exc:
            logger.warning(
                "retention: pruning retrieval_log rows older than %g days "
                "(memory.retention.retrieval_log_max_age_days) failed and was "
                "rolled back (%s); the next retention run retries",
                max_age_days, type(exc).__name__, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Pass 3: Backfill
    # indexed=1 AND redacted=0 AND superseded_by IS NULL AND embedded=0
    # The indexed=1 filter is load-bearing: demotion (pass 1) sets embedded=0,
    # and an unfiltered backfill would re-embed just-demoted records, undoing
    # the vec delete and violating trap #1 / the DoD vec-rows ≤ indexed bound.
    # Skip entirely when embedding is disabled by the embedding_meta mismatch guard.
    # ------------------------------------------------------------------
    if memory._encoder_mismatch:
        logger.debug("retention: backfill skipped (embedding disabled by annotations mismatch)")
        return

    if not memory._embedding_enabled() or not memory._vec_available:
        logger.debug("retention: backfill skipped (embedding not available)")
        return

    async with db.execute(
        "SELECT id, summary FROM memory "
        "WHERE indexed = 1 AND redacted = 0 AND superseded_by IS NULL AND embedded = 0 "
        "LIMIT ?",
        (backfill_batch,),
    ) as cursor:
        backfill_rows = await cursor.fetchall()

    if not backfill_rows:
        return

    attribution_token = set_attribution(stage="retention")
    try:
        import sqlite_vec
        client = memory._llm().get_client("embedding")
        texts = [row[1] for row in backfill_rows]
        embeddings = await client.embed(texts, kind="document")
        # A wrong-width vector would make vec0 reject its INSERT and roll
        # back the whole batch on every run; skip and log that row, keep the
        # rest (same handling as the write path).
        embeddings = [memory._check_embedding_width(e) for e in embeddings]
        # Embeddings are computed above, so no network await runs inside the
        # transaction; the batch commits whole or rolls back whole (#257).
        # Each row is re-checked inside the transaction before its vec row
        # is written: one demoted, redacted or superseded while the
        # embeddings were computed gets no vec row (#309, #272).
        backfilled_count = 0
        async with memory._db_lock, memory._write_txn() as wdb:
            for (mid, _summary), embedding in zip(backfill_rows, embeddings):
                if embedding is None:
                    continue
                cursor = await wdb.execute(
                    "UPDATE memory SET embedded = 1 WHERE id = ? AND indexed = 1 "
                    "AND redacted = 0 AND superseded_by IS NULL AND embedded = 0",
                    (mid,),
                )
                if cursor.rowcount != 1:
                    continue
                # vec0 does not support INSERT OR REPLACE; clear any stale row.
                await wdb.execute(
                    "DELETE FROM memory_vec WHERE memory_id = ?", (mid,)
                )
                await wdb.execute(
                    "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                    (mid, sqlite_vec.serialize_float32(embedding)),
                )
                backfilled_count += 1
        if backfilled_count:
            logger.info("retention: backfilled %d records", backfilled_count)
    except Exception as exc:
        logger.warning(
            "retention: backfill of %d records failed and was rolled back "
            "(%s); it will retry on the next retention run",
            len(backfill_rows), type(exc).__name__, exc_info=True,
        )
    finally:
        reset_attribution(attribution_token)
