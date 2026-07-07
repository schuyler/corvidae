"""Retention scoring, demotion, and re-promotion for MemoryPlugin (WP1b.1).

bootstrap-mapping §3.1: usage-weighted retention, not culling.
Constants are §6-tunable; the job runs as a silent background task
(MemoryPlugin._spawn), never through the TaskQueue.
"""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from corvidae.attribution import reset_attribution, set_attribution

if TYPE_CHECKING:
    from corvidae.memory import MemoryPlugin

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
    """Three-pass retention job: demote → re-promote → backfill.

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

    demoted_ids = []
    for row in rows:
        mid, importance, retrieval_count, last_retrieved_at, created_at = row
        if now - created_at < grace_seconds:
            continue
        if importance >= importance_floor:
            continue
        last_activity = max(created_at, last_retrieved_at or 0.0)
        score = retention_score(importance, retrieval_count, last_activity, now, half_life_days)
        if score < demote_below:
            demoted_ids.append(mid)

    if demoted_ids:
        async with memory._db_lock:
            for mid in demoted_ids:
                await db.execute(
                    "UPDATE memory SET indexed = 0, embedded = 0 WHERE id = ?", (mid,)
                )
                if memory._vec_available:
                    try:
                        await db.execute(
                            "DELETE FROM memory_vec WHERE memory_id = ?", (mid,)
                        )
                    except Exception:
                        logger.debug(
                            "vec row delete failed for memory %d", mid, exc_info=True
                        )
            await db.commit()
        logger.info("retention: demoted %d records", len(demoted_ids))

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
        async with memory._db_lock:
            for mid in repromoted_ids:
                await db.execute(
                    "UPDATE memory SET indexed = 1 WHERE id = ?", (mid,)
                )
            await db.commit()
        logger.info("retention: re-promoted %d records", len(repromoted_ids))

    # ------------------------------------------------------------------
    # Pass 3: Backfill
    # indexed=1 AND redacted=0 AND superseded_by IS NULL AND embedded=0
    # The indexed=1 filter is load-bearing: demotion (pass 1) sets embedded=0,
    # and an unfiltered backfill would re-embed just-demoted records, undoing
    # the vec delete and violating trap #1 / the DoD vec-rows ≤ indexed bound.
    # Skip entirely when embedding is disabled by the embedding_meta mismatch guard.
    # ------------------------------------------------------------------
    if memory._encoder_mismatch:
        logger.debug("retention: backfill skipped (embedding disabled by meta mismatch)")
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
        async with memory._db_lock:
            for (mid, _summary), embedding in zip(backfill_rows, embeddings):
                # vec0 does not support INSERT OR REPLACE; clear any stale row.
                await db.execute(
                    "DELETE FROM memory_vec WHERE memory_id = ?", (mid,)
                )
                await db.execute(
                    "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                    (mid, sqlite_vec.serialize_float32(embedding)),
                )
                await db.execute(
                    "UPDATE memory SET embedded = 1 WHERE id = ?", (mid,)
                )
            await db.commit()
        logger.info("retention: backfilled %d records", len(backfill_rows))
    except Exception:
        logger.warning("retention: backfill embedding failed", exc_info=True)
    finally:
        reset_attribution(attribution_token)
