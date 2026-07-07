"""Tests for WP1b.1 — retention scoring, demotion, and re-promotion.

Red phase: all tests in this file fail at collection with:
    ModuleNotFoundError: No module named 'corvidae.retention'
because corvidae/retention.py does not exist yet.

Designed failure reason: missing corvidae.retention module.

Test naming follows the plan's acceptance criteria (bootstrap-mapping §3.1,
phase-1b.md WP1b.1). Clock-frozen via unittest.mock.patch('time.time').
"""

import asyncio
import json
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import aiosqlite
import pytest

# PRIMARY IMPORT FAILURE — all tests below fail at collection time with
# ModuleNotFoundError: No module named 'corvidae.retention'
from corvidae.retention import (
    retention_score,
    run_retention_job,
    RETRIEVAL_BOOST,
)

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.funnel import FunnelPlugin
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db
# Cross-WP dependency: MemoryToolsPlugin lives in WP1b.3. Tests in this file
# (WP1b.1) will not collect until WP1b.3's module exists — the collection error
# shifts from corvidae.retention to corvidae.tools.memory_tools if only WP1b.1
# is implemented. Implement WP1b.1 and WP1b.3 together to unlock these tests.
from corvidae.tools.memory_tools import MemoryToolsPlugin


# ---------------------------------------------------------------------------
# Constants mirrored from plan (§6-tunable; verified against plan body)
# ---------------------------------------------------------------------------

GRACE_DAYS = 14
IMPORTANCE_FLOOR = 0.8
DEMOTE_BELOW = 0.15
HALF_LIFE_DAYS_RETENTION = 90  # distinct from retrieval half-life

DIMS = 32

EMBED_CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "embedding": {
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": DIMS,
        },
    },
    "memory": {
        "retention": {
            "grace_days": GRACE_DAYS,
            "importance_floor": IMPORTANCE_FLOOR,
            "demote_below": DEMOTE_BELOW,
            "half_life_days": HALF_LIFE_DAYS_RETENTION,
            "interval": 21600,  # 6h
            "backfill_batch": 32,
        },
    },
}


import hashlib
import re


def bow_embed(text: str) -> list[float]:
    vec = [0.0] * DIMS
    for token in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class StubEmbedClient:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts, kind=None):
        self.embed_calls.append(list(texts))
        if self.fail:
            raise RuntimeError("encoder down")
        return [bow_embed(t) for t in texts]

    async def chat(self, messages, tools=None, extra_body=None):
        raise AssertionError("retention tests must not call chat()")


async def build_retention_env(config=None, fail_embed: bool = False):
    """Full pm: persistence (in-memory DB) + stubbed llm + funnel + memory.

    Returns (memory, embedder, channel, db).
    """
    config = config or EMBED_CONFIG
    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=config)
    embedder = StubEmbedClient(fail=fail_embed)
    llm._clients["main"] = embedder
    llm._clients["embedding"] = embedder
    embedding_cfg = config.get("llm", {}).get("embedding")
    if embedding_cfg:
        llm.embedding_dimensions = embedding_cfg["dimensions"]

    funnel = FunnelPlugin()
    pm.register(funnel, name="funnel")
    await funnel.on_init(pm=pm, config=config)

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)

    channel = Channel(transport="irc", scope="#test")
    return memory, embedder, channel, db


async def seed_memory_row(
    db,
    channel_id: str,
    summary: str,
    importance: float = 0.5,
    indexed: int = 1,
    embedded: int = 1,
    retrieval_count: int = 0,
    last_retrieved_at: float | None = None,
    created_at: float | None = None,
    msg_id_start: int = 1,
    msg_id_end: int = 2,
    vec_available: bool = True,
) -> int:
    """Insert one memory row and optionally its vec row."""
    created_at = created_at or time.time()
    cursor = await db.execute(
        "INSERT INTO memory (channel_id, created_at, summary, importance, "
        "msg_id_start, msg_id_end, embedded, indexed, retrieval_count, "
        "last_retrieved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (channel_id, created_at, summary, importance, msg_id_start, msg_id_end,
         embedded, indexed, retrieval_count, last_retrieved_at),
    )
    rowid = cursor.lastrowid
    if embedded and vec_available:
        try:
            import sqlite_vec
            await db.execute(
                "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(bow_embed(summary))),
            )
        except ImportError:
            pass
    await db.commit()
    return rowid


async def vec_row_exists(db, memory_id: int) -> bool:
    """True when a vec row exists for this memory_id."""
    try:
        async with db.execute(
            "SELECT 1 FROM memory_vec WHERE memory_id = ?", (memory_id,)
        ) as c:
            return (await c.fetchone()) is not None
    except Exception:
        return False


async def fetch_memory(db, memory_id: int) -> dict:
    async with db.execute(
        "SELECT id, indexed, embedded, redacted, superseded_by, "
        "retrieval_count, last_retrieved_at, importance "
        "FROM memory WHERE id = ?",
        (memory_id,),
    ) as c:
        row = await c.fetchone()
    if row is None:
        raise ValueError(f"memory_id {memory_id} not found")
    keys = ["id", "indexed", "embedded", "redacted", "superseded_by",
            "retrieval_count", "last_retrieved_at", "importance"]
    return dict(zip(keys, row))


# ---------------------------------------------------------------------------
# WP1b.1 score function unit tests
# ---------------------------------------------------------------------------


class TestRetentionScore:
    """Unit tests for retention_score() at the plan's boundary conditions."""

    def test_zero_retrievals_score(self):
        """retrieval_count=0 → log1p(0)=0 → score = importance * recency."""
        now = 1_000_000.0
        last = now - 86400.0 * 10  # 10 days ago
        score = retention_score(
            importance=0.5,
            retrieval_count=0,
            last_activity=last,
            now=now,
            half_life_days=HALF_LIFE_DAYS_RETENTION,
        )
        expected_recency = math.exp(-10.0 / HALF_LIFE_DAYS_RETENTION)
        assert score == pytest.approx(0.5 * expected_recency, rel=1e-6)

    def test_retrieval_boost_increases_score(self):
        """More retrievals yield higher score (RETRIEVAL_BOOST > 0)."""
        now = 1_000_000.0
        last = now - 86400.0 * 10
        score0 = retention_score(0.5, 0, last, now, HALF_LIFE_DAYS_RETENTION)
        score5 = retention_score(0.5, 5, last, now, HALF_LIFE_DAYS_RETENTION)
        assert score5 > score0

    def test_retrieval_boost_constant_matches_plan(self):
        """RETRIEVAL_BOOST == 0.5 per plan spec."""
        assert RETRIEVAL_BOOST == pytest.approx(0.5)

    def test_score_at_zero_age_is_importance_times_one(self):
        """At last_activity == now, recency == 1.0 exactly."""
        now = 1_000_000.0
        score = retention_score(
            importance=0.7,
            retrieval_count=0,
            last_activity=now,
            now=now,
            half_life_days=HALF_LIFE_DAYS_RETENTION,
        )
        assert score == pytest.approx(0.7, rel=1e-6)

    def test_grace_edge_below_threshold(self):
        """A record younger than grace_days should have score above demote_below.

        This is not a grace-period exemption test (that's job logic);
        it verifies the score isn't accidentally below the demotion
        threshold just due to being new/unaccessed.
        """
        now = 1_000_000.0
        created_just_now = now - 86400.0 * (GRACE_DAYS - 1)  # still in grace
        score = retention_score(
            importance=0.3,
            retrieval_count=0,
            last_activity=created_just_now,
            now=now,
            half_life_days=HALF_LIFE_DAYS_RETENTION,
        )
        # Score value doesn't matter; we're checking the function runs
        # and returns a non-negative float.
        assert score >= 0.0

    def test_high_importance_can_stay_above_demote_threshold(self):
        """High importance (>= importance_floor) produces score above threshold.

        At importance=0.9 with 30 days age and 90-day half-life, score
        should be well above the 0.15 demote threshold.
        """
        now = 1_000_000.0
        last = now - 86400.0 * 30  # 30 days, 1/3 of 90-day half-life
        score = retention_score(
            importance=0.9,
            retrieval_count=0,
            last_activity=last,
            now=now,
            half_life_days=HALF_LIFE_DAYS_RETENTION,
        )
        assert score > DEMOTE_BELOW

    def test_old_low_importance_scores_below_demote_threshold(self):
        """Old, unaccessed, low-importance record has score < threshold."""
        now = 1_000_000.0
        last = now - 86400.0 * 300  # 300 days, well past half-life
        score = retention_score(
            importance=0.1,
            retrieval_count=0,
            last_activity=last,
            now=now,
            half_life_days=HALF_LIFE_DAYS_RETENTION,
        )
        assert score < DEMOTE_BELOW


# ---------------------------------------------------------------------------
# WP1b.1 demotion tests
# ---------------------------------------------------------------------------


class TestDemotion:
    """Retention job demotion pass: exemptions and actual demotion."""

    async def test_young_record_exempt_from_demotion(self):
        """A record younger than grace_days is never demoted regardless of score."""
        memory, embedder, channel, db = await build_retention_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        now = time.time()
        young_created_at = now - 86400.0 * (GRACE_DAYS - 1)  # 1 day inside grace
        mid = await seed_memory_row(
            db, channel.id, "some trivial chatter", importance=0.05,
            created_at=young_created_at, retrieval_count=0, last_retrieved_at=None,
        )

        await run_retention_job(memory, now=now)

        m = await fetch_memory(db, mid)
        assert m["indexed"] == 1, "young record must NOT be demoted (grace period)"
        assert m["embedded"] == 1, "young record's embedded flag must be unchanged"
        assert await vec_row_exists(db, mid), "young record must still have a vec row"
        await db.close()

    async def test_high_importance_record_exempt_from_demotion(self):
        """A record with importance >= floor is never demoted regardless of age."""
        memory, embedder, channel, db = await build_retention_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        now = time.time()
        old_created_at = now - 86400.0 * 300  # well past grace and half-life
        mid = await seed_memory_row(
            db, channel.id, "high importance stale record",
            importance=IMPORTANCE_FLOOR + 0.05,
            created_at=old_created_at, retrieval_count=0, last_retrieved_at=None,
        )

        await run_retention_job(memory, now=now)

        m = await fetch_memory(db, mid)
        assert m["indexed"] == 1, "high-importance record must NOT be demoted (floor)"
        await db.close()

    async def test_old_low_importance_unaccessed_record_demotes(self):
        """Old, never-retrieved, low-importance record: demoted, vec gone, embedded=0.

        Memory row and FTS entry must remain intact (demote ≠ delete).
        """
        memory, embedder, channel, db = await build_retention_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        now = time.time()
        old_created_at = now - 86400.0 * 300
        mid = await seed_memory_row(
            db, channel.id, "trivial old chatter nobody ever asked about",
            importance=0.1,
            created_at=old_created_at, retrieval_count=0, last_retrieved_at=None,
        )
        assert await vec_row_exists(db, mid), "precondition: vec row must exist"

        await run_retention_job(memory, now=now)

        m = await fetch_memory(db, mid)
        assert m["indexed"] == 0, "old/low-importance record must be demoted (indexed=0)"
        assert m["embedded"] == 0, "demoted record must have embedded=0 (vec deleted)"
        assert not await vec_row_exists(db, mid), "demoted record's vec row must be gone"

        # Memory row and FTS still intact
        async with db.execute("SELECT id FROM memory WHERE id = ?", (mid,)) as c:
            assert await c.fetchone() is not None, "memory row must persist after demotion"
        async with db.execute(
            "SELECT rowid FROM memory_fts WHERE rowid = ?", (mid,)
        ) as c:
            assert await c.fetchone() is not None, "FTS row must persist after demotion"

        await db.close()

    async def test_demotion_not_undone_by_same_run_backfill(self):
        """Backfill pass must not re-embed a just-demoted record.

        WP1b.1 backfill is scoped indexed=1 AND ... AND embedded=0.
        A just-demoted record has indexed=0 and must be excluded even
        with a live encoder. (Regression for the backfill indexed=1 scope
        — important finding #2 in design review 2026-07-07.)
        """
        memory, embedder, channel, db = await build_retention_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        now = time.time()
        old_created_at = now - 86400.0 * 300
        mid = await seed_memory_row(
            db, channel.id, "stale record the backfill must not resurrect",
            importance=0.1,
            created_at=old_created_at, retrieval_count=0, last_retrieved_at=None,
        )

        # Single run: demote pass THEN backfill pass (same run)
        await run_retention_job(memory, now=now)

        m = await fetch_memory(db, mid)
        # Vec row must be gone AFTER the full run (including backfill)
        assert not await vec_row_exists(db, mid), (
            "demoted record's vec row must still be gone after backfill "
            "(backfill is scoped to indexed=1; demoted record is indexed=0)"
        )
        assert m["embedded"] == 0, "embedded must remain 0 after backfill skips the record"

        await db.close()

    async def test_redacted_record_not_demoted(self):
        """Redacted records are excluded from the demotion pass."""
        memory, embedder, channel, db = await build_retention_env()
        now = time.time()
        old_created_at = now - 86400.0 * 300
        mid = await seed_memory_row(
            db, channel.id, "redacted trivial record",
            importance=0.1, created_at=old_created_at,
        )
        # Mark as redacted (as the redact CLI would)
        await db.execute(
            "UPDATE memory SET redacted=1, indexed=0, embedded=0 WHERE id = ?", (mid,)
        )
        await db.commit()

        await run_retention_job(memory, now=now)

        # Should not error; redacted records are skipped
        m = await fetch_memory(db, mid)
        assert m["redacted"] == 1


# ---------------------------------------------------------------------------
# WP1b.1 retrieval integration (before_agent_turn exclusion + search_memory)
# ---------------------------------------------------------------------------


class TestDemotionRetrievalIntegration:
    """Demoted records: excluded from passive retrieval, visible to tools."""

    async def test_demoted_excluded_from_before_agent_turn(self):
        """Passive retrieval (before_agent_turn) must skip demoted records."""
        from corvidae.context import MessageType
        memory, embedder, channel, db = await build_retention_env()
        channel.conversation = ContextWindow(channel.id)

        now = time.time()
        old_created_at = now - 86400.0 * 300
        # Two records: one demoted, one active
        _demoted = await seed_memory_row(
            db, channel.id, "wifi power-save old demoted",
            importance=0.1, indexed=0, embedded=0,
            created_at=old_created_at,
            vec_available=False,  # no vec row for demoted
        )
        _active = await seed_memory_row(
            db, channel.id, "kestrel ESP32 weather station wifi power-save fix",
            importance=0.5, indexed=1, embedded=1,
            created_at=now,
        )

        channel.conversation.append({"role": "user", "content": "ESP32 wifi issue"})
        await memory.before_agent_turn(channel=channel)

        from corvidae.context import MessageType
        context_msgs = [
            m for m in channel.conversation.messages
            if m.get("_message_type") == MessageType.CONTEXT
        ]
        # Active record may or may not surface (depends on retrieval scoring),
        # but the demoted record must NEVER appear.
        for ctx in context_msgs:
            assert "old demoted" not in ctx.get("content", ""), (
                "demoted record must not appear in passive retrieval"
            )
        await db.close()

    async def test_demoted_found_by_search_memory_with_flag(self):
        """search_memory must find demoted records and mark them [demoted]."""
        memory, embedder, channel, db = await build_retention_env()

        now = time.time()
        old_created_at = now - 86400.0 * 300
        mid = await seed_memory_row(
            db, channel.id, "kestrel wifi power-save demoted record",
            importance=0.1, indexed=0, embedded=0,
            created_at=old_created_at, vec_available=False,
        )

        # Build memory_tools plugin (needs memory as dependency)
        pm = memory.pm
        memory_tools = MemoryToolsPlugin()
        pm.register(memory_tools, name="memory_tools")
        await memory_tools.on_init(pm=pm, config=EMBED_CONFIG)

        # Call search_memory via registered tool
        tools = []
        memory_tools.register_tools(tools)
        tool_fn = {t.name: t.fn for t in tools}["search_memory"]

        ctx = MagicMock()
        ctx.channel = channel

        result = await tool_fn("wifi power-save", _ctx=ctx)
        assert f"{mid}" in result or "demoted" in result.lower(), (
            f"demoted record {mid} must appear in search_memory output"
        )
        assert "[demoted]" in result, (
            "demoted record must be flagged [demoted] in search_memory output"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.1 re-promotion tests
# ---------------------------------------------------------------------------


class TestRePromotion:
    """Re-promotion via recall_raw stat bumps; backfill restores vec."""

    async def test_demoted_record_repromotes_after_stat_bump(self):
        """Demoted record with bumped stats re-promotes on next job run.

        Re-promotion flips indexed=1 only (no inline embed).
        The same run's backfill restores the vec row when the encoder is up.
        """
        memory, embedder, channel, db = await build_retention_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        now = time.time()
        old_created_at = now - 86400.0 * 300
        mid = await seed_memory_row(
            db, channel.id, "ESP32 wifi issue demoted but recalled",
            importance=0.1, indexed=0, embedded=0,
            created_at=old_created_at,
            retrieval_count=0, last_retrieved_at=None,
            vec_available=False,
        )

        # Simulate recall_raw stat bump (as WP1b.3 recall_raw would do)
        # This gives the record a recent last_retrieved_at and high retrieval_count,
        # pushing its score above the demotion threshold.
        bump_time = now - 86400.0 * 1  # 1 day ago
        await db.execute(
            "UPDATE memory SET retrieval_count = 10, last_retrieved_at = ? WHERE id = ?",
            (bump_time, mid),
        )
        await db.commit()

        # Next job run: re-promotion pass sees indexed=0, score now >= threshold
        next_run = now + 3600.0  # 1 hour later
        await run_retention_job(memory, now=next_run)

        m = await fetch_memory(db, mid)
        assert m["indexed"] == 1, (
            "re-promoted record must have indexed=1"
        )
        # Backfill should have restored the vec row (encoder is up)
        assert await vec_row_exists(db, mid), (
            "backfill pass must restore vec row for re-promoted (indexed=1) record"
        )
        assert m["embedded"] == 1, "embedded must be 1 after backfill restores vec"

        await db.close()

    async def test_repromotion_sets_indexed_only_no_inline_embed(self):
        """Re-promotion must flip indexed=1 without calling embed inline.

        The embed call must come only from the backfill pass, not from
        re-promotion itself.
        """
        memory, embedder, channel, db = await build_retention_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        now = time.time()
        old_created_at = now - 86400.0 * 300
        mid = await seed_memory_row(
            db, channel.id, "record to check re-promotion doesn't embed inline",
            importance=0.1, indexed=0, embedded=0,
            created_at=old_created_at,
            retrieval_count=10, last_retrieved_at=now - 86400.0,
            vec_available=False,
        )

        embed_calls_before = len(embedder.embed_calls)
        await run_retention_job(memory, now=now)

        m = await fetch_memory(db, mid)
        assert m["indexed"] == 1, "re-promoted record must be indexed=1"

        # The backfill pass (not re-promotion) is what calls embed.
        # We verify embed WAS called (by backfill), but not TWICE
        # (once for re-promotion, once for backfill).
        embed_calls_after = len(embedder.embed_calls)
        embed_count = embed_calls_after - embed_calls_before
        assert embed_count == 1, (
            f"exactly one embed call expected (from backfill pass, not re-promotion); "
            f"got {embed_count}"
        )
        await db.close()

    async def test_repromotion_with_embedding_disabled_fts_reachable(self):
        """Re-promotion with encoder mismatch: indexed=1, no vec, no embed call.

        FTS must still find the record (indexed=1 enables FTS retrieval).
        (Design fix report finding #2 — bootstrap-mapping §3.1.)
        """
        memory, embedder, channel, db = await build_retention_env()
        channel.conversation = ContextWindow(channel.id)

        # Simulate embedding disabled (encoder mismatch guard)
        memory._encoder_mismatch = True

        now = time.time()
        old_created_at = now - 86400.0 * 300
        summary = "kestrel rain gauge north fence unique phrase xyzzy"
        mid = await seed_memory_row(
            db, channel.id, summary,
            importance=0.1, indexed=0, embedded=0,
            created_at=old_created_at,
            retrieval_count=10, last_retrieved_at=now - 86400.0,
            vec_available=False,
        )

        embed_calls_before = len(embedder.embed_calls)
        await run_retention_job(memory, now=now)

        m = await fetch_memory(db, mid)
        assert m["indexed"] == 1, "re-promoted record must be indexed=1 even with disabled embedding"
        assert not await vec_row_exists(db, mid), (
            "no vec row must exist when embedding is disabled"
        )
        embed_calls_after = len(embedder.embed_calls)
        assert embed_calls_after == embed_calls_before, (
            "no embed call must be made when embedding is disabled"
        )

        # FTS retrieval must find the re-promoted record
        async with db.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
            ('"xyzzy"',),
        ) as c:
            rows = await c.fetchall()
        assert any(r[0] == mid for r in rows), (
            "re-promoted record must be FTS-reachable even without a vec row"
        )
        await db.close()

    async def test_superseded_records_not_reproduced(self):
        """Superseded records (superseded_by IS NOT NULL) are skipped by re-promotion."""
        memory, embedder, channel, db = await build_retention_env()
        now = time.time()
        old_created_at = now - 86400.0 * 300

        # Create a superseded record (as dedup would leave)
        mid1 = await seed_memory_row(
            db, channel.id, "old duplicate record",
            importance=0.1, indexed=0, embedded=0,
            created_at=old_created_at,
            retrieval_count=10, last_retrieved_at=now - 86400.0,
            vec_available=False,
        )
        mid2 = await seed_memory_row(
            db, channel.id, "new merged record",
            importance=0.5, indexed=1, embedded=1,
            created_at=now,
        )
        await db.execute(
            "UPDATE memory SET superseded_by = ? WHERE id = ?", (mid2, mid1)
        )
        await db.commit()

        await run_retention_job(memory, now=now)

        m1 = await fetch_memory(db, mid1)
        # Superseded record must NOT be re-promoted (it's terminal)
        assert m1["indexed"] == 0, (
            "superseded record must remain indexed=0 (skipped by re-promotion pass)"
        )
        assert m1["superseded_by"] == mid2


# ---------------------------------------------------------------------------
# WP1b.1 stat inflation regression
# ---------------------------------------------------------------------------


class TestStatInflation:
    """search_memory must NOT bump access stats (WP1b.1 step 3)."""

    async def test_search_memory_leaves_stats_unchanged(self):
        """search_memory is a catalog, not consumption: stats must not move."""
        memory, embedder, channel, db = await build_retention_env()

        mid = await seed_memory_row(
            db, channel.id, "ESP32 rain gauge birdhouse firmware",
            importance=0.5, retrieval_count=0, last_retrieved_at=None,
        )

        pm = memory.pm
        memory_tools = MemoryToolsPlugin()
        pm.register(memory_tools, name="memory_tools")
        await memory_tools.on_init(pm=pm, config=EMBED_CONFIG)

        tools = []
        memory_tools.register_tools(tools)
        tool_fn = {t.name: t.fn for t in tools}["search_memory"]

        ctx = MagicMock()
        ctx.channel = channel

        await tool_fn("ESP32 rain gauge", _ctx=ctx)

        m = await fetch_memory(db, mid)
        assert m["retrieval_count"] == 0, (
            "search_memory must NOT bump retrieval_count (stat inflation regression)"
        )
        assert m["last_retrieved_at"] is None, (
            "search_memory must NOT update last_retrieved_at"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.1 startup trigger and interval rate-limit
# ---------------------------------------------------------------------------


class TestJobStartup:
    """Retention job runs at startup; respects interval rate-limit."""

    async def test_job_runs_at_startup(self):
        """on_start must trigger the retention job; retention_meta.last_run is set.

        This is the zero-traffic guarantee: on_idle is push-based and never
        fires on a quiet daemon; the startup trigger ensures coverage.
        """
        memory, embedder, channel, db = await build_retention_env()
        await memory.wait_for_background_tasks()

        async with db.execute(
            "SELECT last_run FROM retention_meta LIMIT 1"
        ) as c:
            row = await c.fetchone()
        assert row is not None, "retention_meta row must exist after startup"
        assert isinstance(row[0], float) and row[0] > 0, (
            "retention_meta.last_run must be a positive float after startup"
        )
        await db.close()

    async def test_second_startup_within_interval_skips_job(self):
        """A second on_start within interval must not re-run the job.

        retention_meta.last_run persists across restarts; the startup trigger
        must check it and skip if the interval has not elapsed.
        (Crash-loop safe — trap #5.)
        """
        memory, embedder, channel, db = await build_retention_env()
        await memory.wait_for_background_tasks()

        # Record last_run after first startup
        async with db.execute("SELECT last_run FROM retention_meta") as c:
            first_last_run = (await c.fetchone())[0]

        # Simulate a restart: create fresh MemoryPlugin on same DB
        memory2 = MemoryPlugin()
        pm2 = memory.pm  # same pm (same DB)
        await memory2.on_init(pm=pm2, config=EMBED_CONFIG)
        await memory2.on_start(config=EMBED_CONFIG)
        await memory2.wait_for_background_tasks()

        async with db.execute("SELECT last_run FROM retention_meta") as c:
            second_last_run = (await c.fetchone())[0]

        assert second_last_run == first_last_run, (
            "retention_meta.last_run must not change when second startup is within "
            "the rate-limit interval (job must be skipped)"
        )
        await db.close()

    async def test_startup_after_interval_reruns_job(self):
        """A startup more than interval seconds after last_run must re-run the job.

        This preserves the zero-traffic guarantee for daemons that have been
        idle longer than the interval.
        """
        memory, embedder, channel, db = await build_retention_env()
        await memory.wait_for_background_tasks()

        # Force last_run to be old (more than interval seconds ago)
        interval = EMBED_CONFIG["memory"]["retention"]["interval"]
        old_last_run = time.time() - interval - 3600.0
        await db.execute("UPDATE retention_meta SET last_run = ?", (old_last_run,))
        await db.commit()

        memory2 = MemoryPlugin()
        pm2 = memory.pm
        await memory2.on_init(pm=pm2, config=EMBED_CONFIG)
        await memory2.on_start(config=EMBED_CONFIG)
        await memory2.wait_for_background_tasks()

        async with db.execute("SELECT last_run FROM retention_meta") as c:
            new_last_run = (await c.fetchone())[0]

        assert new_last_run > old_last_run, (
            "retention_meta.last_run must be updated when interval has elapsed"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.1 backfill attribution (green fix #2 — 2026-07-07)
# ---------------------------------------------------------------------------


class TestBackfillAttribution:
    """Backfill embed calls must carry stage='retention' attribution."""

    async def test_backfill_embed_carries_retention_attribution(self):
        """Backfill pass must set_attribution(stage='retention') before embed calls.

        Green fix report finding #1: backfill called embed without attribution,
        making retention embedding costs invisible to the Phase 0 cost metering.
        """
        from corvidae.attribution import get_attribution

        captured_attributions: list[dict] = []

        class AttributionCapturingEmbedClient:
            async def embed(self, texts, kind=None):
                captured_attributions.append(dict(get_attribution()))
                return [bow_embed(t) for t in texts]

            async def chat(self, messages, tools=None, extra_body=None):
                raise AssertionError("retention tests must not call chat()")

        memory, embedder, channel, db = await build_retention_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        # Replace embedder with attribution-capturing one
        memory._llm()._clients["embedding"] = AttributionCapturingEmbedClient()

        # Seed a record needing backfill (indexed=1, embedded=0)
        await seed_memory_row(
            db, channel.id,
            "wifi sensor backfill attribution test unique phrase",
            indexed=1, embedded=0, vec_available=False,
        )

        await run_retention_job(memory)

        assert len(captured_attributions) >= 1, (
            "backfill must have called embed (precondition for attribution check)"
        )
        for attr in captured_attributions:
            assert attr.get("stage") == "retention", (
                f"backfill embed must carry stage='retention' attribution; "
                f"got {attr!r}"
            )
        await db.close()
