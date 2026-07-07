"""Tests for WP1b.2 — near-duplicate merge at consolidation time.

Red phase failure mode: the near-identical merge test fails at ASSERTION TIME
with AssertionError: expected 1 indexed record, got 2. The dedup logic
(dup query + supersede + stats fold) does not exist yet in corvidae/memory.py.

Other tests (sub-threshold, sibling-channel, embedding-failed, raw-range-intact)
pass in current code because they test 'no merge' scenarios that are already true
when dedup is absent. They are regression guards: they must also pass after 1b.

Designed failure reason (primary test): missing near-dup merge logic in
MemoryPlugin._consolidate_range (corvidae/memory.py).
"""

import asyncio
import json
import math
import hashlib
import re
import time
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.channel import Channel
from corvidae.context import MessageType
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db


DIMS = 4  # small for speed; dedup uses cosine similarity only

EMBED_CONFIG_DEDUP = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "background": {"base_url": "http://localhost:8080", "model": "bg"},
        "embedding": {
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": DIMS,
        },
    },
    "memory": {
        "dup_threshold": 0.95,
        "idle_consolidate_after": 1800,
    },
}

# --- Controllable stub embedder for dedup ---


class DupStubClient:
    """Stub embedder with a per-call embedding map.

    If the text appears in self.fixed_embeddings, the mapped vector is returned.
    Otherwise, a deterministic bow_embed is used.
    """

    def __init__(self):
        self.chat_text: str = "{}"
        self.chat_calls: list = []
        self.embed_calls: list[list[str]] = []
        self.fail_embed: bool = False
        # Map from text substring → forced vector
        self.fixed_embeddings: dict[str, list[float]] = {}

    def _embed_one(self, text: str) -> list[float]:
        for key, vec in self.fixed_embeddings.items():
            if key in text:
                return list(vec)
        return _bow_embed_dims(text, DIMS)

    async def chat(self, messages, tools=None, extra_body=None):
        self.chat_calls.append(messages)
        return {
            "choices": [{"message": {"role": "assistant", "content": self.chat_text}}]
        }

    async def embed(self, texts, kind=None):
        self.embed_calls.append(list(texts))
        if self.fail_embed:
            raise RuntimeError("encoder down for test")
        return [self._embed_one(t) for t in texts]


def _bow_embed_dims(text: str, dims: int) -> list[float]:
    vec = [0.0] * dims
    for token in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dims
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_NEAR_IDENTICAL_SUMMARY = (
    "I helped kestrel debug the ESP32 weather station wifi power-save dropout issue"
)
_DISTINCT_SUMMARY_A = (
    "kestrel's ESP32 weather station had wifi power-save dropout issues"
)
_DISTINCT_SUMMARY_B = (
    "robin ordered BC547 transistors from the usual electronics distributor"
)


async def build_dedup_env(config=None):
    config = config or EMBED_CONFIG_DEDUP
    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=config)

    stub = DupStubClient()
    llm._clients["main"] = stub
    llm._clients["background"] = stub
    llm._clients["embedding"] = stub
    embedding_cfg = config.get("llm", {}).get("embedding")
    if embedding_cfg:
        llm.embedding_dimensions = embedding_cfg["dimensions"]

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)

    prior = MagicMock()
    prior.score = AsyncMock(return_value=0.7)
    memory.importance_prior = prior

    channel = Channel(transport="irc", scope="#test")
    return memory, persistence, stub, channel, db


async def seed_dialog(persistence, channel, texts: list[tuple[str, str]]) -> list[int]:
    rowids = []
    for role, content in texts:
        rowids.append(
            await persistence.on_conversation_event(
                channel=channel,
                message={"role": role, "content": content},
                message_type=MessageType.MESSAGE,
            )
        )
    return rowids


async def fetch_all_memories(db) -> list[dict]:
    async with db.execute(
        "SELECT id, channel_id, summary, importance, retrieval_count, "
        "last_retrieved_at, indexed, embedded, superseded_by, "
        "msg_id_start, msg_id_end "
        "FROM memory ORDER BY id"
    ) as c:
        rows = await c.fetchall()
    keys = ["id", "channel_id", "summary", "importance", "retrieval_count",
            "last_retrieved_at", "indexed", "embedded", "superseded_by",
            "msg_id_start", "msg_id_end"]
    return [dict(zip(keys, row)) for row in rows]


async def vec_count(db) -> int:
    try:
        async with db.execute("SELECT COUNT(*) FROM memory_vec") as c:
            return (await c.fetchone())[0]
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# WP1b.2 near-identical dedup (PRIMARY RED TEST — fails in current code)
# ---------------------------------------------------------------------------


class TestNearIdenticalDupDetection:
    """Primary red test: dedup must supersede the older record on near-identical consolidation."""

    async def test_near_identical_consolidations_supersede(self):
        """Two consolidations with nearly identical summaries → one indexed, one superseded.

        This is the primary RED test for WP1b.2. It FAILS in current code because
        MemoryPlugin._consolidate_range has no dup-detection step. After 1b
        implementation, exactly one record is indexed=1 and one has superseded_by set.

        Failure mode: AssertionError — expected 1 indexed record, got 2.
        """
        memory, persistence, stub, channel, db = await build_dedup_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable; dedup requires vec similarity")

        # Stub returns the same near-identical summary for both consolidations.
        stub.chat_text = json.dumps({
            "summary": _NEAR_IDENTICAL_SUMMARY,
            "topic_tags": ["wifi", "esp32"],
            "participants": ["kestrel"],
        })
        # Fixed embedding: both summaries get the same vector → similarity = 1.0
        stub.fixed_embeddings[_NEAR_IDENTICAL_SUMMARY[:20]] = [1.0, 0.0, 0.0, 0.0]

        rowids1 = await seed_dialog(persistence, channel, [
            ("user", "wifi issue on my ESP32 weather station"),
            ("assistant", "disable WIFI_PS_NONE to fix it"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids1
        )
        await memory.wait_for_background_tasks()

        rowids2 = await seed_dialog(persistence, channel, [
            ("user", "still about the ESP32 wifi power-save issue"),
            ("assistant", "yes WIFI_PS_NONE is the fix here"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids2
        )
        await memory.wait_for_background_tasks()

        memories = await fetch_all_memories(db)
        indexed = [m for m in memories if m["indexed"] == 1 and m["superseded_by"] is None]
        superseded = [m for m in memories if m["superseded_by"] is not None]

        # PRIMARY ASSERTION — fails in current code (no dedup logic)
        assert len(indexed) == 1, (
            f"expected 1 indexed record after near-dup merge, got {len(indexed)}"
        )
        assert len(superseded) == 1, (
            f"expected 1 superseded record after near-dup merge, got {len(superseded)}"
        )
        # The surviving record should carry summed retrieval_count (0+0=0 here,
        # but importance should be max(0.7, 0.7)=0.7 and msg_id ranges merged)
        survivor = indexed[0]
        old = superseded[0]
        assert survivor["importance"] >= old["importance"], (
            "survivor importance must be max(new, old)"
        )
        # Superseded record must have embedded=0 (vec deleted, flag truthful)
        assert old["embedded"] == 0, (
            "superseded record must have embedded=0 (vec row was deleted)"
        )
        assert old["indexed"] == 0, "superseded record must be indexed=0"
        await db.close()

    async def test_survivor_carries_summed_retrieval_count(self):
        """Survivor's retrieval_count = new.retrieval_count + old.retrieval_count.

        Fails in current code (no merge → two separate records each with count 0).
        """
        memory, persistence, stub, channel, db = await build_dedup_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        stub.chat_text = json.dumps({
            "summary": _NEAR_IDENTICAL_SUMMARY,
            "topic_tags": ["wifi"],
            "participants": ["kestrel"],
        })
        stub.fixed_embeddings[_NEAR_IDENTICAL_SUMMARY[:20]] = [1.0, 0.0, 0.0, 0.0]

        # Seed the first consolidation and manually bump its retrieval_count
        rowids1 = await seed_dialog(persistence, channel, [
            ("user", "ESP32 wifi power-save"),
            ("assistant", "WIFI_PS_NONE fix"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids1
        )
        await memory.wait_for_background_tasks()

        # Manually bump retrieval_count to 3 on the first record
        await db.execute("UPDATE memory SET retrieval_count = 3")
        await db.commit()

        rowids2 = await seed_dialog(persistence, channel, [
            ("user", "more ESP32 wifi power-save discussion"),
            ("assistant", "still WIFI_PS_NONE"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids2
        )
        await memory.wait_for_background_tasks()

        memories = await fetch_all_memories(db)
        indexed = [m for m in memories if m["indexed"] == 1 and m["superseded_by"] is None]

        assert len(indexed) == 1, f"expected 1 indexed after merge, got {len(indexed)}"
        assert indexed[0]["retrieval_count"] >= 3, (
            "survivor must carry sum of retrieval_counts (old=3 + new=0 = 3)"
        )
        await db.close()

    async def test_msg_id_range_spans_both_records_after_merge(self):
        """Merged survivor: msg_id_start = min(old, new), msg_id_end = max(old, new).

        Fails in current code (no merge).
        """
        memory, persistence, stub, channel, db = await build_dedup_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        stub.chat_text = json.dumps({
            "summary": _NEAR_IDENTICAL_SUMMARY,
            "topic_tags": [],
            "participants": [],
        })
        stub.fixed_embeddings[_NEAR_IDENTICAL_SUMMARY[:20]] = [1.0, 0.0, 0.0, 0.0]

        rowids1 = await seed_dialog(persistence, channel, [
            ("user", "first segment"), ("assistant", "reply one"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids1
        )
        await memory.wait_for_background_tasks()

        rowids2 = await seed_dialog(persistence, channel, [
            ("user", "second segment"), ("assistant", "reply two"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids2
        )
        await memory.wait_for_background_tasks()

        memories = await fetch_all_memories(db)
        indexed = [m for m in memories if m["indexed"] == 1 and m["superseded_by"] is None]
        assert len(indexed) == 1, f"expected 1 indexed, got {len(indexed)}"

        all_ids = rowids1 + rowids2
        assert indexed[0]["msg_id_start"] == min(all_ids), (
            f"survivor msg_id_start must be min of both ranges, "
            f"got {indexed[0]['msg_id_start']}, expected {min(all_ids)}"
        )
        assert indexed[0]["msg_id_end"] == max(all_ids), (
            f"survivor msg_id_end must be max of both ranges, "
            f"got {indexed[0]['msg_id_end']}, expected {max(all_ids)}"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.2 sub-threshold (regression guard — passes in current code)
# ---------------------------------------------------------------------------


class TestSubThresholdNoDup:
    """Sub-threshold pairs: both records stay indexed. Regression guard."""

    async def test_sub_threshold_pairs_both_indexed(self):
        """Two records with cosine similarity < dup_threshold: both stay indexed=1.

        This test already passes (no dedup = no merge = both indexed).
        It is a regression guard: it must also pass after 1b implementation.
        """
        memory, persistence, stub, channel, db = await build_dedup_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        # First consolidation: electronics summary
        stub.chat_text = json.dumps({
            "summary": _DISTINCT_SUMMARY_A,
            "topic_tags": ["wifi", "esp32"],
            "participants": ["kestrel"],
        })
        rowids1 = await seed_dialog(persistence, channel, [
            ("user", "ESP32 wifi power-save"), ("assistant", "WIFI_PS_NONE"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids1
        )
        await memory.wait_for_background_tasks()

        # Second consolidation: completely different topic (parts ordering)
        stub.chat_text = json.dumps({
            "summary": _DISTINCT_SUMMARY_B,
            "topic_tags": ["parts", "transistors"],
            "participants": ["robin"],
        })
        rowids2 = await seed_dialog(persistence, channel, [
            ("user", "need more BC547s"), ("assistant", "robin will order"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids2
        )
        await memory.wait_for_background_tasks()

        # Verify the two summaries actually have low similarity
        vec_a = _bow_embed_dims(_DISTINCT_SUMMARY_A, DIMS)
        vec_b = _bow_embed_dims(_DISTINCT_SUMMARY_B, DIMS)
        sim = _cosine(vec_a, vec_b)
        assert sim < 0.95, (
            f"test precondition: summaries must be sub-threshold; similarity={sim:.3f}"
        )

        memories = await fetch_all_memories(db)
        indexed = [m for m in memories if m["indexed"] == 1]
        assert len(indexed) == 2, (
            f"sub-threshold pairs must both stay indexed=1, got {len(indexed)} indexed"
        )
        assert all(m["superseded_by"] is None for m in memories), (
            "no record should be superseded below threshold"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.2 compartment boundary (regression guard — passes in current code)
# ---------------------------------------------------------------------------


class TestSiblingChannelCompartmentBoundary:
    """Near-dup on a sibling group channel must NOT be merged (compartment boundary).

    Dup query is scoped to same channel_id, not _channel_scope group.
    Both records stay indexed=1.
    """

    async def test_sibling_channel_not_merged(self):
        """Near-identical records in different (sibling) channels: both stay indexed."""
        config = {
            **EMBED_CONFIG_DEDUP,
            "memory": {
                **EMBED_CONFIG_DEDUP["memory"],
                "channel_groups": {
                    "home": ["irc:#electronics", "irc:#garden"],
                },
            },
        }
        memory, persistence, stub, channel_a, db = await build_dedup_env(config)
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        channel_b = Channel(transport="irc", scope="#garden")

        # Identical summary, but different channels
        summary = "kestrel solar charge controller MQTT power data"
        stub.chat_text = json.dumps({
            "summary": summary,
            "topic_tags": ["solar", "mqtt"],
            "participants": ["kestrel"],
        })

        rowids_a = await seed_dialog(persistence, channel_a, [
            ("user", "solar data on channel a"), ("assistant", "MQTT topic"),
        ])
        await memory.on_compaction(
            channel=channel_a, summary_msg={}, retain_count=0, compacted_ids=rowids_a
        )
        await memory.wait_for_background_tasks()

        rowids_b = await seed_dialog(persistence, channel_b, [
            ("user", "solar data on channel b"), ("assistant", "same MQTT topic"),
        ])
        await memory.on_compaction(
            channel=channel_b, summary_msg={}, retain_count=0, compacted_ids=rowids_b
        )
        await memory.wait_for_background_tasks()

        memories = await fetch_all_memories(db)
        indexed = [m for m in memories if m["indexed"] == 1]
        assert len(indexed) == 2, (
            f"sibling-channel near-dup must NOT be merged (compartment boundary); "
            f"expected 2 indexed, got {len(indexed)}"
        )
        superseded = [m for m in memories if m["superseded_by"] is not None]
        assert len(superseded) == 0, (
            "no record should be superseded when the near-dup is on a sibling channel"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.2 embedding-failed: skips dup detection, inserts with embedded=0
# ---------------------------------------------------------------------------


class TestEmbeddingFailedSkipsDup:
    """When the new record's embedding fails, dup detection is skipped.

    The record is inserted with embedded=0 (same as Phase 1a behavior).
    No dup detection runs when there is no vector to compare.
    """

    async def test_embedding_failed_inserts_embedded_zero_skips_dup(self):
        """Embedding failure → embedded=0, no superseded record created.

        This is a regression guard: in current code embedding failure already
        gives embedded=0. After 1b, the dedup path must not run when embedding
        failed (no vector to compare).
        """
        memory, persistence, stub, channel, db = await build_dedup_env()

        # Seed an existing indexed record that COULD be a near-dup
        stub.chat_text = json.dumps({
            "summary": _NEAR_IDENTICAL_SUMMARY,
            "topic_tags": [],
            "participants": [],
        })
        stub.fixed_embeddings[_NEAR_IDENTICAL_SUMMARY[:20]] = [1.0, 0.0, 0.0, 0.0]

        rowids1 = await seed_dialog(persistence, channel, [
            ("user", "first segment"), ("assistant", "reply"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids1
        )
        await memory.wait_for_background_tasks()

        # Now fail the embedder for the second consolidation
        stub.fail_embed = True
        stub.chat_text = json.dumps({
            "summary": _NEAR_IDENTICAL_SUMMARY,
            "topic_tags": [],
            "participants": [],
        })

        rowids2 = await seed_dialog(persistence, channel, [
            ("user", "second segment about same topic"), ("assistant", "same reply"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids2
        )
        await memory.wait_for_background_tasks()

        memories = await fetch_all_memories(db)
        # The second record should be inserted with embedded=0 (embed failed)
        second_record = [m for m in memories if m["id"] == max(m2["id"] for m2 in memories)][0]
        assert second_record["embedded"] == 0, (
            "record inserted after embedding failure must have embedded=0"
        )
        # No record should be superseded (dedup skipped because no vector)
        superseded = [m for m in memories if m["superseded_by"] is not None]
        assert len(superseded) == 0, (
            "dup detection must be skipped when new record's embedding failed; "
            "no superseded records expected"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.2 raw-range integrity for superseded records
# ---------------------------------------------------------------------------


class TestSupersededRawRangeIntact:
    """Raw message_log range for a superseded record must remain accessible.

    Superseding sets indexed=0 and embedded=0 and deletes the vec row,
    but the memory row and its raw dialog (message_log) are never deleted.
    This is a regression guard.
    """

    async def test_message_log_rows_intact_for_superseded_record(self):
        """message_log rows for a superseded record's range must still exist.

        This verifies 'raw range intact' without calling recall_raw (which
        is implemented in WP1b.3 / test_memory_tools.py). The DB invariant
        is tested directly.
        """
        memory, persistence, stub, channel, db = await build_dedup_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        stub.chat_text = json.dumps({
            "summary": _NEAR_IDENTICAL_SUMMARY,
            "topic_tags": [],
            "participants": [],
        })
        stub.fixed_embeddings[_NEAR_IDENTICAL_SUMMARY[:20]] = [1.0, 0.0, 0.0, 0.0]

        rowids1 = await seed_dialog(persistence, channel, [
            ("user", "first dialog segment"), ("assistant", "first reply"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids1
        )
        await memory.wait_for_background_tasks()

        rowids2 = await seed_dialog(persistence, channel, [
            ("user", "second dialog near-dup"), ("assistant", "second reply"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids2
        )
        await memory.wait_for_background_tasks()

        memories = await fetch_all_memories(db)
        superseded = [m for m in memories if m["superseded_by"] is not None]

        # This assertion passes even in current code (no superseded = empty list)
        # but is a guard for after 1b: the raw range must remain.
        for sup in superseded:
            async with db.execute(
                "SELECT COUNT(*) FROM message_log "
                "WHERE channel_id = ? AND id BETWEEN ? AND ?",
                (channel.id, sup["msg_id_start"], sup["msg_id_end"]),
            ) as c:
                count = (await c.fetchone())[0]
            assert count > 0, (
                f"message_log rows for superseded record {sup['id']} "
                f"(range {sup['msg_id_start']}–{sup['msg_id_end']}) must not be deleted"
            )

        # Also verify the memory row itself is still present
        for sup in superseded:
            async with db.execute(
                "SELECT id FROM memory WHERE id = ?", (sup["id"],)
            ) as c:
                assert await c.fetchone() is not None, (
                    f"memory row {sup['id']} must persist after being superseded"
                )

        await db.close()
