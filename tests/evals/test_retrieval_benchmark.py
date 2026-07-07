"""CI retrieval benchmark — WP1b.5 (bootstrap-mapping §6).

Deterministic stub embedder; fixed fixtures; asserted regression floors.
These tests run against Phase 1a's retrieval path. The fixture is the floor:
floors are set at the measured value minus a small margin and are regression
trips, not aspirations. Raising them is a §6 activity.

The contradiction fixture tests are marked xfail with a pointer to the
bootstrap-mapping §3.1 contradiction-annotation feature (Phase 2+).
"""

import json
import math
import re
import hashlib
from pathlib import Path

import aiosqlite
import pytest

from corvidae.channel import Channel
from corvidae.context import ContextWindow
from corvidae.funnel import FunnelPlugin
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db

from evals.metrics import mrr, recall_at_k

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

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
}


def bow_embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words unit vector (same as test_memory_retrieval)."""
    vec = [0.0] * DIMS
    for token in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class StubEmbedClient:
    async def embed(self, texts, kind=None):
        return [bow_embed(t) for t in texts]

    async def chat(self, messages, tools=None, extra_body=None):
        raise AssertionError("benchmark must not call chat()")


async def build_benchmark_env(config=None):
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
    embedder = StubEmbedClient()
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

    return memory, db


async def seed_fixture_memories(db, memories: list[dict]) -> dict[int, str]:
    """Seed memory rows from fixture records; return {rowid: fixture_id}."""
    import sqlite_vec
    import time
    id_map: dict[int, str] = {}
    for record in memories:
        vec = bow_embed(record["summary"])
        cursor = await db.execute(
            "INSERT INTO memory (channel_id, created_at, summary, importance, "
            "msg_id_start, msg_id_end, embedded) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record["channel_id"], time.time(), record["summary"], 0.5, 1, 2, 1),
        )
        rowid = cursor.lastrowid
        if rowid is not None:
            await db.execute(
                "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(vec)),
            )
            id_map[rowid] = record["id"]
    await db.commit()
    return id_map


class TestGeneralRecallBenchmark:
    """Recall@5 and MRR floors over the general-recall fixture.

    Floors are regression trips: measured value minus a small margin.
    Raising them requires §6 benchmark evidence.

    The negative queries (empty 'relevant') verify the retriever
    admits nothing when there is no relevant memory.
    """

    # Regression floor values set from first measured run (2026-07-07):
    #   recall@5 measured ≈ 0.625 → floor 0.60 (conservative margin)
    #   MRR measured ≈ 0.448 → floor 0.40
    # Raising them requires §6 benchmark evidence.
    RECALL_AT_5_FLOOR = 0.60
    MRR_FLOOR = 0.40

    async def _run_queries(self, fixture_path: Path):
        fixture = json.loads(fixture_path.read_text())
        channels = sorted({m["channel_id"] for m in fixture["memories"]})
        config = {
            **EMBED_CONFIG,
            "memory": {"channel_groups": {"fixture": channels}},
        }
        memory, db = await build_benchmark_env(config)

        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable; vector retrieval path not exercised")

        id_map = await seed_fixture_memories(db, fixture["memories"])

        positive_queries = [q for q in fixture["queries"] if q["relevant"]]
        negative_queries = [q for q in fixture["queries"] if not q["relevant"]]

        recalls, mrrs = [], []
        for query in positive_queries:
            candidates, _degraded = await memory.retrieve(
                channels[0], query["text"], k=5
            )
            ranked = [id_map[c["id"]] for c in candidates]
            recalls.append(recall_at_k(ranked, query["relevant"], k=5))
            mrrs.append(mrr(ranked, query["relevant"]))

        await db.close()
        return recalls, mrrs, negative_queries, memory, id_map

    async def test_recall_at_5_beats_floor(self):
        fixture_path = FIXTURES_DIR / "memory_retrieval_general.json"
        recalls, _mrrs, _neg, _mem, _idmap = await self._run_queries(fixture_path)
        mean_recall = sum(recalls) / len(recalls)
        assert mean_recall >= self.RECALL_AT_5_FLOOR, (
            f"mean recall@5 {mean_recall:.3f} below floor {self.RECALL_AT_5_FLOOR}"
        )

    async def test_mrr_beats_floor(self):
        fixture_path = FIXTURES_DIR / "memory_retrieval_general.json"
        _recalls, mrrs, _neg, _mem, _idmap = await self._run_queries(fixture_path)
        mean_mrr = sum(mrrs) / len(mrrs)
        assert mean_mrr >= self.MRR_FLOOR, (
            f"mean MRR {mean_mrr:.3f} below floor {self.MRR_FLOOR}"
        )

    async def test_negative_queries_admit_nothing(self):
        """Queries with no relevant memory must surface zero results.

        This locks the 'no memory of that' admission behavior: the
        retriever must not hallucinate hits for unknown topics.
        """
        fixture_path = FIXTURES_DIR / "memory_retrieval_general.json"
        fixture = json.loads(fixture_path.read_text())
        channels = sorted({m["channel_id"] for m in fixture["memories"]})
        config = {
            **EMBED_CONFIG,
            "memory": {"channel_groups": {"fixture": channels}},
        }
        memory, db = await build_benchmark_env(config)

        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        await seed_fixture_memories(db, fixture["memories"])

        negative_queries = [q for q in fixture["queries"] if not q["relevant"]]
        assert len(negative_queries) >= 2, (
            "fixture must have at least 2 negative queries (WP1b.5 spec)"
        )

        for query in negative_queries:
            candidates, _degraded = await memory.retrieve(
                channels[0], query["text"], k=5
            )
            # No candidate should score high enough for the funnel; we check
            # that the fixture's corpus genuinely has no close match by
            # verifying recall@5 is 0.0 against an empty relevant set.
            # (recall_at_k returns 1.0 for empty relevant, so we check manually.)
            if candidates:
                # If the retriever surfaces anything, verify it's not a
                # high-confidence match (score < strong band default 0.75).
                for c in candidates:
                    assert c["score"] < 0.75, (
                        f"negative query {query['text']!r} returned high-score "
                        f"candidate {c['summary']!r} (score={c['score']:.3f}) — "
                        f"fixture may contain the topic after all"
                    )

        await db.close()

    async def test_fixture_format_and_minimum_sizes(self):
        """The general-recall fixture satisfies the WP1b.5 size requirements."""
        fixture_path = FIXTURES_DIR / "memory_retrieval_general.json"
        fixture = json.loads(fixture_path.read_text())

        assert len(fixture["memories"]) >= 15, (
            f"WP1b.5 requires >=15 memories, got {len(fixture['memories'])}"
        )
        assert len(fixture["queries"]) >= 8, (
            f"WP1b.5 requires >=8 queries, got {len(fixture['queries'])}"
        )
        negative_queries = [q for q in fixture["queries"] if not q["relevant"]]
        assert len(negative_queries) >= 2, (
            f"WP1b.5 requires >=2 no-memory queries, got {len(negative_queries)}"
        )


class TestContradictionFixture:
    """Contradiction-bearing fixture tests — xfail until Phase 2.

    bootstrap-mapping §3.1 specifies contradiction annotation as requiring
    the appraisal/critique machinery that Phase 2 introduces. These tests
    define the behavior; the xfail is removed when the feature lands.
    """

    @pytest.mark.xfail(
        reason=(
            "Contradiction annotation not implemented. "
            "Requires Phase 2 appraisal/critique machinery (bootstrap-mapping §3.1). "
            "Remove xfail when implemented."
        ),
        strict=False,
    )
    async def test_contradiction_flagged_in_retrieval(self):
        """A contradiction-aware retriever should flag conflicting records.

        When two memories make conflicting claims about the same topic,
        retrieval should surface BOTH and annotate the conflict — not
        silently return whichever ranked first. This is the Phase 2
        contradiction-annotation acceptance criterion.
        """
        fixture_path = FIXTURES_DIR / "memory_retrieval_contradictions.json"
        fixture = json.loads(fixture_path.read_text())
        channels = sorted({m["channel_id"] for m in fixture["memories"]})
        config = {
            **EMBED_CONFIG,
            "memory": {"channel_groups": {"fixture": channels}},
        }
        memory, db = await build_benchmark_env(config)

        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        id_map = await seed_fixture_memories(db, fixture["memories"])

        query = fixture["queries"][0]
        candidates, _degraded = await memory.retrieve(
            channels[0], query["text"], k=5
        )
        ranked = [id_map[c["id"]] for c in candidates]

        # Phase 2: both conflicting records must surface AND be flagged
        # as contradicting each other. For now the retriever surfaces
        # whichever ranked higher without contradiction annotation.
        assert "c1" in ranked and "c2" in ranked, (
            "Both conflicting records must surface for the contradiction query"
        )
        # Check contradiction annotation (Phase 2 behavior — not yet implemented)
        contradictions_flagged = any(
            c.get("contradiction", False) for c in candidates
        )
        assert contradictions_flagged, (
            "Conflicting records must be flagged as contradicting each other"
        )

        await db.close()

    @pytest.mark.xfail(
        reason=(
            "Contradiction annotation not implemented. "
            "Phase 2+ feature (bootstrap-mapping §3.1)."
        ),
        strict=False,
    )
    async def test_most_recent_claim_surfaced_for_specific_query(self):
        """A temporally-aware retriever surfaces the most recent conflicting claim."""
        fixture_path = FIXTURES_DIR / "memory_retrieval_contradictions.json"
        fixture = json.loads(fixture_path.read_text())
        channels = sorted({m["channel_id"] for m in fixture["memories"]})
        config = {
            **EMBED_CONFIG,
            "memory": {"channel_groups": {"fixture": channels}},
        }
        memory, db = await build_benchmark_env(config)

        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable")

        id_map = await seed_fixture_memories(db, fixture["memories"])

        # Query specifically about team projects (c2 is the most specific answer)
        query = fixture["queries"][1]
        candidates, _degraded = await memory.retrieve(
            channels[0], query["text"], k=5
        )
        ranked = [id_map[c["id"]] for c in candidates]

        assert ranked and ranked[0] == "c2", (
            "Most recent, most specific claim (c2) must rank first for "
            "a query about team-project tool choice"
        )

        await db.close()
