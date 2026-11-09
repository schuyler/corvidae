"""Hybrid retrieval candidates (#666; PRD §3.7 Memory 4).

On the vector path the candidate pool is the union of the vec KNN and an
FTS5 search over the same records, ranked by reciprocal rank fusion, so a
record naming a person or identifier is found even when the embedding
doesn't place it near the query. ``memory.retrieval.candidates: vector``
restores the vec-only pool. A candidate only FTS found bands at no less than
``FTS_SIMILARITY_PROXY``; its ``similarity`` stays the real cosine, and each
logged candidate records which search found it (``source``).
"""

import logging

import aiosqlite
import pytest
from corvidae_cognition.memory import (
    CANDIDATES_CHOICES,
    FTS_SIMILARITY_PROXY,
    MemoryPlugin,
    _fts_match_query,
)

from .test_memory_retrieval import (
    EMBED_CONFIG,
    bow_embed,
    build_retrieval_env,
    context_messages,
    seed_memory,
)
from .test_memory_window_exclusion import WINDOW_START, fill_window, seed

pytest.importorskip("sqlite_vec", reason="hybrid fusion needs the vector path")

# The record that names the person, and records that share the query's
# common words (so they win on cosine) but not the name.
NAMED = "Yoz prefers shipping releases on Thursdays"
DISTRACTORS = [
    "what about that one",
    "what about this",
    "what about it then",
    "and what about those",
    "so what about them",
]
QUERY = "what about Yoz"


def config(candidates: str | None = None, k: int = 1, **retrieval) -> dict:
    """EMBED_CONFIG with memory.retrieval.k and, optionally, candidates."""
    retrieval = {"k": k, **retrieval}
    if candidates is not None:
        retrieval["candidates"] = candidates
    return {**EMBED_CONFIG, "memory": {"retrieval": retrieval}}


def cosine(a: str, b: str) -> float:
    """Cosine similarity of two texts under the stub embedder."""
    return sum(x * y for x, y in zip(bow_embed(a), bow_embed(b)))


async def seed_named_and_distractors(db, channel_id: str) -> int:
    """Seed the distractors, then NAMED; return NAMED's id."""
    for text in DISTRACTORS:
        await seed_memory(db, channel_id, text)
    return await seed_memory(db, channel_id, NAMED)


class TestFtsQueryStopwords:
    """The hybrid side's FTS query drops common words."""

    def test_stopwords_dropped_when_asked(self):
        assert _fts_match_query("what about Yoz", stopwords=True) == '"Yoz"'

    def test_all_stopwords_gives_no_query(self):
        assert _fts_match_query("what about that", stopwords=True) is None

    def test_default_keeps_every_word(self):
        """search_memory and the degraded path keep today's builder."""
        assert _fts_match_query("what about Yoz") == '"what" OR "about" OR "Yoz"'

    def test_cap_counts_words_left_after_stopwords(self):
        text = " ".join(["the"] * 30 + [f"w{i}" for i in range(25)])
        match = _fts_match_query(text, stopwords=True)
        assert match is not None
        assert match.count(" OR ") == 19 and '"the"' not in match


class TestFusion:
    def test_choices_default_to_hybrid(self):
        assert CANDIDATES_CHOICES == ("hybrid", "vector")

    async def test_name_query_ranks_the_naming_record_first(self):
        # Sanity: every distractor beats NAMED on cosine, so vec alone loses it.
        assert all(cosine(QUERY, d) > cosine(QUERY, NAMED) for d in DISTRACTORS)
        memory, _e, channel, _conv, db = await build_retrieval_env(config())
        named = await seed_named_and_distractors(db, channel.id)

        pool = await memory._retrieve_pool(QUERY)

        assert pool.ranked[0]["id"] == named
        await db.close()

    async def test_vector_mode_is_the_vec_only_pool(self):
        memory, _e, channel, _conv, db = await build_retrieval_env(config("vector"))
        named = await seed_named_and_distractors(db, channel.id)

        pool = await memory._retrieve_pool(QUERY)

        # k=1 fetches 4 by KNN; NAMED ranks sixth on cosine, so it is absent.
        assert named not in [c["id"] for c in pool.ranked]
        assert {c["source"] for c in pool.ranked} == {"vec"}
        await db.close()

    async def test_fts_only_hit_has_real_cosine_and_fts_source(self):
        memory, _e, channel, _conv, db = await build_retrieval_env(config())
        named = await seed_named_and_distractors(db, channel.id)

        pool = await memory._retrieve_pool(QUERY)

        hit = next(c for c in pool.ranked if c["id"] == named)
        assert hit["source"] == "fts"
        assert hit["similarity"] == pytest.approx(cosine(QUERY, NAMED), abs=1e-5)
        await db.close()

    async def test_record_in_both_lists_is_marked_both(self):
        memory, _e, channel, _conv, db = await build_retrieval_env(config(k=8))
        named = await seed_named_and_distractors(db, channel.id)

        pool = await memory._retrieve_pool(QUERY)

        sources = {c["id"]: c["source"] for c in pool.ranked}
        assert sources[named] == "both"
        assert set(sources.values()) == {"vec", "both"}
        await db.close()

    async def test_fts_only_hit_without_vector_gets_the_proxy(self):
        """A record with no vector (not yet embedded) can still be found by
        FTS; its similarity is the proxy, as on the degraded path."""
        memory, _e, channel, _conv, db = await build_retrieval_env(config())
        for text in DISTRACTORS:
            await seed_memory(db, channel.id, text)
        named = await seed_memory(db, channel.id, NAMED, embedded=False)

        pool = await memory._retrieve_pool(QUERY)

        hit = next(c for c in pool.ranked if c["id"] == named)
        assert hit["similarity"] == FTS_SIMILARITY_PROXY
        assert hit["source"] == "fts"
        await db.close()

    async def test_query_with_no_lexical_hit_ranks_as_vector_only(self):
        """Requirement 2: no FTS match means the fused order is the vec order."""
        orders = []
        for mode in ("hybrid", "vector"):
            memory, _e, channel, _conv, db = await build_retrieval_env(config(mode, k=8))
            for text in [NAMED, *DISTRACTORS]:
                await seed_memory(db, channel.id, text)
            pool = await memory._retrieve_pool("what about those")  # all stopwords
            orders.append([(c["id"], round(c["similarity"], 6)) for c in pool.ranked])
            await db.close()
        assert orders[0] == orders[1]


class TestBandingFloor:
    async def test_fts_hit_bands_at_least_moderate_and_is_admitted(self):
        """The name hit's cosine is weak; the keyword floor keeps it from
        being dropped as weak. k=1 keeps it outside the KNN's k x 4 rows,
        so it is an FTS-only hit, the case the floor is for."""
        assert cosine(QUERY, NAMED) < 0.6
        memory, _e, channel, conv, db = await build_retrieval_env(config(k=1))
        await seed_named_and_distractors(db, channel.id)

        conv.append({"role": "user", "content": QUERY})
        await memory.before_agent_turn(channel=channel, trace_id=None, annotations={})

        recalled = " ".join(m["content"] for m in context_messages(conv))
        assert f"[moderate] (now) {NAMED}" in recalled
        await db.close()

    async def test_floor_raises_the_band_not_similarity_or_score(self):
        memory, _e, channel, _conv, db = await build_retrieval_env(config())
        named = await seed_named_and_distractors(db, channel.id)

        pool = await memory._retrieve_pool(QUERY)

        hit = next(c for c in pool.ranked if c["id"] == named)
        assert hit["band"] == "moderate"
        assert hit["score"] < 0.6 and hit["similarity"] < 0.6
        await db.close()

    async def test_record_found_by_both_keeps_its_measured_band(self):
        """Only FTS-only hits are floored: once the KNN returns the record
        too, its band is the measured one, as before #666."""
        memory, _e, channel, _conv, db = await build_retrieval_env(config(k=8))
        named = await seed_named_and_distractors(db, channel.id)

        pool = await memory._retrieve_pool(QUERY)

        hit = next(c for c in pool.ranked if c["id"] == named)
        assert hit["source"] == "both"
        assert hit["band"] == "weak"
        await db.close()

    async def test_vec_only_candidates_are_banded_as_before(self):
        memory, _e, channel, _conv, db = await build_retrieval_env(config(k=8))
        await seed_named_and_distractors(db, channel.id)

        pool = await memory._retrieve_pool(QUERY)

        for c in pool.ranked:
            if c["source"] == "vec":
                assert c["score"] == pytest.approx(c["similarity"], rel=1e-3)


class TestCandidateLog:
    async def test_source_is_logged_per_candidate(self):
        memory, _e, channel, conv, db = await build_retrieval_env(config(k=1))
        named = await seed_named_and_distractors(db, channel.id)

        conv.append({"role": "user", "content": QUERY})
        await memory.before_agent_turn(channel=channel, trace_id=None, annotations={})

        async with db.execute(
            "SELECT memory_id, source, rank FROM retrieval_candidate ORDER BY rank"
        ) as cursor:
            rows = await cursor.fetchall()
        assert rows[0] == (named, "fts", 1)
        assert {source for _id, source, _rank in rows} <= {"vec", "fts", "both"}
        await db.close()

    async def test_degraded_candidates_log_fts_source(self):
        memory, embedder, channel, conv, db = await build_retrieval_env(config(k=2))
        await seed_named_and_distractors(db, channel.id)
        embedder.fail = True

        conv.append({"role": "user", "content": QUERY})
        await memory.before_agent_turn(channel=channel, trace_id=None, annotations={})

        async with db.execute("SELECT DISTINCT source FROM retrieval_candidate") as cursor:
            assert await cursor.fetchall() == [("fts",)]
        await db.close()

    async def test_existing_table_gains_a_nullable_source_column(self, tmp_path):
        """A retrieval_candidate table from before #666 is upgraded in place;
        its old rows read NULL."""
        path = tmp_path / "sessions.db"
        async with aiosqlite.connect(path) as db:
            await db.execute(
                "CREATE TABLE retrieval_candidate (retrieval_id INTEGER NOT NULL, "
                "memory_id INTEGER NOT NULL, similarity REAL NOT NULL, "
                "score REAL NOT NULL, band TEXT NOT NULL, rank INTEGER NOT NULL, "
                "outcome TEXT NOT NULL, PRIMARY KEY (retrieval_id, memory_id)) "
                "WITHOUT ROWID"
            )
            await db.execute(
                "INSERT INTO retrieval_candidate VALUES (1, 1, 0.5, 0.5, 'weak', 1, 'cut')"
            )
            await db.commit()

        memory, _e, _channel, _conv, db = await build_retrieval_env(db_path=str(path))
        await memory._ensure_schema()
        async with db.execute("SELECT source FROM retrieval_candidate") as cursor:
            assert await cursor.fetchall() == [(None,)]
        await db.close()


class TestWindowExclusion:
    async def test_in_window_fts_hit_is_excluded_before_the_cut(self):
        """An in-window record the FTS side matches doesn't take a slot, and
        is logged as excluded_in_window with its would-be rank."""
        memory, _e, channel, conv, db = await build_retrieval_env(config(k=2))
        for text in DISTRACTORS:
            await seed(db, channel.id, text, 1, 2)
        in_window = await seed(db, channel.id, NAMED, WINDOW_START, WINDOW_START + 1)
        older = await seed(db, channel.id, "Yoz reviewed the battery design", 1, 2)
        fill_window(conv)

        conv.append({"role": "user", "content": QUERY, "_db_id": WINDOW_START + 5})
        await memory.before_agent_turn(channel=channel, trace_id=None, annotations={})

        async with db.execute(
            "SELECT memory_id, outcome, rank, source FROM retrieval_candidate"
        ) as cursor:
            rows = {r[0]: r[1:] for r in await cursor.fetchall()}
        # The out-of-window name hit takes rank 1; the in-window one, which
        # would have ranked 2nd, is excluded rather than admitted.
        assert rows[older][:2] == ("admitted", 1)
        assert rows[in_window][:2] == ("excluded_in_window", 2)
        assert rows[in_window][2] in {"fts", "both"}
        await db.close()


class TestConfig:
    async def test_unknown_candidates_value_warns_and_uses_hybrid(self, caplog):
        with caplog.at_level(logging.WARNING):
            memory, _e, _channel, _conv, db = await build_retrieval_env(
                config("everything")
            )
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "memory.retrieval.candidates" in message
        assert "'everything'" in message and "hybrid" in message and "vector" in message
        assert isinstance(memory, MemoryPlugin) and memory._candidates == "hybrid"
        await db.close()
