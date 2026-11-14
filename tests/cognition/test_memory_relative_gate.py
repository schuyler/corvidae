"""The relative admission gate (#409; PRD §3.7 Memory 4 and 5).

Passive retrieval admits a memory only when it earns its place:

- ``memory.retrieval.band_on: relative`` (the default) bands a record on the
  z-score of its raw cosine to the query against every indexed, unredacted,
  embedded record: ``(cos - mean) / stdev``. ``bands.strong`` /
  ``bands.moderate`` are z thresholds under it (defaults 4.5 / 3.0).
- Fewer than ``RELATIVE_MIN_RECORDS`` embedded records (or no spread) falls
  back to banding on raw similarity with the cosine defaults, logged once.
- Weak records are dropped in every mode, with no fallback: a query with
  nothing above weak admits nothing.
- A keyword-only hit bands on its measured cosine like any other; one with no
  stored vector bands weak, so it is not admitted passively.
- Lines read ``(age, memory <id>) summary`` for a strong record and
  ``[marginal] (age, memory <id>) summary`` for a moderate one (#632). The
  FTS5-degraded path still admits the top k, every line marked ``[marginal]``.
- Each logged candidate carries its z (NULL when no z was computed).
"""

import logging
import statistics
import time

import pytest
from corvidae_cognition.memory import (
    DEFAULT_BAND_MODERATE_Z,
    DEFAULT_BAND_STRONG_Z,
    RELATIVE_MIN_RECORDS,
)

from .test_memory_retrieval import (
    EMBED_CONFIG,
    bow_embed,
    build_retrieval_env,
    context_messages,
    seed_memory,
)

DAY = 86400.0

# The record the queries are about, and filler that shares none of its words.
# Sixty filler records: comfortably past RELATIVE_MIN_RECORDS, and enough that
# the stub embedder's 32 hashed dimensions let an exact match reach the strong
# z threshold (a z over N records can't exceed sqrt(N - 1)).
TARGET = "kestrel's ESP32 weather station dropped off wifi"
FILLER = [f"alpha{i} bravo{i} charlie{i}" for i in range(60)] + [
    "orchid pewter orchard",
]
# An off-topic message that shares no word with any seeded record.
NULL_QUERY = "thanks, that worked perfectly"


def needs_vec() -> None:
    """Skip a test that only means something on the vector path."""
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        pytest.skip("sqlite-vec unavailable; the relative gate needs vectors")


def config(**retrieval) -> dict:
    """EMBED_CONFIG with memory.retrieval keys set."""
    return {**EMBED_CONFIG, "memory": {"retrieval": retrieval}}


def cosine(a: str, b: str) -> float:
    """Cosine similarity of two texts under the stub embedder."""
    return sum(x * y for x, y in zip(bow_embed(a), bow_embed(b)))


def z_of(query: str, summary: str, corpus: list[str]) -> float:
    """The z-score the gate should compute for ``summary`` against ``corpus``."""
    sims = [cosine(query, text) for text in corpus]
    mean = statistics.fmean(sims)
    stdev = statistics.pstdev(sims)
    return (cosine(query, summary) - mean) / stdev


@pytest.fixture
async def env():
    """Factory for a retrieval environment; closes every DB on teardown."""
    dbs = []

    async def _make(cfg=None):
        built = await build_retrieval_env(cfg)
        dbs.append(built[-1])
        return built

    yield _make
    for db in dbs:
        await db.close()


async def seed_corpus(db, channel_id: str, target_age_days: float = 0.0) -> int:
    """Seed the filler records plus TARGET; return TARGET's id."""
    for text in FILLER:
        await seed_memory(db, channel_id, text)
    return await seed_memory(
        db, channel_id, TARGET, created_at=time.time() - target_age_days * DAY,
    )


async def ask(memory, channel, conv, text: str) -> None:
    """Append the user message and run passive retrieval."""
    conv.append({"role": "user", "content": text})
    await memory.before_agent_turn(channel=channel, trace_id=None, annotations={})


def recalled(conv) -> str:
    """Every admitted CONTEXT message's text, joined."""
    return "\n".join(m["content"] for m in context_messages(conv))


async def candidate_rows(db) -> dict[int, dict]:
    """retrieval_candidate rows keyed by memory_id."""
    async with db.execute(
        "SELECT memory_id, band, outcome, z, source FROM retrieval_candidate"
    ) as cursor:
        rows = await cursor.fetchall()
    return {
        r[0]: {"band": r[1], "outcome": r[2], "z": r[3], "source": r[4]}
        for r in rows
    }


class TestDefaults:
    async def test_relative_is_the_default_with_z_thresholds(self, env):
        memory, *_ = await env()
        assert memory._band_on == "relative"
        assert memory._band_strong == DEFAULT_BAND_STRONG_Z
        assert memory._band_moderate == DEFAULT_BAND_MODERATE_Z

    async def test_invalid_band_on_warns_and_defaults_to_relative(self, env, caplog):
        with caplog.at_level(logging.WARNING):
            memory, *_ = await env(config(band_on="vibes"))
        assert "band_on" in caplog.text and "relative" in caplog.text
        assert memory._band_on == "relative"

    async def test_bands_without_band_on_warns_that_they_are_z(self, env, caplog):
        # A deployment that tuned cosine thresholds before #409 would have
        # them silently reread as z values; say so, and how to keep them.
        with caplog.at_level(logging.WARNING):
            await env(config(bands={"strong": 0.75, "moderate": 0.6}))
        assert "memory.retrieval.bands" in caplog.text
        assert "band_on: score" in caplog.text

    async def test_bands_with_an_invalid_band_on_also_warn(self, env, caplog):
        # A misspelt band_on falls back to relative too, so the same rereading
        # applies and the same warning is due.
        with caplog.at_level(logging.WARNING):
            await env(config(band_on="simlarity", bands={"strong": 0.75}))
        assert "memory.retrieval.bands" in caplog.text
        assert "band_on: score" in caplog.text

    async def test_bands_with_band_on_set_do_not_warn(self, env, caplog):
        with caplog.at_level(logging.WARNING):
            await env(config(band_on="score", bands={"strong": 0.75}))
        assert "memory.retrieval.bands" not in caplog.text


class TestRelativeBanding:
    async def test_band_is_z_over_the_whole_memory(self, env):
        needs_vec()
        memory, _, channel, _, db = await env()
        target = await seed_corpus(db, channel.id)
        pool = await memory._retrieve_pool(TARGET)
        hit = next(c for c in pool.ranked if c["id"] == target)
        expected = z_of(TARGET, TARGET, [*FILLER, TARGET])
        assert hit["z"] == pytest.approx(expected, rel=1e-3)
        assert expected >= DEFAULT_BAND_STRONG_Z
        assert hit["band"] == "strong"

    async def test_spread_leaves_out_redacted_and_unindexed_records(self, env):
        # R2: the distribution is over indexed, unredacted records only.
        # Copies of TARGET that are redacted or demoted would drag the mean
        # up and shrink TARGET's z if they were counted.
        needs_vec()
        memory, _, channel, _, db = await env()
        target = await seed_corpus(db, channel.id)
        for column in ("redacted", "indexed"):
            for _ in range(5):
                extra = await seed_memory(db, channel.id, TARGET)
                value = 1 if column == "redacted" else 0
                await db.execute(
                    f"UPDATE memory SET {column} = ? WHERE id = ?", (value, extra),
                )
        await db.commit()
        pool = await memory._retrieve_pool(TARGET)
        hit = next(c for c in pool.ranked if c["id"] == target)
        assert hit["z"] == pytest.approx(
            z_of(TARGET, TARGET, [*FILLER, TARGET]), rel=1e-3,
        )

    async def test_old_apt_record_bands_by_z_not_age(self, env):
        # Under band_on: score a 200-day-old exact match decays to weak; the
        # relative gate bands on relevance alone, so it is still strong.
        needs_vec()
        memory, _, channel, _, db = await env()
        target = await seed_corpus(db, channel.id, target_age_days=200)
        pool = await memory._retrieve_pool(TARGET)
        hit = next(c for c in pool.ranked if c["id"] == target)
        assert hit["score"] < 0.01
        assert hit["band"] == "strong"

    async def test_strong_record_is_admitted_without_a_label(self, env):
        needs_vec()
        memory, _, channel, conv, db = await env()
        target = await seed_corpus(db, channel.id)
        await ask(memory, channel, conv, TARGET)
        # The whole line, from its start: an id-bearing age and no label.
        lines = recalled(conv).splitlines()
        assert f"(now, memory {target}) {TARGET}" in lines
        assert not any(
            line.startswith("[") and TARGET in line for line in lines
        )

    async def test_null_query_admits_nothing(self, env):
        # The whole point of the gate: an off-topic message adds nothing.
        needs_vec()
        memory, _, channel, conv, db = await env()
        await seed_corpus(db, channel.id)
        corpus = [*FILLER, TARGET]
        assert max(z_of(NULL_QUERY, t, corpus) for t in corpus) < DEFAULT_BAND_MODERATE_Z
        await ask(memory, channel, conv, NULL_QUERY)
        assert context_messages(conv) == []
        rows = await candidate_rows(db)
        assert rows
        assert {r["outcome"] for r in rows.values()} <= {"dropped_weak", "cut"}

    async def test_one_shared_keyword_does_not_clear_the_gate(self, env):
        # "orchid" matches one filler record by keyword (the vector search
        # returns it too), but that one shared word doesn't make it stand out
        # from the memory on cosine. The FTS-only case is covered under
        # band_on: similarity in test_memory_hybrid_retrieval.py.
        needs_vec()
        memory, _, channel, conv, db = await env(config(k=1))
        await seed_corpus(db, channel.id)
        query = "where did I put the orchid"
        orchid = next(t for t in FILLER if t.startswith("orchid"))
        assert z_of(query, orchid, [*FILLER, TARGET]) < DEFAULT_BAND_MODERATE_Z
        await ask(memory, channel, conv, query)
        assert orchid not in recalled(conv)

    async def test_unembedded_keyword_hit_is_not_admitted(self, env):
        # No stored vector, so no z: search_memory still finds it, passive
        # retrieval doesn't admit it.
        needs_vec()
        memory, _, channel, conv, db = await env()
        await seed_corpus(db, channel.id)
        loose = "the gazebo latch is rusted"
        loose_id = await seed_memory(db, channel.id, loose, embedded=False)
        await ask(memory, channel, conv, "is the gazebo latch fixed")
        rows = await candidate_rows(db)
        # The keyword search found it; with no measured cosine it gets no z
        # (it isn't banded on the stand-in similarity) and is dropped as weak.
        assert rows[loose_id]["source"] == "fts"
        assert rows[loose_id]["z"] is None
        assert rows[loose_id]["band"] == "weak"
        # Since #705 an unmeasured hit ranks below every measured one, so at
        # k=1 it is cut by rank before the gate ever judges it. Either way it
        # is not admitted; `cut` is the more accurate record, because the
        # reason is its place in the pool and not its band.
        assert rows[loose_id]["outcome"] == "cut"
        assert loose not in recalled(conv)

    async def test_z_is_logged_on_candidate_rows(self, env):
        needs_vec()
        memory, _, channel, conv, db = await env()
        target = await seed_corpus(db, channel.id)
        await ask(memory, channel, conv, TARGET)
        rows = await candidate_rows(db)
        assert rows[target]["z"] == pytest.approx(
            z_of(TARGET, TARGET, [*FILLER, TARGET]), rel=1e-3,
        )


class TestSmallStoreFallback:
    async def test_few_records_band_on_similarity_logged_once(self, env, caplog):
        needs_vec()
        memory, _, channel, _, db = await env()
        await seed_memory(db, channel.id, TARGET)
        assert 1 < RELATIVE_MIN_RECORDS
        with caplog.at_level(logging.WARNING):
            [c], _ = await memory.retrieve(TARGET)
            await memory.retrieve(TARGET)
        assert c["band"] == "strong"
        assert c["z"] is None
        warnings = [r for r in caplog.records if "band_on: similarity" in r.getMessage()]
        assert len(warnings) == 1
        assert str(RELATIVE_MIN_RECORDS) in warnings[0].getMessage()


class TestNoWeakFallback:
    async def test_only_weak_candidates_admit_nothing_under_score(self, env):
        # Before #409 a weak record was admitted when nothing else matched.
        # Vector path only: the FTS5-degraded path admits its top k.
        needs_vec()
        memory, _, channel, conv, db = await env(config(band_on="score"))
        await seed_memory(db, channel.id, TARGET, created_at=time.time() - 60 * DAY)
        await ask(memory, channel, conv, TARGET)
        assert context_messages(conv) == []


class TestLineFormat:
    async def test_moderate_z_record_is_marked_marginal(self, env):
        # R4 under the default relative mode: a record between the moderate
        # and strong z thresholds is admitted as [marginal].
        needs_vec()
        memory, _, channel, conv, db = await env()
        target = await seed_corpus(db, channel.id)
        query = "kestrel's ESP32 weather station"
        z = z_of(query, TARGET, [*FILLER, TARGET])
        assert DEFAULT_BAND_MODERATE_Z <= z < DEFAULT_BAND_STRONG_Z
        await ask(memory, channel, conv, query)
        assert f"[marginal] (now, memory {target}) {TARGET}" in (
            recalled(conv).splitlines()
        )

    async def test_moderate_record_is_marked_marginal(self, env):
        needs_vec()
        memory, _, channel, conv, db = await env(
            config(band_on="similarity", bands={"strong": 2.0, "moderate": 0.0})
        )
        target = await seed_memory(db, channel.id, TARGET)
        await ask(memory, channel, conv, TARGET)
        assert f"[marginal] (now, memory {target}) {TARGET}" in (
            recalled(conv).splitlines()
        )

    async def test_degraded_path_admits_top_k_marked_marginal(self, env):
        memory, embedder, channel, conv, db = await env()
        target = await seed_memory(
            db, channel.id, TARGET, created_at=time.time() - 60 * DAY,
            embedded=_has_vec(),
        )
        embedder.fail = True
        await ask(memory, channel, conv, "weather station wifi")
        assert f"[marginal] (60d, memory {target}) {TARGET}" in (
            recalled(conv).splitlines()
        )


def _has_vec() -> bool:
    """Whether this build can write vec rows."""
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        return False
    return True


def test_min_records_keeps_the_strong_threshold_reachable():
    # A population z over N records can't exceed sqrt(N - 1); below the
    # minimum, nothing could ever band strong.
    assert (RELATIVE_MIN_RECORDS - 1) ** 0.5 > DEFAULT_BAND_STRONG_Z
