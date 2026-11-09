"""Per-candidate retrieval log (issue #407; PRD §3.7 Memory 4).

Every passive retrieval (``before_agent_turn``) writes one
``retrieval_candidate`` row per logged candidate, keyed to its
``retrieval_log`` row: memory_id, similarity, score, band, rank, outcome.
Outcomes: admitted | dropped_weak | dropped_budget | dropped_dup | cut |
excluded_in_window. The rows hold ids and numbers only, never text.

Only passive retrieval logs: a direct ``retrieve()`` call writes nothing.
"""

import logging
import sqlite3

import pytest
from corvidae_cognition.memory import FTS_SIMILARITY_PROXY

from tests.cognition.test_memory_retrieval import (
    EMBED_CONFIG,
    build_retrieval_env,
    context_messages,
)
from tests.cognition.test_memory_retrieval import ask as ask_plain
from tests.cognition.test_memory_window_exclusion import (
    WINDOW_START,
    _sqlite_vec,
    ask,
    fill_window,
    seed,
)

QUERY = "where is the rain gauge"


def needs_vec() -> None:
    """Skip a test that only means something on the vector path."""
    if _sqlite_vec() is None:
        pytest.skip("sqlite-vec unavailable; this test exercises the vec path")


def config_with_k(k: int) -> dict:
    """The stub embedding config with retrieval k overridden."""
    return {**EMBED_CONFIG, "memory": {"retrieval": {"k": k}}}


async def retrieval_ids(db) -> list[int]:
    """Every retrieval_log id, oldest first."""
    async with db.execute("SELECT id FROM retrieval_log ORDER BY id") as cursor:
        return [row[0] for row in await cursor.fetchall()]


async def candidate_rows(db, retrieval_id: int | None = None) -> dict[int, dict]:
    """The retrieval_candidate rows (of one retrieval, or all), keyed by memory_id."""
    sql = (
        "SELECT retrieval_id, memory_id, similarity, score, band, rank, outcome "
        "FROM retrieval_candidate"
    )
    params: tuple = ()
    if retrieval_id is not None:
        sql += " WHERE retrieval_id = ?"
        params = (retrieval_id,)
    async with db.execute(sql, params) as cursor:
        rows = await cursor.fetchall()
    return {
        row[1]: {
            "retrieval_id": row[0], "similarity": row[2], "score": row[3],
            "band": row[4], "rank": row[5], "outcome": row[6],
        }
        for row in rows
    }


class TestCandidateRows:
    async def test_three_candidates_write_three_rows_matching_retrieve(self):
        # L1: a retrieval with exactly 3 candidates writes 3 rows keyed to
        # its retrieval_log id, whose values equal what retrieve() computes
        # for the same query (floats approx: the score depends on time).
        memory, embedder, channel, conv, db = await build_retrieval_env()
        for summary in (
            "the rain gauge is on the north fence",
            "rain gauge wiring notes for the shed",
            "the greenhouse humidity sensor reads four percent high",
        ):
            await seed(db, channel.id, summary, 1, 2)
        await ask_plain(memory, channel, conv, QUERY)

        (retrieval_id,) = await retrieval_ids(db)
        rows = await candidate_rows(db, retrieval_id)
        expected, degraded = await memory.retrieve(QUERY)
        assert len(expected) == 3
        assert set(rows) == {c["id"] for c in expected}
        for rank, cand in enumerate(expected, start=1):
            row = rows[cand["id"]]
            assert row["similarity"] == pytest.approx(cand["similarity"], rel=1e-6)
            assert row["score"] == pytest.approx(cand["score"], rel=1e-6)
            assert row["band"] == cand["band"]
            assert row["rank"] == rank
        await db.close()


class TestOutcomes:
    async def test_admitted_candidate_reads_admitted(self):
        # L2 admitted: the candidate that entered the window as CONTEXT.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        mem = await seed(db, channel.id, QUERY, 1, 2)
        await ask_plain(memory, channel, conv, QUERY)
        assert len(context_messages(conv)) == 1
        (retrieval_id,) = await retrieval_ids(db)
        rows = await candidate_rows(db, retrieval_id)
        assert rows[mem]["outcome"] == "admitted"
        await db.close()

    async def test_weak_candidate_beside_non_weak_reads_dropped_weak(self):
        # L2 dropped_weak: a weak-banded candidate is dropped when a
        # non-weak one exists. Needs real cosine similarity to be weak.
        needs_vec()
        memory, embedder, channel, conv, db = await build_retrieval_env()
        strong = await seed(db, channel.id, QUERY, 1, 2)
        weak = await seed(db, channel.id, "zebra quartz midnight violin harvest", 3, 4)
        await ask_plain(memory, channel, conv, QUERY)
        (retrieval_id,) = await retrieval_ids(db)
        rows = await candidate_rows(db, retrieval_id)
        assert rows[weak]["band"] == "weak"
        assert rows[weak]["outcome"] == "dropped_weak"
        assert rows[strong]["outcome"] == "admitted"
        await db.close()

    async def test_over_budget_candidate_reads_dropped_budget(self):
        # L2 dropped_budget: a selected candidate the funnel's memory budget
        # cannot hold (one token) is neither admitted nor a duplicate.
        config = {**EMBED_CONFIG, "funnel": {"budgets": {"memory": 1}}}
        memory, embedder, channel, conv, db = await build_retrieval_env(config)
        mem = await seed(db, channel.id, QUERY, 1, 2)
        await ask_plain(memory, channel, conv, QUERY)
        assert context_messages(conv) == []
        (retrieval_id,) = await retrieval_ids(db)
        rows = await candidate_rows(db, retrieval_id)
        assert rows[mem]["outcome"] == "dropped_budget"
        await db.close()

    async def test_repeat_retrieval_reads_dropped_dup(self):
        # L2 dropped_dup: the second identical retrieval selects a line the
        # window's CONTEXT already holds, so the funnel dedupes it.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        # Range below the first CONTEXT rowid (1): once that CONTEXT is in
        # the window, #634's exclusion would otherwise hide the record.
        mem = await seed(db, channel.id, QUERY, 0, 0)
        await ask_plain(memory, channel, conv, QUERY)
        await ask_plain(memory, channel, conv, QUERY)
        first, second = await retrieval_ids(db)
        assert (await candidate_rows(db, first))[mem]["outcome"] == "admitted"
        assert (await candidate_rows(db, second))[mem]["outcome"] == "dropped_dup"
        await db.close()


class TestWindowExclusionRows:
    async def test_in_window_record_outscoring_kth_gets_excluded_row(self):
        # L3 (vec): an in-window record that would have placed in the top k
        # (k=2) gets exactly one excluded_in_window row; the returned
        # candidates are unaffected by it.
        needs_vec()
        memory, embedder, channel, conv, db = await build_retrieval_env(config_with_k(2))
        in_window = await seed(
            db, channel.id, QUERY, WINDOW_START + 1, WINDOW_START + 3,
        )
        out_a = await seed(db, channel.id, "rain gauge note one about wiring", 10, 11)
        out_b = await seed(db, channel.id, "rain gauge note two about wiring", 12, 13)
        fill_window(conv)
        await ask(memory, channel, conv, QUERY, WINDOW_START + 5)

        (retrieval_id,) = await retrieval_ids(db)
        rows = await candidate_rows(db, retrieval_id)
        assert rows[in_window]["outcome"] == "excluded_in_window"
        assert {out_a, out_b} <= set(rows)
        async with db.execute(
            "SELECT COUNT(*) FROM retrieval_candidate "
            "WHERE retrieval_id = ? AND memory_id = ?",
            (retrieval_id, in_window),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
        await db.close()

    async def test_in_window_record_below_top_k_gets_no_row(self):
        # L3 (vec): an in-window record that would not have made the top k
        # (k=2, three better out-of-window records) is not logged at all.
        needs_vec()
        memory, embedder, channel, conv, db = await build_retrieval_env(config_with_k(2))
        in_window = await seed(
            db, channel.id, "zebra quartz midnight violin harvest",
            WINDOW_START + 1, WINDOW_START + 3,
        )
        for n in range(3):
            await seed(db, channel.id, QUERY, 10 + n, 11 + n)
        fill_window(conv)
        await ask(memory, channel, conv, QUERY, WINDOW_START + 5)

        (retrieval_id,) = await retrieval_ids(db)
        rows = await candidate_rows(db, retrieval_id)
        assert in_window not in rows
        assert "excluded_in_window" not in {r["outcome"] for r in rows.values()}
        await db.close()

    async def test_fts_degraded_logs_exclusion_count_only(self):
        # L3 (FTS, AC addendum): with the encoder down, in-window matches
        # are only counted in retrieval_log.excluded_in_window; no
        # candidate row reads excluded_in_window.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        for n in range(2):
            await seed(
                db, channel.id, f"the rain gauge reading {n} for the shed",
                WINDOW_START + 1, WINDOW_START + 3,
            )
        out_of_window = await seed(
            db, channel.id, "the rain gauge is on the north fence", 10, 11,
        )
        embedder.fail = True
        fill_window(conv)
        await ask(memory, channel, conv, QUERY, WINDOW_START + 5)

        (retrieval_id,) = await retrieval_ids(db)
        async with db.execute(
            "SELECT degraded_to_fts, excluded_in_window FROM retrieval_log "
            "WHERE id = ?", (retrieval_id,),
        ) as cursor:
            degraded, excluded = await cursor.fetchone()
        assert degraded == 1
        assert excluded == 2
        rows = await candidate_rows(db, retrieval_id)
        assert set(rows) == {out_of_window}
        assert "excluded_in_window" not in {r["outcome"] for r in rows.values()}
        await db.close()


class TestPool:
    async def test_pool_larger_than_k_logs_every_row_ranks_past_k_cut(self):
        # L4 (vec): the whole scored pool is logged, not just the k
        # returned; ranks beyond k read cut, ranks 1..k do not.
        needs_vec()
        memory, embedder, channel, conv, db = await build_retrieval_env(config_with_k(2))
        for n in range(5):
            await seed(db, channel.id, f"{QUERY} number {n}", 10 + n, 11 + n)
        await ask_plain(memory, channel, conv, QUERY)

        (retrieval_id,) = await retrieval_ids(db)
        rows = await candidate_rows(db, retrieval_id)
        assert len(rows) == 5
        assert sorted(r["rank"] for r in rows.values()) == [1, 2, 3, 4, 5]
        for row in rows.values():
            assert (row["outcome"] == "cut") == (row["rank"] > 2), row
        await db.close()


class TestScope:
    async def test_direct_retrieve_writes_no_candidate_rows(self):
        # L7: only passive retrieval logs; calling retrieve() directly
        # (as eval_memory.py does) leaves retrieval_candidate empty.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed(db, channel.id, QUERY, 1, 2)
        candidates, _ = await memory.retrieve(QUERY)
        assert candidates
        assert await candidate_rows(db) == {}
        await db.close()


class TestFtsDegraded:
    async def test_encoder_down_logs_candidates_with_proxy_similarity(self):
        # L8: an encoder-down retrieval logs its candidates with the FTS
        # proxy similarity, and the parent row is marked degraded_to_fts.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        mem = await seed(db, channel.id, "the rain gauge is on the north fence", 1, 2)
        embedder.fail = True
        await ask_plain(memory, channel, conv, QUERY)

        (retrieval_id,) = await retrieval_ids(db)
        async with db.execute(
            "SELECT degraded_to_fts FROM retrieval_log WHERE id = ?", (retrieval_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
        rows = await candidate_rows(db, retrieval_id)
        assert set(rows) == {mem}
        assert rows[mem]["similarity"] == pytest.approx(FTS_SIMILARITY_PROXY)
        await db.close()


class TestFailSoft:
    async def test_failing_candidate_insert_keeps_turn_and_parent_row(
        self, caplog,
    ):
        # L9: when the candidate INSERT raises, the turn still admits
        # CONTEXT, the retrieval_log row is written (PRD §3.7 Memory 4),
        # and the warning names the channel id so an operator can find it.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed(db, channel.id, QUERY, 1, 2)

        def wrap(original):
            """Fail any statement that writes retrieval_candidate."""
            def guarded(sql, *args, **kwargs):
                # Sync on purpose: aiosqlite's execute() returns an object
                # usable with both await and async with.
                lowered = sql.lower()
                if "retrieval_candidate" in lowered and "insert" in lowered:
                    raise sqlite3.OperationalError("disk I/O error (test)")
                return original(sql, *args, **kwargs)
            return guarded

        db.execute = wrap(db.execute)
        db.executemany = wrap(db.executemany)
        with caplog.at_level(logging.WARNING):
            await ask_plain(memory, channel, conv, QUERY)

        assert len(context_messages(conv)) == 1
        # The retrieval_log row was committed, not left in an open transaction.
        assert not db.in_transaction
        ids = await retrieval_ids(db)
        assert len(ids) == 1
        async with db.execute(
            "SELECT admitted_count FROM retrieval_log WHERE id = ?", (ids[0],),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
        warnings = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "candidate" in r.getMessage()
        ]
        assert warnings, [r.getMessage() for r in caplog.records]
        assert any(channel.id in w for w in warnings), warnings
        await db.close()
