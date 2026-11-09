"""Retention prunes old retrieval_log / retrieval_candidate rows (issue #654),
and two logging fixes that ride along: no candidate row for a memory already
redacted, and outcome ``not_admitted`` when no funnel is registered."""

import time

import pytest
from corvidae_cognition.retention import run_retention_job

from tests.cognition.test_memory_retrieval import (
    EMBED_CONFIG,
    build_retrieval_env,
)
from tests.cognition.test_memory_retrieval import ask as ask_plain
from tests.cognition.test_memory_window_exclusion import seed
from tests.cognition.test_retrieval_candidates import only_retrieval

QUERY = "where is the rain gauge"
DAY = 86400.0
KEY = "memory.retention.retrieval_log_max_age_days"


def config_with_retention(retention: dict) -> dict:
    """EMBED_CONFIG with a memory.retention block."""
    return {**EMBED_CONFIG, "memory": {**EMBED_CONFIG.get("memory", {}), "retention": retention}}


async def add_retrieval(db, ts: float, memory_ids: list[int]) -> int:
    """One retrieval_log row at ``ts`` with a candidate row per memory id."""
    cursor = await db.execute(
        "INSERT INTO retrieval_log (ts, channel_id, hit_count, admitted_count) "
        "VALUES (?, 'irc:#test', ?, 0)",
        (ts, len(memory_ids)),
    )
    rid = cursor.lastrowid
    await db.executemany(
        "INSERT INTO retrieval_candidate (retrieval_id, memory_id, similarity, "
        "score, band, rank, outcome) VALUES (?, ?, 0.5, 0.5, 'moderate', ?, 'cut')",
        [(rid, mid, i + 1) for i, mid in enumerate(memory_ids)],
    )
    await db.commit()
    return rid


async def surviving(db) -> tuple[set[int], set[int]]:
    """(retrieval_log ids, retrieval_ids referenced by candidate rows)."""
    async with db.execute("SELECT id FROM retrieval_log") as cursor:
        logs = {r[0] for r in await cursor.fetchall()}
    async with db.execute("SELECT DISTINCT retrieval_id FROM retrieval_candidate") as cursor:
        cands = {r[0] for r in await cursor.fetchall()}
    return logs, cands


async def test_retention_prunes_rows_past_the_configured_age():
    """Parent and child rows older than the age go; newer ones stay."""
    memory, _e, channel, _conv, db = await build_retrieval_env(
        config_with_retention({"retrieval_log_max_age_days": 30})
    )
    mem = await seed(db, channel.id, "the rain gauge is on the north fence", 1, 2)
    now = time.time()
    old = await add_retrieval(db, now - 31 * DAY, [mem])
    new = await add_retrieval(db, now - 1 * DAY, [mem])

    await run_retention_job(memory, now=now)

    assert await surviving(db) == ({new}, {new})
    assert old not in (await surviving(db))[0]
    await db.close()


async def test_no_pruning_without_the_key():
    """Off by default: with no key, nothing is pruned however old."""
    memory, _e, channel, _conv, db = await build_retrieval_env()
    mem = await seed(db, channel.id, "the rain gauge is on the north fence", 1, 2)
    now = time.time()
    old = await add_retrieval(db, now - 3650 * DAY, [mem])

    await run_retention_job(memory, now=now)

    assert await surviving(db) == ({old}, {old})
    await db.close()


@pytest.mark.parametrize("bad", [0, -1, "thirty", True, float("inf")])
async def test_bad_max_age_fails_at_startup_naming_the_key(bad):
    """A malformed value fails at startup with the key, the value and the remedy."""
    with pytest.raises(ValueError) as excinfo:
        await build_retrieval_env(config_with_retention({"retrieval_log_max_age_days": bad}))
    msg = str(excinfo.value)
    assert KEY in msg
    assert repr(bad) in msg
    assert "agent.yaml" in msg


async def test_no_candidate_row_for_a_memory_redacted_after_scoring():
    """Redact racing a live retrieval: the memory is flagged after it was
    scored; its candidate row is not written, the retrieval_log row is."""
    memory, _e, channel, conv, db = await build_retrieval_env()
    mem = await seed(db, channel.id, QUERY, 1, 2)
    original = memory._retrieve_pool

    async def scored_then_redacted(*args, **kwargs):
        # Score as usual, then the redact CLI flags the memory.
        pool = await original(*args, **kwargs)
        await db.execute("UPDATE memory SET redacted = 1 WHERE id = ?", (mem,))
        await db.commit()
        return pool

    memory._retrieve_pool = scored_then_redacted
    await ask_plain(memory, channel, conv, QUERY)

    _rid, rows = await only_retrieval(db)
    assert mem not in rows
    await db.close()


async def test_selected_candidates_read_not_admitted_without_a_funnel():
    """Funnel not registered: a selected candidate logs not_admitted, not
    dropped_budget."""
    memory, _e, channel, conv, db = await build_retrieval_env()
    memory.pm.unregister(name="funnel")
    mem = await seed(db, channel.id, QUERY, 1, 2)

    await ask_plain(memory, channel, conv, QUERY)

    _rid, rows = await only_retrieval(db)
    assert rows[mem]["outcome"] == "not_admitted"
    await db.close()
