"""#1186 R1/R2: every cognition keyword-search site finds unspaced text.

Store ``我喜欢吃苹果``; a search for ``我也喜欢吃`` (a query that *contains*
the word 喜欢吃) and for the 2-character ``苹果`` finds it, on the FTS-only
path and, for memory, the hybrid path. One guard per site that a query of
only common words (``the and with``) still returns rows (R1: no stopword
list). The sites: passive retrieval (``_fts_candidates``, both branches),
``search_memory``, ``search_transcripts`` (markdown recall and critique have
their own files).
"""

import sqlite3
import time

import aiosqlite
import pytest

from cognition.test_memory_retrieval import (
    EMBED_CONFIG,
    build_retrieval_env,
)
from cognition.test_memory_retrieval import seed_memory as seed_retrieval_memory
from cognition.test_memory_tools import (
    build_tools_env,
    get_tools,
    seed_memory,
    seed_message_log_row,
)

TARGET = "我喜欢吃苹果"
# A query that contains the word 喜欢吃 without being the record.
CONTAINING = "我也喜欢吃"
SHORT = "苹果"
COMMON = "the and with"
DINNER = "Dinner: pasta with garlic and basil"


# ---------------------------------------------------------------------------
# The three tables are created with the trigram tokenizer
# ---------------------------------------------------------------------------


async def test_a_fresh_database_creates_trigram_fts_tables():
    """_ensure_schema creates memory_fts and message_fts with the shared
    tokenizer clause (no migration: new tables only)."""
    _memory, _tools, db = await build_tools_env()
    async with db.execute(
        "SELECT name, sql FROM sqlite_master WHERE name IN ('memory_fts', 'message_fts')"
    ) as cursor:
        ddl = dict(await cursor.fetchall())
    assert set(ddl) == {"memory_fts", "message_fts"}
    for name, sql in ddl.items():
        assert "tokenize='trigram remove_diacritics 1'" in sql, (name, sql)
    await db.close()


# ---------------------------------------------------------------------------
# Passive retrieval: _fts_candidates, both branches (degraded / hybrid)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag_window", [False, True], ids=["degraded", "hybrid"])
@pytest.mark.parametrize("query", [CONTAINING, SHORT], ids=["containing", "short"])
async def test_fts_candidates_find_unspaced_text(flag_window, query):
    """The keyword side finds the unspaced record: a query containing 喜欢吃
    through its 3-character piece, 苹果 through the LIKE scan (D2)."""
    memory, _e, channel, _conv, db = await build_retrieval_env()
    target = await seed_retrieval_memory(db, channel.id, TARGET, embedded=False)
    await seed_retrieval_memory(db, channel.id, "kestrel rain gauge", embedded=False)

    rows = await memory._fts_candidates(db, query, 8, flag_window=flag_window)

    assert [r[0] for r in rows] == [target]
    await db.close()


@pytest.mark.parametrize("flag_window", [False, True], ids=["degraded", "hybrid"])
async def test_fts_candidates_run_a_common_words_query(flag_window):
    """R1: `the and with` returns the rows sharing those words."""
    memory, _e, channel, _conv, db = await build_retrieval_env()
    dinner = await seed_retrieval_memory(db, channel.id, DINNER, embedded=False)

    rows = await memory._fts_candidates(db, COMMON, 8, flag_window=flag_window)

    assert [r[0] for r in rows] == [dinner]
    await db.close()


async def test_in_window_count_finds_unspaced_text():
    """The excluded-in-window count uses the same predicate as the candidates."""
    memory, _e, channel, _conv, db = await build_retrieval_env()
    await seed_retrieval_memory(db, channel.id, TARGET, embedded=False)

    count = await memory._fts_in_window_count(db, CONTAINING, (channel.id, 0))

    assert count == 1
    await db.close()


async def test_hybrid_pool_finds_unspaced_text_the_vector_side_cannot():
    """The hybrid path: the record has no vector, so only the FTS side of
    `_retrieve_pool` can return it, as an 'fts' hit."""
    pytest.importorskip("sqlite_vec", reason="hybrid fusion needs the vector path")
    config = {**EMBED_CONFIG, "memory": {"retrieval": {"k": 1}}}
    memory, _e, channel, _conv, db = await build_retrieval_env(config)
    await seed_retrieval_memory(db, channel.id, "kestrel rain gauge calibration")
    target = await seed_retrieval_memory(db, channel.id, TARGET, embedded=False)

    pool = await memory._retrieve_pool(CONTAINING)

    hit = next((c for c in pool.ranked if c["id"] == target), None)
    assert hit is not None and hit["source"] == "fts"
    await db.close()


# ---------------------------------------------------------------------------
# Existing install (D4, D7): old tables are kept, whole words still match
# ---------------------------------------------------------------------------

# The old unicode61 DDL, copied here: the install this guards was built by it.
OLD_MEMORY_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    summary, content='memory', content_rowid='id'
)
"""


async def test_existing_unicode61_install_is_kept_and_whole_words_still_match(tmp_path):
    """D7: no migration, the old table stays unicode61. D4: every term of 4+
    characters is also searched whole, so a whole English word still matches
    there. `yesterday`'s own pieces (yes est ste ter erd rda day) are not
    words in any row, so pieces alone would find nothing on unicode61."""
    from corvidae_cognition.memory import MEMORY_DDL

    path = str(tmp_path / "old.db")
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA journal_mode = WAL")
    raw.execute(MEMORY_DDL)
    raw.execute(OLD_MEMORY_FTS_DDL)
    rows = ["I was reading Dune yesterday evening", "Plans for the weekend hike"]
    for i, text in enumerate(rows, start=1):
        raw.execute(
            "INSERT INTO memory (channel_id, created_at, summary, importance, "
            "msg_id_start, msg_id_end, embedded) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("irc:#a", time.time(), text, 0.5, 1, 2, 0),
        )
        raw.execute("INSERT INTO memory_fts(rowid, summary) VALUES (?, ?)", (i, text))
    raw.commit()
    raw.close()

    memory, _e, _channel, _conv, db = await build_retrieval_env(
        None, db_path=path
    )
    async with db.execute("SELECT sql FROM sqlite_master WHERE name = 'memory_fts'") as c:
        (ddl,) = await c.fetchone()
    assert "trigram" not in ddl, "an existing table must not be migrated"

    found = await memory._fts_candidates(db, "yesterday", 5)

    assert [r[1] for r in found] == [rows[0]]
    await db.close()


# ---------------------------------------------------------------------------
# search_memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", [CONTAINING, SHORT], ids=["containing", "short"])
async def test_search_memory_finds_unspaced_text(query):
    """The record has no embedding, so only the keyword side can find it."""
    _memory, tools, db = await build_tools_env()
    await seed_memory(db, "irc:#a", TARGET, embedded=0)
    await seed_memory(db, "irc:#a", "kestrel rain gauge calibration", embedded=0)

    result = await get_tools(tools)["search_memory"](query)

    assert TARGET in result, result
    assert "kestrel" not in result, result
    await db.close()


async def test_search_memory_keyword_only_mode_finds_unspaced_text():
    """The keyword mode (no embedding profile) uses the same predicate."""
    _memory, tools, db = await build_tools_env({})
    await seed_memory(db, "irc:#a", TARGET, embedded=0)

    result = await get_tools(tools)["search_memory"](CONTAINING)

    assert TARGET in result, result
    await db.close()


async def test_search_memory_runs_a_common_words_query():
    """R1: `the and with` finds the dinner record."""
    _memory, tools, db = await build_tools_env()
    await seed_memory(db, "irc:#a", DINNER, embedded=0)

    result = await get_tools(tools)["search_memory"](COMMON)

    assert "pasta" in result, result
    await db.close()


async def test_search_memory_of_punctuation_only_says_no_searchable_terms():
    """The 'no searchable terms' guard stays for a query with no terms."""
    _memory, tools, db = await build_tools_env()

    result = await get_tools(tools)["search_memory"]("?! ...")

    assert "no searchable terms" in result
    await db.close()


# ---------------------------------------------------------------------------
# search_transcripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", [CONTAINING, SHORT], ids=["containing", "short"])
async def test_search_transcripts_finds_unspaced_text(query):
    _memory, tools, db = await build_tools_env()
    target = await seed_message_log_row(db, "irc:#a", "user", TARGET)
    other = await seed_message_log_row(db, "irc:#a", "user", "kestrel rain gauge")

    result = await get_tools(tools)["search_transcripts"](query)

    assert f"{target}." in result, result
    assert f"{other}." not in result, result
    await db.close()


async def test_search_transcripts_runs_a_common_words_query():
    """R1: `the and with` finds the dinner message."""
    _memory, tools, db = await build_tools_env()
    dinner = await seed_message_log_row(db, "irc:#a", "user", DINNER)

    result = await get_tools(tools)["search_transcripts"](COMMON)

    assert f"{dinner}." in result, result
    await db.close()
