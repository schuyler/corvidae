"""#1186 R1/R2: the shared keyword-query module ``corvidae_cognition.fts``.

Every cognition search site builds its predicate from ``keyword_query`` and
``keyword_where``; the three FTS tables share one tokenizer clause and one
creation-failure message. These tests pin the data-example table of the
design (pieces, dedupe, the short-term rule) and the failure message.
"""

import importlib
import logging
import sqlite3

import aiosqlite
import pytest


class _LazyFts:
    """The module under test, imported on first use so that a missing module
    fails each test (not the collection of the whole suite)."""

    def __getattr__(self, name):
        return getattr(importlib.import_module("corvidae_cognition.fts"), name)


fts = _LazyFts()


def test_constants_pin_the_trigram_tokenizer():
    """All three tables take one tokenizer clause: trigram with diacritics
    removed (unicode61's default, so `cafe` still finds `café`)."""
    assert fts.TRIGRAM_TOKENIZE == "trigram remove_diacritics 1"
    assert fts.TOKENIZE_CLAUSE == "tokenize='trigram remove_diacritics 1'"
    assert fts.MIN_MATCH_CHARS == 3


# ---------------------------------------------------------------------------
# search_terms: maximal runs of letters, marks and numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,terms", [
    # Combining marks (category M) stay inside the term; \w+ would split these.
    ("हिन्दी भाषा", ["हिन्दी", "भाषा"]),
    ("ภาษาไทย ดี", ["ภาษาไทย", "ดี"]),
    # Punctuation separates terms, so no quote or percent can reach a query.
    ("apples, pie", ["apples", "pie"]),
    ('foo"bar 100%', ["foo", "bar", "100"]),
    ("Привет, мир", ["Привет", "мир"]),
    ("?!", []),
])
def test_search_terms_split_by_unicode_category(text, terms):
    assert fts.search_terms(text) == terms


# ---------------------------------------------------------------------------
# keyword_query: the design's data-example table
# ---------------------------------------------------------------------------


def test_unspaced_sentence_is_searched_as_overlapping_pieces_and_whole():
    """D1/D4: a 5-character unspaced term gives its 3-character pieces,
    left to right, then the whole term; no LIKE when pieces exist (D2)."""
    q = fts.keyword_query("我也喜欢吃")
    assert q.terms == ("我也喜欢吃",)
    assert q.pieces == ("我也喜", "也喜欢", "喜欢吃", "我也喜欢吃")
    assert q.match == '"我也喜" OR "也喜欢" OR "喜欢吃" OR "我也喜欢吃"'
    assert q.like == ()


def test_english_query_pieces_then_whole_terms_of_four_plus():
    """Pieces of every 3+ term first (a 3-character term is its own piece),
    then every 4+ term whole (D4); the 2-character `is` is not searched."""
    q = fts.keyword_query("when is the meeting")
    assert q.terms == ("when", "is", "the", "meeting")
    assert q.pieces == (
        "whe", "hen", "the", "mee", "eet", "eti", "tin", "ing", "when", "meeting",
    )
    assert q.match is not None and q.match.count(" OR ") == 9
    assert q.like == ()


def test_only_short_terms_go_to_like():
    """D2: with no term of 3+ characters the query is a LIKE scan."""
    q = fts.keyword_query("苹果")
    assert q.terms == ("苹果",)
    assert q.pieces == ()
    assert q.match is None
    assert q.like == ("%苹果%",)


def test_short_term_beside_a_long_one_is_not_searched():
    """D2: the LIKE scan runs only when the query has no 3+ term."""
    q = fts.keyword_query("苹果 喜欢吃")
    assert q.terms == ("苹果", "喜欢吃")
    assert q.pieces == ("喜欢吃",)
    assert q.match == '"喜欢吃"'
    assert q.like == ()


def test_no_searchable_terms():
    """Punctuation only: no terms, no match, no LIKE (the callers' guard)."""
    q = fts.keyword_query("?!")
    assert q.terms == () and q.pieces == () and q.match is None and q.like == ()


def test_pieces_are_deduplicated_by_casefold_first_occurrence_kept():
    """A repeated piece would count once per repetition in bm25, so the
    commonest pieces would gain the most weight; `The`/`the` are one token."""
    assert fts.keyword_query("The the THE").pieces == ("The",)
    q = fts.keyword_query("running jumping")
    assert q.pieces == (
        "run", "unn", "nni", "nin", "ing", "jum", "ump", "mpi", "pin",
        "running", "jumping",
    )


def test_cap_counts_terms_per_blank_line_separated_paragraph():
    """cap_per_paragraph keeps the first N terms of each paragraph (a
    coalesced burst keeps every message's terms); no cap by default."""
    para_a = " ".join(f"alpha{i}" for i in range(25))
    para_b = " ".join(f"bravo{i}" for i in range(25))
    capped = fts.keyword_query(f"{para_a}\n\n{para_b}", cap_per_paragraph=20)
    assert len(capped.terms) == 40
    assert capped.terms[19] == "alpha19" and capped.terms[20] == "bravo0"
    assert "alpha20" not in capped.terms
    assert len(fts.keyword_query(f"{para_a}\n\n{para_b}").terms) == 50


def test_no_stopword_list_a_common_words_query_still_has_a_match():
    """R1: nothing is dropped for being a common word."""
    q = fts.keyword_query("what about those")
    assert q.terms == ("what", "about", "those")
    assert q.match is not None and '"what"' in q.match and '"about"' in q.match


# ---------------------------------------------------------------------------
# keyword_where
# ---------------------------------------------------------------------------


def test_keyword_where_match_form():
    q = fts.keyword_query("喜欢吃")
    assert fts.keyword_where(q, "memory_fts", "m.summary") == (
        "memory_fts MATCH ?", [q.match],
    )


def test_keyword_where_like_form():
    """With no MATCH the predicate is a LIKE over the caller's column."""
    q = fts.keyword_query("苹果 梨")
    where, params = fts.keyword_where(q, "memory_fts", "m.summary")
    assert params == ["%苹果%", "%梨%"]
    assert where.startswith("(") and where.endswith(")")
    assert where.count("m.summary LIKE ?") == 2


# ---------------------------------------------------------------------------
# create_fts_table: the SQLite floor message
# ---------------------------------------------------------------------------


async def test_create_fts_table_creates_the_table():
    async with aiosqlite.connect(":memory:") as db:
        await fts.create_fts_table(
            db, "demo_fts",
            "CREATE VIRTUAL TABLE IF NOT EXISTS demo_fts USING fts5("
            f"body, {fts.TOKENIZE_CLAUSE})",
        )
        async with db.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'demo_fts'"
        ) as cursor:
            (ddl,) = await cursor.fetchone()
    assert "trigram" in ddl


async def test_create_fts_table_failure_names_table_version_and_floor(caplog):
    """An unknown tokenizer stands in for SQLite < 3.45 (which CI cannot
    run): the ERROR names the table, the SQLite version, the exception type
    and the remedy, and the OperationalError is re-raised."""
    async with aiosqlite.connect(":memory:") as db:
        with caplog.at_level(logging.ERROR):
            with pytest.raises(sqlite3.OperationalError):
                await fts.create_fts_table(
                    db, "demo_fts",
                    "CREATE VIRTUAL TABLE IF NOT EXISTS demo_fts USING fts5("
                    "body, tokenize='nosuch')",
                )
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "no ERROR logged"
    text = errors[0].getMessage()
    assert "demo_fts" in text
    assert sqlite3.sqlite_version in text
    assert "OperationalError" in text
    assert "3.45" in text
