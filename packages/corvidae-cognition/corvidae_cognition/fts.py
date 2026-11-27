"""Keyword-search plumbing shared by every cognition FTS5 site (#1186).

All three FTS tables (``memory_fts``, ``message_fts``, ``chunks_fts``) use
the FTS5 ``trigram`` tokenizer so text written without spaces between words
(Chinese, Japanese) is found by a word it contains. Every search site builds
its predicate here, so the term splitting, the 3-character pieces and the
short-term ``LIKE`` rule are the same everywhere. Nothing in this module
filters by vocabulary: no stopword list, no language-specific rule.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Final

import aiosqlite

from corvidae.session_writer import WriteCursor

logger = logging.getLogger("corvidae.fts")

# One tokenizer for all three tables. remove_diacritics 1 keeps unicode61's
# default (`cafe` finds `café`); it needs SQLite 3.45.
TRIGRAM_TOKENIZE: Final[str] = "trigram remove_diacritics 1"
TOKENIZE_CLAUSE: Final[str] = f"tokenize='{TRIGRAM_TOKENIZE}'"
# Trigram matches nothing shorter than this; it is also the piece length (D1).
MIN_MATCH_CHARS: Final[int] = 3
# A whole term is searched beside its pieces from this length (D4).
_WHOLE_TERM_MIN_CHARS: Final[int] = 4
# Blank-line paragraph separator, for the per-paragraph term cap.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


# Zero-width non-joiner and joiner: format characters that belong inside a
# word, so they must not split it (Persian می‌خواهم is one word).
_JOINERS: Final[str] = "\u200c\u200d"


def search_terms(text: str) -> list[str]:
    """Maximal runs of characters whose Unicode category is L*, M* or N*,
    plus the zero-width joiners (U+200C, U+200D) that sit inside words in
    Persian, Indic and emoji text.

    Every other character (punctuation, whitespace, symbols) separates
    terms, so a term never holds a quote, ``%`` or ``_``.
    """
    terms: list[str] = []
    current: list[str] = []
    for char in text:
        # Letters, combining marks and numbers stay inside a term.
        if unicodedata.category(char)[0] in "LMN" or char in _JOINERS:
            current.append(char)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    return terms


@dataclass(frozen=True)
class KeywordQuery:
    """A keyword query ready for a search site.

    ``terms``: every term in text order, after the per-paragraph cap.
    ``pieces``: overlapping 3-character pieces of each term of 3+ characters
    plus each term of 4+ characters whole, first occurrence kept,
    casefold-deduplicated. ``match``: the pieces as OR'd quoted FTS5
    phrases, None when there are none. ``like``: ``%t%`` patterns, one per
    term, non-empty only when ``match`` is None (D2).
    """

    terms: tuple[str, ...]
    pieces: tuple[str, ...]
    match: str | None
    like: tuple[str, ...]


def keyword_query(text: str, *, cap_per_paragraph: int | None = None) -> KeywordQuery:
    """Split ``text`` into the terms, pieces, MATCH and LIKE a site needs.

    ``cap_per_paragraph`` keeps the first N terms of each blank-line
    separated paragraph, so a coalesced burst keeps every message's terms.
    """
    # Terms in text order, capped per paragraph.
    terms: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        found = search_terms(paragraph)
        terms.extend(found if cap_per_paragraph is None else found[:cap_per_paragraph])

    # Pass 1: every 3-character piece of every term of 3+ characters;
    # pass 2: every term of 4+ characters whole (D4).
    raw: list[str] = []
    for term in terms:
        if len(term) >= MIN_MATCH_CHARS:
            raw.extend(
                term[i : i + MIN_MATCH_CHARS]
                for i in range(len(term) - MIN_MATCH_CHARS + 1)
            )
    raw.extend(term for term in terms if len(term) >= _WHOLE_TERM_MIN_CHARS)

    # Dedupe by casefold: FTS5 counts a repeated phrase once per repetition,
    # which would up-weight the commonest pieces.
    seen: set[str] = set()
    pieces: list[str] = []
    for piece in raw:
        key = piece.casefold()
        if key not in seen:
            seen.add(key)
            pieces.append(piece)

    if pieces:
        match: str | None = " OR ".join(f'"{piece}"' for piece in pieces)
        like: tuple[str, ...] = ()
    else:
        # D2: the LIKE scan runs only when no term reaches 3 characters.
        match = None
        like = tuple(f"%{term}%" for term in terms)
    return KeywordQuery(tuple(terms), tuple(pieces), match, like)


def keyword_where(
    q: KeywordQuery, fts_table: str, like_column: str
) -> tuple[str, list[str]]:
    """The WHERE predicate and its parameters for ``q``.

    Precondition: ``q.terms`` is non-empty; every caller keeps its
    "no searchable terms" guard and returns before calling.
    """
    if q.match is not None:
        return f"{fts_table} MATCH ?", [q.match]
    return (
        "(" + " OR ".join(f"{like_column} LIKE ?" for _ in q.like) + ")",
        list(q.like),
    )


async def create_fts_table(db: aiosqlite.Connection, table: str, ddl: str) -> None:
    """Execute ``ddl`` (a CREATE ... IF NOT EXISTS) for FTS table ``table``.

    On ``sqlite3.OperationalError`` (SQLite older than 3.45 cannot make a
    trigram table with remove_diacritics) log an ERROR naming the table, the
    SQLite version and the floor, then re-raise.
    """
    try:
        await db.execute(ddl)
    except sqlite3.OperationalError as exc:
        logger.error(
            "%s: cannot create the keyword-search index on SQLite %s (%s): the "
            "trigram tokenizer with remove_diacritics needs SQLite 3.45 or "
            "newer — run corvidae on a Python linked against SQLite 3.45+ "
            "(the container image and uv's Python are)",
            table, sqlite3.sqlite_version, type(exc).__name__,
            exc_info=True,
        )
        raise


def create_fts_table_sync(cur: WriteCursor, table: str, ddl: str) -> None:
    """Execute ``ddl`` for FTS table ``table`` inside a write unit.

    Same contract as :func:`create_fts_table`: a CREATE ... IF NOT EXISTS
    that logs an ERROR naming the table and the SQLite floor, then
    re-raises, when the build cannot make a trigram table.
    """
    try:
        cur.execute(ddl)
    except sqlite3.OperationalError as exc:
        logger.error(
            "%s: cannot create the keyword-search index on SQLite %s (%s): the "
            "trigram tokenizer with remove_diacritics needs SQLite 3.45 or "
            "newer — run corvidae on a Python linked against SQLite 3.45+ "
            "(the container image and uv's Python are)",
            table, sqlite3.sqlite_version, type(exc).__name__,
            exc_info=True,
        )
        raise
