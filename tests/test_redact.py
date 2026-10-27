"""Tests for the redact CLI (WP1b.4): the kernel half.

These need no memory plugin. The tests that build the memory schema and
check the memory cascade are in tests/cognition/test_redact_cascade.py,
since they need the cognition extra.

Behaviours tested (bootstrap-mapping §4.11–4.12):
- Non-WAL journal mode aborts with a clear error.
- Pre-1b DB (no memory/message_fts tables): tombstones written, skip
  notices, exit 0.
- The verification helpers: sample-token extraction, and the FTS check on
  databases without the FTS tables.
- CLI error context on a missing database (the affected-count tests need a
  full database and live in the cascade file).

Tests call the redact module's functions directly against a temp DB, not
through the Click runner, except where the CLI's own output is the subject.
"""

import asyncio
import json
import time

import aiosqlite
import pytest

# The redact module's functions, called directly.
from corvidae.commands.redact import (
    redact_command,
    redact_messages,   # async: (db, message_ids, dry_run=False, notices: list | None = None) -> dict
    _verify_fts_clean,
    _extract_sample_token,
)

from corvidae.persistence import init_db


# ---------------------------------------------------------------------------
# DB setup helpers
# ---------------------------------------------------------------------------


async def insert_message(
    db, channel_id: str, role: str, content: str, ts: float | None = None
) -> int:
    msg = json.dumps({"role": role, "content": content})
    cursor = await db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, ?, ?, ?)",
        (channel_id, msg, ts or time.time(), "message"),
    )
    row_id = cursor.lastrowid
    await db.commit()
    return row_id


async def get_message(db, row_id: int) -> dict:
    async with db.execute(
        "SELECT id, channel_id, message FROM message_log WHERE id = ?", (row_id,)
    ) as c:
        row = await c.fetchone()
    if row is None:
        raise ValueError(f"message_log row {row_id} not found")
    return {"id": row[0], "channel_id": row[1], "message": json.loads(row[2])}


# ---------------------------------------------------------------------------
# WP1b.4 non-WAL journal mode
# ---------------------------------------------------------------------------


class TestNonWalJournalMode:
    """Non-WAL journal mode must abort with a clear error message."""

    async def test_non_wal_aborts_with_clear_error(self, tmp_path):
        """Redact on a non-WAL DB must raise RuntimeError or SystemExit with clear message.

        Uses a temp-file DB so the journal_mode is genuinely DELETE (the SQLite
        default for file-based databases).  In-memory DBs always report
        journal_mode = 'memory' and silently ignore PRAGMA journal_mode = WAL,
        so a non-WAL condition cannot be exercised in-memory.
        """
        db_path = str(tmp_path / "non_wal.db")
        db = await aiosqlite.connect(db_path)
        await init_db(db)
        # Do NOT set WAL — file-based SQLite defaults to DELETE journal mode.
        cur = await db.execute("PRAGMA journal_mode")
        row = await cur.fetchone()
        assert row[0].lower() != "wal", (
            f"expected non-WAL journal mode for setup; got: {row[0]!r}"
        )

        msg_id = await insert_message(db, "irc:#test", "user", "some content")

        with pytest.raises((RuntimeError, SystemExit, ValueError)) as exc_info:
            await redact_messages(db, [msg_id])

        error_text = str(exc_info.value).lower()
        assert "wal" in error_text or "journal" in error_text, (
            f"non-WAL error must mention 'wal' or 'journal'; got: {error_text!r}"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.4 pre-Phase-1b DB (schema-presence probe, step 0)
# ---------------------------------------------------------------------------


class TestPrePhase1bDb:
    """Pre-1b DB: message tombstones written, skip notices, no crash."""

    async def test_pre_1b_db_tombstones_messages_skips_absent_tables(self):
        """On a pre-1b DB (only message_log), redact_messages tombstones rows.

        Missing memory/message_fts tables must be skipped with printed notices,
        not aborted. Exit must be clean (no exception). The CLI never creates
        schema.
        """
        # Build a minimal DB: WAL mode + message_log only (no memory schema)
        db = await aiosqlite.connect(":memory:")
        await db.execute("PRAGMA journal_mode = WAL")
        await init_db(db)
        await db.commit()

        msg_id = await insert_message(db, "irc:#test", "user", "pre-1b secret content foobar")

        # Redact must succeed and tombstone the row
        notices = []
        await redact_messages(db, [msg_id], notices=notices)

        msg = await get_message(db, msg_id)
        assert "redacted" in msg["message"]["content"].lower(), (
            "message row must be tombstoned even on pre-1b DB"
        )

        # Skip notices must mention the missing tables
        notices_text = " ".join(notices).lower()
        assert "message_fts" in notices_text or "memory" in notices_text, (
            f"skip notices must mention absent tables; got: {notices!r}"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.4 verification pass (green fix #2 — 2026-07-07)
# ---------------------------------------------------------------------------


class TestVerificationPass:
    """After the cascade, the CLI's verification pass queries FTS for a sample token.

    Covers (kernel half):
    - _extract_sample_token extracts a usable word from JSON and plain text.
    - _verify_fts_clean handles databases without the FTS tables.

    The cases that run a real cascade against FTS tables are in
    tests/cognition/test_redact_cascade.py.
    """

    def test_extract_sample_token_from_json_message(self):
        """_extract_sample_token returns a word from JSON message content."""
        import json
        msg_json = json.dumps({"role": "user", "content": "secret password hunter2 zork"})
        token = _extract_sample_token(msg_json)
        assert token is not None, "must extract a token from JSON message"
        assert len(token) >= 4, "token must be at least 4 characters"
        # Should extract a real word, not 'redacted' or tombstone words
        assert token.lower() not in {"redacted", "operator"}, (
            f"must skip tombstone words; got {token!r}"
        )

    def test_extract_sample_token_from_plain_text(self):
        """_extract_sample_token works on plain text (memory summaries)."""
        token = _extract_sample_token("User discussed the solar panel installation")
        assert token is not None, "must extract a token from plain text"
        assert len(token) >= 4

    def test_extract_sample_token_returns_none_for_empty(self):
        """_extract_sample_token returns None for empty or short-word-only content."""
        assert _extract_sample_token("") is None
        assert _extract_sample_token("{}") is None

    async def test_verify_fts_clean_handles_absent_fts_tables(self):
        """_verify_fts_clean reports no-FTS-surfaces when tables are absent (pre-1b DB)."""
        db = await aiosqlite.connect(":memory:")
        await db.execute("PRAGMA journal_mode = WAL")
        await init_db(db)
        await db.commit()
        # No memory schema — message_fts and memory_fts don't exist

        verify_msg = await _verify_fts_clean(db, "anything")
        assert "no fts" in verify_msg.lower() or "no fts surfaces" in verify_msg.lower() or "no" in verify_msg.lower(), (
            f"_verify_fts_clean must handle absent FTS tables gracefully; got: {verify_msg!r}"
        )
        await db.close()


# ---------------------------------------------------------------------------
# Redact CLI error context
# ---------------------------------------------------------------------------


def test_extract_sample_token_logs_plain_text_parse_failure(caplog):
    """A non-JSON message is still tokenized, and the parse failure is logged."""
    import logging as _logging

    with caplog.at_level(_logging.DEBUG, logger="corvidae.redact"):
        token = _extract_sample_token("User discussed the solar panels")

    assert token == "User"
    assert any(
        "not JSON" in record.getMessage() for record in caplog.records
    ), "a message that is not JSON must be logged, not silently swallowed"


def test_cli_missing_db_raises_clean_error_without_creating_file(tmp_path):
    """A typo in --db names the path and does not create an empty database."""
    from click.testing import CliRunner

    db_path = tmp_path / "typo-sessions.db"
    result = CliRunner().invoke(
        redact_command, ["--db", str(db_path), "message", "1"]
    )

    assert result.exit_code != 0
    assert str(db_path) in result.output
    assert not db_path.exists(), "the CLI must not create an empty database"




# ---------------------------------------------------------------------------
# CLI forms on a database without the memory schema
# ---------------------------------------------------------------------------


async def _make_kernel_db(path: str) -> None:
    """A WAL database with only the kernel's message_log and two rows."""
    db = await aiosqlite.connect(path)
    try:
        await db.execute("PRAGMA journal_mode=WAL")
        await init_db(db)
        for text in ("first secret words here", "second secret words here"):
            await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                ("cli:local", json.dumps({"role": "user", "content": text}), time.time()),
            )
        await db.commit()
    finally:
        await db.close()


@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_range_form_reports_found_of_requested(tmp_path, dry_run):
    """The range form runs to completion and reports what it found.

    It used to raise NameError on every call: the found-vs-requested report
    read an `ids` variable that only exists in the message form.
    """
    from click.testing import CliRunner

    db_path = str(tmp_path / "sessions.db")
    asyncio.run(_make_kernel_db(db_path))
    args = ["--db", db_path] + (["--dry-run"] if dry_run else []) + ["range", "1", "2"]

    result = CliRunner().invoke(redact_command, args)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "2 of 2 messages" in result.output


@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_memory_form_without_memory_table_fails_cleanly(tmp_path, dry_run):
    """Without the memory schema the memory form exits non-zero with one
    clear message, in both modes, instead of a raw SQLite error."""
    from click.testing import CliRunner

    db_path = str(tmp_path / "sessions.db")
    asyncio.run(_make_kernel_db(db_path))
    args = ["--db", db_path] + (["--dry-run"] if dry_run else []) + ["memory", "1"]

    result = CliRunner().invoke(redact_command, args)

    assert result.exit_code != 0
    assert "memory table not present in this database" in result.output
    # A ClickException is a handled exit, not a crash with a traceback.
    assert isinstance(result.exception, SystemExit), repr(result.exception)
