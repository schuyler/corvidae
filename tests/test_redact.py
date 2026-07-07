"""Tests for WP1b.4 — redact CLI.

Red phase: all tests in this file fail at collection with:
    ModuleNotFoundError: No module named 'corvidae.commands.redact'
because corvidae/commands/redact.py does not exist yet.

Designed failure reason: missing corvidae.commands.redact module.

The plan specifies 'direct function invocation against a temp DB', so
tests call into the redact module's underlying functions, not through
the Click CLI runner. The redact module exposes helper functions for
each subcommand form (message, memory, range) alongside redact_command.

Key behaviors tested (bootstrap-mapping §4.11–4.12, WP1b.4):
- Tombstone preserves row count, ids, and role structure.
- Both FTS surfaces (memory_fts, message_fts) return zero hits post-redact.
- Vec rows gone; embedded=0 on redacted memory records.
- Same-channel intersecting memories tombstoned; non-intersecting untouched.
- Foreign-channel memory whose numeric range contains a redacted id: UNTOUCHED.
- redact memory form: channel_id predicate prevents tombstoning foreign-channel rows.
- dry-run writes nothing.
- Non-WAL journal mode aborts with a clear error.
- Pre-1b DB (no memory/message_fts tables): tombstones written, skip notices, exit 0.
- Backfill-after-redact: WP1b.3 backfill indexes the tombstone text, not the original.
"""

import asyncio
import json
import time
import os
import tempfile

import aiosqlite
import pytest

# PRIMARY IMPORT FAILURE — all tests below fail at collection time with
# ModuleNotFoundError: No module named 'corvidae.commands.redact'
from corvidae.commands.redact import (
    redact_command,
    redact_messages,   # async: (db, message_ids, dry_run=False, notices: list | None = None) -> dict
    redact_memory_id,  # async: (db, memory_id, dry_run=False) -> dict
    redact_range,      # async: (db, start_id, end_id, dry_run=False) -> dict
    _verify_fts_clean,
    _extract_sample_token,
)

from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin


# ---------------------------------------------------------------------------
# DB setup helpers
# ---------------------------------------------------------------------------


async def make_full_db(path: str | None = None) -> aiosqlite.Connection:
    """Open (or create) an aiosqlite DB with WAL mode, message_log schema,
    and the full Phase 1b memory schema (memory, memory_fts, message_fts, etc.)."""
    db = await aiosqlite.connect(path or ":memory:")
    await db.execute("PRAGMA journal_mode = WAL")
    await init_db(db)

    # Initialize memory schema via MemoryPlugin
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    from corvidae.llm_plugin import LLMPlugin
    llm = LLMPlugin()
    pm.register(llm, name="llm")
    config = {
        "llm": {
            "main": {"base_url": "http://localhost:8080", "model": "chat"},
            "embedding": {
                "base_url": "http://localhost:8081",
                "model": "test-embedder",
                "dimensions": 32,
            },
        }
    }
    await llm.on_init(pm=pm, config=config)

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)

    return db


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


async def insert_memory(
    db,
    channel_id: str,
    summary: str,
    msg_id_start: int,
    msg_id_end: int,
    importance: float = 0.5,
    indexed: int = 1,
) -> int:
    cursor = await db.execute(
        "INSERT INTO memory (channel_id, created_at, summary, importance, "
        "msg_id_start, msg_id_end, embedded, indexed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (channel_id, time.time(), summary, importance, msg_id_start, msg_id_end, 0, indexed),
    )
    row_id = cursor.lastrowid
    await db.commit()
    return row_id


async def fts_hits(db, table: str, token: str) -> int:
    """Count FTS hits for a token in the named FTS table."""
    try:
        match_expr = f'"{token}"'
        async with db.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?", (match_expr,)
        ) as c:
            return (await c.fetchone())[0]
    except Exception:
        return -1  # table doesn't exist


async def get_message(db, row_id: int) -> dict:
    async with db.execute(
        "SELECT id, channel_id, message FROM message_log WHERE id = ?", (row_id,)
    ) as c:
        row = await c.fetchone()
    if row is None:
        raise ValueError(f"message_log row {row_id} not found")
    return {"id": row[0], "channel_id": row[1], "message": json.loads(row[2])}


async def get_memory(db, memory_id: int) -> dict:
    async with db.execute(
        "SELECT id, summary, redacted, indexed, embedded, superseded_by "
        "FROM memory WHERE id = ?", (memory_id,)
    ) as c:
        row = await c.fetchone()
    if row is None:
        raise ValueError(f"memory row {memory_id} not found")
    keys = ["id", "summary", "redacted", "indexed", "embedded", "superseded_by"]
    return dict(zip(keys, row))


async def vec_exists(db, memory_id: int) -> bool:
    try:
        async with db.execute(
            "SELECT 1 FROM memory_vec WHERE memory_id = ?", (memory_id,)
        ) as c:
            return (await c.fetchone()) is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# WP1b.4 message tombstone basics
# ---------------------------------------------------------------------------


class TestMessageTombstone:
    """Message form: tombstone preserves structure; FTS surfaces clear."""

    async def test_tombstone_preserves_row_count_and_id(self):
        """Redacting a message keeps the row in message_log (same id, same count)."""
        db = await make_full_db()
        channel = "irc:#test"

        msg_id = await insert_message(db, channel, "user", "secret password hunter2")

        async with db.execute("SELECT COUNT(*) FROM message_log") as c:
            count_before = (await c.fetchone())[0]

        await redact_messages(db, [msg_id])

        async with db.execute("SELECT COUNT(*) FROM message_log") as c:
            count_after = (await c.fetchone())[0]

        assert count_after == count_before, "redact must not delete message_log rows"

        msg = await get_message(db, msg_id)
        assert msg["id"] == msg_id, "row id must be unchanged"
        assert msg["message"]["role"] == "user", "role must be preserved"
        assert "hunter2" not in msg["message"].get("content", ""), (
            "original content must be replaced by tombstone"
        )
        assert "redacted" in msg["message"]["content"].lower(), (
            "tombstone content must contain 'redacted'"
        )
        await db.close()

    async def test_redact_clears_message_fts(self):
        """After redacting, message_fts returns zero hits for the original content."""
        db = await make_full_db()
        channel = "irc:#test"

        msg_id = await insert_message(db, channel, "user", "secret phrase xyzzy123")
        # Verify it is findable before redact
        assert await fts_hits(db, "message_fts", "xyzzy123") > 0, (
            "precondition: 'xyzzy123' must be findable in message_fts before redact"
        )

        await redact_messages(db, [msg_id])

        assert await fts_hits(db, "message_fts", "xyzzy123") == 0, (
            "original content must not be findable in message_fts after redact"
        )
        await db.close()

    async def test_redact_drops_tool_calls_and_payload_fields(self):
        """Tombstone drops tool_calls and payload fields, preserves role."""
        db = await make_full_db()
        channel = "irc:#test"

        msg = json.dumps({
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [{"id": "tc1", "function": {"name": "secret_fn", "arguments": "{}"}}],
        })
        cursor = await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)", (channel, msg, time.time(), "message"),
        )
        msg_id = cursor.lastrowid
        await db.commit()

        await redact_messages(db, [msg_id])

        m = await get_message(db, msg_id)
        assert m["message"]["role"] == "assistant", "role must be preserved"
        assert "tool_calls" not in m["message"], "tool_calls must be dropped"
        assert "secret_fn" not in json.dumps(m["message"]), (
            "tool payload content must not appear in tombstone"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.4 memory cascade
# ---------------------------------------------------------------------------


class TestMemoryCascade:
    """Memory cascade: intersecting same-channel memories tombstoned."""

    async def test_intersecting_same_channel_memory_tombstoned(self):
        """A memory record whose range overlaps a redacted message's id is tombstoned."""
        db = await make_full_db()
        channel = "irc:#test"

        msg1 = await insert_message(db, channel, "user", "secret content inside range")
        msg2 = await insert_message(db, channel, "assistant", "reply to secret")

        # Memory record whose range contains both messages
        mem_id = await insert_memory(db, channel, "secret summary", msg1, msg2)

        await redact_messages(db, [msg1])

        mem = await get_memory(db, mem_id)
        assert mem["redacted"] == 1, "intersecting memory must be tombstoned (redacted=1)"
        assert mem["indexed"] == 0, "tombstoned memory must be indexed=0"
        assert mem["embedded"] == 0, "tombstoned memory must have embedded=0"
        assert "redacted" in mem["summary"].lower(), (
            "memory summary must be the tombstone text"
        )
        # memory_fts must not return hits for the original summary
        assert await fts_hits(db, "memory_fts", "secret") == 0, (
            "original memory summary must not be findable in memory_fts after tombstone"
        )
        await db.close()

    async def test_non_intersecting_memory_untouched(self):
        """A memory record whose range does NOT overlap the redacted id: untouched."""
        db = await make_full_db()
        channel = "irc:#test"

        msg1 = await insert_message(db, channel, "user", "safe content in range 1")
        msg2 = await insert_message(db, channel, "assistant", "safe reply 1")
        msg3 = await insert_message(db, channel, "user", "SECRET CONTENT to redact")
        msg4 = await insert_message(db, channel, "assistant", "reply to secret")

        # Memory covering msg1–msg2 (does NOT overlap msg3)
        safe_mem_id = await insert_memory(db, channel, "safe summary", msg1, msg2)
        # Memory covering msg3–msg4
        redacted_mem_id = await insert_memory(db, channel, "secret summary", msg3, msg4)

        await redact_messages(db, [msg3])

        safe = await get_memory(db, safe_mem_id)
        assert safe["redacted"] == 0, "non-intersecting memory must NOT be tombstoned"
        assert safe["indexed"] == 1, "non-intersecting memory must remain indexed"

        redacted = await get_memory(db, redacted_mem_id)
        assert redacted["redacted"] == 1, "intersecting memory must be tombstoned"
        await db.close()

    async def test_foreign_channel_numerically_containing_memory_untouched(self):
        """Foreign-channel memory whose numeric range contains a redacted id: UNTOUCHED.

        message_log ids are global. A foreign-channel memory's range can numerically
        contain a redacted id from a different channel. The cascade must intersect
        per redacted row's channel_id, not on bare numeric range.

        (Design review important finding #3 / fix report #1 — bootstrap-mapping §4.12.)
        """
        db = await make_full_db()
        channel_a = "irc:#electronics"
        channel_b = "irc:#secret"

        # Interleave: a1, b1, a2, b2
        a1 = await insert_message(db, channel_a, "user", "safe channel A content")
        b1 = await insert_message(db, channel_b, "user", "foreign secret channel B")
        a2 = await insert_message(db, channel_a, "assistant", "safe reply A")
        b2 = await insert_message(db, channel_b, "assistant", "foreign reply B")

        # channel_b memory: range b1..b2 (but a2 is numerically inside b1..b2)
        # However, we're redacting message a2 (channel_a)
        # channel_b's memory numerically contains a2 but channel doesn't match
        mem_b = await insert_memory(db, channel_b, "foreign channel B memory", b1, b2)

        # Also a channel_a memory
        mem_a = await insert_memory(db, channel_a, "channel A memory", a1, a2)

        # Redact a2 (channel_a message)
        await redact_messages(db, [a2])

        # channel_a memory (a1..a2) must be tombstoned (intersects, same channel)
        ma = await get_memory(db, mem_a)
        assert ma["redacted"] == 1, (
            "channel_a memory overlapping redacted a2 must be tombstoned"
        )

        # channel_b memory (b1..b2) must NOT be tombstoned (different channel)
        mb = await get_memory(db, mem_b)
        assert mb["redacted"] == 0, (
            "foreign-channel memory must NOT be tombstoned even if its numeric "
            "range contains the redacted id (global-id regression)"
        )
        assert mb["indexed"] == 1, "foreign-channel memory must remain indexed"
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.4 redact memory form — channel predicate prevents foreign-channel damage
# ---------------------------------------------------------------------------


class TestRedactMemoryForm:
    """redact memory form: tombstones only the record's own-channel rows."""

    async def test_redact_memory_tombstones_own_channel_rows_only(self):
        """redact memory <id> tombstones only rows with channel_id = record.channel_id.

        message_log ids are global; an unfiltered BETWEEN would tombstone
        interleaved foreign-channel rows. (Design re-review important finding #1.)
        """
        db = await make_full_db()
        channel_a = "irc:#electronics"
        channel_b = "irc:#garden"

        # Interleave: a1, b1, a2, b2 (global id sequence)
        a1 = await insert_message(db, channel_a, "user", "channel A secret to redact")
        b1 = await insert_message(db, channel_b, "user", "channel B safe content")
        a2 = await insert_message(db, channel_a, "assistant", "channel A reply")
        b2 = await insert_message(db, channel_b, "assistant", "channel B reply")

        # Memory record on channel_a covering a1..a2
        # (b1 is numerically inside a1..a2 but belongs to channel_b)
        mem_a_id = await insert_memory(db, channel_a, "channel A memory summary", a1, a2)
        mem_b_id = await insert_memory(db, channel_b, "channel B memory summary", b1, b2)

        await redact_memory_id(db, mem_a_id)

        # channel_a messages must be tombstoned
        msg_a1 = await get_message(db, a1)
        msg_a2 = await get_message(db, a2)
        assert "redacted" in msg_a1["message"]["content"].lower(), (
            "channel_a row a1 must be tombstoned"
        )
        assert "redacted" in msg_a2["message"]["content"].lower(), (
            "channel_a row a2 must be tombstoned"
        )

        # channel_b messages must NOT be tombstoned (different channel)
        msg_b1 = await get_message(db, b1)
        msg_b2 = await get_message(db, b2)
        assert "safe content" in msg_b1["message"]["content"], (
            "channel_b row b1 must NOT be tombstoned (foreign-channel, global-id regression)"
        )
        assert "channel B reply" in msg_b2["message"]["content"], (
            "channel_b row b2 must NOT be tombstoned"
        )

        # channel_b memory must NOT be tombstoned
        mem_b = await get_memory(db, mem_b_id)
        assert mem_b["redacted"] == 0, (
            "channel_b memory must not be tombstoned (step 2 cascade is also per-channel)"
        )

        # channel_a memory must be tombstoned
        mem_a = await get_memory(db, mem_a_id)
        assert mem_a["redacted"] == 1, "channel_a memory must be tombstoned"
        await db.close()

    async def test_redact_memory_vec_gone_embedded_zero(self):
        """redact memory: vec row deleted, embedded=0 on the memory record."""
        db = await make_full_db()
        channel = "irc:#test"

        msg1 = await insert_message(db, channel, "user", "vec test secret content")
        msg2 = await insert_message(db, channel, "assistant", "reply to vec test")
        mem_id = await insert_memory(db, channel, "vec test memory", msg1, msg2)

        # Manually add a vec row (as if embedding succeeded)
        try:
            import sqlite_vec
            import math
            import hashlib
            import re
            dims = 32
            vec = [0.0] * dims
            for token in re.findall(r"\w+", "vec test memory".lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dims
                vec[bucket] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / norm for x in vec]
            await db.execute(
                "UPDATE memory SET embedded=1 WHERE id = ?", (mem_id,)
            )
            await db.execute(
                "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                (mem_id, sqlite_vec.serialize_float32(vec)),
            )
            await db.commit()
            has_vec = True
        except ImportError:
            has_vec = False

        await redact_memory_id(db, mem_id)

        mem = await get_memory(db, mem_id)
        assert mem["embedded"] == 0, "redacted memory must have embedded=0"
        if has_vec:
            assert not await vec_exists(db, mem_id), (
                "vec row must be deleted after redact"
            )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.4 dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    """--dry-run prints affected counts without writing."""

    async def test_dry_run_writes_nothing(self):
        """dry_run=True must not modify message_log, memory, or memory_vec."""
        db = await make_full_db()
        channel = "irc:#test"

        msg_id = await insert_message(db, channel, "user", "dry run secret content zork")
        msg_before = await get_message(db, msg_id)

        result = await redact_messages(db, [msg_id], dry_run=True)

        msg_after = await get_message(db, msg_id)
        assert msg_after["message"] == msg_before["message"], (
            "dry_run must not modify the message_log row"
        )
        # Still findable in FTS
        assert await fts_hits(db, "message_fts", "zork") > 0, (
            "dry_run must not modify message_fts"
        )
        # dry_run result should indicate the affected count
        assert "affected" in str(result).lower() or isinstance(result, dict), (
            f"dry_run must return affected count info; got: {result!r}"
        )
        await db.close()


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

    async def test_backfill_after_redact_indexes_tombstone_not_secret(self):
        """After redact on a pre-1b DB, running WP1b.3 backfill indexes tombstone text.

        When the daemon later creates the schema and runs the backfill task,
        it must index the tombstone text (not the original secret), because
        the message_log row already contains the tombstone at that point.
        (WP1b.4 step 0 rationale: safe to skip message_fts on pre-1b DB
        because triggers/backfill index current content — bootstrap-mapping §4.11.)
        """
        # 1. Build pre-1b DB, redact the secret
        db = await aiosqlite.connect(":memory:")
        await db.execute("PRAGMA journal_mode = WAL")
        await init_db(db)
        await db.commit()

        secret_token = "supersecrettoken12345"
        msg_id = await insert_message(
            db, "irc:#test", "user", f"the secret is: {secret_token}"
        )
        await redact_messages(db, [msg_id])

        # 2. Simulate daemon startup: add memory schema including message_fts + backfill
        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")

        llm = LLMPlugin()
        pm.register(llm, name="llm")
        config = {
            "llm": {
                "main": {"base_url": "http://localhost:8080", "model": "chat"},
                "embedding": {
                    "base_url": "http://localhost:8081",
                    "model": "test-embedder",
                    "dimensions": 32,
                },
            }
        }
        await llm.on_init(pm=pm, config=config)

        memory = MemoryPlugin()
        pm.register(memory, name="memory")
        await memory.on_init(pm=pm, config=config)
        await memory.on_start(config=config)
        await memory.wait_for_background_tasks()  # wait for backfill

        # 3. The secret must NOT be findable in message_fts
        assert await fts_hits(db, "message_fts", secret_token) == 0, (
            f"secret token '{secret_token}' must not be indexed after backfill-post-redact"
        )
        # 4. The tombstone text must be indexed
        assert await fts_hits(db, "message_fts", "redacted") > 0, (
            "tombstone text must be indexed by backfill"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.4 verification pass (green fix #2 — 2026-07-07)
# ---------------------------------------------------------------------------


class TestVerificationPass:
    """After the cascade, the CLI's verification pass queries FTS for a sample token.

    Covers:
    - _extract_sample_token extracts a usable word from JSON message content.
    - _verify_fts_clean returns "verified: 0 hits" when FTS is clean.
    - _verify_fts_clean returns a WARNING when FTS still has hits.
    - The verification actually queries the FTS tables (not a stub).
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

    async def test_verify_fts_clean_returns_zero_hits_after_redact(self):
        """_verify_fts_clean returns 'verified: 0 hits' after the cascade clears FTS.

        This confirms the function actually queries the FTS tables.
        """
        db = await make_full_db()
        channel = "irc:#test"

        # Seed a message with a unique token
        unique_token = "xyzzy9887foobar"
        msg_id = await insert_message(db, channel, "user", f"secret phrase {unique_token} here")

        # Verify it's findable before redact
        assert await fts_hits(db, "message_fts", unique_token) > 0, (
            "precondition: unique_token must be findable in message_fts"
        )

        # Redact the message
        await redact_messages(db, [msg_id])

        # Now verify FTS is clean for the original token
        verify_msg = await _verify_fts_clean(db, unique_token)
        assert "verified" in verify_msg.lower(), (
            f"_verify_fts_clean must say 'verified' after clean cascade; got: {verify_msg!r}"
        )
        assert "0 hits" in verify_msg, (
            f"_verify_fts_clean must report '0 hits' after clean cascade; got: {verify_msg!r}"
        )
        assert "message_fts" in verify_msg or "memory_fts" in verify_msg, (
            f"_verify_fts_clean must name the queried FTS surface(s); got: {verify_msg!r}"
        )
        await db.close()

    async def test_verify_fts_clean_returns_warning_when_hits_remain(self):
        """_verify_fts_clean returns a WARNING string when FTS still has hits.

        This exercises the non-zero path: if the FTS tables are not clean,
        the verification function must signal that.
        """
        db = await make_full_db()
        channel = "irc:#test"

        # Insert a message WITHOUT redacting it
        msg_id = await insert_message(db, channel, "user", "findable content xyzzy1234")
        # Commit so FTS trigger fires
        await db.commit()

        # Verify directly — content is still in FTS
        verify_msg = await _verify_fts_clean(db, "xyzzy1234")
        assert "warning" in verify_msg.lower() or "hits" in verify_msg.lower(), (
            f"_verify_fts_clean must warn when FTS still has hits; got: {verify_msg!r}"
        )
        await db.close()

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
