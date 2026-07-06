"""Tests for the MemoryPlugin schema (Phase 1a WP1a.4, bootstrap-mapping §4.11–4.12).

Tables are created on the persistence connection: memory,
consolidation_watermark, embedding_meta, memory_fts (external-content FTS5
with content-sync triggers), and memory_vec when the sqlite-vec extension
loads. The plugin starts degraded — vector retrieval disabled, FTS5 only —
when the extension is unavailable, and refuses to mix encoders silently.
"""

import logging
import time
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db


EMBED_CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "embedding": {
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": 4,
        },
    }
}


async def build_memory_plugin(config: dict | None = None):
    """Register persistence + llm + memory on a fresh pm with an in-memory DB.

    Returns (memory_plugin, db). The LLM clients are never started — schema
    tests only need the config-derived dimensions.
    """
    config = config if config is not None else EMBED_CONFIG
    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=config)
    embedding_cfg = config.get("llm", {}).get("embedding")
    if embedding_cfg is not None:
        llm.embedding_dimensions = embedding_cfg["dimensions"]

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)
    return memory, db


async def _table_names(db) -> set[str]:
    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
    ) as cursor:
        rows = await cursor.fetchall()
    return {r[0] for r in rows}


class TestSchemaCreation:
    async def test_tables_and_triggers_exist_after_on_start(self):
        memory, db = await build_memory_plugin()
        names = await _table_names(db)
        for expected in (
            "memory",
            "consolidation_watermark",
            "embedding_meta",
            "memory_fts",
            "memory_ai",
            "memory_au",
        ):
            assert expected in names, f"missing {expected}"
        await db.close()

    async def test_vec_table_created_when_extension_loads(self):
        memory, db = await build_memory_plugin()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable in this environment")
        names = await _table_names(db)
        assert "memory_vec" in names
        await db.close()

    async def test_embedding_meta_recorded(self):
        memory, db = await build_memory_plugin()
        async with db.execute("SELECT encoder, dimensions FROM embedding_meta") as cursor:
            row = await cursor.fetchone()
        assert row == ("test-embedder", 4)
        await db.close()


class TestFtsSync:
    async def _insert_memory(self, db, summary: str) -> int:
        cursor = await db.execute(
            "INSERT INTO memory (channel_id, created_at, summary, importance, "
            "msg_id_start, msg_id_end) VALUES (?, ?, ?, ?, ?, ?)",
            ("irc:#test", time.time(), summary, 0.5, 1, 2),
        )
        await db.commit()
        return cursor.lastrowid

    async def test_fts_row_appears_on_insert(self):
        memory, db = await build_memory_plugin()
        rowid = await self._insert_memory(db, "kestrel fixed the wifi dropout")
        async with db.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'wifi'"
        ) as cursor:
            rows = await cursor.fetchall()
        assert rows == [(rowid,)]
        await db.close()

    async def test_fts_follows_summary_update(self):
        memory, db = await build_memory_plugin()
        rowid = await self._insert_memory(db, "original text about herons")
        await db.execute(
            "UPDATE memory SET summary = ? WHERE id = ?",
            ("replacement text about magpies", rowid),
        )
        await db.commit()
        async with db.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'herons'"
        ) as cursor:
            old_hits = await cursor.fetchall()
        async with db.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'magpies'"
        ) as cursor:
            new_hits = await cursor.fetchall()
        assert old_hits == []
        assert new_hits == [(rowid,)]
        await db.close()


class TestDegradedStart:
    async def test_starts_degraded_when_vec_extension_unavailable(self, monkeypatch, caplog):
        """No sqlite-vec → plugin starts, vector retrieval disabled, one WARNING."""
        async def no_extension(self, db):
            return False

        monkeypatch.setattr(MemoryPlugin, "_load_vec_extension", no_extension)
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            memory, db = await build_memory_plugin()
        assert memory._vec_available is False
        names = await _table_names(db)
        assert "memory_vec" not in names
        assert "memory" in names  # the rest of the schema still lands
        assert any("FTS5" in rec.message or "fts" in rec.message.lower()
                   for rec in caplog.records)
        await db.close()

    async def test_starts_without_embedding_role(self):
        """No llm.embedding config → no vec table, no embedding_meta row."""
        config = {"llm": {"main": {"base_url": "http://localhost:8080", "model": "chat"}}}
        memory, db = await build_memory_plugin(config)
        assert memory._vec_available is False
        async with db.execute("SELECT count(*) FROM embedding_meta") as cursor:
            count = (await cursor.fetchone())[0]
        assert count == 0
        await db.close()


class TestEncoderMismatch:
    async def test_encoder_mismatch_logs_error_and_disables_embedding(self, caplog):
        """A stored encoder differing from config is an ERROR, not silent mixing."""
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        # Pre-seed embedding_meta with a different encoder.
        await db.execute(
            "CREATE TABLE IF NOT EXISTS embedding_meta "
            "(encoder TEXT NOT NULL, dimensions INTEGER NOT NULL)"
        )
        await db.execute(
            "INSERT INTO embedding_meta (encoder, dimensions) VALUES (?, ?)",
            ("old-encoder", 8),
        )
        await db.commit()

        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        llm = LLMPlugin()
        pm.register(llm, name="llm")
        await llm.on_init(pm=pm, config=EMBED_CONFIG)
        llm.embedding_dimensions = 4

        memory = MemoryPlugin()
        pm.register(memory, name="memory")
        await memory.on_init(pm=pm, config=EMBED_CONFIG)
        with caplog.at_level(logging.ERROR, logger="corvidae.memory"):
            await memory.on_start(config=EMBED_CONFIG)

        assert memory._encoder_mismatch is True
        assert any("re-embed" in rec.message for rec in caplog.records)
        await db.close()
