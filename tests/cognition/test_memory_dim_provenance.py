"""Tests for stored-dimension provenance (GitHub issue #597).

The memory vector cache previously had no way to verify that the vectors
actually stored match the dimension the table declares: the encoder-identity
check compares config against meta, and vec0 validates query dimension, not
insert dimension. A differently-widthed encoder (a misnamed model serving the
wrong width) would silently write a wrong-width vector into the
FLOAT[declared] table, and nothing in the stack could confirm or refute a
drift without a manual BLOB read. These tests guard the two detection fixes:

  - Part 2: the write path validates the stored width and refuses a drift,
    leaving the row backfillable and failing loud;
  - Part 1: the stored width is recorded per row as a queryable fact, so a
    drift is a row-level query, not something inferred from a BLOB byte length.
"""

import logging
import time

import aiosqlite
import pytest

from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae_cognition.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db

from helpers import stub_llm_config

DIMS = 4  # small for speed; the width is what matters, not the value

EMBED_CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "embedding": {
            "kind": "embedding",
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": DIMS,
        },
    }
}


class _StubEmbedder:
    """Stub embedder; a text substring can be mapped to a forced vector."""

    def __init__(self) -> None:
        self.fixed: dict[str, list[float]] = {}

    def request_model(self, extra_body=None):
        return (extra_body or {}).get("model", "stub-model")

    async def chat(self, messages, tools=None, extra_body=None):
        return {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}

    async def embed(self, texts, kind=None):
        out = []
        for t in texts:
            for key, vec in self.fixed.items():
                if key in t:
                    out.append(list(vec))
                    break
            else:
                out.append([1.0, 0.0, 0.0, 0.0])
        return out


async def build_env(config=None):
    """Register persistence + llm + memory on a fresh in-memory DB.

    Returns (memory_plugin, db, stub_embedder). The LLM clients are never
    started — only the embedding client is stubbed.
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
    stub = _StubEmbedder()
    llm._clients["embedding"] = stub
    stub_llm_config(llm, config.get("llm"))

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)
    return memory, db, stub


def _requires_vec(memory):
    if not memory._vec_available:
        pytest.skip("sqlite-vec unavailable in this environment")


class TestStoredDimProvenance:
    async def test_insert_records_embedded_dim(self):
        """Part 1: the stored width is recorded per row as a queryable fact."""
        memory, db, stub = await build_env()
        _requires_vec(memory)
        mid = await memory.write_record(
            channel_id="irc:#test", summary="a summary",
            created_at=time.time(), importance=0.5,
            msg_id_start=1, msg_id_end=2,
        )
        async with db.execute(
            "SELECT embedded, embedded_dim FROM memory WHERE id = ?", (mid,)
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        embedded, embedded_dim = row
        assert embedded == 1
        assert embedded_dim == DIMS, "the stored dim must be recorded per row"
        # The vec row exists, and the stored-vs-declared drift query is clean.
        async with db.execute(
            "SELECT count(*) FROM memory_vec WHERE memory_id = ?", (mid,)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
        async with db.execute(
            "SELECT count(*) FROM memory m "
            "JOIN (SELECT dimensions FROM embedding_meta LIMIT 1) AS d "
            "ON m.embedded_dim IS NOT NULL AND m.embedded_dim <> d.dimensions"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0, (
                "a correctly-widthed insert must not register as drift"
            )
        await db.close()

    async def test_insert_refuses_wrong_dim_embedding(self, caplog):
        """Part 2: the write path validates the stored width and refuses a drift."""
        memory, db, stub = await build_env()
        _requires_vec(memory)
        # Force a wrong-width vector (2*DIMS) for this summary.
        stub.fixed["drift summary"] = [0.1] * (DIMS * 2)
        with caplog.at_level(logging.ERROR, logger="corvidae.memory"):
            mid = await memory.write_record(
                channel_id="irc:#test", summary="drift summary",
                created_at=time.time(), importance=0.5,
                msg_id_start=1, msg_id_end=2,
            )
        # The row exists but is left unembedded (backfillable).
        async with db.execute(
            "SELECT embedded, embedded_dim FROM memory WHERE id = ?", (mid,)
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        embedded, embedded_dim = row
        assert embedded == 0, "a drift row must be left unembedded (backfillable)"
        assert embedded_dim is None, "no stored dim is recorded for a refused vector"
        # No vec row was written.
        async with db.execute(
            "SELECT count(*) FROM memory_vec WHERE memory_id = ?", (mid,)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0, (
                "a wrong-width vector must not be inserted"
            )
        # It failed loud.
        assert any(
            rec.levelno == logging.ERROR and "dim" in rec.message.lower()
            for rec in caplog.records
        ), "a stored-dimension drift must fail loud"
        await db.close()
