"""Tests for the memory vector backfill command (#597 part 3).

Parts 1+2 (PR #612) make a stored-dimension drift detectable: the width is
recorded per row, and a wrong-width insert is refused. This is the repair
half: ``backfill_vectors`` re-embeds the rows that still need a vector —
'pending' rows (embedded=0: the embed failed at insert) and, defensively,
'drifted' rows (embedded=1 but embedded_dim differs from the declared
dimension: a wrong-width vec row left by an older sqlite-vec build). It is
crash-safe and idempotent, and it refuses to re-bake a wrong-width vector
while the encoder is still misconfigured.

Note: the current sqlite-vec (0.1.9) rejects a wrong-width *insert*, so the
normal write path cannot produce a drifted row in this environment; the
drifted leg is defensive, and its test simulates the state via a direct row
update (it cannot create the wrong-width vec row that would trigger the
replace).
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
    """Stub embedder; a text substring can be mapped to a forced vector or a failure."""

    def __init__(self) -> None:
        self.fixed: dict[str, list[float]] = {}
        self.fail: set[str] = set()

    def request_model(self, extra_body=None):
        return (extra_body or {}).get("model", "stub-model")

    async def chat(self, messages, tools=None, extra_body=None):
        return {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}

    async def embed(self, texts, kind=None):
        out = []
        for t in texts:
            if any(key in t for key in self.fail):
                raise RuntimeError("stub embed failure")
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


async def _insert_pending_row(db, summary: str) -> int:
    """Insert a memory row left unembedded (embedded=0), as a failed embed would."""
    cur = await db.execute(
        "INSERT INTO memory (channel_id, created_at, summary, importance, "
        "msg_id_start, msg_id_end, embedded) "
        "VALUES (?, ?, ?, ?, ?, ?, 0)",
        ("irc:#test", time.time(), summary, 0.5, 1, 2),
    )
    await db.commit()
    return cur.lastrowid


class TestVectorBackfill:
    async def test_backfill_repairs_pending_row(self):
        """A row left unembedded (embedded=0) gets a vec row on backfill."""
        memory, db, stub = await build_env()
        _requires_vec(memory)
        mid = await _insert_pending_row(db, "pending summary")
        summary = await memory.backfill_vectors()
        assert "error" not in summary
        assert summary["repaired"] == 1
        assert summary["pending"] == 1
        assert summary["drifted"] == 0
        async with db.execute(
            "SELECT embedded, embedded_dim FROM memory WHERE id = ?", (mid,)
        ) as cursor:
            embedded, embedded_dim = await cursor.fetchone()
        assert embedded == 1
        assert embedded_dim == DIMS
        async with db.execute(
            "SELECT count(*) FROM memory_vec WHERE memory_id = ?", (mid,)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1, "the vec row must exist"
        await db.close()

    async def test_backfill_idempotent(self):
        """A second backfill repairs nothing (the rows are already repaired)."""
        memory, db, stub = await build_env()
        _requires_vec(memory)
        await _insert_pending_row(db, "pending summary")
        first = await memory.backfill_vectors()
        assert first["repaired"] == 1
        second = await memory.backfill_vectors()
        assert second["repaired"] == 0, "a rerun must not re-repair"
        await db.close()

    async def test_backfill_refuses_still_misconfigured_encoder(self, caplog):
        """A still-misconfigured encoder (wrong width) is refused, not re-baked."""
        memory, db, stub = await build_env()
        _requires_vec(memory)
        mid = await _insert_pending_row(db, "drift summary")
        # The encoder is still serving the wrong width for this summary.
        stub.fixed["drift summary"] = [0.1] * (DIMS * 2)
        with caplog.at_level(logging.ERROR, logger="corvidae.memory"):
            summary = await memory.backfill_vectors()
        assert summary["repaired"] == 0
        assert summary["skipped"] == 1
        # The row is left unembedded (backfillable), no wrong-width vec row.
        async with db.execute(
            "SELECT embedded, embedded_dim FROM memory WHERE id = ?", (mid,)
        ) as cursor:
            embedded, embedded_dim = await cursor.fetchone()
        assert embedded == 0, "a refused backfill must leave the row unembedded"
        assert embedded_dim is None
        async with db.execute(
            "SELECT count(*) FROM memory_vec WHERE memory_id = ?", (mid,)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0, "no wrong-width vec row"
        # It failed loud.
        assert any(
            rec.levelno == logging.ERROR and "dim" in rec.message.lower()
            for rec in caplog.records
        ), "a still-misconfigured encoder must fail loud"
        await db.close()

    async def test_backfill_skips_embed_failure(self):
        """A row whose embed fails is skipped (left for the next run)."""
        memory, db, stub = await build_env()
        _requires_vec(memory)
        mid = await _insert_pending_row(db, "flaky summary")
        stub.fail.add("flaky summary")
        summary = await memory.backfill_vectors()
        assert summary["repaired"] == 0
        assert summary["skipped"] == 1
        async with db.execute(
            "SELECT embedded FROM memory WHERE id = ?", (mid,)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0, "left unembedded"
        await db.close()

    async def test_backfill_repairs_drifted_row(self):
        """Defensive: a row with embedded_dim != declared is re-embedded.

        The current vec0 rejects a wrong-width insert, so the normal write
        path cannot produce this state; it is simulated via a direct row
        update (the wrong-width vec row that would trigger the replace cannot
        be created in this environment).
        """
        memory, db, stub = await build_env()
        _requires_vec(memory)
        mid = await _insert_pending_row(db, "drifted summary")
        # Simulate a drifted row: embedded=1 but the recorded width is wrong.
        await db.execute(
            "UPDATE memory SET embedded = 1, embedded_dim = ? WHERE id = ?",
            (DIMS * 2, mid),
        )
        await db.commit()
        summary = await memory.backfill_vectors()
        assert "error" not in summary
        assert summary["repaired"] == 1
        assert summary["drifted"] == 1
        assert summary["pending"] == 0
        async with db.execute(
            "SELECT embedded, embedded_dim FROM memory WHERE id = ?", (mid,)
        ) as cursor:
            embedded, embedded_dim = await cursor.fetchone()
        assert embedded == 1
        assert embedded_dim == DIMS, "the recorded width is corrected"
        async with db.execute(
            "SELECT count(*) FROM memory_vec WHERE memory_id = ?", (mid,)
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
        await db.close()

    async def test_backfill_noop_when_all_embedded(self):
        """With no pending/drifted rows, the backfill repairs nothing."""
        memory, db, stub = await build_env()
        _requires_vec(memory)
        # A correctly-embedded row (via the normal write path).
        await memory.write_record(
            channel_id="irc:#test", summary="a summary",
            created_at=time.time(), importance=0.5,
            msg_id_start=1, msg_id_end=2,
        )
        summary = await memory.backfill_vectors()
        assert "error" not in summary
        assert summary["repaired"] == 0
        await db.close()
