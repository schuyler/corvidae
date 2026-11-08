"""Tests: a wrong-width embed is an embed failure (#597, #612).

vec0 rejects a wrong-width insert. Letting one reach the vec INSERT inside a
write transaction makes the ``OperationalError`` roll back the whole
transaction — on the consolidation path that is record, vec row and
watermark — and every later trigger re-summarizes the same range and fails
the same way, stalling the channel for good (records lost, LLM calls
wasted).

A wrong-width embedding is now handled like an embed failure: the row is
stored ``embedded=0`` (backfillable) and the write proceeds. The same check
guards retention's backfill pass, where a single bad vector would have rolled
back the whole batch on every run.
"""

import json
import logging
import time

import aiosqlite
import pytest

from corvidae.channel import Channel
from corvidae.context import MessageType
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae_cognition.memory import MemoryPlugin
from corvidae_cognition.retention import run_retention_job
from corvidae.persistence import PersistencePlugin, init_db

from unittest.mock import AsyncMock, MagicMock

from helpers import FakeForkAgent, stub_llm_config, with_chat_llm

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
    },
    "memory": {
        "idle_consolidate_after": 1800,
        "retention": {
            "grace_days": 14,
            "importance_floor": 0.8,
            "demote_below": 0.15,
            "half_life_days": 90,
            "interval": 21600,  # 6h
            "backfill_batch": 32,
        },
    },
}

SUMMARY_JSON = json.dumps({
    "summary": "the badwidth marker record",
    "topic_tags": ["test"],
    "participants": ["kestrel"],
})


class _WidthStub:
    """Stub LLM client: chat returns a fixed consolidation JSON; embed maps
    a text substring to a forced vector (default is a correct width)."""

    def __init__(self) -> None:
        self.fixed: dict[str, list[float]] = {}
        self.chat_calls = 0
        self.embed_calls = 0

    def request_model(self, extra_body=None):
        return (extra_body or {}).get("model", "stub-model")

    async def chat(self, messages, tools=None, extra_body=None):
        self.chat_calls += 1
        return {"choices": [{"message": {"role": "assistant",
                                         "content": SUMMARY_JSON}}]}

    async def embed(self, texts, kind=None):
        self.embed_calls += 1
        out = []
        for t in texts:
            for key, vec in self.fixed.items():
                if key in t:
                    out.append(list(vec))
                    break
            else:
                out.append([1.0, 0.0, 0.0, 0.0])
        return out


async def build_env():
    """persistence + llm + memory on a fresh in-memory DB.

    Returns (memory, db, stub, channel, persistence).
    """
    config = with_chat_llm(EMBED_CONFIG)
    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=config)
    stub = _WidthStub()
    llm._clients["main"] = stub
    llm._clients["embedding"] = stub
    stub_llm_config(llm, config.get("llm"))

    # A view-capable agent with forks off: a range outside the live window
    # is written from the rendered view (#380, #388).
    pm.register(FakeForkAgent(set()), name="agent")

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)
    # Pin the importance prior for deterministic tests.
    prior = MagicMock()
    prior.score = AsyncMock(return_value=0.7)
    memory.importance_prior = prior

    channel = Channel(transport="irc", scope="#width")
    return memory, db, stub, channel, persistence


def _requires_vec(memory):
    if not memory._vec_available:
        pytest.skip("sqlite-vec unavailable in this environment")


async def seed_dialog(persistence, channel, texts):
    """Persist (role, content) messages; returns their rowids."""
    rowids = []
    for role, content in texts:
        rowids.append(
            await persistence.persist_message(
                channel=channel,
                message={"role": role, "content": content},
                message_type=MessageType.MESSAGE,
            )
        )
    return rowids


async def seed_memory_row(db, channel_id, summary, importance=0.5,
                          created_at=None, embedded=0) -> int:
    """Insert one memory row (indexed=1, unembedded by default)."""
    created_at = created_at or time.time()
    cursor = await db.execute(
        "INSERT INTO memory (channel_id, created_at, summary, importance, "
        "msg_id_start, msg_id_end, embedded, indexed) "
        "VALUES (?, ?, ?, ?, 1, 2, ?, 1)",
        (channel_id, created_at, summary, importance, embedded),
    )
    return cursor.lastrowid


async def fetch_memories(db) -> list[dict]:
    async with db.execute(
        "SELECT id, channel_id, summary, embedded, indexed FROM memory "
        "ORDER BY id"
    ) as cursor:
        rows = await cursor.fetchall()
    return [
        {"id": r[0], "channel_id": r[1], "summary": r[2],
         "embedded": r[3], "indexed": r[4]}
        for r in rows
    ]


async def vec_row_exists(db, memory_id: int) -> bool:
    try:
        async with db.execute(
            "SELECT 1 FROM memory_vec WHERE memory_id = ?", (memory_id,)
        ) as c:
            return (await c.fetchone()) is not None
    except Exception:
        return False


async def watermark(db, channel_id) -> int | None:
    async with db.execute(
        "SELECT last_message_id FROM consolidation_watermark "
        "WHERE channel_id = ?", (channel_id,),
    ) as c:
        row = await c.fetchone()
    return row[0] if row else None


async def compact(memory, channel, ids):
    """``on_compaction`` plus an await on the consolidation task it spawns."""
    before = set(memory._tasks)
    await memory.on_compaction(
        channel=channel, summary_msg={"role": "assistant", "content": "s"},
        retain_count=0, compacted_ids=ids,
    )
    task = next(iter(set(memory._tasks) - before))
    await task


class TestConsolidationWidth:
    """The write path: a wrong-width embed degrades to embedded=0."""

    async def test_on_compaction_stores_pending_and_advances_watermark(
            self, caplog):
        """The #612 regression, through the full consolidation path.

        A wrong-width embed must not roll back the transaction: the record
        is stored embedded=0 (backfillable), there is no vec row, and the
        watermark advances past the range — the channel is not stalled.
        """
        memory, db, stub, channel, persistence = await build_env()
        _requires_vec(memory)
        ids = await seed_dialog(
            persistence, channel, [("user", "hi"), ("assistant", "hello")],
        )
        # Force a wrong-width (2*DIMS) vector for this summary.
        stub.fixed["badwidth"] = [0.1] * (DIMS * 2)
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await compact(memory, channel, ids)

        memories = await fetch_memories(db)
        assert len(memories) == 1, "the record must be stored"
        mid = memories[0]["id"]
        assert memories[0]["embedded"] == 0, (
            "a wrong-width row must be left unembedded (backfillable)"
        )
        assert not await vec_row_exists(db, mid), (
            "a wrong-width vector must not be inserted"
        )
        assert await watermark(db, channel.id) == ids[-1], (
            "the watermark must advance past the range; the channel "
            "must not stall"
        )
        assert any(
            "width mismatch" in rec.message for rec in caplog.records
        ), "a wrong-width embed must fail loud"
        await db.close()

    async def test_channel_not_stalled_after_wrong_width(self):
        """After a bad-width embed, the next consolidation with a
        good-width vector succeeds — no permanent stall."""
        memory, db, stub, channel, persistence = await build_env()
        _requires_vec(memory)
        ids1 = await seed_dialog(persistence, channel, [("user", "first")])
        stub.fixed["badwidth"] = [0.1] * (DIMS * 2)
        await compact(memory, channel, ids1)

        stub.fixed.pop("badwidth")
        ids2 = await seed_dialog(persistence, channel, [("user", "second")])
        await compact(memory, channel, ids2)

        memories = await fetch_memories(db)
        assert [m["embedded"] for m in memories] == [0, 1], (
            "first record pending, second embedded"
        )
        assert await vec_row_exists(db, memories[1]["id"])
        assert await watermark(db, channel.id) == ids2[-1]
        await db.close()

    async def test_correct_width_stored_as_before(self):
        """The common path is unchanged: a correct-width embed is stored
        with its vec row."""
        memory, db, stub, channel, persistence = await build_env()
        _requires_vec(memory)
        mid = await memory.write_record(
            channel_id=channel.id, summary="a normal summary",
            created_at=time.time(), importance=0.5,
            msg_id_start=1, msg_id_end=2,
        )
        memories = await fetch_memories(db)
        assert len(memories) == 1
        assert memories[0]["embedded"] == 1
        assert await vec_row_exists(db, mid)
        await db.close()


class TestRetentionBackfillWidth:
    """Retention pass 3: one bad vector skips its row, keeps the rest."""

    async def test_backfill_skips_wrong_width_row_keeps_rest(self, caplog):
        """A single wrong-width vector must not roll back the batch:
        its row stays backfillable, the rest of the batch is backfilled."""
        memory, db, stub, channel, persistence = await build_env()
        _requires_vec(memory)
        bad_id = await seed_memory_row(db, channel.id, "a badwidth record")
        good_id = await seed_memory_row(db, channel.id, "a normal record")
        stub.fixed["badwidth"] = [0.1] * (DIMS * 2)
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await run_retention_job(memory, now=time.time())

        memories = {m["id"]: m for m in await fetch_memories(db)}
        assert memories[good_id]["embedded"] == 1, (
            "the good row must be backfilled"
        )
        assert await vec_row_exists(db, good_id)
        assert memories[bad_id]["embedded"] == 0, (
            "the bad row must stay backfillable"
        )
        assert not await vec_row_exists(db, bad_id), (
            "a wrong-width vector must not be inserted"
        )
        assert any(
            "width mismatch" in rec.message for rec in caplog.records
        ), "a wrong-width embed must fail loud"
        await db.close()
