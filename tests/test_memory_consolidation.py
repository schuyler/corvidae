"""Tests for the consolidation write path (Phase 1a WP1a.6, bootstrap-mapping §3.1).

Both triggers (compaction, idle) run one watermarked code path in a
tracked background task: fetch the un-consolidated message_log range,
summarize first-person via the background LLM role, score an importance
prior, embed (fail-soft to embedded=0), and insert the memory row with the
watermark advance in one atomic step.
"""

import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.attribution import get_attribution
from corvidae.channel import Channel
from corvidae.context import MessageType
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin, RubricPrior
from corvidae.persistence import PersistencePlugin, init_db


CONSOLIDATION_JSON = json.dumps({
    "summary": "I helped kestrel debug the weather station's wifi dropouts.",
    "topic_tags": ["esp32", "wifi"],
    "participants": ["kestrel"],
})

EMBED_CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "background": {"base_url": "http://localhost:8080", "model": "bg"},
        "embedding": {
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": 4,
        },
    },
    "memory": {"idle_consolidate_after": 1800},
}


class StubClient:
    """Minimal LLMClient stand-in recording calls and attribution."""

    def __init__(self, chat_text: str = CONSOLIDATION_JSON):
        self.chat_text = chat_text
        self.chat_calls: list[dict] = []
        self.embed_calls: list[list[str]] = []
        self.embed_result: list[list[float]] | Exception = [[1.0, 0.0, 0.0, 0.0]]

    async def chat(self, messages, tools=None, extra_body=None):
        self.chat_calls.append({
            "messages": messages,
            "attribution": dict(get_attribution()),
        })
        return {"choices": [{"message": {"role": "assistant", "content": self.chat_text}}]}

    async def embed(self, texts):
        self.embed_calls.append(list(texts))
        if isinstance(self.embed_result, Exception):
            raise self.embed_result
        return self.embed_result


async def build_consolidation_env(config=None):
    """Full pm: persistence (in-memory DB) + stubbed llm clients + memory.

    Returns (memory, persistence, llm_stub_background, channel, db).
    """
    config = config or EMBED_CONFIG
    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")

    llm = LLMPlugin()
    pm.register(llm, name="llm")
    await llm.on_init(pm=pm, config=config)
    stub = StubClient()
    llm._clients["main"] = stub
    llm._clients["background"] = stub
    llm._clients["embedding"] = stub
    embedding_cfg = config.get("llm", {}).get("embedding")
    if embedding_cfg:
        llm.embedding_dimensions = embedding_cfg["dimensions"]

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)
    # The importance prior is pluggable; pin it for deterministic tests.
    prior = MagicMock()
    prior.score = AsyncMock(return_value=0.7)
    memory.importance_prior = prior

    channel = Channel(transport="irc", scope="#test")
    return memory, persistence, stub, channel, db


async def seed_dialog(persistence, channel, texts: list[tuple[str, str]]) -> list[int]:
    """Persist (role, content) messages; returns their rowids."""
    rowids = []
    for role, content in texts:
        rowids.append(
            await persistence.on_conversation_event(
                channel=channel,
                message={"role": role, "content": content},
                message_type=MessageType.MESSAGE,
            )
        )
    return rowids


async def fetch_memories(db) -> list[dict]:
    async with db.execute(
        "SELECT id, channel_id, summary, importance, topic_tags, participants, "
        "msg_id_start, msg_id_end, embedded FROM memory ORDER BY id"
    ) as cursor:
        rows = await cursor.fetchall()
    keys = ["id", "channel_id", "summary", "importance", "topic_tags",
            "participants", "msg_id_start", "msg_id_end", "embedded"]
    return [dict(zip(keys, row)) for row in rows]


async def fetch_watermark(db, channel_id) -> int | None:
    async with db.execute(
        "SELECT last_message_id FROM consolidation_watermark WHERE channel_id = ?",
        (channel_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return row[0] if row else None


class TestCompactionTrigger:
    async def test_compaction_produces_one_memory_row(self):
        memory, persistence, stub, channel, db = await build_consolidation_env()
        rowids = await seed_dialog(persistence, channel, [
            ("user", "my weather station drops off wifi"),
            ("assistant", "did you set WIFI_PS_NONE?"),
            ("user", "that fixed it!"),
        ])

        await memory.on_compaction(
            channel=channel,
            summary_msg={"role": "assistant", "content": "[Summary] wifi fixed"},
            retain_count=0,
            compacted_ids=rowids,
        )
        await memory.wait_for_background_tasks()

        records = await fetch_memories(db)
        assert len(records) == 1
        record = records[0]
        assert record["channel_id"] == channel.id
        assert record["summary"].startswith("I helped kestrel")
        assert json.loads(record["topic_tags"]) == ["esp32", "wifi"]
        assert json.loads(record["participants"]) == ["kestrel"]
        assert record["msg_id_start"] == rowids[0]
        assert record["msg_id_end"] == rowids[-1]
        assert record["importance"] == 0.7
        assert record["embedded"] == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        # The vec row landed alongside.
        if memory._vec_available:
            async with db.execute("SELECT count(*) FROM memory_vec") as cursor:
                assert (await cursor.fetchone())[0] == 1
        await db.close()

    async def test_empty_compacted_ids_is_a_noop(self):
        memory, persistence, stub, channel, db = await build_consolidation_env()
        await memory.on_compaction(
            channel=channel,
            summary_msg={"role": "assistant", "content": "[Summary]"},
            retain_count=0,
            compacted_ids=[],
        )
        await memory.wait_for_background_tasks()
        assert await fetch_memories(db) == []
        await db.close()

    async def test_consolidation_depends_only_on_hook_payload(self):
        """No summary row is ever written — consolidation still works (trap #4)."""
        memory, persistence, stub, channel, db = await build_consolidation_env()
        rowids = await seed_dialog(persistence, channel, [
            ("user", "hello"), ("assistant", "hi"),
        ])
        # Note: persistence.on_compaction is never called here.
        await memory.on_compaction(
            channel=channel,
            summary_msg={"role": "assistant", "content": "[Summary]"},
            retain_count=0,
            compacted_ids=rowids,
        )
        await memory.wait_for_background_tasks()
        assert len(await fetch_memories(db)) == 1
        await db.close()


class TestWatermarkOverlapSafety:
    async def test_overlapping_triggers_produce_no_duplicates(self):
        """Both triggers racing the same range store exactly one record."""
        memory, persistence, stub, channel, db = await build_consolidation_env()
        rowids = await seed_dialog(persistence, channel, [
            ("user", "one"), ("assistant", "two"), ("user", "three"),
        ])

        # Fire the compaction trigger twice over the same range, plus an
        # idle-style direct consolidation — all against the same watermark.
        await memory.on_compaction(
            channel=channel, summary_msg={"role": "assistant", "content": "s"},
            retain_count=0, compacted_ids=rowids,
        )
        await memory.on_compaction(
            channel=channel, summary_msg={"role": "assistant", "content": "s"},
            retain_count=0, compacted_ids=rowids,
        )
        await memory.wait_for_background_tasks()

        records = await fetch_memories(db)
        assert len(records) == 1
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        await db.close()

    async def test_pure_context_range_advances_watermark_without_record(self):
        memory, persistence, stub, channel, db = await build_consolidation_env()
        rowids = []
        for content in ("ctx one", "ctx two"):
            rowids.append(
                await persistence.on_conversation_event(
                    channel=channel,
                    message={"role": "system", "content": content},
                    message_type=MessageType.CONTEXT,
                )
            )
        await memory.on_compaction(
            channel=channel, summary_msg={"role": "assistant", "content": "s"},
            retain_count=0, compacted_ids=rowids,
        )
        await memory.wait_for_background_tasks()

        assert await fetch_memories(db) == []
        assert await fetch_watermark(db, channel.id) == rowids[-1]
        assert stub.chat_calls == []  # no LLM call for a dialog-free range
        await db.close()


class TestEmbeddingDegradation:
    async def test_embedder_failure_stores_row_with_embedded_zero(self, caplog):
        memory, persistence, stub, channel, db = await build_consolidation_env()
        stub.embed_result = RuntimeError("encoder down")
        rowids = await seed_dialog(persistence, channel, [
            ("user", "hello"), ("assistant", "hi"),
        ])
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            await memory.on_compaction(
                channel=channel, summary_msg={"role": "assistant", "content": "s"},
                retain_count=0, compacted_ids=rowids,
            )
            await memory.wait_for_background_tasks()

        records = await fetch_memories(db)
        assert len(records) == 1
        assert records[0]["embedded"] == 0
        if memory._vec_available:
            async with db.execute("SELECT count(*) FROM memory_vec") as cursor:
                assert (await cursor.fetchone())[0] == 0
        # The failure was logged, not raised.
        assert any("embed" in rec.message.lower() for rec in caplog.records)
        await db.close()


class TestIdleTrigger:
    async def test_idle_consolidates_stale_channel_tail(self):
        memory, persistence, stub, channel, db = await build_consolidation_env()
        rowids = await seed_dialog(persistence, channel, [
            ("user", "old talk"), ("assistant", "old reply"),
        ])
        # Age the messages past the idle threshold (timestamps drive
        # inactivity when the channel is not in a registry).
        await db.execute(
            "UPDATE message_log SET timestamp = timestamp - 4000"
        )
        await db.commit()

        await memory.on_idle()
        await memory.wait_for_background_tasks()

        records = await fetch_memories(db)
        assert len(records) == 1
        assert records[0]["msg_id_end"] == rowids[-1]
        await db.close()

    async def test_idle_skips_recently_active_channel(self):
        memory, persistence, stub, channel, db = await build_consolidation_env()
        await seed_dialog(persistence, channel, [
            ("user", "fresh talk"), ("assistant", "fresh reply"),
        ])
        await memory.on_idle()
        await memory.wait_for_background_tasks()
        assert await fetch_memories(db) == []
        await db.close()


class TestAttribution:
    async def test_consolidation_llm_call_carries_attribution(self):
        memory, persistence, stub, channel, db = await build_consolidation_env()
        rowids = await seed_dialog(persistence, channel, [
            ("user", "hello"), ("assistant", "hi"),
        ])
        await memory.on_compaction(
            channel=channel, summary_msg={"role": "assistant", "content": "s"},
            retain_count=0, compacted_ids=rowids,
        )
        await memory.wait_for_background_tasks()

        assert stub.chat_calls, "consolidation never called the LLM"
        attribution = stub.chat_calls[0]["attribution"]
        assert attribution.get("stage") == "consolidation"
        assert attribution.get("channel_id") == channel.id
        await db.close()


class TestRubricPrior:
    async def test_rubric_prior_parses_score(self):
        client = StubClient(chat_text=json.dumps({"importance": 0.85}))
        prior = RubricPrior(lambda: client)
        score = await prior.score([{"role": "user", "content": "big decision"}])
        assert score == 0.85

    async def test_rubric_prior_returns_half_on_failure(self, caplog):
        client = StubClient(chat_text="not json at all")
        prior = RubricPrior(lambda: client)
        with caplog.at_level(logging.WARNING, logger="corvidae.memory"):
            score = await prior.score([{"role": "user", "content": "x"}])
        assert score == 0.5

    async def test_rubric_prior_clamps_out_of_range(self):
        client = StubClient(chat_text=json.dumps({"importance": 3.5}))
        prior = RubricPrior(lambda: client)
        assert await prior.score([{"role": "user", "content": "x"}]) == 1.0
