"""Tests for the retrieval read path (Phase 1a WP1a.7, bootstrap-mapping §3.1).

before_agent_turn embeds the inbound user text, scores vec-KNN candidates
by similarity × recency decay, annotates relevance bands, admits the
winners through the funnel, bumps access stats for what was admitted, and
persists a retrieval_log row. Encoder failure degrades to FTS5.

The stub embedder is a deterministic hashed bag-of-words unit vector, so
related texts have reproducibly higher cosine similarity.
"""

import hashlib
import json
import math
import re
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.funnel import FunnelPlugin
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db

from evals.metrics import recall_at_k


DIMS = 32

EMBED_CONFIG = {
    "llm": {
        "main": {"base_url": "http://localhost:8080", "model": "chat"},
        "embedding": {
            "base_url": "http://localhost:8081",
            "model": "test-embedder",
            "dimensions": DIMS,
        },
    },
}


def bow_embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words unit vector."""
    vec = [0.0] * DIMS
    for token in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class StubEmbedClient:
    """Embedding client stand-in with a controllable failure mode."""

    def __init__(self):
        self.fail = False
        self.calls: list[list[str]] = []

    async def embed(self, texts, kind=None):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("encoder down")
        return [bow_embed(t) for t in texts]

    async def chat(self, messages, tools=None, extra_body=None):
        raise AssertionError("retrieval must not call chat()")


async def build_retrieval_env(config=None):
    """persistence + llm(stub embedder) + funnel + memory on one pm.

    Returns (memory, embedder, channel, conv, db).
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
    embedder = StubEmbedClient()
    llm._clients["main"] = embedder
    llm._clients["embedding"] = embedder
    embedding_cfg = config.get("llm", {}).get("embedding")
    if embedding_cfg:
        llm.embedding_dimensions = embedding_cfg["dimensions"]

    funnel = FunnelPlugin()
    pm.register(funnel, name="funnel")
    await funnel.on_init(pm=pm, config=config)

    memory = MemoryPlugin()
    pm.register(memory, name="memory")
    await memory.on_init(pm=pm, config=config)
    await memory.on_start(config=config)

    channel = Channel(transport="irc", scope="#electronics")
    conv = ContextWindow(channel.id)
    channel.conversation = conv
    return memory, embedder, channel, conv, db


async def seed_memory(
    db, channel_id: str, summary: str, created_at: float | None = None,
    embedded: bool = True,
) -> int:
    """Insert one memory row (+ vec row when embedded)."""
    import sqlite_vec
    cursor = await db.execute(
        "INSERT INTO memory (channel_id, created_at, summary, importance, "
        "msg_id_start, msg_id_end, embedded) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (channel_id, created_at or time.time(), summary, 0.5, 1, 2,
         1 if embedded else 0),
    )
    rowid = cursor.lastrowid
    if embedded:
        await db.execute(
            "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
            (rowid, sqlite_vec.serialize_float32(bow_embed(summary))),
        )
    await db.commit()
    return rowid


async def ask(memory, channel, conv, text: str) -> None:
    """Simulate the turn-loop entry: append the user message, fire retrieval."""
    conv.append({"role": "user", "content": text})
    await memory.before_agent_turn(channel=channel)


def context_messages(conv) -> list[dict]:
    return [m for m in conv.messages
            if m.get("_message_type") == MessageType.CONTEXT]


class TestRetrievalBasics:
    async def test_relevant_memory_ranks_first_with_bands_and_framing(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        relevant = await seed_memory(
            db, channel.id,
            "kestrel's ESP32 weather station kept dropping off wifi until "
            "power-save was disabled",
        )
        await seed_memory(
            db, channel.id, "the greenhouse humidity sensor reads four percent high",
        )

        await ask(memory, channel, conv,
                  "why did the ESP32 weather station drop off wifi?")

        contexts = context_messages(conv)
        assert len(contexts) == 1
        content = contexts[0]["content"]
        # Funnel framing wraps the entries.
        assert content.startswith("[CONTEXT from memory")
        assert content.rstrip().endswith("[end CONTEXT from memory]")
        # Band annotation and age prefix each line.
        assert re.search(r"\[(strong|moderate|weak)\] \(.+?\) .*wifi", content)
        # The relevant memory is the first entry line.
        lines = content.split("\n")[1:-1]
        assert "wifi" in lines[0]
        await db.close()

    async def test_no_memories_admits_nothing(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await ask(memory, channel, conv, "anything at all?")
        assert context_messages(conv) == []
        await db.close()

    async def test_notification_turns_skip_retrieval(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed_memory(db, channel.id, "some memory about wifi networks")
        conv.append({"role": "system", "content": "[task]\n\nwifi wifi wifi"})
        await memory.before_agent_turn(channel=channel)
        assert context_messages(conv) == []
        assert embedder.calls == []
        await db.close()


class TestCompartmentalization:
    async def test_other_channel_memories_never_surface(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed_memory(
            db, "irc:#other", "the wifi password for the other channel is hunter2",
        )
        await ask(memory, channel, conv, "what is the wifi password?")
        assert context_messages(conv) == []
        await db.close()

    async def test_group_configured_channels_share(self):
        config = {
            **EMBED_CONFIG,
            "memory": {
                "channel_groups": {
                    "home": ["irc:#electronics", "irc:#garden"],
                },
            },
        }
        memory, embedder, channel, conv, db = await build_retrieval_env(config)
        await seed_memory(
            db, "irc:#garden",
            "the solar charge controller logs power data to MQTT",
        )
        await ask(memory, channel, conv,
                  "where does the solar charge controller log power data?")
        contexts = context_messages(conv)
        assert len(contexts) == 1
        assert "MQTT" in contexts[0]["content"]
        await db.close()


class TestDedupeAndStats:
    async def test_reasking_does_not_duplicate_context(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed_memory(db, channel.id,
                          "kestrel moved the rain gauge to the north fence")
        await ask(memory, channel, conv, "where is the rain gauge?")
        await ask(memory, channel, conv, "where is the rain gauge?")
        assert len(context_messages(conv)) == 1
        await db.close()

    async def test_access_stats_increment_only_for_admitted(self):
        # A one-token budget admits nothing — stats must not move.
        config = {**EMBED_CONFIG, "funnel": {"budgets": {"memory": 1}}}
        memory, embedder, channel, conv, db = await build_retrieval_env(config)
        rowid = await seed_memory(
            db, channel.id, "kestrel moved the rain gauge to the north fence",
        )
        await ask(memory, channel, conv, "where is the rain gauge?")
        async with db.execute(
            "SELECT retrieval_count, last_retrieved_at FROM memory WHERE id = ?",
            (rowid,),
        ) as cursor:
            count, last = await cursor.fetchone()
        assert count == 0 and last is None
        await db.close()

    async def test_access_stats_increment_for_admitted(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        rowid = await seed_memory(
            db, channel.id, "kestrel moved the rain gauge to the north fence",
        )
        await ask(memory, channel, conv, "where is the rain gauge?")
        async with db.execute(
            "SELECT retrieval_count, last_retrieved_at FROM memory WHERE id = ?",
            (rowid,),
        ) as cursor:
            count, last = await cursor.fetchone()
        assert count == 1
        assert isinstance(last, float)
        await db.close()

    async def test_admitted_ids_no_substring_collision(self):
        """Bug I-1 regression: when A's formatted line is a substring of B's
        formatted line, and only B is admitted (A is budget-dropped), the old
        substring-search code (`line_A in appended`) falsely credits A via the
        containment check.  The fixed code compares against the list of admitted
        lines returned by FunnelPlugin.admit, so only exact matches count."""
        from corvidae.context import count_tokens

        # Both memories use band "strong" (thresholds lowered so scoring
        # differences don't affect band labels).  B's formatted line is
        # "[strong] (now) wifi router fixed the issue"; A's is
        # "[strong] (now) wifi router" — a strict prefix of B's line.
        line_b_formatted = "[strong] (now) wifi router fixed the issue"
        # Budget: admits B exactly; budget-drops A (A would add 8 more tokens).
        budget = count_tokens(line_b_formatted)

        config = {
            **EMBED_CONFIG,
            "memory": {"retrieval": {"bands": {"strong": 0.0, "moderate": 0.0}}},
            "funnel": {"budgets": {"memory": budget}},
        }
        memory, embedder, channel, conv, db = await build_retrieval_env(config)

        rowid_a = await seed_memory(db, channel.id, "wifi router")
        rowid_b = await seed_memory(db, channel.id, "wifi router fixed the issue")

        # Query that scores B (5 shared tokens) higher than A (2 shared tokens).
        await ask(memory, channel, conv, "wifi router fixed the issue")

        async with db.execute(
            "SELECT id, retrieval_count FROM memory ORDER BY id"
        ) as cursor:
            rows = {row[0]: row[1] for row in await cursor.fetchall()}

        assert rows[rowid_b] == 1, "B was admitted; its retrieval_count must be 1"
        # A's formatted line is a substring of B's, so the old `line in appended`
        # check would credit A even though it was budget-dropped.
        assert rows[rowid_a] == 0, (
            f"A was budget-dropped (not admitted); retrieval_count must be 0, "
            f"got {rows[rowid_a]}"
        )
        await db.close()


class TestDegradation:
    async def test_encoder_down_degrades_to_fts(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed_memory(db, channel.id,
                          "kestrel's weather station uses an ESP32 board")
        embedder.fail = True
        await ask(memory, channel, conv, "what board does the weather station use?")

        contexts = context_messages(conv)
        assert len(contexts) == 1
        assert "ESP32" in contexts[0]["content"]
        async with db.execute(
            "SELECT degraded_to_fts FROM retrieval_log ORDER BY id DESC LIMIT 1"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
        await db.close()


class TestRetrievalLog:
    async def test_retrieval_log_row_per_retrieval(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed_memory(db, channel.id,
                          "kestrel moved the rain gauge to the north fence")
        await ask(memory, channel, conv, "where is the rain gauge?")

        async with db.execute(
            "SELECT channel_id, top_score, hit_count, admitted_count, "
            "degraded_to_fts FROM retrieval_log"
        ) as cursor:
            rows = await cursor.fetchall()
        assert len(rows) == 1
        channel_id, top_score, hit_count, admitted_count, degraded = rows[0]
        assert channel_id == channel.id
        assert hit_count == 1
        assert admitted_count == 1
        assert degraded == 0
        assert top_score is not None and top_score > 0
        await db.close()

    async def test_empty_retrieval_still_logged(self):
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await ask(memory, channel, conv, "anything at all?")
        async with db.execute(
            "SELECT hit_count, admitted_count FROM retrieval_log"
        ) as cursor:
            rows = await cursor.fetchall()
        assert rows == [(0, 0)]
        await db.close()


# --- RED: design item 6 (retrieval call site) ---


class _KindTrackingRetrievalStub(StubEmbedClient):
    """StubEmbedClient variant that records the kind argument passed to embed()."""

    def __init__(self):
        super().__init__()
        self.embed_kind_calls: list[str | None] = []

    async def embed(self, texts, kind=None):
        self.embed_kind_calls.append(kind)
        if self.fail:
            raise RuntimeError("encoder down")
        return [bow_embed(t) for t in texts]


class TestEmbedCallSiteKinds:
    async def test_retrieval_embeds_with_query_kind(self):
        """item 6: the retrieval read path calls embed() with kind='query'."""
        memory, _original_embedder, channel, conv, db = await build_retrieval_env()
        if not memory._vec_available:
            pytest.skip("sqlite-vec unavailable; retrieval embed path not exercised")

        tracker = _KindTrackingRetrievalStub()
        llm = memory.pm.get_plugin("llm")
        llm._clients["embedding"] = tracker

        await seed_memory(db, channel.id, "kestrel's ESP32 weather station")
        await ask(memory, channel, conv, "what board does the weather station use?")

        assert tracker.embed_kind_calls, "embed() was never called during retrieval"
        assert tracker.embed_kind_calls[0] == "query", (
            f"expected kind='query', got {tracker.embed_kind_calls[0]!r}"
        )
        await db.close()


class TestRecallBenchmark:
    async def test_recall_at_k_beats_floor_on_seed_fixture(self):
        """The Phase 0 harness doing real work over the real retrieval path."""
        fixture_path = (
            Path(__file__).parent / "fixtures" / "memory_retrieval_basic.json"
        )
        fixture = json.loads(fixture_path.read_text())
        # One group spanning both fixture channels so every query sees the
        # whole corpus.
        channels = sorted({m["channel_id"] for m in fixture["memories"]})
        config = {
            **EMBED_CONFIG,
            "memory": {"channel_groups": {"fixture": channels}},
        }
        memory, embedder, channel, conv, db = await build_retrieval_env(config)

        id_map: dict[int, str] = {}
        for record in fixture["memories"]:
            rowid = await seed_memory(db, record["channel_id"], record["summary"])
            id_map[rowid] = record["id"]

        recalls = []
        for query in fixture["queries"]:
            candidates, degraded = await memory.retrieve(
                channels[0], query["text"], k=5
            )
            assert not degraded
            ranked = [id_map[c["id"]] for c in candidates]
            recalls.append(recall_at_k(ranked, query["relevant"], 5))

        mean_recall = sum(recalls) / len(recalls)
        # Floor, not aspiration: hashed bag-of-words retrieval on the seed
        # fixture measured well above this; a regression below it means the
        # scoring path broke.
        assert mean_recall >= 0.75, f"mean recall@5 {mean_recall:.2f} below floor"
        await db.close()
