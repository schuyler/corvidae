"""Tests for WP1b.3 — message_log FTS (message_fts) and memory tools.

Red phase: all tests in this file fail at collection with:
    ModuleNotFoundError: No module named 'corvidae.tools.memory_tools'
because corvidae/tools/memory_tools.py does not exist yet.

Designed failure reason: missing corvidae.tools.memory_tools module.

Covers:
- search_memory: FTS keyword search, date filters, compartment scoping,
  demoted/superseded flags, include_demoted behavior, out-of-scope errors,
  stat non-mutation.
- recall_raw: verbatim role:content dialog, participants header, token cap,
  channel_id predicate (interleaved-channel regression), out-of-scope rejection,
  stat bump.
- Tool compartment scoping: channel B cannot see channel A's memories;
  group siblings CAN see each other's memories.
- message_fts backfill: pre-existing rows indexed once; partial-backfill
  rerun completes without uniqueness error.
"""

import asyncio
import json
import math
import hashlib
import re
import time
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

# PRIMARY IMPORT FAILURE — all tests below fail at collection time with
# ModuleNotFoundError: No module named 'corvidae.tools.memory_tools'
from corvidae.tools.memory_tools import MemoryToolsPlugin

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType
from corvidae.funnel import FunnelPlugin
from corvidae.hooks import create_plugin_manager
from corvidae.llm_plugin import LLMPlugin
from corvidae.memory import MemoryPlugin
from corvidae.persistence import PersistencePlugin, init_db


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

GROUP_CONFIG = {
    **EMBED_CONFIG,
    "memory": {
        "channel_groups": {
            "home": ["irc:#electronics", "irc:#garden"],
        },
    },
}


def bow_embed(text: str) -> list[float]:
    vec = [0.0] * DIMS
    for token in re.findall(r"\w+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class StubEmbedClient:
    async def embed(self, texts, kind=None):
        return [bow_embed(t) for t in texts]

    async def chat(self, messages, tools=None, extra_body=None):
        raise AssertionError("tools tests must not call chat()")


async def build_tools_env(config=None):
    """Full pm: persistence + llm(stub) + funnel + memory + memory_tools."""
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

    memory_tools = MemoryToolsPlugin()
    pm.register(memory_tools, name="memory_tools")
    await memory_tools.on_init(pm=pm, config=config)

    return memory, memory_tools, db


def get_tools(memory_tools: MemoryToolsPlugin) -> dict:
    """Register tools and return a name→fn dict."""
    tool_list = []
    memory_tools.register_tools(tool_list)
    return {t.name: t.fn for t in tool_list}


def make_ctx(channel: Channel):
    ctx = MagicMock()
    ctx.channel = channel
    return ctx


async def seed_memory(
    db,
    channel_id: str,
    summary: str,
    importance: float = 0.5,
    indexed: int = 1,
    embedded: int = 1,
    retrieval_count: int = 0,
    last_retrieved_at: float | None = None,
    created_at: float | None = None,
    participants: list[str] | None = None,
    msg_id_start: int = 1,
    msg_id_end: int = 2,
) -> int:
    created_at = created_at or time.time()
    cursor = await db.execute(
        "INSERT INTO memory (channel_id, created_at, summary, importance, "
        "msg_id_start, msg_id_end, embedded, indexed, retrieval_count, "
        "last_retrieved_at, participants) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (channel_id, created_at, summary, importance, msg_id_start, msg_id_end,
         embedded, indexed, retrieval_count, last_retrieved_at,
         json.dumps(participants or []) ),
    )
    rowid = cursor.lastrowid
    if embedded:
        try:
            import sqlite_vec
            await db.execute(
                "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(bow_embed(summary))),
            )
        except ImportError:
            pass
    await db.commit()
    return rowid


async def seed_message_log_row(
    db,
    channel_id: str,
    role: str,
    content: str,
    ts: float | None = None,
) -> int:
    """Insert a raw message_log row; returns its id."""
    msg = json.dumps({"role": role, "content": content})
    cursor = await db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, ?, ?, ?)",
        (channel_id, msg, ts or time.time(), "message"),
    )
    rowid = cursor.lastrowid
    await db.commit()
    return rowid


# ---------------------------------------------------------------------------
# WP1b.3 search_memory — keyword search and filters
# ---------------------------------------------------------------------------


class TestSearchMemory:
    """Keyword search with filters; scope; stat non-mutation."""

    async def test_keyword_search_finds_seeded_content(self):
        """search_memory must return results matching the query keyword."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        mid = await seed_memory(
            db, channel.id, "kestrel's ESP32 weather station wifi power-save dropout"
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"]("ESP32 wifi", _ctx=make_ctx(channel))

        assert str(mid) in result or "ESP32" in result or "wifi" in result, (
            f"search_memory must find seeded content; got: {result!r}"
        )
        await db.close()

    async def test_date_filter_after_excludes_old_records(self):
        """search_memory after=<date> excludes records created before that date."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        old_ts = time.time() - 86400.0 * 30  # 30 days ago
        new_ts = time.time()

        _old_mid = await seed_memory(
            db, channel.id, "old ESP32 wifi record", created_at=old_ts
        )
        new_mid = await seed_memory(
            db, channel.id, "new ESP32 wifi record", created_at=new_ts
        )

        # Filter to only records after 7 days ago
        after_date = "2000-01-01"  # epoch-safe; adjust to test specific filtering
        # Use yesterday's date to exclude old_ts (30 days ago) but include new_ts
        import datetime
        yesterday = (
            datetime.datetime.utcfromtimestamp(time.time() - 86400.0)
            .date()
            .isoformat()
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "ESP32 wifi", after=yesterday, _ctx=make_ctx(channel)
        )

        assert str(new_mid) in result or "new" in result, (
            "after filter must allow the recent record"
        )
        # Old record (30 days ago) must not appear
        assert "old ESP32" not in result, (
            f"after filter must exclude old record; got: {result!r}"
        )
        await db.close()

    async def test_date_filter_before_excludes_new_records(self):
        """search_memory before=<date> excludes records created on/after that date."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        import datetime
        old_ts = time.time() - 86400.0 * 30
        _new_ts = time.time()

        old_mid = await seed_memory(
            db, channel.id, "old kestrel rain gauge location", created_at=old_ts
        )
        _new_mid = await seed_memory(
            db, channel.id, "new kestrel rain gauge update", created_at=time.time()
        )

        yesterday = (
            datetime.datetime.utcfromtimestamp(time.time() - 86400.0)
            .date()
            .isoformat()
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "kestrel rain gauge", before=yesterday, _ctx=make_ctx(channel)
        )

        assert str(old_mid) in result or "old" in result, (
            "before filter must include old record"
        )
        assert "new kestrel rain gauge update" not in result, (
            "before filter must exclude new record"
        )
        await db.close()

    async def test_unparseable_date_returns_error_string(self):
        """search_memory with an unparseable date returns an error string."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")
        await seed_memory(db, channel.id, "some summary")

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "some query", after="not-a-date", _ctx=make_ctx(channel)
        )

        assert "error" in result.lower() or "invalid" in result.lower(), (
            f"unparseable date must return error string; got: {result!r}"
        )
        await db.close()

    async def test_demoted_record_flagged_in_search_results(self):
        """search_memory must include demoted records and flag them [demoted]."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        _mid = await seed_memory(
            db, channel.id, "demoted record about kestrel wifi issue",
            indexed=0, embedded=0,
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "kestrel wifi", _ctx=make_ctx(channel)
        )

        assert "[demoted]" in result, (
            f"demoted record must appear with [demoted] flag; got: {result!r}"
        )

    async def test_include_demoted_false_hides_demoted_records(self):
        """include_demoted=False restricts to indexed=1 records."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        _demoted_mid = await seed_memory(
            db, channel.id, "demoted record about rain gauge",
            indexed=0, embedded=0,
        )
        _active_mid = await seed_memory(
            db, channel.id, "active record about weather station",
            indexed=1, embedded=1,
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "rain gauge", include_demoted=False, _ctx=make_ctx(channel)
        )

        assert "[demoted]" not in result, (
            "include_demoted=False must hide demoted records"
        )
        assert "rain gauge" not in result, (
            "demoted rain gauge record must not appear with include_demoted=False"
        )
        await db.close()

    async def test_none_channel_context_returns_error(self):
        """search_memory with _ctx.channel = None returns an error string, never unscoped."""
        memory, memory_tools, db = await build_tools_env()
        await seed_memory(db, "irc:#electronics", "some summary")

        tools = get_tools(memory_tools)
        ctx = MagicMock()
        ctx.channel = None

        result = await tools["search_memory"]("query", _ctx=ctx)
        assert "error" in result.lower() or "channel" in result.lower(), (
            f"None channel must return error string; got: {result!r}"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.3 recall_raw — verbatim dialog, channel predicate, stat bump
# ---------------------------------------------------------------------------


class TestRecallRaw:
    """recall_raw: verbatim dialog, participants header, cap, channel predicate."""

    async def test_recall_raw_returns_verbatim_dialog_with_header(self):
        """recall_raw returns role:content dialog with participants header."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        # Insert message_log rows
        msg1_id = await seed_message_log_row(db, channel.id, "user", "what is the wifi fix?")
        msg2_id = await seed_message_log_row(db, channel.id, "assistant", "disable WIFI_PS_NONE")

        mid = await seed_memory(
            db, channel.id, "kestrel wifi power-save fix",
            participants=["kestrel"],
            msg_id_start=msg1_id, msg_id_end=msg2_id,
        )

        tools = get_tools(memory_tools)
        result = await tools["recall_raw"](mid, _ctx=make_ctx(channel))

        # Header with participants
        assert "kestrel" in result, f"participants header must be present; got: {result!r}"
        # Verbatim dialog
        assert "user: what is the wifi fix?" in result, (
            f"verbatim user message must appear; got: {result!r}"
        )
        assert "assistant: disable WIFI_PS_NONE" in result, (
            f"verbatim assistant message must appear; got: {result!r}"
        )
        await db.close()

    async def test_recall_raw_token_cap_truncates_with_marker(self):
        """recall_raw truncates to max_tokens and adds a truncation marker."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        # Insert many messages
        ids = []
        for i in range(20):
            msg_id = await seed_message_log_row(
                db, channel.id, "user" if i % 2 == 0 else "assistant",
                f"message content number {i} " + "word " * 20
            )
            ids.append(msg_id)

        mid = await seed_memory(
            db, channel.id, "long conversation summary",
            msg_id_start=min(ids), msg_id_end=max(ids),
        )

        tools = get_tools(memory_tools)
        result = await tools["recall_raw"](mid, max_tokens=10, _ctx=make_ctx(channel))

        assert "[truncated" in result or "truncated" in result.lower(), (
            f"token-capped recall must include truncation marker; got: {result!r}"
        )
        await db.close()

    async def test_recall_raw_channel_id_predicate_excludes_foreign_rows(self):
        """recall_raw must filter by channel_id to exclude interleaved foreign-channel rows.

        message_log uses a global AUTOINCREMENT id. With two channels, a record's
        [msg_id_start, msg_id_end] range contains interleaved foreign-channel rows.
        An unfiltered BETWEEN would replay those rows — a compartment leak.

        (Design review important finding #3 — bootstrap-mapping §4.12.)
        """
        memory, memory_tools, db = await build_tools_env()
        channel_a = Channel(transport="irc", scope="#electronics")
        channel_b = Channel(transport="irc", scope="#garden")

        # Interleave messages from both channels in the global id sequence:
        # a1, b1, a2, b2
        a1 = await seed_message_log_row(db, channel_a.id, "user", "ESP32 wifi fix needed")
        b1 = await seed_message_log_row(db, channel_b.id, "user", "garden sensor data")
        a2 = await seed_message_log_row(db, channel_a.id, "assistant", "disable WIFI_PS_NONE")
        b2 = await seed_message_log_row(db, channel_b.id, "assistant", "MQTT garden/power")

        # channel_a's record spans a1..a2 (but b1 is inside that numeric range)
        mid_a = await seed_memory(
            db, channel_a.id, "channel A wifi memory",
            participants=["kestrel"],
            msg_id_start=a1, msg_id_end=a2,
        )

        tools = get_tools(memory_tools)
        result = await tools["recall_raw"](mid_a, _ctx=make_ctx(channel_a))

        # channel_b's rows must NOT appear
        assert "garden sensor data" not in result, (
            f"recall_raw must not include foreign-channel row b1; got: {result!r}"
        )
        assert "MQTT garden/power" not in result, (
            f"recall_raw must not include foreign-channel row b2; got: {result!r}"
        )
        # channel_a's rows must appear
        assert "ESP32 wifi fix needed" in result, (
            f"recall_raw must include own-channel row a1; got: {result!r}"
        )
        assert "disable WIFI_PS_NONE" in result, (
            f"recall_raw must include own-channel row a2; got: {result!r}"
        )
        await db.close()

    async def test_recall_raw_bumps_access_stats(self):
        """recall_raw must bump retrieval_count and set last_retrieved_at."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        msg1 = await seed_message_log_row(db, channel.id, "user", "hello")
        msg2 = await seed_message_log_row(db, channel.id, "assistant", "hi")
        mid = await seed_memory(
            db, channel.id, "greeting exchange",
            retrieval_count=0, last_retrieved_at=None,
            msg_id_start=msg1, msg_id_end=msg2,
        )

        tools = get_tools(memory_tools)
        await tools["recall_raw"](mid, _ctx=make_ctx(channel))

        async with db.execute(
            "SELECT retrieval_count, last_retrieved_at FROM memory WHERE id = ?", (mid,)
        ) as c:
            count, last = await c.fetchone()

        assert count == 1, "recall_raw must increment retrieval_count"
        assert isinstance(last, float) and last > 0, (
            "recall_raw must set last_retrieved_at to current time"
        )
        await db.close()

    async def test_recall_raw_works_on_demoted_record(self):
        """recall_raw can fetch demoted and superseded records (remember harder)."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        msg1 = await seed_message_log_row(db, channel.id, "user", "demoted content here")
        msg2 = await seed_message_log_row(db, channel.id, "assistant", "old reply")
        mid = await seed_memory(
            db, channel.id, "demoted record for recall_raw test",
            indexed=0, embedded=0,
            msg_id_start=msg1, msg_id_end=msg2,
        )

        tools = get_tools(memory_tools)
        result = await tools["recall_raw"](mid, _ctx=make_ctx(channel))

        assert "demoted content here" in result, (
            f"recall_raw must work on demoted records; got: {result!r}"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.3 tool compartment scoping
# ---------------------------------------------------------------------------


class TestToolCompartmentScoping:
    """Both tools enforce _channel_scope; group siblings are in-scope."""

    async def test_search_memory_does_not_see_other_channel_memories(self):
        """Channel B cannot list channel A's memories via search_memory."""
        memory, memory_tools, db = await build_tools_env()
        channel_a = Channel(transport="irc", scope="#electronics")
        channel_b = Channel(transport="irc", scope="#secret")

        await seed_memory(
            db, channel_a.id, "channel A secret wifi password hunter2"
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "wifi password hunter2", _ctx=make_ctx(channel_b)
        )

        assert "hunter2" not in result, (
            f"channel B must not see channel A's memories via search_memory; "
            f"got: {result!r}"
        )

    async def test_recall_raw_rejects_out_of_scope_id_no_stat_bump(self):
        """recall_raw rejects an out-of-scope memory_id without bumping stats."""
        memory, memory_tools, db = await build_tools_env()
        channel_a = Channel(transport="irc", scope="#electronics")
        channel_b = Channel(transport="irc", scope="#secret")

        msg1 = await seed_message_log_row(db, channel_a.id, "user", "secret data")
        msg2 = await seed_message_log_row(db, channel_a.id, "assistant", "secret reply")
        mid_a = await seed_memory(
            db, channel_a.id, "channel A memory",
            retrieval_count=0, last_retrieved_at=None,
            msg_id_start=msg1, msg_id_end=msg2,
        )

        tools = get_tools(memory_tools)
        result = await tools["recall_raw"](mid_a, _ctx=make_ctx(channel_b))

        assert "error" in result.lower() or "scope" in result.lower(), (
            f"recall_raw must return error for out-of-scope id; got: {result!r}"
        )
        # Stats must NOT be bumped
        async with db.execute(
            "SELECT retrieval_count, last_retrieved_at FROM memory WHERE id = ?",
            (mid_a,),
        ) as c:
            count, last = await c.fetchone()
        assert count == 0, "out-of-scope recall_raw must not bump retrieval_count"
        assert last is None, "out-of-scope recall_raw must not set last_retrieved_at"
        await db.close()

    async def test_group_sibling_memories_visible_in_search_memory(self):
        """Group siblings share retrieval scope; a sibling's memory is visible."""
        memory, memory_tools, db = await build_tools_env(config=GROUP_CONFIG)
        channel_a = Channel(transport="irc", scope="#electronics")
        channel_b = Channel(transport="irc", scope="#garden")  # sibling in group

        sibling_mid = await seed_memory(
            db, channel_b.id, "kestrel solar charge MQTT garden power data sibling"
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "solar charge MQTT", _ctx=make_ctx(channel_a)
        )

        assert "solar" in result or str(sibling_mid) in result, (
            f"group-sibling memory must be visible via search_memory; got: {result!r}"
        )
        await db.close()

    async def test_group_sibling_recall_raw_works(self):
        """Group sibling's memory_id is in scope for recall_raw."""
        memory, memory_tools, db = await build_tools_env(config=GROUP_CONFIG)
        channel_a = Channel(transport="irc", scope="#electronics")
        channel_b = Channel(transport="irc", scope="#garden")

        msg1 = await seed_message_log_row(db, channel_b.id, "user", "sibling channel content")
        msg2 = await seed_message_log_row(db, channel_b.id, "assistant", "sibling reply")
        mid_b = await seed_memory(
            db, channel_b.id, "sibling memory about garden solar",
            msg_id_start=msg1, msg_id_end=msg2,
        )

        tools = get_tools(memory_tools)
        result = await tools["recall_raw"](mid_b, _ctx=make_ctx(channel_a))

        assert "error" not in result.lower(), (
            f"group sibling recall_raw must succeed; got: {result!r}"
        )
        assert "sibling channel content" in result, (
            f"recall_raw must return sibling channel's dialog; got: {result!r}"
        )
        await db.close()

    async def test_search_memory_channel_arg_outside_scope_returns_error(self):
        """search_memory(channel=<out-of-scope>) returns error naming visible channels."""
        memory, memory_tools, db = await build_tools_env()
        channel_a = Channel(transport="irc", scope="#electronics")

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "query", channel="irc:#other-channel", _ctx=make_ctx(channel_a)
        )

        assert "error" in result.lower() or "scope" in result.lower() or "visible" in result.lower(), (
            f"out-of-scope channel arg must return error string naming visible channels; "
            f"got: {result!r}"
        )
        # The error string should name the visible channels (the calling channel)
        assert "irc:#electronics" in result, (
            f"error string must name visible channels; got: {result!r}"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.3 stat semantics: recall_raw bumps, search_memory does not
# ---------------------------------------------------------------------------


class TestStatSemantics:
    """search_memory must not bump stats; recall_raw must bump them."""

    async def test_search_memory_does_not_bump_stats(self):
        """search_memory leaves retrieval_count and last_retrieved_at unchanged."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        mid = await seed_memory(
            db, channel.id, "kestrel ESP32 birdhouse firmware PlatformIO",
            retrieval_count=0, last_retrieved_at=None,
        )

        tools = get_tools(memory_tools)
        await tools["search_memory"]("ESP32 PlatformIO", _ctx=make_ctx(channel))

        async with db.execute(
            "SELECT retrieval_count, last_retrieved_at FROM memory WHERE id = ?", (mid,)
        ) as c:
            count, last = await c.fetchone()

        assert count == 0, "search_memory must not increment retrieval_count"
        assert last is None, "search_memory must not set last_retrieved_at"
        await db.close()

    async def test_recall_raw_bumps_stats_search_memory_does_not(self):
        """Two tools, two behaviors: recall_raw bumps, search_memory does not."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        msg1 = await seed_message_log_row(db, channel.id, "user", "kestrel stats test")
        msg2 = await seed_message_log_row(db, channel.id, "assistant", "reply stats")
        mid = await seed_memory(
            db, channel.id, "kestrel stats test memory",
            retrieval_count=0, last_retrieved_at=None,
            msg_id_start=msg1, msg_id_end=msg2,
        )

        tools = get_tools(memory_tools)
        # search_memory first
        await tools["search_memory"]("kestrel stats", _ctx=make_ctx(channel))
        async with db.execute(
            "SELECT retrieval_count FROM memory WHERE id = ?", (mid,)
        ) as c:
            assert (await c.fetchone())[0] == 0, (
                "retrieval_count must be 0 after search_memory"
            )

        # recall_raw second
        await tools["recall_raw"](mid, _ctx=make_ctx(channel))
        async with db.execute(
            "SELECT retrieval_count FROM memory WHERE id = ?", (mid,)
        ) as c:
            assert (await c.fetchone())[0] == 1, (
                "retrieval_count must be 1 after recall_raw"
            )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.3 message_fts backfill
# ---------------------------------------------------------------------------


class TestMessageFtsBackfill:
    """Backfill indexes pre-existing message_log rows exactly once.

    The backfill selects rows absent from message_fts (NOT IN rowid)
    so it is crash-safe: a rerun after partial completion picks up only
    missing rows without hitting FTS5 rowid uniqueness.
    """

    async def test_preexisting_rows_indexed_by_backfill(self):
        """Rows in message_log before MemoryPlugin starts are indexed by backfill."""
        # Build a fresh DB with message_log rows (no FTS yet)
        db = await aiosqlite.connect(":memory:")
        await init_db(db)

        # Insert messages before memory schema (no FTS triggers yet)
        ids = []
        for i in range(5):
            cursor = await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, ?)",
                ("irc:#test", json.dumps({"role": "user", "content": f"preexisting message {i}"}),
                 time.time(), "message"),
            )
            ids.append(cursor.lastrowid)
        await db.commit()

        # Now initialize the memory stack (schema + backfill)
        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")

        llm = LLMPlugin()
        pm.register(llm, name="llm")
        await llm.on_init(pm=pm, config=EMBED_CONFIG)
        llm._clients["embedding"] = StubEmbedClient()

        memory = MemoryPlugin()
        pm.register(memory, name="memory")
        await memory.on_init(pm=pm, config=EMBED_CONFIG)
        await memory.on_start(config=EMBED_CONFIG)
        await memory.wait_for_background_tasks()  # wait for backfill

        # message_fts must now contain the pre-existing rows
        async with db.execute("SELECT COUNT(*) FROM message_fts") as c:
            fts_count = (await c.fetchone())[0]

        assert fts_count >= len(ids), (
            f"backfill must index all {len(ids)} pre-existing message_log rows; "
            f"message_fts has {fts_count}"
        )
        await db.close()

    async def test_backfill_rerun_after_partial_completes_without_uniqueness_error(self):
        """Partial backfill crash-safe: rerun completes the remaining rows.

        The backfill uses 'NOT IN (SELECT rowid FROM message_fts)' so a rerun
        skips already-indexed rows and does not hit FTS5 rowid uniqueness error.
        (Design review cosmetic finding #3 — verified empirically, SQLite 3.47.1.)
        """
        db = await aiosqlite.connect(":memory:")
        await init_db(db)

        # Insert 10 messages
        all_ids = []
        for i in range(10):
            cursor = await db.execute(
                "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                "VALUES (?, ?, ?, ?)",
                ("irc:#test",
                 json.dumps({"role": "user", "content": f"crash-safe test message {i}"}),
                 time.time(), "message"),
            )
            all_ids.append(cursor.lastrowid)
        await db.commit()

        # Initialize memory to create schema (including message_fts)
        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        llm = LLMPlugin()
        pm.register(llm, name="llm")
        await llm.on_init(pm=pm, config=EMBED_CONFIG)
        llm._clients["embedding"] = StubEmbedClient()
        memory = MemoryPlugin()
        pm.register(memory, name="memory")
        await memory.on_init(pm=pm, config=EMBED_CONFIG)
        await memory.on_start(config=EMBED_CONFIG)
        await memory.wait_for_background_tasks()

        # Simulate partial backfill: manually insert only the first 5 rows into
        # message_fts, then delete them so we have a partial state.
        # (The real test: run backfill twice; second run must not raise.)
        # Actually, do it cleanly: delete half the fts rows, simulate crash mid-run.
        first_five = all_ids[:5]
        for rowid in first_five:
            await db.execute("DELETE FROM message_fts WHERE rowid = ?", (rowid,))
        await db.commit()

        # Rerun: a second on_start triggers the backfill task again
        memory2 = MemoryPlugin()
        await memory2.on_init(pm=pm, config=EMBED_CONFIG)
        await memory2.on_start(config=EMBED_CONFIG)
        await memory2.wait_for_background_tasks()  # must not raise uniqueness error

        # All rows must now be in message_fts
        async with db.execute("SELECT COUNT(*) FROM message_fts") as c:
            fts_count = (await c.fetchone())[0]

        assert fts_count == len(all_ids), (
            f"after crash-safe rerun, all {len(all_ids)} rows must be indexed; "
            f"got {fts_count}"
        )
        await db.close()

    async def test_fts_trigger_on_new_insert(self):
        """Newly inserted message_log rows are FTS-indexed via the insert trigger."""
        memory, memory_tools, db = await build_tools_env()

        # Insert after schema is in place (triggers are active)
        cursor = await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            ("irc:#test",
             json.dumps({"role": "user", "content": "unique phrase zorkblarg"}),
             time.time(), "message"),
        )
        new_id = cursor.lastrowid
        await db.commit()

        async with db.execute(
            "SELECT rowid FROM message_fts WHERE message_fts MATCH '\"zorkblarg\"'"
        ) as c:
            rows = await c.fetchall()
        assert any(r[0] == new_id for r in rows), (
            "insert trigger must index new message_log rows into message_fts"
        )
        await db.close()

    async def test_fts_trigger_on_update_message(self):
        """Updating message_log.message re-indexes the row (update trigger).

        The update trigger uses plain DELETE (not FTS5 'delete' command) because
        message_fts is a regular (non-external-content) FTS5 table.
        (Design review important finding #1 — fix verified empirically, SQLite 3.47.1.)
        """
        memory, memory_tools, db = await build_tools_env()

        cursor = await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, ?, ?)",
            ("irc:#test",
             json.dumps({"role": "user", "content": "original secret content"}),
             time.time(), "message"),
        )
        row_id = cursor.lastrowid
        await db.commit()

        # Tombstone the row (as redact would do)
        tombstone = json.dumps({"role": "user", "content": "[redacted by operator 2026-07-07]"})
        # This UPDATE must not raise "SQL logic error" (the bug fixed by using
        # plain DELETE instead of the FTS5 'delete' command in the trigger).
        await db.execute(
            "UPDATE message_log SET message = ? WHERE id = ?", (tombstone, row_id)
        )
        await db.commit()

        # Original content must no longer be findable
        async with db.execute(
            "SELECT rowid FROM message_fts WHERE message_fts MATCH '\"secret\"'"
        ) as c:
            rows = await c.fetchall()
        assert not any(r[0] == row_id for r in rows), (
            "original content must not be FTS-indexed after tombstone UPDATE"
        )

        # Tombstone text must be indexed
        async with db.execute(
            "SELECT rowid FROM message_fts WHERE message_fts MATCH '\"redacted\"'"
        ) as c:
            rows = await c.fetchall()
        assert any(r[0] == row_id for r in rows), (
            "tombstone text must be FTS-indexed after UPDATE trigger fires"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.3 tags filter (green fix #2 — 2026-07-07)
# ---------------------------------------------------------------------------


class TestTagsFilter:
    """search_memory tags= parameter must filter by topic_tags JSON array.

    Green fix report finding #3: tags parameter was accepted but never applied,
    returning unfiltered results when tags were specified.
    """

    async def _seed_with_tags(self, db, channel_id, summary, tags):
        """Seed a memory row with topic_tags as a JSON array."""
        cursor = await db.execute(
            "INSERT INTO memory (channel_id, created_at, summary, importance, "
            "msg_id_start, msg_id_end, embedded, indexed, topic_tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (channel_id, time.time(), summary, 0.5, 1, 2, 0, 1,
             json.dumps(tags)),
        )
        rowid = cursor.lastrowid
        await db.commit()
        return rowid

    async def test_tags_filter_returns_only_matching_records(self):
        """search_memory(tags=['electronics']) must exclude records without that tag.

        Both records share a common query keyword ('kestrel') but differ by tag.
        The tags filter must narrow to only the electronics-tagged one.
        """
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        elec_mid = await self._seed_with_tags(
            db, channel.id,
            "kestrel ESP32 firmware build power-save dropout electronics",
            ["electronics", "firmware"],
        )
        garden_mid = await self._seed_with_tags(
            db, channel.id,
            "kestrel solar panel rain gauge sensor garden mqtt",
            ["garden", "sensors"],
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "kestrel", tags=["electronics"], _ctx=make_ctx(channel)
        )

        # The electronics record must appear (has matching tag)
        assert str(elec_mid) in result or "ESP32" in result or "firmware" in result, (
            f"electronics-tagged record must appear with tags=['electronics']; got: {result!r}"
        )
        # The garden record must not appear (lacks 'electronics' tag)
        assert "rain gauge" not in result and "garden mqtt" not in result, (
            f"garden-tagged record must be excluded by tags=['electronics']; got: {result!r}"
        )
        await db.close()

    async def test_tags_filter_excludes_untagged_and_wrong_tagged_records(self):
        """Tags filter excludes records with no topic_tags or wrong tags."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        # Record with matching tag
        matching_mid = await self._seed_with_tags(
            db, channel.id, "solar charge controller MPPT",
            ["solar", "power"],
        )
        # Record with different tags
        nonmatching_mid = await self._seed_with_tags(
            db, channel.id, "solar charge MPPT algorithm different tags",
            ["network", "mqtt"],
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "solar charge MPPT", tags=["solar"], _ctx=make_ctx(channel)
        )

        assert str(matching_mid) in result or "controller" in result, (
            f"matching-tag record must appear; got: {result!r}"
        )
        assert str(nonmatching_mid) not in result or "different tags" not in result, (
            f"non-matching-tag record must not appear; got: {result!r}"
        )
        await db.close()

    async def test_tags_filter_none_returns_all_records(self):
        """tags=None (default) must not apply any tag filter — all records returned."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        mid1 = await self._seed_with_tags(
            db, channel.id, "ESP32 firmware update kestrel build",
            ["electronics"],
        )
        mid2 = await self._seed_with_tags(
            db, channel.id, "kestrel garden sensor mqtt firmware",
            ["garden"],
        )

        tools = get_tools(memory_tools)
        result = await tools["search_memory"](
            "kestrel firmware", tags=None, _ctx=make_ctx(channel)
        )

        # Both records should appear since tags=None disables filtering
        assert str(mid1) in result or "ESP32" in result or "electronics" in result, (
            f"first record must appear with tags=None; got: {result!r}"
        )
        assert str(mid2) in result or "garden" in result or "sensor" in result, (
            f"second record must appear with tags=None; got: {result!r}"
        )
        await db.close()


# ---------------------------------------------------------------------------
# WP1b.3 token cap (green fix #2 — 2026-07-07)
# ---------------------------------------------------------------------------


class TestTokenCap:
    """search_memory must be token-capped like recall_raw.

    Green fix report finding #4: search_memory had only a LIMIT 50 row cap,
    with no token/character cap. A broad query with 50 long summaries could
    inflate the agent's context window by ~2500 tokens.
    """

    async def test_search_memory_token_cap_truncates_oversized_result(self):
        """search_memory with max_tokens=<small> truncates results with a marker."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        # Seed many records with substantial summaries
        for i in range(15):
            await seed_memory(
                db, channel.id,
                f"kestrel wifi sensor firmware build ESP32 record {i} " + "word " * 30,
            )

        tools = get_tools(memory_tools)
        # max_tokens=1 is extremely small — guarantees truncation
        result = await tools["search_memory"](
            "kestrel wifi", max_tokens=1, _ctx=make_ctx(channel)
        )

        assert "[truncated" in result or "truncated" in result.lower(), (
            f"search_memory with max_tokens=1 must include truncation marker; got: {result!r}"
        )
        await db.close()

    async def test_search_memory_large_token_cap_returns_all(self):
        """search_memory with a large max_tokens returns all matched results."""
        memory, memory_tools, db = await build_tools_env()
        channel = Channel(transport="irc", scope="#electronics")

        # Seed a small number of records
        mids = []
        for i in range(3):
            mid = await seed_memory(
                db, channel.id,
                f"kestrel solar panel sensor record {i}",
            )
            mids.append(mid)

        tools = get_tools(memory_tools)
        # Large cap — all records must appear
        result = await tools["search_memory"](
            "kestrel solar panel", max_tokens=100000, _ctx=make_ctx(channel)
        )

        assert "[truncated" not in result, (
            "large max_tokens must not trigger truncation for a small result set"
        )
        for mid in mids:
            assert str(mid) in result, (
                f"record {mid} must appear with large max_tokens; got: {result!r}"
            )
        await db.close()
