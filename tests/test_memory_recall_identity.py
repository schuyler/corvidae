"""Tests for memory recall under a persistent identity (GitHub issue #52).

- memory.retrieval.band_on: similarity|score — what the relevance band is
  computed from. Ranking always uses the decayed score.
- memory.retrieval.recency_from: created|last_used — where ranking age is
  measured from. Displayed age (age_seconds) always measures from created_at.
- Consolidated records are dated from their source messages, not from the
  time consolidation ran.
"""

import json
import logging
import math
import time

import pytest

from corvidae.memory import FTS_SIMILARITY_PROXY

from test_memory_consolidation import build_consolidation_env, seed_dialog
from test_memory_retrieval import EMBED_CONFIG, build_retrieval_env, seed_memory

DAY = 86400.0
SUMMARY = "kestrel's ESP32 weather station dropped off wifi"


def _config(**retrieval) -> dict:
    """EMBED_CONFIG with memory.retrieval keys set."""
    return {**EMBED_CONFIG, "memory": {"retrieval": retrieval}}


@pytest.fixture
async def env():
    """Factory for a retrieval environment; closes every DB on teardown
    (a leaked aiosqlite thread hangs interpreter exit when a test fails)."""
    dbs = []

    async def _make(config=None):
        built = await build_retrieval_env(config)
        dbs.append(built[-1])
        return built

    yield _make
    for db in dbs:
        await db.close()


async def _set_last_retrieved(db, memory_id: int, ts: float | None) -> None:
    await db.execute(
        "UPDATE memory SET last_retrieved_at = ? WHERE id = ?", (ts, memory_id)
    )
    await db.commit()


# ---------------------------------------------------------------------------
# band_on
# ---------------------------------------------------------------------------


class TestBandOn:
    async def test_default_bands_on_decayed_score(self, env):
        """Today's behavior: a 60-day-old exact match bands weak."""
        memory, _, channel, _, db = await env()
        await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        [c], _ = await memory.retrieve(channel.id, SUMMARY)
        assert c["similarity"] == pytest.approx(1.0, abs=1e-4)
        assert c["score"] == pytest.approx(math.exp(-2), rel=1e-3)
        assert c["band"] == "weak"

    async def test_similarity_bands_on_raw_similarity(self, env):
        memory, _, channel, _, db = await env(_config(band_on="similarity"))
        await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        [c], _ = await memory.retrieve(channel.id, SUMMARY)
        assert c["band"] == "strong"
        # The score still decays — only the band changes.
        assert c["score"] == pytest.approx(math.exp(-2), rel=1e-3)

    async def test_similarity_still_ranks_by_score(self, env):
        """Two equally similar memories: the newer one ranks first."""
        memory, _, channel, _, db = await env(_config(band_on="similarity"))
        old = await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        new = await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 1 * DAY)
        candidates, _ = await memory.retrieve(channel.id, SUMMARY)
        assert [c["id"] for c in candidates] == [new, old]
        assert all(c["band"] == "strong" for c in candidates)

    async def test_similarity_in_fts_degraded_mode_bands_on_proxy(self, env):
        """Keyword hits carry the similarity proxy, so they band on it."""
        memory, embedder, channel, _, db = await env(_config(band_on="similarity"))
        await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        embedder.fail = True
        [c], degraded = await memory.retrieve(channel.id, "weather station wifi")
        assert degraded
        assert c["similarity"] == FTS_SIMILARITY_PROXY
        assert c["band"] == "moderate"

    async def test_invalid_band_on_warns_and_defaults(self, env, caplog):
        with caplog.at_level(logging.WARNING):
            memory, _, channel, _, db = await env(_config(band_on="vibes"))
        assert "band_on" in caplog.text
        await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        [c], _ = await memory.retrieve(channel.id, SUMMARY)
        assert c["band"] == "weak"


# ---------------------------------------------------------------------------
# recency_from
# ---------------------------------------------------------------------------


class TestRecencyFrom:
    async def test_default_ignores_last_retrieved(self, env):
        memory, _, channel, _, db = await env()
        mid = await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        await _set_last_retrieved(db, mid, time.time())
        [c], _ = await memory.retrieve(channel.id, SUMMARY)
        assert c["score"] == pytest.approx(math.exp(-2), rel=1e-3)

    async def test_last_used_ranks_from_last_retrieval(self, env):
        memory, _, channel, _, db = await env(_config(recency_from="last_used"))
        mid = await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        await _set_last_retrieved(db, mid, time.time() - 1 * DAY)
        [c], _ = await memory.retrieve(channel.id, SUMMARY)
        assert c["score"] == pytest.approx(math.exp(-1 / 30), rel=1e-3)
        # Displayed age still measures from created_at.
        assert c["age_seconds"] == pytest.approx(60 * DAY, rel=1e-3)
        assert c["band"] == "strong"

    async def test_last_used_never_retrieved_falls_back_to_created(self, env):
        memory, _, channel, _, db = await env(_config(recency_from="last_used"))
        await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        [c], _ = await memory.retrieve(channel.id, SUMMARY)
        assert c["score"] == pytest.approx(math.exp(-2), rel=1e-3)

    async def test_last_used_applies_in_fts_degraded_mode(self, env):
        memory, embedder, channel, _, db = await env(_config(recency_from="last_used"))
        mid = await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        await _set_last_retrieved(db, mid, time.time())
        embedder.fail = True
        [c], degraded = await memory.retrieve(channel.id, "weather station wifi")
        assert degraded
        assert c["score"] == pytest.approx(FTS_SIMILARITY_PROXY, rel=1e-3)

    async def test_invalid_recency_from_warns_and_defaults(self, env, caplog):
        with caplog.at_level(logging.WARNING):
            memory, _, channel, _, db = await env(_config(recency_from="never"))
        assert "recency_from" in caplog.text
        mid = await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 60 * DAY)
        await _set_last_retrieved(db, mid, time.time())
        [c], _ = await memory.retrieve(channel.id, SUMMARY)
        assert c["score"] == pytest.approx(math.exp(-2), rel=1e-3)

    async def test_both_settings_together(self, env):
        """The Lex configuration: an old memory in recent use reads strong."""
        memory, _, channel, _, db = await env(
            _config(band_on="similarity", recency_from="last_used")
        )
        mid = await seed_memory(db, channel.id, SUMMARY, created_at=time.time() - 180 * DAY)
        await _set_last_retrieved(db, mid, time.time() - 2 * DAY)
        [c], _ = await memory.retrieve(channel.id, SUMMARY)
        assert c["band"] == "strong"
        assert c["age_seconds"] == pytest.approx(180 * DAY, rel=1e-3)


# ---------------------------------------------------------------------------
# Source-dated records
# ---------------------------------------------------------------------------


class TestSourceDatedRecords:
    async def test_record_dated_from_last_source_message(self):
        memory, persistence, _, channel, db = await build_consolidation_env()
        try:
            rowids = await seed_dialog(persistence, channel, [
                ("user", "my weather station drops off wifi"),
                ("assistant", "did you set WIFI_PS_NONE?"),
                ("user", "that fixed it!"),
            ])
            # Backdate the source messages: the conversation happened in April.
            april = time.time() - 180 * DAY
            for offset, rowid in enumerate(rowids):
                await db.execute(
                    "UPDATE message_log SET timestamp = ? WHERE id = ?",
                    (april + offset * 60, rowid),
                )
            await db.commit()

            await memory._consolidate_range(channel.id, rowids[-1], "test")

            async with db.execute("SELECT created_at FROM memory") as cursor:
                [(created_at,)] = await cursor.fetchall()
            assert created_at == pytest.approx(april + 120)
        finally:
            await db.close()

    async def test_range_ending_in_context_row_uses_its_timestamp(self):
        """The range end is dated even if it is not a dialog message."""
        memory, persistence, _, channel, db = await build_consolidation_env()
        try:
            rowids = await seed_dialog(persistence, channel, [
                ("user", "my weather station drops off wifi"),
                ("assistant", "did you set WIFI_PS_NONE?"),
            ])
            from corvidae.context import MessageType
            ctx_id = await persistence.on_conversation_event(
                channel=channel,
                message={"role": "system", "content": "[CONTEXT]"},
                message_type=MessageType.CONTEXT,
            )
            # Dialog a day earlier than the trailing CONTEXT row, so dating
            # from the last dialog row instead of the range end would show.
            then = time.time() - 30 * DAY
            await db.execute(
                "UPDATE message_log SET timestamp = ? WHERE id IN (?, ?)",
                (then - DAY, *rowids),
            )
            await db.execute(
                "UPDATE message_log SET timestamp = ? WHERE id = ?", (then, ctx_id),
            )
            await db.commit()

            await memory._consolidate_range(channel.id, ctx_id, "test")

            async with db.execute("SELECT created_at FROM memory") as cursor:
                [(created_at,)] = await cursor.fetchall()
            assert created_at == pytest.approx(then, abs=1.0)
        finally:
            await db.close()

    async def test_near_dup_merge_keeps_latest_date(self):
        """A backdated record that supersedes a live one carries the later
        date: the merged memory spans both conversations, and the latest of
        them is when it was last true."""
        from test_memory_dedup import _NEAR_IDENTICAL_SUMMARY, build_dedup_env
        from test_memory_dedup import seed_dialog as seed_dedup_dialog

        memory, persistence, stub, channel, db = await build_dedup_env()
        try:
            if not memory._vec_available:
                pytest.skip("sqlite-vec unavailable")
            stub.chat_text = json.dumps({
                "summary": _NEAR_IDENTICAL_SUMMARY, "topic_tags": [], "participants": [],
            })
            stub.fixed_embeddings[_NEAR_IDENTICAL_SUMMARY[:20]] = [1.0, 0.0, 0.0, 0.0]

            # A live conversation consolidates first, dated now.
            live = await seed_dedup_dialog(persistence, channel, [
                ("user", "live segment"), ("assistant", "live reply"),
            ])
            await memory._consolidate_range(channel.id, live[-1], "test")
            async with db.execute("SELECT created_at FROM memory") as cursor:
                [(live_created,)] = await cursor.fetchall()

            # Imported April history arrives later (higher ids, older timestamps).
            seeded = await seed_dedup_dialog(persistence, channel, [
                ("user", "seeded segment"), ("assistant", "seeded reply"),
            ])
            april = time.time() - 180 * DAY
            await db.execute(
                "UPDATE message_log SET timestamp = ? WHERE id IN (?, ?)", (april, *seeded),
            )
            await db.commit()
            await memory._consolidate_range(channel.id, seeded[-1], "test")

            async with db.execute(
                "SELECT created_at FROM memory WHERE superseded_by IS NULL"
            ) as cursor:
                [(survivor_created,)] = await cursor.fetchall()
            assert survivor_created == pytest.approx(live_created)
        finally:
            await db.close()
