"""Passive retrieval skips this channel's records still wholly in its window
(issue #634; design: docs/memory-consolidation-incremental.md §5; PRD §3.7
Memory 4).

A record whose ``msg_id_start`` is at or past the oldest ``message_log`` id
in the current channel's window only echoes what the window already holds,
so ``before_agent_turn`` doesn't admit it. Other channels' records, records
straddling the window's oldest id, and every record when the window carries
no ids are retrieved as before. The exclusion acts before the cut to k.
"""
import time

import pytest

from corvidae_cognition.memory import VEC_OVERFETCH
from tests.cognition.test_memory_retrieval import (
    bow_embed,
    build_retrieval_env,
    context_messages,
)

# The window under test holds message_log ids WINDOW_START .. WINDOW_START+4.
WINDOW_START = 100


def _sqlite_vec():
    """The sqlite_vec module, or None in a build without it (#130)."""
    try:
        import sqlite_vec
    except ImportError:
        return None
    return sqlite_vec


async def seed(
    db, channel_id: str, summary: str, msg_id_start: int, msg_id_end: int,
) -> int:
    """Insert one memory row with an explicit message range.

    Writes the vec row too when sqlite-vec is available, so the same test
    exercises the vector path in the full build and FTS5 in the FTS-only one.
    """
    vec = _sqlite_vec()
    cursor = await db.execute(
        "INSERT INTO memory (channel_id, created_at, summary, importance, "
        "msg_id_start, msg_id_end, embedded) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (channel_id, time.time(), summary, 0.5, msg_id_start, msg_id_end,
         1 if vec else 0),
    )
    rowid = cursor.lastrowid
    if vec is not None:
        await db.execute(
            "INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
            (rowid, vec.serialize_float32(bow_embed(summary))),
        )
    await db.commit()
    return rowid


def fill_window(conv, with_ids: bool = True) -> None:
    """Put five earlier exchange messages in the window, tagged with
    message_log ids WINDOW_START.. unless ``with_ids`` is False."""
    for offset in range(5):
        message = {
            "role": "user" if offset % 2 == 0 else "assistant",
            "content": f"earlier message {offset}",
        }
        if with_ids:
            message["_db_id"] = WINDOW_START + offset
        conv.append(message)


async def ask(memory, channel, conv, text: str, db_id: int | None) -> None:
    """Append the inbound user message (persisted, like the turn loop does
    before before_agent_turn) and fire passive retrieval."""
    message = {"role": "user", "content": text}
    if db_id is not None:
        message["_db_id"] = db_id
    conv.append(message)
    await memory.before_agent_turn(channel=channel, trace_id=None, annotations={})


def admitted_text(conv) -> str:
    """Everything admitted as CONTEXT, joined."""
    return "\n".join(m["content"] for m in context_messages(conv))


class TestWindowExclusion:
    async def test_in_window_record_excluded_other_channel_kept(self):
        # R1: same channel, range wholly in the window -> not admitted;
        # the same content on another channel -> admitted.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed(db, channel.id,
                   "the rain gauge is on the north fence by the shed",
                   WINDOW_START + 1, WINDOW_START + 3)
        await seed(db, "irc:#garden",
                   "the rain gauge is on the north fence by the gate",
                   WINDOW_START + 1, WINDOW_START + 3)
        fill_window(conv)
        await ask(memory, channel, conv, "where is the rain gauge?", WINDOW_START + 5)
        text = admitted_text(conv)
        assert "by the gate" in text
        assert "by the shed" not in text
        await db.close()

    async def test_record_straddling_window_start_admitted(self):
        # R2: msg_id_start below the window's oldest id, end inside it.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed(db, channel.id,
                   "the rain gauge is on the north fence by the shed",
                   WINDOW_START - 3, WINDOW_START + 2)
        fill_window(conv)
        await ask(memory, channel, conv, "where is the rain gauge?", WINDOW_START + 5)
        assert "by the shed" in admitted_text(conv)
        await db.close()

    async def test_exclusion_applies_before_cut_to_k(self):
        # R3: more in-window records than the KNN over-fetch, all ranked
        # above the out-of-window ones; k out-of-window records still return.
        if _sqlite_vec() is None:
            pytest.skip("sqlite-vec unavailable; this test exercises the KNN cut")
        memory, embedder, channel, conv, db = await build_retrieval_env()
        k = 8
        query = "where is the rain gauge"
        in_window = {
            await seed(db, channel.id, query, WINDOW_START + 1, WINDOW_START + 2)
            for _ in range(k * VEC_OVERFETCH + 8)
        }
        out_of_window = {
            await seed(db, channel.id, f"rain gauge note {n} about wiring",
                       10 + n, 11 + n)
            for n in range(k)
        }
        candidates, degraded = await memory.retrieve(
            query, k=k, exclude=(channel.id, WINDOW_START),
        )
        assert not degraded
        ids = {c["id"] for c in candidates}
        assert not ids & in_window
        assert ids == out_of_window
        await db.close()

    async def test_window_without_ids_excludes_nothing(self):
        # R4: no _db_id anywhere in the window -> no exclusion.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed(db, channel.id,
                   "the rain gauge is on the north fence by the shed",
                   WINDOW_START + 1, WINDOW_START + 3)
        fill_window(conv, with_ids=False)
        await ask(memory, channel, conv, "where is the rain gauge?", None)
        assert "by the shed" in admitted_text(conv)
        await db.close()

    async def test_exclusion_holds_when_degraded_to_fts(self):
        # R6: encoder down -> FTS5; the in-window record is still excluded
        # and an out-of-window one on the same channel is admitted.
        memory, embedder, channel, conv, db = await build_retrieval_env()
        await seed(db, channel.id,
                   "the rain gauge is on the north fence by the shed",
                   WINDOW_START + 1, WINDOW_START + 3)
        await seed(db, channel.id,
                   "the rain gauge is on the north fence by the gate",
                   WINDOW_START - 20, WINDOW_START - 10)
        embedder.fail = True
        fill_window(conv)
        await ask(memory, channel, conv, "where is the rain gauge?", WINDOW_START + 5)
        text = admitted_text(conv)
        assert "by the gate" in text
        assert "by the shed" not in text
        await db.close()
