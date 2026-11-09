"""Consolidation reply robustness (issue #655): an out-of-range importance
is absent, a wrong-typed participants is no participants, and entries are
embedded in chunks when the embedding server caps the batch size."""

import logging

import pytest

from .test_memory_consolidation import (
    THREE_ENTRIES,
    build_consolidation_env,
    drive_compaction_consolidation,
    entries_reply,
    fetch_stamped_memories,
)


@pytest.mark.parametrize("importance", [7, -0.5, "1e3"])
async def test_out_of_range_importance_reaches_the_prior_as_absent(importance):
    """R1: a rating outside [0, 1] (a 0-10 scale, a numeric string) is
    treated as absent, so the prior gets hint=None and the rubric runs."""
    memory, persistence, stub, channel, db = await build_consolidation_env()
    try:
        stub.embed_result = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]
        stub.chat_text = entries_reply(THREE_ENTRIES, importance=importance)
        await drive_compaction_consolidation(memory, persistence, channel)

        kwargs = memory.importance_prior.score.await_args.kwargs
        assert kwargs["hint"] is None
    finally:
        await db.close()


async def test_wrong_typed_participants_keeps_the_entries(caplog):
    """R2: participants that is neither a list nor a string reads as no
    participants with a warning naming the key; the entries are stored."""
    memory, persistence, stub, channel, db = await build_consolidation_env()
    try:
        stub.embed_result = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]
        stub.chat_text = entries_reply(THREE_ENTRIES, participants=5)
        caplog.set_level(logging.WARNING)
        caplog.clear()
        await drive_compaction_consolidation(memory, persistence, channel)

        rows = await fetch_stamped_memories(db)
        assert [r["summary"] for r in rows] == [n for n, _ in THREE_ENTRIES]
        assert {r["participants"] for r in rows} == {"[]"}
        warnings = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "participants" in r.getMessage()
        ]
        assert len(warnings) == 1, warnings
        assert "int" in warnings[0]
    finally:
        await db.close()


async def test_entries_over_the_server_batch_cap_are_all_embedded():
    """R3: with an embedder that rejects batches over M texts, a
    consolidation of N > M entries stores every entry embedded=1."""
    memory, persistence, stub, channel, db = await build_consolidation_env()
    try:
        cap = 2
        calls: list[int] = []

        async def capped_embed(texts, kind=None):
            # Reject an oversized batch as a server with a batch cap would.
            calls.append(len(texts))
            if len(texts) > cap:
                raise RuntimeError(f"batch of {len(texts)} exceeds the limit of {cap}")
            return [[1.0, float(i), 0.0, 0.0] for i, _ in enumerate(texts)]

        stub.embed = capped_embed
        stub.chat_text = entries_reply(THREE_ENTRIES)
        await drive_compaction_consolidation(memory, persistence, channel)

        rows = await fetch_stamped_memories(db)
        assert len(rows) == len(THREE_ENTRIES) > cap
        assert {r["embedded"] for r in rows} == {1}
    finally:
        await db.close()
