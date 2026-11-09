"""Window exclusion after a cross-channel near-duplicate merge (issue #627).

A record that superseded another channel's record holds content the current
window doesn't, so passive retrieval on its channel must not exclude it while
its own range is still in the window.
"""

import pytest

from corvidae.channel import Channel

from .test_memory_dedup import (
    _consolidate,
    _same_summary,
    build_dedup_env,
    fetch_all_memories,
)

#: The query embeds to the same vector as the merged records' summary.
QUERY = "kestrel solar charge controller MQTT power data"


@pytest.fixture
async def dedup_env():
    """build_dedup_env's tuple; skips without sqlite-vec (merges need vectors)."""
    memory, persistence, stub, channel, db = await build_dedup_env()
    if not memory._vec_available:
        await db.close()
        pytest.skip("sqlite-vec unavailable")
    yield memory, persistence, stub, db
    await db.close()


async def test_cross_channel_merge_is_not_window_excluded(dedup_env):
    """R1: A on X; B on Y merges over A; retrieval on Y while B's range is in
    Y's window still returns B."""
    memory, persistence, stub, db = dedup_env
    _same_summary(stub)
    x = Channel(transport="irc", scope="#electronics")
    y = Channel(transport="irc", scope="#garden")
    await _consolidate(memory, persistence, x, [("user", "solar data"), ("assistant", "MQTT")])
    rows_y = await _consolidate(
        memory, persistence, y, [("user", "solar again"), ("assistant", "same MQTT")]
    )
    old, merged = await fetch_all_memories(db)
    assert old["superseded_by"] == merged["id"]

    # Y's window starts at B's first row: B's whole range is in it.
    hits, _degraded = await memory.retrieve(QUERY, exclude=(y.id, rows_y[0]))

    assert [h["id"] for h in hits] == [merged["id"]]


async def test_chain_of_merges_keeps_the_foreign_content(dedup_env):
    """R2: C on Y (same window) merges over B, which merged over A on X; C is
    still returned."""
    memory, persistence, stub, db = dedup_env
    _same_summary(stub)
    x = Channel(transport="irc", scope="#electronics")
    y = Channel(transport="irc", scope="#garden")
    await _consolidate(memory, persistence, x, [("user", "solar data"), ("assistant", "MQTT")])
    rows_y = await _consolidate(
        memory, persistence, y, [("user", "solar again"), ("assistant", "same MQTT")]
    )
    await _consolidate(
        memory, persistence, y, [("user", "solar thrice"), ("assistant", "MQTT yet again")]
    )
    *_superseded, latest = await fetch_all_memories(db)
    assert latest["superseded_by"] is None
    assert latest["msg_id_start"] == rows_y[0]

    hits, _degraded = await memory.retrieve(QUERY, exclude=(y.id, rows_y[0]))

    assert [h["id"] for h in hits] == [latest["id"]]


async def test_in_window_record_without_foreign_content_is_still_excluded(dedup_env):
    """R3: a same-channel record whose whole range is in the window, with no
    foreign merge behind it, is still excluded."""
    memory, persistence, stub, db = dedup_env
    _same_summary(stub)
    y = Channel(transport="irc", scope="#garden")
    rows_y = await _consolidate(
        memory, persistence, y, [("user", "solar data"), ("assistant", "MQTT")]
    )

    hits, _degraded = await memory.retrieve(QUERY, exclude=(y.id, rows_y[0]))

    assert hits == []
