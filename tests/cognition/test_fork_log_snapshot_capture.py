"""#1204: a snapshot (compaction) fork records the snapshot, not the live window.

`ff0eab6` changed the capture at memory.py:4219 to `snapshot.conv.messages` when a
snapshot is in play, because the compaction fork renders the pre-compaction window
while the live window has already moved on (#628 runs the fork in the background).
Alfred's own note on that commit: it is code, not proof, and the snapshot path had no
row test. This is that test.

For the assertion to mean anything the two block lists have to differ at record time,
so the live window grows by one message *while the fork is blocked on the gate*. If
the capture reads the live window it stores 5 digests; if it reads the snapshot it
stores 4 — and the fourth one is checked by content, not just by count.
"""

from __future__ import annotations

import copy

import aiosqlite
import pytest

from corvidae_cognition.fork_log import digest_of
from helpers import add_kernel_registry, window_of

from .test_memory_consolidation import (
    EMBED_CONFIG,
    build_consolidation_env,
    seed_dialog,
)
from .test_memory_fork import DIALOG
from .test_memory_turn_trigger import REPLY, GatedForkAgent

pytestmark = pytest.mark.timeout(60)

EXTRA_TURN = ("user", "a turn that arrived after compaction began")
DIALOG_5 = DIALOG + [EXTRA_TURN]
SNAPSHOT_MESSAGES = [c for _, c in DIALOG]          # what the compaction fork renders
LIVE_MESSAGES = [c for _, c in DIALOG_5]


async def _compaction_fork_blocked_on_the_gate(tmp_path):
    cfg = copy.deepcopy(EMBED_CONFIG)
    cfg["daemon"] = {"session_db": str(tmp_path / "sessions.db")}
    memory, persistence, _stub, channel, db = await build_consolidation_env(cfg)
    add_kernel_registry(memory.pm).channels[channel.id] = channel
    # Five rows persisted, only the first four in the window: the fifth is what
    # arrives while the fork is in flight. window_of truncates contents to the
    # id list, so growing the window needs a real id, not just more text.
    ids = await seed_dialog(persistence, channel, DIALOG_5)
    channel.conversation = window_of(channel.id, ids[:4], contents=SNAPSHOT_MESSAGES)

    agent = GatedForkAgent({"consolidation"}, text=REPLY)
    memory.pm.unregister(name="agent")
    memory.pm.register(agent, name="agent")
    await memory.wait_for_background_tasks()

    # Compaction fires: the fork is spawned in the background and blocks on the
    # gate, so nothing has been recorded yet.
    await memory.before_compaction(
        channel=channel, compacted_ids=ids[:4], cut=len(ids[:4])
    )

    # The fork is in flight, not finished: GatedForkAgent appends its call only
    # after the gate opens, so an empty call list here means nothing has been
    # recorded yet. Without this, a fork that completed early would record the
    # 4-block live window and the test would pass without exercising a divergence.
    assert agent.calls == [], "the compaction fork finished before the window moved"

    # The live window moves while the fork is in flight. This is the whole point:
    # the snapshot and the live window now describe different prompts.
    channel.conversation = window_of(channel.id, ids, contents=LIVE_MESSAGES)
    assert len(channel.conversation.messages) == 5, "the live window did not grow"
    agent.gate.set()
    await memory.wait_for_background_tasks()
    return memory, channel, db, agent


async def _close(*handles):
    for h in handles:
        await h.close()


async def test_the_snapshot_fork_records_the_snapshot_it_rendered(tmp_path):
    memory, channel, db, agent = await _compaction_fork_blocked_on_the_gate(tmp_path)
    assert len(agent.calls) == 1, "the compaction fork never ran; nothing was captured"

    log_db = await aiosqlite.connect(str(tmp_path / "sessions.db"))
    try:
        async with log_db.execute(
            "SELECT id, stage, point, block_count FROM fork_log"
            " WHERE stage = 'consolidation' ORDER BY id"
        ) as cur:
            rows = await cur.fetchall()
        assert rows, (
            "the snapshot path captured nothing; fork_log has no consolidation row"
        )
        fork_id = rows[-1][0]

        async with log_db.execute(
            "SELECT digest FROM fork_digest WHERE fork_id = ? ORDER BY ordinal", (fork_id,)
        ) as cur:
            stored = [bytes(r[0]) for r in await cur.fetchall()]
    finally:
        await _close(log_db, db)

    # The fake records the fork's own snapshot, so the comparison is against what
    # was actually sent rather than against a list this test assembled.
    sent = agent.calls[0]["snapshot"]
    assert sent is not None, "the fork ran without a snapshot; this is not the #628 path"
    assert [m["content"] for m in sent.conv.messages] == SNAPSHOT_MESSAGES, (
        "the snapshot the fork was sent is not the pre-compaction window this"
        " test set up"
    )
    expected = [digest_of(m) for m in sent.conv.messages]
    live = [digest_of(m) for m in channel.conversation.messages]
    assert len(expected) != len(live), (
        f"the snapshot ({len(expected)} blocks) and the live window ({len(live)} blocks)"
        " are the same length: the test cannot tell the two captures apart"
    )
    assert stored == expected, (
        f"row stores {len(stored)} digests; the fork rendered {len(expected)}"
        " — a capture that reads the live window describes a prompt that was never sent"
    )
    assert stored != live, "the row describes the live window, not the snapshot"
    # block_count is not checked against len(stored) here: both are written from
    # the same list in one call, so that equality cannot fail and would be a
    # decoration. The sibling consolidation test pins claim-vs-digest drift,
    # which has a cause (a partial digest write) rather than being tautological.
    assert memory._fork_log.write_failures == 0, "the capture counted a silent failure"
