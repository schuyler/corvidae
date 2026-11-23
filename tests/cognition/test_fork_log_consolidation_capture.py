"""#1204: consolidation's capture lands, end to end.

Alfred wired `memory.py` to the shared `ForkRecorder` at `14ed749` and said plainly
that none of the 221 passing tests asserts a consolidation row lands. This is the
test that does: drive the idle trigger the way `test_memory_fork.py` does, then read
the table back.

It goes through the real path — `memory._consolidate_range` with a fork-capable
agent — because that is the only way the assertion covers the *call site* rather than
`record()`. Two things it pins that a "did a row appear" check would not: the row's
`block_count` must equal the digests it actually stored (a claim the read path will
refuse if they drift), and `resolve_chain` must hand the blocks back.

One trap worth naming, because it cost me a run: the harness's own handle is an
**in-memory persistence database**, while `fork_log` lives in the daemon's session
db. Querying the handle the builder returns finds no such table. So this test passes
an explicit `daemon.session_db` under `tmp_path` and opens that file separately. The
builder's default config omits the key, in which case `ForkRecorder` falls back to
`"sessions.db"` in the process CWD — which is how my first run silently wrote a
corpus into the repo root (gitignored, but not what a test should do).
"""

from __future__ import annotations

import aiosqlite
import pytest

from helpers import FakeForkAgent, add_kernel_registry, window_of

from .test_memory_consolidation import (
    EMBED_CONFIG,
    build_consolidation_env,
    seed_dialog,
)
from .test_memory_fork import DIALOG, FORK_JSON

pytestmark = pytest.mark.timeout(60)


async def _consolidate_one_window(tmp_path):
    cfg = {**EMBED_CONFIG, "daemon": {"session_db": str(tmp_path / "sessions.db")}}
    memory, persistence, _stub, channel, db = await build_consolidation_env(config=cfg)
    add_kernel_registry(memory.pm).channels[channel.id] = channel
    ids = await seed_dialog(persistence, channel, DIALOG)
    channel.conversation = window_of(channel.id, ids, contents=[c for _, c in DIALOG])
    agent = FakeForkAgent({"consolidation"}, text=FORK_JSON)
    # the builder registers a forks-off agent; swap in one that can fork
    if memory.pm.get_plugin("agent") is not None:
        memory.pm.unregister(name="agent")
    memory.pm.register(agent, name="agent")

    await memory._consolidate_range(channel.id, ids[-1], "idle")
    await memory.wait_for_background_tasks()
    return memory, agent


async def test_the_consolidation_fork_writes_a_fork_log_row(tmp_path):
    memory, agent = await _consolidate_one_window(tmp_path)
    # Supporting, not load-bearing: both hold even if the capture were deleted.
    # They pin the premise — a fork really ran, and nothing was silently swallowed.
    assert len(agent.calls) == 1, "the fork the capture hangs off never ran"

    log_db = await aiosqlite.connect(str(tmp_path / "sessions.db"))
    try:
        async with log_db.execute(
            "SELECT id, stage, point, block_count, reconstructable, chain_from"
            " FROM fork_log ORDER BY id"
        ) as cur:
            rows = await cur.fetchall()
        assert rows, "fork_log is empty: consolidation captured nothing"
        points = {(r[1], r[2]) for r in rows}
        assert ("consolidation", "consolidation") in points, (
            f"no consolidation row; fork_log holds {points}"
        )

        # Not rows[0]: the table can hold other stages' rows, and a claim checked
        # against the wrong row proves nothing.
        fork_id, stage, point, block_count, reconstructable, chain_from = next(
            r for r in rows if (r[1], r[2]) == ("consolidation", "consolidation")
        )
        assert (stage, point) == ("consolidation", "consolidation")
        assert reconstructable == 1, (
            "a consolidation row that cannot be rebuilt is a counter, not a capture"
        )
        assert chain_from == 0, "the first fork on a fresh channel has no prefix to claim"
        assert block_count == len(DIALOG), (
            f"row claims {block_count} blocks, the window held {len(DIALOG)}"
        )

        async with log_db.execute(
            "SELECT COUNT(*) FROM fork_digest WHERE fork_id = ?", (fork_id,)
        ) as cur:
            digests = (await cur.fetchone())[0]
        assert digests == block_count, (
            f"row claims {block_count} blocks and stores {digests} digests;"
            " resolve_chain refuses a row whose claim and digests disagree"
        )
    finally:
        await log_db.close()

    chain = await memory._fork_log.log.resolve_chain(fork_id)
    assert chain is not None and len(chain) == block_count, (
        f"resolve_chain gave {None if chain is None else f'{len(chain)} blocks'}"
        f" for a {block_count}-block consolidation row"
    )
    assert memory._fork_log.write_failures == 0, "the capture counted a silent failure"


async def test_the_recorder_closes_cleanly(tmp_path):
    """Not a capture check: the recorder opened a file handle on a real database,
    and the sibling harnesses close theirs. A leaked handle is how a WAL file ends
    up locked for the next test in the process."""
    memory, _agent = await _consolidate_one_window(tmp_path)
    await memory._fork_log.close()
