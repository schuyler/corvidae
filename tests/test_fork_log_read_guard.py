"""The two read-time chain guards, each pinned by the mutation that kills it (#1204).

`resolve_chain` refuses to hand back a partial list, because a short chain would
describe a prompt that was never sent. Since `daec8d6` refuses a wrong
`chain_from` at write time, the read-time paths are only reachable behind
`record()` — a row written by an older build, or one edited or damaged directly.
That is still the failure the table exists to prevent, so it gets tests.

There are two guards, not one, and they catch different damage:

    if own_rows and own_rows[0]["ordinal"] != row["chain_from"]: return None
    ...
    blocks = None if inherit is None or len(inherit) != base else inherit

The first fires when a row's own stored ordinals no longer start where the row
claims to start. The second fires when the parent cannot supply the number of
blocks the child claims. Writing tests for these was worth it mainly for what the
mutation runs showed: my first attempt to reach the second guard by rewriting
`chain_from` was caught by the *first* guard instead (deleting the second left
the whole fork_log suite green at 30 passed, 13:00:12Z), so each test below says
which guard it is holding up, and the third reaches the second guard the way it
would actually happen — a parent that lost its digest rows.
"""

from __future__ import annotations

import pytest

from corvidae_cognition.fork_log import ForkLog, digest_of

pytestmark = pytest.mark.timeout(30)

PARENT = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
]
CHILD = [*PARENT, {"role": "user", "content": "u2"}]


@pytest.fixture
async def log(tmp_path):
    log = ForkLog(str(tmp_path / "sessions.db"))
    await log.open()
    try:
        yield log
    finally:
        await log.close()


async def _chain_with_parent(log: ForkLog) -> tuple[int, int]:
    parent = await log.record(
        stage="appraisal",
        channel_id="internal:test",
        messages=PARENT,
        digests=[digest_of(b) for b in PARENT],
    )
    child = await log.record(
        stage="appraisal",
        channel_id="internal:test",
        messages=CHILD,
        # the appended-only shape; the whole-prompt shape is accepted too since
        # daec8d6, and both store one digest row for this child
        digests=[digest_of(b) for b in CHILD][3:],
        chain_parent=parent,
        chain_from=3,
    )
    return parent, child


async def _set_claim(log: ForkLog, fork_id: int, claim: int) -> None:
    db = await log._conn()
    await db.execute("UPDATE fork_log SET chain_from = ? WHERE id = ?", (claim, fork_id))
    await db.commit()


async def _drop_digest_rows(log: ForkLog, fork_id: int) -> None:
    db = await log._conn()
    await db.execute("DELETE FROM fork_digest WHERE fork_id = ?", (fork_id,))
    await db.commit()


async def test_an_honest_chain_still_resolves_after_all_this(log):
    """Control, not a guard test: it pins neither branch.

    With both read-time guards deleted it passes unchanged (13:07:52Z, 4 failed /
    1 passed). It is here so a stricter future guard that rejects an honest chain
    costs someone something.
    """
    _, child = await _chain_with_parent(log)
    chain = await log.resolve_chain(child)
    assert chain is not None and len(chain) == 4


@pytest.mark.parametrize("claim", [2, 5], ids=["claims-fewer", "claims-more"])
async def test_a_row_whose_ordinals_do_not_start_at_its_claim_is_refused(log, claim):
    """Guard 1: ``own_rows[0]["ordinal"] != chain_from``.

    Either direction is a lie about the rebuilt prompt. These two are the case
    where the guards are redundant: delete guard 1 and they still pass, because
    the rewritten claim also breaks the parent-length check (13:02:45Z, 34
    passed). They stay because they document the claim, not because they
    discriminate between the guards — the next two tests do that.
    """
    _, child = await _chain_with_parent(log)
    await _set_claim(log, child, claim)
    assert await log.resolve_chain(child) is None


async def test_a_root_whose_ordinals_skip_a_slot_is_refused(log):
    """Guard 1's own case: a chain root whose diginals start at 1, not 0.

    The parent-length guard cannot see this one. With `chain_from = 0` the walk
    starts from an empty list and never consults a parent, so a skipped slot
    produces a chain of the right *length* whose entries are shifted by one —
    digests that all exist, paired with the wrong positions. That is the shape
    only the ordinal check catches, which is why it is here rather than folded
    into the parametrized pair above.
    """
    root = await log.record(
        stage="appraisal",
        channel_id="internal:test",
        messages=PARENT,
        digests=[digest_of(b) for b in PARENT],
    )
    db = await log._conn()
    # Ordinals can't be shifted: fork_digest has a UNIQUE (fork_id, ordinal)
    # index, which is itself worth knowing. Losing the head row is the way the
    # chain ends up starting at 1.
    await db.execute(
        "DELETE FROM fork_digest WHERE fork_id = ? AND ordinal = 0", (root,)
    )
    await db.commit()
    assert await log.resolve_chain(root) is None


async def test_a_parent_that_lost_its_digest_rows_cannot_supply_the_claimed_base(log):
    """Guard 2: ``len(inherit) != base``.

    Rewriting `chain_from` cannot reach this branch — guard 1 sees that first —
    so it arrives as a parent whose chain got shorter while the child's claim
    stayed. Dropping the parent's digest rows is the stand-in for whatever did
    that in the wild; what matters is that the child returns None rather than a
    one-block chain describing no prompt anyone sent.
    """
    parent, child = await _chain_with_parent(log)
    await _drop_digest_rows(log, parent)
    assert await log.resolve_chain(child) is None


async def test_a_gap_after_the_first_ordinal_is_refused(log):
    """Measured hole (13:07:51Z): drop the middle digest row of a 3-block root and
    ``resolve_chain`` returns 2 blocks for a 3-block prompt, with the row still
    ``reconstructable = 1``. A partial list would describe a different prompt, so
    it is not returned - except when the gap is not first.

    Strict xfail on purpose: it passes and turns the suite red the moment the
    module checks ordinal contiguity, and the marker is then deleted.
    """
    root = await log.record(
        stage="appraisal",
        channel_id="internal:test",
        messages=PARENT,
        digests=[digest_of(b) for b in PARENT],
    )
    db = await log._conn()
    await db.execute("DELETE FROM fork_digest WHERE fork_id = ? AND ordinal = 1", (root,))
    await db.commit()
    assert await log.resolve_chain(root) is None


# A second measured hole is deliberately left untested here because no test can
# close it: a chained child that loses ALL its own digest rows resolves to its
# parent's chain - 3 blocks for a 4-block prompt, reconstructable = 1 (13:08:25Z,
# isolated database; an earlier shared-database run reported None and was wrong).
# A row whose prompt equals its parent's legitimately has zero own digests, so
# "no rows" and "lost rows" are indistinguishable from the chain alone. Closing
# it needs the row to remember how many blocks it claimed - store the chain
# length at write, compare it at read, which subsumes both guards above.


async def test_a_chain_shorter_than_the_row_claims_is_refused(log):
    """Hole B: a 4-block prompt that resolves to 3 must not look intact."""
    root = await log.record(
        stage="appraisal",
        channel_id="discord:1",
        messages=CHILD,
        digests=[digest_of(b) for b in CHILD],
    )
    db = await log._conn()
    # a digest row lost to damage or a partial write
    await db.execute(
        "DELETE FROM fork_digest WHERE fork_id = ? AND ordinal = ?", (root, 3)
    )
    await db.commit()
    assert await log.resolve_chain(root) is None


async def test_rows_written_before_block_count_existed_are_backfilled(tmp_path):
    """A defaulted column lies about old rows; the ladder backfills it."""
    import aiosqlite

    path = str(tmp_path / "legacy.db")
    log = ForkLog(path)
    await log.open()
    fork_id = await log.record(
        stage="appraisal",
        channel_id="discord:1",
        messages=CHILD,
        digests=[digest_of(b) for b in CHILD],
    )
    await log.close()

    # Rewind the row to what an older build wrote: no count, digests intact.
    db = await aiosqlite.connect(path)
    await db.execute("UPDATE fork_log SET block_count = 0 WHERE id = ?", (fork_id,))
    await db.commit()
    await db.close()

    log = ForkLog(path)
    await log.open()
    try:
        assert len(await log.resolve_chain(fork_id)) == len(CHILD)
    finally:
        await log.close()
