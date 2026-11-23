"""fork_log: the content record of a fork (#1204)."""

from __future__ import annotations

import pytest

from corvidae_cognition.fork_log import (
    ForkLog,
    LABELS,
    prompt_sha256,
    question_sha256,
)

pytestmark = pytest.mark.timeout(30)

MESSAGES = [
    {"role": "system", "content": "you are a test"},
    {"role": "user", "content": "hello"},
]


@pytest.fixture
async def log(tmp_path):
    log = ForkLog(str(tmp_path / "sessions.db"))
    await log.open()
    try:
        yield log
    finally:
        await log.close()


async def _row(log: ForkLog, fork_id: int):
    db = await log._conn()
    async with db.execute("SELECT * FROM fork_log WHERE id = ?", (fork_id,)) as cur:
        return await cur.fetchone()


async def test_record_writes_the_row_and_its_prompt_hash(log):
    fork_id = await log.record(
        stage="appraisal",
        channel_id="discord:1",
        messages=MESSAGES,
        instruction="is this fine?",
        verdict="ok",
    )
    row = await _row(log, fork_id)
    assert row["stage"] == "appraisal"
    assert row["channel_id"] == "discord:1"
    assert row["instruction"] == "is this fine?"
    assert row["verdict"] == "ok"
    assert bytes(row["prompt_sha256"]) == prompt_sha256(MESSAGES)


async def test_the_hash_follows_the_rendered_text(log):
    a = prompt_sha256(MESSAGES)
    b = prompt_sha256([*MESSAGES, {"role": "user", "content": "more"}])
    assert a != b
    # key order inside a message does not change what was sent
    assert prompt_sha256([{"content": "x", "role": "user"}]) == prompt_sha256(
        [{"role": "user", "content": "x"}]
    )


async def test_the_question_is_stored_with_its_hash(log):
    fork_id = await log.record(
        stage="appraisal",
        channel_id="discord:1",
        messages=MESSAGES,
        question="does this draft commit us to a date?",
    )
    row = await _row(log, fork_id)
    assert row["question_text"] == "does this draft commit us to a date?"
    assert bytes(row["question_sha256"]) == question_sha256(
        "does this draft commit us to a date?"
    )


async def test_no_question_leaves_no_hash(log):
    fork_id = await log.record(
        stage="critique", channel_id="discord:1", messages=MESSAGES
    )
    row = await _row(log, fork_id)
    assert row["question_sha256"] is None


async def test_findings_are_stored_typed(log):
    finding = {
        "lens": "provenance",
        "class": "provenance",
        "text": "the date has no source",
        "referent": "line 3",
        "cited_records": [],
    }
    fork_id = await log.record(
        stage="critique",
        channel_id="discord:1",
        messages=MESSAGES,
        findings=[finding],
    )
    row = await _row(log, fork_id)
    assert finding["text"] in row["findings_json"]
    assert '"class":"provenance"' in row["findings_json"]


async def test_the_system_prompt_is_stored_once_per_version(log):
    for _ in range(3):
        await log.record(
            stage="appraisal",
            channel_id="discord:1",
            messages=MESSAGES,
            system_prompt="you are a test",
        )
    db = await log._conn()
    async with db.execute("SELECT count(*) AS n FROM fork_block") as cur:
        assert (await cur.fetchone())["n"] == 1


async def test_a_sampled_snapshot_round_trips(log):
    fork_id = await log.record(
        stage="appraisal",
        channel_id="discord:1",
        messages=MESSAGES,
        store_snapshot=True,
    )
    row = await _row(log, fork_id)
    stored = await log.read_block(row["snapshot_sha256"])
    assert stored is not None
    assert "hello" in stored


async def test_labels_are_the_terminal_set_only(log):
    fork_id = await log.record(
        stage="appraisal", channel_id="discord:1", messages=MESSAGES
    )
    await log.set_label(fork_id, "declined", reason="agent said no")
    row = await _row(log, fork_id)
    assert row["label"] == "declined"
    assert row["label_reason"] == "agent said no"
    assert LABELS == {"accepted", "declined", "superseded"}
    with pytest.raises(ValueError, match="unknown label"):
        await log.set_label(fork_id, "ignored")


async def test_counts_report_the_oldest_unlabeled_row(log):
    resolved = await log.record(
        stage="appraisal", channel_id="discord:1", messages=MESSAGES
    )
    await log.record(stage="appraisal", channel_id="discord:1", messages=MESSAGES)
    await log.set_label(resolved, "accepted")
    counts = {(r["label"]): r for r in await log.counts()}
    assert counts["accepted"]["n"] == 1
    assert counts["unlabeled"]["n"] == 1
    assert counts["unlabeled"]["oldest_unlabeled"] is not None


async def test_evict_drops_text_and_never_a_row(log):
    fork_ids = [
        await log.record(
            stage="appraisal",
            channel_id="discord:1",
            messages=MESSAGES,
            system_prompt="you are a test" * 200,
            store_snapshot=True,
        )
        for _ in range(4)
    ]
    await log.set_label(fork_ids[0], "declined")
    before = await log.bytes_used()
    assert before > 0

    freed = await log.evict(max_bytes=before // 2)

    assert freed > 0
    assert await log.bytes_used() <= before // 2
    db = await log._conn()
    async with db.execute("SELECT count(*) AS n FROM fork_log") as cur:
        assert (await cur.fetchone())["n"] == 4
    row = await _row(log, fork_ids[0])
    assert row["label"] == "declined"
    assert bytes(row["prompt_sha256"]) == prompt_sha256(MESSAGES)


async def _fork(log, blocks, *, parent=None, appended=None):
    """One fork whose full block list is ``blocks``.

    Only the appended tail is stored; ``parent`` supplies the prefix.
    """
    from corvidae_cognition.fork_log import digest_of

    chain_from = 0 if parent is None else len(blocks) - (appended or 1)
    return await log.record(
        stage="appraisal",
        channel_id="discord:1",
        messages=blocks,
        digests=[digest_of(b) for b in blocks[chain_from:]] if parent else [
            digest_of(b) for b in blocks
        ],
        chain_parent=parent,
        chain_from=chain_from,
    )


B = [{"role": "user", "content": t} for t in "abcdefg"]


async def test_a_chain_resolves_to_the_full_block_list(log):
    from corvidae_cognition.fork_log import digest_of

    first = await _fork(log, B[:2])
    second = await _fork(log, B[:3], parent=first, appended=1)

    assert await log.resolve_chain(first) == [digest_of(b) for b in B[:2]]
    assert await log.resolve_chain(second) == [digest_of(b) for b in B[:3]]


async def test_a_window_reset_starts_a_new_chain(log):
    from corvidae_cognition.fork_log import digest_of

    first = await _fork(log, B[:2])
    after_reset = await _fork(log, B[2:4])
    later = await _fork(log, B[2:5], parent=after_reset, appended=1)

    assert await log.resolve_chain(after_reset) == [digest_of(b) for b in B[2:4]]
    assert await log.resolve_chain(later) == [digest_of(b) for b in B[2:5]]
    assert await log.resolve_chain(first) == [digest_of(b) for b in B[:2]]


async def test_an_evicted_ancestor_makes_its_descendants_unreconstructable(log):
    first = await _fork(log, B[:2])
    second = await _fork(log, B[:3], parent=first, appended=1)
    await log.mark_unreconstructable([first])

    assert await log.resolve_chain(second) is None
    assert await log.resolve_chain(first) is None


async def test_a_row_without_digests_cannot_claim_reconstructability(log):
    fork_id = await log.record(
        stage="appraisal", channel_id="discord:1", messages=MESSAGES
    )
    row = await _row(log, fork_id)
    assert row["reconstructable"] == 0
    assert await log.resolve_chain(fork_id) is None


async def test_a_wrong_chain_offset_is_refused(log):
    from corvidae_cognition.fork_log import digest_of

    first = await _fork(log, B[:2])
    with pytest.raises(ValueError, match="does not resolve"):
        await log.record(
            stage="appraisal",
            channel_id="discord:1",
            messages=B[:3],
            digests=[digest_of(B[2])],
            chain_parent=first,
            chain_from=7,
        )


async def test_digests_that_do_not_describe_the_blocks_are_refused(log):
    from corvidae_cognition.fork_log import digest_of

    first = await _fork(log, B[:2])
    with pytest.raises(ValueError, match="do not describe"):
        await log.record(
            stage="appraisal",
            channel_id="discord:1",
            messages=B[:3],
            digests=[digest_of("not the block")],
            chain_parent=first,
            chain_from=2,
        )


async def _blocks(log):
    db = await log._conn()
    async with db.execute(
        "SELECT sha256, length(text) AS n FROM fork_block ORDER BY created_at"
    ) as cur:
        return list(await cur.fetchall())


async def test_the_budget_counts_what_eviction_subtracts(log):
    # The budget is stated in stored bytes, independently of bytes_used(): if
    # the loop subtracts a different quantity from the one it measured, one
    # drop frees almost nothing and the loop deletes every block (reported by
    # Lex, 2026-11-23).
    for n in range(3):
        await log.record(
            stage="appraisal",
            channel_id="discord:1",
            messages=MESSAGES,
            system_prompt=f"version {n}\n" + "a line of a system prompt\n" * 800,
        )
    blocks = await _blocks(log)
    assert len(blocks) == 3
    stored = sum(b["n"] for b in blocks)
    await log.evict(max_bytes=stored - 1)

    assert len(await _blocks(log)) == 2


async def test_only_the_rows_that_named_the_dropped_text_are_stranded(log):
    import os

    from corvidae_cognition.fork_log import digest_of

    ids = [
        await log.record(
            stage="appraisal",
            channel_id="discord:1",
            messages=MESSAGES,
            digests=[digest_of(b) for b in MESSAGES],
            system_prompt=os.urandom(2048).hex(),
        )
        for _ in range(2)
    ]
    before = {b["sha256"] for b in await _blocks(log)}
    assert len(before) == 2
    await log.evict(max_bytes=sum(b["n"] for b in await _blocks(log)) - 1)

    after = {b["sha256"] for b in await _blocks(log)}
    assert len(after) == 1
    db = await log._conn()
    for fork_id in ids:
        async with db.execute(
            "SELECT system_sha256, reconstructable FROM fork_log WHERE id = ?",
            (fork_id,),
        ) as cur:
            row = await cur.fetchone()
        kept = row["system_sha256"] in after
        assert row["reconstructable"] == (1 if kept else 0)


async def test_a_row_names_the_system_text_it_used(log):
    fork_id = await log.record(
        stage="appraisal",
        channel_id="discord:1",
        messages=MESSAGES,
        system_prompt="you are a test",
    )
    row = await _row(log, fork_id)
    assert row["system_sha256"] is not None
    assert await log.read_block(row["system_sha256"]) == "you are a test"


async def test_the_whole_prompt_list_is_accepted_for_a_chained_row(log):
    from corvidae_cognition.fork_log import digest_of

    first = await _fork(log, B[:2])
    fork_id = await log.record(
        stage="appraisal",
        channel_id="discord:1",
        messages=B[:3],
        digests=[digest_of(b) for b in B[:3]],
        chain_parent=first,
        chain_from=2,
    )
    assert await log.resolve_chain(fork_id) == [digest_of(b) for b in B[:3]]
    db = await log._conn()
    async with db.execute(
        "SELECT count(*) AS n FROM fork_digest WHERE fork_id = ?", (fork_id,)
    ) as cur:
        assert (await cur.fetchone())["n"] == 1  # the tail only, not the prompt


async def test_a_wrong_length_says_length_not_content(log):
    from corvidae_cognition.fork_log import digest_of

    first = await _fork(log, B[:2])
    with pytest.raises(ValueError, match="wrong length"):
        await log.record(
            stage="appraisal",
            channel_id="discord:1",
            messages=B[:4],
            digests=[digest_of(B[3])],
            chain_parent=first,
            chain_from=2,
        )
