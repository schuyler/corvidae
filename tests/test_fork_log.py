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
    assert row["prompt_sha256"] == prompt_sha256(MESSAGES)


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
    assert row["question_sha256"] == question_sha256(
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
    assert row["prompt_sha256"] == prompt_sha256(MESSAGES)
