"""Catch-up raw-transcript fallback split into admissible records (issue #626).

After two unusable consolidation replies the segment's transcript is stored
as records that each fit the memory admission budget (``funnel.budgets.memory``,
else ``funnel.default_budget``), stamped ``mode = 'raw'``.
"""

import copy

import pytest

from corvidae.context import count_tokens

from .test_memory_consolidation import (
    EMBED_CONFIG,
    _SequenceChatClient,
    build_consolidation_env,
    fetch_stamped_memories,
    seed_dialog,
)

#: The smallest budget the tests use, small enough to force several pieces.
BUDGET = 40

#: A dialog well over BUDGET, with one message that alone exceeds it.
LONG_DIALOG = [
    ("user", "the weather station keeps dropping off wifi every night"),
    ("assistant", "try WIFI_PS_NONE and check the brownout detector"),
    ("user", " ".join(f"reading{i}=21.{i}C" for i in range(60))),
    ("assistant", "the readings look fine; it is the radio resetting"),
]

#: The longest line prefix a memory line gets before its summary.
WORST_PREFIX = "[moderate] (99999d) "


def config_with(funnel: dict) -> dict:
    """EMBED_CONFIG plus a funnel block."""
    config = copy.deepcopy(EMBED_CONFIG)
    config["funnel"] = funnel
    return config


@pytest.mark.parametrize("funnel", [
    {"budgets": {"memory": BUDGET}},
    {"default_budget": BUDGET},
    {"default_budget": 4096, "budgets": {"memory": BUDGET}},
], ids=["budgets.memory", "default_budget", "memory-over-default"])
async def test_raw_fallback_splits_to_the_memory_budget(funnel):
    """Two unusable replies on a segment larger than the budget: several
    records, each admissible as a memory line, together covering the
    transcript, sharing the range, stamped raw with the failed call's
    prompt and model."""
    memory, persistence, _stub, channel, db = await build_consolidation_env(
        config_with(funnel)
    )
    try:
        memory._llm()._clients["main"] = _SequenceChatClient(["nope", "still not json"])
        rowids = await seed_dialog(persistence, channel, LONG_DIALOG)
        await memory.on_compaction(
            channel=channel, summary_msg={}, retain_count=0, compacted_ids=rowids,
        )
        await memory.wait_for_background_tasks()

        rows = await fetch_stamped_memories(db)
        assert len(rows) > 1
        # Each fits the budget as the line the funnel would be offered.
        for r in rows:
            assert count_tokens(WORST_PREFIX + r["summary"]) <= BUDGET, r["summary"]
        # Together they cover the transcript, in order.
        joined = "".join(r["summary"] for r in rows).replace("\n", "")
        for role, content in LONG_DIALOG:
            assert f"{role}: {content}" in joined
        # One range, one stamp: mode raw, with the failed call's prompt and model.
        assert {(r["msg_id_start"], r["msg_id_end"]) for r in rows} == {(rowids[0], rowids[-1])}
        assert {r["mode"] for r in rows} == {"raw"}
        assert all(r["prompt_hash"] for r in rows)
        assert len({r["model"] for r in rows}) == 1 and rows[0]["model"]
    finally:
        await db.close()
