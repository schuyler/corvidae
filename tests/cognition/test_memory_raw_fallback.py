"""Catch-up raw-transcript fallback split into admissible records (issue #626).

After two unusable consolidation replies the segment's transcript is stored
as records that each fit the entry size target (``memory.entry_target_tokens``,
#1176), stamped ``mode = 'raw'``. That the pieces follow the target and not the
funnel's budgets is pinned by ``test_memory_contribution.py``.
"""

import copy

import pytest
from corvidae_cognition.memory import WORST_MEMORY_LINE_PREFIX, prompt_hash

from corvidae.context import count_tokens

from .test_memory_consolidation import (
    EMBED_CONFIG,
    _SequenceChatClient,
    build_consolidation_env,
    fetch_stamped_memories,
    seed_dialog,
)

#: The entry target the test uses, small enough to force several pieces.
BUDGET = 40

#: A dialog well over the target, with one message that alone exceeds it.
LONG_DIALOG = [
    ("user", "the weather station keeps dropping off wifi every night"),
    ("assistant", "try WIFI_PS_NONE and check the brownout detector"),
    ("user", " ".join(f"reading{i}=21.{i}C" for i in range(60))),
    ("assistant", "the readings look fine; it is the radio resetting"),
]

#: The longest line prefix a memory line gets before its summary.
WORST_PREFIX = WORST_MEMORY_LINE_PREFIX


def config_with_target(target: int) -> dict:
    """EMBED_CONFIG with ``memory.entry_target_tokens`` set."""
    config = copy.deepcopy(EMBED_CONFIG)
    config["memory"]["entry_target_tokens"] = target
    return config


async def test_raw_fallback_splits_to_the_entry_target():
    """Two unusable replies on a segment larger than the target: several
    records, each admissible as a memory line, together covering the
    transcript, sharing the range, stamped raw with the failed call's
    prompt and model."""
    memory, persistence, _stub, channel, db = await build_consolidation_env(
        config_with_target(BUDGET)
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
        # Each fits the target as the line the store would be offered.
        for r in rows:
            assert count_tokens(WORST_PREFIX + r["summary"]) <= BUDGET, r["summary"]
        # Together they are the transcript, in order, nothing repeated or
        # lost (line breaks between pieces aside).
        joined = "".join(r["summary"] for r in rows).replace("\n", "")
        transcript = "".join(f"{role}: {content}" for role, content in LONG_DIALOG)
        assert joined == transcript
        # One range, one stamp: mode raw, with the failed call's prompt,
        # model and trigger.
        assert {(r["msg_id_start"], r["msg_id_end"]) for r in rows} == {(rowids[0], rowids[-1])}
        assert {r["mode"] for r in rows} == {"raw"}
        assert {r["prompt_hash"] for r in rows} == {prompt_hash(memory._consolidation_fork_prompt)}
        assert len({r["model"] for r in rows}) == 1 and rows[0]["model"]
        assert {r["trigger"] for r in rows} == {"compaction"}
    finally:
        await db.close()
