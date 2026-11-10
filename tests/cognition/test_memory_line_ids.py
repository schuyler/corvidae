"""Memory lines carry their record id; dedupe on the id (#632, #624).

A retrieved memory is admitted as ``(age, memory <id>) summary``, marked
``[marginal] `` in front when it banded moderate. The id is what the model
passes to ``recall_raw``, and memory hands the funnel ``memory <id>)`` as the
entry's dedupe key, so a record already in the window is not admitted again
when its age or band reads differently now.
"""

import re

from corvidae.channel import Channel
from corvidae.context import ContextWindow
from corvidae_cognition.memory import WORST_MEMORY_LINE_PREFIX, _memory_line

from .test_memory_retrieval import (
    EMBED_CONFIG,
    ask,
    build_retrieval_env,
    context_messages,
    seed_memory,
)
from .test_memory_tools import (
    build_tools_env,
    get_tools,
    seed_message_log_row,
)
from .test_memory_tools import seed_memory as seed_tools_memory

SUMMARY = "kestrel moved the rain gauge to the north fence"
QUERY = "kestrel moved the rain gauge to the north fence"


def banded(strong: float) -> dict:
    """EMBED_CONFIG banding on raw similarity: every measured record is at
    least moderate, and strong when its similarity reaches ``strong``."""
    return {**EMBED_CONFIG, "memory": {"retrieval": {
        "band_on": "similarity",
        "bands": {"strong": strong, "moderate": 0.0},
    }}}


# Every measured record bands strong, or moderate (strong unreachable).
STRONG = banded(0.0)
MARGINAL = banded(2.0)


def lines_of(conv) -> list[str]:
    """Every entry line of the window's CONTEXT rows."""
    return [
        line for m in context_messages(conv) for line in m["content"].splitlines()
    ]


async def outcome_of(db, memory_id: int) -> str:
    """The latest retrieval_candidate outcome logged for ``memory_id``."""
    async with db.execute(
        "SELECT outcome FROM retrieval_candidate WHERE memory_id = ? "
        "ORDER BY retrieval_id DESC LIMIT 1",
        (memory_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return row[0]


async def seed_before_window(db, channel_id: str, summary: str) -> int:
    """Seed a record whose message range (0-0) lies before any row the test
    puts in the window, so in-window exclusion (#634) never hides it from the
    funnel's dedupe (see test_retrieval_candidates.py for the same trap)."""
    rid = await seed_memory(db, channel_id, summary)
    await db.execute(
        "UPDATE memory SET msg_id_start = 0, msg_id_end = 0 WHERE id = ?", (rid,)
    )
    await db.commit()
    return rid


async def retrieval_count(db, memory_id: int) -> int:
    async with db.execute(
        "SELECT retrieval_count FROM memory WHERE id = ?", (memory_id,)
    ) as cursor:
        return (await cursor.fetchone())[0]


class TestLineFormat:
    async def test_strong_line_names_the_record_id(self):
        memory, _, channel, conv, db = await build_retrieval_env(STRONG)
        rid = await seed_memory(db, channel.id, SUMMARY)
        await ask(memory, channel, conv, QUERY)
        assert f"(now, memory {rid}) {SUMMARY}" in lines_of(conv)
        await db.close()

    async def test_marginal_line_names_the_record_id(self):
        memory, _, channel, conv, db = await build_retrieval_env(MARGINAL)
        rid = await seed_memory(db, channel.id, SUMMARY)
        await ask(memory, channel, conv, QUERY)
        assert f"[marginal] (now, memory {rid}) {SUMMARY}" in lines_of(conv)
        await db.close()

    def test_worst_prefix_is_the_longest_line_prefix(self):
        """Raw-fallback pieces are sized against WORST_MEMORY_LINE_PREFIX
        (#626), so it must be the prefix of a marginal line with a five-digit
        day count and a twelve-digit id."""
        worst = {
            "band": "moderate",
            "age_seconds": 99999 * 86400,
            "id": 999_999_999_999,
            "summary": "s",
        }
        assert _memory_line(worst) == WORST_MEMORY_LINE_PREFIX + "s"


class TestDedupeOnId:
    async def test_age_change_does_not_readmit(self):
        """A record already in the window under another age is a duplicate
        (#624): no new CONTEXT row, outcome dropped_dup, stats unmoved."""
        memory, _, channel, conv, db = await build_retrieval_env(STRONG)
        rid = await seed_before_window(db, channel.id, SUMMARY)
        funnel = memory.pm.get_plugin("funnel")
        # The record as an earlier turn admitted it, five hours ago.
        await funnel.admit(channel, conv, "memory", [f"(5h, memory {rid}) {SUMMARY}"])

        await ask(memory, channel, conv, QUERY)

        assert len(context_messages(conv)) == 1
        assert await outcome_of(db, rid) == "dropped_dup"
        assert await retrieval_count(db, rid) == 0
        await db.close()

    async def test_band_change_does_not_readmit(self):
        """Admitted as marginal earlier, strong now: still a duplicate."""
        memory, _, channel, conv, db = await build_retrieval_env(STRONG)
        rid = await seed_before_window(db, channel.id, SUMMARY)
        funnel = memory.pm.get_plugin("funnel")
        await funnel.admit(
            channel, conv, "memory", [f"[marginal] (3h, memory {rid}) {SUMMARY}"],
        )

        await ask(memory, channel, conv, QUERY)

        assert len(context_messages(conv)) == 1
        assert await outcome_of(db, rid) == "dropped_dup"
        await db.close()

    async def test_longer_id_in_window_does_not_block(self):
        """A window holding record <rid>0 doesn't make record <rid> look
        present: the key's closing paren anchors it."""
        memory, _, channel, conv, db = await build_retrieval_env(STRONG)
        rid = await seed_before_window(db, channel.id, SUMMARY)
        funnel = memory.pm.get_plugin("funnel")
        await funnel.admit(
            channel, conv, "memory", [f"(1h, memory {rid}0) something else"],
        )

        await ask(memory, channel, conv, QUERY)

        assert f"(now, memory {rid}) {SUMMARY}" in lines_of(conv)
        assert await outcome_of(db, rid) == "admitted"
        await db.close()


class TestRecallRawRoundTrip:
    async def test_id_on_admitted_line_recalls_the_record(self):
        """The id read off an admitted line is one recall_raw accepts, and it
        returns that record's range."""
        memory, memory_tools, db = await build_tools_env(STRONG)
        channel = Channel(transport="irc", scope="#garden")
        conv = ContextWindow(channel.id)
        channel.conversation = conv

        first = await seed_message_log_row(db, channel.id, "user", "where's the gauge?")
        last = await seed_message_log_row(
            db, channel.id, "assistant", "on the north fence now",
        )
        await seed_tools_memory(
            db, channel.id, SUMMARY, msg_id_start=first, msg_id_end=last,
        )

        await ask(memory, channel, conv, QUERY)
        found = [
            int(m.group(1)) for line in lines_of(conv)
            if (m := re.match(r"^(?:\[marginal\] )?\([^,)]+, memory (\d+)\) ", line))
        ]
        assert len(found) == 1, lines_of(conv)

        result = await get_tools(memory_tools)["recall_raw"](found[0])
        assert "user: where's the gauge?" in result
        assert "assistant: on the north fence now" in result
        await db.close()

    async def test_recall_raw_description_says_where_ids_appear(self):
        """Only the docstring's first line reaches the model as the tool
        description, so that line must name where memory ids appear."""
        _, memory_tools, db = await build_tools_env()
        tools = []
        memory_tools.register_tools(tools)
        schema = next(t.schema for t in tools if t.name == "recall_raw")
        description = schema["function"]["description"]
        assert "(age, memory <id>)" in description
        assert "search_memory" in description
        await db.close()
