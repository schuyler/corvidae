"""#648 raw transcript access: ``search_transcripts`` and ``recall_raw`` by range.

One test per acceptance criterion (R1-R9) of the approved requirements. Rows
are written to ``message_log`` so the real ``message_log_ai`` trigger indexes
them into ``message_fts``, as production does.
"""

import json
import logging
import time

import pytest

from corvidae.commands.redact import redact_messages
from corvidae.tool import declares_untrusted_output

from cognition.test_memory_tools import (
    build_tools_env,
    get_tools,
    seed_memory,
    seed_message_log_row,
)
from helpers import insert_raw_message, skip_warnings

# Fixed instants so timestamp assertions are exact (UTC).
TS_EARLY = 1699999999.0  # 2023-11-14T22:13:19Z
TS_LATE = 1700100000.0  # 2023-11-16T02:00:00Z

LOGGER = "corvidae.tools.memory_tools"


@pytest.fixture
def far_east_tz(monkeypatch):
    """Run the test under UTC+14, so a parser or formatter that used local
    time instead of UTC would move every boundary by fourteen hours."""
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _hit(result: str, rid: int) -> str | None:
    """The hit line for message_log row ``rid``, or None when absent."""
    return next(
        (line for line in result.splitlines() if line.startswith(f"{rid}.")), None,
    )


async def _search_tool(memory_tools):
    """Return the registered search_transcripts fn, asserting it exists."""
    fn = get_tools(memory_tools).get("search_transcripts")
    assert fn is not None, "search_transcripts tool is not registered"
    return fn


class TestSearchScope:
    """R1: whole-transcript keyword search with filters and bm25 order."""

    async def test_finds_word_only_in_other_channel_without_channel_arg(self):
        """R1: a word present only in channel B's rows is found with no channel."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        await seed_message_log_row(db, "irc:#a", "user", "nothing relevant here")
        await seed_message_log_row(db, "irc:#b", "user", "the quokka was spotted")

        result = await search("quokka")

        assert "irc:#b" in result and "quokka" in result, result
        await db.close()

    async def test_channel_arg_filters_by_equality(self):
        """R1: channel narrows to exactly that channel id, not a glob."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        await seed_message_log_row(db, "irc:#a", "user", "wombat in alpha")
        await seed_message_log_row(db, "irc:#b", "user", "wombat in bravo")

        only_b = await search("wombat", channel="irc:#b")
        globbed = await search("wombat", channel="irc:#*")

        assert "bravo" in only_b and "alpha" not in only_b, only_b
        assert "bravo" not in globbed and "alpha" not in globbed, globbed
        await db.close()

    async def test_after_and_before_filter_by_timestamp_naive_is_utc(self, far_east_tz):
        """R1: after/before bound message_log.timestamp; naive values are UTC."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        await seed_message_log_row(db, "irc:#a", "user", "zephyr early", ts=TS_EARLY)
        await seed_message_log_row(db, "irc:#a", "user", "zephyr late", ts=TS_LATE)

        # 2023-11-15 00:00 UTC lies between the two rows.
        after = await search("zephyr", after="2023-11-15")
        before = await search("zephyr", before="2023-11-15")

        assert "late" in after and "early" not in after, after
        assert "early" in before and "late" not in before, before
        await db.close()

    async def test_invalid_date_returns_error_naming_argument_and_value(self):
        """R1: a bad after/before names the argument, the value and the formats."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)

        for name in ("after", "before"):
            result = await search("zephyr", **{name: "next tuesday"})
            assert f"'{name}'" in result, result
            assert "next tuesday" in result, result
            assert "YYYY-MM-DD or ISO-8601" in result, result
        await db.close()

    async def test_hits_in_bm25_order_not_id_order(self):
        """R1: best bm25 match first, even when it has the higher row id."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        # Lower id, weak match: one mention buried in a long message.
        weak = await seed_message_log_row(
            db, "irc:#a", "user",
            "ocelot " + "filler words padding the message out " * 12,
        )
        # Higher id, strong match: short and repeats the term.
        strong = await seed_message_log_row(
            db, "irc:#a", "user", "ocelot ocelot ocelot",
        )

        result = await search("ocelot")

        lines = result.splitlines()
        positions = {
            i: line for i, line in enumerate(lines)
            if line.startswith((f"{weak}.", f"{strong}."))
        }
        order = [int(line.split(".")[0]) for _i, line in sorted(positions.items())]
        assert order == [strong, weak], result
        await db.close()


class TestHitLine:
    """R2: one hit line carries id, channel, UTC timestamp, role, snippet."""

    async def test_hit_line_has_all_five_fields(self, far_east_tz):
        """R2: leading id, channel, ISO UTC timestamp, role, **highlighted** match."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        rid = await seed_message_log_row(
            db, "irc:#b", "assistant", "the kestrel nests at dusk", ts=1700000000.0,
        )

        result = await search("kestrel")

        hit = _hit(result, rid)
        assert hit is not None, result
        assert "irc:#b" in hit, hit
        assert "2023-11-14T22:13:20" in hit, hit
        assert "assistant" in hit, hit
        assert "**kestrel**" in hit, hit
        await db.close()


class TestNoAccessStats:
    """R3: search_transcripts is read-only."""

    async def test_covering_record_stats_unchanged(self):
        """R3: a record covering the hit keeps retrieval_count/last_retrieved_at."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        rid = await seed_message_log_row(db, "irc:#a", "user", "pangolin sighting")
        mid = await seed_memory(
            db, "irc:#a", "a summary", retrieval_count=3, last_retrieved_at=1000.0,
            msg_id_start=rid, msg_id_end=rid,
        )

        result = await search("pangolin")

        assert _hit(result, rid) is not None, result
        async with db.execute(
            "SELECT retrieval_count, last_retrieved_at FROM memory WHERE id = ?",
            (mid,),
        ) as cursor:
            assert await cursor.fetchone() == (3, 1000.0)
        await db.close()


class TestRedaction:
    """R4: redacted text is neither findable nor replayed."""

    async def test_redacted_word_unfindable_and_replay_shows_tombstone(self):
        """R4: after redact_messages, search finds nothing; the range read
        shows the tombstone, not the original."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        rid = await seed_message_log_row(db, "irc:#a", "user", "password is hunter2")
        # Precondition: findable before redaction.
        before = await search("hunter2")
        assert _hit(before, rid) is not None, before

        await redact_messages(db, [rid])

        after = await search("hunter2")
        assert _hit(after, rid) is None and "hunter2" not in after, after
        replay = await get_tools(memory_tools)["recall_raw"](
            memory_id=None, channel="irc:#a", start_id=rid, end_id=rid,
        )
        assert "hunter2" not in replay, replay
        assert "redacted" in replay.lower(), replay
        await db.close()


class TestRecallRange:
    """R5: recall_raw by channel and id range."""

    async def _interleaved(self, db):
        """Rows a1, b1, a2, b2, a3 across two channels; returns the ids."""
        ids = []
        for chan, text in (
            ("irc:#a", "alpha one"), ("irc:#b", "bravo one"),
            ("irc:#a", "alpha two"), ("irc:#b", "bravo two"),
            ("irc:#a", "alpha three"),
        ):
            ids.append(await seed_message_log_row(db, chan, "user", text))
        return ids

    async def test_replays_only_that_channels_rows_inclusive(self):
        """R5: other-channel rows excluded, bounds inclusive, header names range,
        lines are role: content."""
        _memory, memory_tools, db = await build_tools_env()
        a1, _b1, a2, _b2, a3 = await self._interleaved(db)

        result = await get_tools(memory_tools)["recall_raw"](
            memory_id=None, channel="irc:#a", start_id=a1, end_id=a3,
        )

        lines = result.splitlines()
        assert lines[0] == f"[channel irc:#a: messages {a1}–{a3}]", result
        assert lines[1:] == [
            "user: alpha one", "user: alpha two", "user: alpha three",
        ], result
        await db.close()

    async def test_bumps_no_memory_stats(self):
        """R5: the range form changes no record's retrieval stats."""
        _memory, memory_tools, db = await build_tools_env()
        a1, _b1, _a2, _b2, a3 = await self._interleaved(db)
        mid = await seed_memory(
            db, "irc:#a", "summary", retrieval_count=2, last_retrieved_at=500.0,
            msg_id_start=a1, msg_id_end=a3,
        )

        result = await get_tools(memory_tools)["recall_raw"](
            memory_id=None, channel="irc:#a", start_id=a1, end_id=a3,
        )

        assert "alpha one" in result, result
        async with db.execute(
            "SELECT retrieval_count, last_retrieved_at FROM memory WHERE id = ?",
            (mid,),
        ) as cursor:
            assert await cursor.fetchone() == (2, 500.0)
        await db.close()

    async def test_empty_range_names_channel_and_ids(self):
        """R5: an empty result says so, naming the channel and both ids."""
        _memory, memory_tools, db = await build_tools_env()
        a1, *_rest, a3 = await self._interleaved(db)

        result = await get_tools(memory_tools)["recall_raw"](
            memory_id=None, channel="irc:#zzz", start_id=a1, end_id=a3,
        )

        assert (
            f"No messages in channel irc:#zzz between ids {a1} and {a3}." in result
        ), result
        await db.close()


class TestRecallRawForms:
    """R6: exactly one form per call."""

    async def test_mixed_form_error(self):
        """R6: memory_id together with the range arguments is rejected."""
        _memory, memory_tools, db = await build_tools_env()
        result = await get_tools(memory_tools)["recall_raw"](
            memory_id=1, channel="irc:#a", start_id=1, end_id=2,
        )

        assert result.startswith("Error: recall_raw takes either memory_id alone"), result
        assert "channel, start_id and end_id together" in result, result
        assert "pass null" in result, result
        got = result.split("(got ", 1)[1]
        for name in ("memory_id", "channel", "start_id", "end_id"):
            assert name in got, result
        await db.close()

    async def test_partial_range_error(self):
        """R6: channel with start_id but no end_id is rejected."""
        _memory, memory_tools, db = await build_tools_env()
        result = await get_tools(memory_tools)["recall_raw"](
            memory_id=None, channel="irc:#a", start_id=1, end_id=None,
        )

        assert result.startswith("Error: recall_raw takes either memory_id alone"), result
        assert "channel, start_id and end_id together" in result, result
        assert "pass null" in result, result
        got = result.split("(got ", 1)[1]
        assert "channel" in got and "start_id" in got, result
        assert "end_id" not in got and "memory_id" not in got, result
        await db.close()

    async def test_no_arguments_error(self):
        """R6: a call with nothing set says it got no arguments."""
        _memory, memory_tools, db = await build_tools_env()
        result = await get_tools(memory_tools)["recall_raw"]()

        assert result.startswith("Error: recall_raw takes either memory_id alone"), result
        assert "channel, start_id and end_id together" in result, result
        assert "pass null" in result, result
        assert "got no arguments" in result, result
        await db.close()

    async def test_inverted_range_error(self):
        """R6: start_id after end_id names both ids and both forms."""
        _memory, memory_tools, db = await build_tools_env()
        result = await get_tools(memory_tools)["recall_raw"](
            memory_id=None, channel="irc:#a", start_id=9, end_id=3,
        )

        assert result.startswith("Error: recall_raw start_id 9 is after end_id 3"), result
        assert "pass start_id <= end_id" in result, result
        assert "memory_id alone" in result, result
        assert "channel, start_id and end_id together" in result, result
        await db.close()


class TestFence:
    """R7: both tools' output is fenced as untrusted."""

    async def test_both_tools_declare_untrusted_output(self):
        """R7: declares_untrusted_output is true for search_transcripts and recall_raw."""
        _memory, memory_tools, db = await build_tools_env()
        tools = get_tools(memory_tools)

        assert "search_transcripts" in tools, "search_transcripts tool is not registered"
        assert declares_untrusted_output(tools["search_transcripts"])
        assert declares_untrusted_output(tools["recall_raw"])
        await db.close()


class TestRowPattern:
    """R8: search follows recall_raw's row pattern."""

    async def test_unsent_row_is_labelled_withheld(self):
        """R8: a matching unsent row is returned with the withheld label."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        rid = await insert_raw_message(
            db, "irc:#a", "?",
            (json.dumps({"role": "assistant", "content": "a draft about axolotls"}),),
            message_type="unsent",
        )

        result = await search("axolotls")

        hit = _hit(result, rid)
        assert hit is not None, result
        assert "assistant (withheld, not sent)" in hit, hit
        await db.close()

    async def test_corrupt_row_skipped_with_warning_others_returned(self, caplog):
        """R8: a corrupt matching row is skipped with the standard warning; the
        other hits still return."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        good = await seed_message_log_row(db, "irc:#a", "user", "marmot is good")
        # Dropping the trigger lets invalid JSON in; index it by hand so the
        # corrupt row is a search hit.
        bad = await insert_raw_message(
            db, "irc:#a", "?", ("{not valid json",), drop_fts_trigger=True,
        )
        await db.execute(
            "INSERT INTO message_fts(rowid, content_text) VALUES (?, ?)",
            (bad, "marmot is corrupt"),
        )
        await db.commit()

        with caplog.at_level(logging.WARNING):
            result = await search("marmot")

        assert _hit(result, good) is not None, result
        assert _hit(result, bad) is None, result
        assert skip_warnings(caplog, bad, "irc:#a"), [
            r.getMessage() for r in caplog.records
        ]
        await db.close()

    async def test_every_known_type_found_unknown_type_skipped(self, caplog):
        """R8: summary and context rows are searchable like message rows; a
        row of an unknown type is skipped with the standard warning."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        # One matching row per non-message known type, plus an unknown type.
        ids = {}
        for mtype in ("summary", "context", "bogus"):
            ids[mtype] = await insert_raw_message(
                db, "irc:#a", "?",
                (json.dumps({"role": "system", "content": f"capybara {mtype}"}),),
                message_type=mtype,
            )

        with caplog.at_level(logging.WARNING):
            result = await search("capybara")

        assert _hit(result, ids["summary"]) is not None, result
        assert _hit(result, ids["context"]) is not None, result
        assert _hit(result, ids["bogus"]) is None, result
        assert skip_warnings(caplog, ids["bogus"], "irc:#a"), [
            r.getMessage() for r in caplog.records
        ]
        await db.close()


class TestOutputAndErrors:
    """R9: cap, empty results and failure text."""

    async def test_output_capped_with_truncation_marker(self):
        """R9: a small max_tokens truncates with the marker and fewer lines."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        for i in range(10):
            await seed_message_log_row(
                db, "irc:#a", "user", f"lemur number {i} " + "pad " * 10,
            )

        full = await search("lemur", max_tokens=2000)
        capped = await search("lemur", max_tokens=40)

        assert "[truncated — token limit reached]" not in full, full
        assert "[truncated — token limit reached]" in capped, capped
        assert len(capped.splitlines()) < len(full.splitlines()), capped
        await db.close()

    async def test_at_most_fifty_hits(self):
        """R9: no more than 50 hits, even when the token cap would fit more."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        for _ in range(60):
            await seed_message_log_row(db, "irc:#a", "user", "tapir")

        result = await search("tapir", max_tokens=100000)

        hits = [line for line in result.splitlines() if line.split(".")[0].isdigit()]
        assert len(hits) == 50, result
        await db.close()

    async def test_query_without_word_tokens(self):
        """R9: punctuation-only query says it has no searchable terms."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)

        assert await search("?!...") == "No results: query has no searchable terms."
        await db.close()

    async def test_no_hits_text_names_channel_filter(self):
        """R9: no hits says so, and names the channel when one was given."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)

        plain = await search("nonexistentword")
        filtered = await search("nonexistentword", channel="irc:#typo")

        assert "No transcript messages found matching your query." in plain, plain
        assert "irc:#typo" not in plain, plain
        assert "No transcript messages found matching your query" in filtered, filtered
        assert "irc:#typo" in filtered, filtered
        await db.close()

    async def test_failing_fts_query_warns_and_names_remedy(self, caplog):
        """R9: a failing FTS query logs a warning and returns the error naming
        the tool, the query, the exception type and the remedy."""
        _memory, memory_tools, db = await build_tools_env()
        search = await _search_tool(memory_tools)
        await seed_message_log_row(db, "irc:#a", "user", "gecko")
        # Break the index so the MATCH query raises.
        await db.execute("DROP TRIGGER IF EXISTS message_log_ai")
        await db.execute("DROP TRIGGER IF EXISTS message_log_au")
        await db.execute("DROP TABLE message_fts")
        await db.commit()

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = await search("gecko")

        assert "search_transcripts" in result, result
        assert "gecko" in result, result
        assert "OperationalError" in result, result
        assert "retry with a simpler query" in result, result
        assert any(
            r.name == LOGGER and r.levelno == logging.WARNING and r.exc_info
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]
        await db.close()
