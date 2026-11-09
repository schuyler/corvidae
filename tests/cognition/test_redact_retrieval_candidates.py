"""`corvidae redact` reaches retrieval_candidate (#407, requirement L5).

retrieval_candidate rows hold ids and numbers only, but a row that
references a redacted memory still records that the memory was retrieved.
memory owns the table and registers its scrubber under the
``corvidae.redact`` entry-point group; these tests go through the real
entry point. After any redact form, no row references a redacted memory;
a dry run reports the count and deletes nothing.
"""

import asyncio
import time

from click.testing import CliRunner

from corvidae.commands.redact import (
    redact_command,
    redact_memory_id,
    redact_messages,
    redact_range,
)

from cognition.test_redact_cascade import insert_memory, insert_message, make_full_db


async def _add_candidates(
    db, memory_ids: list[int], channel_id: str = "irc:#a",
) -> None:
    """One retrieval_log row (on channel_id) with a candidate row per memory."""
    cursor = await db.execute(
        "INSERT INTO retrieval_log (ts, channel_id, hit_count, admitted_count) "
        "VALUES (?, ?, ?, ?)",
        (time.time(), channel_id, len(memory_ids), 0),
    )
    retrieval_id = cursor.lastrowid
    for rank, memory_id in enumerate(memory_ids, start=1):
        await db.execute(
            "INSERT INTO retrieval_candidate (retrieval_id, memory_id, similarity, "
            "score, band, rank, outcome) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (retrieval_id, memory_id, 0.8, 0.7, "strong", rank, "admitted"),
        )
    await db.commit()


async def _remaining(db) -> set[int]:
    """The memory ids that still have a retrieval_candidate row."""
    async with db.execute("SELECT DISTINCT memory_id FROM retrieval_candidate") as cursor:
        return {row[0] for row in await cursor.fetchall()}


async def _db_with_candidates(path: str | None = None):
    """A full DB with candidate rows for four memories.

    m_target (irc:#a, u1..a1) and m_overlap (irc:#a, u1..u1) both intersect
    a redaction of u1; m_other (irc:#a, u2) and m_chan_b (irc:#b) do not.
    Returns (db, ids) with ids holding message ids and the m_* memory ids.
    """
    db = await make_full_db(path)
    ids: dict[str, int] = {}
    ids["u1"] = await insert_message(db, "irc:#a", "user", "where does the kestrel nest")
    ids["a1"] = await insert_message(db, "irc:#a", "assistant", "Brannan Street")
    ids["u2"] = await insert_message(db, "irc:#a", "user", "thanks")
    ids["b1"] = await insert_message(db, "irc:#b", "user", "unrelated")
    ids["m_target"] = await insert_memory(db, "irc:#a", "kestrel nest", ids["u1"], ids["a1"])
    ids["m_overlap"] = await insert_memory(db, "irc:#a", "nest again", ids["u1"], ids["u1"])
    ids["m_other"] = await insert_memory(db, "irc:#a", "thanks given", ids["u2"], ids["u2"])
    ids["m_chan_b"] = await insert_memory(db, "irc:#b", "other channel", ids["b1"], ids["b1"])
    await _add_candidates(
        db, [ids["m_target"], ids["m_overlap"], ids["m_other"], ids["m_chan_b"]],
    )
    # A retrieval on another channel that also scored the memories the
    # forms redact: its rows reference a redacted memory too (L5), so they go.
    await _add_candidates(db, [ids["m_target"], ids["m_overlap"]], "irc:#b")
    return db, ids


class TestRedactForms:
    async def test_message_form_deletes_rows_of_cascaded_memories(self):
        # L5 message: redacting u1 cascades onto the memories covering it;
        # their candidate rows go, everyone else's stay.
        db, ids = await _db_with_candidates()
        try:
            await redact_messages(db, [ids["u1"]])
            assert await _remaining(db) == {ids["m_other"], ids["m_chan_b"]}
        finally:
            await db.close()

    async def test_memory_form_deletes_target_and_cascaded_rows(self):
        # L5 memory: redacting a memory deletes its rows and those of the
        # records its range cascades onto; unrelated records keep theirs.
        db, ids = await _db_with_candidates()
        try:
            await redact_memory_id(db, ids["m_target"])
            assert await _remaining(db) == {ids["m_other"], ids["m_chan_b"]}
        finally:
            await db.close()

    async def test_range_form_deletes_rows_of_intersecting_memories(self):
        # L5 range: redacting the u1..a1 range deletes the rows of memories
        # intersecting it and keeps the rest.
        db, ids = await _db_with_candidates()
        try:
            await redact_range(db, ids["u1"], ids["a1"])
            assert await _remaining(db) == {ids["m_other"], ids["m_chan_b"]}
        finally:
            await db.close()

    def test_dry_run_cli_reports_count_and_deletes_nothing(self, tmp_path):
        # L5 dry-run: the CLI reports how many retrieval_candidate rows a
        # real run would delete (m_target's and m_overlap's, on both retrievals: 4) and the
        # table is unchanged afterwards.
        path = str(tmp_path / "s.db")

        async def build() -> int:
            db, ids = await _db_with_candidates(path)
            await db.close()
            return ids["u1"]

        async def count_rows() -> int:
            db = await make_full_db(path)
            try:
                async with db.execute("SELECT COUNT(*) FROM retrieval_candidate") as cursor:
                    return (await cursor.fetchone())[0]
            finally:
                await db.close()

        u1 = asyncio.run(build())
        result = CliRunner().invoke(
            redact_command, ["--db", path, "--dry-run", "message", str(u1)]
        )
        assert result.exit_code == 0, result.output
        assert "4 retrieval_candidate rows" in result.output, result.output
        assert asyncio.run(count_rows()) == 6
