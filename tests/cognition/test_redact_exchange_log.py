"""`corvidae redact` reaches exchange_log (#160).

The scrub is cognition's: outcome_log owns exchange_log and registers its
scrubber under the ``corvidae.redact`` entry-point group, which the kernel
redact CLI runs after tombstoning. These tests go through the real entry
point.

exchange_log holds text copied out of the conversation in two columns:
``appraisal.stage2.past_claims`` (claims quoted from the reply, plus search
phrases built from them) and ``provenance_snapshot`` (the CONTEXT messages
in the window at the turn). Redacting message X in channel C must scrub:

- past_claims on the exchange that owns X (in C, the largest
  message_rowid <= X), keeping the number of claims and queries;
- provenance_snapshot on every exchange in C with message_rowid >= X.
"""

import asyncio
import json
import logging
import time

from click.testing import CliRunner

from corvidae.commands.redact import (
    redact_command,
    redact_memory_id,
    redact_messages,
)
from corvidae_cognition.outcome_log import EXCHANGE_LOG_DDL

from cognition.test_redact_cascade import insert_memory, insert_message, make_full_db

CLAIMS = [
    {"claim": "The kestrel nests at Brannan Street", "queries": ["kestrel Brannan", "nest"]},
    {"claim": "Zebulon owes 40 dollars", "queries": ["Zebulon 40"]},
]


async def _add_exchange(
    db, trace_id: str, channel_id: str, message_rowid: int,
    *, claims=CLAIMS, snapshot: str | None = "secret context",
) -> None:
    """Insert an exchange_log row with stage-2 claims and a snapshot."""
    appraisal = {
        "stage1": {"score": 0.4},
        "stage2": {"valence": 0.1, "correction": False, "past_claims": claims},
    }
    await db.execute(
        "INSERT INTO exchange_log (trace_id, channel_id, origin, message_rowid, "
        "created_at, appraisal, provenance_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (trace_id, channel_id, "user", message_rowid, time.time(),
         json.dumps(appraisal),
         None if snapshot is None else json.dumps([{"content": snapshot}])),
    )
    await db.commit()


async def _row(db, trace_id: str) -> tuple[dict, str | None]:
    """Return (appraisal dict, provenance_snapshot) for an exchange."""
    async with db.execute(
        "SELECT appraisal, provenance_snapshot FROM exchange_log WHERE trace_id = ?",
        (trace_id,),
    ) as cursor:
        appraisal, snapshot = await cursor.fetchone()
    return json.loads(appraisal), snapshot


def _assert_scrubbed(appraisal: dict) -> None:
    """past_claims keep their shape and counts but none of the original words."""
    claims = appraisal["stage2"]["past_claims"]
    assert [len(c["queries"]) for c in claims] == [len(c["queries"]) for c in CLAIMS]
    text = json.dumps(claims)
    for word in ("kestrel", "Brannan", "Zebulon", "40"):
        assert word not in text, text
    assert all("redacted" in c["claim"] for c in claims), claims
    assert all("redacted" in q for c in claims for q in c["queries"]), claims
    # Other appraisal fields are untouched.
    assert appraisal["stage1"] == {"score": 0.4}
    assert appraisal["stage2"]["valence"] == 0.1
    assert appraisal["stage2"]["correction"] is False


async def _db_with_dialog(path: str | None = None):
    """A full DB with exchange_log: a two-exchange dialog on irc:#a, with an
    irc:#b exchange interleaved before t2's reply (message_log ids are
    global), and a later one on irc:#b. Returns (db, ids)."""
    db = await make_full_db(path)
    await db.execute(EXCHANGE_LOG_DDL)
    await db.commit()
    ids = {}
    ids["u1"] = await insert_message(db, "irc:#a", "user", "where does the kestrel nest")
    ids["a1"] = await insert_message(db, "irc:#a", "assistant", "Brannan Street")
    ids["u2"] = await insert_message(db, "irc:#a", "user", "thanks")
    ids["bx"] = await insert_message(db, "irc:#b", "user", "meanwhile")
    ids["a2"] = await insert_message(db, "irc:#a", "assistant", "any time")
    ids["b1"] = await insert_message(db, "irc:#b", "user", "unrelated")
    await _add_exchange(db, "t1", "irc:#a", ids["u1"])
    await _add_exchange(db, "t2", "irc:#a", ids["u2"])
    await _add_exchange(db, "tx", "irc:#b", ids["bx"])
    await _add_exchange(db, "tb", "irc:#b", ids["b1"])
    return db, ids


class TestPastClaims:
    async def test_redacting_user_message_scrubs_its_exchange(self):
        db, ids = await _db_with_dialog()
        try:
            await redact_messages(db, [ids["u1"]])
            appraisal, _ = await _row(db, "t1")
            _assert_scrubbed(appraisal)
            # The later exchange's claims are its own reply's, not X's.
            later, _ = await _row(db, "t2")
            assert later["stage2"]["past_claims"] == CLAIMS
        finally:
            await db.close()

    async def test_redacting_reply_scrubs_owning_exchange_only(self):
        db, ids = await _db_with_dialog()
        try:
            # a2 is the reply in t2's exchange; t1 is earlier and untouched.
            # tx (irc:#b) sits between u2 and a2 by id but owns nothing in #a.
            await redact_messages(db, [ids["a2"]])
            appraisal, _ = await _row(db, "t2")
            _assert_scrubbed(appraisal)
            earlier, _ = await _row(db, "t1")
            assert earlier["stage2"]["past_claims"] == CLAIMS
            foreign, foreign_snapshot = await _row(db, "tx")
            assert foreign["stage2"]["past_claims"] == CLAIMS
            assert foreign_snapshot is not None
        finally:
            await db.close()

    async def test_owning_exchange_without_stage2_still_scrubs_snapshot(self):
        # Many exchanges never get stage 2: the claims step has nothing to do,
        # and the snapshot step still runs.
        db, ids = await _db_with_dialog()
        try:
            await db.execute(
                "UPDATE exchange_log SET appraisal = NULL WHERE trace_id = 't1'"
            )
            await db.commit()
            result = await redact_messages(db, [ids["u1"]])
            async with db.execute(
                "SELECT provenance_snapshot FROM exchange_log WHERE trace_id = 't1'"
            ) as cursor:
                assert (await cursor.fetchone())[0] is None
            assert result["scrubbed"]["exchange_log"] == 2
        finally:
            await db.close()


class TestCorruptAppraisalRow:
    """#311: a corrupt appraisal value must not abort the redact cascade.

    The scrubber decodes the appraisal column of every owning exchange; a
    corrupt value (only reachable by out-of-band writes — the daemon's
    writers always produce valid JSON) skips with a WARNING naming the
    trace id, channel and remedy, and the other exchanges still scrub.
    """

    async def test_corrupt_row_skipped_valid_rows_scrubbed(self, caplog):
        db, ids = await _db_with_dialog()
        try:
            await db.execute(
                "UPDATE exchange_log SET appraisal = 'not json{' "
                "WHERE trace_id = 't1'"
            )
            await db.commit()
            with caplog.at_level(
                logging.WARNING, logger="corvidae.outcome_log"
            ):
                # Two owners: the corrupt t1 and the valid t2, so the
                # cascade has a valid row to continue onto after a corrupt
                # one (a break instead of continue would leave t2 unscrubbed).
                result = await redact_messages(db, [ids["u1"], ids["u2"]])
            # t1's snapshot still scrubs (the corrupt appraisal only skips
            # the claims step); t2's claims and snapshot both scrub.
            assert result["scrubbed"]["exchange_log"] == 2
            async with db.execute(
                "SELECT appraisal FROM exchange_log WHERE trace_id = 't1'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == "not json{"
            t2_claims = (await _row(db, "t2"))[0]["stage2"]["past_claims"]
            assert all("redacted" in c["claim"] for c in t2_claims), t2_claims
            for word in ("kestrel", "Brannan", "Zebulon", "40"):
                assert word not in json.dumps(t2_claims), t2_claims
            async with db.execute(
                "SELECT provenance_snapshot FROM exchange_log "
                "WHERE trace_id = 't2'"
            ) as cursor:
                assert (await cursor.fetchone())[0] is None
            warnings = [
                r.getMessage() for r in caplog.records
                if r.name == "corvidae.outcome_log"
                and r.levelno >= logging.WARNING
            ]
            assert len(warnings) == 1, warnings
            assert "t1" in warnings[0], warnings[0]
            assert "irc:#a" in warnings[0], warnings[0]
            assert "inspect or correct" in warnings[0], warnings[0]
            # The unscrubbed row is surfaced to the caller, not only logged:
            # redact turns this into a non-zero exit (#311).
            assert len(result["unscrubbed"]) == 1, result["unscrubbed"]
            assert "exchange_log" in result["unscrubbed"][0]
            assert "t1" in result["unscrubbed"][0]
            assert "irc:#a" in result["unscrubbed"][0]
            assert "may survive" in result["unscrubbed"][0]
        finally:
            await db.close()

    async def test_corrupt_row_snapshot_kept_when_redacted_id_is_above_it(
        self, caplog
    ):
        # The snapshot rule is the channel-wide one, not special-cased for
        # corrupt rows: a1 is the reply inside t1's exchange, so t1's rowid
        # is below X and its snapshot survives, unlike the u1 case above.
        db, ids = await _db_with_dialog()
        try:
            await db.execute(
                "UPDATE exchange_log SET appraisal = 'not json{' "
                "WHERE trace_id = 't1'"
            )
            await db.commit()
            with caplog.at_level(
                logging.WARNING, logger="corvidae.outcome_log"
            ):
                await redact_messages(db, [ids["a1"]])
            async with db.execute(
                "SELECT appraisal, provenance_snapshot FROM exchange_log "
                "WHERE trace_id = 't1'"
            ) as cursor:
                appraisal, snapshot = await cursor.fetchone()
            assert appraisal == "not json{"
            assert snapshot == '[{"content": "secret context"}]'
        finally:
            await db.close()

    async def test_non_dict_appraisal_row_skipped_valid_rows_scrubbed(self, caplog):
        db, ids = await _db_with_dialog()
        try:
            # '[]' decodes as JSON but is not an appraisal envelope, so the
            # scrubber's decode succeeds and .get would raise AttributeError.
            await db.execute(
                "UPDATE exchange_log SET appraisal = '[]' "
                "WHERE trace_id = 't1'"
            )
            await db.commit()
            with caplog.at_level(
                logging.WARNING, logger="corvidae.outcome_log"
            ):
                result = await redact_messages(db, [ids["u1"]])
            # t1's snapshot still scrubs; only its claims step skipped.
            assert result["scrubbed"]["exchange_log"] == 2
            async with db.execute(
                "SELECT appraisal FROM exchange_log WHERE trace_id = 't1'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == "[]"
            warnings = [
                r.getMessage() for r in caplog.records
                if r.name == "corvidae.outcome_log"
                and r.levelno >= logging.WARNING
            ]
            assert len(warnings) == 1, warnings
            assert "t1" in warnings[0], warnings[0]
            assert "inspect or correct" in warnings[0], warnings[0]
        finally:
            await db.close()

    async def test_invalid_utf8_appraisal_row_skipped_valid_rows_scrubbed(
        self, caplog
    ):
        db, ids = await _db_with_dialog()
        try:
            # Valid TEXT to sqlite, but refuses to decode as UTF-8: without
            # CAST(appraisal AS BLOB) the fetch raises and aborts the cascade.
            await db.execute(
                "UPDATE exchange_log "
                "SET appraisal = CAST(x'7b22737461676531223a80ff7d' AS TEXT) "
                "WHERE trace_id = 't1'"
            )
            await db.commit()
            with caplog.at_level(
                logging.WARNING, logger="corvidae.outcome_log"
            ):
                result = await redact_messages(db, [ids["u1"], ids["u2"]])
            # t1's snapshot still scrubs; t2's claims and snapshot both scrub.
            assert result["scrubbed"]["exchange_log"] == 2
            t2_claims = (await _row(db, "t2"))[0]["stage2"]["past_claims"]
            assert all("redacted" in c["claim"] for c in t2_claims), t2_claims
            warnings = [
                r.getMessage() for r in caplog.records
                if r.name == "corvidae.outcome_log"
                and r.levelno >= logging.WARNING
            ]
            assert len(warnings) == 1, warnings
            assert "t1" in warnings[0], warnings[0]
            assert "inspect or correct" in warnings[0], warnings[0]
        finally:
            await db.close()

    async def test_malformed_claims_entry_skipped_valid_rows_scrubbed(
        self, caplog
    ):
        db, ids = await _db_with_dialog()
        try:
            # Valid JSON and a real envelope, but a claim with no "queries"
            # key: the scrubber's claim["queries"] raises KeyError, which
            # must not abort the cascade.
            broken = {"stage1": {"score": 0.4},
                      "stage2": {"valence": 0.1, "past_claims": [
                          {"claim": "The kestrel nests at Brannan Street"}]}}
            await db.execute(
                "UPDATE exchange_log SET appraisal = ? WHERE trace_id = 't1'",
                (json.dumps(broken),),
            )
            await db.commit()
            with caplog.at_level(
                logging.WARNING, logger="corvidae.outcome_log"
            ):
                result = await redact_messages(db, [ids["u1"], ids["u2"]])
            # The malformed row is left as-is; the cascade still reaches
            # and scrubs t2.
            assert result["scrubbed"]["exchange_log"] == 2
            assert (await _row(db, "t1"))[0] == broken
            t2_claims = (await _row(db, "t2"))[0]["stage2"]["past_claims"]
            assert all("redacted" in c["claim"] for c in t2_claims), t2_claims
            warnings = [
                r.getMessage() for r in caplog.records
                if r.name == "corvidae.outcome_log"
                and r.levelno >= logging.WARNING
            ]
            assert len(warnings) == 1, warnings
            assert "t1" in warnings[0], warnings[0]
            assert "inspect or correct" in warnings[0], warnings[0]
        finally:
            await db.close()


class TestProvenanceSnapshot:
    async def test_snapshots_from_x_onward_in_channel_are_nulled(self):
        db, ids = await _db_with_dialog()
        try:
            await redact_messages(db, [ids["a1"]])
            # t1 owns a1 (message_rowid u1 < a1) but starts before it, so its
            # snapshot predates X; t2 is at or after X.
            _, t1_snapshot = await _row(db, "t1")
            _, t2_snapshot = await _row(db, "t2")
            _, tx_snapshot = await _row(db, "tx")
            _, tb_snapshot = await _row(db, "tb")
            assert t1_snapshot is not None
            assert t2_snapshot is None
            # Another channel's exchanges are never touched.
            assert tx_snapshot is not None
            assert tb_snapshot is not None
        finally:
            await db.close()


class TestFormsAndReporting:
    async def test_memory_form_scrubs_exchange_log(self):
        db, ids = await _db_with_dialog()
        try:
            mem_id = await insert_memory(db, "irc:#a", "kestrel nest", ids["u1"], ids["a1"])
            await redact_memory_id(db, mem_id)
            appraisal, t1_snapshot = await _row(db, "t1")
            _assert_scrubbed(appraisal)
            assert t1_snapshot is None
        finally:
            await db.close()

    async def test_memory_form_dry_run_counts_exchanges(self):
        db, ids = await _db_with_dialog()
        try:
            mem_id = await insert_memory(db, "irc:#a", "kestrel nest", ids["u1"], ids["a1"])
            result = await redact_memory_id(db, mem_id, dry_run=True)
            # t1 (claims + snapshot) and t2 (snapshot).
            assert result["scrubbed"]["exchange_log"] == 2
            appraisal, snapshot = await _row(db, "t1")
            assert appraisal["stage2"]["past_claims"] == CLAIMS
            assert snapshot is not None
        finally:
            await db.close()

    async def test_dry_run_counts_and_writes_nothing(self):
        db, ids = await _db_with_dialog()
        try:
            result = await redact_messages(db, [ids["u1"]], dry_run=True)
            # t1 (claims + snapshot) and t2 (snapshot) would change.
            assert result["scrubbed"]["exchange_log"] == 2
            appraisal, snapshot = await _row(db, "t1")
            assert appraisal["stage2"]["past_claims"] == CLAIMS
            assert snapshot is not None
        finally:
            await db.close()

    async def test_result_counts_changed_exchanges(self):
        db, ids = await _db_with_dialog()
        try:
            result = await redact_messages(db, [ids["u1"]])
            assert result["scrubbed"]["exchange_log"] == 2
        finally:
            await db.close()

    def test_cli_reports_exchange_log_rows(self, tmp_path):
        path = str(tmp_path / "s.db")

        async def build() -> int:
            db, ids = await _db_with_dialog(path)
            await db.close()
            return ids["u1"]

        u1 = asyncio.run(build())
        result = CliRunner().invoke(
            redact_command, ["--db", path, "message", str(u1)]
        )
        assert result.exit_code == 0, result.output
        assert "2 exchange_log rows" in result.output, result.output

    def test_cli_range_form_reports_exchange_log_rows(self, tmp_path):
        path = str(tmp_path / "s.db")

        async def build() -> tuple[int, int]:
            db, ids = await _db_with_dialog(path)
            await db.close()
            return ids["u1"], ids["a1"]

        start, end = asyncio.run(build())
        result = CliRunner().invoke(
            redact_command, ["--db", path, "range", str(start), str(end)]
        )
        assert result.exit_code == 0, result.output
        assert "2 exchange_log rows" in result.output, result.output

    def test_cli_memory_form_reports_exchange_log_rows(self, tmp_path):
        path = str(tmp_path / "s.db")

        async def build() -> int:
            db, ids = await _db_with_dialog(path)
            mem_id = await insert_memory(db, "irc:#a", "kestrel nest", ids["u1"], ids["a1"])
            await db.close()
            return mem_id

        mem_id = asyncio.run(build())
        result = CliRunner().invoke(
            redact_command, ["--db", path, "memory", str(mem_id)]
        )
        assert result.exit_code == 0, result.output
        assert "2 exchange_log rows" in result.output, result.output


class TestCliReportsIncompleteRedaction:
    """#311: a row the scrubber cannot clear must fail the command.

    redact used to report a clean count and exit 0 whatever the scrubber
    managed, so a corrupt appraisal row left the redacted claim text in
    place with nothing in the result or the exit status to say so.
    """

    def test_cli_message_form_exits_nonzero_naming_the_row(self, tmp_path):
        path = str(tmp_path / "s.db")

        async def build() -> int:
            db, ids = await _db_with_dialog(path)
            await db.execute(
                "UPDATE exchange_log SET appraisal = 'not json{' "
                "WHERE trace_id = 't1'"
            )
            await db.commit()
            await db.close()
            return ids["u1"]

        u1 = asyncio.run(build())
        result = CliRunner().invoke(
            redact_command, ["--db", path, "message", str(u1)]
        )
        assert result.exit_code == 1, result.output
        assert "redaction incomplete" in result.output, result.output
        assert "t1" in result.output, result.output
        assert "irc:#a" in result.output, result.output
        assert "rerun this command" in result.output, result.output

    def test_cli_dry_run_reports_nothing_incomplete(self, tmp_path):
        # A dry run changed nothing, so nothing is incomplete: it reports
        # the row it would skip and still succeeds.
        path = str(tmp_path / "s.db")

        async def build() -> int:
            db, ids = await _db_with_dialog(path)
            await db.execute(
                "UPDATE exchange_log SET appraisal = 'not json{' "
                "WHERE trace_id = 't1'"
            )
            await db.commit()
            await db.close()
            return ids["u1"]

        u1 = asyncio.run(build())
        result = CliRunner().invoke(
            redact_command, ["--db", path, "--dry-run", "message", str(u1)]
        )
        assert result.exit_code == 0, result.output
        assert "redaction incomplete" not in result.output, result.output

    def test_cli_clean_run_still_exits_zero(self, tmp_path):
        path = str(tmp_path / "s.db")

        async def build() -> int:
            db, ids = await _db_with_dialog(path)
            await db.close()
            return ids["u1"]

        u1 = asyncio.run(build())
        result = CliRunner().invoke(
            redact_command, ["--db", path, "message", str(u1)]
        )
        assert result.exit_code == 0, result.output
        assert "incomplete" not in result.output, result.output


class TestRegistration:
    def test_exchange_log_scrubber_is_registered(self):
        # The kernel finds the scrubber only through the entry point.
        from importlib.metadata import entry_points

        found = {ep.name: ep for ep in entry_points(group="corvidae.redact")}
        assert "exchange_log" in found, sorted(found)
        assert callable(found["exchange_log"].load())

    async def test_db_without_exchange_log_skips_with_notice(self):
        # A DB whose daemon never ran outcome_log has no exchange_log table.
        db = await make_full_db()
        try:
            row_id = await insert_message(db, "irc:#a", "user", "hello there")
            notices: list[str] = []
            result = await redact_messages(db, [row_id], notices=notices)
            assert result["affected_messages"] == 1
            assert result["scrubbed"]["exchange_log"] == 0
            assert any("exchange_log" in n for n in notices), notices
        finally:
            await db.close()
