"""`corvidae redact` reaches outcome_log's JSONL records (#1179, R5).

outcome_log appends records to ``<db stem>.outcomes.jsonl`` beside the
session DB. Appraisal records copy claims from the reply
(``stage2.past_claims``) and critique records copy objection text
(``provenance``/``critique`` ``detail``). Redacting message X on channel C
tombstones that text in the records of the trace that owns X (on C, the
largest persisted ``message_rowid`` <= X), keeping counts, and leaves other
traces alone. These tests go through the real entry point.
"""

import json
import time
from importlib.metadata import entry_points

from cognition.test_redact_cascade import insert_message, make_full_db

from corvidae.commands.redact import redact_messages
from tests.cognition.trace_helpers import outcomes_file

CLAIMS = [
    {"claim": "The kestrel nests at Brannan Street", "queries": ["kestrel Brannan", "nest"]},
    {"claim": "Zebulon owes 40 dollars", "queries": ["Zebulon 40"]},
]
DETAIL = [{"claim": "kestrel at Brannan", "objection": "no record of Brannan"}]


def _line(trace_id, channel_id, source, record, origin=None) -> str:
    # One record as outcome_log writes it.
    return json.dumps({
        "ts": time.time(), "trace_id": trace_id, "channel_id": channel_id,
        "source": source, "origin": origin, "record": record,
    })


def _exchange(trace_id, channel_id, rowid) -> list[str]:
    # A trace's persisted line plus an appraisal and a critique record.
    return [
        _line(trace_id, channel_id, "outcome_log",
              {"event": "persisted", "message_rowid": rowid}, origin="user"),
        _line(trace_id, channel_id, "appraisal", {"stage2": {"past_claims": CLAIMS}}),
        _line(trace_id, channel_id, "critique",
              {"provenance": {"detail": DETAIL}, "critique": {"detail": DETAIL}}),
    ]


async def _db_with_dialog(tmp_path):
    """A file DB with a dialog on irc:#a (t1 owns u1/a1, t2 owns u2/a2) and
    irc:#b interleaved, plus the JSONL records for each trace."""
    path = tmp_path / "sessions.db"
    db = await make_full_db(str(path))
    ids = {}
    ids["u1"] = await insert_message(db, "irc:#a", "user", "where does the kestrel nest")
    ids["a1"] = await insert_message(db, "irc:#a", "assistant", "Brannan Street")
    ids["u2"] = await insert_message(db, "irc:#a", "user", "thanks")
    ids["bx"] = await insert_message(db, "irc:#b", "user", "meanwhile")
    lines = (
        _exchange("t1", "irc:#a", ids["u1"])
        + _exchange("t2", "irc:#a", ids["u2"])
        + _exchange("tx", "irc:#b", ids["bx"])
    )
    outcomes_file(path).write_text("\n".join(lines) + "\n")
    return db, ids, outcomes_file(path)


def _records(path, trace_id, source):
    """One trace's records from one source; lines that do not parse (the
    corrupt-line test keeps one) are skipped."""
    records = []
    for raw in path.read_text().splitlines():
        try:
            line = json.loads(raw)
        except ValueError:
            continue
        if line["trace_id"] == trace_id and line["source"] == source:
            records.append(line["record"])
    return records


class TestScrub:
    async def test_redacting_a_reply_scrubs_its_trace_only(self, tmp_path):
        db, ids, jsonl = await _db_with_dialog(tmp_path)
        try:
            result = await redact_messages(db, [ids["a1"]])
        finally:
            await db.close()
        assert result["scrubbed"]["outcome_log_jsonl"] == 1
        (appraisal,) = _records(jsonl, "t1", "appraisal")
        claims = appraisal["stage2"]["past_claims"]
        # Counts survive; the words do not.
        assert [len(c["queries"]) for c in claims] == [2, 1]
        text = json.dumps(_records(jsonl, "t1", "appraisal") + _records(jsonl, "t1", "critique"))
        for word in ("kestrel", "Brannan", "Zebulon"):
            assert word not in text, text
        (critique,) = _records(jsonl, "t1", "critique")
        assert len(critique["provenance"]["detail"]) == 1
        # The later trace and the other channel keep their text.
        assert _records(jsonl, "t2", "appraisal")[0]["stage2"]["past_claims"] == CLAIMS
        assert _records(jsonl, "tx", "critique")[0]["critique"]["detail"] == DETAIL

    async def test_dry_run_counts_and_leaves_the_file(self, tmp_path):
        db, ids, jsonl = await _db_with_dialog(tmp_path)
        before = jsonl.read_text()
        try:
            result = await redact_messages(db, [ids["u1"]], dry_run=True)
        finally:
            await db.close()
        assert result["scrubbed"]["outcome_log_jsonl"] == 1
        assert jsonl.read_text() == before

    async def test_corrupt_line_is_kept_with_a_warning(self, tmp_path, caplog):
        import logging

        db, ids, jsonl = await _db_with_dialog(tmp_path)
        jsonl.write_text("{not json\n" + jsonl.read_text())
        try:
            with caplog.at_level(logging.WARNING):
                await redact_messages(db, [ids["a1"]])
        finally:
            await db.close()
        assert jsonl.read_text().startswith("{not json\n")
        assert "kestrel" not in json.dumps(_records(jsonl, "t1", "appraisal"))
        messages = [r.getMessage() for r in caplog.records]
        assert any(str(jsonl) in m and "line 1" in m for m in messages), messages


class TestRegistration:
    def test_scrubber_is_registered(self):
        found = {ep.name: ep for ep in entry_points(group="corvidae.redact")}
        assert "outcome_log_jsonl" in found, sorted(found)
        assert callable(found["outcome_log_jsonl"].load())

    async def test_db_without_the_file_skips_with_notice(self, tmp_path):
        db = await make_full_db(str(tmp_path / "sessions.db"))
        try:
            row_id = await insert_message(db, "irc:#a", "user", "hello there")
            notices: list[str] = []
            result = await redact_messages(db, [row_id], notices=notices)
        finally:
            await db.close()
        assert result["scrubbed"]["outcome_log_jsonl"] == 0
        assert any("outcome_log_jsonl" in n for n in notices), notices
