"""`corvidae reconsolidate` (#932): re-consolidate a channel's history into a copy.

Everything here is offline: a real sqlite file stands in for ``sessions.db``,
and the model is a scripted fake. The tests drive the same entry the CLI calls
(`run_reconsolidation`) over data shaped the way the daemon writes it
(`message_log` through ``init_db``, ``memory`` through the memory plugin's own
DDL), so the segmenter, the stamp and the supersede rule are the real ones.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import aiosqlite
import click
import pytest
from click.testing import CliRunner
from corvidae_cognition import reconsolidate as rc
from corvidae_cognition.memory import (
    MEMORY_DDL,
    MEMORY_FTS_DDL,
    MEMORY_FTS_INSERT_TRIGGER,
    MEMORY_FTS_UPDATE_TRIGGER,
    MEMORY_PROMPT_DDL,
    prompt_hash,
)

from corvidae.fork import ViewInputs
from corvidae.persistence import init_db

CHAN = "irc:#lex"
OTHER = "irc:#other"
SYSTEM = "IDENTITY\n\nREGISTER"
INSTRUCTION = "Consolidate this stretch of conversation."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def make_live_db(path: Path, *, per_channel: int = 12) -> None:
    """A sessions.db as the daemon leaves it: ``per_channel`` alternating
    user/assistant rows on each of two channels, interleaved so message ids
    are global (CHAN gets the odd ids), with the memory tables beside them."""
    async with aiosqlite.connect(path) as db:
        await db.execute("PRAGMA journal_mode = WAL")
        await init_db(db)
        for i in range(per_channel):
            for channel in (CHAN, OTHER):
                role = "user" if i % 2 == 0 else "assistant"
                await db.execute(
                    "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                    "VALUES (?, ?, ?, 'message')",
                    (channel, json.dumps({"role": role, "content": f"{channel} line {i} " + "word " * 20}),
                     1000.0 + i * 10),
                )
        for ddl in (MEMORY_DDL, MEMORY_PROMPT_DDL, MEMORY_FTS_DDL,
                    MEMORY_FTS_INSERT_TRIGGER, MEMORY_FTS_UPDATE_TRIGGER):
            await db.execute(ddl)
        await db.commit()


def add_record(path: Path, channel: str, start: int, end: int, **kw) -> int:
    """Insert one memory row (sync; the tests read and write the file between
    runs). Defaults are an active daemon-written ``fork`` record."""
    fields = {
        "mode": "fork", "trigger": "idle", "importance": 0.5,
        "retrieval_count": 0, "last_retrieved_at": None, "created_at": 1000.0 + end,
        "redacted": 0, "merged_foreign": 0, "superseded_by": None, "indexed": 1,
        "summary": f"old record {channel} {start}-{end}",
    }
    fields.update(kw)
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO memory (channel_id, created_at, summary, importance, "
            "msg_id_start, msg_id_end, retrieval_count, last_retrieved_at, "
            "indexed, superseded_by, redacted, mode, trigger, merged_foreign) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (channel, fields["created_at"], fields["summary"], fields["importance"],
             start, end, fields["retrieval_count"], fields["last_retrieved_at"],
             fields["indexed"], fields["superseded_by"], fields["redacted"],
             fields["mode"], fields["trigger"], fields["merged_foreign"]),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def rows(path: Path, sql: str, *args) -> list[tuple]:
    """Read ``sql`` from the file with a throwaway connection."""
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


class FakeClient:
    """Scripted consolidation / rubric / embedding client."""

    model = "fake-model"

    def __init__(self, replies=None, fail_on_call: int | None = None):
        self.replies = replies
        self.calls: list[list[dict]] = []
        self.fail_on_call = fail_on_call

    def request_model(self, extra_body=None) -> str:
        return self.model

    async def chat(self, messages, tools=None, extra_body=None) -> dict:
        self.calls.append(messages)
        n = len(self.calls)
        if self.fail_on_call == n:
            raise ConnectionError("llama-server unreachable")
        if self.replies is not None:
            text = self.replies(n)
        else:
            text = json.dumps({
                "entries": [{"narrative": f"I recalled stretch {n}.", "topic_tags": ["t"]}],
                "participants": ["kestrel"], "importance": 0.7,
            })
        return {"choices": [{"message": {"content": text}}]}


def inputs_for(_budget: int):
    """The per-channel view inputs the CLI resolves. Their budget is the
    channel's max_context_tokens (large here); the run must replace it with the
    --segment-budget flag, so the tests' small budget is the flag's alone."""
    def make(channel_id: str) -> ViewInputs:
        return ViewInputs(SYSTEM, 24000, "system", lambda messages: None, b"k" * 32)
    return make


def settings(**kw) -> rc.RunSettings:
    base = dict(segment_budget=420, instruction=INSTRUCTION, system_prompt=SYSTEM,
                channels=(CHAN,))
    base.update(kw)
    return rc.RunSettings(**base)


async def run(path: Path, *, client=None, dry_run=False, **kw):
    """Open the copy at ``path`` and run the command's driver on it."""
    client = client or FakeClient()
    lines: list[str] = []
    async with aiosqlite.connect(path) as db:
        report = await rc.run_reconsolidation(
            db, settings=settings(**kw), inputs_for=inputs_for(kw.get("segment_budget", 420)),
            clients=rc.Clients(consolidation=client, rubric=client, embedding=None),
            dry_run=dry_run, echo=lines.append,
        )
    return report, client, lines


@pytest.fixture
async def copy(tmp_path):
    """A marked copy of a live db, made the way the command makes one."""
    live = tmp_path / "live.db"
    await make_live_db(live)
    dest = tmp_path / "copy.db"
    await rc.snapshot_database(live, dest)
    return dest


# ---------------------------------------------------------------------------
# Snapshot and refusal of the live db
# ---------------------------------------------------------------------------


async def test_snapshot_copies_history_and_marks_the_copy(tmp_path):
    live = tmp_path / "live.db"
    await make_live_db(live)
    dest = tmp_path / "copy.db"
    await rc.snapshot_database(live, dest)
    assert rows(dest, "SELECT COUNT(*) FROM message_log") == [(24,)]
    assert rows(dest, "SELECT COUNT(*) FROM reconsolidation_copy") == [(1,)]
    # The source is untouched: no marker table appears in it.
    assert rows(live, "SELECT name FROM sqlite_master WHERE name = 'reconsolidation_copy'") == []


async def test_snapshot_refuses_an_existing_destination(tmp_path):
    live = tmp_path / "live.db"
    await make_live_db(live)
    dest = tmp_path / "copy.db"
    dest.write_text("precious")
    with pytest.raises(click.ClickException) as err:
        await rc.snapshot_database(live, dest)
    # Names the path and the remedy.
    assert str(dest) in err.value.message
    assert "choose a new" in err.value.message
    assert dest.read_text() == "precious"


async def test_a_db_that_is_not_a_copy_is_refused(tmp_path):
    live = tmp_path / "live.db"
    await make_live_db(live)
    with pytest.raises(click.ClickException) as err:
        rc.check_is_copy(live, live_path=None)
    assert str(live) in err.value.message
    assert "--snapshot-from" in err.value.message


async def test_the_configured_live_db_is_refused_even_when_marked(tmp_path, copy):
    # A marked file that is also the daemon's configured db is still live.
    with pytest.raises(click.ClickException) as err:
        rc.check_is_copy(copy, live_path=copy)
    assert str(copy) in err.value.message
    assert "daemon.session_db" in err.value.message


# ---------------------------------------------------------------------------
# A full run
# ---------------------------------------------------------------------------


async def test_full_run_writes_stamped_records_and_supersedes_by_intersection(copy):
    old_a = add_record(copy, CHAN, 1, 9, retrieval_count=3, last_retrieved_at=5000.0, importance=0.9)
    old_b = add_record(copy, CHAN, 11, 23, mode="standalone", retrieval_count=2,
                       last_retrieved_at=6000.0, importance=0.2)
    goal = add_record(copy, CHAN, 1, 23, mode="direct", trigger=None)
    elsewhere = add_record(copy, OTHER, 2, 24)

    report, client, _ = await run(copy)

    assert report.failed == []
    assert report.segments_done >= 2, "the budget must split the channel into several segments"
    new = rows(copy, "SELECT id, mode, trigger, segment_budget, prompt_hash, model, "
                     "msg_id_start, msg_id_end, importance, retrieval_count, last_retrieved_at, "
                     "reply_id, indexed, superseded_by FROM memory WHERE trigger = 'reconsolidate'")
    assert len(new) == report.segments_done
    for (_id, mode, trigger, budget, phash, model, start, end, importance,
         count, last, reply_id, indexed, sup) in new:
        assert (mode, trigger, budget, model) == ("standalone", "reconsolidate", 420, "fake-model")
        assert phash == prompt_hash(INSTRUCTION)
        assert (indexed, sup) == (1, None)
        assert reply_id is not None
        # Stats of the old records the segment's range met are folded in: the
        # sum, the latest access, and the larger importance (the reply says 0.7).
        met = [r for r in ((old_a, 1, 9, 3, 5000.0, 0.9), (old_b, 11, 23, 2, 6000.0, 0.2))
               if r[1] <= end and r[2] >= start]
        assert count == sum(r[3] for r in met)
        assert last == (max(r[4] for r in met) if met else None)
        assert importance == max([0.7] + [r[5] for r in met])
    # Both daemon records are demoted (indexed=0) and point at a new record.
    new_ids = {r[0] for r in new}
    for old in (old_a, old_b):
        indexed, sup = rows(copy, "SELECT indexed, superseded_by FROM memory WHERE id = ?", old)[0]
        assert indexed == 0 and sup in new_ids
    # Goal records and another channel's records are left alone.
    assert rows(copy, "SELECT indexed, superseded_by FROM memory WHERE id IN (?, ?) ORDER BY id",
                goal, elsewhere) == [(1, None), (1, None)]


async def test_records_keep_the_full_reply_and_the_prompts(copy):
    def reply(n):
        return json.dumps({
            "entries": [{"narrative": "Entry one.", "topic_tags": ["a"]},
                        {"narrative": "Entry two.", "topic_tags": ["b"]}],
            "participants": ["kestrel"], "importance": 0.4,
            "claims": [{"text": "kestrel likes tea"}],
        })

    report, _, _ = await run(copy, client=FakeClient(reply))
    stored = rows(copy, "SELECT id, channel_id, msg_id_start, msg_id_end, segment_budget, "
                        "prompt_hash, system_prompt_hash, model, reply FROM memory_reply ORDER BY id")
    assert len(stored) == report.segments_done
    for _id, channel, start, end, budget, phash, shash, model, text in stored:
        assert channel == CHAN and budget == 420 and model == "fake-model"
        # The reply is kept verbatim, extra keys (a later `claims` array) included.
        assert json.loads(text)["claims"] == [{"text": "kestrel likes tea"}]
        assert phash == prompt_hash(INSTRUCTION) and shash == prompt_hash(SYSTEM)
    # Two entries per segment, each its own row sharing the segment's range.
    per_reply = rows(copy, "SELECT reply_id, COUNT(*), MIN(msg_id_start), MAX(msg_id_end) "
                           "FROM memory WHERE trigger = 'reconsolidate' GROUP BY reply_id")
    assert all(count == 2 for _, count, _, _ in per_reply)
    # Both instruction texts are kept once per hash (PRD Memory 7).
    texts = dict(rows(copy, "SELECT hash, text FROM memory_prompt"))
    assert texts[prompt_hash(INSTRUCTION)] == INSTRUCTION and texts[prompt_hash(SYSTEM)] == SYSTEM


async def test_records_that_are_not_consolidations_of_this_channel_stay_put(copy):
    redacted = add_record(copy, CHAN, 1, 9, redacted=1)
    foreign = add_record(copy, CHAN, 1, 9, merged_foreign=1)
    folded = add_record(copy, CHAN, 1, 9, indexed=0, superseded_by=foreign)
    raw = add_record(copy, CHAN, 1, 9, mode="raw")
    await run(copy)
    # A tombstone, a record holding another channel's text, an already-folded
    # record and a raw-transcript record are none of the rule's business.
    assert rows(copy, "SELECT id, indexed, superseded_by FROM memory WHERE id IN (?,?,?,?) ORDER BY id",
                redacted, foreign, folded, raw) == [
        (redacted, 1, None), (foreign, 1, None), (folded, 0, foreign), (raw, 1, None)]


async def test_a_record_straddling_two_segments_is_credited_to_both(copy):
    old = add_record(copy, CHAN, 1, 23, retrieval_count=4)
    report, _, _ = await run(copy)
    assert report.segments_done >= 2
    counts = rows(copy, "SELECT retrieval_count FROM memory WHERE trigger = 'reconsolidate'")
    # The owner's sum ruling: each segment it met inherits the full count.
    assert all(c == (4,)[0] for (c,) in counts)
    # It points at the first of them, not the last.
    first = rows(copy, "SELECT MIN(id) FROM memory WHERE trigger = 'reconsolidate'")[0][0]
    assert rows(copy, "SELECT superseded_by FROM memory WHERE id = ?", old) == [(first,)]


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


async def test_since_until_select_by_created_at_and_leave_the_rest(copy):
    early = add_record(copy, CHAN, 1, 7, created_at=100.0)
    inside = add_record(copy, CHAN, 9, 15, created_at=500.0)
    late = add_record(copy, CHAN, 17, 23, created_at=900.0)
    report, client, _ = await run(copy, since=400.0, until=600.0)
    assert [rows(copy, "SELECT indexed FROM memory WHERE id = ?", i)[0][0]
            for i in (early, inside, late)] == [1, 0, 1]
    # Only the selected stretch was sent to the model.
    spans = rows(copy, "SELECT MIN(msg_id_start), MAX(msg_id_end) FROM memory "
                       "WHERE trigger = 'reconsolidate'")[0]
    assert spans[0] >= 9 and spans[1] <= 15
    assert report.records_superseded == 1


async def test_min_source_tokens_narrows_to_big_stretches(copy):
    small = add_record(copy, CHAN, 1, 3, created_at=500.0)
    big = add_record(copy, CHAN, 5, 23, created_at=500.0)
    await run(copy, since=0.0, min_source_tokens=200)
    assert [rows(copy, "SELECT indexed FROM memory WHERE id = ?", i)[0][0] for i in (small, big)] == [1, 0]


async def test_scope_widens_to_cover_an_intersecting_record(copy):
    # `selected` is in the window; `reaching` is not, but its range meets
    # selected's and runs past it, so superseding it would leave 17-23 with no
    # indexed record unless the stretch grows to cover it.
    selected = add_record(copy, CHAN, 9, 15, created_at=500.0)
    reaching = add_record(copy, CHAN, 13, 23, created_at=900.0)
    report, _, lines = await run(copy, since=400.0, until=600.0)
    assert rows(copy, "SELECT indexed FROM memory WHERE id IN (?, ?) ORDER BY id", selected, reaching) \
        == [(0,), (0,)]
    assert rows(copy, "SELECT MAX(msg_id_end) FROM memory WHERE trigger = 'reconsolidate'") == [(23,)]
    assert report.pulled == [reaching]


async def test_dry_run_reports_segments_tokens_and_pulled_records_and_writes_nothing(copy):
    add_record(copy, CHAN, 9, 15, created_at=500.0)
    reaching = add_record(copy, CHAN, 13, 23, created_at=900.0)
    before = rows(copy, "SELECT * FROM memory ORDER BY id")
    tables = rows(copy, "SELECT name FROM sqlite_master ORDER BY name")
    report, client, lines = await run(copy, dry_run=True, since=400.0, until=600.0)
    text = "\n".join(lines)
    assert client.calls == []
    assert report.segments_total >= 1 and report.estimated_tokens > 0
    assert CHAN in text and "tokens" in text
    # The operator sees the records the supersede rule took beyond the selection.
    assert f"memory {reaching}" in text
    assert rows(copy, "SELECT * FROM memory ORDER BY id") == before
    assert rows(copy, "SELECT name FROM sqlite_master ORDER BY name") == tables


# ---------------------------------------------------------------------------
# Resume, failure, identity
# ---------------------------------------------------------------------------


async def test_a_run_that_stops_resumes_without_redoing_finished_segments(copy):
    old = add_record(copy, CHAN, 1, 23)
    first = FakeClient(fail_on_call=2)
    with pytest.raises(ConnectionError):
        await run(copy, client=first)
    done = rows(copy, "SELECT COUNT(*) FROM memory_reply")[0][0]
    assert done == 1
    # The old record is demoted by the first segment already, atomically with it.
    assert rows(copy, "SELECT indexed FROM memory WHERE id = ?", old) == [(0,)]

    second = FakeClient()
    report, _, _ = await run(copy, client=second)
    assert report.segments_skipped == 1
    # The second run calls the model only for what the first did not finish.
    assert len(second.calls) == report.segments_done
    assert report.segments_done + report.segments_skipped == report.segments_total
    total = rows(copy, "SELECT COUNT(*) FROM memory_reply")[0][0]
    assert total == report.segments_total


async def test_resume_with_other_settings_is_refused_naming_the_difference(copy):
    first = FakeClient(fail_on_call=2)
    with pytest.raises(ConnectionError):
        await run(copy, client=first)
    with pytest.raises(click.ClickException) as err:
        await run(copy, segment_budget=900)
    assert "segment_budget" in err.value.message and "420" in err.value.message
    assert "900" in err.value.message
    assert "new --snapshot-from copy" in err.value.message


async def test_an_unusable_reply_leaves_the_segment_and_its_old_records_alone(copy):
    old = add_record(copy, CHAN, 1, 23)
    report, client, _ = await run(copy, client=FakeClient(lambda n: "not json at all"))
    assert report.segments_done == 0 and len(report.failed) == report.segments_total
    # Two attempts per segment, then it is left for a rerun.
    assert len(client.calls) == 2 * report.segments_total
    assert rows(copy, "SELECT indexed, superseded_by FROM memory WHERE id = ?", old) == [(1, None)]
    assert rows(copy, "SELECT COUNT(*) FROM memory_reply") == [(0,)]
    # The report names the channel and the range of what was left.
    first = report.failed[0]
    assert first[0] == CHAN and first[1] <= first[2]


async def test_a_failure_inside_a_segment_leaves_nothing_of_it(copy):
    # The second entry's insert fails after the reply row, the first entry and
    # (were it earlier) the supersede: all of it must roll back on the same
    # connection, not only vanish when the process exits.
    old = add_record(copy, CHAN, 1, 23)
    conn = sqlite3.connect(copy)
    conn.execute(
        "CREATE TRIGGER boom BEFORE INSERT ON memory WHEN new.summary = 'Entry two.' "
        "BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    conn.commit()
    conn.close()

    def reply(n):
        return json.dumps({"entries": [{"narrative": "Entry one."}, {"narrative": "Entry two."}],
                           "participants": [], "importance": 0.5})

    async with aiosqlite.connect(copy) as db:
        with pytest.raises(sqlite3.IntegrityError):
            await rc.run_reconsolidation(
                db, settings=settings(), inputs_for=inputs_for(420),
                clients=rc.Clients(FakeClient(reply), FakeClient(reply)), echo=lambda _: None,
            )
        # Read on the failing connection itself: nothing of the segment is left.
        async with db.execute("SELECT COUNT(*) FROM memory_reply") as cur:
            assert (await cur.fetchone())[0] == 0
        async with db.execute("SELECT COUNT(*) FROM memory WHERE trigger = 'reconsolidate'") as cur:
            assert (await cur.fetchone())[0] == 0
        async with db.execute("SELECT indexed FROM memory WHERE id = ?", (old,)) as cur:
            assert (await cur.fetchone())[0] == 1


async def test_message_log_is_never_written(copy):
    before = rows(copy, "SELECT * FROM message_log ORDER BY id")
    await run(copy)
    assert rows(copy, "SELECT * FROM message_log ORDER BY id") == before


# ---------------------------------------------------------------------------
# Vectors, valence, the rubric, and the remaining refusals
# ---------------------------------------------------------------------------

DIMS = 4
IDENTITY = ("enc", DIMS, "", "")


def add_vec_surface(path: Path, *, meta=IDENTITY) -> None:
    """Give the copy the daemon's vector surface: memory_vec and embedding_meta."""
    import sqlite_vec
    from corvidae_cognition.memory import EMBEDDING_META_DDL

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute(
        f"CREATE VIRTUAL TABLE memory_vec USING vec0(memory_id INTEGER PRIMARY KEY, "
        f"embedding FLOAT[{DIMS}] distance_metric=cosine)"
    )
    conn.execute(EMBEDDING_META_DDL)
    conn.execute("INSERT INTO embedding_meta VALUES (?, ?, ?, ?)", meta)
    conn.commit()
    conn.close()


def vec_ids(path: Path) -> set[int]:
    """The memory ids that have a vector row."""
    import sqlite_vec

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        return {r[0] for r in conn.execute("SELECT memory_id FROM memory_vec")}
    finally:
        conn.close()


class FakeEmbedder:
    """Returns a fixed vector per text, or raises."""

    def __init__(self, vector=None, error: Exception | None = None):
        self.vector = vector or [1.0, 0.0, 0.0, 0.0]
        self.error = error
        self.calls: list[list[str]] = []

    async def embed(self, texts, kind):
        self.calls.append(texts)
        assert kind == "document"
        if self.error:
            raise self.error
        return [list(self.vector) for _ in texts]


async def run_with_embedding(path: Path, embedder, identity=IDENTITY):
    """`run`, with an embedding client configured."""
    client = FakeClient()
    async with aiosqlite.connect(path) as db:
        report = await rc.run_reconsolidation(
            db, settings=settings(), inputs_for=inputs_for(420),
            clients=rc.Clients(client, client, embedder, identity), echo=lambda _: None,
        )
    return report


async def test_vectors_are_written_for_new_records_and_dropped_for_old_ones(copy):
    pytest.importorskip("sqlite_vec")
    add_vec_surface(copy)
    old = add_record(copy, CHAN, 1, 23)
    import sqlite_vec

    conn = sqlite3.connect(copy)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                 (old, sqlite_vec.serialize_float32([0.0, 1.0, 0.0, 0.0])))
    conn.commit()
    conn.close()

    embedder = FakeEmbedder()
    report = await run_with_embedding(copy, embedder)
    new = {r[0] for r in rows(copy, "SELECT id FROM memory WHERE trigger = 'reconsolidate'")}
    # Every new record is embedded and indexed; the superseded one's vector is gone.
    assert len(new) == report.records_written and vec_ids(copy) == new
    assert rows(copy, "SELECT DISTINCT embedded FROM memory WHERE trigger = 'reconsolidate'") == [(1,)]
    assert rows(copy, "SELECT embedded FROM memory WHERE id = ?", old) == [(0,)]


async def test_a_mismatched_encoder_stores_records_unembedded_and_says_so(copy, caplog):
    pytest.importorskip("sqlite_vec")
    add_vec_surface(copy, meta=("other-encoder", DIMS, "", ""))
    embedder = FakeEmbedder()
    with caplog.at_level("WARNING"):
        await run_with_embedding(copy, embedder)
    # Encoders are never mixed: no call, no vector, a warning naming both.
    assert embedder.calls == [] and vec_ids(copy) == set()
    assert rows(copy, "SELECT DISTINCT embedded FROM memory WHERE trigger = 'reconsolidate'") == [(0,)]
    assert "other-encoder" in caplog.text and "enc" in caplog.text


@pytest.mark.parametrize("embedder", [
    FakeEmbedder(error=ConnectionError("embedding server down")),
    FakeEmbedder(vector=[1.0, 0.0]),   # wrong width
])
async def test_a_failed_or_misshapen_embedding_degrades_to_unembedded(copy, embedder, caplog):
    pytest.importorskip("sqlite_vec")
    add_vec_surface(copy)
    with caplog.at_level("WARNING"):
        report = await run_with_embedding(copy, embedder)
    assert report.failed == [] and report.records_written > 0
    assert vec_ids(copy) == set()
    assert rows(copy, "SELECT DISTINCT embedded FROM memory WHERE trigger = 'reconsolidate'") == [(0,)]


async def test_a_vec_table_without_the_extension_is_refused_before_any_write(copy, monkeypatch):
    sqlite_vec = pytest.importorskip("sqlite_vec")
    add_vec_surface(copy)
    tables = rows(copy, "SELECT name FROM sqlite_master ORDER BY name")

    def broken():
        raise AttributeError("no extension here")

    monkeypatch.setattr(sqlite_vec, "loadable_path", broken)
    with pytest.raises(click.ClickException) as err:
        await run(copy)
    assert "memory_vec" in err.value.message and "cognition extra" in err.value.message
    assert rows(copy, "SELECT name FROM sqlite_master ORDER BY name") == tables


async def test_valence_is_the_mean_of_the_range_appraisals(copy):
    conn = sqlite3.connect(copy)
    conn.execute("CREATE TABLE exchange_log (trace_id TEXT, channel_id TEXT, "
                 "message_rowid INTEGER, appraisal TEXT)")
    for trace, rowid, appraisal in [
        ("a", 3, {"stage2": {"valence": 0.2}}),
        ("b", 5, {"stage2": {"valence": 0.6}}),
        ("c", 7, "{not json"),
        ("d", 2, {"stage2": {"valence": -0.9}}),          # the other channel's id space
    ]:
        channel = OTHER if trace == "d" else CHAN
        conn.execute("INSERT INTO exchange_log VALUES (?, ?, ?, ?)", (
            trace, channel, rowid, appraisal if isinstance(appraisal, str) else json.dumps(appraisal)))
    conn.commit()
    conn.close()
    await run(copy, segment_budget=24000)    # one segment spanning the channel
    valence = rows(copy, "SELECT valence FROM memory WHERE trigger = 'reconsolidate'")[0][0]
    # Mean of this channel's readable appraisals (0.2, 0.6); the corrupt row is skipped.
    assert valence == pytest.approx(0.4)


async def test_a_reply_without_importance_is_rated_by_the_rubric(copy):
    consolidation = FakeClient(lambda n: json.dumps(
        {"entries": [{"narrative": "An entry."}], "participants": []}))
    rubric = FakeClient(lambda n: json.dumps({"importance": 0.9}))
    async with aiosqlite.connect(copy) as db:
        report = await rc.run_reconsolidation(
            db, settings=settings(), inputs_for=inputs_for(420),
            clients=rc.Clients(consolidation, rubric), echo=lambda _: None,
        )
    assert len(rubric.calls) == report.segments_done
    assert rows(copy, "SELECT DISTINCT importance FROM memory WHERE trigger = 'reconsolidate'") == [(0.9,)]


async def test_a_segment_without_dialog_is_finished_without_a_model_call(tmp_path):
    live = tmp_path / "live.db"
    await make_live_db(live, per_channel=0)
    async with aiosqlite.connect(live) as db:
        await db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
            "VALUES (?, ?, 1.0, 'message')",
            (CHAN, json.dumps({"role": "tool", "content": "ls output", "tool_call_id": "t1"})),
        )
        await db.commit()
    dest = tmp_path / "copy.db"
    await rc.snapshot_database(live, dest)
    report, client, _ = await run(dest)
    # Nothing to remember: no call, no record, but a resume skips it.
    assert client.calls == [] and report.records_written == 0
    assert rows(dest, "SELECT reply FROM memory_reply") == [(None,)]
    again, _, _ = await run(dest)
    assert again.segments_skipped == 1


async def test_an_unknown_channel_is_refused_naming_it_and_the_known_ones(copy):
    with pytest.raises(click.ClickException) as err:
        await run(copy, channels=("irc:#nowhere",))
    assert "irc:#nowhere" in err.value.message and CHAN in err.value.message
    assert "--channel" in err.value.message


async def test_a_database_without_a_memory_table_is_refused_with_the_remedy(tmp_path):
    live = tmp_path / "live.db"
    async with aiosqlite.connect(live) as db:
        await init_db(db)
    dest = tmp_path / "copy.db"
    await rc.snapshot_database(live, dest)
    with pytest.raises(click.ClickException) as err:
        await run(dest)
    assert "memory table" in err.value.message and "snapshot it again" in err.value.message


# ---------------------------------------------------------------------------
# Redaction reaches the stored replies
# ---------------------------------------------------------------------------


async def test_redacting_a_message_tombstones_the_replies_that_restate_it(copy):
    from corvidae.commands.redact import redact_messages

    await run(copy)
    spans = rows(copy, "SELECT id, msg_id_start, msg_id_end FROM memory_reply ORDER BY id")
    assert len(spans) >= 2
    hit, other = spans[0], spans[-1]
    victim = hit[1]                           # a message inside the first segment only
    async with aiosqlite.connect(copy) as db:
        await db.execute("PRAGMA journal_mode = WAL")
        result = await redact_messages(db, [victim])
    # The scrubber ran through the real entry point and changed one reply.
    assert result["scrubbed"]["memory_reply"] == 1
    replies = dict(rows(copy, "SELECT id, reply FROM memory_reply"))
    assert "redacted" in replies[hit[0]] and "stretch" not in replies[hit[0]]
    assert "stretch" in replies[other[0]]
    # The row stays, so a resume still counts the segment as finished.
    assert rows(copy, "SELECT COUNT(*) FROM memory_reply") == [(len(spans),)]


async def test_a_dry_run_redaction_counts_replies_and_leaves_them(copy):
    from corvidae.commands.redact import redact_messages

    await run(copy)
    victim = rows(copy, "SELECT msg_id_start FROM memory_reply ORDER BY id")[0][0]
    before = rows(copy, "SELECT id, reply FROM memory_reply")
    async with aiosqlite.connect(copy) as db:
        result = await redact_messages(db, [victim], dry_run=True)
    assert result["scrubbed"]["memory_reply"] == 1
    assert rows(copy, "SELECT id, reply FROM memory_reply") == before


async def test_a_database_without_replies_has_nothing_for_the_scrubber(tmp_path):
    live = tmp_path / "live.db"
    await make_live_db(live)
    async with aiosqlite.connect(live) as db:
        assert await rc.redact_memory_reply(db, CHAN, [1], False, "x") is None


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def invoke(*args):
    return CliRunner().invoke(rc.reconsolidate_command, list(args), catch_exceptions=False)


def test_cli_refuses_a_db_that_is_not_a_copy(tmp_path):
    # Sync: the command runs its own event loop, which a running one forbids.
    live = tmp_path / "live.db"
    asyncio.run(make_live_db(live))
    prompt = tmp_path / "sys.txt"
    prompt.write_text(SYSTEM)
    result = invoke("--db", str(live), "--system-prompt", str(prompt),
                    "--segment-budget", "400", "--config", str(tmp_path / "agent.yaml"))
    assert result.exit_code != 0
    assert str(live) in result.output and "--snapshot-from" in result.output


@pytest.mark.parametrize("flag", ["--since", "--until"])
def test_cli_names_a_bad_time_flag_its_value_and_the_accepted_forms(tmp_path, flag):
    prompt = tmp_path / "sys.txt"
    prompt.write_text(SYSTEM)
    result = invoke("--db", str(tmp_path / "c.db"), "--system-prompt", str(prompt),
                    "--segment-budget", "400", flag, "last tuesday")
    assert result.exit_code != 0
    assert flag in result.output and "last tuesday" in result.output
    assert "ISO" in result.output


def test_cli_names_a_bad_segment_budget(tmp_path):
    prompt = tmp_path / "sys.txt"
    prompt.write_text(SYSTEM)
    result = invoke("--db", str(tmp_path / "c.db"), "--system-prompt", str(prompt),
                    "--segment-budget", "0")
    assert result.exit_code != 0
    assert "--segment-budget" in result.output and "0" in result.output


def test_the_command_is_registered_beside_goals():
    """Without its entry point the command is never listed, and nothing says so."""
    import tomllib

    text = Path(__file__).resolve().parents[2].joinpath(
        "packages/corvidae-cognition/pyproject.toml").read_text()
    commands = tomllib.loads(text)["project"]["entry-points"]["corvidae.commands"]
    assert commands["reconsolidate"] == "corvidae_cognition.reconsolidate:reconsolidate_command"


# ---------------------------------------------------------------------------
# The real CLI path: config, clients, per-channel resolution
# ---------------------------------------------------------------------------


class FakeLLMClient(FakeClient):
    """Stands in for ``LLMClient``: takes the profile's kwargs, no network."""

    instances: list["FakeLLMClient"] = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.started = self.stopped = False
        FakeLLMClient.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


def test_cli_end_to_end_snapshots_runs_and_leaves_the_live_db_alone(tmp_path, monkeypatch):
    # The whole command through click: snapshot, mark, resolve config, call
    # the (fake) model, write the copy. The live file never changes.
    live = tmp_path / "sessions.db"
    asyncio.run(make_live_db(live))
    old = add_record(live, CHAN, 1, 23)
    before = rows(live, "SELECT * FROM memory")
    (tmp_path / "agent.yaml").write_text(
        "llm:\n  providers:\n    main:\n      base_url: http://127.0.0.1:1/v1\n"
        "  profiles:\n    main:\n      model: m\n"
    )
    prompt = tmp_path / "sys.txt"
    prompt.write_text(SYSTEM)
    monkeypatch.setattr(rc, "LLMClient", FakeLLMClient)
    FakeLLMClient.instances.clear()

    copy = tmp_path / "variant-a.db"
    result = invoke("--db", str(copy), "--snapshot-from", str(live),
                    "--config", str(tmp_path / "agent.yaml"), "--system-prompt", str(prompt),
                    "--segment-budget", "420", "--channel", CHAN)
    assert result.exit_code == 0, result.output
    assert rows(live, "SELECT * FROM memory") == before
    assert rows(copy, "SELECT indexed FROM memory WHERE id = ?", old) == [(0,)]
    assert rows(copy, "SELECT COUNT(*) FROM memory WHERE trigger = 'reconsolidate'")[0][0] >= 2
    # Clients are started for the run and stopped after it.
    assert FakeLLMClient.instances and all(c.started and c.stopped for c in FakeLLMClient.instances)

    # Rerunning the same command resumes: nothing is redone.
    calls = sum(len(c.calls) for c in FakeLLMClient.instances)
    again = invoke("--db", str(copy), "--config", str(tmp_path / "agent.yaml"),
                   "--system-prompt", str(prompt), "--segment-budget", "420", "--channel", CHAN)
    assert again.exit_code == 0, again.output
    assert sum(len(c.calls) for c in FakeLLMClient.instances) == calls
    assert "already done" in again.output


def test_cli_refuses_the_configured_live_db_even_when_it_is_marked(tmp_path):
    # --db resolves to daemon.session_db (default sessions.db beside the config).
    live = tmp_path / "sessions.db"
    asyncio.run(make_live_db(live))
    conn = sqlite3.connect(live)
    conn.execute(rc.COPY_MARKER_DDL)
    conn.commit()
    conn.close()
    (tmp_path / "agent.yaml").write_text("{}\n")
    prompt = tmp_path / "sys.txt"
    prompt.write_text(SYSTEM)
    result = invoke("--db", str(live), "--config", str(tmp_path / "agent.yaml"),
                    "--system-prompt", str(prompt), "--segment-budget", "420")
    assert result.exit_code != 0
    assert "daemon.session_db" in result.output and str(live) in result.output


def test_a_stop_on_a_transport_error_names_the_remedy_and_keeps_finished_segments(tmp_path, monkeypatch):
    live = tmp_path / "sessions.db"
    asyncio.run(make_live_db(live))
    (tmp_path / "agent.yaml").write_text(
        "llm:\n  providers:\n    main:\n      base_url: http://127.0.0.1:1/v1\n"
        "  profiles:\n    main:\n      model: m\n"
    )
    prompt = tmp_path / "sys.txt"
    prompt.write_text(SYSTEM)

    class Failing(FakeLLMClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fail_on_call = 2

    monkeypatch.setattr(rc, "LLMClient", Failing)
    copy = tmp_path / "c.db"
    result = invoke("--db", str(copy), "--snapshot-from", str(live),
                    "--config", str(tmp_path / "agent.yaml"), "--system-prompt", str(prompt),
                    "--segment-budget", "420", "--channel", CHAN)
    assert result.exit_code != 0
    assert "ConnectionError" in result.output and str(copy) in result.output
    assert "rerun the same command" in result.output
    assert rows(copy, "SELECT COUNT(*) FROM memory_reply") == [(1,)]


def test_per_channel_config_reaches_the_view_inputs():
    # The reasoning policy is resolved per channel the way the daemon layers
    # it: exact entry > transport-wide entry > agent:. The three layers choose
    # three different policies, and the two probes tell all three apart:
    #   completed exchange (user, assistant+reasoning, user): kept only by keep
    #   current exchange   (user, assistant+reasoning):       dropped only by drop
    config = {
        "agent": {"reasoning_history": "keep"},
        "channels": {
            "irc:*": {"reasoning_history": "exchange"},
            "irc:#lex": {"reasoning_history": "drop"},
        },
    }
    resolve = rc._inputs_resolver(config, SYSTEM)

    def survivors(channel_id: str) -> tuple[bool, bool]:
        """(completed exchange keeps its reasoning, current exchange does)."""
        done = [{"role": "user", "content": "a"},
                {"role": "assistant", "content": "b", "reasoning_content": "t"},
                {"role": "user", "content": "c"}]
        current = [{"role": "user", "content": "a"},
                   {"role": "assistant", "content": "b", "reasoning_content": "t"}]
        view = resolve(channel_id)
        view.apply_reasoning_history(done)
        view.apply_reasoning_history(current)
        return "reasoning_content" in done[1], "reasoning_content" in current[1]

    assert survivors("irc:#lex") == (False, False)    # exact entry: drop
    assert survivors("irc:#other") == (False, True)   # transport-wide entry: exchange
    assert survivors("signal:+1") == (True, True)     # no entry: agent keep
    # The prompt is the operator's file, the budget the flag's (set by the run).
    assert resolve("irc:#lex").system_prompt == SYSTEM
