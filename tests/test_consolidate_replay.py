"""Tests for harness/consolidate_replay.py and the selection helpers it
shares with MemoryPlugin.

Pins R2 — replay must reproduce the daemon's consolidation call, not an
approximation. The same rows get selected, the same dialog survives
filtering, and the same two-message payload is assembled. It also pins the
reason the selection is split in two: `_consolidate_range` needs the raw
row endpoints for `memory.msg_id_start`/`msg_id_end`, the importance prior,
and the valence lookup, so a helper that returns only dialog would silently
change what the daemon stores.

Everything here is offline: no llama-server, no remote host, no live daemon.
The
LLM call is exercised through an injected fake.
"""
import json
import sqlite3

import aiosqlite
import pytest

from corvidae.memory import _dialog_transcript


# ---------------------------------------------------------------------------
# Row fixtures — the shape SELECT id, message, message_type returns
# ---------------------------------------------------------------------------


def _row(row_id: int, role: str, content, message_type: str = "message"):
    """One message_log row tuple as the consolidation query yields it."""
    return (row_id, json.dumps({"role": role, "content": content}), message_type)


# Every row class the real #s-compact 76-163 fixture contains, plus the
# degenerate content cases. Only the two dialog rows may survive.
MIXED_ROWS = [
    _row(10, "user", "what were the column names?"),
    _row(11, "assistant", "CRV-1000 onward."),
    _row(12, "assistant", "earlier we discussed...", message_type="summary"),
    _row(13, "tool", "web_fetch result"),
    _row(14, "system", "you are Buster"),
    _row(15, "user", ""),
    _row(16, "user", "   "),
    _row(17, "user", None),
]


def _seed_db_sync(path, rows, channel_id="irc:#s-compact"):
    """Same fixture for tests that only drive the synchronous main().

    Keeping them sync matters: an async test would force main() to detect a
    running loop and hand off to a worker thread, machinery the real CLI —
    which is only ever entered from a shell — never needs.
    """
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE message_log ("
        "id INTEGER PRIMARY KEY, channel_id TEXT, message TEXT, message_type TEXT)"
    )
    db.executemany(
        "INSERT INTO message_log (id, channel_id, message, message_type) "
        "VALUES (?, ?, ?, ?)",
        [(r[0], channel_id, r[1], r[2]) for r in rows],
    )
    db.commit()
    db.close()


async def _seed_db(path, rows, channel_id="irc:#s-compact"):
    """Minimal message_log holding the columns the consolidation query reads."""
    db = await aiosqlite.connect(path)
    await db.execute(
        "CREATE TABLE message_log ("
        "id INTEGER PRIMARY KEY, channel_id TEXT, message TEXT, message_type TEXT)"
    )
    for row_id, message, message_type in rows:
        await db.execute(
            "INSERT INTO message_log (id, channel_id, message, message_type) "
            "VALUES (?, ?, ?, ?)",
            (row_id, channel_id, message, message_type),
        )
    await db.commit()
    return db


# ---------------------------------------------------------------------------
# Fakes — the LLM seam, so nothing here needs a live server
# ---------------------------------------------------------------------------


class _FakeLLMPlugin:
    """Stands in for LLMPlugin: hands out one client, records the role asked for."""

    def __init__(self, client):
        self._client = client
        self.roles_requested = []

    def get_client(self, role):
        self.roles_requested.append(role)
        return self._client


class _FakeClient:
    """Stands in for LLMClient: records payloads, returns canned summaries."""

    def __init__(self, model="qwen3.6:35b", base_url="http://127.0.0.1:8080/v1"):
        self.calls = []
        self.model = model
        self.base_url = base_url

    async def start(self):
        pass

    async def stop(self):
        pass

    async def chat(self, messages, tools=None, extra_body=None):
        self.calls.append(messages)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": f"summary {len(self.calls)}",
                                "topic_tags": ["books"],
                                "participants": ["Schuyler"],
                            }
                        )
                    }
                }
            ]
        }

# ---------------------------------------------------------------------------
# dialog_from_rows — the filter, shared with _consolidate_range
# ---------------------------------------------------------------------------


def test_dialog_from_rows_keeps_only_real_user_and_assistant_dialog():
    from corvidae.memory import dialog_from_rows

    dialog = dialog_from_rows(MIXED_ROWS)

    assert [m["role"] for m in dialog] == ["user", "assistant"]
    assert [m["content"] for m in dialog] == [
        "what were the column names?",
        "CRV-1000 onward.",
    ]


def test_dialog_from_rows_drops_summary_rows():
    """8 such rows sit inside the real 76-163 fixture; none may reach the LLM."""
    from corvidae.memory import dialog_from_rows

    only_summaries = [_row(1, "assistant", "recap", message_type="summary")]
    assert dialog_from_rows(only_summaries) == []


def test_dialog_from_rows_preserves_input_order():
    from corvidae.memory import dialog_from_rows

    rows = [
        _row(3, "user", "third"),
        _row(1, "user", "first"),
        _row(2, "assistant", "second"),
    ]
    assert [m["content"] for m in dialog_from_rows(rows)] == [
        "third",
        "first",
        "second",
    ]


# ---------------------------------------------------------------------------
# fetch_range_rows — the query, shared with _consolidate_range
# ---------------------------------------------------------------------------


async def test_fetch_range_rows_is_half_open_and_id_ordered(tmp_path):
    from corvidae.memory import fetch_range_rows

    rows = [_row(i, "user", f"m{i}") for i in range(1, 6)]
    db = await _seed_db(tmp_path / "t.db", rows)
    try:
        got = await fetch_range_rows(db, "irc:#s-compact", 2, 4)
    finally:
        await db.close()

    assert [r[0] for r in got] == [3, 4]


async def test_fetch_range_rows_excludes_other_channels(tmp_path):
    from corvidae.memory import fetch_range_rows

    db = await _seed_db(tmp_path / "t.db", [_row(1, "user", "mine")])
    await db.execute(
        "INSERT INTO message_log (id, channel_id, message, message_type) "
        "VALUES (?, ?, ?, ?)",
        (2, "irc:#chat", json.dumps({"role": "user", "content": "theirs"}), "message"),
    )
    await db.commit()
    try:
        got = await fetch_range_rows(db, "irc:#s-compact", 0, 99)
    finally:
        await db.close()

    assert [r[0] for r in got] == [1]


async def test_fetch_range_rows_returns_non_dialog_rows_too(tmp_path):
    """The raw endpoints are what _consolidate_range stores as
    memory.msg_id_start/msg_id_end and passes to the importance prior and
    mean_valence. A summary row at the edge of a range still sets that
    endpoint, so the fetch must not pre-filter."""
    from corvidae.memory import fetch_range_rows

    rows = [
        _row(20, "assistant", "recap", message_type="summary"),
        _row(21, "user", "real dialog"),
        _row(22, "tool", "web_fetch result"),
    ]
    db = await _seed_db(tmp_path / "t.db", rows)
    try:
        got = await fetch_range_rows(db, "irc:#s-compact", 19, 22)
    finally:
        await db.close()

    assert [r[0] for r in got] == [20, 21, 22]
    assert got[0][0] == 20 and got[-1][0] == 22


# ---------------------------------------------------------------------------
# Range translation
# ---------------------------------------------------------------------------


def test_parse_range_is_inclusive_and_translates_to_half_open():
    """--range 76:163 reads off a memory row's msg_id_start/msg_id_end, which
    are inclusive; the query bound is (after, through]."""
    from harness.consolidate_replay import parse_range

    assert parse_range("76:163") == (75, 163)



# ---------------------------------------------------------------------------
# Payload assembly — identical to _summarize_range's two-message call
# ---------------------------------------------------------------------------


async def test_build_messages_matches_what_the_plugin_actually_sends():
    """R2: pinned against the production code path, not a hand-copied shape.

    If _summarize_range grows a third message or reorders roles, this fails —
    which a hand-written expected-payload literal would not."""
    from corvidae.memory import MemoryPlugin
    from harness.consolidate_replay import build_messages

    dialog = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    client = _FakeClient()
    plugin = MemoryPlugin()
    plugin._consolidation_prompt = "SYSTEM TEXT"
    plugin._llm = lambda: _FakeLLMPlugin(client)

    await plugin._summarize_range(dialog)

    assert client.calls[0] == build_messages("SYSTEM TEXT", dialog)


async def test_replay_uses_the_same_llm_role_as_the_daemon():
    """D3 reads llm.background; the daemon's consolidation call must agree.

    Characterization test — expected green while the rest of this file is red.
    It pins existing daemon behavior rather than pending work: if
    _summarize_range ever asked for a different role, the replay tool would
    keep asking for `background` and silently exercise a different path than
    production."""
    from corvidae.memory import MemoryPlugin

    client = _FakeClient()
    llm = _FakeLLMPlugin(client)
    plugin = MemoryPlugin()
    plugin._llm = lambda: llm

    await plugin._summarize_range([{"role": "user", "content": "hi"}])

    assert llm.roles_requested == ["background"]


# ---------------------------------------------------------------------------
# R8 — failures name what was observed
# ---------------------------------------------------------------------------


def test_missing_db_exits_naming_the_path(tmp_path, capsys):
    from harness.consolidate_replay import main

    missing = tmp_path / "nope.db"
    rc = main(["--db", str(missing), "--channel", "irc:#s-compact", "--range", "1:2"])

    assert rc != 0
    assert str(missing) in capsys.readouterr().err


def test_unknown_channel_lists_the_channels_present(tmp_path, capsys):
    from harness.consolidate_replay import main

    db_path = tmp_path / "t.db"
    _seed_db_sync(db_path, [_row(1, "user", "hi")])

    rc = main(["--db", str(db_path), "--channel", "irc:#absent", "--range", "0:9"])

    err = capsys.readouterr().err
    assert rc != 0
    assert "irc:#absent" in err
    assert "irc:#s-compact" in err


def test_empty_range_distinguishes_no_rows_from_all_filtered(tmp_path, capsys):
    """'0 rows in range' and 'rows present but none are dialog' are different
    problems; an operator must be able to tell them apart."""
    from harness.consolidate_replay import main

    db_path = tmp_path / "t.db"
    _seed_db_sync(db_path, [_row(5, "assistant", "recap", message_type="summary")])

    rc = main(["--db", str(db_path), "--channel", "irc:#s-compact", "--range", "5:5"])

    err = capsys.readouterr().err
    assert rc != 0
    # Machine-checkable counts: a bare "1"/"0" would match almost any message.
    assert "rows=1" in err
    assert "dialog=0" in err


def test_unreachable_server_exits_naming_the_base_url(tmp_path, capsys):
    """R8's third condition. Port 1 on loopback is reliably refused, and
    max_retries=0 keeps this well inside the 15s global timeout."""
    from harness.consolidate_replay import main

    db_path = tmp_path / "t.db"
    _seed_db_sync(db_path, [_row(1, "user", "hi"), _row(2, "assistant", "yo")])

    config = tmp_path / "agent.yaml"
    config.write_text(
        "llm:\n"
        "  main:\n"
        "    base_url: http://127.0.0.1:1/v1\n"
        "    model: nonesuch\n"
        "    max_retries: 0\n"
    )

    rc = main([
        "--db", str(db_path), "--channel", "irc:#s-compact",
        "--range", "1:2", "--config", str(config), "--out", str(tmp_path / "o"),
    ])

    err = capsys.readouterr().err
    assert rc != 0
    assert "127.0.0.1:1" in err


# ---------------------------------------------------------------------------
# R7 — the source database is never written
# ---------------------------------------------------------------------------


async def test_source_db_is_opened_read_only(tmp_path):
    from harness.consolidate_replay import open_source_db

    db_path = tmp_path / "t.db"
    db = await _seed_db(db_path, [_row(1, "user", "hi")])
    await db.close()

    ro = await open_source_db(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            await ro.execute("DELETE FROM message_log")
            await ro.commit()
    finally:
        await ro.close()


# ---------------------------------------------------------------------------
# R3 / R4 — N trials persisted, input recorded beside the output
# ---------------------------------------------------------------------------


async def test_trials_writes_one_record_per_trial(tmp_path):
    from harness.consolidate_replay import run_replay

    db_path = tmp_path / "t.db"
    db = await _seed_db(db_path, [_row(1, "user", "hi"), _row(2, "assistant", "yo")])
    await db.close()

    out = tmp_path / "out"
    client = _FakeClient()
    await run_replay(
        db_path=db_path,
        channel_id="irc:#s-compact",
        after_id=0,
        through_id=2,
        prompt_text="PROMPT",
        trials=3,
        out_dir=out,
        client=client,
    )

    records = [
        json.loads(line)
        for line in (out / "trials.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 3
    assert [r["summary"] for r in records] == ["summary 1", "summary 2", "summary 3"]


async def test_input_transcript_is_written_beside_the_output(tmp_path):
    """R4: scoring needs the input next to the summary, without re-querying."""
    from harness.consolidate_replay import run_replay

    db_path = tmp_path / "t.db"
    db = await _seed_db(db_path, [_row(1, "user", "hi"), _row(2, "assistant", "yo")])
    await db.close()

    out = tmp_path / "out"
    client = _FakeClient()
    await run_replay(
        db_path=db_path,
        channel_id="irc:#s-compact",
        after_id=0,
        through_id=2,
        prompt_text="PROMPT",
        trials=1,
        out_dir=out,
        client=client,
    )

    expected = _dialog_transcript(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )
    assert (out / "input.txt").read_text() == expected
    # and it is exactly what was sent
    assert client.calls[0][1]["content"] == expected
