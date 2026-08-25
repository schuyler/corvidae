"""Tests for harness/session_driver.py's pure protocol/formatting helpers.

Pins the maildir-style inbox/outbox protocol, transcript line format, and
artifact collection described in .claude/plans/shakedown-session-driver.md
so the implementation is checkable without a live IRC connection, ssh, or
sagan. All fixtures are static (fake clocks / precomputed message lists) —
none of these tests waits on real time, matching the project's concurrency
caution about tests that could hang once the implementation lands.
"""
from harness.driver import IRCMessage, ts_fmt


# ---------------------------------------------------------------------------
# Inbox protocol: sequence numbers, ordering, stop priority
# ---------------------------------------------------------------------------


def test_next_seq_starts_at_1_and_increments(tmp_path):
    from harness.session_driver import next_seq

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    assert next_seq(inbox) == 1

    (inbox / "001.txt").write_text("first")
    (inbox / "002.txt").write_text("second")
    (inbox / "STOP").write_text("")
    (inbox / "003.txt.tmp").write_text("partial write, not yet renamed")
    assert next_seq(inbox) == 3


def test_inbox_processed_in_numeric_order_and_stop_wins(tmp_path):
    from harness.session_driver import next_inbox_action

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "002.txt").write_text("second")
    (inbox / "001.txt").write_text("first")

    action = next_inbox_action(inbox)
    assert action.name == "001.txt"

    (inbox / "STOP").write_text("")
    assert next_inbox_action(inbox) == "STOP"


def test_inbox_order_survives_the_move_past_three_digits(tmp_path):
    from harness.session_driver import next_inbox_action

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "999.txt").write_text("nine ninety-nine")
    (inbox / "1000.txt").write_text("one thousand")

    assert next_inbox_action(inbox).name == "999.txt"


# ---------------------------------------------------------------------------
# Turn-record building: latency anchoring, timeout, settle-window grouping
# ---------------------------------------------------------------------------


def test_turn_latency_is_first_bot_reply_after_send():
    from harness.session_driver import build_turn_record

    sent_at = 1000.0
    messages = [
        IRCMessage(sent_at - 10, "buster", "stale line from before this turn"),
        IRCMessage(sent_at + 1, "schuyler", "(human interjection, not the bot)"),
        IRCMessage(sent_at + 2, "buster", "here is my reply"),
    ]
    record = build_turn_record(
        seq=1, text="hi", sent_at=sent_at, messages=messages,
        bot_nick="buster", settle=2.0,
    )
    assert record["timed_out"] is False
    assert record["first_reply_at"] == sent_at + 2
    assert abs(record["latency_s"] - 2.0) < 1e-9


def test_turn_times_out_with_empty_reply():
    from harness.session_driver import build_turn_record

    sent_at = 1000.0
    record = build_turn_record(
        seq=1, text="hello?", sent_at=sent_at, messages=[],
        bot_nick="buster", settle=2.0,
    )
    assert record["timed_out"] is True
    assert record["reply"] == []


def test_reply_collects_multiline_bot_reply_within_settle_window():
    from harness.session_driver import build_turn_record

    sent_at = 1000.0
    messages = [
        IRCMessage(sent_at + 2.0, "buster", "part one"),
        IRCMessage(sent_at + 3.5, "buster", "part two"),  # 1.5s gap: inside settle
        IRCMessage(sent_at + 6.0, "buster", "part three (too late)"),  # 2.5s gap: outside settle
    ]
    record = build_turn_record(
        seq=1, text="hi", sent_at=sent_at, messages=messages,
        bot_nick="buster", settle=2.0,
    )
    assert record["reply"] == ["buster: part one", "buster: part two"]


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------


def test_transcript_lines_use_ts_fmt_and_direction_markers():
    from harness.session_driver import (
        format_latency_line,
        format_received_line,
        format_sent_line,
    )

    ts = 1700000000.482
    assert format_sent_line(ts, "what do you remember about the door code?") == (
        f"{ts_fmt(ts)} >> what do you remember about the door code?"
    )
    assert format_received_line(ts, "buster", "You told me the door code is 8814.") == (
        f"{ts_fmt(ts)} << buster: You told me the door code is 8814."
    )
    # Interjections from any nick, not just the bot, render as << lines too.
    assert format_received_line(ts, "schuyler", "(human interjection, logged but not a turn)") == (
        f"{ts_fmt(ts)} << schuyler: (human interjection, logged but not a turn)"
    )
    assert format_latency_line(ts, 3, 4.425) == f"{ts_fmt(ts)} -- turn 003 latency 4.425s"


# ---------------------------------------------------------------------------
# Nick derivation
# ---------------------------------------------------------------------------


def test_default_nick_derived_from_channel():
    from harness.session_driver import derive_default_nick

    assert derive_default_nick("#chat") == "chat"
    assert derive_default_nick("#s-compact") == "s-compact"
    # ngircd NICKLEN 9: "p-interleave" (12 chars after stripping '#') truncates.
    assert derive_default_nick("#p-interleave") == "p-interle"


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------


def test_collect_copies_artifacts(tmp_path):
    import sqlite3

    from harness.session_driver import collect_artifacts

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    conn = sqlite3.connect(state_dir / "sessions.db")
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()

    (state_dir / "metrics.jsonl").write_text('{"a": 1}\n')
    (state_dir / "corvidae.log").write_text("log line\n")
    (state_dir / "corvidae.log.1").write_text("rotated log\n")

    collect_artifacts(state_dir, session_dir)

    assert (session_dir / "metrics.jsonl").read_text() == '{"a": 1}\n'
    assert (session_dir / "corvidae.log").read_text() == "log line\n"
    assert (session_dir / "corvidae.log.1").read_text() == "rotated log\n"

    copied = sqlite3.connect(session_dir / "sessions.db")
    try:
        assert copied.execute("SELECT id FROM t").fetchall() == [(1,)]
    finally:
        copied.close()


def test_collect_artifacts_no_error_when_rotated_logs_absent(tmp_path):
    import sqlite3

    from harness.session_driver import collect_artifacts

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    conn = sqlite3.connect(state_dir / "sessions.db")
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    (state_dir / "metrics.jsonl").write_text("{}\n")
    (state_dir / "corvidae.log").write_text("log line\n")
    # No corvidae.log.1, .2, etc. — a short session may not have rotated logs.

    collect_artifacts(state_dir, session_dir)  # must not raise

    assert (session_dir / "corvidae.log").read_text() == "log line\n"
    assert not list(session_dir.glob("corvidae.log.*"))
