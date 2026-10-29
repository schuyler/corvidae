"""Tests for harness/driver.py's pure helpers.

split_irc_text guards against RFC 1459's 512-byte line limit — probe text
that gets sent unsplit is silently dropped: ngircd kills the connection for
"Request too long" instead of delivering it, which cascades into every probe
sent after it failing too.
"""
import time
import types
from pathlib import Path

from harness.driver import split_irc_text


def test_text_under_limit_is_one_chunk():
    text = "short message"
    assert split_irc_text(text, max_bytes=100) == [text]


def test_text_over_limit_splits_into_multiple_chunks():
    text = "a" * 1000
    chunks = split_irc_text(text, max_bytes=100)
    assert len(chunks) > 1
    assert all(len(c.encode("utf-8")) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_multibyte_character_never_split_across_chunks():
    # "é" is 2 bytes in UTF-8 — a byte-oblivious splitter at an odd boundary
    # would sever it and corrupt the text on decode.
    text = "é" * 300
    chunks = split_irc_text(text, max_bytes=101)
    assert all(len(c.encode("utf-8")) <= 101 for c in chunks)
    assert all(c.encode("utf-8").decode("utf-8") == c for c in chunks)
    assert "".join(chunks) == text


# ---------------------------------------------------------------------------
# R7: build_filler_line — one filler paced to exactly one IRC line, so one
# driver "message" is one daemon turn: the compaction probe relies on that
# 1:1 mapping to measure compaction rather than queue depth.
# ---------------------------------------------------------------------------


def test_build_filler_line_fits_one_irc_line():
    from harness.driver import build_filler_line

    line = build_filler_line(3)
    assert len(line.encode("utf-8")) <= 460


def test_build_filler_line_takes_no_ack_token():
    """B: pacing no longer depends on a reply containing a specific token, so
    build_filler_line no longer takes one. Calling it with only an index
    should just work.
    """
    from harness.driver import build_filler_line

    line = build_filler_line(3)
    assert isinstance(line, str)


# ---------------------------------------------------------------------------
# B: compaction probe pacing — advance on any reply, not an exactly-numbered
# ACK-N. Run 8's daemon replied ACK-15 to filler 14; the old send_and_await
# waited for the literal token ACK-14 and hung, and the probe reported "daemon
# stopped replying" — a counting slip read as a liveness failure.
# ---------------------------------------------------------------------------


class _FakeIRCClient:
    """Minimal stand-in for IRCClient: no sockets, no polling delay.

    `privmsg` deposits a reply immediately so `wait_until`'s predicate is
    satisfiable on its very first check.
    """

    def __init__(self, bot_nick: str, reply_text: str = "ack") -> None:
        self.bot_nick = bot_nick
        self.reply_text = reply_text
        self.messages: dict[str, list] = {}

    async def privmsg(self, channel: str, text: str) -> None:
        from harness.driver import IRCMessage

        self.messages.setdefault(channel, []).append(
            IRCMessage(time.time(), self.bot_nick, self.reply_text)
        )

    async def wait_until(self, channel, predicate, timeout, poll_interval=0.5):
        return predicate(self.messages.get(channel, []))


async def test_send_and_await_any_reply_advances_on_mismatched_reply_content():
    """The replacement helper must advance on any reply after the send,
    regardless of what it contains — reproducing run 8's ACK-15-for-filler-14
    skew, where the reply's content no longer matches what a numbered-ack
    scheme expects but the turn still completed.

    Caveat (design review): this assumes Buster's PRIVMSGs are only emitted
    after an LLM call returns (OutboundMessage/OutboundProgress both fire
    post-completion), so "any reply" still means "a turn completed." If a
    filler turn calls a tool, OutboundProgress could fire before the final
    OutboundMessage and drift pacing by one turn — the verdict doesn't depend
    on counting, so correctness holds regardless.
    """
    from harness import driver

    client = _FakeIRCClient("buster", reply_text="ACK-99")  # deliberately mismatched
    ctx = driver.RunContext(
        client=client, bot_nick="buster", state_dir=None, run_dir=None,
        repo_dir=None, session_db=None, channel_ready={}, httpd=None,
    )
    sent: list[tuple[float, str]] = []
    ok = await driver.send_and_await_any_reply(ctx, "#p-compact", "filler text", sent, timeout=1)
    assert ok is True


# ---------------------------------------------------------------------------
# C: filler-corpus determinism — the token-dense filler must provably cross
# the compaction trigger, pinned against the real config and the real token
# counter so a future filler change that silently stops crossing the
# threshold fails here instead of on the harness host.
# ---------------------------------------------------------------------------


def test_filler_corpus_crosses_the_compact_threshold_with_margin():
    import json

    import yaml

    from corvidae.context import count_tokens
    from harness.driver import MAX_FILLERS, build_filler_line

    config_path = Path(__file__).resolve().parent.parent / "harness" / "buster.yaml.in"
    config = yaml.safe_load(config_path.read_text())
    budget = config["channels"]["irc:#p-compact"]["max_context_tokens"]
    trigger = 0.8 * budget

    cumulative = 0
    k = None
    for i in range(1, MAX_FILLERS + 1):
        # Sized the way production measures a message: the JSON-serialized
        # {"role": ..., "content": ...} dict, not raw content alone -- a
        # bare content-length sum understates what actually gets sent.
        line = json.dumps({"role": "user", "content": build_filler_line(i)})
        cumulative += count_tokens(line)
        if cumulative >= trigger:
            k = i
            break

    assert k is not None, f"never reached {trigger} cumulative tokens within {MAX_FILLERS} fillers"
    assert 2 * k <= MAX_FILLERS, f"only {2 * k} <= {MAX_FILLERS} margin at k={k}"


# ---------------------------------------------------------------------------
# C: compaction verdict — the probe must not be able to pass when compaction
# did not mechanically fire. has_summary_row_since and compaction_calls_since
# are two orthogonal observers of one resolve_compaction call; a summary
# row with no compaction-stage usage_log row means the persistence hook wrote
# without a real call behind it, and that must not pass.
# ---------------------------------------------------------------------------


async def test_probe_compaction_verdict_requires_a_compaction_call_row(monkeypatch):
    from harness import driver

    monkeypatch.setattr(driver, "has_summary_row_since", lambda *a, **k: True)
    monkeypatch.setattr(driver, "compaction_calls_since", lambda *a, **k: [])
    monkeypatch.setattr(driver, "usage_rows_since", lambda *a, **k: [])

    async def fake_send_and_await(ctx, channel, text, token, sent, timeout, after_ts=0.0):
        return driver.IRCMessage(time.time(), ctx.bot_nick, token)

    monkeypatch.setattr(driver, "send_and_await", fake_send_and_await)

    ctx = driver.RunContext(
        client=types.SimpleNamespace(messages={}), bot_nick="buster", state_dir=None,
        run_dir=None, repo_dir=None, session_db=None,
        channel_ready={"#p-compact": True}, httpd=None,
    )
    result = await driver.probe_compaction(ctx)
    assert result.passed is False


# ---------------------------------------------------------------------------
# R6: kv-cache reuse pass predicate, extracted as a pure function per the
# file's "pure helpers importable without a live IRC/DB" convention so the
# probe's pass criterion is unit-testable without a live server.
# ---------------------------------------------------------------------------


def test_kv_cache_reused_false_when_nothing_was_cached():
    from harness.driver import kv_cache_reused

    assert kv_cache_reused(cached_tokens=0, prompt_tokens=820) is False


def test_kv_cache_reused_true_when_most_of_the_prefix_was_reused():
    from harness.driver import kv_cache_reused

    assert kv_cache_reused(cached_tokens=803, prompt_tokens=820) is True


# ---------------------------------------------------------------------------
# withheld_restart (buster-gate profile, #131): the sessions.db readers and
# the verdict
# ---------------------------------------------------------------------------


def _withheld_db(tmp_path, rows, exchanges=()):
    import json
    import sqlite3

    path = tmp_path / "sessions.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE message_log (id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT, "
        "message TEXT, timestamp REAL, message_type TEXT)"
    )
    db.execute("CREATE TABLE exchange_log (trace_id TEXT, channel_id TEXT, origin TEXT, outcomes TEXT)")
    for channel_id, msg, mtype in rows:
        db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, 0, ?)",
            (channel_id, json.dumps(msg), mtype),
        )
    for channel_id, origin, outcomes in exchanges:
        db.execute(
            "INSERT INTO exchange_log VALUES ('t', ?, ?, ?)",
            (channel_id, origin, json.dumps(outcomes)),
        )
    db.commit()
    db.close()
    return path


def test_withheld_rowids_and_marker(tmp_path):
    from harness.driver import has_final_marker_after, withheld_rowids

    ch = "irc:#p-withheld"
    path = _withheld_db(tmp_path, [
        (ch, {"role": "system", "content": "[scheduler]\n\nbeat"}, "message"),
        (ch, {"role": "assistant", "content": "PELICAN-7731"}, "unsent"),
        (ch, {"role": "system", "content": "[not sent: final]\n\nYour response above was not sent"}, "context"),
        ("irc:#other", {"role": "assistant", "content": "x"}, "unsent"),
    ])
    assert withheld_rowids(path, ch) == [2]
    assert has_final_marker_after(path, ch, 2) is True
    assert has_final_marker_after(path, ch, 1) is False


def test_vetoed_heartbeat_count(tmp_path):
    from harness.driver import vetoed_heartbeat_count

    ch = "irc:#p-withheld"
    path = _withheld_db(tmp_path, [], exchanges=[
        (ch, "heartbeat", {"send": {"vetoed": True}}),
        (ch, "heartbeat", {"send": {"vetoed": False}}),
        (ch, "user", {"send": {"vetoed": True}}),
    ])
    assert vetoed_heartbeat_count(path, ch) == 1


def test_withheld_restart_verdict_names_first_failing_stage():
    from harness.driver import withheld_restart_verdict

    assert withheld_restart_verdict(True, True, True, True, True, True)[0] is True
    passed, why = withheld_restart_verdict(True, True, True, False, True, True)
    assert passed is False and "sent a withheld reply" in why
    passed, why = withheld_restart_verdict(True, True, True, True, True, False)
    assert passed is False and "did not reload" in why


def test_buster_gate_profile_is_registered():
    from harness.driver import PROFILES, WITHHELD_CHANNEL, probe_withheld_restart

    probes, channels = PROFILES["buster-gate"]
    assert probes == [probe_withheld_restart]
    assert channels == [WITHHELD_CHANNEL]
    template = Path(__file__).resolve().parent.parent / "harness" / "buster-gate.yaml.in"
    assert "min_salience: 1.0" in template.read_text()


# ---------------------------------------------------------------------------
# buster-quiet profile (R10, plans/kernel-clock.md §7): the quiet-period
# probe's verdict, its sessions.db readers and its registration
# ---------------------------------------------------------------------------


def _quiet_db(tmp_path):
    """A sessions.db with the memory, watermark, retention and message_log
    tables the quiet-period readers query (columns as the real DDL has them
    for the ones the probe reads)."""
    import sqlite3

    path = tmp_path / "sessions.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE message_log (id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT, "
        "message TEXT, timestamp REAL, message_type TEXT)"
    )
    db.execute(
        "CREATE TABLE memory (id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT NOT NULL, "
        "created_at REAL NOT NULL, summary TEXT NOT NULL, importance REAL NOT NULL, "
        "msg_id_start INTEGER NOT NULL, msg_id_end INTEGER NOT NULL)"
    )
    db.execute(
        "CREATE TABLE consolidation_watermark (channel_id TEXT PRIMARY KEY, "
        "last_message_id INTEGER NOT NULL)"
    )
    db.execute("CREATE TABLE retention_meta (last_run REAL NOT NULL DEFAULT 0)")
    db.commit()
    return path, db


def test_quiet_period_verdict_names_first_failing_stage():
    """R10, R8, R2: pass only when every stage held; the first failing
    stage in order (consolidated_a, retention_ran, restart_timing_ok,
    consolidated_b, stray_turn_rows) is the one named."""
    from harness.driver import quiet_period_verdict

    assert quiet_period_verdict(True, True, True, True, 0)[0] is True

    # Each single failure fails and gets its own message.
    failures = [
        quiet_period_verdict(False, True, True, True, 0),
        quiet_period_verdict(True, False, True, True, 0),
        quiet_period_verdict(True, True, False, True, 0),
        quiet_period_verdict(True, True, True, False, 0),
        quiet_period_verdict(True, True, True, True, 2),
    ]
    assert all(passed is False for passed, _ in failures)
    reasons = [why for _, why in failures]
    assert len(set(reasons)) == len(reasons)

    # With several stages failing, the earliest is named.
    assert quiet_period_verdict(False, False, False, False, 5)[1] == reasons[0]
    assert quiet_period_verdict(True, False, False, False, 5)[1] == reasons[1]
    assert quiet_period_verdict(True, True, False, False, 5)[1] == reasons[2]
    assert quiet_period_verdict(True, True, True, False, 5)[1] == reasons[3]


def test_quiet_db_helpers_read_real_rows(tmp_path):
    """R10, R1, R4, R8: the four quiet-period readers return what is in
    sessions.db: the channel's watermark, whether a memory row covers a
    message id, retention's last run, and user/assistant rows in an open
    time window."""
    import json

    from harness.driver import (
        consolidation_watermark,
        memory_covering,
        retention_last_run,
        turn_rows,
    )

    ch = "irc:#p-quiet"
    path, db = _quiet_db(tmp_path)

    # Empty database: no watermark reads as 0, no covering row, no retention run.
    db.commit()
    assert consolidation_watermark(path, ch) == 0
    assert memory_covering(path, ch, 5) is False
    assert retention_last_run(path) is None
    assert turn_rows(path, ch, 0, 1000) == 0

    db.execute("INSERT INTO consolidation_watermark VALUES (?, 7)", (ch,))
    db.execute("INSERT INTO consolidation_watermark VALUES ('irc:#other', 99)")
    db.execute("INSERT INTO memory (channel_id, created_at, summary, importance, msg_id_start, msg_id_end) "
               "VALUES (?, 1, 's', 0.5, 3, 7)", (ch,))
    db.execute("INSERT INTO memory (channel_id, created_at, summary, importance, msg_id_start, msg_id_end) "
               "VALUES ('irc:#other', 1, 's', 0.5, 1, 99)")
    db.execute("INSERT INTO retention_meta VALUES (1234.5)")
    for ts, channel, role in [
        (100, ch, "user"),            # on the lower bound: excluded
        (150, ch, "user"),            # counted
        (160, ch, "assistant"),       # counted
        (170, ch, "system"),          # not a turn row
        (180, ch, "tool"),            # not a turn row
        (190, "irc:#other", "user"),  # other channel
        (200, ch, "user"),            # on the upper bound: excluded
    ]:
        db.execute(
            "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, 'message')",
            (channel, json.dumps({"role": role, "content": "x"}), ts),
        )
    db.commit()
    db.close()

    assert consolidation_watermark(path, ch) == 7
    assert memory_covering(path, ch, 7) is True
    assert memory_covering(path, ch, 8) is False
    assert retention_last_run(path) == 1234.5
    assert turn_rows(path, ch, 100, 200) == 2
    assert turn_rows(path, ch, 160, 200) == 0


def test_buster_quiet_profile_is_registered():
    """R10: the profile maps to probe_quiet_period on #p-quiet, and its
    template sets the short periods, disables the scheduler (R4 without
    it), and has no embedding role."""
    import yaml

    from harness.driver import PROFILES, QUIET_CHANNEL, probe_quiet_period

    probes, channels = PROFILES["buster-quiet"]
    assert probes == [probe_quiet_period]
    assert channels == [QUIET_CHANNEL] == ["#p-quiet"]

    harness = Path(__file__).resolve().parent.parent / "harness"
    cfg = yaml.safe_load((harness / "buster-quiet.yaml.in").read_text().replace("@STATE_DIR@", "/state"))
    memrole = yaml.safe_load((harness / "buster-memrole.yaml.in").read_text().replace("@STATE_DIR@", "/state"))
    assert cfg["memory"]["idle_consolidate_after"] == 45
    assert cfg["memory"]["retention"]["interval"] == 60
    assert cfg["memory"]["retention"]["max_delay"] == 30
    assert QUIET_CHANNEL in cfg["irc"]["channels"]
    assert "embedding" not in cfg["llm"]
    assert set(cfg["plugins"]["disabled"]) == set(memrole["plugins"]["disabled"]) | {"scheduler"}
    assert "mid_conversation_system_role" not in cfg["agent"]
