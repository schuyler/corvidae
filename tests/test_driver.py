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
    after an LLM call returns (send_message/send_progress both fire
    post-completion), so "any reply" still means "a turn completed." If a
    filler turn calls a tool, send_progress could fire before the final
    send_message and drift pacing by one turn — the verdict doesn't depend
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
# are two orthogonal observers of one compact_conversation call; a summary
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
