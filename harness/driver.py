#!/usr/bin/env python3
"""Buster harness probe driver.

Stdlib-only asyncio IRC client (plain socket NICK/USER/JOIN/PRIVMSG/PING
handling) that runs the probe corpus (design.md §5.5) against a live
Buster instance over local IRC, then writes a self-contained run report.

Every probe is objectively checkable: planted exact tokens grepped out of
IRC replies, or mechanical facts read from sessions.db. No LLM-judge
assertions.

Runs on the Buster host, invoked by harness/run.sh after the buster-daemon screen
session is (re)started. See harness/README.md for prerequisites.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import http.server
import json
import random
import re
import sqlite3
import string
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

IRC_MAX_LINE_BYTES = 512  # RFC 1459 §2.3, including the trailing CRLF

SLOW_SERVER_PORT = 8931
# Must stay comfortably inside web_fetch's 15s default per-request timeout
# (corvidae/tools/web.py:20) — at 20s the tool always abandoned the request
# before the response existed, so the token could never arrive. At 13s, the
# ~3s interleaved answer leaves a >3x margin before the fetch is served.
SLOW_SERVER_DELAY_S = 13
SLOW_SERVER_TOKEN = "TOKEN-EELGRASS-71"
MAX_FILLERS = 12

# Ordinary single turn. Observed 2-7s (run 7), worst observed 35s on an
# inflated context. 90s is ~2.5x the worst observation.
TURN_TIMEOUT_S = 90
# A turn plus a tool round trip, including the 13s slow fetch.
TOOL_TURN_TIMEOUT_S = 120
# A turn that may carry a compaction summary call. Observed compaction turn
# stack ~108s in run 7 with an 80s LLM turn.
COMPACTION_TURN_TIMEOUT_S = 150
# Whole filler phase; ~10 paced turns at 3-5s plus one compaction turn.
COMPACTION_PHASE_BUDGET_S = 300


# ---------------------------------------------------------------------------
# Minimal IRC client
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IRCMessage:
    ts: float
    nick: str
    text: str


class IRCRegistrationError(Exception):
    """Server rejected NICK/USER registration (432/433) before we got a nick."""


class IRCClient:
    """Plain-socket IRC client: NICK/USER registration, JOIN, PRIVMSG, PING.

    No dependency on pydle (the transport corvidae itself uses) — the
    driver is a separate, minimal IRC participant so the probe corpus
    exercises the wire protocol independently of the plugin under test.
    """

    def __init__(self, host: str, port: int, nick: str) -> None:
        self.host = host
        self.port = port
        self.nick = nick
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._listen_task: asyncio.Task | None = None
        # Resolved on 001 (success) or 432/433 (rejected) — see connect().
        self._registration: asyncio.Future[None] | None = None
        # channel -> list of received PRIVMSGs
        self.messages: dict[str, list[IRCMessage]] = {}
        # channel -> list of (ts, nick) JOINs observed on that channel
        self.channel_joins: dict[str, list[tuple[float, str]]] = {}

    async def connect(self, timeout: float = 30) -> None:
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        self._registration = asyncio.get_event_loop().create_future()
        self._send(f"NICK {self.nick}")
        self._send(f"USER {self.nick} 0 * :Buster harness driver")
        self._listen_task = asyncio.create_task(self._listen())
        try:
            await asyncio.wait_for(self._registration, timeout=timeout)
        except asyncio.TimeoutError:
            # A bare TimeoutError tells you nothing; say what was being
            # waited for so an abort report is legible without re-deriving
            # it from scratch.
            raise TimeoutError(
                f"IRC registration timed out after {timeout}s waiting for "
                f"nick {self.nick!r} on {self.host}:{self.port}"
            ) from None

    def _send(self, line: str) -> None:
        assert self._writer is not None
        self._writer.write((line + "\r\n").encode("utf-8"))

    async def _listen(self) -> None:
        assert self._reader is not None
        while True:
            raw = await self._reader.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        if line.startswith("PING"):
            token = line.split(" ", 1)[1] if " " in line else ""
            self._send(f"PONG {token}")
            return
        prefix = None
        rest = line
        if line.startswith(":"):
            prefix, rest = line[1:].split(" ", 1)
        parts = rest.split(" ", 2)
        command = parts[0] if parts else ""
        if command == "001":
            if self._registration is not None and not self._registration.done():
                self._registration.set_result(None)
        elif command in ("432", "433"):
            # ERR_ERRONEUSNICKNAME / ERR_NICKNAMEINUSE — registration can
            # never complete with this nick. Fail immediately with the
            # server's own text instead of waiting out the connect timeout.
            if self._registration is not None and not self._registration.done():
                self._registration.set_exception(IRCRegistrationError(f"{command} {rest}"))
        elif command == "JOIN" and prefix:
            nick = prefix.split("!", 1)[0]
            chan = (parts[1] if len(parts) > 1 else "").lstrip(":")
            self.channel_joins.setdefault(chan, []).append((time.time(), nick))
        elif command == "PRIVMSG" and prefix and len(parts) >= 3:
            nick = prefix.split("!", 1)[0]
            target = parts[1]
            text = parts[2][1:] if parts[2].startswith(":") else parts[2]
            self.messages.setdefault(target, []).append(IRCMessage(time.time(), nick, text))

    async def join(self, channel: str) -> None:
        self._send(f"JOIN {channel}")

    async def privmsg(self, channel: str, text: str) -> None:
        prefix_bytes = len(f"PRIVMSG {channel} :".encode("utf-8"))
        max_text_bytes = IRC_MAX_LINE_BYTES - prefix_bytes - len("\r\n")
        for chunk in split_irc_text(text, max_text_bytes):
            self._send(f"PRIVMSG {channel} :{chunk}")

    async def close(self) -> None:
        if self._writer is not None:
            self._send("QUIT :done")
            self._writer.close()
        if self._listen_task is not None:
            self._listen_task.cancel()

    async def wait_for_nick_join(
        self, channel: str, nick: str, timeout: float, after_ts: float = 0.0
    ) -> bool:
        """Poll until `nick` has JOINed `channel` at a time > after_ts, or timeout expires.

        `after_ts` defaults to 0.0 (any historical join counts) for the
        startup check; the restart probe passes the pre-restart wall-clock
        time so a stale JOIN from before the daemon bounced can't satisfy it.
        """
        deadline = time.monotonic() + timeout
        while True:
            if any(n == nick and ts > after_ts for ts, n in self.channel_joins.get(channel, [])):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.5)

    async def wait_until(
        self,
        channel: str,
        predicate: Callable[[list[IRCMessage]], bool],
        timeout: float,
        poll_interval: float = 0.5,
    ) -> bool:
        """Poll received messages on `channel` until predicate is True or timeout."""
        deadline = time.monotonic() + timeout
        while True:
            if predicate(self.messages.get(channel, [])):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Pure helper functions (importable for unit tests without a live IRC/DB)
# ---------------------------------------------------------------------------


def derive_driver_nick(bot_nick: str, max_len: int = 9) -> str:
    """A nick for the driver's own IRC connection, distinct from `bot_nick`
    and within ngircd's default NICKLEN (9) — a literal f"{bot_nick}-driver"
    overflows that for any bot_nick longer than 3 chars (e.g. "buster" ->
    "buster-driver", 13 chars), which ngircd rejects with 432/433.
    """
    suffix = "d"
    return (bot_nick[: max_len - len(suffix)] + suffix)[:max_len]


def split_irc_text(text: str, max_bytes: int) -> list[str]:
    """Split `text` into chunks whose UTF-8 encoding is each at most `max_bytes`,
    without splitting a multi-byte character across chunks.

    Probe text that runs past IRC's 512-byte line limit gets the connection
    killed by the server ("Request too long") instead of delivered, if sent
    unsplit.
    """
    chunks = []
    current: list[str] = []
    current_bytes = 0
    for ch in text:
        ch_bytes = len(ch.encode("utf-8"))
        if current and current_bytes + ch_bytes > max_bytes:
            chunks.append("".join(current))
            current = []
            current_bytes = 0
        current.append(ch)
        current_bytes += ch_bytes
    if current:
        chunks.append("".join(current))
    return chunks


def reply_from(
    messages: list[IRCMessage], nick: str, token: str, after_ts: float = 0.0
) -> IRCMessage | None:
    """First message from `nick` after `after_ts` on the channel containing `token`.

    `after_ts` defaults to 0.0 (any historical message counts); restart_recovery
    passes the pre-restart wall-clock time so a stale reply can't satisfy a
    post-restart wait.
    """
    for msg in messages:
        if msg.nick == nick and msg.ts > after_ts and token in msg.text:
            return msg
    return None


def build_filler_line(index: int) -> str:
    """A single filler line, sized to fit one IRC PRIVMSG (<=460 UTF-8 bytes).

    One driver line is then one daemon `on_message` is then one turn — the
    filler paragraph this replaces produced ~14 IRC lines per call, which is
    what made the compaction probe measure queue depth instead of compaction
    (§M). The body is pseudo-random 4-char alphanumeric groups
    (`random.Random(index)`, reproducible run to run) rather than a repeated
    character, so the corpus is token-dense enough to provably cross the
    compaction trigger within a handful of fillers — a run of one repeated
    character encodes far fewer tokens per byte under BPE.
    """
    header = f"Filler line {index}. "
    trailer = " Acknowledge briefly."
    budget = 460 - len(header.encode("utf-8")) - len(trailer.encode("utf-8"))
    rng = random.Random(index)
    alphabet = string.ascii_letters + string.digits
    n_groups = (budget + 1) // 5  # each group is 4 chars + 1 joining space
    groups = ["".join(rng.choice(alphabet) for _ in range(4)) for _ in range(n_groups)]
    return f"{header}{' '.join(groups)}{trailer}"


def kv_cache_reused(cached_tokens: int, prompt_tokens: int) -> bool:
    """Whether a follow-up turn reused most of the cached prompt prefix.

    llama.cpp never reuses the final few tokens of the prefix, so the
    predicate is a ratio rather than equality: healthy is ~98%, a regression
    is 0%.
    """
    return cached_tokens >= 0.5 * prompt_tokens


def render_report_txt(run_dims: dict, results: list["ProbeResult"]) -> str:
    """Human-readable run report: run-dimension header + one line per probe.

    finished_at/daemon_crashed are unknown to the driver (only run.sh learns
    them after the driver exits) and stored as None — dict.get's default
    only fires on a *missing* key, not a None value, so both fields are
    rendered explicitly against `is None` and run.sh's text-replace patch
    step has a real placeholder to find.
    """
    finished_at = run_dims.get("finished_at")
    daemon_crashed = run_dims.get("daemon_crashed")
    lines = [
        "Buster harness run report",
        f"  started_at:  {run_dims.get('started_at')}",
        f"  finished_at: {'PENDING_FINISHED_AT' if finished_at is None else finished_at}",
        f"  git_rev:     {run_dims.get('git_rev')}",
        f"  llama_endpoint: {run_dims.get('llama_endpoint')}",
        f"  config_rendered_sha256: {run_dims.get('config_rendered_sha256')}",
        f"  daemon_crashed: {'PENDING_DAEMON_CRASHED' if daemon_crashed is None else daemon_crashed}",
        "",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"{status}  {r.name:<20} {r.duration_s:7.1f}s  {r.details}")
    return "\n".join(lines) + "\n"


def ts_fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3] + "Z"


@dataclasses.dataclass
class ProbeResult:
    name: str
    passed: bool
    details: str
    duration_s: float
    transcript: list[str]


# ---------------------------------------------------------------------------
# Local slow HTTP server for the interleaving probe
# ---------------------------------------------------------------------------


class _SlowHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        if self.path == "/slow":
            self.server.request_times.append(time.time())
            time.sleep(SLOW_SERVER_DELAY_S)
            self.server.serve_times.append(time.time())
            body = SLOW_SERVER_TOKEN.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args) -> None:  # silence per-request stderr noise
        pass


def start_slow_server(port: int = SLOW_SERVER_PORT) -> http.server.ThreadingHTTPServer:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), _SlowHandler)
    httpd.request_times = []
    httpd.serve_times = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


# ---------------------------------------------------------------------------
# sessions.db readers (WAL mode: safe to read concurrently with the daemon)
# ---------------------------------------------------------------------------


def _query(db_path: Path, sql: str, params: tuple) -> list[tuple]:
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
            try:
                return conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            last_exc = exc
            time.sleep(1)
    raise RuntimeError(f"sessions.db read failed after retries: {last_exc}")


def count_main_calls_since(db_path: Path, channel_id: str, since_ts: float) -> int:
    rows = _query(
        db_path,
        "SELECT COUNT(*) FROM usage_log WHERE role = ? AND channel_id = ? AND ts >= ?",
        ("main", channel_id, since_ts),
    )
    return rows[0][0] if rows else 0


def has_summary_row_since(db_path: Path, channel_id: str, since_ts: float) -> bool:
    rows = _query(
        db_path,
        "SELECT COUNT(*) FROM message_log WHERE channel_id = ? "
        "AND message_type = 'summary' AND timestamp >= ?",
        (channel_id, since_ts),
    )
    return bool(rows and rows[0][0] > 0)


def usage_rows_since(db_path: Path, channel_id: str, since_ts: float) -> list[tuple[int, int]]:
    """(prompt_tokens, cached_tokens) for main-role calls, oldest first."""
    return _query(
        db_path,
        "SELECT prompt_tokens, cached_tokens FROM usage_log "
        "WHERE role = 'main' AND channel_id = ? AND ts >= ? ORDER BY ts",
        (channel_id, since_ts),
    )


def compaction_calls_since(db_path: Path, channel_id: str, since_ts: float) -> list[float]:
    return [
        r[0]
        for r in _query(
            db_path,
            "SELECT latency_ms FROM usage_log WHERE stage = 'compaction' "
            "AND channel_id = ? AND ts >= ? ORDER BY ts",
            (channel_id, since_ts),
        )
    ]


def tool_messages_since(db_path: Path, channel_id: str, since_ts: float) -> int:
    return _query(
        db_path,
        "SELECT COUNT(*) FROM message_log WHERE channel_id = ? AND timestamp >= ? "
        "AND json_extract(message, '$.role') = 'tool'",
        (channel_id, since_ts),
    )[0][0]


# ---------------------------------------------------------------------------
# Run context and probes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunContext:
    client: IRCClient
    bot_nick: str
    state_dir: Path
    run_dir: Path
    repo_dir: Path
    session_db: Path
    channel_ready: dict[str, bool]
    httpd: http.server.ThreadingHTTPServer


def _transcript(ctx: RunContext, channel: str, sent: list[tuple[float, str]]) -> list[str]:
    items: list[tuple[float, str, str]] = [(ts, ">>", text) for ts, text in sent]
    for msg in ctx.client.messages.get(channel, []):
        items.append((msg.ts, "<<", f"{msg.nick}: {msg.text}"))
    items.sort(key=lambda x: x[0])
    return [f"{ts_fmt(ts)} {d} {t}" for ts, d, t in items]


async def _send(ctx: RunContext, channel: str, text: str, sent_log: list[tuple[float, str]]) -> None:
    await ctx.client.privmsg(channel, text)
    sent_log.append((time.time(), text))


async def send_and_await(
    ctx: RunContext,
    channel: str,
    text: str,
    token: str,
    sent: list[tuple[float, str]],
    timeout: float,
    after_ts: float = 0.0,
) -> IRCMessage | None:
    """Send one line; block until the bot replies with `token`. None on timeout.

    The harness introduces concurrency only where concurrency is the thing
    under test — every probe but `probe_interleave` sends then waits, so this
    is the shared shape.
    """
    await _send(ctx, channel, text, sent)
    if not await ctx.client.wait_until(
        channel,
        lambda msgs: reply_from(msgs, ctx.bot_nick, token, after_ts) is not None,
        timeout,
    ):
        return None
    return reply_from(ctx.client.messages.get(channel, []), ctx.bot_nick, token, after_ts)


async def send_and_await_any_reply(
    ctx: RunContext,
    channel: str,
    text: str,
    sent: list[tuple[float, str]],
    timeout: float,
) -> bool:
    """Send one line; block until the bot replies with anything. False on timeout.

    Unlike `send_and_await`, the reply's content is not load-bearing — only
    its existence, timestamped after the send. Paces `probe_compaction`
    against a model that answers a filler with an off-by-one counted ack
    (run 8: ACK-15 for filler 14), which a token match would wait on forever.
    """
    # The empty token matches any reply; after_ts is the only thing separating
    # this send's reply from stale lines and progress emissions.
    after_ts = time.time()
    return await send_and_await(ctx, channel, text, "", sent, timeout, after_ts=after_ts) is not None


def _skip_result(name: str, channel: str) -> ProbeResult:
    return ProbeResult(
        name=name,
        passed=False,
        details=f"buster never joined {channel} — probe skipped",
        duration_s=0.0,
        transcript=[],
    )


async def probe_smoke_echo(ctx: RunContext) -> ProbeResult:
    channel = "#p-smoke"
    if not ctx.channel_ready.get(channel):
        return _skip_result("smoke_echo", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()
    matched = await send_and_await(
        ctx, channel, "The anchovy code is 7291. Reply with that code.", "7291", sent, TURN_TIMEOUT_S,
    )
    return ProbeResult(
        "smoke_echo",
        matched is not None,
        "reply containing 7291 observed" if matched else "no reply containing 7291 within timeout",
        time.monotonic() - t0,
        _transcript(ctx, channel, sent),
    )


async def probe_tool_loop(ctx: RunContext) -> ProbeResult:
    channel = "#p-toolloop"
    if not ctx.channel_ready.get(channel):
        return _skip_result("tool_loop", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()
    ts0 = time.time()

    probe_data = ctx.state_dir / "probe_data"
    probe_data.mkdir(parents=True, exist_ok=True)
    a_path = probe_data / "a.txt"
    b_path = probe_data / "b.txt"
    a_path.write_text("first half is QRT-; second half is in b.txt\n")
    b_path.write_text("9083\n")

    matched = await send_and_await(
        ctx, channel,
        f"Use your read_file tool to read {a_path} and follow its instructions "
        "to determine a full code (you may need to read a second file it "
        "references). Reply with just the code.",
        "QRT-9083", sent, TURN_TIMEOUT_S,
    )
    duration = time.monotonic() - t0
    if matched is None:
        return ProbeResult(
            "tool_loop", False, "no reply containing QRT-9083 within timeout", duration,
            _transcript(ctx, channel, sent),
        )
    call_count = count_main_calls_since(ctx.session_db, f"irc:{channel}", ts0)
    passed = call_count >= 3
    details = f"reply contained QRT-9083; usage_log main-role calls since probe start: {call_count}"
    return ProbeResult("tool_loop", passed, details, duration, _transcript(ctx, channel, sent))


async def probe_interleave(ctx: RunContext) -> ProbeResult:
    channel = "#p-interleave"
    if not ctx.channel_ready.get(channel):
        return _skip_result("interleave", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()
    ts0 = time.time()

    # Deliberate departure from the send-then-await principle every other
    # probe follows: sending M2 while M1's tool call is still in flight is
    # this probe's subject, so it calls _send directly rather than
    # send_and_await.
    await _send(
        ctx, channel,
        f"Fetch http://127.0.0.1:{SLOW_SERVER_PORT}/slow with web_fetch and tell me the token.",
        sent,
    )

    deadline = time.monotonic() + TURN_TIMEOUT_S
    while len(ctx.httpd.request_times) < 1:
        if time.monotonic() >= deadline:
            return ProbeResult(
                "interleave", False, "M1 never triggered web_fetch", time.monotonic() - t0,
                _transcript(ctx, channel, sent),
            )
        await asyncio.sleep(0.5)

    await _send(ctx, channel, "What is 17+25? Reply with just the number.", sent)

    forty_two_re = re.compile(r"\b42\b")
    eelgrass_re = re.compile(re.escape(SLOW_SERVER_TOKEN))

    def find_reply(token_regex: re.Pattern[str]) -> IRCMessage | None:
        for msg in ctx.client.messages.get(channel, []):
            if msg.nick == ctx.bot_nick and token_regex.search(msg.text):
                return msg
        return None

    matched_42 = await ctx.client.wait_until(
        channel, lambda _msgs: find_reply(forty_two_re) is not None, TURN_TIMEOUT_S,
    )
    if not matched_42:
        return ProbeResult(
            "interleave", False, "no reply containing 42 within timeout", time.monotonic() - t0,
            _transcript(ctx, channel, sent),
        )

    matched_token = await ctx.client.wait_until(
        channel, lambda _msgs: find_reply(eelgrass_re) is not None, TOOL_TURN_TIMEOUT_S,
    )
    duration = time.monotonic() - t0
    if not matched_token:
        return ProbeResult(
            "interleave", False, f"no reply containing {SLOW_SERVER_TOKEN} within timeout", duration,
            _transcript(ctx, channel, sent),
        )

    forty_two = find_reply(forty_two_re)
    assert forty_two is not None
    serve_ts = ctx.httpd.serve_times[0]
    passed = forty_two.ts < serve_ts
    tool_msgs = tool_messages_since(ctx.session_db, f"irc:{channel}", ts0)
    details = (
        f"42-reply at {ts_fmt(forty_two.ts)}, slow server served at {ts_fmt(serve_ts)}; "
        f"interleaved={passed}; tool messages since probe start: {tool_msgs}"
    )
    return ProbeResult("interleave", passed, details, duration, _transcript(ctx, channel, sent))


async def probe_compaction(ctx: RunContext) -> ProbeResult:
    channel = "#p-compact"
    if not ctx.channel_ready.get(channel):
        return _skip_result("compaction", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()
    ts0 = time.time()

    fillers_sent = 0
    summarized = False
    for i in range(1, MAX_FILLERS + 1):
        line = build_filler_line(i)
        if not await send_and_await_any_reply(ctx, channel, line, sent, COMPACTION_TURN_TIMEOUT_S):
            return ProbeResult(
                "compaction", False, f"no reply to filler {i} within {COMPACTION_TURN_TIMEOUT_S}s",
                time.monotonic() - t0, _transcript(ctx, channel, sent),
            )
        fillers_sent = i
        if has_summary_row_since(ctx.session_db, f"irc:{channel}", ts0) and compaction_calls_since(
            ctx.session_db, f"irc:{channel}", ts0
        ):
            summarized = True
            break
        if time.monotonic() - t0 > COMPACTION_PHASE_BUDGET_S:
            break

    rows = usage_rows_since(ctx.session_db, f"irc:{channel}", ts0)
    max_prompt_tokens = max((r[0] for r in rows), default=0)
    calls = compaction_calls_since(ctx.session_db, f"irc:{channel}", ts0)
    longest = max(calls, default=0)
    if not summarized:
        return ProbeResult(
            "compaction", False,
            f"compaction did not fire after {fillers_sent} paced fillers "
            f"(max prompt_tokens {max_prompt_tokens}, {len(calls)} compaction-stage calls)",
            time.monotonic() - t0, _transcript(ctx, channel, sent),
        )

    checkpoint = await send_and_await(
        ctx, channel, "Reply with the word CHECKPOINT-OK.", "CHECKPOINT-OK", sent,
        COMPACTION_TURN_TIMEOUT_S,
    )
    duration = time.monotonic() - t0
    if checkpoint is None:
        return ProbeResult(
            "compaction", False,
            f"compaction fired ({len(calls)} compaction-stage LLM calls, longest {longest:.0f} ms) "
            f"but no CHECKPOINT-OK within {COMPACTION_TURN_TIMEOUT_S}s",
            duration, _transcript(ctx, channel, sent),
        )
    details = (
        f"compaction fired after {fillers_sent} fillers ({len(calls)} compaction-stage calls, "
        f"longest {longest:.0f} ms); CHECKPOINT-OK observed"
    )
    return ProbeResult("compaction", True, details, duration, _transcript(ctx, channel, sent))


async def probe_restart_recovery(ctx: RunContext) -> ProbeResult:
    channel = "#p-restart"
    if not ctx.channel_ready.get(channel):
        return _skip_result("restart_recovery", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()

    acked = await send_and_await(
        ctx, channel, "My door code is 8814. Acknowledge with the word ACKNOWLEDGED.",
        "ACKNOWLEDGED", sent, TURN_TIMEOUT_S,
    )
    if acked is None:
        return ProbeResult(
            "restart_recovery", False, "no acknowledgement reply before restart", time.monotonic() - t0,
            _transcript(ctx, channel, sent),
        )

    restart_script = ctx.repo_dir / "harness" / "restart-daemon.sh"
    daemon_log = ctx.run_dir / "daemon.log"
    restart_ts = time.time()
    subprocess.run([str(restart_script), str(daemon_log)], cwd=ctx.repo_dir, check=True, timeout=120)

    rejoined = await ctx.client.wait_for_nick_join(
        channel, ctx.bot_nick, timeout=90, after_ts=restart_ts
    )
    if not rejoined:
        return ProbeResult(
            "restart_recovery", False, "buster did not rejoin channel after restart",
            time.monotonic() - t0, _transcript(ctx, channel, sent),
        )
    # only replies after the restart are evidence of recovery, not the pre-restart ack
    matched = await send_and_await(
        ctx, channel, "What door code did I give you earlier in this conversation?",
        "8814", sent, TURN_TIMEOUT_S, after_ts=restart_ts,
    )
    duration = time.monotonic() - t0
    details = (
        "post-restart reply contained 8814 (history reloaded)" if matched
        else "post-restart reply did not contain 8814 within timeout"
    )
    return ProbeResult("restart_recovery", matched is not None, details, duration, _transcript(ctx, channel, sent))


async def probe_kv_slot(ctx: RunContext) -> ProbeResult:
    channel = "#p-slots"
    if not ctx.channel_ready.get(channel):
        return _skip_result("kv_slot", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()
    ts0 = time.time()
    token = "SLOTPROBE-3319"

    r1 = await send_and_await(
        ctx, channel, f"Remember this token: {token}. Reply with the word RECEIVED.",
        "RECEIVED", sent, TURN_TIMEOUT_S,
    )
    if r1 is None:
        return ProbeResult(
            "kv_slot", False, "no RECEIVED reply within timeout", time.monotonic() - t0,
            _transcript(ctx, channel, sent),
        )
    # Nothing may be sent between the two turns: --parallel 1 means one KV
    # slot, and an interposed call on another channel would evict the prefix.
    r2 = await send_and_await(ctx, channel, "Reply with the word OK.", "OK", sent, TURN_TIMEOUT_S)
    duration = time.monotonic() - t0
    if r2 is None:
        return ProbeResult(
            "kv_slot", False, "no OK reply within timeout", duration,
            _transcript(ctx, channel, sent),
        )

    rows = usage_rows_since(ctx.session_db, f"irc:{channel}", ts0)
    if len(rows) < 2:
        return ProbeResult(
            "kv_slot", False, f"expected at least 2 usage_log rows, got {len(rows)}", duration,
            _transcript(ctx, channel, sent),
        )
    prompt_tokens, cached_tokens = rows[1]
    passed = cached_tokens is not None and kv_cache_reused(cached_tokens, prompt_tokens)
    details = (
        f"turn1: prompt_tokens={rows[0][0]}, cached_tokens={rows[0][1]}; "
        f"turn2: prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}"
    )
    return ProbeResult("kv_slot", passed, details, duration, _transcript(ctx, channel, sent))


PROBES: list[Callable[[RunContext], "asyncio.Future[ProbeResult]"]] = [
    probe_smoke_echo,
    probe_tool_loop,
    probe_interleave,
    probe_compaction,
    probe_restart_recovery,
    probe_kv_slot,
]

ALL_CHANNELS = ["#p-smoke", "#p-toolloop", "#p-interleave", "#p-compact", "#p-restart", "#p-slots"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--state-dir", required=True, type=Path)
    p.add_argument("--repo-dir", required=True, type=Path)
    p.add_argument("--irc-host", default="127.0.0.1")
    p.add_argument("--irc-port", type=int, default=6667)
    p.add_argument("--bot-nick", default="buster")
    p.add_argument("--llama-url", default="http://127.0.0.1:8080")
    p.add_argument("--join-timeout", type=float, default=180)
    p.add_argument("--started-at", required=True)
    p.add_argument("--git-rev", required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--server-props-file", required=True, type=Path)
    p.add_argument("--server-models-file", required=True, type=Path)
    return p.parse_args(argv)


async def async_main(args: argparse.Namespace, run_dims: dict) -> int:
    # Daemon thread; intentionally never shut down — it lives for the
    # process, and the process exits when the driver is done.
    slow_httpd = start_slow_server()

    # server_props/server_models are recorded up front, inside the crash
    # guard in main() — a bad file here is as much a "driver crashed"
    # condition as an IRC connect failure.
    run_dims["server_props"] = json.loads(args.server_props_file.read_text())
    run_dims["server_models"] = json.loads(args.server_models_file.read_text())

    driver_nick = derive_driver_nick(args.bot_nick)
    client = IRCClient(args.irc_host, args.irc_port, driver_nick)
    await client.connect()
    for channel in ALL_CHANNELS:
        await client.join(channel)

    channel_ready = {
        channel: await client.wait_for_nick_join(channel, args.bot_nick, args.join_timeout)
        for channel in ALL_CHANNELS
    }

    ctx = RunContext(
        client=client,
        bot_nick=args.bot_nick,
        state_dir=args.state_dir,
        run_dir=args.run_dir,
        repo_dir=args.repo_dir,
        session_db=args.state_dir / "sessions.db",
        channel_ready=channel_ready,
        httpd=slow_httpd,
    )

    results: list[ProbeResult] = []
    for probe in PROBES:
        # A probe raising (e.g. restart_recovery's subprocess.run of
        # restart-daemon.sh hitting CalledProcessError/TimeoutExpired) must
        # not take down the whole run — record it as a failed probe and
        # keep going, so the report is always written (§5.6: no silent
        # empty run directories).
        try:
            results.append(await probe(ctx))
        except Exception as exc:
            name = probe.__name__[len("probe_"):] if probe.__name__.startswith("probe_") else probe.__name__
            results.append(
                ProbeResult(name, False, f"probe raised an unhandled exception: {exc!r}", 0.0, [])
            )

    await client.close()
    _write_report(args.run_dir, run_dims, results)
    return 0 if all(r.passed for r in results) else 1


def _base_run_dims(args: argparse.Namespace) -> dict:
    """Run-dimension fields the driver can fill in on its own.

    finished_at/daemon_crashed stay None — only run.sh knows them, after
    the driver has exited and it has checked the daemon's screen session.
    """
    return {
        "started_at": args.started_at,
        "finished_at": None,
        "git_rev": args.git_rev,
        "llama_endpoint": args.llama_url,
        "config_rendered_sha256": args.config_sha256,
        "daemon_crashed": None,
    }


def _write_report(run_dir: Path, run_dims: dict, results: list[ProbeResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run": run_dims,
        "probes": [
            {
                "name": r.name,
                "passed": r.passed,
                "duration_s": r.duration_s,
                "details": r.details,
                "transcript": r.transcript,
            }
            for r in results
        ],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "report.txt").write_text(render_report_txt(run_dims, results))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dims = _base_run_dims(args)
    try:
        return asyncio.run(async_main(args, run_dims))
    except Exception as exc:
        # Anything that escapes async_main's own per-probe guard (IRC
        # connect failure, a bug in the setup/report-writing code itself,
        # ...) still gets a legible aborted report instead of a bare
        # traceback and an empty run directory.
        run_dims["aborted"] = True
        run_dims["abort_reason"] = f"driver crashed: {exc!r}"
        _write_report(args.run_dir, run_dims, [])
        return 1


if __name__ == "__main__":
    sys.exit(main())
