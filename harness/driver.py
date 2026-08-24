#!/usr/bin/env python3
"""Buster harness probe driver.

Stdlib-only asyncio IRC client (plain socket NICK/USER/JOIN/PRIVMSG/PING
handling) that runs the probe corpus (design.md §5.5) against a live
Buster instance over local IRC, then writes a self-contained run report.

Every probe is objectively checkable: planted exact tokens grepped out of
IRC replies, or mechanical facts read from sessions.db / the llama-server
/slots endpoint. No LLM-judge assertions.

Runs on sagan, invoked by harness/run.sh after the buster-daemon screen
session is (re)started. See harness/README.md for prerequisites.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import http.server
import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

IRC_MAX_LINE_BYTES = 512  # RFC 1459 §2.3, including the trailing CRLF

SLOW_SERVER_PORT = 8931
# Must stay comfortably inside web_fetch's 15s default per-request timeout
# (corvidae/tools/web.py:20) — at 20s the tool always abandoned the request
# before the response existed, so the token could never arrive.
SLOW_SERVER_DELAY_S = 10
SLOW_SERVER_TOKEN = "TOKEN-EELGRASS-71"


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

    Probe text (e.g. build_filler_paragraph's output) can run well past IRC's
    512-byte line limit; sending it unsplit gets the connection killed by the
    server ("Request too long") instead of delivered.
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


def reply_from(messages: list[IRCMessage], nick: str, token: str) -> IRCMessage | None:
    """First message from `nick` on the channel containing `token` (word or exact substring)."""
    for msg in messages:
        if msg.nick == nick and token in msg.text:
            return msg
    return None


def any_reply_from(messages: list[IRCMessage], nick: str) -> IRCMessage | None:
    for msg in messages:
        if msg.nick == nick:
            return msg
    return None


def build_filler_paragraph(index: int, planted_token: str) -> str:
    """Deterministic, verbose filler paragraph ending in a planted token.

    Used to inflate a channel's context past its max_context_tokens so the
    compaction probe (P3) forces a compaction pass.
    """
    sentence = (
        f"This is filler paragraph number {index}, written to consume context "
        "budget in a controlled and repeatable way. It restates itself "
        "several times so that the channel's token count grows steadily "
        "without depending on any external content. "
    )
    body = sentence * 6
    return f"{body}Planted marker: {planted_token}"


def busy_slot_ids(slots_response: list[dict]) -> list[int]:
    """id of every slot reported as currently processing."""
    return [slot["id"] for slot in slots_response if slot.get("is_processing")]


def prompt_contains_token(slots_response: list[dict], slot_id: int, token: str) -> bool:
    """Whether the given slot's prompt field (if present) contains `token`."""
    for slot in slots_response:
        if slot.get("id") == slot_id and token in str(slot.get("prompt", "")):
            return True
    return False


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
            time.sleep(SLOW_SERVER_DELAY_S)
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


# ---------------------------------------------------------------------------
# llama-server /slots polling
# ---------------------------------------------------------------------------


def fetch_slots(llama_url: str) -> list[dict]:
    with urllib.request.urlopen(f"{llama_url}/slots", timeout=5) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Run context and probes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunContext:
    client: IRCClient
    bot_nick: str
    state_dir: Path
    run_dir: Path
    llama_url: str
    repo_dir: Path
    session_db: Path
    channel_ready: dict[str, bool]


def _transcript(ctx: RunContext, channel: str, sent: list[tuple[float, str]]) -> list[str]:
    items: list[tuple[float, str, str]] = [(ts, ">>", text) for ts, text in sent]
    for msg in ctx.client.messages.get(channel, []):
        items.append((msg.ts, "<<", f"{msg.nick}: {msg.text}"))
    items.sort(key=lambda x: x[0])
    return [f"{ts_fmt(ts)} {d} {t}" for ts, d, t in items]


async def _send(ctx: RunContext, channel: str, text: str, sent_log: list[tuple[float, str]]) -> None:
    await ctx.client.privmsg(channel, text)
    sent_log.append((time.time(), text))


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
    await _send(ctx, channel, "The anchovy code is 7291. Reply with that code.", sent)
    matched = await ctx.client.wait_until(
        channel, lambda msgs: reply_from(msgs, ctx.bot_nick, "7291") is not None, timeout=300
    )
    return ProbeResult(
        "smoke_echo",
        matched,
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

    await _send(
        ctx,
        channel,
        f"Use your read_file tool to read {a_path} and follow its instructions "
        "to determine a full code (you may need to read a second file it "
        "references). Reply with just the code.",
        sent,
    )
    matched = await ctx.client.wait_until(
        channel, lambda msgs: reply_from(msgs, ctx.bot_nick, "QRT-9083") is not None, timeout=300
    )
    duration = time.monotonic() - t0
    if not matched:
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

    await _send(
        ctx, channel,
        f"Fetch http://127.0.0.1:{SLOW_SERVER_PORT}/slow with web_fetch and tell me the token.",
        sent,
    )
    await asyncio.sleep(3)  # scenario-clarity heuristic, not a correctness gate (queue re-entrancy
                             # makes the pass criteria valid wherever M2 lands relative to M1's tool cycle)
    await _send(ctx, channel, "What is 17+25? Reply with just the number.", sent)

    async def find_reply(token_regex: re.Pattern[str]) -> IRCMessage | None:
        for msg in ctx.client.messages.get(channel, []):
            if msg.nick == ctx.bot_nick and token_regex.search(msg.text):
                return msg
        return None

    forty_two_re = re.compile(r"\b42\b")
    eelgrass_re = re.compile(re.escape(SLOW_SERVER_TOKEN))

    def both_seen(_msgs: list[IRCMessage]) -> bool:
        return (
            any(m.nick == ctx.bot_nick and forty_two_re.search(m.text) for m in ctx.client.messages.get(channel, []))
            and any(m.nick == ctx.bot_nick and eelgrass_re.search(m.text) for m in ctx.client.messages.get(channel, []))
        )

    matched = await ctx.client.wait_until(channel, both_seen, timeout=300)
    duration = time.monotonic() - t0
    if not matched:
        forty_two = await find_reply(forty_two_re)
        eelgrass = await find_reply(eelgrass_re)
        details = (
            f"timed out waiting for both replies (42 seen: {forty_two is not None}, "
            f"token seen: {eelgrass is not None})"
        )
        return ProbeResult("interleave", False, details, duration, _transcript(ctx, channel, sent))

    forty_two = await find_reply(forty_two_re)
    eelgrass = await find_reply(eelgrass_re)
    assert forty_two is not None and eelgrass is not None
    passed = forty_two.ts < eelgrass.ts
    details = (
        f"42-reply at {ts_fmt(forty_two.ts)}, token-reply at {ts_fmt(eelgrass.ts)}; "
        f"interleaved={passed}"
    )
    return ProbeResult("interleave", passed, details, duration, _transcript(ctx, channel, sent))


async def probe_compaction(ctx: RunContext) -> ProbeResult:
    channel = "#p-compact"
    if not ctx.channel_ready.get(channel):
        return _skip_result("compaction", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()
    ts0 = time.time() - 5  # small buffer against clock/DB timestamp skew

    for i in range(1, 16):
        await _send(ctx, channel, build_filler_paragraph(i, f"FILLER-PLANT-{i:02d}"), sent)
        await asyncio.sleep(0.3)

    await _send(ctx, channel, "Reply with the word CHECKPOINT-OK.", sent)
    matched = await ctx.client.wait_until(
        channel, lambda msgs: reply_from(msgs, ctx.bot_nick, "CHECKPOINT-OK") is not None, timeout=900
    )
    duration = time.monotonic() - t0
    if not matched:
        return ProbeResult(
            "compaction", False, "no reply containing CHECKPOINT-OK within timeout", duration,
            _transcript(ctx, channel, sent),
        )
    summarized = has_summary_row_since(ctx.session_db, f"irc:{channel}", ts0)
    passed = summarized
    details = (
        f"CHECKPOINT-OK reply observed; message_log summary row for {channel} "
        f"since probe start: {summarized}"
    )
    return ProbeResult("compaction", passed, details, duration, _transcript(ctx, channel, sent))


async def probe_restart_recovery(ctx: RunContext) -> ProbeResult:
    channel = "#p-restart"
    if not ctx.channel_ready.get(channel):
        return _skip_result("restart_recovery", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()

    await _send(ctx, channel, "My door code is 8814. Acknowledge.", sent)
    acked = await ctx.client.wait_until(
        channel, lambda msgs: any_reply_from(msgs, ctx.bot_nick) is not None, timeout=300
    )
    if not acked:
        return ProbeResult(
            "restart_recovery", False, "no acknowledgement reply before restart", time.monotonic() - t0,
            _transcript(ctx, channel, sent),
        )

    restart_script = ctx.repo_dir / "harness" / "restart-daemon.sh"
    daemon_log = ctx.run_dir / "daemon.log"
    restart_ts = time.time()
    subprocess.run([str(restart_script), str(daemon_log)], cwd=ctx.repo_dir, check=True, timeout=120)

    rejoined = await ctx.client.wait_for_nick_join(
        channel, ctx.bot_nick, timeout=180, after_ts=restart_ts
    )
    if not rejoined:
        return ProbeResult(
            "restart_recovery", False, "buster did not rejoin channel after restart",
            time.monotonic() - t0, _transcript(ctx, channel, sent),
        )
    # only replies after the restart are evidence of recovery, not the pre-restart ack
    await _send(ctx, channel, "What door code did I give you earlier in this conversation?", sent)
    matched = await ctx.client.wait_until(
        channel,
        lambda msgs: any(m.nick == ctx.bot_nick and m.ts > restart_ts and "8814" in m.text for m in msgs),
        timeout=300,
    )
    duration = time.monotonic() - t0
    details = (
        "post-restart reply contained 8814 (history reloaded)" if matched
        else "post-restart reply did not contain 8814 within timeout"
    )
    return ProbeResult("restart_recovery", matched, details, duration, _transcript(ctx, channel, sent))


class SlotsFetchError(Exception):
    """GET /slots failed while polling during a probe turn."""


async def probe_kv_slot(ctx: RunContext) -> ProbeResult:
    channel = "#p-slots"
    if not ctx.channel_ready.get(channel):
        return _skip_result("kv_slot", channel)
    sent: list[tuple[float, str]] = []
    t0 = time.monotonic()
    token = "SLOTPROBE-3319"
    turn1_samples: list[list[dict]] = []
    turn2_samples: list[list[dict]] = []

    async def poll_until_reply(text_marker: str, sink: list[list[dict]]) -> bool:
        deadline = time.monotonic() + 300
        while True:
            if reply_from(ctx.client.messages.get(channel, []), ctx.bot_nick, text_marker) is not None:
                return True
            if time.monotonic() >= deadline:
                return False
            try:
                sink.append(fetch_slots(ctx.llama_url))
            except (urllib.error.URLError, OSError, ValueError) as exc:
                raise SlotsFetchError(str(exc)) from exc
            await asyncio.sleep(0.5)

    await _send(ctx, channel, f"Remember this token: {token}. Reply with the word RECEIVED.", sent)
    try:
        r1 = await poll_until_reply("RECEIVED", turn1_samples)
        await _send(ctx, channel, "Reply with the word OK.", sent)
        r2 = await poll_until_reply("OK", turn2_samples)
    except SlotsFetchError as exc:
        return ProbeResult(
            "kv_slot", False, f"GET /slots failed: {exc}", time.monotonic() - t0,
            _transcript(ctx, channel, sent),
        )

    duration = time.monotonic() - t0
    if not (r1 and r2):
        return ProbeResult(
            "kv_slot", False, "did not observe both probe replies within timeout", duration,
            _transcript(ctx, channel, sent),
        )

    all_samples = turn1_samples + turn2_samples
    all_busy_ids: list[int] = []
    for sample in all_samples:
        all_busy_ids.extend(busy_slot_ids(sample))
    pinned = all(sid == 0 for sid in all_busy_ids)
    # §5.5 P5: the planted token is expected in slot 0's prompt during turn
    # 1 specifically (the message that contains it) — turn 2 is checked too,
    # but only for the report's diagnostic detail, not the pass criterion.
    token_seen_turn1 = any(prompt_contains_token(sample, 0, token) for sample in turn1_samples)
    token_seen_turn2 = any(prompt_contains_token(sample, 0, token) for sample in turn2_samples)
    passed = pinned and token_seen_turn1 and len(all_samples) > 0
    details = (
        f"{len(all_samples)} /slots samples ({len(turn1_samples)} turn1, {len(turn2_samples)} turn2); "
        f"busy slot ids observed: {sorted(set(all_busy_ids))}; "
        f"token in slot 0 prompt — turn1: {token_seen_turn1}, turn2: {token_seen_turn2}"
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
    _slow_httpd = start_slow_server()  # noqa: F841

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
        llama_url=args.llama_url,
        repo_dir=args.repo_dir,
        session_db=args.state_dir / "sessions.db",
        channel_ready=channel_ready,
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
