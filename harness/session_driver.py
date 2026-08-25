#!/usr/bin/env python3
"""Buster harness session driver — shakedown conversations, not probes.

Three modes sharing one maildir-style file protocol (see
.claude/plans/shakedown-session-driver.md):

  serve --session-dir D --channel '#chat' [...]   long-running IRC presence
  send  --session-dir D --text "..."              one turn, blocks for reply
  stop  --session-dir D --state-dir S              stop serve, collect artifacts

`serve` holds one IRC connection to one channel for the whole session and
drains numbered `inbox/NNN.txt` files one at a time, writing `outbox/NNN.json`
per turn and appending to `transcript.log`. `send`/`stop` are the buster-host-side
halves of harness/session.sh's ssh calls — they never touch IRC directly.

Stdlib-only, Python 3.10-compatible: runs under buster-host's system python3, same
as driver.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from harness.driver import IRCClient, IRCMessage, reply_from, ts_fmt
except ImportError:
    from driver import IRCClient, IRCMessage, reply_from, ts_fmt


# ---------------------------------------------------------------------------
# Inbox protocol: sequence numbers, ordering, stop priority
# ---------------------------------------------------------------------------


def next_seq(inbox_dir: Path) -> int:
    """Next turn number: max existing `NNN.txt` + 1, 1 if none exist.

    Turn files are never removed once written (send's numbering depends on
    them staying put), so this always scans the full inbox, not just
    unprocessed entries.
    """
    seqs = [int(p.stem) for p in inbox_dir.glob("*.txt")]
    return max(seqs, default=0) + 1


def next_inbox_action(inbox_dir: Path) -> Path | str | None:
    """`"STOP"` if inbox/STOP exists (wins over any numbered turn file),
    else the lowest-numbered `NNN.txt`, else None if nothing is pending.
    """
    if (inbox_dir / "STOP").exists():
        return "STOP"
    turns = sorted(inbox_dir.glob("*.txt"), key=lambda p: p.name)
    return turns[0] if turns else None


# ---------------------------------------------------------------------------
# Turn-record building: latency anchoring, timeout, settle-window grouping
# ---------------------------------------------------------------------------


def build_turn_record(
    seq: int,
    text: str,
    sent_at: float,
    messages: list[IRCMessage],
    bot_nick: str,
    timeout: float,
    settle: float,
    now: float,
) -> dict:
    """Pure post-hoc summary over an already-final message list.

    Assumes the caller only invokes this once its own wait-for-reply polling
    has already resolved as reply-or-timeout — it does not itself model a
    "still waiting" state, so `timeout` and `now` are accepted for the
    outbox record's shape but not evaluated here.
    """
    bot_msgs = [m for m in messages if m.nick == bot_nick and m.ts > sent_at]
    if not bot_msgs:
        return {
            "seq": seq,
            "text": text,
            "sent_at": sent_at,
            "first_reply_at": None,
            "latency_s": None,
            "reply": [],
            "timed_out": True,
        }

    first = bot_msgs[0]
    reply = [f"{first.nick}: {first.text}"]
    prev_ts = first.ts
    for m in bot_msgs[1:]:
        if m.ts - prev_ts >= settle:
            break
        reply.append(f"{m.nick}: {m.text}")
        prev_ts = m.ts

    return {
        "seq": seq,
        "text": text,
        "sent_at": sent_at,
        "first_reply_at": first.ts,
        "latency_s": first.ts - sent_at,
        "reply": reply,
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------


def format_sent_line(ts: float, text: str) -> str:
    return f"{ts_fmt(ts)} >> {text}"


def format_received_line(ts: float, nick: str, text: str) -> str:
    return f"{ts_fmt(ts)} << {nick}: {text}"


def format_latency_line(ts: float, seq: int, latency_s: float) -> str:
    return f"{ts_fmt(ts)} -- turn {seq:03d} latency {latency_s:.3f}s"


# ---------------------------------------------------------------------------
# Nick derivation
# ---------------------------------------------------------------------------


def derive_default_nick(channel: str, max_len: int = 9) -> str:
    """Channel name, `#` stripped, truncated to ngircd's NICKLEN (9)."""
    name = channel[1:] if channel.startswith("#") else channel
    return name[:max_len]


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------


def collect_artifacts(state_dir: Path, session_dir: Path) -> None:
    """Copy sessions.db, metrics.jsonl, corvidae.log* from state_dir into
    session_dir, flat. Missing rotated logs are not an error — a short
    session may not have rotated any.
    """
    db_path = state_dir / "sessions.db"
    if db_path.exists():
        src = sqlite3.connect(db_path)
        try:
            dst = sqlite3.connect(session_dir / "sessions.db")
            try:
                # backup() takes a consistent snapshot even while the daemon
                # holds the WAL open; a plain file copy could grab a torn write.
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()

    metrics_path = state_dir / "metrics.jsonl"
    if metrics_path.exists():
        shutil.copy2(metrics_path, session_dir / "metrics.jsonl")

    for log_path in state_dir.glob("corvidae.log*"):
        shutil.copy2(log_path, session_dir / log_path.name)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def _append_line(path: Path, line: str) -> None:
    with path.open("a") as f:
        f.write(line + "\n")


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


async def _process_turn(
    client: IRCClient,
    channel: str,
    transcript_path: Path,
    turn_path: Path,
    outbox_dir: Path,
    seq: int,
    bot_nick: str,
    turn_timeout: float,
    settle: float,
) -> None:
    text = turn_path.read_text()
    sent_at = time.time()
    for line in text.splitlines():
        await client.privmsg(channel, line)
        _append_line(transcript_path, format_sent_line(sent_at, line))

    # Empty token: existence-after-send is load-bearing, not content — same
    # reasoning as driver.py's send_and_await_any_reply.
    got_reply = await client.wait_until(
        channel,
        lambda msgs: reply_from(msgs, bot_nick, "", after_ts=sent_at) is not None,
        turn_timeout,
    )
    if got_reply:
        # Keep collecting while bot lines keep arriving inside the settle
        # gap; Buster's replies can span several PRIVMSGs (512-byte splits,
        # progress lines) that would otherwise bleed into the next turn.
        while True:
            bot_msgs = [
                m for m in client.messages.get(channel, [])
                if m.nick == bot_nick and m.ts > sent_at
            ]
            if time.time() - bot_msgs[-1].ts >= settle:
                break
            await asyncio.sleep(0.5)

    record = build_turn_record(
        seq=seq,
        text=text,
        sent_at=sent_at,
        messages=client.messages.get(channel, []),
        bot_nick=bot_nick,
        timeout=turn_timeout,
        settle=settle,
        now=time.time(),
    )
    _write_json_atomic(outbox_dir / f"{seq:03d}.json", record)
    if not record["timed_out"]:
        _append_line(
            transcript_path,
            format_latency_line(record["first_reply_at"], seq, record["latency_s"]),
        )


async def _serve_async(args: argparse.Namespace) -> int:
    session_dir: Path = args.session_dir
    inbox = session_dir / "inbox"
    outbox = session_dir / "outbox"
    session_dir.mkdir(parents=True, exist_ok=True)
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    transcript_path = session_dir / "transcript.log"

    nick = args.nick or derive_default_nick(args.channel)
    client = IRCClient(args.irc_host, args.irc_port, nick)
    await client.connect()
    await client.join(args.channel)

    session_json_path = session_dir / "session.json"
    session_json_path.write_text(json.dumps({
        "channel": args.channel,
        "nick": nick,
        "bot_nick": args.bot_nick,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    flushed = 0
    last_seq = 0

    def flush_transcript() -> None:
        nonlocal flushed
        msgs = client.messages.get(args.channel, [])
        new = msgs[flushed:]
        if new:
            for m in new:
                _append_line(transcript_path, format_received_line(m.ts, m.nick, m.text))
            flushed = len(msgs)

    while True:
        flush_transcript()
        action = next_inbox_action(inbox)

        if action == "STOP":
            await client.close()
            data = json.loads(session_json_path.read_text())
            data["stopped_at"] = datetime.now(timezone.utc).isoformat()
            session_json_path.write_text(json.dumps(data, indent=2))
            return 0

        if action is not None:
            seq = int(action.stem)
            # Turns are single-sender and processed strictly in order (send
            # blocks for the prior outbox record before writing the next
            # one), so the lowest-numbered file is only ever new the first
            # time we see it — last_seq is what tells "already handled"
            # apart from "next up", since turn files are never removed.
            if seq > last_seq:
                await _process_turn(
                    client, args.channel, transcript_path, action, outbox,
                    seq, args.bot_nick, args.turn_timeout, args.settle,
                )
                last_seq = seq
                continue

        await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# send / stop
# ---------------------------------------------------------------------------


def cmd_send(args: argparse.Namespace) -> int:
    session_dir: Path = args.session_dir
    inbox = session_dir / "inbox"
    outbox = session_dir / "outbox"
    text = args.text if args.text is not None else sys.stdin.read()

    seq = next_seq(inbox)
    turn_path = inbox / f"{seq:03d}.txt"
    tmp_path = turn_path.with_suffix(turn_path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.rename(turn_path)

    outbox_path = outbox / f"{seq:03d}.json"
    deadline = time.monotonic() + args.poll_timeout
    while not outbox_path.exists():
        if time.monotonic() >= deadline:
            print(
                f"session driver did not respond to turn {seq:03d} within "
                f"{args.poll_timeout}s (driver dead?)",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.5)

    record = json.loads(outbox_path.read_text())
    print(json.dumps(record, indent=2))
    return 1 if record.get("timed_out") else 0


def cmd_stop(args: argparse.Namespace) -> int:
    session_dir: Path = args.session_dir
    (session_dir / "inbox" / "STOP").write_text("")

    session_json_path = session_dir / "session.json"
    deadline = time.monotonic() + 30
    confirmed = False
    while time.monotonic() < deadline:
        if session_json_path.exists():
            data = json.loads(session_json_path.read_text())
            if "stopped_at" in data:
                confirmed = True
                break
        time.sleep(0.5)
    if not confirmed:
        # Collection must still happen for a dead driver — it's the only
        # place artifacts get pulled out before the next probe run wipes them.
        print("session driver did not confirm stop within 30s; collecting anyway", file=sys.stderr)

    collect_artifacts(args.state_dir, session_dir)
    print(str(session_dir))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--session-dir", required=True, type=Path)
    serve_p.add_argument("--channel", required=True)
    serve_p.add_argument("--nick", default=None)
    serve_p.add_argument("--irc-host", default="127.0.0.1")
    serve_p.add_argument("--irc-port", type=int, default=6667)
    serve_p.add_argument("--bot-nick", default="buster")
    serve_p.add_argument("--turn-timeout", type=float, default=150.0)
    serve_p.add_argument("--settle", type=float, default=2.0)

    send_p = sub.add_parser("send")
    send_p.add_argument("--session-dir", required=True, type=Path)
    # Omit --text to read the turn from stdin — session.sh pipes it that
    # way so apostrophes/quotes in chat text never have to survive shell
    # interpolation across the ssh hop.
    send_p.add_argument("--text", default=None)
    # turn-timeout (150) + settle (2) + margin.
    send_p.add_argument("--poll-timeout", type=float, default=180.0)

    stop_p = sub.add_parser("stop")
    stop_p.add_argument("--session-dir", required=True, type=Path)
    stop_p.add_argument("--state-dir", required=True, type=Path)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "serve":
        return asyncio.run(_serve_async(args))
    if args.mode == "send":
        return cmd_send(args)
    return cmd_stop(args)


if __name__ == "__main__":
    sys.exit(main())
