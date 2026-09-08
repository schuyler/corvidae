#!/usr/bin/env python3
"""Replay a recorded consolidation range against prompt variants.

Reproduces the exact call `MemoryPlugin._consolidate_range` /
`_summarize_range` make for a given channel and message-id range, against a
frozen snapshot database, so that `consolidation_prompt` variants can be
compared on identical input without a live IRC session (see
.claude/plans/consolidation-replay.md).

    uv run python harness/consolidate_replay.py --db <snapshot.db> \\
        --channel 'irc:#s-compact' --range 76:163 [--prompt-file F] [--trials N]

Prerequisite: an SSH tunnel to the Buster host's llama-server —
    ssh -N -L 8080:127.0.0.1:8080 "$BUSTER_HOST"
torn down with `ssh -O cancel -L 8080:127.0.0.1:8080 "$BUSTER_HOST"` (not a kill —
see harness/README.md). `buster.yaml.in`'s `base_url` is then correct
verbatim on the Mac.

Runs Mac-side inside the project venv, unlike harness/session_driver.py:
this imports corvidae.* directly, which is what makes reproducing the
daemon's own call path possible at all.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import yaml

from corvidae.llm import LLMClient
from corvidae.memory import (
    DEFAULT_CONSOLIDATION_PROMPT,
    _dialog_transcript,
    _parse_json_block,
    dialog_from_rows,
    fetch_range_rows,
)

# Role configs carry keys LLMClient does not take (e.g. `dimensions` on the
# embedding role). Derived, not hand-listed, so a constructor rename cannot
# silently drop a key and leave replay diverging from the daemon.
_LLM_CLIENT_KEYS = frozenset(
    inspect.signature(LLMClient.__init__).parameters
) - {"self"}

TUNNEL_HINT = (
    f"ssh -N -L 8080:127.0.0.1:8080 {os.environ.get('BUSTER_HOST', '<buster-host>')}"
)


# ---------------------------------------------------------------------------
# Range translation (D4)
# ---------------------------------------------------------------------------


def parse_range(s: str) -> tuple[int, int]:
    """'76:163' (inclusive, as read off a memory row) -> (after_id, through_id)
    = (75, 163), the half-open (after, through] bound fetch_range_rows uses.
    """
    start_s, end_s = s.split(":")
    start, end = int(start_s), int(end_s)
    return start - 1, end


# ---------------------------------------------------------------------------
# Call assembly (D2) — identical to _summarize_range's payload
# ---------------------------------------------------------------------------


def build_messages(prompt_text: str, dialog: list[dict]) -> list[dict]:
    """The exact two-message payload _summarize_range sends."""
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": _dialog_transcript(dialog)},
    ]


# ---------------------------------------------------------------------------
# Source database (D7 / R7 — never written)
# ---------------------------------------------------------------------------


async def open_source_db(path) -> aiosqlite.Connection:
    """Open the snapshot read-only. Needs the -shm index alongside the -wal
    file (see harness/README.md) — snapshot while the daemon is running."""
    return await aiosqlite.connect(f"file:{path}?mode=ro", uri=True)


async def _distinct_channels(db: aiosqlite.Connection) -> list[str]:
    async with db.execute("SELECT DISTINCT channel_id FROM message_log") as cur:
        return [row[0] for row in await cur.fetchall()]


# ---------------------------------------------------------------------------
# LLM client (D3)
# ---------------------------------------------------------------------------


def _build_client(config_path: Path) -> LLMClient:
    """llm.background, falling back to llm.main (LLMPlugin.get_client's
    documented fallback — background is the role _summarize_range uses)."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except OSError as e:
        raise ValueError(f"cannot read config {config_path}: {e}") from e
    llm_config = config.get("llm") or {}
    cfg = llm_config.get("background") or llm_config.get("main")
    if cfg is None:
        raise ValueError(f"{config_path}: no llm.main configured")
    kwargs = {k: cfg[k] for k in _LLM_CLIENT_KEYS if k in cfg}
    return LLMClient(**kwargs)


# ---------------------------------------------------------------------------
# Replay (R3 / R4 / D5)
# ---------------------------------------------------------------------------


async def run_replay(
    *,
    db_path,
    channel_id: str,
    after_id: int,
    through_id: int,
    prompt_text: str,
    trials: int,
    out_dir,
    client: LLMClient,
    prompt_path: str | None = None,
) -> None:
    """Fetch the range, assemble the daemon's call, run `trials` chat()
    calls, and write input.txt/meta.json/trials.jsonl into out_dir.

    A chat() failure propagates uncaught — the caller (main) turns it into
    R8's loud, base_url-naming failure. Only per-trial JSON-parse failures
    are caught and recorded, since those are a legitimate trial outcome
    (the thing under study), not an infrastructure failure.
    """
    db = await open_source_db(db_path)
    try:
        rows = await fetch_range_rows(db, channel_id, after_id, through_id)
    finally:
        await db.close()
    dialog = dialog_from_rows(rows)
    transcript = _dialog_transcript(dialog)
    messages = build_messages(prompt_text, dialog)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input.txt").write_text(transcript)

    meta = {
        "channel": channel_id,
        "range": f"{after_id + 1}:{through_id}",
        "after_id": after_id,
        "through_id": through_id,
        "raw_row_count": len(rows),
        "dialog_count": len(dialog),
        "prompt_path": prompt_path,
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "model": client.model,
        "base_url": client.base_url,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    await client.start()
    try:
        with (out_dir / "trials.jsonl").open("w") as f:
            for i in range(trials):
                start = time.monotonic()
                response = await client.chat(messages)
                latency_s = time.monotonic() - start
                text = response["choices"][0]["message"]["content"]
                record: dict = {"index": i, "latency_s": latency_s, "raw_output": text}
                try:
                    data = _parse_json_block(text)
                    record["summary"] = data.get("summary")
                    record["topic_tags"] = data.get("topic_tags")
                    record["participants"] = data.get("participants")
                except Exception as e:
                    record["error"] = str(e)
                f.write(json.dumps(record) + "\n")
    finally:
        await client.stop()


# ---------------------------------------------------------------------------
# CLI (R8 — each failure names what was observed)
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="path to a read-only snapshot database")
    p.add_argument("--channel", required=True, help="channel_id, e.g. irc:#s-compact")
    p.add_argument("--range", required=True, help="inclusive msg_id range, e.g. 76:163")
    p.add_argument("--prompt-file", default=None, help="defaults to DEFAULT_CONSOLIDATION_PROMPT")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--out", default="harness/replay_out")
    # Default matches buster.yaml.in's convention; a missing default file is
    # fine — db/channel/range are checked first and fail before it's read.
    p.add_argument("--config", default="agent.yaml")
    return p.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"db not found: {db_path}", file=sys.stderr)
        return 1

    after_id, through_id = parse_range(args.range)

    # DB-side checks first: they don't need a working LLM config, and
    # ordering them first lets a bad --channel/--range surface without
    # also demanding a valid --config.
    db = await open_source_db(db_path)
    try:
        channels = await _distinct_channels(db)
        if args.channel not in channels:
            print(
                f"channel {args.channel!r} not found in {db_path}; "
                f"channels present: {channels}",
                file=sys.stderr,
            )
            return 1
        rows = await fetch_range_rows(db, args.channel, after_id, through_id)
        dialog = dialog_from_rows(rows)
        if not dialog:
            print(
                f"range {args.range} for {args.channel} yields no dialog: "
                f"rows={len(rows)} dialog={len(dialog)}",
                file=sys.stderr,
            )
            return 1
    finally:
        await db.close()

    try:
        client = _build_client(Path(args.config))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    prompt_path = args.prompt_file
    prompt_text = (
        Path(prompt_path).read_text() if prompt_path else DEFAULT_CONSOLIDATION_PROMPT
    )
    out_dir = Path(args.out) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        await run_replay(
            db_path=db_path,
            channel_id=args.channel,
            after_id=after_id,
            through_id=through_id,
            prompt_text=prompt_text,
            trials=args.trials,
            out_dir=out_dir,
            client=client,
            prompt_path=prompt_path,
        )
    except Exception as e:
        print(
            f"replay against {client.base_url} failed: "
            f"{type(e).__name__}: {e}\n"
            f"if the server is unreachable, check the ssh tunnel ({TUNNEL_HINT})",
            file=sys.stderr,
        )
        return 1

    print(str(out_dir))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
