#!/usr/bin/env python3
"""Replay a recorded consolidation range against instruction variants.

Reproduces the exact request `MemoryPlugin._consolidate_range` sends for a
range that has left the live window (its catch-up: the agent's rebuilt view
plus the consolidation instruction, on the consolidation stage's default profile), against a frozen
snapshot database, so that `memory.consolidation_fork_prompt` variants can
be compared on identical input without a live IRC session (see
.claude/plans/consolidation-replay.md). Both build the request with
`build_catchup_segments`.

    uv run python harness/consolidate_replay.py --db <snapshot.db> \\
        --channel 'irc:#s-compact' --range 76:163 \\
        --system-prompt-file <identity+register.txt> [--prompt-file F] [--trials N]

A range that the daemon would split into more than one view (it spans a
summary boundary or exceeds the budget) is rejected, listing the segments'
id ranges; replay those ranges one at a time.

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

from corvidae.agent import _resolve_mid_conversation_system_role
from corvidae.channel import ChannelConfig
from corvidae.fork import ViewInputs
from corvidae.llm import LLMClient
from corvidae.llm_plugin import parse_llm_config
from corvidae.thinking import apply_reasoning_history
from corvidae_cognition.memory import (
    CONSOLIDATION_EXTRA_BODY,
    resolve_consolidation_fork_prompt,
    _parse_json_block,
    build_catchup_segments,
    dialog_from_rows,
    fetch_range_rows,
)

# Profile configs carry keys LLMClient does not take (e.g. `dimensions` on the
# embedding profile). Derived, not hand-listed, so a constructor rename cannot
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


def _load_config(config_path: Path) -> dict:
    """Read the agent.yaml at ``config_path``; ValueError names it when it
    cannot be read."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except OSError as e:
        raise ValueError(f"cannot read config {config_path}: {e}") from e


def _build_client(config: dict, config_path: Path) -> LLMClient:
    """The client for the profile the daemon's consolidation stage resolves
    to (llm.stages, else the default chat profile), as
    ``LLMPlugin.get_client("consolidation")`` builds it."""
    # The same parse the daemon runs, so the stage resolves identically.
    try:
        llm_config = parse_llm_config(config.get("llm"))
    except ValueError as e:
        raise ValueError(f"{config_path}: {e}") from e
    profile = llm_config.profile_for("consolidation", "chat")
    cfg = llm_config.profiles[profile]
    kwargs = {k: cfg[k] for k in _LLM_CLIENT_KEYS if k in cfg}
    return LLMClient(**kwargs)


# ---------------------------------------------------------------------------
# Replay (R3 / R4 / D5)
# ---------------------------------------------------------------------------


class RangeNotOneSegment(ValueError):
    """The range would be sent as more than one view (or none) by the daemon."""


def view_inputs_from_config(config: dict, system_prompt: str) -> ViewInputs:
    """The ViewInputs the replay builds the daemon's request with.

    shortcut: agent-level config only, skips exact and transport-wide channel
    entries; upgrade per Deferred when an operator replays a channel with its
    own entries. ``system_prompt`` is the channel's resolved identity +
    register, supplied by the operator.
    """
    agent_cfg = config.get("agent") or {}
    # ChannelConfig() with no overrides resolves over the agent: section.
    resolved = ChannelConfig().resolve(agent_cfg)
    policy = resolved["reasoning_history"]
    return ViewInputs(
        system_prompt=system_prompt,
        max_tokens=resolved["max_context_tokens"],
        mid_role=_resolve_mid_conversation_system_role(agent_cfg),
        apply_reasoning_history=lambda messages: apply_reasoning_history(messages, policy),
    )


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
    inputs: ViewInputs,
    prompt_path: str | None = None,
) -> None:
    """Build the daemon's catch-up request for the range, run `trials` chat()
    calls, and write input.txt/meta.json/trials.jsonl into out_dir.

    ``prompt_text`` is the consolidation instruction (the fork prompt).
    A chat() failure propagates uncaught — the caller (main) turns it into
    R8's loud, base_url-naming failure. Only per-trial JSON-parse failures
    are caught and recorded, since those are a legitimate trial outcome
    (the thing under study), not an infrastructure failure.

    Raises:
        RangeNotOneSegment: the daemon would send the range as other than
            exactly one request; the message lists the segments' id ranges.
    """
    db = await open_source_db(db_path)
    try:
        rows = await fetch_range_rows(db, channel_id, after_id, through_id)
        # The same builder the daemon's catch-up calls.
        segments = await build_catchup_segments(
            db, channel_id, after_id, through_id, inputs, prompt_text
        )
    finally:
        await db.close()
    dialog = dialog_from_rows(rows, channel_id)
    if len(segments) != 1:
        spans = ", ".join(f"{seg.rows[0][0]}:{seg.end_id}" for seg in segments) or "none"
        raise RangeNotOneSegment(
            f"range {after_id + 1}:{through_id} for {channel_id} is {len(segments)} "
            f"segment(s) in the daemon (id ranges: {spans}); replay one of those "
            "ranges instead"
        )
    messages = segments[0].messages

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The request exactly as sent (there is no transcript any more).
    (out_dir / "input.txt").write_text(json.dumps(messages, indent=2))

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
                # Same response_format constraint the daemon sends (#199).
                response = await client.chat(
                    messages, extra_body=CONSOLIDATION_EXTRA_BODY
                )
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
    p.add_argument("--prompt-file", default=None,
                   help="consolidation instruction; defaults to memory.consolidation_fork_prompt "
                        "from --config, else DEFAULT_CONSOLIDATION_FORK_PROMPT, as the daemon does")
    p.add_argument("--system-prompt-file", required=True,
                   help="the channel's resolved identity + register (heads the view); "
                        "the budget and reasoning policy come from agent: in --config")
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
        dialog = dialog_from_rows(rows, args.channel)
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
        config = _load_config(Path(args.config))
        client = _build_client(config, Path(args.config))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    # The system prompt and the instruction are operator-supplied files.
    prompt_path = args.prompt_file
    try:
        system_prompt = Path(args.system_prompt_file).read_text()
        # No file given: the config's instruction (or the default), as the daemon uses.
        prompt_text = (
            Path(prompt_path).read_text() if prompt_path
            else resolve_consolidation_fork_prompt(config)
        )
        inputs = view_inputs_from_config(config, system_prompt)
    except (OSError, ValueError) as e:
        print(
            f"cannot build the view inputs ({type(e).__name__}: {e}); check "
            f"--system-prompt-file {args.system_prompt_file}, --prompt-file and "
            f"the agent: section of {args.config}",
            file=sys.stderr,
        )
        return 1
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
            inputs=inputs,
            prompt_path=prompt_path,
        )
    except RangeNotOneSegment as e:
        print(str(e), file=sys.stderr)
        return 1
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
