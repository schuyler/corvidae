#!/usr/bin/env python3
"""Regenerate the death spiral compaction fixture from sessions.db.

Usage:
    python scripts/regen_death_spiral_fixture.py [--db sessions.db.backup]

This extracts the death spiral conversation from the database and writes it
to tests/fixtures/death_spiral_compaction.json. Run this if sessions.db changes
and you need to update the fixture.
"""

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Regenerate death spiral fixture")
    parser.add_argument("--db", default="sessions.db.backup", help="Path to sessions database")
    parser.add_argument("--output", default=None, help="Output path (default: tests/fixtures/death_spiral_compaction.json)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "death_spiral_compaction.json"
    )

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT id, message, timestamp, message_type FROM message_log "
        "WHERE channel_id = ? ORDER BY id",
        ("cli:local",),
    ).fetchall()
    conn.close()

    all_msgs = []
    for row in rows:
        msg = json.loads(row[1])
        all_msgs.append({
            "id": row[0],
            "role": msg.get("role", "unknown"),
            "content": msg.get("content", ""),
            "message_type": row[3],
            "timestamp": row[2],
            "tool_calls": msg.get("tool_calls"),
        })

    # Find summary rows
    summary_ids = [m["id"] for m in all_msgs if m["message_type"] == "summary"]
    print(f"Found {len(all_msgs)} messages, {len(summary_ids)} summaries (ids: {summary_ids})")

    if len(summary_ids) < 2:
        print("❌ Expected at least 2 summaries for the death spiral fixture")
        return

    # Build truncated message list for the fixture
    truncated_msgs = []
    for m in all_msgs:
        entry = dict(m)
        if len(entry.get("content", "")) > 500:
            entry["content"] = entry["content"][:500] + "...[truncated]"
        truncated_msgs.append(entry)

    # Build compaction segments
    fixture = {
        "description": (
            "Death spiral conversation from sessions.db (real Corvidae session). "
            "Two compaction events occurred. The second compaction produced a summary "
            'that said "No user instructions, questions, or decisions have been made yet" '
            "even though extensive user instructions existed in the first summary."
        ),
        "source": f"sessions.db.backup, channel cli:local, extracted {all_msgs[0]['timestamp']:.0f}-{all_msgs[-1]['timestamp']:.0f}",
        "all_messages_truncated": truncated_msgs,
        "compaction_segments": [],
    }

    # Segment 1: everything before first summary
    first_sum = summary_ids[0]
    seg1_compacted = [m for m in all_msgs if m["id"] < first_sum and m["message_type"] == "message"]
    seg1_summary = next(m for m in all_msgs if m["id"] == first_sum)
    seg1_retained = [m for m in all_msgs if first_sum < m["id"] <= summary_ids[1]]

    fixture["compaction_segments"].append({
        "segment_id": "first_compaction",
        "compacted_message_ids": [seg1_compacted[0]["id"], seg1_compacted[-1]["id"]] if seg1_compacted else [],
        "summary_row_id": first_sum,
        "compacted": seg1_compacted,
        "summary": seg1_summary,
        "retained": seg1_retained,
    })

    # Segment 2: the death spiral
    second_sum = summary_ids[1]
    seg2_compacted = [m for m in all_msgs if first_sum < m["id"] < second_sum and m["message_type"] == "message"]
    seg2_summary = next(m for m in all_msgs if m["id"] == second_sum)
    # Retained: up to 30 messages after (enough for evaluation context)
    seg2_retained = [m for m in all_msgs if second_sum < m["id"] <= second_sum + 30]

    fixture["compaction_segments"].append({
        "segment_id": "second_compaction_death_spiral",
        "description": (
            "This is the death spiral: only tool outputs were compacted, "
            "but the resulting summary said no user instructions existed."
        ),
        "compacted_message_ids": [seg2_compacted[0]["id"], seg2_compacted[-1]["id"]] if seg2_compacted else [],
        "summary_row_id": second_sum,
        "compacted": seg2_compacted,
        "summary": seg2_summary,
        "retained": seg2_retained,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)

    size = output_path.stat().st_size
    print(f"\n✅ Written to {output_path}")
    print(f"   Size: {size/1024:.0f} KB")
    print(f"   Segment 1: {len(seg1_compacted)} compacted, {len(seg1_retained)} retained")
    print(f"   Segment 2: {len(seg2_compacted)} compacted, {len(seg2_retained)} retained")


if __name__ == "__main__":
    main()
