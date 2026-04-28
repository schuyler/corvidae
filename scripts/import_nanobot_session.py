#!/usr/bin/env python3
"""Import messages from nanobot's JSONL session files into corvidae's SQLite DB.

Usage:
    python scripts/import_nanobot_session.py <nanobot_jsonl_path> <corvidae_db_path> [channel_id]

The channel_id defaults to the key from the metadata line.

Skips the metadata line and any messages with tool_call_id (orphan tool results).
"""

import asyncio
import json
import sys
import time
from datetime import datetime

import aiosqlite


async def convert_timestamp(ts_str: str) -> float:
    """Convert ISO 8601 timestamp string to Unix float."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time()


async def import_session(jsonl_path: str, db_path: str, channel_id: str | None = None) -> int:
    """Import messages from a nanobot JSONL session into corvidae SQLite.

    Returns the number of messages imported.
    """
    # Read all messages from JSONL
    messages: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Skip metadata line
            if data.get("_type") == "metadata":
                if channel_id is None:
                    channel_id = data.get("key", f"imported_{line_num}")
                continue

            # Skip orphan tool results (no role or has tool_call_id without matching turn)
            if data.get("tool_call_id"):
                continue

            messages.append(data)

    print(f"Read {len(messages)} message rows from {jsonl_path}")

    # Connect to corvidae DB and create table if needed
    db = await aiosqlite.connect(db_path)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS message_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp REAL NOT NULL,
            message_type TEXT NOT NULL DEFAULT 'message'
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_log_channel ON message_log (channel_id, timestamp)"
    )

    # Insert messages in batches
    batch = []
    imported = 0
    for msg in messages:
        ts = await convert_timestamp(msg.get("timestamp", ""))
        # Strip nanobot-internal fields that would confuse corvidae's conversation builder
        cleaned = {k: v for k, v in msg.items() if k not in ("_message_type", "reasoning_content")}
        batch.append((channel_id, json.dumps(cleaned), ts, "message"))
        imported += 1

    await db.executemany(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) VALUES (?, ?, ?, ?)",
        batch,
    )
    await db.commit()
    print(f"Imported {imported} messages into '{channel_id}' in {db_path}")

    await db.close()
    return imported


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    jsonl_path = sys.argv[1]
    db_path = sys.argv[2]
    channel_id = sys.argv[3] if len(sys.argv) > 3 else None

    count = asyncio.run(import_session(jsonl_path, db_path, channel_id))
    print(f"Done. {count} messages imported.")
