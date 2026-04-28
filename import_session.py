#!/usr/bin/env python3
"""One-off import of nanobot session JSONL into corvidae SQLite.

Run:
    python /home/nanobot/code/nanobot/corvidae/import_session.py \
        /home/nanobot/code/nanobot/sessions/irc_#general.jsonl \
        /home/nanobot/code/nanobot/corvidae/sessions.db \
        irc:#general

Uses synchronous sqlite3 for speed on a one-shot import.
"""

import json
import sqlite3
import sys
import time
from datetime import datetime


def convert_ts(ts_str: str) -> float:
    """Convert ISO 8601 timestamp to Unix epoch float."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time()


def main():
    if len(sys.argv) < 4:
        print("Usage: import_session.py <jsonl_path> <db_path> <channel_id>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    db_path = sys.argv[2]
    channel_id = sys.argv[3]

    # Connect to corvidae DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table if not exists (matches corvidae's existing schema)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS message_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_log_channel ON message_log (channel_id, timestamp)"
    )

    # Check if channel already has data
    cur.execute("SELECT COUNT(*) FROM message_log WHERE channel_id = ?", (channel_id,))
    existing_count = cur.fetchone()[0]

    if existing_count > 0:
        print(f"Channel '{channel_id}' already has {existing_count} messages. Skipping.")
        conn.close()
        return

    # Read and batch insert messages from JSONL
    batch = []
    count = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Skip metadata line
            if data.get("_type") == "metadata":
                continue

            ts = convert_ts(data.get("timestamp", ""))
            # Strip nanobot-internal fields that would confuse corvidae
            cleaned = {k: v for k, v in data.items() if k not in ("_message_type",)}
            batch.append((channel_id, json.dumps(cleaned), ts))
            count += 1

            # Insert in batches of 500 for performance
            if len(batch) >= 500:
                cur.executemany(
                    "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
                    batch,
                )
                conn.commit()
                batch.clear()

    # Insert remaining
    if batch:
        cur.executemany(
            "INSERT INTO message_log (channel_id, message, timestamp) VALUES (?, ?, ?)",
            batch,
        )
        conn.commit()

    print(f"Imported {count} messages into '{channel_id}' in {db_path}")
    conn.close()


if __name__ == "__main__":
    main()
