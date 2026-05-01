"""Schema migrations for Corvidae SQLite database.

This module defines a forward-only sequential migration system. Each migration
is a SQL string stored in the ``MIGRATIONS`` list. Migrations are numbered by
their 1-based index (MIGRATIONS[0] is migration #1, MIGRATIONS[1] is #2, …).

To add a new migration, simply append a new SQL string to ``MIGRATIONS``.
The ``init_db`` function in ``persistence.py`` will automatically detect and
apply any pending migrations on next startup.
"""

# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------

MIGRATIONS: list[str] = [
    # Migration 001 — Initial schema: message_log table and index
    """\
CREATE TABLE IF NOT EXISTS message_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp REAL NOT NULL,
    message_type TEXT NOT NULL DEFAULT 'message'
);

CREATE INDEX IF NOT EXISTS idx_log_channel ON message_log (channel_id, timestamp);
""",
]
