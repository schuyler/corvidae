"""Load sqlite-vec into a seeding connection.

Memory's tests seed ``memory_vec`` rows through the read-write ``db`` fixture
connection. The daemon loads the extension on the session writer and on the
reader, never on a test's seeding connection, so a test that writes vec rows
loads it here first.
"""

import aiosqlite


async def load_vec_for_seed(db: aiosqlite.Connection) -> bool:
    """Load sqlite-vec on ``db``; False when the build or install lacks it."""
    try:
        import sqlite_vec

        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        return True
    except (ImportError, AttributeError):
        # No sqlite-vec or no extension support: the caller's own guards skip.
        return False
