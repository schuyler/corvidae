"""Tests for SQLite schema versioning and sequential migration runner.

Covers the requirements from issue #14:
  1. A `schema_version` meta table tracks the current migration version.
  2. Existing DDL is the first numbered migration (migration 001).
  3. `init_db` checks the current version and applies pending migrations
     in sequential order.
  4. Each migration runs inside a transaction (BEGIN / COMMIT).
  5. Already-applied migrations are never re-applied (idempotent).
  6. A new `corvidae.migrations` module exposes the migration registry.
"""

import aiosqlite
import pytest
import pytest_asyncio

from corvidae.migrations import MIGRATIONS
from corvidae.persistence import get_schema_version, init_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def raw_db():
    """A bare in-memory SQLite connection — no tables, no init_db."""
    async with aiosqlite.connect(":memory:") as conn:
        yield conn


@pytest_asyncio.fixture
async def initialized_db():
    """An in-memory database that has already been initialised via init_db."""
    async with aiosqlite.connect(":memory:") as db:
        await init_db(db)
        yield db


# Helper SQL for seeding a database that is already at version 1
# (migration 001 applied manually). Used by three test classes.
_SEED_VERSION_1_SQL = """\
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version (version) VALUES (1);
CREATE TABLE IF NOT EXISTS message_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp REAL NOT NULL,
    message_type TEXT NOT NULL DEFAULT 'message'
);
CREATE INDEX IF NOT EXISTS idx_log_channel
    ON message_log (channel_id, timestamp);
"""


@pytest_asyncio.fixture
async def version1_db():
    """An in-memory database seeded at schema version 1 (migration 001 applied)."""
    async with aiosqlite.connect(":memory:") as db:
        await db.executescript(_SEED_VERSION_1_SQL)
        await db.commit()
        yield db


# ---------------------------------------------------------------------------
# 1. schema_version meta table
# ---------------------------------------------------------------------------


class TestSchemaVersionTable:
    """The schema_version table must be created by init_db and contain exactly
    one row holding the current migration version number."""

    async def test_table_exists_after_init(self, initialized_db):
        async with initialized_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None, "schema_version table must exist after init_db"

    async def test_has_single_row(self, initialized_db):
        async with initialized_db.execute(
            "SELECT COUNT(*) FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, "schema_version must contain exactly one row"

    async def test_version_is_integer(self, initialized_db):
        async with initialized_db.execute(
            "SELECT version FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert isinstance(row[0], int), f"version must be int, got {type(row[0])}"
        assert row[0] >= 0

    async def test_version_equals_migration_count(self, initialized_db):
        async with initialized_db.execute(
            "SELECT version FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == len(MIGRATIONS)


# ---------------------------------------------------------------------------
# 2. Existing DDL becomes migration 001
# ---------------------------------------------------------------------------


class TestMigration001:
    """The first migration must create the existing message_log table and
    index — exactly what the old init_db used to do."""

    async def test_creates_message_log_table(self, raw_db):
        await raw_db.executescript(MIGRATIONS[0])
        await raw_db.commit()

        async with raw_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_log'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None

    async def test_creates_index(self, raw_db):
        await raw_db.executescript(MIGRATIONS[0])
        await raw_db.commit()

        async with raw_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_log_channel'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None

    async def test_expected_columns(self, raw_db):
        await raw_db.executescript(MIGRATIONS[0])
        await raw_db.commit()

        async with raw_db.execute("PRAGMA table_info(message_log)") as cursor:
            columns = await cursor.fetchall()

        col_names = {col[1] for col in columns}
        expected = {"id", "channel_id", "message", "timestamp", "message_type"}
        assert expected.issubset(col_names), f"Missing columns: {expected - col_names}"


# ---------------------------------------------------------------------------
# 3. corvidae.migrations module
# ---------------------------------------------------------------------------


class TestMigrationsModule:
    """The migrations module must expose a list of numbered migration scripts."""

    def test_module_importable(self):
        import corvidae.migrations  # noqa: F401

    def test_migrations_is_list_of_strings(self):
        assert isinstance(MIGRATIONS, (list, tuple))
        for i, m in enumerate(MIGRATIONS):
            assert isinstance(m, str), f"MIGRATIONS[{i}] must be a string, got {type(m)}"
            assert m.strip(), f"MIGRATIONS[{i}] must not be empty"

    def test_at_least_one_migration_exists(self):
        assert len(MIGRATIONS) >= 1

    def test_migrations_indexable_by_position(self):
        for i in range(len(MIGRATIONS)):
            assert MIGRATIONS[i] is not None


# ---------------------------------------------------------------------------
# 4. init_db applies pending migrations
# ---------------------------------------------------------------------------


class TestInitDbMigrations:
    """init_db must check schema_version, apply pending migrations, and update
    the version number."""

    async def test_fresh_db_runs_all_migrations(self, initialized_db):
        async with initialized_db.execute(
            "SELECT version FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == len(MIGRATIONS)

    async def test_idempotent_reinit(self, initialized_db):
        async with initialized_db.execute(
            "SELECT COUNT(*) FROM message_log"
        ) as cursor:
            count_before = (await cursor.fetchone())[0]

        await init_db(initialized_db)

        async with initialized_db.execute(
            "SELECT version FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == len(MIGRATIONS), "Version must not increase on re-init"

        async with initialized_db.execute(
            "SELECT COUNT(*) FROM message_log"
        ) as cursor:
            count_after = (await cursor.fetchone())[0]
        assert count_after == count_before, "Tables must not be duplicated"

    async def test_applies_only_new_migrations(self, version1_db):
        await init_db(version1_db)

        async with version1_db.execute(
            "SELECT version FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == len(MIGRATIONS)

    async def test_existing_db_without_schema_version(self):
        """An existing database that has message_log but no schema_version
        must be handled gracefully — treated as version 0."""
        async with aiosqlite.connect(":memory:") as db:
            # Simulate an old database: message_log exists, no schema_version
            await db.execute(
                """CREATE TABLE message_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    message_type TEXT NOT NULL DEFAULT 'message'
                )"""
            )
            await db.execute(
                "CREATE INDEX idx_log_channel ON message_log (channel_id, timestamp)"
            )
            await db.commit()

            await init_db(db)

            async with db.execute(
                "SELECT version FROM schema_version"
            ) as cursor:
                row = await cursor.fetchone()
            assert row[0] == len(MIGRATIONS)


# ---------------------------------------------------------------------------
# 5. Transactional safety
# ---------------------------------------------------------------------------


class TestMigrationTransactions:
    """Each migration must run inside a transaction so a partial failure is
    rolled back cleanly."""

    async def test_failed_migration_does_not_corrupt_version(self, version1_db):
        if len(MIGRATIONS) <= 1:
            pytest.skip("Needs at least 2 migrations to test failure scenario")

        original_m2 = MIGRATIONS[1]
        MIGRATIONS[1] = "INVALID SQL THAT WILL FAIL;"

        try:
            await init_db(version1_db)
        except Exception:
            pass  # Expected to fail

        async with version1_db.execute(
            "SELECT version FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1, f"Version must remain 1 after failed migration, got {row[0]}"

        MIGRATIONS[1] = original_m2


# ---------------------------------------------------------------------------
# 6. get_schema_version helper
# ---------------------------------------------------------------------------


class TestGetSchemaVersion:
    """get_schema_version reads the current schema version from the DB."""

    async def test_returns_zero_on_fresh_db(self, raw_db):
        assert await get_schema_version(raw_db) == 0

    async def test_returns_version_after_init(self, initialized_db):
        assert await get_schema_version(initialized_db) == len(MIGRATIONS)

    async def test_reads_explicit_value(self, raw_db):
        await raw_db.execute(
            "CREATE TABLE schema_version (version INTEGER NOT NULL)"
        )
        await raw_db.execute("INSERT INTO schema_version (version) VALUES (42)")
        await raw_db.commit()
        assert await get_schema_version(raw_db) == 42


# ---------------------------------------------------------------------------
# 7. Logging
# ---------------------------------------------------------------------------


class TestMigrationLogging:
    """Migration activity must be logged at INFO level."""

    async def test_logs_applied_migrations(self, caplog):
        import logging

        async with aiosqlite.connect(":memory:") as db:
            with caplog.at_level(logging.INFO, logger="corvidae.persistence"):
                await init_db(db)

        migration_logs = [
            r for r in caplog.records
            if r.name == "corvidae.persistence"
            and ("migration" in r.getMessage().lower()
                 or "schema" in r.getMessage().lower())
        ]
        assert migration_logs, "init_db must log at least one message about migrations/schema"

    async def test_no_apply_log_on_reinit(self, initialized_db, caplog):
        import logging

        caplog.clear()

        with caplog.at_level(logging.INFO, logger="corvidae.persistence"):
            await init_db(initialized_db)

        apply_logs = [
            r for r in caplog.records
            if r.name == "corvidae.persistence"
            and "applied migration" in r.getMessage().lower()
        ]
        assert len(apply_logs) == 0, (
            "init_db must not log 'Applied migration' when no migrations are pending"
        )


# ---------------------------------------------------------------------------
# 8. Extensibility: adding a new migration
# ---------------------------------------------------------------------------


class TestExtensibility:
    """Verify the migration system is designed for easy extension."""

    def test_migrations_list_is_mutable(self):
        assert isinstance(MIGRATIONS, list), (
            "MIGRATIONS must be a list to support appending new migrations"
        )

    async def test_hypothetical_second_migration(self, version1_db):
        if len(MIGRATIONS) < 2:
            pytest.skip("Needs at least 2 migrations to test")

        await init_db(version1_db)

        async with version1_db.execute(
            "SELECT version FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == len(MIGRATIONS)
