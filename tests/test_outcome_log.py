"""Tests for corvidae.outcome_log — the exchange_log table and writer API."""

import pytest

from corvidae.hooks import create_plugin_manager


async def _setup(db):
    from corvidae.outcome_log import OutcomeLogPlugin
    from corvidae.persistence import PersistencePlugin

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    plugin = OutcomeLogPlugin()
    pm.register(plugin, name="outcome_log")
    await plugin.on_init(pm, {})
    await plugin.on_start({})
    return plugin


class TestExchangeLogWriter:
    async def test_record_and_update_round_trip(self, db):
        plugin = await _setup(db)

        await plugin.record_exchange(
            "ex1", "irc:#general", origin="user", message_rowid=7
        )
        await plugin.update_exchange(
            "ex1", retrieval_top_score=0.91, retrieval_hit_count=3
        )

        async with db.execute(
            "SELECT channel_id, origin, message_rowid, retrieval_top_score, "
            "retrieval_hit_count, probe_score FROM exchange_log "
            "WHERE exchange_key = ?",
            ("ex1",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row == ("irc:#general", "user", 7, 0.91, 3, None)

    async def test_record_minimal_defaults(self, db):
        plugin = await _setup(db)
        await plugin.record_exchange("ex2", "cli:local")

        async with db.execute(
            "SELECT origin, message_rowid, created_at FROM exchange_log "
            "WHERE exchange_key = ?",
            ("ex2",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] is None
        assert row[1] is None
        assert row[2] > 0  # created_at is stamped

    async def test_duplicate_record_is_idempotent(self, db):
        plugin = await _setup(db)
        await plugin.record_exchange("ex3", "cli:local", origin="user")
        # Second insert with different values is ignored (INSERT OR IGNORE).
        await plugin.record_exchange("ex3", "other", origin="task")

        async with db.execute(
            "SELECT COUNT(*), MAX(channel_id) FROM exchange_log "
            "WHERE exchange_key = ?",
            ("ex3",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row == (1, "cli:local")

    async def test_update_unknown_column_rejected(self, db):
        plugin = await _setup(db)
        await plugin.record_exchange("ex4", "cli:local")

        with pytest.raises(ValueError):
            await plugin.update_exchange("ex4", not_a_column="x")

    async def test_update_immutable_column_rejected(self, db):
        # The key/channel/created_at identity columns are not updatable.
        plugin = await _setup(db)
        await plugin.record_exchange("ex5", "cli:local")

        with pytest.raises(ValueError):
            await plugin.update_exchange("ex5", channel_id="hijacked")

    async def test_update_json_columns(self, db):
        plugin = await _setup(db)
        await plugin.record_exchange("ex6", "cli:local")
        await plugin.update_exchange(
            "ex6", appraisal='{"valence": 0.2}', outcomes='{"verdict": "good"}'
        )

        async with db.execute(
            "SELECT appraisal, outcomes FROM exchange_log WHERE exchange_key = ?",
            ("ex6",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row == ('{"valence": 0.2}', '{"verdict": "good"}')
