"""Tests for the #350 usage_log columns: call timing, attempts, backend split, context counts."""

import logging

import aiosqlite
import pytest

from corvidae.channel import ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.metrics import UsageLogPlugin
from corvidae.persistence import PersistencePlugin

USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
ATTRIBUTION = {"stage": "turn", "channel_id": "irc:#general"}

# The usage_log table as it stood before #350: the current DDL minus the nine new columns.
PRE_350_DDL = """
CREATE TABLE usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    model TEXT NOT NULL,
    stage TEXT,
    channel_id TEXT,
    trace_id TEXT,
    trigger TEXT,
    mode TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER,
    cache_n INTEGER,
    prompt_n INTEGER,
    latency_ms REAL,
    error TEXT
)
"""


async def _fire_timed_response(pm, *, request_id="req1", started_at=1000.0,
                               total_ms=50.0, attempts=1, message_count=2,
                               tool_count=1, usage=USAGE, error=None,
                               attribution=ATTRIBUTION, timings=None,
                               latency_ms=42.0, profile="main", model="m1"):
    """Dispatch on_llm_response with every #350 argument, the way LLMClient's observer does."""
    await pm.ahook.on_llm_response(
        profile=profile,
        model=model,
        request_id=request_id,
        usage=usage,
        latency_ms=latency_ms,
        attribution=attribution,
        error=error,
        timings=timings,
        started_at=started_at,
        total_ms=total_ms,
        attempts=attempts,
        message_count=message_count,
        tool_count=tool_count,
    )


async def _setup(db, *, registry=None):
    """Plugin manager with a persistence DB, an optional registry, and UsageLogPlugin.

    on_start is not called, so the usage_log table is created or upgraded by the first write.
    """
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    if registry is not None:
        pm.register(registry, name="registry")
    plugin = UsageLogPlugin()
    pm.register(plugin, name="usage_log")
    await plugin.on_init(pm, {})
    return pm


async def _fetch(db, request_id, columns):
    """Return the named columns of the usage_log row for request_id, as a dict."""
    async with db.execute(
        f"SELECT {columns} FROM usage_log WHERE request_id = ?", (request_id,)
    ) as cursor:
        names = [d[0] for d in cursor.description]
        row = await cursor.fetchone()
    assert row is not None, f"no usage_log row for request_id {request_id!r}"
    return dict(zip(names, row))


class TestTimingColumns:
    """A5: the row stores the timing, attempt and count values it was passed."""

    async def test_row_stores_timing_attempt_and_count_columns(self, db):
        pm = await _setup(db)
        await _fire_timed_response(
            pm, request_id="r5", started_at=1700000000.25, total_ms=350.5,
            attempts=3, message_count=7, tool_count=4,
        )

        row = await _fetch(
            db, "r5", "started_at, ended_at, total_ms, attempts, message_count, tool_count"
        )

        assert row["started_at"] == 1700000000.25
        assert row["total_ms"] == 350.5
        assert row["attempts"] == 3
        assert row["message_count"] == 7
        assert row["tool_count"] == 4
        # Absolute tolerance: the default relative one is ~1700 s at epoch
        # magnitudes and would accept ended_at == started_at.
        assert row["ended_at"] == pytest.approx(
            1700000000.25 + 350.5 / 1000, abs=1e-6
        )


class TestBackendTimingSplit:
    """A6: prompt_ms and predicted_ms come from timings; NULL when absent, 0 stays 0."""

    @pytest.mark.parametrize(
        "timings, expected",
        [
            ({"prompt_ms": 12.5, "predicted_ms": 80.0}, (12.5, 80.0)),
            (None, (None, None)),
            ({"prompt_ms": 0, "predicted_ms": 0.0}, (0, 0.0)),
            ({"cache_n": 5, "prompt_n": 5}, (None, None)),
        ],
        ids=["present", "timings-none", "reported-zero", "keys-missing"],
    )
    async def test_prompt_and_predicted_ms_from_timings(self, db, timings, expected):
        pm = await _setup(db)
        await _fire_timed_response(pm, request_id="r6", timings=timings)

        row = await _fetch(db, "r6", "prompt_ms, predicted_ms")

        # Exact tuple equality: a reported zero must stay a stored zero, not
        # collapse to NULL (None != 0).
        assert (row["prompt_ms"], row["predicted_ms"]) == expected


class TestMaxContextTokens:
    """A7: max_context_tokens is the channel's resolved budget, or NULL with a warning on failure."""

    async def test_follows_channel_override_not_only_default(self, db):
        registry = ChannelRegistry({"max_context_tokens": 24000})
        registry.get_or_create(
            "irc", "#general", config=ChannelConfig(max_context_tokens=5000)
        )
        pm = await _setup(db, registry=registry)
        await _fire_timed_response(pm, request_id="r7a", attribution={"channel_id": "irc:#general"})

        row = await _fetch(db, "r7a", "max_context_tokens")

        assert row["max_context_tokens"] == 5000

    async def test_no_channel_id_stores_null(self, db):
        registry = ChannelRegistry({"max_context_tokens": 24000})
        pm = await _setup(db, registry=registry)
        await _fire_timed_response(pm, request_id="r7b", attribution={})

        row = await _fetch(db, "r7b", "max_context_tokens")

        assert row["max_context_tokens"] is None

    async def test_unknown_channel_stores_null(self, db):
        registry = ChannelRegistry({"max_context_tokens": 24000})
        pm = await _setup(db, registry=registry)
        await _fire_timed_response(pm, request_id="r7c", attribution={"channel_id": "irc:#ghost"})

        row = await _fetch(db, "r7c", "max_context_tokens")

        assert row["max_context_tokens"] is None

    async def test_no_registry_plugin_stores_null(self, db):
        pm = await _setup(db, registry=None)
        await _fire_timed_response(pm, request_id="r7d", attribution={"channel_id": "irc:#general"})

        row = await _fetch(db, "r7d", "max_context_tokens")

        assert row["max_context_tokens"] is None

    async def test_failed_lookup_writes_row_with_null_and_warns(self, db, caplog):
        registry = ChannelRegistry({"max_context_tokens": 24000})
        registry.get_or_create("irc", "#boom")

        def _raise(channel):
            """Stand-in for a resolve_config that fails for this channel."""
            raise RuntimeError("resolve exploded")

        registry.resolve_config = _raise
        pm = await _setup(db, registry=registry)

        with caplog.at_level(logging.WARNING):
            await _fire_timed_response(pm, request_id="r7e", attribution={"channel_id": "irc:#boom"})

        row = await _fetch(db, "r7e", "request_id, max_context_tokens")
        assert row["request_id"] == "r7e"
        assert row["max_context_tokens"] is None
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "irc:#boom" in r.getMessage()]
        assert warnings, "expected a WARNING naming channel irc:#boom"


class TestUsageLogMigration:
    """A8: a pre-#350 table is upgraded in place on first write, keeping its rows."""

    async def test_old_table_upgrades_in_place_and_matches_fresh_schema(self, db):
        await db.execute(PRE_350_DDL)
        await db.execute(
            "INSERT INTO usage_log (ts, request_id, profile, model, latency_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (1.0, "old", "main", "m1", 10.0),
        )
        await db.commit()
        pm = await _setup(db)

        await _fire_timed_response(pm, request_id="new")

        old = await _fetch(
            db, "old",
            "request_id, latency_ms, started_at, ended_at, total_ms, attempts, "
            "message_count, tool_count, max_context_tokens, prompt_ms, predicted_ms",
        )
        assert old["request_id"] == "old"
        assert old["latency_ms"] == 10.0
        for column in ("started_at", "ended_at", "total_ms", "attempts", "message_count",
                       "tool_count", "max_context_tokens", "prompt_ms", "predicted_ms"):
            assert old[column] is None, f"old row column {column} should be NULL"

        async with db.execute("PRAGMA table_info(usage_log)") as cursor:
            migrated = {row[1] for row in await cursor.fetchall()}

        # A table created fresh by the same plugin, for the column-set comparison.
        async with aiosqlite.connect(":memory:") as fresh_db:
            fresh_pm = await _setup(fresh_db)
            await _fire_timed_response(fresh_pm, request_id="fresh")
            async with fresh_db.execute("PRAGMA table_info(usage_log)") as cursor:
                fresh = {row[1] for row in await cursor.fetchall()}

        assert migrated == fresh


class TestOverlapSelfJoin:
    """A9: rows for overlapping calls are found by a self-join on started_at/ended_at."""

    async def test_overlapping_calls_join_and_disjoint_call_does_not(self, db):
        pm = await _setup(db)
        # a [1000.0, 1000.5] overlaps b [1000.2, 1001.0]; c [1002.0, 1002.1] overlaps neither.
        await _fire_timed_response(pm, request_id="a", started_at=1000.0, total_ms=500.0)
        await _fire_timed_response(pm, request_id="b", started_at=1000.2, total_ms=800.0)
        await _fire_timed_response(pm, request_id="c", started_at=1002.0, total_ms=100.0)

        async with db.execute(
            "SELECT x.request_id, y.request_id FROM usage_log x "
            "JOIN usage_log y ON x.id < y.id "
            "AND x.started_at < y.ended_at AND y.started_at < x.ended_at "
            "ORDER BY x.id, y.id"
        ) as cursor:
            pairs = await cursor.fetchall()

        assert pairs == [("a", "b")]
