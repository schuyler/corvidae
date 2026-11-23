"""outcome_log as the sink of ``on_trace_record`` (#1179, R4, R6).

outcome_log appends one JSONL line per record to ``<db stem>.outcomes.jsonl``
beside the session DB, writes its own admitted/rejected/persisted records
the same way, answers ``resolve_trace_origin`` from a bounded in-memory
index, and no longer creates or writes ``exchange_log``.
"""

import logging

import aiosqlite
import pytest

from corvidae.channel import Channel, ChannelRegistry
from corvidae.hooks import create_plugin_manager
from tests.cognition.trace_helpers import add_trace, outcomes_file, read_outcomes

CH = Channel(transport="irc", scope="#a")


async def _setup(db, *, registry: ChannelRegistry | None = None):
    """outcome_log on a pm with persistence (on ``db``) and the trace plugin."""
    from corvidae_cognition.outcome_log import OutcomeLogPlugin

    from corvidae.persistence import PersistencePlugin

    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    add_trace(pm)
    if registry is not None:
        pm.register(registry, name="registry")
    plugin = OutcomeLogPlugin()
    pm.register(plugin, name="outcome_log")
    await plugin.on_init(pm, {})
    await plugin.on_start({})
    return plugin


@pytest.fixture
async def file_db(tmp_path):
    """A file-backed session DB, so outcome_log has a place to write."""
    from corvidae.persistence import init_db

    path = tmp_path / "sessions.db"
    async with aiosqlite.connect(str(path)) as conn:
        await init_db(conn)
        yield conn, path


async def _fire(pm, trace_id="t1", channel_id=CH.id, source="gates",
                origin=None, record=None):
    # Fire the hook the way a reporter does.
    await pm.ahook.on_trace_record(
        trace_id=trace_id, channel_id=channel_id, source=source,
        origin=origin, record=record if record is not None else {"send": {"vetoed": True}},
    )


class TestRecords:
    async def test_one_line_per_record_beside_the_db(self, file_db):
        db, path = file_db
        plugin = await _setup(db)
        await _fire(plugin.pm, record={"send": {"vetoed": True}})
        await _fire(plugin.pm, record={"suppressed": {"progress": True}})

        lines = read_outcomes(outcomes_file(path))
        assert [line["record"] for line in lines] == [
            {"send": {"vetoed": True}}, {"suppressed": {"progress": True}},
        ]
        first = lines[0]
        assert set(first) == {"ts", "trace_id", "channel_id", "source", "origin", "record"}
        assert (first["trace_id"], first["channel_id"], first["source"], first["origin"]) == (
            "t1", "irc:#a", "gates", None,
        )
        assert first["ts"] > 0

    async def test_path_follows_the_db_file(self, tmp_path):
        # The file sits beside whatever file the persistence connection
        # opened, so redact (which only has the DB) can find it.
        from corvidae.persistence import init_db

        sub = tmp_path / "sub"
        sub.mkdir()
        async with aiosqlite.connect(str(sub / "other.db")) as db:
            await init_db(db)
            plugin = await _setup(db)
            await _fire(plugin.pm)
        assert len(read_outcomes(sub / "other.outcomes.jsonl")) == 1

    async def test_no_exchange_log_table_is_created(self, file_db):
        db, _ = file_db
        plugin = await _setup(db)
        await plugin.on_message_admitted(channel=CH, trace_id="t1", sender="u", text="hi")
        await _fire(plugin.pm)
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE name = 'exchange_log'"
        ) as cursor:
            assert await cursor.fetchone() is None

    def test_writer_api_is_gone(self):
        from corvidae_cognition.outcome_log import OutcomeLogPlugin

        for name in ("record_exchange", "update_exchange", "upsert_exchange"):
            assert not hasattr(OutcomeLogPlugin, name), name


class TestOwnRecords:
    async def test_admitted_then_persisted(self, file_db):
        db, path = file_db
        plugin = await _setup(db)
        await plugin.on_message_admitted(channel=CH, trace_id="t1", sender="u", text="hi")
        await plugin.on_message_persisted(
            channel=CH, trace_id="t1", rowid=7, text="hi", annotations={},
        )
        lines = read_outcomes(outcomes_file(path))
        assert [(line["source"], line["origin"], line["record"]) for line in lines] == [
            ("outcome_log", "user", {"event": "admitted"}),
            # First origin wins: a user trace's persisted line says "user".
            ("outcome_log", "user", {"event": "persisted", "message_rowid": 7}),
        ]

    async def test_persisted_without_admission_takes_annotation_origin(self, file_db):
        db, path = file_db
        plugin = await _setup(db)
        await plugin.on_message_persisted(
            channel=CH, trace_id="hb", rowid=3, text="beat",
            annotations={"origin": "heartbeat"},
        )
        await plugin.on_message_persisted(
            channel=CH, trace_id="tk", rowid=None, text="note", annotations={},
        )
        lines = {line["trace_id"]: line for line in read_outcomes(outcomes_file(path))}
        assert lines["hb"]["origin"] == "heartbeat"
        assert lines["tk"]["origin"] == "task"
        assert lines["tk"]["record"] == {"event": "persisted", "message_rowid": None}

    async def test_rejected(self, file_db):
        db, path = file_db
        plugin = await _setup(db)
        await plugin.on_message_rejected(channel=CH, trace_id="t1", sender="u", text="no")
        (line,) = read_outcomes(outcomes_file(path))
        assert (line["origin"], line["record"]) == ("user", {"event": "rejected"})


class TestOriginIndex:
    async def test_resolves_first_origin(self, file_db):
        db, _ = file_db
        plugin = await _setup(db)
        pm = plugin.pm
        await plugin.on_message_admitted(channel=CH, trace_id="t1", sender="u", text="hi")
        await plugin.on_message_persisted(
            channel=CH, trace_id="t1", rowid=1, text="hi",
            annotations={"origin": "heartbeat"},
        )
        await _fire(pm, trace_id="t2", origin="reminder")
        assert await pm.ahook.resolve_trace_origin(trace_id="t1") == "user"
        assert await pm.ahook.resolve_trace_origin(trace_id="t2") == "reminder"
        assert await pm.ahook.resolve_trace_origin(trace_id="nope") is None

    async def test_index_is_bounded(self, file_db, monkeypatch):
        from corvidae_cognition import outcome_log

        monkeypatch.setattr(outcome_log, "ORIGIN_INDEX_MAX", 2)
        db, _ = file_db
        plugin = await _setup(db)
        for trace_id in ("a", "b", "c"):
            await plugin.on_message_admitted(channel=CH, trace_id=trace_id, sender="u", text="x")
        pm = plugin.pm
        assert await pm.ahook.resolve_trace_origin(trace_id="a") is None
        assert await pm.ahook.resolve_trace_origin(trace_id="c") == "user"


class TestUnregisteredChannel:
    """#261: nothing is recorded for a channel the registry never held."""

    async def test_refused_stranger_and_reporters_leave_no_line(self, file_db):
        db, path = file_db
        plugin = await _setup(db, registry=ChannelRegistry({}))
        stranger = Channel(transport="irc", scope="mallory")
        await plugin.on_message_rejected(channel=stranger, trace_id="t1", sender="m", text="hi")
        await _fire(plugin.pm, channel_id=stranger.id)
        assert read_outcomes(outcomes_file(path)) == []

    async def test_known_channel_is_recorded(self, file_db):
        db, path = file_db
        registry = ChannelRegistry({})
        plugin = await _setup(db, registry=registry)
        known = registry.get_or_create("irc", "alice")
        await plugin.on_message_rejected(channel=known, trace_id="t1", sender="a", text="ok")
        assert len(read_outcomes(outcomes_file(path))) == 1


class TestNowhereToWrite:
    async def test_in_memory_db_writes_nothing_and_warns_once(self, db, tmp_path, monkeypatch, caplog):
        # An in-memory session DB has no file to sit beside: nothing is
        # written (not even into the working directory) and the operator is
        # told once, naming the config key.
        monkeypatch.chdir(tmp_path)
        plugin = await _setup(db)
        with caplog.at_level(logging.WARNING, logger="corvidae.outcome_log"):
            await _fire(plugin.pm, trace_id="t1")
            await _fire(plugin.pm, trace_id="t2")
        warnings = [r for r in caplog.records if r.name == "corvidae.outcome_log"]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert "daemon.session_db" in warnings[0].getMessage()
        assert list(tmp_path.iterdir()) == []

    async def test_write_failure_warns_naming_path_trace_and_remedy(self, file_db, caplog):
        # A failed append is logged with the file, the trace and what to
        # check; the reporter is not affected.
        db, path = file_db
        target = outcomes_file(path)
        target.mkdir()  # appending to a directory fails with an OSError
        plugin = await _setup(db)
        with caplog.at_level(logging.WARNING, logger="corvidae.outcome_log"):
            await _fire(plugin.pm, trace_id="t-fail", source="silence")
        messages = [r.getMessage() for r in caplog.records if r.name == "corvidae.outcome_log"]
        assert len(messages) == 1, messages
        assert str(target) in messages[0]
        assert "t-fail" in messages[0] and "silence" in messages[0]
        assert "permission" in messages[0] or "disk" in messages[0]

