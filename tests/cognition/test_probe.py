"""ReadOnlyProbe: lazy, rate-limited read-only connection (#159, #237)."""

import asyncio
import logging
import sqlite3

import aiosqlite
import pytest

pytest.importorskip("corvidae_cognition")

from corvidae_cognition import probe as probe_mod  # noqa: E402
from corvidae_cognition.probe import PROBE_RETRY_SECONDS, ReadOnlyProbe  # noqa: E402

CONSEQUENCE = "the widget stays off until things_fts exists"


class FakeClock:
    """A manually stepped clock, starting at 0 on purpose."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _sync_exec(path, *statements: str) -> None:
    """Run statements against the file synchronously (bypasses the
    aiosqlite.connect counter)."""
    db = sqlite3.connect(path)
    for stmt in statements:
        db.execute(stmt)
    db.commit()
    db.close()


async def _make_db(path, with_table: bool) -> None:
    """Create a SQLite file, optionally with the required table."""
    _sync_exec(path, "CREATE TABLE other (x)")
    if with_table:
        _sync_exec(path, "CREATE TABLE things_fts (x)")


def _probe(path, clock=None) -> ReadOnlyProbe:
    return ReadOnlyProbe(
        str(path), "things_fts", "widget", CONSEQUENCE,
        clock=clock or FakeClock(),
    )


@pytest.fixture
def counting_connect(monkeypatch):
    """Count aiosqlite.connect calls made by the probe module."""
    real = aiosqlite.connect
    calls = {"n": 0, "conns": []}

    def counting(*args, **kwargs):
        calls["n"] += 1
        conn = real(*args, **kwargs)
        calls["conns"].append(conn)
        return conn

    monkeypatch.setattr(probe_mod.aiosqlite, "connect", counting)
    return calls


async def test_construction_opens_nothing(tmp_path, counting_connect):
    p = _probe(tmp_path / "s.db")
    assert p.connection is None
    assert counting_connect["n"] == 0


async def test_first_get_opens_and_caches(tmp_path, counting_connect):
    path = tmp_path / "s.db"
    await _make_db(path, True)
    p = _probe(path)
    try:
        conn = await p.get()
        assert conn is not None
        assert p.connection is conn
        assert await p.get() is conn
        assert counting_connect["n"] == 1
    finally:
        await p.close()


async def test_table_appearing_later_is_picked_up_after_interval(tmp_path):
    path = tmp_path / "s.db"
    await _make_db(path, False)
    clock = FakeClock()
    p = _probe(path, clock)
    try:
        assert await p.get() is None
        await _make_db_add_table(path)
        # Still inside the interval: no new attempt.
        clock.now = PROBE_RETRY_SECONDS - 1
        assert await p.get() is None
        clock.now = PROBE_RETRY_SECONDS + 1
        assert await p.get() is not None
    finally:
        await p.close()


async def _make_db_add_table(path) -> None:
    _sync_exec(path, "CREATE TABLE things_fts (x)")


async def test_rate_limit_one_attempt_per_interval(tmp_path, counting_connect):
    path = tmp_path / "s.db"
    await _make_db(path, False)
    clock = FakeClock()
    p = _probe(path, clock)
    for _ in range(5):
        assert await p.get() is None
    assert counting_connect["n"] == 1
    clock.now = PROBE_RETRY_SECONDS
    assert await p.get() is None
    assert counting_connect["n"] == 2


async def test_missing_table_closes_half_open_connection(tmp_path, counting_connect):
    path = tmp_path / "s.db"
    await _make_db(path, False)
    p = _probe(path)
    assert await p.get() is None
    [conn] = counting_connect["conns"]
    assert conn._connection is None  # closed, not leaked


async def test_missing_table_warning_names_path_reason_consequence(tmp_path, caplog):
    path = tmp_path / "s.db"
    await _make_db(path, False)
    p = _probe(path)
    with caplog.at_level(logging.WARNING, logger="corvidae.probe"):
        await p.get()
    assert str(path) in caplog.text
    assert "things_fts missing" in caplog.text
    assert CONSEQUENCE in caplog.text
    assert "retrying automatically" in caplog.text


async def test_open_error_warning_names_path_and_exception(tmp_path, caplog):
    path = tmp_path / "nope" / "s.db"
    p = _probe(path)
    with caplog.at_level(logging.WARNING, logger="corvidae.probe"):
        assert await p.get() is None
    assert str(path) in caplog.text
    assert "OperationalError" in caplog.text
    assert CONSEQUENCE in caplog.text


async def test_recovery_logs_info_once(tmp_path, caplog):
    path = tmp_path / "s.db"
    await _make_db(path, False)
    clock = FakeClock()
    p = _probe(path, clock)
    try:
        assert await p.get() is None
        await _make_db_add_table(path)
        clock.now = PROBE_RETRY_SECONDS + 1
        with caplog.at_level(logging.INFO, logger="corvidae.probe"):
            assert await p.get() is not None
            assert await p.get() is not None
        recovered = [r for r in caplog.records if "recovered" in r.getMessage()]
        assert len(recovered) == 1
        assert str(path) in recovered[0].getMessage()
        assert "things_fts" in recovered[0].getMessage()
    finally:
        await p.close()


async def test_first_try_success_logs_no_recovery(tmp_path, caplog):
    path = tmp_path / "s.db"
    await _make_db(path, True)
    p = _probe(path)
    try:
        with caplog.at_level(logging.INFO, logger="corvidae.probe"):
            assert await p.get() is not None
        assert "recovered" not in caplog.text
    finally:
        await p.close()


async def test_concurrent_get_opens_one_connection(tmp_path, counting_connect):
    path = tmp_path / "s.db"
    await _make_db(path, True)
    p = _probe(path)
    try:
        results = await asyncio.gather(*(p.get() for _ in range(8)))
        assert all(r is results[0] and r is not None for r in results)
        assert counting_connect["n"] == 1
    finally:
        await p.close()


async def test_concurrent_failing_get_one_attempt_one_warning(
    tmp_path, counting_connect, caplog
):
    path = tmp_path / "s.db"
    await _make_db(path, False)
    p = _probe(path)
    with caplog.at_level(logging.WARNING, logger="corvidae.probe"):
        results = await asyncio.gather(*(p.get() for _ in range(8)))
    assert results == [None] * 8
    assert counting_connect["n"] == 1
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


async def test_cancellation_mid_open_closes_connection(tmp_path, monkeypatch):
    path = tmp_path / "s.db"
    await _make_db(path, True)
    p = _probe(path)
    opened: list[aiosqlite.Connection] = []
    in_query = asyncio.Event()
    real_connect = aiosqlite.connect

    async def slow_connect(*args, **kwargs):
        conn = await real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    real_execute = aiosqlite.Connection.execute

    def stalled_execute(self, *args, **kwargs):
        # Signal that the open completed, then hang the table check.
        in_query.set()
        return _Hang()

    class _Hang:
        async def __aenter__(self):
            await asyncio.sleep(3600)

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(probe_mod.aiosqlite, "connect", slow_connect)
    monkeypatch.setattr(aiosqlite.Connection, "execute", stalled_execute)
    task = asyncio.create_task(p.get())
    await asyncio.wait_for(in_query.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(aiosqlite.Connection, "execute", real_execute)
    assert p.connection is None
    [conn] = opened
    assert conn._connection is None  # closed on cancellation


async def test_get_after_close_returns_none_and_opens_nothing(
    tmp_path, counting_connect
):
    path = tmp_path / "s.db"
    await _make_db(path, True)
    p = _probe(path)
    conn = await p.get()
    await p.close()
    assert p.connection is None
    assert conn._connection is None
    assert await p.get() is None
    assert counting_connect["n"] == 1


async def test_close_before_get_blocks_open(tmp_path, counting_connect):
    path = tmp_path / "s.db"
    await _make_db(path, True)
    p = _probe(path)
    await p.close()
    assert await p.get() is None
    assert counting_connect["n"] == 0


async def test_close_racing_get_leaves_no_open_connection(tmp_path, counting_connect):
    path = tmp_path / "s.db"
    await _make_db(path, True)
    p = _probe(path)
    await asyncio.gather(p.get(), p.close())
    assert p.connection is None
    for conn in counting_connect["conns"]:
        assert conn._connection is None
