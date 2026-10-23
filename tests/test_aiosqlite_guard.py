"""Tests for the aiosqlite leak guard (GitHub issue #64).

aiosqlite runs each connection on a non-daemon worker thread. pytest stores
each failure's traceback in sys.last_traceback and clears it only when the
next test starts, so if the last test to run fails with a connection still
open, that connection is never collected and the interpreter waits forever
for its thread at exit (with -x, -k, or a single file this is common).
Exit-time hangs can only be observed from outside, so these tests run pytest
in a subprocess on a probe directory that uses the guard fixture.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The probe: a test that closes its own connection (the guard must not
# double-close), a sync test (the async autouse fixture must not break sync
# tests), and last, a failing test that leaks a connection. It must run last:
# a later test would clear sys.last_traceback and let GC stop the thread.
PROBE_TESTS = '''
import aiosqlite


async def test_closes_its_own_connection():
    db = await aiosqlite.connect(":memory:")
    await db.execute("SELECT 1")
    await db.close()


def test_sync():
    assert True


async def test_fails_with_open_connection():
    db = await aiosqlite.connect(":memory:")
    assert False, "fails before db.close()"
    await db.close()
'''

PROBE_CONFTEST = "from aiosqlite_guard import close_leaked_aiosqlite_connections  # noqa: F401\n"


def test_failing_test_with_open_connection_does_not_hang(tmp_path):
    """pytest exits promptly after a failure that left a connection open."""
    (tmp_path / "conftest.py").write_text(PROBE_CONFTEST)
    (tmp_path / "test_probe.py").write_text(PROBE_TESTS)

    # Run the probe with the repo's pytest config (asyncio auto mode,
    # pythonpath) so the guard module is importable.
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                "-c", str(REPO / "pyproject.toml"), "--rootdir", str(REPO),
                str(tmp_path),
            ],
            capture_output=True, text=True, timeout=10, cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("pytest hung at exit after a failing test leaked an aiosqlite connection")

    assert "1 failed, 2 passed" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 1
