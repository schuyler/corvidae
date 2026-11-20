"""Tests for `corvidae goals repair-statuses` (#205).

Pre-migration rows carry lying statuses: retired_at set but status left at
its DDL default 'open' (147 such rows in production on 2026-11-19). The
beat list filters on retired_at so the board is right, but direct SQL
queries against `status` mislead. This command reconciles status with the
retired_at partition — dry-run by default, --apply to write.
"""

import time

import pytest
from click.testing import CliRunner
from corvidae_cognition.goals import GOALS_DDL, goals_command

pytest.importorskip("aiosqlite")
import sqlite3


def _make_db(tmp_path):
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=wal")
    conn.execute(GOALS_DDL)
    now = time.time()
    rows = [
        # (text, retired_at, status) — the lying production shapes:
        ("honest open", None, "open"),            # untouched
        ("honest blocked", None, "blocked"),      # untouched
        ("lying retired", now - 100, "open"),     # -> retired
        ("also lying", now - 50, "waiting"),      # -> retired
        ("reverse break", None, "retired"),       # -> open
        ("honest retired", now - 10, "retired"),  # untouched
    ]
    conn.executemany(
        "INSERT INTO goals (text, created_at, updated_at, retired_at, status, position)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [(t, now, now, r, s, i) for i, (t, r, s) in enumerate(rows)],
    )
    conn.commit()
    conn.close()
    return db


def _statuses(db):
    conn = sqlite3.connect(db)
    out = dict(
        conn.execute(
            "SELECT text, status FROM goals"
        ).fetchall()
    )
    conn.close()
    return out


def test_dry_run_reports_and_does_not_write(tmp_path):
    db = _make_db(tmp_path)
    before = _statuses(db)
    result = CliRunner().invoke(
        goals_command, ["--db", str(db), "repair-statuses"], catch_exceptions=False
    )
    assert result.exit_code == 0
    # Reports the mismatch counts it found, naming both directions.
    assert "2" in result.output and "1" in result.output
    assert _statuses(db) == before  # untouched without --apply


def test_apply_reconciles_both_directions(tmp_path):
    db = _make_db(tmp_path)
    result = CliRunner().invoke(
        goals_command, ["--db", str(db), "repair-statuses", "--apply"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    after = _statuses(db)
    assert after["lying retired"] == "retired"
    assert after["also lying"] == "retired"
    assert after["reverse break"] == "open"
    # honest rows untouched
    assert after["honest open"] == "open"
    assert after["honest blocked"] == "blocked"
    assert after["honest retired"] == "retired"


def test_apply_is_idempotent_and_verifies(tmp_path):
    db = _make_db(tmp_path)
    args = ["--db", str(db), "repair-statuses", "--apply"]
    CliRunner().invoke(goals_command, args, catch_exceptions=False)
    second = CliRunner().invoke(
        goals_command, args, catch_exceptions=False
    )
    assert second.exit_code == 0
    assert "0 mismatches" in second.output
    conn = sqlite3.connect(db)
    mismatch = conn.execute(
        "SELECT COUNT(*) FROM goals WHERE "
        "(retired_at IS NOT NULL AND status != 'retired') OR "
        "(retired_at IS NULL AND status = 'retired')"
    ).fetchone()[0]
    conn.close()
    assert mismatch == 0


def test_requires_existing_db(tmp_path):
    result = CliRunner().invoke(
        goals_command, ["--db", str(tmp_path / "nope.db"), "repair-statuses"]
    )
    assert result.exit_code != 0
