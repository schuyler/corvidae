"""Tests for the scheduler YAML file reader/writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corvidae.scheduler_file import read_jobs, write_jobs


def test_read_missing_file_returns_empty(tmp_path: Path):
    result = read_jobs(tmp_path / "nonexistent.yaml")
    assert result == []


def test_read_empty_file_returns_empty(tmp_path: Path):
    f = tmp_path / "jobs.yaml"
    f.write_text("")
    result = read_jobs(f)
    assert result == []


def test_read_whitespace_file_returns_empty(tmp_path: Path):
    f = tmp_path / "jobs.yaml"
    f.write_text("   \n  \n")
    result = read_jobs(f)
    assert result == []


def test_write_and_read_roundtrip(tmp_path: Path):
    f = tmp_path / "jobs.yaml"
    jobs = [
        {"name": "test-job", "channel": "signal:123", "cron": "0 8 * * *",
         "origin": "reminder", "prompt": "Hello"},
        {"name": "another", "channel": "irc:#general", "cron": "*/15 * * * *",
         "origin": "heartbeat", "prompt": "Tick"},
    ]
    write_jobs(f, jobs)
    result = read_jobs(f)
    assert result == jobs


def test_write_creates_parent_dirs(tmp_path: Path):
    f = tmp_path / "nested" / "dir" / "jobs.yaml"
    write_jobs(f, [{"name": "test", "channel": "x", "cron": "* * * * *"}])
    assert f.exists()


def test_write_is_atomic_no_partial(tmp_path: Path):
    f = tmp_path / "jobs.yaml"
    # Write initial content
    write_jobs(f, [{"name": "initial", "channel": "x", "cron": "* * * * *"}])
    # Verify it reads back
    assert read_jobs(f)[0]["name"] == "initial"
    # No temp files left behind
    leftovers = list(tmp_path.glob(".jobs-*.tmp"))
    assert leftovers == []


    """Our writer produces JSON; reading JSON should work."""
    f = tmp_path / "jobs.yaml"
    data = {"jobs": [{"name": "json-job", "channel": "x", "cron": "0 0 * * *"}]}
    f.write_text(json.dumps(data))
    result = read_jobs(f)
    assert len(result) == 1
    assert result[0]["name"] == "json-job"


def test_read_minimal_yaml_format(tmp_path: Path):
    """Hand-edited YAML (not JSON) should also parse."""
    f = tmp_path / "jobs.yaml"
    f.write_text(
        "jobs:\n"
        "  - name: yaml-job\n"
        "    channel: signal:123\n"
        "    cron: \"0 8 * * *\"\n"
        "    origin: reminder\n"
        "    prompt: Hello\n"
    )
    result = read_jobs(f)
    assert len(result) == 1
    assert result[0]["name"] == "yaml-job"
    assert result[0]["channel"] == "signal:123"
    assert result[0]["cron"] == "0 8 * * *"


def test_read_yaml_multiple_jobs(tmp_path: Path):
    f = tmp_path / "jobs.yaml"
    f.write_text(
        "jobs:\n"
        "  - name: job1\n"
        "    channel: a\n"
        "    cron: \"* * * * *\"\n"
        "  - name: job2\n"
        "    channel: b\n"
        "    cron: \"0 0 * * *\"\n"
    )
    result = read_jobs(f)
    assert len(result) == 2
    assert result[0]["name"] == "job1"
    assert result[1]["name"] == "job2"


def test_write_preserves_all_fields(tmp_path: Path):
    f = tmp_path / "jobs.yaml"
    job = {"name": "full", "channel": "signal:123", "cron": "0 8 * * 1",
           "origin": "heartbeat", "prompt": "Weekly review",
           "extra_field": "preserved"}
    write_jobs(f, [job])
    result = read_jobs(f)
    assert result[0] == job
