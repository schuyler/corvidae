"""Tests for the scheduler CRUD operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from corvidae.scheduler_crud import add_job, list_jobs, remove_job, update_job
from corvidae.scheduler_file import read_jobs, write_jobs


@pytest.fixture
def jobs_file(tmp_path: Path) -> Path:
    """A jobs file with one pre-existing job."""
    f = tmp_path / "jobs.yaml"
    write_jobs(f, [
        {"name": "existing", "channel": "signal:123", "cron": "0 8 * * *",
         "origin": "reminder", "prompt": "Existing job"},
    ])
    return f


# --- add_job ---

def test_add_job(jobs_file: Path):
    result = add_job(jobs_file, "new-job", "signal:456", "0 9 * * 1", "Monday morning")
    assert "Added job" in result
    jobs = read_jobs(jobs_file)
    assert len(jobs) == 2
    assert jobs[1]["name"] == "new-job"
    assert jobs[1]["cron"] == "0 9 * * 1"


def test_add_job_duplicate_name(jobs_file: Path):
    result = add_job(jobs_file, "existing", "signal:456", "0 9 * * *", "Dup")
    assert "already exists" in result
    jobs = read_jobs(jobs_file)
    assert len(jobs) == 1


def test_add_job_invalid_cron(jobs_file: Path):
    result = add_job(jobs_file, "bad", "signal:123", "not a cron", "Test")
    assert "Could not add" in result
    assert "invalid cron" in result


def test_add_job_missing_channel(jobs_file: Path):
    result = add_job(jobs_file, "nochan", "", "0 8 * * *", "Test")
    assert "Could not add" in result
    assert "channel" in result


def test_add_job_bad_origin(jobs_file: Path):
    result = add_job(jobs_file, "badorigin", "signal:123", "0 8 * * *",
                     "Test", origin="invalid")
    assert "Could not add" in result
    assert "origin" in result


def test_add_job_no_prompt(jobs_file: Path):
    result = add_job(jobs_file, "noprompt", "signal:123", "0 8 * * *", "")
    assert "Could not add" in result
    assert "prompt" in result


def test_add_job_channel_needs_colon(jobs_file: Path):
    result = add_job(jobs_file, "nocolon", "signal123", "0 8 * * *", "Test")
    assert "Could not add" in result
    assert "transport:scope" in result


# --- update_job ---

def test_update_job_cron(jobs_file: Path):
    result = update_job(jobs_file, "existing", cron="0 10 * * *")
    assert "Updated job" in result
    jobs = read_jobs(jobs_file)
    assert jobs[0]["cron"] == "0 10 * * *"


def test_update_job_prompt(jobs_file: Path):
    result = update_job(jobs_file, "existing", prompt="New prompt")
    assert "Updated job" in result
    jobs = read_jobs(jobs_file)
    assert jobs[0]["prompt"] == "New prompt"


def test_update_job_missing(jobs_file: Path):
    result = update_job(jobs_file, "nonexistent", cron="0 10 * * *")
    assert "Could not update" in result
    assert "no job named" in result


def test_update_job_invalid_field(jobs_file: Path):
    result = update_job(jobs_file, "existing", color="red")
    assert "Could not update" in result
    assert "not an updatable field" in result


def test_update_job_bad_cron(jobs_file: Path):
    result = update_job(jobs_file, "existing", cron="invalid")
    assert "Could not update" in result
    assert "invalid cron" in result


# --- remove_job ---

def test_remove_job(jobs_file: Path):
    result = remove_job(jobs_file, "existing")
    assert "Removed job" in result
    jobs = read_jobs(jobs_file)
    assert len(jobs) == 0


def test_remove_job_missing(jobs_file: Path):
    result = remove_job(jobs_file, "nonexistent")
    assert "Could not remove" in result
    assert "no job named" in result


# --- list_jobs ---

def test_list_jobs_empty(tmp_path: Path):
    f = tmp_path / "empty.yaml"
    result = list_jobs(f)
    assert "No jobs" in result


def test_list_jobs_with_jobs(jobs_file: Path):
    result = list_jobs(jobs_file)
    assert "1 job(s)" in result
    assert "existing" in result
    assert "0 8 * * *" in result
    assert "signal:123" in result


def test_list_jobs_multiple(jobs_file: Path):
    add_job(jobs_file, "second", "irc:#general", "*/15 * * * *", "Fifteen minutes")
    result = list_jobs(jobs_file)
    assert "2 job(s)" in result
    assert "existing" in result
    assert "second" in result


def test_list_jobs_truncates_long_prompt(jobs_file: Path):
    update_job(jobs_file, "existing", prompt="x" * 100)
    result = list_jobs(jobs_file)
    assert "..." in result
    assert "x" * 100 not in result
