"""CRUD operations for scheduler jobs in the YAML file.

Four operations: add, update, remove, list. Each reads the current
jobs, modifies the list, and writes back atomically. Returns a
human-readable confirmation or error string (the model sees these).

The caller (SchedulerPlugin) is responsible for triggering a reload
after a mutation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corvidae.cron import CronParseError, parse_cron
from corvidae.scheduler_file import read_jobs, write_jobs

VALID_ORIGINS = ("reminder", "heartbeat")


def _validate_job(job: dict[str, Any]) -> str | None:
    """Validate a job dict. Returns an error string or None if valid."""
    if not job.get("name") or not isinstance(job["name"], str):
        return "job must have a non-empty 'name' string"
    if not job.get("channel") or not isinstance(job["channel"], str):
        return f"job {job['name']!r} must have a 'channel' string"
    if ":" not in job["channel"]:
        return f"job {job['name']!r} channel must be 'transport:scope'"
    cron = job.get("cron")
    if not cron or not isinstance(cron, str):
        return f"job {job['name']!r} must have a 'cron' string (5-field expression)"
    try:
        parse_cron(cron)
    except CronParseError as exc:
        return f"job {job['name']!r} has invalid cron {cron!r}: {exc}"
    origin = job.get("origin", "reminder")
    if origin not in VALID_ORIGINS:
        return f"job {job['name']!r} origin must be one of {VALID_ORIGINS}, got {origin!r}"
    if not job.get("prompt") and not job.get("prompt_file"):
        return f"job {job['name']!r} must have a 'prompt' or 'prompt_file'"
    return None


def add_job(
    path: Path,
    name: str,
    channel: str,
    cron: str,
    prompt: str,
    origin: str = "reminder",
) -> str:
    """Add a new job. Returns a confirmation or error string."""
    jobs = read_jobs(path)

    # Check for duplicate name.
    if any(j["name"] == name for j in jobs):
        return f"Could not add: a job named {name!r} already exists."

    job: dict[str, Any] = {
        "name": name,
        "channel": channel,
        "cron": cron,
        "origin": origin,
        "prompt": prompt,
    }
    err = _validate_job(job)
    if err:
        return f"Could not add: {err}"

    jobs.append(job)
    write_jobs(path, jobs)
    return f"Added job {name!r}: {cron} → {channel} ({origin})"


def update_job(path: Path, name: str, **changes: Any) -> str:
    """Update an existing job's fields. Returns a confirmation or error."""
    jobs = read_jobs(path)

    target = None
    for job in jobs:
        if job["name"] == name:
            target = job
            break

    if target is None:
        return f"Could not update: no job named {name!r} found."

    # Apply changes.
    allowed_fields = {"channel", "cron", "origin", "prompt", "prompt_file"}
    for key, value in changes.items():
        if key not in allowed_fields:
            return f"Could not update: {key!r} is not an updatable field " \
                   f"(use one of {sorted(allowed_fields)})"
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value

    err = _validate_job(target)
    if err:
        return f"Could not update: {err}"

    write_jobs(path, jobs)
    changed = ", ".join(f"{k}={v!r}" for k, v in changes.items() if v is not None)
    return f"Updated job {name!r}: {changed}"


def remove_job(path: Path, name: str) -> str:
    """Remove a job by name. Returns a confirmation or error."""
    jobs = read_jobs(path)

    target_idx = None
    for i, job in enumerate(jobs):
        if job["name"] == name:
            target_idx = i
            break

    if target_idx is None:
        return f"Could not remove: no job named {name!r} found."

    removed = jobs.pop(target_idx)
    write_jobs(path, jobs)
    return f"Removed job {name!r} ({removed.get('cron', '?')} → {removed.get('channel', '?')})"


def list_jobs(path: Path) -> str:
    """List all jobs. Returns a formatted summary string."""
    jobs = read_jobs(path)

    if not jobs:
        return "No jobs configured."

    lines = [f"{len(jobs)} job(s):"]
    for job in jobs:
        cron = job.get("cron", "?")
        channel = job.get("channel", "?")
        origin = job.get("origin", "reminder")
        prompt = job.get("prompt", "")
        if len(prompt) > 40:
            prompt = prompt[:37] + "..."
        lines.append(f"  {job['name']}: {cron} → {channel} [{origin}] \"{prompt}\"")

    return "\n".join(lines)
