"""YAML job file reader/writer for the scheduler extension.

Reads and writes `state/scheduler/jobs.yaml` (or a configured path).
The file format:

    jobs:
      - name: morning-checkin
        channel: signal:4d8d...
        origin: reminder
        cron: "0 8 * * *"
        prompt: "Good morning."

Uses pyyaml (already a core dependency). Atomic write: write to a temp
file in the same directory, then rename. Comments are lost on write.
If the file is empty or missing, returns an empty list.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def read_jobs(path: Path) -> list[dict[str, Any]]:
    """Read the jobs list from a YAML file. Returns [] if the file is
    missing, empty, or has no `jobs` key."""
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError) as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        return []
    jobs = data.get("jobs", [])
    if not isinstance(jobs, list):
        return []
    return jobs


def write_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    """Write the jobs list to a YAML file atomically. Comments are lost."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"jobs": jobs}
    content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
