"""YAML job file reader/writer for the scheduler extension.

Reads and writes `state/scheduler/jobs.yaml` (or a configured path).
The file format:

    jobs:
      - name: morning-checkin
        channel: signal:4d8d...
        origin: reminder
        cron: "0 8 * * *"
        prompt: "Good morning."

Atomic write: write to a temp file in the same directory, then rename.
This prevents a partial write from corrupting the file.

No external YAML dependency: uses a minimal parser/writer for the
specific structure we need (a top-level `jobs` key with a list of
flat dicts). If the file is empty or missing, returns an empty list.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# We use JSON-compatible YAML (a subset). The file is YAML, but we
# write it as JSON-with-YAML-extension for simplicity. YAML is a
# superset of JSON, so JSON is valid YAML. This avoids a PyYAML
# dependency. If we need full YAML later, we can add PyYAML.


def read_jobs(path: Path) -> list[dict[str, Any]]:
    """Read the jobs YAML file. Returns an empty list if the file is
    missing or empty. Raises ValueError on malformed content."""
    if not path.exists():
        return []

    text = path.read_text().strip()
    if not text:
        return []

    # Try JSON first (our writer produces JSON).
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to a minimal YAML parser for the specific structure.
        data = _minimal_yaml_parse(text)

    if not isinstance(data, dict):
        raise ValueError(f"jobs file must be a mapping, got {type(data).__name__}")

    jobs = data.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError(f"jobs must be a list, got {type(jobs).__name__}")

    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"job[{i}] must be a mapping, got {type(job).__name__}")
        if "name" not in job or not isinstance(job["name"], str):
            raise ValueError(f"job[{i}] missing 'name' (string)")

    return jobs


def write_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    """Write the jobs YAML file atomically. Creates parent directories
    if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"jobs": jobs}
    text = json.dumps(data, indent=2) + "\n"

    # Atomic write: temp file in the same directory, then rename.
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".jobs-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _minimal_yaml_parse(text: str) -> dict[str, Any]:
    """Parse the minimal YAML subset we use: a top-level mapping with a
    `jobs` key containing a list of flat mappings.

    This is a fallback for hand-edited files. Our writer produces JSON,
    so this path is rarely hit.
    """
    result: dict[str, Any] = {"jobs": []}
    in_jobs = False
    current_job: dict[str, Any] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped == "jobs:":
            in_jobs = True
            continue

        if in_jobs:
            if stripped.startswith("- "):
                # New job entry
                if current_job is not None:
                    result["jobs"].append(current_job)
                current_job = {}
                stripped = stripped[2:]  # Remove "- "

            if current_job is not None and ":" in stripped:
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip()
                # Strip quotes
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                current_job[key] = value

    if current_job is not None:
        result["jobs"].append(current_job)

    return result
