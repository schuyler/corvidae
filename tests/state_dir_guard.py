"""Fail any test that writes files into the working directory's state/.

Transports and plugins default their output to relative paths under
``state/`` (attachments, notification spill files). A test that forgets to
point that config at ``tmp_path`` writes into the checkout instead, and those
files can end up committed: 39 Signal test PNGs reached main that way in #522.

The fixture lists the files under ``<cwd>/state`` before and after each test
and fails the test, naming the new paths, if any appeared.

shortcut: a daemon running from the same checkout while the suite runs can
write into state/ and trip this; run the daemon from another directory, or
upgrade to per-test chdir if that becomes common.
"""

import os
from pathlib import Path

import pytest


def _files_under(root: Path) -> set[Path]:
    """Return every file path under root (empty set if root is absent)."""
    # os.walk on a missing directory yields nothing, so a clean checkout
    # costs one failed stat per test.
    found: set[Path] = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            found.add(Path(dirpath) / name)
    return found


@pytest.fixture(autouse=True)
def forbid_writes_to_cwd_state(request: pytest.FixtureRequest):
    """Fail the test if it created files under the working directory's state/."""
    # Resolve the directory once at setup, so a test that chdirs is still
    # checked against the directory it started in.
    state_dir = Path.cwd() / "state"
    before = _files_under(state_dir)

    yield

    # Report only files the test added; pre-existing files are not its fault.
    added = sorted(_files_under(state_dir) - before)
    if added:
        listed = "\n  ".join(str(p) for p in added[:10])
        more = f"\n  ... and {len(added) - 10} more" if len(added) > 10 else ""
        pytest.fail(
            f"{request.node.nodeid} wrote {len(added)} file(s) into {state_dir}:\n"
            f"  {listed}{more}\n"
            "Point the plugin's output path (e.g. attachment_dir) at tmp_path, "
            "then delete these files.",
            pytrace=False,
        )
