"""Tests for the scheduler file watcher (hot reload)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from corvidae.scheduler_watch import FileWatcher


@pytest.mark.asyncio
async def test_watcher_fires_initial_sync(tmp_path: Path):
    """The callback fires once at startup (initial sync)."""
    f = tmp_path / "jobs.yaml"
    f.write_text('{"jobs": []}')

    calls: list[Path] = []

    async def on_change(path: Path) -> None:
        calls.append(path)

    watcher = FileWatcher(f, on_change, poll_interval=0.1)
    await watcher.start()
    await asyncio.sleep(0.3)  # Let the initial sync fire
    await watcher.stop()

    assert len(calls) == 1
    assert calls[0] == f


@pytest.mark.asyncio
async def test_watcher_detects_change(tmp_path: Path):
    """The callback fires when the file's mtime changes."""
    f = tmp_path / "jobs.yaml"
    f.write_text('{"jobs": []}')

    calls: list[int] = [0]

    async def on_change(path: Path) -> None:
        calls[0] += 1

    watcher = FileWatcher(f, on_change, poll_interval=0.1)
    await watcher.start()
    await asyncio.sleep(0.2)  # Initial sync
    initial_count = calls[0]

    # Modify the file (change mtime)
    await asyncio.sleep(0.15)  # Ensure mtime differs
    f.write_text('{"jobs": [{"name": "new", "channel": "x", "cron": "* * * * *"}]}')

    await asyncio.sleep(0.3)  # Wait for the watcher to detect the change
    await watcher.stop()

    assert calls[0] > initial_count


@pytest.mark.asyncio
async def test_watcher_idempotent_start(tmp_path: Path):
    """Starting the watcher twice is a no-op."""
    f = tmp_path / "jobs.yaml"
    f.write_text('{"jobs": []}')

    calls: list[int] = [0]

    async def on_change(path: Path) -> None:
        calls[0] += 1

    watcher = FileWatcher(f, on_change, poll_interval=0.1)
    await watcher.start()
    await watcher.start()  # Second call should be a no-op
    await asyncio.sleep(0.3)
    await watcher.stop()

    # Only one initial sync (not two)
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_watcher_stop_cancels_task(tmp_path: Path):
    """Stopping the watcher cancels the background task."""
    f = tmp_path / "jobs.yaml"
    f.write_text('{"jobs": []}')

    async def on_change(path: Path) -> None:
        pass

    watcher = FileWatcher(f, on_change, poll_interval=0.1)
    await watcher.start()
    assert watcher.running

    await watcher.stop()
    assert not watcher.running


@pytest.mark.asyncio
async def test_watcher_missing_file(tmp_path: Path):
    """The watcher handles a missing file gracefully."""
    f = tmp_path / "nonexistent.yaml"

    calls: list[int] = [0]

    async def on_change(path: Path) -> None:
        calls[0] += 1

    watcher = FileWatcher(f, on_change, poll_interval=0.1)
    await watcher.start()
    await asyncio.sleep(0.3)
    await watcher.stop()

    # Initial sync fires even for a missing file (the callback handles it)
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_watcher_callback_exception_does_not_kill_loop(tmp_path: Path):
    """A callback exception is logged, not propagated."""
    f = tmp_path / "jobs.yaml"
    f.write_text('{"jobs": []}')

    call_count: list[int] = [0]

    async def on_change(path: Path) -> None:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated failure")

    watcher = FileWatcher(f, on_change, poll_interval=0.1)
    await watcher.start()
    await asyncio.sleep(0.1)  # Initial sync (raises)
    # Modify the file to trigger another call
    await asyncio.sleep(0.15)
    f.write_text('{"jobs": [{"name": "x", "channel": "y", "cron": "* * * * *"}]}')
    await asyncio.sleep(0.3)  # Should fire again despite the earlier exception
    await watcher.stop()

    # The loop survived the exception and fired again
    assert call_count[0] >= 2
