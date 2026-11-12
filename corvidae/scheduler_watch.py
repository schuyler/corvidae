"""Hot reload for the scheduler YAML job file.

Watches the jobs file for changes (mtime-based polling) and triggers
a callback when the file is modified. The callback (provided by the
SchedulerPlugin) re-reads the file, re-syncs the SQLite table, and
wakes the timer.

Uses a background asyncio task that checks the file's mtime every
`poll_interval` seconds. No inotify dependency — works on all
platforms, including the LXD container.

The watcher is idempotent: starting it twice is a no-op. Stopping it
cancels the background task.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 5.0  # seconds


class FileWatcher:
    """Watches a file for mtime changes and calls a callback.

    The callback is async and receives the file path. It is called
    once at startup (initial sync) and then on every detected change.
    """

    def __init__(
        self,
        path: Path,
        on_change: Callable[[Path], Awaitable[None]],
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self._path = path
        self._on_change = on_change
        self._poll_interval = poll_interval
        self._task: asyncio.Task | None = None
        self._last_mtime: float = 0.0
        self._stopped = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start watching. Idempotent: a second call is a no-op."""
        if self.running:
            return
        self._stopped = False
        # Record the current mtime before starting the loop.
        self._last_mtime = self._current_mtime()
        self._task = asyncio.create_task(
            self._watch_loop(), name="scheduler-file-watcher"
        )
        logger.info(
            "scheduler: file watcher started for %s (poll %.0fs)",
            self._path, self._poll_interval,
        )

    async def stop(self) -> None:
        """Stop watching. Cancels the background task."""
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("scheduler: file watcher stopped for %s", self._path)

    def _current_mtime(self) -> float:
        """Return the file's mtime, or 0.0 if it doesn't exist."""
        try:
            return self._path.stat().st_mtime
        except (OSError, FileNotFoundError):
            return 0.0

    async def _watch_loop(self) -> None:
        """The background polling loop."""
        # Initial sync: fire the callback once at startup.
        try:
            await self._on_change(self._path)
        except Exception:
            logger.exception("scheduler: initial file sync failed")

        while not self._stopped:
            await asyncio.sleep(self._poll_interval)
            if self._stopped:
                break

            mtime = self._current_mtime()
            if mtime != self._last_mtime:
                self._last_mtime = mtime
                logger.info(
                    "scheduler: file change detected for %s (mtime %.0f)",
                    self._path, mtime,
                )
                try:
                    await self._on_change(self._path)
                except Exception:
                    logger.exception(
                        "scheduler: file change handler failed for %s",
                        self._path,
                    )
