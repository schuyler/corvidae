"""Size-capped append for the JSONL sinks (#404).

``JsonlLogPlugin`` (per-channel transcripts) and ``MetricsJsonlPlugin`` both
append one JSON line per event to a file that otherwise grows without bound.
``RotatingAppender`` caps such a file the way ``logging.file`` is capped: when
a write would pass ``max_bytes`` the file becomes ``x.jsonl.1``, the previous
``.1`` becomes ``.2``, and so on up to ``backup_count``, the oldest dropped.

Config (shared by both sinks; restart-only):
    logging:
      jsonl:
        max_bytes: 10485760   # per file; 0 = never rotate
        backup_count: 5       # rotated files kept; 0 = keep none
"""

import logging
import os
import threading
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

# Match logging.file (corvidae/logging.py): 10 MB x 5 backups.
DEFAULT_MAX_BYTES = 10_485_760
DEFAULT_BACKUP_COUNT = 5


def parse_rotation(config: dict) -> tuple[int, int]:
    """The (max_bytes, backup_count) from ``logging.jsonl``, defaults filled in.

    Raises:
        ValueError: naming the key, the rejected value and the accepted values,
            for a section that is not a mapping or a value that is not an
            integer of 0 or greater.
    """
    section = (config.get("logging") or {}).get("jsonl")
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError(
            f"logging.jsonl is {section!r}; use a mapping with max_bytes and "
            "backup_count, or omit it for the defaults"
        )
    values = []
    for key, default in (("max_bytes", DEFAULT_MAX_BYTES), ("backup_count", DEFAULT_BACKUP_COUNT)):
        value = section.get(key, default)
        # bool is an int subclass, but `max_bytes: true` is a typo, not a size.
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"logging.jsonl.{key} is {value!r}; use an integer of 0 or greater"
                f" (default {default})"
            )
        values.append(value)
    return values[0], values[1]


class RotatingAppender:
    """An append-mode JSONL file that rotates itself before it passes a cap.

    Synchronous: callers run it in worker threads (``asyncio.to_thread``), and
    one appender can be written from several at once (concurrent metrics
    events), so a lock serialises writes: two threads must not both rotate.
    The handle is opened lazily and flushed per write, so an external
    ``copytruncate`` rotation still works on a file this class has open.
    """

    def __init__(self, path: Path, max_bytes: int, backup_count: int) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._fh: IO[str] | None = None
        self._lock = threading.Lock()
        # True while rotation keeps failing, so the warning is logged once per
        # streak and not once per record.
        self._rotation_failing = False

    def write_line(self, line: str) -> None:
        """Append ``line`` and a newline, rotating first when it would not fit."""
        with self._lock:
            self._write_line_locked(line)

    def _write_line_locked(self, line: str) -> None:
        """The body of ``write_line``; the caller holds the lock."""
        if self._fh is None:
            self._fh = open(self._path, "a", encoding="utf-8")
        # Size from the file itself, not a counter: the file may already be over
        # the cap when the daemon starts, or have been truncated externally.
        size = os.fstat(self._fh.fileno()).st_size
        incoming = len(line.encode("utf-8")) + 1
        # `size > 0` keeps a single record larger than the cap from rotating an
        # empty file forever; it is written whole.
        if self._max_bytes and size > 0 and size + incoming > self._max_bytes:
            try:
                self._rotate()
                self._rotation_failing = False
            except OSError as exc:
                # shortcut: keep appending past the cap while rotation fails (a
                # read-only directory, say). Losing every later record is worse
                # than a file over its cap; rotation is retried on each write.
                if not self._rotation_failing:
                    logger.warning(
                        "could not rotate JSONL file %s (%s: %s); appending past its "
                        "%d-byte cap until rotation works — check the directory's "
                        "permissions and free space",
                        self._path, type(exc).__name__, exc, self._max_bytes,
                        exc_info=True,
                    )
                self._rotation_failing = True
            self._fh = open(self._path, "a", encoding="utf-8")
        self._fh.write(line + "\n")
        self._fh.flush()

    def _rotate(self) -> None:
        """Shift ``x`` -> ``x.1`` -> ... -> ``x.N``, dropping the oldest."""
        assert self._fh is not None
        self._fh.close()
        self._fh = None
        if self._backup_count == 0:
            os.remove(self._path)
        else:
            # Oldest first, so each rename lands on a name just vacated;
            # os.replace overwrites the file that falls off the end.
            for n in range(self._backup_count - 1, 0, -1):
                src = self._path.with_name(f"{self._path.name}.{n}")
                if src.exists():
                    os.replace(src, self._path.with_name(f"{self._path.name}.{n + 1}"))
            os.replace(self._path, self._path.with_name(f"{self._path.name}.1"))
        logger.info(
            "rotated JSONL file %s (cap %d bytes, keeping %d)",
            self._path, self._max_bytes, self._backup_count,
        )

    def close(self) -> None:
        """Flush and close the handle; a later write reopens the file."""
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
