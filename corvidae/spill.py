"""The spill service: a notification or tool result too long to show inline
is cut, and its full text is written once to a file the agent reads on demand.

A plugin that cuts notification text calls ``NotificationSpill.notify``
instead of firing ``on_notify`` itself. The Agent cuts a tool result over
``tools.max_result_chars`` through ``NotificationSpill.cut`` and records the
returned ``Cut`` in the result row's provenance (#575); a subagent loop,
which stores no rows, renders the same words as plain text. The service decides, per channel,
whether the channel's tools can read a spill file, writes one file for the
whole delivery, and tells ``Agent.on_notify`` what it did through a context
variable (``current_cut``), so the kernel renders the cut marker from the
stored row and no plugin can forge it.

Configuration (``daemon`` section, applied at restart):
    notification_spill_dir: directory for spill files (default
        ``state/notification-spill``, relative to the config directory)
    notification_spill_max_chars: characters written at most (default 1000000)
    notification_spill_max_age_days: age after which files are deleted (default 7)

Logging:
    - INFO: a cut notification or tool result was spilled (characters, path)
    - INFO: the age sweep deleted files
    - WARNING: the spill write failed; the item is delivered cut
    - WARNING: the age sweep could not list the directory or delete a file
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from corvidae.channel import Channel
from corvidae.clock import Clock
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.provenance import clean_value
from corvidae.tool import declares_spill_reader
from corvidae.tool_collection import ToolCollectionPlugin
from corvidae.tool_policy import disabled_on

logger = logging.getLogger(__name__)

DEFAULT_DIR = "state/notification-spill"
DEFAULT_MAX_CHARS = 1_000_000
DEFAULT_MAX_AGE_DAYS: float = 7
CLEANUP_INTERVAL_SECONDS = 3600.0
#: The only names the service writes or deletes.
SPILL_NAME = re.compile(r"[0-9a-f]{32}\.txt")
#: A spill file name inside longer text, such as a tool call's argument.
SPILL_NAME_IN_TEXT = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}\.txt")


@dataclass(frozen=True)
class Cut:
    """What the kernel did with one channel's copy of a cut notification or tool result."""

    shown: int  # characters passed inline
    total: int  # characters in the full text
    path: str | None = None  # spill file path; set only when this channel gets a pointer
    spilled: int | None = None  # characters in the file; set with path
    failed: bool = False  # this channel can read spills, but the write failed
    source: str | None = None  # who the item came from, kept on the row (#576)


_current_cut: contextvars.ContextVar[Cut | None] = contextvars.ContextVar(
    "corvidae_notification_cut", default=None
)


def current_cut() -> Cut | None:
    """The cut of the notification being delivered, or None; read by Agent.on_notify."""
    return _current_cut.get()


class NotificationSpill(CorvidaePlugin):
    """Kernel service that spills the full text of cut notifications to files."""

    depends_on = frozenset({"clock", "tools"})

    def __init__(self) -> None:
        self.dir = Path(DEFAULT_DIR)
        self.max_chars = DEFAULT_MAX_CHARS
        self.max_age_days: float = DEFAULT_MAX_AGE_DAYS

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        daemon = config.get("daemon") or {}

        directory = daemon.get("notification_spill_dir", DEFAULT_DIR)
        if not isinstance(directory, str):
            raise ValueError(
                f"daemon.notification_spill_dir is {directory!r}; "
                f"use a directory path such as {DEFAULT_DIR}"
            )
        # Relative paths resolve against the config directory, like jsonl_log_dir.
        resolved = (Path(config.get("_base_dir", Path("."))) / directory).resolve()
        # The header renders the path through clean_value; refuse a directory
        # it would alter, so the pointer is always a path read_file accepts.
        if clean_value(str(resolved)) != str(resolved):
            raise ValueError(
                f"daemon.notification_spill_dir resolves to {str(resolved)!r}, which a "
                f"notification header would not render verbatim (it would show "
                f"{clean_value(str(resolved))!r}); choose a directory whose path has no "
                "brackets, <, >, ;, control characters, doubled or edge whitespace, or "
                "characters NFKC normalisation changes"
            )
        self.dir = resolved

        # The spill cap: a positive integer number of characters.
        max_chars = daemon.get("notification_spill_max_chars", DEFAULT_MAX_CHARS)
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError(
                f"daemon.notification_spill_max_chars is {max_chars!r}; "
                f"use a positive integer number of characters (default {DEFAULT_MAX_CHARS})"
            )
        self.max_chars = max_chars

        # The age policy in days; `not > 0` also refuses NaN, which would
        # silently stop every sweep.
        max_age = daemon.get("notification_spill_max_age_days", DEFAULT_MAX_AGE_DAYS)
        if isinstance(max_age, bool) or not isinstance(max_age, (int, float)) or not max_age > 0:
            raise ValueError(
                f"daemon.notification_spill_max_age_days is {max_age!r}; "
                f"use a positive number of days (default {DEFAULT_MAX_AGE_DAYS:g})"
            )
        self.max_age_days = max_age

    @hookimpl
    async def on_start(self, config: dict) -> None:
        # Clear what aged out while the daemon was down, then keep clearing hourly.
        await self._sweep_now()
        self._arm_timer()

    async def notify(
        self,
        channels: Sequence[Channel],
        *,
        source: str,
        text: str,
        annotations: dict | None,
        limit: int,
    ) -> str | None:
        """Notify every channel of ``text``, cut to ``limit`` characters when longer.

        Returns the spill file path, or None when no file was written.
        """
        # Text that fits goes out unchanged, with no cut recorded.
        if len(text) <= limit:
            for channel in channels:
                await self._on_notify(channel, source, text, annotations)
            return None

        readers = [self._can_read(channel) for channel in channels]
        # One file per item, written only when some channel can read it.
        path, spilled, failed = await self._spill(
            text, source=source, limit=limit,
            what=f"for {sum(readers)} of {len(channels)} channels",
        ) if any(readers) else (None, None, False)

        # Each channel's row records its own cut: a pointer only where it can
        # read the file, "not saved" only where it could have.
        for channel, reads in zip(channels, readers):
            cut = Cut(
                shown=limit,
                total=len(text),
                path=str(path) if reads and path else None,
                spilled=spilled if reads and path else None,
                failed=reads and failed,
                source=source,
            )
            token = _current_cut.set(cut)
            try:
                await self._on_notify(channel, source, text[:limit], annotations)
            finally:
                _current_cut.reset(token)
        return str(path) if path else None

    async def cut(self, text: str, *, limit: int, source: str, can_read: bool) -> Cut:
        """Spill ``text``, which is longer than ``limit``, for one destination.

        The caller passes the first ``limit`` characters on and records the
        returned ``Cut``. The file is written only when ``can_read``.
        """
        path, spilled, failed = await self._spill(
            text, source=source, limit=limit, what="for one destination",
        ) if can_read else (None, None, False)
        return Cut(
            shown=limit, total=len(text),
            path=str(path) if path else None, spilled=spilled, failed=failed,
            source=source,
        )

    def can_read(self, channel: Channel, tools: dict | None = None) -> bool:
        """Whether an enabled tool on ``channel`` declares it reads spill files.

        ``tools`` is the tool set to check (a subagent's, say); by default the
        daemon's registry.
        """
        if tools is None:
            return self._can_read(channel)
        disabled = disabled_on(channel) if channel is not None else frozenset()
        return any(
            declares_spill_reader(fn) and name not in disabled for name, fn in tools.items()
        )

    def contains(self, path: str) -> bool:
        """Whether ``path`` lies inside the spill directory."""
        return Path(os.path.abspath(path)).is_relative_to(self.dir)

    def stale_pointer_error(self, path: str) -> str | None:
        """The error text for a missing file inside the spill directory, else None."""
        absolute = Path(os.path.abspath(path))
        if not absolute.is_relative_to(self.dir) or os.path.exists(absolute):
            return None
        return (
            f"Error: file not found: {path}; it was a spill file (the full text of a cut "
            f"notification or tool result), and spill files are deleted "
            f"{self.max_age_days:g} days after they are written "
            "(daemon.notification_spill_max_age_days)"
        )

    async def _spill(
        self, text: str, *, source: str, limit: int, what: str
    ) -> tuple[Path | None, int | None, bool]:
        """Write up to the cap of ``text``; return (path, characters, failed)."""
        content = text[: self.max_chars]
        try:
            path = await asyncio.to_thread(self._write, content)
        except OSError as exc:
            logger.warning(
                "spill: could not write the full text of an item from %s (%d characters) "
                "to %s (%s); delivering the first %d characters with a header saying the "
                "full text was not saved — check that the directory is writable and has space",
                source, len(text), self.dir, type(exc).__name__, limit,
                exc_info=True,
            )
            return None, None, True
        logger.info(
            "item from %s cut from %d to %d characters; %d characters written to %s %s",
            source, len(text), limit, len(content), path, what,
        )
        return path, len(content), False

    async def _on_notify(
        self, channel: Channel, source: str, text: str, annotations: dict | None
    ) -> None:
        """Fire on_notify for one channel with a fresh copy of the annotations."""
        await self.pm.ahook.on_notify(
            channel=channel,
            source=source,
            text=text,
            tool_call_id=None,
            annotations=dict(annotations or {}),
        )

    def _can_read(self, channel: Channel) -> bool:
        """Whether an enabled tool on ``channel`` declares it reads spill files."""
        registry = get_dependency(self.pm, "tools", ToolCollectionPlugin).get_registry()
        # Before the registry is built (the webhook ingress can deliver first)
        # no channel is known to have a reader.
        if registry is None:
            return False
        disabled = disabled_on(channel)
        return any(
            declares_spill_reader(fn) and name not in disabled
            for name, fn in registry.as_dict().items()
        )

    def _write(self, content: str) -> Path:
        """Write ``content`` to a new service-named file and return its path."""
        # The directory may have been removed while the daemon ran.
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{uuid.uuid4().hex}.txt"
        # Mode 0600: the content is untrusted and may carry secrets. Replacement
        # keeps one character per code point (a lone surrogate would otherwise
        # refuse to encode), so the header's counts hold.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        return path

    def _arm_timer(self) -> None:
        """Schedule the next hourly sweep."""
        get_dependency(self.pm, "clock", Clock).call_later(
            CLEANUP_INTERVAL_SECONDS, self._hourly, name="notification-spill-cleanup"
        )

    async def _hourly(self) -> None:
        """Timer callback: re-arm first so a failed sweep never stops the timer."""
        self._arm_timer()
        await self._sweep_now()

    async def _sweep_now(self) -> None:
        """Delete spill files older than the age policy."""
        now = get_dependency(self.pm, "clock", Clock).now()
        await asyncio.to_thread(self._sweep, now)

    def _sweep(self, now: float) -> None:
        """Unlink service-named regular files whose mtime is past the age."""
        cutoff = now - self.max_age_days * 86400
        try:
            entries = list(self.dir.iterdir())
        except FileNotFoundError:
            # Nothing has been spilled yet.
            return
        except OSError as exc:
            logger.warning(
                "notification spill: could not list %s (%s); old spill files are not "
                "being deleted — check the directory's permissions",
                self.dir, type(exc).__name__, exc_info=True,
            )
            return
        deleted = 0
        for entry in entries:
            try:
                if (
                    SPILL_NAME.fullmatch(entry.name)
                    and entry.is_file()
                    and entry.stat().st_mtime < cutoff
                ):
                    entry.unlink()
                    deleted += 1
            except OSError as exc:
                logger.warning(
                    "notification spill: could not delete %s (%s); it will be retried "
                    "at the next sweep — check the file's permissions",
                    entry, type(exc).__name__, exc_info=True,
                )
        if deleted:
            logger.info(
                "notification spill: deleted %d files older than %g days from %s",
                deleted, self.max_age_days, self.dir,
            )
