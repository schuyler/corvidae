"""inotify_ping: event-driven completion pings for background jobs (goal #47).

Replaces the "set a reminder for N minutes and hope the job finished by then"
pattern with a ping that fires on the actual artifact. When enabled, the plugin
watches configured file paths (libc inotify, no dependency) and, when a watched
file is written and then goes quiet for a debounce window (it "settles"), it
injects a notification into a target channel via ``on_notify`` — the same
mechanism the scheduler uses for reminders.

The load-bearing signal is *settle*, not *modify*: a running job writes many
times, so a raw inotify event is noise. The plugin coalesces a burst of events
into one ping fired after the file is quiet for ``debounce_seconds``. One ping
per activity burst, re-arming after each fire, so a long-lived log (or a job
that runs again) pings again on the next quiet. No sentinel file and no
job-side cooperation — it works on plain ``systemd-run`` logs as they are today.
An optional ``terminal_marker`` tightens the criterion from *quiet* to *complete*: when set, a
settle only pings if the file's last non-empty line contains the marker (e.g. a ``done`` line), so a
mid-run pause longer than the debounce does not fire a false completion. Unset, the quiet-period
behavior above stands.

If a configured file does not exist when the plugin arms (a job that has not
yet created its log), the plugin watches the file's parent directory and
filters for the basename, so the file can appear and settle after boot instead
of arming a silent 0/N from ENOENT.

Surface: a background watcher thread bridges inotify events onto the event
loop; the async side does the debounce and the ``on_notify``. The target
channel is resolved at fire time via ``resolve_persisted_channel`` (live
registry -> transport hook -> bare ``internal:*``/IRC/CLI fallback), so it
works after a restart; an unresolvable channel is a dead ping, so it is dropped with a
WARNING. Dormant by default (``inotify_ping.enabled: false``).
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import logging
import os
import select
import struct
import threading
from threading import Lock
from dataclasses import dataclass, field
from typing import Any

from corvidae.channel import ChannelRegistry, resolve_persisted_channel
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl

logger = logging.getLogger("corvidae.inotify_ping")

# --- libc inotify bindings (no pip dependency) --------------------------------

_libc: ctypes.CDLL | None = None


def _get_libc() -> ctypes.CDLL:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    return _libc

IN_MODIFY = 0x2
IN_CLOSE_WRITE = 0x8
IN_MOVED_TO = 0x40
IN_DELETE = 0x200
# The masks that mean "the file's content changed" and should count as activity.
WRITE_MASK = IN_MODIFY | IN_CLOSE_WRITE | IN_MOVED_TO

# struct inotify_event { int wd; uint32_t mask; uint32_t cookie; uint32_t len; char name[]; }
_EVENT_FMT = "iIII"
_EVENT_BASE = struct.calcsize(_EVENT_FMT)  # 16 bytes


def _inotify_init() -> int:
    fd = _get_libc().inotify_init()
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init failed")
    return fd


def _inotify_add_watch(fd: int, path: str, mask: int) -> int:
    if isinstance(path, str):
        path = path.encode("utf-8")
    wd = _get_libc().inotify_add_watch(fd, path, mask)
    if wd < 0:
        raise OSError(ctypes.get_errno(), f"inotify_add_watch({path!r}) failed")
    return wd


def _inotify_rm_watch(fd: int, wd: int) -> None:
    _get_libc().inotify_rm_watch(fd, wd)


def parse_events(buf: bytes) -> list[tuple[int, int, str]]:
    """Parse a raw inotify read buffer into (wd, mask, name) tuples.

    Pure: the whole buffer is parsed, variable-length ``name`` fields included.
    The ``name`` is a null-terminated C string (``len`` bytes, NUL included);
    the trailing NUL is stripped, so a directory-watch event for ``job.log``
    yields ``"job.log"`` and a file-watch event (no name) yields ``""``.
    """
    events: list[tuple[int, int, str]] = []
    off = 0
    n = len(buf)
    while off + _EVENT_BASE <= n:
        wd, mask, _cookie, length = struct.unpack_from(_EVENT_FMT, buf, off)
        name = (
            buf[off + _EVENT_BASE : off + _EVENT_BASE + length].rstrip(b"\x00").decode("utf-8", "replace")
            if length
            else ""
        )
        events.append((wd, mask, name))
        off += _EVENT_BASE + length
    return events


def classify(mask: int) -> str | None:
    """Return "write" for a content-changing mask, None otherwise.

    Pure and dependency-free so the event-to-activity rule is unit-testable.
    """
    if mask & WRITE_MASK:
        return "write"
    return None


def watch_target(path: str) -> tuple[str, str | None]:
    """Return (watch_path, filter_name) for a target file path.

    If the file exists, watch it directly (filter_name is None — every
    content-changing event counts). If the file is absent but its parent
    directory exists, watch the directory and filter for the basename, so a
    file created *after* arming still settles (graceful degradation instead of
    a silent 0/N from ENOENT). If neither exists, return the path unchanged and
    let the caller surface the failure.
    """
    if os.path.exists(path):
        return path, None
    parent = os.path.dirname(path) or "."
    if os.path.isdir(parent):
        return parent, os.path.basename(path)
    return path, None


def has_terminal_marker(path: str, marker: str, tail_bytes: int = 4096) -> bool:
    """Return True if the file's last non-empty line contains the marker.

    The terminal marker is a completion signal, so it lives at the end of the
    log — read only the tail (a log can be large). A missing/unreadable/empty
    file returns False (no completion signal). Pure and dependency-free so the
    criterion is unit-testable.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            data = f.read()
    except OSError:
        return False
    lines = [ln for ln in data.decode("utf-8", "replace").splitlines() if ln.strip()]
    if not lines:
        return False
    return marker in lines[-1]


# --- config ------------------------------------------------------------------


@dataclass
class InotifyConfig:
    enabled: bool
    channel: str = ""
    debounce_seconds: float = 5.0
    paths: list[str] = field(default_factory=list)
    terminal_marker: str | None = None


def load_config(raw: dict[str, Any]) -> InotifyConfig:
    """Parse and validate the ``inotify_ping`` config section.

    A disabled section is inert: only ``enabled`` is validated, so a present
    but off section does not need channel/paths. Enabled requires a non-empty
    channel and a non-empty list of path strings, and a debounce >= 0.
    """
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"inotify_ping.enabled must be a bool, got {enabled!r}")
    if not enabled:
        return InotifyConfig(enabled=False)

    channel = raw.get("channel", "")
    if not isinstance(channel, str) or not channel:
        raise ValueError(f"inotify_ping.channel must be a non-empty string, got {channel!r}")

    debounce = raw.get("debounce_seconds", 5.0)
    if isinstance(debounce, bool) or not isinstance(debounce, (int, float)) or debounce < 0:
        raise ValueError(f"inotify_ping.debounce_seconds must be a number >= 0, got {debounce!r}")

    paths = raw.get("paths", [])
    if (
        not isinstance(paths, list)
        or not paths
        or not all(isinstance(p, str) and p for p in paths)
    ):
        raise ValueError(f"inotify_ping.paths must be a non-empty list of strings, got {paths!r}")

    terminal_marker = raw.get("terminal_marker")
    if terminal_marker is not None and (
        not isinstance(terminal_marker, str) or not terminal_marker
    ):
        raise ValueError(
            f"inotify_ping.terminal_marker must be a non-empty string or null, got {terminal_marker!r}"
        )

    return InotifyConfig(
        enabled=True,
        channel=channel,
        debounce_seconds=float(debounce),
        paths=list(paths),
        terminal_marker=terminal_marker,
    )


# --- plugin ------------------------------------------------------------------


class InotifyPingPlugin(CorvidaePlugin):
    """Watch file paths and ping a channel when a watched file settles.

    A path whose file exists is watched directly; a path whose file is absent
    (but whose parent directory exists) degrades to a directory watch filtered
    on the basename, so the file can appear and settle after boot.
    """

    # on_config_reload re-parses the whole section and re-arms, so declare the
    # section reloadable — otherwise config_watcher logs a false-positive
    # "restart-only" warning for inotify_ping.* changes that DO take effect live.
    reloadable_keys = ("inotify_ping",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._config: InotifyConfig | None = None
        self._enabled: bool = False
        self._fd: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wd_info: dict[int, list[tuple[str, str | None]]] = {}
        self._wd_lock = Lock()
        self._settle_tasks: dict[str, asyncio.Task] = {}
        self._arm_task: asyncio.Task | None = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self._config = load_config(config.get("inotify_ping", {}))
        self._enabled = self._config.enabled
        logger.info(
            "inotify_ping: on_init enabled=%s paths=%s channel=%r",
            self._enabled, self._config.paths, self._config.channel,
        )

    @hookimpl
    async def on_start(self, config: dict) -> None:
        if not self._enabled:
            return
        # Spawn a background task so on_start returns immediately.
        # Runtime.start awaits the on_start broadcast before calling
        # agent.on_start() (which sets _started_event), so blocking here
        # would deadlock the entire startup.
        self._arm_task = asyncio.ensure_future(self._wait_and_arm())

    async def _wait_and_arm(self) -> None:
        await self._wait_for_agent()
        try:
            self._arm_watches()
        except OSError as exc:
            logger.error("inotify_ping: arm failed: %s", exc)
            self._fd = None

    async def _wait_for_agent(self) -> None:
        """Block until the Agent can take notifications (the guide's requirement).

        ``agent.started`` is a bool property and ``agent._started_event`` is the
        asyncio.Event set on start — the same idiom the Agent's own on_message
        uses. Defensive getattr so a renamed attribute degrades to "proceed"
        rather than crashing startup.
        """
        agent = self.pm.get_plugin("agent") if self.pm is not None else None
        if agent is None:
            return
        if not getattr(agent, "started", True):
            event = getattr(agent, "_started_event", None)
            if event is not None:
                await event.wait()

    def _arm_watches(self) -> None:
        if self._fd is None:
            self._fd = _inotify_init()
            self._loop = asyncio.get_running_loop()
            self._stop.clear()
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()
        self._reset_watches()

    def _reset_watches(self) -> None:
        with self._wd_lock:
            for wd in list(self._wd_info):
                try:
                    _inotify_rm_watch(self._fd, wd)
                except OSError:
                    pass
            self._wd_info.clear()
            n_file = n_dir = 0
            for p in self._config.paths:
                watch_path, filter_name = watch_target(p)
                try:
                    wd = _inotify_add_watch(self._fd, watch_path, WRITE_MASK)
                except OSError as exc:
                    logger.warning("inotify_ping: could not watch %s: %s", p, exc)
                    continue
                self._wd_info.setdefault(wd, []).append((p, filter_name))
                if filter_name is None:
                    n_file += 1
                else:
                    n_dir += 1
        logger.info(
            "inotify_ping: armed %d/%d watches (%d file, %d dir) -> channel %r (debounce %.1fs)",
            len(self._wd_info), len(self._config.paths), n_file, n_dir,
            self._config.channel, self._config.debounce_seconds,
        )

    def _watch_loop(self) -> None:
        """Blocking inotify reader on a worker thread; bridges to the loop."""
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                data = os.read(self._fd, 65536)
            except OSError:
                break
            if not data:
                continue
            for wd, mask, name in parse_events(data):
                with self._wd_lock:
                    targets = list(self._wd_info.get(wd, []))
                if not targets or classify(mask) is None:
                    continue
                for target_path, filter_name in targets:
                    if filter_name is not None and name != filter_name:
                        continue
                    if self._loop is not None:
                        asyncio.run_coroutine_threadsafe(
                            self._on_fs_event(target_path), self._loop
                        )
        # Thread is exiting: close the fd (the thread owns it for its
        # lifetime) and clear it so a subsequent reload can re-arm.
        logger.warning("inotify_ping: watcher thread exiting")
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    async def _on_fs_event(self, path: str) -> None:
        """Coalesce: cancel any pending settle for this path, start a fresh one."""
        task = self._settle_tasks.get(path)
        if task is not None and not task.done():
            task.cancel()
        self._settle_tasks[path] = asyncio.ensure_future(self._settle(path))

    async def _settle(self, path: str) -> None:
        try:
            await asyncio.sleep(self._config.debounce_seconds)
        except asyncio.CancelledError:
            return
        self._settle_tasks.pop(path, None)
        marker = self._config.terminal_marker
        if marker and not has_terminal_marker(path, marker):
            logger.debug(
                "inotify_ping: %s quiet but no terminal marker %r; not pinging",
                path, marker,
            )
            return
        await self._emit_ping(path)

    async def _emit_ping(self, path: str) -> None:
        if not self._enabled:
            return
        channel = None
        try:
            registry = get_dependency(self.pm, "registry", ChannelRegistry)
            channel = await resolve_persisted_channel(
                self.pm, registry, self._config.channel
            )
        except Exception:  # noqa: BLE001 - a resolution failure is a dropped ping
            channel = None
        if channel is None:
            logger.warning(
                "inotify_ping: target channel %r not resolvable; dropping ping for %s",
                self._config.channel, path,
            )
            return
        await self.pm.ahook.on_notify(
            channel=channel,
            source="inotify",
            text=f"[inotify] {path} settled — background work likely complete",
            tool_call_id=None,
            annotations={"origin": "inotify", "path": path, "event": "settled"},
        )

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        new_config = load_config(config.get("inotify_ping", {}))
        if not new_config.enabled:
            self._enabled = False
            await self._cancel_arm()
            for task in list(self._settle_tasks.values()):
                task.cancel()
            self._settle_tasks.clear()
            self._disarm()
            return
        self._config = new_config
        self._enabled = True
        if self._fd is not None:
            self._reset_watches()
        else:
            # No active fd: spawn a fresh arm task (covers disabled->enabled
            # and enabled->disabled->enabled transitions).
            self._arm_task = asyncio.ensure_future(self._wait_and_arm())

    def _disarm(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        # The thread closes the fd on exit; if it was never started
        # (fd is None), there is nothing to close.
        self._fd = None
        self._thread = None
        with self._wd_lock:
            self._wd_info.clear()

    @hookimpl
    async def on_stop(self) -> None:
        await self._cancel_arm()
        for task in list(self._settle_tasks.values()):
            task.cancel()
        self._settle_tasks.clear()
        self._disarm()

    async def _cancel_arm(self) -> None:
        if self._arm_task is not None and not self._arm_task.done():
            self._arm_task.cancel()
            try:
                await self._arm_task
            except (asyncio.CancelledError, Exception):
                pass
        self._arm_task = None
