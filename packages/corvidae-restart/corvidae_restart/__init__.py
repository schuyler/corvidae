"""Corvidae plugin: restart.

The ``restart`` tool, with its loop guard and pre-flight, and the
notification delivered after a restart or an abnormal boot.

Config (``restart:`` in agent.yaml), read in ``on_init``:

    home_channel                channel id for notices with no originating tool call
    config_path                 config the pre-flight validates (default agent.yaml)
    min_uptime_seconds          120
    max_restarts                3 requested restarts ...
    window_seconds              ... per 21600 seconds
    preflight_timeout_seconds   120
    notify_delay_seconds        5

State, under ``<_base_dir>/state/``: ``restart.json`` (pending marker),
``restart-history.jsonl``, and ``boot-history.jsonl`` (written by the
boot supervisor; read here to learn the boot mode and whether the
previous run ended badly).

Test seams, instance attributes set in ``__init__``:

    _preflight           async (config_path, timeout) -> (int, str)
    _killer              (pid, sig) -> None; delivers the SIGTERM
    _monotonic           () -> float; process-uptime clock
    _wall                () -> float; epoch seconds for every timestamp written
    _ready_poll_seconds  polling interval while waiting for ``agent.tools``

Loop guard constraints (checked in order, first failure wins):

    1. No human message since boot -> refused. The ``_user_message_seen``
       latch is set by ``on_message_admitted``. After a restart, the first
       restart attempt will always be refused until a human sends a message.
    2. Uptime < min_uptime_seconds -> refused.
    3. >= max_restarts restarts in window_seconds -> refused.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.tool import Tool, ToolContext

logger = logging.getLogger(__name__)

# Maximum number of characters of pre-flight stderr returned in a refusal.
STDERR_LIMIT = 2000
STDERR_HEAD = 400


def truncate_stderr(text: str) -> str:
    """Fit ``text`` in STDERR_LIMIT characters: head and tail with elision."""
    if len(text) <= STDERR_LIMIT:
        return text
    tail = STDERR_LIMIT - STDERR_HEAD
    elided = len(text) - STDERR_LIMIT
    return f"{text[:STDERR_HEAD]}\n[... {elided} characters truncated ...]\n{text[-tail:]}"


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def run_preflight(config_path: Path, timeout: float) -> tuple[int, str]:
    """Run the pre-flight check in a subprocess.

    Returns ``(returncode, stderr)``. On timeout the process group is
    killed; a timeout yields a non-zero code and a stderr string
    containing "timed out".
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-P",
        "-m",
        "corvidae_restart.preflight",
        "--config",
        str(config_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        _kill_group(proc.pid)
        await proc.wait()
        return 1, f"pre-flight timed out after {timeout:g} seconds"
    except BaseException:
        _kill_group(proc.pid)
        if proc.returncode is None:
            await proc.wait()
        raise
    return proc.returncode, stderr.decode(errors="replace")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping blank, malformed and non-object lines."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except (OSError, ValueError, UnicodeError) as exc:
        logger.warning("cannot read %s, treating it as empty: %s", path, exc)
        return []
    records = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def read_boot_status(path: Path) -> tuple[str | None, bool]:
    """Return ``(mode, previous_run_abnormal)`` for the current boot."""
    records = [r for r in read_jsonl(path) if r.get("reset") is not True]
    starts = [i for i, record in enumerate(records) if "mode" in record]
    if not starts:
        return None, False
    current = starts[-1]
    mode = records[current]["mode"]
    if not isinstance(mode, str):
        logger.warning(
            "boot-history start record has a non-string mode %r; ignoring it", mode
        )
        mode = None
    if current == 0:
        return mode, False
    previous = records[current - 1]
    return mode, previous.get("service_result") != "success"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class RestartPlugin(CorvidaePlugin):
    """Restart tool, startup notification and loop guard."""

    depends_on = frozenset({"registry", "agent"})

    def __init__(self) -> None:
        self._preflight = run_preflight
        self._killer = os.kill
        self._monotonic = time.monotonic
        self._wall = time.time
        self._ready_poll_seconds = 0.25
        self._delivery_task: asyncio.Task | None = None
        self._user_message_seen = False

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read config and boot state."""
        await super().on_init(pm, config)
        section = config.get("restart") or {}
        self._base_dir = Path(config["_base_dir"])
        self._state_dir = self._base_dir / "state"
        self._home_channel_id = section.get("home_channel")
        self._normal_config = self._base_dir / section.get("config_path", "agent.yaml")
        self._min_uptime = section.get("min_uptime_seconds", 120)
        self._max_restarts = section.get("max_restarts", 3)
        self._window = section.get("window_seconds", 21600)
        self._preflight_timeout = section.get("preflight_timeout_seconds", 120)
        self._notify_delay = section.get("notify_delay_seconds", 5)

        self._blocked = list((config.get("plugins") or {}).get("disabled") or [])
        self._loaded = sorted(
            name
            for name, plugin in pm.list_name_plugin()
            if plugin is not None and not name.startswith("_")
        )
        self._boot_mode, self._previous_abnormal = read_boot_status(
            self._state_dir / "boot-history.jsonl"
        )
        self._booted_mono = self._monotonic()
        self._booted_wall = self._wall()

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Start the notification task."""
        self._delivery_task = asyncio.create_task(self._deliver_notices())

    @hookimpl
    async def on_stop(self) -> None:
        task = self._delivery_task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Re-read restart settings when agent.yaml changes."""
        section = config.get("restart") or {}
        self._home_channel_id = section.get("home_channel")
        self._normal_config = self._base_dir / section.get("config_path", "agent.yaml")
        self._min_uptime = section.get("min_uptime_seconds", 120)
        self._max_restarts = section.get("max_restarts", 3)
        self._window = section.get("window_seconds", 21600)
        self._preflight_timeout = section.get("preflight_timeout_seconds", 120)
        self._notify_delay = section.get("notify_delay_seconds", 5)

    @hookimpl
    async def on_message_admitted(self, channel, trace_id, sender, text) -> None:
        """Latch: a human has spoken since this process booted."""
        self._user_message_seen = True

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self

        async def restart(reason: str, _ctx: ToolContext) -> str:
            """Restart this daemon, after a pre-flight check, so plugin, prompt and config changes take effect; the result arrives after the restart.

            Refused until a human has messaged since boot. Also refused
            within min_uptime_seconds of boot, and after max_restarts
            restarts in window_seconds.
            """
            return await plugin._restart(reason, _ctx)

        tool_registry.append(Tool.from_function(restart))

    # -- state files -------------------------------------------------------

    @property
    def _marker_path(self) -> Path:
        return self._state_dir / "restart.json"

    @property
    def _history_path(self) -> Path:
        return self._state_dir / "restart-history.jsonl"

    def _write_marker(self, marker: dict) -> None:
        """Write restart.json atomically."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._marker_path.with_name("restart.json.tmp")
        try:
            tmp.write_text(json.dumps(marker))
            os.replace(tmp, self._marker_path)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _append_history(self, entry: dict) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        with open(self._history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _take_marker(self) -> dict | None:
        """Read the pending marker, record it as delivered and delete it."""
        try:
            marker = json.loads(self._marker_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("cannot read restart marker %s: %s", self._marker_path, exc)
            return None
        except ValueError:
            logger.warning("discarding unreadable restart marker %s", self._marker_path)
            self._discard_marker()
            return None
        if not isinstance(marker, dict):
            logger.warning(
                "discarding unreadable restart marker %s: not a JSON object",
                self._marker_path,
            )
            self._discard_marker()
            return None
        try:
            self._append_history(
                {"event": "delivered", **marker, "delivered_at": self._wall()}
            )
        except OSError:
            logger.warning("cannot record the delivered restart marker", exc_info=True)
        self._discard_marker()
        return marker

    def _discard_marker(self) -> None:
        try:
            self._marker_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "cannot delete restart marker %s", self._marker_path, exc_info=True
            )

    # -- the tool ----------------------------------------------------------

    @property
    def _safe_mode(self) -> bool:
        return bool(self._boot_mode) and self._boot_mode.startswith("safe")

    @property
    def _safe_mode_label(self) -> str:
        return self._boot_mode.removeprefix("safe-").replace("-", " ")

    def _safe_mode_suffix(self) -> str:
        if not self._safe_mode:
            return ""
        blocked = ", ".join(self._blocked) or "none"
        return (
            f"\nSafe mode ({self._safe_mode_label}); "
            f"this boot blocks these plugins: {blocked}"
        )

    def _requested_in_window(self) -> int:
        cutoff = self._wall() - self._window
        return sum(
            1
            for entry in read_jsonl(self._history_path)
            if entry.get("event") == "requested"
            and _is_number(entry.get("requested_at"))
            and entry["requested_at"] > cutoff
        )

    def _loop_guard_refusal(self) -> str | None:
        """Return the reason a restart must be refused now, or None."""
        if not self._user_message_seen:
            return (
                "Refused: no human message since boot. A restart is allowed "
                "only after a human has sent a message to this process."
            )
        uptime = self._monotonic() - self._booted_mono
        if uptime < self._min_uptime:
            return (
                f"Refused: uptime is {uptime:.0f}s, under the minimum of "
                f"{self._min_uptime:g}s since boot."
            )
        requested = self._requested_in_window()
        if requested >= self._max_restarts:
            return (
                f"Refused: {requested} restarts were already requested within "
                f"the {self._window:g}-second window (maximum "
                f"{self._max_restarts}). A human can run systemctl restart."
            )
        return None

    async def _restart(self, reason: str, ctx: ToolContext) -> str:
        """The restart tool. Returns only to refuse; on success it never returns."""
        if ctx.channel is None:
            return "Refused: this tool call has no channel to restart against."
        refusal = self._loop_guard_refusal()
        if refusal:
            return refusal + self._safe_mode_suffix()

        returncode, stderr = await self._preflight(
            self._normal_config, self._preflight_timeout
        )
        if returncode != 0:
            stderr = truncate_stderr(stderr)
            return f"Refused: pre-flight failed.\n{stderr}" + self._safe_mode_suffix()

        marker = {
            "requested_at": self._wall(),
            "reason": reason,
            "channel_id": ctx.channel.id,
            "tool_call_id": ctx.tool_call_id,
            "pid": os.getpid(),
        }
        self._write_marker(marker)
        try:
            self._append_history({"event": "requested", **marker})
        except BaseException:
            try:
                self._marker_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        await self._send_courtesy_message(ctx.channel, reason)

        asyncio.get_running_loop().call_soon(
            self._killer, os.getpid(), signal.SIGTERM
        )
        await asyncio.Event().wait()

    async def _send_courtesy_message(self, channel, reason: str) -> None:
        """Tell the user now; the real reply comes after the restart."""
        text = f"Restarting: {reason}" + self._safe_mode_suffix()
        try:
            await self.pm.ahook.send(
                channel=channel, event=OutboundMessage(text=text)
            )
        except Exception:
            logger.warning("restart courtesy message failed", exc_info=True)

    # -- notifications -----------------------------------------------------

    def _home_notice_reason(self) -> str | None:
        """Why a marker-less boot should notify the home channel, or None."""
        if self._safe_mode:
            return f"safe mode ({self._safe_mode_label})"
        if self._previous_abnormal:
            return "the previous run ended abnormally"
        return None

    async def _wait_until_ready(self) -> None:
        """Wait for the agent's tools, then for the transports to connect."""
        agent = self.pm.get_plugin("agent")
        while not agent.tools:
            await asyncio.sleep(self._ready_poll_seconds)
        await asyncio.sleep(self._notify_delay)

    async def _deliver_notices(self) -> None:
        """Deliver the restart notice, or the home-channel notice."""
        try:
            reason = self._home_notice_reason()
            if not self._marker_path.exists() and reason is None:
                return
            await self._wait_until_ready()
            marker = self._take_marker()
            if marker is not None:
                await self._notify_restart(marker)
            elif reason is not None:
                await self._notify_home(
                    "\n".join([f"Boot notice: {reason}.", *self._summary_lines()])
                )
        except Exception:
            logger.exception("startup notification failed")

    async def _notify_restart(self, marker: dict) -> None:
        """Deliver the marker's result against its originating tool call."""
        registry = self.pm.get_plugin("registry")
        channel_id = marker.get("channel_id")
        channel = registry.get(channel_id)
        requested_at = marker.get("requested_at")
        if not _is_number(requested_at):
            if requested_at is not None:
                logger.warning(
                    "restart marker has a non-numeric requested_at: %r",
                    requested_at,
                )
            requested_at = self._booted_wall
        downtime = max(0.0, self._booted_wall - requested_at)
        lines = [
            f"Restart complete. Reason: {marker.get('reason')}",
            f"Downtime: {downtime:.0f}s",
            *self._summary_lines(marker),
        ]
        if channel is None:
            logger.warning(
                "restart marker channel %s is not registered; "
                "falling back to the home channel",
                channel_id,
            )
            lines.insert(
                1, f"The originating channel {channel_id} is no longer registered."
            )
            await self._notify_home("\n".join(lines))
            return
        await self._notify(channel, "\n".join(lines), marker.get("tool_call_id"))

    async def _notify_home(self, text: str) -> None:
        """Send a notice (no tool call) to the home channel."""
        headline = text.splitlines()[0]
        if not self._home_channel_id:
            logger.info("no home_channel configured; not sending: %s", headline)
            return
        channel = self.pm.get_plugin("registry").get(self._home_channel_id)
        if channel is None:
            logger.warning(
                "home channel %s is not registered; not sending: %s",
                self._home_channel_id,
                headline,
            )
            return
        await self._notify(channel, text, None)

    async def _notify(self, channel, text: str, tool_call_id: str | None) -> None:
        await self.pm.ahook.on_notify(
            channel=channel,
            source="restart",
            text=text,
            tool_call_id=tool_call_id,
            meta=None,
        )

    def _summary_lines(self, marker: dict | None = None) -> list[str]:
        """What this boot is running: plugins and safe mode."""
        lines = [
            f"Plugins loaded: {', '.join(self._loaded) or 'none'}",
            f"Plugins blocked: {', '.join(self._blocked) or 'none'}",
        ]
        if self._safe_mode:
            lines.append(f"Safe mode: {self._safe_mode_label}")
        return lines
