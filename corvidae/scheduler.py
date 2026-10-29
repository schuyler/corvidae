"""SchedulerPlugin — reminders, periodic jobs, and the heartbeat (#59).

The clock-driven producer: one ``schedule`` table in the persistence
database, one asyncio timer task, and three tools. A due row fires as a
plain ``on_notify`` call, so the whole notification pipeline (serial
queue, persistence, origin-keyed consumers, attribution, the outbound send)
runs unchanged; the model's reply on the row's channel is the unprompted
message. Design: plans/scheduler.md.

Rows come from two sources:
    - reminders, created by the ``remind_me`` tool, pinned to the channel
      the tool ran on (there is no channel argument — scoping is a
      security property, see docs/threat-model.md);
    - jobs, declared by the operator under ``scheduler.jobs`` and synced by
      ``name`` on startup and on every config reload.

Config:
    scheduler:
      timezone: America/Los_Angeles   # resolves offset-less times; default UTC
      catch_up_window: 3600           # seconds a missed one-shot may be late and still fire
      max_reminders_per_channel: 50
      jobs:
        - name: heartbeat             # unique; the sync key across reloads
          channel: internal:heartbeat
          origin: heartbeat           # 'reminder' (default) | 'heartbeat'
          every: 6h                   # required
          at: "08:00"                 # optional anchor: HH:MM or ISO datetime
          prompt: "..."               # or prompt_file: prompts/heartbeat.md

Logging:
    - INFO: row fired, config job inserted/updated/disabled, catch-up decisions
    - WARNING: bad timezone, channel resolution failure, hook failures
    - ERROR: invalid job config, timer loop errors
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiosqlite

from corvidae.channel import Channel, ChannelRegistry
from corvidae.clock import MAX_WALL_SLEEP_SECONDS, Clock
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.tool import Tool, tool_to_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults and limits
# ---------------------------------------------------------------------------

DEFAULT_TIMEZONE = "UTC"
DEFAULT_CATCH_UP_WINDOW = 3600.0
DEFAULT_MAX_REMINDERS_PER_CHANNEL = 50

#: Origins a row may carry; each becomes the "[origin]" notification prefix
#: and meta["origin"], which origin-aware plugins key on.
VALID_ORIGINS = ("reminder", "heartbeat")

#: Longest the timer sleeps in one go even with nothing due sooner; the
#: clock owns the cap so a wall-clock jump is noticed within the hour.
MAX_SLEEP_SECONDS = MAX_WALL_SLEEP_SECONDS
#: Re-check interval while the persistence DB or Agent is not up yet.
READY_POLL_SECONDS = 1.0
#: Back-off after an unexpected error in a timer pass.
ERROR_RETRY_SECONDS = 30.0
#: Re-check interval while a due row's owning transport cannot resolve its
#: channel yet (Signal before its first alias resolution, #76).
CHANNEL_RETRY_SECONDS = 5.0
#: Upper bound on reminder text a tool call may store; together with
#: max_reminders_per_channel this bounds what a hostile speaker can park
#: in the table.
MAX_REMINDER_TEXT_CHARS = 2000

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEDULE_DDL = """
CREATE TABLE IF NOT EXISTS schedule (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  channel_id TEXT NOT NULL,
  origin TEXT NOT NULL,
  text TEXT NOT NULL,
  next_fire_at REAL NOT NULL,
  interval_seconds REAL,
  created_at REAL NOT NULL,
  created_by TEXT NOT NULL,
  last_fired_at REAL,
  enabled INTEGER NOT NULL DEFAULT 1,
  disabled_reason TEXT,
  schedule_spec TEXT
)
"""
SCHEDULE_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS schedule_due ON schedule (enabled, next_fire_at)"
)

_ROW_COLUMNS = (
    "id, name, channel_id, origin, text, next_fire_at, interval_seconds, "
    "created_at, created_by, last_fired_at, enabled, disabled_reason, schedule_spec"
)


class ChannelNotReady(Exception):
    """A due row's owning transport cannot build its channel yet; the row
    stays due and the timer retries after CHANNEL_RETRY_SECONDS."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(f"channel {channel_id} cannot be resolved yet")
        self.channel_id = channel_id


@dataclass
class ScheduleRow:
    """One row of the schedule table."""

    id: int
    name: str | None
    channel_id: str
    origin: str
    text: str
    next_fire_at: float
    interval_seconds: float | None
    created_at: float
    created_by: str
    last_fired_at: float | None
    enabled: bool
    disabled_reason: str | None
    schedule_spec: str | None

    @classmethod
    def from_db(cls, row: tuple) -> "ScheduleRow":
        """Build from a SELECT of _ROW_COLUMNS."""
        return cls(
            id=row[0], name=row[1], channel_id=row[2], origin=row[3],
            text=row[4], next_fire_at=row[5], interval_seconds=row[6],
            created_at=row[7], created_by=row[8], last_fired_at=row[9],
            enabled=bool(row[10]), disabled_reason=row[11], schedule_spec=row[12],
        )


# ---------------------------------------------------------------------------
# Time parsing and arithmetic
# ---------------------------------------------------------------------------

# Unit suffixes accepted in durations, mapped to seconds.
_UNIT_SECONDS: dict[str, float] = {}
for _names, _secs in (
    (("s", "sec", "secs", "second", "seconds"), 1.0),
    (("m", "min", "mins", "minute", "minutes"), 60.0),
    (("h", "hr", "hrs", "hour", "hours"), 3600.0),
    (("d", "day", "days"), 86400.0),
    (("w", "wk", "wks", "week", "weeks"), 604800.0),
):
    for _name in _names:
        _UNIT_SECONDS[_name] = _secs

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]+)")
_DURATION_FULL = re.compile(r"^(?:\d+(?:\.\d+)?\s*[a-z]+\s*,?\s*)+$")
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?$")

WHEN_FORMS_HELP = (
    "a relative duration (20m, 2h, 3d, 1h30m), an ISO 8601 time "
    "(2026-10-24T09:00 or 2026-10-24T09:00-07:00), or HH:MM for the next occurrence"
)


def parse_duration(text: str | int | float) -> float:
    """Parse a duration like ``20m``, ``2h``, ``1h30m``, ``3 days`` to seconds.

    A bare number is seconds. Raises ValueError on anything else, or on a
    non-positive result.
    """
    # Bare numbers (YAML gives ints/floats for `every: 3600`) are seconds.
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        seconds = float(text)
    else:
        cleaned = str(text).strip().lower()
        if cleaned.startswith("in "):
            cleaned = cleaned[3:].strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
            seconds = float(cleaned)
        elif _DURATION_FULL.match(cleaned):
            seconds = 0.0
            for amount, unit in _DURATION_PART.findall(cleaned):
                if unit not in _UNIT_SECONDS:
                    raise ValueError(f"unknown duration unit {unit!r} in {text!r}")
                seconds += float(amount) * _UNIT_SECONDS[unit]
        else:
            raise ValueError(f"not a duration: {text!r}")
    if seconds <= 0:
        raise ValueError(f"duration must be positive: {text!r}")
    return seconds


def _next_wall_clock(now: float, hour: int, minute: int, second: int, tz: tzinfo) -> float:
    """The next instant strictly after ``now`` whose wall clock in ``tz``
    reads hour:minute:second."""
    local_now = datetime.fromtimestamp(now, tz)
    candidate = local_now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    # Already passed (or exactly now) today: the next occurrence is tomorrow,
    # computed in wall-clock time so it stays HH:MM across a DST change.
    if candidate.timestamp() <= now:
        candidate = candidate + timedelta(days=1)
    return candidate.timestamp()


def parse_when(when: str, now: float, tz: tzinfo) -> float:
    """Resolve a reminder's ``when`` to an absolute unix time.

    Accepts a relative duration, an ISO 8601 timestamp (no offset means
    ``tz``), or a bare ``HH:MM`` meaning the next occurrence in ``tz``.
    The tool does this arithmetic, not the model: small models are
    unreliable at dates. Raises ValueError naming the accepted forms.
    """
    text = (when or "").strip()
    if not text:
        raise ValueError(f"empty time; use {WHEN_FORMS_HELP}")

    # Bare HH:MM[:SS] — the next occurrence in the configured timezone.
    match = _HHMM.match(text)
    if match:
        return _next_wall_clock(
            now, int(match.group(1)), int(match.group(2)), int(match.group(3) or 0), tz,
        )

    # A bare number is ambiguous (seconds? minutes?) — make the caller say.
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        raise ValueError(f"{when!r} has no unit; use {WHEN_FORMS_HELP}")

    # Relative duration from now.
    try:
        return now + parse_duration(text)
    except ValueError:
        pass

    # ISO 8601; a naive timestamp is read in the configured timezone.
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"could not understand {when!r}; use {WHEN_FORMS_HELP}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed.timestamp()


def advance(scheduled: float, interval: float, now: float, tz: tzinfo) -> float:
    """The first slot ``scheduled + k*interval`` strictly after ``now``.

    Computed from the scheduled time, never the actual fire time, so drift
    never accumulates; a row more than one interval behind skips to the
    next future slot rather than replaying every missed one. Steps are
    taken in wall-clock time in ``tz`` (aware-datetime arithmetic), so an
    ``every: 1d`` job anchored at 08:30 stays at 08:30 across DST.
    """
    base = datetime.fromtimestamp(scheduled, tz)
    step = timedelta(seconds=interval)
    # Jump close to the answer in one step, then walk the last slot or two;
    # keeps a long outage from costing one iteration per missed slot.
    k = max(1, int((now - scheduled) // interval))
    candidate = base + step * k
    while candidate.timestamp() <= now:
        k += 1
        candidate = base + step * k
    # A DST transition can shift wall-clock arithmetic by up to an hour;
    # step back while the previous slot is still in the future.
    while k > 1 and (base + step * (k - 1)).timestamp() > now:
        k -= 1
        candidate = base + step * k
    return candidate.timestamp()


def format_time(ts: float, tz: tzinfo) -> str:
    """Render a unix time in ``tz``, e.g. ``Fri 2026-10-23 14:22 PDT``."""
    return datetime.fromtimestamp(ts, tz).strftime("%a %Y-%m-%d %H:%M %Z")


def format_duration(seconds: float) -> str:
    """Render a span compactly, e.g. ``42m``, ``3h 5m``, ``2d 4h``, ``45s``."""
    total = int(round(max(seconds, 0)))
    if total < 60:
        return f"{total}s"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and not days:
        parts.append(f"{minutes}m")
    return " ".join(parts) or "0m"


def _quote_as_data(text: str) -> str:
    """Wrap stored text in «» for the firing frame, neutralizing any «/»
    inside it so the text cannot close the quote and continue as framing."""
    return "«" + text.replace("«", '"').replace("»", '"') + "»"


def frame_firing(row: ScheduleRow, tz: tzinfo, overdue: float | None) -> str:
    """Build the notification text for a due row.

    Reminder text is framed as data, not instruction: it is text the agent
    wrote down from a possibly untrusted speaker, re-entering later in a
    system-role row (threat-model finding 3, with a delay). The wording
    reads correctly whether it renders as a system message or, under
    ``agent.mid_conversation_system_role: user``, as the last user turn.
    ``overdue`` (seconds) adds the stale marker for a row the catch-up
    policy fires late after the daemon was down.
    """
    if row.created_by == "config":
        every = format_duration(row.interval_seconds) if row.interval_seconds else "once"
        body = (
            f"Scheduled job \"{row.name}\" (set by the operator, every {every}) is due. "
            f"Job prompt: {_quote_as_data(row.text)}. "
            "Decide whether and how to act on it."
        )
    else:
        created = datetime.fromtimestamp(row.created_at, tz).strftime("%Y-%m-%d %H:%M (%Z)")
        body = (
            f"A reminder you set on {created} for this conversation is due. "
            f"Reminder text: {_quote_as_data(row.text)}. "
            "Decide whether and how to act on it."
        )
    if overdue is not None:
        return f"[overdue by {format_duration(overdue)}, daemon was down]\n{body}"
    return body


# ---------------------------------------------------------------------------
# Config jobs
# ---------------------------------------------------------------------------


@dataclass
class JobSpec:
    """A validated ``scheduler.jobs`` entry."""

    name: str
    channel_id: str
    origin: str
    interval_seconds: float
    at: str | None
    text: str

    def schedule_spec(self, tz_name: str) -> str:
        """Canonical string of everything that determines the schedule, so
        a reload recomputes next_fire_at only when it actually changed."""
        return f"every={self.interval_seconds!r};at={self.at or ''};tz={tz_name}"

    def first_fire(self, now: float, tz: tzinfo) -> float:
        """The first fire time for a newly (re)scheduled job.

        No anchor: one interval from now. ``HH:MM``: the next occurrence.
        ISO datetime: the anchor itself if future, else the first slot
        ``anchor + k*every`` after now (``every: 7d`` anchored on a Friday
        09:00 is "every Friday at 9am").
        """
        if self.at is None:
            return now + self.interval_seconds
        match = _HHMM.match(self.at)
        if match:
            return _next_wall_clock(
                now, int(match.group(1)), int(match.group(2)), int(match.group(3) or 0), tz,
            )
        anchor = datetime.fromisoformat(self.at)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=tz)
        anchor_ts = anchor.timestamp()
        if anchor_ts > now:
            return anchor_ts
        return advance(anchor_ts, self.interval_seconds, now, tz)


def parse_job(raw: object, base_dir: Path) -> JobSpec:
    """Validate one ``scheduler.jobs`` entry. Raises ValueError naming the
    problem; the caller logs it and leaves any existing row untouched."""
    if not isinstance(raw, dict):
        raise ValueError(f"job must be a mapping, got {type(raw).__name__}")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("job is missing required field 'name'")
    channel_id = raw.get("channel")
    if not isinstance(channel_id, str):
        raise ValueError(f"job {name!r} is missing required field 'channel'")
    transport, sep, scope = channel_id.partition(":")
    if not sep or not transport or not scope:
        raise ValueError(f"job {name!r} channel {channel_id!r} is not 'transport:scope'")
    origin = raw.get("origin", "reminder")
    if origin not in VALID_ORIGINS:
        raise ValueError(f"job {name!r} origin {origin!r} is not one of {', '.join(VALID_ORIGINS)}")
    if raw.get("every") is None:
        raise ValueError(f"job {name!r} is missing required field 'every'")
    interval = parse_duration(raw["every"])
    at = raw.get("at")
    if at is not None:
        at = str(at).strip()
        if not _HHMM.match(at):
            try:
                datetime.fromisoformat(at)
            except ValueError:
                raise ValueError(f"job {name!r} 'at' {at!r} is not HH:MM or an ISO datetime") from None

    # The objective text comes inline or from a file relative to agent.yaml.
    prompt = raw.get("prompt")
    prompt_file = raw.get("prompt_file")
    if prompt is not None and prompt_file is not None:
        raise ValueError(f"job {name!r} sets both 'prompt' and 'prompt_file'")
    if prompt_file is not None:
        path = Path(prompt_file)
        if not path.is_absolute():
            path = base_dir / path
        try:
            prompt = path.read_text().strip()
        except OSError as exc:
            raise ValueError(f"job {name!r} prompt_file {str(path)!r} unreadable: {exc}") from exc
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"job {name!r} needs a non-empty 'prompt' or 'prompt_file'")

    return JobSpec(
        name=name.strip(), channel_id=channel_id, origin=origin,
        interval_seconds=interval, at=at, text=prompt.strip(),
    )


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class SchedulerPlugin(CorvidaePlugin):
    """Clock-driven producer of reminder and heartbeat notifications.

    Attributes:
        _clock: The kernel clock, resolved from the plugin manager in on_init.
        _wake: Set whenever a row is added, cancelled or re-synced, so the
            timer re-arms on the new earliest row immediately.
        _started_up: Startup work (schema, job sync, catch-up) has run.
            Deferred until the persistence DB is open, which on_start
            ordering does not guarantee.
        _stale_overdue: row id -> seconds overdue, for rows the startup
            catch-up policy decided to fire late; consumed on firing.
        _deferred_rows: row ids whose channel the owning transport could
            not resolve yet; each is warned about once, not every retry.
    """

    depends_on = frozenset({"registry", "persistence", "clock"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        # Set by on_init, once the clock plugin is registered.
        self._clock: Clock | None = None
        self._registry: ChannelRegistry | None = None
        self._base_dir: Path = Path(".")

        # Config, populated by on_init and refreshed by on_config_reload.
        self._tz_name: str = DEFAULT_TIMEZONE
        self._tz: tzinfo = timezone.utc
        self._catch_up_window: float = DEFAULT_CATCH_UP_WINDOW
        self._max_per_channel: int = DEFAULT_MAX_REMINDERS_PER_CHANNEL
        self._jobs_raw: object = []

        # Runtime state.
        self._schema_ready = False
        self._started_up = False
        self._startup_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stale_overdue: dict[int, float] = {}
        self._deferred_rows: set[int | None] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read the scheduler: block."""
        await super().on_init(pm, config)
        # Strict lookup: the scheduler cannot run without the kernel clock.
        self._clock = get_dependency(self.pm, "clock", Clock)
        self._apply_config(config)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Resolve dependencies, run startup work if the DB is open, and
        launch the timer task (which retries startup if it is not)."""
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)
        self._base_dir = config.get("_base_dir", self._base_dir)
        try:
            await self._startup_if_ready()
        except Exception:
            logger.warning("scheduler startup deferred to the timer task", exc_info=True)
        self._task = asyncio.create_task(self._timer_loop())

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the timer task. Firings already enqueued drain with the
        channel queues."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("scheduler: timer task raised during shutdown", exc_info=True)

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Re-read config, re-sync jobs by name, and wake the timer so the
        change takes effect immediately."""
        self._apply_config(config)
        if self._started_up:
            await self._sync_jobs()
        self.wake()

    def _apply_config(self, config: dict) -> None:
        """Extract scheduler settings from a full config dict."""
        sched = config.get("scheduler") or {}
        if not isinstance(sched, dict):
            logger.error("scheduler: config block must be a mapping; using defaults")
            sched = {}
        self._base_dir = config.get("_base_dir", self._base_dir)

        # Timezone: an unknown name degrades to UTC rather than failing startup.
        tz_name = sched.get("timezone", DEFAULT_TIMEZONE)
        try:
            self._tz = ZoneInfo(str(tz_name))
            self._tz_name = str(tz_name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("scheduler.timezone %r is unknown; using UTC", tz_name)
            self._tz = timezone.utc
            self._tz_name = DEFAULT_TIMEZONE

        self._catch_up_window = float(sched.get("catch_up_window", DEFAULT_CATCH_UP_WINDOW))
        self._max_per_channel = int(
            sched.get("max_reminders_per_channel", DEFAULT_MAX_REMINDERS_PER_CHANNEL)
        )
        self._jobs_raw = sched.get("jobs", []) if sched.get("jobs") is not None else []

    def wake(self) -> None:
        """Wake the timer so it re-reads the earliest due row."""
        self._wake.set()

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _resolve_db(self) -> aiosqlite.Connection | None:
        """Return the persistence plugin's connection, or None if not open yet."""
        from corvidae.persistence import PersistencePlugin
        persistence = get_dependency(self.pm, "persistence", PersistencePlugin)
        return persistence.db

    async def _ensure_schema(self) -> aiosqlite.Connection | None:
        """Create the schedule table on first use; None while the DB is closed.

        Lazy because the on_start broadcast runs plugins concurrently, so
        PersistencePlugin may not have opened the DB during ours.
        """
        db = self._resolve_db()
        if db is None:
            return None
        if not self._schema_ready:
            await db.execute(SCHEDULE_DDL)
            await db.execute(SCHEDULE_INDEX_DDL)
            await db.commit()
            self._schema_ready = True
        return db

    async def _fetch(self, db: aiosqlite.Connection, where: str, params: tuple) -> list[ScheduleRow]:
        """SELECT rows matching a WHERE clause (a code constant, never input)."""
        async with db.execute(f"SELECT {_ROW_COLUMNS} FROM schedule WHERE {where}", params) as cursor:
            return [ScheduleRow.from_db(r) for r in await cursor.fetchall()]

    # ------------------------------------------------------------------
    # Startup: job sync and catch-up
    # ------------------------------------------------------------------

    async def _startup_if_ready(self) -> bool:
        """Run startup work once the DB is open; True once it has run.

        Startup syncs config jobs and applies the catch-up policy. It never
        fires anything itself: firing waits for the timer's readiness gate.
        """
        if self._started_up:
            return True
        async with self._startup_lock:
            if self._started_up:
                return True
            db = await self._ensure_schema()
            if db is None:
                return False
            await self._sync_jobs()
            await self._catch_up()
            self._started_up = True
        return True

    async def _sync_jobs(self) -> None:
        """Sync ``scheduler.jobs`` into the table by name.

        New names insert; existing names update text/channel/origin, and
        recompute next_fire_at only when the schedule spec changed or the
        row was disabled; names no longer configured are disabled with
        ``removed_from_config``, a durable stop that survives restart. An
        invalid entry is logged and its existing row (if any) left as is,
        so a typo does not silently stop a running job.
        """
        db = await self._ensure_schema()
        if db is None:
            return
        if not isinstance(self._jobs_raw, list):
            logger.error("scheduler.jobs must be a list; job sync skipped")
            return

        # Validate every entry; invalid names are protected from removal.
        specs: dict[str, JobSpec] = {}
        protected: set[str] = set()
        for raw in self._jobs_raw:
            try:
                spec = parse_job(raw, Path(self._base_dir))
            except ValueError as exc:
                logger.error("scheduler: invalid job skipped: %s", exc)
                if isinstance(raw, dict) and isinstance(raw.get("name"), str):
                    protected.add(raw["name"].strip())
                continue
            if spec.name in specs:
                logger.error("scheduler: duplicate job name %r; keeping the first", spec.name)
                continue
            specs[spec.name] = spec

        now = self._clock.now()
        async with self._db_lock:
            existing = {r.name: r for r in await self._fetch(db, "created_by = 'config'", ())}

            # Insert or update every configured job.
            for name, spec in specs.items():
                schedule_spec = spec.schedule_spec(self._tz_name)
                row = existing.get(name)
                if row is None:
                    next_fire = spec.first_fire(now, self._tz)
                    await db.execute(
                        "INSERT INTO schedule (name, channel_id, origin, text, next_fire_at, "
                        "interval_seconds, created_at, created_by, enabled, schedule_spec) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, 'config', 1, ?)",
                        (name, spec.channel_id, spec.origin, spec.text, next_fire,
                         spec.interval_seconds, now, schedule_spec),
                    )
                    logger.info(
                        "scheduler: config job inserted",
                        extra={"job": name, "next_fire_at": format_time(next_fire, self._tz)},
                    )
                    continue
                reschedule = row.schedule_spec != schedule_spec or not row.enabled
                next_fire = spec.first_fire(now, self._tz) if reschedule else row.next_fire_at
                await db.execute(
                    "UPDATE schedule SET channel_id = ?, origin = ?, text = ?, "
                    "interval_seconds = ?, next_fire_at = ?, schedule_spec = ?, "
                    "enabled = 1, disabled_reason = NULL WHERE id = ?",
                    (spec.channel_id, spec.origin, spec.text, spec.interval_seconds,
                     next_fire, schedule_spec, row.id),
                )
                if reschedule:
                    self._stale_overdue.pop(row.id, None)
                    logger.info(
                        "scheduler: config job rescheduled",
                        extra={"job": name, "next_fire_at": format_time(next_fire, self._tz)},
                    )

            # Disable jobs that are no longer configured (and not merely invalid).
            for name, row in existing.items():
                if name in specs or name in protected or not row.enabled:
                    continue
                await db.execute(
                    "UPDATE schedule SET enabled = 0, disabled_reason = 'removed_from_config' "
                    "WHERE id = ?",
                    (row.id,),
                )
                self._stale_overdue.pop(row.id, None)
                logger.info("scheduler: config job disabled (removed from config)", extra={"job": name})
            await db.commit()

    async def _catch_up(self) -> None:
        """Apply the startup catch-up policy to rows that came due while the
        daemon was down.

        One-shot: overdue by less than ``catch_up_window`` fires once with
        the stale marker; older is disabled as ``missed``. Recurring: less
        than one interval behind fires once with the marker; further behind
        skips to the next future slot without firing.
        """
        db = await self._ensure_schema()
        if db is None:
            return
        now = self._clock.now()
        async with self._db_lock:
            overdue_rows = await self._fetch(db, "enabled = 1 AND next_fire_at <= ?", (now,))
            for row in overdue_rows:
                overdue = now - row.next_fire_at
                if row.interval_seconds is None:
                    # One-shot reminder: fire late within the window, else missed.
                    if overdue < self._catch_up_window:
                        self._stale_overdue[row.id] = overdue
                        logger.info("scheduler: overdue reminder will fire late",
                                    extra={"schedule_id": row.id, "overdue_s": round(overdue)})
                    else:
                        await db.execute(
                            "UPDATE schedule SET enabled = 0, disabled_reason = 'missed' WHERE id = ?",
                            (row.id,),
                        )
                        logger.info("scheduler: reminder missed while daemon was down",
                                    extra={"schedule_id": row.id, "overdue_s": round(overdue)})
                elif overdue < row.interval_seconds:
                    # Recurring, less than one slot behind: fire the missed slot once.
                    self._stale_overdue[row.id] = overdue
                else:
                    # Recurring, more than a slot behind: skip ahead, no replay.
                    next_fire = advance(row.next_fire_at, row.interval_seconds, now, self._tz)
                    await db.execute(
                        "UPDATE schedule SET next_fire_at = ? WHERE id = ?", (next_fire, row.id),
                    )
                    logger.info("scheduler: recurring job skipped ahead after downtime",
                                extra={"schedule_id": row.id,
                                       "next_fire_at": format_time(next_fire, self._tz)})
            await db.commit()

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _agent_ready(self) -> bool:
        """True when no Agent is registered or it has finished on_start.

        Runtime starts Agent after the on_start broadcast this plugin's
        timer task begins in; firing before then would enqueue onto a
        queue whose consumer has no registry yet.
        """
        agent = self.pm.get_plugin("agent")
        return agent is None or bool(getattr(agent, "started", True))

    async def _timer_loop(self) -> None:
        """Sleep until the earliest enabled row is due, fire, repeat.

        The wake event is cleared before each pass, so a row added during
        a pass makes the following sleep return at once. ``except
        Exception`` deliberately lets CancelledError (on_stop) through.
        """
        while True:
            self._wake.clear()
            try:
                delay = await self._tick()
            except Exception:
                logger.exception("scheduler: timer pass failed; retrying")
                delay = ERROR_RETRY_SECONDS
            await self._clock.sleep(delay, self._wake)

    async def _tick(self) -> float:
        """One timer pass: fire everything due, return seconds to sleep."""
        # Gate: the DB must be open (startup work done) and Agent started.
        if not await self._startup_if_ready() or not self._agent_ready():
            return READY_POLL_SECONDS
        db = await self._ensure_schema()
        if db is None:
            return READY_POLL_SECONDS

        now = self._clock.now()
        due = await self._fetch(db, "enabled = 1 AND next_fire_at <= ? ORDER BY next_fire_at", (now,))
        failed = False
        deferred = False
        for row in due:
            try:
                await self._fire(db, row, now)
            except ChannelNotReady:
                deferred = True
            except Exception:
                failed = True
                logger.exception("scheduler: firing failed", extra={"schedule_id": row.id})

        # A firing that failed before its row was advanced leaves the row
        # due; re-arming at zero would spin, so back off instead.
        if failed:
            return ERROR_RETRY_SECONDS
        # A deferred row is still due; check back soon for its channel.
        if deferred:
            return CHANNEL_RETRY_SECONDS

        # Re-arm on the earliest remaining enabled row.
        async with db.execute(
            "SELECT MIN(next_fire_at) FROM schedule WHERE enabled = 1"
        ) as cursor:
            (earliest,) = await cursor.fetchone()
        if earliest is None:
            return MAX_SLEEP_SECONDS
        return min(max(earliest - self._clock.now(), 0.0), MAX_SLEEP_SECONDS)

    async def _fire(self, db: aiosqlite.Connection, row: ScheduleRow, now: float) -> None:
        """Fire one due row through on_notify.

        The row is advanced (recurring) or disabled as ``fired`` (one-shot)
        before the notification is sent: at-most-once, so a crash between
        the two can lose a firing but never repeats one on restart.
        """
        # Resolve the target channel before anything is recorded, so a row
        # whose transport cannot build the channel yet stays due (raises
        # ChannelNotReady) rather than being consumed; an unresolvable id
        # is logged and dropped below, after the row is recorded.
        channel = await self._channel_for(row.channel_id, schedule_id=row.id)
        overdue = self._stale_overdue.pop(row.id, None)

        # Record the firing first. The update is conditional on the row
        # still being enabled, so a cancel or config removal that landed
        # after the due rows were fetched wins and nothing fires.
        async with self._db_lock:
            if row.interval_seconds is None:
                cursor = await db.execute(
                    "UPDATE schedule SET enabled = 0, disabled_reason = 'fired', "
                    "last_fired_at = ? WHERE id = ? AND enabled = 1",
                    (now, row.id),
                )
            else:
                next_fire = advance(row.next_fire_at, row.interval_seconds, now, self._tz)
                cursor = await db.execute(
                    "UPDATE schedule SET next_fire_at = ?, last_fired_at = ? "
                    "WHERE id = ? AND enabled = 1",
                    (next_fire, now, row.id),
                )
            await db.commit()
        if cursor.rowcount == 0:
            logger.info("scheduler: row disabled before it fired; skipped",
                        extra={"schedule_id": row.id})
            return

        # An unresolvable id is logged and dropped.
        if channel is None:
            logger.warning("scheduler: cannot resolve channel; firing dropped",
                           extra={"schedule_id": row.id, "channel": row.channel_id})
            return

        # Touch so last_active, which orders the web and ACP channel lists,
        # reflects the firing.
        channel.touch()
        logger.info(
            "scheduler: firing",
            extra={"schedule_id": row.id, "channel": channel.id, "origin": row.origin,
                   "overdue_s": round(overdue) if overdue is not None else None},
        )
        await self.pm.ahook.on_notify(
            channel=channel,
            source=row.origin,
            text=frame_firing(row, self._tz, overdue),
            tool_call_id=None,
            meta={"origin": row.origin, "schedule_id": row.id},
        )

    async def _channel_for(self, channel_id: str, schedule_id: int | None = None) -> Channel | None:
        """Resolve a persisted channel id: live registry entry, then the
        transport's resolve_channel (Signal applies alias config), then a
        bare registry channel from the split id (IRC, CLI, internal:*).

        Raises ChannelNotReady when the transport that owns the id
        implements resolve_channel but declined: it cannot apply the
        channel's config yet (PRD §3.2.7), and a bare channel would cache
        the defaults for the process lifetime (#76).
        """
        channel = self._registry.get(channel_id)
        if channel is not None:
            self._deferred_rows.discard(schedule_id)
            return channel
        try:
            channel = await self.pm.ahook.resolve_channel(channel_id=channel_id)
        except Exception:
            logger.warning("resolve_channel hook failed", exc_info=True,
                           extra={"channel": channel_id})
            channel = None
        if channel is not None:
            self._deferred_rows.discard(schedule_id)
            return channel
        transport, sep, scope = channel_id.partition(":")
        if not sep or not transport or not scope:
            return None
        if self._transport_resolves(transport):
            # Warn once per row, not once per retry pass.
            if schedule_id not in self._deferred_rows:
                self._deferred_rows.add(schedule_id)
                logger.warning(
                    "scheduler: %s transport cannot resolve %s yet; holding schedule "
                    "row %s and retrying every %ss",
                    transport, channel_id, schedule_id, CHANNEL_RETRY_SECONDS,
                    extra={"schedule_id": schedule_id, "channel": channel_id},
                )
            raise ChannelNotReady(channel_id)
        return self._registry.get_or_create(transport, scope)

    def _transport_resolves(self, transport: str) -> bool:
        """Whether a loaded plugin for ``transport`` implements
        resolve_channel, so its None means "not yet" rather than "not
        mine". Transports set a ``transport`` attribute (TransportPlugin)."""
        return any(
            getattr(impl.plugin, "transport", None) == transport
            for impl in self.pm.hook.resolve_channel.get_hookimpls()
        )

    # ------------------------------------------------------------------
    # Reminder operations (the tools' backing)
    # ------------------------------------------------------------------

    async def add_reminder(self, channel_id: str, text: str, fire_at: float) -> int:
        """Insert a one-shot reminder; raises ValueError over the channel cap
        or when the DB is not open."""
        db = await self._ensure_schema()
        if db is None:
            raise ValueError("the schedule database is not available yet")
        async with self._db_lock:
            # Cap check and insert under one lock so concurrent tool calls
            # in a batch cannot both slip under the limit.
            async with db.execute(
                "SELECT COUNT(*) FROM schedule WHERE channel_id = ? AND enabled = 1 "
                "AND created_by = 'tool'",
                (channel_id,),
            ) as cursor:
                (count,) = await cursor.fetchone()
            if count >= self._max_per_channel:
                raise ValueError(
                    f"this conversation already has {count} pending reminders "
                    f"(limit {self._max_per_channel}); cancel one first"
                )
            cursor = await db.execute(
                "INSERT INTO schedule (channel_id, origin, text, next_fire_at, created_at, "
                "created_by, enabled) VALUES (?, 'reminder', ?, ?, ?, 'tool', 1)",
                (channel_id, text, fire_at, self._clock.now()),
            )
            row_id = cursor.lastrowid
            await db.commit()
        self.wake()
        return row_id

    async def list_reminders_for(self, channel_id: str) -> list[ScheduleRow]:
        """Enabled rows on one channel, earliest first."""
        db = await self._ensure_schema()
        if db is None:
            return []
        return await self._fetch(
            db, "channel_id = ? AND enabled = 1 ORDER BY next_fire_at", (channel_id,),
        )

    async def cancel_reminder_on(self, row_id: int, channel_id: str) -> str:
        """Disable a tool-created row on ``channel_id``; returns the tool's
        reply. Refuses config jobs and rows on other channels; another
        channel's row reads as not found so its existence is not leaked."""
        db = await self._ensure_schema()
        if db is None:
            return "Error: the schedule database is not available yet."
        async with self._db_lock:
            rows = await self._fetch(db, "id = ?", (row_id,))
            row = rows[0] if rows else None
            if row is None or row.channel_id != channel_id:
                return f"Error: no reminder #{row_id} in this conversation."
            if row.created_by == "config":
                return (f"Error: #{row_id} is the operator's scheduled job "
                        f"\"{row.name}\" and cannot be cancelled from here.")
            if not row.enabled:
                return f"Reminder #{row_id} is not pending ({row.disabled_reason})."
            await db.execute(
                "UPDATE schedule SET enabled = 0, disabled_reason = 'cancelled' WHERE id = ?",
                (row_id,),
            )
            await db.commit()
            self._stale_overdue.pop(row_id, None)
        self.wake()
        return f"Cancelled reminder #{row_id}."

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        """Register remind_me, list_reminders, cancel_reminder.

        All three act on ``_ctx.channel`` only; there is no channel
        argument, so a speaker on one transport cannot schedule a message
        to another (docs/threat-model.md).
        """
        plugin = self

        async def remind_me(when: str, text: str, _ctx) -> str:
            """Set a one-shot reminder in this conversation. `when` is a relative duration (20m, 2h, 3d, 1h30m), an ISO 8601 time (2026-10-24T09:00, optional offset), or HH:MM for the next occurrence; times without an offset use the configured timezone. Returns the resolved time to confirm to the person."""
            channel = _ctx.channel
            if channel is None:
                return "Error: no channel context available"
            text = (text or "").strip()
            if not text:
                return "Error: reminder text is empty"
            if len(text) > MAX_REMINDER_TEXT_CHARS:
                return f"Error: reminder text is longer than {MAX_REMINDER_TEXT_CHARS} characters"
            now = plugin._clock.now()
            try:
                fire_at = parse_when(when, now, plugin._tz)
            except ValueError as exc:
                return f"Error: {exc}"
            if fire_at <= now:
                return (f"Error: {format_time(fire_at, plugin._tz)} is in the past; "
                        f"it is now {format_time(now, plugin._tz)}")
            try:
                row_id = await plugin.add_reminder(channel.id, text, fire_at)
            except ValueError as exc:
                return f"Error: {exc}"
            logger.info("scheduler: reminder set",
                        extra={"schedule_id": row_id, "channel": channel.id})
            return (f"Reminder #{row_id} set for {format_time(fire_at, plugin._tz)} "
                    f"(in {format_duration(fire_at - now)}).")

        async def list_reminders(_ctx) -> str:
            """List the pending reminders and scheduled jobs in this conversation, with their ids and due times. Scheduled jobs set by the operator are read-only."""
            channel = _ctx.channel
            if channel is None:
                return "Error: no channel context available"
            rows = await plugin.list_reminders_for(channel.id)
            if not rows:
                return "No pending reminders in this conversation."
            lines = []
            for row in rows:
                due = format_time(row.next_fire_at, plugin._tz)
                if row.created_by == "config":
                    every = format_duration(row.interval_seconds or 0)
                    preview = row.text if len(row.text) <= 80 else row.text[:77] + "..."
                    lines.append(f"#{row.id} {due} [operator job \"{row.name}\", every {every}, "
                                 f"read-only] {preview}")
                else:
                    lines.append(f"#{row.id} {due}: {row.text}")
            return "\n".join(lines)

        async def cancel_reminder(id: int, _ctx) -> str:  # noqa: A002 - tool arg name per plan
            """Cancel a pending reminder in this conversation by its id (from list_reminders). Operator scheduled jobs cannot be cancelled."""
            channel = _ctx.channel
            if channel is None:
                return "Error: no channel context available"
            # Tool args are not type-checked; accept "12" and "#12" as well.
            try:
                row_id = int(str(id).strip().lstrip("#"))
            except ValueError:
                return f"Error: {id!r} is not a reminder id; use the number from list_reminders"
            return await plugin.cancel_reminder_on(row_id, channel.id)

        for fn in (remind_me, list_reminders, cancel_reminder):
            tool_registry.append(Tool(name=fn.__name__, fn=fn, schema=tool_to_schema(fn)))
