"""5-field cron expression parser and next-fire-time calculator.

Fields: minute (0-59), hour (0-23), day-of-month (1-31), month (1-12),
day-of-week (0-6, 0=Sunday).

Supports: * (any), N (specific), N-M (range), */N (step), N,M,K (list).
Combinations like 1-5/2 are also supported (range with step).

Timezone-aware: all calculations use a ZoneInfo.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

# Field bounds: (min, max) for each of the 5 fields.
_FIELD_BOUNDS = [
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 6),    # day of week (0=Sunday)
]

_FIELD_NAMES = ["minute", "hour", "day-of-month", "month", "day-of-week"]


class CronParseError(ValueError):
    """Raised when a cron expression cannot be parsed."""


def _parse_field(field: str, lo: int, hi: int, name: str) -> set[int]:
    """Parse one cron field into a set of allowed values."""
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise CronParseError(f"empty element in {name} field: {field!r}")

        # Split off step: "N/M" or "*/M" or "N-M/M"
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            try:
                step = int(step_str)
            except ValueError:
                raise CronParseError(f"invalid step {step_str!r} in {name} field")
            if step < 1:
                raise CronParseError(f"step must be >= 1 in {name} field, got {step}")

        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            range_parts = part.split("-", 1)
            try:
                start = int(range_parts[0])
                end = int(range_parts[1])
            except ValueError:
                raise CronParseError(f"invalid range {part!r} in {name} field")
            if start < 0 or end < 0:
                raise CronParseError(f"negative value in {name} field: {part!r}")
            if start > end:
                raise CronParseError(f"range start > end in {name} field: {part!r}")
        else:
            try:
                start = int(part)
            except ValueError:
                raise CronParseError(f"invalid value {part!r} in {name} field")
            if start < 0:
                raise CronParseError(f"negative value in {name} field: {part!r}")
            end = start

        if start < lo or end > hi:
            raise CronParseError(
                f"{name} value out of range [{lo}, {hi}]: {part!r}"
            )

        values.update(range(start, end + 1, step))

    return values


def parse_cron(expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    """Parse a 5-field cron expression into sets of allowed values.

    Returns (minutes, hours, days_of_month, months, days_of_week).
    Raises CronParseError on invalid input.
    """
    fields = expr.split()
    if len(fields) != 5:
        raise CronParseError(
            f"expected 5 fields, got {len(fields)}: {expr!r}"
        )

    sets = []
    for i, field in enumerate(fields):
        lo, hi = _FIELD_BOUNDS[i]
        sets.append(_parse_field(field, lo, hi, _FIELD_NAMES[i]))

    return tuple(sets)  # type: ignore[return-value]


def next_fire(
    cron: tuple[set[int], set[int], set[int], set[int], set[int]],
    after: datetime,
    tz: ZoneInfo,
) -> datetime:
    """Compute the next fire time strictly after `after`.

    `after` must be timezone-aware (in `tz`). Returns a timezone-aware
    datetime in `tz`.

    Stepping is done in UTC to avoid the `fold` reset that wall-clock
    `timedelta` arithmetic causes during DST transitions. Each candidate
    is converted back to `tz` for field checks.

    Search bounded to 4 years ahead (covers Feb 29 edge cases).
    """
    minutes, hours, doms, months, dows = cron

    # Start from the next minute, in UTC.
    t_utc = (after + timedelta(minutes=1)).replace(second=0, microsecond=0).astimezone(UTC)

    # Bounded search: 4 years * 366 days * 24 hours * 60 minutes ≈ 2.1M iterations
    # In practice we skip whole days/hours that don't match, so this is fast.
    limit_utc = t_utc + timedelta(days=4 * 366)

    while t_utc < limit_utc:
        # Convert to local time for field checks.
        t_local = t_utc.astimezone(tz)

        if t_local.month not in months:
            # Advance to the 1st of the next month (in local time), then back to UTC.
            if t_local.month == 12:
                next_local = t_local.replace(year=t_local.year + 1, month=1, day=1, hour=0, minute=0)
            else:
                next_local = t_local.replace(month=t_local.month + 1, day=1, hour=0, minute=0)
            t_utc = next_local.astimezone(UTC)
            continue

        # Day-of-week: Python's weekday() is 0=Monday..6=Sunday;
        # cron is 0=Sunday..6=Saturday. Convert.
        cron_dow = (t_local.weekday() + 1) % 7  # 0=Sun, 1=Mon, ..., 6=Sat
        dow_match = cron_dow in dows
        dom_match = t_local.day in doms

        # Standard cron semantics: if both dom and dow are restricted
        # (not *), the day matches if EITHER matches. If only one is
        # restricted, that one must match.
        dom_star = doms == set(range(1, 32))
        dow_star = dows == set(range(0, 7))
        if dom_star and dow_star:
            day_match = True
        elif dom_star:
            day_match = dow_match
        elif dow_star:
            day_match = dom_match
        else:
            day_match = dom_match or dow_match

        if not day_match:
            next_local = (t_local + timedelta(days=1)).replace(hour=0, minute=0)
            t_utc = next_local.astimezone(UTC)
            continue

        if t_local.hour not in hours:
            next_local = (t_local + timedelta(hours=1)).replace(minute=0)
            t_utc = next_local.astimezone(UTC)
            continue

        if t_local.minute not in minutes:
            t_utc += timedelta(minutes=1)
            continue

        # Guard: the result must be strictly after `after` in real time (UTC).
        # Wall-clock comparison (result <= after) is wrong during the
        # fall-back hour: 01:30 PDT (fold=0) > 01:10 PST (fold=1) as
        # wall-clock, but 08:30Z < 09:10Z in real time.
        result = t_utc.astimezone(tz)
        if result.astimezone(UTC) <= after.astimezone(UTC):
            t_utc += timedelta(minutes=1)
            continue
        return result

    raise CronParseError(
        f"no matching time found within 4 years of {after.isoformat()}"
    )


def cron_description(expr: str) -> str:
    """Human-readable description of a cron expression (for list_jobs)."""
    minutes, hours, doms, months, dows = parse_cron(expr)

    def fmt(values: set[int], lo: int, hi: int) -> str:
        if values == set(range(lo, hi + 1)):
            return "*"
        if len(values) == 1:
            return str(next(iter(values)))
        if len(values) <= 4:
            return ",".join(str(v) for v in sorted(values))
        return f"{min(values)}-{max(values)}"

    parts = [
        fmt(minutes, 0, 59),
        fmt(hours, 0, 23),
        fmt(doms, 1, 31),
        fmt(months, 1, 12),
        fmt(dows, 0, 6),
    ]
    return " ".join(parts)
