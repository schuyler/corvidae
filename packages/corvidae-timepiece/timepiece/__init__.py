"""Corvidae plugin: timepiece — ambient wall-clock stamps in the window.

Before each LLM turn, if the channel's last stamp is older than
``timepiece.interval_minutes`` (default 10), a short CONTEXT marker
(``[time 2026-11-07 19:15 PST (Sat)]``) is appended to the
conversation. CONTEXT entries appear in the prompt for that turn and
are persisted by the agent's post-hook sweep (agent.py, after
``before_agent_turn``); they survive compaction because compaction
only summarizes MESSAGE entries. The result is ambient time awareness
for the model: no turn is triggered, no tool is exposed, and at most
one ~12-token line enters each channel per interval.

Config (own section; absent block = defaults):
    timepiece.interval_minutes: positive number, default 10.
    timepiece.timezone: IANA timezone name, default the daemon's local time.
    timepiece.format: strftime format string, default
        "%A %Y-%m-%d at %H:%M %Z" (produces e.g. "Monday 2026-11-09
        at 17:12 PST").
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from corvidae.context import MessageType
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tuning import config_value

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_MINUTES = 10
DEFAULT_FORMAT = "%A %Y-%m-%d at %H:%M %Z"


class TimepiecePlugin(CorvidaePlugin):
    """Injects a wall-clock CONTEXT stamp into each channel, at most once per interval."""

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        # config is the WHOLE agent.yaml dict: read only our section.
        raw = config_value(config, "timepiece.interval_minutes", DEFAULT_INTERVAL_MINUTES)
        if type(raw) not in (int, float) or isinstance(raw, bool) or raw <= 0:
            # Malformed block: loud, names the key and the rejected value.
            raise ValueError(
                "Config error: timepiece.interval_minutes must be a positive "
                f"number of minutes, got {raw!r} — check agent.yaml"
            )
        self._interval = timedelta(minutes=raw)
        tz_name = config_value(config, "timepiece.timezone", None)
        if tz_name is not None:
            if not isinstance(tz_name, str) or not tz_name:
                raise ValueError(
                    "Config error: timepiece.timezone must be a non-empty IANA "
                    f"timezone name, got {tz_name!r} — check agent.yaml"
                )
            try:
                self._tz = ZoneInfo(tz_name)
            except Exception as exc:
                raise ValueError(
                    "Config error: timepiece.timezone is not a known IANA "
                    f"timezone: {tz_name!r} — check agent.yaml"
                ) from exc
        else:
            self._tz = None  # fall back to the daemon's local time
        fmt = config_value(config, "timepiece.format", DEFAULT_FORMAT)
        if not isinstance(fmt, str) or not fmt:
            raise ValueError(
                "Config error: timepiece.format must be a non-empty "
                f"strftime format string, got {fmt!r} \u2014 check agent.yaml"
            )
        self._format = fmt
        # channel id -> wall time of the last stamp injected there.
        # In-memory only: after a restart the first turn re-stamps, which
        # is the correct behavior (the window is fresh, so is the clock).
        self._last: dict[str, datetime] = {}

    def _now(self) -> datetime:
        """Wall clock with the configured timezone (seam for tests)."""
        if self._tz is None:
            return datetime.now().astimezone()
        return datetime.now(self._tz)

    @hookimpl
    async def before_agent_turn(
        self, channel, trace_id: str | None, annotations: dict
    ) -> None:
        conv = channel.conversation
        if conv is None:
            return
        now = self._now()
        last = self._last.get(channel.id)
        # Stamp on the first turn and whenever the interval has fully
        # elapsed; strictly inside it the last stamp is still fresh.
        if last is not None and now - last < self._interval:
            return
        stamp = now.strftime(self._format)
        marker = {"role": "system", "content": f"The current date and time are now {stamp}"}
        conv.append(marker, MessageType.CONTEXT)
        # No manual persist: the agent fires persist_message for entries
        # appended during this hook that lack a _db_id (agent.py sweep).
        self._last[channel.id] = now
