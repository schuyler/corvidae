"""Backlog wording shared by the transports (#644, PRD §3.2).

A message that reached the agent long after it was sent must read as old, or
the agent answers it as though it were current. Signal has done this since its
first version; Zulip's catch-up needs the same words, so they live here rather
than in one transport importing the other.
"""

from datetime import datetime, timezone

#: Wall-clock age past which an inbound item is backlog and says so.
STALE_MESSAGE_THRESHOLD_SECONDS = 5 * 60


def format_stale_prefix(timestamp_ms: int) -> str:
    """Render a send timestamp as a ``[sent YYYY-MM-DD HH:MM UTC]`` prefix."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return f"[sent {dt.strftime('%Y-%m-%d %H:%M')} UTC]\n"


def maybe_prefix_stale(text: str, timestamp_ms: int | None, now_ms: float) -> str:
    """``text`` prefixed with its send time when it is older than the threshold."""
    if timestamp_ms is None:
        return text
    if (now_ms - timestamp_ms) / 1000 <= STALE_MESSAGE_THRESHOLD_SECONDS:
        return text
    return format_stale_prefix(timestamp_ms) + text
