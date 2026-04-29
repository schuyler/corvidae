"""Performance monitoring plugin for Corvidae.

Hooks into the agent generation loop to track:
- Per-channel latency (ms) from user message to response
- Tokens in (prompt) and out (completion) per turn
- Cache rebuild frequency (compaction events)
- Prompt size evolution over time

Outputs live stats via `/perf_stats` tool.
"""
from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.channel import Channel

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """Metrics for a single agent turn."""
    timestamp: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    compaction_triggered: bool = False
    cache_rebuild: bool = False


@dataclass
class ChannelStats:
    """Rolling metrics for a single channel."""
    turns: deque[TurnMetrics] = field(default_factory=lambda: deque(maxlen=100))
    total_turns: int = 0
    total_compactions: int = 0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_tokens_in: int = 0
    avg_tokens_out: int = 0


class PerfMonPlugin:
    """Performance monitoring plugin.

    Hooks into the generation loop to track latency, token usage, and
    compaction frequency per channel. Provides a /perf_stats tool for
    querying current metrics.
    """

    name = "perf_mon"
    depends_on: list[str] = []

    def __init__(self):
        self._channel_stats: dict[str, ChannelStats] = {}
        self._turn_start_times: dict[str, float] = {}  # channel_id -> timestamp
        self._turn_tokens_in: dict[str, int] = {}      # channel_id -> tokens_in from prompt
        self.compaction_count: int = 0                 # global compaction counter

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    async def on_start(self, plugin_manager) -> None:
        """Register the /perf_stats tool."""
        pm = plugin_manager
        pm.register(self)
        logger.info("PerfMonPlugin registered")

    async def on_stop(self, plugin_manager) -> None:
        pass

    async def before_agent_turn(
        self, channel: "Channel"
    ) -> None:
        """Record turn start time and prompt size.

        Called before each LLM invocation after compaction, so we can measure
        the full latency from prompt build to response.
        """
        ch_id = channel.id
        self._turn_start_times[ch_id] = time.monotonic()

        # Estimate tokens_in from conversation size (rough approximation)
        conv = channel.conversation if channel.conversation else []
        self._turn_tokens_in[ch_id] = len(conv)  # message count as proxy

    async def after_agent_response(
        self,
        channel: "Channel",
        result_message: dict,
    ) -> None:
        """Record turn completion metrics.

        Called after the LLM response is received and before it's persisted.
        Captures latency and output token count.
        """
        ch_id = channel.id
        start = self._turn_start_times.pop(ch_id, 0)
        tokens_in = self._turn_tokens_in.pop(ch_id, 0)

        if not start:
            return  # missed before_agent_turn

        latency_ms = (time.monotonic() - start) * 1000

        # Extract output tokens from result
        tokens_out = 0
        if "usage" in result_message:
            usage = result_message["usage"]
            if isinstance(usage, dict):
                tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        # Create metrics record
        metrics = TurnMetrics(
            timestamp=time.time(),
            latency_ms=round(latency_ms, 1),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        stats = self._channel_stats.setdefault(ch_id, ChannelStats())
        stats.turns.append(metrics)
        stats.total_turns += 1
        stats.last_latency_ms = latency_ms

        # Update rolling averages and percentiles
        latencies = [m.latency_ms for m in stats.turns]
        if latencies:
            stats.avg_latency_ms = round(statistics.mean(latencies), 1)
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            stats.p50_latency_ms = round(sorted_lat[n // 2], 1)
            stats.p95_latency_ms = round(sorted_lat[int(n * 0.95)], 1) if n > 1 else stats.p50_latency_ms
            stats.p99_latency_ms = round(sorted_lat[int(n * 0.99)], 1) if n > 1 else stats.p50_latency_ms

        tokens_in_list = [m.tokens_in for m in stats.turns]
        tokens_out_list = [m.tokens_out for m in stats.turns]
        if tokens_in_list:
            stats.avg_tokens_in = round(statistics.mean(tokens_in_list))
        if tokens_out_list:
            stats.avg_tokens_out = round(statistics.mean(tokens_out_list))

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def perf_stats(self, channel: "Channel") -> str:
        """Show performance metrics for all channels.

        Returns a formatted string with latency percentiles, token usage,
        and turn counts per channel.
        """
        if not self._channel_stats:
            return "No performance data recorded yet."

        lines = ["**Performance Statistics**"]
        lines.append(f"Total tracked channels: {len(self._channel_stats)}")
        lines.append("")

        for ch_id, stats in sorted(self._channel_stats.items()):
            if not stats.turns:
                continue

            lines.append(f"**{ch_id}** (turns: {stats.total_turns})")
            lines.append(
                f"  Latency: avg={stats.avg_latency_ms:.0f}ms "
                f"p50={stats.p50_latency_ms:.0f}ms "
                f"p95={stats.p95_latency_ms:.0f}ms "
                f"p99={stats.p99_latency_ms:.0f}ms "
                f"last={stats.last_latency_ms:.0f}ms"
            )
            lines.append(
                f"  Tokens: avg_in={stats.avg_tokens_in} "
                f"avg_out={stats.avg_tokens_out}"
            )

        return "\n".join(lines)
