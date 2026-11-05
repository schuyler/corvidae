"""Cross-channel send: the agent can message any channel from any context (#234).

The send_message tool lets any turn send an instruction to another channel
context. The message triggers an inference turn on the target channel; the
receiving context sees a provenance frame identifying the source channel
and mechanism.

The list_channels tool returns registered channel IDs so the sending
context knows its available targets.

Loop prevention: each send increments a hop_count tracked in the item's
meta. The receiving context's attribution carries the hop_count, so the
next send from that context increments it again. When hop_count exceeds
max_hops (config, default 3), the send is refused.
"""

from __future__ import annotations

import logging

from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.attribution import get_attribution

logger = logging.getLogger("cross_channel_send")

DEFAULT_MAX_HOPS = 3


class CrossChannelSendPlugin(CorvidaePlugin):
    """Provides send_message and list_channels tools for cross-channel delivery."""

    depends_on = frozenset({"agent"})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._agent = None
        self._max_hops: int = DEFAULT_MAX_HOPS

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        raw = config.get("max_hops", DEFAULT_MAX_HOPS)
        if type(raw) is bool or isinstance(raw, float):
            raise ValueError(f"max_hops must be an integer, got {raw!r}")
        try:
            self._max_hops = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"max_hops must be an integer, got {raw!r}") from exc
        if self._max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {self._max_hops}")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        from corvidae.agent import Agent
        self._agent = get_dependency(self.pm, "agent", Agent)

    @hookimpl
    def register_tools(self, tool_registry):
        tool_registry.append(self.send_message)
        tool_registry.append(self.list_channels)

    async def send_message(self, channel_id: str, text: str) -> str:
        """Send an instruction to another channel context.

        The message triggers an inference turn on the target channel. The
        receiving context sees a provenance frame identifying this channel
        as the source. Use this to ask another context of yourself to take
        action on a different channel (e.g. heartbeat asking Signal to ping
        the user with news).

        Args:
            channel_id: The target channel in "transport:scope" form,
                e.g. "signal:4d8d4445-..." or "web:main".
            text: The instruction for the receiving context.

        Returns:
            A confirmation string, or an error naming the channel.
        """
        if self._agent is None:
            return "Could not send: agent is not available."
        if not isinstance(channel_id, str) or not isinstance(text, str):
            return "Could not send: channel_id and text must be strings."
        try:
            trace_id = None
            source_channel = None
            hop_count = 1
            attribution = get_attribution()
            if attribution:
                trace_id = attribution.get("trace_id")
                source_channel = attribution.get("channel_id")
                meta = attribution.get("meta") or {}
                if isinstance(meta, dict):
                    prev_hops = meta.get("hop_count")
                    if type(prev_hops) is int:
                        hop_count = prev_hops + 1
            if not source_channel:
                return "Could not send: no source channel in current context."
            if hop_count > self._max_hops:
                return (
                    f"Could not send: cross-channel loop detected "
                    f"(hop {hop_count} > max {self._max_hops})."
                )
            await self._agent.enqueue_agent_send(
                channel_id, text,
                source_channel_id=source_channel,
                trace_id=trace_id,
                hop_count=hop_count,
            )
            return f"Queued on {channel_id}."
        except KeyError:
            return (
                f"Could not send: channel {channel_id!r} is not registered. "
                "Use list_channels to see available targets."
            )
        except ValueError as exc:
            return f"Could not send: {exc}"
        except RuntimeError as exc:
            return f"Could not send: {exc}"
        except Exception:
            logger.exception("cross_channel_send failed")
            return "Could not send: unexpected error."

    async def list_channels(self) -> str:
        """List registered channel IDs.

        Returns the IDs of all channels currently registered in the
        ChannelRegistry. Use this to discover valid targets for
        send_message.

        Returns:
            A newline-separated list of channel IDs (sorted), or a message
            indicating no channels are registered.
        """
        if self._agent is None:
            return "Could not list channels: agent is not available."
        try:
            channels = self._agent.list_channels()
            if not channels:
                return "No channels registered."
            return "\n".join(channels)
        except Exception:
            logger.exception("list_channels failed")
            return "Could not list channels: unexpected error."
