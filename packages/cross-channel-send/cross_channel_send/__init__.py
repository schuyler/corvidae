"""Cross-channel send: the agent can message any channel from any context (#234).

The send_message tool lets any turn send an instruction to another channel
context. The message is delivered as a system-role notification (via
on_notify), the same mechanism the scheduler uses for reminders. The
target mints its own trace and gets a fresh turn budget.

The list_channels tool returns registered channel IDs so the sending
context knows its available targets.

Loop prevention: each send increments a hop_count tracked in the
notification's annotations. The receiving context's attribution carries the
hop_count, so the next send from that context increments it again. When
hop_count exceeds max_hops (config, default 3), the send is refused.

Cognition consumers see origin "agent_send" in annotations. Verified against the
cognition source (2026-11-05):
- critique._eligible: exempt. Eligible iff origin == "reminder", or origin
  is absent and the trace was admitted (user). "agent_send" matches
  neither, so a sent instruction never forks a critique.
- funnel: neutral. Origin keys the deferred-registration queue as
  (channel.id, origin); "agent_send" is just a new vocabulary value,
  queued and drained on the next same-origin turn of the channel.
- gates: the send's final response is a non-user origin, so the decide
  gate's salience withholding applies to it exactly as to a reminder's
  reply (engaged inbound messages are unaffected; the engagement gate is
  inbound-only).
- appraisal: stage 2 runs (it skips only origin "critique") — one silent
  appraisal task per sent instruction.
- outcome_log: the exchange row's origin column records "agent_send"
  (record_exchange takes annotations["origin"] verbatim, default "task").
"""

from __future__ import annotations

import logging

from corvidae.attribution import get_attribution
from corvidae.channel import ChannelRegistry
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.tuning import config_value

logger = logging.getLogger("cross_channel_send")

DEFAULT_MAX_HOPS = 3
MAX_TEXT_CHARS = 8000


class CrossChannelSendPlugin(CorvidaePlugin):
    """Provides send_message and list_channels tools for cross-channel delivery."""

    depends_on = frozenset({"registry"})
    model_visible_annotations = frozenset({"origin"})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._registry: ChannelRegistry | None = None
        self._max_hops: int = DEFAULT_MAX_HOPS

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        raw = config_value(config, "cross_channel_send.max_hops", DEFAULT_MAX_HOPS)
        if type(raw) is bool or isinstance(raw, float):
            raise ValueError(f"cross_channel_send.max_hops must be an integer, got {raw!r}")
        try:
            self._max_hops = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"cross_channel_send.max_hops must be an integer, got {raw!r}"
            ) from exc
        if self._max_hops < 1:
            raise ValueError(f"cross_channel_send.max_hops must be >= 1, got {self._max_hops}")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        self._registry = get_dependency(self.pm, "registry", ChannelRegistry)

    @hookimpl
    def register_tools(self, tool_registry):
        tool_registry.append(self.send_message)
        tool_registry.append(self.list_channels)

    async def send_message(self, channel_id: str, text: str) -> str:
        """Send an instruction to another channel context.

        The message triggers an inference turn on the target channel. The
        receiving context sees a system notification identifying this
        channel as the source. Use this to ask another context of yourself
        to take action on a different channel (e.g. heartbeat asking Signal
        to ping the user with news).

        Args:
            channel_id: The target channel in "transport:scope" form,
                e.g. "signal:4d8d4445-..." or "web:main".
            text: The instruction for the receiving context.

        Returns:
            A confirmation string, or an error naming the channel.
        """
        if self._registry is None:
            return "Could not send: channel registry is not available."
        if not isinstance(channel_id, str) or not channel_id:
            return "Could not send: channel_id must be a non-empty string."
        if not isinstance(text, str) or not text.strip():
            return "Could not send: text must be a non-empty string."
        if len(text) > MAX_TEXT_CHARS:
            return f"Could not send: text is {len(text)} chars, limit is {MAX_TEXT_CHARS}."

        try:
            attribution = get_attribution()
            if not attribution:
                return "Could not send: no source channel in the current context."
            source_channel = attribution.get("channel_id")
            if not source_channel:
                return "Could not send: no source channel in the current context."

            annotations = attribution.get("annotations")
            if not isinstance(annotations, dict):
                annotations = {}
            prev_hops = annotations.get("hop_count")
            if type(prev_hops) is int and prev_hops >= 0:
                hop = prev_hops + 1
            else:
                hop = 1
            if hop > self._max_hops:
                return (
                    f"Could not send: cross-channel loop detected on "
                    f"{channel_id!r} (hop {hop} > max {self._max_hops})."
                )

            target = self._registry.get(channel_id)
            if target is None:
                return (
                    f"Could not send: channel {channel_id!r} is not registered; "
                    "use list_channels to see available targets."
                )

            await self.pm.ahook.on_notify(
                channel=target,
                source=f"agent_send from {source_channel}",
                text=text,
                tool_call_id=None,
                annotations={
                    "origin": "agent_send",
                    "source_channel": source_channel,
                    "source_trace_id": attribution.get("trace_id"),
                    "hop_count": hop,
                },
            )
            return f"Queued on {channel_id}."
        except Exception:
            logger.exception("cross_channel_send failed for %s", channel_id)
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
        if self._registry is None:
            return "Could not list channels: channel registry is not available."
        try:
            channels = self._registry.channels
            if not channels:
                return "No channels registered."
            return "\n".join(sorted(channels))
        except Exception:
            logger.exception("list_channels failed")
            return "Could not list channels: unexpected error."
