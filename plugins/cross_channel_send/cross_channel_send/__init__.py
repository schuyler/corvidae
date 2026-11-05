"""Cross-channel send: the agent can message any channel from any context (#234).

The send_message tool lets any turn send an instruction to another channel
context. The message triggers an inference turn on the target channel; the
receiving context sees a provenance frame identifying the source channel
and mechanism.

The list_channels tool returns registered channel IDs so the sending
context knows its available targets.
"""

from __future__ import annotations

import logging

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.attribution import get_attribution

logger = logging.getLogger("cross_channel_send")


class CrossChannelSendPlugin(CorvidaePlugin):
    """Provides send_message and list_channels tools for cross-channel delivery."""

    depends_on = frozenset({"agent"})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._agent = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        self._agent = self.pm.get_plugin("agent")
        if self._agent is None:
            raise RuntimeError("cross_channel_send requires the agent plugin")

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
            attribution = get_attribution()
            if attribution:
                trace_id = attribution.get("trace_id")
                source_channel = attribution.get("channel_id")
            if not source_channel:
                return "Could not send: no source channel in current context."
            await self._agent.enqueue_agent_send(
                channel_id, text,
                source_channel_id=source_channel,
                trace_id=trace_id,
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
            return "No channels registered."
        try:
            channels = self._agent.list_channels()
            if not channels:
                return "No channels registered."
            return "\n".join(channels)
        except Exception:
            logger.exception("list_channels failed")
            return "Could not list channels: unexpected error."
