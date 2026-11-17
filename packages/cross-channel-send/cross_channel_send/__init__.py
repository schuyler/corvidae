"""Cross-channel send: the agent can notify another channel's context from any turn (#234).

The notify_other_channel tool lets any turn notify a different channel
context. The message is delivered as a system-role notification (via
on_notify), the same mechanism the scheduler uses for reminders. The
target mints its own trace and gets a fresh turn budget.

The list_channels tool returns the live (registered) channel IDs. A target
need not be live: notify_other_channel resolves any well-formed
"transport:scope" id with resolve_persisted_channel (#938), so a channel
nobody has spoken in since the restart is still reachable.

Loop prevention: each send increments a hop_count tracked in the
notification's annotations. The receiving context's attribution carries the
hop_count, so the next send from that context increments it again. When
hop_count exceeds max_hops (config, default 3), the send is refused.

Origin: the note carries the sending turn's ``annotations["origin"]``
verbatim and never stamps one of its own (PRD §3.5.10, #1034), so a note sent
from a reminder turn wakes a reminder-origin turn. A user turn has no origin
("user" is inferred from admission), so its note carries none. The sender is
named by ``source_channel``/``source_trace_id`` and by the kernel's
``origin_channel``/``origin_trace`` metadata. How the cognition consumers read
the woken turn (its trace is minted at dequeue, never admitted), checked
against their source on 2026-11-17:
- critique._eligible: eligible iff origin == "reminder", or origin is absent
  and the trace was admitted. A reminder chain's woken turn is eligible; a
  user chain's is not. (#1031 decided that every main-loop turn is judged;
  not built yet.)
- funnel: origin keys the deferred-registration queue as (channel.id,
  origin); a woken turn with no origin drains only the next-turn queue.
- gates: no origin reads as "task", so the decide gate's salience
  withholding applies to a user chain's woken turn; a reminder chain's is
  never withheld; a ``gate.send.quiet_origins`` origin stays quiet on the
  channel it sends to.
- appraisal: stage 2 runs unless the origin is "critique".
- outcome_log: the exchange row's origin column records the carried origin,
  or "task" when there is none.
"""

from __future__ import annotations

import logging

from corvidae.admission import offer_work
from corvidae.attribution import get_attribution
from corvidae.channel import ChannelNotReady, ChannelRegistry, resolve_persisted_channel
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.tuning import config_value

logger = logging.getLogger("cross_channel_send")

DEFAULT_MAX_HOPS = 3
MAX_TEXT_CHARS = 8000


class CrossChannelSendPlugin(CorvidaePlugin):
    """Provides notify_other_channel and list_channels tools for cross-channel delivery."""

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
        tool_registry.append(self.notify_other_channel)
        tool_registry.append(self.list_channels)

    async def notify_other_channel(self, channel_id: str, text: str) -> str:
        """Notify a *different* channel context, waking it for an inference turn.

        This does NOT message the user on your current channel — your normal
        reply already reaches them. It queues a system notification on the
        *target* channel, which wakes that context for a fresh turn (it mints
        its own trace and gets a new turn budget). Use it to ask another
        context of yourself to act on a different channel (e.g. heartbeat
        asking Signal to ping the user with news). Never use it to reply on
        the channel you are already on: a self-send is refused.

        Args:
            channel_id: The target channel in "transport:scope" form,
                e.g. "signal:4d8d4445-..." or "web:main". It need not appear
                in list_channels: any well-formed id is resolved. Must differ
                from the current channel.
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

            # Resolve the target the way persisted ids are resolved elsewhere
            # (#938): live registry, then the owning transport, then a bare
            # channel, so a channel that isn't live yet is still reachable.
            try:
                target = await resolve_persisted_channel(
                    self.pm, self._registry, channel_id
                )
            except ChannelNotReady as exc:
                # The owning transport can't apply the channel's config yet.
                logger.warning(
                    "notify_other_channel: channel %s not ready (%s); refused",
                    channel_id,
                    type(exc).__name__,
                )
                return (
                    f"Could not send: channel {channel_id!r} can't be resolved yet "
                    "(its transport isn't ready to apply its config); try again "
                    "once the transport is ready."
                )
            # None means the id is not in transport:scope form.
            if target is None:
                return (
                    f"Could not send: {channel_id!r} is not a channel id; expected "
                    '"transport:scope" (e.g. "signal:+15551234567"); use '
                    "list_channels to see available targets."
                )
            if source_channel == target.id:
                logger.warning(
                    "notify_other_channel refused: target %s is the source channel",
                    target.id,
                )
                return (
                    f"Could not send: target {channel_id!r} is the same as the "
                    "source channel; reply directly instead of using "
                    "notify_other_channel."
                )

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

            # The note names its sender and its hop; it carries the sending
            # turn's origin verbatim and never stamps one of its own, so the
            # woken turn follows how the chain started (PRD §3.5.10, #1034).
            # A turn with no origin (a user turn) sends a note with none.
            note_annotations = {
                "source_channel": source_channel,
                "source_trace_id": attribution.get("trace_id"),
                "hop_count": hop,
            }
            if "origin" in annotations:
                note_annotations["origin"] = annotations["origin"]

            # Offer the note to the admission seam; a refusal is the result
            # and nothing reaches on_notify.
            refusal = await offer_work(
                self.pm,
                channel=target,
                kind="note",
                source_channel=source_channel,
                trace_id=attribution.get("trace_id"),
                annotations=annotations,
            )
            if refusal is not None:
                return refusal

            await self.pm.ahook.on_notify(
                channel=target,
                source=f"agent_send from {source_channel}",
                text=text,
                tool_call_id=None,
                annotations=note_annotations,
            )
            return f"Queued on {channel_id}."
        except Exception:
            logger.exception("cross_channel_send failed for %s", channel_id)
            return "Could not send: unexpected error."

    async def list_channels(self) -> str:
        """List registered channel IDs.

        Returns the IDs of all channels currently registered in the
        ChannelRegistry. Use this to discover targets for
        notify_other_channel; a well-formed id that isn't listed (a channel
        nobody has spoken in since the restart) is also accepted.

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
