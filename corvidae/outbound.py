"""Outbound events: what the agent asks a transport to show.

The ``send(channel, event)`` hook carries one of these dataclasses.
``TransportPlugin.send`` (corvidae/hooks.py) filters the event to the
transport that owns the channel and dispatches it by type to that
transport's ``send_message``, ``send_thinking``, ``send_tool_status`` or
``send_progress`` method. A new kind of outbound event is a new dataclass
here plus a no-op method on the base class, not a new hookspec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    """The agent's reply to a channel.

    Attributes:
        text: The message content to deliver.
        latency_ms: Agent-loop latency for this reply, or None. Transports
            that display timing (the CLI) use it; the rest ignore it.
    """

    text: str
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class OutboundThinking:
    """Reasoning content from an LLM response, for transports that show it.

    Attributes:
        text: The reasoning text.
    """

    text: str


@dataclass(frozen=True, slots=True)
class OutboundToolStatus:
    """A tool call's lifecycle: dispatched, then completed.

    Attributes:
        tool_name: The tool's name.
        status: "dispatched" or "completed".
        args_summary: Truncated arguments, on dispatch.
        result_summary: Truncated result, on completion.
        tool_call_id: The LLM's id for this call, or None when the sender
            does not know it. Lets a transport pair the two events of one
            call even when the same tool runs twice at once.
    """

    tool_name: str
    status: str
    args_summary: str | None = None
    result_summary: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class OutboundProgress:
    """Intermediate assistant text sent alongside tool calls.

    Attributes:
        text: The intermediate text.
    """

    text: str


# Every event the ``send`` hook can carry.
Outbound = OutboundMessage | OutboundThinking | OutboundToolStatus | OutboundProgress
