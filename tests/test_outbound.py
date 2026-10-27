"""The outbound ``send`` hook and the TransportPlugin broadcast filter.

``send(channel, event)`` replaces the four ``send_*`` hookspecs
(plans/kernel-split.md, Part 1 §3). Every transport receives every event;
``TransportPlugin.send`` drops events for channels the transport does not
own and dispatches the rest to a plain ``send_*`` method by event type, so a
transport cannot forget the filter (R3).
"""

from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import TransportPlugin, create_plugin_manager
from corvidae.outbound import (
    Outbound,
    OutboundMessage,
    OutboundProgress,
    OutboundThinking,
    OutboundToolStatus,
)


class _Recorder(TransportPlugin):
    """A transport that overrides only send_message and records each call.

    Thinking, tool status and progress fall through to the base class's
    no-op defaults.
    """

    def __init__(self, transport: str) -> None:
        self.transport = transport
        self.seen: list[tuple[str, Outbound]] = []

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        self.seen.append((channel.id, event))


class _AllKinds(TransportPlugin):
    """A transport that overrides every send_* method and records which
    method received which event."""

    def __init__(self, transport: str) -> None:
        self.transport = transport
        self.seen: list[tuple[str, Outbound]] = []

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        self.seen.append(("send_message", event))

    async def send_thinking(self, channel: Channel, event: OutboundThinking) -> None:
        self.seen.append(("send_thinking", event))

    async def send_tool_status(self, channel: Channel, event: OutboundToolStatus) -> None:
        self.seen.append(("send_tool_status", event))

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        self.seen.append(("send_progress", event))


class _Disowning(_AllKinds):
    """A transport whose owns() narrows the filter to nothing (WebPlugin's
    disabled case)."""

    def owns(self, channel: Channel) -> bool:
        return False


def _channel(transport: str, scope: str = "room") -> Channel:
    """Build a bare channel owned by ``transport``."""
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


# One event of each kind, so every dispatch arm is exercised, paired with
# the method each one must reach.
EVENTS: list[Outbound] = [
    OutboundMessage(text="hello", latency_ms=12.5),
    OutboundThinking(text="hmm"),
    OutboundToolStatus(tool_name="shell", status="dispatched", args_summary="ls"),
    OutboundProgress(text="working"),
]
EXPECTED_METHODS = ["send_message", "send_thinking", "send_tool_status", "send_progress"]


async def test_filter_and_noop_defaults() -> None:
    """Each transport sees exactly its own channel's message, nothing else.

    Thinking, tool-status and progress events hit the no-op defaults
    without raising; no event for one transport's channel reaches the
    other, and nothing for a third transport's channel reaches either.
    """
    # Two message-only transports on one plugin manager.
    pm = create_plugin_manager()
    fake, other = _Recorder("fake"), _Recorder("other")
    pm.register(fake, name="fake")
    pm.register(other, name="other")

    # Send every kind to each transport's own channel and to a channel
    # owned by a third transport that neither plugin owns.
    mine, theirs, foreign = _channel("fake"), _channel("other"), _channel("third")
    for event in EVENTS:
        for channel in (mine, theirs, foreign):
            await pm.ahook.send(channel=channel, event=event)

    # Only the message event, and only on each plugin's own channel.
    assert fake.seen == [(mine.id, EVENTS[0])]
    assert other.seen == [(theirs.id, EVENTS[0])]


async def test_dispatch_by_event_type() -> None:
    """Each event kind reaches the send_* method for that kind, in order."""
    # One transport that records which method each event reached.
    pm = create_plugin_manager()
    plugin = _AllKinds("fake")
    pm.register(plugin, name="fake")

    # Send one event of each kind to its own channel.
    for event in EVENTS:
        await pm.ahook.send(channel=_channel("fake"), event=event)

    # A swapped dispatch arm shows up as a mismatched (method, event) pair.
    assert plugin.seen == list(zip(EXPECTED_METHODS, EVENTS))


async def test_owns_override_narrows_the_filter() -> None:
    """A transport whose owns() returns False sees nothing, even for a
    channel whose transport name matches its own.

    The plugin overrides all four methods, so a filter that only guarded
    some dispatch arms would show up here.
    """
    pm = create_plugin_manager()
    plugin = _Disowning("fake")
    pm.register(plugin, name="fake")

    # Every kind, to a channel the transport name says it owns.
    for event in EVENTS:
        await pm.ahook.send(channel=_channel("fake"), event=event)

    assert plugin.seen == []


async def test_tool_status_carries_tool_call_id() -> None:
    """tool_call_id reaches send_tool_status intact, and defaults to None."""
    pm = create_plugin_manager()
    plugin = _AllKinds("fake")
    pm.register(plugin, name="fake")

    # An event built with an id arrives with that id.
    with_id = OutboundToolStatus(tool_name="shell", status="completed", tool_call_id="call_1")
    await pm.ahook.send(channel=_channel("fake"), event=with_id)
    assert plugin.seen == [("send_tool_status", with_id)]
    assert plugin.seen[0][1].tool_call_id == "call_1"

    # Senders that do not know the id may leave it out.
    assert OutboundToolStatus(tool_name="t", status="dispatched").tool_call_id is None


def test_transport_plugin_depends_on_registry() -> None:
    """Every transport needs the registry; the base class declares it."""
    assert TransportPlugin.depends_on == frozenset({"registry"})
