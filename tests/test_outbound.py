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
    """A transport that overrides every send_* method and records each call."""

    def __init__(self, transport: str) -> None:
        self.transport = transport
        self.seen: list[Outbound] = []

    async def send_message(self, channel: Channel, event: OutboundMessage) -> None:
        self.seen.append(event)

    async def send_thinking(self, channel: Channel, event: OutboundThinking) -> None:
        self.seen.append(event)

    async def send_tool_status(self, channel: Channel, event: OutboundToolStatus) -> None:
        self.seen.append(event)

    async def send_progress(self, channel: Channel, event: OutboundProgress) -> None:
        self.seen.append(event)


class _Disowning(_AllKinds):
    """A transport whose owns() narrows the filter to nothing (WebPlugin's
    disabled case)."""

    def owns(self, channel: Channel) -> bool:
        return False


def _channel(transport: str, scope: str = "room") -> Channel:
    """Build a bare channel owned by ``transport``."""
    return Channel(transport=transport, scope=scope, config=ChannelConfig())


# One event of each kind, so every dispatch arm is exercised.
EVENTS: list[Outbound] = [
    OutboundMessage(text="hello", latency_ms=12.5),
    OutboundThinking(text="hmm"),
    OutboundToolStatus(tool_name="shell", status="dispatched", args_summary="ls"),
    OutboundProgress(text="working"),
]


async def test_filter_and_noop_defaults() -> None:
    """The fake transport sees exactly its own message; the other sees nothing.

    Thinking, tool-status and progress events for the fake transport's own
    channel hit the no-op defaults without raising, and no event for the
    other transport's channel reaches either plugin.
    """
    pm = create_plugin_manager()
    fake, other = _Recorder("fake"), _Recorder("other")
    pm.register(fake, name="fake")
    pm.register(other, name="other")

    # Send every kind to a channel owned by "fake", then to one owned by a
    # third transport that neither plugin owns.
    mine, foreign = _channel("fake"), _channel("third")
    for event in EVENTS:
        await pm.ahook.send(channel=mine, event=event)
        await pm.ahook.send(channel=foreign, event=event)

    assert fake.seen == [(mine.id, EVENTS[0])]
    assert other.seen == []


async def test_dispatch_by_event_type() -> None:
    """Each event kind reaches the matching send_* method, in order."""
    pm = create_plugin_manager()
    plugin = _AllKinds("fake")
    pm.register(plugin, name="fake")

    for event in EVENTS:
        await pm.ahook.send(channel=_channel("fake"), event=event)

    assert plugin.seen == EVENTS


async def test_owns_override_narrows_the_filter() -> None:
    """A transport whose owns() returns False sees nothing, even for a
    channel whose transport name matches its own."""
    pm = create_plugin_manager()
    plugin = _Disowning("fake")
    pm.register(plugin, name="fake")

    for event in EVENTS:
        await pm.ahook.send(channel=_channel("fake"), event=event)

    assert plugin.seen == []


async def test_tool_status_carries_tool_call_id() -> None:
    """tool_call_id reaches send_tool_status intact, and defaults to None."""
    pm = create_plugin_manager()
    plugin = _AllKinds("fake")
    pm.register(plugin, name="fake")

    with_id = OutboundToolStatus(tool_name="shell", status="completed", tool_call_id="call_1")
    await pm.ahook.send(channel=_channel("fake"), event=with_id)

    assert plugin.seen == [with_id]
    assert plugin.seen[0].tool_call_id == "call_1"
    assert OutboundToolStatus(tool_name="t", status="dispatched").tool_call_id is None


def test_transport_plugin_depends_on_registry() -> None:
    """Every transport needs the registry; the base class declares it."""
    assert TransportPlugin.depends_on == frozenset({"registry"})
