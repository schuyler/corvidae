"""The kernel answers which transport owns a channel, as plain facts (#1193).

Plugins read a channel's transport facts through ``owning_transport`` instead
of walking ``pm.get_plugins()`` for a ``TransportPlugin`` (docs/plugin-guide.md,
"Plugin boundaries").
"""

from __future__ import annotations

from corvidae.channel import ChannelConfig, ChannelRegistry
from corvidae.hooks import TransportFacts, TransportPlugin, create_plugin_manager, owning_transport


class _ChatTransport(TransportPlugin):
    """A conversational transport with a declared check allowance."""

    transport = "chat"
    conversational = True
    check_allowance = 7.5


class _PlainTransport(TransportPlugin):
    """A transport that declares nothing beyond its name."""

    transport = "plain"


def _env():
    """(pm, registry) with both test transports loaded."""
    pm = create_plugin_manager()
    registry = ChannelRegistry()
    pm.register(registry, name="registry")
    pm.register(_ChatTransport(), name="chat")
    pm.register(_PlainTransport(), name="plain")
    return pm, registry


def test_facts_of_the_owning_transport():
    # Each channel gets its own transport's facts, as values.
    pm, registry = _env()
    chat = registry.get_or_create("chat", "room", config=ChannelConfig())
    plain = registry.get_or_create("plain", "room", config=ChannelConfig())
    assert owning_transport(pm, chat) == TransportFacts("chat", True, 7.5)
    assert owning_transport(pm, plain) == TransportFacts("plain", False, None)


def test_none_for_a_channel_no_transport_owns():
    # An internal channel (no loaded transport claims it) has no owner.
    pm, registry = _env()
    internal = registry.get_or_create("internal", "heartbeat", config=ChannelConfig())
    assert owning_transport(pm, internal) is None
