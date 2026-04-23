"""Channel abstraction for conversation scoping.

This module provides the channel registry that maps transport:scope identifiers
to Channel objects. Each channel has its own conversation history and config
overrides. Channels are created on-demand when messages arrive, or pre-registered
from the config file.

Logging:
    - INFO: channel registered from config
    - WARNING: invalid channel key format
    - DEBUG: channel config resolved
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sherman.conversation import ConversationLog

logger = logging.getLogger(__name__)


@dataclass
class ChannelConfig:
    """Per-channel configuration. Falls back to agent-level defaults
    for any field set to None."""

    system_prompt: str | list[str] | None = None
    max_context_tokens: int | None = None
    keep_thinking_in_history: bool | None = None
    # Future: tool allowlist/denylist, response length, etc.

    def resolve(self, agent_defaults: dict) -> dict:
        """Merge channel overrides with agent-level defaults.

        Channel config wins where set; agent defaults fill the gaps.
        Uses `is not None` for all fields so that falsy values (empty
        string, 0) are treated as intentional overrides.
        """
        resolved = {
            "system_prompt": (
                self.system_prompt if self.system_prompt is not None
                else agent_defaults.get("system_prompt", "You are a helpful assistant.")
            ),
            "max_context_tokens": (
                self.max_context_tokens if self.max_context_tokens is not None
                else agent_defaults.get("max_context_tokens", 24000)
            ),
            "keep_thinking_in_history": (
                self.keep_thinking_in_history
                if self.keep_thinking_in_history is not None
                else agent_defaults.get("keep_thinking_in_history", False)
            ),
        }

        logger.debug(
            "channel config resolved",
            extra={"channel_id": "(unknown)", "resolved": resolved},
        )

        return resolved


@dataclass
class Channel:
    """A conversation scope tied to a transport."""

    transport: str          # "irc", "signal", "cli"
    scope: str              # "#lex", "+15551234567", "local"
    config: ChannelConfig = field(default_factory=ChannelConfig)
    conversation: ConversationLog | None = None
    created_at: float = field(default_factory=time)
    last_active: float = field(default_factory=time)

    @property
    def id(self) -> str:
        """The string key used for storage and routing."""
        return f"{self.transport}:{self.scope}"

    def touch(self) -> None:
        """Update last_active to now."""
        self.last_active = time()

    def matches_transport(self, transport_name: str) -> bool:
        """Check if this channel belongs to the given transport."""
        return self.transport == transport_name


class ChannelRegistry:
    """Manages the lifecycle of channels."""

    def __init__(self, agent_defaults: dict) -> None:
        """
        Args:
            agent_defaults: Agent-level config dict used as fallback when
                resolving per-channel configuration overrides.
        """
        self.agent_defaults = agent_defaults
        self.channels: dict[str, Channel] = {}

    def get_or_create(
        self,
        transport: str,
        scope: str,
        config: ChannelConfig | None = None,
    ) -> Channel:
        """Look up an existing channel or create a new one.

        Transports call this when a message arrives. The first message
        on a new scope creates the channel.
        """
        channel_id = f"{transport}:{scope}"
        if channel_id not in self.channels:
            self.channels[channel_id] = Channel(
                transport=transport,
                scope=scope,
                config=config or ChannelConfig(),
            )
        channel = self.channels[channel_id]
        channel.touch()
        return channel

    def get(self, channel_id: str) -> Channel | None:
        """Look up a channel by its string ID. Returns None if not found."""
        return self.channels.get(channel_id)

    def resolve_config(self, channel: Channel) -> dict:
        """Resolve channel config with agent-level fallbacks."""
        return channel.config.resolve(self.agent_defaults)

    def all(self) -> list[Channel]:
        """Return all registered channels."""
        return list(self.channels.values())

    def by_transport(self, transport: str) -> list[Channel]:
        """Return all channels for a given transport."""
        return [c for c in self.channels.values() if c.transport == transport]


def load_channel_config(config: dict, registry: ChannelRegistry) -> None:
    """Pre-register channels from the config file.

    Precondition: This must run before on_start is called. The ordering
    in main.py enforces this — load_channel_config is called between
    plugin manager creation and the on_start hook.

    Reads the top-level "channels" dict from config. Each key must be
    in "transport:scope" format.

    Raises:
        ValueError: If a channel key does not contain a colon separator.
    """
    channels_config = config.get("channels", {})
    for channel_id, overrides in channels_config.items():
        parts = channel_id.split(":", 1)
        if len(parts) != 2:
            logger.warning("invalid channel key format: %s", channel_id)
            raise ValueError(
                f"Invalid channel key {channel_id!r} — expected 'transport:scope'"
            )
        transport, scope = parts
        if overrides is None:
            overrides = {}
        elif not isinstance(overrides, dict):
            raise ValueError(
                f"Channel config for {channel_id!r} must be a mapping, got {type(overrides).__name__!r}"
            )
        channel_config = ChannelConfig(
            system_prompt=overrides.get("system_prompt"),
            max_context_tokens=overrides.get("max_context_tokens"),
            keep_thinking_in_history=overrides.get("keep_thinking_in_history"),
        )
        registry.get_or_create(transport, scope, config=channel_config)

        logger.info(
            "channel registered from config",
            extra={"channel_id": channel_id},
        )
