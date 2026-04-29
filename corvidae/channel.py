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
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.context import ContextWindow

logger = logging.getLogger(__name__)
_prompt_logger = logging.getLogger("corvidae.prompt")


@dataclass
class ChannelConfig:
    """Per-channel configuration. Falls back to agent-level defaults
    for any field set to None."""

    system_prompt: str | list[str] | None = None
    max_context_tokens: int | None = None
    keep_thinking_in_history: bool | None = None
    max_turns: int | None = None
    # Future: tool allowlist/denylist, response length, etc.

    def resolve(self, agent_defaults: dict, runtime_overrides: dict | None = None) -> dict:
        """Merge channel overrides with agent-level defaults.

        Channel config wins where set; agent defaults fill the gaps.
        Uses `is not None` for all fields so that falsy values (empty
        string, 0) are treated as intentional overrides.

        Resolution order (last wins):
        1. Built-in defaults
        2. agent_defaults from YAML
        3. ChannelConfig fields (per-channel YAML)
        4. runtime_overrides (set by tool at runtime)
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
            "max_turns": (
                self.max_turns if self.max_turns is not None
                else agent_defaults.get("max_turns", 10)
            ),
        }

        if runtime_overrides:
            resolved.update(runtime_overrides)

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
    conversation: "ContextWindow | None" = None
    created_at: float = field(default_factory=time)
    last_active: float = field(default_factory=time)
    turn_counter: int = 0  # Consecutive LLM turns without user message. Lives on Channel
    pending_tool_call_ids: set = field(default_factory=set)  # Tool call IDs awaiting results. Cleared when all results collected.
    runtime_overrides: dict = field(default_factory=dict)  # Per-channel runtime overrides set by the set_settings tool.
    # (not on Agent) because the re-entrant agent loop design means tool
    # results re-enter via on_notify → serial queue → _process_queue_item.
    # The counter must be accessible across re-entries; Channel is the only
    # per-conversation object that persists across these re-entries.

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

    def __init__(self, agent_defaults: dict | None = None) -> None:
        """
        Args:
            agent_defaults: Agent-level config dict used as fallback when
                resolving per-channel configuration overrides. Defaults to {}
                when not provided (no-arg construction for entry-point loading).
        """
        self.agent_defaults = agent_defaults if agent_defaults is not None else {}
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
        """Resolve channel config with agent-level fallbacks and runtime overrides."""
        return channel.config.resolve(self.agent_defaults, runtime_overrides=channel.runtime_overrides)

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
            max_turns=overrides.get("max_turns"),
        )
        registry.get_or_create(transport, scope, config=channel_config)

        logger.info(
            "channel registered from config",
            extra={"channel_id": channel_id},
        )


def resolve_system_prompt(value: str | list[str], base_dir: Path) -> str:
    """Resolve a system_prompt config value to a string.

    If value is a string, return it directly.
    If value is a list of paths, read each file and concatenate
    with double newlines. Relative paths are resolved against
    base_dir. Absolute paths are used as-is.

    Args:
        value: Either a literal prompt string or a list of file paths
        base_dir: Base directory for resolving relative paths

    Returns:
        The resolved system prompt as a single string

    Raises:
        FileNotFoundError: If any path in the list does not exist.
        TypeError: If value is neither str nor list.

    Logs:
        DEBUG: Resolution method (string/file list) and result length
        WARNING: Empty list resolves to empty string
    """
    if isinstance(value, str):
        _prompt_logger.debug(
            "system prompt resolved from literal string",
            extra={"length": len(value)},
        )
        return value
    if isinstance(value, list):
        if not value:
            _prompt_logger.warning("empty system prompt list resolved to empty string")
            return ""
        parts = []
        for entry in value:
            path = Path(entry)
            if not path.is_absolute():
                path = base_dir / path
            parts.append(path.read_text().strip())
        result = "\n\n".join(parts)
        _prompt_logger.debug(
            "system prompt resolved from file list",
            extra={"file_count": len(value), "length": len(result)},
        )
        return result
    raise TypeError(
        f"system_prompt must be str or list[str], got {type(value).__name__!r}"
    )
