"""Channel abstraction for conversation scoping.

This module provides the channel registry that maps transport:scope identifiers
to Channel objects. Each channel has its own conversation history and config
overrides. Channels are created on-demand when messages arrive, or pre-registered
from the config file.

Logging:
    - INFO: channel registered from config
    - INFO: a live channel's transport-wide entries changed on reload
    - WARNING: invalid channel key format
    - DEBUG: channel config resolved
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from corvidae.fork import parse_forks_config
from corvidae.tool_policy import parse_channel_tools
# Transport/session metadata parked on Channel.runtime_overrides — never
# merge into resolved ChannelConfig or forward to the LLM request body.
RUNTIME_META_KEYS = frozenset({"cwd", "tool_backend"})

# The inference parameters set_settings may change and the agent forwards
# into the LLM request body (GitHub issue #206). An allowlist, because the
# body accepts keys that replace the conversation (messages), repoint the
# server (model) or rewrite the tool list (tools, tool_choice), and no
# blocklist can name every key a given server honors.
INFERENCE_KEYS = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens",
    "seed",
})

if TYPE_CHECKING:
    from corvidae.context import ContextWindow

logger = logging.getLogger(__name__)
_prompt_logger = logging.getLogger("corvidae.prompt")


class ReasoningHistory(str, enum.Enum):
    """What happens to ``reasoning_content`` in the in-memory history.

    DROP: strip it from every assistant message.
    EXCHANGE: keep it only on assistant messages after the last user message
        (the current exchange, including a tool chain in progress); strip it
        from completed exchanges. Pair with the server's
        ``preserve_thinking`` off.
    KEEP: never strip it.

    The DB record always keeps it regardless of policy.
    """

    DROP = "drop"
    EXCHANGE = "exchange"
    KEEP = "keep"


#: Config key replaced by ``reasoning_history``; rejected so a stale
#: agent.yaml fails loudly instead of silently getting the default.
_REMOVED_REASONING_KEY = "keep_thinking_in_history"


def parse_reasoning_history(value: object) -> ReasoningHistory:
    """Return ``value`` as a ReasoningHistory, or raise ValueError.

    Only the exact strings "drop", "exchange", and "keep" are accepted —
    booleans from the removed keep_thinking_in_history key are rejected
    rather than mapped.
    """
    # Only strings can name a policy; anything else falls through to the error.
    if isinstance(value, str):
        try:
            return ReasoningHistory(value)
        except ValueError:
            pass
    choices = ", ".join(m.value for m in ReasoningHistory)
    raise ValueError(
        f"reasoning_history must be one of {choices}; got {value!r}"
    )


def _check_reasoning_keys(section: dict, where: str) -> None:
    """Reject the removed key and an invalid reasoning_history in a config
    section (the ``agent`` block or one ``channels`` entry)."""
    if _REMOVED_REASONING_KEY in section:
        raise ValueError(
            f"{where}: {_REMOVED_REASONING_KEY} was removed; use "
            f"reasoning_history: drop | exchange | keep"
        )
    if section.get("reasoning_history") is not None:
        try:
            parse_reasoning_history(section["reasoning_history"])
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc


def validate_agent_config(agent: dict) -> None:
    """Validate the ``agent`` config section before it becomes agent_defaults.

    Raises:
        ValueError: on the removed keep_thinking_in_history key, an
            invalid reasoning_history value, or an invalid agent.forks
            section.
    """
    _check_reasoning_keys(agent, "agent")
    parse_forks_config(agent)


@dataclass
class ChannelConfig:
    """Per-channel configuration. Falls back to agent-level defaults
    for any field set to None."""

    system_prompt: str | list[str] | None = None
    max_context_tokens: int | None = None
    reasoning_history: str | None = None
    max_turns: int | None = None
    # channels.<id>.tools.disabled, layered on the global tools.disabled
    # (issue #149). Unlike the fields above it is not merged by resolve():
    # the Agent snapshots it per conversation (corvidae/tool_policy.py),
    # and a hot reload updates it on channels already registered. The
    # channel's effective list (Channel.disabled_tools) also unions every
    # matching transport-wide entry's list (issue #203).
    disabled_tools: frozenset[str] = frozenset()

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
            "reasoning_history": (
                self.reasoning_history
                if self.reasoning_history is not None
                else agent_defaults.get("reasoning_history", ReasoningHistory.DROP)
            ),
            "max_turns": (
                self.max_turns if self.max_turns is not None
                else agent_defaults.get("max_turns", 10)
            ),
        }

        if runtime_overrides:
            # Skip transport metadata (cwd, tool_backend) — not config fields.
            resolved.update(
                {k: v for k, v in runtime_overrides.items() if k not in RUNTIME_META_KEYS}
            )

        # Normalize the reasoning policy to the enum after every layer has
        # applied, so a bad value from any layer raises here.
        resolved["reasoning_history"] = parse_reasoning_history(
            resolved["reasoning_history"]
        )

        logger.debug(
            "channel config resolved",
            extra={"channel_id": "(unknown)", "resolved": resolved},
        )

        return resolved


#: ChannelConfig fields resolved by ChannelConfig.resolve (disabled_tools
#: is unioned by Channel.disabled_tools instead).
_SCALAR_FIELDS = ("system_prompt", "max_context_tokens", "reasoning_history", "max_turns")

#: The suffix that turns a ``channels:`` key into a transport-wide entry.
WIDE_SUFFIX = "*"

#: The key forms a transport-wide entry accepts, for error messages.
_WIDE_FORMS = "'transport:*' or 'transport:<scope-prefix>*' (e.g. 'signal:*', 'discord:dm:*')"


@dataclass(frozen=True)
class WideEntry:
    """A ``channels:`` entry whose key ends in ``*`` (issue #203).

    It covers every channel whose id starts with ``prefix`` (the key
    without its ``*``), including channels config never names: Signal DMs
    keyed on ACIs, ACP sessions with minted ids, Discord channels the bot
    is mentioned in. Never hashed (``config`` is a mutable dataclass);
    entries are compared by equality only.
    """

    key: str
    prefix: str
    config: ChannelConfig

    def matches(self, channel_id: str) -> bool:
        """Whether this entry covers ``channel_id``."""
        return channel_id.startswith(self.prefix)


def parse_wide_key(key: str) -> str:
    """Return the id prefix of a transport-wide ``channels:`` key, or raise
    ValueError naming the key and the accepted forms.

    Only the trailing ``*`` is special; a second ``*`` or a key without a
    ``transport:`` in front of the ``*`` is rejected.
    """
    prefix = key[: -len(WIDE_SUFFIX)]
    transport, colon, _ = prefix.partition(":")
    if WIDE_SUFFIX in prefix or not colon or not transport:
        raise ValueError(
            f"Invalid transport-wide channel key {key!r} in channels — expected "
            f"{_WIDE_FORMS}, with '*' only at the end"
        )
    return prefix


@dataclass
class Channel:
    """A conversation scope tied to a transport."""

    transport: str          # "irc", "signal", "cli"
    scope: str              # "#lex", "+15551234567", "local"
    config: ChannelConfig = field(default_factory=ChannelConfig)
    conversation: "ContextWindow | None" = None
    created_at: float = field(default_factory=time)
    last_active: float = field(default_factory=time)
    # Consecutive LLM turns without user message. Lives on Channel
    # (not on Agent) because the re-entrant agent loop design means tool
    # results re-enter via on_notify → serial queue → _process_queue_item.
    # The counter must be accessible across re-entries; Channel is the only
    # per-conversation object that persists across these re-entries.
    turn_counter: int = 0
    pending_tool_call_ids: set = field(default_factory=set)  # Tool call IDs awaiting results. Cleared when all results collected.
    runtime_overrides: dict = field(default_factory=dict)  # Per-channel runtime overrides set by the set_settings tool.
    # Transport-wide entries matching this channel, least specific first.
    # ChannelRegistry assigns them at creation and on every config load, so
    # a reload reaches live channels the way an agent: edit does.
    wide_entries: tuple["WideEntry", ...] = ()

    @property
    def id(self) -> str:
        """The string key used for storage and routing."""
        return f"{self.transport}:{self.scope}"

    @property
    def disabled_tools(self) -> frozenset[str]:
        """Tools this channel's config disables: its own list plus every
        matching transport-wide entry's. A union, so no layer can re-enable
        a tool another layer disabled."""
        disabled = set(self.config.disabled_tools)
        for entry in self.wide_entries:
            disabled |= entry.config.disabled_tools
        return frozenset(disabled)

    def touch(self) -> None:
        """Update last_active to now."""
        self.last_active = time()

    def matches_transport(self, transport_name: str) -> bool:
        """Check if this channel belongs to the given transport."""
        return self.transport == transport_name


class ChannelRegistry:
    """Manages the lifecycle of channels."""

    # Class-level agent_defaults allows ConfigWatcherPlugin to update the default
    # config globally by setting ChannelRegistry.agent_defaults = new_config.get("agent", {}).
    # Instance __init__ sets self.agent_defaults which shadows this for existing instances;
    # the class attribute is the target for hot-reload updates via ConfigWatcherPlugin.
    agent_defaults: dict = {}

    def __init__(self, agent_defaults: dict | None = None) -> None:
        """
        Args:
            agent_defaults: Agent-level config dict used as fallback when
                resolving per-channel configuration overrides. Defaults to {}
                when not provided (no-arg construction for entry-point loading).
        """
        self.agent_defaults = agent_defaults if agent_defaults is not None else {}
        self.channels: dict[str, Channel] = {}
        # Transport-wide channels: entries, set by load_channel_config.
        self.wide_entries: tuple[WideEntry, ...] = ()

    def _matching(self, channel_id: str) -> tuple[WideEntry, ...]:
        """The wide entries covering ``channel_id``, least specific first.

        Two matching prefixes of one id are always prefixes of each other,
        so ordering by length is a total order; equal prefixes would be the
        same key, which a mapping can't repeat.
        """
        return tuple(sorted(
            (e for e in self.wide_entries if e.matches(channel_id)),
            key=lambda e: len(e.prefix),
        ))

    def set_wide_entries(self, entries: tuple[WideEntry, ...]) -> None:
        """Replace the transport-wide entries and re-match every live channel.

        Wide entries reload like ``agent:``: a live channel sees the new
        values at its next resolution (a conversation's system prompt and
        tool schemas stay as they started until compaction or restart; see
        docs/configuration.md). Each live channel whose matches change is
        logged, so a reload never changes what a channel gets silently.
        """
        self.wide_entries = tuple(entries)
        for channel in self.channels.values():
            matches = self._matching(channel.id)
            if matches == channel.wide_entries:
                continue
            channel.wide_entries = matches
            logger.info(
                "channel %s: transport-wide entries now %s; a narrowed tools.disabled "
                "is refused at dispatch at once (widening waits for compaction), "
                "other keys apply at the next turn, the system prompt at the next "
                "new conversation",
                channel.id, ", ".join(e.key for e in matches) or "(none)",
                extra={"channel_id": channel.id, "wide_keys": [e.key for e in matches],
                       "disabled": sorted(channel.disabled_tools)},
            )

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
                wide_entries=self._matching(channel_id),
            )
        channel = self.channels[channel_id]
        channel.touch()
        return channel

    def get(self, channel_id: str) -> Channel | None:
        """Look up a channel by its string ID. Returns None if not found."""
        return self.channels.get(channel_id)

    def resolve_config(self, channel: Channel) -> dict:
        """Resolve channel config with agent-level fallbacks and runtime overrides.

        Matching transport-wide entries are overlaid on the agent defaults,
        least specific first, so ChannelConfig.resolve's per-key fallback
        gives: exact entry > wide entries > agent > built-in.
        """
        layered = dict(self.agent_defaults)
        for entry in channel.wide_entries:
            for name in _SCALAR_FIELDS:
                value = getattr(entry.config, name)
                if value is not None:
                    layered[name] = value
        return channel.config.resolve(layered, runtime_overrides=channel.runtime_overrides)

    def all(self) -> list[Channel]:
        """Return all registered channels."""
        return list(self.channels.values())

    def by_transport(self, transport: str) -> list[Channel]:
        """Return all channels for a given transport."""
        return [c for c in self.channels.values() if c.transport == transport]


@dataclass(frozen=True)
class ParsedChannels:
    """The validated result of parse_channel_config, ready to apply.

    ``exact`` holds ``(channel_id, transport, scope, config)`` per exact
    entry; ``wide`` holds the transport-wide entries (issue #203).
    """

    exact: tuple[tuple[str, str, str, ChannelConfig], ...] = ()
    wide: tuple[WideEntry, ...] = ()


def parse_channel_config(config: dict) -> ParsedChannels:
    """Validate the top-level ``channels`` dict and build its entries.

    Pure: touches no registry or other state, so a hot reload can validate
    the channels section before it applies anything (#208). Each key is
    either an exact channel id in "transport:scope" format, or a
    transport-wide key ending in ``*`` ("signal:*", "discord:dm:*") that
    covers every channel whose id starts with the text before the ``*``
    (issue #203). Both kinds accept the same keys.

    Raises:
        ValueError: If a channel key does not contain a colon separator, a
            transport-wide key is malformed, or an entry is not a mapping,
            uses the removed keep_thinking_in_history key, an invalid
            reasoning_history value, or a malformed tools block.
    """
    channels_config = config.get("channels", {})
    parsed: list[tuple[str, str, str, ChannelConfig]] = []
    wide: list[WideEntry] = []
    for channel_id, overrides in channels_config.items():
        # A trailing '*' makes a transport-wide key; anything else is an
        # exact channel id (which may itself contain '*', as IRC allows).
        is_wide = channel_id.endswith(WIDE_SUFFIX)
        if is_wide:
            prefix = parse_wide_key(channel_id)
        else:
            parts = channel_id.split(":", 1)
            if len(parts) != 2:
                logger.warning("invalid channel key format: %s", channel_id)
                raise ValueError(
                    f"Invalid channel key {channel_id!r} — expected 'transport:scope' "
                    f"or a transport-wide {_WIDE_FORMS}"
                )
            transport, scope = parts
        if overrides is None:
            overrides = {}
        elif not isinstance(overrides, dict):
            raise ValueError(
                f"Channel config for {channel_id!r} must be a mapping, got {type(overrides).__name__!r}"
            )
        _check_reasoning_keys(overrides, f"channels.{channel_id}")
        channel_config = ChannelConfig(
            system_prompt=overrides.get("system_prompt"),
            max_context_tokens=overrides.get("max_context_tokens"),
            reasoning_history=overrides.get("reasoning_history"),
            max_turns=overrides.get("max_turns"),
            disabled_tools=parse_channel_tools(
                overrides.get("tools"), f"channels.{channel_id}"
            ),
        )
        if is_wide:
            wide.append(WideEntry(key=channel_id, prefix=prefix, config=channel_config))
        else:
            parsed.append((channel_id, transport, scope, channel_config))
    return ParsedChannels(exact=tuple(parsed), wide=tuple(wide))


def apply_channel_config(parsed: ParsedChannels, registry: ChannelRegistry) -> None:
    """Apply entries from parse_channel_config to the registry.

    Does no validation, so it cannot raise ValueError on parsed input.
    Transport-wide entries are replaced and re-matched against every live
    channel (ChannelRegistry.set_wide_entries), so they reload like
    ``agent:``. A channel already registered keeps its own ChannelConfig
    except for ``tools.disabled``, which is updated in place (shared with
    any Signal ACI channel created from an E.164 alias); the Agent applies
    it to schemas at the next conversation start or compaction, and to
    dispatch at once (corvidae/tool_policy.py).
    """
    # Wide entries first, so exact channels registered below are matched
    # against the new set at creation. A wide key is never a channel.
    registry.set_wide_entries(parsed.wide)

    for channel_id, transport, scope, channel_config in parsed.exact:
        existing = registry.get(channel_id)
        if existing is not None:
            # Reload: only the tool list is live for a registered channel.
            if existing.config.disabled_tools != channel_config.disabled_tools:
                existing.config.disabled_tools = channel_config.disabled_tools
                logger.info(
                    "channel tools.disabled updated by reload; schemas follow at "
                    "the next conversation start or compaction",
                    extra={"channel_id": channel_id,
                           "disabled": sorted(channel_config.disabled_tools)},
                )
            continue
        registry.get_or_create(transport, scope, config=channel_config)

        logger.info(
            "channel registered from config",
            extra={"channel_id": channel_id},
        )


def load_channel_config(config: dict, registry: ChannelRegistry) -> None:
    """Pre-register channels from the config file: parse, then apply.

    Precondition: This must run before on_start is called. The ordering
    in Runtime.start() enforces this — load_channel_config is called between
    plugin manager creation and the on_start hook.

    Every entry is validated (parse_channel_config) before any is applied,
    so a malformed file changes nothing. Hot reload calls the two halves
    itself so it can validate before touching ``agent:`` defaults.

    Raises:
        ValueError: As parse_channel_config.
    """
    apply_channel_config(parse_channel_config(config), registry)


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
