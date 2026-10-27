"""Per-channel tool sets (GitHub issue #149).

``tools.disabled`` drops tools for the whole daemon, in
ToolCollectionPlugin. ``channels.<id>.tools.disabled`` narrows one
channel further, layered on top through the channel-settings resolution
order (PRD §3.11.2): built-in nothing, the global list, then the
channel's list as a union. A channel can only narrow; ``set_settings``
has no key for it, so the model cannot widen its own tools.

Two views of a channel's list exist on purpose:

- The conversation's **ToolView**, resolved when the conversation starts
  and again at a compaction boundary, is what the schemas are built from.
  Tool schemas head the prompt, so they never change mid-conversation
  (principle 2), and forks reuse them (prefix parity, PRD §3.8.2).
- The **live** ChannelConfig list, which a hot reload updates.

Dispatch refuses a tool disabled in either, so a narrowing reload takes
effect at once even while the schemas still list the tool until the next
boundary; a widening one waits for the boundary. Hiding a tool is not
access control: the model can call a tool it saw earlier or one named in
injected text, which is why dispatch enforces the list.

This module imports nothing from the agent or the channel module so both
can use it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.channel import Channel

#: Tools that are the same capability under different names: task_pipeline
#: shells out on its own, so it is shell by another name (PRD §3.5.3).
#: Naming any member of a group disables the whole group.
COUPLED_TOOLS: tuple[frozenset[str], ...] = (frozenset({"shell", "task_pipeline"}),)

#: Keys accepted under ``channels.<id>.tools``.
CHANNEL_TOOLS_KEYS = ("disabled",)


def expand_disabled(names: Iterable[str]) -> frozenset[str]:
    """Return ``names`` plus every tool coupled to one of them."""
    disabled = set(names)
    for group in COUPLED_TOOLS:
        if disabled & group:
            disabled |= group
    return frozenset(disabled)


def parse_disabled_list(value: object, key: str) -> frozenset[str]:
    """Validate a ``disabled`` list of tool names; raise ValueError naming
    ``key``, the rejected value, and what is accepted.

    A bare string is rejected rather than read as a list of characters.
    """
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(
            f"{key} must be a list of tool names (e.g. [shell, web_fetch]); "
            f"got {value!r} ({type(value).__name__})"
        )
    return frozenset(value)


def parse_channel_tools(value: object, where: str) -> frozenset[str]:
    """Parse a ``channels.<id>.tools`` block into the channel's disabled set.

    ``where`` is the channel's config path (``channels.irc:#lex``).
    """
    key = f"{where}.tools"
    if value is None:
        return frozenset()
    accepted = ", ".join(CHANNEL_TOOLS_KEYS)
    if not isinstance(value, dict):
        raise ValueError(
            f"{key} must be a mapping with key(s) {accepted}; "
            f"got {value!r} ({type(value).__name__})"
        )
    unknown = sorted(k for k in value if k not in CHANNEL_TOOLS_KEYS)
    if unknown:
        raise ValueError(
            f"{key}: unknown key(s) {', '.join(repr(k) for k in unknown)}; "
            f"accepted: {accepted}"
        )
    return parse_disabled_list(value.get("disabled"), f"{key}.disabled")


@dataclass(frozen=True)
class ToolView:
    """One conversation's tool set: what its schemas offer.

    Built once per conversation start or compaction boundary and reused
    every turn, so the schema list sent is the same object, byte for byte.
    """

    disabled: frozenset[str] = frozenset()
    tools: dict[str, Callable] = field(default_factory=dict)
    schemas: list[dict] = field(default_factory=list)


def build_tool_view(
    tools: dict[str, Callable], schemas: list[dict], disabled: Iterable[str]
) -> ToolView:
    """Filter the daemon's tools and schemas down to one channel's set."""
    blocked = expand_disabled(disabled)
    return ToolView(
        disabled=blocked,
        tools={name: fn for name, fn in tools.items() if name not in blocked},
        schemas=[s for s in schemas if _schema_name(s) not in blocked],
    )


def _schema_name(schema: dict) -> str | None:
    """The tool name an OpenAI-style function schema declares."""
    function = schema.get("function") if isinstance(schema, dict) else None
    return function.get("name") if isinstance(function, dict) else None


def disabled_on(channel: "Channel") -> frozenset[str]:
    """Tools refused on ``channel`` right now: the conversation's view
    plus the live config, so a narrowing reload applies at once."""
    live = expand_disabled(channel.config.disabled_tools)
    conversation = channel.conversation
    view = getattr(conversation, "tool_view", None) if conversation is not None else None
    return live | view.disabled if view is not None else live


def refusal_message(tool_name: str, channel: "Channel") -> str:
    """The tool result the model gets for a call to a disabled tool."""
    return (
        f"Error: tool '{tool_name}' is disabled on channel {channel.id} "
        f"(channels.{channel.id}.tools.disabled); it was not run. "
        "Do not retry it here; use another tool or answer without it."
    )
