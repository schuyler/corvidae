"""RuntimeSettingsPlugin — exposes a set_settings tool for per-channel runtime configuration.

The set_settings tool lets the agent update LLM inference parameters (e.g.
temperature, top_p) and framework parameters (e.g. max_turns) for the current
channel at runtime. An operator-configurable blocklist prevents mutation of
sensitive keys. "system_prompt" is always blocked regardless of operator config.
"""

import logging

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tool import Tool, tool_to_schema

logger = logging.getLogger(__name__)


class RuntimeSettingsPlugin(CorvidaePlugin):
    """Plugin that registers the set_settings tool.

    The default blocklist always includes "system_prompt". Additional
    immutable settings are read from config in on_init.

    Legacy usage: RuntimeSettingsPlugin(pm, immutable_settings=...) is still
    accepted for backward compatibility with existing tests.
    """

    depends_on = frozenset()

    def __init__(self, pm=None, *, immutable_settings: set | None = None) -> None:
        # Backward-compatible: accept optional pm positional arg and
        # immutable_settings keyword arg so existing test code still works.
        if pm is not None:
            self.pm = pm
        self.blocklist: set = {"system_prompt"}
        if immutable_settings is not None:
            self.blocklist |= set(immutable_settings)

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        immutable = config.get("agent", {}).get("immutable_settings", [])
        self.blocklist |= set(immutable)

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        plugin = self

        async def set_settings(settings: dict, _ctx) -> str:
            """Update runtime settings for this conversation. Keys include LLM
            inference parameters (temperature, top_p, top_k, frequency_penalty,
            presence_penalty, max_tokens) and framework parameters (max_turns,
            max_context_tokens, keep_thinking_in_history). Pass null for a key
            to clear that override and revert to the static config value.
            Returns the current settings after the update."""
            # Blocklist check runs first — before any mutations.
            blocked = [k for k in settings if k in plugin.blocklist]
            if blocked:
                logger.warning(
                    "set_settings: blocked keys rejected",
                    extra={"blocked": sorted(blocked)},
                )
                return (
                    f"Error: the following settings are immutable and cannot be changed: "
                    f"{', '.join(sorted(blocked))}"
                )

            channel = _ctx.channel
            if channel is None:
                return "Error: no channel context available"

            for key, value in settings.items():
                if value is None:
                    # Null means "clear this override"; ignore missing keys gracefully.
                    channel.runtime_overrides.pop(key, None)
                else:
                    channel.runtime_overrides[key] = value

            logger.info(
                "set_settings: runtime overrides updated",
                extra={
                    "channel": channel.id,
                    "overrides": dict(channel.runtime_overrides),
                },
            )

            if channel.runtime_overrides:
                overrides_str = ", ".join(
                    f"{k}={v!r}" for k, v in sorted(channel.runtime_overrides.items())
                )
                return f"Settings updated. Current overrides: {overrides_str}"
            return "Settings updated. No active overrides (all reverted to defaults)."

        tool_registry.append(Tool(
            name="set_settings",
            fn=set_settings,
            schema=tool_to_schema(set_settings),
        ))
