"""ThinkingPlugin — strips <think> blocks and reasoning_content for display.

Extracts the thinking-token handling that was previously inlined in agent.py
into a standalone plugin that can be unregistered without crashing the system.
"""
from __future__ import annotations

import re

from corvidae.hooks import CorvidaePlugin, hookimpl


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_reasoning_content(messages: list[dict]) -> None:
    """Remove reasoning_content from assistant messages in place."""
    for msg in messages:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)


class ThinkingPlugin(CorvidaePlugin):
    """Plugin that handles <think> block and reasoning_content stripping.

    Implements two hooks:
    - after_persist_assistant: strips reasoning_content from the in-memory
      message when keep_thinking_in_history is False.
    - transform_display_text: strips <think>...</think> blocks from the
      response text before it is sent to the channel.
    """

    depends_on = frozenset({"registry"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm

    @hookimpl
    async def after_persist_assistant(self, channel, message) -> None:
        """Strip reasoning_content from in-memory message when configured.

        Reads keep_thinking_in_history from the resolved channel config.
        If False, calls strip_reasoning_content on the in-memory message.
        Returns early (no-op) if the registry plugin is not registered.
        """
        registry = self.pm.get_plugin("registry")
        if registry is None:
            return
        resolved = registry.resolve_config(channel)
        if not resolved["keep_thinking_in_history"]:
            strip_reasoning_content([message])

    @hookimpl(wrapper=True)
    def transform_display_text(self, **kwargs):
        """Wrap the transform_display_text chain to strip <think> tags.

        Receives the chain result (from inner hooks or seed) and strips
        <think>...</think> blocks if present.
        """
        result = yield
        if result is not None:
            return strip_thinking(result)
        return result
