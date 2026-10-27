"""ThinkingPlugin — strips <think> blocks and reasoning_content for display.

Extracts the thinking-token handling that was previously inlined in agent.py
into a standalone plugin that can be unregistered without crashing the system.
"""
from __future__ import annotations

import re

from corvidae.channel import ReasoningHistory
from corvidae.hooks import CorvidaePlugin, hookimpl


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_reasoning_content(messages: list[dict]) -> None:
    """Remove reasoning_content from assistant messages in place."""
    for msg in messages:
        if msg.get("role") == "assistant":
            msg.pop("reasoning_content", None)


def completed_exchange_end(messages: list[dict]) -> int:
    """Index of the last user message, or 0 if there is none.

    Assistant messages before this index belong to completed exchanges.
    Only role "user" counts: tool results and system notifications continue
    the current exchange. In window order this boundary is never later
    than the one a Qwen-family chat template computes, so with the server's
    preserve_thinking off the template drops everything stripped here. The
    exception is a tool-call block that build_prompt moves past an
    interleaving user message (docs/configuration.md).
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return 0


class ThinkingPlugin(CorvidaePlugin):
    """Plugin that handles <think> block and reasoning_content stripping.

    Implements three hooks, all driven by the channel's resolved
    ``reasoning_history`` policy (see ReasoningHistory):
    - before_agent_turn: applies the policy to the whole in-memory window
      before each LLM call — strips everything under "drop", completed
      exchanges under "exchange". Also covers history reloaded from the DB,
      which always carries reasoning_content.
    - after_persist_assistant: strips the just-appended message under "drop"
      so the window never holds it even between turns.
    - transform_display_text: strips <think>...</think> blocks from the
      response text before it is sent to the channel.
    """

    depends_on = frozenset({"registry"})

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm

    def _policy(self, channel) -> ReasoningHistory | None:
        """Resolve the channel's reasoning_history, or None without a registry."""
        registry = self.pm.get_plugin("registry")
        if registry is None:
            return None
        return registry.resolve_config(channel)["reasoning_history"]

    @hookimpl
    async def before_agent_turn(self, channel, trace_id, meta) -> None:
        """Apply the reasoning_history policy to the whole window.

        No-op without a registry or a conversation window, and under "keep".
        """
        policy = self._policy(channel)
        conv = channel.conversation
        if policy is None or conv is None:
            return
        # "drop" strips the entire window; "exchange" strips only what
        # precedes the last user message.
        if policy == ReasoningHistory.DROP:
            strip_reasoning_content(conv.messages)
        elif policy == ReasoningHistory.EXCHANGE:
            end = completed_exchange_end(conv.messages)
            strip_reasoning_content(conv.messages[:end])

    @hookimpl
    async def after_persist_assistant(self, channel, message) -> None:
        """Strip reasoning_content from the just-appended message under "drop".

        Under "exchange" the message belongs to the current exchange and
        keeps its reasoning; before_agent_turn strips it once a later user
        message completes the exchange. Returns early (no-op) if the registry
        plugin is not registered.
        """
        if self._policy(channel) == ReasoningHistory.DROP:
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
