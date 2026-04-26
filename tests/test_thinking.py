"""Tests for sherman.thinking.ThinkingPlugin."""

from unittest.mock import MagicMock, AsyncMock

import pytest

from sherman.thinking import ThinkingPlugin
from sherman.hooks import create_plugin_manager, call_firstresult_hook


def _make_pm(keep_thinking_in_history=False, registry_missing=False):
    """Build a minimal mock PluginManager for ThinkingPlugin tests.

    If registry_missing=True, pm.get_plugin("registry") returns None.
    Otherwise returns a mock registry whose resolved config includes
    keep_thinking_in_history.
    """
    pm = MagicMock()
    if registry_missing:
        pm.get_plugin.return_value = None
    else:
        registry = MagicMock()
        registry.resolve_config.return_value = {
            "keep_thinking_in_history": keep_thinking_in_history,
        }

        def _get_plugin(name):
            if name == "registry":
                return registry
            return None

        pm.get_plugin.side_effect = _get_plugin

    return pm


def _make_channel():
    channel = MagicMock()
    return channel


class TestAfterPersistAssistant:
    async def test_after_persist_strips_reasoning_content(self):
        """Plugin with keep_thinking_in_history=False removes reasoning_content
        from the in-memory message dict."""
        pm = _make_pm(keep_thinking_in_history=False)
        plugin = ThinkingPlugin(pm)
        channel = _make_channel()

        message = {
            "role": "assistant",
            "content": "hello",
            "reasoning_content": "internal thoughts",
        }

        await plugin.after_persist_assistant(channel=channel, message=message)

        assert "reasoning_content" not in message
        assert message["content"] == "hello"

    async def test_after_persist_preserves_reasoning_content_when_configured(self):
        """Plugin with keep_thinking_in_history=True leaves reasoning_content
        in the in-memory message dict."""
        pm = _make_pm(keep_thinking_in_history=True)
        plugin = ThinkingPlugin(pm)
        channel = _make_channel()

        message = {
            "role": "assistant",
            "content": "hello",
            "reasoning_content": "internal thoughts",
        }

        await plugin.after_persist_assistant(channel=channel, message=message)

        assert message["reasoning_content"] == "internal thoughts"
        assert message["content"] == "hello"

    async def test_after_persist_no_registry_is_noop(self):
        """When pm.get_plugin('registry') returns None, after_persist_assistant
        must not crash and must leave the message unchanged."""
        pm = _make_pm(registry_missing=True)
        plugin = ThinkingPlugin(pm)
        channel = _make_channel()

        message = {
            "role": "assistant",
            "content": "hello",
            "reasoning_content": "internal thoughts",
        }

        # Must not raise
        await plugin.after_persist_assistant(channel=channel, message=message)

        # Message is untouched
        assert message["reasoning_content"] == "internal thoughts"


class TestTransformDisplayText:
    async def test_transform_display_text_strips_thinking_tags(self):
        """Text containing <think>...</think> blocks is stripped; the
        transformed string (without the tags) is returned."""
        pm = _make_pm()
        plugin = ThinkingPlugin(pm)
        channel = _make_channel()
        result_message = MagicMock()

        text = "<think>internal reasoning</think>clean answer"
        result = await plugin.transform_display_text(
            channel=channel, text=text, result_message=result_message
        )

        assert result == "clean answer"
        assert result is not None

    async def test_transform_display_text_no_tags(self):
        """Text without <think> blocks returns None (signal: no transformation
        needed; caller keeps original text)."""
        pm = _make_pm()
        plugin = ThinkingPlugin(pm)
        channel = _make_channel()
        result_message = MagicMock()

        text = "plain answer with no think tags"
        result = await plugin.transform_display_text(
            channel=channel, text=text, result_message=result_message
        )

        assert result is None

    async def test_transform_display_text_empty_after_strip(self):
        """Text that is entirely a <think> block yields an empty string after
        stripping. The plugin must return "" (not None) so the caller can
        distinguish 'intentional empty result' from 'no transformation'."""
        pm = _make_pm()
        plugin = ThinkingPlugin(pm)
        channel = _make_channel()
        result_message = MagicMock()

        text = "<think>only thinking, no answer</think>"
        result = await plugin.transform_display_text(
            channel=channel, text=text, result_message=result_message
        )

        assert result == ""
        assert result is not None


class TestGracefulDegradationWithoutThinkingPlugin:
    async def test_no_thinking_plugin_no_crash(self):
        """When no ThinkingPlugin is registered, the system does not crash.

        after_persist_assistant is a broadcast hook — zero implementations is a no-op.
        transform_display_text via call_firstresult_hook returns None when no impl exists.
        """
        pm = create_plugin_manager()
        # ThinkingPlugin intentionally NOT registered.

        mock_channel = MagicMock()

        # Broadcast hook with no implementations — must not raise.
        await pm.ahook.after_persist_assistant(channel=mock_channel, message={})

        # firstresult hook with no implementations — must return None.
        result = await call_firstresult_hook(
            pm,
            "transform_display_text",
            channel=mock_channel,
            text="<think>foo</think>bar",
            result_message={},
        )
        assert result is None
