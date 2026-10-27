"""Tests for corvidae.thinking — ThinkingPlugin and strip_* utilities."""

from unittest.mock import MagicMock, AsyncMock

import pytest

from corvidae.thinking import ThinkingPlugin, strip_reasoning_content, strip_thinking
from corvidae.hooks import create_plugin_manager


# ---------------------------------------------------------------------------
# strip_thinking tests
# ---------------------------------------------------------------------------


def test_strip_thinking_removes_tags():
    text = "<think>internal monologue</think>actual response"
    assert strip_thinking(text) == "actual response"


def test_strip_thinking_no_tags():
    text = "plain response without thinking"
    assert strip_thinking(text) == text


def test_strip_thinking_multiline():
    text = "<think>\nline one\nline two\n</think>\nfinal answer"
    assert strip_thinking(text) == "final answer"


# ---------------------------------------------------------------------------
# strip_reasoning_content tests
# ---------------------------------------------------------------------------


def test_strip_reasoning_content_removes_from_assistant_messages():
    """strip_reasoning_content removes reasoning_content from assistant messages
    in place, leaves user/tool/system messages alone."""
    messages = [
        {"role": "user", "content": "hi", "reasoning_content": "should not be touched"},
        {"role": "assistant", "content": "hello", "reasoning_content": "thinking..."},
        {"role": "tool", "content": "result", "reasoning_content": "shouldn't be here either"},
    ]
    strip_reasoning_content(messages)

    assert "reasoning_content" not in messages[1]
    assert messages[1]["content"] == "hello"
    # Non-assistant messages must be untouched
    assert messages[0]["reasoning_content"] == "should not be touched"
    assert messages[2]["reasoning_content"] == "shouldn't be here either"


def test_strip_reasoning_content_no_reasoning_content_is_noop():
    """Assistant messages without reasoning_content are left unchanged."""
    messages = [{"role": "assistant", "content": "hi"}]
    strip_reasoning_content(messages)
    assert messages == [{"role": "assistant", "content": "hi"}]


def _make_pm(reasoning_history="drop", registry_missing=False):
    """Build a minimal mock PluginManager for ThinkingPlugin tests.

    If registry_missing=True, pm.get_plugin("registry") returns None.
    Otherwise returns a mock registry whose resolved config includes
    reasoning_history.
    """
    pm = MagicMock()
    if registry_missing:
        pm.get_plugin.return_value = None
    else:
        registry = MagicMock()
        registry.resolve_config.return_value = {
            "reasoning_history": reasoning_history,
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
        """Plugin with reasoning_history="drop" removes reasoning_content
        from the in-memory message dict."""
        pm = _make_pm(reasoning_history="drop")
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
        """Plugin with reasoning_history="keep" leaves reasoning_content
        in the in-memory message dict."""
        pm = _make_pm(reasoning_history="keep")
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


    async def test_after_persist_exchange_keeps_reasoning_content(self):
        """Under "exchange" the just-produced message is part of the current
        exchange, so after_persist_assistant leaves its reasoning alone."""
        pm = _make_pm(reasoning_history="exchange")
        plugin = ThinkingPlugin(pm)
        message = {"role": "assistant", "content": "", "reasoning_content": "mid-chain"}

        await plugin.after_persist_assistant(channel=_make_channel(), message=message)

        assert message["reasoning_content"] == "mid-chain"


# ---------------------------------------------------------------------------
# before_agent_turn — window-wide reasoning policy (#57)
# ---------------------------------------------------------------------------


def _window_channel(messages):
    """A mock channel whose conversation window holds ``messages``."""
    channel = MagicMock()
    channel.conversation.messages = messages
    return channel


def _two_exchange_window():
    """A completed tool-chain exchange followed by a new user message and an
    in-progress tool chain — the shape the "exchange" policy cares about."""
    return [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "", "reasoning_content": "old-1",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "assistant", "content": "first answer", "reasoning_content": "old-2"},
        {"role": "system", "content": "[task] a notification"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "", "reasoning_content": "new-1",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]


def _reasoning(messages):
    """reasoning_content of each assistant message, None where absent."""
    return [m.get("reasoning_content") for m in messages if m["role"] == "assistant"]


class TestBeforeAgentTurn:
    async def test_exchange_strips_only_before_last_user_message(self):
        """"exchange" strips reasoning from completed exchanges and keeps it
        on every assistant message after the last user message."""
        messages = _two_exchange_window()
        plugin = ThinkingPlugin(_make_pm(reasoning_history="exchange"))

        await plugin.before_agent_turn(
            channel=_window_channel(messages), trace_id=None, meta={},
        )

        assert _reasoning(messages) == [None, None, "new-1"]

    async def test_exchange_keeps_whole_chain_when_no_new_user_message(self):
        """Mid tool chain with no newer user message, nothing is stripped —
        the case "drop" gets wrong for Qwen-family templates."""
        messages = _two_exchange_window()[:4]
        plugin = ThinkingPlugin(_make_pm(reasoning_history="exchange"))

        await plugin.before_agent_turn(
            channel=_window_channel(messages), trace_id=None, meta={},
        )

        assert _reasoning(messages) == ["old-1", "old-2"]

    async def test_exchange_ignores_system_and_tool_messages_as_boundary(self):
        """Only role "user" marks an exchange boundary: a system notification
        or tool result after the final answer does not complete the exchange."""
        messages = _two_exchange_window()[:5]  # ends on the system notice
        plugin = ThinkingPlugin(_make_pm(reasoning_history="exchange"))

        await plugin.before_agent_turn(
            channel=_window_channel(messages), trace_id=None, meta={},
        )

        assert _reasoning(messages) == ["old-1", "old-2"]

    async def test_drop_strips_every_assistant_message(self):
        """"drop" strips the whole window, including history reloaded from the
        DB (which carries reasoning_content because the DB copy keeps it)."""
        messages = _two_exchange_window()
        plugin = ThinkingPlugin(_make_pm(reasoning_history="drop"))

        await plugin.before_agent_turn(
            channel=_window_channel(messages), trace_id=None, meta={},
        )

        assert _reasoning(messages) == [None, None, None]

    async def test_keep_is_noop(self):
        """"keep" leaves every message's reasoning in place."""
        messages = _two_exchange_window()
        plugin = ThinkingPlugin(_make_pm(reasoning_history="keep"))

        await plugin.before_agent_turn(
            channel=_window_channel(messages), trace_id=None, meta={},
        )

        assert _reasoning(messages) == ["old-1", "old-2", "new-1"]

    async def test_no_registry_is_noop(self):
        """Without a registry the plugin cannot resolve policy; no-op."""
        messages = _two_exchange_window()
        plugin = ThinkingPlugin(_make_pm(registry_missing=True))

        await plugin.before_agent_turn(
            channel=_window_channel(messages), trace_id=None, meta={},
        )

        assert _reasoning(messages) == ["old-1", "old-2", "new-1"]

    async def test_no_conversation_is_noop(self):
        """A channel without a conversation window must not crash."""
        channel = MagicMock()
        channel.conversation = None
        plugin = ThinkingPlugin(_make_pm(reasoning_history="exchange"))

        await plugin.before_agent_turn(channel=channel, trace_id=None, meta={})


class TestTransformDisplayText:
    async def test_transform_display_text_strips_thinking_tags(self):
        """Text containing <think>...</think> blocks is stripped by the
        ThinkingPlugin wrapper. The wrapper receives the seed's return value
        (the input text) and strips <think> tags from it."""
        pm = create_plugin_manager()
        plugin = ThinkingPlugin(pm)
        pm.register(plugin, name="thinking")
        channel = _make_channel()

        text = "<think>internal reasoning</think>clean answer"
        result = await pm.ahook.transform_display_text(
            channel=channel, text=text, result_message={},
        )

        assert result == "clean answer"

    async def test_transform_display_text_no_tags(self):
        """Text without <think> blocks passes through the ThinkingPlugin
        wrapper unchanged — the wrapper returns the seed value as-is."""
        pm = create_plugin_manager()
        plugin = ThinkingPlugin(pm)
        pm.register(plugin, name="thinking")
        channel = _make_channel()

        text = "plain answer with no think tags"
        result = await pm.ahook.transform_display_text(
            channel=channel, text=text, result_message={},
        )

        assert result == text

    async def test_transform_display_text_empty_after_strip(self):
        """Text that is entirely a <think> block yields an empty string after
        stripping. The wrapper returns "" (not the original text)."""
        pm = create_plugin_manager()
        plugin = ThinkingPlugin(pm)
        pm.register(plugin, name="thinking")
        channel = _make_channel()

        text = "<think>only thinking, no answer</think>"
        result = await pm.ahook.transform_display_text(
            channel=channel, text=text, result_message={},
        )

        assert result == ""


class TestGracefulDegradationWithoutThinkingPlugin:
    async def test_no_thinking_plugin_no_crash(self):
        """When no ThinkingPlugin is registered, the system does not crash.

        after_persist_assistant is a broadcast hook — zero implementations is a no-op.
        transform_display_text with firstresult=True and a seed plugin returns the
        input text unchanged when no other hookimpl is registered.
        """
        pm = create_plugin_manager()
        # ThinkingPlugin intentionally NOT registered.

        mock_channel = MagicMock()

        # Broadcast hook with no implementations — must not raise.
        await pm.ahook.after_persist_assistant(channel=mock_channel, message={})

        # firstresult=True with seed plugin — returns input text unchanged.
        input_text = "<think>foo</think>bar"
        result = await pm.ahook.transform_display_text(
            channel=mock_channel,
            text=input_text,
            result_message={},
        )
        assert result == input_text
