"""Tests for anchoring the post-compaction window at a user message (GitHub issue #56).

The compaction summary is an assistant message. If the retained tail is all
tool traffic, the rebuilt prompt has no user message and strict chat
templates (Swift-1.5 / Qwen 3.8) raise "No user query found in messages."
Compaction must start the retained block at a user message.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.context import ContextWindow, MessageType, message_tokens

from test_mid_conversation_system_role import TemplateError, render_swift


def _channel() -> MagicMock:
    """A minimal stand-in channel for compact_conversation."""
    channel = MagicMock()
    channel.id = "test:anchor"
    return channel


def _call(n: int) -> dict:
    """An assistant tool-call message with call id c<n>."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": f"c{n}",
            "type": "function",
            "function": {"name": "shell", "arguments": '{"cmd": "ls"}'},
        }],
    }


def _result(n: int) -> dict:
    """The tool result answering call c<n>."""
    return {"role": "tool", "tool_call_id": f"c{n}", "content": "x" * 200}


def _window(messages: list[dict]) -> ContextWindow:
    """A ContextWindow holding messages as MESSAGE-typed entries."""
    conv = ContextWindow("test:anchor")
    conv.system_prompt = "You are Lex."
    for m in messages:
        conv.append(m, MessageType.MESSAGE)
    return conv


def _plugin_retaining(conv: ContextWindow, tail: int, max_tokens: int) -> CompactionPlugin:
    """A CompactionPlugin whose retention budget fits exactly the last `tail` messages."""
    plugin = CompactionPlugin(pm=None)
    tail_tokens = sum(message_tokens(m) for m in conv.messages[-tail:])
    plugin._compaction_retention = (tail_tokens + 0.5) / max_tokens
    return plugin


def _as_server_renders(prompt: list[dict]) -> list[dict]:
    """Parse tool-call arguments into objects, as llama-server does before templating."""
    out = []
    for m in prompt:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [
                {**c, "function": {**c["function"],
                                   "arguments": json.loads(c["function"]["arguments"])}}
                for c in m["tool_calls"]
            ]}
        out.append(m)
    return out


async def _compact(plugin: CompactionPlugin, conv: ContextWindow, max_tokens: int):
    """Run compaction with a stub summarizer."""
    async def fake_summarize(messages, prior_summaries=None, max_tokens=None):
        return "summary text"

    with patch.object(plugin, "_summarize", side_effect=fake_summarize):
        return await plugin.compact_conversation(
            channel=_channel(), conversation=conv, max_tokens=max_tokens
        )


def _long_turn() -> list[dict]:
    """A history whose latest turn is one user message followed by a long tool loop."""
    return [
        {"role": "user", "content": "hello " * 40},
        {"role": "assistant", "content": "hi " * 40},
        {"role": "user", "content": "list the files"},
        _call(1), _result(1),
        _call(2), _result(2),
        _call(3), _result(3),
        _call(4), _result(4),
    ]


class TestRetainedBlockStartsAtUser:
    async def test_tool_only_tail_extends_back_to_user(self):
        """Budget covers only tool traffic: retention extends back to the user turn."""
        conv = _window(_long_turn())
        max_tokens = conv.token_estimate()
        plugin = _plugin_retaining(conv, tail=4, max_tokens=max_tokens)

        assert await _compact(plugin, conv, max_tokens) is True

        # Summary first, then the retained block opening at the user message.
        assert conv.messages[0]["_message_type"] == MessageType.SUMMARY
        assert conv.messages[1]["role"] == "user"
        assert conv.messages[1]["content"] == "list the files"
        assert len(conv.messages) == 1 + 9

    async def test_compacted_prompt_renders_through_strict_template(self):
        """The rebuilt prompt passes the Swift-1.5 template's user-query check."""
        conv = _window(_long_turn())
        max_tokens = conv.token_estimate()
        plugin = _plugin_retaining(conv, tail=4, max_tokens=max_tokens)

        await _compact(plugin, conv, max_tokens)

        render_swift(_as_server_renders(conv.build_prompt()))  # must not raise

    async def test_tail_containing_user_trims_forward_to_it(self):
        """Budget starts mid tool loop but includes a later user turn: trim to that user."""
        conv = _window([
            {"role": "user", "content": "hello " * 40},
            {"role": "assistant", "content": "hi " * 40},
            {"role": "user", "content": "first task"},
            _call(1), _result(1),
            _call(2), _result(2),
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "second task"},
            {"role": "assistant", "content": "ok"},
        ])
        max_tokens = conv.token_estimate()
        # Last 5 messages: result(2), "done", "second task", "ok" plus call(2).
        plugin = _plugin_retaining(conv, tail=5, max_tokens=max_tokens)

        assert await _compact(plugin, conv, max_tokens) is True

        assert [m["role"] for m in conv.messages] == ["assistant", "user", "assistant"]
        assert conv.messages[1]["content"] == "second task"

    async def test_only_user_at_start_skips_compaction(self):
        """If anchoring would retain everything, compaction is a no-op."""
        conv = _window([
            {"role": "user", "content": "do a long job"},
            _call(1), _result(1),
            _call(2), _result(2),
            _call(3), _result(3),
        ])
        before = list(conv.messages)
        max_tokens = conv.token_estimate()
        plugin = _plugin_retaining(conv, tail=2, max_tokens=max_tokens)

        assert await _compact(plugin, conv, max_tokens) is None
        assert conv.messages == before


class TestTemplateReproduction:
    def test_tool_only_window_after_summary_is_rejected(self):
        """Guard: the fixture template rejects a summary followed by tool traffic only."""
        prompt = [
            {"role": "system", "content": "You are Lex."},
            {"role": "assistant", "content": "[Summary of earlier conversation]\n..."},
            _call(1), _result(1),
        ]
        with pytest.raises(TemplateError, match="No user query"):
            render_swift(_as_server_renders(prompt))
