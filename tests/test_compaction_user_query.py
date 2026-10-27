"""Tests for keeping a user query in the post-compaction prompt (GitHub issue #56).

The compaction summary is an assistant message. If the retained tail is all
tool traffic, the rebuilt prompt has no user message and strict chat
templates (Swift-1.5 / Qwen 3.8) raise "No user query found in messages."
Compaction then renders the summary as a user message instead, leaving the
budgeted cut where it is.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.context import ContextWindow, MessageType, message_tokens

from test_mid_conversation_system_role import TemplateError, render_swift


def _channel() -> MagicMock:
    """A minimal stand-in channel for resolve_compaction."""
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
        return await plugin.resolve_compaction(
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


class TestSummaryRoleKeepsUserQuery:
    async def test_tool_only_tail_gets_user_role_summary(self):
        """Budget covers only tool traffic: the cut stays put; the summary becomes user."""
        conv = _window(_long_turn())
        max_tokens = conv.token_estimate()
        plugin = _plugin_retaining(conv, tail=4, max_tokens=max_tokens)

        assert await _compact(plugin, conv, max_tokens) is True

        # Summary first, then exactly the budgeted tail.
        assert len(conv.messages) == 1 + 4
        assert conv.messages[0]["_message_type"] == MessageType.SUMMARY
        assert conv.messages[0]["role"] == "user"
        assert conv.messages[0]["content"].startswith("[Summary of earlier conversation]")

    async def test_compacted_prompt_renders_through_strict_template(self):
        """The rebuilt prompt passes the Swift-1.5 template's user-query check."""
        conv = _window(_long_turn())
        max_tokens = conv.token_estimate()
        plugin = _plugin_retaining(conv, tail=4, max_tokens=max_tokens)

        await _compact(plugin, conv, max_tokens)

        render_swift(_as_server_renders(conv.build_prompt()))  # must not raise

    async def test_tail_with_user_keeps_assistant_summary_and_budget(self):
        """A tail containing a user message is untouched; the summary stays assistant."""
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
        tail = conv.messages[-5:]
        plugin = _plugin_retaining(conv, tail=5, max_tokens=max_tokens)

        assert await _compact(plugin, conv, max_tokens) is True

        assert conv.messages[0]["role"] == "assistant"
        assert conv.messages[1:] == tail

    async def test_single_turn_tool_loop_still_shrinks(self):
        """One request followed by a long tool loop is compacted, not skipped."""
        conv = _window(
            [{"role": "user", "content": "do a long job"}]
            + [m for n in range(1, 13) for m in (_call(n), _result(n))]
        )
        before = conv.token_estimate()
        max_tokens = before
        plugin = _plugin_retaining(conv, tail=4, max_tokens=max_tokens)

        assert await _compact(plugin, conv, max_tokens) is True

        assert conv.token_estimate() < before
        assert conv.messages[0]["role"] == "user"
        render_swift(_as_server_renders(conv.build_prompt()))  # must not raise


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
