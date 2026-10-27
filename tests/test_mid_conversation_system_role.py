"""Tests for agent.mid_conversation_system_role (GitHub issue #49).

Strict chat templates (Swift-1.5 / Qwen 3.8) raise on any system message
that is not first. With ``mid_conversation_system_role: user`` the prompt
builder renders every non-leading system message as ``user``; the stored
window and the DB rows keep ``role: system``.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import jinja2
import pytest

from corvidae.correlation import mint_correlation_id
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import hookimpl

from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response

TEMPLATE_PATH = Path(__file__).parent / "fixtures" / "chat_templates" / "swift-1.5.jinja"


# ---------------------------------------------------------------------------
# Template rendering helper
# ---------------------------------------------------------------------------


class TemplateError(Exception):
    """Raised by the raise_exception shim (llama.cpp's minja provides it)."""


def _raise_exception(message: str):
    raise TemplateError(message)


def render_swift(messages: list[dict]) -> str:
    """Render messages through the Swift-1.5 chat template as llama-server would."""
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = _raise_exception
    template = env.from_string(TEMPLATE_PATH.read_text())
    return template.render(messages=messages, add_generation_prompt=True)


def _window_with_context() -> ContextWindow:
    """A window holding a user turn followed by a funnel-style CONTEXT message."""
    conv = ContextWindow("test:scope")
    conv.system_prompt = "You are Lex."
    conv.append({"role": "user", "content": "hello"})
    conv.append({"role": "assistant", "content": "hi"})
    conv.append({"role": "user", "content": "what did we discuss?"})
    conv.append(
        {"role": "system", "content": "[CONTEXT from memory]\nfact\n[end CONTEXT from memory]"},
        MessageType.CONTEXT,
    )
    return conv


# ---------------------------------------------------------------------------
# build_prompt unit behavior
# ---------------------------------------------------------------------------


class TestBuildPromptRole:
    def test_default_keeps_system_role(self):
        """With no argument, non-leading system messages stay system."""
        prompt = _window_with_context().build_prompt()
        assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user", "system"]

    def test_explicit_system_keeps_system_role(self):
        prompt = _window_with_context().build_prompt(mid_conversation_system_role="system")
        assert prompt[-1]["role"] == "system"

    def test_user_rewrites_non_leading_system(self):
        prompt = _window_with_context().build_prompt(mid_conversation_system_role="user")
        assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user", "user"]
        # Content is carried over unchanged.
        assert prompt[-1]["content"].startswith("[CONTEXT from memory]")

    def test_user_leaves_leading_system_untouched(self):
        prompt = _window_with_context().build_prompt(mid_conversation_system_role="user")
        assert prompt[0] == {"role": "system", "content": "You are Lex."}

    def test_window_messages_unchanged(self):
        """The rewrite applies to the prompt copy only."""
        conv = _window_with_context()
        before = [dict(m) for m in conv.messages]
        conv.build_prompt(mid_conversation_system_role="user")
        assert conv.messages == before
        assert conv.messages[-1]["role"] == "system"

    def test_prefix_stable_across_turns(self):
        """Appending a turn leaves the previously rendered prefix identical."""
        conv = _window_with_context()
        first = conv.build_prompt(mid_conversation_system_role="user")
        conv.append({"role": "assistant", "content": "we discussed X"})
        conv.append({"role": "system", "content": "[notice]\n\nlater"})
        second = conv.build_prompt(mid_conversation_system_role="user")
        assert second[: len(first)] == first
        assert second[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Swift-1.5 template rendering
# ---------------------------------------------------------------------------


class TestSwiftTemplate:
    def test_stock_context_message_raises(self):
        prompt = _window_with_context().build_prompt()
        with pytest.raises(TemplateError, match="System message must be at the beginning"):
            render_swift(prompt)

    def test_user_role_context_message_renders(self):
        prompt = _window_with_context().build_prompt(mid_conversation_system_role="user")
        rendered = render_swift(prompt)
        assert rendered.startswith("<|im_start|>system\n")
        assert "<|im_start|>user\n[CONTEXT from memory]" in rendered

    def test_notification_only_window(self):
        """A window holding only a notification has no user query today."""
        conv = ContextWindow("test:scope")
        conv.system_prompt = "You are Lex."
        conv.append({"role": "system", "content": "[task]\n\nbackground event"})
        # Stock: the template's user-query scan runs before its per-message
        # role check, so this window fails on the missing user turn.
        with pytest.raises(TemplateError, match="No user query found"):
            render_swift(conv.build_prompt())
        rendered = render_swift(conv.build_prompt(mid_conversation_system_role="user"))
        assert "<|im_start|>user\n[task]\n\nbackground event<|im_end|>" in rendered


# ---------------------------------------------------------------------------
# Agent wiring: config, and the funnel / notification paths
# ---------------------------------------------------------------------------


class _ContextInjector:
    """Stand-in for a funnel producer: appends a CONTEXT system message."""

    @hookimpl
    async def before_agent_turn(self, channel, correlation_id, meta):
        channel.conversation.append(
            {"role": "system", "content": "[CONTEXT from memory]\nfact\n[end CONTEXT from memory]"},
            MessageType.CONTEXT,
        )


@pytest.fixture
async def make_agent():
    """Factory for an agent with mid_conversation_system_role configured
    (None = unset). Teardown closes the DB and stops the task plugin even
    when the test fails — a leaked aiosqlite thread hangs interpreter exit."""
    built = []

    async def _make(role: str | None):
        plugin, channel, db = await build_plugin_and_channel()
        built.append((plugin, db))
        agent_cfg = {} if role is None else {"mid_conversation_system_role": role}
        await plugin.on_init(pm=plugin.pm, config={"agent": agent_cfg})
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        plugin._client = mock_client
        return plugin, channel, db, mock_client

    yield _make
    for plugin, db in built:
        await plugin.pm.get_plugin("task").on_stop()
        await db.close()


def _sent_roles(mock_client) -> list[str]:
    messages = mock_client.chat.call_args.args[0]
    return [m["role"] for m in messages]


async def _db_roles(db) -> list[str]:
    async with db.execute("SELECT message FROM message_log ORDER BY id") as cur:
        return [json.loads(row[0])["role"] for row in await cur.fetchall()]


class TestAgentWiring:
    async def test_funnel_context_sent_as_user(self, make_agent):
        plugin, channel, db, client = await make_agent("user")
        plugin.pm.register(_ContextInjector(), name="injector")
        await plugin.on_message(channel=channel, sender="u", text="hi", correlation_id=mint_correlation_id())
        await drain(plugin, channel)
        assert _sent_roles(client) == ["system", "user", "user"]
        # Window and DB keep the system role.
        assert channel.conversation.messages[1]["role"] == "system"
        assert "system" in await _db_roles(db)

    async def test_notification_sent_as_user(self, make_agent):
        plugin, channel, db, client = await make_agent("user")
        await plugin.pm.ahook.on_notify(
            channel=channel, source="task", text="background event",
            tool_call_id=None, meta=None,
        )
        await drain(plugin, channel)
        sent = client.chat.call_args.args[0]
        assert [m["role"] for m in sent] == ["system", "user"]
        assert sent[1]["content"] == "[task]\n\nbackground event"
        assert await _db_roles(db) == ["system", "assistant"]

    async def test_default_is_system(self, make_agent):
        plugin, channel, db, client = await make_agent(None)
        plugin.pm.register(_ContextInjector(), name="injector")
        await plugin.on_message(channel=channel, sender="u", text="hi", correlation_id=mint_correlation_id())
        await drain(plugin, channel)
        assert _sent_roles(client) == ["system", "user", "system"]

    async def test_invalid_value_falls_back_to_system(self, caplog, make_agent):
        plugin, channel, db, client = await make_agent("assistant")
        assert "mid_conversation_system_role" in caplog.text
        plugin.pm.register(_ContextInjector(), name="injector")
        await plugin.on_message(channel=channel, sender="u", text="hi", correlation_id=mint_correlation_id())
        await drain(plugin, channel)
        assert _sent_roles(client) == ["system", "user", "system"]

    async def test_config_reload_updates_role(self, make_agent):
        plugin, channel, db, client = await make_agent(None)
        plugin.pm.register(_ContextInjector(), name="injector")
        await plugin.on_config_reload(config={"agent": {"mid_conversation_system_role": "user"}})
        plugin._client = client  # reload re-borrows from the (absent) llm plugin
        await plugin.on_message(channel=channel, sender="u", text="hi", correlation_id=mint_correlation_id())
        await drain(plugin, channel)
        assert _sent_roles(client)[-1] == "user"
