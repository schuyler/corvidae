"""Tests for sherman.agent.AgentPlugin."""

import json
from pathlib import Path
import unittest.mock
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from sherman.channel import Channel, ChannelConfig, ChannelRegistry
from sherman.conversation import ConversationLog, init_db
from sherman.hooks import create_plugin_manager

from sherman.agent import AgentPlugin
from sherman.task import TaskPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# BASE_CONFIG migrated from flat llm block to llm.main / llm.background
# per the channel-queue design (breaking config change, design review Co3).
BASE_CONFIG = {
    "llm": {
        "main": {
            "base_url": "http://localhost:8080",
            "model": "test-model",
        },
    },
    "daemon": {
        "session_db": ":memory:",
    },
}

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "keep_thinking_in_history": False,
}


def _make_text_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _make_tool_call_response(calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": calls,
                }
            }
        ]
    }


def _make_tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _build_plugin_and_channel(agent_defaults=None, channel_config=None):
    """Helper: create plugin manager, registry, in-memory DB, plugin, and a channel.

    Returns (plugin, channel, db) so tests can inspect state directly.
    Callers must call task_plugin.on_stop() and db.close() in teardown.
    """
    if agent_defaults is None:
        agent_defaults = AGENT_DEFAULTS

    db = await aiosqlite.connect(":memory:")
    await init_db(db)

    pm = create_plugin_manager()
    registry = ChannelRegistry(agent_defaults)
    pm.register(registry, name="registry")

    # Async mocks for outbound hooks.
    pm.ahook.send_message = AsyncMock()
    pm.ahook.on_agent_response = AsyncMock()
    pm.ahook.on_task_complete = AsyncMock()
    # NOTE: on_notify is NOT mocked — both AgentPlugin and TaskPlugin use it.

    # Register TaskPlugin.
    task_plugin = TaskPlugin(pm)
    pm.register(task_plugin, name="task")
    await task_plugin.on_start(config={})

    plugin = AgentPlugin(pm)
    pm.register(plugin, name="agent_loop")

    # Inject a pre-opened DB so we don't open a second connection inside on_start.
    plugin.db = db
    # Inject the registry directly (tests bypass on_start where _registry is resolved).
    plugin._registry = registry

    channel = registry.get_or_create(
        "test",
        "scope1",
        config=channel_config or ChannelConfig(),
    )

    return plugin, channel, db


async def _drain(plugin, channel):
    """Drain the channel's queue after on_message returns.

    With the channel queue design, on_message is fire-and-enqueue, so
    callers must drain to wait for processing before asserting on results.

    Guards against missing queue (CR2: avoids KeyError when queue was
    never created, e.g. in test_on_task_complete_does_not_call_send_message_directly).
    """
    if channel.id in plugin._queues:
        await plugin._queues[channel.id].drain()


# ---------------------------------------------------------------------------
# Section 1 — on_start
# ---------------------------------------------------------------------------


class TestOnStart:
    async def test_on_start_creates_client_and_db(self):
        """on_start with valid config creates LLMClient and opens DB.
        Verify client.start() is called."""
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client) as mock_cls, \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):

            mock_connect.return_value = MagicMock()

            await plugin.on_start(config=BASE_CONFIG)

        mock_cls.assert_called_once_with(
            base_url="http://localhost:8080",
            model="test-model",
            api_key=None,
            extra_body=None,
        )
        mock_client.start.assert_awaited_once()

    async def test_on_start_collects_tools(self):
        """Register a test plugin implementing register_tools. Verify tools
        are collected in the AgentPlugin and schemas generated."""
        def my_test_tool(x: str) -> str:
            """A test tool."""
            return f"result: {x}"

        from sherman.hooks import hookimpl

        class ToolPlugin:
            @hookimpl
            def register_tools(self, tool_registry):
                tool_registry.append(my_test_tool)

        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        tool_plugin = ToolPlugin()
        pm.register(tool_plugin, name="tool_plugin")

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client), \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config=BASE_CONFIG)

        assert "my_test_tool" in plugin.tools
        assert plugin.tools["my_test_tool"] is my_test_tool
        assert len(plugin.tool_schemas) >= 1
        assert plugin.tool_schemas[0]["function"]["name"] == "my_test_tool"

    async def test_on_start_stores_base_dir_from_config(self):
        """on_start reads config["_base_dir"] and stores it as plugin.base_dir."""
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        config_with_base_dir = dict(BASE_CONFIG)
        config_with_base_dir["_base_dir"] = Path("/some/dir")

        with patch("sherman.agent.LLMClient", return_value=mock_client), \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config=config_with_base_dir)

        assert plugin.base_dir == Path("/some/dir")

    async def test_on_start_defaults_base_dir_to_cwd(self):
        """on_start sets plugin.base_dir to Path(".") when _base_dir is absent."""
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client), \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config=BASE_CONFIG)

        assert plugin.base_dir == Path(".")

    async def test_on_start_missing_llm_main_raises(self):
        """Config without llm.main raises a clear error (flat llm block is rejected)."""
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")
        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        # Old flat config — must be rejected with a clear error
        bad_config = {
            "llm": {
                "base_url": "http://localhost:8080",
                "model": "test-model",
            }
        }

        with pytest.raises((KeyError, ValueError)):
            await plugin.on_start(config=bad_config)

    async def test_on_start_config_parses_llm_main(self):
        """on_start with llm.main creates self.client."""
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        mock_main_client = MagicMock()
        mock_main_client.start = AsyncMock()

        config_with_bg = {
            "llm": {
                "main": {
                    "base_url": "http://localhost:8080",
                    "model": "main-model",
                },
                "background": {
                    "base_url": "http://localhost:8081",
                    "model": "bg-model",
                },
            },
            "daemon": {"session_db": ":memory:"},
        }

        with patch("sherman.agent.LLMClient", return_value=mock_main_client), \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config=config_with_bg)

        # AgentPlugin only creates the main client
        assert plugin.client is mock_main_client
        mock_main_client.start.assert_awaited()


# ---------------------------------------------------------------------------
# Section 2 — on_message: conversation initialization
# ---------------------------------------------------------------------------


class TestOnMessageConversationInit:
    async def test_on_message_initializes_conversation(self):
        """First message on a channel creates ConversationLog, sets
        system_prompt from resolved config, calls load()."""
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hello"))
        plugin.client = mock_client

        assert channel.conversation is None

        await plugin.on_message(channel=channel, sender="user", text="hi")
        await _drain(plugin, channel)

        assert channel.conversation is not None
        assert isinstance(channel.conversation, ConversationLog)
        assert channel.conversation.system_prompt == "You are a test assistant."

        await db.close()

    async def test_on_message_reuses_existing_conversation(self):
        """Second message on the same channel reuses the ConversationLog."""
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("hello"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="first")
        await _drain(plugin, channel)
        conv_first = channel.conversation

        await plugin.on_message(channel=channel, sender="user", text="second")
        await _drain(plugin, channel)
        conv_second = channel.conversation

        assert conv_first is conv_second

        await db.close()


# ---------------------------------------------------------------------------
# Section 3 — on_message: message persistence and loop interaction
# ---------------------------------------------------------------------------


class TestOnMessagePersistenceAndLoop:
    async def test_on_message_appends_user_message(self):
        """User message is persisted in the conversation log before the
        agent loop runs."""
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("response"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="alice", text="test message")
        await _drain(plugin, channel)

        conv = channel.conversation
        user_messages = [m for m in conv.messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert user_messages[0]["content"] == "test message"

        await db.close()

    async def test_on_message_calls_run_agent_turn(self):
        """Verify run_agent_turn is called with the correct arguments."""
        from sherman.agent_loop import AgentTurnResult

        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        plugin.client = mock_client

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "answer"},
            tool_calls=[],
            text="answer",
            latency_ms=1.0,
        )

        with patch("sherman.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            mock_turn.return_value = turn_result
            await plugin.on_message(channel=channel, sender="user", text="query")
            await _drain(plugin, channel)

        mock_turn.assert_awaited_once()
        call_args = mock_turn.call_args
        # First positional arg is the client
        assert call_args[0][0] is mock_client
        # Second positional arg is the messages (build_prompt output, not raw conv.messages)
        messages_arg = call_args[0][1]
        assert messages_arg[0]["role"] == "system"
        assert any(m.get("content") == "query" for m in messages_arg)
        # Third positional arg is tool_schemas
        assert call_args[0][2] is plugin.tool_schemas

        await db.close()

    async def test_on_message_persists_agent_response(self):
        """New messages appended by run_agent_turn are persisted to the
        conversation log."""
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("agent reply"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await _drain(plugin, channel)

        conv = channel.conversation
        assistant_messages = [m for m in conv.messages if m["role"] == "assistant"]
        assert len(assistant_messages) >= 1
        assert assistant_messages[-1]["content"] == "agent reply"

        await db.close()

    async def test_on_message_sends_response(self):
        """Verify send_message hook and on_agent_response hook are called
        with the display text."""
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("my answer"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="question")
        await _drain(plugin, channel)

        plugin.pm.ahook.send_message.assert_awaited_once_with(
            channel=channel,
            text="my answer",
            latency_ms=ANY,
        )
        plugin.pm.ahook.on_agent_response.assert_awaited_once_with(
            channel=channel,
            request_text="question",
            response_text="my answer",
        )

        await db.close()

    async def test_on_message_with_extra_body_from_config(self):
        """Plugin should read extra_body from llm.main config and pass it to LLMClient."""
        extra_body_value = {"id_slot": 1, "cache_prompt": True}
        config = {
            "llm": {
                "main": {
                    "base_url": "http://localhost:8080",
                    "model": "test-model",
                    "extra_body": extra_body_value,
                },
            },
            "daemon": {"session_db": ":memory:"},
            "_base_dir": Path(".")
        }

        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client) as mock_cls, \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config)

        # Verify LLMClient was created with extra_body
        mock_cls.assert_called_once_with(
            base_url="http://localhost:8080",
            model="test-model",
            api_key=None,
            extra_body=extra_body_value,
        )

        await db.close()

    async def test_on_message_without_extra_body_config(self):
        """Missing extra_body in config should create LLMClient with extra_body=None."""
        config = {
            "llm": {
                "main": {
                    "base_url": "http://localhost:8080",
                    "model": "test-model",
                    # Note: no extra_body key
                },
            },
            "daemon": {"session_db": ":memory:"},
            "_base_dir": Path(".")
        }

        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client) as mock_cls, \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config)

        # Verify LLMClient was created with extra_body=None
        mock_cls.assert_called_once_with(
            base_url="http://localhost:8080",
            model="test-model",
            api_key=None,
            extra_body=None,
        )

        await db.close()

    async def test_on_message_with_extra_body_none(self):
        """extra_body=None in config should create LLMClient with extra_body=None."""
        config = {
            "llm": {
                "main": {
                    "base_url": "http://localhost:8080",
                    "model": "test-model",
                    "extra_body": None,
                },
            },
            "daemon": {"session_db": ":memory:"},
            "_base_dir": Path(".")
        }

        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client) as mock_cls, \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config)

        # Verify LLMClient was created with extra_body=None
        mock_cls.assert_called_once_with(
            base_url="http://localhost:8080",
            model="test-model",
            api_key=None,
            extra_body=None,
        )

        await db.close()

    async def test_on_message_with_extra_body_empty(self):
        """extra_body={} in config should create LLMClient with extra_body={}."""
        config = {
            "llm": {
                "main": {
                    "base_url": "http://localhost:8080",
                    "model": "test-model",
                    "extra_body": {},
                },
            },
            "daemon": {"session_db": ":memory:"},
            "_base_dir": Path(".")
        }

        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with patch("sherman.agent.LLMClient", return_value=mock_client) as mock_cls, \
             patch("sherman.agent.aiosqlite.connect", new_callable=AsyncMock) as mock_connect, \
             patch("sherman.agent.init_db", new_callable=AsyncMock):
            mock_connect.return_value = MagicMock()
            await plugin.on_start(config)

        # Verify LLMClient was created with extra_body={}
        mock_cls.assert_called_once_with(
            base_url="http://localhost:8080",
            model="test-model",
            api_key=None,
            extra_body={},
        )

        await db.close()


# ---------------------------------------------------------------------------
# Section 4 — on_message: thinking token handling
# ---------------------------------------------------------------------------


class TestOnMessageThinkingTokens:
    async def test_on_message_strips_thinking_for_display(self):
        """If raw response contains <think> tags, the displayed text
        (sent via send_message and on_agent_response) has them stripped."""
        plugin, channel, db = await _build_plugin_and_channel()

        raw = "<think>internal reasoning</think>clean answer"
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response(raw))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hi")
        await _drain(plugin, channel)

        plugin.pm.ahook.send_message.assert_awaited_once_with(
            channel=channel,
            text="clean answer",
            latency_ms=ANY,
        )

        await db.close()

    async def test_on_message_strips_reasoning_content_from_history(self):
        """When keep_thinking_in_history is False, reasoning_content is
        removed from the newly appended in-memory assistant messages.
        Pre-existing assistant messages are not touched."""
        # keep_thinking_in_history=False is the default in AGENT_DEFAULTS
        plugin, channel, db = await _build_plugin_and_channel()

        # Pre-existing message already in the conversation (simulated prior turn)
        prior_msg = {"role": "assistant", "content": "prior", "reasoning_content": "prior thinking"}

        # Inject a pre-existing conversation state
        conv = ConversationLog(db, channel.id)
        conv.system_prompt = "You are a test assistant."
        conv.messages = [prior_msg]
        channel.conversation = conv

        # New response with reasoning_content
        response_msg = {
            "role": "assistant",
            "content": "new answer",
            "reasoning_content": "new thinking",
        }
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": response_msg}]}
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="query")
        await _drain(plugin, channel)

        # Prior message still has reasoning_content (not touched)
        assert conv.messages[0].get("reasoning_content") == "prior thinking"

        # New assistant message should have reasoning_content stripped
        new_assistant = [m for m in conv.messages[1:] if m.get("role") == "assistant"]
        assert len(new_assistant) >= 1
        assert "reasoning_content" not in new_assistant[-1]

        await db.close()

    async def test_on_message_preserves_reasoning_content_in_history(self):
        """When keep_thinking_in_history is True, reasoning_content stays
        in the in-memory messages."""
        agent_defaults_with_thinking = {
            "system_prompt": "You are a test assistant.",
            "max_context_tokens": 8000,
            "keep_thinking_in_history": True,
        }
        plugin, channel, db = await _build_plugin_and_channel(
            agent_defaults=agent_defaults_with_thinking
        )

        response_msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "kept thinking",
        }
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": response_msg}]}
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="query")
        await _drain(plugin, channel)

        conv = channel.conversation
        assistant_messages = [m for m in conv.messages if m.get("role") == "assistant"]
        assert len(assistant_messages) >= 1
        assert assistant_messages[-1].get("reasoning_content") == "kept thinking"

        await db.close()

    async def test_on_message_preserves_reasoning_content_in_persistent_log(self):
        """Regardless of keep_thinking_in_history, the persisted record in
        the message_log SQLite table must contain reasoning_content.

        Queries message_log directly — does NOT rely on conv.messages, which
        may have been stripped in memory."""
        # keep_thinking_in_history=False (default) strips in-memory, not on disk
        plugin, channel, db = await _build_plugin_and_channel()

        response_msg = {
            "role": "assistant",
            "content": "disk answer",
            "reasoning_content": "disk thinking",
        }
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": response_msg}]}
        )
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="query")
        await _drain(plugin, channel)

        # Query the DB directly for this channel's assistant messages
        async with db.execute(
            "SELECT message FROM message_log WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            rows = await cursor.fetchall()

        persisted = [json.loads(row[0]) for row in rows]
        assistant_persisted = [m for m in persisted if m.get("role") == "assistant"]
        assert len(assistant_persisted) >= 1
        assert assistant_persisted[-1].get("reasoning_content") == "disk thinking"

        await db.close()


# ---------------------------------------------------------------------------
# Section 5 — on_message: compaction and per-channel config
# ---------------------------------------------------------------------------


class TestOnMessageCompactionAndConfig:
    async def test_on_message_compacts_before_agent_turn(self):
        """When token estimate is high, compact_if_needed runs before
        run_agent_turn."""
        from sherman.agent_loop import AgentTurnResult

        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("answer"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hi")
        await _drain(plugin, channel)

        # Verify compact_if_needed was called by patching it on the conversation
        # object. We do this on a second message after the conversation exists.
        conv = channel.conversation
        original_compact = conv.compact_if_needed
        compact_calls = []
        call_order = []

        async def mock_compact(client, max_tokens):
            compact_calls.append((client, max_tokens))
            call_order.append("compact")
            await original_compact(client, max_tokens)

        conv.compact_if_needed = mock_compact

        turn_result = AgentTurnResult(
            message={"role": "assistant", "content": "answer"},
            tool_calls=[],
            text="answer",
            latency_ms=1.0,
        )

        with patch("sherman.agent.run_agent_turn", new_callable=AsyncMock) as mock_turn:
            async def tracking_turn(*args, **kwargs):
                call_order.append("turn")
                return turn_result

            mock_turn.side_effect = tracking_turn

            await plugin.on_message(channel=channel, sender="user", text="second message")
            await _drain(plugin, channel)

        assert len(compact_calls) == 1
        assert compact_calls[0][0] is mock_client
        assert compact_calls[0][1] == 8000  # max_context_tokens from AGENT_DEFAULTS
        assert call_order == ["compact", "turn"]

        await db.close()

    async def test_on_message_per_channel_config(self):
        """Two channels with different configs behave according to their
        resolved config (different system prompts)."""
        db = await aiosqlite.connect(":memory:")
        await init_db(db)

        pm = create_plugin_manager()
        registry = ChannelRegistry({
            "system_prompt": "Default system prompt.",
            "max_context_tokens": 8000,
            "keep_thinking_in_history": False,
        })
        pm.register(registry, name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")
        plugin.db = db
        plugin._registry = registry

        cfg_a = ChannelConfig(system_prompt="Channel A prompt.")
        cfg_b = ChannelConfig(system_prompt="Channel B prompt.")

        channel_a = registry.get_or_create("test", "a", config=cfg_a)
        channel_b = registry.get_or_create("test", "b", config=cfg_b)

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel_a, sender="user", text="hello")
        await plugin.on_message(channel=channel_b, sender="user", text="hello")
        await _drain(plugin, channel_a)
        await _drain(plugin, channel_b)

        assert channel_a.conversation.system_prompt == "Channel A prompt."
        assert channel_b.conversation.system_prompt == "Channel B prompt."

        await db.close()


# ---------------------------------------------------------------------------
# Section 6 — on_stop
# ---------------------------------------------------------------------------


class TestOnStop:
    async def test_on_stop_cleans_up(self):
        """on_stop closes the LLM client and the DB connection."""
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        mock_client = MagicMock()
        mock_client.stop = AsyncMock()
        plugin.client = mock_client

        mock_db = MagicMock()
        mock_db.close = AsyncMock()
        plugin.db = mock_db

        await plugin.on_stop()

        mock_client.stop.assert_awaited_once()
        mock_db.close.assert_awaited_once()

    async def test_on_stop_cancels_all_queues(self):
        """on_stop cancels all channel queue consumers."""
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")
        pm.ahook.send_message = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        plugin = AgentPlugin(pm)
        pm.register(plugin, name="agent_loop")

        mock_client = MagicMock()
        mock_client.stop = AsyncMock()
        plugin.client = mock_client

        mock_db = MagicMock()
        mock_db.close = AsyncMock()
        plugin.db = mock_db

        # Create a couple of channel queues on the plugin so we can verify they are stopped.
        from sherman.queue import SerialQueue, QueueItem

        async def noop(item):
            pass

        q1 = SerialQueue()
        q1.start(noop)
        q2 = SerialQueue()
        q2.start(noop)

        plugin._queues = {
            "test:ch1": q1,
            "test:ch2": q2,
        }

        await plugin.on_stop()

        # All queue consumers must be stopped
        for q in [q1, q2]:
            assert q._consumer_task is None or q._consumer_task.done(), \
                "Queue consumer was not stopped by on_stop()"


# ---------------------------------------------------------------------------
# Section 7 — on_message: tool call round trip
# ---------------------------------------------------------------------------


class TestOnMessageToolCallRoundTrip:
    async def test_on_message_tool_call_round_trip(self):
        """Mock LLM returns a tool call, the tool executes via TaskQueue, then
        LLM returns a final text response via the notification path."""
        plugin, channel, db = await _build_plugin_and_channel()

        tool_result_store = {}

        async def my_plugin_tool(query: str) -> str:
            """A plugin-provided tool."""
            tool_result_store["called_with"] = query
            return "tool output"

        plugin.tools = {"my_plugin_tool": my_plugin_tool}
        from sherman.agent_loop import tool_to_schema
        plugin.tool_schemas = [tool_to_schema(my_plugin_tool)]

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                # First call: tool call dispatched
                _make_tool_call_response([
                    _make_tool_call("call_1", "my_plugin_tool", {"query": "test query"})
                ]),
                # Second call: final text response after tool result arrives
                _make_text_response("final response after tool"),
            ]
        )
        plugin.client = mock_client

        import asyncio

        # Phase 3 async dispatch cycle:
        # 1. on_message enqueues user QueueItem
        await plugin.on_message(channel=channel, sender="user", text="use the tool")
        # 2. Drain: _process_queue_item runs, dispatches tool Task
        await _drain(plugin, channel)
        # 3. Wait for TaskQueue to execute the tool task
        task_plugin = plugin.pm.get_plugin("task")
        await task_plugin.task_queue.queue.join()
        # 4. queue.join() unblocks after task_done() but before on_complete fires.
        #    Yield to let worker call on_complete -> on_notify -> enqueue notification.
        await asyncio.sleep(0)
        # 5. Drain: second _process_queue_item runs with tool result, sends response
        await _drain(plugin, channel)

        assert tool_result_store.get("called_with") == "test query"
        plugin.pm.ahook.send_message.assert_awaited_once_with(
            channel=channel,
            text="final response after tool",
            latency_ms=ANY,
        )

        await task_plugin.on_stop()
        await db.close()


# ---------------------------------------------------------------------------
# Section 8 — system prompt resolution via _ensure_conversation (Phase 2.5)
# ---------------------------------------------------------------------------


class TestEnsureConversationPromptResolution:
    async def test_ensure_conversation_resolves_file_list(self, tmp_path):
        """_ensure_conversation resolves a list of prompt file paths into a
        concatenated string and assigns it to conv.system_prompt.

        Setup: create temp files with known content, set plugin.base_dir to
        tmp_path, then call on_message to trigger _ensure_conversation.
        """
        soul = tmp_path / "soul.md"
        irc = tmp_path / "irc.md"
        soul.write_text("You are Sherman.")
        irc.write_text("This is IRC.")

        agent_defaults = {
            "system_prompt": ["soul.md", "irc.md"],
            "max_context_tokens": 8000,
            "keep_thinking_in_history": False,
        }
        plugin, channel, db = await _build_plugin_and_channel(
            agent_defaults=agent_defaults
        )
        plugin.base_dir = tmp_path

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await _drain(plugin, channel)

        assert channel.conversation is not None
        assert channel.conversation.system_prompt == "You are Sherman.\n\nThis is IRC."

        await db.close()

    async def test_ensure_conversation_string_prompt_unchanged(self):
        """String system_prompt passes through _ensure_conversation unchanged.

        This is a regression test — existing string-based config must
        continue to work identically after Phase 2.5.
        """
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await _drain(plugin, channel)

        assert channel.conversation is not None
        assert channel.conversation.system_prompt == "You are a test assistant."

        await db.close()

    async def test_mixed_config_agent_list_channel_string(self, tmp_path):
        """Agent-level list + channel string override: channel string wins.

        resolve_config() gives the channel string; _ensure_conversation should
        pass it through as a string, not attempt file reads.
        """
        # Create a file that would be read if the agent list were used — it
        # should NOT be read because the channel override wins.
        soul = tmp_path / "soul.md"
        soul.write_text("Agent level content.")

        agent_defaults = {
            "system_prompt": ["soul.md"],
            "max_context_tokens": 8000,
            "keep_thinking_in_history": False,
        }
        channel_cfg = ChannelConfig(system_prompt="Channel string wins.")
        plugin, channel, db = await _build_plugin_and_channel(
            agent_defaults=agent_defaults,
            channel_config=channel_cfg,
        )
        plugin.base_dir = tmp_path

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await _drain(plugin, channel)

        assert channel.conversation is not None
        assert channel.conversation.system_prompt == "Channel string wins."

        await db.close()

    async def test_mixed_config_agent_string_channel_list(self, tmp_path):
        """Agent-level string + channel list override: channel list wins.

        _ensure_conversation should read the channel's file list and set
        conv.system_prompt to the concatenated content.
        """
        f = tmp_path / "channel.md"
        f.write_text("Channel list content.")

        agent_defaults = {
            "system_prompt": "Agent string.",
            "max_context_tokens": 8000,
            "keep_thinking_in_history": False,
        }
        channel_cfg = ChannelConfig(system_prompt=["channel.md"])
        plugin, channel, db = await _build_plugin_and_channel(
            agent_defaults=agent_defaults,
            channel_config=channel_cfg,
        )
        plugin.base_dir = tmp_path

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        plugin.client = mock_client

        await plugin.on_message(channel=channel, sender="user", text="hello")
        await _drain(plugin, channel)

        assert channel.conversation is not None
        assert channel.conversation.system_prompt == "Channel list content."

        await db.close()


# ---------------------------------------------------------------------------
# Section 9 — on_message: error handling
# ---------------------------------------------------------------------------


class TestOnMessageErrorHandling:
    async def test_on_message_run_agent_turn_error(self):
        """When run_agent_turn raises an exception:
        - send_message is called with an error message
        - on_agent_response is NOT called
        - on_message returns without re-raising
        - The user message persisted before the failure remains in the log
        """
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        # chat raises to simulate run_agent_turn failure
        mock_client.chat = AsyncMock(side_effect=RuntimeError("LLM exploded"))
        plugin.client = mock_client

        # Should not raise
        await plugin.on_message(channel=channel, sender="user", text="trigger error")
        await _drain(plugin, channel)

        # send_message should be called with the error text
        plugin.pm.ahook.send_message.assert_awaited_once()
        call_kwargs = plugin.pm.ahook.send_message.call_args
        sent_text = call_kwargs.kwargs["text"]
        assert "error" in sent_text.lower() or "sorry" in sent_text.lower()

        # on_agent_response must NOT be called
        plugin.pm.ahook.on_agent_response.assert_not_awaited()

        # The user message must still be in the persistent log
        conv = channel.conversation
        assert conv is not None
        user_messages = [m for m in conv.messages if m["role"] == "user"]
        assert len(user_messages) >= 1
        assert user_messages[0]["content"] == "trigger error"

        await db.close()


# ---------------------------------------------------------------------------
# Section 10 — on_notify hook
# ---------------------------------------------------------------------------


class TestOnNotify:
    async def test_on_notify_enqueues_and_triggers_agent_loop(self):
        """on_notify enqueues a notification item; after draining, send_message
        is called with the agent's response to the notification."""
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("notified response"))
        plugin.client = mock_client

        await plugin.pm.ahook.on_notify(
            channel=channel, source="test", text="background event happened",
            tool_call_id=None, meta=None,
        )
        await _drain(plugin, channel)

        plugin.pm.ahook.send_message.assert_awaited_once_with(
            channel=channel,
            text="notified response",
            latency_ms=ANY,
        )

        await db.close()

    async def test_on_notify_with_tool_call_id(self):
        """on_notify with tool_call_id passes it through so the conversation
        message uses role=tool."""
        plugin, channel, db = await _build_plugin_and_channel()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("tool result acknowledged"))
        plugin.client = mock_client

        await plugin.pm.ahook.on_notify(
            channel=channel,
            source="task",
            text="task done: success",
            tool_call_id="call_abc123",
            meta=None,
        )
        await _drain(plugin, channel)

        # After draining, agent responded — verify the conversation contained
        # a tool-role message with the right call_id.
        assert channel.conversation is not None
        tool_messages = [
            m for m in channel.conversation.messages
            if m.get("role") == "tool"
        ]
        assert any(
            m.get("tool_call_id") == "call_abc123" for m in tool_messages
        ), f"Expected tool message with call_id=call_abc123 in {tool_messages}"

        await db.close()



# ---------------------------------------------------------------------------
# Section 13 — tool schema: underscore param exclusion (design review C3)
# ---------------------------------------------------------------------------


class TestToolSchema:
    def test_underscore_params_excluded_from_schema(self):
        """Any parameter whose name starts with _ is excluded from the schema."""
        from sherman.agent_loop import tool_to_schema

        async def my_tool(x: str, _injected: str = None, y: int = 0) -> str:
            """A tool with injected params."""
            pass

        schema = tool_to_schema(my_tool)
        properties = schema["function"]["parameters"].get("properties", {})
        assert "_injected" not in properties, \
            "_injected must not appear in schema"
        assert "x" in properties, "x must appear in schema"
        # y may or may not appear depending on whether optional params are included,
        # but the key assertion is that _injected is absent

