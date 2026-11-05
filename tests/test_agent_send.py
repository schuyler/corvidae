"""Tests for cross-channel agent send (#234).

AGENT_SEND queue items deliver a message on a channel without starting
a new agent turn. The message is recorded in the channel's history,
persisted, and delivered through the owning transport.
"""

from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from corvidae.agent import Agent, QueueItem, QueueItemRole
from corvidae.channel import ChannelConfig, ChannelRegistry
from corvidae.context import ContextWindow
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

from helpers import build_plugin_and_channel, drain, sent, sent_texts


# ---------------------------------------------------------------------------
# Section 1 — enqueue_agent_send
# ---------------------------------------------------------------------------


class TestEnqueueAgentSend:
    async def test_enqueues_agent_send_item(self):
        """enqueue_agent_send puts an AGENT_SEND item on the target queue."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        processed = []
        orig_process = plugin._process_agent_send
        async def spy_process(item, ch):
            processed.append(item)
            await orig_process(item, ch)
        plugin._process_agent_send = spy_process

        await plugin.enqueue_agent_send(channel.id, "hello from elsewhere", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert len(processed) == 1
        assert processed[0].role == QueueItemRole.AGENT_SEND
        assert processed[0].content == "hello from elsewhere"

        await db.close()

    async def test_raises_for_unregistered_channel(self):
        """enqueue_agent_send raises KeyError for a channel not in the registry."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        with pytest.raises(KeyError, match="not registered"):
            await plugin.enqueue_agent_send("nonexistent:channel", "hello")

        await db.close()

    async def test_raises_when_not_started(self):
        """enqueue_agent_send raises RuntimeError when the agent has not started."""
        plugin, channel, db = await build_plugin_and_channel()
        # _mark_started is NOT called — the agent is not started.

        with pytest.raises(RuntimeError, match="not started"):
            await plugin.enqueue_agent_send(channel.id, "hello")

        await db.close()


# ---------------------------------------------------------------------------
# Section 2 — _process_agent_send: history and persistence
# ---------------------------------------------------------------------------


class TestProcessAgentSendHistory:
    async def test_appends_assistant_message_to_conversation(self):
        """The agent's message is appended to the target channel's conversation
        as an assistant message."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        # Initialize the conversation with a prior exchange.
        conv = ContextWindow(channel.id)
        conv.system_prompt = "You are a test assistant."
        conv.messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        channel.conversation = conv

        await plugin.enqueue_agent_send(channel.id, "cross-channel message", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assistant_messages = [m for m in conv.messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 2
        assert assistant_messages[-1]["content"] == "cross-channel message"

        await db.close()

    async def test_initializes_conversation_if_needed(self):
        """If the target channel has no conversation, one is created."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        assert channel.conversation is None

        await plugin.enqueue_agent_send(channel.id, "first message", trace_id=mint_trace_id())
        await drain(plugin, channel)

        assert channel.conversation is not None
        assert isinstance(channel.conversation, ContextWindow)
        assistant_messages = [m for m in channel.conversation.messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["content"] == "first message"

        await db.close()

    async def test_persists_message_to_log(self):
        """The agent's message is persisted to the message_log."""
        import json
        import aiosqlite

        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        await plugin.enqueue_agent_send(channel.id, "persisted message", trace_id=mint_trace_id())
        await drain(plugin, channel)

        async with db.execute(
            "SELECT message FROM message_log WHERE channel_id = ?",
            (channel.id,),
        ) as cursor:
            rows = await cursor.fetchall()

        persisted = [json.loads(row[0]) for row in rows]
        assistant_persisted = [m for m in persisted if m.get("role") == "assistant"]
        assert len(assistant_persisted) == 1
        assert assistant_persisted[0]["content"] == "persisted message"

        await db.close()


# ---------------------------------------------------------------------------
# Section 3 — _process_agent_send: transport delivery
# ---------------------------------------------------------------------------


class TestProcessAgentSendDelivery:
    async def test_sends_outbound_message_to_transport(self):
        """The message is delivered through the send hook as an OutboundMessage."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        trace = mint_trace_id()
        await plugin.enqueue_agent_send(channel.id, "delivered message", trace_id=trace)
        await drain(plugin, channel)

        sent_messages = sent(plugin.pm, OutboundMessage)
        assert len(sent_messages) == 1
        sent_channel, event = sent_messages[0]
        assert sent_channel is channel
        assert event.text == "delivered message"
        assert event.trace_id == trace

        await db.close()

    async def test_no_llm_call_for_agent_send(self):
        """An AGENT_SEND item does not trigger an LLM call (no new turn)."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock()
        plugin._client = mock_client

        await plugin.enqueue_agent_send(channel.id, "no turn here", trace_id=mint_trace_id())
        await drain(plugin, channel)

        mock_client.chat.assert_not_awaited()

        await db.close()


# ---------------------------------------------------------------------------
# Section 4 — serialization: lands between exchanges
# ---------------------------------------------------------------------------


class TestAgentSendSerialization:
    async def test_agent_send_waits_for_in_flight_turn(self):
        """An AGENT_SEND item queued while a turn is in flight is processed
        after the turn completes, not interleaved with it."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        # Start a user turn.
        mock_client = MagicMock()
        from llm_response_fixtures import _make_text_response
        mock_client.chat = AsyncMock(return_value=_make_text_response("turn response"))
        plugin._client = mock_client

        # Enqueue the user message and the agent send in sequence.
        user_trace = mint_trace_id()
        send_trace = mint_trace_id()
        await plugin.on_message(channel=channel, sender="user", text="do work", trace_id=user_trace)
        await plugin.enqueue_agent_send(channel.id, "queued while working", trace_id=send_trace)

        await drain(plugin, channel)

        conv = channel.conversation
        # The user message, the turn's response, and the agent send are all
        # in the conversation, in order.
        user_msgs = [m for m in conv.messages if m["role"] == "user"]
        assistant_msgs = [m for m in conv.messages if m["role"] == "assistant"]
        assert user_msgs[0]["content"] == "do work"
        assert assistant_msgs[-1]["content"] == "queued while working"

        await db.close()

# ---------------------------------------------------------------------------
# Section 5 — edge cases
# ---------------------------------------------------------------------------


class TestAgentSendEdgeCases:
    async def test_consecutive_assistant_messages(self):
        """An AGENT_SEND after a completed turn produces consecutive assistant
        messages. This is acceptable: ContextWindow does not require
        alternating roles, and the agent's send is a distinct utterance."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        # Simulate a completed turn: user message + assistant response.
        conv = ContextWindow(channel.id)
        conv.system_prompt = "You are a test assistant."
        conv.messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        channel.conversation = conv

        await plugin.enqueue_agent_send(channel.id, "cross-channel note", trace_id=mint_trace_id())
        await drain(plugin, channel)

        # Two consecutive assistant messages — this is the expected shape.
        assistant_msgs = [m for m in conv.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0]["content"] == "earlier answer"
        assert assistant_msgs[1]["content"] == "cross-channel note"

        await db.close()

    async def test_transport_failure_still_records_in_history(self):
        """If transport delivery fails, the message stays in the channel's
        conversation history. The agent's intent to send is recorded
        regardless of transport success (same semantics as a normal turn
        whose reply fails to deliver)."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        # Make the send hook raise (the helper already mocks it as AsyncMock).
        plugin.pm.ahook.send.side_effect = RuntimeError("transport down")

        await plugin.enqueue_agent_send(channel.id, "undelivered message", trace_id=mint_trace_id())
        await drain(plugin, channel)

        # The message is in the conversation despite the transport failure.
        conv = channel.conversation
        assistant_msgs = [m for m in conv.messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "undelivered message"

        await db.close()

    async def test_empty_text_rejected(self):
        """enqueue_agent_send rejects empty or whitespace-only text."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        with pytest.raises(ValueError, match="must not be empty"):
            await plugin.enqueue_agent_send(channel.id, "", trace_id=mint_trace_id())

        with pytest.raises(ValueError, match="must not be empty"):
            await plugin.enqueue_agent_send(channel.id, "   ", trace_id=mint_trace_id())

        await db.close()

# ---------------------------------------------------------------------------
# Section 6 — second review pass: branch coverage
# ---------------------------------------------------------------------------


class TestAgentSendBranchCoverage:
    async def test_conversation_init_loads_existing_history(self):
        """When the target channel has no conversation but resolve_conversation
        returns prior history, that history is loaded before the agent send."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        assert channel.conversation is None
        # Make resolve_conversation return prior messages.
        plugin.pm.ahook.resolve_conversation = AsyncMock(return_value=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ])

        await plugin.enqueue_agent_send(channel.id, "new message", trace_id=mint_trace_id())
        await drain(plugin, channel)

        conv = channel.conversation
        assert conv is not None
        # Prior history loaded, then the agent send appended.
        assert conv.messages[0]["content"] == "old question"
        assert conv.messages[1]["content"] == "old answer"
        assert conv.messages[-1]["role"] == "assistant"
        assert conv.messages[-1]["content"] == "new message"

        await db.close()

    async def test_db_id_set_on_persist(self):
        """When persist_message returns a rowid, _db_id is set on the message."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        await plugin.enqueue_agent_send(channel.id, "tracked message", trace_id=mint_trace_id())
        await drain(plugin, channel)

        conv = channel.conversation
        last = conv.messages[-1]
        assert last["role"] == "assistant"
        assert "_db_id" in last
        assert last["_db_id"] is not None

        await db.close()

    async def test_persist_failure_still_delivers(self):
        """When persist_message raises, the message stays in the conversation
        and delivery still occurs."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        plugin.pm.ahook.persist_message = AsyncMock(
            side_effect=RuntimeError("db write failed")
        )

        await plugin.enqueue_agent_send(channel.id, "unpersisted message", trace_id=mint_trace_id())
        await drain(plugin, channel)

        conv = channel.conversation
        # Message is in the conversation.
        assert conv.messages[-1]["content"] == "unpersisted message"
        # No _db_id since persistence failed.
        assert "_db_id" not in conv.messages[-1]
        # Delivery still happened.
        sent_messages = sent(plugin.pm, OutboundMessage)
        assert len(sent_messages) == 1
        assert sent_messages[0][1].text == "unpersisted message"

        await db.close()

    async def test_serialization_order_explicit(self):
        """The serialization test, strengthened: assert the turn response
        precedes the agent send in the conversation."""
        plugin, channel, db = await build_plugin_and_channel()
        plugin._mark_started()

        mock_client = MagicMock()
        from llm_response_fixtures import _make_text_response
        mock_client.chat = AsyncMock(return_value=_make_text_response("turn response"))
        plugin._client = mock_client

        user_trace = mint_trace_id()
        send_trace = mint_trace_id()
        await plugin.on_message(channel=channel, sender="user", text="do work", trace_id=user_trace)
        await plugin.enqueue_agent_send(channel.id, "queued while working", trace_id=send_trace)
        await drain(plugin, channel)

        conv = channel.conversation
        assistant_msgs = [m for m in conv.messages if m["role"] == "assistant"]
        # The turn response comes before the agent send.
        assert assistant_msgs[-2]["content"] == "turn response"
        assert assistant_msgs[-1]["content"] == "queued while working"

        await db.close()
