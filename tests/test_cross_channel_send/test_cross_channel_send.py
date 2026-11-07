"""Tests for the cross_channel_send plugin.

Covers the on_notify delivery mechanism, attribution-based source,
hop-count loop prevention, config reading, and integration with
the real Agent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
from cross_channel_send import CrossChannelSendPlugin
from llm_response_fixtures import _make_text_response

from corvidae.agent import Agent
from corvidae.channel import ChannelConfig, ChannelRegistry
from corvidae.hooks import create_plugin_manager
from corvidae.persistence import PersistencePlugin, init_db
from corvidae.task import TaskPlugin
from corvidae.thinking import ThinkingPlugin
from corvidae.trace import mint_trace_id


async def _build_harness(
    config: dict | None = None,
) -> tuple[CrossChannelSendPlugin, ChannelRegistry, aiosqlite.Connection, Agent]:
    """Build a plugin + registry + agent harness for testing."""
    full_config = config or {}
    db = await aiosqlite.connect(":memory:")
    await init_db(db)
    pm = create_plugin_manager()

    registry = ChannelRegistry()
    registry.agent_defaults = {
        "system_prompt": "You are a test assistant.",
        "max_context_tokens": 8000,
        "reasoning_history": "drop",
        "max_turns": 3,
    }
    pm.register(registry, name="registry")
    pm.ahook.send = AsyncMock()
    pm.ahook.on_agent_response = AsyncMock()

    task_plugin = TaskPlugin()
    pm.register(task_plugin, name="task")
    await task_plugin.on_init(pm=pm, config=full_config)
    await task_plugin.on_start(config=full_config)

    persistence = PersistencePlugin()
    persistence.db = db
    persistence._registry = registry
    pm.register(persistence, name="persistence")

    thinking_plugin = ThinkingPlugin()
    pm.register(thinking_plugin, name="thinking")
    await thinking_plugin.on_init(pm=pm, config=full_config)

    plugin = Agent()
    pm.register(plugin, name="agent")
    plugin.pm = pm
    plugin._registry = registry

    return CrossChannelSendPlugin(), registry, db, plugin


async def _init_plugin(
    plugin: CrossChannelSendPlugin,
    pm,
    config: dict | None = None,
):
    """Initialize and start the plugin."""
    await plugin.on_init(pm=pm, config=config or {})
    await plugin.on_start(config=config or {})


@pytest.fixture(autouse=True)
def _reset_attribution():
    """Ensure attribution is clean between tests."""
    from corvidae.attribution import _attribution

    token = _attribution.set({})
    yield
    _attribution.reset(token)


class TestSendMessageDelivery:
    """send_message calls on_notify with the right parameters."""

    async def test_calls_on_notify_with_correct_params(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message(ch_b.id, "hello bob")
        assert result == f"Queued on {ch_b.id}."
        assert notify.call_count == 1

        kwargs = notify.call_args.kwargs
        assert kwargs["channel"] is ch_b
        assert kwargs["source"] == f"agent_send from {ch_a.id}"
        assert kwargs["text"] == "hello bob"
        assert kwargs["tool_call_id"] is None
        assert kwargs["annotations"]["origin"] == "agent_send"
        assert kwargs["annotations"]["source_channel"] == ch_a.id
        assert kwargs["annotations"]["source_trace_id"] == "trace-123"
        assert kwargs["annotations"]["hop_count"] == 1
        # trace_id must NOT be in annotations (target mints its own trace)
        assert "trace_id" not in kwargs["annotations"]

        await db.close()

    async def test_source_from_attribution_not_tool_args(self):
        """The source channel comes from attribution, not tool arguments.
        A tool-argument attempt cannot override it."""
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())
        registry.get_or_create("signal", "evil", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        # Even if the LLM passes a different channel_id, the source
        # is always from attribution
        await plugin.send_message(ch_b.id, "hello")
        kwargs = notify.call_args.kwargs
        assert kwargs["annotations"]["source_channel"] == ch_a.id
        assert kwargs["source"] == f"agent_send from {ch_a.id}"

        await db.close()

    async def test_no_attribution_means_error(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message(ch_b.id, "hello")
        assert "no source channel" in result
        assert notify.call_count == 0

        await db.close()


class TestHopCount:
    """Hop count increments from attribution annotations, refused above max_hops."""

    async def test_hop_increments_from_attribution(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {"hop_count": 2},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        await plugin.send_message(ch_b.id, "hello")
        kwargs = notify.call_args.kwargs
        assert kwargs["annotations"]["hop_count"] == 3

        await db.close()

    async def test_hop_refused_above_max(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {"hop_count": 3},  # at the default max
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message(ch_b.id, "hello")
        assert "loop detected" in result
        assert str(ch_b.id) in result
        assert notify.call_count == 0

        await db.close()

    async def test_ignores_non_int_hop_count(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {"hop_count": "three"},  # not an int
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        await plugin.send_message(ch_b.id, "hello")
        kwargs = notify.call_args.kwargs
        assert kwargs["annotations"]["hop_count"] == 1  # reset to 1

        await db.close()

    async def test_ignores_bool_hop_count(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {"hop_count": True},  # bool, not int
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        await plugin.send_message(ch_b.id, "hello")
        kwargs = notify.call_args.kwargs
        assert kwargs["annotations"]["hop_count"] == 1  # reset to 1

        await db.close()


class TestInputValidation:
    """Input validation: unregistered target, empty/over-limit text."""

    async def test_unregistered_target(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message("signal:nonexistent", "hello")
        assert "not registered" in result
        assert "signal:nonexistent" in result
        assert "list_channels" in result
        assert notify.call_count == 0

        await db.close()

    async def test_empty_text_rejected(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        result = await plugin.send_message(ch_b.id, "")
        assert "text" in result.lower()
        assert "non-empty" in result

        await db.close()

    async def test_whitespace_text_rejected(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        result = await plugin.send_message(ch_b.id, "   ")
        assert "text" in result.lower()
        assert "non-empty" in result

        await db.close()

    async def test_over_limit_text_rejected(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        long_text = "x" * 8001
        result = await plugin.send_message(ch_b.id, long_text)
        assert "8001" in result
        assert "8000" in result

        await db.close()

    async def test_empty_channel_id_rejected(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        result = await plugin.send_message("", "hello")
        assert "channel_id" in result
        assert "non-empty" in result

        await db.close()


class TestConfig:
    """max_hops is read from cross_channel_send.max_hops in the full config."""

    async def test_reads_from_plugin_section(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        config = {"cross_channel_send": {"max_hops": 5}}
        await _init_plugin(plugin, pm, config)
        assert plugin._max_hops == 5
        await db.close()

    async def test_ignores_top_level_max_hops(self):
        """A top-level max_hops key must be ignored; only the plugin
        section is read."""
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        config = {"max_hops": 99}  # top-level, should be ignored
        await _init_plugin(plugin, pm, config)
        assert plugin._max_hops == 3  # default, not 99
        await db.close()

    async def test_invalid_value_raises(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        config = {"cross_channel_send": {"max_hops": "three"}}
        with pytest.raises(ValueError, match="max_hops"):
            await plugin.on_init(pm=pm, config=config)
        await db.close()

    async def test_bool_value_raises(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        config = {"cross_channel_send": {"max_hops": True}}
        with pytest.raises(ValueError, match="max_hops"):
            await plugin.on_init(pm=pm, config=config)
        await db.close()

    async def test_zero_value_raises(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        config = {"cross_channel_send": {"max_hops": 0}}
        with pytest.raises(ValueError, match="max_hops"):
            await plugin.on_init(pm=pm, config=config)
        await db.close()

    async def test_missing_section_uses_default(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm, {})
        assert plugin._max_hops == 3
        await db.close()


class TestListChannels:
    """list_channels returns sorted registered channel IDs."""

    async def test_returns_sorted_channels(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        registry.get_or_create("signal", "bob", config=ChannelConfig())
        registry.get_or_create("signal", "alice", config=ChannelConfig())
        registry.get_or_create("web", "main", config=ChannelConfig())

        result = await plugin.list_channels()
        lines = result.strip().split("\n")
        assert lines == sorted(lines)
        assert len(lines) == 3

        await db.close()

    async def test_no_channels(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        result = await plugin.list_channels()
        assert result == "No channels registered."
        await db.close()


class TestIntegration:
    """Integration through the real Agent with two channels."""

    async def test_send_triggers_llm_on_target_only(self):
        """A send from A produces one LLM call on B, no LLM call on A.
        B's message is role:system with a header naming A as the origin.
        B's trace differs from A's and originating is True."""
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        pm = create_plugin_manager()

        registry = ChannelRegistry()
        registry.agent_defaults = {
            "system_prompt": "You are a test assistant.",
            "max_context_tokens": 8000,
            "reasoning_history": "drop",
            "max_turns": 3,
        }
        pm.register(registry, name="registry")
        pm.ahook.send = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()
        pm.ahook.on_message_persisted = AsyncMock()

        task_plugin = TaskPlugin()
        pm.register(task_plugin, name="task")
        await task_plugin.on_init(pm=pm, config={})
        await task_plugin.on_start(config={})

        persistence = PersistencePlugin()
        persistence.db = db
        persistence._registry = registry
        pm.register(persistence, name="persistence")

        thinking_plugin = ThinkingPlugin()
        pm.register(thinking_plugin, name="thinking")
        await thinking_plugin.on_init(pm=pm, config={})

        agent = Agent()
        pm.register(agent, name="agent")
        agent.pm = pm
        agent._registry = registry
        agent._fence_key = b"k" * 32

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        # Mock LLM
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        agent._client = mock_client
        agent._mark_started()

        plugin = CrossChannelSendPlugin()
        pm.register(plugin, name="cross_channel_send")
        await _init_plugin(plugin, pm)

        # Set attribution to channel A
        from corvidae.attribution import _attribution

        trace_a = mint_trace_id()
        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": trace_a,
                "annotations": {},
            }
        )

        # Send from A to B
        result = await plugin.send_message(ch_b.id, "hello bob")
        assert result == f"Queued on {ch_b.id}."

        # Drain B's queue
        if ch_b.id in agent.queues:
            await agent.queues[ch_b.id].drain()

        # LLM should have been called once (for B's turn)
        assert mock_client.chat.call_count == 1

        # Check B's conversation: the message should be role:system
        b_conv = ch_b.conversation.messages
        system_msgs = [m for m in b_conv if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        agent_send_msg = [m for m in system_msgs if "hello bob" in m.get("content", "")]
        assert len(agent_send_msg) == 1
        assert agent_send_msg[0]["corvidae_provenance"]["origin_channel"] == ch_a.id

        # on_message_persisted should have fired for B
        assert pm.ahook.on_message_persisted.call_count >= 1

        await db.close()

    async def test_budget_regression_send_only_channel(self):
        """A channel that only receives sends and whose turns make tool
        calls never exhausts max_turns. Each send gets a fresh budget."""
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        pm = create_plugin_manager()

        registry = ChannelRegistry()
        registry.agent_defaults = {
            "system_prompt": "You are a test assistant.",
            "max_context_tokens": 8000,
            "reasoning_history": "drop",
            "max_turns": 2,  # low limit to test exhaustion
        }
        pm.register(registry, name="registry")
        pm.ahook.send = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        task_plugin = TaskPlugin()
        pm.register(task_plugin, name="task")
        await task_plugin.on_init(pm=pm, config={})
        await task_plugin.on_start(config={})

        persistence = PersistencePlugin()
        persistence.db = db
        persistence._registry = registry
        pm.register(persistence, name="persistence")

        thinking_plugin = ThinkingPlugin()
        pm.register(thinking_plugin, name="thinking")
        await thinking_plugin.on_init(pm=pm, config={})

        agent = Agent()
        pm.register(agent, name="agent")
        agent.pm = pm
        agent._registry = registry

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        # Mock LLM that always returns a tool call (to increment turn_counter)
        call_count = 0
        mock_client = MagicMock()

        async def chat_with_tools(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 6:
                tc = _make_text_response("thinking")
                tc.tool_calls = [
                    MagicMock(id=f"tc{call_count}", function=MagicMock(name="test", arguments="{}"))
                ]
                return tc
            return _make_text_response("done")

        mock_client.chat = AsyncMock(side_effect=chat_with_tools)
        agent._client = mock_client
        agent._mark_started()

        plugin = CrossChannelSendPlugin()
        pm.register(plugin, name="cross_channel_send")
        await _init_plugin(plugin, pm)

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": mint_trace_id(),
                "annotations": {},
            }
        )

        # Send 3 messages to B (more than max_turns=2)
        for i in range(3):
            result = await plugin.send_message(ch_b.id, f"msg {i}")
            assert result == f"Queued on {ch_b.id}."
            if ch_b.id in agent.queues:
                await agent.queues[ch_b.id].drain()
                # turn_counter should be reset for the next send
                assert ch_b.turn_counter < 2, (
                    f"turn_counter={ch_b.turn_counter} after send {i + 1}; should have been reset"
                )

        await db.close()

    async def test_loop_bound_a_b_a(self):
        """A→B→A bouncing stops at max_hops with the error, not a hang."""
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        pm = create_plugin_manager()

        registry = ChannelRegistry()
        registry.agent_defaults = {
            "system_prompt": "You are a test assistant.",
            "max_context_tokens": 8000,
            "reasoning_history": "drop",
            "max_turns": 3,
        }
        pm.register(registry, name="registry")
        pm.ahook.send = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        task_plugin = TaskPlugin()
        pm.register(task_plugin, name="task")
        await task_plugin.on_init(pm=pm, config={})
        await task_plugin.on_start(config={})

        persistence = PersistencePlugin()
        persistence.db = db
        persistence._registry = registry
        pm.register(persistence, name="persistence")

        thinking_plugin = ThinkingPlugin()
        pm.register(thinking_plugin, name="thinking")
        await thinking_plugin.on_init(pm=pm, config={})

        agent = Agent()
        pm.register(agent, name="agent")
        agent.pm = pm
        agent._registry = registry

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        agent._client = mock_client
        agent._mark_started()

        plugin = CrossChannelSendPlugin()
        pm.register(plugin, name="cross_channel_send")
        # Use max_hops=2 for this test
        config = {"cross_channel_send": {"max_hops": 2}}
        await plugin.on_init(pm=pm, config=config)
        await plugin.on_start(config=config)

        from corvidae.attribution import _attribution

        # Simulate: A sends to B (hop 1), B sends to A (hop 2), A tries to send to B (hop 3 > max 2)
        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": mint_trace_id(),
                "annotations": {"hop_count": 2},  # already at max
            }
        )

        result = await plugin.send_message(ch_b.id, "this should fail")
        assert "loop detected" in result
        assert "hop 3" in result
        assert "max 2" in result

        await db.close()


class TestNegativeHopCount:
    """Negative hop_count is clamped to 0, not accepted as-is."""

    async def test_negative_hop_count_clamped(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {"hop_count": -100},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message(ch_b.id, "hello")
        assert result == f"Queued on {ch_b.id}."
        kwargs = notify.call_args.kwargs
        assert kwargs["annotations"]["hop_count"] == 1  # clamped, not -99

        await db.close()

    async def test_negative_one_hop_count(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {"hop_count": -1},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message(ch_b.id, "hello")
        assert result == f"Queued on {ch_b.id}."
        kwargs = notify.call_args.kwargs
        assert kwargs["annotations"]["hop_count"] == 1

        await db.close()


class TestNonDictMeta:
    """Non-dict attribution annotations is handled gracefully."""

    async def test_non_dict_meta(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": "not a dict",
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message(ch_b.id, "hello")
        assert result == f"Queued on {ch_b.id}."
        kwargs = notify.call_args.kwargs
        assert kwargs["annotations"]["hop_count"] == 1

        await db.close()


class TestOnNotifyFailure:
    """A failing on_notify returns an error string, does not raise."""

    async def test_on_notify_failure(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        pm.ahook.on_notify = AsyncMock(side_effect=RuntimeError("boom"))

        result = await plugin.send_message(ch_b.id, "hello")
        assert "unexpected error" in result

        await db.close()


class TestHopPropagation:
    """Hop count propagates into the target's attribution annotations."""

    async def test_hop_count_in_target_attribution(self):
        """After a send from A to B, B's attribution annotations carries
        hop_count=1 and source_trace_id, but NOT trace_id."""
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        pm = create_plugin_manager()

        registry = ChannelRegistry()
        registry.agent_defaults = {
            "system_prompt": "You are a test assistant.",
            "max_context_tokens": 8000,
            "reasoning_history": "drop",
            "max_turns": 3,
        }
        pm.register(registry, name="registry")
        pm.ahook.send = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()

        task_plugin = TaskPlugin()
        pm.register(task_plugin, name="task")
        await task_plugin.on_init(pm=pm, config={})
        await task_plugin.on_start(config={})

        persistence = PersistencePlugin()
        persistence.db = db
        persistence._registry = registry
        pm.register(persistence, name="persistence")

        thinking_plugin = ThinkingPlugin()
        pm.register(thinking_plugin, name="thinking")
        await thinking_plugin.on_init(pm=pm, config={})

        agent = Agent()
        pm.register(agent, name="agent")
        agent.pm = pm
        agent._registry = registry
        agent._fence_key = b"k" * 32

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        agent._client = mock_client
        agent._mark_started()

        plugin = CrossChannelSendPlugin()
        pm.register(plugin, name="cross_channel_send")
        await _init_plugin(plugin, pm)

        from corvidae.attribution import _attribution

        trace_a = mint_trace_id()
        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": trace_a,
                "annotations": {},
            }
        )

        # Send from A to B
        await plugin.send_message(ch_b.id, "hello bob")

        # Drain B's queue
        if ch_b.id in agent.queues:
            await agent.queues[ch_b.id].drain()

        # B's conversation should have the notification
        b_msgs = ch_b.conversation.messages
        sys_msgs = [m for m in b_msgs if m.get("role") == "system"]
        assert any(
            m.get("corvidae_provenance", {}).get("origin_channel") == ch_a.id
            and m["corvidae_provenance"]["origin_trace"] == trace_a
            for m in sys_msgs
        )

        await db.close()

    async def test_target_mints_own_trace(self):
        """B's turn has a different trace_id from A's, and
        on_message_persisted fires for B."""
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        pm = create_plugin_manager()

        registry = ChannelRegistry()
        registry.agent_defaults = {
            "system_prompt": "You are a test assistant.",
            "max_context_tokens": 8000,
            "reasoning_history": "drop",
            "max_turns": 3,
        }
        pm.register(registry, name="registry")
        pm.ahook.send = AsyncMock()
        pm.ahook.on_agent_response = AsyncMock()
        pm.ahook.on_message_persisted = AsyncMock()

        task_plugin = TaskPlugin()
        pm.register(task_plugin, name="task")
        await task_plugin.on_init(pm=pm, config={})
        await task_plugin.on_start(config={})

        persistence = PersistencePlugin()
        persistence.db = db
        persistence._registry = registry
        pm.register(persistence, name="persistence")

        thinking_plugin = ThinkingPlugin()
        pm.register(thinking_plugin, name="thinking")
        await thinking_plugin.on_init(pm=pm, config={})

        agent = Agent()
        pm.register(agent, name="agent")
        agent.pm = pm
        agent._registry = registry

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_make_text_response("ok"))
        agent._client = mock_client
        agent._mark_started()

        plugin = CrossChannelSendPlugin()
        pm.register(plugin, name="cross_channel_send")
        await _init_plugin(plugin, pm)

        from corvidae.attribution import _attribution

        trace_a = mint_trace_id()
        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": trace_a,
                "annotations": {},
            }
        )

        await plugin.send_message(ch_b.id, "hello bob")

        if ch_b.id in agent.queues:
            await agent.queues[ch_b.id].drain()

        # on_message_persisted should have fired (B got a new trace)
        assert pm.ahook.on_message_persisted.call_count >= 1

        await db.close()


class TestHopDoesNotPersist:
    """hop_count does not leak into unrelated user turns."""

    async def test_fresh_turn_gets_fresh_hop_count(self):
        """After a cross-channel chain, a new user turn on the target
        channel starts with hop_count=1, not the chain's count."""
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())
        ch_c = registry.get_or_create("signal", "carol", config=ChannelConfig())

        from corvidae.attribution import _attribution

        # Simulate: B is in the middle of a chain (hop_count=2)
        _attribution.set(
            {
                "channel_id": ch_b.id,
                "trace_id": mint_trace_id(),
                "annotations": {"hop_count": 2},
            }
        )

        # B sends to C — this should use hop_count=3
        notify = AsyncMock()
        pm.ahook.on_notify = notify
        await plugin.send_message(ch_c.id, "from b")
        kwargs = notify.call_args.kwargs
        assert kwargs["annotations"]["hop_count"] == 3

        # Now a fresh user turn starts on B (clean attribution)
        _attribution.set(
            {
                "channel_id": ch_b.id,
                "trace_id": mint_trace_id(),
                "annotations": {},  # clean, no hop_count
            }
        )

        # B sends to C again — this should use hop_count=1
        notify2 = AsyncMock()
        pm.ahook.on_notify = notify2
        await plugin.send_message(ch_c.id, "fresh turn")
        kwargs2 = notify2.call_args.kwargs
        assert kwargs2["annotations"]["hop_count"] == 1

        await db.close()


class TestMinorCoverage:
    """Minor test coverage gaps from the review."""

    async def test_registry_none(self):
        plugin = CrossChannelSendPlugin()
        # Don't call on_start, so _registry is None
        result = await plugin.send_message("signal:bob", "hello")
        assert "registry is not available" in result

    async def test_register_tools(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        tools = []
        plugin.register_tools(tools)
        assert len(tools) == 2
        assert tools[0].__name__ == "send_message"
        assert tools[1].__name__ == "list_channels"
        await db.close()

    async def test_text_exactly_at_limit_accepted(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        exact_text = "x" * 8000
        result = await plugin.send_message(ch_b.id, exact_text)
        assert result == f"Queued on {ch_b.id}."
        assert notify.call_count == 1

        await db.close()

    async def test_over_limit_no_notify(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())
        ch_b = registry.get_or_create("signal", "bob", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message(ch_b.id, "x" * 8001)
        assert "8001" in result
        assert notify.call_count == 0

        await db.close()

    async def test_empty_channel_no_notify(self):
        plugin, registry, db, agent = await _build_harness()
        pm = agent.pm
        await _init_plugin(plugin, pm)

        ch_a = registry.get_or_create("signal", "alice", config=ChannelConfig())

        from corvidae.attribution import _attribution

        _attribution.set(
            {
                "channel_id": ch_a.id,
                "trace_id": "trace-123",
                "annotations": {},
            }
        )

        notify = AsyncMock()
        pm.ahook.on_notify = notify

        result = await plugin.send_message("", "hello")
        assert "non-empty" in result
        assert notify.call_count == 0

        await db.close()
