"""Tests for corvidae.context_compact.ContextCompactPlugin.

Tests background block generation, injection, token tracking, and the stats tool.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.context import ContextWindow, MessageType


@pytest.fixture
def plugin():
    """Create a ContextCompactPlugin with default config."""
    from corvidae.context_compact import ContextCompactPlugin
    p = ContextCompactPlugin(None)
    # Set very low thresholds for testing.
    p._enabled = True
    p._bg_block_threshold = 5
    p._bg_compaction_threshold = 0.75
    p._min_background_blocks = 1
    p._max_bg_block_chars = 2048
    p._chars_per_token = 3.5
    return p


def _make_mock_get_dependency(mock_client):
    """Return a mock LLMPlugin with main_client set to mock_client."""
    from corvidae.llm_plugin import LLMPlugin
    mock_llm = MagicMock(spec=LLMPlugin)
    mock_llm.main_client = mock_client

    def _get_dependency(pm, name, expected_type):
        return mock_llm

    return _get_dependency


class TestContextCompactPlugin:
    """Tests for ContextCompactPlugin."""

    # -- Background block generation --

    async def test_no_block_below_threshold(self, plugin):
        """No background block generated when turn count is below threshold."""
        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        # Few messages — won't meet bg_block_threshold of 5.
        conv.messages = [{"role": "user", "content": f"message {i}"} for i in range(3)]

        mock_client = AsyncMock()
        with patch("corvidae.context_compact.get_dependency", _make_mock_get_dependency(mock_client)):
            result = await plugin.compact_conversation(conversation=conv, max_tokens=10000)

        assert result is None
        mock_client.chat.assert_not_called()
        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) == 0

    async def test_block_generated_above_threshold(self, plugin):
        """Background block generated when message count exceeds threshold."""
        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        conv.messages = [{"role": "user", "content": f"important context about project X"} for _ in range(8)]

        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "choices": [{"message": {"content": "Project X: ongoing development."}}]
        }

        with patch("corvidae.context_compact.get_dependency", _make_mock_get_dependency(mock_client)):
            result = await plugin.compact_conversation(conversation=conv, max_tokens=10000)

        assert result is None  # Side effects only
        mock_client.chat.assert_called_once()
        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) == 1
        assert "[Background Context]" in ctx_msgs[0]["content"]

    async def test_block_truncated_to_max_chars(self, plugin):
        """Background block is truncated when it exceeds max_background_block_chars."""
        plugin._max_bg_block_chars = 50

        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        conv.messages = [{"role": "user", "content": f"x" * 200} for _ in range(8)]

        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "choices": [{"message": {"content": "a" * 300}}]
        }

        with patch("corvidae.context_compact.get_dependency", _make_mock_get_dependency(mock_client)):
            await plugin.compact_conversation(conversation=conv, max_tokens=10000)

        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) == 1
        content = ctx_msgs[0]["content"]
        assert "[Background Context]" in content
        block_part = content.replace("[Background Context]\n", "")
        assert len(block_part) == 50  # truncated to max_bg_block_chars

    async def test_block_generation_failure_logged(self, plugin):
        """Background block generation failure is logged, not raised."""
        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        conv.messages = [{"role": "user", "content": f"msg {idx}"} for idx in range(8)]

        mock_client = AsyncMock()
        mock_client.chat.side_effect = RuntimeError("LLM API error")

        with patch("corvidae.context_compact.get_dependency", _make_mock_get_dependency(mock_client)):
            result = await plugin.compact_conversation(conversation=conv, max_tokens=10000)

        assert result is None
        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) == 0

    async def test_no_duplicate_blocks(self, plugin):
        """Subsequent compaction does not create duplicate blocks."""
        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        conv.messages = [{"role": "user", "content": f"msg {_i}"} for _i in range(8)]

        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "choices": [{"message": {"content": "block 1"}}]
        }

        with patch("corvidae.context_compact.get_dependency", _make_mock_get_dependency(mock_client)):
            # First call generates a block.
            await plugin.compact_conversation(conversation=conv, max_tokens=10000)
            ctx_msgs_1 = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
            assert len(ctx_msgs_1) == 1

            # Second call — last_block_ts is now set, so older messages are excluded.
            mock_client.chat.reset_mock()
            await plugin.compact_conversation(conversation=conv, max_tokens=10000)

        ctx_msgs_2 = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs_2) == 1
        mock_client.chat.assert_not_called()

    async def test_disabled_plugin_noop(self, plugin):
        """Disabled plugin does nothing."""
        plugin._enabled = False

        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        conv.messages = [{"role": "user", "content": f"msg {idx}"} for idx in range(8)]

        mock_client = AsyncMock()
        with patch("corvidae.context_compact.get_dependency", _make_mock_get_dependency(mock_client)):
            result = await plugin.compact_conversation(conversation=conv, max_tokens=10000)

        assert result is None
        mock_client.chat.assert_not_called()

    # -- before_agent_turn injection --

    async def test_injects_background_block(self, plugin):
        """before_agent_turn injects the most recent background block."""
        conv = ContextWindow("chan1")
        conv.system_prompt = ""

        # Simulate a block that was generated earlier.
        conv.append(
            {"role": "user", "content": "[Background Context]\nProject X is underway."},
            message_type=MessageType.CONTEXT,
        )
        plugin._last_block_ts["chan1"] = 1000.0

        mock_channel = MagicMock()
        mock_channel.id = "chan1"
        mock_channel.conversation = conv

        await plugin.before_agent_turn(mock_channel)

        # Should have cleaned old entries and re-injected.
        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) == 1
        assert "[Background Context]" in ctx_msgs[0]["content"]

    async def test_no_injection_without_block(self, plugin):
        """No injection when no background block exists."""
        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        assert "chan1" not in plugin._last_block_ts

        mock_channel = MagicMock()
        mock_channel.id = "chan1"
        mock_channel.conversation = conv

        await plugin.before_agent_turn(mock_channel)

        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) == 0

    async def test_no_injection_when_disabled(self, plugin):
        """No injection when plugin is disabled."""
        plugin._enabled = False
        plugin._last_block_ts["chan1"] = 1000.0

        conv = ContextWindow("chan1")
        conv.system_prompt = ""

        mock_channel = MagicMock()
        mock_channel.id = "chan1"
        mock_channel.conversation = conv

        await plugin.before_agent_turn(mock_channel)

        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) == 0

    # -- Turn tracking --

    async def test_tracked_turns_increment(self, plugin):
        """Turn counter increments on each agent response."""
        mock_channel = MagicMock()
        mock_channel.id = "chan1"

        await plugin.on_agent_response(mock_channel, "request", "response")
        assert plugin._turn_counter["chan1"] == 1

        await plugin.on_agent_response(mock_channel, "request2", "response2")
        assert plugin._turn_counter["chan1"] == 2

    # -- Tool registration --

    async def test_stats_tool_returns_info(self, plugin):
        """The context_stats tool returns stats when channels are tracked."""
        plugin._turn_counter = {"chan1": 5, "chan2": 3}
        plugin._last_block_ts = {"chan1": 1000.0, "chan2": 0}

        registry = []
        plugin.register_tools(registry)

        assert len(registry) == 1
        tool = registry[0]
        assert tool.name == "context_stats"
        result = await tool.fn()
        assert "chan1: 5 turns" in result
        assert "chan2: 3 turns" in result

    async def test_stats_tool_disabled(self, plugin):
        """Stats tool reports disabled when plugin is off."""
        plugin._enabled = False
        plugin._turn_counter = {"chan1": 5}

        registry = []
        plugin.register_tools(registry)

        tool = registry[0]
        result = await tool.fn()
        assert "disabled" in result.lower()

    # -- Config loading --

    async def test_config_defaults(self):
        """Plugin loads sensible defaults when no config provided."""
        from corvidae.context_compact import ContextCompactPlugin
        p = ContextCompactPlugin(None)
        await p.on_start({"agent": {}})

        assert p._enabled is True
        assert p._bg_block_threshold == 20
        assert p._bg_compaction_threshold == 0.75
        assert p._min_background_blocks == 1
        assert p._max_bg_block_chars == 2048

    async def test_config_custom_values(self):
        """Plugin respects custom config values."""
        from corvidae.context_compact import ContextCompactPlugin
        p = ContextCompactPlugin(None)
        await p.on_start({
            "agent": {
                "context_compact": {
                    "enabled": False,
                    "bg_block_threshold": 10,
                    "bg_compaction_threshold": 0.9,
                    "min_background_blocks": 2,
                    "max_background_block_chars": 4096,
                }
            }
        })

        assert p._enabled is False
        assert p._bg_block_threshold == 10
        assert p._bg_compaction_threshold == 0.9
        assert p._min_background_blocks == 2
        assert p._max_bg_block_chars == 4096


class TestContextCompactIntegration:
    """Integration tests for ContextCompactPlugin with full message flow."""

    async def test_full_flow_generate_and_inject(self):
        """End-to-end: generate a block, then inject it on next turn."""
        from corvidae.context_compact import ContextCompactPlugin

        plugin = ContextCompactPlugin(None)
        plugin._enabled = True
        plugin._bg_block_threshold = 5
        plugin._chars_per_token = 3.5

        conv = ContextWindow("chan1")
        conv.system_prompt = "You are a helpful assistant."
        conv.messages = [{"role": "user", "content": f"Let's discuss topic {i} for a while"} for i in range(8)]

        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "choices": [{"message": {"content": "Topics 0-7 discussed. User wants to continue."}}]
        }

        # Step 1: Generate background block via compact_conversation.
        with patch("corvidae.context_compact.get_dependency", _make_mock_get_dependency(mock_client)):
            await plugin.compact_conversation(conversation=conv, max_tokens=10000)

        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) == 1

        # Step 2: Simulate agent response to track turn count.
        mock_channel = MagicMock()
        mock_channel.id = "chan1"
        mock_channel.conversation = conv

        await plugin.on_agent_response(mock_channel, "request", "response")
        assert plugin._turn_counter["chan1"] == 1

        # Step 3: Simulate next agent turn — block should be re-injected.
        await plugin.before_agent_turn(mock_channel)

        ctx_after = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_after) == 1
        assert "[Background Context]" in ctx_after[0]["content"]

    async def test_block_survives_compaction(self):
        """Background block persists through basic compaction."""
        from corvidae.compaction import CompactionPlugin
        from corvidae.context_compact import ContextCompactPlugin

        cc_plugin = ContextCompactPlugin(None)
        cc_plugin._enabled = True
        cc_plugin._bg_block_threshold = 5
        cc_plugin._chars_per_token = 3.5

        comp_plugin = CompactionPlugin(None)
        comp_plugin._compaction_threshold = 0.8
        comp_plugin._compaction_retention = 0.5
        comp_plugin._min_messages = 3
        comp_plugin._chars_per_token = 3.5

        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        long_msg = "a" * 100
        conv.messages = [{"role": "user", "content": long_msg} for _ in range(20)]

        cc_mock_client = AsyncMock()
        cc_mock_client.chat.return_value = {
            "choices": [{"message": {"content": "Long discussion summarized."}}]
        }

        comp_mock_client = AsyncMock()
        comp_mock_client.chat.return_value = {
            "choices": [{"message": {"content": "Compacted summary."}}]
        }
        comp_plugin._llm_client = comp_mock_client

        mock_channel = MagicMock()
        mock_channel.id = "chan1"
        mock_channel.conversation = conv

        # Generate background block first.
        with patch("corvidae.context_compact.get_dependency", _make_mock_get_dependency(cc_mock_client)):
            await cc_plugin.compact_conversation(conversation=conv, max_tokens=1000)

        # Then trigger compaction.
        await comp_plugin.compact_conversation(
            channel=mock_channel, conversation=conv, max_tokens=1000
        )

        # The background block should still be present (as a CONTEXT entry).
        ctx_msgs = [m for m in conv.messages if m.get("_message_type") == MessageType.CONTEXT]
        assert len(ctx_msgs) >= 1
        assert "[Background Context]" in ctx_msgs[0]["content"]
