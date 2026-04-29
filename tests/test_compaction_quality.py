"""Tests for compaction guardrails: cooldown, empty summary rejection, configurable prompts."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

from corvidae.compaction import CompactionPlugin
from corvidae.context import ContextWindow, MessageType


def _make_channel(channel_id: str = "test:scope1") -> object:
    """Build a minimal mock channel for compaction tests."""
    channel = MagicMock()
    channel.id = channel_id
    return channel


def _make_conversation(channel_id: str = "chan1", msg_count: int = 10, chars: int = 35):
    """Build a conversation with msg_count messages of chars characters each."""
    conv = ContextWindow(channel_id)
    conv.system_prompt = ""
    conv.messages = [
        {"role": "user", "content": "a" * chars, "_message_type": MessageType.MESSAGE}
        for _ in range(msg_count)
    ]
    return conv


# ---------------------------------------------------------------------------
# Failed compaction cooldown
# ---------------------------------------------------------------------------


class TestFailedCompactionCooldown:
    """Tests that failed compaction triggers a cooldown period."""

    async def test_failed_compaction_sets_cooldown(self):
        """After a summarization failure, compaction is skipped during cooldown."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        # First call: simulate a summarization failure.
        conv1 = _make_conversation(msg_count=10)
        with patch.object(plugin, "_summarize", side_effect=RuntimeError("LLM down")):
            try:
                await plugin.compact_conversation(
                    channel=channel, conversation=conv1, max_tokens=50
                )
            except RuntimeError:
                pass

        # Cooldown should be set.
        assert "chan1" in plugin._last_failed_compaction

        # Second call with fresh conversation: should be skipped due to cooldown.
        conv2 = _make_conversation(msg_count=10)
        result = await plugin.compact_conversation(
            channel=channel, conversation=conv2, max_tokens=50
        )
        assert result is None

    async def test_cooldown_clears_on_success(self):
        """After a successful compaction, the failure cooldown is cleared."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        # Set a fake failure cooldown that has expired so compaction can proceed.
        plugin._last_failed_compaction["chan1"] = time.monotonic() - 60

        conv = _make_conversation(msg_count=10)
        with patch.object(plugin, "_summarize", return_value="A good summary"):
            result = await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert result is True
        assert "chan1" not in plugin._last_failed_compaction

    async def test_cooldown_expires_after_time(self):
        """Compaction retries after the cooldown period expires."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        # Set a failure cooldown that has already expired.
        plugin._last_failed_compaction["chan1"] = time.monotonic() - 60

        conv = _make_conversation(msg_count=10)
        with patch.object(plugin, "_summarize", return_value="A good summary"):
            result = await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert result is True


# ---------------------------------------------------------------------------
# Minimum messages between compactions
# ---------------------------------------------------------------------------


class TestMinimumMessagesBetweenCompactions:
    """Tests for the minimum-messages-between-compactions guard."""

    async def test_compaction_skipped_when_too_few_new_messages(self):
        """Compaction is skipped if fewer than 6 messages since last compaction."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        # First compaction succeeds.
        conv1 = _make_conversation(msg_count=10)
        with patch.object(plugin, "_summarize", return_value="Summary 1"):
            result1 = await plugin.compact_conversation(
                channel=channel, conversation=conv1, max_tokens=50
            )
        assert result1 is True

        # Simulate only 3 new messages (below the minimum of 6).
        conv2 = _make_conversation(msg_count=5)
        plugin._last_compaction_msg_count["chan1"] = 3

        result2 = await plugin.compact_conversation(
            channel=channel, conversation=conv2, max_tokens=50
        )
        assert result2 is None  # skipped

    async def test_compaction_proceeds_when_enough_new_messages(self):
        """Compaction proceeds when enough messages have accumulated since last one."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        # Record a previous compaction at 2 messages.
        plugin._last_compaction_msg_count["chan1"] = 2

        # Now we have 10 messages — that's 8 new, which exceeds the minimum of 6.
        conv = _make_conversation(msg_count=10)
        with patch.object(plugin, "_summarize", return_value="Summary"):
            result = await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )
        assert result is True


# ---------------------------------------------------------------------------
# Empty summary rejection
# ---------------------------------------------------------------------------


class TestEmptySummaryRejection:
    """Tests that empty or blank summaries are rejected."""

    async def test_empty_summary_rejected(self):
        """Compaction is aborted when the LLM returns an empty string."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()
        conv = _make_conversation(msg_count=10)

        with patch.object(plugin, "_summarize", return_value=""):
            result = await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert result is None
        # Original messages should still be intact (not replaced).
        assert len(conv.messages) == 10

    async def test_whitespace_only_summary_rejected(self):
        """Compaction is aborted when the LLM returns only whitespace."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()
        conv = _make_conversation(msg_count=10)

        with patch.object(plugin, "_summarize", return_value="   \n\t  "):
            result = await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert result is None
        assert len(conv.messages) == 10

    async def test_empty_summary_triggers_failure_cooldown(self):
        """An empty summary sets the failure cooldown to prevent rapid re-attempts."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()
        conv = _make_conversation(msg_count=10)

        with patch.object(plugin, "_summarize", return_value=""):
            await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert "chan1" in plugin._last_failed_compaction

    async def test_substantive_summary_accepted(self):
        """A non-empty summary is accepted normally."""
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()
        conv = _make_conversation(msg_count=10)

        with patch.object(plugin, "_summarize", return_value="This is a good summary"):
            result = await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert result is True


# ---------------------------------------------------------------------------
# Configurable prompts
# ---------------------------------------------------------------------------


class TestConfigurablePrompts:
    """Tests that summary prompts are configurable via agent.yaml."""

    async def test_default_prompt_used_when_not_configured(self):
        """The default summary prompt is used when no config override exists."""
        plugin = CompactionPlugin(pm=None)
        await plugin.on_start(config={})

        assert plugin._summary_prompt == CompactionPlugin.DEFAULT_SUMMARY_PROMPT

    async def test_custom_prompt_from_config(self):
        """A custom summary prompt is loaded from config."""
        custom = "Custom prompt: be very detailed."
        plugin = CompactionPlugin(pm=None)
        await plugin.on_start(config={"agent": {"compaction_prompt": custom}})

        assert plugin._summary_prompt == custom

    async def test_custom_prompt_used_in_summarize(self):
        """The configured prompt is passed to the LLM in _summarize."""
        custom = "Custom prompt: preserve debugging state."
        plugin = CompactionPlugin(pm=None)
        plugin._summary_prompt = custom

        mock_client = AsyncMock()
        captured = {}

        async def fake_chat(payload):
            captured["system"] = payload[0]["content"]
            return {"choices": [{"message": {"content": "summary"}}]}

        mock_client.chat = fake_chat
        plugin._llm_client = mock_client

        await plugin._summarize([{"role": "user", "content": "hello"}])

        assert captured["system"] == custom

    async def test_bg_block_prompt_default(self):
        """ContextCompactPlugin has a default background block prompt."""
        from corvidae.context_compact import ContextCompactPlugin
        plugin = ContextCompactPlugin(pm=None)

        assert plugin._bg_block_prompt == ContextCompactPlugin.DEFAULT_BG_BLOCK_PROMPT

    async def test_bg_block_prompt_configurable(self):
        """ContextCompactPlugin prompt can be overridden via config."""
        from corvidae.context_compact import ContextCompactPlugin
        custom = "Custom bg block prompt."
        plugin = ContextCompactPlugin(pm=None)
        await plugin.on_start(config={
            "agent": {"context_compact": {"bg_block_prompt": custom}}
        })

        assert plugin._bg_block_prompt == custom

    async def test_default_prompt_is_not_empty(self):
        """Sanity check: the default summary prompt is substantive."""
        assert len(CompactionPlugin.DEFAULT_SUMMARY_PROMPT) > 50

    async def test_bg_block_default_prompt_is_not_empty(self):
        """Sanity check: the default bg block prompt is substantive."""
        from corvidae.context_compact import ContextCompactPlugin
        assert len(ContextCompactPlugin.DEFAULT_BG_BLOCK_PROMPT) > 50


class TestDefaultPromptDesign:
    """Tests that the default prompts follow the Schuyler-style design principles."""

    def test_default_prompt_mentions_flow_into_retained(self):
        """Default prompt should instruct the LLM that its summary flows into retained context."""
        prompt = CompactionPlugin.DEFAULT_SUMMARY_PROMPT.lower()
        assert "second half" in prompt or "flows naturally" in prompt or "verbatim" in prompt

    def test_default_prompt_has_word_limit(self):
        """Default prompt should include a word/length limit."""
        prompt = CompactionPlugin.DEFAULT_SUMMARY_PROMPT.lower()
        assert "words" in prompt or "characters" in prompt or "limit" in prompt

    def test_default_prompt_mentions_specific_details(self):
        """Default prompt should instruct preservation of specific details."""
        prompt = CompactionPlugin.DEFAULT_SUMMARY_PROMPT.lower()
        assert "file path" in prompt or "variable" in prompt or "error" in prompt

    def test_bg_block_prompt_has_word_limit(self):
        """Background block prompt should include a word/length limit."""
        from corvidae.context_compact import ContextCompactPlugin
        prompt = ContextCompactPlugin.DEFAULT_BG_BLOCK_PROMPT.lower()
        assert "words" in prompt or "characters" in prompt or "limit" in prompt
