"""Tests for corvidae.compaction.CompactionPlugin."""

from unittest.mock import AsyncMock, MagicMock, patch

from corvidae.context import ContextWindow, MessageType


def _make_channel(channel_id: str = "test:scope1") -> object:
    """Build a minimal mock channel for compaction tests."""
    channel = MagicMock()
    channel.id = channel_id
    return channel


class TestCompactFiltersNonMessage:
    """Tests for compaction filtering behavior from TestMessageType."""

    async def test_compact_filters_non_message_from_older(self):
        """compact_conversation must exclude SUMMARY-typed entries from the older
        portion passed to the summarizer.

        Setup: 10 messages that trigger compaction, with the first message
        (index 0, which ends up in 'older') typed as SUMMARY. The summarizer
        must only receive MESSAGE-typed entries, so the SUMMARY-typed dict
        must not appear in the messages list sent to the LLM.
        """
        conv = ContextWindow("chan1")
        conv.system_prompt = ""

        # Build 10 messages. The first one is typed SUMMARY to simulate a
        # pre-existing summary entry in the older portion.
        # Token math: 10 msgs x 35 chars → int(350/3.5)=100 ≥ 40 → triggers; len=10>5.
        # retain_count=2 → older = messages[0:8], retained = messages[-2:].
        # The SUMMARY-typed message at index 0 falls in older and must be filtered out.
        conv.messages = [
            {"role": "assistant", "content": "a" * 35, "_message_type": MessageType.SUMMARY},
        ] + [
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}
            for _ in range(9)
        ]

        captured_older = []

        async def capture_summarize(client, messages):
            captured_older.extend(messages)
            return "filtered summary"

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "filtered summary"}}]}
        )

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        # Patch the plugin's _summarize to capture what older list is passed in
        with patch.object(plugin, "_summarize", side_effect=capture_summarize):
            await plugin.compact_conversation(
                channel=channel, conversation=conv, client=mock_client, max_tokens=50
            )

        # The SUMMARY-typed dict (index 0) must not appear in older passed to summarizer
        for msg in captured_older:
            assert msg.get("_message_type") != MessageType.SUMMARY, (
                f"SUMMARY-typed message must be filtered from older before summarization: {msg!r}"
            )
            # _message_type must also be stripped before LLM serialization
            assert "_message_type" not in msg, (
                f"_message_type key must be stripped before passing to _summarize: {msg!r}"
            )


# ---------------------------------------------------------------------------
# TestOnCompactionHookFired
# ---------------------------------------------------------------------------


class TestOnCompactionHookFired:
    """Tests that CompactionPlugin fires the on_compaction hook after replace_with_summary."""

    async def test_on_compaction_hook_fired_after_compaction(self):
        """compact_conversation must fire pm.ahook.on_compaction after replace_with_summary."""
        from corvidae.compaction import CompactionPlugin

        mock_pm = MagicMock()
        mock_pm.ahook = MagicMock()
        mock_pm.ahook.on_compaction = AsyncMock()

        plugin = CompactionPlugin(pm=mock_pm)
        channel = _make_channel()

        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "a" * 35}
            for _ in range(10)
        ]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "hook test summary"}}]}
        )

        await plugin.compact_conversation(
            channel=channel, conversation=conv, client=mock_client, max_tokens=50
        )

        mock_pm.ahook.on_compaction.assert_awaited_once()
        call_kwargs = mock_pm.ahook.on_compaction.call_args.kwargs
        assert call_kwargs["channel"] is channel
        assert "summary_msg" in call_kwargs
        assert "retain_count" in call_kwargs

    async def test_on_compaction_not_fired_when_no_compaction(self):
        """compact_conversation must NOT fire on_compaction when below threshold."""
        from corvidae.compaction import CompactionPlugin

        mock_pm = MagicMock()
        mock_pm.ahook = MagicMock()
        mock_pm.ahook.on_compaction = AsyncMock()

        plugin = CompactionPlugin(pm=mock_pm)
        channel = _make_channel()

        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        # Small messages — will not trigger compaction
        conv.messages = [{"role": "user", "content": "hi"} for _ in range(5)]

        mock_client = AsyncMock()

        await plugin.compact_conversation(
            channel=channel, conversation=conv, client=mock_client, max_tokens=10000
        )

        mock_pm.ahook.on_compaction.assert_not_awaited()

    async def test_compaction_plugin_init_accepts_pm_none(self):
        """CompactionPlugin(pm=None) must work without errors."""
        from corvidae.compaction import CompactionPlugin
        # Must not raise
        plugin = CompactionPlugin(pm=None)
        assert plugin.pm is None

    async def test_compaction_plugin_stores_pm(self):
        """CompactionPlugin(pm=mock_pm) must store pm as self.pm."""
        from corvidae.compaction import CompactionPlugin

        mock_pm = MagicMock()
        plugin = CompactionPlugin(pm=mock_pm)
        assert plugin.pm is mock_pm
