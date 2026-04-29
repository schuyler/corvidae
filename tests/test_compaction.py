"""Tests for corvidae.compaction.CompactionPlugin."""

from unittest.mock import AsyncMock, MagicMock, patch

from corvidae.context import ContextWindow, MessageType


def _make_channel(channel_id: str = "test:scope1") -> object:
    """Build a minimal mock channel for compaction tests."""
    channel = MagicMock()
    channel.id = channel_id
    return channel


class TestCompactCarriesForwardSummaries:
    """Tests for compaction handling of prior summaries in the older portion."""

    async def test_compact_carries_forward_prior_summaries(self):
        """compact_conversation must carry forward existing SUMMARY entries
        from the older portion into the new summary.

        This prevents the "death spiral" where repeated compaction of
        tool-only messages loses all accumulated user context.

        Setup: 10 messages that trigger compaction, with the first message
        (index 0, which ends up in 'older') typed as SUMMARY. The summarizer
        must receive the SUMMARY content as prior context.
        """
        conv = ContextWindow("chan1")
        conv.system_prompt = ""

        # Build 10 messages. The first one is typed SUMMARY to simulate a
        # pre-existing summary entry in the older portion.
        # Token math: 10 msgs x 35 chars → int(350/3.5)=100 ≥ 40 → triggers; len=10>5.
        # retain_count=2 → older = messages[0:8], retained = messages[-2:].
        # The SUMMARY-typed message at index 0 falls in older and must be carried forward.
        conv.messages = [
            {"role": "assistant", "content": "a" * 35, "_message_type": MessageType.SUMMARY},
        ] + [
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}
            for _ in range(9)
        ]

        captured_new = []
        captured_prior = []

        async def capture_summarize(messages, prior_summaries=None):
            captured_new.extend(messages)
            if prior_summaries:
                captured_prior.extend(prior_summaries)
            return "carried forward summary"

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        with patch.object(plugin, "_summarize", side_effect=capture_summarize):
            await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        # The SUMMARY-typed dict must be passed as prior_summaries
        assert len(captured_prior) == 1, (
            f"Expected 1 prior summary, got {len(captured_prior)}"
        )
        # Prior summary content should be preserved
        assert captured_prior[0]["content"] == "a" * 35
        # _message_type must be stripped from both new messages and prior summaries
        for msg in captured_new + captured_prior:
            assert "_message_type" not in msg, (
                f"_message_type key must be stripped: {msg!r}"
            )

    async def test_compact_filters_non_message_non_summary_from_older(self):
        """compact_conversation must exclude CONTEXT-typed entries from both
        the new messages and prior summaries passed to the summarizer.
        """
        conv = ContextWindow("chan1")
        conv.system_prompt = ""

        conv.messages = [
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.CONTEXT},
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE},
        ] + [
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}
            for _ in range(8)
        ]

        captured_new = []
        captured_prior = []

        async def capture_summarize(messages, prior_summaries=None):
            captured_new.extend(messages)
            if prior_summaries:
                captured_prior.extend(prior_summaries)
            return "filtered summary"

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        with patch.object(plugin, "_summarize", side_effect=capture_summarize):
            await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        # CONTEXT-typed messages should be filtered out
        for msg in captured_new:
            assert msg["content"] == "a" * 35  # all should be MESSAGE-type content
        # No prior summaries (the CONTEXT entry shouldn't appear as one)
        assert len(captured_prior) == 0


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
        plugin._llm_client = mock_client

        await plugin.compact_conversation(
            channel=channel, conversation=conv, max_tokens=50
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

        await plugin.compact_conversation(
            channel=channel, conversation=conv, max_tokens=10000
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


# ---------------------------------------------------------------------------
# TestCompactionPluginPart3 — Part 3 red-phase tests
# ---------------------------------------------------------------------------


class TestCompactionPluginPart3:
    """RED-phase tests for Part 3 of the agent-decomposition refactor.

    These tests verify the new interface for CompactionPlugin after
    LLMPlugin extraction. They fail until Part 3 is implemented.

    See plans/agent-decomposition-parts-3-4.md §Part 3.
    """

    def test_compaction_plugin_has_depends_on(self):
        """CompactionPlugin must have a 'depends_on' class attribute.

        Currently CompactionPlugin has no depends_on attribute at all.
        This test fails until the attribute is added.
        """
        from corvidae.compaction import CompactionPlugin

        assert hasattr(CompactionPlugin, "depends_on"), (
            "CompactionPlugin must have a 'depends_on' class attribute — "
            "see plans/agent-decomposition-parts-3-4.md §Part 3"
        )

    def test_compaction_plugin_depends_on_includes_llm(self):
        """CompactionPlugin.depends_on must include 'llm'.

        Currently CompactionPlugin has no depends_on at all.
        This test fails until depends_on = {'llm'} is added.
        """
        from corvidae.compaction import CompactionPlugin

        assert "llm" in CompactionPlugin.depends_on, (
            "CompactionPlugin.depends_on must include 'llm' — "
            "see plans/agent-decomposition-parts-3-4.md §Part 3"
        )

class TestDeathSpiralPrevention:
    """Tests that repeated compaction carries forward accumulated context.

    The "death spiral" occurs when:
    1. First compaction produces a good summary with user instructions.
    2. Second compaction only has tool outputs to summarize (no user messages).
    3. The second summary says "no user instructions" because it can't see
       the user messages that were in the first summary.

    The fix: pass prior summaries to the summarizer so accumulated context
    is carried forward.
    """

    async def test_tool_only_compaction_carries_forward_prior_summary(self):
        """When compacting only tool outputs, the prior summary must be
        passed to the summarizer so user context is preserved.

        Simulates the death spiral: [SUMMARY, tool, tool, tool, tool, tool, ...retained]
        where the second compaction only sees tool outputs as new messages.
        """
        conv = ContextWindow("chan1")
        conv.system_prompt = ""

        # Build: 1 SUMMARY (from prior compaction) + 5 tool outputs + 4 recent messages
        # Token math: each msg ~35 chars, 10 msgs total → ~100 tokens → triggers at 80%
        # of max_tokens=50. retain_count will be ~2-3.
        conv.messages = [
            # Prior summary with user context
            {
                "role": "assistant",
                "content": "[Summary] User asked to investigate duplicate LLM responses bug. Branch: yozlet/first-fixes.",
                "_message_type": MessageType.SUMMARY,
            },
            # Tool outputs (no user messages in here!)
            {
                "role": "tool",
                "content": "[Task abc123] file contents of agent_loop.py",
                "_message_type": MessageType.MESSAGE,
            },
            {
                "role": "tool",
                "content": "[Task def456] file contents of cli.py",
                "_message_type": MessageType.MESSAGE,
            },
            {
                "role": "tool",
                "content": "[Task ghi789] file contents of channel.py",
                "_message_type": MessageType.MESSAGE,
            },
            {
                "role": "tool",
                "content": "[Task jkl012] file contents of agent_loop.py again",
                "_message_type": MessageType.MESSAGE,
            },
            {
                "role": "tool",
                "content": "[Task mno345] file contents of channel.py again",
                "_message_type": MessageType.MESSAGE,
            },
            # Recent messages (retained)
            {
                "role": "assistant",
                "content": "b" * 35,
                "_message_type": MessageType.MESSAGE,
            },
            {
                "role": "tool",
                "content": "c" * 35,
                "_message_type": MessageType.MESSAGE,
            },
            {
                "role": "assistant",
                "content": "d" * 35,
                "_message_type": MessageType.MESSAGE,
            },
            {
                "role": "user",
                "content": "e" * 35,
                "_message_type": MessageType.MESSAGE,
            },
        ]

        captured_new = []
        captured_prior = []

        async def capture_summarize(messages, prior_summaries=None):
            captured_new.extend(messages)
            if prior_summaries:
                captured_prior.extend(prior_summaries)
            return "User asked to investigate duplicate responses. Reading source files."

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        with patch.object(plugin, "_summarize", side_effect=capture_summarize):
            result = await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert result is True, "Compaction should have been performed"

        # The prior SUMMARY must be passed to the summarizer
        assert len(captured_prior) == 1, (
            f"Expected 1 prior summary (the earlier compaction result), "
            f"got {len(captured_prior)}"
        )
        assert "duplicate LLM responses" in captured_prior[0]["content"], (
            "Prior summary content must include the user context about the bug investigation"
        )

        # New messages should not include the SUMMARY (it was separated to prior_summaries)
        for msg in captured_new:
            assert msg.get("content", "") != captured_prior[0]["content"], (
                "SUMMARY content should not be duplicated in new messages"
            )

    async def test_no_prior_summaries_passes_none(self):
        """First compaction (no prior summaries) should pass None for prior_summaries."""
        conv = ContextWindow("chan1")
        conv.system_prompt = ""

        # All MESSAGE type, no summaries
        conv.messages = [
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}
            for _ in range(10)
        ]

        captured_prior = None

        async def capture_summarize(messages, prior_summaries=None):
            nonlocal captured_prior
            captured_prior = prior_summaries
            return "first summary"

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        with patch.object(plugin, "_summarize", side_effect=capture_summarize):
            await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert captured_prior == [], (
            f"First compaction should have empty prior_summaries, got {captured_prior}"
        )


class TestSummarizeTruncation:
    """Tests for _summarize message truncation when input is large."""

    async def test_summarize_truncates_large_input(self):
        """_summarize must truncate to head + marker + tail when given >100 messages."""
        from corvidae.compaction import CompactionPlugin

        plugin = CompactionPlugin(pm=None)
        plugin._llm_client = AsyncMock()

        # Build 200 messages so truncation kicks in.
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(200)]

        captured_payload = {}

        async def fake_chat(payload):
            captured_payload["payload"] = payload
            return {"choices": [{"message": {"content": "summary"}}]}

        plugin._llm_client.chat = fake_chat
        await plugin._summarize(messages)

        user_msg = captured_payload["payload"][1]  # index 0 is system prompt
        import json
        sent = json.loads(user_msg["content"])

        # head (50) + marker (1) + tail (50) = 101 entries
        assert len(sent) == 101
        assert sent[50]["content"] == "[...100 messages omitted...]"
        assert sent[0]["content"] == "msg 0"
        assert sent[-1]["content"] == "msg 199"

    async def test_summarize_does_not_truncate_small_input(self):
        """_summarize must not truncate when given ≤100 messages."""
        from corvidae.compaction import CompactionPlugin

        plugin = CompactionPlugin(pm=None)
        plugin._llm_client = AsyncMock()

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(50)]

        captured_payload = {}

        async def fake_chat(payload):
            captured_payload["payload"] = payload
            return {"choices": [{"message": {"content": "summary"}}]}

        plugin._llm_client.chat = fake_chat
        await plugin._summarize(messages)

        import json
        sent = json.loads(captured_payload["payload"][1]["content"])
        assert len(sent) == 50
        assert all("messages omitted" not in m.get("content", "") for m in sent)

    def test_compact_conversation_hookimpl_has_no_client_parameter(self):
        """compact_conversation hookimpl must NOT have a 'client' parameter.

        After Part 3, client is obtained from LLMPlugin internally, so the
        hook signature drops the 'client' parameter. This test inspects the
        actual method signature.
        """
        import inspect
        from corvidae.compaction import CompactionPlugin

        method = CompactionPlugin.compact_conversation
        sig = inspect.signature(method)
        param_names = list(sig.parameters.keys())

        assert "client" not in param_names, (
            f"compact_conversation hookimpl must not have a 'client' parameter "
            f"after Part 3. Current parameters: {param_names}"
        )
