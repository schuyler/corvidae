"""Tests for corvidae.compaction.CompactionPlugin."""

from unittest.mock import AsyncMock, MagicMock, patch

from corvidae.context import ContextWindow, MessageType


def _make_channel(channel_id: str = "test:scope1") -> object:
    """Build a minimal mock channel for compaction tests."""
    channel = MagicMock()
    channel.id = channel_id
    return channel


class TestCompactCarriesForwardSummaries:
    """Compaction hands the view call the live window prefix, summaries
    included (the prior-summaries split was removed with the standalone
    summarizer; #380/#388, R5)."""

    async def test_prior_summary_is_in_the_prefix_the_view_receives(self):
        """resolve_compaction passes ``_view_summary`` the conversation and the
        cut, and the prefix ``conversation.messages[:cut]`` still holds the
        existing SUMMARY row, so repeated compaction carries it forward
        (the "death spiral" guard) by rendering it, not by a split.

        Token math: 10 msgs x 35 chars -> int(350/3.5)=100 >= 40 triggers.
        """
        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "assistant", "content": "a" * 35, "_message_type": MessageType.SUMMARY},
        ] + [
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}
            for _ in range(9)
        ]

        seen: dict = {}

        async def capture_view(channel, conversation, cut):
            # Snapshot the prefix the view call would render.
            seen["conversation"] = conversation
            seen["cut"] = cut
            seen["prefix"] = list(conversation.messages[:cut])
            return "carried forward summary"

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        with patch.object(plugin, "_view_summary", side_effect=capture_view):
            await plugin.resolve_compaction(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert seen["conversation"] is conv
        assert any(
            m.get("_message_type") == MessageType.SUMMARY for m in seen["prefix"]
        ), "the existing summary row must be in the prefix the view renders"


# ---------------------------------------------------------------------------
# TestOnCompactionHookFired
# ---------------------------------------------------------------------------


class TestOnCompactionHookFired:
    """Tests that CompactionPlugin fires the on_compaction hook after replace_with_summary."""

    async def test_on_compaction_hook_fired_after_compaction(self):
        """resolve_compaction must fire pm.ahook.on_compaction after replace_with_summary."""
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

        # The summary comes from the view-call seam; no LLM is involved.
        with patch.object(plugin, "_view_summary", AsyncMock(return_value="hook test summary")):
            await plugin.resolve_compaction(
                channel=channel, conversation=conv, max_tokens=50
            )

        mock_pm.ahook.on_compaction.assert_awaited_once()
        call_kwargs = mock_pm.ahook.on_compaction.call_args.kwargs
        assert call_kwargs["channel"] is channel
        assert "summary_msg" in call_kwargs
        assert "retain_count" in call_kwargs

    async def test_on_compaction_not_fired_when_no_compaction(self):
        """resolve_compaction must NOT fire on_compaction when below threshold."""
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

        await plugin.resolve_compaction(
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
        in the prefix the view call renders so user context is preserved.

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

        captured_prefix: list[dict] = []

        async def capture_view(channel, conversation, cut):
            # The prefix the view call renders: the summary must be in it.
            captured_prefix.extend(conversation.messages[:cut])
            return "User asked to investigate duplicate responses. Reading source files."

        from corvidae.compaction import CompactionPlugin
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        with patch.object(plugin, "_view_summary", side_effect=capture_view):
            result = await plugin.resolve_compaction(
                channel=channel, conversation=conv, max_tokens=50
            )

        assert result is True, "Compaction should have been performed"

        # The prior SUMMARY is in the prefix the view renders
        summaries = [
            m for m in captured_prefix
            if m.get("_message_type") == MessageType.SUMMARY
        ]
        assert len(summaries) == 1
        assert "duplicate LLM responses" in summaries[0]["content"], (
            "Prior summary content must include the user context about the bug investigation"
        )


class TestResolveCompactionSignature:
    """Signature of the resolve_compaction hookimpl."""

    def test_resolve_compaction_hookimpl_has_no_client_parameter(self):
        """resolve_compaction hookimpl must NOT have a 'client' parameter.

        After Part 3, client is obtained from LLMPlugin internally, so the
        hook signature drops the 'client' parameter. This test inspects the
        actual method signature.
        """
        import inspect
        from corvidae.compaction import CompactionPlugin

        method = CompactionPlugin.resolve_compaction
        sig = inspect.signature(method)
        param_names = list(sig.parameters.keys())

        assert "client" not in param_names, (
            f"resolve_compaction hookimpl must not have a 'client' parameter "
            f"after Part 3. Current parameters: {param_names}"
        )


class TestSizingUnifiesAcrossSites:
    """Compaction must see size that hides in reasoning_content/tool_calls,
    not just visible content -- and the fix must actually bound what gets
    sent to the LLM, not merely trigger.
    """

    def _reasoning_shaped_conversation(self, count: int = 16) -> ContextWindow:
        """Messages shaped like a thinking-model channel: small `content`
        per message, but assistant turns also carry a large persisted
        `reasoning_content` field -- invisible to content-only sizing.
        """
        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        messages = []
        for i in range(count):
            role = "assistant" if i % 2 == 0 else "user"
            msg = {
                "role": role,
                "content": f"short reply {i}",
                "_message_type": MessageType.MESSAGE,
            }
            if role == "assistant":
                # ~3200 chars of persisted reasoning, invisible to
                # content-only sizing but present in what's actually stored
                # and sent.
                msg["reasoning_content"] = "reasoning filler text " * 140
            messages.append(msg)
        conv.messages = messages
        return conv

    async def test_compaction_succeeds_despite_large_reasoning_content(self):
        """resolve_compaction must trigger and complete without raising
        when content is small but reasoning_content is large (the trigger
        sizes what is stored, not just visible content). The summary call
        itself is the view seam, covered by tests/test_compaction_fork.py."""
        from corvidae.compaction import CompactionPlugin

        conv = self._reasoning_shaped_conversation()
        plugin = CompactionPlugin(pm=None)
        channel = _make_channel()

        max_tokens = 4000
        with patch.object(plugin, "_view_summary", AsyncMock(return_value="summary")):
            result = await plugin.resolve_compaction(
                channel=channel, conversation=conv, max_tokens=max_tokens
            )

        # Content alone is tiny (16 short strings) -- a content-only trigger
        # check would never fire even though the persisted messages are
        # large once reasoning_content is included. Compaction must trigger
        # and succeed, not silently skip while the real, oversized window
        # keeps growing.
        assert result is True, (
            "compaction must trigger and succeed when size is hidden in "
            "reasoning_content, not just visible content"
        )


class TestCooldownWithoutPriorFailure:
    """A channel that never failed compaction is never in cooldown.

    The cooldown used to default "never failed" to timestamp 0. Since
    time.monotonic() counts from boot, compaction was skipped for the first
    30 seconds of a machine's uptime: a daemon started at boot, or a CI
    runner that had just booted.
    """

    async def test_compacts_within_30s_of_boot(self):
        """With the monotonic clock at 5 s, an over-threshold window compacts."""
        from corvidae.compaction import CompactionPlugin

        # Ten short messages against a 50-token budget: over the threshold.
        conv = ContextWindow("chan1")
        conv.system_prompt = ""
        conv.messages = [
            {"role": "user", "content": "a" * 35, "_message_type": MessageType.MESSAGE}
            for _ in range(10)
        ]
        plugin = CompactionPlugin(pm=None)

        # Pretend the machine booted five seconds ago.
        with patch("corvidae.compaction.time.monotonic", return_value=5.0), \
                patch.object(plugin, "_view_summary", AsyncMock(return_value="summary")):
            result = await plugin.resolve_compaction(
                channel=_make_channel(), conversation=conv, max_tokens=50
            )

        assert result is True
