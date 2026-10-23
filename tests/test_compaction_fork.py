"""Compaction as a fork of the dropped prefix (GitHub issue #51, phase 2).

Covers the tool-block snap of the retain walk, the before_compaction hook
(fired after the summary succeeds and before replace_with_summary), the
fork summary path, and its fallback to the standalone summarizer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.attribution import get_attribution
from corvidae.compaction import CompactionPlugin
from corvidae.context import ContextWindow, MessageType
from corvidae.fork import ForkRefused, ForkResult
from corvidae.hooks import create_plugin_manager, hookimpl

from helpers import build_plugin_and_channel
from llm_response_fixtures import _make_text_response


def _msg(role: str, content: str, db_id: int, **extra) -> dict:
    """A persisted window message."""
    return {"role": role, "content": content, "_db_id": db_id,
            "_message_type": MessageType.MESSAGE, **extra}


def _call(call_id: str) -> dict:
    return {"id": call_id, "function": {"name": "shell", "arguments": "{}"}}


def _straddling_window() -> ContextWindow:
    """Ten messages; with max_tokens=200 and the default 0.5 retention the
    retain walk keeps the last three, cutting between the call at index 6
    and its result at index 7."""
    conv = ContextWindow("test:scope")
    conv.system_prompt = "You are Lex."
    big = "x" * 120
    conv.messages = [
        _msg("user", big, 1),
        _msg("assistant", big, 2),
        _msg("user", big, 3),
        _msg("assistant", big, 4),
        _msg("user", big, 5),
        _msg("assistant", big, 6),
        _msg("assistant", "checking " + "y" * 150, 7, tool_calls=[_call("c1")]),
        _msg("tool", "short", 8, tool_call_id="c1"),
        _msg("assistant", "ok", 9),
        _msg("user", "next", 10),
    ]
    return conv


MAX_TOKENS = 200


def _retain_walk_count(conv: ContextWindow) -> int:
    """The unsnapped retain count, computed the way compaction does."""
    from corvidae.context import message_tokens

    budget = int(MAX_TOKENS * 0.5)
    count = tokens = 0
    for m in reversed(conv.messages):
        t = message_tokens(m)
        if tokens + t > budget and count > 0:
            break
        tokens += t
        count += 1
    return count


class _Order:
    """Records before_compaction / on_compaction calls and window length."""

    def __init__(self, conv):
        self.conv = conv
        self.calls = []

    @hookimpl
    async def before_compaction(self, channel, compacted_ids, cut):
        self.calls.append(("before", list(compacted_ids), cut, len(self.conv.messages)))

    @hookimpl
    async def on_compaction(self, channel, summary_msg, retain_count, compacted_ids):
        self.calls.append(("on", list(compacted_ids), retain_count, len(self.conv.messages)))


class _FakeAgent:
    """Stands in for Agent: forks enabled for the named stages."""

    def __init__(self, stages=("compaction",), text="forked summary"):
        self.stages = set(stages)
        self.fork = AsyncMock(return_value=ForkResult(text=text, message={}))

    def fork_enabled(self, stage):
        return stage in self.stages


def _plugin(conv, agent=None, config=None):
    """A CompactionPlugin on a real pm with an order recorder (and agent)."""
    pm = create_plugin_manager()
    order = _Order(conv)
    pm.register(order, name="order")
    if agent is not None:
        pm.register(agent, name="agent")
    plugin = CompactionPlugin(pm=pm)
    if config is not None:
        plugin._fork_prompt = config
    return plugin, order


def _channel():
    channel = MagicMock()
    channel.id = "test:scope"
    return channel


# ---------------------------------------------------------------------------
# Tool-block snap
# ---------------------------------------------------------------------------


class TestRetainSnap:
    def test_fixture_really_straddles(self):
        conv = _straddling_window()
        assert _retain_walk_count(conv) == 3

    async def test_cut_backs_up_to_keep_the_tool_cycle_whole(self):
        conv = _straddling_window()
        plugin, order = _plugin(conv)
        with patch.object(plugin, "_summarize", AsyncMock(return_value="sum")):
            assert await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
            )
        # The call (db 7) stays with its result (db 8).
        assert [m.get("_db_id") for m in conv.messages[1:]] == [7, 8, 9, 10]
        before = order.calls[0]
        assert before[1] == [1, 2, 3, 4, 5, 6]
        assert before[2] == 6
        assert order.calls[1][2] == 4  # retain_count

    async def test_cycle_spanning_the_window_still_compacts(self):
        """No clean cut keeps anything: backing up reaches the head and the
        forward snap reaches the tail. The walk's cut stands rather than
        skipping compaction (which would stall it every turn)."""
        conv = ContextWindow("test:scope")
        big = "x" * 200
        conv.messages = [
            _msg("assistant", "", 1, tool_calls=[_call("c1")]),
            *[_msg("user", big, i) for i in range(2, 7)],
            _msg("tool", "r", 7, tool_call_id="c1"),
        ]
        plugin, order = _plugin(conv)
        with patch.object(plugin, "_summarize", AsyncMock(return_value="sum")):
            assert await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
            )
        assert order.calls[0][1][0] == 1  # the call was dropped


def _long_span_window(trailing_exchanges: int) -> ContextWindow:
    """A long-running call near the head, 40 exchanges of chat while it
    runs (R4), its result, then ``trailing_exchanges`` more exchanges."""
    conv = ContextWindow("test:scope")
    conv.system_prompt = "sys"
    big = "word " * 120
    msgs = [
        _msg("user", "run the long job", 1),
        _msg("assistant", "", 2, tool_calls=[_call("c1")]),
    ]
    i = 3
    for _ in range(40):
        msgs += [_msg("user", big, i), _msg("assistant", big, i + 1)]
        i += 2
    msgs.append(_msg("tool", "job done", i, tool_call_id="c1"))
    i += 1
    for _ in range(trailing_exchanges):
        msgs += [_msg("user", big, i), _msg("assistant", big, i + 1)]
        i += 2
    conv.messages = msgs
    return conv


def _orphans(messages: list[dict]) -> list[dict]:
    """Tool results whose call is not in the window."""
    calls = {c["id"] for m in messages for c in (m.get("tool_calls") or [])}
    return [m for m in messages if m.get("role") == "tool" and m["tool_call_id"] not in calls]


class TestSnapLiveness:
    """Backing the cut up to a long-span call must not stall compaction."""

    async def test_long_span_cycle_is_dropped_whole_when_the_tail_survives(self):
        conv = _long_span_window(trailing_exchanges=20)
        result_id = 83
        max_tokens = int(conv.token_estimate() / 0.9)
        plugin, order = _plugin(conv)
        with patch.object(plugin, "_summarize", AsyncMock(return_value="sum")):
            assert await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=max_tokens
            )
        assert conv.token_estimate() < 0.8 * max_tokens
        # The cut moved forward: the call and its result both went.
        assert all(m.get("_db_id") not in (2, result_id) for m in conv.messages)
        assert result_id in order.calls[0][1]
        assert _orphans(conv.messages) == []
        assert len(conv.messages) == 1 + 40  # summary + the 20 trailing exchanges

    @pytest.mark.parametrize("trailing", [0, 3])
    async def test_short_tail_keeps_the_planned_cut(self, trailing):
        """When dropping the whole cycle would leave too little (the result
        is at or near the tail), the walk's own cut stands, as before #51:
        compaction still makes progress."""
        conv = _long_span_window(trailing_exchanges=trailing)
        max_tokens = int(conv.token_estimate() / 0.9)
        plugin, _ = _plugin(conv)
        with patch.object(plugin, "_summarize", AsyncMock(return_value="sum")):
            assert await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=max_tokens
            )
        assert conv.token_estimate() < 0.8 * max_tokens
        assert conv.token_estimate() > 0.3 * max_tokens

    async def test_short_span_still_backs_up(self):
        """The small straddle keeps backing up: the retained tail stays well
        under the trigger."""
        conv = _straddling_window()
        plugin, _ = _plugin(conv)
        with patch.object(plugin, "_summarize", AsyncMock(return_value="sum")):
            await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
            )
        assert [m.get("_db_id") for m in conv.messages[1:]] == [7, 8, 9, 10]


# ---------------------------------------------------------------------------
# before_compaction
# ---------------------------------------------------------------------------


class TestBeforeCompaction:
    async def test_fires_before_replace_with_window_intact(self):
        conv = _straddling_window()
        plugin, order = _plugin(conv)
        with patch.object(plugin, "_summarize", AsyncMock(return_value="sum")):
            await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
            )
        assert [c[0] for c in order.calls] == ["before", "on"]
        assert order.calls[0][3] == 10  # window still whole at before_compaction
        assert order.calls[1][3] == 5   # summary + 4 retained at on_compaction
        assert order.calls[0][1] == order.calls[1][1]

    @pytest.mark.parametrize("summary", ["", "   "])
    async def test_not_fired_for_empty_summary(self, summary):
        conv = _straddling_window()
        plugin, order = _plugin(conv)
        with patch.object(plugin, "_summarize", AsyncMock(return_value=summary)):
            await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
            )
        assert order.calls == []
        assert len(conv.messages) == 10

    async def test_not_fired_when_summarizer_raises(self):
        conv = _straddling_window()
        plugin, order = _plugin(conv)
        with patch.object(plugin, "_summarize", AsyncMock(side_effect=RuntimeError("down"))):
            with pytest.raises(RuntimeError):
                await plugin.compact_conversation(
                    channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
                )
        assert order.calls == []

    async def test_hook_failure_does_not_block_compaction(self):
        conv = _straddling_window()
        plugin, order = _plugin(conv)

        class Boom:
            @hookimpl
            async def before_compaction(self, channel, compacted_ids, cut):
                raise RuntimeError("boom")

        plugin.pm.register(Boom(), name="boom")
        with patch.object(plugin, "_summarize", AsyncMock(return_value="sum")):
            assert await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
            )
        assert len(conv.messages) == 5


# ---------------------------------------------------------------------------
# Fork summary path
# ---------------------------------------------------------------------------


class TestForkSummary:
    async def test_fork_produces_summary_and_skips_standalone(self):
        conv = _straddling_window()
        agent = _FakeAgent()
        plugin, order = _plugin(conv, agent)
        summarize = AsyncMock(return_value="standalone")
        channel = _channel()
        with patch.object(plugin, "_summarize", summarize):
            await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=MAX_TOKENS
            )
        summarize.assert_not_called()
        agent.fork.assert_awaited_once()
        args, kwargs = agent.fork.call_args
        assert args[0] is channel
        assert args[1] == CompactionPlugin.DEFAULT_FORK_PROMPT
        assert kwargs["cut"] == 6
        assert conv.messages[0]["content"] == "[Summary of earlier conversation]\nforked summary"
        assert conv.messages[0]["_message_type"] == MessageType.SUMMARY

    async def test_configured_fork_prompt_is_used(self):
        conv = _straddling_window()
        agent = _FakeAgent()
        plugin, _ = _plugin(conv, agent)
        await plugin.on_init(plugin.pm, {"agent": {"compaction_fork_prompt": "Sum it up."}})
        await plugin.compact_conversation(
            channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
        )
        assert agent.fork.call_args.args[1] == "Sum it up."

    async def test_fork_runs_under_compaction_attribution(self):
        conv = _straddling_window()
        agent = _FakeAgent()
        seen = {}

        async def fork(channel, instruction, *, cut=None, extra_body=None):
            seen.update(get_attribution())
            return ForkResult(text="s", message={})

        agent.fork = fork
        plugin, _ = _plugin(conv, agent)
        await plugin.compact_conversation(
            channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
        )
        assert seen["stage"] == "compaction"

    @pytest.mark.parametrize("failure", [
        ForkRefused("tool call"), RuntimeError("http 500"), "",
    ])
    async def test_fork_failure_falls_back_to_standalone(self, failure):
        conv = _straddling_window()
        agent = _FakeAgent()
        if isinstance(failure, str):
            agent.fork = AsyncMock(return_value=ForkResult(text=failure, message={}))
        else:
            agent.fork = AsyncMock(side_effect=failure)
        plugin, order = _plugin(conv, agent)
        seen = {}

        async def summarize(messages, prior_summaries=None, *, max_tokens):
            seen.update(get_attribution())
            return "standalone summary"

        with patch.object(plugin, "_summarize", side_effect=summarize):
            assert await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
            )
        assert conv.messages[0]["content"].endswith("standalone summary")
        assert seen["mode"] == "standalone"
        assert seen["stage"] == "compaction"

    async def test_disabled_stage_uses_standalone(self):
        conv = _straddling_window()
        agent = _FakeAgent(stages=())
        plugin, _ = _plugin(conv, agent)
        summarize = AsyncMock(return_value="standalone")
        with patch.object(plugin, "_summarize", summarize):
            await plugin.compact_conversation(
                channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
            )
        agent.fork.assert_not_called()
        summarize.assert_awaited_once()

    async def test_summary_role_rule_unchanged(self):
        """#56: no user message in the retained tail → user-role summary."""
        conv = _straddling_window()
        conv.messages[-1] = _msg("assistant", "next", 10)
        agent = _FakeAgent()
        plugin, _ = _plugin(conv, agent)
        await plugin.compact_conversation(
            channel=_channel(), conversation=conv, max_tokens=MAX_TOKENS
        )
        assert conv.messages[0]["role"] == "user"


class TestForkThroughRealAgent:
    async def test_fork_request_is_prefix_of_main_render(self):
        """End to end through Agent.fork: the summary request is the main
        turn's render up to the cut plus the instruction."""
        agent, channel, db = await build_plugin_and_channel()
        try:
            await agent.on_init(agent.pm, {"agent": {"forks": {"enabled": True}}})
            client = MagicMock()
            client.chat = AsyncMock(return_value=_make_text_response("real fork"))
            agent._client = client
            conv = _straddling_window()
            channel.conversation = conv
            full = conv.build_prompt(frozenset())

            plugin = CompactionPlugin(pm=agent.pm)
            assert await plugin.compact_conversation(
                channel=channel, conversation=conv, max_tokens=MAX_TOKENS
            )
            messages = client.chat.call_args.args[0]
            assert messages[:-1] == full[:7]  # system + 6 dropped messages
            assert messages[-1] == {"role": "user", "content": CompactionPlugin.DEFAULT_FORK_PROMPT}
            assert conv.messages[0]["content"].endswith("real fork")
        finally:
            await db.close()
