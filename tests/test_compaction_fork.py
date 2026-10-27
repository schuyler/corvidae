"""Compaction as a fork of the dropped prefix (GitHub issue #51, phase 2):
the cut moved off straddling tool cycles without stalling, the
before_compaction hook, the fork summary and its standalone fallback."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corvidae.compaction import CompactionPlugin
from corvidae.context import ContextWindow, MessageType
from corvidae.fork import ForkRefused
from corvidae.hooks import create_plugin_manager, hookimpl

from helpers import FakeForkAgent, build_plugin_and_channel
from llm_response_fixtures import _make_text_response

MAX_TOKENS = 20000


def _msg(role, content, db_id, **extra):
    return {"role": role, "content": content, "_db_id": db_id,
            "_message_type": MessageType.MESSAGE, **extra}


def _call(call_id):
    return {"id": call_id, "function": {"name": "shell", "arguments": "{}"}}


def _straddling_window() -> ContextWindow:
    """Ten messages, ~16.9k tokens. At MAX_TOKENS the retain walk keeps the
    last three, cutting between the large call message (index 6, ~10k
    tokens) and its result (index 7); backing up to keep the cycle leaves
    the window well under the trigger, summary allowance included."""
    conv = ContextWindow("test:scope")
    conv.system_prompt = "You are Lex."
    big = "word " * 1100
    conv.messages = [_msg("user" if i % 2 == 0 else "assistant", big, i + 1) for i in range(6)] + [
        _msg("assistant", "checking " + "word " * 10100, 7, tool_calls=[_call("c1")]),
        _msg("tool", "short", 8, tool_call_id="c1"),
        _msg("assistant", "ok", 9),
        _msg("user", "next", 10),
    ]
    return conv


def _long_span_window(trailing_exchanges: int) -> ContextWindow:
    """A long-running call near the head (db 2), 40 exchanges of chat while
    it runs, its result (db 83), then ``trailing_exchanges`` more."""
    conv = ContextWindow("test:scope")
    big = "word " * 120
    msgs = [_msg("user", "run the long job", 1), _msg("assistant", "", 2, tool_calls=[_call("c1")])]
    msgs += [_msg("user" if i % 2 == 0 else "assistant", big, 3 + i) for i in range(80)]
    msgs.append(_msg("tool", "job done", 83, tool_call_id="c1"))
    msgs += [_msg("user" if i % 2 == 0 else "assistant", big, 84 + i) for i in range(2 * trailing_exchanges)]
    conv.messages = msgs
    return conv


class _Order:
    """Records before_compaction / on_compaction with the window length."""

    def __init__(self, conv):
        self.conv = conv
        self.calls = []

    @hookimpl
    async def before_compaction(self, channel, compacted_ids, cut):
        self.calls.append(("before", list(compacted_ids), cut, len(self.conv.messages)))

    @hookimpl
    async def on_compaction(self, channel, summary_msg, retain_count, compacted_ids):
        self.calls.append(("on", list(compacted_ids), retain_count, len(self.conv.messages)))


async def _compact(conv, agent=None, summary="sum", max_tokens=MAX_TOKENS):
    """Run one compaction with a patched standalone summarizer; returns
    (result, order recorder, summarize mock)."""
    pm = create_plugin_manager()
    order = _Order(conv)
    pm.register(order, name="order")
    if agent is not None:
        pm.register(agent, name="agent")
    plugin = CompactionPlugin(pm=pm)
    summarize = AsyncMock(side_effect=summary) if isinstance(summary, Exception) else AsyncMock(return_value=summary)
    channel = MagicMock()
    channel.id = "test:scope"
    with patch.object(plugin, "_summarize", summarize):
        result = await plugin.resolve_compaction(channel=channel, conversation=conv, max_tokens=max_tokens)
    return result, order, summarize


def _orphans(messages):
    calls = {c["id"] for m in messages for c in (m.get("tool_calls") or [])}
    return [m for m in messages if m.get("role") == "tool" and m["tool_call_id"] not in calls]


# ---------------------------------------------------------------------------
# Where the cut lands
# ---------------------------------------------------------------------------


async def test_short_span_cycle_backs_up_and_hooks_fire_in_order():
    conv = _straddling_window()
    result, order, _ = await _compact(conv)
    assert result
    assert [m.get("_db_id") for m in conv.messages[1:]] == [7, 8, 9, 10]
    # before_compaction sees the whole window; on_compaction the truncated one.
    assert order.calls == [
        ("before", [1, 2, 3, 4, 5, 6], 6, 10),
        ("on", [1, 2, 3, 4, 5, 6], 4, 5),
    ]


async def test_long_span_cycle_is_dropped_whole_when_the_tail_survives():
    conv = _long_span_window(trailing_exchanges=20)
    max_tokens = int(conv.token_estimate() / 0.9)
    result, order, _ = await _compact(conv, max_tokens=max_tokens)
    assert result and conv.token_estimate() < 0.8 * max_tokens
    assert len(conv.messages) == 1 + 40  # summary + the trailing exchanges
    assert 83 in order.calls[0][1] and _orphans(conv.messages) == []


@pytest.mark.parametrize("trailing", [0, 3])
async def test_short_tail_keeps_the_planned_cut(trailing):
    """Backing up would stall compaction and moving forward would keep too
    little: the walk's own cut stands, as before #51."""
    conv = _long_span_window(trailing_exchanges=trailing)
    max_tokens = int(conv.token_estimate() / 0.9)
    result, _, _ = await _compact(conv, max_tokens=max_tokens)
    assert result
    assert 0.3 * max_tokens < conv.token_estimate() < 0.8 * max_tokens


@pytest.mark.parametrize("trailing", [0, 3])
async def test_short_tail_cut_renders_no_unclaimed_tool_message(trailing):
    """#79 case 2: the planned cut drops the call and keeps its result. The
    rendered prompt must still carry no tool message without its call."""
    conv = _long_span_window(trailing_exchanges=trailing)
    max_tokens = int(conv.token_estimate() / 0.9)
    result, _, _ = await _compact(conv, max_tokens=max_tokens)
    assert result
    # The window really is in the orphaned state the render must handle.
    assert any(m.get("tool_call_id") == "c1" for m in conv.messages)
    assert not any(m.get("tool_calls") for m in conv.messages)

    prompt = conv.build_prompt(frozenset())
    seen_calls: set[str] = set()
    for msg in prompt:
        for call in msg.get("tool_calls") or []:
            seen_calls.add(call["id"])
        if msg.get("role") == "tool":
            assert msg["tool_call_id"] in seen_calls, msg
    # The result is reframed, not dropped.
    assert any(m["role"] == "system" and m["content"].endswith("job done") for m in prompt[1:])


async def test_large_system_prompt_does_not_stall_compaction():
    """Repeated rounds with a long-span call behind the summary and a system
    prompt of ~22% of max_tokens: the back-snap has to count the system
    prompt and the summary, or it keeps dropping only the old summary while
    the window climbs toward max_tokens."""
    pm = create_plugin_manager()
    plugin = CompactionPlugin(pm=pm)
    conv = ContextWindow("test:scope")
    conv.system_prompt = "w " * 1800
    channel = MagicMock()
    channel.id = "test:scope"
    rounds = []
    for step in range(1, 261):
        if step == 16:
            conv.messages.append(_msg("assistant", "", step, tool_calls=[_call("c1")]))
        elif step == 61:
            conv.messages.append(_msg("tool", "done", step, tool_call_id="c1"))
        else:
            conv.messages.append(_msg("user" if step % 2 else "assistant", "word " * 60, step))
        before, count = conv.token_estimate(), len(conv.messages)
        with patch.object(plugin, "_summarize", AsyncMock(return_value="s " * 600)):
            if await plugin.resolve_compaction(channel=channel, conversation=conv, max_tokens=8192):
                rounds.append((count - len(conv.messages) + 1, before / 8192))
    assert rounds
    assert all(dropped > 2 for dropped, _ in rounds)
    assert max(ratio for _, ratio in rounds) < 0.9


@pytest.mark.parametrize("summary", ["", "  ", RuntimeError("down")])
async def test_failed_summary_fires_no_hooks(summary):
    conv = _straddling_window()
    if isinstance(summary, Exception):
        with pytest.raises(RuntimeError):
            await _compact(conv, summary=summary)
    else:
        result, order, _ = await _compact(conv, summary=summary)
        assert order.calls == []
    assert len(conv.messages) == 10


# ---------------------------------------------------------------------------
# Fork summary and fallback
# ---------------------------------------------------------------------------


async def test_fork_produces_summary():
    conv = _straddling_window()
    conv.messages[-1] = _msg("assistant", "next", 10)  # no user in tail: #56 role rule
    agent = FakeForkAgent({"compaction"}, text="forked summary")
    result, _, summarize = await _compact(conv, agent)
    summarize.assert_not_called()
    call = agent.calls[0]
    assert (call["instruction"], call["cut"]) == (CompactionPlugin.DEFAULT_FORK_PROMPT, 6)
    assert call["attribution"]["stage"] == "compaction"
    assert conv.messages[0] == {
        "role": "user", "content": "[Summary of earlier conversation]\nforked summary",
        "_message_type": MessageType.SUMMARY,
    }


@pytest.mark.parametrize("agent", [
    FakeForkAgent({"compaction"}, error=ForkRefused("tool call")),
    FakeForkAgent({"compaction"}, error=RuntimeError("http 500")),
    FakeForkAgent({"compaction"}, text=""),
    FakeForkAgent(set(), text="forks off"),
])
async def test_fork_miss_falls_back_to_standalone(agent):
    conv = _straddling_window()
    seen = {}

    async def summarize(*args, **kwargs):
        from corvidae.attribution import get_attribution
        seen.update(get_attribution())
        return "standalone summary"

    pm = create_plugin_manager()
    pm.register(agent, name="agent")
    plugin = CompactionPlugin(pm=pm)
    channel = MagicMock()
    channel.id = "test:scope"
    with patch.object(plugin, "_summarize", side_effect=summarize):
        assert await plugin.resolve_compaction(channel=channel, conversation=conv, max_tokens=MAX_TOKENS)
    assert conv.messages[0]["content"].endswith("standalone summary")
    assert (seen["stage"], seen["mode"]) == ("compaction", "standalone")


async def test_fork_through_real_agent_is_prefix_of_main_render():
    agent, channel, db = await build_plugin_and_channel()
    try:
        await agent.on_init(agent.pm, {"agent": {
            "forks": {"enabled": True}, "compaction_fork_prompt": "Sum it up.",
        }})
        client = MagicMock()
        client.chat = AsyncMock(return_value=_make_text_response("real fork"))
        agent._client = client
        conv = _straddling_window()
        channel.conversation = conv
        full = conv.build_prompt(frozenset())

        plugin = CompactionPlugin(pm=agent.pm)
        await plugin.on_init(agent.pm, {"agent": {"compaction_fork_prompt": "Sum it up."}})
        assert await plugin.resolve_compaction(channel=channel, conversation=conv, max_tokens=MAX_TOKENS)
        assert client.chat.call_args.args[0] == full[:7] + [{"role": "user", "content": "Sum it up."}]
        assert conv.messages[0]["content"].endswith("real fork")
    finally:
        await db.close()
