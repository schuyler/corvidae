"""The fork primitive (GitHub issue #51, plans/forks.md Part 1 §1).

A fork's prompt must equal the main turn's render up to the cut, plus one
user instruction, so llama-server reuses the cached prefix. It mutates,
persists, fires and dispatches nothing.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.attribution import get_attribution
from corvidae.context import ContextWindow, MessageType
from corvidae.fork import (
    ForkRefused,
    ForksConfig,
    ForkUnavailable,
    build_fork_prompt,
    parse_forks_config,
    snap_cut_forward,
    snap_cut_to_tool_boundary,
)
from corvidae.hooks import hookimpl

from helpers import build_plugin_and_channel
from llm_response_fixtures import _make_text_response, _make_tool_call, _make_tool_call_response

INSTRUCTION = "Summarize the conversation above."
TOOL_SCHEMAS = [{"type": "function", "function": {"name": "shell", "parameters": {}}}]


def _call(call_id):
    return {"id": call_id, "function": {"name": "shell", "arguments": "{}"}}


def _tool_window() -> ContextWindow:
    """Summary head, a two-call cycle (index 2) whose results (3, 5) are
    interleaved with a user message, then a CONTEXT row and a user turn.
    Clean cuts: 0-2 and 6-9; 3-5 straddle the cycle."""
    conv = ContextWindow("test:scope")
    conv.system_prompt = "You are Lex."
    rows = [
        ({"role": "assistant", "content": "[Summary]\nold"}, MessageType.SUMMARY),
        ({"role": "user", "content": "run two things"}, MessageType.MESSAGE),
        ({"role": "assistant", "content": "", "tool_calls": [_call("a"), _call("b")]}, MessageType.MESSAGE),
        ({"role": "tool", "tool_call_id": "a", "content": "A"}, MessageType.MESSAGE),
        ({"role": "user", "content": "still there?"}, MessageType.MESSAGE),
        ({"role": "tool", "tool_call_id": "b", "content": "B"}, MessageType.MESSAGE),
        ({"role": "assistant", "content": "both done"}, MessageType.MESSAGE),
        ({"role": "system", "content": "[CONTEXT]\nfact"}, MessageType.CONTEXT),
        ({"role": "user", "content": "thanks"}, MessageType.MESSAGE),
    ]
    conv.messages = [{**m, "_db_id": 10 + i, "_message_type": t} for i, (m, t) in enumerate(rows)]
    return conv


@pytest.mark.parametrize("mid_role", ["system", "user"])
@pytest.mark.parametrize("cut", [None, 1, 2, 6, 7, 9])
def test_fork_prompt_is_prefix_of_main_render(cut, mid_role):
    conv = _tool_window()
    full = conv.build_prompt(frozenset({"zz"}), mid_conversation_system_role=mid_role)
    prompt = build_fork_prompt(conv, cut, INSTRUCTION, frozenset({"zz"}), mid_role)
    assert prompt[:-1] == full[: len(prompt) - 1]
    assert prompt[-1] == {"role": "user", "content": INSTRUCTION}


@pytest.mark.parametrize("cut", [-1, 10])
def test_fork_prompt_rejects_out_of_range_cut(cut):
    with pytest.raises(ValueError):
        build_fork_prompt(_tool_window(), cut, INSTRUCTION, frozenset(), "system")


@pytest.mark.parametrize("cut,back,forward", [
    (0, 0, 0), (2, 2, 2), (3, 2, 6), (4, 2, 6), (5, 2, 6), (6, 6, 6), (9, 9, 9),
])
def test_snap_both_directions(cut, back, forward):
    messages = _tool_window().messages
    assert snap_cut_to_tool_boundary(messages, cut) == back
    assert snap_cut_forward(messages, cut) == forward


def test_snap_reaches_a_fixpoint_and_ignores_unanswered_calls():
    """Two overlapping cycles x (call 1, result 3) and y (call 2, result 4);
    z (call 6) is never answered and never straddles."""
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("x")]},
        {"role": "assistant", "content": "", "tool_calls": [_call("y")]},
        {"role": "tool", "tool_call_id": "x", "content": "X"},
        {"role": "tool", "tool_call_id": "y", "content": "Y"},
        {"role": "assistant", "content": "done"},
        {"role": "assistant", "content": "", "tool_calls": [_call("z")]},
        {"role": "user", "content": "hello?"},
    ]
    assert snap_cut_to_tool_boundary(messages, 4) == 1
    assert snap_cut_forward(messages, 2) == 5
    assert snap_cut_to_tool_boundary(messages, 7) == 7


# ---------------------------------------------------------------------------
# agent.forks config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ({}, ForksConfig()),
    ({"forks": {"enabled": True, "reasoning_budget_tokens": 512, "stages": ["x"]}},
     ForksConfig(True, 512, frozenset({"x"}))),
    ({"forks": {"reasoning_budget_tokens": None, "stages": None}}, ForksConfig(False, None, None)),
])
def test_parse_forks_config(raw, expected):
    assert parse_forks_config(raw) == expected


def test_stage_enabled():
    assert ForksConfig(enabled=True).stage_enabled("anything")
    assert not ForksConfig(enabled=True, stages=frozenset({"a"})).stage_enabled("b")
    assert not ForksConfig(enabled=False).stage_enabled("a")


@pytest.mark.parametrize("forks", [
    {"enabled": "yes"}, {"reasoning_budget_tokens": -5}, {"reasoning_budget_tokens": True},
    {"stages": "compaction"}, {"stages": ["a", 3]}, {"bogus": 1}, "on",
])
def test_invalid_forks_config_fails_validation(forks):
    from corvidae.channel import validate_agent_config

    with pytest.raises(ValueError, match="forks"):
        validate_agent_config({"forks": forks})


# ---------------------------------------------------------------------------
# Agent.fork
# ---------------------------------------------------------------------------


class _EventRecorder:
    def __init__(self):
        self.events = []

    @hookimpl
    async def on_conversation_event(self, channel, message, message_type):
        self.events.append(message)


@pytest.fixture
async def forking_agent():
    """(agent, channel, client) with forks on, a fake main client, a live window."""
    plugin, channel, db = await build_plugin_and_channel()
    await plugin.on_init(plugin.pm, {"agent": {
        "forks": {"enabled": True, "reasoning_budget_tokens": 1024},
        "mid_conversation_system_role": "user",
    }})
    plugin._tool_schemas = TOOL_SCHEMAS
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("fork says hi"))
    plugin._client = client
    channel.conversation = _tool_window()
    yield plugin, channel, client
    await db.close()


async def test_fork_request_matches_main_turn(forking_agent):
    plugin, channel, client = forking_agent
    channel.runtime_overrides = {"temperature": 0.3, "memory.k": 4, "tool_choice": "required"}
    channel.pending_tool_call_ids = {"zz"}
    recorder = _EventRecorder()
    plugin.pm.register(recorder, name="recorder")
    before = [dict(m) for m in channel.conversation.messages]
    seen = {}

    async def chat(messages, **kwargs):
        seen.update(get_attribution())
        return _make_text_response("fork says hi")

    client.chat = AsyncMock(side_effect=chat)
    fmt = {"type": "json_schema"}
    result = await plugin.fork(channel, INSTRUCTION, cut=2, extra_body={"response_format": fmt})

    messages = client.chat.call_args.args[0]
    full = channel.conversation.build_prompt(frozenset({"zz"}), mid_conversation_system_role="user")
    assert messages == full[:3] + [{"role": "user", "content": INSTRUCTION}]
    kwargs = client.chat.call_args.kwargs
    assert kwargs["tools"] == TOOL_SCHEMAS
    # Inference overrides ride along; plugin tunables don't; fork keys win.
    assert kwargs["extra_body"] == {
        "temperature": 0.3, "tool_choice": "none",
        "reasoning_budget_tokens": 1024, "response_format": fmt,
    }
    assert result.text == "fork says hi"
    assert result.usage["prompt_tokens"] == 100
    assert seen["mode"] == "fork" and "mode" not in get_attribution()
    # Nothing mutated, persisted, or sent.
    assert channel.conversation.messages == before
    assert recorder.events == []
    plugin.pm.ahook.send_message.assert_not_called()


@pytest.mark.parametrize("response", [
    _make_tool_call_response([_make_tool_call("c1", "shell", {"command": "ls"})]),
    _make_text_response('<tool_call>\n{"name": "shell", "arguments": {}}\n</tool_call>'),
    _make_text_response('  {"name": "shell", "arguments": {"command": "ls"}}'),
    _make_text_response("<function=shell><parameter=command>ls</parameter></function>"),
    _make_text_response('<function name="shell">{}</function>'),
    _make_text_response("<function_call>{}</function_call>"),
    _make_text_response("<functions.shell>{}"),
])
async def test_tool_calls_raise_fork_refused(forking_agent, response):
    plugin, channel, client = forking_agent
    client.chat = AsyncMock(return_value=response)
    with pytest.raises(ForkRefused):
        await plugin.fork(channel, INSTRUCTION)
    assert channel.pending_tool_call_ids == set()


@pytest.mark.parametrize("text", [
    '{"summary": "I helped", "topic_tags": []}', '{"objections": []}',
    "We debugged the name and arguments of a function.",
    "<functional> components came up; I summarized them.",
])
async def test_ordinary_replies_are_not_refused(forking_agent, text):
    plugin, channel, client = forking_agent
    client.chat = AsyncMock(return_value=_make_text_response(text))
    assert (await plugin.fork(channel, INSTRUCTION)).text == text


@pytest.mark.parametrize("missing", ["conversation", "client"])
async def test_fork_unavailable(forking_agent, missing):
    plugin, channel, client = forking_agent
    if missing == "conversation":
        channel.conversation = None
    else:
        plugin._client = None
    with pytest.raises(ForkUnavailable):
        await plugin.fork(channel, INSTRUCTION)


async def test_fork_config_reloads_and_null_budget_is_omitted(forking_agent):
    plugin, channel, client = forking_agent
    await plugin.on_config_reload({"agent": {"forks": {
        "enabled": True, "stages": ["appraisal"], "reasoning_budget_tokens": None,
    }}})
    assert plugin.fork_enabled("appraisal") and not plugin.fork_enabled("compaction")
    await plugin.fork(channel, INSTRUCTION)
    assert "reasoning_budget_tokens" not in client.chat.call_args.kwargs["extra_body"]


@pytest.mark.parametrize("enabled,background,warns", [
    (True, "small", True), (False, "small", False), (True, "big", False), (True, None, False),
])
async def test_startup_warns_when_forks_move_background_work(caplog, enabled, background, warns):
    from corvidae.llm_plugin import LLMPlugin
    from corvidae.tool import ToolRegistry
    from corvidae.tool_collection import ToolCollectionPlugin

    plugin, channel, db = await build_plugin_and_channel()
    llm = LLMPlugin()
    llm._clients["main"] = MagicMock(model="big")
    if background:
        llm._clients["background"] = MagicMock(model=background)
    plugin.pm.register(llm, name="llm")
    tools = ToolCollectionPlugin()
    tools.registry = ToolRegistry()
    plugin.pm.register(tools, name="tools")
    await plugin.on_init(plugin.pm, {"agent": {"forks": {"enabled": enabled}}})
    with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
        await plugin.on_start({})
    await db.close()
    assert any("fork" in r.getMessage() for r in caplog.records) == warns


async def test_usage_log_mode_column_upgrades_and_records(db):
    from corvidae.hooks import create_plugin_manager
    from corvidae.metrics import USAGE_LOG_DDL, UsageLogPlugin
    from corvidae.persistence import PersistencePlugin

    await db.execute(USAGE_LOG_DDL.replace("    mode TEXT,\n", ""))
    await db.execute("INSERT INTO usage_log (ts, request_id, role, model) VALUES (1, 'old', 'main', 'm')")
    await db.commit()
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    pm.register(UsageLogPlugin(pm), name="usage_log")
    for rid, attribution in [("f", {"mode": "fork"}), ("s", {"mode": "standalone"}), ("t", {})]:
        await pm.ahook.on_llm_response(role="main", model="m", request_id=rid, usage={},
                                       latency_ms=1.0, attribution=attribution, error=None)
    async with db.execute("SELECT request_id, mode FROM usage_log ORDER BY id") as cursor:
        assert await cursor.fetchall() == [("old", None), ("f", "fork"), ("s", "standalone"), ("t", None)]
