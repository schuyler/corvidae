"""The fork primitive (GitHub issue #51, plans/forks.md Part 1 §1).

A fork's prompt must equal the main turn's render up to the cut, plus one
user instruction, so llama-server reuses the cached prefix. It mutates,
persists, fires and dispatches nothing.
"""

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

from corvidae.outbound import OutboundMessage
from helpers import LLMResponseRecorder, build_plugin_and_channel, sent, start_llm_plugin
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


def _straddle_window(with_result: bool = True) -> ContextWindow:
    """user, call c1, final reply, and (optionally) c1's result after the reply (#185)."""
    from corvidae.context import MessageType

    conv = ContextWindow("test:chan")
    conv.system_prompt = "system"
    rows = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1")]},
        {"role": "assistant", "content": "working on it"},
    ]
    if with_result:
        rows.append({"role": "tool", "tool_call_id": "c1", "content": "late"})
    conv.messages = [{**m, "_db_id": i, "_message_type": MessageType.MESSAGE} for i, m in enumerate(rows)]
    return conv


def _tool_content(prompt: list[dict], call_id: str) -> str:
    """Content of the rendered tool message answering ``call_id``."""
    return next(m["content"] for m in prompt if m.get("tool_call_id") == call_id)


def test_fork_cut_renders_call_pending_at_cut():
    """A result landing after the reply must not turn the call into 'abandoned' (#185, R1, R2)."""
    from corvidae.context import TOOL_RESULT_PENDING

    conv = _straddle_window()
    prompt = build_fork_prompt(conv, 3, INSTRUCTION, frozenset(), "system")
    assert _tool_content(prompt, "c1") == TOOL_RESULT_PENDING
    # Equal to what the main turn rendered at reply time.
    main = conv.build_prompt(frozenset({"c1"}))
    reply_render = ContextWindow("test:chan")
    reply_render.system_prompt = "system"
    reply_render.messages = conv.messages[:3]
    assert prompt[:-1] == reply_render.build_prompt(frozenset({"c1"}), mid_conversation_system_role="system")
    assert main  # full render still works


def test_fork_cut_leaves_unanswered_call_abandoned():
    """No result anywhere and not live-pending stays 'abandoned' (R3)."""
    from corvidae.context import TOOL_RESULT_ABANDONED

    conv = _straddle_window(with_result=False)
    prompt = build_fork_prompt(conv, 3, INSTRUCTION, frozenset(), "system")
    assert _tool_content(prompt, "c1") == TOOL_RESULT_ABANDONED


def test_fork_cut_after_result_renders_the_result():
    """A call whose result is before the cut renders that result (R3)."""
    conv = _straddle_window()
    prompt = build_fork_prompt(conv, 4, INSTRUCTION, frozenset(), "system")
    assert _tool_content(prompt, "c1") == "late"


def test_fork_without_cut_unchanged_and_pure():
    """cut=None renders as before and the window is untouched (R3, R4)."""
    conv = _straddle_window()
    before = [dict(m) for m in conv.messages]
    prompt = build_fork_prompt(conv, None, INSTRUCTION, frozenset(), "system")
    assert prompt[:-1] == conv.build_prompt(frozenset(), mid_conversation_system_role="system")
    build_fork_prompt(conv, 3, INSTRUCTION, frozenset(), "system")
    assert conv.messages == before


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
    ({"forks": {"enabled": True, "stages": ["x"]}}, ForksConfig(True, stages=frozenset({"x"}))),
    ({"forks": {"stages": None}}, ForksConfig()),
])
def test_parse_forks_config(raw, expected):
    assert parse_forks_config(raw) == expected


@pytest.mark.parametrize("raw", [{}, {"forks": {}}, {"forks": {"stages": None}}])
def test_forks_default_on(raw):
    """#320: with no ``enabled`` key every stage forks (PRD §3.8)."""
    # Parse the config the way the daemon does; no key means forks on.
    cfg = parse_forks_config(raw)
    assert cfg.enabled is True
    for stage in ("compaction", "consolidation", "appraisal", "critique"):
        assert cfg.stage_enabled(stage), f"stage {stage!r} should fork by default"


async def test_agent_forks_default_on_without_forks_block():
    """#320: an Agent started with no ``agent.forks`` block forks every stage."""
    plugin, channel, db = await build_plugin_and_channel()
    # Initialise with an agent config that has no ``forks`` key at all.
    await plugin.on_init(plugin.pm, {"agent": {}})
    await db.close()
    for stage in ("compaction", "consolidation", "appraisal", "critique"):
        assert plugin.fork_enabled(stage), f"stage {stage!r} should fork by default"


def test_stage_enabled():
    assert ForksConfig(enabled=True).stage_enabled("anything")
    assert not ForksConfig(enabled=True, stages=frozenset({"a"})).stage_enabled("b")
    assert not ForksConfig(enabled=False).stage_enabled("a")


@pytest.mark.parametrize("forks", [
    {"enabled": "yes"},
    {"stages": "compaction"}, {"stages": ["a", 3]}, {"bogus": 1}, "on",
])
def test_invalid_forks_config_fails_validation(forks):
    from corvidae.channel import validate_agent_config

    with pytest.raises(ValueError, match="forks"):
        validate_agent_config({"forks": forks})


@pytest.mark.parametrize("budget", [2048, None])
def test_forks_reasoning_budget_key_is_an_error_pointing_to_the_stage_map(budget):
    """R9: agent.forks.reasoning_budget_tokens is removed; leaving it in
    config names the key and points to llm.stages. Any value, null included."""
    from corvidae.channel import validate_agent_config

    for parse in (parse_forks_config, validate_agent_config):
        with pytest.raises(ValueError) as exc:
            parse({"forks": {"reasoning_budget_tokens": budget}})
        message = str(exc.value)
        assert "agent.forks.reasoning_budget_tokens" in message
        assert "llm.stages" in message


# ---------------------------------------------------------------------------
# fork_body: the body a fork sends on top of the stage profile's (R8)
# ---------------------------------------------------------------------------


def test_fork_body_profile_beats_runtime_override():
    """R8: a runtime override of a key the stage profile's extra_body also
    sets is not sent; the others ride along."""
    from corvidae.fork import fork_body

    body = fork_body({"max_tokens": 200, "temperature": 0.3}, {"max_tokens": 4096}, None)

    assert "max_tokens" not in body
    assert body["temperature"] == 0.3


def test_fork_body_stage_schema_and_tool_choice_win():
    """R8: the stage's own body beats the profile and the override, and
    tool_choice none beats everything."""
    from corvidae.fork import fork_body

    fmt = {"type": "json_schema"}
    body = fork_body(
        {"tool_choice": "required", "response_format": {"type": "text"}},
        {"response_format": {"type": "json_object"}, "tool_choice": "auto"},
        {"response_format": fmt},
    )

    assert body["response_format"] == fmt
    assert body["tool_choice"] == "none"


# ---------------------------------------------------------------------------
# Agent.fork
# ---------------------------------------------------------------------------


class _EventRecorder:
    def __init__(self):
        self.events = []

    @hookimpl
    async def persist_message(self, channel, message, message_type):
        self.events.append(message)


FORK_STAGES = {"compaction": "chat", "consolidation": "chat", "appraisal": "chat"}
MAIN_PROFILE = {"base_url": "http://main.test/v1", "model": "main-m"}


@pytest.fixture
async def forking_agent(monkeypatch):
    """(agent, channel, client) with forks on, a live window, and a started
    LLMPlugin whose default chat client (profile main) is a stub."""
    plugin, channel, db = await build_plugin_and_channel()
    await plugin.on_init(plugin.pm, {"agent": {
        "forks": {"enabled": True},
        "mid_conversation_system_role": "user",
    }})
    plugin._tool_schemas = TOOL_SCHEMAS
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("fork says hi"))
    llm, _ = await start_llm_plugin(
        monkeypatch, {"main": MAIN_PROFILE}, FORK_STAGES, pm=plugin.pm
    )
    llm._clients["main"] = client
    plugin._client = client
    plugin._mark_started()
    channel.conversation = _tool_window()
    yield plugin, channel, client
    await db.close()


MAPPED_LLM = {
    "main": MAIN_PROFILE,
    # max_tokens lives in a base that forks extends: R8 counts inherited keys.
    "capped": {**MAIN_PROFILE, "extra_body": {"max_tokens": 4096}},
    "forks": {
        "extends": "capped",
        "base_url": "http://forks.test/v1",
        "model": "forks-m",
        "extra_body": {"reasoning_budget_tokens": 8192},
    },
    "stages": {"consolidation": "forks"},
}


@pytest.fixture
async def mapped_agent(monkeypatch):
    """(agent, channel, posts, recorder): real clients over a recording HTTP
    session, stage consolidation mapped to profile ``forks``, appraisal not."""
    plugin, channel, db = await build_plugin_and_channel()
    await plugin.on_init(plugin.pm, {"agent": {"forks": {"enabled": True}}})
    plugin._tool_schemas = TOOL_SCHEMAS
    recorder = LLMResponseRecorder()
    plugin.pm.register(recorder, name="recorder")
    llm, posts = await start_llm_plugin(monkeypatch, MAPPED_LLM, FORK_STAGES, pm=plugin.pm)
    plugin._client = llm.get_client()
    plugin._mark_started()
    channel.conversation = _tool_window()
    yield plugin, channel, posts, recorder
    await llm.on_stop()
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
    result = await plugin.fork(
        channel, INSTRUCTION, stage="consolidation", cut=2, extra_body={"response_format": fmt}
    )

    messages = client.chat.call_args.args[0]
    full = channel.conversation.build_prompt(frozenset({"zz"}), mid_conversation_system_role="user")
    assert messages == full[:3] + [{"role": "user", "content": INSTRUCTION}]
    kwargs = client.chat.call_args.kwargs
    assert kwargs["tools"] == TOOL_SCHEMAS
    # Inference overrides ride along; plugin tunables don't; fork keys win.
    # An unmapped fork sends no reasoning budget (R9).
    assert kwargs["extra_body"] == {
        "temperature": 0.3, "tool_choice": "none", "response_format": fmt,
    }
    assert result.text == "fork says hi"
    assert result.usage["prompt_tokens"] == 100
    assert seen["mode"] == "fork" and "mode" not in get_attribution()
    # Nothing mutated, persisted, or sent.
    assert channel.conversation.messages == before
    assert recorder.events == []
    assert sent(plugin.pm, OutboundMessage) == []


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
        await plugin.fork(channel, INSTRUCTION, stage="consolidation")
    assert channel.pending_tool_call_ids == set()


@pytest.mark.parametrize("text", [
    '{"summary": "I helped", "topic_tags": []}', '{"objections": []}',
    "We debugged the name and arguments of a function.",
    "<functional> components came up; I summarized them.",
])
async def test_ordinary_replies_are_not_refused(forking_agent, text):
    plugin, channel, client = forking_agent
    client.chat = AsyncMock(return_value=_make_text_response(text))
    assert (await plugin.fork(channel, INSTRUCTION, stage="consolidation")).text == text


@pytest.mark.parametrize("missing", ["conversation", "client"])
async def test_fork_unavailable(forking_agent, missing):
    plugin, channel, client = forking_agent
    if missing == "conversation":
        channel.conversation = None
    else:
        plugin.pm.get_plugin("llm")._clients.clear()
    with pytest.raises(ForkUnavailable, match="consolidation"):
        await plugin.fork(channel, INSTRUCTION, stage="consolidation")


async def test_fork_config_reloads_and_unmapped_fork_sends_no_budget(forking_agent):
    plugin, channel, client = forking_agent
    await plugin.on_config_reload({"agent": {"forks": {"enabled": True, "stages": ["appraisal"]}}})
    assert plugin.fork_enabled("appraisal") and not plugin.fork_enabled("compaction")
    await plugin.fork(channel, INSTRUCTION, stage="appraisal")
    assert "reasoning_budget_tokens" not in client.chat.call_args.kwargs["extra_body"]


async def test_fork_requires_a_stage(forking_agent):
    """R7: stage is a required keyword; a missing one fails loudly instead of
    routing to the main profile."""
    plugin, channel, client = forking_agent
    with pytest.raises(TypeError, match="stage"):
        await plugin.fork(channel, INSTRUCTION)


async def test_usage_log_mode_column_upgrades_and_records(db):
    from corvidae.hooks import create_plugin_manager
    from corvidae.metrics import USAGE_LOG_DDL, UsageLogPlugin
    from corvidae.persistence import PersistencePlugin

    await db.execute(USAGE_LOG_DDL.replace("    mode TEXT,\n", ""))
    await db.execute("INSERT INTO usage_log (ts, request_id, profile, model) VALUES (1, 'old', 'main', 'm')")
    await db.commit()
    pm = create_plugin_manager()
    persistence = PersistencePlugin()
    persistence.db = db
    pm.register(persistence, name="persistence")
    pm.register(UsageLogPlugin(pm), name="usage_log")
    for rid, attribution in [("f", {"mode": "fork"}), ("s", {"mode": "standalone"}), ("t", {})]:
        await pm.ahook.on_llm_response(profile="main", model="m", request_id=rid, usage={},
                                       latency_ms=1.0, attribution=attribution, error=None,
                                       timings=None, started_at=1000.0, total_ms=1.0,
                                       attempts=1, message_count=1, tool_count=0)
    async with db.execute("SELECT request_id, mode FROM usage_log ORDER BY id") as cursor:
        assert await cursor.fetchall() == [("old", None), ("f", "fork"), ("s", "standalone"), ("t", None)]


async def test_fork_result_carries_request_model(forking_agent):
    """Issue #299 R1: ForkResult.model is what the fork was sent with, taken from
    the client's request_model for the fork's body, not a constant."""
    plugin, channel, client = forking_agent
    # Echo the body's model override, else a fixed default.
    client.request_model = lambda extra_body=None: (extra_body or {}).get("model", "m")

    overridden = await plugin.fork(
        channel, INSTRUCTION, stage="consolidation", extra_body={"model": "fork-m"}
    )
    plain = await plugin.fork(channel, INSTRUCTION, stage="consolidation")

    assert overridden.model == "fork-m"
    assert plain.model == "m"


async def test_fork_model_is_the_client_the_call_was_sent_with(forking_agent):
    """Issue #299 R1: a config reload that swaps the client mid-call must not
    change the model the record is stamped with."""
    plugin, channel, client = forking_agent
    client.request_model = lambda extra_body=None: "original-m"
    replacement = MagicMock()
    replacement.request_model = lambda extra_body=None: "replacement-m"

    async def chat(messages, **kwargs):
        # The profile's client is replaced while the request is in flight.
        plugin.pm.get_plugin("llm")._clients["main"] = replacement
        return _make_text_response("fork says hi")

    client.chat = AsyncMock(side_effect=chat)
    result = await plugin.fork(channel, INSTRUCTION, stage="consolidation")

    assert result.model == "original-m"


async def test_fork_survives_client_cleared_mid_call(forking_agent):
    """Issue #299 R1: a plugin stop that clears the client mid-call must not
    make a successful fork raise; it keeps the original model."""
    plugin, channel, client = forking_agent
    client.request_model = lambda extra_body=None: "original-m"

    async def chat(messages, **kwargs):
        # A stop clears the clients while the request is in flight.
        plugin.pm.get_plugin("llm")._clients.clear()
        return _make_text_response("fork says hi")

    client.chat = AsyncMock(side_effect=chat)
    result = await plugin.fork(channel, INSTRUCTION, stage="consolidation")

    assert result.model == "original-m"


# ---------------------------------------------------------------------------
# Stage profiles (#399): R7, R8, R11
# ---------------------------------------------------------------------------


async def test_mapped_stage_fork_goes_to_the_mapped_profile(mapped_agent):
    """R7: a mapped stage's fork is sent to its profile's endpoint and model;
    an unmapped stage's fork goes to the default chat profile. The render is
    the main turn's: same messages and tools."""
    plugin, channel, posts, _ = mapped_agent

    await plugin.fork(channel, INSTRUCTION, stage="consolidation", cut=2)
    await plugin.fork(channel, INSTRUCTION, stage="appraisal", cut=2)

    mapped, unmapped = posts
    assert mapped["url"] == "http://forks.test/v1/chat/completions"
    assert mapped["json"]["model"] == "forks-m"
    assert unmapped["url"] == "http://main.test/v1/chat/completions"
    assert unmapped["json"]["model"] == "main-m"
    full = channel.conversation.build_prompt(
        frozenset(), mid_conversation_system_role=plugin._mid_conversation_system_role
    )
    expected = full[:3] + [{"role": "user", "content": INSTRUCTION}]
    for post in posts:
        assert post["json"]["messages"] == expected
        assert post["json"]["tools"] == TOOL_SCHEMAS


async def test_unmapped_fork_request_body_has_no_reasoning_budget(mapped_agent):
    """R9: nothing injects a budget; an unmapped fork's profile sets none."""
    plugin, channel, posts, _ = mapped_agent

    await plugin.fork(channel, INSTRUCTION, stage="appraisal")

    assert "reasoning_budget_tokens" not in posts[-1]["json"]


async def test_profile_max_tokens_beats_channel_override_on_a_fork(mapped_agent):
    """R8: a channel's runtime max_tokens 200 against the stage profile's
    extra_body max_tokens 4096 sends 4096; an override the profile doesn't set
    still applies."""
    plugin, channel, posts, _ = mapped_agent
    channel.runtime_overrides = {"max_tokens": 200, "temperature": 0.3}

    await plugin.fork(channel, INSTRUCTION, stage="consolidation")

    sent_body = posts[-1]["json"]
    assert sent_body["max_tokens"] == 4096
    assert sent_body["temperature"] == 0.3
    assert sent_body["reasoning_budget_tokens"] == 8192


async def test_stage_schema_and_tool_choice_still_win_on_a_fork(mapped_agent):
    """R8: with a stage profile in play, the stage's response_format and
    tool_choice none are still what is sent."""
    plugin, channel, posts, _ = mapped_agent
    channel.runtime_overrides = {"tool_choice": "required"}
    fmt = {"type": "json_schema"}

    await plugin.fork(channel, INSTRUCTION, stage="consolidation", extra_body={"response_format": fmt})

    assert posts[-1]["json"]["response_format"] == fmt
    assert posts[-1]["json"]["tool_choice"] == "none"


async def test_fork_attribution_names_the_mapped_profile(mapped_agent):
    """R11: a fork's on_llm_response names the stage's profile."""
    plugin, channel, _, recorder = mapped_agent

    await plugin.fork(channel, INSTRUCTION, stage="consolidation")
    await plugin.fork(channel, INSTRUCTION, stage="appraisal")

    assert [r["profile"] for r in recorder.responses] == ["forks", "main"]


async def test_fork_renders_a_snapshot_not_the_live_window(forking_agent):
    """#628: a fork given a snapshot renders the window as it stood when the
    snapshot was taken, even after the live window has changed (a compaction
    replaced it); the live window is not touched."""
    from corvidae.fork import snapshot_window

    plugin, channel, client = forking_agent
    channel.pending_tool_call_ids = {"zz"}
    expected = channel.conversation.build_prompt(
        frozenset({"zz"}), mid_conversation_system_role="user"
    )[:3] + [{"role": "user", "content": INSTRUCTION}]
    snapshot = snapshot_window(channel.conversation, frozenset(channel.pending_tool_call_ids))
    # The live window changes after the snapshot: a compaction replaced it.
    channel.conversation.messages[:] = [
        {"role": "assistant", "content": "[Summary] it was about wifi"}
    ]
    channel.pending_tool_call_ids = set()
    after = [dict(m) for m in channel.conversation.messages]

    await plugin.fork(channel, INSTRUCTION, stage="consolidation", cut=2, snapshot=snapshot)

    assert client.chat.call_args.args[0] == expected
    assert channel.conversation.messages == after
