"""Tests for the fork primitive (GitHub issue #51, plans/forks.md Part 1 §1).

A fork is one LLM call whose prompt is a prefix of the channel's rendered
window plus one appended ``user`` instruction. It must render the same
prefix the main turn renders (same client, tools, build_prompt arguments,
and runtime-override filter) so llama-server reuses the cached KV prefix.
It mutates nothing, persists nothing, fires no hooks, and dispatches no
tools.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.attribution import get_attribution
from corvidae.context import ContextWindow, MessageType
from corvidae.fork import (
    ForkRefused,
    ForkResult,
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


def _call(call_id: str, name: str = "shell") -> dict:
    """An OpenAI-shaped tool call."""
    return {"id": call_id, "function": {"name": name, "arguments": "{}"}}


def _tool_window() -> ContextWindow:
    """A window with a CONTEXT row, a summary head, and a two-call tool cycle
    whose results are interleaved with a user message."""
    conv = ContextWindow("test:scope")
    conv.system_prompt = "You are Lex."
    conv.messages = [
        {"role": "assistant", "content": "[Summary of earlier conversation]\nold",
         "_message_type": MessageType.SUMMARY},
        {"role": "user", "content": "run two things", "_db_id": 10,
         "_message_type": MessageType.MESSAGE},
        {"role": "assistant", "content": "", "tool_calls": [_call("a"), _call("b")],
         "_db_id": 11, "_message_type": MessageType.MESSAGE},
        {"role": "tool", "tool_call_id": "a", "content": "A", "_db_id": 12,
         "_message_type": MessageType.MESSAGE},
        {"role": "user", "content": "still there?", "_db_id": 13,
         "_message_type": MessageType.MESSAGE},
        {"role": "tool", "tool_call_id": "b", "content": "B", "_db_id": 14,
         "_message_type": MessageType.MESSAGE},
        {"role": "assistant", "content": "both done", "_db_id": 15,
         "_message_type": MessageType.MESSAGE},
        {"role": "system", "content": "[CONTEXT]\nfact", "_db_id": 16,
         "_message_type": MessageType.CONTEXT},
        {"role": "user", "content": "thanks", "_db_id": 17,
         "_message_type": MessageType.MESSAGE},
    ]
    return conv


# ---------------------------------------------------------------------------
# build_fork_prompt — prefix parity
# ---------------------------------------------------------------------------


class TestBuildForkPrompt:
    @pytest.mark.parametrize("mid_role", ["system", "user"])
    def test_full_window_fork_is_main_prompt_plus_instruction(self, mid_role):
        conv = _tool_window()
        pending = frozenset()
        prompt = build_fork_prompt(conv, len(conv.messages), INSTRUCTION, pending, mid_role)
        assert prompt[:-1] == conv.build_prompt(pending, mid_conversation_system_role=mid_role)
        assert prompt[-1] == {"role": "user", "content": INSTRUCTION}

    def test_cut_none_means_full_window(self):
        conv = _tool_window()
        prompt = build_fork_prompt(conv, None, INSTRUCTION, frozenset(), "system")
        assert prompt[:-1] == conv.build_prompt(frozenset())

    @pytest.mark.parametrize("cut", [1, 2, 6, 7, 8])
    def test_prefix_fork_at_clean_boundary_is_prefix_of_main_prompt(self, cut):
        """At a cut no tool cycle straddles, the prefix render is exactly the
        start of the full render: the server already holds it."""
        conv = _tool_window()
        full = conv.build_prompt(frozenset())
        prompt = build_fork_prompt(conv, cut, INSTRUCTION, frozenset(), "system")
        assert prompt[:-1] == full[: len(prompt) - 1]
        assert prompt[-1] == {"role": "user", "content": INSTRUCTION}

    def test_does_not_mutate_window(self):
        conv = _tool_window()
        before = [dict(m) for m in conv.messages]
        build_fork_prompt(conv, 4, INSTRUCTION, frozenset(), "user")
        assert conv.messages == before

    def test_internal_tags_never_reach_prompt(self):
        conv = _tool_window()
        prompt = build_fork_prompt(conv, None, INSTRUCTION, frozenset(), "system")
        assert not any(k.startswith("_") for m in prompt for k in m)

    def test_cut_out_of_range_raises(self):
        conv = _tool_window()
        with pytest.raises(ValueError):
            build_fork_prompt(conv, len(conv.messages) + 1, INSTRUCTION, frozenset(), "system")
        with pytest.raises(ValueError):
            build_fork_prompt(conv, -1, INSTRUCTION, frozenset(), "system")


# ---------------------------------------------------------------------------
# snap_cut_to_tool_boundary
# ---------------------------------------------------------------------------


class TestSnapCut:
    def test_clean_cut_is_unchanged(self):
        conv = _tool_window()
        for cut in (0, 1, 2, 6, 7, 8, 9):
            assert snap_cut_to_tool_boundary(conv.messages, cut) == cut

    @pytest.mark.parametrize("cut", [3, 4, 5])
    def test_cut_inside_a_tool_cycle_backs_up_to_the_call(self, cut):
        """Cuts between the call (index 2) and its last result (index 5)
        back up to the call itself."""
        conv = _tool_window()
        assert snap_cut_to_tool_boundary(conv.messages, cut) == 2

    def test_nested_cycles_back_up_to_a_fixpoint(self):
        """Backing up past one call can land inside an earlier cycle whose
        result sits between the two; the snap keeps going."""
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_call("x")]},
            {"role": "assistant", "content": "", "tool_calls": [_call("y")]},
            {"role": "tool", "tool_call_id": "x", "content": "X"},
            {"role": "tool", "tool_call_id": "y", "content": "Y"},
            {"role": "assistant", "content": "done"},
        ]
        # cut=4: call y (index 2) straddles → back up to 2; call x (index 1)
        # has its result at 3 ≥ 2 → back up to 1.
        assert snap_cut_to_tool_boundary(messages, 4) == 1

    @pytest.mark.parametrize("cut,expected", [(3, 6), (4, 6), (5, 6), (2, 2), (7, 7)])
    def test_forward_snap_passes_the_last_result(self, cut, expected):
        conv = _tool_window()
        assert snap_cut_forward(conv.messages, cut) == expected

    def test_forward_snap_chains_to_a_fixpoint(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [_call("x")]},
            {"role": "assistant", "content": "", "tool_calls": [_call("y")]},
            {"role": "tool", "tool_call_id": "x", "content": "X"},
            {"role": "user", "content": "hm"},
            {"role": "tool", "tool_call_id": "y", "content": "Y"},
            {"role": "user", "content": "done?"},
        ]
        # cut=1: x straddles → 3; then y (index 1) straddles → 5.
        assert snap_cut_forward(messages, 1) == 5

    def test_unanswered_call_does_not_straddle(self):
        """A call with no result anywhere renders in place either way."""
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_call("x")]},
            {"role": "user", "content": "hello?"},
        ]
        assert snap_cut_to_tool_boundary(messages, 2) == 2

    def test_snapped_cut_restores_prefix_parity(self):
        conv = _tool_window()
        full = conv.build_prompt(frozenset())
        cut = snap_cut_to_tool_boundary(conv.messages, 4)
        prompt = build_fork_prompt(conv, cut, INSTRUCTION, frozenset(), "system")
        assert prompt[:-1] == full[: len(prompt) - 1]


# ---------------------------------------------------------------------------
# parse_forks_config
# ---------------------------------------------------------------------------


class TestParseForksConfig:
    def test_defaults_are_off(self):
        cfg = parse_forks_config({})
        assert cfg == ForksConfig()
        assert cfg.enabled is False
        assert cfg.reasoning_budget_tokens == 2048
        assert cfg.stages is None

    def test_enabled_with_budget_and_stages(self):
        cfg = parse_forks_config({"forks": {
            "enabled": True, "reasoning_budget_tokens": 512, "stages": ["appraisal"],
        }})
        assert cfg.enabled is True
        assert cfg.reasoning_budget_tokens == 512
        assert cfg.stages == frozenset({"appraisal"})

    def test_null_budget_means_server_default(self):
        cfg = parse_forks_config({"forks": {"reasoning_budget_tokens": None}})
        assert cfg.reasoning_budget_tokens is None

    def test_null_stages_means_every_stage(self):
        cfg = parse_forks_config({"forks": {"enabled": True, "stages": None}})
        assert cfg.stage_enabled("compaction")
        assert cfg.stage_enabled("anything")

    def test_stage_enabled_needs_both_flags(self):
        cfg = ForksConfig(enabled=True, stages=frozenset({"critique"}))
        assert cfg.stage_enabled("critique")
        assert not cfg.stage_enabled("compaction")
        assert not ForksConfig(enabled=False).stage_enabled("critique")

    @pytest.mark.parametrize("forks", [
        {"enabled": "yes"},
        {"reasoning_budget_tokens": -5},
        {"reasoning_budget_tokens": 1.5},
        {"reasoning_budget_tokens": True},
        {"stages": ["compaction", 3]},
        {"stages": "compaction"},
        {"bogus": 1},
        "on",
    ])
    def test_invalid_values_raise(self, forks):
        with pytest.raises(ValueError):
            parse_forks_config({"forks": forks})

    def test_validate_agent_config_rejects_bad_forks(self):
        from corvidae.channel import validate_agent_config

        with pytest.raises(ValueError, match="forks"):
            validate_agent_config({"forks": {"enabled": 1}})


# ---------------------------------------------------------------------------
# Agent.fork
# ---------------------------------------------------------------------------


class _EventRecorder:
    """Records on_conversation_event calls (a fork must fire none)."""

    def __init__(self):
        self.events = []

    @hookimpl
    async def on_conversation_event(self, channel, message, message_type):
        self.events.append(message)
        return None


TOOL_SCHEMAS = [{"type": "function", "function": {"name": "shell", "parameters": {}}}]


async def _agent_with_window(forks: dict | None = None):
    """An Agent with a fake main client and a live window on its channel."""
    plugin, channel, db = await build_plugin_and_channel()
    agent_cfg = {"forks": forks} if forks is not None else {}
    await plugin.on_init(plugin.pm, {"agent": agent_cfg})
    plugin._tool_schemas = TOOL_SCHEMAS
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("fork says hi"))
    plugin._client = client
    channel.conversation = _tool_window()
    return plugin, channel, db, client


class TestAgentFork:
    async def test_request_shares_main_prefix_tools_and_overrides(self):
        plugin, channel, db, client = await _agent_with_window(
            {"enabled": True, "reasoning_budget_tokens": 1024}
        )
        try:
            # A non-dotted override is an inference param (sent on both main
            # and fork); a dotted one is a plugin tunable (sent on neither).
            channel.runtime_overrides = {"temperature": 0.3, "memory.k": 4}
            result = await plugin.fork(channel, INSTRUCTION)

            args, kwargs = client.chat.call_args
            messages = args[0]
            assert messages[:-1] == channel.conversation.build_prompt(frozenset())
            assert messages[-1] == {"role": "user", "content": INSTRUCTION}
            assert kwargs["tools"] == TOOL_SCHEMAS
            body = kwargs["extra_body"]
            assert body["tool_choice"] == "none"
            assert body["reasoning_budget_tokens"] == 1024
            assert body["temperature"] == 0.3
            assert "memory.k" not in body
            assert "reasoning_effort" not in body

            assert isinstance(result, ForkResult)
            assert result.text == "fork says hi"
            assert result.message["content"] == "fork says hi"
            assert result.usage["prompt_tokens"] == 100
        finally:
            await db.close()

    async def test_uses_mid_conversation_role_and_pending_ids(self):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        await plugin.on_init(plugin.pm, {"agent": {
            "forks": {"enabled": True}, "mid_conversation_system_role": "user",
        }})
        try:
            channel.pending_tool_call_ids = {"zz"}
            await plugin.fork(channel, INSTRUCTION)
            messages = client.chat.call_args.args[0]
            assert messages[:-1] == channel.conversation.build_prompt(
                frozenset({"zz"}), mid_conversation_system_role="user"
            )
        finally:
            await db.close()

    async def test_cut_forks_from_prefix(self):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        try:
            await plugin.fork(channel, INSTRUCTION, cut=2)
            messages = client.chat.call_args.args[0]
            full = channel.conversation.build_prompt(frozenset())
            assert messages[:-1] == full[:3]
        finally:
            await db.close()

    async def test_fork_keys_win_over_overrides_and_stage_body_merges(self):
        """set_settings can put tool_choice in runtime_overrides; a fork must
        still never be allowed to call tools."""
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        try:
            channel.runtime_overrides = {"tool_choice": "required"}
            fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
            await plugin.fork(channel, INSTRUCTION, extra_body={
                "response_format": fmt, "tool_choice": "auto",
            })
            body = client.chat.call_args.kwargs["extra_body"]
            assert body["tool_choice"] == "none"
            assert body["response_format"] == fmt
        finally:
            await db.close()

    async def test_null_budget_omits_the_field(self):
        plugin, channel, db, client = await _agent_with_window(
            {"enabled": True, "reasoning_budget_tokens": None}
        )
        try:
            await plugin.fork(channel, INSTRUCTION)
            assert "reasoning_budget_tokens" not in client.chat.call_args.kwargs["extra_body"]
        finally:
            await db.close()

    async def test_no_tools_sends_none_like_the_main_turn(self):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        try:
            plugin._tool_schemas = []
            await plugin.fork(channel, INSTRUCTION)
            assert client.chat.call_args.kwargs["tools"] is None
        finally:
            await db.close()

    async def test_window_untouched_and_no_hooks_fire(self):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        recorder = _EventRecorder()
        plugin.pm.register(recorder, name="recorder")
        try:
            before = [dict(m) for m in channel.conversation.messages]
            await plugin.fork(channel, INSTRUCTION)
            assert channel.conversation.messages == before
            assert recorder.events == []
            plugin.pm.ahook.send_message.assert_not_called()
            plugin.pm.ahook.on_agent_response.assert_not_called()
        finally:
            await db.close()

    async def test_tool_calls_in_reply_raise_fork_refused(self):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        client.chat = AsyncMock(return_value=_make_tool_call_response(
            [_make_tool_call("c1", "shell", {"command": "rm -rf /"})]
        ))
        try:
            with pytest.raises(ForkRefused):
                await plugin.fork(channel, INSTRUCTION)
            assert channel.pending_tool_call_ids == set()
        finally:
            await db.close()

    @pytest.mark.parametrize("text", [
        '<tool_call>\n{"name": "shell", "arguments": {"command": "ls"}}\n</tool_call>',
        '  {"name": "shell", "arguments": {"command": "ls"}}',
        "<function=shell><parameter=command>ls</parameter></function>",
    ])
    async def test_tool_call_written_as_text_raises_fork_refused(self, text):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        client.chat = AsyncMock(return_value=_make_text_response(text))
        try:
            with pytest.raises(ForkRefused):
                await plugin.fork(channel, INSTRUCTION)
        finally:
            await db.close()

    @pytest.mark.parametrize("text", [
        '{"summary": "I helped", "topic_tags": [], "participants": []}',
        '{"objections": []}',
        "We were debugging the name and arguments of a function.",
    ])
    async def test_ordinary_replies_are_not_refused(self, text):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        client.chat = AsyncMock(return_value=_make_text_response(text))
        try:
            assert (await plugin.fork(channel, INSTRUCTION)).text == text
        finally:
            await db.close()

    async def test_attribution_carries_mode_fork_during_call_only(self):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        seen = {}

        async def chat(messages, **kwargs):
            seen.update(get_attribution())
            return _make_text_response("ok")

        client.chat = chat
        try:
            await plugin.fork(channel, INSTRUCTION)
            assert seen["mode"] == "fork"
            assert seen["channel_id"] == channel.id
            assert "mode" not in get_attribution()
        finally:
            await db.close()

    async def test_no_window_raises_unavailable(self):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        channel.conversation = None
        try:
            with pytest.raises(ForkUnavailable):
                await plugin.fork(channel, INSTRUCTION)
            client.chat.assert_not_called()
        finally:
            await db.close()

    async def test_no_client_raises_unavailable(self):
        plugin, channel, db, client = await _agent_with_window({"enabled": True})
        plugin._client = None
        try:
            with pytest.raises(ForkUnavailable):
                await plugin.fork(channel, INSTRUCTION)
        finally:
            await db.close()

    async def test_fork_enabled_reflects_config_and_reload(self):
        plugin, channel, db, client = await _agent_with_window()
        try:
            assert not plugin.fork_enabled("appraisal")
            await plugin.on_config_reload({"agent": {"forks": {
                "enabled": True, "stages": ["appraisal"],
            }}})
            assert plugin.fork_enabled("appraisal")
            assert not plugin.fork_enabled("compaction")
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Startup warning: forks run on llm.main
# ---------------------------------------------------------------------------


class TestBackgroundModelWarning:
    def _llm(self, main_model: str, background_model: str | None):
        from corvidae.llm_plugin import LLMPlugin

        llm = LLMPlugin()
        main = MagicMock()
        main.model = main_model
        llm._clients["main"] = main
        if background_model is not None:
            bg = MagicMock()
            bg.model = background_model
            llm._clients["background"] = bg
        return llm

    async def _start(self, forks: dict, background_model: str | None, caplog):
        from corvidae.tool_collection import ToolCollectionPlugin

        plugin, channel, db = await build_plugin_and_channel()
        pm = plugin.pm
        pm.register(self._llm("big", background_model), name="llm")
        from corvidae.tool import ToolRegistry

        tools = ToolCollectionPlugin()
        tools.registry = ToolRegistry()
        pm.register(tools, name="tools")
        await plugin.on_init(pm, {"agent": {"forks": forks}})
        with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
            await plugin.on_start({})
        await db.close()
        return [r for r in caplog.records if "fork" in r.getMessage()]

    async def test_warns_when_enabled_and_background_differs(self, caplog):
        records = await self._start({"enabled": True}, "small", caplog)
        assert len(records) == 1
        assert "small" in records[0].getMessage()

    async def test_quiet_when_disabled(self, caplog):
        assert await self._start({"enabled": False}, "small", caplog) == []

    async def test_quiet_when_background_matches_or_absent(self, caplog):
        assert await self._start({"enabled": True}, "big", caplog) == []
        caplog.clear()
        assert await self._start({"enabled": True}, None, caplog) == []


# ---------------------------------------------------------------------------
# usage_log.mode
# ---------------------------------------------------------------------------


class TestUsageLogMode:
    async def _setup(self, db):
        from corvidae.hooks import create_plugin_manager
        from corvidae.metrics import UsageLogPlugin
        from corvidae.persistence import PersistencePlugin

        pm = create_plugin_manager()
        persistence = PersistencePlugin()
        persistence.db = db
        pm.register(persistence, name="persistence")
        plugin = UsageLogPlugin()
        pm.register(plugin, name="usage_log")
        await plugin.on_init(pm, {})
        return pm

    async def _fire(self, pm, attribution, request_id):
        await pm.ahook.on_llm_response(
            role="main", model="m", request_id=request_id, usage={},
            latency_ms=1.0, attribution=attribution, error=None,
        )

    async def test_mode_written_from_attribution(self, db):
        pm = await self._setup(db)
        await self._fire(pm, {"stage": "appraisal", "mode": "fork"}, "r1")
        await self._fire(pm, {"stage": "appraisal", "mode": "standalone"}, "r2")
        await self._fire(pm, {"stage": "turn"}, "r3")
        async with db.execute(
            "SELECT request_id, mode FROM usage_log ORDER BY id"
        ) as cursor:
            rows = await cursor.fetchall()
        assert rows == [("r1", "fork"), ("r2", "standalone"), ("r3", None)]

    async def test_table_predating_mode_column_is_upgraded(self, db):
        from corvidae.metrics import USAGE_LOG_DDL

        await db.execute(USAGE_LOG_DDL.replace("    mode TEXT,\n", ""))
        await db.execute(
            "INSERT INTO usage_log (ts, request_id, role, model) "
            "VALUES (1.0, 'old', 'main', 'm')"
        )
        await db.commit()
        pm = await self._setup(db)
        await self._fire(pm, {"stage": "critique", "mode": "fork"}, "new")
        async with db.execute(
            "SELECT request_id, mode FROM usage_log ORDER BY id"
        ) as cursor:
            rows = await cursor.fetchall()
        assert rows == [("old", None), ("new", "fork")]
