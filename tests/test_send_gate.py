"""RED tests for WP2.8: the should_send_response hook, MessageType.UNSENT and
its four firing sites (plans/implementation/phase-2.md WP2.8; docs/prd.md
§3.7 "Output gate and the right to silence").

Kernel deviations agreed on #131: the hook carries ``meta`` instead of an
``origin`` param (origin is a plugin-side convention), the stored type is
``MessageType.UNSENT`` (the kernel vocabulary rule), and core appends the
not-sent marker itself (the funnel is cognition).

Interface pinned here:
  hookspec  should_send_response(channel, text, emission, trace_id, meta)
            -> bool | None, reject-wins; emission in
            {"final", "progress", "thinking", "error"}
  corvidae.context.MessageType.UNSENT == "unsent"
  corvidae.agent.UNSENT_MARKER_LABEL — the label the marker row opens with
"""

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pluggy
import pytest

from corvidae.agent import MAX_TURNS_FALLBACK_MESSAGE
from corvidae.channel import ChannelConfig
from corvidae.context import MessageType, unsent_marker_emission
from corvidae.hooks import _check_hook_arg_binding, create_plugin_manager, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress, OutboundThinking
from corvidae.trace import mint_trace_id
from helpers import build_plugin_and_channel, drain, drain_task_queue, sent_texts
from llm_response_fixtures import (
    _make_mixed_response,
    _make_text_response,
    _make_tool_call,
    _make_tool_call_response,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class SpyGate:
    """A should_send_response impl that records every firing and vetoes the
    emissions named in ``veto``."""

    def __init__(self, veto=(), raises=False):
        self.veto = set(veto)
        self.raises = raises
        self.calls: list[dict] = []

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, meta):
        self.calls.append({
            "channel": channel, "text": text, "emission": emission,
            "trace_id": trace_id, "meta": dict(meta),
        })
        if self.raises:
            raise RuntimeError("gate exploded")
        if emission in self.veto:
            return False
        return None

    def emissions(self) -> list[str]:
        return [c["emission"] for c in self.calls]


@pytest.fixture
async def env(request):
    """(plugin, channel, db) with TaskPlugin teardown; optional ChannelConfig
    via indirect param."""
    params = getattr(request, "param", {}) or {}
    plugin, channel, db = await build_plugin_and_channel(
        channel_config=params.get("channel_config"),
    )
    yield plugin, channel, db
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


def _client(*responses):
    client = MagicMock()
    client.chat = AsyncMock(side_effect=list(responses))
    return client


async def _rows(db, channel_id):
    """(message dict, message_type) for every message_log row, in order."""
    async with db.execute(
        "SELECT message, message_type FROM message_log WHERE channel_id = ? ORDER BY id",
        (channel_id,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [(json.loads(m), t) for m, t in rows]


def _markers(messages):
    """Window CONTEXT messages that are not-sent markers."""
    return [
        m for m in messages
        if m.get("_message_type") == MessageType.CONTEXT
        and unsent_marker_emission(m.get("content")) is not None
    ]


async def _say(plugin, channel, text="hello"):
    await plugin.on_message(channel=channel, sender="user", text=text, trace_id=mint_trace_id())
    await drain(plugin, channel)


def _install_tool(plugin, calls):
    async def my_tool(x: str) -> str:
        """A test tool."""
        calls.append(x)
        return f"result:{x}"

    plugin._tools = {"my_tool": my_tool}


# ---------------------------------------------------------------------------
# Hookspec shape
# ---------------------------------------------------------------------------


class TestHookspec:
    def test_signature(self):
        import inspect

        from corvidae.hooks import AgentSpec

        params = [p for p in inspect.signature(AgentSpec.should_send_response).parameters if p != "self"]
        assert params == ["channel", "text", "emission", "trace_id", "meta"]

    def test_unsent_message_type(self):
        assert MessageType.UNSENT.value == "unsent"

    def test_impl_defaulting_meta_is_flagged(self):
        pm = create_plugin_manager()

        class Defaulting:
            @hookimpl
            async def should_send_response(self, channel, text, emission, trace_id, meta=None):
                return None

        pm.register(Defaulting(), name="defaulting_gate")
        with pytest.raises(pluggy.PluginValidationError) as excinfo:
            _check_hook_arg_binding(pm)
        assert "meta" in str(excinfo.value)
        assert "defaulting_gate" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Persistence-controlling firing (emission="final")
# ---------------------------------------------------------------------------


class TestFinalVeto:
    async def test_vetoed_final_is_stored_unsent_and_never_sent(self, env):
        plugin, channel, db = env
        gate = SpyGate(veto={"final"})
        plugin.pm.register(gate, name="spy_gate")
        plugin._client = _client(_make_text_response("the secret answer"))

        await _say(plugin, channel)

        # Transports never see it.
        assert sent_texts(plugin.pm, OutboundMessage) == []
        # The row is stored, tagged UNSENT.
        rows = await _rows(db, channel.id)
        assistant = [(m, t) for m, t in rows if m.get("role") == "assistant"]
        assert assistant == [({"role": "assistant", "content": "the secret answer"}, "unsent")]
        # The window holds the text, tagged, followed by a marker quoting it.
        conv = channel.conversation
        tagged = [m for m in conv.messages if m.get("role") == "assistant"]
        assert tagged[0]["content"] == "the secret answer"
        assert tagged[0]["_message_type"] == MessageType.UNSENT
        markers = _markers(conv.messages)
        assert len(markers) == 1
        assert unsent_marker_emission(markers[0]["content"]) == "final"
        assert "the secret answer" not in markers[0]["content"]
        assert conv.messages[-1] is markers[0]
        assert conv.messages[-2] is tagged[0]
        # The marker row is persisted too (it must survive restart).
        assert any(
            t == "context" and unsent_marker_emission(m["content"]) == "final"
            for m, t in rows
        )
        # The gate saw exactly what would have been sent.
        assert gate.calls[-1]["emission"] == "final"
        assert gate.calls[-1]["text"] == "the secret answer"

    async def test_vetoed_final_still_fires_on_agent_response(self, env):
        """Critique stays eligible for withheld responses: on_agent_response
        fires on a veto exactly as on a pass."""
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(_make_text_response("kept to myself"))

        await _say(plugin, channel)

        plugin.pm.ahook.on_agent_response.assert_awaited_once()
        assert plugin.pm.ahook.on_agent_response.await_args.kwargs["response_text"] == "kept to myself"

    async def test_unsent_row_and_marker_reload_into_window(self, env):
        """Restart: the reloaded window holds the UNSENT row, tagged, and the
        marker — window identity across restarts."""
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(_make_text_response("the secret answer"))
        await _say(plugin, channel)

        live = [
            {k: v for k, v in m.items() if k != "_db_id"}
            for m in channel.conversation.messages
        ]

        persistence = plugin.pm.get_plugin("persistence")
        reloaded = await persistence.resolve_conversation(channel=channel)
        reloaded_clean = [{k: v for k, v in m.items() if k != "_db_id"} for m in reloaded]
        assert reloaded_clean == live
        unsent = [m for m in reloaded if m["_message_type"] == MessageType.UNSENT]
        assert [m["content"] for m in unsent] == ["the secret answer"]
        assert len(_markers(reloaded)) == 1

    async def test_fresh_agent_after_restart_prompts_with_unsent_row(self, env):
        """A restart in miniature: a new Agent and a new registry over the
        same sessions DB. The next LLM call's prompt carries the withheld
        text as the assistant turn, followed by its marker."""
        from corvidae.agent import Agent
        from corvidae.channel import ChannelConfig, ChannelRegistry

        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(_make_text_response("the secret answer"))
        await _say(plugin, channel)

        pm = plugin.pm
        pm.unregister(plugin)
        # The spy vetoes every final; the restarted daemon runs without it.
        pm.unregister(name="spy_gate")
        registry = ChannelRegistry()
        registry.agent_defaults = plugin._registry.agent_defaults
        fresh = Agent()
        pm.register(fresh, name="agent")
        fresh.pm = pm
        fresh._registry = registry
        client = _client(_make_text_response("I kept it to myself."))
        fresh._client = client
        reborn = registry.get_or_create(channel.transport, channel.scope, config=ChannelConfig())
        assert reborn is not channel and reborn.conversation is None

        await fresh.on_message(channel=reborn, sender="user", text="what did you say?", trace_id=mint_trace_id())
        await drain(fresh, reborn)

        prompt = client.chat.await_args.args[0]
        idx = prompt.index({"role": "assistant", "content": "the secret answer"})
        assert unsent_marker_emission(prompt[idx + 1]["content"]) == "final"
        assert sent_texts(pm, OutboundMessage)[-1] == "I kept it to myself."

    async def test_build_prompt_includes_unsent_row(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(_make_text_response("the secret answer"))
        await _say(plugin, channel)

        prompt = channel.conversation.build_prompt(frozenset())
        assert {"role": "assistant", "content": "the secret answer"} in prompt
        # The prompt never ends on the assistant message (PRD §3.1.6).
        assert prompt[-1]["role"] != "assistant"

    async def test_passed_final_is_sent_and_stored_as_message(self, env):
        plugin, channel, db = env
        gate = SpyGate()
        plugin.pm.register(gate, name="spy_gate")
        plugin._client = _client(_make_text_response("hi there"))

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == ["hi there"]
        rows = await _rows(db, channel.id)
        assert [t for m, t in rows] == ["message", "message"]
        assert _markers(channel.conversation.messages) == []
        assert gate.emissions() == ["final"]

    async def test_gate_receives_trace_id_and_meta(self, env):
        """Origin is not a kernel param; the gate reads it from meta."""
        plugin, channel, db = env
        gate = SpyGate()
        plugin.pm.register(gate, name="spy_gate")
        plugin._client = _client(_make_text_response("reminder done"))

        await plugin.on_notify(
            channel=channel, source="scheduler", text="time to stretch",
            tool_call_id=None, meta={"origin": "reminder"},
        )
        await drain(plugin, channel)

        call = gate.calls[-1]
        assert call["emission"] == "final"
        assert call["trace_id"]
        assert call["meta"]["origin"] == "reminder"
        assert call["channel"] is channel

    async def test_raising_gate_fails_open(self, env, caplog):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(raises=True), name="spy_gate")
        plugin._client = _client(_make_text_response("still delivered"))

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == ["still delivered"]
        rows = await _rows(db, channel.id)
        assert "unsent" not in [t for m, t in rows]
        assert any(
            "should_send_response" in r.getMessage() and r.exc_info
            for r in caplog.records
        )


class TestToolCallsResultNeverFacesFinalGate:
    async def test_dispatching_turn_fires_only_progress(self, env):
        plugin, channel, db = env
        gate = SpyGate()
        plugin.pm.register(gate, name="spy_gate")
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        plugin._client = _client(
            _make_mixed_response("Let me look that up", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
            _make_text_response("Found it"),
        )

        await _say(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)

        # First turn (tool calls dispatch) fired progress only; the final
        # firing belongs to the second turn.
        assert gate.emissions() == ["progress", "final"]
        assert tool_calls == ["a"]


# ---------------------------------------------------------------------------
# Max-turns branch
# ---------------------------------------------------------------------------


class TestMaxTurns:
    @pytest.mark.parametrize(
        "env", [{"channel_config": ChannelConfig(max_turns=0)}], indirect=True,
    )
    async def test_tool_calls_stripped_from_row_and_window(self, env):
        plugin, channel, db = env
        gate = SpyGate()
        plugin.pm.register(gate, name="spy_gate")
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        plugin._client = _client(
            _make_tool_call_response([_make_tool_call("call-1", "my_tool", {"x": "a"})]),
        )

        await _say(plugin, channel)
        await drain_task_queue(plugin)

        rows = await _rows(db, channel.id)
        assistant_rows = [m for m, t in rows if m.get("role") == "assistant"]
        assert len(assistant_rows) == 1
        assert "tool_calls" not in assistant_rows[0]
        window_assistant = [m for m in channel.conversation.messages if m.get("role") == "assistant"]
        assert "tool_calls" not in window_assistant[0]
        # No dispatch, fallback sent, and the fallback is what the gate checked.
        assert tool_calls == []
        assert sent_texts(plugin.pm, OutboundMessage) == [MAX_TURNS_FALLBACK_MESSAGE]
        assert gate.calls[-1]["emission"] == "final"
        assert gate.calls[-1]["text"] == MAX_TURNS_FALLBACK_MESSAGE

    @pytest.mark.parametrize(
        "env", [{"channel_config": ChannelConfig(max_turns=0)}], indirect=True,
    )
    async def test_strip_happens_with_no_gate_registered(self, env):
        plugin, channel, db = env
        plugin._client = _client(
            _make_tool_call_response([_make_tool_call("call-1", "my_tool", {"x": "a"})]),
        )

        await _say(plugin, channel)

        rows = await _rows(db, channel.id)
        assert all("tool_calls" not in m for m, t in rows)
        assert all("tool_calls" not in m for m in channel.conversation.messages)
        # A stripped assistant message is still well-formed for the server.
        prompt = channel.conversation.build_prompt(frozenset())
        assistant = [m for m in prompt if m["role"] == "assistant"]
        assert isinstance(assistant[0]["content"], str)

    @pytest.mark.parametrize(
        "env", [{"channel_config": ChannelConfig(max_turns=0)}], indirect=True,
    )
    async def test_vetoed_max_turns_fallback_is_unsent(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(
            _make_tool_call_response([_make_tool_call("call-1", "my_tool", {"x": "a"})]),
        )

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == []
        rows = await _rows(db, channel.id)
        assistant = [(m, t) for m, t in rows if m.get("role") == "assistant"]
        assert [t for _m, t in assistant] == ["unsent"]
        assert "tool_calls" not in assistant[0][0]
        assert len(_markers(channel.conversation.messages)) == 1


# ---------------------------------------------------------------------------
# Per-emission firings
# ---------------------------------------------------------------------------


class TestProgressVeto:
    async def test_progress_suppressed_tools_dispatch_row_intact(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"progress"}), name="spy_gate")
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        plugin._client = _client(
            _make_mixed_response("Let me look that up", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
            _make_text_response("Found it"),
        )

        await _say(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)

        assert sent_texts(plugin.pm, OutboundProgress) == []
        assert tool_calls == ["a"]
        assert sent_texts(plugin.pm, OutboundMessage) == ["Found it"]
        rows = await _rows(db, channel.id)
        first_assistant = next((m, t) for m, t in rows if m.get("role") == "assistant")
        assert first_assistant[1] == "message"
        assert first_assistant[0]["content"] == "Let me look that up"
        assert first_assistant[0]["tool_calls"][0]["id"] == "call-1"
        markers = _markers(channel.conversation.messages)
        assert len(markers) == 1
        assert "Let me look that up" in markers[0]["content"]
        # The tool call and its result still pair in the prompt.
        prompt = channel.conversation.build_prompt(frozenset())
        idx = next(i for i, m in enumerate(prompt) if m.get("tool_calls"))
        assert prompt[idx + 1]["role"] == "tool"
        assert prompt[idx + 1]["tool_call_id"] == "call-1"


class TestThinkingVeto:
    async def test_thinking_suppressed_message_sent(self, env):
        plugin, channel, db = env
        gate = SpyGate(veto={"thinking"})
        plugin.pm.register(gate, name="spy_gate")
        plugin._client = _client(_make_text_response("answer", reasoning="pondering deeply"))

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundThinking) == []
        assert sent_texts(plugin.pm, OutboundMessage) == ["answer"]
        rows = await _rows(db, channel.id)
        assistant = [(m, t) for m, t in rows if m.get("role") == "assistant"]
        assert [t for _m, t in assistant] == ["message"]
        markers = _markers(channel.conversation.messages)
        assert len(markers) == 1
        assert unsent_marker_emission(markers[0]["content"]) == "thinking"
        # Reasoning must not outlive reasoning_history: drop in a marker.
        assert "pondering deeply" not in markers[0]["content"]
        assert "thinking" in gate.emissions()


class TestErrorVeto:
    async def test_error_fallback_suppressed_nothing_persisted(self, env):
        plugin, channel, db = env
        gate = SpyGate(veto={"error"})
        plugin.pm.register(gate, name="spy_gate")
        client = MagicMock()
        client.chat = AsyncMock(side_effect=RuntimeError("llm down"))
        plugin._client = client

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == []
        assert gate.emissions() == ["error"]
        rows = await _rows(db, channel.id)
        assert [m.get("role") for m, t in rows] == ["user"]
        assert _markers(channel.conversation.messages) == []

    async def test_error_fallback_passes_through_gate(self, env):
        plugin, channel, db = env
        gate = SpyGate()
        plugin.pm.register(gate, name="spy_gate")
        client = MagicMock()
        client.chat = AsyncMock(side_effect=RuntimeError("llm down"))
        plugin._client = client

        await _say(plugin, channel)

        texts = sent_texts(plugin.pm, OutboundMessage)
        assert len(texts) == 1
        assert gate.calls[0]["text"] == texts[0]
        assert gate.calls[0]["trace_id"]


# ---------------------------------------------------------------------------
# No gate registered: today's behavior
# ---------------------------------------------------------------------------


class TestNoGate:
    async def test_final_progress_thinking_all_sent(self, env):
        plugin, channel, db = env
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        plugin._client = _client(
            _make_mixed_response("Let me look that up", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
            _make_text_response("Found it", reasoning="hmm"),
        )

        await _say(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)

        assert sent_texts(plugin.pm, OutboundProgress) == ["Let me look that up"]
        assert sent_texts(plugin.pm, OutboundThinking) == ["hmm"]
        assert sent_texts(plugin.pm, OutboundMessage) == ["Found it"]
        rows = await _rows(db, channel.id)
        assert {t for m, t in rows} == {"message"}
        assert _markers(channel.conversation.messages) == []


class TestReviewGaps:
    async def test_vetoed_final_withholds_its_reasoning(self, env):
        plugin, channel, db = env
        gate = SpyGate(veto={"final"})
        plugin.pm.register(gate, name="spy_gate")
        plugin._client = _client(_make_text_response("the answer", reasoning="the answer is 42"))

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundThinking) == []
        assert sent_texts(plugin.pm, OutboundMessage) == []
        # The thinking gate is not consulted for a withheld response.
        assert gate.emissions() == ["final"]
        assert [unsent_marker_emission(m["content"]) for m in _markers(channel.conversation.messages)] == ["final"]

    async def test_progress_veto_reloads_identically(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"progress", "thinking"}), name="spy_gate")
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        plugin._client = _client(
            _make_mixed_response("Let me look that up", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
            _make_text_response("Found it", reasoning="hmm"),
        )
        await _say(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)

        live = [{k: v for k, v in m.items() if k != "_db_id"} for m in channel.conversation.messages]
        reloaded = await plugin.pm.get_plugin("persistence").resolve_conversation(channel=channel)
        # reasoning_history: drop strips reasoning from the live window only.
        reloaded_clean = [
            {k: v for k, v in m.items() if k not in ("_db_id", "reasoning_content")}
            for m in reloaded
        ]
        assert reloaded_clean == live
        assert [unsent_marker_emission(m["content"]) for m in _markers(reloaded)] == ["progress", "thinking"]

    @pytest.mark.parametrize("emission", ["progress", "thinking", "error"])
    async def test_raising_gate_fails_open_for_every_emission(self, env, emission):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(raises=True), name="spy_gate")
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        if emission == "error":
            client = MagicMock()
            client.chat = AsyncMock(side_effect=RuntimeError("llm down"))
            plugin._client = client
        else:
            plugin._client = _client(
                _make_mixed_response("Let me look that up", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
                _make_text_response("Found it", reasoning="hmm"),
            )
        await _say(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)

        if emission == "progress":
            assert sent_texts(plugin.pm, OutboundProgress) == ["Let me look that up"]
        elif emission == "thinking":
            assert sent_texts(plugin.pm, OutboundThinking) == ["hmm"]
        else:
            assert len(sent_texts(plugin.pm, OutboundMessage)) == 1
        assert _markers(channel.conversation.messages) == []

    async def test_reject_wins_across_gates(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(), name="no_opinion_gate")
        plugin.pm.register(SpyGate(veto={"final"}), name="vetoing_gate")

        class Accepting:
            @hookimpl
            async def should_send_response(self, channel, text, emission, trace_id, meta):
                return True

        plugin.pm.register(Accepting(), name="accepting_gate")
        plugin._client = _client(_make_text_response("the secret answer"))

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == []


# ---------------------------------------------------------------------------
# Replay paths: transports never see an UNSENT row
# ---------------------------------------------------------------------------


class TestReplay:
    def test_web_history_skips_unsent_rows(self):
        from corvidae.channels.web import WebPlugin

        rows = [
            (1, json.dumps({"role": "user", "content": "hi"}), "message"),
            (2, json.dumps({"role": "assistant", "content": "secret"}), "unsent"),
            (3, json.dumps({"role": "system", "content": "marker"}), "context"),
        ]
        frames = WebPlugin()._rows_to_frames(rows)
        assert [f.get("text") for f in frames] == ["hi"]

    def test_web_history_holds_back_suppressed_progress_and_thinking(self):
        from corvidae.channels.web import WebPlugin
        from corvidae.context import unsent_marker_text

        call = {"id": "call-1", "function": {"name": "my_tool", "arguments": "{}"}}
        rows = [
            (1, json.dumps({"role": "user", "content": "hi"}), "message"),
            (2, json.dumps({
                "role": "assistant", "content": "Let me look", "tool_calls": [call],
                "reasoning_content": "private plan",
            }), "message"),
            (3, json.dumps({"role": "system", "content": unsent_marker_text("thinking", "private plan")}), "context"),
            (4, json.dumps({"role": "system", "content": unsent_marker_text("progress", "Let me look")}), "context"),
            (5, json.dumps({"role": "tool", "tool_call_id": "call-1", "content": "r"}), "message"),
            (6, json.dumps({"role": "assistant", "content": "Found it"}), "message"),
        ]
        frames = WebPlugin()._rows_to_frames(rows)
        types = [f["type"] for f in frames]
        assert "thinking" not in types
        assert "progress" not in types
        assert [f.get("text") for f in frames if f["type"] in ("user", "message")] == ["hi", "Found it"]

    def test_acp_markers_after_reads_the_whole_context_run(self):
        from corvidae.channels.acp import _markers_after
        from corvidae.context import unsent_marker_text

        history = [
            {"role": "assistant", "content": "Let me look", "_message_type": MessageType.MESSAGE},
            {"role": "system", "content": unsent_marker_text("thinking", "x"), "_message_type": MessageType.CONTEXT},
            {"role": "system", "content": unsent_marker_text("progress", "Let me look"), "_message_type": MessageType.CONTEXT},
            {"role": "tool", "content": "r", "_message_type": MessageType.MESSAGE},
            {"role": "system", "content": unsent_marker_text("final", "y"), "_message_type": MessageType.CONTEXT},
        ]
        assert _markers_after(history, 0) == {"thinking", "progress"}

    async def test_acp_load_session_skips_unsent_rows(self, tmp_path):
        from acp import PROTOCOL_VERSION

        from corvidae.channel import ChannelRegistry
        from corvidae.channels.acp import AcpPlugin, CorvidaeAcpAgent
        from corvidae.persistence import PersistencePlugin, init_db

        pm = create_plugin_manager()
        registry = ChannelRegistry({"system_prompt": "x", "max_context_tokens": 8000})
        pm.register(registry, name="registry")
        persistence = PersistencePlugin(pm)
        db = await aiosqlite.connect(":memory:")
        await init_db(db)
        persistence.db = db
        pm.register(persistence, name="persistence")
        plugin = AcpPlugin()
        pm.register(plugin, name="acp")
        await plugin.on_init(pm=pm, config={"_acp_mode": True})
        agent = CorvidaeAcpAgent(
            agent_info={"name": "corvidae", "title": "Corvidae", "version": "0.0.1"},
            plugin=plugin,
        )
        conn = MagicMock()
        conn.session_update = AsyncMock()
        plugin._conn = conn
        agent.on_connect(conn)
        try:
            await agent.initialize(protocol_version=PROTOCOL_VERSION)
            session_id = "unsentreplay1"
            from corvidae.context import unsent_marker_text

            for role, text, ts, mtype in (
                ("user", "visible-user", 1.0, "message"),
                ("assistant", "never-sent", 2.0, "unsent"),
                ("system", unsent_marker_text("final", "never-sent"), 2.5, "context"),
                ("assistant", "suppressed-progress", 3.0, "message"),
                ("system", unsent_marker_text("progress", "suppressed-progress"), 3.5, "context"),
                ("assistant", "visible-answer", 4.0, "message"),
            ):
                await db.execute(
                    "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                    "VALUES (?, ?, ?, ?)",
                    (f"acp:{session_id}", json.dumps({"role": role, "content": text}), ts, mtype),
                )
            await db.commit()

            await agent.load_session(cwd=str(tmp_path), session_id=session_id)

            texts = [c.kwargs["update"].content.text for c in conn.session_update.await_args_list]
            assert texts == ["visible-user", "visible-answer"]
        finally:
            await db.close()
