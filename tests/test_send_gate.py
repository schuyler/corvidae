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

import pluggy
import pytest

from corvidae.agent import MAX_TURNS_FALLBACK_MESSAGE
from corvidae.channel import ChannelConfig
from corvidae.context import MessageType, unsent_marker_emission, unsent_marker_text
from corvidae.hooks import _check_hook_arg_binding, create_plugin_manager, hookimpl
from corvidae.outbound import OutboundMessage, OutboundProgress, OutboundThinking
from corvidae.trace import mint_trace_id
from helpers import acp_agent_env, build_plugin_and_channel, drain, drain_task_queue, sent_texts
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
        plugin._client = _client(_make_text_response("the secret answer", reasoning="the answer is 42"))
        plugin._mark_started()

        await _say(plugin, channel)

        # Transports never see it, nor its reasoning: the thinking gate is
        # not even consulted for a withheld response.
        assert sent_texts(plugin.pm, OutboundMessage) == []
        assert sent_texts(plugin.pm, OutboundThinking) == []
        assert gate.emissions() == ["final"]
        # The gate saw exactly what would have been sent.
        assert gate.calls[-1]["text"] == "the secret answer"
        # The row is stored, tagged UNSENT.
        rows = await _rows(db, channel.id)
        assistant = [(m.get("content"), t) for m, t in rows if m.get("role") == "assistant"]
        assert assistant == [("the secret answer", "unsent")]
        # The window holds the text, tagged, followed by a marker that does
        # not repeat it; the marker row is persisted too.
        conv = channel.conversation
        assert conv.messages[-2]["content"] == "the secret answer"
        assert conv.messages[-2]["_message_type"] == MessageType.UNSENT
        markers = _markers(conv.messages)
        assert markers == [conv.messages[-1]]
        assert unsent_marker_emission(markers[0]["content"]) == "final"
        assert "the secret answer" not in markers[0]["content"]
        assert any(t == "context" and unsent_marker_emission(m["content"]) == "final" for m, t in rows)

    async def test_vetoed_final_still_fires_on_agent_response(self, env):
        """Critique stays eligible for withheld responses: on_agent_response
        fires on a veto exactly as on a pass."""
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(_make_text_response("kept to myself"))
        plugin._mark_started()

        await _say(plugin, channel)

        plugin.pm.ahook.on_agent_response.assert_awaited_once()
        assert plugin.pm.ahook.on_agent_response.await_args.kwargs["response_text"] == "kept to myself"

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
        plugin._mark_started()

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == []

    async def test_gate_receives_trace_id_and_meta(self, env):
        """Origin is not a kernel param; the gate reads it from meta."""
        plugin, channel, db = env
        gate = SpyGate()
        plugin.pm.register(gate, name="spy_gate")
        plugin._client = _client(_make_text_response("reminder done"))
        plugin._mark_started()

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

    async def test_dispatching_turn_never_faces_the_final_gate(self, env):
        plugin, channel, db = env
        gate = SpyGate()
        plugin.pm.register(gate, name="spy_gate")
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        plugin._client = _client(
            _make_mixed_response("Let me look that up", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
            _make_text_response("Found it"),
        )
        plugin._mark_started()

        await _say(plugin, channel)
        await drain_task_queue(plugin)
        await drain(plugin, channel)

        # The first turn (its tool calls dispatch) fired progress only; the
        # final firing belongs to the second turn.
        assert gate.emissions() == ["progress", "final"]
        assert tool_calls == ["a"]


class TestRestart:
    async def test_unsent_row_and_marker_reload_into_window(self, env):
        """Window identity across restarts: the reloaded window equals the
        live one, UNSENT tag and marker included."""
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(_make_text_response("the secret answer"))
        plugin._mark_started()
        await _say(plugin, channel)

        live = [{k: v for k, v in m.items() if k != "_db_id"} for m in channel.conversation.messages]
        reloaded = await plugin.pm.get_plugin("persistence").resolve_conversation(channel=channel)
        assert [{k: v for k, v in m.items() if k != "_db_id"} for m in reloaded] == live

    async def test_fresh_agent_after_restart_prompts_with_unsent_row(self, env):
        """A restart in miniature: a new Agent and a new registry over the
        same sessions DB. The next LLM call's prompt carries the withheld
        text as the assistant turn, followed by its marker."""
        from corvidae.agent import Agent
        from corvidae.channel import ChannelRegistry

        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(_make_text_response("the secret answer"))
        plugin._mark_started()
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
        fresh._mark_started()
        reborn = registry.get_or_create(channel.transport, channel.scope, config=ChannelConfig())
        assert reborn is not channel and reborn.conversation is None

        await fresh.on_message(channel=reborn, sender="user", text="what did you say?", trace_id=mint_trace_id())
        await drain(fresh, reborn)

        prompt = client.chat.await_args.args[0]
        idx = prompt.index({"role": "assistant", "content": "the secret answer"})
        assert unsent_marker_emission(prompt[idx + 1]["content"]) == "final"
        assert sent_texts(pm, OutboundMessage)[-1] == "I kept it to myself."


# ---------------------------------------------------------------------------
# Max-turns branch
# ---------------------------------------------------------------------------


_MAX_TURNS_ZERO = pytest.mark.parametrize(
    "env", [{"channel_config": ChannelConfig(max_turns=0)}], indirect=True,
)


class TestMaxTurns:
    @_MAX_TURNS_ZERO
    async def test_tool_calls_stripped_from_row_and_window(self, env):
        plugin, channel, db = env
        gate = SpyGate()
        plugin.pm.register(gate, name="spy_gate")
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        plugin._client = _client(
            _make_tool_call_response([_make_tool_call("call-1", "my_tool", {"x": "a"})]),
        )
        plugin._mark_started()

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

    @_MAX_TURNS_ZERO
    async def test_strip_happens_with_no_gate_registered(self, env):
        plugin, channel, db = env
        plugin._client = _client(
            _make_tool_call_response([_make_tool_call("call-1", "my_tool", {"x": "a"})]),
        )
        plugin._mark_started()

        await _say(plugin, channel)

        rows = await _rows(db, channel.id)
        assert all("tool_calls" not in m for m, t in rows)
        assert all("tool_calls" not in m for m in channel.conversation.messages)
        # A stripped assistant message is still well-formed for the server.
        prompt = channel.conversation.build_prompt(frozenset())
        assistant = [m for m in prompt if m["role"] == "assistant"]
        assert isinstance(assistant[0]["content"], str)

    @_MAX_TURNS_ZERO
    async def test_vetoed_max_turns_fallback_is_unsent(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"final"}), name="spy_gate")
        plugin._client = _client(
            _make_tool_call_response([_make_tool_call("call-1", "my_tool", {"x": "a"})]),
        )
        plugin._mark_started()

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


def _tool_turn_client():
    """A progress-text turn with one tool call, then a final with reasoning."""
    return _client(
        _make_mixed_response("Let me look that up", [_make_tool_call("call-1", "my_tool", {"x": "a"})]),
        _make_text_response("Found it", reasoning="hmm"),
    )


async def _run_tool_turn(plugin, channel):
    await _say(plugin, channel)
    await drain_task_queue(plugin)
    await drain(plugin, channel)


class TestProgressAndThinkingVetoes:
    async def test_progress_suppressed_tools_dispatch_row_intact(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"progress"}), name="spy_gate")
        tool_calls: list[str] = []
        _install_tool(plugin, tool_calls)
        plugin._client = _tool_turn_client()
        plugin._mark_started()

        await _run_tool_turn(plugin, channel)

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

    async def test_thinking_suppressed_message_sent(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"thinking"}), name="spy_gate")
        plugin._client = _client(_make_text_response("answer", reasoning="pondering deeply"))
        plugin._mark_started()

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundThinking) == []
        assert sent_texts(plugin.pm, OutboundMessage) == ["answer"]
        rows = await _rows(db, channel.id)
        assert [t for m, t in rows if m.get("role") == "assistant"] == ["message"]
        markers = _markers(channel.conversation.messages)
        assert [unsent_marker_emission(m["content"]) for m in markers] == ["thinking"]
        # Reasoning must not outlive reasoning_history: drop in a marker.
        assert "pondering deeply" not in markers[0]["content"]

    async def test_both_vetoes_reload_identically(self, env):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(veto={"progress", "thinking"}), name="spy_gate")
        _install_tool(plugin, [])
        plugin._client = _tool_turn_client()
        plugin._mark_started()
        await _run_tool_turn(plugin, channel)

        live = [{k: v for k, v in m.items() if k != "_db_id"} for m in channel.conversation.messages]
        reloaded = await plugin.pm.get_plugin("persistence").resolve_conversation(channel=channel)
        # reasoning_history: drop strips reasoning from the live window only.
        reloaded_clean = [
            {k: v for k, v in m.items() if k not in ("_db_id", "reasoning_content")}
            for m in reloaded
        ]
        assert reloaded_clean == live
        assert [unsent_marker_emission(m["content"]) for m in _markers(reloaded)] == ["progress", "thinking"]


class TestErrorVeto:
    async def test_error_fallback_suppressed_nothing_persisted(self, env):
        plugin, channel, db = env
        gate = SpyGate(veto={"error"})
        plugin.pm.register(gate, name="spy_gate")
        client = MagicMock()
        client.chat = AsyncMock(side_effect=RuntimeError("llm down"))
        plugin._client = client
        plugin._mark_started()

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == []
        assert gate.emissions() == ["error"]
        assert gate.calls[0]["trace_id"]
        rows = await _rows(db, channel.id)
        assert [m.get("role") for m, t in rows] == ["user"]
        assert _markers(channel.conversation.messages) == []


# ---------------------------------------------------------------------------
# Failure modes and the no-opinion baseline
# ---------------------------------------------------------------------------


class TestFailOpen:
    @pytest.mark.parametrize("emission", ["final", "progress", "thinking", "error"])
    async def test_raising_gate_sends_every_emission(self, env, emission, caplog):
        plugin, channel, db = env
        plugin.pm.register(SpyGate(raises=True), name="spy_gate")
        _install_tool(plugin, [])
        if emission == "error":
            client = MagicMock()
            client.chat = AsyncMock(side_effect=RuntimeError("llm down"))
            plugin._client = client
            plugin._mark_started()
        else:
            plugin._client = _tool_turn_client()
            plugin._mark_started()

        await _run_tool_turn(plugin, channel)

        sent = {
            "final": (OutboundMessage, ["Found it"]),
            "progress": (OutboundProgress, ["Let me look that up"]),
            "thinking": (OutboundThinking, ["hmm"]),
        }
        if emission == "error":
            assert len(sent_texts(plugin.pm, OutboundMessage)) == 1
        else:
            kind, texts = sent[emission]
            assert sent_texts(plugin.pm, kind) == texts
        assert _markers(channel.conversation.messages) == []
        assert "unsent" not in [t for m, t in await _rows(db, channel.id)]
        assert any(
            "should_send_response" in r.getMessage() and emission in r.getMessage() and r.exc_info
            for r in caplog.records
        )


class TestNoOpinion:
    """No gate registered, or one with no opinion: everything is sent and
    stored as today."""

    @pytest.mark.parametrize("with_gate", [False, True], ids=["no-gate", "no-opinion-gate"])
    async def test_final_progress_thinking_all_sent(self, env, with_gate):
        plugin, channel, db = env
        gate = SpyGate()
        if with_gate:
            plugin.pm.register(gate, name="spy_gate")
        _install_tool(plugin, [])
        plugin._client = _tool_turn_client()
        plugin._mark_started()

        await _run_tool_turn(plugin, channel)

        assert sent_texts(plugin.pm, OutboundProgress) == ["Let me look that up"]
        assert sent_texts(plugin.pm, OutboundThinking) == ["hmm"]
        assert sent_texts(plugin.pm, OutboundMessage) == ["Found it"]
        rows = await _rows(db, channel.id)
        assert {t for m, t in rows} == {"message"}
        assert _markers(channel.conversation.messages) == []
        if with_gate:
            assert gate.emissions() == ["progress", "final", "thinking"]


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
        frames = WebPlugin()._rows_to_frames(rows, "web:t")
        assert [f.get("text") for f in frames] == ["hi"]

    def test_web_history_holds_back_suppressed_progress_and_thinking(self):
        from corvidae.channels.web import WebPlugin

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
        frames = WebPlugin()._rows_to_frames(rows, "web:t")
        types = [f["type"] for f in frames]
        assert "thinking" not in types
        assert "progress" not in types
        assert [f.get("text") for f in frames if f["type"] in ("user", "message")] == ["hi", "Found it"]

    def test_acp_markers_after_reads_the_whole_context_run(self):
        from corvidae.channels.acp import _markers_after

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

        async with acp_agent_env({"system_prompt": "x", "max_context_tokens": 8000}) as acp:
            await acp.agent.initialize(protocol_version=PROTOCOL_VERSION)
            session_id = "unsentreplay1"
            for role, text, ts, mtype in (
                ("user", "visible-user", 1.0, "message"),
                ("assistant", "never-sent", 2.0, "unsent"),
                ("system", unsent_marker_text("final", "never-sent"), 2.5, "context"),
                ("assistant", "suppressed-progress", 3.0, "message"),
                ("system", unsent_marker_text("progress", "suppressed-progress"), 3.5, "context"),
                ("assistant", "visible-answer", 4.0, "message"),
            ):
                await acp.db.execute(
                    "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
                    "VALUES (?, ?, ?, ?)",
                    (f"acp:{session_id}", json.dumps({"role": role, "content": text}), ts, mtype),
                )
            await acp.db.commit()

            await acp.agent.load_session(cwd=str(tmp_path), session_id=session_id)

            texts = [c.kwargs["update"].content.text for c in acp.conn.session_update.await_args_list]
            assert texts == ["visible-user", "visible-answer"]
