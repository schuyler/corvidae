"""Tests for issue #215, the cognition half: SilencePlugin (the instruction
on conversational transports, S3/S7/S9; the NO_REPLY veto, S1; the decline
record, S5).

Design: .claude/plans/215-silence.md. Interface pinned here:
  corvidae_cognition.silence.NO_REPLY == "NO_REPLY"
  corvidae_cognition.silence.REGISTER_FRAGMENT: str
  corvidae_cognition.silence.SilencePlugin (entry point "silence",
      optional_depends_on = {"outcome_log"}):
      transform_system_prompt (wrapper) — on a channel owned by a
          conversational transport (IRC, Discord, Signal, Zulip) appends a
          blank line and the instruction for the mode (REGISTER_FRAGMENT in
          token mode); any other channel is unchanged
      silence.mode (token | tool | both) — token registers no tool; tool and
          both register end_turn(reason) through the register_tools hook;
          a bad mode or a non-mapping silence block fails on_init naming the
          key and value; the instruction text follows the mode (#797)
      end_turn — on a conversational channel returns the result asking for
          an exact NO_REPLY echo and records outcomes {"silence": "tool",
          "silence_reason": <reason cut at 200 characters>}; on any other
          channel returns an Error naming the channel and saying to answer
          normally; the turn's final never overwrites the tool's record
      should_send_response — final emission carrying the NO_REPLY token
          (an exact reply, or a reply embedding it), any origin -> False;
          anything else -> None
      on_agent_response — when the newest assistant row is UNSENT: a
          response carrying the NO_REPLY token records outcomes
          {"silence": "no_reply"}, an empty one {"silence": "empty"}, any
          other unsent final nothing; a sent final records nothing. Writes
          go through outcome_log.upsert_exchange from tracked tasks that
          on_stop awaits.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from helpers import build_plugin_and_channel, drain, sent_texts
from llm_response_fixtures import _make_text_response

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType, unsent_marker_emission
from corvidae.hooks import create_plugin_manager, resolve_reject_wins
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

# ---------------------------------------------------------------------------
# The veto (S1)
# ---------------------------------------------------------------------------


class TestVeto:
    async def _verdict(self, text="NO_REPLY", emission="final", origin="user"):
        """Dispatch through the plugin manager, as the kernel's _gate_send
        does, so a method missing @hookimpl is never consulted."""
        from corvidae_cognition.silence import SilencePlugin

        pm = create_plugin_manager()
        pm.register(SilencePlugin(), name="silence")
        results = await pm.ahook.should_send_response(
            channel=Channel(transport="test", scope="veto"),
            text=text, emission=emission, trace_id="sveto", annotations={"origin": origin},
        )
        return resolve_reject_wins(results)

    def test_constants(self):
        from corvidae_cognition.silence import (
            NO_REPLY,
            REGISTER_FRAGMENT,
            SilencePlugin,
        )

        assert NO_REPLY == "NO_REPLY"
        assert NO_REPLY in REGISTER_FRAGMENT
        assert not hasattr(SilencePlugin, "register_fragment")

    @pytest.mark.parametrize(
        "text",
        [
            "NO_REPLY",
            "  NO_REPLY\n",
            "I'll be quiet: NO_REPLY",
            "NO_REPLY (standing by)",
            "nothing to add — NO_REPLY",
        ],
    )
    @pytest.mark.parametrize("origin", ["user", "heartbeat", "critique"])
    async def test_marker_final_is_vetoed_for_any_origin(self, text, origin):
        assert await self._verdict(text=text, origin=origin) is False

    @pytest.mark.parametrize(
        "text, emission",
        [
            ("The deadline is Thursday.", "final"),
            ("No replies from CI yet.", "final"),
            ("The marker is NO_REPL, a near miss.", "final"),
            ("NO_REPLY", "progress"),
            ("NO_REPLY", "thinking"),
            ("NO_REPLY", "error"),
        ],
    )
    async def test_anything_else_has_no_opinion(self, text, emission):
        assert await self._verdict(text=text, emission=emission) is None


# ---------------------------------------------------------------------------
# S3/S7/S9: the system prompt carries the instruction on conversational
# transports only
# ---------------------------------------------------------------------------


async def _prompt_pm(config: dict | None = None):
    """A manager with the real SilencePlugin and every real transport; the
    web plugin is enabled so it owns its channel. ``config`` reaches the
    plugin's on_init."""
    from corvidae.channels.acp import AcpPlugin
    from corvidae.channels.cli import CLIPlugin
    from corvidae.channels.discord import DiscordPlugin
    from corvidae.channels.irc import IRCPlugin
    from corvidae.channels.signal import SignalPlugin
    from corvidae.channels.web import WebPlugin
    from corvidae_cognition.silence import SilencePlugin

    pm = create_plugin_manager()
    silence = SilencePlugin()
    pm.register(silence, name="silence")
    await silence.on_init(pm=pm, config=config or {})
    web = WebPlugin()
    web._enabled = True
    for name, plugin in (
        ("irc", IRCPlugin()), ("discord", DiscordPlugin()), ("signal", SignalPlugin()),
        ("acp", AcpPlugin()), ("web", web), ("cli", CLIPlugin()),
    ):
        pm.register(plugin, name=name)
    return pm


class TestSystemPromptWrapper:
    @pytest.mark.parametrize(
        "transport, scope",
        [("irc", "#lex"), ("discord", "dm:7"), ("signal", "aci-1")],
    )
    async def test_conversational_channel_gets_the_fragment(self, transport, scope):
        from corvidae_cognition.silence import REGISTER_FRAGMENT

        result = await (await _prompt_pm()).ahook.transform_system_prompt(
            channel=Channel(transport=transport, scope=scope), text="IDENTITY"
        )

        assert result == "IDENTITY\n\n" + REGISTER_FRAGMENT

    @pytest.mark.parametrize(
        "transport, scope",
        [("acp", "sess1"), ("web", "tab1"), ("cli", "local")],
    )
    async def test_always_answer_channel_is_unchanged(self, transport, scope):
        result = await (await _prompt_pm()).ahook.transform_system_prompt(
            channel=Channel(transport=transport, scope=scope), text="IDENTITY"
        )

        assert result == "IDENTITY"


class TestInstructionOverride:
    """PRD principle 15: the instruction is a prompt the operator can change,
    through the silence.instruction key and corvidae.prompts.resolve_prompt."""

    async def test_inline_string_replaces_the_instruction(self):
        pm = await _prompt_pm({"silence": {"instruction": "CUSTOM INSTRUCTION"}})

        result = await pm.ahook.transform_system_prompt(
            channel=Channel(transport="irc", scope="#lex"), text="IDENTITY"
        )

        assert result == "IDENTITY\n\nCUSTOM INSTRUCTION"

    async def test_file_reference_is_read_relative_to_the_config_dir(self, tmp_path):
        (tmp_path / "silence.md").write_text("FROM A FILE\n")
        pm = await _prompt_pm(
            {"_base_dir": str(tmp_path), "silence": {"instruction": {"file": "silence.md"}}}
        )

        result = await pm.ahook.transform_system_prompt(
            channel=Channel(transport="signal", scope="aci-1"), text="IDENTITY"
        )

        assert result == "IDENTITY\n\nFROM A FILE"


# ---------------------------------------------------------------------------
# S1 at agent level: nothing reaches the transport, the row is UNSENT
# ---------------------------------------------------------------------------


@pytest.fixture
async def agent_env():
    """A real Agent with the real on_agent_response hook and SilencePlugin."""
    from corvidae_cognition.silence import SilencePlugin

    plugin, channel, db = await build_plugin_and_channel(mock_on_agent_response=False)
    silence = SilencePlugin()
    plugin.pm.register(silence, name="silence")
    await silence.on_init(pm=plugin.pm, config={})
    yield plugin, channel, db, silence
    await silence.on_stop()
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


class TestDeclineAtAgentLevel:
    @pytest.mark.parametrize("reply", ["NO_REPLY", " NO_REPLY\n"])
    async def test_user_turn_no_reply_reaches_no_transport(self, agent_env, reply):
        plugin, channel, db, _silence = agent_env
        client = MagicMock()
        client.chat = AsyncMock(side_effect=[_make_text_response(reply)])
        plugin._client = client
        plugin._mark_started()

        await plugin.on_message(
            channel=channel, sender="user", text="thanks, night!", trace_id=mint_trace_id()
        )
        await drain(plugin, channel)

        # Nothing was sent.
        assert sent_texts(plugin.pm, OutboundMessage) == []
        # The row is stored UNSENT, with the final marker after it.
        async with db.execute(
            "SELECT message, message_type FROM message_log WHERE channel_id = ? ORDER BY id",
            (channel.id,),
        ) as cursor:
            rows = [(json.loads(m), t) for m, t in await cursor.fetchall()]
        assert [t for m, t in rows if m.get("role") == "assistant"] == ["unsent"]
        assert any(
            t == "context" and unsent_marker_emission(m["content"]) == "final" for m, t in rows
        )
        # And the window tags it, so later turns can see the agent stayed silent.
        window_assistant = [m for m in channel.conversation.messages if m.get("role") == "assistant"]
        assert window_assistant[-1]["_message_type"] == MessageType.UNSENT


# ---------------------------------------------------------------------------
# S5: every decline is recorded in exchange_log
# ---------------------------------------------------------------------------


class _StubOutcomeLog:
    """Records the writes SilencePlugin makes through upsert_exchange."""

    def __init__(self):
        self.writes: list[dict] = []

    async def upsert_exchange(self, trace_id, channel_id, origin=None, **columns):
        self.writes.append(
            {"trace_id": trace_id, "channel_id": channel_id, "origin": origin, **columns}
        )


def _channel_ending_in(final_type: MessageType) -> Channel:
    """A channel whose newest assistant row is of ``final_type``."""
    channel = Channel(transport="test", scope="s5")
    conv = ContextWindow(channel.id)
    conv.append({"role": "user", "content": "thanks, night!"})
    conv.append({"role": "assistant", "content": "x"}, final_type)
    channel.conversation = conv
    return channel


async def _respond(response_text: str, final_type: MessageType) -> list[dict]:
    """Fire on_agent_response for a final and return the recorded writes."""
    from corvidae_cognition.silence import SilencePlugin

    pm = create_plugin_manager()
    outcome_log = _StubOutcomeLog()
    pm.register(outcome_log, name="outcome_log")
    silence = SilencePlugin()
    pm.register(silence, name="silence")
    await silence.on_init(pm=pm, config={})

    # Dispatch through the manager so a method missing @hookimpl never runs.
    await pm.ahook.on_agent_response(
        channel=_channel_ending_in(final_type), request_text="thanks, night!",
        response_text=response_text, trace_id="t-s5", annotations={}, logprobs=None,
    )
    # The write is a tracked fire-and-forget task; on_stop drains it.
    await silence.on_stop()
    return outcome_log.writes


class TestDeclineRecord:
    @pytest.mark.parametrize(
        "response_text, expected",
        [
            ("NO_REPLY", "no_reply"),
            (" NO_REPLY\n", "no_reply"),
            ("quiet, standing by: NO_REPLY", "no_reply"),
            ("", "empty"),
        ],
    )
    async def test_unsent_decline_is_recorded(self, response_text, expected):
        writes = await _respond(response_text, MessageType.UNSENT)

        assert len(writes) == 1
        assert writes[0]["trace_id"] == "t-s5"
        assert writes[0]["channel_id"] == "test:s5"
        assert writes[0]["outcomes"] == {"silence": expected}

    async def test_sent_reply_records_nothing(self):
        assert await _respond("Thursday works for me.", MessageType.MESSAGE) == []

    async def test_sent_no_reply_records_nothing(self):
        """A NO_REPLY that went out (a sibling gate failed open) is not a
        decline: its newest row is MESSAGE."""
        assert await _respond("NO_REPLY", MessageType.MESSAGE) == []

    async def test_other_unsent_final_records_nothing(self):
        """Another gate's veto is not this plugin's decline."""
        assert await _respond("a vetoed answer", MessageType.UNSENT) == []

    async def test_near_miss_not_recorded(self):
        """A near-miss substring (NO_REPL, No replies) is not a no_reply decline."""
        writes = await _respond('The marker is NO_REPL', MessageType.UNSENT)
        assert writes == []

        writes = await _respond('No replies from CI yet.', MessageType.UNSENT)
        assert writes == []


class TestEndToEndConsistency:
    """A contained final is both dropped (UNSENT) and recorded (no_reply)."""

    async def test_contained_final_dropped_and_recorded(self):
        from corvidae_cognition.silence import SilencePlugin

        text = 'done, NO_REPLY'

        # The veto: should_send_response returns False
        pm = create_plugin_manager()
        pm.register(SilencePlugin(), name='silence')
        results = await pm.ahook.should_send_response(
            channel=Channel(transport='test', scope='e2e'),
            text=text, emission='final', trace_id='e2e', annotations={'origin': 'user'},
        )
        assert resolve_reject_wins(results) is False

        # The record: on_agent_response records no_reply
        writes = await _respond(text, MessageType.UNSENT)
        assert len(writes) == 1
        assert writes[0]['outcomes'] == {'silence': 'no_reply'}

# ---------------------------------------------------------------------------
# silence.mode: the end_turn tool (#797)
# ---------------------------------------------------------------------------


async def _tool_pm(mode: str | None = "tool", outcome_log=None, instruction=None):
    """A plugin manager with SilencePlugin in ``mode`` and real transports
    (IRC is conversational; the enabled web plugin and the CLI are not),
    plus the plugin and the tools it registered, keyed by name."""
    from corvidae.channels.cli import CLIPlugin
    from corvidae.channels.irc import IRCPlugin
    from corvidae.channels.web import WebPlugin
    from corvidae_cognition.silence import SilencePlugin

    pm = create_plugin_manager()
    silence = SilencePlugin()
    pm.register(silence, name="silence")
    web = WebPlugin()
    web._enabled = True
    for name, plugin in (("irc", IRCPlugin()), ("web", web), ("cli", CLIPlugin())):
        pm.register(plugin, name=name)
    if outcome_log is not None:
        pm.register(outcome_log, name="outcome_log")
    block = {"mode": mode} if mode else {}
    if instruction:
        block["instruction"] = instruction
    await silence.on_init(pm=pm, config={"silence": block} if block else {})
    # Dispatch through the manager so a register_tools missing @hookimpl
    # never loads.
    registry: list = []
    pm.hook.register_tools(tool_registry=registry)
    return pm, silence, {tool.name: tool for tool in registry}


async def _call(tools, channel, trace_id=None, **kwargs):
    """Call end_turn as the kernel's tool Task does: with the turn's trace id
    in the attribution and a ToolContext for ``channel``."""
    from corvidae.attribution import reset_attribution, set_attribution
    from corvidae.tool import ToolContext

    ctx = ToolContext(channel=channel, tool_call_id="call-1", task_queue=None)
    token = set_attribution(trace_id=trace_id)
    try:
        return await tools["end_turn"].fn(_ctx=ctx, **kwargs)
    finally:
        reset_attribution(token)


IRC = Channel(transport="irc", scope="#lex")


class TestModeConfig:
    @pytest.mark.parametrize(
        "mode, registers", [("token", False), ("tool", True), ("both", True)]
    )
    async def test_only_tool_modes_register_end_turn(self, mode, registers):
        _pm, _silence, tools = await _tool_pm(mode=mode)

        assert set(tools) == ({"end_turn"} if registers else set())
        if registers:
            assert "reason" in tools["end_turn"].schema["function"]["parameters"]["properties"]

    @pytest.mark.parametrize(
        "block, key, value, wanted",
        [
            ({"mode": "tools"}, "silence.mode", "'tools'", ("token", "tool", "both")),
            ("tool", "silence", "'tool'", ("mapping", "plugins.disabled")),
        ],
    )
    async def test_bad_config_names_key_value_and_remedy(self, block, key, value, wanted):
        from corvidae_cognition.silence import SilencePlugin

        pm = create_plugin_manager()
        silence = SilencePlugin()
        config = {"silence": block}

        with pytest.raises(ValueError) as exc:
            await silence.on_init(pm=pm, config=config)

        message = str(exc.value)
        assert key in message and value in message
        assert all(word in message for word in wanted)
        # The runtime adds "Config error:" and the path; no doubling here.
        assert "Config error" not in message


class TestModeInstruction:
    async def _prompt(self, **kwargs):
        pm, _silence, _tools = await _tool_pm(**kwargs)
        result = await pm.ahook.transform_system_prompt(channel=IRC, text="IDENTITY")
        return result.removeprefix("IDENTITY\n\n")

    @pytest.mark.parametrize(
        "mode, names_tool, asks_for_token",
        [("tool", True, False), ("both", True, True)],
    )
    async def test_instruction_follows_the_mode(self, mode, names_tool, asks_for_token):
        fragment = await self._prompt(mode=mode)

        assert ("end_turn" in fragment) is names_tool
        assert ("reply with exactly NO_REPLY" in fragment) is asks_for_token

    async def test_operator_instruction_still_wins_in_tool_mode(self):
        assert await self._prompt(mode="tool", instruction="CUSTOM") == "CUSTOM"


class TestEndTurnTool:
    @pytest.mark.parametrize("kwargs", [{"reason": "they said thanks"}, {}])
    async def test_result_tells_the_model_to_echo_the_token(self, kwargs):
        """Also with no reason: a model that omits the argument must still
        reach the echo."""
        _pm, _silence, tools = await _tool_pm()

        result = await _call(tools, IRC, **kwargs)

        assert "Nothing was sent" in result
        assert "exactly NO_REPLY" in result

    @pytest.mark.parametrize(
        "reason, recorded", [("they said thanks", "they said thanks"), ("x" * 500, "x" * 200)]
    )
    async def test_call_is_recorded_with_its_reason_cut_at_200(self, reason, recorded):
        log = _StubOutcomeLog()
        _pm, silence, tools = await _tool_pm(outcome_log=log)

        await _call(tools, IRC, trace_id="t-tool", reason=reason)
        await silence.on_stop()

        assert len(log.writes) == 1
        assert log.writes[0]["trace_id"] == "t-tool"
        assert log.writes[0]["channel_id"] == "irc:#lex"
        assert log.writes[0]["outcomes"] == {"silence": "tool", "silence_reason": recorded}

    @pytest.mark.parametrize(
        "transport, scope, names",
        [
            # web is owned by a non-conversational transport; internal has
            # no owning transport; None is no channel at all.
            ("web", "tab1", "web:tab1"),
            ("internal", "heartbeat", "internal:heartbeat"),
            (None, None, "no channel context"),
        ],
    )
    async def test_refused_with_a_remedy_and_nothing_recorded(self, transport, scope, names):
        log = _StubOutcomeLog()
        _pm, silence, tools = await _tool_pm(outcome_log=log)
        channel = Channel(transport=transport, scope=scope) if transport else None

        result = await _call(tools, channel, trace_id="t-refused", reason="x")
        await silence.on_stop()

        assert result.startswith("Error")
        assert names in result and "answer normally" in result
        assert log.writes == []

    @pytest.mark.parametrize(
        "reason, final_type, final_text",
        [
            # The guard is membership, not the reason's value: a null reason
            # must not let the echo overwrite the record.
            (None, MessageType.UNSENT, "NO_REPLY"),
            # A model that answers anyway still leaves no stale entry.
            ("done", MessageType.MESSAGE, "Goodnight!"),
        ],
    )
    async def test_the_turns_final_never_overwrites_the_tool_record(
        self, reason, final_type, final_text
    ):
        log = _StubOutcomeLog()
        pm, silence, tools = await _tool_pm(outcome_log=log)
        # end_turn only works on a conversational channel, so rebuild the
        # helper's window on an IRC channel.
        channel = Channel(transport="irc", scope="s5")
        channel.conversation = _channel_ending_in(final_type).conversation

        await _call(tools, channel, trace_id="t-s5", reason=reason)
        await pm.ahook.on_agent_response(
            channel=channel, request_text="thanks, night!", response_text=final_text,
            trace_id="t-s5", annotations={}, logprobs=None,
        )
        await silence.on_stop()

        assert [w["outcomes"]["silence"] for w in log.writes] == ["tool"]
        assert silence._ended == set()


class TestEndTurnAtAgentLevel:
    """Through the real agent loop: the tool call runs as a Task, its result
    brings the model back, and the echoed NO_REPLY reaches no transport."""

    async def test_end_turn_then_echo_sends_nothing_and_records_the_call(self):
        import asyncio

        from corvidae.hooks import TransportPlugin
        from corvidae_cognition.silence import SilencePlugin
        from helpers import drain_task_queue
        from llm_response_fixtures import _make_tool_call, _make_tool_call_response

        class Chat(TransportPlugin):
            transport = "chat"
            conversational = True

        plugin, _channel, db = await build_plugin_and_channel(mock_on_agent_response=False)
        pm = plugin.pm
        log = _StubOutcomeLog()
        pm.register(log, name="outcome_log")
        silence = SilencePlugin()
        pm.register(silence, name="silence")
        pm.register(Chat(), name="chat")
        await silence.on_init(pm=pm, config={"silence": {"mode": "tool"}})
        registry: list = []
        pm.hook.register_tools(tool_registry=registry)
        plugin._tools = {t.name: t.fn for t in registry}
        plugin._tool_schemas = [t.schema for t in registry]
        channel = plugin._registry.get_or_create("chat", "#lex")

        client = MagicMock()
        client.chat = AsyncMock(side_effect=[
            _make_tool_call_response(
                [_make_tool_call("c1", "end_turn", {"reason": "they said thanks"})]
            ),
            _make_text_response("NO_REPLY"),
        ])
        plugin._client = client
        plugin._mark_started()
        trace_id = mint_trace_id()

        await plugin.on_message(
            channel=channel, sender="user", text="thanks, night!", trace_id=trace_id
        )
        for _ in range(6):
            await drain(plugin, channel)
            await drain_task_queue(plugin)
            await asyncio.sleep(0.01)
        await silence.on_stop()

        try:
            assert sent_texts(pm, OutboundMessage) == []
            assert [w["outcomes"] for w in log.writes] == [
                {"silence": "tool", "silence_reason": "they said thanks"}
            ]
            assert log.writes[0]["trace_id"] == trace_id
        finally:
            await pm.get_plugin("task").on_stop()
            await db.close()
