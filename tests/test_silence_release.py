"""RED tests for issue #215, the kernel half: a withheld final releases
transport state (S8), and the transports' default registers carry the
silence instruction only while a plugin named "silence" is loaded
(S3/S7/S9), with the no-plugin behaviour unchanged (S4).

Design: .claude/plans/215-silence.md. Interface pinned here:
  corvidae.hooks.final_was_withheld(channel) -> bool
      True when the newest assistant row in channel.conversation is UNSENT.
  DiscordPlugin / SignalPlugin / AcpPlugin / CLIPlugin gain an
      ``owns()``-guarded ``on_agent_response`` hookimpl that releases the
      per-turn state a withheld final would otherwise leak.
  IRC / Discord / Signal ``resolve_register_prompt`` append the fragment of
      the plugin registered as "silence" (``register_fragment()``), after a
      blank line, when there is one.

Kernel tests must not import corvidae_cognition (tests/test_kernel_boundary.py),
so the silence plugin is a stub exposing ``register_fragment()`` here; the
real plugin is covered in tests/cognition/test_silence.py.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from helpers import build_plugin_and_channel, drain, sent_texts
from llm_response_fixtures import _make_text_response

from corvidae.channel import Channel
from corvidae.channels._registers import shipped_register
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

# The text the stub silence plugin contributes to a register.
FRAGMENT = "STUB-SILENCE-FRAGMENT: reply with exactly NO_REPLY when nothing is left to say."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubSilence:
    """Stands in for the cognition SilencePlugin: only the fragment accessor
    the transports look up by the name "silence"."""

    def register_fragment(self) -> str:
        """The instruction text a transport appends to its register."""
        return FRAGMENT


def _pm_with_silence(loaded: bool) -> object:
    """A plugin manager, with the stub registered as "silence" when loaded."""
    pm = create_plugin_manager()
    if loaded:
        pm.register(_StubSilence(), name="silence")
    return pm


def _channel_ending_in(transport: str, scope: str, final_type: MessageType) -> Channel:
    """A channel whose window ends with an assistant row of ``final_type``:
    UNSENT for a withheld final, MESSAGE for a sent one."""
    channel = Channel(transport=transport, scope=scope)
    conv = ContextWindow(channel.id)
    conv.append({"role": "user", "content": "thanks, night!"})
    conv.append({"role": "assistant", "content": "NO_REPLY"}, final_type)
    channel.conversation = conv
    return channel


def _on_agent_response_kwargs(channel: Channel) -> dict:
    """The arguments the kernel passes when a final is produced."""
    return {
        "channel": channel, "request_text": "thanks, night!", "response_text": "NO_REPLY",
        "trace_id": mint_trace_id(), "meta": {}, "logprobs": None,
    }


async def _dispatch(plugin: object, channel: Channel) -> None:
    """Fire on_agent_response through a plugin manager, as the kernel does,
    so a transport method missing @hookimpl is never called."""
    # A fresh manager holding only this transport: the hook reaches it only
    # if the method is a registered hookimpl.
    pm = create_plugin_manager()
    pm.register(plugin)
    await pm.ahook.on_agent_response(**_on_agent_response_kwargs(channel))


def _irc_plugin():
    """An IRCPlugin imported at test time (see tests/test_register_prompt.py:
    tests/test_irc_plugin.py mocks pydle at collection)."""
    from corvidae.channels.irc import IRCPlugin

    return IRCPlugin()


# ---------------------------------------------------------------------------
# final_was_withheld
# ---------------------------------------------------------------------------


class TestFinalWasWithheld:
    def test_true_when_newest_assistant_row_is_unsent(self):
        from corvidae.hooks import final_was_withheld

        channel = _channel_ending_in("test", "s", MessageType.UNSENT)
        assert final_was_withheld(channel) is True

    def test_false_when_newest_assistant_row_is_message(self):
        from corvidae.hooks import final_was_withheld

        channel = _channel_ending_in("test", "s", MessageType.MESSAGE)
        assert final_was_withheld(channel) is False

    def test_marker_row_after_the_final_does_not_hide_it(self):
        """The kernel follows a withheld final with a system CONTEXT marker;
        the newest *assistant* row is still the final."""
        from corvidae.hooks import final_was_withheld

        channel = _channel_ending_in("test", "s", MessageType.UNSENT)
        channel.conversation.append(
            {"role": "system", "content": "marker"}, MessageType.CONTEXT
        )
        assert final_was_withheld(channel) is True

    def test_false_without_a_conversation_or_assistant_row(self):
        from corvidae.hooks import final_was_withheld

        channel = Channel(transport="test", scope="s")
        assert final_was_withheld(channel) is False
        channel.conversation = ContextWindow(channel.id)
        channel.conversation.append({"role": "user", "content": "hi"})
        assert final_was_withheld(channel) is False


# ---------------------------------------------------------------------------
# S8, Discord and Signal: the typing slot
# ---------------------------------------------------------------------------


def _slot_transports():
    """(id, plugin factory, transport, scope) for the two slot-counting
    transports."""
    from corvidae.channels.discord import DiscordPlugin
    from corvidae.channels.signal import SignalPlugin

    return [
        pytest.param(DiscordPlugin, "discord", "dm:1", id="discord"),
        pytest.param(SignalPlugin, "signal", "aci-1", id="signal"),
    ]


class TestSlotRelease:
    @pytest.mark.parametrize("factory, transport, scope", _slot_transports())
    async def test_withheld_final_releases_exactly_one_slot(self, factory, transport, scope):
        """Two slots outstanding, one withheld final: one left. (Starting at
        one would hide a double release behind the floor at zero.)"""
        plugin = factory()
        channel = _channel_ending_in(transport, scope, MessageType.UNSENT)
        plugin._outstanding[channel.id] = 2

        await _dispatch(plugin, channel)

        assert plugin._outstanding[channel.id] == 1

    @pytest.mark.parametrize("factory, transport, scope", _slot_transports())
    async def test_sent_final_releases_once_in_kernel_order(self, factory, transport, scope):
        """on_agent_response (row MESSAGE) then send_message, as the kernel
        orders them: two slots become one, not zero."""
        plugin = factory()
        plugin._deliver = AsyncMock()
        channel = _channel_ending_in(transport, scope, MessageType.MESSAGE)
        plugin._outstanding[channel.id] = 2

        await _dispatch(plugin, channel)
        assert plugin._outstanding[channel.id] == 2  # a sent final waits for send_message
        await plugin.send_message(channel, OutboundMessage(text="hello"))

        assert plugin._outstanding[channel.id] == 1


# ---------------------------------------------------------------------------
# S8, ACP: the prompt future
# ---------------------------------------------------------------------------


class TestAcpRelease:
    async def test_withheld_final_completes_the_prompt_with_end_turn(self):
        from corvidae.channels.acp import AcpPlugin

        plugin = AcpPlugin()
        channel = _channel_ending_in("acp", "sess1", MessageType.UNSENT)
        future = asyncio.get_running_loop().create_future()
        plugin._active_prompts["sess1"] = future

        await _dispatch(plugin, channel)

        assert future.done()
        assert future.result().stop_reason == "end_turn"

    async def test_sent_final_completes_only_at_send_message(self):
        from corvidae.channels.acp import AcpPlugin

        plugin = AcpPlugin()
        channel = _channel_ending_in("acp", "sess1", MessageType.MESSAGE)
        future = asyncio.get_running_loop().create_future()
        plugin._active_prompts["sess1"] = future

        await _dispatch(plugin, channel)
        assert not future.done()  # the text has not been streamed yet
        await plugin.send_message(channel, OutboundMessage(text="hello"))

        assert future.done()
        assert future.result().stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# S8, CLI: the next prompt
# ---------------------------------------------------------------------------


class TestCliRelease:
    async def test_withheld_final_prints_the_prompt_once(self, capsys):
        from corvidae.channels.cli import CLIPlugin

        plugin = CLIPlugin()
        plugin._interactive = True
        channel = _channel_ending_in("cli", "local", MessageType.UNSENT)

        await _dispatch(plugin, channel)

        assert capsys.readouterr().out.count("> ") == 1

    async def test_sent_final_prints_the_prompt_once_in_kernel_order(self, capsys):
        from corvidae.channels.cli import CLIPlugin

        plugin = CLIPlugin()
        plugin._interactive = True
        channel = _channel_ending_in("cli", "local", MessageType.MESSAGE)

        await _dispatch(plugin, channel)
        await plugin.send_message(channel, OutboundMessage(text="hello"))

        assert capsys.readouterr().out.count("> ") == 1

    async def test_withheld_final_on_a_channel_cli_does_not_own_prints_nothing(self, capsys):
        """The owns() guard: on_agent_response reaches every transport, and
        CLI is the one whose missing guard would show on a terminal."""
        from corvidae.channels.cli import CLIPlugin

        plugin = CLIPlugin()
        plugin._interactive = True
        channel = _channel_ending_in("irc", "#lex", MessageType.UNSENT)

        await _dispatch(plugin, channel)

        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# S8, agent level: the row is appended before on_agent_response fires
# ---------------------------------------------------------------------------


class _RowAtResponse:
    """Records the newest assistant row's tag each time on_agent_response
    fires, read straight from the window (not through final_was_withheld)."""

    def __init__(self):
        self.tags: list[MessageType] = []
        self.contents: list[str] = []

    @hookimpl
    async def on_agent_response(self, channel, request_text, response_text, trace_id, meta, logprobs):
        newest = next(
            m for m in reversed(channel.conversation.messages) if m.get("role") == "assistant"
        )
        self.tags.append(newest["_message_type"])
        self.contents.append(newest["content"])


class _VetoFinal:
    """A gate that withholds every final."""

    @hookimpl
    async def should_send_response(self, channel, text, emission, trace_id, meta):
        return False if emission == "final" else None


@pytest.fixture
async def agent_env():
    """A real Agent with the real on_agent_response hook (not mocked)."""
    plugin, channel, db = await build_plugin_and_channel(mock_on_agent_response=False)
    yield plugin, channel
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


def _client(*responses: object) -> MagicMock:
    """A mock LLM client that returns ``responses`` in order."""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=list(responses))
    return client


async def _say(plugin: object, channel: Channel, text: str = "thanks, night!") -> None:
    """Deliver one user message and run the channel's queue to completion."""
    await plugin.on_message(channel=channel, sender="user", text=text, trace_id=mint_trace_id())
    await drain(plugin, channel)


class TestOrderingAtAgentLevel:
    async def test_vetoed_final_is_unsent_when_on_agent_response_fires(self, agent_env):
        plugin, channel = agent_env
        spy = _RowAtResponse()
        plugin.pm.register(spy, name="row_spy")
        plugin.pm.register(_VetoFinal(), name="veto_final")
        plugin._client = _client(_make_text_response("kept to myself"))
        plugin._mark_started()

        await _say(plugin, channel)

        assert spy.contents == ["kept to myself"]
        assert spy.tags == [MessageType.UNSENT]

    async def test_empty_final_is_unsent_when_on_agent_response_fires(self, agent_env):
        plugin, channel = agent_env
        spy = _RowAtResponse()
        plugin.pm.register(spy, name="row_spy")
        plugin._client = _client(_make_text_response(""))
        plugin._mark_started()

        await _say(plugin, channel)

        assert spy.tags == [MessageType.UNSENT]

    async def test_sent_final_is_message_when_on_agent_response_fires(self, agent_env):
        plugin, channel = agent_env
        spy = _RowAtResponse()
        plugin.pm.register(spy, name="row_spy")
        plugin._client = _client(_make_text_response("a real answer"))
        plugin._mark_started()

        await _say(plugin, channel)

        assert spy.contents == ["a real answer"]
        assert spy.tags == [MessageType.MESSAGE]


# ---------------------------------------------------------------------------
# S3/S7: registers carry the fragment while "silence" is loaded
# ---------------------------------------------------------------------------


def _register_cases():
    """(id, plugin factory, transport, scope, shipped names) for every
    transport that ships a register."""
    from corvidae.channels.discord import DiscordPlugin
    from corvidae.channels.signal import SignalPlugin

    return [
        pytest.param(_irc_plugin, "irc", "#lex", ("chat", "irc"), id="irc"),
        pytest.param(DiscordPlugin, "discord", "dm:7", ("chat",), id="discord-dm"),
        pytest.param(DiscordPlugin, "discord", "1234", ("chat", "discord_group"), id="discord-group"),
        pytest.param(SignalPlugin, "signal", "aci-1", ("chat", "signal_dm"), id="signal"),
    ]


async def _register(factory, transport, scope, loaded: bool) -> str | None:
    """The register a transport resolves for a channel, with the stub silence
    plugin loaded or not."""
    plugin = factory()
    plugin.pm = _pm_with_silence(loaded)
    return await plugin.resolve_register_prompt(channel=Channel(transport=transport, scope=scope))


class TestRegisterCarriesFragment:
    @pytest.mark.parametrize("factory, transport, scope, names", _register_cases())
    async def test_register_contains_the_fragment_when_loaded(self, factory, transport, scope, names):
        register = await _register(factory, transport, scope, loaded=True)

        assert register is not None and FRAGMENT in register
        # Appended after a blank line to the register the transport ships.
        assert register == shipped_register(*names) + "\n\n" + FRAGMENT

    async def test_silence_fragment_helper_reads_the_plugin_by_name(self):
        from corvidae.channels._registers import silence_fragment

        assert silence_fragment(_pm_with_silence(True)) == FRAGMENT
        assert silence_fragment(_pm_with_silence(False)) is None


# ---------------------------------------------------------------------------
# S9: ACP, web and CLI never learn the instruction
# ---------------------------------------------------------------------------


class TestAlwaysAnswerTransports:
    @pytest.mark.parametrize(
        "transport, scope",
        [("acp", "sess1"), ("web", "tab1"), ("cli", "local")],
    )
    async def test_no_register_resolves_with_silence_loaded(self, transport, scope):
        """Guard: ACP, web and CLI ship no register, so nothing carries the
        fragment to them (passes before Green; must keep passing)."""
        from corvidae.channels.acp import AcpPlugin
        from corvidae.channels.cli import CLIPlugin
        from corvidae.channels.web import WebPlugin

        pm = _pm_with_silence(True)
        for name, plugin in (("acp", AcpPlugin()), ("web", WebPlugin()), ("cli", CLIPlugin())):
            pm.register(plugin, name=name)

        register = await pm.ahook.resolve_register_prompt(
            channel=Channel(transport=transport, scope=scope)
        )

        assert register is None


# ---------------------------------------------------------------------------
# S4: with the plugin not loaded, behaviour is unchanged
# ---------------------------------------------------------------------------


class TestSilenceNotLoaded:
    """Regression guards, expected to PASS before Green: without a plugin
    named "silence" the registers are today's shipped text and a NO_REPLY
    final is sent like any other reply."""

    @pytest.mark.parametrize("factory, transport, scope, names", _register_cases())
    async def test_registers_equal_todays_shipped_text(self, factory, transport, scope, names):
        register = await _register(factory, transport, scope, loaded=False)

        assert register == shipped_register(*names)

    async def test_no_reply_user_final_is_sent_unchanged(self, agent_env):
        plugin, channel = agent_env
        plugin._client = _client(_make_text_response("NO_REPLY"))
        plugin._mark_started()

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == ["NO_REPLY"]
