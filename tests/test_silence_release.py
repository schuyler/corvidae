"""Kernel tests for issue #215: an unsent final releases transport state
(S8), the kernel's ``transform_system_prompt`` hook (S7), and core's freedom
from the silence plugin (S10), with the no-plugin behaviour unchanged (S4).

Design: .claude/plans/215-silence.md. Interface pinned here:
  corvidae.hooks.final_was_unsent(channel) -> bool
      True when the newest assistant row in channel.conversation is UNSENT.
  DiscordPlugin / SignalPlugin / AcpPlugin / CLIPlugin gain an
      ``owns()``-guarded ``on_agent_response`` hookimpl that releases the
      per-turn state an unsent final would otherwise leak.
  Agent.channel_system_prompt passes identity + register through the
      ``transform_system_prompt`` hook.

Kernel tests must not import corvidae_cognition (tests/test_kernel_boundary.py);
the real plugin is covered in tests/cognition/test_silence.py.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from helpers import build_plugin_and_channel, drain, sent_texts
from llm_response_fixtures import _make_text_response

from corvidae.channel import Channel
from corvidae.channels._registers import shipped_register
from corvidae.channel import resolve_system_prompt
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

# The root of the kernel package, searched recursively by the S10 test.
CORVIDAE_ROOT = Path(__file__).resolve().parent.parent / "corvidae"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _channel_ending_in(transport: str, scope: str, final_type: MessageType) -> Channel:
    """A channel whose window ends with an assistant row of ``final_type``:
    UNSENT for an unsent final, MESSAGE for a sent one."""
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
# final_was_unsent
# ---------------------------------------------------------------------------


class TestFinalWasUnsent:
    def test_true_when_newest_assistant_row_is_unsent(self):
        from corvidae.hooks import final_was_unsent

        channel = _channel_ending_in("test", "s", MessageType.UNSENT)
        assert final_was_unsent(channel) is True

    def test_false_when_newest_assistant_row_is_message(self):
        from corvidae.hooks import final_was_unsent

        channel = _channel_ending_in("test", "s", MessageType.MESSAGE)
        assert final_was_unsent(channel) is False

    def test_marker_row_after_the_final_does_not_hide_it(self):
        """The kernel follows an unsent final with a system CONTEXT marker;
        the newest *assistant* row is still the final."""
        from corvidae.hooks import final_was_unsent

        channel = _channel_ending_in("test", "s", MessageType.UNSENT)
        channel.conversation.append(
            {"role": "system", "content": "marker"}, MessageType.CONTEXT
        )
        assert final_was_unsent(channel) is True

    def test_false_without_a_conversation_or_assistant_row(self):
        from corvidae.hooks import final_was_unsent

        channel = Channel(transport="test", scope="s")
        assert final_was_unsent(channel) is False
        channel.conversation = ContextWindow(channel.id)
        channel.conversation.append({"role": "user", "content": "hi"})
        assert final_was_unsent(channel) is False


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
    async def test_unsent_final_releases_exactly_one_slot(self, factory, transport, scope):
        """Two slots outstanding, one unsent final: one left. (Starting at
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
    async def test_unsent_final_completes_the_prompt_with_end_turn(self):
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
    async def test_unsent_final_prints_the_prompt_once(self, capsys):
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

    async def test_unsent_final_on_a_channel_cli_does_not_own_prints_nothing(self, capsys):
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
    fires, read straight from the window (not through final_was_unsent)."""

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
# S7: the system prompt passes through transform_system_prompt
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


class _AppendMarker:
    """A stub wrapper that appends a marker to the composed system prompt."""

    @hookimpl(wrapper=True)
    def transform_system_prompt(self, channel, text):
        result = yield
        return result + "\n\nSTUB-MARKER"


class TestSystemPromptHook:
    async def test_wrapper_output_ends_the_channel_system_prompt(self, agent_env):
        plugin, channel = agent_env
        plain = await plugin.channel_system_prompt(channel)
        plugin.pm.register(_AppendMarker(), name="marker")

        prompt = await plugin.channel_system_prompt(channel)

        assert prompt == plain + "\n\nSTUB-MARKER"

    async def test_no_wrapper_leaves_identity_plus_register(self, agent_env):
        plugin, channel = agent_env
        resolved = plugin._registry.resolve_config(channel)
        identity = resolve_system_prompt(resolved["system_prompt"], Path("."))
        register = await plugin._resolve_register(channel, resolved["register_prompt"], Path("."))

        prompt = await plugin.channel_system_prompt(channel)

        assert prompt == (f"{identity}\n\n{register}" if register else identity)

    async def test_raising_wrapper_falls_back_to_the_untransformed_prompt(self, agent_env, caplog):
        plugin, channel = agent_env
        plain = await plugin.channel_system_prompt(channel)

        class _Raises:
            @hookimpl(wrapper=True)
            def transform_system_prompt(self, channel, text):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        plugin.pm.register(_Raises(), name="raises")
        with caplog.at_level("WARNING"):
            prompt = await plugin.channel_system_prompt(channel)

        assert prompt == plain
        assert channel.id in caplog.text and "transform_system_prompt" in caplog.text


# ---------------------------------------------------------------------------
# S10: core names nothing from cognition
# ---------------------------------------------------------------------------


class TestCoreIsUnawareOfSilence:
    def test_no_core_file_names_the_plugin_or_its_helpers(self):
        # Built from parts so this file's own text is not the offender.
        banned = [
            '"' + "silence" + '"', "'" + "silence" + "'",
            "silence" + "_fragment", "with" + "_silence", "register" + "_fragment",
        ]
        offenders = [
            f"{path.relative_to(CORVIDAE_ROOT).as_posix()}: {token}"
            for path in sorted(CORVIDAE_ROOT.rglob("*.py"))
            if "__pycache__" not in path.parts
            for token in banned
            if token in path.read_text()
        ]
        assert not offenders, "core must not couple to the silence plugin:\n" + "\n".join(offenders)

    def test_hooks_exports_the_neutral_helper_only(self):
        import corvidae.hooks as hooks

        assert hasattr(hooks, "final_was_unsent")
        assert not hasattr(hooks, "final_was_" + "withheld")


# ---------------------------------------------------------------------------
# S4: with the plugin not loaded, behaviour is unchanged
# ---------------------------------------------------------------------------


class TestSilenceNotLoaded:
    """Regression guards: without the silence plugin the registers are the
    shipped text and a NO_REPLY final is sent like any other reply."""

    @pytest.mark.parametrize("factory, transport, scope, names", _register_cases())
    async def test_registers_equal_todays_shipped_text(self, factory, transport, scope, names):
        plugin = factory()
        register = await plugin.resolve_register_prompt(
            channel=Channel(transport=transport, scope=scope)
        )

        assert register == shipped_register(*names)

    async def test_no_reply_user_final_is_sent_unchanged(self, agent_env):
        plugin, channel = agent_env
        plugin._client = _client(_make_text_response("NO_REPLY"))
        plugin._mark_started()

        await _say(plugin, channel)

        assert sent_texts(plugin.pm, OutboundMessage) == ["NO_REPLY"]
