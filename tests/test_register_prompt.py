"""Identity plus per-channel register (GitHub issue #316).

Every channel's system prompt starts with ``agent.system_prompt`` (the
identity). A channel's register is appended after it: from
``register_prompt`` config (exact entry, transport-wide entries, then the
agent level), else from the owning transport's shipped default (the
``resolve_register_prompt`` hook), else blank. No channel setting replaces
the identity.
"""

from __future__ import annotations

import logging
from importlib import resources
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.agent import mint_trace_id
from corvidae.channel import (
    Channel,
    ChannelRegistry,
    load_channel_config,
    parse_channel_config,
    validate_agent_config,
)
from corvidae.channels.acp import AcpPlugin
from corvidae.channels.cli import CLIPlugin
from corvidae.channels.discord import DiscordPlugin
from corvidae.channels.irc import IRCPlugin
from corvidae.channels.signal import SignalPlugin
from corvidae.channels.web import WebPlugin
from corvidae.hooks import AgentSpec, hookimpl

from helpers import build_plugin_and_channel, drain
from llm_response_fixtures import _make_text_response

AGENT = {"system_prompt": "AGENT", "max_turns": 10, "max_context_tokens": 8000}


def _shipped(*names: str) -> str:
    """The shipped register text for ``names``, read independently of the
    code under test: each package file stripped, joined by a blank line."""
    base = resources.files("corvidae.channels").joinpath("registers")
    return "\n\n".join(base.joinpath(f"{n}.md").read_text().strip() for n in names)


def _resolved_register(config: dict, transport: str, scope: str, agent: dict | None = None):
    """The resolved ``register_prompt`` for a channel under ``config``."""
    registry = ChannelRegistry(dict(AGENT if agent is None else agent))
    load_channel_config(config, registry)
    return registry.resolve_config(registry.get_or_create(transport, scope))["register_prompt"]


# ---------------------------------------------------------------------------
# R1: no channel setting replaces the identity
# ---------------------------------------------------------------------------


class TestIdentityCannotBeReplaced:
    @pytest.mark.parametrize("key", ["irc:#lex", "signal:*"])
    def test_channel_system_prompt_is_rejected_with_the_remedy(self, key):
        with pytest.raises(ValueError) as exc:
            parse_channel_config({"channels": {key: {"system_prompt": "X"}}})
        message = str(exc.value)
        assert f"channels.{key}.system_prompt" in message
        assert "agent.system_prompt" in message
        assert f"channels.{key}.register_prompt" in message

    def test_identity_resolves_from_the_agent_only(self):
        registry = ChannelRegistry(dict(AGENT))
        load_channel_config(
            {"channels": {"signal:*": {"register_prompt": "W"}, "signal:a": {"register_prompt": "E"}}},
            registry,
        )
        resolved = registry.resolve_config(registry.get_or_create("signal", "a"))
        assert resolved["system_prompt"] == "AGENT"


# ---------------------------------------------------------------------------
# R11: wrong types are rejected at config load
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("value", [5, ["ok.md", 3], {"a": 1}])
    def test_channel_register_prompt_of_wrong_type_is_rejected(self, value):
        with pytest.raises(ValueError) as exc:
            parse_channel_config({"channels": {"irc:#lex": {"register_prompt": value}}})
        message = str(exc.value)
        assert "channels.irc:#lex.register_prompt" in message
        assert repr(value)[:20] in message
        assert "string" in message and "list" in message

    def test_agent_register_prompt_of_wrong_type_is_rejected(self):
        with pytest.raises(ValueError) as exc:
            validate_agent_config({"register_prompt": 5})
        message = str(exc.value)
        assert "agent.register_prompt" in message
        assert "5" in message
        assert "string" in message and "list" in message

    @pytest.mark.parametrize("value", ["text", ["a.md", "b.md"], "", []])
    def test_valid_register_prompt_values_load(self, value):
        parse_channel_config({"channels": {"signal:*": {"register_prompt": value}}})
        validate_agent_config({"register_prompt": value})


# ---------------------------------------------------------------------------
# R4, R6, R7: precedence among configured layers
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_exact_beats_wide_beats_agent(self):
        config = {
            "channels": {
                "discord:*": {"register_prompt": "WIDE"},
                "discord:dm:*": {"register_prompt": "DM-WIDE"},
                "discord:dm:7": {"register_prompt": "EXACT"},
            }
        }
        agent = dict(AGENT, register_prompt="AGENT-REG")
        assert _resolved_register(config, "discord", "dm:7", agent) == "EXACT"
        assert _resolved_register(config, "discord", "dm:8", agent) == "DM-WIDE"
        assert _resolved_register(config, "discord", "9", agent) == "WIDE"
        assert _resolved_register(config, "irc", "#x", agent) == "AGENT-REG"

    def test_unconfigured_register_is_none(self):
        assert _resolved_register({}, "signal", "a") is None

    @pytest.mark.parametrize("empty", ["", []])
    def test_explicit_empty_is_kept_not_treated_as_unset(self, empty):
        config = {"channels": {"signal:*": {"register_prompt": empty}}}
        assert _resolved_register(config, "signal", "a") == empty


# ---------------------------------------------------------------------------
# R3, R5, R9: shipped defaults from the owning transport
# ---------------------------------------------------------------------------


async def _default(plugin, transport: str, scope: str):
    return await plugin.resolve_register_prompt(channel=Channel(transport=transport, scope=scope))


class TestShippedDefaults:
    async def test_signal_gets_chat_and_signal_dm(self):
        assert await _default(SignalPlugin(), "signal", "aci-1") == _shipped("chat", "signal_dm")

    async def test_discord_dm_gets_chat(self):
        assert await _default(DiscordPlugin(), "discord", "dm:42") == _shipped("chat")

    async def test_discord_server_gets_chat_and_group(self):
        assert await _default(DiscordPlugin(), "discord", "42") == _shipped("chat", "discord_group")

    @pytest.mark.parametrize("scope", ["#lex", "somenick"])
    async def test_irc_gets_chat_and_irc(self, scope):
        assert await _default(IRCPlugin(), "irc", scope) == _shipped("chat", "irc")

    @pytest.mark.parametrize(
        "plugin, transport",
        [(SignalPlugin(), "discord"), (DiscordPlugin(), "signal"), (IRCPlugin(), "signal")],
    )
    async def test_a_transport_does_not_answer_for_another(self, plugin, transport):
        assert await _default(plugin, transport, "x") is None

    @pytest.mark.parametrize("cls", [WebPlugin, CLIPlugin, AcpPlugin])
    def test_web_cli_acp_ship_no_default(self, cls):
        assert not hasattr(cls, "resolve_register_prompt")

    def test_hookspec_is_firstresult_with_required_channel(self):
        spec = AgentSpec.resolve_register_prompt
        assert spec.corvidae_spec["firstresult"] is True
        import inspect

        params = inspect.signature(spec).parameters
        assert list(params) == ["self", "channel"]
        assert params["channel"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# R2, R5, R7, R8: what the model receives, end to end through the Agent
# ---------------------------------------------------------------------------


class _DefaultRegister:
    """Stands in for a transport that ships ``text`` as its default."""

    def __init__(self, transport: str, text: str) -> None:
        self.transport = transport
        self.text = text

    @hookimpl
    async def resolve_register_prompt(self, channel):
        if channel.transport != self.transport:
            return None
        return self.text


async def _system_sent(plugin, channel) -> tuple[str, MagicMock]:
    """Send one message on ``channel``; return the system prompt the LLM got."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("ok"))
    plugin._client = client
    plugin._mark_started()
    await plugin.on_message(channel=channel, sender="u", text="hi", trace_id=mint_trace_id())
    await drain(plugin, channel)
    messages = client.chat.call_args.args[0]
    assert messages[0]["role"] == "system"
    return messages[0]["content"], client


@pytest.fixture
async def agent():
    plugin, _, db = await build_plugin_and_channel(agent_defaults=dict(AGENT))
    plugin.pm.register(_DefaultRegister("signal", "DEFAULT"), name="fake_signal_register")
    try:
        yield plugin
    finally:
        await plugin.pm.get_plugin("task").on_stop()
        await db.close()


class TestThroughTheAgent:
    async def test_no_register_sends_the_identity_alone(self, agent):
        channel = agent._registry.get_or_create("web", "default")
        prompt, _ = await _system_sent(agent, channel)
        assert prompt == "AGENT"

    async def test_shipped_default_is_appended_after_the_identity(self, agent):
        channel = agent._registry.get_or_create("signal", "aci-1")
        prompt, _ = await _system_sent(agent, channel)
        assert prompt == "AGENT\n\nDEFAULT"

    async def test_configured_register_replaces_the_default(self, agent):
        load_channel_config({"channels": {"signal:*": {"register_prompt": "OVR"}}}, agent._registry)
        channel = agent._registry.get_or_create("signal", "aci-2")
        prompt, _ = await _system_sent(agent, channel)
        assert prompt == "AGENT\n\nOVR"

    @pytest.mark.parametrize("empty", ["", []])
    async def test_explicit_empty_switches_the_default_off(self, agent, empty, caplog):
        load_channel_config({"channels": {"signal:*": {"register_prompt": empty}}}, agent._registry)
        channel = agent._registry.get_or_create("signal", "aci-3")
        with caplog.at_level(logging.WARNING):
            prompt, _ = await _system_sent(agent, channel)
        assert prompt == "AGENT"
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    async def test_agent_level_register_beats_the_shipped_default(self, agent):
        agent._registry.agent_defaults = dict(AGENT, register_prompt="AGENT-REG")
        channel = agent._registry.get_or_create("signal", "aci-4")
        prompt, _ = await _system_sent(agent, channel)
        assert prompt == "AGENT\n\nAGENT-REG"

    async def test_register_on_a_transport_with_no_default_applies(self, agent):
        load_channel_config({"channels": {"web:*": {"register_prompt": "WEB"}}}, agent._registry)
        channel = agent._registry.get_or_create("web", "default")
        prompt, _ = await _system_sent(agent, channel)
        assert prompt == "AGENT\n\nWEB"

    async def test_register_files_resolve_against_the_config_dir(self, agent, tmp_path):
        (tmp_path / "a.md").write_text("ONE\n")
        (tmp_path / "b.md").write_text("TWO\n")
        agent._base_dir = tmp_path
        load_channel_config(
            {"channels": {"signal:*": {"register_prompt": ["a.md", "b.md"]}}}, agent._registry
        )
        channel = agent._registry.get_or_create("signal", "aci-5")
        prompt, _ = await _system_sent(agent, channel)
        assert prompt == "AGENT\n\nONE\n\nTWO"

    @pytest.mark.parametrize("content", [None, b"\xff\xfe\x00bad"])
    async def test_unreadable_register_file_degrades_with_a_warning(
        self, agent, tmp_path, caplog, content
    ):
        path = tmp_path / "reg.md"
        if content is not None:
            path.write_bytes(content)
        agent._base_dir = tmp_path
        load_channel_config({"channels": {"signal:*": {"register_prompt": ["reg.md"]}}}, agent._registry)
        channel = agent._registry.get_or_create("signal", "aci-6")
        with caplog.at_level(logging.WARNING):
            prompt, client = await _system_sent(agent, channel)
        assert prompt == "AGENT"
        client.chat.assert_called()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        expected_type = "FileNotFoundError" if content is None else "UnicodeDecodeError"
        assert any(
            channel.id in w and "reg.md" in w and "register_prompt" in w and expected_type in w
            for w in warnings
        ), warnings


# ---------------------------------------------------------------------------
# R10: moved fragments are gone from prompts/
# ---------------------------------------------------------------------------


def test_moved_fragments_are_deleted_from_prompts(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "prompts"
    for name in ("chat.md", "signal_dm.md", "discord_group.md", "IRC.md"):
        assert not (root / name).exists(), f"prompts/{name} should have moved into the package"
