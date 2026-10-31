"""Transport-wide channel entries (GitHub issue #203).

A ``channels:`` key ending in ``*`` covers every channel whose id starts
with the text before the ``*``: ``signal:*`` for every Signal DM,
``discord:dm:*`` for Discord DMs, ``acp:*`` for ACP sessions whose ids are
minted per session. Entries layer per key between ``agent:`` and the exact
entry, more specific prefixes over less specific ones; ``tools.disabled``
is the union of every layer, so a channel can only narrow.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.agent import mint_trace_id
from corvidae.channel import ChannelConfig, ChannelRegistry, load_channel_config
from corvidae.tool import tool_to_schema
from corvidae.tool_policy import disabled_on, refusal_message

from helpers import build_plugin_and_channel, drain, drain_task_queue
from llm_response_fixtures import _make_text_response, _make_tool_call, _make_tool_call_response

AGENT = {"system_prompt": "AGENT", "max_turns": 10, "max_context_tokens": 8000}

# The deployment the issue describes: Signal DMs and Discord groups get a
# chat register, Discord DMs their own, web and ACP keep the agent default.
CHAT_CONFIG = {
    "channels": {
        "signal:*": {"system_prompt": "SIGNAL"},
        "discord:*": {"system_prompt": "DISCORD-GROUP", "max_turns": 4},
        "discord:dm:*": {"system_prompt": "DISCORD-DM"},
    }
}


def _registry(config: dict, agent: dict | None = None) -> ChannelRegistry:
    """A registry loaded from ``config`` with ``agent`` defaults."""
    registry = ChannelRegistry(dict(AGENT if agent is None else agent))
    load_channel_config(config, registry)
    return registry


def _resolved(registry: ChannelRegistry, transport: str, scope: str) -> dict:
    """Resolved settings for a channel created the way a transport would."""
    return registry.resolve_config(registry.get_or_create(transport, scope))


# ---------------------------------------------------------------------------
# Coverage: channels config never named
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_unseen_signal_dm_gets_the_signal_entry(self):
        """R1: a contact who first writes after the config is loaded."""
        registry = _registry(CHAT_CONFIG)
        assert _resolved(registry, "signal", "0b1e-new-aci")["system_prompt"] == "SIGNAL"

    def test_unlisted_discord_server_channel_gets_the_discord_entry(self):
        """R2: server channels and threads key on a numeric id."""
        registry = _registry(CHAT_CONFIG)
        resolved = _resolved(registry, "discord", "123456789")
        assert resolved["system_prompt"] == "DISCORD-GROUP"
        assert resolved["max_turns"] == 4

    def test_discord_dm_gets_its_own_prompt_and_inherits_the_rest(self):
        """R3: the more specific prefix wins per key; keys it doesn't set
        come from the less specific entry."""
        registry = _registry(CHAT_CONFIG)
        resolved = _resolved(registry, "discord", "dm:42")
        assert resolved["system_prompt"] == "DISCORD-DM"
        assert resolved["max_turns"] == 4

    def test_more_specific_wins_whatever_the_config_order(self):
        """R3: specificity, not the order keys appear in agent.yaml."""
        config = {
            "channels": {
                "discord:dm:*": {"system_prompt": "DISCORD-DM"},
                "discord:*": {"system_prompt": "DISCORD-GROUP", "max_turns": 4},
            }
        }
        registry = _registry(config)
        resolved = _resolved(registry, "discord", "dm:42")
        assert resolved["system_prompt"] == "DISCORD-DM"
        assert resolved["max_turns"] == 4
        assert _resolved(registry, "discord", "77")["system_prompt"] == "DISCORD-GROUP"

    def test_web_keeps_the_agent_default(self):
        """R4."""
        registry = _registry(CHAT_CONFIG)
        assert _resolved(registry, "web", "default")["system_prompt"] == "AGENT"

    def test_acp_keeps_the_agent_default_until_targeted(self):
        """R5: no leak by default; an acp:* entry reaches minted sessions."""
        registry = _registry(CHAT_CONFIG)
        assert _resolved(registry, "acp", "sess-1")["system_prompt"] == "AGENT"

        config = {"channels": {**CHAT_CONFIG["channels"], "acp:*": {"system_prompt": "ACP"}}}
        registry = _registry(config)
        assert _resolved(registry, "acp", "sess-2")["system_prompt"] == "ACP"

    def test_wide_key_is_never_registered_as_a_channel(self):
        """R8: today ``signal:*`` would become a channel with scope ``*``."""
        registry = _registry(CHAT_CONFIG)
        assert [c.id for c in registry.all()] == []

    def test_prefix_matches_on_the_whole_id(self):
        """``signal:*`` is about Signal, not every transport named signal-ish."""
        registry = _registry({"channels": {"signal:*": {"system_prompt": "SIGNAL"}}})
        assert _resolved(registry, "signalx", "a")["system_prompt"] == "AGENT"


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_exact_entry_beats_wide_per_key(self):
        """R6: the exact entry overrides what it sets and nothing else."""
        config = {
            "channels": {
                "signal:*": {"system_prompt": "SIGNAL", "max_turns": 3},
                "signal:aci-1": {"max_turns": 7},
            }
        }
        registry = _registry(config)
        resolved = registry.resolve_config(registry.get("signal:aci-1"))
        assert resolved["max_turns"] == 7
        assert resolved["system_prompt"] == "SIGNAL"

    def test_every_scalar_key_resolves_through_a_wide_entry(self):
        """R7: a wide entry accepts every key an exact entry does. One
        assertion per ChannelConfig scalar field, so a field the overlay
        forgets fails here."""
        from dataclasses import fields

        from corvidae.channel import ReasoningHistory

        values = {
            "system_prompt": "WIDE",
            "max_context_tokens": 4000,
            "reasoning_history": "keep",
            "max_turns": 2,
        }
        scalar_fields = {f.name for f in fields(ChannelConfig)} - {"disabled_tools"}
        assert scalar_fields == set(values), "a new ChannelConfig field needs a case here"
        registry = _registry({"channels": {"signal:*": values}})
        resolved = _resolved(registry, "signal", "aci-15")
        assert resolved["system_prompt"] == "WIDE"
        assert resolved["max_context_tokens"] == 4000
        assert resolved["reasoning_history"] is ReasoningHistory.KEEP
        assert resolved["max_turns"] == 2

    def test_wide_beats_agent_and_agent_fills_the_rest(self):
        registry = _registry({"channels": {"signal:*": {"max_turns": 3}}})
        resolved = _resolved(registry, "signal", "aci-2")
        assert resolved["max_turns"] == 3
        assert resolved["system_prompt"] == "AGENT"
        assert resolved["max_context_tokens"] == 8000

    def test_runtime_override_beats_every_layer(self):
        registry = _registry({"channels": {"signal:*": {"max_turns": 3}}})
        channel = registry.get_or_create("signal", "aci-3")
        channel.runtime_overrides["max_turns"] = 12
        assert registry.resolve_config(channel)["max_turns"] == 12

    def test_signal_alias_config_beats_the_wide_entry(self):
        """D7: an E.164 alias's ChannelConfig arrives as ``config=`` and is
        the exact layer for the ACI channel."""
        registry = _registry({"channels": {"signal:*": {"system_prompt": "SIGNAL", "max_turns": 3}}})
        alias = ChannelConfig(max_turns=9)
        channel = registry.get_or_create("signal", "aci-4", config=alias)
        resolved = registry.resolve_config(channel)
        assert resolved["max_turns"] == 9
        assert resolved["system_prompt"] == "SIGNAL"

    def test_existing_exact_keys_with_a_star_inside_still_load(self):
        """R10: IRC allows ``*`` in channel names; only a trailing ``*``
        makes a wide key."""
        registry = _registry({"channels": {"irc:#c*t": {"max_turns": 2}}})
        channel = registry.get("irc:#c*t")
        assert channel is not None
        assert registry.resolve_config(channel)["max_turns"] == 2


# ---------------------------------------------------------------------------
# tools.disabled: union of every layer
# ---------------------------------------------------------------------------


class TestDisabledTools:
    def test_union_across_wide_chain_and_exact(self):
        """R7: a more specific entry cannot re-enable a tool."""
        config = {
            "channels": {
                "discord:*": {"tools": {"disabled": ["shell"]}},
                "discord:dm:*": {"tools": {"disabled": ["web_fetch"]}},
                "discord:dm:42": {"tools": {"disabled": ["read_file"]}},
            }
        }
        registry = _registry(config)
        channel = registry.get("discord:dm:42")
        assert channel.disabled_tools == frozenset({"shell", "web_fetch", "read_file"})
        # Coupled tools follow (shell takes task_pipeline with it).
        assert {"shell", "task_pipeline", "web_fetch", "read_file"} <= disabled_on(channel)

    def test_unmatched_channel_is_not_narrowed(self):
        registry = _registry({"channels": {"discord:*": {"tools": {"disabled": ["shell"]}}}})
        channel = registry.get_or_create("signal", "aci-5")
        assert channel.disabled_tools == frozenset()
        assert "shell" not in disabled_on(channel)

    def test_refusal_names_the_wide_key(self):
        registry = _registry({"channels": {"signal:*": {"tools": {"disabled": ["shell"]}}}})
        channel = registry.get_or_create("signal", "aci-6")
        message = refusal_message("shell", channel)
        assert "'shell'" in message
        assert "channels.signal:*.tools.disabled" in message
        assert channel.id in message

    def test_refusal_for_a_coupled_tool_names_the_wide_key(self):
        """``shell`` disabled widely takes ``task_pipeline`` with it; the
        refusal still names the key that did it."""
        registry = _registry({"channels": {"signal:*": {"tools": {"disabled": ["shell"]}}}})
        channel = registry.get_or_create("signal", "aci-16")
        assert "channels.signal:*.tools.disabled" in refusal_message("task_pipeline", channel)

    def test_refusal_names_the_exact_key_when_the_exact_entry_disabled_it(self):
        config = {
            "channels": {
                "signal:*": {"tools": {"disabled": ["web_fetch"]}},
                "signal:aci-7": {"tools": {"disabled": ["shell"]}},
            }
        }
        registry = _registry(config)
        message = refusal_message("shell", registry.get("signal:aci-7"))
        assert "channels.signal:aci-7.tools.disabled" in message
        assert "channels.signal:*.tools.disabled" not in message


# ---------------------------------------------------------------------------
# Validation: loud, named, all-or-nothing
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("key", ["signal*", "*", ":*", "sig*nal:*", "signal:**"])
    def test_malformed_wide_key_is_rejected_with_key_and_accepted_forms(self, key):
        registry = ChannelRegistry(dict(AGENT))
        with pytest.raises(ValueError) as excinfo:
            load_channel_config({"channels": {key: {"system_prompt": "x"}}}, registry)
        message = str(excinfo.value)
        assert repr(key) in message
        # The remedy: the accepted forms.
        assert "transport:*" in message
        assert "transport:<scope-prefix>*" in message

    def test_invalid_value_in_a_wide_entry_names_the_wide_key(self):
        registry = ChannelRegistry(dict(AGENT))
        with pytest.raises(ValueError, match=r"channels\.signal:\*"):
            load_channel_config(
                {"channels": {"signal:*": {"reasoning_history": "sometimes"}}}, registry
            )

    def test_invalid_tools_in_a_wide_entry_names_the_wide_key(self):
        registry = ChannelRegistry(dict(AGENT))
        with pytest.raises(ValueError, match=r"channels\.signal:\*\.tools"):
            load_channel_config(
                {"channels": {"signal:*": {"tools": {"disabled": "shell"}}}}, registry
            )

    def test_non_mapping_wide_entry_is_rejected(self):
        registry = ChannelRegistry(dict(AGENT))
        with pytest.raises(ValueError, match=r"signal:\*"):
            load_channel_config({"channels": {"signal:*": "SIGNAL"}}, registry)

    def test_invalid_wide_entry_applies_nothing(self):
        """A valid exact entry alongside an invalid wide one registers no
        channel and sets no wide entries."""
        registry = ChannelRegistry(dict(AGENT))
        config = {
            "channels": {
                "irc:#lex": {"max_turns": 3},
                "signal:*": {"reasoning_history": "sometimes"},
            }
        }
        with pytest.raises(ValueError):
            load_channel_config(config, registry)
        assert registry.all() == []
        assert _resolved(registry, "signal", "aci-8")["system_prompt"] == "AGENT"

    def test_invalid_exact_entry_keeps_the_previous_wide_entries(self):
        """D4: a valid wide entry beside an invalid exact one is not applied."""
        registry = _registry({"channels": {"signal:*": {"system_prompt": "OLD"}}})
        with pytest.raises(ValueError, match=r"channels\.irc:#lex"):
            load_channel_config(
                {
                    "channels": {
                        "signal:*": {"system_prompt": "NEW"},
                        "irc:#lex": {"reasoning_history": "sometimes"},
                    }
                },
                registry,
            )
        assert _resolved(registry, "signal", "aci-17")["system_prompt"] == "OLD"

    def test_invalid_reload_keeps_the_previous_wide_entries(self):
        registry = _registry({"channels": {"signal:*": {"system_prompt": "SIGNAL"}}})
        with pytest.raises(ValueError):
            load_channel_config({"channels": {"signal:*": {"max_turns": 1}, "x*": {}}}, registry)
        assert _resolved(registry, "signal", "aci-9")["system_prompt"] == "SIGNAL"


# ---------------------------------------------------------------------------
# Reload: wide entries reload like agent: does
# ---------------------------------------------------------------------------


class TestReload:
    def test_new_wide_entry_reaches_channels_created_after_the_reload(self):
        """R9."""
        registry = _registry({})
        load_channel_config(CHAT_CONFIG, registry)
        assert _resolved(registry, "signal", "aci-10")["system_prompt"] == "SIGNAL"

    def test_edited_scalar_applies_to_a_live_channel(self):
        """D5: scalars are re-resolved every turn, as agent: values are."""
        registry = _registry({"channels": {"signal:*": {"max_turns": 3}}})
        channel = registry.get_or_create("signal", "aci-11")
        load_channel_config({"channels": {"signal:*": {"max_turns": 5}}}, registry)
        assert registry.resolve_config(channel)["max_turns"] == 5

    def test_narrowing_wide_reload_applies_to_a_live_channel_at_once(self):
        registry = _registry({"channels": {"signal:*": {}}})
        channel = registry.get_or_create("signal", "aci-12")
        assert "shell" not in disabled_on(channel)
        load_channel_config({"channels": {"signal:*": {"tools": {"disabled": ["shell"]}}}}, registry)
        assert "shell" in disabled_on(channel)

    def test_reload_that_changes_a_live_channel_is_logged(self, caplog):
        """D5: a reload that changes what a live channel gets is never silent."""
        registry = _registry({"channels": {"signal:*": {}}})
        channel = registry.get_or_create("signal", "aci-14")
        with caplog.at_level(logging.INFO, logger="corvidae.channel"):
            load_channel_config(
                {"channels": {"signal:*": {"tools": {"disabled": ["shell"]}}}}, registry
            )
        assert any(
            r.levelno == logging.INFO
            and channel.id in r.getMessage()
            and "signal:*" in r.getMessage()
            for r in caplog.records
        )

    def test_removing_a_wide_entry_takes_effect_live(self):
        registry = _registry({"channels": {"signal:*": {"max_turns": 3}}})
        channel = registry.get_or_create("signal", "aci-13")
        load_channel_config({"channels": {}}, registry)
        assert registry.resolve_config(channel)["max_turns"] == 10
        assert channel.wide_entries == ()


# ---------------------------------------------------------------------------
# End to end through the Agent: the prompt the model actually receives
# ---------------------------------------------------------------------------


def _text_client() -> MagicMock:
    """An LLM client that answers every call with plain text."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=_make_text_response("ok"))
    return client


async def _system_sent(plugin, channel) -> str:
    """Send one message on ``channel``; return the system prompt the LLM got."""
    client = _text_client()
    plugin._client = client
    plugin._mark_started()
    await plugin.on_message(channel=channel, sender="u", text="hi", trace_id=mint_trace_id())
    await drain(plugin, channel)
    messages = client.chat.call_args.args[0]
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


async def _teardown(plugin, db) -> None:
    await plugin.pm.get_plugin("task").on_stop()
    await db.close()


class TestThroughTheAgent:
    async def test_signal_dm_conversation_starts_with_the_wide_prompt(self):
        plugin, _, db = await build_plugin_and_channel(agent_defaults=dict(AGENT))
        try:
            registry = plugin._registry
            load_channel_config(CHAT_CONFIG, registry)
            signal = registry.get_or_create("signal", "aci-20")
            web = registry.get_or_create("web", "default")
            assert await _system_sent(plugin, signal) == "SIGNAL"
            assert await _system_sent(plugin, web) == "AGENT"
        finally:
            await _teardown(plugin, db)

    async def test_prompt_edit_reaches_new_conversations_not_live_ones(self):
        """D5: a live conversation keeps the prompt it started with; a
        channel created after the reload starts with the new one."""
        plugin, _, db = await build_plugin_and_channel(agent_defaults=dict(AGENT))
        try:
            registry = plugin._registry
            load_channel_config({"channels": {"signal:*": {"system_prompt": "OLD"}}}, registry)
            live = registry.get_or_create("signal", "aci-21")
            assert await _system_sent(plugin, live) == "OLD"

            load_channel_config({"channels": {"signal:*": {"system_prompt": "NEW"}}}, registry)
            assert await _system_sent(plugin, live) == "OLD"
            fresh = registry.get_or_create("signal", "aci-22")
            assert await _system_sent(plugin, fresh) == "NEW"
        finally:
            await _teardown(plugin, db)

    async def test_unknown_name_warning_names_the_wide_key(self, caplog):
        """D6: a typo in a wide entry points the operator at the key that
        is actually in their config."""
        plugin, _, db = await build_plugin_and_channel(agent_defaults=dict(AGENT))

        async def read_file(path: str) -> str:
            """Read a file."""
            return "contents"

        plugin._tools = {"read_file": read_file}
        plugin._tool_schemas = [tool_to_schema(read_file)]
        try:
            registry = plugin._registry
            load_channel_config(
                {"channels": {"signal:*": {"tools": {"disabled": ["shel"]}}}}, registry
            )
            channel = registry.get_or_create("signal", "aci-24")
            with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
                await _system_sent(plugin, channel)
            warnings = [
                r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "shel" in r.getMessage()
            ]
            assert warnings, "expected a warning naming the unknown tool"
            assert any("channels.signal:*.tools.disabled" in w for w in warnings)
            assert not any(f"channels.{channel.id}.tools.disabled" in w for w in warnings)
        finally:
            await _teardown(plugin, db)

    async def test_wide_disabled_tool_is_hidden_and_refused(self, caplog):
        plugin, _, db = await build_plugin_and_channel(agent_defaults=dict(AGENT))
        calls: list[str] = []

        async def shell(command: str) -> str:
            """Run a command."""
            calls.append("shell")
            return "ran"

        async def read_file(path: str) -> str:
            """Read a file."""
            calls.append("read_file")
            return "contents"

        plugin._tools = {"shell": shell, "read_file": read_file}
        plugin._tool_schemas = [tool_to_schema(fn) for fn in plugin._tools.values()]
        try:
            registry = plugin._registry
            load_channel_config(
                {"channels": {"signal:*": {"tools": {"disabled": ["shell"]}}}}, registry
            )
            channel = registry.get_or_create("signal", "aci-23")
            client = MagicMock()
            client.chat = AsyncMock(side_effect=[
                _make_tool_call_response([_make_tool_call("call_1", "shell", {"command": "id"})]),
                _make_text_response("done"),
            ])
            plugin._client = client
            plugin._mark_started()
            with caplog.at_level(logging.WARNING, logger="corvidae.agent"):
                await plugin.on_message(
                    channel=channel, sender="u", text="go", trace_id=mint_trace_id()
                )
                await drain(plugin, channel)
                await drain_task_queue(plugin)
                await drain(plugin, channel)

            # The operator's log names the key that disabled it (D6).
            refusals = [
                r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "refused tool shell" in r.getMessage()
            ]
            assert refusals
            assert all("channels.signal:*.tools.disabled" in m for m in refusals)

            first = client.chat.call_args_list[0]
            names = [s["function"]["name"] for s in first.kwargs["tools"] or []]
            assert names == ["read_file"]
            assert calls == []
            messages = client.chat.call_args_list[1].args[0]
            (tool_msg,) = [m for m in messages if m.get("role") == "tool"]
            assert "channels.signal:*.tools.disabled" in tool_msg["content"]
        finally:
            await _teardown(plugin, db)
