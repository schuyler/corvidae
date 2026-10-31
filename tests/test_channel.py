"""Tests for corvidae.channel: ChannelConfig, Channel, ChannelRegistry, load_channel_config."""

import pytest
from time import time
from corvidae.channel import (
    Channel,
    ChannelConfig,
    CHANNEL_ENTRY_KEYS,
    ChannelRegistry,
    ReasoningHistory,
    apply_channel_config,
    load_channel_config,
    parse_channel_config,
    validate_agent_config,
)


# ---------------------------------------------------------------------------
# Section 6.1 — ChannelConfig.resolve tests
# ---------------------------------------------------------------------------

class TestChannelConfigResolve:
    def test_resolve_all_defaults(self):
        """ChannelConfig() with no overrides resolves all fields from agent_defaults."""
        cfg = ChannelConfig()
        agent_defaults = {
            "system_prompt": "Be helpful.",
            "max_context_tokens": 8000,
            "reasoning_history": "drop",
        }
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == "Be helpful."
        assert result["max_context_tokens"] == 8000
        assert result["reasoning_history"] == "drop"

    def test_resolve_channel_overrides_win(self):
        """Channel-level values override agent defaults for all three fields."""
        cfg = ChannelConfig(
            system_prompt="Channel prompt.",
            max_context_tokens=4000,
            reasoning_history="drop",
        )
        agent_defaults = {
            "system_prompt": "Agent prompt.",
            "max_context_tokens": 24000,
            "reasoning_history": "keep",
        }
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == "Channel prompt."
        assert result["max_context_tokens"] == 4000
        assert result["reasoning_history"] == "drop"

    def test_resolve_partial_override(self):
        """Only some fields overridden; rest fall back to agent defaults."""
        cfg = ChannelConfig(system_prompt="Custom prompt.")
        agent_defaults = {
            "system_prompt": "Agent prompt.",
            "max_context_tokens": 16000,
            "reasoning_history": "keep",
        }
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == "Custom prompt."
        assert result["max_context_tokens"] == 16000
        assert result["reasoning_history"] == "keep"

    def test_resolve_empty_string_is_not_none(self):
        """Empty string system_prompt is an intentional override, not a fallback."""
        cfg = ChannelConfig(system_prompt="")
        agent_defaults = {"system_prompt": "Agent prompt."}
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == ""

    def test_resolve_zero_max_context_tokens_is_not_none(self):
        """Zero max_context_tokens is an intentional override."""
        cfg = ChannelConfig(max_context_tokens=0)
        agent_defaults = {"max_context_tokens": 24000}
        result = cfg.resolve(agent_defaults)
        assert result["max_context_tokens"] == 0

    def test_resolve_channel_reasoning_history_wins(self):
        """A channel-level reasoning_history beats the agent default."""
        cfg = ChannelConfig(reasoning_history="drop")
        agent_defaults = {"reasoning_history": "keep"}
        result = cfg.resolve(agent_defaults)
        assert result["reasoning_history"] == "drop"

    def test_resolve_missing_agent_defaults(self):
        """Empty agent_defaults dict falls back to hardcoded defaults."""
        cfg = ChannelConfig()
        result = cfg.resolve({})
        assert result["system_prompt"] == "You are a helpful assistant."
        assert result["max_context_tokens"] == 24000
        assert result["reasoning_history"] == "drop"

    def test_resolve_returns_reasoning_history_enum(self):
        """The resolved value is a ReasoningHistory member, so consumers
        compare against one definition of the three policies."""
        result = ChannelConfig(reasoning_history="exchange").resolve({})
        assert result["reasoning_history"] is ReasoningHistory.EXCHANGE

    @pytest.mark.parametrize("bad", [True, False, "strip", "Drop"])
    def test_resolve_rejects_invalid_reasoning_history(self, bad):
        """Anything outside drop/exchange/keep — including the booleans the
        old keep_thinking_in_history key took — raises instead of silently
        picking a policy."""
        with pytest.raises(ValueError, match="reasoning_history"):
            ChannelConfig().resolve({"reasoning_history": bad})


# ---------------------------------------------------------------------------
# validate_agent_config — fail loudly on stale or bad reasoning config (#57)
# ---------------------------------------------------------------------------

class TestValidateAgentConfig:
    @pytest.mark.parametrize("value", ["drop", "exchange", "keep"])
    def test_accepts_valid_values(self, value):
        validate_agent_config({"reasoning_history": value})

    def test_accepts_missing_key(self):
        validate_agent_config({})

    def test_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="reasoning_history"):
            validate_agent_config({"reasoning_history": "sometimes"})

    def test_rejects_removed_keep_thinking_key(self):
        """The old boolean key was removed, not aliased; a stale agent.yaml
        must not quietly fall back to the default."""
        with pytest.raises(ValueError, match="keep_thinking_in_history"):
            validate_agent_config({"keep_thinking_in_history": True})


# ---------------------------------------------------------------------------
# Section 6.2 — Channel tests
# ---------------------------------------------------------------------------

class TestChannel:
    def test_channel_id_property(self):
        """`channel.id` returns `"transport:scope"`."""
        ch = Channel(transport="irc", scope="#lex")
        assert ch.id == "irc:#lex"

    def test_channel_matches_transport(self):
        """Returns True for matching transport, False otherwise."""
        ch = Channel(transport="signal", scope="+15551234567")
        assert ch.matches_transport("signal") is True
        assert ch.matches_transport("irc") is False

    def test_channel_touch_updates_last_active(self):
        """`touch()` updates `last_active` to a more recent time."""
        ch = Channel(transport="cli", scope="local")
        before = ch.last_active
        # Force time to advance by manipulating last_active directly so the
        # test is deterministic without sleeping.
        ch.last_active = before - 1.0
        ch.touch()
        assert ch.last_active >= before

    def test_channel_default_conversation_is_none(self):
        """Default `conversation` is None."""
        ch = Channel(transport="irc", scope="#general")
        assert ch.conversation is None

    def test_channel_default_config(self):
        """Default config has all None fields."""
        ch = Channel(transport="irc", scope="#general")
        assert ch.config.system_prompt is None
        assert ch.config.max_context_tokens is None
        assert ch.config.reasoning_history is None


# ---------------------------------------------------------------------------
# Section 6.3 — ChannelRegistry tests
# ---------------------------------------------------------------------------

class TestChannelRegistry:
    def test_get_or_create_new_channel(self):
        """Creates a new channel on first call."""
        registry = ChannelRegistry({})
        ch = registry.get_or_create("irc", "#lex")
        assert ch.transport == "irc"
        assert ch.scope == "#lex"
        assert ch.id == "irc:#lex"

    def test_get_or_create_returns_existing(self):
        """Second call with same transport+scope returns same object."""
        registry = ChannelRegistry({})
        ch1 = registry.get_or_create("irc", "#lex")
        ch2 = registry.get_or_create("irc", "#lex")
        assert ch1 is ch2

    def test_get_or_create_touches_channel(self):
        """`get_or_create` calls `touch()` on both new and existing channels."""
        registry = ChannelRegistry({})
        ch = registry.get_or_create("irc", "#lex")
        # Backdate last_active to confirm it's updated on subsequent call.
        ch.last_active = ch.last_active - 10.0
        before = ch.last_active
        registry.get_or_create("irc", "#lex")
        assert ch.last_active > before

    def test_get_or_create_with_config(self):
        """Config is applied to new channel."""
        registry = ChannelRegistry({})
        cfg = ChannelConfig(system_prompt="Custom.", max_context_tokens=2000)
        ch = registry.get_or_create("irc", "#lex", config=cfg)
        assert ch.config.system_prompt == "Custom."
        assert ch.config.max_context_tokens == 2000

    def test_get_or_create_ignores_config_for_existing_channel(self):
        """Config passed to get_or_create for an existing channel is ignored."""
        registry = ChannelRegistry({})
        cfg1 = ChannelConfig(system_prompt="First.")
        ch1 = registry.get_or_create("irc", "#lex", config=cfg1)
        cfg2 = ChannelConfig(system_prompt="Second.")
        ch2 = registry.get_or_create("irc", "#lex", config=cfg2)
        assert ch2.config.system_prompt == "First."

    def test_get_existing(self):
        """`get()` returns channel by string ID."""
        registry = ChannelRegistry({})
        registry.get_or_create("signal", "+15551234567")
        ch = registry.get("signal:+15551234567")
        assert ch is not None
        assert ch.id == "signal:+15551234567"

    def test_get_nonexistent(self):
        """`get()` returns None for unknown ID."""
        registry = ChannelRegistry({})
        assert registry.get("irc:#nothere") is None

    def test_resolve_config(self):
        """Delegates to `channel.config.resolve()` with registry defaults."""
        agent_defaults = {"system_prompt": "Agent.", "max_context_tokens": 12000}
        registry = ChannelRegistry(agent_defaults)
        cfg = ChannelConfig(system_prompt="Channel.")
        ch = registry.get_or_create("irc", "#lex", config=cfg)
        result = registry.resolve_config(ch)
        assert result["system_prompt"] == "Channel."
        assert result["max_context_tokens"] == 12000

    def test_all(self):
        """Returns all registered channels."""
        registry = ChannelRegistry({})
        registry.get_or_create("irc", "#a")
        registry.get_or_create("irc", "#b")
        registry.get_or_create("signal", "+1")
        channels = registry.all()
        assert len(channels) == 3
        ids = {ch.id for ch in channels}
        assert ids == {"irc:#a", "irc:#b", "signal:+1"}

    def test_by_transport(self):
        """Filters channels by transport name."""
        registry = ChannelRegistry({})
        registry.get_or_create("irc", "#a")
        registry.get_or_create("irc", "#b")
        registry.get_or_create("signal", "+1")
        irc_channels = registry.by_transport("irc")
        assert len(irc_channels) == 2
        assert all(ch.transport == "irc" for ch in irc_channels)
        signal_channels = registry.by_transport("signal")
        assert len(signal_channels) == 1
        assert signal_channels[0].scope == "+1"

    def test_get_or_create_touches_new_channel(self):
        """touch() is called even on initial channel creation."""
        registry = ChannelRegistry({})
        before = time()
        ch = registry.get_or_create("irc", "#lex")
        assert ch.last_active >= before


# ---------------------------------------------------------------------------
# Section 6.4 — load_channel_config tests
# ---------------------------------------------------------------------------

class TestLoadChannelConfig:
    def test_load_channel_config_basic(self):
        """Pre-registers channels from config."""
        config = {
            "channels": {
                "irc:#lex": {"system_prompt": "IRC prompt.", "max_context_tokens": 8000},
                "signal:+15551234567": {"reasoning_history": "exchange"},
            }
        }
        registry = ChannelRegistry({})
        load_channel_config(config, registry)

        irc = registry.get("irc:#lex")
        assert irc is not None
        assert irc.config.system_prompt == "IRC prompt."
        assert irc.config.max_context_tokens == 8000

        signal = registry.get("signal:+15551234567")
        assert signal is not None
        assert signal.config.reasoning_history == "exchange"

    def test_load_channel_config_rejects_removed_keep_thinking_key(self):
        """A channel entry still using keep_thinking_in_history raises."""
        config = {"channels": {"irc:#lex": {"keep_thinking_in_history": False}}}
        with pytest.raises(ValueError, match="keep_thinking_in_history"):
            load_channel_config(config, ChannelRegistry({}))

    def test_load_channel_config_rejects_invalid_reasoning_history(self):
        """An invalid per-channel reasoning_history raises at load time."""
        config = {"channels": {"irc:#lex": {"reasoning_history": "yes"}}}
        with pytest.raises(ValueError, match="reasoning_history"):
            load_channel_config(config, ChannelRegistry({}))

    def test_parse_channel_config_touches_no_registry(self):
        """#208 R3: parse_channel_config is pure; it returns the exact and
        wide entries and makes no call on any registry."""
        from unittest.mock import MagicMock

        config = {"channels": {
            "irc:#lex": {"max_turns": 3},
            "signal:*": {"tools": {"disabled": ["shell"]}},
        }}
        registry = MagicMock()
        parsed = parse_channel_config(config)
        assert registry.method_calls == []
        assert [p[0] for p in parsed.exact] == ["irc:#lex"]
        assert [w.key for w in parsed.wide] == ["signal:*"]
        # Applying the parsed result is what touches the registry.
        apply_channel_config(parsed, registry)
        registry.set_wide_entries.assert_called_once()

    @pytest.mark.parametrize("config", [
        {"channels": {"nocolon": {}}},
        {"channels": {"irc:#a": "text"}},
        {"channels": {"irc:#a": {"keep_thinking_in_history": True}}},
        {"channels": {"irc:#a": {"reasoning_history": "yes"}}},
        {"channels": {"irc:#a": {"tools": ["shell"]}}},
        {"channels": {"signal:*x*": {}}},
    ])
    def test_parse_channel_config_raises_on_malformed(self, config):
        """#208 R3: parse_channel_config rejects what load_channel_config did."""
        with pytest.raises(ValueError):
            parse_channel_config(config)

    def test_apply_channel_config_never_raises_value_error(self):
        """#208 risk: apply on parsed input cannot fail validation, so a reload
        that parsed cleanly cannot half-apply."""
        registry = ChannelRegistry({})
        parsed = parse_channel_config({"channels": {
            "irc:#a": {"max_turns": 2}, "irc:*": {"max_turns": 4},
            "signal:+1555": None,
        }})
        apply_channel_config(parsed, registry)
        apply_channel_config(parsed, registry)  # a reload over existing channels
        assert registry.get("irc:#a") is not None

    @pytest.mark.parametrize("key", ["irc:#a", "signal:*"])
    def test_unknown_entry_key_is_rejected(self, key):
        """#205 R1: an unknown key in an exact or wide entry raises, naming
        channels.<id>.<key>, the value and every accepted key."""
        config = {"channels": {key: {"max_turn": 5}}}
        with pytest.raises(ValueError) as excinfo:
            load_channel_config(config, ChannelRegistry({}))
        msg = str(excinfo.value)
        assert f"channels.{key}.max_turn" in msg
        assert "5" in msg
        for accepted in ("system_prompt", "max_context_tokens", "max_turns",
                         "reasoning_history", "tools"):
            assert accepted in msg

    def test_several_unknown_entry_keys_named_sorted_once(self):
        """#205 R1: all unknown keys appear in one message, sorted."""
        config = {"channels": {"irc:#a": {"zeta": 1, "max_tokens": 200, "max_turns": 3}}}
        with pytest.raises(ValueError) as excinfo:
            parse_channel_config(config)
        msg = str(excinfo.value)
        assert msg.index("channels.irc:#a.max_tokens") < msg.index("channels.irc:#a.zeta")
        assert "200" in msg

    def test_unknown_entry_key_value_is_truncated(self):
        """#205: a long rejected value is repr-truncated, not dumped."""
        config = {"channels": {"irc:#a": {"bogus": "x" * 5000}}}
        with pytest.raises(ValueError) as excinfo:
            parse_channel_config(config)
        assert "channels.irc:#a.bogus" in str(excinfo.value)
        assert len(str(excinfo.value)) < 600

    def test_removed_keep_thinking_key_keeps_migration_message(self):
        """#205 R4: the specific migration text wins over the generic one."""
        config = {"channels": {"irc:#a": {"keep_thinking_in_history": True}}}
        with pytest.raises(ValueError, match="use reasoning_history") as excinfo:
            parse_channel_config(config)
        assert "unknown key" not in str(excinfo.value)

    def test_entry_keys_match_channel_config_fields(self):
        """CHANNEL_ENTRY_KEYS must track ChannelConfig, or a new field would be
        rejected as unknown."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ChannelConfig)}
        assert set(CHANNEL_ENTRY_KEYS) == (fields - {"disabled_tools"}) | {"tools"}

    @pytest.mark.parametrize("path", [
        "agent.yaml.example", "agent.minimal.yaml.example", "harness/buster.yaml.in",
    ])
    def test_shipped_channel_entries_still_load(self, path):
        """#205 R5: every channels: entry shipped in the repo, including the
        commented examples of agent.yaml.example, passes validation."""
        import re
        from pathlib import Path

        import yaml

        text = (Path(__file__).resolve().parent.parent / path).read_text()
        # Uncomment example entries: '  # key:' and '  #   indented' lines
        # (prose comments have spaces before any colon, so they stay comments).
        lines = ["channels:"]
        in_section = False
        for line in text.splitlines():
            # Only the top-level channels: section is of interest.
            if line == "channels:":
                in_section = True
                continue
            if in_section and line and not line.startswith((" ", "#")):
                break
            if not in_section:
                continue
            m = re.match(r"^  # (\s*\S+:.*|\s{2,}.*)$", line)
            if m and (m.group(1).startswith(" ") or " " not in m.group(1).split(":")[0]):
                line = "  " + m.group(1)
            lines.append(line)
        doc = yaml.safe_load("\n".join(lines))
        assert doc["channels"], path
        parse_channel_config({"channels": doc["channels"]})

    def test_load_channel_config_no_channels_section(self):
        """Missing 'channels' key is a no-op."""
        config = {"agent": {"system_prompt": "Hello."}}
        registry = ChannelRegistry({})
        load_channel_config(config, registry)
        assert registry.all() == []

    def test_load_channel_config_empty_channels(self):
        """Empty channels dict is a no-op."""
        config = {"channels": {}}
        registry = ChannelRegistry({})
        load_channel_config(config, registry)
        assert registry.all() == []

    def test_load_channel_config_invalid_key_no_colon(self):
        """Key without colon raises ValueError."""
        config = {"channels": {"nocolon": {}}}
        registry = ChannelRegistry({})
        with pytest.raises(ValueError, match="nocolon"):
            load_channel_config(config, registry)

    def test_load_channel_config_colon_in_scope(self):
        """Key like 'signal:+1:555' splits correctly on first colon only."""
        config = {"channels": {"signal:+1:555": {}}}
        registry = ChannelRegistry({})
        load_channel_config(config, registry)
        ch = registry.get("signal:+1:555")
        assert ch is not None
        assert ch.transport == "signal"
        assert ch.scope == "+1:555"

    def test_load_channel_config_null_overrides(self):
        """Null YAML value for a channel is treated as empty config."""
        config = {"channels": {"irc:#lex": None}}
        registry = ChannelRegistry({})
        load_channel_config(config, registry)
        ch = registry.get("irc:#lex")
        assert ch is not None
        assert ch.config.system_prompt is None


# ---------------------------------------------------------------------------
# Section 6.5 — system_prompt list passthrough (Phase 2.5)
# ---------------------------------------------------------------------------


class TestChannelConfigListSystemPrompt:
    def test_resolve_list_system_prompt_passthrough(self):
        """ChannelConfig with a list system_prompt preserves the list through resolve().

        The list is NOT resolved to a string at this layer — that happens later
        in Agent._process_queue_item via resolve_system_prompt().
        """
        cfg = ChannelConfig(system_prompt=["a.md", "b.md"])
        result = cfg.resolve({})
        assert result["system_prompt"] == ["a.md", "b.md"]

    def test_resolve_list_agent_default_passthrough(self):
        """Agent-level list system_prompt is passed through when the channel
        has no override (system_prompt is None).

        resolve() should not attempt to read files — the raw list is preserved.
        """
        agent_defaults = {
            "system_prompt": ["soul.md", "irc.md"],
            "max_context_tokens": 8000,
            "reasoning_history": "drop",
        }
        cfg = ChannelConfig()  # no override
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == ["soul.md", "irc.md"]


# ---------------------------------------------------------------------------
# Item 8 red-phase: resolve_system_prompt must be importable from corvidae.channel
# ---------------------------------------------------------------------------


class TestResolveSystemPromptInChannel:
    def test_resolve_system_prompt_importable_from_channel(self):
        """resolve_system_prompt must be importable from corvidae.channel after relocation.

        RED phase: this test fails while the function still lives in corvidae.conversation.
        It passes once the function is moved to corvidae.channel.
        """
        from corvidae.channel import resolve_system_prompt  # noqa: F401

    def test_resolve_system_prompt_basic_string(self):
        """resolve_system_prompt imported from corvidae.channel works for a string input.

        RED phase: fails until the function is available in corvidae.channel.
        """
        from pathlib import Path

        from corvidae.channel import resolve_system_prompt

        result = resolve_system_prompt("You are a helpful assistant.", Path("/tmp"))
        assert result == "You are a helpful assistant."

    def test_conversation_module_removed(self):
        """corvidae.conversation must not exist after the context/persistence split."""
        import importlib
        import sys
        # Remove cached reference if present
        sys.modules.pop("corvidae.conversation", None)
        try:
            importlib.import_module("corvidae.conversation")
            assert False, "corvidae.conversation should not be importable after deletion"
        except ModuleNotFoundError:
            pass  # expected

    def test_prompt_logger_in_channel(self):
        """_prompt_logger must live in corvidae.channel after relocation."""
        import corvidae.channel as mod

        assert hasattr(mod, "_prompt_logger"), (
            "channel.py must define _prompt_logger"
        )
        assert mod._prompt_logger.name == "corvidae.prompt"


class TestChannelsShapeAndScalarTypes:
    """#242: channels: section shape and scalar types fail as ValueError at parse time."""

    @pytest.mark.parametrize("section", [[{"irc:#x": {}}], "irc:#x", 5])
    def test_non_mapping_section_rejected(self, section):
        """A list/scalar channels: section names the section and expected type."""
        from corvidae.channel import parse_channel_config

        with pytest.raises(ValueError) as exc:
            parse_channel_config({"channels": section})
        assert "channels" in str(exc.value)
        assert "mapping" in str(exc.value)

    def test_null_section_is_empty(self):
        """An empty (null) channels: section means no channels."""
        from corvidae.channel import parse_channel_config

        parsed = parse_channel_config({"channels": None})
        assert parsed.exact == () and parsed.wide == ()

    def test_non_string_key_rejected(self):
        """A non-string YAML key is named with its type and the fix."""
        from corvidae.channel import parse_channel_config

        with pytest.raises(ValueError) as exc:
            parse_channel_config({"channels": {123: {}}})
        msg = str(exc.value)
        assert "channels.123" in msg and "int" in msg and "string" in msg

    @pytest.mark.parametrize("key", ["max_turns", "max_context_tokens"])
    @pytest.mark.parametrize("bad", ["five", 1.5, True, [1]])
    def test_scalar_must_be_integer(self, key, bad):
        """A non-integer scalar names channels.<id>.<key>, the value and 'integer'."""
        from corvidae.channel import parse_channel_config

        with pytest.raises(ValueError) as exc:
            parse_channel_config({"channels": {"irc:#x": {key: bad}}})
        msg = str(exc.value)
        assert f"channels.irc:#x.{key}" in msg
        assert repr(bad) in msg
        assert "integer" in msg

    def test_integer_and_null_scalars_accepted(self):
        """Integers and null (inherit) still parse."""
        from corvidae.channel import parse_channel_config

        parsed = parse_channel_config(
            {"channels": {"irc:#x": {"max_turns": 5, "max_context_tokens": None}}}
        )
        assert parsed.exact[0][3].max_turns == 5
