"""Tests for sherman.channel: ChannelConfig, Channel, ChannelRegistry, load_channel_config."""

import pytest
from time import time
from sherman.channel import Channel, ChannelConfig, ChannelRegistry, load_channel_config


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
            "keep_thinking_in_history": False,
        }
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == "Be helpful."
        assert result["max_context_tokens"] == 8000
        assert result["keep_thinking_in_history"] is False

    def test_resolve_channel_overrides_win(self):
        """Channel-level values override agent defaults for all three fields."""
        cfg = ChannelConfig(
            system_prompt="Channel prompt.",
            max_context_tokens=4000,
            keep_thinking_in_history=False,
        )
        agent_defaults = {
            "system_prompt": "Agent prompt.",
            "max_context_tokens": 24000,
            "keep_thinking_in_history": True,
        }
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == "Channel prompt."
        assert result["max_context_tokens"] == 4000
        assert result["keep_thinking_in_history"] is False

    def test_resolve_partial_override(self):
        """Only some fields overridden; rest fall back to agent defaults."""
        cfg = ChannelConfig(system_prompt="Custom prompt.")
        agent_defaults = {
            "system_prompt": "Agent prompt.",
            "max_context_tokens": 16000,
            "keep_thinking_in_history": True,
        }
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == "Custom prompt."
        assert result["max_context_tokens"] == 16000
        assert result["keep_thinking_in_history"] is True

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

    def test_resolve_false_keep_thinking_is_not_none(self):
        """`False` keep_thinking_in_history is an intentional override."""
        cfg = ChannelConfig(keep_thinking_in_history=False)
        agent_defaults = {"keep_thinking_in_history": True}
        result = cfg.resolve(agent_defaults)
        assert result["keep_thinking_in_history"] is False

    def test_resolve_missing_agent_defaults(self):
        """Empty agent_defaults dict falls back to hardcoded defaults."""
        cfg = ChannelConfig()
        result = cfg.resolve({})
        assert result["system_prompt"] == "You are a helpful assistant."
        assert result["max_context_tokens"] == 24000
        assert result["keep_thinking_in_history"] is False


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
        assert ch.config.keep_thinking_in_history is None


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
                "signal:+15551234567": {"keep_thinking_in_history": False},
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
        assert signal.config.keep_thinking_in_history is False

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
        in _ensure_conversation via resolve_system_prompt().
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
            "keep_thinking_in_history": False,
        }
        cfg = ChannelConfig()  # no override
        result = cfg.resolve(agent_defaults)
        assert result["system_prompt"] == ["soul.md", "irc.md"]
