"""Tests for the runtime settings tool feature.

Covers:
- set_settings tool: valid params applied to channel.runtime_overrides
- set_settings tool: blocked params rejected (whole call rejected if any key blocked)
- set_settings tool: null clears override (del from runtime_overrides)
- set_settings tool: null for blocked key is rejected
- ChannelConfig.resolve() with runtime_overrides parameter (step 4 in resolution order)
- run_agent_turn passes extra_body through to client.chat()
- FRAMEWORK_KEYS derived from dataclasses.fields(ChannelConfig) filters correctly
- Default blocklist always includes "system_prompt" even if operator config doesn't list it
- Partial updates: settings merge into existing overrides, not replace

All tests are expected to FAIL until the implementation exists.
"""

import dataclasses
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.tool import ToolContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(runtime_overrides=None):
    """Return a minimal Channel with optional pre-seeded runtime_overrides."""
    channel = Channel(transport="test", scope="scope1", config=ChannelConfig())
    if runtime_overrides is not None:
        channel.runtime_overrides = runtime_overrides
    return channel


def _make_tool_context(channel):
    """Return a ToolContext wired to the given channel."""
    return ToolContext(
        channel=channel,
        tool_call_id="call_test",
        task_queue=None,
    )


def _get_set_settings_fn(immutable_settings=None):
    """Instantiate RuntimeSettingsPlugin and return the registered set_settings callable."""
    from corvidae.tool import Tool
    from corvidae.tools.settings import RuntimeSettingsPlugin

    if immutable_settings is None:
        immutable_settings = set()
    plugin = RuntimeSettingsPlugin(immutable_settings=immutable_settings)
    registry = []
    plugin.register_tools(tool_registry=registry)
    # Find set_settings in registry
    for item in registry:
        name = item.name if isinstance(item, Tool) else item.__name__
        if name == "set_settings":
            fn = item.fn if isinstance(item, Tool) else item
            return fn
    raise AssertionError("set_settings not found in registry")


# ---------------------------------------------------------------------------
# Channel.runtime_overrides field
# ---------------------------------------------------------------------------


class TestChannelRuntimeOverrides:
    def test_channel_has_runtime_overrides_field(self):
        """Channel must have a runtime_overrides field that defaults to an empty dict."""
        channel = Channel(transport="test", scope="scope1", config=ChannelConfig())
        assert hasattr(channel, "runtime_overrides"), (
            "Channel must have a runtime_overrides attribute"
        )
        assert isinstance(channel.runtime_overrides, dict), (
            "runtime_overrides must be a dict"
        )
        assert channel.runtime_overrides == {}, (
            "runtime_overrides must default to empty dict"
        )

    def test_runtime_overrides_is_independent_per_instance(self):
        """Each Channel instance must have its own runtime_overrides dict (not shared)."""
        ch1 = Channel(transport="test", scope="a", config=ChannelConfig())
        ch2 = Channel(transport="test", scope="b", config=ChannelConfig())
        ch1.runtime_overrides["temperature"] = 0.5
        assert ch2.runtime_overrides == {}, (
            "Mutating ch1.runtime_overrides must not affect ch2.runtime_overrides"
        )

    def test_runtime_overrides_uses_default_factory(self):
        """runtime_overrides must be declared with field(default_factory=dict)."""
        fields = {f.name: f for f in dataclasses.fields(Channel)}
        assert "runtime_overrides" in fields, "runtime_overrides must be a dataclass field"
        field_obj = fields["runtime_overrides"]
        assert field_obj.default_factory is dict, (  # type: ignore[union-attr]
            "runtime_overrides must use field(default_factory=dict)"
        )


# ---------------------------------------------------------------------------
# ChannelConfig.resolve() with runtime_overrides
# ---------------------------------------------------------------------------


class TestChannelConfigResolveWithOverrides:
    def test_resolve_accepts_runtime_overrides_kwarg(self):
        """ChannelConfig.resolve() must accept a runtime_overrides keyword argument."""
        config = ChannelConfig()
        agent_defaults = {}
        resolved = config.resolve(agent_defaults, runtime_overrides={})
        assert isinstance(resolved, dict)

    def test_runtime_overrides_win_over_channel_config(self):
        """runtime_overrides (step 4) must override per-channel ChannelConfig (step 3)."""
        config = ChannelConfig(max_turns=5)
        agent_defaults = {}
        resolved = config.resolve(agent_defaults, runtime_overrides={"max_turns": 20})
        assert resolved["max_turns"] == 20, (
            "runtime_overrides must override ChannelConfig field"
        )

    def test_runtime_overrides_win_over_agent_defaults(self):
        """runtime_overrides must override agent_defaults (step 2)."""
        config = ChannelConfig()
        agent_defaults = {"max_turns": 7}
        resolved = config.resolve(agent_defaults, runtime_overrides={"max_turns": 15})
        assert resolved["max_turns"] == 15

    def test_runtime_overrides_win_over_builtin_defaults(self):
        """runtime_overrides must override built-in defaults (step 1)."""
        config = ChannelConfig()
        agent_defaults = {}
        resolved = config.resolve(agent_defaults, runtime_overrides={"max_turns": 99})
        assert resolved["max_turns"] == 99

    def test_empty_runtime_overrides_does_not_change_result(self):
        """Passing an empty runtime_overrides must produce the same result as no overrides."""
        config = ChannelConfig(max_turns=3)
        agent_defaults = {"max_context_tokens": 8000}
        resolved_without = config.resolve(agent_defaults)
        resolved_with = config.resolve(agent_defaults, runtime_overrides={})
        assert resolved_without == resolved_with

    def test_multiple_keys_in_runtime_overrides(self):
        """Multiple keys in runtime_overrides must all be applied."""
        config = ChannelConfig()
        agent_defaults = {}
        resolved = config.resolve(
            agent_defaults,
            runtime_overrides={"max_turns": 50, "max_context_tokens": 100},
        )
        assert resolved["max_turns"] == 50
        assert resolved["max_context_tokens"] == 100

    def test_non_framework_keys_in_overrides_pass_through(self):
        """LLM inference params (e.g. temperature) in runtime_overrides are included
        in the resolved dict even though they are not ChannelConfig fields."""
        config = ChannelConfig()
        agent_defaults = {}
        resolved = config.resolve(
            agent_defaults,
            runtime_overrides={"temperature": 0.9},
        )
        assert resolved.get("temperature") == 0.9


# ---------------------------------------------------------------------------
# ChannelRegistry.resolve_config passes runtime_overrides
# ---------------------------------------------------------------------------


class TestChannelRegistryResolveConfig:
    def test_resolve_config_passes_runtime_overrides(self):
        """ChannelRegistry.resolve_config must pass channel.runtime_overrides to resolve()."""
        from corvidae.channel import ChannelRegistry

        channel = _make_channel(runtime_overrides={"max_turns": 42})
        registry = ChannelRegistry(agent_defaults={})

        resolved = registry.resolve_config(channel)
        assert resolved["max_turns"] == 42, (
            "ChannelRegistry.resolve_config must thread channel.runtime_overrides through"
        )


# ---------------------------------------------------------------------------
# FRAMEWORK_KEYS in agent.py
# ---------------------------------------------------------------------------


class TestFrameworkKeys:
    def test_framework_keys_exists_in_agent_module(self):
        """agent.py must export FRAMEWORK_KEYS at module level."""
        from corvidae import agent as agent_module
        assert hasattr(agent_module, "FRAMEWORK_KEYS"), (
            "corvidae.agent must export FRAMEWORK_KEYS"
        )

    def test_framework_keys_is_a_set(self):
        """FRAMEWORK_KEYS must be a set."""
        from corvidae.agent import FRAMEWORK_KEYS
        assert isinstance(FRAMEWORK_KEYS, set)

    def test_framework_keys_contains_channelconfig_fields(self):
        """FRAMEWORK_KEYS must contain all field names from ChannelConfig."""
        from corvidae.agent import FRAMEWORK_KEYS
        expected = {f.name for f in dataclasses.fields(ChannelConfig)}
        assert expected.issubset(FRAMEWORK_KEYS), (
            f"FRAMEWORK_KEYS must contain all ChannelConfig fields. "
            f"Missing: {expected - FRAMEWORK_KEYS}"
        )

    def test_framework_keys_does_not_include_temperature(self):
        """FRAMEWORK_KEYS must NOT include LLM inference params like 'temperature'."""
        from corvidae.agent import FRAMEWORK_KEYS
        assert "temperature" not in FRAMEWORK_KEYS

    def test_framework_keys_equals_channelconfig_field_names(self):
        """FRAMEWORK_KEYS must be exactly the set of ChannelConfig field names
        (derived from dataclasses.fields, not hardcoded)."""
        from corvidae.agent import FRAMEWORK_KEYS
        expected = {f.name for f in dataclasses.fields(ChannelConfig)}
        assert FRAMEWORK_KEYS == expected, (
            f"FRAMEWORK_KEYS must equal ChannelConfig fields exactly. "
            f"Expected: {expected}, Got: {FRAMEWORK_KEYS}"
        )


# ---------------------------------------------------------------------------
# run_agent_turn extra_body plumbing
# ---------------------------------------------------------------------------


class TestRunAgentTurnExtraBody:
    async def test_run_agent_turn_accepts_extra_body_kwarg(self):
        """run_agent_turn must accept an extra_body keyword argument."""
        from corvidae.agent_loop import run_agent_turn

        client = MagicMock()
        client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "ok"}}]}
        )
        messages = [{"role": "user", "content": "hi"}]
        # Must not raise TypeError
        await run_agent_turn(client, messages, tool_schemas=[], extra_body={"temperature": 0.7})

    async def test_run_agent_turn_passes_extra_body_to_client_chat(self):
        """run_agent_turn must forward extra_body to client.chat()."""
        from corvidae.agent_loop import run_agent_turn

        client = MagicMock()
        client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "ok"}}]}
        )
        messages = [{"role": "user", "content": "hi"}]
        extra_body = {"temperature": 0.9, "top_p": 0.95}
        await run_agent_turn(client, messages, tool_schemas=[], extra_body=extra_body)

        client.chat.assert_awaited_once()
        call_kwargs = client.chat.call_args.kwargs
        assert call_kwargs.get("extra_body") == extra_body, (
            f"run_agent_turn must pass extra_body to client.chat(). "
            f"Got call_args: {client.chat.call_args}"
        )

    async def test_run_agent_turn_no_extra_body_does_not_forward_it(self):
        """When extra_body is not provided, client.chat() must not receive extra_body kwarg."""
        from corvidae.agent_loop import run_agent_turn

        client = MagicMock()
        client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "ok"}}]}
        )
        messages = [{"role": "user", "content": "hi"}]
        await run_agent_turn(client, messages, tool_schemas=[])

        call_kwargs = client.chat.call_args.kwargs
        assert "extra_body" not in call_kwargs or call_kwargs["extra_body"] is None, (
            "When extra_body is not provided, it must not be forwarded to client.chat()"
        )

    async def test_run_agent_turn_extra_body_none_not_forwarded(self):
        """When extra_body=None is passed, client.chat() must not receive extra_body."""
        from corvidae.agent_loop import run_agent_turn

        client = MagicMock()
        client.chat = AsyncMock(
            return_value={"choices": [{"message": {"content": "ok"}}]}
        )
        messages = [{"role": "user", "content": "hi"}]
        await run_agent_turn(client, messages, tool_schemas=[], extra_body=None)

        call_kwargs = client.chat.call_args.kwargs
        assert "extra_body" not in call_kwargs or call_kwargs["extra_body"] is None, (
            "extra_body=None must not be forwarded to client.chat()"
        )


# ---------------------------------------------------------------------------
# RuntimeSettingsPlugin
# ---------------------------------------------------------------------------


class TestRuntimeSettingsPlugin:
    def test_import_runtime_settings_plugin(self):
        """RuntimeSettingsPlugin must be importable from corvidae.tools.settings."""
        from corvidae.tools.settings import RuntimeSettingsPlugin  # noqa: F401

    def test_plugin_has_register_tools(self):
        """RuntimeSettingsPlugin must implement the register_tools hook."""
        from corvidae.tools.settings import RuntimeSettingsPlugin
        plugin = RuntimeSettingsPlugin(immutable_settings=set())
        assert hasattr(plugin, "register_tools"), "must implement register_tools"
        assert callable(plugin.register_tools)

    def test_register_tools_adds_set_settings(self):
        """register_tools must register a tool named 'set_settings'."""
        from corvidae.tool import Tool
        from corvidae.tools.settings import RuntimeSettingsPlugin

        plugin = RuntimeSettingsPlugin(immutable_settings=set())
        registry = []
        plugin.register_tools(tool_registry=registry)

        names = {item.name if isinstance(item, Tool) else item.__name__ for item in registry}
        assert "set_settings" in names, (
            f"set_settings must be registered. Got: {names}"
        )


# ---------------------------------------------------------------------------
# Default blocklist always includes system_prompt
# ---------------------------------------------------------------------------


class TestDefaultBlocklist:
    def test_runtime_settings_plugin_default_blocklist_includes_system_prompt(self):
        """RuntimeSettingsPlugin must always include 'system_prompt' in its blocklist
        even when no operator config is provided."""
        from corvidae.tools.settings import RuntimeSettingsPlugin

        plugin = RuntimeSettingsPlugin(immutable_settings=set())
        assert "system_prompt" in plugin.blocklist, (
            "system_prompt must always be in the blocklist"
        )

    def test_operator_can_extend_blocklist(self):
        """Operator-configured keys must be added to the blocklist."""
        from corvidae.tools.settings import RuntimeSettingsPlugin

        plugin = RuntimeSettingsPlugin(immutable_settings={"max_turns"})
        assert "system_prompt" in plugin.blocklist
        assert "max_turns" in plugin.blocklist

    def test_operator_cannot_remove_system_prompt(self):
        """system_prompt must be in the blocklist regardless of operator config."""
        from corvidae.tools.settings import RuntimeSettingsPlugin

        plugin = RuntimeSettingsPlugin(immutable_settings=set())
        assert "system_prompt" in plugin.blocklist

    def test_blocklist_is_a_set(self):
        """blocklist must be a set for O(1) lookup."""
        from corvidae.tools.settings import RuntimeSettingsPlugin

        plugin = RuntimeSettingsPlugin(immutable_settings={"foo"})
        assert isinstance(plugin.blocklist, set)


# ---------------------------------------------------------------------------
# set_settings tool: valid params applied
# ---------------------------------------------------------------------------


class TestSetSettingsValid:
    async def test_valid_settings_applied_to_runtime_overrides(self):
        """Valid settings must be merged into channel.runtime_overrides."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        await set_settings(settings={"temperature": 0.8}, _ctx=ctx)

        assert channel.runtime_overrides.get("temperature") == 0.8, (
            f"temperature must be set in runtime_overrides. Got: {channel.runtime_overrides}"
        )

    async def test_result_is_string(self):
        """set_settings must return a string."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        result = await set_settings(settings={"temperature": 0.5}, _ctx=ctx)
        assert isinstance(result, str)

    async def test_multiple_valid_settings_applied(self):
        """Multiple valid settings must all be applied."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        await set_settings(settings={"temperature": 0.7, "max_turns": 20}, _ctx=ctx)

        assert channel.runtime_overrides["temperature"] == 0.7
        assert channel.runtime_overrides["max_turns"] == 20

    async def test_result_contains_current_overrides(self):
        """Return value must describe the current state of runtime_overrides."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        result = await set_settings(settings={"temperature": 0.6}, _ctx=ctx)
        assert "temperature" in result, (
            f"Return value must mention the set key. Got: {result!r}"
        )


# ---------------------------------------------------------------------------
# set_settings tool: partial updates (merge, not replace)
# ---------------------------------------------------------------------------


class TestSetSettingsPartialUpdate:
    async def test_settings_merged_not_replaced(self):
        """set_settings must merge new settings into existing overrides, not replace them."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel(runtime_overrides={"temperature": 0.5})
        ctx = _make_tool_context(channel)

        await set_settings(settings={"max_turns": 15}, _ctx=ctx)

        assert channel.runtime_overrides["temperature"] == 0.5, (
            "Existing override must not be removed on partial update"
        )
        assert channel.runtime_overrides["max_turns"] == 15


# ---------------------------------------------------------------------------
# set_settings tool: blocked params rejected
# ---------------------------------------------------------------------------


class TestSetSettingsBlocked:
    async def test_blocked_key_rejects_entire_call(self):
        """When any key is blocked, the entire call must be rejected and
        no settings must be applied."""
        set_settings = _get_set_settings_fn()  # default blocklist: {system_prompt}
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        result = await set_settings(
            settings={"system_prompt": "evil", "temperature": 0.9},
            _ctx=ctx,
        )

        assert channel.runtime_overrides == {}, (
            "No settings must be applied when any key is blocked"
        )
        assert "system_prompt" in result, (
            f"Result must name the blocked key. Got: {result!r}"
        )

    async def test_blocked_key_error_message_mentions_all_blocked_keys(self):
        """Error result must name all blocked keys."""
        set_settings = _get_set_settings_fn(immutable_settings={"max_turns"})
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        result = await set_settings(
            settings={"system_prompt": "x", "max_turns": 99},
            _ctx=ctx,
        )

        assert "system_prompt" in result
        assert "max_turns" in result

    async def test_single_blocked_key_rejects_call(self):
        """A single blocked key must reject the entire call."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        result = await set_settings(
            settings={"system_prompt": "x"},
            _ctx=ctx,
        )
        assert channel.runtime_overrides == {}, (
            "Call must be rejected when the only key is blocked"
        )

    async def test_unblocked_key_is_applied(self):
        """Keys not in the blocklist must be applied."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        result = await set_settings(settings={"temperature": 0.3}, _ctx=ctx)
        assert channel.runtime_overrides["temperature"] == 0.3


# ---------------------------------------------------------------------------
# set_settings tool: null clears override
# ---------------------------------------------------------------------------


class TestSetSettingsNullClears:
    async def test_null_value_clears_existing_override(self):
        """Passing null (None) for a key must delete it from runtime_overrides."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel(runtime_overrides={"temperature": 0.9})
        ctx = _make_tool_context(channel)

        await set_settings(settings={"temperature": None}, _ctx=ctx)
        assert "temperature" not in channel.runtime_overrides, (
            "null must remove the key from runtime_overrides"
        )

    async def test_null_for_nonexistent_key_does_not_raise(self):
        """Passing null for a key not in runtime_overrides must not raise."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        result = await set_settings(settings={"temperature": None}, _ctx=ctx)
        assert isinstance(result, str)
        assert "temperature" not in channel.runtime_overrides

    async def test_none_not_stored_in_runtime_overrides(self):
        """None values must be deleted, never stored as None."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel(runtime_overrides={"temperature": 0.5})
        ctx = _make_tool_context(channel)

        await set_settings(settings={"temperature": None}, _ctx=ctx)
        assert "temperature" not in channel.runtime_overrides, (
            "None must not be stored — key must be absent"
        )


# ---------------------------------------------------------------------------
# set_settings tool: null for blocked key is rejected
# ---------------------------------------------------------------------------


class TestSetSettingsNullBlockedKey:
    async def test_null_for_blocked_key_is_rejected(self):
        """Passing null for a blocked key must be rejected — blocked keys can
        never be in runtime_overrides, so there is nothing to clear."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        result = await set_settings(
            settings={"system_prompt": None},
            _ctx=ctx,
        )
        assert "system_prompt" in result, (
            f"Error must mention the blocked key. Got: {result!r}"
        )
        assert channel.runtime_overrides == {}, (
            "runtime_overrides must remain empty when blocked null key rejected"
        )

    async def test_blocklist_check_runs_before_null_delete(self):
        """Order of operations: blocklist check runs before null-delete.
        Even null for a blocked key is rejected."""
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        # system_prompt is always blocked; passing None must still be rejected
        result = await set_settings(settings={"system_prompt": None}, _ctx=ctx)
        assert channel.runtime_overrides == {}


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


class TestSetSettingsAuditLogging:
    async def test_info_log_emitted_on_successful_change(self, caplog):
        """set_settings must emit an INFO log when a setting is successfully changed."""
        import logging
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        with caplog.at_level(logging.INFO, logger="corvidae.tools.settings"):
            await set_settings(settings={"temperature": 0.7}, _ctx=ctx)

        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and r.name == "corvidae.tools.settings"
        ]
        assert info_records, (
            "set_settings must emit at least one INFO log record on a successful change"
        )

    async def test_info_log_contains_channel_id(self, caplog):
        """The INFO audit log must include the channel id in its extra fields."""
        import logging
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        with caplog.at_level(logging.INFO, logger="corvidae.tools.settings"):
            await set_settings(settings={"temperature": 0.7}, _ctx=ctx)

        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and r.name == "corvidae.tools.settings"
        ]
        assert info_records, "Expected at least one INFO record"
        record = info_records[0]
        assert hasattr(record, "channel"), (
            f"INFO log record must have 'channel' in extras. Record.__dict__: {record.__dict__}"
        )
        assert record.channel == channel.id, (
            f"INFO log 'channel' extra must equal channel.id ({channel.id!r}). "
            f"Got: {record.channel!r}"
        )

    async def test_info_log_contains_overrides(self, caplog):
        """The INFO audit log must include the current overrides in its extra fields."""
        import logging
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        with caplog.at_level(logging.INFO, logger="corvidae.tools.settings"):
            await set_settings(settings={"temperature": 0.7}, _ctx=ctx)

        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and r.name == "corvidae.tools.settings"
        ]
        assert info_records, "Expected at least one INFO record"
        record = info_records[0]
        assert hasattr(record, "overrides"), (
            f"INFO log record must have 'overrides' in extras. Record.__dict__: {record.__dict__}"
        )
        assert record.overrides == {"temperature": 0.7}, (
            f"INFO log 'overrides' extra must reflect current runtime_overrides. "
            f"Got: {record.overrides!r}"
        )

    async def test_warning_log_emitted_on_blocked_key(self, caplog):
        """set_settings must emit a WARNING log when a blocked key is rejected."""
        import logging
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        with caplog.at_level(logging.WARNING, logger="corvidae.tools.settings"):
            await set_settings(settings={"system_prompt": "x"}, _ctx=ctx)

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "corvidae.tools.settings"
        ]
        assert warning_records, (
            "set_settings must emit at least one WARNING log record when a blocked key is rejected"
        )

    async def test_warning_log_contains_blocked_keys(self, caplog):
        """The WARNING audit log must include the rejected key names in its extra fields."""
        import logging
        set_settings = _get_set_settings_fn(immutable_settings={"max_turns"})
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        with caplog.at_level(logging.WARNING, logger="corvidae.tools.settings"):
            await set_settings(settings={"system_prompt": "x", "max_turns": 99}, _ctx=ctx)

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "corvidae.tools.settings"
        ]
        assert warning_records, "Expected at least one WARNING record"
        record = warning_records[0]
        assert hasattr(record, "blocked"), (
            f"WARNING log record must have 'blocked' in extras. Record.__dict__: {record.__dict__}"
        )
        assert "system_prompt" in record.blocked, (
            f"'blocked' extra must contain 'system_prompt'. Got: {record.blocked!r}"
        )
        assert "max_turns" in record.blocked, (
            f"'blocked' extra must contain 'max_turns'. Got: {record.blocked!r}"
        )

    async def test_warning_log_blocked_is_sorted(self, caplog):
        """The 'blocked' extra in the WARNING log must be a sorted list."""
        import logging
        set_settings = _get_set_settings_fn(immutable_settings={"max_turns"})
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        with caplog.at_level(logging.WARNING, logger="corvidae.tools.settings"):
            await set_settings(settings={"max_turns": 99, "system_prompt": "x"}, _ctx=ctx)

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "corvidae.tools.settings"
        ]
        assert warning_records, "Expected at least one WARNING record"
        record = warning_records[0]
        assert hasattr(record, "blocked"), "Expected 'blocked' in extras"
        assert record.blocked == sorted(record.blocked), (
            f"'blocked' extra must be sorted. Got: {record.blocked!r}"
        )

    async def test_no_warning_on_successful_change(self, caplog):
        """set_settings must NOT emit a WARNING log when the call succeeds."""
        import logging
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        with caplog.at_level(logging.WARNING, logger="corvidae.tools.settings"):
            await set_settings(settings={"temperature": 0.5}, _ctx=ctx)

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "corvidae.tools.settings"
        ]
        assert not warning_records, (
            f"No WARNING must be emitted on a successful change. Got: {warning_records}"
        )

    async def test_no_info_on_blocked_key(self, caplog):
        """set_settings must NOT emit an INFO audit log when the call is rejected."""
        import logging
        set_settings = _get_set_settings_fn()
        channel = _make_channel()
        ctx = _make_tool_context(channel)

        with caplog.at_level(logging.INFO, logger="corvidae.tools.settings"):
            await set_settings(settings={"system_prompt": "x"}, _ctx=ctx)

        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and r.name == "corvidae.tools.settings"
        ]
        assert not info_records, (
            f"No INFO must be emitted when the call is rejected. Got: {info_records}"
        )
