"""Tests for config hot-reload (Issue #32).

Coverage:
    - on_config_reload hookspec on AgentSpec
    - ConfigWatcherPlugin: importable, file-change detection, YAML reload,
      hook dispatch, error handling (invalid YAML, missing file),
      deep_merge with CLI overrides, on_stop task cancellation,
      _last_mtime initialization (no spurious reload)
    - Per-plugin error isolation: one failing on_config_reload doesn't block
      others
    - LLMPlugin.on_config_reload: new client created, old client closed
    - RuntimeSettingsPlugin.on_config_reload: blocklist reset then re-applied
    - Agent.on_config_reload: system_prompt unchanged, chars_per_token updated
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import yaml

from helpers import stub_llm_config


# ---------------------------------------------------------------------------
# 1. on_config_reload hookspec exists on AgentSpec
# ---------------------------------------------------------------------------


def test_on_config_reload_hookspec_exists():
    """AgentSpec must define an on_config_reload hookspec."""
    from corvidae.hooks import AgentSpec

    assert hasattr(AgentSpec, "on_config_reload"), (
        "AgentSpec must have an on_config_reload hookspec"
    )


def test_on_config_reload_is_callable():
    """on_config_reload on AgentSpec must be callable."""
    from corvidae.hooks import AgentSpec

    assert callable(AgentSpec.on_config_reload)


def test_on_config_reload_hookspec_registered_in_pm():
    """create_plugin_manager() must include on_config_reload in the hook list."""
    from corvidae.hooks import create_plugin_manager

    pm = create_plugin_manager()
    hook = pm.hook.on_config_reload
    assert hook is not None, "on_config_reload must be a registered hook in the plugin manager"


# ---------------------------------------------------------------------------
# 2. ConfigWatcherPlugin is importable and has expected interface
# ---------------------------------------------------------------------------


def test_config_watcher_plugin_importable():
    """corvidae.config_watcher.ConfigWatcherPlugin must be importable."""
    from corvidae.config_watcher import ConfigWatcherPlugin  # noqa: F401


def test_config_watcher_plugin_has_on_start():
    """ConfigWatcherPlugin must have an on_start hookimpl."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    assert hasattr(ConfigWatcherPlugin, "on_start"), (
        "ConfigWatcherPlugin must have an on_start method"
    )


def test_config_watcher_plugin_has_on_stop():
    """ConfigWatcherPlugin must have an on_stop hookimpl."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    assert hasattr(ConfigWatcherPlugin, "on_stop"), (
        "ConfigWatcherPlugin must have an on_stop method"
    )


def test_config_watcher_plugin_has_on_init():
    """ConfigWatcherPlugin must have an on_init hookimpl."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    assert hasattr(ConfigWatcherPlugin, "on_init"), (
        "ConfigWatcherPlugin must have an on_init method"
    )


# ---------------------------------------------------------------------------
# 2b. ConfigWatcherPlugin reads configurable poll interval from config
# ---------------------------------------------------------------------------


async def test_config_watcher_reads_poll_interval_from_config(tmp_path):
    """ConfigWatcherPlugin.on_init must read daemon.config_poll_interval from config
    and store it as self._poll_interval.

    If the key is absent, a sensible default (e.g. 1.0 second) must be used.
    """
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "daemon": {"config_poll_interval": 5.0},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    assert hasattr(plugin, "_poll_interval"), (
        "ConfigWatcherPlugin must have a _poll_interval attribute after on_init"
    )
    assert plugin._poll_interval == 5.0, (
        f"ConfigWatcherPlugin must read _poll_interval from daemon.config_poll_interval. "
        f"Got {plugin._poll_interval!r}"
    )


async def test_config_watcher_poll_interval_default(tmp_path):
    """ConfigWatcherPlugin._poll_interval must have a sensible default when
    daemon.config_poll_interval is absent from config.
    """
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    assert hasattr(plugin, "_poll_interval"), (
        "ConfigWatcherPlugin must have a _poll_interval attribute after on_init"
    )
    assert isinstance(plugin._poll_interval, (int, float)), (
        "_poll_interval must be numeric"
    )
    assert plugin._poll_interval > 0, (
        "_poll_interval default must be a positive number"
    )


# ---------------------------------------------------------------------------
# 3. ConfigWatcherPlugin detects file changes (mtime change triggers reload)
# ---------------------------------------------------------------------------


async def test_config_watcher_detects_mtime_change(tmp_path):
    """ConfigWatcherPlugin must call _reload_config when file mtime changes."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    # Patch _reload_config to track calls
    reload_calls = []

    async def fake_reload():
        reload_calls.append(True)

    plugin._reload_config = fake_reload

    # Set _last_mtime to a value older than actual file mtime to trigger reload
    plugin._last_mtime = 0.0

    # Run _watch_loop for one iteration: return False on first call (enter loop),
    # True on subsequent calls (exit loop after one pass).
    call_count = 0

    def side_effect():
        nonlocal call_count
        call_count += 1
        return call_count > 1

    plugin._stop_event = MagicMock()
    plugin._stop_event.is_set.side_effect = side_effect

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await plugin._watch_loop()

    assert reload_calls, "ConfigWatcherPlugin must call _reload_config when mtime changes"


# ---------------------------------------------------------------------------
# 4. ConfigWatcherPlugin does NOT reload on start (no spurious reload)
# ---------------------------------------------------------------------------


async def test_config_watcher_no_spurious_reload_on_start(tmp_path):
    """on_start must initialize _last_mtime from actual file mtime to avoid
    a reload on the first poll cycle (M1 from design review)."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    actual_mtime = config_file.stat().st_mtime

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    reload_calls = []

    async def fake_reload():
        reload_calls.append(True)

    plugin._reload_config = fake_reload

    # on_start must set _last_mtime to actual file mtime
    with patch.object(plugin, "_watch_loop", new_callable=AsyncMock):
        await plugin.on_start(config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        })

    assert plugin._last_mtime == actual_mtime, (
        f"on_start must initialize _last_mtime to file's actual mtime ({actual_mtime}), "
        f"got {plugin._last_mtime}"
    )


# ---------------------------------------------------------------------------
# 5. ConfigWatcherPlugin reloads and parses YAML on change
# ---------------------------------------------------------------------------


async def test_config_watcher_reloads_yaml_on_change(tmp_path):
    """_reload_config must read and parse the YAML file when mtime changes."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m1\n")

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m1"}},
        },
    )

    # Write new YAML content
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m2\n")

    # Patch _dispatch_config_reload to capture what config it receives
    dispatched_configs = []

    async def capture_dispatch(config: dict) -> None:
        dispatched_configs.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    await plugin._reload_config()

    assert dispatched_configs, "_reload_config must dispatch to plugins"
    assert dispatched_configs[0].get("llm", {}).get("main", {}).get("model") == "m2", (
        "Reloaded config must contain updated YAML values"
    )


# ---------------------------------------------------------------------------
# 6. ConfigWatcherPlugin applies deep_merge with CLI overrides on reload
# ---------------------------------------------------------------------------


async def test_config_watcher_applies_cli_overrides_on_reload(tmp_path):
    """_reload_config must deep_merge CLI overrides on top of new YAML."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: from-yaml\n")

    cli_overrides = {"llm": {"main": {"model": "from-cli"}}}

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": cli_overrides,
            "llm": {"main": {"base_url": "http://localhost", "model": "from-cli"}},
        },
    )

    dispatched_configs = []

    async def capture_dispatch(config: dict) -> None:
        dispatched_configs.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    await plugin._reload_config()

    assert dispatched_configs, "_reload_config must dispatch"
    merged_model = dispatched_configs[0].get("llm", {}).get("main", {}).get("model")
    assert merged_model == "from-cli", (
        f"CLI overrides must win over YAML content after merge. Got model={merged_model!r}"
    )


# ---------------------------------------------------------------------------
# 6b. ConfigWatcherPlugin updates ChannelRegistry.agent_defaults on reload
# ---------------------------------------------------------------------------


async def test_config_watcher_updates_channel_registry_agent_defaults(tmp_path):
    """_reload_config must update ChannelRegistry.agent_defaults from the new config.

    The design specifies: ChannelRegistry.agent_defaults = new_config.get("agent", {})
    This update is performed directly in ConfigWatcherPlugin._reload_config, not
    via a per-plugin hookimpl.
    """
    from corvidae.config_watcher import ConfigWatcherPlugin
    from corvidae.channel import ChannelRegistry

    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "agent:\n  max_turns: 5\nllm:\n  main:\n    base_url: http://localhost\n    model: m\n"
    )

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "agent": {"max_turns": 5},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    # Write updated YAML with new agent defaults
    config_file.write_text(
        "agent:\n  max_turns: 20\nllm:\n  main:\n    base_url: http://localhost\n    model: m\n"
    )

    async def no_op_dispatch(config: dict) -> None:
        pass

    plugin._dispatch_config_reload = no_op_dispatch

    original = ChannelRegistry.agent_defaults
    try:
        await plugin._reload_config()

        assert ChannelRegistry.agent_defaults.get("max_turns") == 20, (
            "_reload_config must update ChannelRegistry.agent_defaults with the new agent config"
        )
    finally:
        ChannelRegistry.agent_defaults = original


# ---------------------------------------------------------------------------
# 7. ConfigWatcherPlugin handles invalid YAML gracefully
# ---------------------------------------------------------------------------


async def test_config_watcher_handles_invalid_yaml(tmp_path, caplog):
    """_reload_config must log an error and not crash on invalid YAML."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    # Write invalid YAML
    config_file.write_text(": invalid: yaml: [unclosed")

    dispatched = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        # Must not raise
        await plugin._reload_config()

    assert not dispatched, "Invalid YAML must not result in a dispatch"
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "Invalid YAML must produce an ERROR log"


async def test_a_non_string_llm_key_refuses_the_reload(tmp_path, caplog):
    """R471.2: a reload carrying a non-string `llm:` key changes nothing.

    YAML turns an unquoted `2:` into an int. That used to raise a TypeError
    from inside the config error's own message, which the watcher does not
    catch, so the reload escaped with a traceback instead of the skip line.
    """
    from corvidae.config_watcher import ConfigWatcherPlugin

    good = "llm:\n  main:\n    base_url: http://localhost\n    model: m\n"
    config_file = tmp_path / "agent.yaml"
    config_file.write_text(good)

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    running = {"llm": {"main": {"base_url": "http://localhost", "model": "m"}}}
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            **running,
        },
    )

    # An unquoted integer profile key, which YAML reads as an int.
    config_file.write_text(
        "llm:\n  main:\n    base_url: http://localhost\n    model: m\n"
        "  2:\n    base_url: http://other\n    model: n\n"
    )

    dispatched = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        await plugin._reload_config()

    assert not dispatched, "a rejected llm section must not dispatch"
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    joined = "\n".join(errors)
    assert "reload skipped" in joined, f"must say the reload was skipped: {errors}"
    assert "2" in joined, f"must name the offending key: {errors}"


async def _reload_refused(tmp_path: Path, caplog, payload: bytes):
    """Write ``payload`` over a loaded config, reload, and return
    (dispatched configs, ERROR messages, config path) (#310)."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    # Start from a valid config so there is a running config to keep.
    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")
    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    # Overwrite with the bad bytes and capture any dispatch.
    config_file.write_bytes(payload)
    dispatched: list[dict] = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch
    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        # Must not raise.
        await plugin._reload_config()
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    return dispatched, errors, config_file


async def test_config_watcher_refuses_non_utf8_reload(tmp_path, caplog):
    """A non-UTF-8 file is refused with an ERROR naming the path, the cause
    and the remedy; nothing is dispatched (#310)."""
    payload = b"# " + b"x" * 8998 + b"\n" + b"k: caf\xe9\n"
    dispatched, errors, path = await _reload_refused(tmp_path, caplog, payload)

    assert not dispatched, "A non-UTF-8 file must not result in a dispatch"
    text = "\n".join(errors)
    assert "not valid UTF-8" in text
    assert f"at byte {payload.index(b'\xe9')}" in text
    assert "re-save it as UTF-8" in text
    assert str(path) in text
    assert "running config stays in force" in text


async def test_config_watcher_refuses_deeply_nested_reload(tmp_path, caplog):
    """YAML that raises RecursionError is refused with an ERROR naming the
    path, the cause and the remedy; nothing is dispatched (#310)."""
    dispatched, errors, path = await _reload_refused(
        tmp_path, caplog, b"[" * 10000
    )

    assert not dispatched, "Too-deep YAML must not result in a dispatch"
    text = "\n".join(errors)
    assert "nested too deeply" in text
    assert "reduce the nesting" in text
    assert str(path) in text
    assert "running config stays in force" in text


# ---------------------------------------------------------------------------
# 8. ConfigWatcherPlugin handles missing file gracefully
# ---------------------------------------------------------------------------


async def test_config_watcher_handles_missing_file(tmp_path, caplog):
    """_watch_loop must log a warning and continue when the config file is missing."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    # File does not exist

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
        },
    )

    plugin._last_mtime = 0.0

    # Return False on first call (enter loop body), True on subsequent calls (exit).
    call_count = 0

    def side_effect():
        nonlocal call_count
        call_count += 1
        return call_count > 1

    plugin._stop_event = MagicMock()
    plugin._stop_event.is_set.side_effect = side_effect

    with caplog.at_level(logging.WARNING, logger="corvidae.config_watcher"):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Must not raise even though the file is missing
            await plugin._watch_loop()

    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records, "Missing config file must produce a WARNING log"


# ---------------------------------------------------------------------------
# 9. ConfigWatcherPlugin validates config (rejects if llm.main absent)
# ---------------------------------------------------------------------------


async def test_config_watcher_rejects_invalid_agent_config(tmp_path, caplog):
    """_reload_config must reject a reload whose agent section fails
    validate_agent_config (here: the removed keep_thinking_in_history key)
    without touching the registry's agent_defaults or dispatching."""
    from corvidae.channel import ChannelRegistry
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "llm:\n  main:\n    base_url: http://localhost\n    model: m\n"
        "agent:\n  keep_thinking_in_history: true\n"
    )

    registry = ChannelRegistry({"reasoning_history": "drop"})
    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    pm.get_plugin.return_value = registry
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    dispatched = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        await plugin._reload_config()

    assert not dispatched
    assert registry.agent_defaults == {"reasoning_history": "drop"}
    assert any("invalid agent config" in r.getMessage() for r in caplog.records)


async def test_reload_with_invalid_channels_entry_changes_nothing(tmp_path, caplog):
    """#208 R1/R2: a reload whose channels: section is rejected changes
    nothing: agent_defaults (instance and class), wide entries, a registered
    channel's tools.disabled and plugin on_config_reload all stay put, and the
    ERROR log names the key, the file and that the reload was skipped."""
    from corvidae.channel import ChannelConfig, ChannelRegistry, WideEntry
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    # New file: changed agent.max_turns, changed wide entry, malformed tools.
    config_file.write_text(
        "llm:\n  main:\n    base_url: http://localhost\n    model: m\n"
        "agent:\n  max_turns: 99\n"
        "channels:\n"
        "  'irc:#a':\n    tools:\n      disabled: [web]\n"
        "  'irc:*':\n    max_turns: 7\n"
        "  'irc:#x':\n    tools: [shell]\n"
    )

    old_defaults = {"max_turns": 5}
    saved_class_defaults = ChannelRegistry.agent_defaults
    registry = ChannelRegistry(old_defaults)
    old_wide = (
        WideEntry(key="irc:*", prefix="irc:", config=ChannelConfig(max_turns=3)),
    )
    registry.set_wide_entries(old_wide)
    channel = registry.get_or_create(
        "irc", "#a", config=ChannelConfig(disabled_tools=frozenset({"shell"}))
    )
    class_defaults_before = ChannelRegistry.agent_defaults

    spy = MagicMock()
    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    pm.get_plugin.return_value = registry
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    async def capture_dispatch(config: dict) -> None:
        spy(config)

    plugin._dispatch_config_reload = capture_dispatch

    try:
        with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
            await plugin._reload_config()

        assert registry.agent_defaults is old_defaults
        assert registry.agent_defaults == {"max_turns": 5}
        assert ChannelRegistry.agent_defaults is class_defaults_before
        assert registry.wide_entries == old_wide
        assert channel.config.disabled_tools == frozenset({"shell"})
        spy.assert_not_called()
    finally:
        # The class attribute is shared state; put it back for other tests.
        ChannelRegistry.agent_defaults = saved_class_defaults

    # R2: the message (not only the traceback) names key, file, and the skip.
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        "channels.irc:#x.tools" in m
        and str(config_file) in m
        and "reload skipped" in m
        for m in errors
    ), errors
    assert "agent defaults still applied" not in caplog.text


async def test_reload_with_unknown_channel_key_changes_nothing(tmp_path, caplog):
    """#205 R3: a reload that adds channels.<id>.max_tokens is refused whole:
    the ERROR names the key and agent_defaults stays as it was."""
    from corvidae.channel import ChannelRegistry
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "llm:\n  main:\n    base_url: http://localhost\n    model: m\n"
        "agent:\n  max_turns: 99\n"
        "channels:\n  'irc:#new':\n    max_tokens: 200\n"
    )
    saved_class_defaults = ChannelRegistry.agent_defaults
    registry = ChannelRegistry({"max_turns": 5})
    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    pm.get_plugin.return_value = registry
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )
    try:
        with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
            await plugin._reload_config()
        assert registry.agent_defaults == {"max_turns": 5}
        assert registry.get("irc:#new") is None
        assert registry.wide_entries == ()
    finally:
        ChannelRegistry.agent_defaults = saved_class_defaults
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("channels.irc:#new.max_tokens" in m and "reload skipped" in m
               for m in errors), errors


async def test_reload_with_removed_daemon_key_is_rejected(tmp_path, caplog):
    """R12: _reload_config rejects a reload that carries a removed daemon
    idle key: it logs the key, leaves agent_defaults alone and dispatches
    nothing."""
    from corvidae.channel import ChannelRegistry
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "llm:\n  main:\n    base_url: http://localhost\n    model: m\n"
        "agent:\n  max_turns: 10\n"
        "daemon:\n  idle_cooldown_seconds: 30\n"
    )

    registry = ChannelRegistry({"reasoning_history": "drop"})
    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    pm.get_plugin.return_value = registry
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    dispatched = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        await plugin._reload_config()

    # The reload is rejected before anything is applied or dispatched.
    assert not dispatched
    assert registry.agent_defaults == {"reasoning_history": "drop"}
    # The log names the removed key so the operator can delete it.
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert "idle_cooldown_seconds" in caplog.text


async def test_config_watcher_rejects_config_without_llm_main(tmp_path, caplog):
    """_reload_config must skip dispatch when llm.main is absent after merge."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    # YAML without llm.main
    config_file.write_text("agent:\n  max_turns: 10\n")

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            # Original config had llm.main
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    dispatched = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        await plugin._reload_config()

    assert not dispatched, "Config without llm.main must not be dispatched"
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "Missing llm.main must produce an ERROR log"


async def test_config_watcher_rejects_malformed_llm_section(tmp_path, caplog):
    """R2 (#399): a reload whose llm section is malformed (here an unknown
    ``extends`` base) is rejected as a whole, as one without llm.main is."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "llm:\n"
        "  main:\n    base_url: http://localhost\n    model: m\n"
        "  forks:\n    extends: nope\n"
    )

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    dispatched = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        await plugin._reload_config()

    assert not dispatched, "A malformed llm section must not be dispatched"
    assert "llm.forks.extends" in caplog.text


async def test_config_watcher_accepts_llm_section_with_defaults_chat_and_no_main(tmp_path):
    """R5 (#399): a reload whose llm section names its default chat profile
    through ``defaults.chat`` is accepted without an ``llm.main`` block."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "llm:\n"
        "  foo:\n    base_url: http://localhost\n    model: m\n"
        "  defaults:\n    chat: foo\n"
    )

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"foo": {"base_url": "http://localhost", "model": "m"},
                    "defaults": {"chat": "foo"}},
        },
    )

    dispatched = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    await plugin._reload_config()

    assert len(dispatched) == 1, "defaults.chat without llm.main must be accepted"


# ---------------------------------------------------------------------------
# 10. ConfigWatcherPlugin.on_stop cancels the watcher task
# ---------------------------------------------------------------------------


async def test_config_watcher_on_stop_cancels_task(tmp_path):
    """on_stop must cancel the watcher task started by on_start."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    # Simulate on_start by creating a real asyncio task that blocks
    async def infinite_loop():
        try:
            while True:
                await asyncio.sleep(100)
        except asyncio.CancelledError:
            raise

    plugin._watcher_task = asyncio.create_task(infinite_loop())
    plugin._stop_event = asyncio.Event()

    # on_stop must cancel the task and not hang
    await plugin.on_stop()

    assert plugin._watcher_task is None or plugin._watcher_task.cancelled() or plugin._watcher_task.done(), (
        "on_stop must cancel the watcher task"
    )


# ---------------------------------------------------------------------------
# 11. Per-plugin error isolation: one failing plugin doesn't block others
# ---------------------------------------------------------------------------


async def test_config_watcher_per_plugin_error_isolation(tmp_path, caplog):
    """_dispatch_config_reload must call each plugin independently.
    A plugin that raises must not prevent other plugins from receiving the config.
    """
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    # Create two mock hookimpls: first raises, second succeeds
    received_configs = []

    async def failing_impl(config: dict) -> None:
        raise RuntimeError("intentional failure in on_config_reload")

    async def succeeding_impl(config: dict) -> None:
        received_configs.append(config)

    hookimpl_failing = MagicMock()
    hookimpl_failing.function = failing_impl
    hookimpl_failing.argnames = ["config"]
    hookimpl_failing.plugin_name = "failing_plugin"

    hookimpl_succeeding = MagicMock()
    hookimpl_succeeding.function = succeeding_impl
    hookimpl_succeeding.argnames = ["config"]
    hookimpl_succeeding.plugin_name = "succeeding_plugin"

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = [
        hookimpl_failing,
        hookimpl_succeeding,
    ]

    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    test_config = {"llm": {"main": {"base_url": "http://localhost", "model": "m"}}}

    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        await plugin._dispatch_config_reload(test_config)

    assert received_configs, (
        "succeeding plugin must receive config even though failing plugin raised"
    )
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "Failing plugin must produce an ERROR log"


# ---------------------------------------------------------------------------
# 12. _dispatch_config_reload filters kwargs by argnames
# ---------------------------------------------------------------------------


async def test_dispatch_config_reload_filters_kwargs_by_argnames(tmp_path):
    """_dispatch_config_reload must filter kwargs by hook_impl.argnames.
    Hookimpls that omit 'config' from their signature must receive no arguments
    rather than raising TypeError (NEW-I1 from design review Fix #2).
    """
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    called = []

    async def impl_no_config_arg() -> None:
        # This impl deliberately has no 'config' param
        called.append(True)

    hookimpl_no_config = MagicMock()
    hookimpl_no_config.function = impl_no_config_arg
    hookimpl_no_config.argnames = []  # no 'config' in argnames
    hookimpl_no_config.plugin_name = "no_config_plugin"

    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = [hookimpl_no_config]

    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    # Must not raise TypeError
    await plugin._dispatch_config_reload(
        {"llm": {"main": {"base_url": "http://localhost", "model": "m"}}}
    )

    assert called, "Plugin with no 'config' arg must still be called"


# ---------------------------------------------------------------------------
# 13. LLMPlugin.on_config_reload: llm config is restart-only (#399)
# ---------------------------------------------------------------------------


async def test_llm_plugin_on_config_reload_warns_and_keeps_client_on_change(caplog):
    """A reload that changes the llm section logs one WARNING that it applies
    on restart; no client is created or swapped."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    stub_llm_config(plugin, {"main": {"base_url": "http://old", "model": "old-model"}})

    old_client = MagicMock(name="old_client")
    plugin._clients["main"] = old_client

    new_config = {"llm": {"main": {"base_url": "http://new", "model": "new-model"}}}

    with patch.object(LLMPlugin, "_create_client") as create:
        with caplog.at_level(logging.WARNING, logger="corvidae.llm_plugin"):
            await plugin.on_config_reload(config=new_config)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "apply on restart" in warnings[0], warnings
    create.assert_not_called()
    assert plugin.get_client() is old_client


async def test_llm_plugin_on_config_reload_no_change_no_warning(caplog):
    """A reload that leaves the llm section unchanged is silent."""
    from corvidae.llm_plugin import LLMPlugin

    same_section = {"main": {"base_url": "http://localhost", "model": "m"}}
    plugin = LLMPlugin(pm=None)
    stub_llm_config(plugin, same_section)

    with caplog.at_level(logging.WARNING, logger="corvidae.llm_plugin"):
        await plugin.on_config_reload(config={"llm": same_section})

    assert not caplog.records


# ---------------------------------------------------------------------------
# 14. CompactionPlugin.on_config_reload re-reads thresholds (the cached
#     _llm_client and its invalidation were removed with _summarize, #380/#388)
# ---------------------------------------------------------------------------


async def test_compaction_plugin_on_config_reload_updates_threshold():
    """CompactionPlugin.on_config_reload must re-read compaction_threshold from config."""
    from corvidae.compaction import CompactionPlugin

    plugin = CompactionPlugin(pm=None)
    await plugin.on_init(
        pm=None,
        config={"agent": {"compaction_threshold": 0.8}, "llm": {"main": {"base_url": "http://localhost", "model": "m"}}},
    )

    assert plugin._compaction_threshold == 0.8

    await plugin.on_config_reload(config={
        "agent": {"compaction_threshold": 0.9},
        "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
    })

    assert plugin._compaction_threshold == 0.9, (
        "CompactionPlugin.on_config_reload must update _compaction_threshold"
    )


# ---------------------------------------------------------------------------
# 15. RuntimeSettingsPlugin.on_config_reload resets and re-applies blocklist
# ---------------------------------------------------------------------------


async def test_runtime_settings_plugin_on_config_reload_resets_blocklist():
    """RuntimeSettingsPlugin.on_config_reload must reset blocklist to
    _constructor_immutable | {'system_prompt'} before re-applying config.
    Removed entries must not persist (I3 from design review Fix #1)."""
    from corvidae.tools.settings import RuntimeSettingsPlugin

    plugin = RuntimeSettingsPlugin(pm=None, immutable_settings={"constructor_key"})
    await plugin.on_init(
        pm=None,
        config={"agent": {"immutable_settings": ["config_key"]}},
    )

    # Both constructor_key and config_key should be in blocklist
    assert "constructor_key" in plugin.blocklist
    assert "config_key" in plugin.blocklist

    # Reload with config that no longer has config_key
    await plugin.on_config_reload(config={"agent": {"immutable_settings": []}})

    # config_key must be removed; constructor_key and system_prompt must remain
    assert "config_key" not in plugin.blocklist, (
        "config_key must be removed from blocklist when removed from config on reload"
    )
    assert "constructor_key" in plugin.blocklist, (
        "constructor_key must remain (was provided in constructor)"
    )
    assert "system_prompt" in plugin.blocklist, (
        "system_prompt must always remain in blocklist"
    )


async def test_runtime_settings_plugin_stores_constructor_immutable():
    """RuntimeSettingsPlugin must store constructor-supplied immutable_settings
    separately in _constructor_immutable so they survive reloads (NEW-M1)."""
    from corvidae.tools.settings import RuntimeSettingsPlugin

    plugin = RuntimeSettingsPlugin(pm=None, immutable_settings={"ctor_key"})

    assert hasattr(plugin, "_constructor_immutable"), (
        "RuntimeSettingsPlugin must have a _constructor_immutable attribute"
    )
    assert "ctor_key" in plugin._constructor_immutable, (
        "_constructor_immutable must contain keys passed to the constructor"
    )


# ---------------------------------------------------------------------------
# 16. Agent.on_config_reload updates chars_per_token
# ---------------------------------------------------------------------------


async def test_agent_on_config_reload_updates_chars_per_token():
    """Agent.on_config_reload must re-read chars_per_token from the new config."""
    from corvidae.agent import Agent

    pm = MagicMock()
    plugin = Agent(pm=pm)
    await plugin.on_init(
        pm=pm,
        config={"agent": {"chars_per_token": 4.0}},
    )

    assert plugin._chars_per_token == 4.0

    await plugin.on_config_reload(config={"agent": {"chars_per_token": 5.0}})

    assert plugin._chars_per_token == 5.0, (
        "Agent.on_config_reload must update _chars_per_token"
    )


async def test_agent_on_config_reload_reborrow_llm_client():
    """Agent.on_config_reload must re-borrow _client from LLMPlugin after
    LLMPlugin may have swapped its client (correction in design Key Decision #3)."""
    from corvidae.agent import Agent
    from corvidae.llm_plugin import LLMPlugin

    pm = MagicMock()
    plugin = Agent(pm=pm)

    # Set up a mock LLMPlugin reachable from pm
    mock_llm_plugin = MagicMock(spec=LLMPlugin)
    new_mock_client = MagicMock(name="new_llm_client")
    mock_llm_plugin.get_client.return_value = new_mock_client
    pm.get_plugin.return_value = mock_llm_plugin

    old_client = MagicMock(name="old_llm_client")
    plugin._client = old_client
    plugin._mark_started()

    await plugin.on_config_reload(config={"agent": {}, "llm": {"main": {"base_url": "http://localhost", "model": "m"}}})

    assert plugin._client is new_mock_client, (
        "Agent.on_config_reload must re-borrow _client from LLMPlugin"
    )


# ---------------------------------------------------------------------------
# 17. ConfigWatcherPlugin reads _config_path and _cli_overrides from config
# ---------------------------------------------------------------------------


async def test_config_watcher_reads_config_path_from_reserved_key(tmp_path):
    """ConfigWatcherPlugin.on_init must read _config_path from config dict
    (I4 from design review Fix #1 — consistent with _base_dir pattern)."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {"llm": {"main": {"api_key": "override-key"}}},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    assert plugin._config_path == config_file, (
        "ConfigWatcherPlugin must store _config_path from config dict"
    )
    assert plugin._cli_overrides == {"llm": {"main": {"api_key": "override-key"}}}, (
        "ConfigWatcherPlugin must store _cli_overrides from config dict"
    )


# ---------------------------------------------------------------------------
# 18. ConfigWatcherPlugin handles FileNotFoundError on on_start gracefully
# ---------------------------------------------------------------------------


async def test_config_watcher_on_start_handles_missing_file(tmp_path):
    """on_start must set _last_mtime to 0.0 when the config file is missing
    (FileNotFoundError fallback from design doc on_start code)."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "nonexistent.yaml"
    # File does not exist

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={"_config_path": config_file, "_cli_overrides": {}},
    )

    with patch.object(plugin, "_watch_loop", new_callable=AsyncMock):
        await plugin.on_start(config={"_config_path": config_file, "_cli_overrides": {}})

    assert plugin._last_mtime == 0.0, (
        "on_start must set _last_mtime to 0.0 when config file does not exist"
    )


# ---------------------------------------------------------------------------
# 19. ConfigWatcherPlugin on_start creates _watcher_task
# ---------------------------------------------------------------------------


async def test_config_watcher_on_start_creates_watcher_task(tmp_path):
    """on_start must create an asyncio.Task for the polling loop."""
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text("llm:\n  main:\n    base_url: http://localhost\n    model: m\n")

    pm = MagicMock()
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    # Patch _watch_loop so the task doesn't actually run the real loop
    loop_started = []

    async def fake_loop():
        loop_started.append(True)
        # Yield control to allow the task to be inspected, then return
        await asyncio.sleep(0)

    plugin._watch_loop = fake_loop

    await plugin.on_start(config={
        "_config_path": config_file,
        "_cli_overrides": {},
        "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
    })

    assert plugin._watcher_task is not None, (
        "on_start must create a _watcher_task"
    )
    assert isinstance(plugin._watcher_task, asyncio.Task), (
        "_watcher_task must be an asyncio.Task"
    )

    # Clean up
    plugin._watcher_task.cancel()
    try:
        await plugin._watcher_task
    except (asyncio.CancelledError, Exception):
        pass


async def test_reload_with_bad_tools_timeouts_is_rejected(tmp_path, caplog):
    """Issue #94 R6: a reload carrying a non-mapping tools.timeouts is skipped:
    nothing dispatched, defaults untouched, ERROR naming the key."""
    from corvidae.channel import ChannelRegistry
    from corvidae.config_watcher import ConfigWatcherPlugin

    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        "llm:\n  main:\n    base_url: http://localhost\n    model: m\n"
        "tools:\n  timeouts: [a]\n"
    )

    registry = ChannelRegistry({"reasoning_history": "drop"})
    pm = MagicMock()
    pm.hook.on_config_reload.get_hookimpls.return_value = []
    pm.get_plugin.return_value = registry
    plugin = ConfigWatcherPlugin()
    await plugin.on_init(
        pm=pm,
        config={
            "_config_path": config_file,
            "_cli_overrides": {},
            "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
        },
    )

    dispatched = []

    async def capture_dispatch(config: dict) -> None:
        dispatched.append(config)

    plugin._dispatch_config_reload = capture_dispatch

    with caplog.at_level(logging.ERROR, logger="corvidae.config_watcher"):
        await plugin._reload_config()

    # Rejected before anything is applied or dispatched.
    assert not dispatched
    assert registry.agent_defaults == {"reasoning_history": "drop"}
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    # R6: the refusal names the key, the rejected value and what is accepted.
    assert "tools.timeouts" in caplog.text
    assert "['a']" in caplog.text
    assert "tool names to seconds" in caplog.text
    assert "reload skipped" in caplog.text


# ---------------------------------------------------------------------------
# #408: a restart-only key that changes must be reported, and a plugin's
# declared live keys must be the ones its reload hook actually re-reads.
# ---------------------------------------------------------------------------

def _make_watcher(config_path: Path):
    """A ConfigWatcherPlugin wired to a plugin manager, not started."""
    from corvidae.config_watcher import ConfigWatcherPlugin
    from corvidae.hooks import create_plugin_manager

    from corvidae.compaction import CompactionPlugin

    pm = create_plugin_manager()
    watcher = ConfigWatcherPlugin()
    pm.register(watcher, name="config_watcher")
    # Compaction is the plugin whose live keys the warning test relies on.
    pm.register(CompactionPlugin(), name="compaction")
    watcher.pm = pm
    watcher._config_path = config_path
    watcher._cli_overrides = {}
    return watcher


class _RecordingConfig(dict):
    """A config dict that records every key read, including nested ones.

    A source scan cannot see a key a hook reads through a helper
    (`_resolve_mid_conversation_system_role(agent_config)`), so the guard
    drives the hook instead and observes what it actually reads (#408
    review).
    """

    def __init__(self, data: dict, prefix: str = "", log: set[str] | None = None):
        super().__init__(data)
        self._prefix = prefix
        self.reads = log if log is not None else set()

    def get(self, key, default=None):
        dotted = f"{self._prefix}.{key}" if self._prefix else key
        self.reads.add(dotted)
        value = super().get(key, default)
        if isinstance(value, dict):
            return _RecordingConfig(value, dotted, self.reads)
        return value

    def __getitem__(self, key):
        dotted = f"{self._prefix}.{key}" if self._prefix else key
        self.reads.add(dotted)
        value = super().__getitem__(key)
        if isinstance(value, dict):
            return _RecordingConfig(value, dotted, self.reads)
        return value


async def _keys_read_by_hook(plugin, config: dict) -> set[str]:
    """Dotted keys the plugin's on_config_reload actually reads."""
    recorder = _RecordingConfig(config)
    await plugin.on_config_reload(recorder)
    # A section read like `config.get("funnel")` is also recorded; keep it
    # only when nothing was read beneath it, since a parent that led to
    # leaves is not itself a key an operator sets.
    reads = {k for k in recorder.reads if not k.startswith("_")}
    leaves = {
        k for k in reads
        if not any(other.startswith(k + ".") for other in reads)
    }
    return leaves


def _make_watcher(config_path: Path):
    """A ConfigWatcherPlugin wired to a plugin manager, not started."""
    from corvidae.config_watcher import ConfigWatcherPlugin
    from corvidae.hooks import create_plugin_manager

    from corvidae.compaction import CompactionPlugin

    pm = create_plugin_manager()
    watcher = ConfigWatcherPlugin()
    pm.register(watcher, name="config_watcher")
    # Compaction is the plugin whose live keys the warning test relies on.
    pm.register(CompactionPlugin(), name="compaction")
    watcher.pm = pm
    watcher._config_path = config_path
    watcher._cli_overrides = {}
    return watcher


def _make_watcher(config_path: Path):
    """A ConfigWatcherPlugin wired to a plugin manager, not started."""
    from corvidae.config_watcher import ConfigWatcherPlugin
    from corvidae.hooks import create_plugin_manager

    from corvidae.compaction import CompactionPlugin

    pm = create_plugin_manager()
    watcher = ConfigWatcherPlugin()
    pm.register(watcher, name="config_watcher")
    # Compaction is the plugin whose live keys the warning test relies on.
    pm.register(CompactionPlugin(), name="compaction")
    watcher.pm = pm
    watcher._config_path = config_path
    watcher._cli_overrides = {}
    return watcher


class TestRestartOnlyWarning:
    async def test_changing_a_restart_only_key_warns_with_the_key(
        self, tmp_path, caplog
    ):
        """A key no plugin reloads is reported as pending a restart.

        The failure this prevents: changing `memory.retrieval.bands` logged
        only that compaction and agent config were updated, so the operator
        had no way to learn the memory change was waiting for a restart.
        """
        cfg = tmp_path / "agent.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "m"}},
                    "memory": {"retrieval": {"bands": [1, 2, 3]}},
                }
            )
        )
        watcher = _make_watcher(cfg)
        watcher.config = {
            "_config_path": str(cfg),
            "_base_dir": str(tmp_path),
            "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "m"}},
            "memory": {"retrieval": {"bands": [1, 2, 4]}},
        }

        with caplog.at_level(logging.WARNING):
            # The helper is the unit under R2; the full reload path needs a
            # registry, which is not what this test is about.
            watcher._warn_restart_only_changes(
                {
                    "_config_path": str(cfg),
                    "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "m"}},
                    "memory": {"retrieval": {"bands": [1, 2, 3]}},
                }
            )

        pending = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        named = [m for m in pending if "memory.retrieval.bands" in m]
        assert named, f"a restart-only change must name the key; got {pending}"
        assert "restart" in named[0].lower(), named[0]

    async def test_changing_a_live_key_does_not_warn(self, tmp_path, caplog):
        """A key a plugin declares reloadable applies at once, so it is not
        reported as pending."""
        cfg = tmp_path / "agent.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "m"}},
                    "agent": {"compaction_threshold": 0.9},
                }
            )
        )
        watcher = _make_watcher(cfg)
        watcher.config = {
            "_config_path": str(cfg),
            "_base_dir": str(tmp_path),
            "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "m"}},
            "agent": {"compaction_threshold": 0.5},
        }

        with caplog.at_level(logging.WARNING):
            watcher._warn_restart_only_changes(
                {
                    "_config_path": str(cfg),
                    "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "m"}},
                    "agent": {"compaction_threshold": 0.9},
                }
            )

        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not [m for m in messages if "compaction_threshold" in m], messages


# A config with every section the declared hooks below read, so the
# recorder observes each key they take. Values are defaults-shaped.
_FULL_SECTION_CFG = {
    "agent": {
        "chars_per_token": 4,
        "request_logprobs": False,
        "mid_conversation_system_role": "system",
        "compaction_threshold": 0.8,
        "compaction_retention": 0.5,
        "min_messages_to_compact": 5,
        "compaction_fork_prompt": "x",
        "forks": {},
    },
    "funnel": {"default_budget": 512, "budgets": {}},
}


class TestReloadableKeyRegistry:
    """R3: the declared live keys are the ones the hook actually re-reads.

    A declared key that the hook never reads is a lie the operator pays for:
    they change it, see no warning, and wait for an effect that never comes.
    """

    async def test_compaction_declares_what_its_hook_re_reads(self):
        from corvidae.compaction import CompactionPlugin

        hook_keys = await _keys_read_by_hook(CompactionPlugin(None), _FULL_SECTION_CFG)
        assert hook_keys, "could not find any config keys in the hook"
        missing = {
            k for k in hook_keys
            if not any(k == d or k.startswith(d + ".") for d in CompactionPlugin.reloadable_keys)
        }
        assert not missing, (
            f"CompactionPlugin.on_config_reload reads {sorted(missing)} but "
            f"reloadable_keys is {CompactionPlugin.reloadable_keys}"
        )

    async def test_agent_declares_what_its_hook_re_reads(self):
        from corvidae.agent import Agent

        agent_plugin = Agent()
        # The hook re-borrows the LLM client at the end; only reads matter.
        agent_plugin.pm = MagicMock()
        agent_plugin.pm.get_plugin.return_value = None
        hook_keys = await _keys_read_by_hook(agent_plugin, _FULL_SECTION_CFG)
        missing = {
            k for k in hook_keys
            if not any(k == d or k.startswith(d + ".") for d in Agent.reloadable_keys)
        }
        assert not missing, (
            f"Agent.on_config_reload reads {sorted(missing)} but "
            f"reloadable_keys is {Agent.reloadable_keys}"
        )

    async def test_funnel_declares_what_its_hook_re_reads(self):
        # The kernel job runs without the cognition extra, so this test must
        # skip there rather than fail to import (#408).
        pytest.importorskip("corvidae_cognition")
        from corvidae_cognition.funnel import FunnelPlugin

        hook_keys = await _keys_read_by_hook(FunnelPlugin(None), _FULL_SECTION_CFG)
        missing = {
            k for k in hook_keys
            if not any(k == d or k.startswith(d + ".") for d in FunnelPlugin.reloadable_keys)
        }
        assert not missing, (
            f"FunnelPlugin.on_config_reload reads {sorted(missing)} but "
            f"reloadable_keys is {FunnelPlugin.reloadable_keys}"
        )

    async def test_a_second_reload_does_not_re_report_the_first_changes(
        self, tmp_path, caplog
    ):
        """The warning compares against the last reload, not against startup.

        `self.config` is set once in on_init and never advances, so using it
        as the baseline would make reload N re-report everything reload 1
        changed, on every reload, forever.
        """
        cfg = tmp_path / "agent.yaml"
        cfg.write_text(yaml.safe_dump({"llm": {"main": {}}}))
        watcher = _make_watcher(cfg)
        watcher.config = {
            "_config_path": str(cfg),
            "llm": {"main": {"model": "m"}},
            "memory": {"retrieval": {"bands": [1]}},
        }
        changed = {
            "_config_path": str(cfg),
            "llm": {"main": {"model": "m"}},
            "memory": {"retrieval": {"bands": [2]}},
        }

        with caplog.at_level(logging.WARNING):
            watcher._warn_restart_only_changes(changed)
            first = [r.getMessage() for r in caplog.records if "bands" in r.getMessage()]
            assert first, "the first reload must report the change"

            caplog.clear()
            # Same config again: the reload watcher saw this one already.
            watcher._warn_restart_only_changes(changed)
            second = [r.getMessage() for r in caplog.records if "bands" in r.getMessage()]

        assert not second, f"an unchanged key must not be re-reported: {second}"

    async def test_the_reload_path_calls_the_restart_only_report(
        self, tmp_path, caplog
    ):
        """Step 5b runs on a real reload, not only when called directly.

        The other warning tests drive the helper; without this one, deleting
        its call site in `_reload_config` leaves every test green (#408
        review).
        """
        cfg = tmp_path / "agent.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "m"}},
                    "memory": {"retrieval": {"bands": [1, 2, 3]}},
                }
            )
        )
        watcher = _make_watcher(cfg)
        registry = MagicMock()
        watcher._registry = registry
        watcher.config = {
            "_config_path": str(cfg),
            "_base_dir": str(tmp_path),
            "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "m"}},
            "memory": {"retrieval": {"bands": [9, 9, 9]}},
        }

        with patch(
            "corvidae.config_watcher.apply_channel_config", create=True
        ), caplog.at_level(logging.WARNING):
            await watcher._reload_config()

        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("memory.retrieval.bands" in m for m in messages), (
            f"the reload path must report the restart-only change; got {messages}"
        )
