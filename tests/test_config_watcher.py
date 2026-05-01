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
    - CompactionPlugin.on_config_reload: _llm_client cache invalidated
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
# 13. LLMPlugin.on_config_reload creates new client when llm config changes
# ---------------------------------------------------------------------------


async def test_llm_plugin_on_config_reload_creates_new_client_on_change():
    """LLMPlugin.on_config_reload must create a new main_client when llm.main changes."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    await plugin.on_init(
        pm=None,
        config={"llm": {"main": {"base_url": "http://old", "model": "old-model"}}},
    )

    old_client = MagicMock(name="old_client")
    old_client.start = AsyncMock()
    old_client.stop = AsyncMock()
    plugin.main_client = old_client

    new_client = MagicMock(name="new_client")
    new_client.start = AsyncMock()

    new_config = {"llm": {"main": {"base_url": "http://new", "model": "new-model"}}}

    with patch.object(LLMPlugin, "_create_client", return_value=new_client):
        await plugin.on_config_reload(config=new_config)

    assert plugin.main_client is new_client, (
        "on_config_reload must swap main_client to the new client"
    )
    new_client.start.assert_awaited_once()


async def test_llm_plugin_on_config_reload_schedules_old_client_close():
    """LLMPlugin.on_config_reload must schedule closure of the old client
    via asyncio.create_task (to allow in-flight requests to finish)."""
    from corvidae.llm_plugin import LLMPlugin

    plugin = LLMPlugin(pm=None)
    await plugin.on_init(
        pm=None,
        config={"llm": {"main": {"base_url": "http://old", "model": "old-model"}}},
    )

    old_client = MagicMock(name="old_client")
    old_client.start = AsyncMock()
    old_client.stop = AsyncMock()
    plugin.main_client = old_client

    new_client = MagicMock(name="new_client")
    new_client.start = AsyncMock()

    new_config = {"llm": {"main": {"base_url": "http://new", "model": "new-model"}}}

    tasks_created = []
    original_create_task = asyncio.create_task

    def tracking_create_task(coro, **kwargs):
        task = original_create_task(coro, **kwargs)
        tasks_created.append(task)
        return task

    with patch.object(LLMPlugin, "_create_client", return_value=new_client):
        with patch("asyncio.create_task", side_effect=tracking_create_task):
            await plugin.on_config_reload(config=new_config)

    assert tasks_created, (
        "on_config_reload must schedule old client closure via asyncio.create_task"
    )


async def test_llm_plugin_on_config_reload_no_change_no_swap():
    """LLMPlugin.on_config_reload must NOT create a new client when llm.main is unchanged."""
    from corvidae.llm_plugin import LLMPlugin

    same_config = {"llm": {"main": {"base_url": "http://localhost", "model": "m"}}}
    plugin = LLMPlugin(pm=None)
    await plugin.on_init(pm=None, config=same_config)

    original_client = MagicMock(name="original_client")
    original_client.start = AsyncMock()
    plugin.main_client = original_client

    create_calls = []
    original_create = LLMPlugin._create_client

    # Use module-level patch to avoid staticmethod binding issue where
    # patch.object would pass `self` as the first argument.
    with patch("corvidae.llm_plugin.LLMPlugin._create_client", side_effect=lambda cfg: (create_calls.append(cfg), original_create(cfg))[1]):
        await plugin.on_config_reload(config=same_config)

    assert not create_calls, (
        "on_config_reload must not create a new client when llm.main config is unchanged"
    )
    assert plugin.main_client is original_client


# ---------------------------------------------------------------------------
# 14. CompactionPlugin.on_config_reload invalidates _llm_client cache
# ---------------------------------------------------------------------------


async def test_compaction_plugin_on_config_reload_invalidates_llm_client():
    """CompactionPlugin.on_config_reload must set _llm_client = None
    so it is re-resolved from LLMPlugin on next use (I1 from design review Fix #1)."""
    from corvidae.compaction import CompactionPlugin

    plugin = CompactionPlugin(pm=None)
    await plugin.on_init(
        pm=None,
        config={"agent": {}, "llm": {"main": {"base_url": "http://localhost", "model": "m"}}},
    )

    # Simulate a cached LLM client
    plugin._llm_client = MagicMock(name="stale_client")

    new_config = {
        "agent": {"compaction_threshold": 0.9},
        "llm": {"main": {"base_url": "http://localhost", "model": "m"}},
    }
    await plugin.on_config_reload(config=new_config)

    assert plugin._llm_client is None, (
        "CompactionPlugin.on_config_reload must reset _llm_client to None"
    )


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


# ---------------------------------------------------------------------------
# 20. HotReloadPlugin.on_config_reload updates self.config
# ---------------------------------------------------------------------------


async def test_hot_reload_plugin_on_config_reload_updates_config():
    """HotReloadPlugin.on_config_reload must update self.config for newly loaded plugins."""
    from corvidae.hot_reload import HotReloadPlugin
    plugin = HotReloadPlugin()
    plugin.config = {"old": "config"}
    new_config = {"new": "config", "llm": {"main": {"model": "test"}}}
    await plugin.on_config_reload(config=new_config)
    assert plugin.config == new_config
