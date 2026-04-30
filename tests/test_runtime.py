"""Tests for corvidae.runtime — Runtime lifecycle and deep_merge utility.

All tests in this file are RED (failing) until corvidae/runtime.py is
implemented. They fail with ImportError or AttributeError, not test bugs.

Patch targets:
    corvidae.runtime.asyncio.wait_for   — shutdown timeout
    corvidae.runtime.create_plugin_manager
    corvidae.runtime.validate_dependencies
    corvidae.runtime.configure_logging
"""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(path: str, data: dict | None = None) -> None:
    """Write a minimal YAML config file to path."""
    config = data or {
        "llm": {
            "main": {
                "base_url": "http://localhost:8080",
                "model": "test-model",
            },
        },
    }
    with open(path, "w") as f:
        yaml.dump(config, f)


def _make_mock_pm_and_agent():
    """Return a (mock_pm, mock_agent) pair suitable for Runtime patches."""
    mock_pm = MagicMock()
    mock_pm.ahook.on_init = AsyncMock(return_value=[])
    mock_pm.ahook.on_start = AsyncMock(return_value=[])
    mock_pm.ahook.on_stop = AsyncMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.on_start = AsyncMock()
    mock_agent.on_stop = AsyncMock()
    mock_pm.get_plugin.return_value = mock_agent
    return mock_pm, mock_agent


# ---------------------------------------------------------------------------
# Section 1 — deep_merge utility
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """Unit tests for the standalone deep_merge(base, overrides) function."""

    def test_dict_merge_is_recursive(self):
        """Nested dicts are merged recursively, not replaced wholesale."""
        from corvidae.runtime import deep_merge

        base = {"channels": {"irc:#lex": {"system_prompt": "old"}}}
        overrides = {"channels": {"cli:local": {}}}
        result = deep_merge(base, overrides)

        # Both keys survive; the override adds without removing the base key.
        assert "irc:#lex" in result["channels"]
        assert "cli:local" in result["channels"]

    def test_non_dict_value_replaces(self):
        """A non-dict override value replaces the base value."""
        from corvidae.runtime import deep_merge

        base = {"logging": {"file": "old.log", "level": "INFO"}}
        overrides = {"logging": {"file": "new.log"}}
        result = deep_merge(base, overrides)

        assert result["logging"]["file"] == "new.log"
        # Other keys in the nested dict are preserved.
        assert result["logging"]["level"] == "INFO"

    def test_none_override_is_skipped(self):
        """An override value of None leaves the base value unchanged."""
        from corvidae.runtime import deep_merge

        base = {"logging": {"file": "keep.log"}}
        overrides = {"logging": {"file": None}}
        result = deep_merge(base, overrides)

        # None does not delete or replace the YAML value.
        assert result["logging"]["file"] == "keep.log"

    def test_top_level_key_added_by_override(self):
        """An override key absent from base is added to the result."""
        from corvidae.runtime import deep_merge

        base = {"llm": {"main": {"model": "gpt-4"}}}
        overrides = {"channels": {"cli:local": {}}}
        result = deep_merge(base, overrides)

        assert "channels" in result
        assert result["channels"] == {"cli:local": {}}

    def test_base_unchanged_when_none_override(self):
        """deep_merge with a top-level None override preserves the base key."""
        from corvidae.runtime import deep_merge

        base = {"key": "value"}
        overrides = {"key": None}
        result = deep_merge(base, overrides)

        assert result["key"] == "value"

    def test_empty_overrides_returns_copy_of_base(self):
        """Empty overrides dict returns a result equal to base."""
        from corvidae.runtime import deep_merge

        base = {"a": 1, "b": {"c": 2}}
        result = deep_merge(base, {})

        assert result == base

    def test_original_base_not_mutated(self):
        """deep_merge must not mutate its base argument."""
        from corvidae.runtime import deep_merge

        base = {"channels": {"irc:#lex": {}}}
        original_base = {"channels": {"irc:#lex": {}}}
        deep_merge(base, {"channels": {"cli:local": {}}})

        assert base == original_base


# ---------------------------------------------------------------------------
# Section 2 — Runtime.__init__
# ---------------------------------------------------------------------------


class TestRuntimeInit:
    """Runtime constructor stores config_path and overrides correctly."""

    def test_stores_config_path_default(self):
        """Runtime() with no args defaults config_path to 'agent.yaml'."""
        from corvidae.runtime import Runtime

        rt = Runtime()
        assert rt.config_path == "agent.yaml"

    def test_stores_custom_config_path(self):
        """Runtime(config_path=...) stores the supplied path."""
        from corvidae.runtime import Runtime

        rt = Runtime(config_path="custom.yaml")
        assert rt.config_path == "custom.yaml"

    def test_overrides_default_is_empty_dict(self):
        """Runtime() with no overrides defaults to {}."""
        from corvidae.runtime import Runtime

        rt = Runtime()
        assert rt.overrides == {}

    def test_stores_overrides(self):
        """Runtime(overrides={...}) stores the supplied overrides dict."""
        from corvidae.runtime import Runtime

        overrides = {"logging": {"file": "corvidae.log"}}
        rt = Runtime(overrides=overrides)
        assert rt.overrides == overrides

    def test_pm_starts_none(self):
        """pm attribute is None before start() is called."""
        from corvidae.runtime import Runtime

        rt = Runtime()
        assert rt.pm is None

    def test_registry_starts_none(self):
        """registry attribute is None before start() is called."""
        from corvidae.runtime import Runtime

        rt = Runtime()
        assert rt.registry is None


# ---------------------------------------------------------------------------
# Section 3 — Runtime.start() lifecycle
# ---------------------------------------------------------------------------


class TestRuntimeStart:
    """Runtime.start() lifecycle: config load, merge, logging, PM creation."""

    async def test_start_loads_config_and_creates_pm(self):
        """start() should call create_plugin_manager after loading the config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()
            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ) as mock_create, patch(
                "corvidae.runtime.validate_dependencies"
            ), patch(
                "corvidae.runtime.configure_logging"
            ):
                from corvidae.runtime import Runtime

                rt = Runtime(config_path=config_path)
                await rt.start()

                mock_create.assert_called_once()
        finally:
            os.unlink(config_path)

    async def test_start_calls_on_init_before_on_start(self):
        """start() must call pm.ahook.on_init before pm.ahook.on_start."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        call_order: list[str] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def record_on_init(**kwargs):
                call_order.append("on_init")
                return []

            async def record_on_start(**kwargs):
                call_order.append("on_start")
                return []

            mock_pm.ahook.on_init = AsyncMock(side_effect=record_on_init)
            mock_pm.ahook.on_start = AsyncMock(side_effect=record_on_start)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()

            assert "on_init" in call_order
            assert "on_start" in call_order
            assert call_order.index("on_init") < call_order.index("on_start")
        finally:
            os.unlink(config_path)

    async def test_start_calls_agent_on_start_after_broadcast(self):
        """start() must call agent.on_start() explicitly AFTER pm.ahook.on_start().

        Agent.on_start has no @hookimpl so it is not reached by the broadcast.
        start() must call it explicitly after the broadcast completes.
        """
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        call_order: list[str] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def record_broadcast_on_start(**kwargs):
                call_order.append("broadcast_on_start")
                return []

            async def record_agent_on_start(**kwargs):
                call_order.append("agent_on_start")

            mock_pm.ahook.on_start = AsyncMock(
                side_effect=record_broadcast_on_start
            )
            mock_agent.on_start = AsyncMock(side_effect=record_agent_on_start)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()

            assert "broadcast_on_start" in call_order
            assert "agent_on_start" in call_order
            assert call_order.index("broadcast_on_start") < call_order.index(
                "agent_on_start"
            ), f"agent.on_start() must come AFTER pm.ahook.on_start(); order: {call_order}"
        finally:
            os.unlink(config_path)

    async def test_start_raises_if_no_agent_plugin(self):
        """start() raises RuntimeError when no 'agent' plugin is registered."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm = MagicMock()
            mock_pm.ahook.on_init = AsyncMock(return_value=[])
            mock_pm.ahook.on_start = AsyncMock(return_value=[])
            mock_pm.get_plugin.return_value = None  # no agent registered

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                with pytest.raises(RuntimeError, match="[Aa]gent"):
                    await rt.start()
        finally:
            os.unlink(config_path)

    async def test_start_missing_config_raises_file_not_found(self):
        """start() raises FileNotFoundError when config_path does not exist."""
        from corvidae.runtime import Runtime

        rt = Runtime(config_path="nonexistent_config_that_does_not_exist.yaml")
        with pytest.raises(FileNotFoundError):
            await rt.start()

    async def test_start_sets_base_dir_from_config_path(self):
        """start() sets config['_base_dir'] to the directory of config_path."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        captured_config: list[dict] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def capture_config(**kwargs):
                # Capture the config dict passed to on_init
                captured_config.append(kwargs.get("config", {}))
                return []

            mock_pm.ahook.on_init = AsyncMock(side_effect=capture_config)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()

            assert len(captured_config) == 1
            config = captured_config[0]
            assert "_base_dir" in config
            assert config["_base_dir"] == Path(config_path).parent
        finally:
            os.unlink(config_path)

    async def test_base_dir_not_overrideable_via_overrides(self):
        """_base_dir in overrides does not persist — it's always set from config_path."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        captured_config: list[dict] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def capture_config(**kwargs):
                captured_config.append(kwargs.get("config", {}))
                return []

            mock_pm.ahook.on_init = AsyncMock(side_effect=capture_config)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(
                    config_path=config_path,
                    overrides={"_base_dir": Path("/should/not/survive")},
                )
                await rt.start()

            config = captured_config[0]
            # _base_dir must always be derived from config_path, never from overrides
            assert config["_base_dir"] == Path(config_path).parent
        finally:
            os.unlink(config_path)

    async def test_overrides_are_merged_into_config(self):
        """start() deep-merges overrides on top of the YAML config."""
        from corvidae.runtime import Runtime

        base_config_data = {
            "llm": {"main": {"base_url": "http://localhost:8080", "model": "gpt-4"}},
            "channels": {"irc:#lex": {}},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(base_config_data, f)
            config_path = f.name

        captured_config: list[dict] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def capture_config(**kwargs):
                captured_config.append(kwargs.get("config", {}))
                return []

            mock_pm.ahook.on_init = AsyncMock(side_effect=capture_config)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(
                    config_path=config_path,
                    overrides={"channels": {"cli:local": {}}},
                )
                await rt.start()

            config = captured_config[0]
            # Override added cli:local
            assert "cli:local" in config["channels"]
            # YAML-defined irc:#lex survived the merge
            assert "irc:#lex" in config["channels"]
        finally:
            os.unlink(config_path)

    async def test_configure_logging_called_first(self):
        """configure_logging() is called before create_plugin_manager()."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        call_order: list[str] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            def record_configure_logging(**kwargs):
                call_order.append("configure_logging")

            def record_create_pm():
                call_order.append("create_pm")
                return mock_pm

            with patch(
                "corvidae.runtime.configure_logging",
                side_effect=record_configure_logging,
            ), patch(
                "corvidae.runtime.create_plugin_manager",
                side_effect=record_create_pm,
            ), patch(
                "corvidae.runtime.validate_dependencies"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()

            assert "configure_logging" in call_order
            assert "create_pm" in call_order
            assert call_order.index("configure_logging") < call_order.index(
                "create_pm"
            ), f"configure_logging must be called before create_plugin_manager; order: {call_order}"
        finally:
            os.unlink(config_path)

    async def test_start_validates_dependencies(self):
        """start() calls validate_dependencies after loading plugins."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch(
                "corvidae.runtime.validate_dependencies"
            ) as mock_validate, patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()

            mock_validate.assert_called_once_with(mock_pm)
        finally:
            os.unlink(config_path)

    async def test_start_loads_setuptools_entrypoints(self):
        """start() calls pm.load_setuptools_entrypoints('corvidae') to load plugins."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()

            mock_pm.load_setuptools_entrypoints.assert_called_once_with("corvidae")
        finally:
            os.unlink(config_path)

    async def test_start_creates_channel_registry(self):
        """start() creates a ChannelRegistry and registers it with the plugin manager."""
        from corvidae.channel import ChannelRegistry
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        registered_plugins: list = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            original_register = mock_pm.register

            def capture_register(plugin, **kwargs):
                registered_plugins.append(plugin)
                return original_register(plugin, **kwargs)

            mock_pm.register = capture_register

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()

            # A ChannelRegistry must have been registered with the PM.
            registry_plugins = [
                p for p in registered_plugins if isinstance(p, ChannelRegistry)
            ]
            assert len(registry_plugins) == 1, (
                "start() must register exactly one ChannelRegistry with the PM"
            )
        finally:
            os.unlink(config_path)

    async def test_start_calls_load_channel_config(self):
        """start() calls load_channel_config with the merged config."""
        from corvidae.runtime import Runtime

        config_data = {
            "llm": {"main": {"base_url": "http://localhost:8080", "model": "m"}},
            "channels": {"irc:#lex": {"system_prompt": "test"}},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ), patch(
                "corvidae.runtime.load_channel_config"
            ) as mock_load_channel_config:
                rt = Runtime(config_path=config_path)
                await rt.start()

            mock_load_channel_config.assert_called_once()
            # The merged config (with channels section) must be passed as
            # the first positional argument or as 'config' keyword argument.
            call_args = mock_load_channel_config.call_args
            if call_args[0]:
                passed_config = call_args[0][0]
            else:
                passed_config = call_args[1].get("config")
            assert passed_config is not None, "load_channel_config must receive the config"
            assert "channels" in passed_config
        finally:
            os.unlink(config_path)


# ---------------------------------------------------------------------------
# Section 4 — Runtime.stop() graceful shutdown
# ---------------------------------------------------------------------------


class TestRuntimeStop:
    """Runtime.stop() calls agent.on_stop before pm.ahook.on_stop, with timeout."""

    async def test_stop_calls_agent_on_stop_before_broadcast(self):
        """stop() calls agent.on_stop() before pm.ahook.on_stop()."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        call_order: list[str] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def record_agent_on_stop(**kwargs):
                call_order.append("agent_on_stop")

            async def record_broadcast_on_stop(**kwargs):
                call_order.append("broadcast_on_stop")
                return []

            mock_agent.on_stop = AsyncMock(side_effect=record_agent_on_stop)
            mock_pm.ahook.on_stop = AsyncMock(side_effect=record_broadcast_on_stop)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()
                await rt.stop()

            assert "agent_on_stop" in call_order
            assert "broadcast_on_stop" in call_order
            assert call_order.index("agent_on_stop") < call_order.index(
                "broadcast_on_stop"
            ), f"agent.on_stop() must precede broadcast; order: {call_order}"
        finally:
            os.unlink(config_path)

    async def test_stop_raises_if_no_agent_registered(self):
        """stop() raises RuntimeError when no 'agent' plugin is registered."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, _ = _make_mock_pm_and_agent()

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(config_path=config_path)
                await rt.start()
                # Remove agent so stop() cannot find it
                mock_pm.get_plugin.return_value = None
                with pytest.raises(RuntimeError):
                    await rt.stop()
        finally:
            os.unlink(config_path)

    async def test_slow_shutdown_calls_os_exit_after_timeout(self):
        """When shutdown hangs, os._exit(1) is called after the timeout.

        Patches corvidae.runtime.asyncio.wait_for to raise TimeoutError
        immediately so the test stays fast.
        """
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def _patched_wait_for(coro, timeout):
                coro.close()  # prevent "coroutine never awaited" warning
                raise asyncio.TimeoutError()

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ), patch(
                "corvidae.runtime.asyncio.wait_for",
                side_effect=_patched_wait_for,
            ), patch(
                "os._exit"
            ) as mock_os_exit:
                rt = Runtime(config_path=config_path)
                await rt.start()
                await rt.stop()

            mock_os_exit.assert_called_once_with(1)
        finally:
            os.unlink(config_path)


# ---------------------------------------------------------------------------
# Section 5 — Runtime.run() signal handling
# ---------------------------------------------------------------------------


class TestRuntimeRun:
    """Runtime.run() calls start(), waits for signal, then stop()."""

    async def test_run_calls_start_and_stop(self):
        """run() calls start() then stop() after a SIGINT."""
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        call_order: list[str] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def record_broadcast_on_start(**kwargs):
                call_order.append("on_start")
                return []

            async def record_broadcast_on_stop(**kwargs):
                call_order.append("on_stop")
                return []

            mock_pm.ahook.on_start = AsyncMock(side_effect=record_broadcast_on_start)
            mock_pm.ahook.on_stop = AsyncMock(side_effect=record_broadcast_on_stop)

            loop = asyncio.get_running_loop()

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                # Schedule SIGINT after a brief delay so run() has time to start
                loop.call_later(0.05, os.kill, os.getpid(), signal.SIGINT)
                rt = Runtime(config_path=config_path)
                await rt.run()

            assert "on_start" in call_order
            assert "on_stop" in call_order
            assert call_order.index("on_start") < call_order.index("on_stop")
        finally:
            os.unlink(config_path)

    async def test_double_sigint_calls_os_exit(self):
        """Second SIGINT during shutdown triggers os._exit(1) — no graceful cleanup.

        Uses event-gating to ensure agent.on_stop() blocks until the second
        signal has been sent, preventing a race where run() exits before the
        second signal fires. Polls mock_os_exit.called before unblocking to
        account for asyncio signal-handler dispatch latency.
        """
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            # Block on_stop until the second signal has been sent.
            shutdown_started = asyncio.Event()
            shutdown_proceed = asyncio.Event()

            async def _blocking_on_stop(**kwargs):
                shutdown_started.set()
                await shutdown_proceed.wait()

            mock_agent.on_stop = AsyncMock(side_effect=_blocking_on_stop)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ), patch(
                "os._exit"
            ) as mock_os_exit:
                loop = asyncio.get_running_loop()

                async def _orchestrate():
                    os.kill(os.getpid(), signal.SIGINT)       # first signal
                    await shutdown_started.wait()              # wait for shutdown entry
                    os.kill(os.getpid(), signal.SIGINT)        # second signal while hung
                    # Signal handler fires after multiple event-loop iterations
                    # (wakeup fd -> reader callback -> _handle_signal -> call_soon).
                    # Poll until the mock records the call before unblocking run().
                    while not mock_os_exit.called:
                        await asyncio.sleep(0)
                    shutdown_proceed.set()                     # unblock so run() can exit

                loop.call_later(0.05, lambda: asyncio.ensure_future(_orchestrate()))
                rt = Runtime(config_path=config_path)
                await rt.run()

            mock_os_exit.assert_called_once_with(1)
        finally:
            os.unlink(config_path)

    async def test_sigterm_then_sigint_calls_os_exit(self):
        """SIGTERM followed by SIGINT triggers os._exit(1).

        Any second interrupt signal (regardless of type) should force-exit.
        Uses event-gating to ensure agent.on_stop() blocks until the second
        signal has been sent.
        """
        from corvidae.runtime import Runtime

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            shutdown_started = asyncio.Event()
            shutdown_proceed = asyncio.Event()

            async def _blocking_on_stop(**kwargs):
                shutdown_started.set()
                await shutdown_proceed.wait()

            mock_agent.on_stop = AsyncMock(side_effect=_blocking_on_stop)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ), patch(
                "os._exit"
            ) as mock_os_exit:
                loop = asyncio.get_running_loop()

                async def _orchestrate():
                    os.kill(os.getpid(), signal.SIGTERM)       # first signal
                    await shutdown_started.wait()               # wait for shutdown entry
                    os.kill(os.getpid(), signal.SIGINT)         # second signal while hung
                    while not mock_os_exit.called:
                        await asyncio.sleep(0)
                    shutdown_proceed.set()                      # unblock so run() can exit

                loop.call_later(0.05, lambda: asyncio.ensure_future(_orchestrate()))
                rt = Runtime(config_path=config_path)
                await rt.run()

            mock_os_exit.assert_called_once_with(1)
        finally:
            os.unlink(config_path)


# ---------------------------------------------------------------------------
# Section 6 — Config merge semantics (integration-level)
# ---------------------------------------------------------------------------


class TestConfigMergeSemantics:
    """Verify the full merge rules from design section 4."""

    async def test_logging_file_override_replaces_yaml(self):
        """Overrides can replace a scalar in the logging section."""
        from corvidae.runtime import Runtime

        base_config_data = {
            "llm": {"main": {"base_url": "http://localhost:8080", "model": "m"}},
            "logging": {"file": "original.log", "level": "DEBUG"},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(base_config_data, f)
            config_path = f.name

        captured_config: list[dict] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def capture_config(**kwargs):
                captured_config.append(kwargs.get("config", {}))
                return []

            mock_pm.ahook.on_init = AsyncMock(side_effect=capture_config)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(
                    config_path=config_path,
                    overrides={"logging": {"file": "corvidae.log"}},
                )
                await rt.start()

            config = captured_config[0]
            # Override wins for the scalar field
            assert config["logging"]["file"] == "corvidae.log"
            # Other logging fields survive
            assert config["logging"]["level"] == "DEBUG"
        finally:
            os.unlink(config_path)

    async def test_none_override_preserves_yaml_value(self):
        """A None override does not delete the YAML-configured value."""
        from corvidae.runtime import Runtime

        base_config_data = {
            "llm": {"main": {"base_url": "http://localhost:8080", "model": "m"}},
            "logging": {"file": "important.log"},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(base_config_data, f)
            config_path = f.name

        captured_config: list[dict] = []

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def capture_config(**kwargs):
                captured_config.append(kwargs.get("config", {}))
                return []

            mock_pm.ahook.on_init = AsyncMock(side_effect=capture_config)

            with patch(
                "corvidae.runtime.create_plugin_manager", return_value=mock_pm
            ), patch("corvidae.runtime.validate_dependencies"), patch(
                "corvidae.runtime.configure_logging"
            ):
                rt = Runtime(
                    config_path=config_path,
                    overrides={"logging": {"file": None}},
                )
                await rt.start()

            config = captured_config[0]
            # None override must not wipe out the YAML value
            assert config["logging"]["file"] == "important.log"
        finally:
            os.unlink(config_path)
