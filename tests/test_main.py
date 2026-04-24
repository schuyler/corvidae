"""Tests for sherman.main."""

import asyncio
import os
import signal
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from sherman.channel import ChannelRegistry
from sherman.main import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(path: str, data: dict | None = None) -> None:
    """Write a minimal YAML config file."""
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


def _schedule_sigint(delay: float = 0.05) -> None:
    """Schedule a SIGINT to the current process after *delay* seconds."""
    loop = asyncio.get_running_loop()
    loop.call_later(delay, os.kill, os.getpid(), signal.SIGINT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMainLoadsConfigAndCreatesPM:
    async def test_main_loads_config_and_creates_pm(self):
        """main() should call create_plugin_manager after loading the config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            with patch("sherman.main.create_plugin_manager") as mock_create:
                # Return a minimal object that satisfies pm.ahook.on_start / on_stop
                mock_pm = MagicMock()
                mock_pm.ahook.on_start = AsyncMock(return_value=[])
                mock_pm.ahook.on_stop = AsyncMock(return_value=[])
                mock_create.return_value = mock_pm

                _schedule_sigint()
                await main(config_path)

                mock_create.assert_called_once()
        finally:
            os.unlink(config_path)


class TestMainCallsOnStartAndOnStop:
    async def test_main_calls_on_start_and_on_stop(self):
        """main() should call on_start and on_stop on all registered plugins."""
        from sherman.hooks import hookimpl
        from sherman.plugin_manager import create_plugin_manager

        on_start_mock = AsyncMock()
        on_stop_mock = AsyncMock()

        class MockPlugin:
            @hookimpl
            async def on_start(self, config):
                await on_start_mock(config=config)

            @hookimpl
            async def on_stop(self):
                await on_stop_mock()

        def patched_create_plugin_manager():
            pm = create_plugin_manager()
            pm.register(MockPlugin())
            return pm

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_client = MagicMock()
            mock_client.start = AsyncMock()
            mock_client.stop = AsyncMock()
            with patch(
                "sherman.main.create_plugin_manager",
                side_effect=patched_create_plugin_manager,
            ), patch(
                "sherman.agent_loop_plugin.LLMClient",
                return_value=mock_client,
            ), patch(
                "sherman.agent_loop_plugin.aiosqlite.connect",
                new_callable=AsyncMock,
            ) as mock_connect, patch(
                "sherman.agent_loop_plugin.init_db",
                new_callable=AsyncMock,
            ):
                mock_db = MagicMock()
                mock_db.close = AsyncMock()
                mock_connect.return_value = mock_db
                _schedule_sigint()
                await main(config_path)

            on_start_mock.assert_called_once()
            on_stop_mock.assert_called_once()
        finally:
            os.unlink(config_path)


class TestRegistryPopulatedBeforeOnStart:
    async def test_registry_populated_before_on_start(self):
        """pm.registry must be a ChannelRegistry with pre-configured channels
        by the time on_start fires."""
        from sherman.hooks import hookimpl
        from sherman.plugin_manager import create_plugin_manager

        registry_snapshot: list = []

        class InspectorPlugin:
            @hookimpl
            async def on_start(self, config):
                # Capture the registry state at on_start time via closure over pm
                registry_snapshot.append(pm.registry)

        inspector = InspectorPlugin()

        def patched_create_plugin_manager():
            nonlocal pm
            pm = create_plugin_manager()
            pm.register(inspector)
            return pm

        pm = None

        config_data = {
            "llm": {
                "main": {
                    "base_url": "http://localhost:8080",
                    "model": "test-model",
                },
            },
            "channels": {
                "irc:#lex": {
                    "system_prompt": "You are a helpful assistant for #lex.",
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            mock_client = MagicMock()
            mock_client.start = AsyncMock()
            mock_client.stop = AsyncMock()
            with patch(
                "sherman.main.create_plugin_manager",
                side_effect=patched_create_plugin_manager,
            ), patch(
                "sherman.agent_loop_plugin.LLMClient",
                return_value=mock_client,
            ), patch(
                "sherman.agent_loop_plugin.aiosqlite.connect",
                new_callable=AsyncMock,
            ) as mock_connect, patch(
                "sherman.agent_loop_plugin.init_db",
                new_callable=AsyncMock,
            ):
                mock_db = MagicMock()
                mock_db.close = AsyncMock()
                mock_connect.return_value = mock_db
                _schedule_sigint()
                await main(config_path)
        finally:
            os.unlink(config_path)

        assert len(registry_snapshot) == 1, "on_start was not called exactly once"
        registry = registry_snapshot[0]
        assert isinstance(registry, ChannelRegistry)
        channel = registry.get("irc:#lex")
        assert channel is not None, "irc:#lex not found in registry at on_start time"
        assert channel.config.system_prompt == "You are a helpful assistant for #lex."


class TestMainMissingConfig:
    async def test_main_missing_config_raises(self):
        """main() should raise FileNotFoundError when the config file is absent."""
        with pytest.raises(FileNotFoundError):
            await main("nonexistent_config_file_that_does_not_exist.yaml")
