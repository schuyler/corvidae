"""Tests for corvidae.main."""

import asyncio
import os
import signal
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from corvidae.channel import ChannelRegistry
from corvidae.main import main


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
            mock_pm, mock_agent = _make_mock_pm_and_agent()
            with patch("corvidae.main.create_plugin_manager", return_value=mock_pm) as mock_create:
                _schedule_sigint()
                await main(config_path)

                mock_create.assert_called_once()
        finally:
            os.unlink(config_path)


class TestAgentLifecycleOrdering:
    async def test_agent_on_start_called_after_broadcast(self):
        """main() must call agent_loop.on_start() explicitly AFTER pm.ahook.on_start().

        After the race-condition fix, Agent.on_start no longer participates
        in the broadcast.  main() calls pm.ahook.on_start() first (all other plugins
        init), then calls agent_loop.on_start() explicitly so that tool-providing
        plugins (e.g. McpClientPlugin) are ready before tools are collected.

        This test fails against the current code because the current main() only
        calls pm.ahook.on_start() and relies on the @hookimpl broadcast — it never
        makes the explicit agent_loop.on_start() call recorded here.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        call_order: list[str] = []

        try:
            mock_pm = MagicMock()
            mock_agent = MagicMock()

            async def broadcast_on_init(**kwargs):
                call_order.append("broadcast_on_init")
                return []

            async def broadcast_on_start(**kwargs):
                call_order.append("broadcast_on_start")
                return []

            async def broadcast_on_stop(**kwargs):
                call_order.append("broadcast_on_stop")
                return []

            mock_pm.ahook.on_init = AsyncMock(side_effect=broadcast_on_init)
            mock_pm.ahook.on_start = AsyncMock(side_effect=broadcast_on_start)
            mock_pm.ahook.on_stop = AsyncMock(side_effect=broadcast_on_stop)
            mock_pm.get_plugin.return_value = mock_agent

            async def agent_on_start(**kwargs):
                call_order.append("agent_on_start")

            async def agent_on_stop(**kwargs):
                call_order.append("agent_on_stop")

            mock_agent.on_start = AsyncMock(side_effect=agent_on_start)
            mock_agent.on_stop = AsyncMock(side_effect=agent_on_stop)

            with patch("corvidae.main.create_plugin_manager", return_value=mock_pm):
                _schedule_sigint()
                await main(config_path)

            # Startup: broadcast must fire before explicit agent.on_start
            assert "broadcast_on_start" in call_order, (
                "pm.ahook.on_start() was not called"
            )
            assert "agent_on_start" in call_order, (
                "agent_loop.on_start() was not called explicitly — "
                "after the fix main() must call it after the broadcast"
            )
            broadcast_start_idx = call_order.index("broadcast_on_start")
            agent_start_idx = call_order.index("agent_on_start")
            assert broadcast_start_idx < agent_start_idx, (
                f"agent_loop.on_start() must be called AFTER pm.ahook.on_start(), "
                f"but order was: {call_order}"
            )

            # Shutdown: explicit agent.on_stop must fire before broadcast
            assert "broadcast_on_stop" in call_order, (
                "pm.ahook.on_stop() was not called"
            )
            assert "agent_on_stop" in call_order, (
                "agent_loop.on_stop() was not called explicitly — "
                "after the fix main() must call it before the broadcast"
            )
            agent_stop_idx = call_order.index("agent_on_stop")
            broadcast_stop_idx = call_order.index("broadcast_on_stop")
            assert agent_stop_idx < broadcast_stop_idx, (
                f"agent_loop.on_stop() must be called BEFORE pm.ahook.on_stop(), "
                f"but order was: {call_order}"
            )
        finally:
            os.unlink(config_path)


class TestMainCallsOnStartAndOnStop:
    async def test_main_calls_on_start_and_on_stop(self):
        """main() should call on_start and on_stop on all registered plugins."""
        from corvidae.hooks import hookimpl
        from corvidae.hooks import create_plugin_manager

        on_start_mock = AsyncMock()
        on_stop_mock = AsyncMock()

        class MockPlugin:
            @hookimpl
            async def on_start(self, config):
                await on_start_mock(config=config)

            @hookimpl
            async def on_stop(self):
                await on_stop_mock()

        class _MockAgent:
            async def on_start(self, config):
                pass

            async def on_stop(self):
                pass

        mock_agent = _MockAgent()

        def patched_create_plugin_manager():
            pm = create_plugin_manager()
            pm.register(MockPlugin())
            # Suppress real entry-point loading and register a mock agent so
            # main() can find pm.get_plugin("agent") without loading the full
            # plugin stack (which requires mocking LLM, DB, aiohttp, etc.).
            pm.load_setuptools_entrypoints = lambda *a, **kw: None
            pm.register(mock_agent, name="agent")
            return pm

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            with patch(
                "corvidae.main.create_plugin_manager",
                side_effect=patched_create_plugin_manager,
            ), patch(
                "corvidae.main.validate_dependencies",
            ):
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
        from corvidae.hooks import hookimpl
        from corvidae.hooks import create_plugin_manager

        registry_snapshot: list = []

        class InspectorPlugin:
            @hookimpl
            async def on_start(self, config):
                # Capture the registry state at on_start time via closure over pm
                registry_snapshot.append(pm.get_plugin("registry"))

        inspector = InspectorPlugin()

        class _MockAgent:
            async def on_start(self, config):
                pass

            async def on_stop(self):
                pass

        mock_agent = _MockAgent()

        def patched_create_plugin_manager():
            nonlocal pm
            pm = create_plugin_manager()
            pm.register(inspector)
            # Suppress real entry-point loading and register a mock agent so
            # main() can find pm.get_plugin("agent") without loading the full
            # plugin stack.
            pm.load_setuptools_entrypoints = lambda *a, **kw: None
            pm.register(mock_agent, name="agent")
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
            with patch(
                "corvidae.main.create_plugin_manager",
                side_effect=patched_create_plugin_manager,
            ), patch(
                "corvidae.main.validate_dependencies",
            ):
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


# ---------------------------------------------------------------------------
# Helpers shared by graceful-shutdown tests
# ---------------------------------------------------------------------------


def _make_mock_pm_and_agent():
    """Return a (mock_pm, mock_agent) pair suitable for main() patches."""
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
# TestDoubleSignalForceExit
# ---------------------------------------------------------------------------


class TestDoubleSignalForceExit:
    """A second interrupt signal must call os._exit(1) immediately."""

    async def test_double_sigint_calls_os_exit(self):
        """Second SIGINT triggers os._exit(1) — no graceful cleanup.

        Uses event-gating to ensure agent.on_stop() blocks until the second
        signal has been sent, preventing a race where main() exits before
        the second signal fires. Polls mock_os_exit.called before unblocking
        to account for asyncio signal-handler dispatch latency.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            # Block on_stop until the second signal has been sent.
            shutdown_started = asyncio.Event()
            shutdown_proceed = asyncio.Event()

            async def _blocking_on_stop():
                shutdown_started.set()
                await shutdown_proceed.wait()

            mock_agent.on_stop = AsyncMock(side_effect=_blocking_on_stop)

            with (
                patch("corvidae.main.create_plugin_manager", return_value=mock_pm),
                patch("os._exit") as mock_os_exit,
            ):
                loop = asyncio.get_running_loop()

                async def _orchestrate():
                    os.kill(os.getpid(), signal.SIGINT)       # first signal
                    await shutdown_started.wait()              # wait for shutdown entry
                    os.kill(os.getpid(), signal.SIGINT)        # second signal while hung
                    # Signal handler fires after multiple event-loop iterations
                    # (wakeup fd → reader callback → _handle_signal → call_soon).
                    # Poll until the mock records the call before unblocking main().
                    while not mock_os_exit.called:
                        await asyncio.sleep(0)
                    shutdown_proceed.set()                     # unblock so main() can exit

                loop.call_later(0.05, lambda: asyncio.ensure_future(_orchestrate()))
                await main(config_path)

            mock_os_exit.assert_called_once_with(1)
        finally:
            os.unlink(config_path)

    async def test_sigterm_then_sigint_calls_os_exit(self):
        """SIGTERM followed by SIGINT triggers os._exit(1).

        Any second interrupt signal (regardless of type) should force-exit.
        Uses event-gating to ensure agent.on_stop() blocks until the second
        signal has been sent.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            shutdown_started = asyncio.Event()
            shutdown_proceed = asyncio.Event()

            async def _blocking_on_stop():
                shutdown_started.set()
                await shutdown_proceed.wait()

            mock_agent.on_stop = AsyncMock(side_effect=_blocking_on_stop)

            with (
                patch("corvidae.main.create_plugin_manager", return_value=mock_pm),
                patch("os._exit") as mock_os_exit,
            ):
                loop = asyncio.get_running_loop()

                async def _orchestrate():
                    os.kill(os.getpid(), signal.SIGTERM)       # first signal
                    await shutdown_started.wait()               # wait for shutdown entry
                    os.kill(os.getpid(), signal.SIGINT)         # second signal while hung
                    while not mock_os_exit.called:
                        await asyncio.sleep(0)
                    shutdown_proceed.set()                      # unblock so main() can exit

                loop.call_later(0.05, lambda: asyncio.ensure_future(_orchestrate()))
                await main(config_path)

            mock_os_exit.assert_called_once_with(1)
        finally:
            os.unlink(config_path)


# ---------------------------------------------------------------------------
# TestShutdownTimeout
# ---------------------------------------------------------------------------


class TestShutdownTimeout:
    """If _run_shutdown takes too long, asyncio.wait_for triggers os._exit(1)."""

    async def test_slow_shutdown_calls_os_exit_after_timeout(self):
        """When _run_shutdown hangs, os._exit(1) is called after the timeout.

        Patches asyncio.wait_for to raise TimeoutError immediately so the
        test stays fast (no actual sleep).
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _write_config(f.name)
            config_path = f.name

        try:
            mock_pm, mock_agent = _make_mock_pm_and_agent()

            async def _patched_wait_for(coro, timeout):
                coro.close()  # prevent "coroutine never awaited" warning
                raise asyncio.TimeoutError()

            with (
                patch("corvidae.main.create_plugin_manager", return_value=mock_pm),
                patch("os._exit") as mock_os_exit,
                patch("corvidae.main.asyncio.wait_for", side_effect=_patched_wait_for),
            ):
                _schedule_sigint()
                await main(config_path)

            mock_os_exit.assert_called_once_with(1)
        finally:
            os.unlink(config_path)
