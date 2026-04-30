"""Tests for corvidae.main — click dispatcher and subcommand discovery.

All tests in this file are RED (failing) until the implementation lands.

Test isolation requirement: patch importlib.metadata.entry_points to return
an empty list for "corvidae.commands" so real command entry points are not
loaded during test setup.

Patch targets:
    importlib.metadata.entry_points  — prevent real command loading at import time
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TestDiscoverCommands
# ---------------------------------------------------------------------------


class TestDiscoverCommands:
    """discover_commands() loads entry points from the 'corvidae.commands' group."""

    @pytest.fixture(autouse=True)
    def reload_main_clean(self):
        """Reload corvidae.main with empty entry points before each test,
        then restore the module to a clean state after.

        This prevents real entry-point loading from interfering with tests
        that call discover_commands() with controlled fake entry points.
        """
        with patch("importlib.metadata.entry_points", return_value=[]):
            import corvidae.main as main_module

            importlib.reload(main_module)
        yield main_module

    def test_discover_commands_registers_command_from_entry_point(
        self, reload_main_clean
    ):
        """discover_commands() loads an entry point and adds it as a subcommand."""
        import click

        main_module = reload_main_clean

        @click.command("fake")
        def _fake_cmd():
            """Fake command for testing."""

        fake_ep = MagicMock()
        fake_ep.name = "fake"
        fake_ep.load.return_value = _fake_cmd

        with patch("importlib.metadata.entry_points", return_value=[fake_ep]):
            main_module.discover_commands()

        assert "fake" in main_module.corvidae.commands

    def test_discover_commands_logs_warning_on_load_failure(
        self, reload_main_clean, caplog
    ):
        """discover_commands() logs a warning and continues when an entry point fails to load."""
        import logging

        main_module = reload_main_clean

        bad_ep = MagicMock()
        bad_ep.name = "broken"
        bad_ep.load.side_effect = Exception("intentional load failure")

        with caplog.at_level(logging.WARNING, logger="corvidae.main"), patch(
            "importlib.metadata.entry_points", return_value=[bad_ep]
        ):
            main_module.discover_commands()

        # Use caplog.text to avoid relying on record.message which may not be
        # populated before formatting.
        assert "broken" in caplog.text

    def test_bare_corvidae_shows_help(self, reload_main_clean):
        """Invoking the corvidae group with no subcommand outputs help text."""
        from click.testing import CliRunner

        main_module = reload_main_clean

        runner = CliRunner()
        result = runner.invoke(main_module.corvidae, [])

        # Exit code 0; help text must appear.
        assert result.exit_code == 0
        assert "Usage" in result.output or "help" in result.output.lower()

    def test_corvidae_group_is_click_group(self, reload_main_clean):
        """corvidae in corvidae.main is a click.Group after the refactor."""
        import click

        main_module = reload_main_clean

        assert isinstance(main_module.corvidae, click.Group)


# ---------------------------------------------------------------------------
# Tests for CLI subcommand: corvidae.channels.cli.cli_command
#
# Patch targets: corvidae.channels.cli.asyncio.run
# Assumption: corvidae/channels/cli.py uses module-level `import asyncio`
# (not a local import inside the function body). Module-level imports create
# a module attribute that unittest.mock.patch can target. If the implementation
# uses a local import instead, these patch targets will raise AttributeError
# and the tests must be updated to patch `asyncio.run` directly.
# ---------------------------------------------------------------------------


class TestCliCommand:
    """cli_command is a click.Command that passes the correct overrides to Runtime."""

    def test_cli_command_exists_and_is_click_command(self):
        """corvidae.channels.cli exposes cli_command as a click.BaseCommand."""
        import click
        from corvidae.channels.cli import cli_command

        assert isinstance(cli_command, click.BaseCommand)

    def test_cli_command_name_is_cli(self):
        """cli_command.name is 'cli'."""
        from corvidae.channels.cli import cli_command

        assert cli_command.name == "cli"

    def test_cli_command_passes_channel_override_to_runtime(self):
        """cli_command instantiates Runtime with channels.cli:local in overrides."""
        from click.testing import CliRunner
        from corvidae.channels.cli import cli_command

        captured_overrides: list[dict] = []

        class _CapturingRuntime:
            def __init__(self, config_path="agent.yaml", overrides=None):
                captured_overrides.append(overrides or {})

            def run(self):
                pass  # stub — asyncio.run is patched below

        # Assumption: corvidae/channels/cli.py imports asyncio at module level.
        # The patch target corvidae.channels.cli.asyncio.run is valid only if
        # `import asyncio` appears at module scope in cli.py.
        with patch("corvidae.channels.cli.Runtime", _CapturingRuntime), patch(
            "corvidae.channels.cli.asyncio.run", side_effect=lambda coro: None
        ):
            runner = CliRunner()
            runner.invoke(cli_command, ["--config", "test.yaml"])

        assert len(captured_overrides) == 1, "Runtime was not instantiated"
        overrides = captured_overrides[0]
        assert "channels" in overrides, "overrides must include 'channels'"
        assert "cli:local" in overrides["channels"], (
            "overrides must include cli:local channel"
        )

    def test_cli_command_passes_logging_file_override(self):
        """cli_command passes logging.file override to Runtime."""
        from click.testing import CliRunner
        from corvidae.channels.cli import cli_command

        captured_overrides: list[dict] = []

        class _CapturingRuntime:
            def __init__(self, config_path="agent.yaml", overrides=None):
                captured_overrides.append(overrides or {})

            def run(self):
                pass

        # Assumption: corvidae/channels/cli.py imports asyncio at module level.
        with patch("corvidae.channels.cli.Runtime", _CapturingRuntime), patch(
            "corvidae.channels.cli.asyncio.run", side_effect=lambda coro: None
        ):
            runner = CliRunner()
            runner.invoke(cli_command, [])

        overrides = captured_overrides[0]
        assert "logging" in overrides, "overrides must include 'logging'"
        assert "file" in overrides["logging"], (
            "overrides['logging'] must include 'file'"
        )


# ---------------------------------------------------------------------------
# Tests for serve subcommand: corvidae.commands.serve.serve_command
#
# Patch targets: corvidae.commands.serve.asyncio.run
# Assumption: corvidae/commands/serve.py uses module-level `import asyncio`
# (not a local import inside the function body). Module-level imports create
# a module attribute that unittest.mock.patch can target. If the implementation
# uses a local import instead, these patch targets will raise AttributeError
# and the tests must be updated to patch `asyncio.run` directly.
# ---------------------------------------------------------------------------


class TestServeCommand:
    """serve_command is a click.Command that boots Runtime with no channel overrides."""

    def test_serve_command_exists_and_is_click_command(self):
        """corvidae.commands.serve exposes serve_command as a click.BaseCommand."""
        import click
        from corvidae.commands.serve import serve_command

        assert isinstance(serve_command, click.BaseCommand)

    def test_serve_command_name_is_serve(self):
        """serve_command.name is 'serve'."""
        from corvidae.commands.serve import serve_command

        assert serve_command.name == "serve"

    def test_serve_command_passes_no_channel_overrides(self):
        """serve_command does not inject any channel overrides into Runtime."""
        from click.testing import CliRunner
        from corvidae.commands.serve import serve_command

        captured_kwargs: list[dict] = []

        class _CapturingRuntime:
            def __init__(self, config_path="agent.yaml", overrides=None):
                captured_kwargs.append({"config_path": config_path, "overrides": overrides})

            def run(self):
                pass

        # Assumption: corvidae/commands/serve.py imports asyncio at module level.
        with patch("corvidae.commands.serve.Runtime", _CapturingRuntime), patch(
            "corvidae.commands.serve.asyncio.run", side_effect=lambda coro: None
        ):
            runner = CliRunner()
            runner.invoke(serve_command, [])

        assert len(captured_kwargs) == 1, "Runtime was not instantiated"
        overrides = captured_kwargs[0].get("overrides") or {}
        assert "channels" not in overrides, (
            "serve_command must not inject channel overrides"
        )

    def test_serve_command_accepts_config_option(self):
        """serve_command accepts --config option and passes it to Runtime."""
        from click.testing import CliRunner
        from corvidae.commands.serve import serve_command

        captured_kwargs: list[dict] = []

        class _CapturingRuntime:
            def __init__(self, config_path="agent.yaml", overrides=None):
                captured_kwargs.append({"config_path": config_path})

            def run(self):
                pass

        # Assumption: corvidae/commands/serve.py imports asyncio at module level.
        with patch("corvidae.commands.serve.Runtime", _CapturingRuntime), patch(
            "corvidae.commands.serve.asyncio.run", side_effect=lambda coro: None
        ):
            runner = CliRunner()
            runner.invoke(serve_command, ["--config", "custom.yaml"])

        assert captured_kwargs[0]["config_path"] == "custom.yaml"
