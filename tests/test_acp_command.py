"""Tests for corvidae acp CLI command packaging (WP-A0.1: A1, A2, A3).

Red before green: these tests must fail until the optional `acp` extra,
command entry point, and SDK-missing guard land.
"""

from __future__ import annotations

import logging
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from click.testing import CliRunner


class TestAcpCommandHelp:
    def test_acp_help_lists_subcommand(self):
        """The top-level corvidae group advertises the acp subcommand."""
        # Reload so entry-point discovery picks up a newly installed acp command.
        import corvidae.main as main_module

        importlib.reload(main_module)
        runner = CliRunner()
        result = runner.invoke(main_module.corvidae, ["--help"])
        assert result.exit_code == 0
        assert "acp" in result.output


class TestAcpMissingSdk:
    def test_acp_missing_sdk_exits_with_hint(self):
        """Invoking acp without agent-client-protocol exits non-zero with an install hint."""
        from corvidae.channels.acp import acp_command

        runner = CliRunner()
        # Force the SDK import inside the command to fail even if the extra is installed.
        real_import = __import__

        def _import_blocker(name, *args, **kwargs):
            if name == "acp" or name.startswith("acp."):
                raise ImportError("simulated missing agent-client-protocol")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_blocker):
            result = runner.invoke(acp_command, [])

        assert result.exit_code != 0
        combined = (result.output or "") + (result.stderr or "")
        assert "acp" in combined.lower()
        assert "uv sync" in combined or "extra" in combined.lower()


class TestServeAndCliDoNotImportAcp:
    def test_serve_and_cli_modules_do_not_import_acp(self):
        """serve and cli must load without pulling in the acp SDK package."""
        # Drop any prior import so we can detect a fresh dependency.
        for key in list(sys.modules):
            if key == "acp" or key.startswith("acp."):
                del sys.modules[key]

        importlib.import_module("corvidae.commands.serve")
        importlib.import_module("corvidae.channels.cli")

        assert "acp" not in sys.modules


class TestAcpConfigErrors:
    """Startup misconfiguration must be one clean line, not a traceback."""

    @pytest.fixture(autouse=True)
    def _restore_logging_state(self):
        """Restore global logging after acp_command reconfigures it on startup."""
        root = logging.getLogger()
        corvidae_logger = logging.getLogger("corvidae")
        before = (
            list(root.handlers),
            root.level,
            list(corvidae_logger.handlers),
            corvidae_logger.level,
            corvidae_logger.propagate,
        )
        yield
        # Close only the handlers configure_logging installed, then restore.
        installed = (set(root.handlers) | set(corvidae_logger.handlers)) - (
            set(before[0]) | set(before[2])
        )
        for handler in installed:
            handler.close()
        root.handlers[:] = before[0]
        root.setLevel(before[1])
        corvidae_logger.handlers[:] = before[2]
        corvidae_logger.setLevel(before[3])
        corvidae_logger.propagate = before[4]

    def _invoke(self, config_path):
        from corvidae.channels.acp import acp_command

        runner = CliRunner()
        result = runner.invoke(acp_command, ["--config", str(config_path)])
        combined = (result.output or "") + (result.stderr or "")
        return result, combined

    def test_missing_config_file_reports_the_path(self, tmp_path):
        missing = tmp_path / "nope.yaml"

        result, combined = self._invoke(missing)

        assert result.exit_code != 0
        assert "config file not found" in combined.lower()
        assert str(missing) in combined
        assert "Traceback" not in combined

    def test_invalid_yaml_reports_the_path(self, tmp_path):
        bad = tmp_path / "agent.yaml"
        bad.write_text("agent: [unclosed\n")

        result, combined = self._invoke(bad)

        assert result.exit_code != 0
        assert "config error" in combined.lower()
        assert str(bad) in combined
        assert "Traceback" not in combined

    def test_removed_config_key_reports_the_key(self, tmp_path):
        bad = tmp_path / "agent.yaml"
        bad.write_text(
            "agent:\n"
            "  system_prompt: hi\n"
            "  max_context_tokens: 8000\n"
            "  keep_thinking_in_history: true\n"
        )

        result, combined = self._invoke(bad)

        assert result.exit_code != 0
        # Runtime.start reports agent/channels validation itself, as the
        # same "Config error: ... — check <path>" line serve prints.
        assert "config error" in combined.lower()
        assert str(bad) in combined
        assert "reasoning_history" in combined
        assert "Traceback" not in combined

    def test_directory_as_config_reports_that(self, tmp_path):
        result, combined = self._invoke(tmp_path)

        assert result.exit_code != 0
        assert str(tmp_path) in combined
        assert "Traceback" not in combined


class TestAcpRunFailure:
    """A crashed ACP loop must exit non-zero, not leave a live-looking process."""

    def _invoke_with_fake_runtime(self, monkeypatch, tmp_path, *, failed: bool):
        cfg = tmp_path / "agent.yaml"
        cfg.write_text("agent:\n  system_prompt: hi\n  max_context_tokens: 8000\n")

        plugin = type("_FakePlugin", (), {})()
        plugin._failed = failed

        class _FakeRuntime:
            def __init__(self, *args, **kwargs):
                self.pm = MagicMock()
                self.pm.get_plugin.return_value = plugin

            async def run(self):
                return

        monkeypatch.setattr("corvidae.channels.acp.Runtime", _FakeRuntime)

        from corvidae.channels.acp import acp_command

        result = CliRunner().invoke(acp_command, ["--config", str(cfg)])
        combined = (result.output or "") + (result.stderr or "")
        return result, combined

    def test_crashed_server_exits_nonzero(self, monkeypatch, tmp_path):
        result, combined = self._invoke_with_fake_runtime(
            monkeypatch, tmp_path, failed=True
        )

        assert result.exit_code != 0
        assert "exited unexpectedly" in combined.lower()

    def test_clean_exit_is_zero(self, monkeypatch, tmp_path):
        result, _combined = self._invoke_with_fake_runtime(
            monkeypatch, tmp_path, failed=False
        )

        assert result.exit_code == 0
