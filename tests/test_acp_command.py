"""Tests for corvidae acp CLI command packaging (WP-A0.1: A1, A2, A3).

Red before green: these tests must fail until the optional `acp` extra,
command entry point, and SDK-missing guard land.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

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
