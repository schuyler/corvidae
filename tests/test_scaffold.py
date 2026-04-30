"""Tests for corvidae.scaffold — plugin package generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from corvidae.scaffold import (
    _to_class_name,
    _to_entry_point_name,
    _to_package_name,
    scaffold,
)


# ---------------------------------------------------------------------------
# Name-munging helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name,expected", [
    ("my-tool-plugin", "my_tool_plugin"),
    ("corvidae-weather", "corvidae_weather"),
    ("simple", "simple"),
])
def test_to_package_name(dist_name, expected):
    assert _to_package_name(dist_name) == expected


@pytest.mark.parametrize("dist_name,expected", [
    ("my-tool-plugin", "MyToolPlugin"),
    ("corvidae-weather", "CorvidaeWeather"),
    ("simple", "Simple"),
])
def test_to_class_name(dist_name, expected):
    assert _to_class_name(dist_name) == expected


@pytest.mark.parametrize("dist_name,expected", [
    ("my-tool-plugin", "my_tool"),    # strips trailing "plugin"
    ("corvidae-weather-tools", "corvidae_weather"),  # strips trailing "tools"
    ("weather", "weather"),            # no suffix to strip
])
def test_to_entry_point_name(dist_name, expected):
    assert _to_entry_point_name(dist_name) == expected


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

def test_scaffold_creates_expected_files(tmp_path):
    scaffold("my-tool-plugin", tmp_path)

    root = tmp_path / "my-tool-plugin"
    assert (root / "pyproject.toml").exists()
    assert (root / "my_tool_plugin" / "__init__.py").exists()
    assert (root / "tests" / "conftest.py").exists()
    assert (root / "tests" / "test_my_tool_plugin.py").exists()


def test_scaffold_pyproject_contains_entry_point(tmp_path):
    scaffold("my-tool-plugin", tmp_path)
    content = (tmp_path / "my-tool-plugin" / "pyproject.toml").read_text()
    assert 'my_tool = "my_tool_plugin:MyToolPlugin"' in content
    assert 'corvidae' in content


def test_scaffold_pyproject_contains_corvidae_dependency(tmp_path):
    scaffold("my-tool-plugin", tmp_path)
    content = (tmp_path / "my-tool-plugin" / "pyproject.toml").read_text()
    assert '"corvidae"' in content


def test_scaffold_plugin_module_contains_class(tmp_path):
    scaffold("my-tool-plugin", tmp_path)
    content = (tmp_path / "my-tool-plugin" / "my_tool_plugin" / "__init__.py").read_text()
    assert "class MyToolPlugin(CorvidaePlugin):" in content
    assert "register_tools" in content
    assert "Tool.from_function" in content


def test_scaffold_test_file_imports_plugin(tmp_path):
    scaffold("my-tool-plugin", tmp_path)
    content = (tmp_path / "my-tool-plugin" / "tests" / "test_my_tool_plugin.py").read_text()
    assert "from my_tool_plugin import MyToolPlugin" in content
    assert "def test_register_tools" in content


def test_scaffold_conftest_has_plugin_manager_fixture(tmp_path):
    scaffold("my-tool-plugin", tmp_path)
    content = (tmp_path / "my-tool-plugin" / "tests" / "conftest.py").read_text()
    assert "create_plugin_manager" in content
    assert "def plugin_manager" in content


def test_scaffold_idempotent(tmp_path):
    scaffold("my-tool-plugin", tmp_path)
    scaffold("my-tool-plugin", tmp_path)  # should not raise


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

def test_scaffold_cli_rejects_invalid_name():
    result = subprocess.run(
        [sys.executable, "-m", "corvidae.scaffold_runner", "123-bad"],
        capture_output=True, text=True,
    )
    # The scaffold_cli function raises SystemExit on invalid name; we test it
    # via the public API instead.
    from corvidae.scaffold import scaffold_cli
    with pytest.raises(SystemExit) as exc_info:
        scaffold_cli(["123-bad"])
    assert exc_info.value.code != 0


def test_scaffold_cli_generates_package(tmp_path):
    from corvidae.scaffold import scaffold_cli
    scaffold_cli(["my-weather-plugin", "--output-dir", str(tmp_path)])
    assert (tmp_path / "my-weather-plugin" / "pyproject.toml").exists()


def test_corvidae_scaffold_subcommand(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.argv = ['corvidae', 'scaffold', 'demo-plugin', '--output-dir', {str(tmp_path)!r}];"
         "from corvidae.main import cli; cli()"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "demo-plugin" / "pyproject.toml").exists()
