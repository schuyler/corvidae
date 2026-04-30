"""Tests for corvidae.scaffold — plugin scaffolding tool.

Unit tests for name helpers, scaffold logic, and the click CLI command.
"""

from __future__ import annotations

import os

import pytest

from corvidae.scaffold import (
    _to_class_name,
    _to_entry_point_name,
    _to_package_name,
    _write,
    scaffold,
    scaffold_command,
)


# ---------------------------------------------------------------------------
# Name conversion helpers
# ---------------------------------------------------------------------------


class TestToPackageName:
    def test_simple_name(self):
        assert _to_package_name("MyPlugin") == "myplugin"

    def test_spaces_become_underscores(self):
        assert _to_package_name("My Cool Plugin") == "my_cool_plugin"

    def test_special_chars_become_underscores(self):
        assert _to_package_name("my-cool-plugin!") == "my_cool_plugin"

    def test_leading_trailing_underscores_stripped(self):
        assert _to_package_name("__foo__") == "foo"

    def test_multiple_nonalnum_collapsed(self):
        assert _to_package_name("a---b___c") == "a_b_c"


class TestToClassName:
    def test_simple(self):
        assert _to_class_name("my_plugin") == "MyPlugin"

    def test_single_segment(self):
        assert _to_class_name("weather") == "WeatherPlugin"

    def test_multiple_segments(self):
        assert _to_class_name("cool_weather_thing") == "CoolWeatherThingPlugin"


class TestToEntryPointName:
    def test_strips_corvidae_prefix(self):
        assert _to_entry_point_name("corvidae_weather") == "weather"

    def test_no_prefix(self):
        assert _to_entry_point_name("weather") == "weather"

    def test_corvidae_only(self):
        assert _to_entry_point_name("corvidae_") == ""


# ---------------------------------------------------------------------------
# _write helper
# ---------------------------------------------------------------------------


class TestWrite:
    def test_writes_file(self, tmp_path):
        path = str(tmp_path / "out.txt")
        _write(path, "hello")
        assert open(path).read() == "hello"

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "out.txt")
        _write(path, "nested")
        assert open(path).read() == "nested"

    def test_refuses_to_overwrite(self, tmp_path):
        path = str(tmp_path / "exists.txt")
        _write(path, "first")
        with pytest.raises(FileExistsError):
            _write(path, "second")


# ---------------------------------------------------------------------------
# scaffold() function
# ---------------------------------------------------------------------------


class TestScaffold:
    def test_creates_project_directory(self, tmp_path):
        project_dir = scaffold("My Weather Plugin", output_dir=str(tmp_path))
        assert os.path.isdir(project_dir)
        assert project_dir == str(tmp_path / "my_weather_plugin")

    def test_creates_pyproject_toml(self, tmp_path):
        project_dir = scaffold("Weather", output_dir=str(tmp_path))
        pyproject = os.path.join(project_dir, "pyproject.toml")
        assert os.path.isfile(pyproject)
        content = open(pyproject).read()
        assert 'name = "weather"' in content
        assert "weather = " in content  # entry point line

    def test_creates_plugin_module(self, tmp_path):
        project_dir = scaffold("Weather", output_dir=str(tmp_path))
        init = os.path.join(project_dir, "weather", "__init__.py")
        assert os.path.isfile(init)
        content = open(init).read()
        assert "class WeatherPlugin" in content
        assert "CorvidaePlugin" in content

    def test_creates_test_file(self, tmp_path):
        project_dir = scaffold("Weather", output_dir=str(tmp_path))
        test_file = os.path.join(project_dir, "tests", "test_weather.py")
        assert os.path.isfile(test_file)
        content = open(test_file).read()
        assert "from weather import WeatherPlugin" in content

    def test_refuses_to_overwrite_existing_files(self, tmp_path):
        scaffold("Weather", output_dir=str(tmp_path))
        with pytest.raises(FileExistsError):
            scaffold("Weather", output_dir=str(tmp_path))

    def test_corvidae_prefix_handling(self, tmp_path):
        """A plugin named 'corvidae_foo' uses 'foo' as the entry point name."""
        project_dir = scaffold("corvidae_foo", output_dir=str(tmp_path))
        pyproject = os.path.join(project_dir, "pyproject.toml")
        content = open(pyproject).read()
        # Entry point should be 'foo', not 'corvidae_foo'
        assert 'foo = "corvidae_foo:CorvidaeFooPlugin"' in content


# ---------------------------------------------------------------------------
# Click CLI command
# ---------------------------------------------------------------------------


class TestScaffoldCommand:
    """Tests for the click command entry point."""

    def test_scaffold_command_exists_and_is_click_command(self):
        import click

        assert isinstance(scaffold_command, click.BaseCommand)

    def test_scaffold_command_name(self):
        assert scaffold_command.name == "scaffold"

    def test_scaffold_command_creates_project(self, tmp_path):
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            scaffold_command, ["MyTestPlugin", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        assert os.path.isdir(tmp_path / "mytestplugin")

    def test_scaffold_command_prints_path(self, tmp_path):
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            scaffold_command, ["MyTestPlugin", "-o", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "mytestplugin" in result.output

    def test_scaffold_command_default_output_dir(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(scaffold_command, ["SomePlugin"])
        assert result.exit_code == 0
        assert os.path.isdir(tmp_path / "someplugin")

    def test_scaffold_command_error_on_existing(self, tmp_path):
        from click.testing import CliRunner

        runner = CliRunner()
        # First invocation succeeds
        runner.invoke(scaffold_command, ["Dupe", "-o", str(tmp_path)])
        # Second invocation fails with a human-readable error message
        result = runner.invoke(scaffold_command, ["Dupe", "-o", str(tmp_path)])
        assert result.exit_code == 1
        assert "Refusing to overwrite" in result.output
        # ClickException → sys.exit(1) → CliRunner stores SystemExit, not a raw exception
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_scaffold_command_missing_name_shows_usage(self):
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(scaffold_command, [])
        assert result.exit_code != 0
        assert "Usage" in result.output or "Missing" in result.output
