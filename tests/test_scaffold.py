"""Tests for corvidae.scaffold — plugin scaffolding tool.

Unit tests for name helpers, scaffold logic, and the click CLI command.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib

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

    def test_scaffold_command_prints_next_steps(self, tmp_path):
        """#812: the operator needs to know how to install it and see it load."""
        from click.testing import CliRunner

        result = CliRunner().invoke(scaffold_command, ["Weather", "-o", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "uv pip install" in result.output            # how to install it
        assert "plugins loaded" in result.output            # whether it was picked up
        assert "uv sync" in result.output and "removes" in result.output  # the sync caveat


# ---------------------------------------------------------------------------
# What the generated project contains (#812)
# ---------------------------------------------------------------------------


def _generated(tmp_path, name: str = "Echo"):
    """Scaffold a project into tmp_path and return its path."""
    return pathlib.Path(scaffold(name, output_dir=str(tmp_path)))


def _generated_pyproject(tmp_path, name: str = "Echo") -> dict:
    """Parse the generated pyproject.toml."""
    return tomllib.loads((_generated(tmp_path, name) / "pyproject.toml").read_text())


def _checklist_python_blocks() -> list[str]:
    """The ```python blocks in docs/plugin-checklist.md §3, in order.

    §3 is the skeleton the scaffold writes, and this test is the link between
    the two: the doc and the generator cannot drift apart.
    """
    doc = pathlib.Path(__file__).resolve().parent.parent / "docs" / "plugin-checklist.md"
    section = doc.read_text().split("\n## 3.", 1)[1].split("\n## 4.", 1)[0]
    return re.findall(r"```python\n(.*?)```", section, re.S)


class TestGeneratedProjectContents:
    def test_dependency_declares_a_source_not_a_bare_pypi_name(self, tmp_path):
        """A bare `corvidae` requirement installs an unrelated PyPI package (#812)."""
        deps = _generated_pyproject(tmp_path)["project"]["dependencies"]
        assert "corvidae" not in deps
        assert deps == ["corvidae @ git+https://github.com/schuyler/corvidae"]

    def test_generated_project_runs_tests_under_a_timeout(self, tmp_path):
        """AGENTS.md: every async test runs under a timeout."""
        pyproject = _generated_pyproject(tmp_path)
        assert pyproject["tool"]["pytest"]["ini_options"]["timeout"] == 15
        dev = pyproject["project"]["optional-dependencies"]["dev"]
        assert any(dep.startswith("pytest-timeout") for dep in dev)

    def test_generated_plugin_reads_its_config_and_registers_a_tool(self, tmp_path):
        """The skeleton is the checklist's §3 skeleton, not empty hooks."""
        module = (_generated(tmp_path) / "echo" / "__init__.py").read_text()
        assert 'config_value(config, "echo.max_chars", DEFAULT_MAX_CHARS)' in module
        assert "Config error: echo.max_chars" in module
        assert "tool_registry.append(self.echo)" in module

    def test_generated_test_uses_the_real_plugin_manager(self, tmp_path):
        """A test that only instantiates the class misses hook-binding bugs."""
        test_file = (_generated(tmp_path) / "tests" / "test_echo.py").read_text()
        assert "create_plugin_manager()" in test_file
        assert "register_tools(tool_registry=registry)" in test_file

    def test_generated_skeleton_matches_the_checklist(self, tmp_path):
        """docs/plugin-checklist.md §3 is exactly what `scaffold echo` writes."""
        module_block, test_block = _checklist_python_blocks()
        project = _generated(tmp_path, "echo")
        assert (project / "echo" / "__init__.py").read_text() == module_block
        assert (project / "tests" / "test_echo.py").read_text() == test_block

    @pytest.mark.timeout(120)
    def test_generated_project_tests_pass(self, tmp_path):
        """The generated project's own tests run green against this corvidae."""
        project = _generated(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=110,
            env={**os.environ, "PYTHONPATH": str(project)},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "2 passed" in proc.stdout

    @pytest.mark.timeout(180)
    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
    def test_generated_project_installs_and_loads_through_the_real_pm(self, tmp_path):
        """#812: install it against a local corvidae, then load it for real."""
        project = _generated(tmp_path)
        target = tmp_path / "site"
        install = subprocess.run(
            [
                "uv", "pip", "install", "--no-deps", "--target", str(target), str(project),
            ],
            capture_output=True,
            text=True,
            timeout=170,
        )
        assert install.returncode == 0, install.stdout + install.stderr
        assert list(target.glob("echo-*.dist-info")), "no installed metadata"

        # A separate interpreter with the target on PYTHONPATH: the entry
        # point has to resolve from installed metadata, not from a source tree.
        script = textwrap.dedent(
            """
            import asyncio
            import importlib.metadata as md

            from corvidae.hooks import create_plugin_manager
            from corvidae.tool import Tool

            eps = [
                e for e in md.entry_points(group="corvidae")
                if e.dist is not None and e.dist.name == "echo"
            ]
            assert eps, "the installed project registers no corvidae entry point"
            pm = create_plugin_manager()
            pm.register(eps[0].load()(), name=eps[0].name)
            asyncio.run(pm.ahook.on_init(pm=pm, config={}))
            registry = []
            pm.hook.register_tools(tool_registry=registry)
            assert [Tool.from_function(f).name for f in registry] == ["echo"], registry
            """
        )
        load = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=170,
            env={**os.environ, "PYTHONPATH": str(target)},
        )
        assert load.returncode == 0, load.stdout + load.stderr
