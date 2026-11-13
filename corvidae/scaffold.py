"""Plugin scaffolding tool for corvidae.

Generates a new plugin project directory with pyproject.toml, plugin module,
and test file. Does not require the agent runtime.

The generated plugin module and its test are the skeleton in
``docs/plugin-checklist.md`` §3, kept identical by
``tests/test_scaffold.py::test_generated_skeleton_matches_the_checklist``.
"""

from __future__ import annotations

import os
import re
import string

import click

# ---------------------------------------------------------------------------
# Name conversion helpers
# ---------------------------------------------------------------------------

def _to_package_name(plugin_name: str) -> str:
    """Convert a plugin display name to a Python package name.

    Lowercases, replaces non-alphanumeric runs with underscores, strips
    leading/trailing underscores.
    """
    return re.sub(r"[^a-z0-9]+", "_", plugin_name.lower()).strip("_")


def _to_class_name(package_name: str) -> str:
    """Convert a package name to a PascalCase class name.

    Splits on underscores, capitalizes each segment. Appends 'Plugin'
    if the result does not already end with 'Plugin'.
    """
    result = "".join(seg.capitalize() for seg in package_name.split("_"))
    if not result.endswith("Plugin"):
        result += "Plugin"
    return result


def _to_entry_point_name(package_name: str) -> str:
    """Convert a package name to an entry point name.

    Strips the 'corvidae_' prefix if present.
    """
    return package_name.removeprefix("corvidae_")


# ---------------------------------------------------------------------------
# Templates
#
# `${name}` placeholders, substituted with string.Template: the templates are
# Python source, so they are full of braces that str.format would eat.
# ---------------------------------------------------------------------------

PYPROJECT_TEMPLATE = string.Template("""\
[project]
name = "${package_name}"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    # Not a bare "corvidae": PyPI has an unrelated package by that name, and
    # a bare requirement installs it. The git URL is this project's source.
    "corvidae @ git+https://github.com/schuyler/corvidae",
]

# Developing beside a corvidae checkout? Point the dependency at it instead:
# [tool.uv.sources]
# corvidae = { path = "../corvidae" }

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "pytest-timeout>=2.4",
]

[project.entry-points.corvidae]
${entry_point_name} = "${package_name}:${class_name}"

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
asyncio_mode = "auto"
# Every async test runs under a timeout (AGENTS.md).
timeout = 15
""")

PLUGIN_TEMPLATE = string.Template('''\
"""Corvidae plugin: ${plugin_name}."""

from __future__ import annotations

import logging

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tuning import config_value

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 200


class ${class_name}(CorvidaePlugin):
    """Registers the `${entry_point_name}` tool."""

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)  # stores self.pm and self.config
        # config is the WHOLE agent.yaml dict: read your own section.
        raw = config_value(config, "${entry_point_name}.max_chars", DEFAULT_MAX_CHARS)
        if type(raw) is not int or raw < 1:  # malformed: loud, names key + value
            raise ValueError(
                f"Config error: ${entry_point_name}.max_chars must be a positive "
                f"integer, got {raw!r} — check agent.yaml"
            )
        self._max_chars = raw

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        tool_registry.append(self.${entry_point_name})  # bare async callables are auto-wrapped

    async def ${entry_point_name}(self, text: str) -> str:
        """TODO: say what this tool does.

        Args:
            text: TODO: say what this argument is.
        """
        # TODO: do the work here. Until then it echoes its input, which is
        # enough to show the wiring (config read, registration) works.
        if len(text) > self._max_chars:
            # Says what failed, which limit, and the next step.
            return (
                f"Could not ${entry_point_name}: text is {len(text)} chars, the limit "
                f"is {self._max_chars}; shorten it."
            )
        return text
''')

TEST_TEMPLATE = string.Template('''\
"""Tests for ${package_name}: the skeleton loads through the real plugin manager."""

from __future__ import annotations

import pytest

from corvidae.hooks import create_plugin_manager
from corvidae.tool import Tool

from ${package_name} import ${class_name}


async def test_registers_tool_through_the_real_pm():
    pm = create_plugin_manager()
    pm.register(${class_name}(), name="${entry_point_name}")
    await pm.ahook.on_init(pm=pm, config={})
    registry: list = []
    pm.hook.register_tools(tool_registry=registry)  # register_tools is sync
    assert [Tool.from_function(f).name for f in registry] == ["${entry_point_name}"]


async def test_malformed_config_names_the_key_and_the_remedy():
    pm = create_plugin_manager()
    pm.register(${class_name}(), name="${entry_point_name}")
    with pytest.raises(ValueError, match=r"Config error: ${entry_point_name}\\.max_chars"):
        await pm.ahook.on_init(
            pm=pm, config={"${entry_point_name}": {"max_chars": 0}}
        )
''')

NEXT_STEPS_TEMPLATE = string.Template("""\

Next steps:
  1. Install it into the corvidae environment:
       uv pip install --python <corvidae-venv>/bin/python -e .
  2. Start corvidae and check that the "plugins loaded" log line names
     ${entry_point_name}. If it does not, the entry point in pyproject.toml is
     wrong and the plugin never loads.
  3. Optionally add a config section to agent.yaml (the default is 200):
       ${entry_point_name}:
         max_chars: 200

A later `uv sync` in the corvidae environment removes a plugin installed with
`uv pip install`; reinstall after one.
""")


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

def _write(path: str, content: str) -> None:
    """Write content to path, creating parent directories as needed.

    Refuses to overwrite an existing file.
    """
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Scaffold logic
# ---------------------------------------------------------------------------

def scaffold(plugin_name: str, output_dir: str = ".") -> str:
    """Generate a new corvidae plugin project.

    Creates a directory under output_dir named after the package, containing:
    - pyproject.toml
    - <package_name>/__init__.py  (plugin module)
    - tests/test_<package_name>.py

    Returns the path to the created project directory.
    """
    package_name = _to_package_name(plugin_name)
    class_name = _to_class_name(package_name)
    entry_point_name = _to_entry_point_name(package_name)

    # Project directory
    project_dir = os.path.join(output_dir, package_name)
    os.makedirs(project_dir, exist_ok=True)

    # Template variables
    vars_ = {
        "plugin_name": plugin_name,
        "package_name": package_name,
        "class_name": class_name,
        "entry_point_name": entry_point_name,
    }

    # Write pyproject.toml
    _write(
        os.path.join(project_dir, "pyproject.toml"),
        PYPROJECT_TEMPLATE.substitute(**vars_),
    )

    # Write plugin module
    module_dir = os.path.join(project_dir, package_name)
    os.makedirs(module_dir, exist_ok=True)
    _write(
        os.path.join(module_dir, "__init__.py"),
        PLUGIN_TEMPLATE.substitute(**vars_),
    )

    # Write test file
    test_dir = os.path.join(project_dir, "tests")
    os.makedirs(test_dir, exist_ok=True)
    _write(
        os.path.join(test_dir, f"test_{package_name}.py"),
        TEST_TEMPLATE.substitute(**vars_),
    )

    return project_dir


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command("scaffold")
@click.argument("plugin_name")
@click.option(
    "--output-dir", "-o",
    default=".",
    help="Parent directory for the generated project (default: current directory).",
)
def scaffold_command(plugin_name: str, output_dir: str) -> None:
    """Generate a new corvidae plugin project."""
    try:
        project_dir = scaffold(plugin_name, output_dir=output_dir)
    except FileExistsError as e:
        raise click.ClickException(str(e))
    click.echo(f"Created plugin project: {project_dir}")
    click.echo(
        NEXT_STEPS_TEMPLATE.substitute(
            entry_point_name=_to_entry_point_name(_to_package_name(plugin_name))
        )
    )
