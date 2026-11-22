"""Plugin scaffolding tool for corvidae.

Generates a new plugin project directory with pyproject.toml, plugin module,
and test file. Does not require the agent runtime.

The generated plugin module and its test are the skeleton in
``docs/plugin-checklist.md`` §3, kept identical by
``tests/test_scaffold.py::test_generated_skeleton_matches_the_checklist``.
"""

from __future__ import annotations

import keyword
import os
import re
import string

import click

from corvidae.hooks import CorvidaePlugin

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


# Names the kernel registers itself: a plugin entry point with one of these is
# shadowed or refused at load (#849). Keep in step with Runtime.start().
_KERNEL_ENTRY_POINT_NAMES = frozenset({"registry", "clock", "idle", "spill"})

# CorvidaePlugin sets these on every plugin in on_init; a plugin method or
# attribute with one of them shadows the kernel's (#849).
_PLUGIN_INSTANCE_ATTRIBUTES = frozenset({"pm", "config"})


def validate_plugin_name(plugin_name: str) -> None:
    """Refuse a name that cannot produce a loadable plugin, naming it (#849).

    A scaffold is the first thing a plugin author sees, so a name that generates
    a project which cannot import or load must fail here, with the reason and the
    accepted form, rather than at their first run.
    """
    package_name = _to_package_name(plugin_name)
    if not package_name:
        raise ValueError(
            f"plugin name {plugin_name!r} produces an empty package name; use "
            f"letters or digits, e.g. 'weather-report'"
        )
    if package_name[0].isdigit():
        raise ValueError(
            f"plugin name {plugin_name!r} produces the package name "
            f"{package_name!r}, which starts with a digit and cannot be "
            f"imported; start it with a letter, e.g. 'weather-{plugin_name}'"
        )
    if keyword.iskeyword(package_name):
        raise ValueError(
            f"plugin name {plugin_name!r} produces the package name "
            f"{package_name!r}, which is a Python keyword; choose another name"
        )
    # Instance attributes CorvidaePlugin sets on every plugin; a method or
    # attribute with one of these names shadows it (#849).
    if package_name in _PLUGIN_INSTANCE_ATTRIBUTES or hasattr(
        CorvidaePlugin, package_name
    ):
        raise ValueError(
            f"plugin name {plugin_name!r} produces the package name "
            f"{package_name!r}, which is an attribute of CorvidaePlugin and "
            f"would shadow it; choose another name"
        )
    if _to_entry_point_name(package_name) in _KERNEL_ENTRY_POINT_NAMES:
        raise ValueError(
            f"plugin name {plugin_name!r} produces the entry point name "
            f"{_to_entry_point_name(package_name)!r}, which the kernel already "
            f"registers; choose another name"
        )


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
    # The message must name the key, the rejected value and the remedy.
    with pytest.raises(
        ValueError,
        match=r"Config error: ${entry_point_name}\\.max_chars must be a positive "
        r"integer, got 0 — check agent\\.yaml",
    ):
        await pm.ahook.on_init(
            pm=pm, config={"${entry_point_name}": {"max_chars": 0}}
        )
''')

NEXT_STEPS_TEMPLATE = string.Template("""\

Next steps:
  1. Install it into the corvidae environment:
       uv pip install --no-deps --python <corvidae-venv>/bin/python -e .
     (--no-deps keeps the corvidae already installed there instead of
     replacing it with the git source the generated pyproject names.)
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
    validate_plugin_name(plugin_name)
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
    except (FileExistsError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"Created plugin project: {project_dir}")
    click.echo(
        NEXT_STEPS_TEMPLATE.substitute(
            entry_point_name=_to_entry_point_name(_to_package_name(plugin_name))
        )
    )
