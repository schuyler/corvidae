"""Plugin scaffolding tool for corvidae.

Generates a new plugin project directory with pyproject.toml, plugin module,
and test file. Does not require the agent runtime.
"""

from __future__ import annotations

import os
import re
import textwrap

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
# ---------------------------------------------------------------------------

PYPROJECT_TEMPLATE = textwrap.dedent("""\
    [project]
    name = "{package_name}"
    version = "0.1.0"
    requires-python = ">=3.13"
    dependencies = [
        "corvidae",
    ]

    [project.optional-dependencies]
    dev = [
        "pytest>=8",
        "pytest-asyncio>=0.23",
    ]

    [project.entry-points.corvidae]
    {entry_point_name} = "{package_name}:{class_name}"

    [build-system]
    requires = ["setuptools>=75"]
    build-backend = "setuptools.build_meta"

    [tool.pytest.ini_options]
    asyncio_mode = "auto"
""")

PLUGIN_TEMPLATE = textwrap.dedent("""\
    \"\"\"Corvidae plugin: {plugin_name}.\"\"\"

    from __future__ import annotations

    from corvidae.hooks import CorvidaePlugin, hookimpl


    class {class_name}(CorvidaePlugin):
        \"\"\"TODO: describe what this plugin does.\"\"\"

        @hookimpl
        async def on_init(self, pm, config: dict) -> None:
            await super().on_init(pm, config)

        @hookimpl
        async def on_start(self, config: dict) -> None:
            pass

        @hookimpl
        async def on_stop(self) -> None:
            pass
""")

TEST_TEMPLATE = textwrap.dedent("""\
    \"\"\"Tests for {package_name}.\"\"\"

    import pytest

    from {package_name} import {class_name}


    @pytest.fixture
    def plugin():
        return {class_name}()


    class TestInit:
        async def test_plugin_instantiates(self, plugin):
            assert isinstance(plugin, {class_name})
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
        PYPROJECT_TEMPLATE.format(**vars_),
    )

    # Write plugin module
    module_dir = os.path.join(project_dir, package_name)
    os.makedirs(module_dir, exist_ok=True)
    _write(
        os.path.join(module_dir, "__init__.py"),
        PLUGIN_TEMPLATE.format(**vars_),
    )

    # Write test file
    test_dir = os.path.join(project_dir, "tests")
    os.makedirs(test_dir, exist_ok=True)
    _write(
        os.path.join(test_dir, f"test_{package_name}.py"),
        TEST_TEMPLATE.format(**vars_),
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
