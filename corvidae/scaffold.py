"""Scaffold a new corvidae tool plugin package.

Generates a ready-to-install Python package with:
  - pyproject.toml (corvidae dependency + entry point)
  - Plugin module implementing register_tools
  - tests/conftest.py with plugin_manager fixture
  - tests/test_<pkg>.py with basic fixture wiring
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _to_package_name(dist_name: str) -> str:
    return dist_name.replace("-", "_")


def _to_class_name(dist_name: str) -> str:
    return "".join(part.capitalize() for part in dist_name.replace("-", "_").split("_"))


def _to_entry_point_name(dist_name: str) -> str:
    parts = dist_name.replace("-", "_").split("_")
    if parts[-1] in ("plugin", "tools"):
        parts = parts[:-1]
    return "_".join(parts)


_PYPROJECT_TEMPLATE = """\
[project]
name = "{dist_name}"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "corvidae",
]

[project.entry-points.corvidae]
{ep_name} = "{pkg}:{cls}"

[tool.setuptools.packages.find]
include = ["{pkg}*"]

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
"""

_PLUGIN_TEMPLATE = """\
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.tool import Tool


class {cls}(CorvidaePlugin):
    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def my_tool(input: str) -> str:
            \"\"\"TODO: describe what this tool does.\"\"\"
            raise NotImplementedError("Implement me")

        tool_registry.append(Tool.from_function(my_tool))
"""

_CONFTEST_TEMPLATE = """\
import pytest
from corvidae.hooks import create_plugin_manager


@pytest.fixture
def plugin_manager():
    return create_plugin_manager()
"""

_TEST_TEMPLATE = """\
import pytest
from unittest.mock import MagicMock

from {pkg} import {cls}


def test_plugin_importable():
    from {pkg} import {cls}  # noqa: F401


def test_plugin_instantiable():
    plugin = {cls}()
    assert plugin is not None


def test_register_tools():
    plugin = {cls}()
    registry = []
    plugin.register_tools(registry)
    assert len(registry) > 0


async def test_register_tools_via_pluginmanager(plugin_manager):
    plugin = {cls}()
    plugin_manager.register(plugin, name="{ep_name}")
    registry = []
    plugin_manager.hook.register_tools(tool_registry=registry)
    assert len(registry) > 0
"""


def scaffold(dist_name: str, output_dir: Path) -> None:
    """Generate a plugin package directory tree under output_dir."""
    pkg = _to_package_name(dist_name)
    cls = _to_class_name(dist_name)
    ep_name = _to_entry_point_name(dist_name)

    root = output_dir / dist_name
    root.mkdir(parents=True, exist_ok=True)
    (root / pkg).mkdir(exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)

    _write(root / "pyproject.toml", _PYPROJECT_TEMPLATE.format(dist_name=dist_name, pkg=pkg, cls=cls, ep_name=ep_name))
    _write(root / pkg / "__init__.py", _PLUGIN_TEMPLATE.format(cls=cls))
    _write(root / "tests" / "conftest.py", _CONFTEST_TEMPLATE)
    _write(root / "tests" / f"test_{pkg}.py", _TEST_TEMPLATE.format(pkg=pkg, cls=cls, ep_name=ep_name))

    created = [
        f"  {dist_name}/pyproject.toml",
        f"  {dist_name}/{pkg}/__init__.py",
        f"  {dist_name}/tests/conftest.py",
        f"  {dist_name}/tests/test_{pkg}.py",
    ]
    print(f"Scaffolded plugin package: {root}/")
    print("\n".join(created))
    print()
    print("Next steps:")
    print(f"  cd {root}")
    print( "  pip install -e .")
    print(f"  # Edit {pkg}/__init__.py to implement your tools")
    print( "  pytest")


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def scaffold_cli(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="corvidae scaffold",
        description="Generate a new corvidae tool plugin package.",
    )
    parser.add_argument(
        "name",
        help="Distribution name for the new plugin (e.g. my-tool-plugin)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        metavar="DIR",
        help="Parent directory for the generated package tree (default: .)",
    )
    ns = parser.parse_args(args)

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", ns.name):
        print(
            f"Error: invalid name {ns.name!r} — must start with a letter and"
            " contain only letters, digits, hyphens, or underscores.",
            file=sys.stderr,
        )
        sys.exit(1)

    scaffold(ns.name, Path(ns.output_dir))
