"""The built-in tools, split across three plugins (#16).

ShellToolPlugin (shell, task_pipeline), FileToolsPlugin (read_file,
write_file) and WebToolsPlugin (web_fetch, web_search) each own their config
and lifecycle, load from their own entry point, and can be disabled by name
without taking the others with them.
"""

import logging
from importlib.metadata import entry_points
from unittest.mock import MagicMock

import pytest
import yaml

from corvidae.tool import Tool

pytestmark = pytest.mark.timeout(15)

# Cognition plugins left out of the real-Runtime boots below, as in
# tests/test_minimal_profile.py, so the boot needs only a stub LLM config.
_COGNITION_DISABLED = [
    "memory", "memory_tools", "funnel", "appraisal", "critique", "outcome_log",
    "gates", "silence", "skills", "heartbeat", "goals",
]


def _tool_names(plugin) -> set[str]:
    """Names of the tools a plugin's register_tools adds."""
    registry: list = []
    plugin.register_tools(tool_registry=registry)
    return {item.name if isinstance(item, Tool) else item.__name__ for item in registry}


def _plugins():
    """The three plugin classes, imported lazily so each test fails on its own."""
    from corvidae.tools import FileToolsPlugin, ShellToolPlugin, WebToolsPlugin

    return ShellToolPlugin, FileToolsPlugin, WebToolsPlugin


# ---------------------------------------------------------------------------
# R1: which plugin registers which tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, {"shell", "task_pipeline"}),
        (1, {"read_file", "write_file"}),
        (2, {"web_fetch", "web_search"}),
    ],
    ids=["shell", "file", "web"],
)
def test_each_plugin_registers_exactly_its_tools(index: int, expected: set[str]) -> None:
    """Each plugin registers its own tools and no others (R1).

    task_pipeline rides with shell: PRD §3.5.3 makes them one capability.
    """
    plugin_cls = _plugins()[index]
    assert _tool_names(plugin_cls()) == expected


# ---------------------------------------------------------------------------
# R2: each plugin reads its own keys from the unchanged tools.* block
# ---------------------------------------------------------------------------

_TOOLS_CONFIG = {
    "tools": {
        "shell_timeout": 41,
        "max_file_read_bytes": 4242,
        "web_fetch_timeout": 43,
        "web_max_response_bytes": 4444,
        "web_search_max_results": 45,
    }
}


async def test_shell_plugin_reads_its_config() -> None:
    """ShellToolPlugin reads tools.shell_timeout (R2)."""
    shell_cls, _, _ = _plugins()
    plugin = shell_cls()
    await plugin.on_init(pm=MagicMock(), config=_TOOLS_CONFIG)
    assert plugin._shell_timeout == 41


async def test_file_plugin_reads_its_config() -> None:
    """FileToolsPlugin reads tools.max_file_read_bytes (R2)."""
    _, file_cls, _ = _plugins()
    plugin = file_cls()
    await plugin.on_init(pm=MagicMock(), config=_TOOLS_CONFIG)
    assert plugin._max_file_read_bytes == 4242


async def test_web_plugin_reads_its_config() -> None:
    """WebToolsPlugin reads the three web_* keys (R2)."""
    _, _, web_cls = _plugins()
    plugin = web_cls()
    await plugin.on_init(pm=MagicMock(), config=_TOOLS_CONFIG)
    assert (
        plugin._web_fetch_timeout,
        plugin._web_max_response_bytes,
        plugin._web_search_max_results,
    ) == (43, 4444, 45)


@pytest.mark.parametrize(
    "index, foreign",
    [
        (0, ["_max_file_read_bytes", "_web_fetch_timeout", "_web_max_response_bytes",
             "_web_search_max_results", "_session"]),
        (1, ["_shell_timeout", "_web_fetch_timeout", "_web_max_response_bytes",
             "_web_search_max_results", "_session"]),
        (2, ["_shell_timeout", "_max_file_read_bytes"]),
    ],
    ids=["shell", "file", "web"],
)
async def test_plugin_carries_no_other_plugins_config(index: int, foreign: list[str]) -> None:
    """A plugin holds none of the other plugins' settings (R2, R3)."""
    plugin = _plugins()[index]()
    await plugin.on_init(pm=MagicMock(), config=_TOOLS_CONFIG)
    present = [name for name in foreign if hasattr(plugin, name)]
    assert not present, f"{type(plugin).__name__} carries {present}"


# ---------------------------------------------------------------------------
# R3: WebToolsPlugin owns the aiohttp session
# ---------------------------------------------------------------------------


async def test_web_plugin_opens_and_closes_its_session() -> None:
    """The session opens in on_start and closes in on_stop (R3)."""
    _, _, web_cls = _plugins()
    plugin = web_cls()
    await plugin.on_init(pm=MagicMock(), config={})
    await plugin.on_start(config={})
    session = plugin._session
    try:
        assert session is not None and not session.closed
    finally:
        await plugin.on_stop()
    assert session.closed


@pytest.mark.parametrize("index", [0, 1], ids=["shell", "file"])
def test_shell_and_file_plugins_have_no_lifecycle_hooks(index: int) -> None:
    """Only the web plugin has a session to start and stop (R3)."""
    plugin_cls = _plugins()[index]
    assert "on_start" not in plugin_cls.__dict__
    assert "on_stop" not in plugin_cls.__dict__


# ---------------------------------------------------------------------------
# R4: entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, value",
    [
        ("shell_tools", "corvidae.tools:ShellToolPlugin"),
        ("file_tools", "corvidae.tools:FileToolsPlugin"),
        ("web_tools", "corvidae.tools:WebToolsPlugin"),
    ],
)
def test_entry_point_names_its_plugin(name: str, value: str) -> None:
    """Each plugin loads from its own entry point (R4)."""
    eps = {ep.name: ep.value for ep in entry_points(group="corvidae")}
    assert eps.get(name) == value, (
        f"entry point {name!r} should be {value!r}, got {eps.get(name)!r}; "
        "if pyproject.toml lists it, re-sync with `uv sync`"
    )


# ---------------------------------------------------------------------------
# R5 / R6: a real Runtime boot
# ---------------------------------------------------------------------------


def _write_config(tmp_path, disabled: list[str]) -> str:
    """A minimal agent.yaml for a real Runtime boot, with plugins.disabled."""
    config = {
        "agent": {"system_prompt": "test", "max_context_tokens": 8000},
        "llm": {"main": {"base_url": "http://127.0.0.1:1/v1", "model": "test-model"}},
        "daemon": {"session_db": str(tmp_path / "sessions.db")},
        "plugins": {"disabled": _COGNITION_DISABLED + disabled},
    }
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


@pytest.fixture
def keep_test_logging(monkeypatch):
    """Stop Runtime.start() from reconfiguring logging.

    configure_logging replaces the root handlers and stops the corvidae
    logger propagating, which hides every later record from caplog and leaks
    a handler into later tests (same patch as tests/test_runtime.py).
    """
    monkeypatch.setattr("corvidae.runtime.configure_logging", lambda **kwargs: None)


async def test_disabling_web_tools_keeps_shell_and_file_tools(
    tmp_path, keep_test_logging
) -> None:
    """plugins.disabled: [web_tools] drops only the web tools (R5)."""
    from corvidae.runtime import Runtime

    rt = Runtime(config_path=_write_config(tmp_path, ["web_tools"]))
    await rt.start()
    try:
        names = set(rt.pm.get_plugin("tools").registry.as_dict())
    finally:
        await rt.stop()
    assert {"shell", "task_pipeline", "read_file", "write_file"} <= names
    assert not names & {"web_fetch", "web_search"}, names


async def test_unknown_disabled_name_warns_with_key_value_and_choices(
    tmp_path, caplog, keep_test_logging
) -> None:
    """A plugins.disabled entry naming no plugin logs one WARNING naming the
    key, the value and the installed plugins (R6)."""
    from corvidae.runtime import Runtime

    rt = Runtime(config_path=_write_config(tmp_path, ["no_such_plugin"]))
    try:
        with caplog.at_level(logging.WARNING, logger="corvidae.runtime"):
            await rt.start()
    finally:
        await rt.stop()

    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and "no_such_plugin" in r.getMessage()
    ]
    assert len(warnings) == 1, warnings
    assert "plugins.disabled" in warnings[0]
    assert "shell_tools" in warnings[0], "the warning must list the installed plugins"


async def test_known_disabled_name_does_not_warn(
    tmp_path, caplog, keep_test_logging
) -> None:
    """A real plugin name in plugins.disabled raises no unknown-name warning (R6)."""
    from corvidae.runtime import Runtime

    rt = Runtime(config_path=_write_config(tmp_path, ["web_tools"]))
    try:
        with caplog.at_level(logging.WARNING, logger="corvidae.runtime"):
            await rt.start()
    finally:
        await rt.stop()

    # Match the quoted value, not the installed-plugins list every unknown-name
    # warning carries (a kernel-only install warns for the cognition names).
    assert not [
        r for r in caplog.records
        if r.levelno == logging.WARNING and '"web_tools"' in r.getMessage()
    ]
