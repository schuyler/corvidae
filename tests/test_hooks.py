import pytest
import apluggy as pluggy
from pluggy import HookimplMarker, HookspecMarker

from sherman.hooks import AgentSpec, hookimpl, hookspec
from sherman.hooks import create_plugin_manager
from sherman.hooks import get_dependency, validate_dependencies


def test_hookspec_marker_exists():
    assert isinstance(hookspec, HookspecMarker)


def test_hookimpl_marker_exists():
    assert isinstance(hookimpl, HookimplMarker)


async def test_register_plugin_with_hookimpl():
    called_with = {}

    class MyPlugin:
        @hookimpl
        async def on_start(self, config):
            called_with["config"] = config

    pm = create_plugin_manager()
    pm.register(MyPlugin())
    await pm.ahook.on_start(config={"key": "value"})

    assert called_with == {"config": {"key": "value"}}


def test_sync_hook_register_tools():
    tool_registry = []

    class MyPlugin:
        @hookimpl
        def register_tools(self, tool_registry):
            tool_registry.append("my_tool")

    pm = create_plugin_manager()
    pm.register(MyPlugin())
    pm.hook.register_tools(tool_registry=tool_registry)

    assert tool_registry == ["my_tool"]


async def test_multiple_plugins_receive_hook():
    from sherman.channel import Channel

    received = []

    class PluginA:
        @hookimpl
        async def on_message(self, channel, sender, text):
            received.append("A")

    class PluginB:
        @hookimpl
        async def on_message(self, channel, sender, text):
            received.append("B")

    pm = create_plugin_manager()
    pm.register(PluginA())
    pm.register(PluginB())
    ch = Channel(transport="test", scope="scope")
    await pm.ahook.on_message(channel=ch, sender="user", text="hi")

    assert sorted(received) == ["A", "B"]


# ---------------------------------------------------------------------------
# get_dependency
# ---------------------------------------------------------------------------

class _FakePlugin:
    """Minimal plugin class used as a type anchor in get_dependency tests."""


def test_get_dependency_returns_plugin_when_found_and_type_matches():
    pm = create_plugin_manager()
    plugin = _FakePlugin()
    pm.register(plugin, name="fake")

    result = get_dependency(pm, "fake", _FakePlugin)

    assert result is plugin


def test_get_dependency_raises_runtime_error_when_not_registered():
    pm = create_plugin_manager()

    with pytest.raises(RuntimeError):
        get_dependency(pm, "nonexistent", _FakePlugin)


def test_get_dependency_raises_type_error_when_wrong_type():
    class _OtherPlugin:
        pass

    pm = create_plugin_manager()
    pm.register(_OtherPlugin(), name="other")

    with pytest.raises(TypeError):
        get_dependency(pm, "other", _FakePlugin)


# ---------------------------------------------------------------------------
# validate_dependencies
# ---------------------------------------------------------------------------

def test_validate_dependencies_passes_when_all_declared_deps_are_registered():
    class _Dep:
        pass

    class _Consumer:
        depends_on = {"dep"}

    pm = create_plugin_manager()
    pm.register(_Dep(), name="dep")
    pm.register(_Consumer(), name="consumer")

    # Should not raise.
    validate_dependencies(pm)


def test_validate_dependencies_raises_runtime_error_when_dependency_missing():
    class _Consumer:
        depends_on = {"missing_dep"}

    pm = create_plugin_manager()
    pm.register(_Consumer(), name="consumer")

    with pytest.raises(RuntimeError):
        validate_dependencies(pm)


def test_validate_dependencies_passes_when_depends_on_is_empty_set():
    class _NoDeclaredDeps:
        depends_on: set[str] = set()

    pm = create_plugin_manager()
    pm.register(_NoDeclaredDeps(), name="nodeclared")

    # Should not raise — empty set means no dependencies.
    validate_dependencies(pm)


def test_validate_dependencies_skips_plugins_without_depends_on():
    class _NoDeps:
        pass

    pm = create_plugin_manager()
    pm.register(_NoDeps(), name="nodeps")

    # Should not raise.
    validate_dependencies(pm)
