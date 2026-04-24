import apluggy as pluggy
from pluggy import HookimplMarker, HookspecMarker

from sherman.hooks import AgentSpec, hookimpl, hookspec
from sherman.hooks import create_plugin_manager


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
