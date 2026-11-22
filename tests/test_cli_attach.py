"""The CLI can be attached to another channel (#1059).

Attaching is additive: stdin reads for that channel and the CLI displays its
outbound events, while a CLI that never attaches behaves exactly as before.
"""

from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock, patch

from corvidae.channel import ChannelRegistry
from corvidae.channels.cli import CLIPlugin
from corvidae.hooks import CorvidaePlugin, create_plugin_manager, hookimpl
from corvidae.outbound import OutboundMessage


class Recorder(CorvidaePlugin):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    @hookimpl
    async def on_message(self, channel, sender, text, trace_id):
        self.calls.append({"channel": channel, "text": text})


def _plugin():
    pm = create_plugin_manager()
    plugin = CLIPlugin(pm)
    pm.register(plugin, name="cli")
    registry = ChannelRegistry()
    plugin._registry = registry
    return pm, plugin, registry


async def test_attach_routes_stdin_to_the_attached_channel():
    pm, plugin, registry = _plugin()
    recorder = Recorder()
    pm.register(recorder, name="recorder")
    core = registry.get_or_create("discord", "room")
    plugin.attach(core)
    plugin._lines = asyncio.Queue()
    plugin._lines.put_nowait("hello\n")
    plugin._lines.put_nowait("")  # EOF

    fake_stdin = MagicMock()
    fake_stdin.isatty.return_value = False  # a pipe, not a terminal
    with patch("sys.stdin", fake_stdin), patch("sys.stdout"), \
         patch("corvidae.channels.cli.os.kill") as kill:
        await plugin._read_loop()

    kill.assert_not_called()  # a non-TTY EOF ends the transport, not the daemon

    assert recorder.calls == [{"channel": core, "text": "hello"}]


async def test_attach_does_not_change_the_channel_id():
    _, plugin, registry = _plugin()
    core = registry.get_or_create("discord", "room")
    before = core.id
    plugin.attach(core)
    assert core.id == before


async def test_an_attached_channel_is_owned_for_display():
    _, plugin, registry = _plugin()
    core = registry.get_or_create("discord", "room")
    assert plugin.owns(core) is False  # guard: nothing changes un-attached
    plugin.attach(core)
    assert plugin.owns(core) is True
    # The transport's own channels stay owned, attached or not.
    assert plugin.owns(registry.get_or_create("cli", "local")) is True


async def test_attach_makes_the_cli_display_the_attached_channel():
    _, plugin, registry = _plugin()
    core = registry.get_or_create("discord", "room")
    plugin.attach(core)

    with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
        await plugin.send(core, OutboundMessage(text="hi there"))

    assert "hi there" in fake_stdout.getvalue()


async def test_without_attach_the_cli_ignores_a_foreign_channel():
    _, plugin, registry = _plugin()
    core = registry.get_or_create("discord", "room")

    with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
        await plugin.send(core, OutboundMessage(text="hi there"))

    assert "hi there" not in fake_stdout.getvalue()
