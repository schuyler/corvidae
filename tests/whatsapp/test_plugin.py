"""The WhatsApp transport against a fake sidecar (#1220).

These tests run the plugin the way the daemon does: a real plugin manager, a
real channel registry, and a unix socket standing in for the sidecar. They pin
the seam the two halves agreed on -- method names, params and the error codes --
so a change on either side breaks here rather than at an operator's first
message.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from corvidae.channel import ChannelRegistry
from corvidae.hooks import hookimpl
from corvidae.outbound import OutboundMessage
from corvidae.hooks import create_plugin_manager
from corvidae_whatsapp import WhatsAppPlugin

pytestmark = pytest.mark.timeout(30)

CHAT = "15551234567@s.whatsapp.net"


class FakeSidecar:
    """A unix-socket server speaking the sidecar's half of the seam."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.requests: list[dict] = []
        self.writer: asyncio.StreamWriter | None = None
        self._server: asyncio.AbstractServer | None = None
        #: method -> result, or an ("error", {...}) tuple for an error reply
        self.replies: dict[str, object] = {"status": {"state": "connected"}}

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(self._handle, path=self.path)

    async def stop(self) -> None:
        # Close the connection first: since 3.12 `server.wait_closed()` also
        # waits for its open connections, so the order decides whether this hangs.
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except (ConnectionError, OSError):
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        while line := await reader.readline():
            request = json.loads(line)
            self.requests.append(request)
            reply = self.replies.get(request["method"], {})
            if isinstance(reply, tuple) and reply[0] == "error":
                payload = {"jsonrpc": "2.0", "id": request["id"], "error": reply[1]}
            else:
                payload = {"jsonrpc": "2.0", "id": request["id"], "result": reply}
            writer.write(json.dumps(payload).encode() + b"\n")
            await writer.drain()

    async def notify(self, method: str, params: dict) -> None:
        assert self.writer is not None, "the plugin never connected"
        self.writer.write(
            json.dumps({"jsonrpc": "2.0", "method": method, "params": params}).encode() + b"\n"
        )
        await self.writer.drain()


class Recorder:
    """Records the channel messages the plugin hands to the agent."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []

    @hookimpl
    async def on_message(self, channel, sender: str, text: str, trace_id: str) -> None:
        self.messages.append((channel.id, sender, text))


async def _plugin(tmp_path, *, config: dict | None = None):
    """A registered plugin, its recorder, and the pm it lives in."""
    pm = create_plugin_manager()
    pm.register(ChannelRegistry(), name="registry")
    recorder = Recorder()
    pm.register(recorder, name="recorder")
    plugin = WhatsAppPlugin()
    pm.register(plugin, name="whatsapp")
    cfg = {"whatsapp": {"socket": str(tmp_path / "whatsapp.sock")}} if config is None else config
    await plugin.on_init(pm=pm, config=cfg)
    await plugin.on_start(config=cfg)
    return plugin, recorder, pm


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    """Wait until predicate() is true, or fail the test with the wait named."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0.01)


async def test_an_incoming_message_becomes_a_channel_message(tmp_path):
    """The sidecar's `message` notification reaches the agent's on_message."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        await sidecar.notify("message", {
            "id": "ABC123", "chat_jid": CHAT, "sender_jid": "15551234567@s.whatsapp.net",
            "from_me": False, "text": "hello there", "ts": 1,
        })
        await _wait_for(lambda: bool(recorder.messages))
        channel_id, sender, text = recorder.messages[0]
        assert channel_id == f"whatsapp:{CHAT}"
        assert sender == "15551234567@s.whatsapp.net"
        assert text == "hello there"
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_reply_is_sent_as_send_text(tmp_path):
    """Outbound text goes to the sidecar as send_text with the chat's JID."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, _recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        channel = plugin._registry.get_or_create("whatsapp", CHAT)
        await plugin.send_message(channel, OutboundMessage(text="hi back"))
        await _wait_for(lambda: any(r["method"] == "send_text" for r in sidecar.requests))
        [request] = [r for r in sidecar.requests if r["method"] == "send_text"]
        assert request["params"]["chat_jid"] == CHAT
        assert request["params"]["text"] == "hi back"
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_logged_out_link_names_the_fix(tmp_path, caplog):
    """-32001 must say the link is logged out and how to fix it, not raise."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    sidecar.replies["send_text"] = (
        "error",
        {"code": -32001, "message": "not linked", "data": {"state": "not_linked",
         "fix": "run the pair tool to relink"}},
    )
    await sidecar.start()
    plugin, _recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        channel = plugin._registry.get_or_create("whatsapp", CHAT)
        with caplog.at_level(logging.WARNING, logger="corvidae.whatsapp"):
            await plugin.send_message(channel, OutboundMessage(text="hi"))
        [line] = [r.getMessage() for r in caplog.records if "logged out" in r.getMessage()]
        assert "pair tool" in line
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_media_only_message_is_logged_not_silently_dropped(tmp_path, caplog):
    """v1 carries text only; a caption-less photo must leave a trace."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        with caplog.at_level(logging.INFO, logger="corvidae.whatsapp"):
            await sidecar.notify("message", {
                "id": "M1", "chat_jid": CHAT, "sender_jid": CHAT,
                "from_me": False, "text": "", "ts": 2, "media": {"type": "image"},
            })
            await asyncio.sleep(0.1)
        assert recorder.messages == []
        assert any("no text" in r.getMessage() for r in caplog.records)
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_block_without_a_socket_names_the_key(tmp_path):
    """A present-but-malformed block fails at startup, naming the key."""
    pm = create_plugin_manager()
    pm.register(ChannelRegistry(), name="registry")
    plugin = WhatsAppPlugin()
    pm.register(plugin, name="whatsapp")
    with pytest.raises(ValueError, match="'socket'"):
        await plugin.on_init(pm=pm, config={"whatsapp": {}})


async def test_no_block_means_no_connection(tmp_path):
    """Absence is inert: nothing connects and nothing is logged above DEBUG."""
    plugin, recorder, _pm = await _plugin(tmp_path, config={})
    try:
        assert plugin._client is None
        assert plugin._connect_task is None
        assert recorder.messages == []
    finally:
        await plugin.on_stop()


async def test_a_not_connected_link_is_logged_without_alarming(tmp_path, caplog):
    """-32002 is normal while the link comes up: INFO naming the fix, not WARNING."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    sidecar.replies["send_text"] = (
        "error",
        {"code": -32002, "message": "not connected", "data": {
            "state": "not_connected", "fix": "the sidecar is not connected yet; wait and try again"}},
    )
    await sidecar.start()
    plugin, _recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        channel = plugin._registry.get_or_create("whatsapp", CHAT)
        with caplog.at_level(logging.INFO, logger="corvidae.whatsapp"):
            await plugin.send_message(channel, OutboundMessage(text="hi"))
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING
                    and "send on" in r.getMessage()]
        assert any("not ready" in r.getMessage() for r in caplog.records)
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_the_link_state_is_read_from_state_not_connected(tmp_path, caplog):
    """A `reconnecting` link answers connected=false; `state` is the authority (#1220)."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    sidecar.replies["status"] = {"linked": False, "connected": False, "state": "reconnecting"}
    await sidecar.start()
    with caplog.at_level(logging.INFO, logger="corvidae.whatsapp"):
        plugin, _recorder, _pm = await _plugin(tmp_path)
        try:
            await _wait_for(lambda: plugin._session_state != "unknown")
            assert plugin._session_state == "reconnecting"
            assert any("reconnecting" in r.getMessage() for r in caplog.records)
        finally:
            await plugin.on_stop()
            await sidecar.stop()


async def test_a_replayed_message_id_starts_one_turn(tmp_path, caplog):
    """A reconnect replays unacked messages; the duplicate wamid is dropped (#1220)."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        message = {"id": "SAME1", "chat_jid": CHAT, "sender_jid": CHAT,
                   "from_me": False, "text": "hello", "ts": 1}
        with caplog.at_level(logging.INFO, logger="corvidae.whatsapp"):
            await sidecar.notify("message", message)
            await _wait_for(lambda: bool(recorder.messages))
            await sidecar.notify("message", message)
            await asyncio.sleep(0.1)
        assert len(recorder.messages) == 1
        assert any("duplicate message SAME1" in r.getMessage() for r in caplog.records)
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_sidecar_that_stays_down_warns_once(tmp_path, caplog):
    """One WARNING for a long outage, the rest at DEBUG — the log stays readable."""
    config = {"whatsapp": {"socket": str(tmp_path / "absent.sock")}}
    with caplog.at_level(logging.DEBUG, logger="corvidae.whatsapp"):
        plugin, _recorder, _pm = await _plugin(tmp_path, config=config)
        try:
            await _wait_for(lambda: plugin._connect_failures >= 3, timeout=20)
            warnings = [r for r in caplog.records
                        if r.levelno == logging.WARNING and "cannot reach" in r.getMessage()]
            assert len(warnings) == 1, [r.getMessage() for r in warnings]
            assert any(r.levelno == logging.DEBUG and "still unreachable" in r.getMessage()
                       for r in caplog.records)
        finally:
            await plugin.on_stop()
