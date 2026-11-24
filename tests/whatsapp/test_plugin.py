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
    cfg = {"whatsapp": {"socket": str(tmp_path / "whatsapp.sock"), "allow": [CHAT]}} if config is None else config
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


async def test_a_media_only_message_names_the_fetched_file(tmp_path):
    """get_media's path is named in the text, so a caption-less photo still arrives."""
    media_dir = tmp_path / "state" / "whatsapp-media"
    media_dir.mkdir(parents=True)
    photo = media_dir / "wamid.M1.jpg"
    photo.write_bytes(b"jpegbytes")
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    sidecar.replies["get_media"] = {
        "path": str(photo), "kind": "image", "name": "IMG_1.jpg",
        "size": 9, "mime": "image/jpeg",
    }
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(
        tmp_path, config={"whatsapp": {"socket": str(tmp_path / "whatsapp.sock"),
                                       "media_dir": str(media_dir), "allow": [CHAT]}},
    )
    try:
        await _wait_for(lambda: plugin._client is not None)
        await sidecar.notify("message", {
            "id": "M1", "chat_jid": CHAT, "sender_jid": CHAT, "from_me": False,
            "text": "", "ts": 2,
            "media": [{"kind": "image", "name": "IMG_1.jpg", "mime": "image/jpeg", "size": 9}],
        })
        await _wait_for(lambda: bool(recorder.messages))
        assert recorder.messages[0][2] == f"[image: {photo}]"
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_media_path_outside_the_root_is_refused(tmp_path, caplog):
    """The sidecar's path is untrusted: anything outside media_dir never reaches the text."""
    media_dir = tmp_path / "state" / "whatsapp-media"
    media_dir.mkdir(parents=True)
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    sidecar.replies["get_media"] = {
        "path": "/etc/passwd", "kind": "image", "name": "x", "size": 1, "mime": "image/png",
    }
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(
        tmp_path, config={"whatsapp": {"socket": str(tmp_path / "whatsapp.sock"),
                                       "media_dir": str(media_dir), "allow": [CHAT]}},
    )
    try:
        await _wait_for(lambda: plugin._client is not None)
        with caplog.at_level(logging.WARNING, logger="corvidae.whatsapp"):
            await sidecar.notify("message", {
                "id": "M3", "chat_jid": CHAT, "sender_jid": CHAT, "from_me": False,
                "text": "look", "ts": 3,
                "media": [{"kind": "image", "name": "x", "mime": "image/png", "size": 1}],
            })
            await _wait_for(lambda: bool(recorder.messages))
        text = recorder.messages[0][2]
        assert "/etc/passwd" not in text
        assert "refused" in text
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_media_unavailable_is_named_by_state(tmp_path):
    """-32003 means bytes are unavailable, not pair/wait: the note says which state."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    sidecar.replies["get_media"] = (
        "error",
        {"code": -32003, "message": "too large", "data": {
            "state": "too_large", "fix": "none"}},
    )
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        await sidecar.notify("message", {
            "id": "M4", "chat_jid": CHAT, "sender_jid": CHAT, "from_me": False,
            "text": "big one", "ts": 4,
            "media": [{"kind": "document", "name": "a.zip", "mime": None, "size": 99999999}],
        })
        await _wait_for(lambda: bool(recorder.messages))
        assert "[document: unavailable (too_large)]" in recorder.messages[0][2]
        assert recorder.messages[0][2].endswith("big one")
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


async def test_a_dropped_message_does_not_consume_its_id(tmp_path):
    """A media-only message burns no id, so a replay with text still arrives (#1220)."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        await sidecar.notify("message", {
            "id": "M2", "chat_jid": CHAT, "sender_jid": CHAT,
            "from_me": False, "text": "", "ts": 3, "media": {"type": "image"},
        })
        await asyncio.sleep(0.1)
        assert recorder.messages == []
        # The same id with text now: an early insert would call this a duplicate.
        await sidecar.notify("message", {
            "id": "M2", "chat_jid": CHAT, "sender_jid": CHAT,
            "from_me": False, "text": "the caption", "ts": 3,
        })
        await _wait_for(lambda: bool(recorder.messages))
        assert recorder.messages[0][2] == "the caption"
    finally:
        await plugin.on_stop()
        await sidecar.stop()


class FlakyRecorder(Recorder):
    """Raises on the first message it sees, then records normally."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    @hookimpl
    async def on_message(self, channel, sender: str, text: str, trace_id: str) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("agent not ready")
        await super().on_message(channel, sender, text, trace_id)


async def test_a_raising_recorder_does_not_kill_the_read_loop(tmp_path, caplog):
    """A raising impl is swallowed by corvidae's hook guard: at-most-once.

    The guard converts the raise into a logged skip, so `await on_message`
    returns normally and the wamid is remembered anyway. The failed message is
    not redelivered; a later, distinct message still arrives.
    """
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    pm = create_plugin_manager()
    pm.register(ChannelRegistry(), name="registry")
    recorder = FlakyRecorder()
    pm.register(recorder, name="recorder")
    plugin = WhatsAppPlugin()
    pm.register(plugin, name="whatsapp")
    config = {"whatsapp": {"socket": str(tmp_path / "whatsapp.sock"), "allow": [CHAT]}}
    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)
    message = {"id": "R1", "chat_jid": CHAT, "sender_jid": CHAT,
               "from_me": False, "text": "hello", "ts": 4}
    try:
        await _wait_for(lambda: plugin._client is not None)
        await sidecar.notify("message", message)
        await _wait_for(lambda: recorder.failed)
        # At-most-once, pinned: the guard swallowed the raise inside the await,
        # so the id is remembered even though the hand-over failed.
        await _wait_for(lambda: "R1" in plugin._seen, timeout=2)
        await sidecar.notify("message", {**message, "id": "R2", "text": "second"})
        await _wait_for(lambda: bool(recorder.messages))
        assert recorder.messages[0][2] == "second"
    finally:
        await plugin.on_stop()
        await sidecar.stop()


class SlowSidecar(FakeSidecar):
    """Delays the get_media reply, so a second notification could overtake it."""

    async def _handle(self, reader, writer):
        self.writer = writer
        while line := await reader.readline():
            request = json.loads(line)
            self.requests.append(request)
            if request["method"] == "get_media":
                await asyncio.sleep(0.3)
            reply = self.replies.get(request["method"], {})
            if isinstance(reply, tuple) and reply[0] == "error":
                payload = {"jsonrpc": "2.0", "id": request["id"], "error": reply[1]}
            else:
                payload = {"jsonrpc": "2.0", "id": request["id"], "result": reply}
            writer.write(json.dumps(payload).encode() + b"\n")
            await writer.drain()


async def test_a_slow_media_fetch_does_not_let_the_next_message_overtake(tmp_path):
    """One worker: a chat's messages keep arrival order even while a fetch waits."""
    media_dir = tmp_path / "state" / "whatsapp-media"
    media_dir.mkdir(parents=True)
    photo = media_dir / "wamid.O1.jpg"
    photo.write_bytes(b"x")
    sidecar = SlowSidecar(str(tmp_path / "whatsapp.sock"))
    sidecar.replies["get_media"] = {"path": str(photo), "kind": "image",
                                    "name": "o.jpg", "size": 1, "mime": "image/jpeg"}
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(
        tmp_path, config={"whatsapp": {"socket": str(tmp_path / "whatsapp.sock"),
                                       "media_dir": str(media_dir), "allow": [CHAT]}},
    )
    try:
        await _wait_for(lambda: plugin._client is not None)
        await sidecar.notify("message", {
            "id": "O1", "chat_jid": CHAT, "sender_jid": CHAT, "from_me": False,
            "text": "first", "ts": 5,
            "media": [{"kind": "image", "name": "o.jpg", "mime": "image/jpeg", "size": 1}],
        })
        await sidecar.notify("message", {
            "id": "O2", "chat_jid": CHAT, "sender_jid": CHAT, "from_me": False,
            "text": "second", "ts": 6,
        })
        await _wait_for(lambda: len(recorder.messages) == 2, timeout=10)
        assert [text.splitlines()[-1] for _cid, _sender, text in recorder.messages] == [
            "first", "second",
        ]
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_media_message_without_an_id_is_named_not_sent_as_null(tmp_path, caplog):
    """No wamid means the seam cannot fetch; say so instead of sending null."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        with caplog.at_level(logging.WARNING, logger="corvidae.whatsapp"):
            await sidecar.notify("message", {
                "id": None, "chat_jid": CHAT, "sender_jid": CHAT, "from_me": False,
                "text": "no id", "ts": 7,
                "media": [{"kind": "image", "name": "x", "mime": "image/png", "size": 3}],
            })
            await _wait_for(lambda: bool(recorder.messages))
        assert "[image: unavailable (no id)]" in recorder.messages[0][2]
        assert not any(r["method"] == "get_media" for r in sidecar.requests)
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_send_media_sends_a_file_under_the_root(tmp_path):
    """The tool passes an absolute path inside the root and the caption through."""
    root = tmp_path / "state"
    root.mkdir()
    report = root / "report.pdf"
    report.write_bytes(b"%PDF-1.4")
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, _recorder, _pm = await _plugin(
        tmp_path, config={"whatsapp": {"socket": str(tmp_path / "whatsapp.sock"),
                                       "send_media_root": str(root), "allow": [CHAT]}},
    )
    try:
        await _wait_for(lambda: plugin._client is not None)
        channel = plugin._registry.get_or_create("whatsapp", CHAT)
        result = await plugin.send_whatsapp_media(
            f"whatsapp:{CHAT}", "report.pdf", "the numbers",
        )
        assert result.startswith("Sent report.pdf")
        sent = [r for r in sidecar.requests if r["method"] == "send_media"]
        assert sent[0]["params"] == {
            "chat_jid": CHAT, "path": str(report), "caption": "the numbers",
        }
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_send_media_refuses_a_path_outside_the_root(tmp_path):
    """A path outside send_media_root never reaches the sidecar."""
    root = tmp_path / "state"
    root.mkdir()
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, _recorder, _pm = await _plugin(
        tmp_path, config={"whatsapp": {"socket": str(tmp_path / "whatsapp.sock"),
                                       "send_media_root": str(root), "allow": [CHAT]}},
    )
    try:
        await _wait_for(lambda: plugin._client is not None)
        plugin._registry.get_or_create("whatsapp", CHAT)
        result = await plugin.send_whatsapp_media(f"whatsapp:{CHAT}", "/etc/passwd")
        assert "outside the allowed root" in result
        assert not [r for r in sidecar.requests if r["method"] == "send_media"]
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_stranger_is_refused_before_any_download(tmp_path, caplog):
    """Default-deny (#1220 R2): a number not in whatsapp.allow reaches nothing."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        with caplog.at_level(logging.WARNING, logger="corvidae.whatsapp"):
            await sidecar.notify("message", {
                "id": "S1", "chat_jid": "15559999999@s.whatsapp.net",
                "sender_jid": "15559999999@s.whatsapp.net", "from_me": False,
                "text": "hello from a stranger", "ts": 8,
                "media": [{"kind": "image", "name": "x", "mime": "image/png", "size": 9}],
            })
            await _wait_for(lambda: any("refused a message" in r.getMessage()
                                        for r in caplog.records))
        assert recorder.messages == []
        assert not [r for r in sidecar.requests if r["method"] == "get_media"]
        assert plugin._registry.get("whatsapp:15559999999@s.whatsapp.net") is None
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_the_allowlist_matches_digits_not_jid_spelling(tmp_path):
    """A number written as +E.164 allows the same person as their JID."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, recorder, _pm = await _plugin(
        tmp_path, config={"whatsapp": {"socket": str(tmp_path / "whatsapp.sock"),
                                       "allow": ["+15551234567"]}},
    )
    try:
        await _wait_for(lambda: plugin._client is not None)
        await sidecar.notify("message", {
            "id": "A1", "chat_jid": CHAT, "sender_jid": CHAT, "from_me": False,
            "text": "hi", "ts": 9,
        })
        await _wait_for(lambda: bool(recorder.messages))
        assert recorder.messages[0][2] == "hi"
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_session_logged_out_notification_warns_and_names_repairing(tmp_path, caplog):
    """The `session` notification is {linked, reason}; loggedOut must warn (#1220 R5).

    Reading a `state` key instead made every notification "unknown", so the one
    event that needs the operator saw an INFO line and nothing else.
    """
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, _recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        with caplog.at_level(logging.INFO, logger="corvidae.whatsapp"):
            await plugin._on_notification("session", {"linked": False, "reason": "loggedOut"})
        [line] = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "logged out" in r.getMessage()
        ]
        assert "re-pair" in line
        assert plugin._session_state == "loggedOut"
    finally:
        await plugin.on_stop()
        await sidecar.stop()


async def test_a_session_open_notification_is_quiet(tmp_path, caplog):
    """A healthy link says nothing at WARNING; the state is still recorded."""
    sidecar = FakeSidecar(str(tmp_path / "whatsapp.sock"))
    await sidecar.start()
    plugin, _recorder, _pm = await _plugin(tmp_path)
    try:
        await _wait_for(lambda: plugin._client is not None)
        with caplog.at_level(logging.INFO, logger="corvidae.whatsapp"):
            await plugin._on_notification("session", {"linked": True, "reason": "open"})
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert plugin._session_state == "open"
    finally:
        await plugin.on_stop()
        await sidecar.stop()
