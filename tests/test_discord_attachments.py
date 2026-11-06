"""Discord attachments: inbound download and the outbound send tool (#521).

The three behaviours that matter are the allowlist gate before any fetch, the
path jail on the way out, and that attachment-only messages are no longer
dropped. Everything else here is the note wording and the size caps.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from corvidae.attachments import safe_suffix, save_attachment
from corvidae.channels import discord as dmod
from tests.test_discord_plugin import (
    AUTH_ID, BOT_ID, OTHER_ID, FakeMessageable, _start_plugin, make_dm, make_message,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


class _Resp:
    """A minimal aiohttp response: status plus a readable body."""

    def __init__(self, body: bytes = PNG, status: int = 200) -> None:
        self.status = status
        self.content = SimpleNamespace(read=self._read)
        self._body = body

    async def _read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


class _Session:
    def __init__(self, resp: _Resp, calls: list) -> None:
        self._resp = resp
        self._calls = calls

    def get(self, url):
        self._calls.append(url)
        return self._resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


@pytest.fixture()
def fetches(monkeypatch):
    """Record attachment URLs fetched; serve a PNG by default."""
    calls: list[str] = []
    state = {"resp": _Resp()}

    def _session(*_a, **_kw):
        return _Session(state["resp"], calls)

    monkeypatch.setattr(dmod.aiohttp, "ClientSession", _session)
    return SimpleNamespace(calls=calls, set=lambda resp: state.__setitem__("resp", resp))


def _att(filename="photo.png", content_type="image/png", size=len(PNG), url="https://cdn.example/a"):
    return SimpleNamespace(filename=filename, content_type=content_type, size=size, url=url)


class TestInbound:
    async def test_attachment_only_message_reaches_the_agent(self, tmp_path, fetches):
        h = await _start_plugin(discord_overrides={"attachment_dir": str(tmp_path)})
        try:
            await h.plugin._handle_message(
                make_dm("", attachments=[_att()]), BOT_ID,
            )
            assert h.pm.ahook.on_message.await_count == 1
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert "[image: " in text
            path = Path(text.split("[image: ")[1].rstrip("]"))
            assert path.parent == tmp_path
            assert path.read_bytes() == PNG
            assert path.suffix == ".png"
        finally:
            await h.plugin.on_stop()

    async def test_attachment_plus_text_keeps_both(self, tmp_path, fetches):
        h = await _start_plugin(discord_overrides={"attachment_dir": str(tmp_path)})
        try:
            await h.plugin._handle_message(
                make_dm("look at this", attachments=[_att()]), BOT_ID,
            )
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text.startswith("[image: ")
            assert text.endswith("look at this")
        finally:
            await h.plugin.on_stop()

    async def test_refused_sender_is_never_fetched(self, tmp_path, fetches):
        h = await _start_plugin(discord_overrides={"attachment_dir": str(tmp_path)})
        try:
            await h.plugin._handle_message(
                make_dm("", author_id=OTHER_ID, attachments=[_att()]), BOT_ID,
            )
            assert h.pm.ahook.on_message.await_count == 1
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text == "[image attached]"
            assert fetches.calls == []
            assert list(tmp_path.iterdir()) == []
        finally:
            await h.plugin.on_stop()

    async def test_oversize_attachment_is_refused_without_fetching(self, tmp_path, fetches):
        h = await _start_plugin(
            discord_overrides={"attachment_dir": str(tmp_path), "max_attachment_size": 8},
        )
        try:
            await h.plugin._handle_message(
                make_dm("", attachments=[_att()]), BOT_ID,
            )
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text == "[image: download failed]"
            assert fetches.calls == []
        finally:
            await h.plugin.on_stop()

    async def test_body_over_the_cap_is_not_written(self, tmp_path, fetches):
        # Discord's stated size lies; the body is what gets capped.
        h = await _start_plugin(
            discord_overrides={"attachment_dir": str(tmp_path), "max_attachment_size": 8},
        )
        try:
            await h.plugin._handle_message(
                make_dm("", attachments=[_att(size=4)]), BOT_ID,
            )
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text == "[image: download failed]"
            assert list(tmp_path.iterdir()) == []
        finally:
            await h.plugin.on_stop()

    async def test_traversal_filename_stays_inside(self, tmp_path, fetches):
        h = await _start_plugin(discord_overrides={"attachment_dir": str(tmp_path / "att")})
        try:
            await h.plugin._handle_message(
                make_dm("", attachments=[_att(filename="../../etc/passwd.png")]), BOT_ID,
            )
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            path = Path(text.split("[image: ")[1].rstrip("]"))
            assert path.parent == tmp_path / "att"
            assert path.suffix == ".png"
        finally:
            await h.plugin.on_stop()

    async def test_failed_download_is_noted_not_raised(self, tmp_path, fetches):
        fetches.set(_Resp(status=404, body=b""))
        h = await _start_plugin(discord_overrides={"attachment_dir": str(tmp_path)})
        try:
            await h.plugin._handle_message(make_dm("", attachments=[_att()]), BOT_ID)
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text == "[image: download failed]"
        finally:
            await h.plugin.on_stop()

    async def test_other_kinds_are_named_and_not_fetched(self, tmp_path, fetches):
        h = await _start_plugin(discord_overrides={"attachment_dir": str(tmp_path)})
        try:
            await h.plugin._handle_message(
                make_dm("", attachments=[_att(filename="clip.mp4", content_type="video/mp4")]),
                BOT_ID,
            )
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text == "[video attached]"
            assert fetches.calls == []
        finally:
            await h.plugin.on_stop()


class TestOutbound:
    async def _ready(self, tmp_path):
        h = await _start_plugin(discord_overrides={"send_attachment_root": str(tmp_path)})
        channel = h.registry.get_or_create("discord", str(4242))
        target = FakeMessageable()
        h.plugin._messageable = AsyncMock(return_value=target)
        return h, channel, target

    async def test_sends_a_file_with_a_caption(self, tmp_path, monkeypatch):
        sent: dict = {}

        class _File:
            def __init__(self, path): sent["path"] = path

        monkeypatch.setattr(
            dmod, "_sdk", lambda: SimpleNamespace(
                File=_File, AllowedMentions=SimpleNamespace(none=lambda: "none"),
            ),
        )
        h, channel, target = await self._ready(tmp_path)
        try:
            f = tmp_path / "shot.png"
            f.write_bytes(PNG)
            out = await h.plugin.send_attachment(channel.id, str(f), "here you go")
            assert "Sent shot.png" in out
            assert sent["path"] == str(f)
            assert target.sends[0][0] == "here you go"
            assert target.sends[0][1]["allowed_mentions"] == "none"
        finally:
            await h.plugin.on_stop()

    async def test_outside_the_root_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dmod, "_sdk", lambda: SimpleNamespace())
        h, channel, target = await self._ready(tmp_path)
        try:
            out = await h.plugin.send_attachment(channel.id, "/etc/passwd")
            assert "outside the allowed root" in out
            assert target.sends == []
        finally:
            await h.plugin.on_stop()

    async def test_missing_file_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dmod, "_sdk", lambda: SimpleNamespace())
        h, channel, target = await self._ready(tmp_path)
        try:
            out = await h.plugin.send_attachment(channel.id, str(tmp_path / "nope.png"))
            assert "not a readable file" in out
            assert target.sends == []
        finally:
            await h.plugin.on_stop()

    async def test_oversize_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dmod, "_sdk", lambda: SimpleNamespace())
        h, channel, target = await self._ready(tmp_path)
        try:
            h.plugin._max_attachment_size = 4
            f = tmp_path / "big.png"
            f.write_bytes(PNG)
            out = await h.plugin.send_attachment(channel.id, str(f))
            assert "byte limit" in out
            assert target.sends == []
        finally:
            await h.plugin.on_stop()

    async def test_non_discord_channel_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dmod, "_sdk", lambda: SimpleNamespace())
        h, _, _ = await self._ready(tmp_path)
        try:
            out = await h.plugin.send_attachment("signal:abc", str(tmp_path / "x.png"))
            assert "not a Discord channel" in out
        finally:
            await h.plugin.on_stop()

    async def test_tool_is_registered(self):
        h = await _start_plugin()
        try:
            tools: list = []
            h.plugin.register_tools(tools)
            assert h.plugin.send_attachment in tools
        finally:
            await h.plugin.on_stop()


class TestSharedHelper:
    def test_safe_suffix(self):
        assert safe_suffix("a/b/photo.JPG") == ".jpg"
        assert safe_suffix("no-extension") == ""
        assert safe_suffix("evil.py/../../x") == ""
        assert safe_suffix(None) == ""

    async def test_stem_with_a_separator_is_refused(self, tmp_path):
        assert await save_attachment(PNG, tmp_path, "../escape") is None
        assert not any(tmp_path.parent.glob("escape*"))


def test_bytes_are_base64_safe_png():
    """Guard: the fixture above is the PNG signature, not random bytes."""
    assert base64.b64encode(PNG)[:6] == b"iVBORw"
