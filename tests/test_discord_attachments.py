"""Discord attachments: inbound download and the outbound send tool (#521).

The three behaviours that matter are the allowlist gate before any fetch, the
path jail on the way out, and that attachment-only messages are no longer
dropped. Everything else here is the note wording and the size caps.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from corvidae.attachments import safe_suffix, save_attachment
from corvidae.channels import discord as dmod
from tests.test_discord_plugin import (
    BOT_ID,
    OTHER_ID,
    FakeMessageable,
    _start_plugin,
    make_dm,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


class _Content:
    """A StreamReader double: the body arrives in pieces, like a socket.

    ``iter_chunked(n)`` yields at most ``n`` bytes per chunk, in several
    chunks, which is what aiohttp does for any body bigger than one read —
    the bug this models is a client that keeps only the first chunk.
    """

    def __init__(self, body: bytes, chunks: int = 4) -> None:
        self._body = body
        self._chunks = max(1, chunks)
        # How many chunks the client pulled, so a test can tell a download
        # that stopped at the cap from one that read the whole body.
        self.consumed = 0

    async def iter_chunked(self, n: int):
        size = max(1, len(self._body) // self._chunks)
        for start in range(0, len(self._body), size):
            self.consumed += 1
            yield self._body[start:start + size]


class _Resp:
    """A minimal aiohttp response: status plus a streaming body."""

    def __init__(self, body: bytes = PNG, status: int = 200, chunks: int = 4) -> None:
        self.status = status
        self.content = _Content(body, chunks)
        self._body = body

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

    async def close(self):
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

    async def test_traversal_filename_contributes_only_its_extension(self, tmp_path, fetches):
        # With no content type the suffix falls back to the remote filename, so
        # this is the one path where that name reaches save_attachment. Only
        # its extension may land in the path: a name used whole would put the
        # file in a subdirectory of the attachment dir (or outside it).
        h = await _start_plugin(discord_overrides={"attachment_dir": str(tmp_path / "att")})
        try:
            msg = make_dm("", attachments=[_att(filename="../../etc/passwd.png", content_type="")])
            await h.plugin._handle_message(msg, BOT_ID)
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text.startswith("[image: "), text
            path = Path(text.split("[image: ")[1].rstrip("]"))
            # The name is ours (<message id>-<index>) plus the extension, and
            # nothing else from the remote name.
            assert path == (tmp_path / "att").resolve() / f"{msg.id}-0.png"
            assert path.read_bytes() == PNG
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
        # A valid extension followed by more must not pass on its prefix.
        assert safe_suffix("a.png/x") == ""
        assert safe_suffix(None) == ""

    async def test_data_over_max_bytes_is_not_written(self, tmp_path):
        assert await save_attachment(b"x" * 9, tmp_path, "big", max_bytes=8) is None
        assert list(tmp_path.iterdir()) == []

    async def test_stem_with_a_separator_is_refused(self, tmp_path):
        assert await save_attachment(PNG, tmp_path, "../escape") is None
        assert not any(tmp_path.parent.glob("escape*"))


class TestOutboundArguments:
    """The tool is called by a model, so its arguments are untrusted too."""

    async def _ready(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            dmod, "_sdk", lambda: SimpleNamespace(
                File=lambda p: p, AllowedMentions=SimpleNamespace(none=lambda: "none"),
            ),
        )
        h = await _start_plugin(discord_overrides={"send_attachment_root": str(tmp_path)})
        channel = h.registry.get_or_create("discord", str(4343))
        target = FakeMessageable()
        h.plugin._messageable = AsyncMock(return_value=target)
        return h, channel, target

    async def test_null_caption_is_allowed(self, tmp_path, monkeypatch):
        h, channel, target = await self._ready(tmp_path, monkeypatch)
        try:
            f = tmp_path / "a.png"
            f.write_bytes(PNG)
            out = await h.plugin.send_attachment(channel.id, str(f), None)
            assert "Sent a.png" in out
            assert target.sends[0][0] is None
        finally:
            await h.plugin.on_stop()

    async def test_non_string_caption_is_refused(self, tmp_path, monkeypatch):
        h, channel, target = await self._ready(tmp_path, monkeypatch)
        try:
            out = await h.plugin.send_attachment(channel.id, str(tmp_path), 42)
            assert "caption must be a string" in out
            assert target.sends == []
        finally:
            await h.plugin.on_stop()

    async def test_long_caption_splits_and_the_file_rides_first(self, tmp_path, monkeypatch):
        h, channel, target = await self._ready(tmp_path, monkeypatch)
        try:
            f = tmp_path / "a.png"
            f.write_bytes(PNG)
            out = await h.plugin.send_attachment(channel.id, str(f), "x" * 2500)
            assert "Sent a.png" in out
            assert len(target.sends) == 2
            assert "file" in target.sends[0][1]
            assert "file" not in target.sends[1][1]
        finally:
            await h.plugin.on_stop()


class TestTruncation:
    """A body that arrives in several chunks must be stored whole.

    Regression for the cold review's blocking finding: the first version used
    ``resp.content.read(cap + 1)``, which returns only what one socket read
    delivered, so every real screenshot was written truncated and reported as
    a success.
    """

    async def test_a_multi_chunk_body_is_written_whole(self, tmp_path, fetches):
        big = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 900  # ~230 KB
        fetches.set(_Resp(body=big, chunks=7))
        h = await _start_plugin(
            discord_overrides={"attachment_dir": str(tmp_path), "max_attachment_size": len(big) + 1},
        )
        try:
            await h.plugin._handle_message(make_dm("", attachments=[_att(size=len(big))]), BOT_ID)
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            path = Path(text.split("[image: ")[1].rstrip("]"))
            assert path.read_bytes() == big
        finally:
            await h.plugin.on_stop()

    async def test_a_body_that_grows_past_the_cap_is_refused(self, tmp_path, fetches):
        big = b"\x00" * 5000
        resp = _Resp(body=big, chunks=10)
        fetches.set(resp)
        h = await _start_plugin(
            discord_overrides={"attachment_dir": str(tmp_path), "max_attachment_size": 1000},
        )
        try:
            # Discord's stated size says it is small; the body is not.
            await h.plugin._handle_message(make_dm("", attachments=[_att(size=10)]), BOT_ID)
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text == "[image: download failed]"
            assert list(tmp_path.iterdir()) == []
            # The cap is enforced while streaming: 500-byte chunks against a
            # 1000-byte cap stop at the third, not after all ten were buffered.
            assert resp.content.consumed == 3
        finally:
            await h.plugin.on_stop()


class TestSuffixAndJail:
    """The extension is ours to choose, and the root is a real jail."""

    async def test_suffix_comes_from_the_content_type_not_the_name(self, tmp_path, fetches):
        h = await _start_plugin(discord_overrides={"attachment_dir": str(tmp_path)})
        try:
            await h.plugin._handle_message(
                make_dm("", attachments=[_att(filename="x.html", content_type="image/png")]), BOT_ID,
            )
            text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert text.rstrip("]").endswith(".png")
        finally:
            await h.plugin.on_stop()

    async def test_relative_path_is_relative_to_the_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            dmod, "_sdk", lambda: SimpleNamespace(
                File=lambda p: p, AllowedMentions=SimpleNamespace(none=lambda: "none"),
            ),
        )
        h = await _start_plugin(discord_overrides={"send_attachment_root": str(tmp_path)})
        channel = h.registry.get_or_create("discord", str(5151))
        target = FakeMessageable()
        h.plugin._messageable = AsyncMock(return_value=target)
        try:
            (tmp_path / "shot.png").write_bytes(PNG)
            out = await h.plugin.send_attachment(channel.id, "shot.png")
            assert "Sent shot.png" in out
        finally:
            await h.plugin.on_stop()

    async def test_a_symlink_out_of_the_root_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dmod, "_sdk", lambda: SimpleNamespace())
        h = await _start_plugin(discord_overrides={"send_attachment_root": str(tmp_path)})
        channel = h.registry.get_or_create("discord", str(5252))
        target = FakeMessageable()
        h.plugin._messageable = AsyncMock(return_value=target)
        try:
            (tmp_path / "escape.png").symlink_to("/etc/passwd")
            out = await h.plugin.send_attachment(channel.id, "escape.png")
            assert "outside the allowed root" in out
            assert target.sends == []
        finally:
            await h.plugin.on_stop()
