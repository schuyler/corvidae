"""Signal send_signal_attachment tool (#544).

One test per acceptance criterion in the design (R1-R8): the wire frame the
fake signal-cli daemon sees, the root jail and size cap, the target checks,
the failure messages (and how far a send got before it failed), caption
splitting, config validation, and registration.

The fake daemon (tests/signal_fake.py) matches methods by substring, so
"send" also matches sendTyping/sendReceipt. Every assertion here filters on
``frame["method"] == "send"``, and the target contact is registered directly
with ``registry.get_or_create`` rather than by pushing an inbound envelope
(which would start typing frames).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from corvidae.tool import Tool
from tests.test_discord_plugin import _start_plugin as _start_discord
from tests.test_signal_plugin import AUTH_ACI, _start_plugin, _stop_plugin, _wait_until

# A few bytes standing in for an image; the daemon is fake, so content is moot.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


async def _start(tmp_path, **overrides):
    """Start a plugin whose send root is ``tmp_path/out`` (kept out of the checkout).

    Returns (harness, root). Extra keyword arguments become signal: config
    overrides, e.g. max_attachment_size or message_chunk_size.
    """
    root = tmp_path / "out"
    root.mkdir()
    signal_overrides = {"send_attachment_root": str(root), **overrides}
    h = await _start_plugin(tmp_path, signal_overrides=signal_overrides)
    return h, root


def _send_frames(h) -> list[dict]:
    """The `send` frames the daemon received, excluding typing and receipts."""
    return [f for f in h.server.received if f.get("method") == "send"]


class TestRegistration:
    """R8, and the name half of R1: the tool exists only when configured and
    cannot collide with Discord's."""

    async def test_tool_is_registered_when_configured(self, tmp_path):
        h, _ = await _start(tmp_path)
        try:
            tools: list = []
            h.plugin.register_tools(tools)
            assert h.plugin.send_signal_attachment in tools
        finally:
            await _stop_plugin(h)

    async def test_tool_is_not_registered_when_inert(self, tmp_path):
        # No signal: block means an inert plugin: nothing to send through.
        h = await _start_plugin(tmp_path, omit_signal_block=True, wait_connect=False)
        try:
            tools: list = []
            h.plugin.register_tools(tools)
            assert tools == []
        finally:
            await _stop_plugin(h)

    async def test_tool_name_does_not_collide_with_discords(self, tmp_path):
        # The registry keeps the last tool of a given name while still
        # advertising both schemas, so two transports must not share a name.
        h, _ = await _start(tmp_path)
        d = await _start_discord()
        try:
            tools: list = []
            h.plugin.register_tools(tools)
            d.plugin.register_tools(tools)
            names = [Tool.from_function(t).name for t in tools]
            assert "send_signal_attachment" in names
            assert len(names) == len(set(names)), names
        finally:
            await _stop_plugin(h)
            await d.plugin.on_stop()


class TestSend:
    """R1, R7: the frame on the wire."""

    async def test_sends_file_and_caption_in_one_frame(self, tmp_path):
        h, root = await _start(tmp_path)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            out = await h.plugin.send_signal_attachment(channel.id, str(f), "here you go")
            frames = _send_frames(h)
            assert len(frames) == 1
            assert frames[0]["params"] == {
                "recipient": AUTH_ACI,
                "attachments": [str(f.resolve())],
                "message": "here you go",
            }
            # The confirmation names the file and the channel.
            assert not out.startswith("Could not send")
            assert "pic.png" in out and channel.id in out
        finally:
            await _stop_plugin(h)

    @pytest.mark.parametrize("caption", ["", "   "])
    async def test_blank_caption_sends_no_message_key(self, tmp_path, caption):
        h, root = await _start(tmp_path)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            await h.plugin.send_signal_attachment(channel.id, str(f), caption)
            frames = _send_frames(h)
            # One frame: the file, with no message and no blank follow-up.
            assert len(frames) == 1
            assert frames[0]["params"] == {
                "recipient": AUTH_ACI,
                "attachments": [str(f.resolve())],
            }
        finally:
            await _stop_plugin(h)

    async def test_long_caption_is_split_not_truncated(self, tmp_path):
        h, root = await _start(tmp_path, message_chunk_size=100)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            # 50 distinct tokens, about 2.5 chunks of 100 bytes.
            tokens = [f"w{i:03d}" for i in range(50)]
            await h.plugin.send_signal_attachment(channel.id, str(f), " ".join(tokens))
            frames = _send_frames(h)
            assert len(frames) >= 3
            # Only the first frame carries the file; the rest are text.
            assert "attachments" in frames[0]["params"]
            assert all("attachments" not in fr["params"] for fr in frames[1:])
            messages = [fr["params"]["message"] for fr in frames]
            assert all(len(m.encode("utf-8")) <= 100 for m in messages)
            # Every token arrives, in order.
            joined = " ".join(messages)
            positions = [joined.index(t) for t in tokens]
            assert positions == sorted(positions)
        finally:
            await _stop_plugin(h)


class TestRootAndSize:
    """R2, R3: the jail and the cap, each refused before any frame is sent."""

    async def test_relative_path_is_read_under_the_root(self, tmp_path):
        h, root = await _start(tmp_path)
        try:
            (root / "pic.png").write_bytes(PNG)
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            out = await h.plugin.send_signal_attachment(channel.id, "pic.png", "")
            assert not out.startswith("Could not send"), out
            frames = _send_frames(h)
            assert frames[0]["params"]["attachments"] == [str((root / "pic.png").resolve())]
        finally:
            await _stop_plugin(h)

    async def test_path_outside_root_is_refused_naming_path_root_and_setting(self, tmp_path):
        h, root = await _start(tmp_path)
        try:
            outside = tmp_path / "secret.txt"
            outside.write_bytes(b"x")
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            out = await h.plugin.send_signal_attachment(channel.id, str(outside), "")
            assert out.startswith("Could not send:")
            assert str(outside) in out
            assert str(root.resolve()) in out
            assert "signal.send_attachment_root" in out
            assert _send_frames(h) == []
        finally:
            await _stop_plugin(h)

    async def test_missing_path_outside_root_gets_the_root_message(self, tmp_path):
        # The jail check runs before any existence check, so a path that does
        # not exist and is outside the root still reads as an out-of-root error.
        h, root = await _start(tmp_path)
        try:
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            out = await h.plugin.send_signal_attachment(
                channel.id, str(tmp_path / "nope.png"), ""
            )
            assert "signal.send_attachment_root" in out
            assert _send_frames(h) == []
        finally:
            await _stop_plugin(h)

    async def test_oversized_file_is_refused_naming_file_size_cap_and_setting(self, tmp_path):
        h, root = await _start(tmp_path, max_attachment_size=10)
        try:
            f = root / "big.png"
            f.write_bytes(b"x" * 11)
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            out = await h.plugin.send_signal_attachment(channel.id, str(f), "")
            assert out.startswith("Could not send:")
            assert "big.png" in out and "11" in out and "10" in out
            assert "signal.max_attachment_size" in out
            assert _send_frames(h) == []
        finally:
            await _stop_plugin(h)

    async def test_directory_in_root_is_not_a_file(self, tmp_path):
        h, root = await _start(tmp_path)
        try:
            (root / "sub").mkdir()
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            out = await h.plugin.send_signal_attachment(channel.id, "sub", "")
            assert out.startswith("Could not send:")
            assert "sub" in out and "is not a file" in out
            assert _send_frames(h) == []
        finally:
            await _stop_plugin(h)


class TestTarget:
    """R5: only a known Signal channel is a valid target, and each refusal
    says which case it is."""

    async def test_foreign_transport_id_is_refused(self, tmp_path):
        h, root = await _start(tmp_path)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            # Whether or not the registry knows the id, the transport is wrong.
            h.registry.get_or_create("discord", "123")
            for target in ("discord:123", "discord:999"):
                out = await h.plugin.send_signal_attachment(target, str(f), "")
                assert out.startswith("Could not send:")
                assert target in out and "Signal channel" in out
            assert _send_frames(h) == []
        finally:
            await _stop_plugin(h)

    async def test_unknown_signal_id_is_refused(self, tmp_path):
        h, root = await _start(tmp_path)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            out = await h.plugin.send_signal_attachment("signal:nobody", str(f), "")
            assert out.startswith("Could not send:")
            assert "signal:nobody" in out and "not a known Signal channel" in out
            assert _send_frames(h) == []
        finally:
            await _stop_plugin(h)

    async def test_missing_registry_returns_an_error(self, tmp_path):
        h, root = await _start(tmp_path)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            h.plugin._registry = None
            out = await h.plugin.send_signal_attachment(f"signal:{AUTH_ACI}", str(f), "")
            assert out.startswith("Could not send:")
            assert _send_frames(h) == []
        finally:
            await _stop_plugin(h)


class TestFailures:
    """R6: a failed send returns an error that says how far it got and what to
    do, logs with a traceback, and never raises into the turn."""

    async def test_file_frame_error_says_nothing_was_delivered(self, tmp_path, caplog):
        h, root = await _start(tmp_path)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            h.server.fail_next_matching("send", "boom")
            with caplog.at_level(logging.WARNING):
                out = await h.plugin.send_signal_attachment(channel.id, str(f), "hi")
            assert out.startswith("Could not send:")
            assert "pic.png" in out and channel.id in out
            assert "nothing was delivered" in out
            # The remedy for a daemon that cannot see the file.
            assert "read" in out and str(f.resolve()) in out
            # Logged at warning with the traceback and the identifiers.
            records = [r for r in caplog.records if r.levelno >= logging.WARNING and r.exc_info]
            assert any("pic.png" in r.getMessage() for r in records)
        finally:
            await _stop_plugin(h)

    async def test_later_caption_chunk_error_says_the_file_was_delivered(self, tmp_path):
        h, root = await _start(tmp_path, message_chunk_size=100)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            sends: list[dict] = []

            def fail_second_send(frame: dict) -> None:
                # on_request runs before the fake answers this same frame, so
                # arming here fails exactly the second `send`.
                if frame.get("method") == "send":
                    sends.append(frame)
                    if len(sends) == 2:
                        h.server.fail_next_matching("send", "boom")

            h.server.on_request = fail_second_send
            caption = " ".join(f"w{i:03d}" for i in range(50))
            out = await h.plugin.send_signal_attachment(channel.id, str(f), caption)
            # Not the success form, and honest about what already went out.
            assert out.startswith("Could not send:")
            assert not out.startswith(f"Sent pic.png to {channel.id}")
            assert "pic.png" in out and channel.id in out
            assert "caption part 2 of" in out
            assert "do not resend the file" in out
        finally:
            await _stop_plugin(h)

    async def test_lost_connection_says_signal_cli_is_not_connected(self, tmp_path):
        h, root = await _start(tmp_path)
        try:
            f = root / "pic.png"
            f.write_bytes(PNG)
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            await h.server.drop_connection()
            await _wait_until(lambda: h.plugin._writer is None)
            out = await h.plugin.send_signal_attachment(channel.id, str(f), "")
            assert out.startswith("Could not send:")
            assert "not connected" in out
            assert "pic.png" in out and channel.id in out
        finally:
            await _stop_plugin(h)


class TestConfig:
    """R4: signal.send_attachment_root defaults to `state` and rejects
    anything that is not a non-empty string, naming the key and the value."""

    async def test_default_root_is_state(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            assert h.plugin._send_attachment_root == Path("state")
        finally:
            await _stop_plugin(h)

    async def test_configured_root_is_honoured(self, tmp_path):
        h = await _start_plugin(
            tmp_path, signal_overrides={"send_attachment_root": str(tmp_path / "out")}
        )
        try:
            assert h.plugin._send_attachment_root == tmp_path / "out"
        finally:
            await _stop_plugin(h)

    @pytest.mark.parametrize("bad", ["", 5, None])
    async def test_bad_root_is_rejected_naming_key_and_value(self, tmp_path, bad):
        with pytest.raises(ValueError) as excinfo:
            await _start_plugin(
                tmp_path,
                signal_overrides={"send_attachment_root": bad},
                wait_connect=False,
            )
        message = str(excinfo.value)
        assert "send_attachment_root" in message
        assert f"got {bad!r}" in message
        assert "non-empty string" in message
