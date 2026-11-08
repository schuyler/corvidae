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
from dataclasses import dataclass
from pathlib import Path

import pytest

from corvidae.channel import Channel
from corvidae.tool import Tool
from tests.test_signal_plugin import (
    AUTH_ACI,
    Harness,
    _start_plugin,
    _stop_plugin,
    _wait_until,
)

# A few bytes standing in for an image; the daemon is fake, so content is moot.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


@dataclass
class SendEnv:
    """A running plugin with a send root, a small file in it, and a known
    Signal channel to send to."""

    h: Harness
    root: Path
    f: Path
    channel: Channel


@pytest.fixture
async def start_env(tmp_path):
    """Factory: start a plugin whose send root is ``tmp_path/out`` (kept out
    of the checkout), write ``pic.png`` there and register the target contact.

    Keyword arguments become signal: config overrides, e.g.
    max_attachment_size or message_chunk_size. Every plugin started is
    stopped at teardown, whether or not the test passed.
    """
    started: list[Harness] = []

    async def start(**overrides) -> SendEnv:
        root = tmp_path / "out"
        root.mkdir()
        signal_overrides = {"send_attachment_root": str(root), **overrides}
        h = await _start_plugin(tmp_path, signal_overrides=signal_overrides)
        started.append(h)
        f = root / "pic.png"
        f.write_bytes(PNG)
        channel = h.registry.get_or_create("signal", AUTH_ACI)
        return SendEnv(h=h, root=root, f=f, channel=channel)

    yield start
    # Teardown: stop every plugin this test started.
    for h in started:
        await _stop_plugin(h)


@pytest.fixture
async def env(start_env) -> SendEnv:
    """A started plugin with the default config (see start_env)."""
    return await start_env()


def _send_frames(h) -> list[dict]:
    """The `send` frames the daemon received, excluding typing and receipts."""
    return [f for f in h.server.received if f.get("method") == "send"]


class TestRegistration:
    """R8, and the name half of R1: the tool exists only when configured and
    cannot collide with Discord's."""

    async def test_tool_is_registered_when_configured(self, env):
        tools: list = []
        env.h.plugin.register_tools(tools)
        assert env.h.plugin.send_signal_attachment in tools

    async def test_tool_is_not_registered_when_inert(self, tmp_path):
        # No signal: block means an inert plugin: nothing to send through.
        h = await _start_plugin(tmp_path, omit_signal_block=True, wait_connect=False)
        try:
            tools: list = []
            h.plugin.register_tools(tools)
            assert tools == []
        finally:
            await _stop_plugin(h)

    async def test_tool_name_does_not_collide_with_discords(self, env):
        # The registry keeps the last tool of a given name while still
        # advertising both schemas, so two transports must not share a name.
        # Importing the Discord harness skips the whole module when the
        # optional discord extra is absent (the kernel CI job), so it is
        # imported here, where only this test needs it.
        pytest.importorskip("discord")
        from tests.test_discord_plugin import _start_plugin as _start_discord

        d = await _start_discord()
        try:
            tools: list = []
            env.h.plugin.register_tools(tools)
            d.plugin.register_tools(tools)
            names = [Tool.from_function(t).name for t in tools]
            assert "send_signal_attachment" in names
            assert len(names) == len(set(names)), names
        finally:
            await d.plugin.on_stop()


class TestSend:
    """R1, R7: the frame on the wire."""

    async def test_sends_file_and_caption_in_one_frame(self, env):
        out = await env.h.plugin.send_signal_attachment(
            env.channel.id, str(env.f), "here you go"
        )
        frames = _send_frames(env.h)
        assert len(frames) == 1
        assert frames[0]["params"] == {
            "recipient": AUTH_ACI,
            "attachments": [str(env.f.resolve())],
            "message": "here you go",
        }
        # The confirmation names the file and the channel.
        assert not out.startswith("Could not send")
        assert "pic.png" in out and env.channel.id in out

    @pytest.mark.parametrize("caption", ["", "   "])
    async def test_blank_caption_sends_no_message_key(self, env, caption):
        await env.h.plugin.send_signal_attachment(env.channel.id, str(env.f), caption)
        frames = _send_frames(env.h)
        # One frame: the file, with no message and no blank follow-up.
        assert len(frames) == 1
        assert frames[0]["params"] == {
            "recipient": AUTH_ACI,
            "attachments": [str(env.f.resolve())],
        }

    async def test_long_caption_is_split_not_truncated(self, start_env):
        env = await start_env(message_chunk_size=100)
        # 50 distinct tokens, about 2.5 chunks of 100 bytes.
        tokens = [f"w{i:03d}" for i in range(50)]
        await env.h.plugin.send_signal_attachment(
            env.channel.id, str(env.f), " ".join(tokens)
        )
        frames = _send_frames(env.h)
        # 100 bytes per chunk over ~250 bytes of text: exactly three frames.
        assert len(frames) == 3
        # Only the first frame carries the file; the rest are text.
        assert "attachments" in frames[0]["params"]
        assert all("attachments" not in fr["params"] for fr in frames[1:])
        messages = [fr["params"]["message"] for fr in frames]
        assert all(len(m.encode("utf-8")) <= 100 for m in messages)
        # Every token arrives, in order.
        joined = " ".join(messages)
        positions = [joined.index(t) for t in tokens]
        assert positions == sorted(positions)


class TestRootAndSize:
    """R2, R3: the jail and the cap, each refused before any frame is sent."""

    async def test_relative_path_is_read_under_the_root(self, env):
        out = await env.h.plugin.send_signal_attachment(env.channel.id, "pic.png", "")
        assert not out.startswith("Could not send"), out
        frames = _send_frames(env.h)
        assert frames[0]["params"]["attachments"] == [str(env.f.resolve())]

    async def test_path_outside_root_is_refused_naming_path_root_and_setting(
        self, env, tmp_path
    ):
        outside = tmp_path / "secret.txt"
        outside.write_bytes(b"x")
        out = await env.h.plugin.send_signal_attachment(env.channel.id, str(outside), "")
        assert out.startswith("Could not send:")
        assert str(outside) in out
        assert str(env.root.resolve()) in out
        assert "signal.send_attachment_root" in out
        assert _send_frames(env.h) == []

    async def test_missing_path_outside_root_gets_the_root_message(self, env, tmp_path):
        # The jail check runs before any existence check, so a missing path
        # outside the root gets the same error as an existing one. Otherwise
        # "not a file" vs "outside the root" would let the model probe which
        # files exist outside the root.
        out = await env.h.plugin.send_signal_attachment(
            env.channel.id, str(tmp_path / "nope.png"), ""
        )
        assert "signal.send_attachment_root" in out
        assert _send_frames(env.h) == []

    async def test_oversized_file_is_refused_naming_file_size_cap_and_setting(
        self, start_env
    ):
        # A cap of 1000 and a file of 1001 bytes: distinctive numbers that a
        # tmp path cannot contain by accident.
        env = await start_env(max_attachment_size=1000)
        f = env.root / "big.png"
        f.write_bytes(b"x" * 1001)
        out = await env.h.plugin.send_signal_attachment(env.channel.id, str(f), "")
        assert out.startswith("Could not send:")
        assert "big.png" in out and "1001" in out and "1000" in out
        assert "signal.max_attachment_size" in out
        assert _send_frames(env.h) == []

    async def test_file_exactly_at_the_cap_is_sent(self, start_env):
        # R3 refuses a file *over* the cap, so one exactly at it must go.
        env = await start_env(max_attachment_size=1000)
        f = env.root / "edge.png"
        f.write_bytes(b"x" * 1000)
        out = await env.h.plugin.send_signal_attachment(env.channel.id, str(f), "")
        assert not out.startswith("Could not send"), out
        assert len(_send_frames(env.h)) == 1

    async def test_directory_in_root_is_not_a_file(self, env):
        (env.root / "sub").mkdir()
        out = await env.h.plugin.send_signal_attachment(env.channel.id, "sub", "")
        assert out.startswith("Could not send:")
        assert "sub" in out and "is not a file" in out
        assert _send_frames(env.h) == []

    async def test_unresolvable_path_is_refused_naming_the_path(self, env):
        # An embedded NUL makes Path.resolve() raise ValueError, the R6 case of
        # a path that cannot be resolved: an error naming it, and no frame.
        out = await env.h.plugin.send_signal_attachment(env.channel.id, "pic\x00.png", "")
        assert out.startswith("Could not send:")
        assert "pic" in out and ".png" in out and "cannot read" in out
        assert _send_frames(env.h) == []


class TestDaemonPrefix:
    """R9: signal.daemon_outbound_prefix translates the path on the wire
    when the signal-cli daemon runs on a filesystem that does not share
    corvidae's view of the send root (a host daemon with the root
    bind-mounted into the container). The jail, the existence check, and
    the size cap always run on the local path."""

    async def test_prefix_translates_the_wire_path(self, start_env):
        env = await start_env(daemon_outbound_prefix="/home/sderle/signal-out")
        out = await env.h.plugin.send_signal_attachment(
            env.channel.id, "pic.png", ""
        )
        frames = _send_frames(env.h)
        assert len(frames) == 1
        # The daemon receives the host-side path; the local file is the
        # same file, reached through the mount.
        assert frames[0]["params"] == {
            "recipient": AUTH_ACI,
            "attachments": ["/home/sderle/signal-out/pic.png"],
        }
        assert not out.startswith("Could not send")
        assert "pic.png" in out

    async def test_prefix_applies_to_an_absolute_local_path(self, start_env):
        env = await start_env(daemon_outbound_prefix="/home/sderle/signal-out")
        await env.h.plugin.send_signal_attachment(
            env.channel.id, str(env.f), ""
        )
        frames = _send_frames(env.h)
        assert frames[0]["params"]["attachments"] == [
            "/home/sderle/signal-out/pic.png"
        ]

    async def test_prefix_preserves_subdirectories(self, start_env):
        env = await start_env(daemon_outbound_prefix="/home/sderle/signal-out")
        sub = env.root / "sub"
        sub.mkdir()
        (sub / "deep.png").write_bytes(PNG)
        await env.h.plugin.send_signal_attachment(
            env.channel.id, "sub/deep.png", ""
        )
        frames = _send_frames(env.h)
        assert frames[0]["params"]["attachments"] == [
            "/home/sderle/signal-out/sub/deep.png"
        ]

    async def test_prefix_does_not_widen_the_root_jail(self, start_env, tmp_path):
        env = await start_env(daemon_outbound_prefix="/home/sderle/signal-out")
        outside = tmp_path / "outside.png"
        outside.write_bytes(PNG)
        out = await env.h.plugin.send_signal_attachment(
            env.channel.id, str(outside), ""
        )
        assert out.startswith("Could not send")
        assert "outside the allowed root" in out
        assert _send_frames(env.h) == []

    @pytest.mark.parametrize("bad", ["", 5])
    async def test_bad_prefix_is_rejected_naming_key_and_value(
        self, tmp_path, bad
    ):
        with pytest.raises(ValueError) as excinfo:
            await _start_plugin(
                tmp_path,
                signal_overrides={"daemon_outbound_prefix": bad},
                wait_connect=False,
            )
        message = str(excinfo.value)
        assert "daemon_outbound_prefix" in message
        assert f"got {bad!r}" in message
        assert "non-empty string" in message


class TestTarget:
    """R5: only a known Signal channel is a valid target, and each refusal
    says which case it is."""

    async def test_foreign_transport_id_is_refused(self, env):
        # Whether or not the registry knows the id, the transport is wrong.
        env.h.registry.get_or_create("discord", "123")
        for target in ("discord:123", "discord:999"):
            out = await env.h.plugin.send_signal_attachment(target, str(env.f), "")
            assert out.startswith("Could not send:")
            assert target in out and "Signal channel" in out
        assert _send_frames(env.h) == []

    async def test_unknown_signal_id_is_refused(self, env):
        out = await env.h.plugin.send_signal_attachment("signal:nobody", str(env.f), "")
        assert out.startswith("Could not send:")
        assert "signal:nobody" in out and "not a known Signal channel" in out
        assert _send_frames(env.h) == []

    async def test_missing_registry_returns_an_error(self, env):
        env.h.plugin._registry = None
        out = await env.h.plugin.send_signal_attachment(env.channel.id, str(env.f), "")
        assert out.startswith("Could not send:")
        assert _send_frames(env.h) == []


class TestFailures:
    """R6: a failed send returns an error that says how far it got and what to
    do, logs with a traceback, and never raises into the turn."""

    async def test_file_frame_error_says_nothing_was_delivered(self, env, caplog):
        env.h.server.fail_next_matching("send", "boom")
        # The resolved path, so the remedy's path can be matched exactly.
        path = str(env.f.resolve())
        with caplog.at_level(logging.WARNING):
            out = await env.h.plugin.send_signal_attachment(env.channel.id, path, "hi")
        assert out.startswith("Could not send:")
        assert "pic.png" in out and env.channel.id in out
        assert "nothing was delivered" in out
        # The remedy for a daemon that cannot see the file.
        assert "read" in out and path in out
        # Logged at warning with the traceback and the identifiers.
        records = [r for r in caplog.records if r.levelno >= logging.WARNING and r.exc_info]
        assert any("pic.png" in r.getMessage() for r in records)

    async def test_later_caption_chunk_error_says_the_file_was_delivered(self, start_env):
        env = await start_env(message_chunk_size=100)
        sends: list[dict] = []

        def fail_second_send(frame: dict) -> None:
            # on_request runs before the fake answers this same frame, so
            # arming here fails exactly the second `send`.
            if frame.get("method") == "send":
                sends.append(frame)
                if len(sends) == 2:
                    env.h.server.fail_next_matching("send", "boom")

        env.h.server.on_request = fail_second_send
        caption = " ".join(f"w{i:03d}" for i in range(50))
        out = await env.h.plugin.send_signal_attachment(env.channel.id, str(env.f), caption)
        # Not the success form, and honest about what already went out.
        assert out.startswith("Could not send:")
        assert not out.startswith(f"Sent pic.png to {env.channel.id}")
        assert "pic.png" in out and env.channel.id in out
        assert "caption part 2 of" in out
        assert "do not resend the file" in out

    async def test_lost_connection_says_signal_cli_is_not_connected(self, env):
        await env.h.server.drop_connection()
        await _wait_until(lambda: env.h.plugin._writer is None)
        out = await env.h.plugin.send_signal_attachment(env.channel.id, str(env.f), "")
        assert out.startswith("Could not send:")
        assert "not connected" in out
        assert "pic.png" in out and env.channel.id in out


class TestConfig:
    """R4: signal.send_attachment_root defaults to `state` and rejects
    anything that is not a non-empty string, naming the key and the value."""

    async def test_default_root_is_state(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            assert h.plugin._send_attachment_root == Path("state")
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
