"""Tests for corvidae.channels.signal.SignalPlugin.

Drives a real SignalPlugin against a FakeSignalServer (tests/signal_fake.py)
-- a stdlib-asyncio double of signal-cli's JSON-RPC daemon on a unix socket
in tmp_path. No signal-cli process and no library mocking: the plugin
connects to a real (fake) socket and speaks real (canned) JSON-RPC frames,
mirroring the pattern in test_irc_plugin.py.

Wire-protocol assumptions pinned by these tests (adjust here and in
signal_fake.py together if signal-cli's actual behavior differs -- several
of these are explicitly left for implementation-time verification by the
design):
  - Envelope fields: "sourceUuid" (the sender's ACI), "sourceNumber"
    (E.164, may be absent), "dataMessage" (with "message", "attachments",
    "groupInfo", "expiresInSeconds"), "syncMessage".
  - Outbound send: JSON-RPC method "send", with the outbound text under
    params["message"] -- explicit in the design ("outbound is a `send`
    request").
  - Typing indicator and read receipt: JSON-RPC method names are not
    specified by the design ("verify the method name"); tests match on
    "typing"/"receipt" appearing in the method name rather than an exact
    string, and match request content by searching the whole frame
    (signal_fake.frame_contains) rather than a specific param key, so
    they don't overspecify a wire detail the design deliberately left
    open.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from corvidae.channel import ChannelRegistry, load_channel_config
from corvidae.hooks import (
    create_plugin_manager,
    hookimpl,
    resolve_reject_wins,
    _check_hook_arg_binding,
)
from corvidae.outbound import OutboundMessage

# The transport under test does not exist yet -- this import is expected to
# fail until the green phase creates corvidae/channels/signal.py. That
# failure is the point of this file.
from corvidae.channels.signal import SignalPlugin

from signal_fake import FakeSignalServer, frame_contains


# ---------------------------------------------------------------------------
# Constants and envelope builders
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "reasoning_history": "drop",
}

BOT_NUMBER = "+15550001111"
BOT_ACI = "aaaaaaaa-0000-0000-0000-000000000000"
AUTH_ACI = "8f2c1111-1111-1111-1111-111111111111"
AUTH_NUMBER = "+15551234567"
OTHER_ACI = "9e3d2222-2222-2222-2222-222222222222"
OTHER_NUMBER = "+15559998888"

# The typing-refresh cadence the liveness tests run at. The transport's
# production cadence is ~10s (Signal clients expire a typing indicator after
# roughly 15s without a refresh); running the tests at that cadence would
# cost a minute and a half of wall clock and would never exercise the
# shortening hook at all. The hook is a module-level constant,
# corvidae.channels.signal.TYPING_REFRESH_SECONDS, which the transport must
# consult no earlier than plugin construction so that patching it here takes
# effect. patch() raises AttributeError if the constant is absent, so an
# implementation that hardcodes its interval fails these tests.
TYPING_REFRESH_FOR_TESTS = 0.25

# How long to watch for typing traffic before concluding the refresh loop
# has stopped: several shortened intervals, so a loop that is still running
# would certainly have fired.
TYPING_SILENCE_WINDOW = TYPING_REFRESH_FOR_TESTS * 6


def _fast_typing_refresh():
    """Patch the transport's typing-refresh interval down for a test.

    Must wrap plugin construction, not just the body, so an implementation
    that reads the constant once in __init__ still sees the short value.
    """
    return patch("corvidae.channels.signal.TYPING_REFRESH_SECONDS", TYPING_REFRESH_FOR_TESTS)


def _now_ms() -> int:
    """Current time as signal-cli-style epoch milliseconds."""
    return int(time.time() * 1000)


def make_data_envelope(
    *,
    source_uuid: str | None = None,
    source_number: str | None = None,
    text: str | None = "hello",
    timestamp: int | None = None,
    expires_in_seconds: int = 0,
    group_info: dict | None = None,
    attachments: list | None = None,
    account: str = BOT_NUMBER,
) -> dict:
    """Build a 'receive' notification's params for an inbound dataMessage."""
    ts = timestamp if timestamp is not None else _now_ms()
    data_message: dict = {"timestamp": ts, "expiresInSeconds": expires_in_seconds}
    if text is not None:
        data_message["message"] = text
    if group_info is not None:
        data_message["groupInfo"] = group_info
    if attachments is not None:
        data_message["attachments"] = attachments

    envelope: dict = {"timestamp": ts, "dataMessage": data_message}
    if source_uuid is not None:
        envelope["sourceUuid"] = source_uuid
    if source_number is not None:
        envelope["sourceNumber"] = source_number

    return {"envelope": envelope, "account": account}


def make_sync_envelope(*, text: str = "hi", destination: str = AUTH_NUMBER, timestamp: int | None = None) -> dict:
    """A syncMessage notification: the bot's own send, echoed from another
    device signed into the same account. Never legitimate input."""
    ts = timestamp if timestamp is not None else _now_ms()
    return {
        "envelope": {
            "sourceUuid": BOT_ACI,
            "sourceNumber": BOT_NUMBER,
            "timestamp": ts,
            "syncMessage": {
                "sentMessage": {"destination": destination, "message": text, "timestamp": ts}
            },
        },
        "account": BOT_NUMBER,
    }


def _record_text(record: logging.LogRecord) -> str:
    """A log record's message plus any exception text, for substring checks."""
    text = record.getMessage()
    if record.exc_info:
        text += " " + "".join(traceback.format_exception(*record.exc_info))
    return text


def _extract_text(send_frame: dict) -> str | None:
    """Pull the outbound message text out of a captured 'send' request."""
    params = send_frame.get("params", {})
    if isinstance(params, dict) and isinstance(params.get("message"), str):
        return params["message"]
    return None


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    """Poll predicate() until truthy, or raise asyncio.TimeoutError."""
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(interval)
    await asyncio.wait_for(_poll(), timeout=timeout)


async def _drive_gate(pm, *, channel, sender: str, text: str, trace_id: str) -> bool | None:
    """Run the message gate the way Agent.on_message runs it.

    Broadcasts should_process_message through pluggy, resolves it with
    reject-wins, and on a veto broadcasts on_message_rejected -- the same
    sequence corvidae/agent.py performs. Tests use this rather than calling
    the plugin's hookimpl directly so that registration, pluggy argument
    binding, and the follow-on rejection broadcast are all exercised.
    """
    results = await pm.ahook.should_process_message(
        channel=channel, sender=sender, text=text, trace_id=trace_id,
    )
    decision = resolve_reject_wins(results)
    if decision is False:
        await pm.ahook.on_message_rejected(
            channel=channel, trace_id=trace_id, sender=sender, text=text,
        )
    return decision


class _VetoGatePlugin:
    """A second gate plugin on signal channels, of the kind the design warns
    about: it vetoes a message the transport's own allowlist check has
    already admitted and counted as outstanding."""

    def __init__(self, veto_text: str) -> None:
        self.veto_text = veto_text

    @hookimpl
    async def should_process_message(self, channel, sender, text, trace_id):
        return False if text == self.veto_text else None


# ---------------------------------------------------------------------------
# Fake agent: the harness's on_message
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Stands in for Agent.on_message, mirroring corvidae/agent.py's sequence:
    should_process_message -> reject-wins -> on_message_rejected | admitted,
    all using the trace id the *caller passed*.

    ``ready=False`` models the agent-not-ready path: return without firing
    any hook. ``block`` holds an admitted call open. Installed as ``pm.ahook.on_message = AsyncMock(side_effect=...)``
    so the transport's real call (including its keyword names) is exercised.
    """

    def __init__(self, pm, ready: bool = True) -> None:
        self.pm = pm
        self.ready = ready
        # When set, the call blocks on this event after the message has been
        # admitted (a rejected message returns before it), modelling a turn
        # that is still running.
        self.block: asyncio.Event | None = None
        # (text, trace_id, channel) for each on_message call.
        self.calls: list[tuple[str, str, object]] = []

    def install(self) -> None:
        """Install this fake agent as the harness's ``on_message``."""
        # A bound async method, not the instance: AsyncMock only awaits a
        # side_effect it recognizes as a coroutine function.
        self.pm.ahook.on_message = AsyncMock(side_effect=self.__call__)

    def tid_for(self, text: str) -> str:
        """The trace id the transport passed for the message `text`."""
        return next(tid for t, tid, _ in self.calls if t == text)

    async def __call__(self, channel, sender, text, trace_id):
        # Record first so tests can find the id even on early return.
        self.calls.append((text, trace_id, channel))
        if not self.ready:
            return
        # Gate, then reject-wins, then the matching lifecycle hook.
        results = await self.pm.ahook.should_process_message(
            channel=channel, sender=sender, text=text, trace_id=trace_id,
        )
        if resolve_reject_wins(results) is False:
            await self.pm.ahook.on_message_rejected(
                channel=channel, trace_id=trace_id, sender=sender, text=text,
            )
            return
        await self.pm.ahook.on_message_admitted(
            channel=channel, trace_id=trace_id, sender=sender, text=text,
        )
        if self.block is not None:
            await self.block.wait()


# ---------------------------------------------------------------------------
# Harness: wires a FakeSignalServer + SignalPlugin together
# ---------------------------------------------------------------------------


@dataclass
class Harness:
    pm: object
    registry: ChannelRegistry
    plugin: SignalPlugin
    server: FakeSignalServer
    config: dict
    socket_dir: str
    # The fake agent installed as on_message (see _FakeAgent).
    agent: "_FakeAgent"


def _short_socket_path() -> Path:
    """A unix socket path short enough for AF_UNIX's ~104-byte sun_path limit.

    pytest's tmp_path nests under a directory named after the test, which
    routinely blows that limit (especially under macOS's long default
    TMPDIR) -- so this deliberately does not live under tmp_path.
    """
    socket_dir = tempfile.mkdtemp(dir="/tmp", prefix="cvsig-")
    return Path(socket_dir) / "s.sock"


async def _start_plugin(
    tmp_path,
    *,
    signal_overrides: dict | None = None,
    channels: dict | None = None,
    omit_signal_block: bool = False,
    wait_connect: bool = True,
    pre_queue: list[tuple[str, dict]] | None = None,
    pre_arm: list[tuple] | None = None,
    pre_fail: list[tuple[str, str]] | None = None,
) -> Harness:
    """Start a FakeSignalServer and a SignalPlugin connected to it.

    Mirrors what Runtime does: load_channel_config runs before on_start,
    on_init runs before on_start. Callers must call _stop_plugin in a
    finally block.

    pre_arm/pre_fail arm the server's canned responses/failures before the
    plugin connects -- needed for anything the plugin RPCs immediately on
    connect (allowlist alias resolution), since arming after _start_plugin
    returns would race the request it's meant to answer.
    """
    socket_path = _short_socket_path()
    server = FakeSignalServer(socket_path)
    await server.start()
    for method, params in (pre_queue or []):
        server.queue_before_connect(method, params)
    for value, result in (pre_arm or []):
        server.respond_when_contains(value, result)
    for method_substring, message in (pre_fail or []):
        server.fail_next_matching(method_substring, message)

    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")
    # on_message is a fake agent that runs the gate and fires the lifecycle
    # hooks with the transport's id, as the real Agent does; it is an
    # AsyncMock underneath so await_count / await_args still work.
    agent = _FakeAgent(pm)
    agent.install()

    config: dict = {}
    if not omit_signal_block:
        signal_config = {
            "socket": str(socket_path),
            "account": BOT_NUMBER,
            "allow": [AUTH_ACI],
            "message_chunk_size": 2000,
        }
        if signal_overrides:
            signal_config.update(signal_overrides)
        config["signal"] = signal_config
    if channels:
        config["channels"] = channels
        load_channel_config(config, registry)

    plugin = SignalPlugin(pm)
    pm.register(plugin, name="signal")
    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)
    if wait_connect:
        await server.wait_for_connection()

    return Harness(
        pm=pm, registry=registry, plugin=plugin, server=server, config=config,
        socket_dir=str(socket_path.parent), agent=agent,
    )


async def _stop_plugin(h: Harness) -> None:
    await h.plugin.on_stop()
    await h.server.stop()
    shutil.rmtree(h.socket_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


class TestConnectionLifecycle:
    async def test_no_signal_config_skips_silently(self, tmp_path, caplog):
        """An absent `signal:` block means no connection attempt, no error."""
        with caplog.at_level(logging.INFO):
            h = await _start_plugin(tmp_path, omit_signal_block=True, wait_connect=False)
        try:
            assert h.plugin._connect_task is None
            assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        finally:
            await _stop_plugin(h)

    async def test_connects_and_processes_a_frame(self, tmp_path):
        """Plugin opens the socket and dispatches a decoded inbound frame."""
        h = await _start_plugin(tmp_path)
        try:
            await h.server.push_notification(
                "receive",
                make_data_envelope(source_uuid=AUTH_ACI, source_number=AUTH_NUMBER, text="ping"),
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            assert h.pm.ahook.on_message.await_args.kwargs["text"] == "ping"
        finally:
            await _stop_plugin(h)

    async def test_dropped_connection_retries_with_backoff(self, tmp_path):
        """A dropped connection is retried, starting at the same ~10s backoff IRC uses."""
        h = await _start_plugin(tmp_path)
        try:
            assert h.server.connection_count == 1
            delays: list[float] = []
            real_sleep = asyncio.sleep

            async def tracking_sleep(delay, *a, **kw):
                delays.append(delay)
                await real_sleep(0.01 if delay > 0.5 else delay)

            with patch("corvidae.channels.signal.asyncio.sleep", side_effect=tracking_sleep):
                await h.server.drop_connection()
                # Poll with the captured, unpatched sleep -- asyncio.sleep is
                # patched process-wide for the duration of this block, so
                # _wait_until's own polling sleep would otherwise pollute
                # `delays` with its poll interval.
                for _ in range(500):
                    if h.server.connection_count >= 2:
                        break
                    await real_sleep(0.01)

            assert h.server.connection_count >= 2, "expected a reconnect attempt"
            assert delays, "expected the plugin to sleep before reconnecting"
            assert delays[0] == pytest.approx(10, rel=0.5)
        finally:
            await _stop_plugin(h)

    @pytest.mark.timeout(30)
    async def test_reconnect_backoff_doubles_and_caps(self, tmp_path):
        """With the daemon gone for good, the retry ladder climbs by doubling
        and flattens at the 300s ceiling -- the same ladder IRC uses."""
        h = await _start_plugin(tmp_path)
        try:
            delays: list[float] = []
            real_sleep = asyncio.sleep

            async def tracking_sleep(delay, *a, **kw):
                delays.append(delay)
                await real_sleep(0.001 if delay > 0.5 else delay)

            with patch("corvidae.channels.signal.asyncio.sleep", side_effect=tracking_sleep):
                # Take the socket away entirely so every reconnect fails and
                # the ladder is allowed to climb to its cap.
                await h.server.stop()
                # Poll with the captured, unpatched sleep (see the note in
                # the test above).
                for _ in range(4000):
                    if len(delays) >= 8:
                        break
                    await real_sleep(0.005)

            # Drop any long sentinel sleep the connection loop may use while
            # a connection is held; the retry ladder itself is what matters.
            retry_delays = [d for d in delays if d < 1000]
            assert len(retry_delays) >= 8, f"expected the retry ladder to keep climbing: {delays}"
            assert retry_delays[0] == pytest.approx(10, rel=0.5)
            for prev, nxt in zip(retry_delays, retry_delays[1:]):
                assert nxt >= min(prev * 2, 300), f"backoff must double until capped: {retry_delays}"
            assert max(retry_delays) <= 300, f"backoff must not exceed the 300s cap: {retry_delays}"
            assert max(retry_delays) == pytest.approx(300, rel=0.01), (
                f"the ladder must actually reach its 300s cap: {retry_delays}"
            )
        finally:
            await _stop_plugin(h)

    async def test_on_stop_cancels_cleanly_with_no_pending_tasks(self, tmp_path):
        """on_stop cancels the read task and leaves no task running behind it."""
        h = await _start_plugin(tmp_path)
        tasks_before = {t for t in asyncio.all_tasks() if not t.done()}
        await h.plugin.on_stop()
        await asyncio.sleep(0)  # let cancellation callbacks settle
        leaked = {t for t in asyncio.all_tasks() if not t.done()} - tasks_before
        assert not leaked, f"on_stop left tasks running: {leaked}"
        assert h.plugin._connect_task is None
        await h.server.stop()
        shutil.rmtree(h.socket_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Inbound decoding and filtering
# ---------------------------------------------------------------------------


class TestInboundFiltering:
    async def test_data_message_fires_on_message_with_decoded_text(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            await h.server.push_notification(
                "receive",
                make_data_envelope(source_uuid=AUTH_ACI, source_number=AUTH_NUMBER, text="hello there"),
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            kwargs = h.pm.ahook.on_message.await_args.kwargs
            assert kwargs["text"] == "hello there"
            assert kwargs["channel"].transport == "signal"
            assert kwargs["sender"] == AUTH_ACI
        finally:
            await _stop_plugin(h)

    async def test_sync_message_is_not_treated_as_input(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            await h.server.push_notification("receive", make_sync_envelope(text="mirrored from another device"))
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="a real message")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            assert h.pm.ahook.on_message.await_count == 1
            assert h.pm.ahook.on_message.await_args.kwargs["text"] == "a real message"
        finally:
            await _stop_plugin(h)

    async def test_group_message_is_ignored(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            await h.server.push_notification(
                "receive",
                make_data_envelope(
                    source_uuid=AUTH_ACI, source_number=None, text="group chatter",
                    group_info={"groupId": "abc123", "type": "DELIVER"},
                ),
            )
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="direct message")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            assert h.pm.ahook.on_message.await_count == 1
            assert h.pm.ahook.on_message.await_args.kwargs["text"] == "direct message"
        finally:
            await _stop_plugin(h)

    async def test_message_from_bot_own_account_is_ignored(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            await h.server.push_notification(
                "receive",
                make_data_envelope(source_uuid=BOT_ACI, source_number=BOT_NUMBER, text="talking to myself"),
            )
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="a real message")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            assert h.pm.ahook.on_message.await_count == 1
            assert h.pm.ahook.on_message.await_args.kwargs["text"] == "a real message"
        finally:
            await _stop_plugin(h)

    async def test_attachment_only_message_is_dropped_without_error(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            await h.server.push_notification(
                "receive",
                make_data_envelope(
                    source_uuid=AUTH_ACI, source_number=None, text=None,
                    attachments=[{"id": "att-1", "contentType": "image/jpeg"}],
                ),
            )
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="after the photo")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            assert h.pm.ahook.on_message.await_count == 1
            assert h.pm.ahook.on_message.await_args.kwargs["text"] == "after the photo"
        finally:
            await _stop_plugin(h)

    async def test_attachment_plus_text_processes_the_text(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            await h.server.push_notification(
                "receive",
                make_data_envelope(
                    source_uuid=AUTH_ACI, source_number=None, text="check this out",
                    attachments=[{"id": "att-2", "contentType": "image/png"}],
                ),
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            assert h.pm.ahook.on_message.await_args.kwargs["text"] == "check this out"
        finally:
            await _stop_plugin(h)

    async def test_malformed_frame_is_logged_and_loop_survives(self, tmp_path, caplog):
        h = await _start_plugin(tmp_path)
        try:
            with caplog.at_level(logging.WARNING):
                await h.server.push_raw_line("{not valid json")
                await h.server.push_notification(
                    "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="still alive")
                )
                await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            assert h.pm.ahook.on_message.await_args.kwargs["text"] == "still alive"
            assert any(r.levelno >= logging.WARNING for r in caplog.records)
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Channel scope and identity
# ---------------------------------------------------------------------------


class TestChannelScopeAndIdentity:
    async def test_scope_is_aci_not_number(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=AUTH_NUMBER, text="hi")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            channel = h.pm.ahook.on_message.await_args.kwargs["channel"]
            assert channel.id == f"signal:{AUTH_ACI}"
            assert AUTH_NUMBER not in channel.id
        finally:
            await _stop_plugin(h)

    async def test_alias_system_prompt_applies_via_direct_number_match(self, tmp_path):
        """A `channels:` alias keyed by E.164 lands its config on the ACI channel
        when the envelope carries a matching sourceNumber."""
        channels_config = {f"signal:{OTHER_NUMBER}": {"system_prompt": "alias prompt"}}
        # The sender is allowlisted: a refused one gets no registered
        # channel at all (#103), so identity is pinned on an allowed one.
        h = await _start_plugin(
            tmp_path, signal_overrides={"allow": [AUTH_ACI, OTHER_ACI]}, channels=channels_config,
        )
        try:
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=OTHER_ACI, source_number=OTHER_NUMBER, text="hi")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            channel = h.pm.ahook.on_message.await_args.kwargs["channel"]
            assert channel.id == f"signal:{OTHER_ACI}"
            resolved = h.registry.resolve_config(channel)
            assert resolved["system_prompt"] == "alias prompt"
        finally:
            await _stop_plugin(h)

    async def test_alias_system_prompt_applies_via_forward_resolution_when_number_absent(self, tmp_path):
        """When phone-number privacy hides sourceNumber, the alias still
        resolves -- via the same forward E.164->ACI getUserStatus
        resolution the allowlist uses, not a reverse ACI->number lookup.
        A reverse lookup can't reliably tell "not yet a contact" apart
        from "no such account" against a real daemon, so the alias's
        number must also be configured in allow: for this to resolve; an
        unauthorized sender never gets a reply, so an override that never
        resolves for them is harmless."""
        channels_config = {f"signal:{OTHER_NUMBER}": {"system_prompt": "alias prompt"}}
        h = await _start_plugin(
            tmp_path,
            signal_overrides={"allow": [OTHER_NUMBER]},
            channels=channels_config,
            pre_arm=[(
                OTHER_NUMBER,
                [{"recipient": OTHER_NUMBER, "number": OTHER_NUMBER, "username": None,
                  "uuid": OTHER_ACI, "isRegistered": True}],
            )],
        )
        try:
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=OTHER_ACI, source_number=None, text="hi")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            channel = h.pm.ahook.on_message.await_args.kwargs["channel"]
            assert channel.id == f"signal:{OTHER_ACI}"
            resolved = h.registry.resolve_config(channel)
            assert resolved["system_prompt"] == "alias prompt"
        finally:
            await _stop_plugin(h)

    async def test_number_form_and_aci_form_yield_a_byte_identical_channel_id(self, tmp_path):
        """The E.164 form in `channels:` and the ACI form name one
        conversation, not two: the id the inbound path lands on is
        byte-identical to the one get_or_create produces from the ACI alone,
        and it is not the alias id. Two ids here would fork the history that
        sessions.db, the jsonl logs, and memory are all keyed on."""
        channels_config = {f"signal:{OTHER_NUMBER}": {"system_prompt": "alias prompt"}}
        # The sender is allowlisted: a refused one gets no registered
        # channel at all (#103), so identity is pinned on an allowed one.
        h = await _start_plugin(
            tmp_path, signal_overrides={"allow": [AUTH_ACI, OTHER_ACI]}, channels=channels_config,
        )
        try:
            # The number form is pre-registered by load_channel_config.
            alias_channel = h.registry.get(f"signal:{OTHER_NUMBER}")
            assert alias_channel is not None

            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=OTHER_ACI, source_number=OTHER_NUMBER, text="hi")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            inbound_channel = h.pm.ahook.on_message.await_args.kwargs["channel"]

            # The direct-ACI form: what a config with no alias would produce.
            direct_channel = h.registry.get_or_create("signal", OTHER_ACI)

            assert inbound_channel.id.encode("utf-8") == direct_channel.id.encode("utf-8")
            assert inbound_channel is direct_channel
            assert inbound_channel.id != alias_channel.id
            assert OTHER_NUMBER not in inbound_channel.id
            # The alias entry's config landed on the channel traffic uses.
            assert h.registry.resolve_config(inbound_channel)["system_prompt"] == "alias prompt"
        finally:
            await _stop_plugin(h)

    async def test_channel_identity_stable_across_source_number_presence(self, tmp_path):
        """The same sender maps to the same channel whether or not a given
        message carries sourceNumber -- an identity change must not fork
        the history."""
        channels_config = {f"signal:{OTHER_NUMBER}": {"system_prompt": "alias prompt"}}
        # The sender is allowlisted: a refused one gets no registered
        # channel at all (#103), so identity is pinned on an allowed one.
        h = await _start_plugin(
            tmp_path, signal_overrides={"allow": [AUTH_ACI, OTHER_ACI]}, channels=channels_config,
        )
        try:
            h.server.respond_when_contains(OTHER_ACI, [{"number": OTHER_NUMBER, "uuid": OTHER_ACI}])

            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=OTHER_ACI, source_number=OTHER_NUMBER, text="first")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            first_channel = h.pm.ahook.on_message.await_args.kwargs["channel"]

            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=OTHER_ACI, source_number=None, text="second")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 2, timeout=5)
            second_channel = h.pm.ahook.on_message.await_args.kwargs["channel"]

            assert first_channel is second_channel
            assert first_channel.id == f"signal:{OTHER_ACI}"
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Authorization gate
# ---------------------------------------------------------------------------


class TestAuthorizationGate:
    async def test_allowlisted_sender_is_admitted(self, tmp_path):
        h = await _start_plugin(tmp_path, signal_overrides={"allow": [AUTH_ACI]})
        try:
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            result = await h.plugin.should_process_message(
                channel=channel, sender=AUTH_ACI, text="hi", trace_id="corr-1"
            )
            assert result is True
        finally:
            await _stop_plugin(h)

    async def test_non_allowlisted_sender_is_rejected_and_the_rejection_is_visible(self, tmp_path):
        """Default-deny, and the refusal is observable rather than a silent
        drop: driven through pluggy the way the agent drives it, the gate
        resolves to a veto and the follow-on on_message_rejected broadcast
        reaches its observers."""
        h = await _start_plugin(tmp_path, signal_overrides={"allow": [AUTH_ACI]})
        rejected = AsyncMock()
        h.pm.ahook.on_message_rejected = rejected
        try:
            channel = h.registry.get_or_create("signal", OTHER_ACI)
            decision = await _drive_gate(
                h.pm, channel=channel, sender=OTHER_ACI, text="hi", trace_id="corr-2",
            )
            assert decision is False
            rejected.assert_awaited_once()
            assert rejected.await_args.kwargs["trace_id"] == "corr-2"
            assert rejected.await_args.kwargs["sender"] == OTHER_ACI
        finally:
            await _stop_plugin(h)

    async def test_empty_allowlist_rejects_everyone(self, tmp_path):
        h = await _start_plugin(tmp_path, signal_overrides={"allow": []})
        try:
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            result = await h.plugin.should_process_message(
                channel=channel, sender=AUTH_ACI, text="hi", trace_id="corr-3"
            )
            assert result is False
        finally:
            await _stop_plugin(h)

    def test_should_process_message_binds_trace_id_without_default(self):
        """Registration-time guard: trace_id is spec-required. A
        hookimpl that defaults it would silently receive None from pluggy
        instead of the caller's value (see hooks.py:489)."""
        pm = create_plugin_manager()
        registry = ChannelRegistry(AGENT_DEFAULTS)
        pm.register(registry, name="registry")
        plugin = SignalPlugin(pm)
        pm.register(plugin, name="signal")

        _check_hook_arg_binding(pm)  # must not raise

    async def test_unauthorized_sender_gets_no_typing_or_receipt(self, tmp_path):
        """A refused sender leaves no signal that the account is live: no
        typing indicator, no read receipt."""
        h = await _start_plugin(tmp_path, signal_overrides={"allow": [AUTH_ACI]})
        try:
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=OTHER_ACI, source_number=None, text="uninvited")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)

            # Prove the absence is discriminating: a message right after,
            # from the allowlisted sender, does produce typing traffic --
            # and it is the very first frame the fake server ever read.
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="hello")
            )
            request = await h.server.next_request(timeout=5)
            assert "typing" in str(request.get("method", "")).lower()
            assert len(h.server.received) == 1
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Allowlist alias resolution: E.164 entries in `allow` resolve forward to
# an ACI via a single batched getUserStatus call, so the operator can write
# phone numbers into agent.yaml while matching still happens on the ACI --
# the persistence key -- alone.
# ---------------------------------------------------------------------------


class TestAllowlistAliasResolution:
    async def test_aci_allowlist_entry_needs_no_resolution(self, tmp_path):
        """A raw ACI entry in allow: never triggers any RPC at all -- not
        even a resolution attempt with nothing to resolve."""
        h = await _start_plugin(tmp_path, signal_overrides={"allow": [AUTH_ACI]})
        try:
            result = await h.plugin.should_process_message(
                channel=h.registry.get_or_create("signal", AUTH_ACI),
                sender=AUTH_ACI, text="hi", trace_id="corr-1",
            )
            assert result is True
            assert h.plugin._allow_resolution_task is None
            assert h.server.received == []
        finally:
            await _stop_plugin(h)

    async def test_e164_allowlist_alias_resolves_forward_and_admits_matching_aci(self, tmp_path):
        """A configured E.164 alias resolves forward via getUserStatus, and
        the sender's ACI -- not its number -- is what admits the message."""
        h = await _start_plugin(
            tmp_path,
            signal_overrides={"allow": [OTHER_NUMBER]},
            pre_arm=[(
                OTHER_NUMBER,
                [{"recipient": OTHER_NUMBER, "number": OTHER_NUMBER, "username": None,
                  "uuid": OTHER_ACI, "isRegistered": True}],
            )],
        )
        try:
            request = await h.server.next_request(timeout=5)
            assert request.get("method") == "getUserStatus"
            assert request["params"]["recipient"] == [OTHER_NUMBER]

            result = await h.plugin.should_process_message(
                channel=h.registry.get_or_create("signal", OTHER_ACI),
                sender=OTHER_ACI, text="hi", trace_id="corr-1",
            )
            assert result is True
        finally:
            await _stop_plugin(h)

    async def test_multiple_aliases_resolved_in_a_single_batched_call(self, tmp_path):
        """Resolving several configured E.164 aliases costs one CDS call,
        not one per alias -- CDS lookups are server-side rate-limited."""
        third_number = "+15557778888"
        third_aci = "cccccccc-3333-3333-3333-333333333333"
        h = await _start_plugin(
            tmp_path,
            signal_overrides={"allow": [OTHER_NUMBER, third_number]},
            pre_arm=[(
                OTHER_NUMBER,
                [
                    {"recipient": OTHER_NUMBER, "number": OTHER_NUMBER, "username": None,
                     "uuid": OTHER_ACI, "isRegistered": True},
                    {"recipient": third_number, "number": third_number, "username": None,
                     "uuid": third_aci, "isRegistered": True},
                ],
            )],
        )
        try:
            requests = []
            while True:
                try:
                    requests.append(await h.server.next_request(timeout=0.5))
                except asyncio.TimeoutError:
                    break
            status_requests = [r for r in requests if r.get("method") == "getUserStatus"]
            assert len(status_requests) == 1, "expected a single batched call, not one per alias"
            assert set(status_requests[0]["params"]["recipient"]) == {OTHER_NUMBER, third_number}

            first = await h.plugin.should_process_message(
                channel=h.registry.get_or_create("signal", OTHER_ACI),
                sender=OTHER_ACI, text="hi", trace_id="corr-1",
            )
            second = await h.plugin.should_process_message(
                channel=h.registry.get_or_create("signal", third_aci),
                sender=third_aci, text="hi", trace_id="corr-2",
            )
            assert first is True
            assert second is True
        finally:
            await _stop_plugin(h)

    async def test_unresolvable_allowlist_alias_logs_error_naming_it(self, tmp_path, caplog):
        """An alias the daemon reports as not registered is refused, and the
        log names the specific alias rather than reading as an ordinary
        unauthorized-sender refusal."""
        with caplog.at_level(logging.ERROR):
            h = await _start_plugin(
                tmp_path,
                signal_overrides={"allow": [OTHER_NUMBER]},
                pre_arm=[(
                    OTHER_NUMBER,
                    [{"recipient": OTHER_NUMBER, "number": None, "username": None,
                      "uuid": None, "isRegistered": False}],
                )],
            )
            try:
                result = await h.plugin.should_process_message(
                    channel=h.registry.get_or_create("signal", OTHER_ACI),
                    sender=OTHER_ACI, text="hi", trace_id="corr-1",
                )
                assert result is False
                assert any(
                    r.levelno == logging.ERROR and OTHER_NUMBER in r.getMessage()
                    for r in caplog.records
                )
            finally:
                await _stop_plugin(h)

    async def test_rpc_level_resolution_failure_is_distinguishable_from_not_registered(
        self, tmp_path, caplog
    ):
        """A failure of the batched call itself (e.g. a CDS rate limit) logs
        the underlying error text, and must not also produce the separate
        per-alias "did not resolve" message that means the number was
        looked up successfully and simply isn't on Signal."""
        with caplog.at_level(logging.ERROR):
            h = await _start_plugin(
                tmp_path,
                signal_overrides={"allow": [OTHER_NUMBER]},
                pre_fail=[("getUserStatus", "rate limit exceeded")],
            )
            try:
                result = await h.plugin.should_process_message(
                    channel=h.registry.get_or_create("signal", OTHER_ACI),
                    sender=OTHER_ACI, text="hi", trace_id="corr-1",
                )
                assert result is False
                rpc_failure_logs = [
                    r for r in caplog.records
                    if r.levelno == logging.ERROR and "rate limit exceeded" in r.getMessage()
                ]
                assert rpc_failure_logs, "expected the RPC failure logged with the underlying error text"
                not_registered_logs = [r for r in caplog.records if "did not resolve" in r.getMessage()]
                assert not not_registered_logs, (
                    "a whole-batch RPC failure must not also log a per-alias "
                    "not-registered message"
                )
            finally:
                await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Outbound send and splitting
# ---------------------------------------------------------------------------


class TestOutboundSend:
    async def test_send_message_on_non_signal_channel_is_noop(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            cli_channel = h.registry.get_or_create("cli", "local")
            await h.plugin.send(channel=cli_channel, event=OutboundMessage(text="ignored"))
            await asyncio.sleep(0.1)
            assert h.server.received == []
        finally:
            await _stop_plugin(h)

    async def test_reply_with_no_outstanding_message_is_delivered(self, tmp_path):
        """An agent-started turn (a reminder, say) answers no inbound
        message: the reply still goes out and the outstanding count stays
        at zero rather than going negative (design.md, Signal section)."""
        h = await _start_plugin(tmp_path)
        try:
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            assert h.plugin._outstanding.get(channel.id, 0) == 0
            await h.plugin.send(channel=channel, event=OutboundMessage(text="reminder: stand up"))
            # Wait for the send frame and check its text.
            while True:
                frame = await h.server.next_request(timeout=5)
                if frame.get("method") == "send":
                    break
            assert "reminder: stand up" in (_extract_text(frame) or "")
            assert h.plugin._outstanding.get(channel.id, 0) == 0
        finally:
            await _stop_plugin(h)

    async def test_send_message_failure_is_logged_and_does_not_raise(self, tmp_path, caplog):
        h = await _start_plugin(tmp_path)
        try:
            h.server.fail_next_matching("send", "delivery boom")
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            with caplog.at_level(logging.WARNING):
                await h.plugin.send(channel=channel, event=OutboundMessage(text="hello"))  # must not raise
                await _wait_until(
                    lambda: any("delivery boom" in _record_text(r) for r in caplog.records), timeout=5
                )
        finally:
            await _stop_plugin(h)

    async def test_long_reply_is_split_under_limit_and_delivered_in_full(self, tmp_path):
        h = await _start_plugin(tmp_path, signal_overrides={"message_chunk_size": 200})
        try:
            channel = h.registry.get_or_create("signal", AUTH_ACI)
            paragraphs = [f"Paragraph {i}. " + ("word " * 20) for i in range(6)]
            long_text = "\n\n".join(paragraphs)
            assert len(long_text) > 200

            await h.plugin.send(channel=channel, event=OutboundMessage(text=long_text))

            sends = []
            while True:
                try:
                    frame = await h.server.next_request(timeout=0.5)
                except asyncio.TimeoutError:
                    break
                if frame.get("method") == "send":
                    sends.append(frame)

            assert len(sends) > 1, "expected the reply to be split into multiple sends"
            chunks = [_extract_text(f) for f in sends]
            assert all(chunk is not None for chunk in chunks), sends
            assert all(len(chunk) <= 200 for chunk in chunks)

            # Delivered in full: every paragraph appears, in order, in the
            # concatenation of the delivered chunks. Not an equality
            # assertion, because whether the "\n\n" separator rides along
            # into a chunk is a splitter detail this transport does not
            # pin -- but a splitter that drops or truncates any paragraph
            # still fails here.
            joined = "".join(chunks)
            cursor = 0
            for para in paragraphs:
                found = joined.find(para, cursor)
                assert found >= 0, f"paragraph missing or truncated in delivery: {para!r}"
                cursor = found + len(para)
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Liveness: typing indicator and read receipt
# ---------------------------------------------------------------------------


class TestLiveness:
    async def test_typing_starts_before_on_message_is_dequeued(self, tmp_path):
        """Typing fires from the inbound handler itself, before the message
        is handed off -- covering queueing delay, not just generation."""
        h = await _start_plugin(tmp_path)
        order: list[str] = []
        h.server.on_request = lambda frame: order.append(f"request:{frame.get('method')}")
        block = asyncio.Event()
        # Wrap the fake agent: note the call, then admit and hold the turn open.
        h.agent.block = block

        async def blocking_on_message(*args, **kwargs):
            order.append("on_message_start")
            await h.agent(*args, **kwargs)

        h.pm.ahook.on_message = AsyncMock(side_effect=blocking_on_message)
        try:
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="hi")
            )
            await _wait_until(
                lambda: "on_message_start" in order and any(e.startswith("request:") for e in order),
                timeout=5,
            )
            typing_index = next(i for i, e in enumerate(order) if e.startswith("request:"))
            dequeue_index = order.index("on_message_start")
            assert typing_index < dequeue_index, order
        finally:
            block.set()
            await asyncio.sleep(0)
            await _stop_plugin(h)

    async def test_read_receipt_fires_at_processing_start_not_arrival(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            ts = _now_ms()
            await h.server.push_notification(
                "receive",
                make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="hi", timestamp=ts),
            )

            typing_request = await h.server.next_request(timeout=5)
            assert "typing" in str(typing_request.get("method", "")).lower()

            # Arrival alone must not produce a receipt.
            with pytest.raises(asyncio.TimeoutError):
                await h.server.next_request(timeout=0.5)

            # The fake agent already admitted the message under the id the
            # transport minted; persist it under that same id, as the
            # agent's queue consumer would when processing starts.
            _, trace_id, channel = h.agent.calls[0]
            await h.plugin.on_message_persisted(
                channel=channel, trace_id=trace_id, rowid=1, text="hi", meta={}
            )

            receipt_request = await h.server.next_request(timeout=5)
            assert "receipt" in str(receipt_request.get("method", "")).lower()
            assert frame_contains(receipt_request, ts)
        finally:
            await _stop_plugin(h)

    async def test_typing_refreshes_then_stops_after_final_reply(self, tmp_path):
        """A turn that outlasts the refresh interval gets typing sent at least
        twice, and stops for good once the owed reply is sent.

        Runs at the shortened interval, so the second frame is due within a
        fraction of a second; an implementation with no refresh loop sends
        one typing frame and never a second, and one that ignores
        TYPING_REFRESH_SECONDS does not send the second in time.
        """
        with _fast_typing_refresh():
            h = await _start_plugin(tmp_path)
            # Admit the message, then hold the turn open.
            block = asyncio.Event()
            h.agent.block = block
            try:
                await h.server.push_notification(
                    "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="slow turn")
                )

                first = await h.server.next_request(timeout=5)
                assert "typing" in str(first.get("method", "")).lower()

                # Deliberately far shorter than the production cadence: this
                # is what fails an implementation that never shortens.
                second = await h.server.next_request(timeout=TYPING_SILENCE_WINDOW)
                assert "typing" in str(second.get("method", "")).lower()

                block.set()
                channel = h.pm.ahook.on_message.await_args.kwargs["channel"]
                await h.plugin.send(channel=channel, event=OutboundMessage(text="final reply"))

                send_request = await h.server.next_request(timeout=5)
                assert send_request.get("method") == "send"

                with pytest.raises(asyncio.TimeoutError):
                    await h.server.next_request(timeout=TYPING_SILENCE_WINDOW)
            finally:
                await _stop_plugin(h)

    async def test_rejection_decrements_the_outstanding_typing_count(self, tmp_path):
        """Typing runs while the per-channel count of outstanding admitted
        messages is positive. A second gate plugin can veto a message the
        transport already counted in, so a rejection on the channel has to
        give that count back: three inbound messages, one vetoed, must stop
        typing after the second reply rather than sitting at a permanently
        positive count waiting for a third that will never come.

        Runs at the shortened refresh interval; both observation windows
        below span several of those intervals."""
        with _fast_typing_refresh():
            h = await _start_plugin(tmp_path)
            h.pm.register(_VetoGatePlugin("vetoed"), name="veto_gate")
            try:
                for text in ("one", "vetoed", "two"):
                    await h.server.push_notification(
                        "receive",
                        make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text=text),
                    )
                await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 3, timeout=5)
                channel = h.pm.ahook.on_message.await_args.kwargs["channel"]
                # The fake agent ran the gate for each message: the veto has
                # been broadcast, and the transport gave its slot back once
                # on_message returned without admitting it.
                await _wait_until(lambda: h.plugin._outstanding.get(channel.id) == 2, timeout=5)

                await h.plugin.send(channel=channel, event=OutboundMessage(text="reply to one"))
                # One reply is still owed, so typing must keep refreshing. This
                # is what makes the assertion below discriminating: a transport
                # that simply stopped typing on any send would pass that one.
                await asyncio.sleep(0.5)
                before = len(h.server.requests_matching("typing"))
                await asyncio.sleep(TYPING_SILENCE_WINDOW)
                assert len(h.server.requests_matching("typing")) > before, (
                    "typing must keep refreshing while a reply is still owed"
                )

                await h.plugin.send(channel=channel, event=OutboundMessage(text="reply to two"))
                await asyncio.sleep(0.5)
                settled = len(h.server.requests_matching("typing"))
                await asyncio.sleep(TYPING_SILENCE_WINDOW)
                assert len(h.server.requests_matching("typing")) == settled, (
                    "typing must stop once the rejection and both replies have "
                    "cleared the outstanding count; without a decrement on "
                    "rejection the count never reaches zero and typing leaks"
                )
            finally:
                await _stop_plugin(h)

    async def test_one_refresh_loop_per_channel_and_stop_cancels_it(self, tmp_path):
        """A message arriving while the previous turn's refresh loop is still
        asleep reuses that loop rather than starting a second one -- a
        second loop would double the refreshes, and once the first exits it
        would drop the second from tracking, so on_stop would leave it
        running."""

        def _live_loops() -> list[asyncio.Task]:
            # Every not-yet-finished task running the refresh coroutine.
            return [
                t for t in asyncio.all_tasks()
                if not t.done() and "_typing_refresh_loop" in repr(t.get_coro())
            ]

        with _fast_typing_refresh():
            h = await _start_plugin(tmp_path)
            try:
                # First turn: typing starts and the refresh loop goes to sleep.
                await h.server.push_notification(
                    "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="one"),
                )
                await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
                channel = h.pm.ahook.on_message.await_args.kwargs["channel"]

                # Reply, then a new message before the loop's sleep ends.
                await h.plugin.send(channel=channel, event=OutboundMessage(text="reply to one"))
                await h.server.push_notification(
                    "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="two"),
                )
                await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 2, timeout=5)
                assert len(_live_loops()) == 1

                # Let the loop wake: it must still be the tracked one.
                await asyncio.sleep(TYPING_REFRESH_FOR_TESTS * 2)
                assert len(_live_loops()) == 1
            finally:
                await _stop_plugin(h)
            assert _live_loops() == []

    async def test_typing_send_failure_is_logged_and_does_not_block_reply(self, tmp_path, caplog):
        h = await _start_plugin(tmp_path)
        try:
            h.server.fail_next_matching("typing", "typing-boom")
            with caplog.at_level(logging.WARNING):
                await h.server.push_notification(
                    "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="hi")
                )
                await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
                await _wait_until(
                    lambda: any("typing-boom" in _record_text(r) for r in caplog.records), timeout=5
                )
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Backlog timestamps
# ---------------------------------------------------------------------------


class TestBacklogTimestamps:
    async def test_backlog_messages_processed_in_order(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            base = _now_ms() - 3 * 24 * 3600 * 1000
            texts = ["first", "second", "third"]
            for i, text in enumerate(texts):
                await h.server.push_notification(
                    "receive",
                    make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text=text, timestamp=base + i * 1000),
                )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 3, timeout=5)

            calls = h.pm.ahook.on_message.await_args_list
            assert len(calls) == 3
            received_order = [c.kwargs["text"].splitlines()[-1] for c in calls]
            assert received_order == texts
        finally:
            await _stop_plugin(h)

    async def test_stale_message_gets_prefix_fresh_message_does_not(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            stale_ts = _now_ms() - 10 * 60 * 1000  # 10 minutes old
            await h.server.push_notification(
                "receive",
                make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="old news", timestamp=stale_ts),
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            stale_text = h.pm.ahook.on_message.await_args.kwargs["text"]

            expected_dt = datetime.fromtimestamp(stale_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            assert stale_text.startswith("[sent ")
            assert expected_dt in stale_text
            assert stale_text.endswith("old news")

            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text="fresh news")
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 2, timeout=5)
            fresh_text = h.pm.ahook.on_message.await_args.kwargs["text"]
            assert fresh_text == "fresh news"
        finally:
            await _stop_plugin(h)

    async def test_backlog_message_from_unauthorized_sender_still_gated(self, tmp_path):
        """Backlog does not bypass authorization: the same gate applies."""
        h = await _start_plugin(tmp_path, signal_overrides={"allow": [AUTH_ACI]})
        try:
            stale_ts = _now_ms() - 3 * 24 * 3600 * 1000
            await h.server.push_notification(
                "receive",
                make_data_envelope(
                    source_uuid=OTHER_ACI, source_number=None, text="uninvited backlog", timestamp=stale_ts
                ),
            )
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 1, timeout=5)
            channel = h.pm.ahook.on_message.await_args.kwargs["channel"]

            result = await h.plugin.should_process_message(
                channel=channel, sender=OTHER_ACI, text="uninvited backlog", trace_id="corr-x"
            )
            assert result is False
        finally:
            await _stop_plugin(h)

    async def test_queued_before_connect_backlog_delivered_in_order_on_connect(self, tmp_path):
        """Models --receive-mode=on-connection: messages queued while no
        client was attached are flushed, in order, the moment one attaches."""
        base = _now_ms() - 2 * 24 * 3600 * 1000
        texts = ["queued one", "queued two", "queued three"]
        pre_queue = [
            ("receive", make_data_envelope(source_uuid=AUTH_ACI, source_number=None, text=t, timestamp=base + i * 1000))
            for i, t in enumerate(texts)
        ]
        h = await _start_plugin(tmp_path, pre_queue=pre_queue)
        try:
            await _wait_until(lambda: h.pm.ahook.on_message.await_count >= 3, timeout=5)
            calls = h.pm.ahook.on_message.await_args_list
            order = [c.kwargs["text"].splitlines()[-1] for c in calls]
            assert order == texts
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# resolve_channel (#59): channel resolution for a persisted id after restart
# ---------------------------------------------------------------------------


class TestResolveChannel:
    async def test_unseen_aci_gets_the_e164_alias_config(self, tmp_path):
        """A scheduler row due after a restart carries only a channel id.
        resolve_channel must route through the alias resolver (ACI as
        scope, no source_number) so the E.164 alias's ChannelConfig lands
        on the channel, not a bare default one."""
        channels_config = {f"signal:{OTHER_NUMBER}": {"system_prompt": "alias prompt"}}
        h = await _start_plugin(
            tmp_path,
            signal_overrides={"allow": [OTHER_NUMBER]},
            channels=channels_config,
            pre_arm=[(
                OTHER_NUMBER,
                [{"recipient": OTHER_NUMBER, "number": OTHER_NUMBER, "username": None,
                  "uuid": OTHER_ACI, "isRegistered": True}],
            )],
        )
        try:
            # Forward resolution starts on connect; wait for it to be in
            # flight so the hook awaits it rather than racing its start.
            await _wait_until(lambda: h.plugin._allow_resolution_task is not None, timeout=5)
            assert h.registry.get(f"signal:{OTHER_ACI}") is None
            channel = await h.pm.ahook.resolve_channel(channel_id=f"signal:{OTHER_ACI}")
            assert channel is not None
            assert channel.id == f"signal:{OTHER_ACI}"
            assert h.registry.get(channel.id) is channel
            resolved = h.registry.resolve_config(channel)
            assert resolved["system_prompt"] == "alias prompt"
        finally:
            await _stop_plugin(h)

    async def test_known_channel_is_returned_as_is(self, tmp_path):
        h = await _start_plugin(tmp_path)
        try:
            existing = h.registry.get_or_create("signal", AUTH_ACI)
            channel = await h.pm.ahook.resolve_channel(channel_id=f"signal:{AUTH_ACI}")
            assert channel is existing
        finally:
            await _stop_plugin(h)

    async def test_non_signal_ids_defer(self, tmp_path):
        """Ids Signal does not own return None so the next impl (or the
        caller's registry fallback) handles them."""
        h = await _start_plugin(tmp_path)
        try:
            for channel_id in ("internal:heartbeat", "irc:#lex", "signal:", "nocolon"):
                assert await h.pm.ahook.resolve_channel(channel_id=channel_id) is None
            assert h.registry.get("internal:heartbeat") is None
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Issue #95: the transport mints the trace id
# ---------------------------------------------------------------------------


async def _push_text(h: Harness, text: str, **envelope_kwargs) -> None:
    """Push one allowlisted inbound message."""
    await h.server.push_notification(
        "receive",
        make_data_envelope(
            source_uuid=AUTH_ACI, source_number=None, text=text, **envelope_kwargs
        ),
    )


class TestTransportMintedIds:
    async def test_receipt_targets_right_message_after_veto(self, tmp_path):
        """O4: two allowlisted messages, the first vetoed by a second gate;
        persisting the second yields a read receipt for the SECOND's
        timestamp, not the vetoed first's."""
        h = await _start_plugin(tmp_path)
        agent = _FakeAgent(h.pm)
        agent.install()
        h.pm.register(_VetoGatePlugin("vetoed"), name="veto_gate")
        try:
            # Fresh timestamps: stale ones get a "[sent ...]" prefix, which
            # would stop the text-matching veto gate from firing.
            ts_first = _now_ms()
            ts_second = ts_first + 1
            # Send the to-be-vetoed message, then the one that is admitted.
            await _push_text(h, "vetoed", timestamp=ts_first)
            await _push_text(h, "kept", timestamp=ts_second)
            await _wait_until(lambda: len(agent.calls) >= 2, timeout=3)

            # Persist the second message, as the agent's queue consumer would.
            channel = agent.calls[1][2]
            await h.plugin.on_message_persisted(
                channel=channel, trace_id=agent.tid_for("kept"),
                rowid=1, text="kept", meta={},
            )

            await _wait_until(lambda: h.server.requests_matching("receipt"), timeout=3)
            receipt = h.server.requests_matching("receipt")[0]
            assert frame_contains(receipt, ts_second)
            assert not frame_contains(receipt, ts_first)
        finally:
            await _stop_plugin(h)

    async def test_agent_not_ready_leaves_no_residue(self, tmp_path):
        """O6: on_message returning without any hook (agent not ready)
        leaves no pending receipt, no outstanding slot, and typing stops."""
        with _fast_typing_refresh():
            h = await _start_plugin(tmp_path)
            agent = _FakeAgent(h.pm, ready=False)
            agent.install()
            try:
                await _push_text(h, "anyone home")
                await _wait_until(lambda: len(agent.calls) >= 1, timeout=3)
                channel = agent.calls[0][2]
                # Wait for the transport's post-call cleanup to settle.
                await _wait_until(
                    lambda: not h.plugin._pending_receipts.get(channel.id)
                    and h.plugin._outstanding.get(channel.id, 0) == 0,
                    timeout=3,
                )

                # Typing must go quiet: nothing new across several intervals.
                settled = len(h.server.requests_matching("typing"))
                await asyncio.sleep(TYPING_SILENCE_WINDOW)
                assert len(h.server.requests_matching("typing")) == settled
            finally:
                await _stop_plugin(h)

    async def test_vetoed_message_leaves_no_residue(self, tmp_path):
        """O6: a message vetoed by a second gate leaves no pending receipt,
        no outstanding slot, and typing stops."""
        with _fast_typing_refresh():
            h = await _start_plugin(tmp_path)
            agent = _FakeAgent(h.pm)
            agent.install()
            h.pm.register(_VetoGatePlugin("vetoed"), name="veto_gate")
            try:
                await _push_text(h, "vetoed")
                await _wait_until(lambda: len(agent.calls) >= 1, timeout=3)
                channel = agent.calls[0][2]
                # Wait for the transport's post-call cleanup to settle.
                await _wait_until(
                    lambda: not h.plugin._pending_receipts.get(channel.id)
                    and h.plugin._outstanding.get(channel.id, 0) == 0,
                    timeout=3,
                )

                # Typing must go quiet: nothing new across several intervals.
                settled = len(h.server.requests_matching("typing"))
                await asyncio.sleep(TYPING_SILENCE_WINDOW)
                assert len(h.server.requests_matching("typing")) == settled
            finally:
                await _stop_plugin(h)

    async def test_veto_without_timestamp_keeps_other_messages_slot(self, tmp_path):
        """O6/O1: an allowlisted sender's envelope with NO timestamp never
        took a typing slot, so its veto must not give back the slot owed to
        another admitted message."""
        h = await _start_plugin(tmp_path)
        agent = _FakeAgent(h.pm)
        agent.install()
        h.pm.register(_VetoGatePlugin("vetoed"), name="veto_gate")
        try:
            # An admitted message that is still owed a reply.
            await _push_text(h, "owed")
            await _wait_until(lambda: len(agent.calls) >= 1, timeout=3)
            channel = agent.calls[0][2]
            assert h.plugin._outstanding[channel.id] == 1

            # A vetoed message whose envelope carries no timestamp at all.
            envelope = make_data_envelope(
                source_uuid=AUTH_ACI, source_number=None, text="vetoed",
            )
            envelope["envelope"].pop("timestamp")
            envelope["envelope"]["dataMessage"].pop("timestamp")
            await h.server.push_notification("receive", envelope)
            await _wait_until(lambda: len(agent.calls) >= 2, timeout=3)
            await asyncio.sleep(0.2)

            # The owed message's slot is still held.
            assert h.plugin._outstanding[channel.id] == 1
        finally:
            await _stop_plugin(h)

    async def test_persist_for_unknown_id_sends_no_receipt(self, tmp_path):
        """O1/O3: receipts are looked up by id, not popped oldest-first. A
        notification-born turn's on_message_persisted carries an id Signal
        never minted; it must send nothing and leave the owed message's
        timestamp for that message's own persist."""
        h = await _start_plugin(tmp_path)
        agent = _FakeAgent(h.pm)
        agent.install()
        try:
            # An admitted message still waiting for its turn.
            ts = _now_ms()
            await _push_text(h, "owed", timestamp=ts)
            await _wait_until(lambda: len(agent.calls) >= 1, timeout=3)
            channel = agent.calls[0][2]

            # A notification-born turn on the same channel persists first.
            await h.plugin.on_message_persisted(
                channel=channel, trace_id="notif-id", rowid=9, text="n", meta={},
            )
            await asyncio.sleep(0.2)
            assert h.server.requests_matching("receipt") == []

            # The owed message's own persist sends its receipt.
            await h.plugin.on_message_persisted(
                channel=channel, trace_id=agent.tid_for("owed"),
                rowid=10, text="owed", meta={},
            )
            await _wait_until(lambda: h.server.requests_matching("receipt"), timeout=3)
            assert frame_contains(h.server.requests_matching("receipt")[0], ts)
            # The in-flight admitted set is drained once on_message returns.
            assert not h.plugin._admitted
        finally:
            await _stop_plugin(h)

    async def test_no_inference_state_remains(self, tmp_path):
        """O3: the FIFO timestamp queue and the admitted-id registry that
        inferred message identity are gone."""
        h = await _start_plugin(tmp_path)
        try:
            assert not hasattr(h.plugin, "_pending_timestamps")
            assert not hasattr(h.plugin, "_admitted_correlation_ids")
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Issue #103: a refused sender never creates a persistent channel
# ---------------------------------------------------------------------------


class TestRefusedSenderCreatesNoChannel:
    async def test_stranger_does_not_enter_the_registry(self, tmp_path):
        """PRD §3.2.6: an envelope from a sender outside allow: reaches the
        gate (on_message still fires) but leaves no channel behind."""
        h = await _start_plugin(tmp_path, signal_overrides={"allow": [AUTH_ACI]})
        try:
            await h.server.push_notification(
                "receive", make_data_envelope(source_uuid=OTHER_ACI, text="uninvited"),
            )
            await _wait_until(lambda: len(h.agent.calls) >= 1, timeout=5)
            assert h.registry.get(f"signal:{OTHER_ACI}") is None
            assert h.registry.by_transport("signal") == []
        finally:
            await _stop_plugin(h)

    async def test_refusal_still_reaches_the_gate_with_the_aci_channel_id(self, tmp_path, caplog):
        """The transient channel carries the same id a registered one would,
        so the gate's veto and its INFO line name the right sender."""
        h = await _start_plugin(tmp_path, signal_overrides={"allow": [AUTH_ACI]})
        rejected = AsyncMock()
        h.pm.ahook.on_message_rejected = rejected
        try:
            with caplog.at_level(logging.INFO, logger="corvidae.signal_plugin"):
                await h.server.push_notification(
                    "receive", make_data_envelope(source_uuid=OTHER_ACI, text="uninvited"),
                )
                await _wait_until(lambda: rejected.await_count >= 1, timeout=5)
            assert rejected.await_args.kwargs["channel"].id == f"signal:{OTHER_ACI}"
            assert rejected.await_args.kwargs["sender"] == OTHER_ACI
            assert any(
                "unauthorized sender" in r.getMessage() and OTHER_ACI in r.getMessage()
                for r in caplog.records
            )
        finally:
            await _stop_plugin(h)

    async def test_allowlisted_sender_still_gets_a_registered_channel(self, tmp_path):
        """Discriminating control: the allowed sender's channel is created."""
        h = await _start_plugin(tmp_path, signal_overrides={"allow": [AUTH_ACI]})
        try:
            await _push_text(h, "hello")
            await _wait_until(lambda: len(h.agent.calls) >= 1, timeout=5)
            channel = h.registry.get(f"signal:{AUTH_ACI}")
            assert channel is not None
            assert h.agent.calls[0][2] is channel
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Issue #104: an admitted message that is never persisted is swept
# ---------------------------------------------------------------------------


class TestOrphanedPendingReceipts:
    async def test_later_persist_sweeps_an_earlier_never_persisted_message(self, tmp_path, caplog):
        """A and B are admitted; A's turn dies before persistence. When B
        persists, the serial queue guarantees A is finished, so A's pending
        timestamp and typing slot are released: no receipt for A, one for
        B, and the outstanding count drops by A's slot."""
        h = await _start_plugin(tmp_path)
        try:
            ts_a = _now_ms()
            ts_b = ts_a + 1
            await _push_text(h, "a", timestamp=ts_a)
            await _push_text(h, "b", timestamp=ts_b)
            await _wait_until(lambda: len(h.agent.calls) >= 2, timeout=3)
            channel = h.agent.calls[0][2]
            assert h.plugin._outstanding[channel.id] == 2

            with caplog.at_level(logging.WARNING, logger="corvidae.signal_plugin"):
                await h.plugin.on_message_persisted(
                    channel=channel, trace_id=h.agent.tid_for("b"),
                    rowid=2, text="b", meta={},
                )

            await _wait_until(lambda: h.server.requests_matching("receipt"), timeout=3)
            receipts = h.server.requests_matching("receipt")
            assert len(receipts) == 1
            assert frame_contains(receipts[0], ts_b)
            assert not frame_contains(receipts[0], ts_a)
            # A's entry is gone and its slot given back; B still owes a reply.
            assert channel.id not in h.plugin._pending_receipts
            assert h.plugin._outstanding[channel.id] == 1
            # The rescue is logged at WARNING naming the channel and A's id.
            assert any(
                channel.id in r.getMessage() and h.agent.tid_for("a") in r.getMessage()
                for r in caplog.records if r.levelno == logging.WARNING
            )
        finally:
            await _stop_plugin(h)

    async def test_persist_does_not_sweep_messages_admitted_after_it(self, tmp_path):
        """Only entries older than the persisted id are orphans; a later
        message still queued keeps its receipt and slot."""
        h = await _start_plugin(tmp_path)
        try:
            ts_a = _now_ms()
            ts_b = ts_a + 1
            await _push_text(h, "a", timestamp=ts_a)
            await _push_text(h, "b", timestamp=ts_b)
            await _wait_until(lambda: len(h.agent.calls) >= 2, timeout=3)
            channel = h.agent.calls[0][2]

            await h.plugin.on_message_persisted(
                channel=channel, trace_id=h.agent.tid_for("a"),
                rowid=1, text="a", meta={},
            )
            await _wait_until(lambda: h.server.requests_matching("receipt"), timeout=3)
            assert h.plugin._pending_receipts[channel.id] == {h.agent.tid_for("b"): ts_b}
            assert h.plugin._outstanding[channel.id] == 2
        finally:
            await _stop_plugin(h)


# ---------------------------------------------------------------------------
# Issue #76: resolve_channel defers until alias config can be applied
# ---------------------------------------------------------------------------


class TestResolveChannelBeforeConnect:
    async def test_defers_before_first_connect_when_an_alias_is_configured(self, tmp_path):
        """PRD §3.2.7: before the first connect the E.164 alias cannot be
        matched to an ACI, so resolve_channel returns None and caches
        nothing, rather than a default-configured channel."""
        channels_config = {f"signal:{OTHER_NUMBER}": {"system_prompt": "alias prompt"}}
        h = await _start_plugin(
            tmp_path,
            signal_overrides={"allow": [OTHER_NUMBER]},
            channels=channels_config,
            wait_connect=False,
        )
        # Hold the connect off: the plugin is started but signal-cli has
        # not been reached, so no alias resolution has run.
        await h.server.stop()
        try:
            channel = await h.pm.ahook.resolve_channel(channel_id=f"signal:{OTHER_ACI}")
            assert channel is None
            assert h.registry.get(f"signal:{OTHER_ACI}") is None
        finally:
            await _stop_plugin(h)

    async def test_no_alias_configured_resolves_without_waiting(self, tmp_path):
        """With no E.164 channels: key there is nothing to wait for."""
        h = await _start_plugin(tmp_path, wait_connect=False)
        await h.server.stop()
        try:
            channel = await h.pm.ahook.resolve_channel(channel_id=f"signal:{OTHER_ACI}")
            assert channel is not None and channel.id == f"signal:{OTHER_ACI}"
        finally:
            await _stop_plugin(h)

    async def test_alias_outside_allow_does_not_block(self, tmp_path):
        """An E.164 channels: key that is not in allow: is never resolved
        (it is inert), so it must not hold resolve_channel back."""
        channels_config = {f"signal:{OTHER_NUMBER}": {"system_prompt": "alias prompt"}}
        h = await _start_plugin(
            tmp_path, signal_overrides={"allow": [AUTH_ACI]},
            channels=channels_config, wait_connect=False,
        )
        await h.server.stop()
        try:
            channel = await h.pm.ahook.resolve_channel(channel_id=f"signal:{AUTH_ACI}")
            assert channel is not None
        finally:
            await _stop_plugin(h)

    async def test_failed_resolution_attempt_no_longer_blocks(self, tmp_path, caplog):
        """Once an attempt has completed, a batch failure (retried only on
        the next connect) must not defer reminders indefinitely: the
        channel is created and the reduced config is logged at WARNING."""
        channels_config = {f"signal:{OTHER_NUMBER}": {"system_prompt": "alias prompt"}}
        h = await _start_plugin(
            tmp_path,
            signal_overrides={"allow": [OTHER_NUMBER]},
            channels=channels_config,
            pre_fail=[("getUserStatus", "rate limited")],
        )
        try:
            await _wait_until(
                lambda: h.plugin._allow_resolution_task is not None
                and h.plugin._allow_resolution_task.done(),
                timeout=5,
            )
            with caplog.at_level(logging.WARNING, logger="corvidae.signal_plugin"):
                channel = await h.pm.ahook.resolve_channel(channel_id=f"signal:{OTHER_ACI}")
            assert channel is not None
            assert any(
                OTHER_NUMBER in r.getMessage() and channel.id in r.getMessage()
                for r in caplog.records if r.levelno == logging.WARNING
            )
        finally:
            await _stop_plugin(h)
