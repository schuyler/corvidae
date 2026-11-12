"""Signal reaction tools signal_react / signal_unreact and their trigger map
(design: .claude/plans/reactions-design.md, R2-R6).

The trigger map (trace_id -> (author ACI, timestamp), recorded for allowlisted
senders only) does not have a public accessor, so these tests drive it the way
production does: push an inbound envelope through the fake daemon, read the
trace id the transport minted off the fake agent's on_message call, then call
the tool under ``set_attribution`` for that trace, as a turn would.

The fake daemon matches methods by substring, so every frame assertion
filters on ``frame["method"] == "sendReaction"``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from corvidae.attribution import reset_attribution, set_attribution
from tests.test_signal_plugin import (
    AUTH_ACI,
    OTHER_ACI,
    Harness,
    _now_ms,
    _start_plugin,
    _stop_plugin,
    _wait_until,
    make_data_envelope,
)


@pytest.fixture
def ts():
    """Distinctive timestamps (ms) computed at call time: recent so the
    transport does not treat the envelope as backlog and prefix its text
    (that happens past 300 s), distinct so a wrong target shows."""
    now = _now_ms()
    return SimpleNamespace(first=now - 2000, second=now - 1000)


@pytest.fixture
async def h(tmp_path):
    """A started plugin connected to the fake daemon; stopped at teardown."""
    harness = await _start_plugin(tmp_path)
    yield harness
    # Teardown: stop the plugin and server whether or not the test passed.
    await _stop_plugin(harness)


async def _inbound(h: Harness, text: str, *, ts, sender: str = AUTH_ACI) -> str:
    """Push one inbound text envelope and return the trace id the transport
    minted for it (read off the fake agent's on_message call)."""
    params = make_data_envelope(source_uuid=sender, text=text, timestamp=ts)
    await h.server.push_notification("receive", params)
    # Wait until the transport has fired on_message for this text.
    await _wait_until(lambda: any(t == text for t, _, _ in h.agent.calls))
    return h.agent.tid_for(text)


async def _inbound_raw_ts(h: Harness, text: str, ts_value) -> str:
    """Like _inbound but with the envelope's timestamp replaced by an
    arbitrary value (missing or non-int), after make_data_envelope built it."""
    params = make_data_envelope(source_uuid=AUTH_ACI, text=text, timestamp=_now_ms() - 2000)
    if ts_value is None:
        # Missing: drop the key from both the envelope and the dataMessage.
        params["envelope"].pop("timestamp")
        params["envelope"]["dataMessage"].pop("timestamp")
    else:
        params["envelope"]["timestamp"] = ts_value
        params["envelope"]["dataMessage"]["timestamp"] = ts_value
    await h.server.push_notification("receive", params)
    await _wait_until(lambda: any(t == text for t, _, _ in h.agent.calls))
    return h.agent.tid_for(text)


async def _react_as(h: Harness, fn_name: str, emoji: str, *, channel_id, trace_id):
    """Call a reaction tool under the attribution a turn would carry."""
    # Fetch the tool first so a missing tool surfaces as AttributeError.
    fn = getattr(h.plugin, fn_name)
    fields = {"channel_id": channel_id}
    if trace_id is not None:
        fields["trace_id"] = trace_id
    token = set_attribution(**fields)
    try:
        return await fn(emoji)
    finally:
        reset_attribution(token)


def _frames(h: Harness) -> list[dict]:
    """The sendReaction frames the daemon received."""
    return [f for f in h.server.received if f.get("method") == "sendReaction"]


# The channel id a turn from the allowlisted sender carries.
CHANNEL = f"signal:{AUTH_ACI}"


def _assert_no_message_error(out: str) -> None:
    """R5 error: names the missing target, the cause and the text-reply
    alternative. Case-insensitive and alternative-tolerant, like
    tests/test_discord_react.py."""
    low = out.lower()
    assert "no message to react to" in low
    # Names the cause: not started from a message, or forgotten.
    assert "did not start from a message" in low or "forgotten" in low
    # Names the alternative.
    assert "text reply" in low or "reply in text" in low


class TestReactFrame:
    """R2, R4: the RPC the daemon sees."""

    async def test_react_sends_sendreaction_frame_for_the_triggering_message(self, h, ts):
        """A react call sends one sendReaction frame aimed at the inbound message."""
        trace = await _inbound(h, "hello", ts=ts.first)
        out = await _react_as(h, "signal_react", "👍", channel_id=CHANNEL, trace_id=trace)
        frames = _frames(h)
        assert len(frames) == 1
        assert frames[0]["params"] == {
            "recipient": AUTH_ACI,
            "emoji": "👍",
            "targetAuthor": AUTH_ACI,
            "targetTimestamp": ts.first,
        }
        # Not an error string.
        assert "could not" not in out.lower()

    async def test_unreact_sends_the_same_frame_with_remove_true(self, h, ts):
        """Unreact sends the same target with remove=True."""
        trace = await _inbound(h, "hello", ts=ts.first)
        await _react_as(h, "signal_unreact", "👍", channel_id=CHANNEL, trace_id=trace)
        frames = _frames(h)
        assert len(frames) == 1
        assert frames[0]["params"] == {
            "recipient": AUTH_ACI,
            "emoji": "👍",
            "targetAuthor": AUTH_ACI,
            "targetTimestamp": ts.first,
            "remove": True,
        }


class TestTargeting:
    """R3: the target comes from turn provenance, never an argument."""

    async def test_tool_schema_exposes_only_emoji(self, h):
        """Neither tool's schema offers a target argument, only the emoji."""
        from corvidae.tool import Tool

        # Register through the real plugin manager, as startup does.
        tools: list = []
        h.pm.hook.register_tools(tool_registry=tools)
        by_name = {t.__name__: t for t in tools if hasattr(t, "__name__")}
        for name in ("signal_react", "signal_unreact"):
            assert name in by_name, f"{name} not registered"
            schema = Tool.from_function(by_name[name]).schema
            assert set(schema["function"]["parameters"]["properties"]) == {"emoji"}

    async def test_each_trace_reacts_to_its_own_message(self, h, ts):
        """Two interleaved messages each resolve to their own timestamp."""
        first = await _inbound(h, "one", ts=ts.first)
        second = await _inbound(h, "two", ts=ts.second)
        await _react_as(h, "signal_react", "👍", channel_id=CHANNEL, trace_id=first)
        await _react_as(h, "signal_react", "❤️", channel_id=CHANNEL, trace_id=second)
        stamps = [f["params"]["targetTimestamp"] for f in _frames(h)]
        assert stamps == [ts.first, ts.second]

    async def test_entry_for_a_different_channel_is_not_found(self, h, ts):
        """A trace recorded for one channel does not resolve under another."""
        trace = await _inbound(h, "hello", ts=ts.first)
        # Same trace, but the attribution names another channel.
        out = await _react_as(
            h, "signal_react", "👍", channel_id="signal:someone-else", trace_id=trace
        )
        _assert_no_message_error(out)
        assert _frames(h) == []

    async def test_message_from_non_allowlisted_sender_is_not_recorded(self, h, ts):
        """Only allowlisted senders' messages become reaction targets."""
        trace = await _inbound(h, "intruder", ts=ts.first, sender=OTHER_ACI)
        out = await _react_as(
            h, "signal_react", "👍", channel_id=f"signal:{OTHER_ACI}", trace_id=trace
        )
        _assert_no_message_error(out)
        assert _frames(h) == []

    @pytest.mark.parametrize("kind", ["missing", "float"])
    async def test_envelope_timestamp_missing_or_not_int_records_nothing(
        self, h, ts, kind
    ):
        """A timestamp that is absent or not an int cannot be a reaction
        target, so the trigger map must hold no entry for the message."""
        bad_ts = None if kind == "missing" else ts.first + 0.5
        trace = await _inbound_raw_ts(h, "odd", bad_ts)
        out = await _react_as(h, "signal_react", "👍", channel_id=CHANNEL, trace_id=trace)
        _assert_no_message_error(out)
        assert _frames(h) == []


class TestNoTarget:
    """R5: no message to react to is an error string naming the cause and the
    text-reply alternative; nothing is raised."""

    @pytest.mark.parametrize("tool", ["signal_react", "signal_unreact"])
    @pytest.mark.parametrize(
        "trace_id, channel_id",
        [
            (None, CHANNEL),  # a reminder or heartbeat turn: no trace at all
            ("trace-never-seen", CHANNEL),  # a trace the transport forgot
            ("trace-on-discord", "discord:123"),  # a turn on another transport
        ],
        ids=["no-trace", "unknown-trace", "non-signal-channel"],
    )
    async def test_returns_the_no_message_error(self, h, tool, trace_id, channel_id):
        """Both tools return the R5 error, send no frame, and do not raise."""
        out = await _react_as(h, tool, "👍", channel_id=channel_id, trace_id=trace_id)
        _assert_no_message_error(out)
        assert _frames(h) == []


class TestRegistration:
    """R6: tools register only when the signal block configures the transport."""

    async def test_registered_through_the_plugin_manager_when_configured(self, h):
        """Both tools appear in register_tools once signal is configured."""
        tools: list = []
        h.pm.hook.register_tools(tool_registry=tools)
        names = {t.__name__ for t in tools if hasattr(t, "__name__")}
        assert {"signal_react", "signal_unreact"} <= names

    async def test_not_registered_when_inert(self, tmp_path):
        """An inert plugin (no signal block) registers neither tool."""
        # No signal: block means an inert plugin: nothing to react through.
        inert = await _start_plugin(tmp_path, omit_signal_block=True, wait_connect=False)
        try:
            tools: list = []
            inert.pm.hook.register_tools(tool_registry=tools)
            names = {t.__name__ for t in tools if hasattr(t, "__name__")}
            assert not names & {"signal_react", "signal_unreact"}
        finally:
            await _stop_plugin(inert)


class TestFailures:
    """Daemon rejection and a lost connection: an error naming the channel and
    the reason, logged at WARNING, never raised."""

    async def test_daemon_rejection_is_returned_with_its_text_and_logged(self, h, ts, caplog):
        """A daemon error is returned with its text and logged at WARNING."""
        trace = await _inbound(h, "hello", ts=ts.first)
        h.server.fail_next_matching("sendReaction", "bad emoji from daemon")
        with caplog.at_level(logging.WARNING):
            out = await _react_as(
                h, "signal_react", "👍", channel_id=CHANNEL, trace_id=trace
            )
        assert CHANNEL in out
        assert "bad emoji from daemon" in out
        # The remedy: the error must say what to do next (AGENTS.md).
        low = out.lower()
        assert "retry" in low or "reply in text" in low or "text reply" in low
        # Logged at WARNING naming the channel.
        assert any(
            r.levelno >= logging.WARNING and CHANNEL in r.getMessage()
            for r in caplog.records
        )

    async def test_not_connected_is_returned_naming_channel_and_reason(self, h, ts, caplog):
        """With no connection the error names the channel and says not connected."""
        trace = await _inbound(h, "hello", ts=ts.first)
        await h.server.drop_connection()
        await _wait_until(lambda: h.plugin._writer is None)
        with caplog.at_level(logging.WARNING):
            out = await _react_as(
                h, "signal_react", "👍", channel_id=CHANNEL, trace_id=trace
            )
        assert CHANNEL in out
        assert "not connected" in out.lower()
        # The remedy: the error must say what to do next (AGENTS.md).
        low = out.lower()
        assert "retry" in low or "reply in text" in low or "text reply" in low
        assert any(
            r.levelno >= logging.WARNING and CHANNEL in r.getMessage()
            for r in caplog.records
        )


class TestEmojiCheck:
    """Local emoji check: empty or over 16 code points is refused before any
    frame, with the emoji named and a next step."""

    @pytest.mark.parametrize(
        "emoji",
        ["", "a" * 17],
        ids=["empty", "17-code-points"],
    )
    async def test_bad_emoji_is_refused_naming_it_and_a_next_step(self, h, ts, emoji):
        """A bad emoji is refused locally: no frame, the value named, a next step."""
        trace = await _inbound(h, "hello", ts=ts.first)
        out = await _react_as(
            h, "signal_react", emoji, channel_id=CHANNEL, trace_id=trace
        )
        # Names the rejected value (repr or raw string) and the word emoji.
        assert repr(emoji) in out or (emoji and emoji in out)
        assert "emoji" in out.lower()
        # States a remedy.
        low = out.lower()
        assert "try" in low or "reply in text" in low or "text reply" in low
        assert _frames(h) == []
