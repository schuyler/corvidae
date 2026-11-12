"""R8: Signal reaction events delivered as waiting CONTEXT rows.

A reaction (a dataMessage carrying a ``reaction`` object and no ``message``)
from an allowlisted sender must reach the agent's context as a framed
``[reaction: ...]`` row at the tail of that sender's channel, appended by
``before_agent_turn``, without starting a turn (no on_message, no on_notify).

Everything runs through the real plugin manager and a real SignalPlugin
against the fake daemon from tests/test_signal_plugin.py. The plugin manager
has no Agent, so there is no queue; "no turn" is a spy on the two hooks that
would start one.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import hookimpl
from tests.test_signal_plugin import (
    AUTH_ACI,
    BOT_ACI,
    OTHER_ACI,
    Harness,
    _start_plugin,
    _stop_plugin,
    make_data_envelope,
)

# A fixed target-message time: 2023-11-14 22:13:20 UTC, rendered to the minute.
TARGET_TS = 1_700_000_000_000
TARGET_TEXT = "2023-11-14 22:13 UTC"


class _TurnSpy:
    """Records on_notify calls; a reaction must never start a turn."""

    def __init__(self) -> None:
        # One entry per on_notify the plugin manager delivered.
        self.notifies: list[dict] = []

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, annotations):
        # Capture the call so a test can assert there was none.
        self.notifies.append({"channel": channel, "source": source, "text": text})


async def _start(tmp_path) -> tuple[Harness, _TurnSpy]:
    """Start the plugin with a notify spy registered on the real pm."""
    h = await _start_plugin(tmp_path)
    spy = _TurnSpy()
    h.pm.register(spy, name="turn_spy")
    return h, spy


def _reaction_params(
    *,
    sender: str | None = AUTH_ACI,
    reaction: object = None,
    emoji: object = "👍",
    target_author: object = "ABSENT",
    target_ts: object = TARGET_TS,
    is_remove: bool = False,
    text: str | None = None,
    timestamp: int = TARGET_TS + 5_000,
) -> dict:
    """Build a 'receive' notification carrying a reaction dataMessage."""
    params = make_data_envelope(source_uuid=sender, text=text, timestamp=timestamp)
    if reaction is None:
        # Default shape signal-cli emits for an added reaction.
        reaction = {
            "emoji": emoji,
            "targetSentTimestamp": target_ts,
            "isRemove": is_remove,
        }
        if target_author != "ABSENT":
            reaction["targetAuthorUuid"] = target_author
    params["envelope"]["dataMessage"]["reaction"] = reaction
    return params


def _channel_with_conversation(h: Harness, sender: str = AUTH_ACI):
    """The sender's channel, with a conversation set by hand (no Agent here)."""
    channel = h.registry.get_or_create("signal", sender)
    channel.conversation = ContextWindow(channel.id)
    return channel


async def _drain(h: Harness, channel) -> list[dict]:
    """Fire before_agent_turn as the agent does and return the CONTEXT rows."""
    await h.pm.ahook.before_agent_turn(channel=channel, trace_id="t-1", annotations={})
    return [
        m for m in channel.conversation.messages
        if m.get("_message_type") == MessageType.CONTEXT
    ]


def _no_turn_started(h: Harness, spy: _TurnSpy) -> bool:
    """True when neither on_message nor on_notify was called."""
    return h.pm.ahook.on_message.await_count == 0 and not spy.notifies


def _minute_text(ts_ms: int) -> str:
    """Independent rendering of a ms epoch as YYYY-MM-DD HH:MM UTC."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# (1) allowlisted reaction -> waiting context row, no turn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target_author", "whose"),
    [
        (BOT_ACI, "your"),
        (AUTH_ACI, "their own"),
        ("ABSENT", "a"),
        # An explicit JSON null (key present, value None) reads like an absent key.
        (None, "a"),
    ],
)
async def test_allowlisted_reaction_becomes_context_row_without_a_turn(
    tmp_path, target_author, whose
):
    """The row text is built in code from emoji, whose message and its time."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        await h.plugin._handle_notification(
            "receive", _reaction_params(emoji="👍", target_author=target_author)
        )
        # A reaction waits as context; it must not start a turn.
        assert _no_turn_started(h, spy)
        rows = await _drain(h, channel)
        assert len(rows) == 1
        assert rows[0]["role"] == "system"
        assert rows[0]["content"] == f"[reaction: 👍 on {whose} message from {TARGET_TEXT}]"
    finally:
        await _stop_plugin(h)


# ---------------------------------------------------------------------------
# (2) per-channel scoping and drain-once
# ---------------------------------------------------------------------------


async def test_reaction_is_not_drained_for_another_channel(tmp_path):
    """before_agent_turn on a different channel adds nothing; the owner still gets it."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        other = _channel_with_conversation(h, OTHER_ACI)
        await h.plugin._handle_notification("receive", _reaction_params(emoji="❤️"))
        # The other channel's turn must not see the sender's reaction.
        assert await _drain(h, other) == []
        # The sender's own channel still holds it, untouched by that drain.
        rows = await _drain(h, channel)
        assert [r["content"] for r in rows] == [
            f"[reaction: ❤️ on a message from {TARGET_TEXT}]"
        ]
    finally:
        await _stop_plugin(h)


async def test_before_agent_turn_drains_the_list_once(tmp_path):
    """A second before_agent_turn adds no duplicate row."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        await h.plugin._handle_notification("receive", _reaction_params())
        first = await _drain(h, channel)
        second = await _drain(h, channel)
        assert len(first) == 1
        # Same single row after the second call: nothing was re-appended.
        assert len(second) == 1
    finally:
        await _stop_plugin(h)


# ---------------------------------------------------------------------------
# (3) non-allowlisted sender: nothing, one INFO line, no channel
# ---------------------------------------------------------------------------


async def test_refused_sender_reaction_logs_one_info_and_creates_nothing(tmp_path, caplog):
    """A sender off the allowlist leaves one INFO line naming them and no state."""
    h, spy = await _start(tmp_path)
    try:
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification(
                "receive", _reaction_params(sender=OTHER_ACI)
            )
        infos = [
            r for r in caplog.records
            if r.levelno == logging.INFO and OTHER_ACI in r.getMessage()
        ]
        assert len(infos) == 1, [r.getMessage() for r in caplog.records]
        # The refusal must not mint a channel for the stranger (PRD 3.2.6).
        assert h.registry.get(f"signal:{OTHER_ACI}") is None
        assert _no_turn_started(h, spy)
        # And nothing worse than INFO about it.
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    finally:
        await _stop_plugin(h)


# ---------------------------------------------------------------------------
# (4) removals are dropped
# ---------------------------------------------------------------------------


async def test_reaction_removal_is_dropped(tmp_path, caplog):
    """isRemove true produces no context row and no turn, and a DEBUG line says why."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification("receive", _reaction_params(is_remove=True))
        # The drop is deliberate, so it is logged at DEBUG (design step 2). Only a
        # mention of the removal distinguishes it from the existing "no content" line.
        debug = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.DEBUG and r.name == "corvidae.signal_plugin"
        ]
        assert any("remov" in m.lower() for m in debug), debug
        assert await _drain(h, channel) == []
        assert _no_turn_started(h, spy)
    finally:
        await _stop_plugin(h)


# ---------------------------------------------------------------------------
# (5) the emoji validator
# ---------------------------------------------------------------------------

TAGS_16 = "".join(chr(0xE0020 + i) for i in range(16))

ACCEPTED_EMOJI = [
    "👍",
    "❤️",
    "👍🏽",
    "👍🏽️",
    "‼️",
    "ℹ️",
    "\U0001fae9",
]

REJECTED_EMOJI = [
    pytest.param("ignore", id="word"),
    pytest.param("^^^^", id="carets"),
    pytest.param("###", id="hashes"),
    pytest.param("ⓘⓖⓝ", id="enclosed-letters"),
    pytest.param(TAGS_16, id="sixteen-tag-chars"),
    pytest.param("\U0001f3f4" + TAGS_16 + "\U000e007f", id="tag-sequence-flag-payload"),
    pytest.param("ⓘ‍ⓖ‍ⓝ", id="enclosed-letters-zwj"),
    pytest.param("👍\n", id="trailing-newline"),
    pytest.param("\U0001f1ef\U0001f1f5", id="regional-indicator-pair"),
    pytest.param("1️⃣", id="keycap"),
    pytest.param("\U0001f468‍\U0001f469‍\U0001f467", id="zwj-family"),
    pytest.param("\U0001f3fd", id="lone-skin-tone"),
    pytest.param("\U0001f3fd\U0001f3fd", id="modifier-modifier"),
]


@pytest.mark.parametrize("emoji", ACCEPTED_EMOJI)
def test_is_plain_emoji_accepts_single_symbols(emoji):
    """A single pictograph, optional skin tone, optional VS16 passes."""
    # Imported here so the file collects before the predicate exists.
    from corvidae.channels.signal import _is_plain_emoji

    assert _is_plain_emoji(emoji) is True, repr(emoji)


@pytest.mark.parametrize("emoji", REJECTED_EMOJI)
def test_is_plain_emoji_rejects_everything_else(emoji):
    """Words, symbols, tag payloads, compounds and stray modifiers fail."""
    from corvidae.channels.signal import _is_plain_emoji

    assert _is_plain_emoji(emoji) is False, repr(emoji)


@pytest.mark.parametrize("emoji", REJECTED_EMOJI)
async def test_rejected_emoji_warns_naming_sender_and_adds_no_row(tmp_path, caplog, emoji):
    """End to end: a rejected value is dropped with a WARNING carrying the sender and repr."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification("receive", _reaction_params(emoji=emoji))
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert AUTH_ACI in warnings[0]
        assert repr(emoji[:32]) in warnings[0]
        assert await _drain(h, channel) == []
        assert _no_turn_started(h, spy)
    finally:
        await _stop_plugin(h)


async def test_rejected_emoji_is_truncated_to_32_chars_in_the_warning(tmp_path, caplog):
    """An oversized value is logged cut to 32 characters, not in full."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        oversized = "x" * 100
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification("receive", _reaction_params(emoji=oversized))
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert AUTH_ACI in warnings[0]
        assert repr("x" * 32) in warnings[0]
        # The 33rd character must not leak into the log line.
        assert "x" * 33 not in warnings[0]
        assert await _drain(h, channel) == []
    finally:
        await _stop_plugin(h)


# ---------------------------------------------------------------------------
# (6) fix-now edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_ts",
    [
        pytest.param("1700000000000", id="string"),
        pytest.param(True, id="bool"),
        pytest.param(0, id="zero"),
        pytest.param(10**18, id="beyond-year-9999"),
    ],
)
async def test_bad_target_timestamp_is_dropped_with_warning(tmp_path, caplog, bad_ts):
    """An implausible targetSentTimestamp drops the reaction and names the sender."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification("receive", _reaction_params(target_ts=bad_ts))
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert AUTH_ACI in warnings[0]
        assert await _drain(h, channel) == []
        assert _no_turn_started(h, spy)
    finally:
        await _stop_plugin(h)


@pytest.mark.parametrize(
    "emoji",
    [pytest.param(123, id="int"), pytest.param(["👍"], id="list")],
)
async def test_non_string_emoji_is_dropped_with_warning(tmp_path, caplog, emoji):
    """An emoji that is not a string is dropped, naming the sender."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification("receive", _reaction_params(emoji=emoji))
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert AUTH_ACI in warnings[0]
        assert await _drain(h, channel) == []
    finally:
        await _stop_plugin(h)


@pytest.mark.parametrize(
    "reaction",
    [pytest.param("👍", id="string"), pytest.param(["👍"], id="list"), pytest.param(7, id="int")],
)
async def test_non_object_reaction_is_dropped_with_warning(tmp_path, caplog, reaction):
    """A reaction that is not an object is dropped, naming the sender."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification(
                "receive", _reaction_params(reaction=reaction)
            )
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert AUTH_ACI in warnings[0]
        assert await _drain(h, channel) == []
        assert _no_turn_started(h, spy)
    finally:
        await _stop_plugin(h)


async def test_pending_overflow_drops_oldest_with_warning_naming_channel(tmp_path, caplog):
    """The 21st waiting reaction evicts the oldest and warns, naming the channel."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        # Distinct target minutes so each row's text identifies its reaction.
        stamps = [TARGET_TS + i * 60_000 for i in range(21)]
        with caplog.at_level(logging.DEBUG):
            for ts in stamps:
                await h.plugin._handle_notification("receive", _reaction_params(target_ts=ts))
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert channel.id in warnings[0]
        rows = await _drain(h, channel)
        contents = [r["content"] for r in rows]
        assert len(contents) == 20
        # The oldest (index 0) is gone; the newest 20 remain, oldest first.
        assert contents == [
            f"[reaction: 👍 on a message from {_minute_text(ts)}]" for ts in stamps[1:]
        ]
    finally:
        await _stop_plugin(h)


async def test_reaction_without_source_uuid_is_dropped_with_warning(tmp_path, caplog):
    """No sourceUuid means no channel can be scoped: WARNING, nothing queued."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification("receive", _reaction_params(sender=None))
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert "sourceUuid" in warnings[0]
        assert await _drain(h, channel) == []
        assert _no_turn_started(h, spy)
    finally:
        await _stop_plugin(h)


# ---------------------------------------------------------------------------
# (7) a null reaction with real text is an ordinary message
# ---------------------------------------------------------------------------


async def test_null_reaction_with_text_is_delivered_as_a_message(tmp_path, caplog):
    """reaction: null must fall through to the text path unchanged."""
    h, spy = await _start(tmp_path)
    try:
        channel = _channel_with_conversation(h)
        params = make_data_envelope(source_uuid=AUTH_ACI, text="hello there")
        params["envelope"]["dataMessage"]["reaction"] = None
        with caplog.at_level(logging.DEBUG):
            await h.plugin._handle_notification("receive", params)
        # The text reached on_message untouched, as today.
        assert h.pm.ahook.on_message.await_count == 1
        assert h.pm.ahook.on_message.await_args.kwargs["text"] == "hello there"
        assert h.pm.ahook.on_message.await_args.kwargs["sender"] == AUTH_ACI
        # No reaction row, and no warning about a reaction.
        assert await _drain(h, channel) == []
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    finally:
        await _stop_plugin(h)
