"""RED-phase tests for the Discord reaction tools (design: reactions-design.md).

``discord_react(emoji)`` and ``discord_unreact(emoji)`` act on the message that
triggered the current turn. The target is never a tool argument: the transport
records each admitted message under the trace id it minted, and a tool resolves
its target from ``get_attribution()``. These tests drive that map through the
public inbound path (``_enqueue`` -> ``on_message``), then call the tool under
``set_attribution`` the way a running turn would.

No network and no gateway: messages are duck-typed SimpleNamespaces whose
``add_reaction`` / ``remove_reaction`` are AsyncMocks, and the client is the
FakeClient from the transport's own test module.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# The SDK is the optional ``discord`` extra; without it this file skips (the
# kernel CI job has no Discord SDK).
discord = pytest.importorskip("discord")

from corvidae.attribution import reset_attribution, set_attribution
from corvidae.channels import discord as dmod
from corvidae.tool import tool_to_schema

from .test_discord_plugin import (
    AUTH_ID,
    BOT_ID,
    OTHER_ID,
    Harness,
    _deliver,
    _start_plugin,
    make_dm,
)

# The channel id a DM from the allowlisted user gets, and the refused user's.
ALLOWED_CHANNEL = f"discord:dm:{AUTH_ID}"
REFUSED_CHANNEL = f"discord:dm:{OTHER_ID}"


class FakeForbidden(Exception):
    """Stand-in for discord.Forbidden. One class so the plugin's isinstance
    check against ``_sdk().Forbidden`` can match it."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reactable_dm(text: str, *, author_id: int = AUTH_ID) -> SimpleNamespace:
    """A DM whose reaction methods are recorded, so a test can see which
    message received (or lost) a reaction."""
    msg = make_dm(text, author_id=author_id)
    msg.add_reaction = AsyncMock()
    msg.remove_reaction = AsyncMock()
    return msg


def _trace_of(h: Harness, text: str) -> str:
    """The trace id the transport minted for the delivered message ``text``."""
    for call in h.on_message.await_args_list:
        if call.kwargs["text"] == text:
            return call.kwargs["trace_id"]
    raise AssertionError(f"message {text!r} never reached on_message")


def _tool(h: Harness, name: str):
    """Fetch a tool through the real plugin manager's register_tools hook."""
    tools: list = []
    h.pm.hook.register_tools(tool_registry=tools)
    by_name = {t.__name__: t for t in tools}
    assert name in by_name, f"{name} was not registered; got {sorted(by_name)}"
    return by_name[name]


@contextmanager
def _turn(**attribution):
    """Run the body as a turn carrying this attribution, then restore it."""
    token = set_attribution(**attribution)
    try:
        yield
    finally:
        reset_attribution(token)


def _assert_no_message_error(out: str) -> None:
    """The R5 text: names the cause and offers the text-reply alternative."""
    low = out.lower()
    assert "no message to react to" in low
    # The cause: the turn did not start from a message, or it was forgotten.
    assert "did not start from a message" in low or "forgotten" in low
    # The remedy: answer in text instead.
    assert "text reply" in low or "reply in text" in low


# ---------------------------------------------------------------------------
# R1: add the reaction to the triggering message
# ---------------------------------------------------------------------------


class TestReact:
    async def test_reacts_on_the_triggering_message(self):
        """R1: add_reaction is awaited on the message that started the turn."""
        h = await _start_plugin()
        try:
            msg = _reactable_dm("hello")
            await _deliver(h, msg)
            react = _tool(h, "discord_react")
            with _turn(trace_id=_trace_of(h, "hello"), channel_id=ALLOWED_CHANNEL):
                out = await react("👍")
            # The SDK was asked for exactly this emoji on exactly this message.
            msg.add_reaction.assert_awaited_once_with("👍")
            assert "error" not in out.lower() and "could not" not in out.lower()
        finally:
            await h.plugin.on_stop()

    @pytest.mark.parametrize("emoji", ["", "x" * 65], ids=["empty", "over-64-chars"])
    async def test_bad_emoji_is_refused_naming_it_and_the_next_step(self, emoji):
        """R1: an empty or over-long emoji is rejected locally, naming the
        emoji and saying what to do instead; Discord is never asked."""
        h = await _start_plugin()
        try:
            msg = _reactable_dm("hello")
            await _deliver(h, msg)
            react = _tool(h, "discord_react")
            with _turn(trace_id=_trace_of(h, "hello"), channel_id=ALLOWED_CHANNEL):
                out = await react(emoji)
            msg.add_reaction.assert_not_awaited()
            # Names the rejected value (an empty one is named by its repr).
            assert repr(emoji) in out or (emoji and emoji in out)
            assert "emoji" in out.lower()
            # Says what to do next.
            low = out.lower()
            assert "try" in low or "reply in text" in low or "text reply" in low
        finally:
            await h.plugin.on_stop()

    async def test_forbidden_names_both_permissions(self, monkeypatch):
        """R1: a permission refusal names Add Reactions and Read Message History."""
        monkeypatch.setattr(
            dmod, "_sdk", lambda: SimpleNamespace(Forbidden=FakeForbidden)
        )
        h = await _start_plugin()
        try:
            msg = _reactable_dm("hello")
            msg.add_reaction.side_effect = FakeForbidden("Missing Permissions")
            await _deliver(h, msg)
            react = _tool(h, "discord_react")
            with _turn(trace_id=_trace_of(h, "hello"), channel_id=ALLOWED_CHANNEL):
                out = await react("👍")
            assert "Add Reactions" in out
            assert "Read Message History" in out
        finally:
            await h.plugin.on_stop()

    async def test_other_failure_is_caught_named_and_logged(self, caplog):
        """R1: any other exception is returned as text naming the emoji and the
        exception type, and logged at WARNING with its traceback."""
        h = await _start_plugin()
        try:
            msg = _reactable_dm("hello")
            msg.add_reaction.side_effect = RuntimeError("client is closed")
            await _deliver(h, msg)
            react = _tool(h, "discord_react")
            with caplog.at_level(logging.WARNING):
                with _turn(trace_id=_trace_of(h, "hello"), channel_id=ALLOWED_CHANNEL):
                    # Must not raise: the model reads the returned text.
                    out = await react("🫥")
            assert "🫥" in out
            assert "RuntimeError" in out
            low = out.lower()
            assert "different emoji" in low or "reply in text" in low or "text reply" in low
            # Logged for the operator, with the traceback attached.
            records = [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING and r.exc_info is not None
            ]
            assert records, "no WARNING with a traceback was logged"
            assert any(r.exc_info[0] is RuntimeError for r in records)
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R3: the target comes from turn provenance
# ---------------------------------------------------------------------------


class TestTargeting:
    async def test_schema_exposes_only_emoji(self):
        """R3: neither tool lets the model name a target; the only argument
        is ``emoji``."""
        h = await _start_plugin()
        try:
            for name in ("discord_react", "discord_unreact"):
                schema = tool_to_schema(_tool(h, name))
                params = schema["function"]["parameters"]
                assert set(params["properties"]) == {"emoji"}, name
                assert params.get("required") == ["emoji"], name
        finally:
            await h.plugin.on_stop()

    async def test_the_right_entry_of_two_is_hit(self):
        """R3: with two recorded messages, the turn on the FIRST message's
        trace reacts to the first and leaves the second alone."""
        h = await _start_plugin()
        try:
            first, second = _reactable_dm("first"), _reactable_dm("second")
            await _deliver(h, first)
            await _deliver(h, second)
            react = _tool(h, "discord_react")
            with _turn(trace_id=_trace_of(h, "first"), channel_id=ALLOWED_CHANNEL):
                await react("👍")
            first.add_reaction.assert_awaited_once_with("👍")
            second.add_reaction.assert_not_awaited()
        finally:
            await h.plugin.on_stop()

    async def test_coalesced_burst_resolves_to_the_last_message(self):
        """R3: a burst is one turn owned by the LAST message's trace, so the
        tool called under that trace hits the last message, not an earlier one."""
        h = await _start_plugin()
        try:
            msgs = [_reactable_dm(f"burst-{i}") for i in range(3)]
            await _deliver(h, *msgs)
            react = _tool(h, "discord_react")
            with _turn(trace_id=_trace_of(h, "burst-2"), channel_id=ALLOWED_CHANNEL):
                await react("🎉")
            msgs[2].add_reaction.assert_awaited_once_with("🎉")
            msgs[0].add_reaction.assert_not_awaited()
            msgs[1].add_reaction.assert_not_awaited()
        finally:
            await h.plugin.on_stop()

    async def test_entry_for_another_channel_is_treated_as_not_found(self):
        """R3: a trace recorded for one channel, carried into a turn on a
        different channel (a notification can inherit a trace), is not found."""
        h = await _start_plugin()
        try:
            msg = _reactable_dm("hello")
            await _deliver(h, msg)
            react = _tool(h, "discord_react")
            with _turn(trace_id=_trace_of(h, "hello"), channel_id="discord:999"):
                out = await react("👍")
            msg.add_reaction.assert_not_awaited()
            _assert_no_message_error(out)
        finally:
            await h.plugin.on_stop()

    async def test_refused_sender_is_not_recorded(self):
        """R3 (design): a message from a non-allowlisted sender is never held,
        so a tool called under its trace finds nothing and reacts to nothing."""
        h = await _start_plugin()
        try:
            stranger = _reactable_dm("let me in", author_id=OTHER_ID)
            await _deliver(h, stranger)
            # The refused message still reached on_message (the gate vetoes it
            # later), so its trace exists; only the trigger map must lack it.
            trace = _trace_of(h, "let me in")
            react = _tool(h, "discord_react")
            with _turn(trace_id=trace, channel_id=REFUSED_CHANNEL):
                out = await react("👍")
            stranger.add_reaction.assert_not_awaited()
            _assert_no_message_error(out)
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R4: unreact removes the bot's own reaction
# ---------------------------------------------------------------------------


class TestUnreact:
    async def test_removes_the_bots_own_reaction(self):
        """R4: remove_reaction(emoji, <the bot's user>) on the triggering message."""
        h = await _start_plugin()
        try:
            msg = _reactable_dm("hello")
            await _deliver(h, msg)
            unreact = _tool(h, "discord_unreact")
            with _turn(trace_id=_trace_of(h, "hello"), channel_id=ALLOWED_CHANNEL):
                await unreact("👍")
            # The member argument is the client's own user, not the sender.
            msg.remove_reaction.assert_awaited_once_with("👍", h.clients[0].user)
            assert h.clients[0].user.id == BOT_ID
        finally:
            await h.plugin.on_stop()

    async def test_unknown_client_user_says_not_connected_yet_retry(self):
        """R4: before the gateway is ready the client's user is None; the tool
        says it is not connected yet and to retry, and removes nothing."""
        h = await _start_plugin()
        try:
            msg = _reactable_dm("hello")
            await _deliver(h, msg)
            # The gateway has not identified the bot yet.
            h.clients[0].user = None
            unreact = _tool(h, "discord_unreact")
            with _turn(trace_id=_trace_of(h, "hello"), channel_id=ALLOWED_CHANNEL):
                out = await unreact("👍")
            msg.remove_reaction.assert_not_awaited()
            low = out.lower()
            assert "not connected yet" in low
            assert "retry" in low
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R5: a turn with no message to react to
# ---------------------------------------------------------------------------


class TestNoMessage:
    @pytest.mark.parametrize("tool_name", ["discord_react", "discord_unreact"])
    @pytest.mark.parametrize(
        "case", ["no-trace", "no-attribution", "unknown-trace", "non-discord-channel"]
    )
    async def test_returns_the_no_message_error_and_raises_nothing(
        self, case, tool_name
    ):
        """R5: reminder/heartbeat turns (no trace), a forgotten trace, and a turn
        on another transport's channel each get the same explanatory error."""
        h = await _start_plugin()
        try:
            msg = _reactable_dm("hello")
            await _deliver(h, msg)
            tool = _tool(h, tool_name)
            real_trace = _trace_of(h, "hello")
            attribution = {
                # A turn with a channel but no trace (a reminder).
                "no-trace": {"channel_id": ALLOWED_CHANNEL},
                # A turn with no provenance at all.
                "no-attribution": {},
                # A trace this transport never recorded (forgotten/evicted).
                "unknown-trace": {
                    "trace_id": "never-recorded",
                    "channel_id": ALLOWED_CHANNEL,
                },
                # A real Discord trace, but the turn is on a Signal channel.
                "non-discord-channel": {
                    "trace_id": real_trace,
                    "channel_id": "signal:abc",
                },
            }[case]
            with _turn(**attribution):
                out = await tool("👍")
            # Nothing was touched on the real message.
            msg.add_reaction.assert_not_awaited()
            msg.remove_reaction.assert_not_awaited()
            _assert_no_message_error(out)
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R6: registration only when the transport is configured
# ---------------------------------------------------------------------------


class TestRegistration:
    async def test_registered_when_configured(self):
        """R6: a configured transport offers both tools through the real
        plugin manager's register_tools hook."""
        h = await _start_plugin()
        try:
            tools: list = []
            h.pm.hook.register_tools(tool_registry=tools)
            names = {t.__name__ for t in tools}
            assert {"discord_react", "discord_unreact"} <= names
        finally:
            await h.plugin.on_stop()

    async def test_not_registered_without_a_discord_block(self):
        """R6: with no discord: block there is no Discord to react on, so the
        tools are not offered."""
        h = await _start_plugin(omit_block=True)
        try:
            tools: list = []
            h.pm.hook.register_tools(tool_registry=tools)
            names = {t.__name__ for t in tools}
            assert "discord_react" not in names
            assert "discord_unreact" not in names
        finally:
            await h.plugin.on_stop()
