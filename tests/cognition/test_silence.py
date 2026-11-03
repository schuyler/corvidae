"""RED tests for issue #215, the cognition half: SilencePlugin (the NO_REPLY
veto, S1; the decline record, S5).

Design: .claude/plans/215-silence.md. Interface pinned here:
  corvidae_cognition.silence.NO_REPLY == "NO_REPLY"
  corvidae_cognition.silence.REGISTER_FRAGMENT: str
  corvidae_cognition.silence.SilencePlugin (entry point "silence",
      optional_depends_on = {"outcome_log"}):
      register_fragment() -> REGISTER_FRAGMENT
      should_send_response — final emission, text equal to NO_REPLY after
          stripping, any origin -> False; anything else -> None
      on_agent_response — when the newest assistant row is UNSENT: a
          NO_REPLY response records outcomes {"silence": "no_reply"}, an
          empty one {"silence": "empty"}, any other withheld final nothing;
          a sent final records nothing. Writes go through
          outcome_log.upsert_exchange from tracked tasks that on_stop awaits.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from helpers import build_plugin_and_channel, drain, sent_texts
from llm_response_fixtures import _make_text_response

from corvidae.channel import Channel
from corvidae.context import ContextWindow, MessageType, unsent_marker_emission
from corvidae.hooks import create_plugin_manager, resolve_reject_wins
from corvidae.outbound import OutboundMessage
from corvidae.trace import mint_trace_id

# ---------------------------------------------------------------------------
# The veto (S1)
# ---------------------------------------------------------------------------


class TestVeto:
    async def _verdict(self, text="NO_REPLY", emission="final", origin="user"):
        """Dispatch through the plugin manager, as the kernel's _gate_send
        does, so a method missing @hookimpl is never consulted."""
        from corvidae_cognition.silence import SilencePlugin

        pm = create_plugin_manager()
        pm.register(SilencePlugin(), name="silence")
        results = await pm.ahook.should_send_response(
            channel=Channel(transport="test", scope="veto"),
            text=text, emission=emission, trace_id="sveto", meta={"origin": origin},
        )
        return resolve_reject_wins(results)

    def test_constants(self):
        from corvidae_cognition.silence import (
            NO_REPLY,
            REGISTER_FRAGMENT,
            SilencePlugin,
        )

        assert NO_REPLY == "NO_REPLY"
        assert NO_REPLY in REGISTER_FRAGMENT
        assert SilencePlugin().register_fragment() == REGISTER_FRAGMENT

    @pytest.mark.parametrize("text", ["NO_REPLY", "  NO_REPLY\n"])
    @pytest.mark.parametrize("origin", ["user", "heartbeat", "critique"])
    async def test_marker_final_is_vetoed_for_any_origin(self, text, origin):
        assert await self._verdict(text=text, origin=origin) is False

    @pytest.mark.parametrize(
        "text, emission",
        [
            ("The deadline is Thursday.", "final"),
            ("NO_REPLY", "progress"),
            ("NO_REPLY", "thinking"),
            ("NO_REPLY", "error"),
        ],
    )
    async def test_anything_else_has_no_opinion(self, text, emission):
        assert await self._verdict(text=text, emission=emission) is None


# ---------------------------------------------------------------------------
# S1 at agent level: nothing reaches the transport, the row is UNSENT
# ---------------------------------------------------------------------------


@pytest.fixture
async def agent_env():
    """A real Agent with the real on_agent_response hook and SilencePlugin."""
    from corvidae_cognition.silence import SilencePlugin

    plugin, channel, db = await build_plugin_and_channel(mock_on_agent_response=False)
    silence = SilencePlugin()
    plugin.pm.register(silence, name="silence")
    await silence.on_init(pm=plugin.pm, config={})
    yield plugin, channel, db, silence
    await silence.on_stop()
    task_plugin = plugin.pm.get_plugin("task")
    if task_plugin:
        await task_plugin.on_stop()
    await db.close()


class TestDeclineAtAgentLevel:
    @pytest.mark.parametrize("reply", ["NO_REPLY", " NO_REPLY\n"])
    async def test_user_turn_no_reply_reaches_no_transport(self, agent_env, reply):
        plugin, channel, db, _silence = agent_env
        client = MagicMock()
        client.chat = AsyncMock(side_effect=[_make_text_response(reply)])
        plugin._client = client
        plugin._mark_started()

        await plugin.on_message(
            channel=channel, sender="user", text="thanks, night!", trace_id=mint_trace_id()
        )
        await drain(plugin, channel)

        # Nothing was sent.
        assert sent_texts(plugin.pm, OutboundMessage) == []
        # The row is stored UNSENT, with the final marker after it.
        async with db.execute(
            "SELECT message, message_type FROM message_log WHERE channel_id = ? ORDER BY id",
            (channel.id,),
        ) as cursor:
            rows = [(json.loads(m), t) for m, t in await cursor.fetchall()]
        assert [t for m, t in rows if m.get("role") == "assistant"] == ["unsent"]
        assert any(
            t == "context" and unsent_marker_emission(m["content"]) == "final" for m, t in rows
        )
        # And the window tags it, so later turns can see the agent stayed silent.
        window_assistant = [m for m in channel.conversation.messages if m.get("role") == "assistant"]
        assert window_assistant[-1]["_message_type"] == MessageType.UNSENT


# ---------------------------------------------------------------------------
# S5: every decline is recorded in exchange_log
# ---------------------------------------------------------------------------


class _StubOutcomeLog:
    """Records the writes SilencePlugin makes through upsert_exchange."""

    def __init__(self):
        self.writes: list[dict] = []

    async def upsert_exchange(self, trace_id, channel_id, origin=None, **columns):
        self.writes.append(
            {"trace_id": trace_id, "channel_id": channel_id, "origin": origin, **columns}
        )


def _channel_ending_in(final_type: MessageType) -> Channel:
    """A channel whose newest assistant row is of ``final_type``."""
    channel = Channel(transport="test", scope="s5")
    conv = ContextWindow(channel.id)
    conv.append({"role": "user", "content": "thanks, night!"})
    conv.append({"role": "assistant", "content": "x"}, final_type)
    channel.conversation = conv
    return channel


async def _respond(response_text: str, final_type: MessageType) -> list[dict]:
    """Fire on_agent_response for a final and return the recorded writes."""
    from corvidae_cognition.silence import SilencePlugin

    pm = create_plugin_manager()
    outcome_log = _StubOutcomeLog()
    pm.register(outcome_log, name="outcome_log")
    silence = SilencePlugin()
    pm.register(silence, name="silence")
    await silence.on_init(pm=pm, config={})

    # Dispatch through the manager so a method missing @hookimpl never runs.
    await pm.ahook.on_agent_response(
        channel=_channel_ending_in(final_type), request_text="thanks, night!",
        response_text=response_text, trace_id="t-s5", meta={}, logprobs=None,
    )
    # The write is a tracked fire-and-forget task; on_stop drains it.
    await silence.on_stop()
    return outcome_log.writes


class TestDeclineRecord:
    @pytest.mark.parametrize(
        "response_text, expected",
        [("NO_REPLY", "no_reply"), (" NO_REPLY\n", "no_reply"), ("", "empty")],
    )
    async def test_withheld_decline_is_recorded(self, response_text, expected):
        writes = await _respond(response_text, MessageType.UNSENT)

        assert len(writes) == 1
        assert writes[0]["trace_id"] == "t-s5"
        assert writes[0]["channel_id"] == "test:s5"
        assert writes[0]["outcomes"] == {"silence": expected}

    async def test_sent_reply_records_nothing(self):
        assert await _respond("Thursday works for me.", MessageType.MESSAGE) == []

    async def test_sent_no_reply_records_nothing(self):
        """A NO_REPLY that went out (a sibling gate failed open) is not a
        decline: its newest row is MESSAGE."""
        assert await _respond("NO_REPLY", MessageType.MESSAGE) == []

    async def test_other_withheld_final_records_nothing(self):
        """Another gate's veto is not this plugin's decline."""
        assert await _respond("a vetoed answer", MessageType.UNSENT) == []
