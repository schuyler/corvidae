"""`allow_bots`: a bot account's message is dropped by default.

Two corvidae agents in one channel hear nothing from each other, because every
message authored by a bot is discarded before the allowlist is consulted. The
flag opens that door for one named bot at a time: the allowlist still decides
who, the agent still never answers itself, and in a mention-only channel the
other bot still has to @-mention it.
"""

from __future__ import annotations

import pytest

from tests.test_discord_plugin import (
    AUTH_ID,
    BOT_ID,
    BOT_MENTION,
    OTHER_ID,
    _deliver,
    _delivered,
    _start_plugin,
    make_dm,
    make_message,
)

pytestmark = pytest.mark.timeout(30)


async def _deliver_bot(h, content: str, **kw) -> None:
    """A guild message from another bot that @-mentions this one."""
    await _deliver(
        h,
        make_message(
            content, author_id=OTHER_ID, bot=True, mentions=[BOT_MENTION], **kw
        ),
    )


class TestAllowBots:
    async def test_a_bot_message_is_dropped_by_default(self):
        """The default is off even for an allowed bot: opt in explicitly."""
        h = await _start_plugin(discord_overrides={"allow": [AUTH_ID, OTHER_ID]})
        try:
            await _deliver_bot(h, "from another bot")
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_an_allowed_bot_reaches_the_agent_when_the_flag_is_on(self):
        h = await _start_plugin(
            discord_overrides={"allow": [AUTH_ID, OTHER_ID], "allow_bots": True}
        )
        try:
            await _deliver_bot(h, "from another bot")
            assert _delivered(h) == ["from another bot"]
        finally:
            await h.plugin.on_stop()

    async def test_the_allowlist_still_decides_which_bot(self):
        """The flag widens who may speak, it does not widen who is trusted: a
        bot outside discord.allow reaches the gate on a transient channel, is
        vetoed there, and leaves no registry entry (#274)."""
        h = await _start_plugin(
            discord_overrides={"allow": [AUTH_ID], "allow_bots": True}
        )
        try:
            await _deliver_bot(h, "from a stranger")
            ch = h.on_message.await_args_list[0].kwargs["channel"]
            assert await h.plugin.should_process_message(
                channel=ch, sender=str(OTHER_ID), text="from a stranger",
                trace_id="tid",
            ) is False
            assert h.registry.get(ch.id) is None
        finally:
            await h.plugin.on_stop()

    async def test_a_mention_only_channel_still_wants_the_mention(self):
        """The flag is not a second require_mention: open the transport too."""
        h = await _start_plugin(
            discord_overrides={"allow": [AUTH_ID, OTHER_ID], "allow_bots": True}
        )
        try:
            await _deliver(
                h, make_message("unmentioned", author_id=OTHER_ID, bot=True)
            )
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_the_agent_never_answers_itself(self):
        h = await _start_plugin(
            discord_overrides={"allow": [BOT_ID], "allow_bots": True}
        )
        try:
            await _deliver(
                h,
                make_message(
                    "my own message", author_id=BOT_ID, bot=True,
                    mentions=[BOT_MENTION],
                ),
            )
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_a_dm_from_an_allowed_bot_reaches_the_agent(self):
        h = await _start_plugin(
            discord_overrides={"allow": [AUTH_ID, OTHER_ID], "allow_bots": True}
        )
        try:
            await _deliver(h, make_dm("a bot DM", author_id=OTHER_ID, bot=True))
            assert _delivered(h) == ["a bot DM"]
        finally:
            await h.plugin.on_stop()

    async def test_a_non_boolean_raises_naming_the_field(self):
        """`allow_bots: "false"` is a truthy string; refuse it, don't coerce."""
        with pytest.raises(ValueError, match="allow_bots"):
            await _start_plugin(
                discord_overrides={"allow_bots": "false"}, wait_connect=False
            )
