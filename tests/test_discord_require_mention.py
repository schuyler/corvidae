"""Per-channel `require_mention` (#512).

One boolean for the whole transport cannot fit a bot that is a participant in
one channel and a guest in others. The transport value stays the default; a
`discord.channels.<channel id>` entry overrides it, threads inherit their
parent's entry, and a thread whose parent is not listed stays at the default
rather than silently becoming conversational.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_discord_plugin import (
    BOT_ID,
    BOT_MENTION,
    CHAN_ID,
    _deliver,
    _delivered,
    _start_plugin,
    make_message,
)

pytestmark = pytest.mark.timeout(30)


def thread_message(content: str = "in a thread", *, thread_id: int, parent_id: int):
    """A guild message in a thread: its own channel id, with a parent."""
    message = make_message(content, channel_id=thread_id)
    message.channel = SimpleNamespace(id=thread_id, parent_id=parent_id)
    return message


class TestPerChannelRequireMention:
    async def test_listed_channel_answers_unmentioned_when_the_default_is_mention(
        self,
    ):
        h = await _start_plugin(
            discord_overrides={
                "require_mention": True,
                "channels": {str(CHAN_ID): {"require_mention": False}},
            }
        )
        try:
            await _deliver(h, make_message("just chatting"))
            assert _delivered(h) == ["just chatting"]
        finally:
            await h.plugin.on_stop()

    async def test_unlisted_channel_keeps_the_transport_default(self):
        h = await _start_plugin(
            discord_overrides={
                "require_mention": True,
                "channels": {"999": {"require_mention": False}},
            }
        )
        try:
            await _deliver(h, make_message("just chatting"))
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_a_thread_inherits_its_parents_setting(self):
        parent_id = 4242
        h = await _start_plugin(
            discord_overrides={
                "require_mention": True,
                "channels": {str(parent_id): {"require_mention": False}},
            }
        )
        try:
            await _deliver(
                h, thread_message(thread_id=CHAN_ID, parent_id=parent_id)
            )
            assert _delivered(h) == ["in a thread"]
        finally:
            await h.plugin.on_stop()

    async def test_a_thread_whose_parent_is_unlisted_stays_mention_only(self):
        h = await _start_plugin(
            discord_overrides={
                "require_mention": True,
                "channels": {"4242": {"require_mention": False}},
            }
        )
        try:
            await _deliver(
                h, thread_message(thread_id=CHAN_ID, parent_id=7777)
            )
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_the_channel_setting_can_tighten_a_conversational_default(self):
        h = await _start_plugin(
            discord_overrides={
                "require_mention": False,
                "channels": {str(CHAN_ID): {"require_mention": True}},
            }
        )
        try:
            # One delivery: a second `_deliver` would return as soon as the
            # first call's sentinel was still in the texts.
            await _deliver(
                h,
                make_message("just chatting"),
                make_message(f"<@{BOT_ID}> hi", mentions=[BOT_MENTION]),
            )
            assert _delivered(h) == ["hi"]
        finally:
            await h.plugin.on_stop()

    async def test_a_dm_is_unaffected_by_a_channel_entry(self):
        """An allowed user's DM is answered, whatever the channel entry says."""
        h = await _start_plugin(
            discord_overrides={
                "require_mention": True,
                "channels": {str(CHAN_ID): {"require_mention": True}},
            }
        )
        try:
            await _deliver(h, make_message("dm hi", guild=False))
            assert _delivered(h) == ["dm hi"]
        finally:
            await h.plugin.on_stop()

    async def test_message_content_intent_is_requested_for_any_conversational_channel(
        self,
    ):
        """The intent is per client, not per channel: one conversational channel
        means the privileged intent is needed, and the docs say so."""
        h = await _start_plugin(
            discord_overrides={
                "require_mention": True,
                "channels": {str(CHAN_ID): {"require_mention": False}},
            }
        )
        try:
            # The class's own factory: the instance method is faked in tests.
            from corvidae.channels.discord import DiscordPlugin

            assert (
                DiscordPlugin._make_client(h.plugin).intents.message_content
                is True
            )
        finally:
            await h.plugin.on_stop()


class TestConfigValidation:
    async def test_channels_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="channels") as excinfo:
            await _start_plugin(
                discord_overrides={"channels": ["1568093636044787772"]},
                wait_connect=False,
            )
        assert "discord" in str(excinfo.value)

    async def test_each_channel_entry_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="channels") as excinfo:
            await _start_plugin(
                discord_overrides={"channels": {str(CHAN_ID): False}},
                wait_connect=False,
            )
        assert str(CHAN_ID) in str(excinfo.value)

    async def test_the_override_must_be_a_bool(self):
        with pytest.raises(ValueError, match="require_mention") as excinfo:
            await _start_plugin(
                discord_overrides={
                    "channels": {str(CHAN_ID): {"require_mention": "false"}}
                },
                wait_connect=False,
            )
        assert str(CHAN_ID) in str(excinfo.value)

    async def test_an_unknown_key_is_named_rather_than_ignored(self):
        with pytest.raises(ValueError, match="requre_mention") as excinfo:
            await _start_plugin(
                discord_overrides={
                    "channels": {str(CHAN_ID): {"requre_mention": False}}
                },
                wait_connect=False,
            )
        assert str(CHAN_ID) in str(excinfo.value)
