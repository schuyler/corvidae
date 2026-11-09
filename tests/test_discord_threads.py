"""The transport's thread tool (#593).

A thread is a channel in its own right (#513), so starting one has to leave the
registry saying so: a later send to that scope must find it. These tests drive
the real ``start_thread`` against duck-typed Discord objects, the way the rest of
the transport's tests do.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from corvidae.channels import discord as dmod

from .test_discord_plugin import CHAN_ID, FakeMessageable, _start_plugin


class FakeThread:
    """A duck-typed discord.Thread: an id, a title, a parent, and its own sends."""

    def __init__(self, thread_id: int, name: str, parent=None) -> None:
        self.id = thread_id
        self.name = name
        self.parent = parent
        self.sends: list[tuple[str, dict]] = []

    refuse_send_with: Exception | None = None

    async def send(self, content, **kwargs):
        if self.refuse_send_with is not None:
            raise self.refuse_send_with
        self.sends.append((content, kwargs))


class FakeParent(FakeMessageable):
    """A messageable that can also create a thread of its own."""

    def __init__(self, name: str = "home") -> None:
        super().__init__()
        self.name = name
        self.created: list[str] = []
        self.thread: FakeThread | None = None
        self.refuse_with: Exception | None = None
        #: Applied to the thread the channel creates, to fail its first message.
        self.thread_refuse_with: Exception | None = None

    async def create_thread(self, *, name: str, **kwargs) -> FakeThread:
        if self.refuse_with is not None:
            raise self.refuse_with
        self.created.append(name)
        self.thread = FakeThread(9001, name, parent=self)
        self.thread.refuse_send_with = self.thread_refuse_with
        return self.thread

    async def fetch_message(self, message_id: int):
        return FakeSourceMessage(message_id, self)


class FakeSourceMessage:
    """A message a thread can hang off."""

    def __init__(self, message_id: int, channel: FakeParent) -> None:
        self.id = message_id
        self.channel = channel
        self.created: list[str] = []

    async def create_thread(self, *, name: str, **kwargs) -> FakeThread:
        self.created.append(name)
        return FakeThread(9002, name, parent=self.channel)


class Forbidden(Exception):
    """The SDK's permission refusal. One class, because the plugin does an
    ``isinstance`` check and a fresh class per call would never match."""


class ForumChannel:
    """The SDK's forum type. Present so the forum case is a value, not a guess."""


_SDK = SimpleNamespace(
    AllowedMentions=SimpleNamespace(none=lambda: "none"),
    Forbidden=Forbidden,
    ForumChannel=ForumChannel,
)


def _sdk():
    return _SDK


class TestStartThread:
    async def _ready(self, monkeypatch):
        monkeypatch.setattr(dmod, "_sdk", _sdk)
        h = await _start_plugin()
        channel = h.registry.get_or_create("discord", str(CHAN_ID))
        target = FakeParent()
        h.plugin._messageable = AsyncMock(return_value=target)
        return h, channel, target

    async def test_starts_a_standalone_thread_and_registers_it(self, monkeypatch):
        h, channel, target = await self._ready(monkeypatch)
        try:
            out = await h.plugin.start_thread(channel.id, "Deploy notes")
            assert "Started thread discord:9001" in out
            assert target.created == ["Deploy notes"]
            # A later send to that scope must find the thread, not a fresh
            # channel with no name and no history.
            thread_channel = h.registry.get("discord:9001")
            assert thread_channel is not None
            assert thread_channel.name == "thread Deploy notes in #home"
        finally:
            await h.plugin.on_stop()

    async def test_hangs_the_thread_off_a_named_message(self, monkeypatch):
        h, channel, target = await self._ready(monkeypatch)
        try:
            out = await h.plugin.start_thread(
                channel.id, "About this", from_message_id="4242"
            )
            assert "Started thread discord:9002" in out
            assert h.registry.get("discord:9002") is not None
            # The channel's own create_thread is not the path taken.
            assert target.created == []
        finally:
            await h.plugin.on_stop()

    async def test_posts_the_first_message_into_the_thread(self, monkeypatch):
        h, channel, target = await self._ready(monkeypatch)
        try:
            await h.plugin.start_thread(channel.id, "Deploy notes", text="step 3")
            assert target.thread is not None
            assert target.thread.sends[0][0] == "step 3"
            # Generated text must not ping anyone, the first message included.
            assert target.thread.sends[0][1]["allowed_mentions"] == "none"
        finally:
            await h.plugin.on_stop()

    async def test_a_name_over_the_limit_is_refused_before_discord_sees_it(
        self, monkeypatch
    ):
        h, channel, target = await self._ready(monkeypatch)
        try:
            out = await h.plugin.start_thread(channel.id, "x" * 101)
            assert "over Discord's 100 limit" in out
            assert target.created == []
        finally:
            await h.plugin.on_stop()

    async def test_an_empty_name_is_refused(self, monkeypatch):
        h, channel, target = await self._ready(monkeypatch)
        try:
            out = await h.plugin.start_thread(channel.id, "   ")
            assert "name must be a non-empty string" in out
            assert target.created == []
        finally:
            await h.plugin.on_stop()

    async def test_a_refused_permission_names_the_permission(self, monkeypatch):
        h, channel, target = await self._ready(monkeypatch)
        try:
            target.refuse_with = _sdk().Forbidden("Missing Permissions")
            out = await h.plugin.start_thread(channel.id, "Deploy notes")
            assert "Create Public Threads" in out
            assert h.registry.get("discord:9001") is None
        finally:
            await h.plugin.on_stop()

    async def test_a_dm_has_no_threads(self, monkeypatch):
        h, _, target = await self._ready(monkeypatch)
        try:
            dm = h.registry.get_or_create("discord", "dm:77")
            out = await h.plugin.start_thread(dm.id, "Deploy notes")
            assert "DMs have no threads" in out
            assert target.created == []
        finally:
            await h.plugin.on_stop()

    async def test_a_non_discord_channel_is_refused(self, monkeypatch):
        h, _, _ = await self._ready(monkeypatch)
        try:
            out = await h.plugin.start_thread("signal:abc", "Deploy notes")
            assert "not a Discord channel" in out
        finally:
            await h.plugin.on_stop()

    async def test_a_failed_first_message_still_names_the_thread(self, monkeypatch):
        h, channel, target = await self._ready(monkeypatch)
        try:
            target.thread_refuse_with = Forbidden("Missing Permissions")
            out = await h.plugin.start_thread(channel.id, "Deploy notes", text="step 3")
            # The thread exists and is registered, so the message must say which
            # one: a retry would otherwise create a second thread.
            assert "Started thread discord:9001" in out
            assert "could not post in it" in out
            assert h.registry.get("discord:9001") is not None
        finally:
            await h.plugin.on_stop()

    async def test_a_forum_is_refused_not_mistaken_for_a_text_channel(self, monkeypatch):
        h, channel, _ = await self._ready(monkeypatch)

        class Forum(ForumChannel):
            async def create_thread(self, *, name, **kwargs):
                raise AssertionError("a forum takes posts, not threads")

        h.plugin._messageable = AsyncMock(return_value=Forum())
        try:
            out = await h.plugin.start_thread(channel.id, "Deploy notes")
            assert "is a forum" in out
            assert h.registry.get("discord:9001") is None
        finally:
            await h.plugin.on_stop()

    async def test_the_tool_is_registered(self):
        h = await _start_plugin()
        try:
            tools: list = []
            h.plugin.register_tools(tools)
            assert h.plugin.start_thread in tools
        finally:
            await h.plugin.on_stop()
