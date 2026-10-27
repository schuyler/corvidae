"""Tests for corvidae.channels.discord.DiscordPlugin (RED phase, issue #80).

The Discord transport does not exist yet -- the import below is expected to
fail until the green phase creates corvidae/channels/discord.py. The design
is .claude/plans/discord-transport.md (requirements R1-R17).

No network and no discord.py gateway: the plugin is driven through small
fakes.

  - FakeClient stands in for the ``_DiscordClient`` the plugin builds in
    ``DiscordPlugin._make_client()``. Its ``start(token)`` raises a chosen
    exception, returns, or blocks until ``close()``; ``close()`` records calls;
    ``.user.id`` is the bot's id.
  - Inbound traffic is injected through the plugin's synchronous
    ``_enqueue(message, bot_user_id)`` entry point (what the client's
    ``on_message`` calls) with SimpleNamespace messages (duck typed: ``guild``
    is None for DMs, thread channels just have their own id).
  - Outbound traffic goes to a FakeMessageable returned by a patched
    ``DiscordPlugin._messageable(channel)``; the real ``_messageable`` is
    exercised separately against a fake client with ``get_channel`` /
    ``fetch_channel`` / ``get_user`` / ``fetch_user`` / ``create_dm``.

Names this file assumes the implementation exposes (from the design and the
Signal sibling): module constants ``TYPING_REFRESH_SECONDS``,
``_FENCE_RESERVE``, ``_INITIAL_BACKOFF``, ``_BACKOFF_CAP`` (ladder 10 / x2 /
300); ``DiscordPlugin._connect_task``; ``DiscordPlugin._client``;
``DiscordPlugin._make_client()`` (no args); the ``asyncio`` module imported
as ``asyncio`` in the plugin module (backoff sleeps are observed by patching
``corvidae.channels.discord.asyncio.sleep``).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import tomllib
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord
import pytest

from corvidae.channel import ChannelRegistry, load_channel_config
from corvidae.hooks import (
    _check_hook_arg_binding,
    create_plugin_manager,
    resolve_reject_wins,
)

# Expected to fail at import until corvidae/channels/discord.py exists.
import corvidae.channels.discord as dmod
from corvidae.channels.discord import (
    DiscordPlugin,
    _FENCE_RESERVE,
    _split_discord_segments,
    split_discord_message,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_DEFAULTS = {
    "system_prompt": "You are a test assistant.",
    "max_context_tokens": 8000,
    "reasoning_history": "drop",
}

TOKEN = "SECRET-TOKEN-abc123.xyz"
BOT_ID = 900000000000000001
AUTH_ID = 111111111111111111
OTHER_ID = 222222222222222222
CHAN_ID = 333333333333333333
THREAD_ID = 444444444444444444
DM_CHAN_ID = 555555555555555555

BOT_MENTION = SimpleNamespace(id=BOT_ID)
OTHER_MENTION = SimpleNamespace(id=OTHER_ID)

TYPING_REFRESH_FOR_TESTS = 0.05

_message_ids = itertools.count(1)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_message(
    content: str = "hello",
    *,
    author_id: int = AUTH_ID,
    bot: bool = False,
    guild: bool = True,
    channel_id: int = CHAN_ID,
    mentions: list | None = None,
    mention_everyone: bool = False,
    attachments: list | None = None,
) -> SimpleNamespace:
    """Build a duck-typed discord.Message. ``guild=False`` makes it a DM."""
    return SimpleNamespace(
        id=next(_message_ids),
        author=SimpleNamespace(id=author_id, bot=bot),
        content=content,
        channel=SimpleNamespace(id=channel_id),
        guild=SimpleNamespace(id=1) if guild else None,
        mentions=list(mentions or []),
        mention_everyone=mention_everyone,
        attachments=list(attachments or []),
    )


def make_dm(content: str = "hello", *, author_id: int = AUTH_ID, **kw) -> SimpleNamespace:
    """A DM message: no guild, channel id unrelated to the author id."""
    return make_message(content, author_id=author_id, guild=False, channel_id=DM_CHAN_ID, **kw)


class FakeTyping:
    """Stand-in for discord's Typing: counts when awaited or entered."""

    def __init__(self, owner: "FakeMessageable") -> None:
        self.owner = owner

    def __await__(self):
        # Awaiting counts as one typing indicator sent.
        async def _go() -> None:
            self.owner.typing_count += 1
        return _go().__await__()

    async def __aenter__(self) -> None:
        self.owner.typing_count += 1

    async def __aexit__(self, *exc) -> None:
        return None


class FakeMessageable:
    """Records every send (content + kwargs) and every typing indicator."""

    def __init__(self) -> None:
        self.sends: list[tuple[str | None, dict]] = []
        self.typing_count = 0
        self.fail_with: Exception | None = None

    async def send(self, *args, **kwargs):
        """Record a send; optionally fail to exercise the error path."""
        if self.fail_with is not None:
            raise self.fail_with
        content = args[0] if args else kwargs.get("content")
        self.sends.append((content, kwargs))

    def typing(self) -> FakeTyping:
        return FakeTyping(self)

    @property
    def contents(self) -> list[str]:
        return [c for c, _ in self.sends]


class FakeClient:
    """Stand-in for the discord client the connection loop drives.

    ``behavior`` is an Exception (raised from start), the string "return"
    (start returns immediately), an async callable taking the client (run
    inside start; may raise) or None (start blocks until close()).
    """

    def __init__(self, behavior=None) -> None:
        self.behavior = behavior
        self.user = SimpleNamespace(id=BOT_ID)
        self.started_with: list[str] = []
        self.close_calls = 0
        self._closed = asyncio.Event()

    async def start(self, token: str, *args, **kwargs) -> None:
        self.started_with.append(token)
        if isinstance(self.behavior, BaseException):
            raise self.behavior
        if self.behavior == "return":
            return
        if callable(self.behavior):
            # A scripted coroutine function: run it (it may raise).
            await self.behavior(self)
            return
        await self._closed.wait()

    async def close(self) -> None:
        self.close_calls += 1
        self._closed.set()


class LookupClient:
    """Fake client for the real ``_messageable``: cached + remote lookups."""

    def __init__(self) -> None:
        self.cached_channels: dict[int, object] = {}
        self.remote_channels: dict[int, object] = {}
        self.cached_users: dict[int, object] = {}
        self.remote_users: dict[int, object] = {}
        self.fail_remote: Exception | None = None
        self.calls: list[tuple[str, int]] = []
        self.user = SimpleNamespace(id=BOT_ID)

    def get_channel(self, channel_id: int):
        self.calls.append(("get_channel", channel_id))
        return self.cached_channels.get(channel_id)

    async def fetch_channel(self, channel_id: int):
        self.calls.append(("fetch_channel", channel_id))
        if self.fail_remote is not None:
            raise self.fail_remote
        return self.remote_channels[channel_id]

    def get_user(self, user_id: int):
        self.calls.append(("get_user", user_id))
        return self.cached_users.get(user_id)

    async def fetch_user(self, user_id: int):
        self.calls.append(("fetch_user", user_id))
        if self.fail_remote is not None:
            raise self.fail_remote
        return self.remote_users[user_id]

    async def close(self) -> None:
        return None


def make_user(dm: FakeMessageable) -> SimpleNamespace:
    """A fake discord User whose create_dm() yields the given messageable."""
    async def create_dm():
        return dm
    return SimpleNamespace(create_dm=create_dm)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_text(record: logging.LogRecord) -> str:
    """A log record's message plus any exception text, for substring checks."""
    text = record.getMessage()
    if record.exc_info:
        text += " " + "".join(traceback.format_exception(*record.exc_info))
    return text


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    """Poll predicate() until truthy, or raise asyncio.TimeoutError."""
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(interval)
    await asyncio.wait_for(_poll(), timeout=timeout)


async def _drive_gate(pm, *, channel, sender: str, text: str, trace_id: str = "tid") -> bool | None:
    """Run the message gate the way Agent.on_message does (reject-wins, then
    on_message_rejected on a veto)."""
    results = await pm.ahook.should_process_message(
        channel=channel, sender=sender, text=text, trace_id=trace_id,
    )
    decision = resolve_reject_wins(results)
    if decision is False:
        await pm.ahook.on_message_rejected(
            channel=channel, trace_id=trace_id, sender=sender, text=text,
        )
    return decision


@dataclass
class Harness:
    pm: object
    registry: ChannelRegistry
    plugin: DiscordPlugin
    config: dict
    clients: list[FakeClient] = field(default_factory=list)
    messageable: FakeMessageable = field(default_factory=FakeMessageable)
    messageable_requests: list[str] = field(default_factory=list)

    @property
    def on_message(self) -> AsyncMock:
        return self.pm.ahook.on_message

    def texts(self) -> list[str]:
        """Texts of every on_message call so far, in order."""
        return [c.kwargs["text"] for c in self.on_message.await_args_list]


async def _start_plugin(
    *,
    discord_overrides: dict | None = None,
    omit_block: bool = False,
    behaviors: list | None = None,
    wait_connect: bool = True,
    patch_messageable: bool = True,
    extra_config: dict | None = None,
) -> Harness:
    """Build a real pm + registry + DiscordPlugin wired to FakeClients.

    ``behaviors`` is consumed one per client the plugin builds; once it is
    exhausted further clients block until closed. Callers stop the plugin
    with ``await h.plugin.on_stop()`` in a finally block.
    """
    pm = create_plugin_manager()
    registry = ChannelRegistry(AGENT_DEFAULTS)
    pm.register(registry, name="registry")
    pm.ahook.on_message = AsyncMock()

    config: dict = dict(extra_config or {})
    if not omit_block:
        block: dict = {"token": TOKEN, "allow": [AUTH_ID]}
        block.update(discord_overrides or {})
        config["discord"] = block

    # Mirror Runtime: per-channel config is registered before on_start.
    if "channels" in config:
        load_channel_config(config, registry)

    plugin = DiscordPlugin(pm)
    pm.register(plugin, name="discord")
    h = Harness(pm=pm, registry=registry, plugin=plugin, config=config)

    queue = list(behaviors or [])

    def make_client(*_a, **_kw) -> FakeClient:
        # Each construction pops the next scripted behavior (default: block).
        client = FakeClient(queue.pop(0) if queue else None)
        h.clients.append(client)
        return client

    plugin._make_client = make_client

    if patch_messageable:
        async def fake_messageable(channel):
            h.messageable_requests.append(channel.id)
            return h.messageable
        plugin._messageable = fake_messageable

    await plugin.on_init(pm=pm, config=config)
    await plugin.on_start(config=config)
    if wait_connect and not omit_block:
        await _wait_until(lambda: h.clients and h.clients[0].started_with)
    return h


async def _deliver(h: Harness, *messages, bot_id: int = BOT_ID) -> None:
    """Enqueue messages, then a sentinel, and wait until the sentinel has
    reached on_message -- i.e. every earlier message has been fully
    processed (dropped or delivered), in order."""
    for m in messages:
        h.plugin._enqueue(m, bot_id)
    h.plugin._enqueue(make_dm("__sentinel__"), bot_id)
    await _wait_until(lambda: "__sentinel__" in h.texts())


def _delivered(h: Harness) -> list[str]:
    """on_message texts excluding the sentinel."""
    return [t for t in h.texts() if t != "__sentinel__"]


@pytest.fixture
def fast_typing(monkeypatch):
    """Shorten the typing refresh interval (module constant, like Signal)."""
    monkeypatch.setattr(dmod, "TYPING_REFRESH_SECONDS", TYPING_REFRESH_FOR_TESTS)


@pytest.fixture
def fast_backoff(monkeypatch):
    """Shrink the reconnect ladder so retry tests run in milliseconds."""
    monkeypatch.setattr(dmod, "_INITIAL_BACKOFF", 0.01)
    monkeypatch.setattr(dmod, "_BACKOFF_CAP", 0.05)


@pytest.fixture(autouse=True)
def no_ambient_token(monkeypatch):
    """Tests must not inherit a real DISCORD_TOKEN from the environment."""
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# R2, R3: configuration
# ---------------------------------------------------------------------------


class TestConfig:
    async def test_absent_block_is_inert(self, caplog):
        """No `discord:` block: no client constructed, no tasks, nothing loud."""
        with caplog.at_level(logging.INFO):
            h = await _start_plugin(omit_block=True)
        try:
            assert h.clients == []
            assert h.plugin._connect_task is None
            assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        finally:
            await h.plugin.on_stop()

    async def test_missing_token_raises_naming_token(self):
        """A block with no token and no DISCORD_TOKEN env fails startup."""
        with pytest.raises(ValueError, match="token"):
            await _start_plugin(discord_overrides={"token": None}, wait_connect=False)

    async def test_env_token_fallback(self, monkeypatch):
        """DISCORD_TOKEN stands in for discord.token and reaches client.start."""
        monkeypatch.setenv("DISCORD_TOKEN", "env-token-value")
        h = await _start_plugin(discord_overrides={"token": None})
        try:
            assert h.clients[0].started_with == ["env-token-value"]
        finally:
            await h.plugin.on_stop()

    async def test_config_token_reaches_client_start(self):
        """The configured token is what client.start receives."""
        h = await _start_plugin()
        try:
            assert h.clients[0].started_with == [TOKEN]
        finally:
            await h.plugin.on_stop()

    @pytest.mark.parametrize("bad", ["123", 123, {"a": 1}])
    async def test_allow_not_a_list_raises(self, bad):
        """`allow` must be a list; the error names the field."""
        with pytest.raises(ValueError, match="allow"):
            await _start_plugin(discord_overrides={"allow": bad}, wait_connect=False)

    @pytest.mark.parametrize("bad", [0, 99, 2001, "2000", 100.5, None])
    async def test_chunk_size_out_of_range_raises(self, bad):
        """message_chunk_size must be an int in [100, 2000]."""
        with pytest.raises(ValueError, match="message_chunk_size"):
            await _start_plugin(discord_overrides={"message_chunk_size": bad}, wait_connect=False)

    @pytest.mark.parametrize("ok", [100, 1500, 2000])
    async def test_chunk_size_in_range_accepted(self, ok):
        """The boundaries of [100, 2000] are valid."""
        h = await _start_plugin(discord_overrides={"message_chunk_size": ok})
        await h.plugin.on_stop()

    async def test_int_allow_entries_are_normalized_to_str(self):
        """YAML parses ids as ints; the gate compares them to str(author.id)."""
        h = await _start_plugin(discord_overrides={"allow": [AUTH_ID, str(OTHER_ID)]})
        try:
            ch = h.registry.get_or_create("discord", f"dm:{AUTH_ID}")
            assert await _drive_gate(h.pm, channel=ch, sender=str(AUTH_ID), text="x") is True
            assert await _drive_gate(h.pm, channel=ch, sender=str(OTHER_ID), text="x") is True
            assert await _drive_gate(h.pm, channel=ch, sender="1", text="x") is False
        finally:
            await h.plugin.on_stop()

    def test_no_send_thinking_or_tool_status(self):
        """R15: like IRC/Signal, thinking and tool status are not rendered."""
        assert not hasattr(DiscordPlugin, "send_thinking")
        assert not hasattr(DiscordPlugin, "send_tool_status")


# ---------------------------------------------------------------------------
# R6, R3: connection lifecycle
# ---------------------------------------------------------------------------


class TestConnectionLifecycle:
    async def test_start_and_stop_closes_client_and_leaves_no_tasks(self):
        """on_stop closes the client and leaves no task of ours pending."""
        before = set(asyncio.all_tasks())
        h = await _start_plugin()
        client = h.clients[0]
        assert h.plugin._connect_task is not None
        await h.plugin.on_stop()
        assert client.close_calls >= 1
        leftover = {t for t in asyncio.all_tasks() if t not in before and not t.done()}
        assert leftover == set()

    async def test_login_failure_logs_one_error_and_never_retries(self, caplog, fast_backoff):
        """Bad token: exactly one ERROR, start called once, no retry."""
        failure = discord.errors.LoginFailure("Improper token has been passed.")
        with caplog.at_level(logging.DEBUG):
            h = await _start_plugin(behaviors=[failure])
            try:
                await _wait_until(lambda: h.plugin._connect_task.done())
                # Give a (wrong) retry ample time to happen.
                await asyncio.sleep(0.2)
                assert len(h.clients) == 1
                assert h.clients[0].started_with == [TOKEN]
                errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
                assert len(errors) == 1
                assert "token" in errors[0].getMessage().lower()
                assert h.clients[0].close_calls >= 1
            finally:
                await h.plugin.on_stop()

    async def test_other_exception_warns_closes_client_and_retries(self, caplog, fast_backoff):
        """A transient failure: WARNING, failed client closed, new client tried."""
        with caplog.at_level(logging.DEBUG):
            h = await _start_plugin(behaviors=[RuntimeError("boom")])
            try:
                await _wait_until(lambda: len(h.clients) >= 2 and h.clients[1].started_with)
                assert h.clients[0].close_calls >= 1
                assert h.clients[1].started_with == [TOKEN]
                assert any(r.levelno == logging.WARNING for r in caplog.records)
                assert not any(r.levelno >= logging.ERROR for r in caplog.records)
            finally:
                await h.plugin.on_stop()

    async def test_start_returning_while_running_is_retried(self, fast_backoff):
        """client.start() returning without a shutdown is a drop: reconnect."""
        h = await _start_plugin(behaviors=["return"])
        try:
            await _wait_until(lambda: len(h.clients) >= 2 and h.clients[1].started_with)
            assert h.clients[0].close_calls >= 1
        finally:
            await h.plugin.on_stop()

    async def test_backoff_doubles_from_ten_seconds(self):
        """Retries wait 10s, then double (same ladder as IRC/Signal)."""
        delays: list[float] = []
        real_sleep = asyncio.sleep

        async def tracking_sleep(delay, *a, **kw):
            delays.append(delay)
            await real_sleep(0.001)

        boom = [RuntimeError("boom")] * 4
        with patch("corvidae.channels.discord.asyncio.sleep", side_effect=tracking_sleep):
            h = await _start_plugin(behaviors=boom, wait_connect=False)
            try:
                # Poll with the captured real sleep; asyncio.sleep is patched
                # process-wide inside this block.
                for _ in range(1000):
                    if len(h.clients) >= 5:
                        break
                    await real_sleep(0.005)
                assert len(h.clients) >= 5
            finally:
                await h.plugin.on_stop()
        assert delays[0] == pytest.approx(10, rel=0.5)
        for prev, nxt in zip(delays[:3], delays[1:4]):
            assert nxt >= min(prev * 2, 300), delays

    async def test_token_never_logged(self, caplog, fast_backoff):
        """R3: no record at any level contains the token, on any path."""
        failure = discord.errors.LoginFailure("Improper token has been passed.")
        with caplog.at_level(logging.DEBUG):
            h = await _start_plugin(behaviors=[RuntimeError("boom"), failure])
            try:
                await _wait_until(lambda: h.plugin._connect_task.done())
                # Exercise inbound, gate and outbound paths too.
                h2 = await _start_plugin()
                try:
                    await _deliver(h2, make_dm("hi"))
                    ch = h2.registry.get_or_create("discord", f"dm:{AUTH_ID}")
                    await h2.plugin.send_message(channel=ch, text="reply")
                    await _drive_gate(h2.pm, channel=ch, sender=str(OTHER_ID), text="no")
                finally:
                    await h2.plugin.on_stop()
            finally:
                await h.plugin.on_stop()
        assert caplog.records, "expected some log output to inspect"
        for record in caplog.records:
            assert TOKEN not in _record_text(record)
            assert TOKEN not in str(record.args)


# ---------------------------------------------------------------------------
# R1, R4, R5, R7: inbound filtering
# ---------------------------------------------------------------------------


class TestInboundFiltering:
    async def test_dm_fires_on_message_once(self):
        """R1: a DM from an allowed user reaches on_message exactly once."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_dm("hello there"))
            assert _delivered(h) == ["hello there"]
        finally:
            await h.plugin.on_stop()

    async def test_own_message_dropped(self):
        """R4: the bot's own messages never fire on_message."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_dm("echo", author_id=BOT_ID))
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_bot_author_dropped(self):
        """R4: other bots are ignored, even when they mention us."""
        h = await _start_plugin()
        try:
            await _deliver(
                h,
                make_dm("beep", bot=True),
                make_message(f"<@{BOT_ID}> beep", bot=True, mentions=[BOT_MENTION]),
            )
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    @pytest.mark.parametrize("content", ["", "   ", "\n\t \n"])
    async def test_empty_or_whitespace_dropped(self, content):
        """R5: nothing to say, nothing delivered."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_dm(content))
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_attachment_only_dropped_with_debug_log(self, caplog):
        """R5: attachment-only messages are dropped; attachments are logged at DEBUG."""
        h = await _start_plugin()
        try:
            with caplog.at_level(logging.DEBUG):
                await _deliver(h, make_dm("", attachments=[SimpleNamespace(filename="a.png")]))
            assert _delivered(h) == []
            assert any(
                r.levelno == logging.DEBUG and "attach" in r.getMessage().lower()
                for r in caplog.records
            )
        finally:
            await h.plugin.on_stop()

    async def test_text_with_attachments_kept(self):
        """R5: text sent along with attachments is delivered (attachments discarded)."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_dm("look", attachments=[SimpleNamespace(filename="a.png")]))
            assert _delivered(h) == ["look"]
        finally:
            await h.plugin.on_stop()

    async def test_guild_message_without_mention_dropped(self):
        """R7: in a server channel the bot only reacts when addressed."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_message("just chatting"))
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_mention_of_someone_else_dropped(self):
        """R7: mentioning another user is not addressing the bot."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_message(f"<@{OTHER_ID}> hi", mentions=[OTHER_MENTION]))
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_everyone_only_not_treated_as_mention(self):
        """R7: @everyone/@here (mention_everyone, bot not in mentions) is dropped."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_message("@everyone look", mention_everyone=True))
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_mention_only_content_dropped(self):
        """R5/R7: after stripping the bot's mention nothing is left -> drop."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_message(f"<@{BOT_ID}>", mentions=[BOT_MENTION]))
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    @pytest.mark.parametrize("prefix", ["<@{id}>", "<@!{id}>"])
    async def test_mention_is_stripped_from_text(self, prefix):
        """R7: both mention forms are removed from the text the agent sees."""
        h = await _start_plugin()
        try:
            content = prefix.format(id=BOT_ID) + " hi"
            await _deliver(h, make_message(content, mentions=[BOT_MENTION]))
            assert _delivered(h) == ["hi"]
        finally:
            await h.plugin.on_stop()

    async def test_pinging_reply_without_token_is_admitted(self):
        """R7: a reply that pings the bot lists it in mentions but carries no
        <@id> token in the content; it is admitted with the text unchanged."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_message("hi", mentions=[BOT_MENTION]))
            assert _delivered(h) == ["hi"]
        finally:
            await h.plugin.on_stop()

    async def test_other_mentions_are_left_in_text(self):
        """Only the bot's own mention token is stripped."""
        h = await _start_plugin()
        try:
            content = f"<@{BOT_ID}> ask <@{OTHER_ID}> x"
            await _deliver(h, make_message(content, mentions=[BOT_MENTION, OTHER_MENTION]))
            assert _delivered(h) == [f"ask <@{OTHER_ID}> x"]
        finally:
            await h.plugin.on_stop()

    async def test_dm_needs_no_mention(self):
        """R7: DMs always pass without a mention."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_dm("plain dm"))
            assert _delivered(h) == ["plain dm"]
        finally:
            await h.plugin.on_stop()

    async def test_guild_mention_with_empty_content_warns_and_drops(self, caplog):
        """R6: a mention that arrives with empty content is a Discord-side
        change worth a WARNING; the message is dropped."""
        h = await _start_plugin()
        try:
            with caplog.at_level(logging.DEBUG):
                await _deliver(h, make_message("", mentions=[BOT_MENTION]))
            assert _delivered(h) == []
            assert any(r.levelno == logging.WARNING for r in caplog.records)
        finally:
            await h.plugin.on_stop()

    async def test_handler_exception_does_not_stop_processing(self, caplog):
        """A failing on_message for one message is logged; the next still runs."""
        h = await _start_plugin()
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("handler blew up")

        h.pm.ahook.on_message = AsyncMock(side_effect=flaky)
        try:
            with caplog.at_level(logging.DEBUG):
                await _deliver(h, make_dm("first"))
            assert "first" in h.texts() and "__sentinel__" in h.texts()
            assert any(r.levelno >= logging.WARNING for r in caplog.records)
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R8: scope derivation
# ---------------------------------------------------------------------------


class TestScope:
    async def _single_call(self, h: Harness):
        return h.on_message.await_args_list[0].kwargs

    async def test_guild_channel_scope(self):
        """Server channel -> discord:<channel_id>; sender is str(author.id)."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_message(f"<@{BOT_ID}> hi", mentions=[BOT_MENTION], channel_id=CHAN_ID))
            kw = await self._single_call(h)
            assert kw["channel"].id == f"discord:{CHAN_ID}"
            assert kw["sender"] == str(AUTH_ID)
            assert h.registry.get(f"discord:{CHAN_ID}") is kw["channel"]
            # The transport mints a real (non-empty string) trace id.
            assert isinstance(kw["trace_id"], str) and kw["trace_id"]
        finally:
            await h.plugin.on_stop()

    async def test_thread_has_its_own_scope(self):
        """A thread is just a channel with its own id -> its own history."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_message(f"<@{BOT_ID}> hi", mentions=[BOT_MENTION], channel_id=THREAD_ID))
            kw = await self._single_call(h)
            assert kw["channel"].id == f"discord:{THREAD_ID}"
            assert kw["channel"].id != f"discord:{CHAN_ID}"
        finally:
            await h.plugin.on_stop()

    async def test_dm_scope_is_keyed_on_user_not_dm_channel(self):
        """DM -> discord:dm:<user_id> (stable across DM channel ids)."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_dm("hi"))
            kw = await self._single_call(h)
            assert kw["channel"].id == f"discord:dm:{AUTH_ID}"
            assert kw["sender"] == str(AUTH_ID)
        finally:
            await h.plugin.on_stop()

    async def test_channel_uses_discord_transport(self):
        """Channels created here match the 'discord' transport."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_dm("hi"))
            ch = h.on_message.await_args_list[0].kwargs["channel"]
            assert ch.matches_transport("discord")
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R14: ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    async def test_messages_reach_on_message_in_arrival_order(self):
        """Enqueued messages are processed one at a time, in order, even
        when on_message awaits (discord.py runs each event in its own task)."""
        h = await _start_plugin()
        seen: list[str] = []

        async def slow(**kwargs):
            # Earlier messages take longer, so concurrency would reorder.
            n = int(kwargs["text"].split()[-1]) if kwargs["text"] != "__sentinel__" else 0
            await asyncio.sleep(0.02 if n % 2 else 0.001)
            seen.append(kwargs["text"])

        h.pm.ahook.on_message = AsyncMock(side_effect=slow)
        try:
            await _deliver(h, *[make_dm(f"msg {i}") for i in range(1, 9)])
            # The side effect records after its sleep, so wait for the last one.
            await _wait_until(lambda: len(seen) == 9)
            assert seen == [f"msg {i}" for i in range(1, 9)] + ["__sentinel__"]
        finally:
            await h.plugin.on_stop()

    def test_enqueue_is_synchronous(self):
        """_enqueue is a plain function (put_nowait), not a coroutine."""
        assert not asyncio.iscoroutinefunction(DiscordPlugin._enqueue)


# ---------------------------------------------------------------------------
# R10: authorization gate
# ---------------------------------------------------------------------------


class TestAuthorizationGate:
    async def test_allowed_sender_accepted(self):
        h = await _start_plugin()
        try:
            ch = h.registry.get_or_create("discord", f"dm:{AUTH_ID}")
            assert await _drive_gate(h.pm, channel=ch, sender=str(AUTH_ID), text="x") is True
        finally:
            await h.plugin.on_stop()

    async def test_other_sender_rejected_with_info_log(self, caplog):
        h = await _start_plugin()
        try:
            ch = h.registry.get_or_create("discord", f"dm:{OTHER_ID}")
            with caplog.at_level(logging.INFO):
                assert await _drive_gate(h.pm, channel=ch, sender=str(OTHER_ID), text="x") is False
            assert any(
                r.levelno == logging.INFO and str(OTHER_ID) in r.getMessage()
                for r in caplog.records
            )
        finally:
            await h.plugin.on_stop()

    async def test_non_discord_channel_gets_no_opinion(self):
        h = await _start_plugin()
        try:
            ch = h.registry.get_or_create("irc", "#chan")
            assert await _drive_gate(h.pm, channel=ch, sender=str(OTHER_ID), text="x") is None
        finally:
            await h.plugin.on_stop()

    async def test_empty_allowlist_denies_everyone(self):
        """Default-deny: an empty allow list admits nobody."""
        h = await _start_plugin(discord_overrides={"allow": []})
        try:
            ch = h.registry.get_or_create("discord", f"dm:{AUTH_ID}")
            assert await _drive_gate(h.pm, channel=ch, sender=str(AUTH_ID), text="x") is False
        finally:
            await h.plugin.on_stop()

    async def test_typing_only_for_allowed_senders(self, fast_typing):
        """Denied senders get no typing indicator (no sign the bot is live)."""
        h = await _start_plugin()
        try:
            h.plugin._enqueue(make_dm("let me in", author_id=OTHER_ID), BOT_ID)
            await _wait_until(lambda: h.on_message.await_count >= 1)
            await asyncio.sleep(TYPING_REFRESH_FOR_TESTS * 4)
            assert h.messageable.typing_count == 0

            h.plugin._enqueue(make_dm("hello", author_id=AUTH_ID), BOT_ID)
            await _wait_until(lambda: h.messageable.typing_count >= 1)
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R13: typing
# ---------------------------------------------------------------------------


class TestTyping:
    async def test_typing_refreshes_while_outstanding(self, fast_typing):
        """Typing is re-sent every refresh interval while a reply is owed."""
        h = await _start_plugin()
        try:
            await _deliver(h, make_dm("work on this"))
            await _wait_until(lambda: h.messageable.typing_count >= 3)
        finally:
            await h.plugin.on_stop()

    async def test_typing_stops_after_send_message(self, fast_typing):
        """send_message decrements outstanding; the refresh loop falls silent."""
        h = await _start_plugin(discord_overrides={"allow": [AUTH_ID]})
        try:
            h.plugin._enqueue(make_dm("work"), BOT_ID)
            await _wait_until(lambda: h.messageable.typing_count >= 2)
            ch = h.registry.get(f"discord:dm:{AUTH_ID}")
            await h.plugin.send_message(channel=ch, text="done")
            await asyncio.sleep(TYPING_REFRESH_FOR_TESTS * 3)  # let in-flight refresh settle
            settled = h.messageable.typing_count
            await asyncio.sleep(TYPING_REFRESH_FOR_TESTS * 6)
            assert h.messageable.typing_count == settled
        finally:
            await h.plugin.on_stop()

    async def test_refused_stranger_does_not_stop_owed_typing(self, fast_typing):
        """A refused sender never took an outstanding slot, so their
        rejection must not give one back: typing for an allowlisted user's
        turn in the same channel keeps running."""
        h = await _start_plugin()
        try:
            h.plugin._enqueue(make_dm("work"), BOT_ID)
            await _wait_until(lambda: h.messageable.typing_count >= 2)
            ch = h.registry.get(f"discord:dm:{AUTH_ID}")
            await h.plugin.on_message_rejected(
                channel=ch, trace_id="tid", sender=str(OTHER_ID), text="hi",
            )
            assert h.plugin._outstanding[ch.id] == 1
            before = h.messageable.typing_count
            await _wait_until(lambda: h.messageable.typing_count >= before + 2)
        finally:
            await h.plugin.on_stop()

    async def test_one_refresh_loop_per_channel_and_stop_cancels_it(self, fast_typing):
        """Re-arming typing while the previous refresh loop is still asleep
        reuses that loop rather than starting a second one, and on_stop
        leaves no refresh loop running."""

        def _live_loops() -> list[asyncio.Task]:
            # Every not-yet-finished task running the refresh coroutine.
            return [
                t for t in asyncio.all_tasks()
                if not t.done() and "_typing_refresh_loop" in repr(t.get_coro())
            ]

        h = await _start_plugin()
        try:
            h.plugin._enqueue(make_dm("one"), BOT_ID)
            await _wait_until(lambda: h.messageable.typing_count >= 1)
            ch = h.registry.get(f"discord:dm:{AUTH_ID}")
            # Reply, then a new message before the loop's sleep ends.
            h.plugin._decrement_outstanding(ch)
            await h.plugin._begin_typing(ch)
            assert len(_live_loops()) == 1
            # Reply again, wait out the old loop, and re-arm once more.
            h.plugin._decrement_outstanding(ch)
            await asyncio.sleep(TYPING_REFRESH_FOR_TESTS * 3)
            await h.plugin._begin_typing(ch)
            assert len(_live_loops()) == 1
        finally:
            await h.plugin.on_stop()
        assert _live_loops() == []

    async def test_typing_stops_after_rejection(self, fast_typing):
        """on_message_rejected also gives the outstanding slot back."""
        h = await _start_plugin()
        try:
            h.plugin._enqueue(make_dm("work"), BOT_ID)
            await _wait_until(lambda: h.messageable.typing_count >= 2)
            ch = h.registry.get(f"discord:dm:{AUTH_ID}")
            await h.plugin.on_message_rejected(
                channel=ch, trace_id="tid", sender=str(AUTH_ID), text="work",
            )
            await asyncio.sleep(TYPING_REFRESH_FOR_TESTS * 3)
            settled = h.messageable.typing_count
            await asyncio.sleep(TYPING_REFRESH_FOR_TESTS * 6)
            assert h.messageable.typing_count == settled
        finally:
            await h.plugin.on_stop()

    async def test_send_progress_keeps_typing_alive(self, fast_typing):
        """R13: send_progress does not decrement outstanding."""
        h = await _start_plugin()
        try:
            h.plugin._enqueue(make_dm("work"), BOT_ID)
            await _wait_until(lambda: h.messageable.typing_count >= 1)
            ch = h.registry.get(f"discord:dm:{AUTH_ID}")
            await h.plugin.send_progress(channel=ch, text="thinking out loud")
            before = h.messageable.typing_count
            await _wait_until(lambda: h.messageable.typing_count >= before + 2)
        finally:
            await h.plugin.on_stop()

    async def test_typing_failure_is_logged_not_raised(self, fast_typing, caplog):
        """A typing call that raises must not take down message processing."""
        h = await _start_plugin()

        async def broken(channel):
            raise RuntimeError("no channel")

        h.plugin._messageable = broken
        try:
            with caplog.at_level(logging.DEBUG):
                await _deliver(h, make_dm("hi"))
            assert "hi" in h.texts()
            assert any(r.levelno >= logging.WARNING for r in caplog.records)
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R9: resolve_channel
# ---------------------------------------------------------------------------


class TestResolveChannel:
    async def test_guild_id_resolves(self):
        h = await _start_plugin()
        try:
            ch = await h.plugin.resolve_channel(channel_id=f"discord:{CHAN_ID}")
            assert ch is not None
            assert ch.id == f"discord:{CHAN_ID}"
            assert h.registry.get(f"discord:{CHAN_ID}") is ch
        finally:
            await h.plugin.on_stop()

    async def test_dm_id_resolves(self):
        h = await _start_plugin()
        try:
            ch = await h.plugin.resolve_channel(channel_id=f"discord:dm:{AUTH_ID}")
            assert ch is not None
            assert ch.id == f"discord:dm:{AUTH_ID}"
        finally:
            await h.plugin.on_stop()

    @pytest.mark.parametrize(
        "channel_id",
        ["irc:#chan", "signal:abc", "cli:", "discord:", "discord:abc", "discord:dm:",
         "discord:dm:abc", "discord:12a", "discord:dm:12:3", "discord"],
    )
    async def test_other_or_malformed_ids_defer(self, channel_id):
        h = await _start_plugin()
        try:
            assert await h.plugin.resolve_channel(channel_id=channel_id) is None
        finally:
            await h.plugin.on_stop()

    async def test_none_before_on_start(self):
        """Before the registry is resolved the hook defers."""
        pm = create_plugin_manager()
        plugin = DiscordPlugin(pm)
        assert await plugin.resolve_channel(channel_id=f"discord:{CHAN_ID}") is None

    async def test_resolve_through_pluggy(self):
        """Dispatch through the real firstresult hook returns the Channel."""
        h = await _start_plugin()
        try:
            ch = await h.pm.ahook.resolve_channel(channel_id=f"discord:dm:{AUTH_ID}")
            assert ch is not None and ch.id == f"discord:dm:{AUTH_ID}"
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R11, R12: outbound
# ---------------------------------------------------------------------------


class TestOutbound:
    async def _channel(self, h: Harness):
        return h.registry.get_or_create("discord", f"dm:{AUTH_ID}")

    async def test_send_message_delivers_with_no_mentions(self):
        """R11: every send passes AllowedMentions.none() explicitly."""
        h = await _start_plugin()
        try:
            await h.plugin.send_message(channel=await self._channel(h), text="hello @everyone <@123>")
            assert h.messageable.contents == ["hello @everyone <@123>"]
            for _, kwargs in h.messageable.sends:
                assert _is_no_mentions(kwargs.get("allowed_mentions"))
        finally:
            await h.plugin.on_stop()

    async def test_every_chunk_of_a_long_message_has_no_mentions(self):
        """R11 holds for every chunk, not just the first."""
        h = await _start_plugin()
        try:
            await h.plugin.send_message(channel=await self._channel(h), text="word " * 1500)
            assert len(h.messageable.sends) >= 3
            for _, kwargs in h.messageable.sends:
                assert _is_no_mentions(kwargs.get("allowed_mentions"))
        finally:
            await h.plugin.on_stop()

    async def test_send_progress_has_no_mentions(self):
        h = await _start_plugin()
        try:
            await h.plugin.send_progress(channel=await self._channel(h), text="@everyone")
            assert len(h.messageable.sends) == 1
            assert _is_no_mentions(h.messageable.sends[0][1].get("allowed_mentions"))
        finally:
            await h.plugin.on_stop()

    async def test_long_text_split_into_chunks_within_limit(self):
        """R12: a 5000-char reply goes out whole, each send <= 2000 chars."""
        h = await _start_plugin()
        text = ("A sentence of moderate length. " * 40 + "\n\n") * 4
        try:
            await h.plugin.send_message(channel=await self._channel(h), text=text)
            sent = h.messageable.contents
            assert len(sent) >= 3
            assert all(len(c) <= 2000 for c in sent)
            assert "".join(sent) == text
        finally:
            await h.plugin.on_stop()

    async def test_configured_chunk_size_is_honoured(self):
        h = await _start_plugin(discord_overrides={"message_chunk_size": 100})
        text = "word " * 100
        try:
            await h.plugin.send_message(channel=await self._channel(h), text=text)
            sent = h.messageable.contents
            assert len(sent) >= 5
            assert all(len(c.encode("utf-8")) <= 100 for c in sent)
            assert "".join(sent) == text
        finally:
            await h.plugin.on_stop()

    async def test_code_fence_closed_and_reopened_across_sends(self):
        """R12: a chunk boundary inside ``` closes and reopens the fence."""
        h = await _start_plugin()
        code = "".join(f"value_{i:04d} = compute({i})\n" for i in range(120))
        text = "```python\n" + code + "```\n"
        try:
            await h.plugin.send_message(channel=await self._channel(h), text=text)
            sent = h.messageable.contents
            assert len(sent) >= 2
            assert sent[0].endswith("```")
            assert sent[1].startswith("```python\n")
        finally:
            await h.plugin.on_stop()

    async def test_whitespace_only_chunks_never_sent(self):
        """Discord rejects empty messages: whitespace-only chunks are skipped."""
        h = await _start_plugin(discord_overrides={"message_chunk_size": 100})
        try:
            await h.plugin.send_message(
                channel=await self._channel(h), text="a" * 100 + "\n" * 150 + "b" * 100,
            )
            assert h.messageable.sends, "real content must still be sent"
            assert all(c is not None and c.strip() for c in h.messageable.contents)

            h.messageable.sends.clear()
            await h.plugin.send_message(channel=await self._channel(h), text="  \n\n  ")
            assert h.messageable.sends == []
        finally:
            await h.plugin.on_stop()

    @pytest.mark.parametrize("hook", ["send_message", "send_progress"])
    async def test_broadcast_filter_ignores_other_transports(self, hook):
        """Every send_* opens with a matches_transport('discord') filter."""
        h = await _start_plugin()
        try:
            irc = h.registry.get_or_create("irc", "#chan")
            await getattr(h.plugin, hook)(channel=irc, text="not yours")
            assert h.messageable.sends == []
            assert h.messageable_requests == []
        finally:
            await h.plugin.on_stop()

    async def test_send_message_via_pluggy_broadcast_ignores_other_transports(self):
        h = await _start_plugin()
        try:
            irc = h.registry.get_or_create("irc", "#chan")
            await h.pm.ahook.send_message(channel=irc, text="hi", latency_ms=None)
            assert h.messageable.sends == []
        finally:
            await h.plugin.on_stop()

    async def test_send_progress_does_not_decrement_outstanding(self, fast_typing):
        """send_progress delivers but leaves the reply owed (see TestTyping);
        here: the message goes out and typing keeps going afterwards."""
        h = await _start_plugin()
        try:
            # Exactly one owed reply (no sentinel, which would add a second).
            h.plugin._enqueue(make_dm("go"), BOT_ID)
            await _wait_until(lambda: h.on_message.await_count >= 1)
            ch = h.registry.get(f"discord:dm:{AUTH_ID}")
            await h.plugin.send_progress(channel=ch, text="step 1")
            assert h.messageable.contents == ["step 1"]
            mark = h.messageable.typing_count
            await _wait_until(lambda: h.messageable.typing_count > mark + 1)
        finally:
            await h.plugin.on_stop()

    async def test_send_failure_is_logged_not_raised(self, caplog):
        """A failing send neither raises into the agent nor goes unlogged."""
        h = await _start_plugin()
        h.messageable.fail_with = discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Missing Permissions")
        try:
            with caplog.at_level(logging.DEBUG):
                await h.plugin.send_message(channel=await self._channel(h), text="hi")
            assert any(r.levelno >= logging.WARNING for r in caplog.records)
        finally:
            await h.plugin.on_stop()

    async def test_unresolvable_channel_is_logged_not_raised(self, caplog):
        """If no messageable can be found the send is dropped with a log."""
        h = await _start_plugin()

        async def nowhere(channel):
            return None

        h.plugin._messageable = nowhere
        try:
            with caplog.at_level(logging.DEBUG):
                await h.plugin.send_message(channel=await self._channel(h), text="hi")
            assert any(r.levelno >= logging.WARNING for r in caplog.records)
        finally:
            await h.plugin.on_stop()


class TestMessageableLookup:
    """The real ``_messageable`` against a fake client."""

    async def _plugin(self, lookup: LookupClient | None):
        pm = create_plugin_manager()
        registry = ChannelRegistry(AGENT_DEFAULTS)
        pm.register(registry, name="registry")
        plugin = DiscordPlugin(pm)
        await plugin.on_init(pm=pm, config={"discord": {"token": TOKEN, "allow": []}})
        plugin._client = lookup
        return plugin, registry

    async def test_guild_channel_from_cache(self):
        lookup, target = LookupClient(), FakeMessageable()
        lookup.cached_channels[CHAN_ID] = target
        plugin, reg = await self._plugin(lookup)
        got = await plugin._messageable(reg.get_or_create("discord", str(CHAN_ID)))
        assert got is target
        assert ("fetch_channel", CHAN_ID) not in lookup.calls

    async def test_guild_channel_falls_back_to_fetch(self):
        lookup, target = LookupClient(), FakeMessageable()
        lookup.remote_channels[CHAN_ID] = target
        plugin, reg = await self._plugin(lookup)
        got = await plugin._messageable(reg.get_or_create("discord", str(CHAN_ID)))
        assert got is target
        assert ("get_channel", CHAN_ID) in lookup.calls
        assert ("fetch_channel", CHAN_ID) in lookup.calls

    async def test_dm_uses_cached_user_and_create_dm(self):
        lookup, dm = LookupClient(), FakeMessageable()
        lookup.cached_users[AUTH_ID] = make_user(dm)
        plugin, reg = await self._plugin(lookup)
        got = await plugin._messageable(reg.get_or_create("discord", f"dm:{AUTH_ID}"))
        assert got is dm
        assert ("fetch_user", AUTH_ID) not in lookup.calls

    async def test_dm_falls_back_to_fetch_user(self):
        lookup, dm = LookupClient(), FakeMessageable()
        lookup.remote_users[AUTH_ID] = make_user(dm)
        plugin, reg = await self._plugin(lookup)
        got = await plugin._messageable(reg.get_or_create("discord", f"dm:{AUTH_ID}"))
        assert got is dm
        assert ("fetch_user", AUTH_ID) in lookup.calls

    async def test_lookup_error_returns_none_and_logs(self, caplog):
        lookup = LookupClient()
        lookup.fail_remote = discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "unknown")
        plugin, reg = await self._plugin(lookup)
        with caplog.at_level(logging.DEBUG):
            got = await plugin._messageable(reg.get_or_create("discord", str(CHAN_ID)))
            got_dm = await plugin._messageable(reg.get_or_create("discord", f"dm:{AUTH_ID}"))
        assert got is None and got_dm is None
        assert any(r.levelno >= logging.INFO for r in caplog.records)

    async def test_no_client_returns_none(self):
        plugin, reg = await self._plugin(None)
        assert await plugin._messageable(reg.get_or_create("discord", str(CHAN_ID))) is None


# ---------------------------------------------------------------------------
# R6: the real client (intents) and discord.py -> plugin glue
# ---------------------------------------------------------------------------


class TestRealClient:
    def test_client_does_not_request_message_content_intent(self):
        """R6: no privileged Message Content intent is requested."""
        client = DiscordPlugin(create_plugin_manager())._make_client()
        assert isinstance(client, discord.Client)
        assert client.intents.message_content is False

    async def test_client_on_message_reaches_on_message_hook(self):
        """The real client's on_message feeds the plugin's inbound queue."""
        h = await _start_plugin()
        try:
            # The class's real factory, bypassing the instance-level fake.
            client = DiscordPlugin._make_client(h.plugin)
            client._connection.user = SimpleNamespace(id=BOT_ID)
            await client.on_message(make_dm("through the glue"))
            await _wait_until(lambda: "through the glue" in h.texts())
            kw = h.on_message.await_args_list[0].kwargs
            assert kw["channel"].id == f"discord:dm:{AUTH_ID}"
        finally:
            await h.plugin.on_stop()

    async def test_client_on_message_drops_own_message(self):
        """The bot id the glue passes is the client's own user id (R4)."""
        h = await _start_plugin()
        try:
            client = DiscordPlugin._make_client(h.plugin)
            client._connection.user = SimpleNamespace(id=BOT_ID)
            await client.on_message(make_dm("echo", author_id=BOT_ID))
            await _deliver(h)
            assert _delivered(h) == []
        finally:
            await h.plugin.on_stop()

    async def test_on_ready_logs_info_with_user_id(self, caplog):
        """on_ready announces itself at INFO with the bot's user id."""
        h = await _start_plugin()
        try:
            client = DiscordPlugin._make_client(h.plugin)
            client._connection.user = SimpleNamespace(id=BOT_ID)
            with caplog.at_level(logging.INFO):
                await client.on_ready()
            assert any(
                r.levelno == logging.INFO and str(BOT_ID) in r.getMessage()
                for r in caplog.records
            )
        finally:
            await h.plugin.on_stop()

    async def test_on_ready_resets_backoff(self):
        """Design: backoff resets to the initial value on on_ready. A
        failure after a successful ready waits the initial delay again
        instead of continuing the doubling ladder."""
        delays: list[float] = []
        real_sleep = asyncio.sleep
        holder: dict = {}

        async def ready_then_fail(_fake_client) -> None:
            # Drive the real client's on_ready, then drop the connection.
            real = DiscordPlugin._make_client(holder["h"].plugin)
            real._connection.user = SimpleNamespace(id=BOT_ID)
            await real.on_ready()
            raise RuntimeError("connection lost after ready")

        async def tracking_sleep(delay, *a, **kw):
            delays.append(delay)
            await real_sleep(0.001)

        behaviors = [RuntimeError("fail 1"), RuntimeError("fail 2"), ready_then_fail, RuntimeError("fail 4")]
        with patch("corvidae.channels.discord.asyncio.sleep", side_effect=tracking_sleep):
            h = await _start_plugin(behaviors=behaviors, wait_connect=False)
            holder["h"] = h
            try:
                for _ in range(1000):
                    if len(h.clients) >= 5:
                        break
                    await real_sleep(0.005)
                assert len(h.clients) >= 5
            finally:
                await h.plugin.on_stop()
        # Failures: #1 -> 10, #2 -> 20, ready+fail #3 -> reset to 10, #4 -> 20.
        assert delays[0] == pytest.approx(10, rel=0.5)
        assert delays[1] == pytest.approx(2 * delays[0], rel=0.01)
        assert delays[2] == pytest.approx(delays[0], rel=0.01), delays
        assert delays[3] == pytest.approx(2 * delays[0], rel=0.01), delays


# ---------------------------------------------------------------------------
# R16: per-channel config keyed on raw ids
# ---------------------------------------------------------------------------


class TestChannelConfigKeys:
    async def test_raw_id_keys_round_trip_and_apply_to_inbound(self):
        """`channels: discord:<id>` / `discord:dm:<id>` register channels whose
        ids round-trip, and an inbound DM lands on the pre-registered one."""
        channels = {
            "discord:123": {"system_prompt": "guild prompt"},
            "discord:dm:456": {"max_turns": 7},
        }
        h = await _start_plugin(
            discord_overrides={"allow": [456]},
            extra_config={"channels": channels},
        )
        try:
            guild_ch = h.registry.get("discord:123")
            dm_ch = h.registry.get("discord:dm:456")
            assert guild_ch is not None and guild_ch.id == "discord:123"
            assert dm_ch is not None and dm_ch.id == "discord:dm:456"
            assert guild_ch.config.system_prompt == "guild prompt"

            await _deliver(h, make_dm("hi", author_id=456))
            landed = h.on_message.await_args_list[0].kwargs["channel"]
            assert landed is dm_ch
            assert landed.config.max_turns == 7
        finally:
            await h.plugin.on_stop()


# ---------------------------------------------------------------------------
# R12: splitter
# ---------------------------------------------------------------------------

_SPLIT_FRAGMENTS = [
    "alpha", "beta", "gamma", "x", "hello",
    "中", "文。", "\U0001f600", "é",
    " ", " ", " ", "\n", "\n", "\n\n",
    ".", "!", "?",
    "`", "``", "```", "```", "```python\n", "```\n", "```js\n", "~~~",
]


def _random_markdown(rng: random.Random, n: int) -> str:
    """Random mixed text (fences, backticks, CJK, emoji, long words)."""
    parts = [rng.choice(_SPLIT_FRAGMENTS) for _ in range(rng.randint(0, 80))]
    if rng.random() < 0.3:
        word = rng.choice(["a", "中", "\U0001f600", "`"]) * rng.randint(1, n * 2)
        parts.insert(rng.randint(0, len(parts)), word)
    return "".join(parts)


def _check_segments(text: str, n: int) -> None:
    """Assert the three splitter invariants for one (text, n) case."""
    segments = _split_discord_segments(text, n)
    assert "".join(body for _, body, _ in segments) == text, "bodies must tile the text"
    for prefix, body, suffix in segments:
        chunk = prefix + body + suffix
        assert len(chunk.encode("utf-8")) <= n, f"{len(chunk.encode('utf-8'))} bytes > {n}"
        assert chunk.count("```") % 2 == 0, f"unbalanced fences in {chunk!r}"


class TestSplitDiscordMessage:
    def test_fence_reserve_is_24_bytes(self):
        assert _FENCE_RESERVE == len("```") + 16 + len("\n") + len("\n```") == 24

    @pytest.mark.parametrize("seed", [10, 11, 12, 13])
    def test_invariants_small_limits(self, seed):
        """Seeded property test: n in [30, 200]."""
        rng = random.Random(seed)
        for _ in range(500):
            n = rng.randint(30, 200)
            _check_segments(_random_markdown(rng, n), n)

    def test_invariants_production_limit(self):
        """Seeded property test at n = 2000 with longer texts."""
        rng = random.Random(2000)
        for _ in range(250):
            text = "".join(_random_markdown(rng, 2000) for _ in range(rng.randint(1, 8)))
            _check_segments(text, 2000)

    def test_chunks_are_prefix_body_suffix(self):
        """split_discord_message is exactly the prefix+body+suffix join."""
        text = "```python\n" + "x = 1\n" * 600 + "```\n"
        segments = _split_discord_segments(text, 2000)
        assert split_discord_message(text, 2000) == [p + b + s for p, b, s in segments]

    def test_reopened_fence_keeps_language(self):
        """```python + ~3000 bytes @2000: chunk 1 closes, chunk 2 reopens python."""
        code = "".join(f"total_{i:03d} = {i} * 2\n" for i in range(160))
        assert 2900 <= len(code.encode()) <= 3300
        text = "```python\n" + code + "```\n"
        chunks = split_discord_message(text, 2000)
        assert len(chunks) == 2
        assert chunks[0].startswith("```python\n")
        assert chunks[0].endswith("```")
        assert chunks[1].startswith("```python\n")
        assert chunks[1].endswith("```\n")
        assert all(len(c.encode()) <= 2000 for c in chunks)

    def test_reopened_fence_without_language(self):
        """A bare ``` fence reopens as a bare ``` line."""
        code = "".join(f"line {i:04d} of output\n" for i in range(200))
        chunks = split_discord_message("```\n" + code + "```\n", 2000)
        assert len(chunks) >= 2
        assert chunks[0].endswith("```")
        assert chunks[1].startswith("```\n")

    def test_overlong_language_is_not_carried(self):
        """An info string over 16 bytes is dropped on reopen (reserve safety)."""
        lang = "x" * 40
        code = "".join(f"line {i:04d} of output\n" for i in range(200))
        chunks = split_discord_message(f"```{lang}\n" + code + "```\n", 2000)
        assert len(chunks) >= 2
        assert chunks[1].startswith("```\n")
        assert all(len(c.encode()) <= 2000 for c in chunks)

    def test_one_line_fence_does_not_open_a_block(self):
        """```x``` on one line leaves the parity even: no suffix, no reopen."""
        text = "see ```x``` here\n" + "plain text line\n" * 300
        segments = _split_discord_segments(text, 2000)
        assert len(segments) >= 2
        assert all(prefix == "" and suffix == "" for prefix, _, suffix in segments)

    def test_plain_text_is_unchanged(self):
        """No fences: no prefix, no suffix, content reproduced exactly."""
        text = ("A plain sentence without any code. " * 30 + "\n") * 8
        segments = _split_discord_segments(text, 2000)
        assert len(segments) >= 2
        assert all(prefix == "" and suffix == "" for prefix, _, suffix in segments)
        chunks = split_discord_message(text, 2000)
        assert "".join(chunks) == text
        assert all("```" not in c for c in chunks)

    def test_short_text_is_one_chunk(self):
        assert split_discord_message("hello", 2000) == ["hello"]

    def test_whitespace_only_chunks_are_kept_by_the_splitter(self):
        """Filtering is a send-time concern; the splitter reconstructs exactly."""
        text = "a" * 100 + "\n" * 300 + "b"
        segments = _split_discord_segments(text, 100)
        assert "".join(body for _, body, _ in segments) == text
        chunks = split_discord_message(text, 100)
        assert chunks == [p + b + s for p, b, s in segments]
        assert "".join(chunks) == text
        assert any(not c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# Wiring: hook arg binding and entry point
# ---------------------------------------------------------------------------


def _is_no_mentions(value: object) -> bool:
    """True when value is an AllowedMentions that pings nobody.

    AllowedMentions defines no __eq__ in discord.py 2.7.1, so compare the
    serialized form rather than the objects themselves.
    """
    return (
        isinstance(value, discord.AllowedMentions)
        and value.to_dict() == discord.AllowedMentions.none().to_dict()
    )


class TestWiring:
    def test_production_constants(self):
        """R13/R12: Discord's typing indicator lasts ~10s, so refresh at 8s;
        Discord's message limit is 2000."""
        assert dmod.TYPING_REFRESH_SECONDS == 8
        assert dmod.DISCORD_MESSAGE_LIMIT == 2000

    def test_hook_arg_binding_guard_passes(self):
        """No hookimpl defaults a spec-required param (pluggy would drop it)."""
        pm = create_plugin_manager()
        pm.register(ChannelRegistry(AGENT_DEFAULTS), name="registry")
        pm.register(DiscordPlugin(pm), name="discord")
        _check_hook_arg_binding(pm)  # must not raise

    def test_entry_point_registered(self):
        """Without the entry point the plugin is silently never loaded."""
        root = Path(__file__).resolve().parent.parent
        data = tomllib.loads((root / "pyproject.toml").read_text())
        eps = data["project"]["entry-points"]["corvidae"]
        assert eps["discord"] == "corvidae.channels.discord:DiscordPlugin"

    def test_depends_on_registry(self):
        assert "registry" in DiscordPlugin.depends_on
