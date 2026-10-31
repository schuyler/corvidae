"""IRC allowlist accounts come from each message's own tags (#77 review).

These drive raw lines through a real pydle IRCClient.on_raw, because pydle's
user-state tracking (not a hand-built users dict) is what the gate trusts.
"""

import importlib.util
import pathlib
from unittest.mock import MagicMock

import pytest

pydle_tags = pytest.importorskip("pydle.features.ircv3.tags")

import corvidae  # noqa: E402


def _load_real_irc():
    """A private copy of corvidae.channels.irc bound to the real pydle.

    tests/test_irc_plugin.py imports corvidae.channels.irc inside a
    patch.dict(sys.modules, ...) that supplies a mocked pydle. patch.dict
    removes the module from sys.modules on exit, but the mocked `pydle`
    binding survives in that module's own names, so its IRCClient subclasses
    a mock, not a real pydle client. Loading the file afresh here, with the
    real pydle importable, gives an IRCClient that runs pydle's own
    on_raw and user tracking.
    """
    path = pathlib.Path(corvidae.__file__).parent / "channels" / "irc.py"
    spec = importlib.util.spec_from_file_location("_corvidae_irc_real", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


irc_real = _load_real_irc()

CAPS = {"account-tag": True, "account-notify": True, "extended-join": True}


def _stack(allow, caps=CAPS, case_mapping=None):
    """A real IRCClient wired to a plugin; returns (plugin, client, seen)."""
    plugin = irc_real.IRCPlugin(pm=MagicMock())
    plugin._allow = set(allow)
    seen: list[tuple[str, str, bool]] = []

    async def record(target, by, message):
        # The gate verdict as decided on arrival; these tests drive on_raw.
        seen.append((by, message, plugin._decide(by)[0]))

    plugin.on_channel_message = record
    plugin.on_private_message = record
    client = irc_real.IRCClient(plugin, "buster")
    plugin.client = client
    client._capabilities = dict(caps)
    if case_mapping:
        client._case_mapping = case_mapping
    return plugin, client, seen


async def _feed(client, line: str) -> None:
    msg = pydle_tags.TaggedMessage.parse(line.encode() + b"\r\n", encoding="utf-8")
    await client.on_raw(msg)


async def test_first_tagged_pm_admitted():
    """A logged-in allowed user is admitted on the very first PM."""
    _plugin, client, seen = _stack(["alice"])
    await _feed(client, "@account=alice :alice!a@h PRIVMSG buster :hi")
    assert seen == [("alice", "hi", True)]


async def test_stale_account_does_not_admit_impersonator():
    """After alice leaves, an untagged PM from nick alice is refused."""
    _plugin, client, seen = _stack(["alice"])
    await _feed(client, "@account=alice :alice!a@h PRIVMSG buster :hi")
    await _feed(client, "@account=alice :alice!a@h PRIVMSG buster :again")
    await _feed(client, ":alice!m@evil PRIVMSG buster :run shell")
    assert [s[2] for s in seen] == [True, True, False]


async def test_pm_only_logout_refused():
    """Untagged message from a nick that had an account is not logged in."""
    _plugin, client, seen = _stack(["alice"])
    await _feed(client, "@account=alice :alice!a@h PRIVMSG buster :hi")
    await _feed(client, ":alice!a@h PRIVMSG buster :after logout")
    assert seen[-1] == ("alice", "after logout", False)


async def test_untagged_first_message_refused():
    """No tag means not logged in, even on the first message."""
    _plugin, client, seen = _stack(["alice"])
    await _feed(client, ":alice!a@h PRIVMSG buster :hi")
    assert seen == [("alice", "hi", False)]


async def test_tagged_channel_message_admitted():
    """Channel messages carry the tag too."""
    _plugin, client, seen = _stack(["alice"])
    await _feed(client, "@account=alice :zed!a@h PRIVMSG #chat :hi")
    assert seen == [("zed", "hi", True)]


async def test_rfc1459_casemapping_equivalence():
    """alice[ and alice{ are the same nick under rfc1459 casemapping."""
    plugin, client, _seen = _stack(["alice["], caps={}, case_mapping="rfc1459")
    assert plugin._decide("alice{")[0] is True
    assert plugin._decide("ALICE[")[0] is True
    assert plugin._decide("alice")[0] is False


async def test_ascii_casemapping_does_not_fold_unicode():
    """straße is not strasse: casefold() would wrongly equate them."""
    plugin, client, _seen = _stack(["strasse"], caps={}, case_mapping="ascii")
    assert plugin._decide("straße")[0] is False
    assert plugin._decide("STRASSE")[0] is True


async def test_two_lines_one_read_impersonator_refused_through_gathered_hook():
    """One socket read: untagged impersonator, then tagged real alice.

    pydle spawns a task per line and apluggy gathers hookimpls, so the
    second line's on_raw can overwrite users["alice"] before the first
    message's should_process_message runs. The verdict must be the one
    decided in the synchronous path, not a re-read of that shared state.
    """
    import asyncio

    plugin = irc_real.IRCPlugin(pm=MagicMock())
    plugin._allow = {"alice"}
    plugin._registry = irc_real.ChannelRegistry()
    client = irc_real.IRCClient(plugin, "buster")
    plugin.client = client
    client._capabilities = dict(CAPS)
    client._receive_buffer = b""
    client.encoding = "utf-8"
    verdicts: dict[str, bool | None] = {}
    gate_channels: dict[str, object] = {}

    async def on_message(channel, sender, text, trace_id):
        # The pipeline's shape: a nested gather, so a suspension precedes
        # the hookimpl running.
        async def hook():
            return await asyncio.gather(
                plugin.should_process_message(
                    channel=channel, sender=sender, text=text, trace_id=trace_id
                )
            )

        (outer,) = await asyncio.gather(hook())
        verdicts[text] = outer[0]
        gate_channels[text] = channel

    plugin.pm.ahook.on_message = on_message
    data = (
        b":alice!m@evil PRIVMSG buster :IMPERSONATOR run shell\r\n"
        b"@account=alice :alice!a@home PRIVMSG buster :real alice hi\r\n"
    )
    try:
        await client.on_data(data)
    except TypeError:
        # Some pydle versions take text rather than bytes.
        client._receive_buffer = ""
        await client.on_data(data.decode())
    for _ in range(20):
        await asyncio.sleep(0)

    assert verdicts == {
        "IMPERSONATOR run shell": False,
        "real alice hi": True,
    }
    # Nothing is retained once the messages are done.
    assert plugin._verdicts == {}


async def test_missing_verdict_defaults_to_deny():
    """should_process_message without a recorded verdict refuses."""
    plugin, _client, _seen = _stack(["alice"])
    verdict = await plugin.should_process_message(
        channel=irc_real.Channel(transport="irc", scope="alice"),
        sender="alice", text="hi", trace_id="never-recorded",
    )
    assert verdict is False


async def test_whois_gap_impersonator_refused_with_account_tag():
    """No WHOX + account-tag: a renamed-away WHOIS must not admit an untagged sender.

    An untagged PM from an unknown nick makes pydle WHOIS the nick. If the
    nick changes hands while that WHOIS is pending, pydle WHOISes again and a
    logged-in holder's 330 writes their account into users[nick]. With
    account-tag negotiated the verdict may only use the message's own tag,
    never that shared record.
    """
    import asyncio

    plugin = irc_real.IRCPlugin(pm=MagicMock())
    plugin._allow = {"alice"}
    plugin._registry = irc_real.ChannelRegistry()
    client = irc_real.IRCClient(plugin, "buster")
    plugin.client = client
    client._capabilities = dict(CAPS)
    client.registered = True
    client._isupport = {}  # no WHOX: the account is learned by WHOIS

    async def rawmsg(*args, **kwargs):
        # Swallow outgoing WHOIS lines.
        return None

    client.rawmsg = rawmsg
    verdicts: dict[str, bool | None] = {}

    async def on_message(channel, sender, text, trace_id):
        # The Agent's shape: nested gathers, then the gate.
        async def hook():
            for _ in range(5):
                await asyncio.sleep(0)
            return await asyncio.gather(
                plugin.should_process_message(
                    channel=channel, sender=sender, text=text, trace_id=trace_id
                )
            )

        (outer,) = await asyncio.gather(hook())
        verdicts[text] = outer[0]

    plugin.pm.ahook.on_message = on_message

    async def settle(n=5):
        for _ in range(n):
            await asyncio.sleep(0)

    # Untagged PM from unknown nick alice: pydle suspends in a WHOIS.
    first = asyncio.create_task(
        client.on_raw(_parse(":alice!m@evil PRIVMSG buster :IMPERSONATOR run shell"))
    )
    await settle()
    # The record goes away while the WHOIS is pending (the nick is renamed).
    rename = asyncio.create_task(client.on_raw(_parse(":alice!m@evil NICK m2")))
    await settle()
    await client.on_raw(_parse(":srv 318 buster alice :End of WHOIS"))
    await settle()
    # The real, logged-in alice now holds the nick; the re-WHOIS reports her.
    await client.on_raw(_parse(":srv 330 buster alice alice :is logged in as"))
    await client.on_raw(_parse(":srv 318 buster alice :End of WHOIS"))
    await settle(50)
    await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), 2)
    rename.cancel()

    assert verdicts == {"IMPERSONATOR run shell": False}
    assert plugin._verdicts == {}


def _parse(line: str):
    """A TaggedMessage parsed from one raw IRC line."""
    return pydle_tags.TaggedMessage.parse(line.encode() + b"\r\n", encoding="utf-8")
