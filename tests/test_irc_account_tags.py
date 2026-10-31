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

    tests/test_irc_plugin.py imports the module against a mocked pydle, so
    the cached module's IRCClient is not a real pydle client.
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
        # Evaluate the gate at delivery time, as the real pipeline does.
        seen.append((by, message, plugin._is_allowed(by)))

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
    assert plugin._is_allowed("alice{") is True
    assert plugin._is_allowed("ALICE[") is True
    assert plugin._is_allowed("alice") is False


async def test_ascii_casemapping_does_not_fold_unicode():
    """straße is not strasse: casefold() would wrongly equate them."""
    plugin, client, _seen = _stack(["strasse"], caps={}, case_mapping="ascii")
    assert plugin._is_allowed("straße") is False
    assert plugin._is_allowed("STRASSE") is True
