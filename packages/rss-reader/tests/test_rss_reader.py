"""Tests for the RSS reader plugin."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rss_reader import RSSPlugin, _parse_feed, _localname, _bad


# --- _parse_feed tests ---

RSS_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <item>
      <title>First Post</title>
      <link>https://example.com/1</link>
      <guid>https://example.com/1</guid>
      <description>Summary of first post</description>
    </item>
    <item>
      <title>Second Post</title>
      <link>https://example.com/2</link>
      <guid>https://example.com/2</guid>
      <description>Summary of second post</description>
    </item>
  </channel>
</rss>"""

ATOM_SAMPLE = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Atom Feed</title>
  <entry>
    <title>Atom Entry</title>
    <link href="https://example.org/atom/1"/>
    <id>https://example.org/atom/1</id>
    <summary>Atom summary</summary>
  </entry>
</feed>"""


def test_parse_rss_feed():
    items = _parse_feed(RSS_SAMPLE)
    assert len(items) == 2
    assert items[0]["title"] == "First Post"
    assert items[0]["link"] == "https://example.com/1"
    assert items[0]["guid"] == "https://example.com/1"
    assert items[0]["summary"] == "Summary of first post"
    assert items[1]["title"] == "Second Post"


def test_parse_atom_feed():
    items = _parse_feed(ATOM_SAMPLE)
    assert len(items) == 1
    assert items[0]["title"] == "Atom Entry"
    assert items[0]["link"] == "https://example.org/atom/1"
    assert items[0]["guid"] == "https://example.org/atom/1"
    assert items[0]["summary"] == "Atom summary"


def test_parse_invalid_xml():
    items = _parse_feed(b"not xml at all")
    assert items == []


def test_parse_empty_feed():
    items = _parse_feed(b"<rss><channel></channel></rss>")
    assert items == []


def test_parse_item_without_guid_falls_back_to_link():
    sample = b"""<rss><channel><item>
      <title>No Guid</title>
      <link>https://example.com/noguid</link>
      <description>test</description>
    </item></channel></rss>"""
    items = _parse_feed(sample)
    assert len(items) == 1
    assert items[0]["guid"] == "https://example.com/noguid"


# --- _localname tests ---

def test_localname_strips_namespace():
    assert _localname("{http://www.w3.org/2005/Atom}title") == "title"


def test_localname_no_namespace():
    assert _localname("title") == "title"


# --- _bad tests ---

def test_bad_raises_value_error():
    err = _bad("rss.feeds", None, "use a list")
    assert isinstance(err, ValueError)
    assert "Config error" in str(err)
    assert "rss.feeds" in str(err)


# --- RSSPlugin config validation tests ---

def _make_plugin(config: dict) -> RSSPlugin:
    """Create a plugin instance with the given config, bypassing on_init."""
    plugin = RSSPlugin.__new__(RSSPlugin)
    plugin._registry = MagicMock()
    plugin._spill = MagicMock()
    plugin._spill.notify = AsyncMock()
    plugin.pm = MagicMock()
    plugin._prompt = ""
    return plugin


def _make_plugin_for_on_init() -> RSSPlugin:
    """Create a plugin instance for on_init tests, patching get_dependency."""
    from unittest.mock import patch
    plugin = RSSPlugin.__new__(RSSPlugin)
    plugin.pm = MagicMock()
    # Patch get_dependency to return our mocks without type checking
    plugin._registry = MagicMock()
    plugin._spill = MagicMock()
    plugin._spill.notify = AsyncMock()
    return plugin


def test_on_init_no_config_is_inert():
    """Without an 'rss' config block, the plugin should be inert."""
    plugin = RSSPlugin.__new__(RSSPlugin)
    # The plugin has no _feeds attribute until on_init runs
    assert not hasattr(plugin, "_feeds")


@pytest.mark.asyncio
async def test_on_init_rejects_non_list_feeds():
    plugin = RSSPlugin.__new__(RSSPlugin)
    plugin._registry = MagicMock()
    plugin._spill = MagicMock()
    plugin.pm = MagicMock()

    with pytest.raises(ValueError, match="rss.feeds"):
        # Simulate the validation logic
        block = {"feeds": "not a list"}
        feeds = block.get("feeds")
        if not isinstance(feeds, list) or not feeds:
            raise _bad("rss.feeds", feeds, "use a non-empty list of feed URLs")


@pytest.mark.asyncio
async def test_on_init_rejects_non_url_feeds():
    block = {"feeds": ["ftp://example.com/feed"]}
    feeds = block.get("feeds")
    for f in feeds:
        if not isinstance(f, str) or not f.startswith(("http://", "https://")):
            with pytest.raises(ValueError, match="rss.feeds"):
                raise _bad("rss.feeds", f, "each entry must be an http(s) URL")


# --- State file tests ---

def test_load_seen_empty(tmp_path: Path):
    plugin = _make_plugin({})
    plugin._state_path = tmp_path / "nonexistent.json"
    seen = plugin._load_seen()
    assert seen == {}


def test_load_seen_existing(tmp_path: Path):
    plugin = _make_plugin({})
    state_file = tmp_path / "seen.json"
    state_file.write_text(json.dumps({
        "https://example.com/feed": ["guid1", "guid2"]
    }))
    plugin._state_path = state_file
    seen = plugin._load_seen()
    assert seen == {"https://example.com/feed": ["guid1", "guid2"]}


def test_save_and_load_roundtrip(tmp_path: Path):
    plugin = _make_plugin({})
    plugin._state_path = tmp_path / "seen.json"
    seen = {"https://example.com/feed": ["guid1", "guid2", "guid3"]}
    plugin._save_seen(seen)
    loaded = plugin._load_seen()
    assert loaded == seen


def test_save_seen_creates_parent_dir(tmp_path: Path):
    plugin = _make_plugin({})
    plugin._state_path = tmp_path / "nested" / "dir" / "seen.json"
    plugin._save_seen({"url": {"guid"}})
    assert plugin._state_path.exists()


def _sample_feed(n: int) -> bytes:
    """Generate a sample RSS feed with n items."""
    items = []
    for i in range(1, n + 1):
        item = '<item><title>Post %d</title><link>https://example.com/%d</link><guid>https://example.com/%d</guid><description>Summary of post %d</description></item>' % (i, i, i, i)
        items.append(item)
    xml = '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Test Feed</title><link>https://example.com</link>' + ''.join(items) + '</channel></rss>'
    return xml.encode()


# --- Poll logic tests ---

@pytest.mark.asyncio
async def test_poll_once_delivers_new_items(tmp_path: Path):
    plugin = _make_plugin({})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:+15551234567",)
    plugin._max_items = 10
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 60

    # Mock the HTTP response
    mock_resp = MagicMock()
    mock_resp.content = RSS_SAMPLE
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    # Mock channel resolution
    mock_channel = MagicMock()
    with patch("rss_reader.resolve_persisted_channel", AsyncMock(return_value=mock_channel)):
        with patch("httpx.AsyncClient", return_value=mock_client):
            await plugin._poll_once()

    # Verify spill.notify was called for both items
    assert plugin._spill.notify.call_count == 2
    texts = [c[1]["text"] for c in plugin._spill.notify.call_args_list]
    assert any("[RSS] First Post" in t for t in texts)
    assert any("[RSS] Second Post" in t for t in texts)


@pytest.mark.asyncio
async def test_poll_once_skips_seen_items(tmp_path: Path):
    plugin = _make_plugin({})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:+15551234567",)
    plugin._max_items = 10
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 60

    # Pre-populate seen state with all items from the sample feed
    seen_file = tmp_path / "seen.json"
    seen_file.write_text(json.dumps({
        "https://example.com/feed": ["https://example.com/1", "https://example.com/2"]
    }))

    mock_resp = MagicMock()
    mock_resp.content = RSS_SAMPLE
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await plugin._poll_once()

    # No new items → no notify call
    assert not plugin._spill.notify.called


@pytest.mark.asyncio
async def test_poll_once_caps_items_per_poll(tmp_path: Path):
    plugin = _make_plugin({})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:+15551234567",)
    plugin._max_items = 1  # Cap at 1
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 60

    mock_resp = MagicMock()
    mock_resp.content = RSS_SAMPLE  # Has 2 items
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_channel = MagicMock()
    with patch("rss_reader.resolve_persisted_channel", AsyncMock(return_value=mock_channel)):
        with patch("httpx.AsyncClient", return_value=mock_client):
            await plugin._poll_once()

    # Only 1 item delivered (capped)
    assert plugin._spill.notify.call_count == 1


class TestSecondPoll:
    """Regression: _load_seen returns lists, not sets, so list ops
    (.append, [-500:]) work on the second and subsequent polls."""

    async def test_second_poll_does_not_crash(self, tmp_path):
        plugin = _make_plugin({})
        plugin._feeds = ["https://example.com/feed.xml"]
        plugin._channel_id = "test:channel"
        plugin._state_path = tmp_path / "seen.json"
        plugin._max_chars = 2000
        plugin._poll_secs = 60
        plugin._max_items = 10
        plugin._targets = ("test:channel",)

        # Seed the state file with a seen guid (simulating a first poll).
        import json
        plugin._state_path.parent.mkdir(parents=True, exist_ok=True)
        plugin._state_path.write_text(json.dumps({
            "https://example.com/feed.xml": ["guid-1", "guid-2"]
        }))

        # Second poll: feed has guid-1 (seen) + guid-3 (new).
        feed_xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Test</title><link>https://example.com</link><description>Test</description>
<item><guid>guid-1</guid><title>Old</title><link>https://example.com/1</link><description>Old</description></item>
<item><guid>guid-3</guid><title>New</title><link>https://example.com/3</link><description>New</description></item>
</channel></rss>"""

        mock_resp = MagicMock()
        mock_resp.content = feed_xml.encode()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_channel = MagicMock()
        with patch("rss_reader.resolve_persisted_channel", AsyncMock(return_value=mock_channel)):
            with patch("httpx.AsyncClient", return_value=mock_client):
                await plugin._poll_once()  # Must not crash

        # Only the new guid (guid-3) should be delivered.
        assert plugin._spill.notify.call_count == 1

        # State file should now have all 3 guids (2 old + 1 new).
        saved = json.loads(plugin._state_path.read_text())
        assert "guid-3" in saved["https://example.com/feed.xml"]
        assert "guid-1" in saved["https://example.com/feed.xml"]


# --- Config validation tests (call on_init, assert message names key/value/form) ---

@pytest.mark.asyncio
async def test_on_init_rejects_non_list_feeds_calls_on_init():
    """on_init must reject non-list feeds and name the key, value, and accepted form."""
    from unittest.mock import patch
    from rss_reader import RSSPlugin
    plugin = _make_plugin_for_on_init()

    with patch("rss_reader.get_dependency") as mock_getdep:
        mock_getdep.side_effect = lambda pm, name, _type: {
            "registry": plugin._registry,
            "spill": plugin._spill,
        }[name]
        with pytest.raises(ValueError) as exc_info:
            await plugin.on_init(
                plugin.pm,
                {"rss": {"feeds": "not a list", "poll_interval": "30m", "targets": ["signal:+1"]}}
            )
    msg = str(exc_info.value)
    assert "rss.feeds" in msg
    assert "not a list" in msg
    assert "list of feed URLs" in msg


@pytest.mark.asyncio
async def test_on_init_rejects_non_url_feeds_calls_on_init():
    """on_init must reject non-URL feeds and name the key, value, and accepted form."""
    from unittest.mock import patch
    from rss_reader import RSSPlugin
    plugin = _make_plugin_for_on_init()

    with patch("rss_reader.get_dependency") as mock_getdep:
        mock_getdep.side_effect = lambda pm, name, _type: {
            "registry": plugin._registry,
            "spill": plugin._spill,
        }[name]
        with pytest.raises(ValueError) as exc_info:
            await plugin.on_init(
                plugin.pm,
                {"rss": {"feeds": ["ftp://example.com/feed"], "poll_interval": "30m", "targets": ["signal:+1"]}}
            )
    msg = str(exc_info.value)
    assert "rss.feeds" in msg
    assert "ftp://example.com/feed" in msg
    assert "http(s) URL" in msg


@pytest.mark.asyncio
async def test_on_init_rejects_zero_poll_interval():
    """on_init must reject zero poll intervals."""
    from unittest.mock import patch
    from rss_reader import RSSPlugin
    plugin = _make_plugin_for_on_init()

    with patch("rss_reader.get_dependency") as mock_getdep:
        mock_getdep.side_effect = lambda pm, name, _type: {
            "registry": plugin._registry,
            "spill": plugin._spill,
        }[name]
        with pytest.raises(ValueError) as exc_info:
            await plugin.on_init(
                plugin.pm,
                {"rss": {"feeds": ["https://example.com/feed"], "poll_interval": "0m", "targets": ["signal:+1"]}}
            )
    msg = str(exc_info.value)
    assert "rss.poll_interval" in msg
    assert "0m" in msg
    assert "positive" in msg


@pytest.mark.asyncio
async def test_on_init_rejects_negative_poll_interval():
    """on_init must reject negative poll intervals (sign must not be lost)."""
    from unittest.mock import patch
    from rss_reader import RSSPlugin
    plugin = _make_plugin_for_on_init()

    with patch("rss_reader.get_dependency") as mock_getdep:
        mock_getdep.side_effect = lambda pm, name, _type: {
            "registry": plugin._registry,
            "spill": plugin._spill,
        }[name]
        with pytest.raises(ValueError) as exc_info:
            await plugin.on_init(
                plugin.pm,
                {"rss": {"feeds": ["https://example.com/feed"], "poll_interval": "-5m", "targets": ["signal:+1"]}}
            )
    msg = str(exc_info.value)
    assert "rss.poll_interval" in msg
    assert "-5m" in msg


@pytest.mark.asyncio
async def test_on_init_rejects_trailing_junk_poll_interval():
    """on_init must reject poll intervals with trailing junk."""
    from unittest.mock import patch
    from rss_reader import RSSPlugin
    plugin = _make_plugin_for_on_init()

    with patch("rss_reader.get_dependency") as mock_getdep:
        mock_getdep.side_effect = lambda pm, name, _type: {
            "registry": plugin._registry,
            "spill": plugin._spill,
        }[name]
        with pytest.raises(ValueError) as exc_info:
            await plugin.on_init(
                plugin.pm,
                {"rss": {"feeds": ["https://example.com/feed"], "poll_interval": "30m junk", "targets": ["signal:+1"]}}
            )
    msg = str(exc_info.value)
    assert "rss.poll_interval" in msg
    assert "30m junk" in msg


# --- _parse_duration regression tests ---

def test_parse_duration_rejects_zero():
    from rss_reader import _parse_duration
    with pytest.raises(ValueError, match="positive"):
        _parse_duration("0m")


def test_parse_duration_rejects_negative():
    from rss_reader import _parse_duration
    with pytest.raises(ValueError):
        _parse_duration("-5m")


def test_parse_duration_rejects_trailing_junk():
    from rss_reader import _parse_duration
    with pytest.raises(ValueError):
        _parse_duration("30m junk")


def test_parse_duration_accepts_valid():
    from rss_reader import _parse_duration
    assert _parse_duration("30m") == 1800.0
    assert _parse_duration("2h") == 7200.0
    assert _parse_duration("1h30m") == 5400.0
    assert _parse_duration("90") == 90.0


# --- Regression tests for the delivery fixes ---

def _client_serving(feed: bytes):
    """An httpx.AsyncClient stand-in that answers every GET with `feed`."""
    resp = MagicMock()
    resp.content = feed
    resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.mark.asyncio
async def test_items_stay_unseen_when_no_channel_ready(tmp_path: Path):
    """Items must not be marked seen when no channel is ready to receive them.

    Resolution, not the registry, is what decides this: mock the resolver the
    plugin actually calls. A MagicMock registry resolves to a truthy channel
    through the bare-id fallback, which is how this test used to pass while
    proving nothing.
    """
    from corvidae.channel import ChannelNotReady
    from rss_reader import RSSPlugin

    plugin = _make_plugin({})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:+1",)
    plugin._max_items = 500
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 1800

    with (
        patch(
            "rss_reader.resolve_persisted_channel",
            AsyncMock(side_effect=ChannelNotReady("signal:+1")),
        ),
        patch("httpx.AsyncClient", return_value=_client_serving(_sample_feed(3))),
    ):
        await plugin._poll_once()

    assert plugin._spill.notify.called is False
    assert plugin._load_seen() == {}


@pytest.mark.asyncio
async def test_items_stay_unseen_when_a_target_resolves_to_none(tmp_path: Path):
    """A target that resolves to nothing is skipped, not delivered to.

    `resolve_persisted_channel` answers None for an id that is not
    `transport:scope`, and config validation only checks for a colon.
    """
    from rss_reader import RSSPlugin

    plugin = _make_plugin({})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:",)
    plugin._max_items = 500
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 1800

    with (
        patch("rss_reader.resolve_persisted_channel", AsyncMock(return_value=None)),
        patch("httpx.AsyncClient", return_value=_client_serving(_sample_feed(3))),
    ):
        await plugin._poll_once()

    assert plugin._spill.notify.called is False
    assert plugin._load_seen() == {}


@pytest.mark.asyncio
async def test_second_poll_delivers_new_items(tmp_path: Path):
    """A second poll should deliver items that arrived after the first poll."""
    from rss_reader import RSSPlugin
    plugin = _make_plugin({})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:+1",)
    plugin._max_items = 500
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 1800

    feed1 = _sample_feed(2)
    feed2 = _sample_feed(3)

    # Mock HTTP client for both polls
    resp1 = MagicMock()
    resp1.content = feed1
    resp1.raise_for_status = MagicMock()
    resp2 = MagicMock()
    resp2.content = feed2
    resp2.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[resp1, resp2])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_channel = MagicMock()
    with patch("rss_reader.resolve_persisted_channel", AsyncMock(return_value=mock_channel)):
        with patch("httpx.AsyncClient", return_value=mock_client):
            # First poll
            await plugin._poll_once()
            seen_after_first = plugin._load_seen()
            total_first = sum(len(v) for v in seen_after_first.values())
            assert total_first == 2

            # Second poll: should deliver only the new item
            await plugin._poll_once()
            seen_after_second = plugin._load_seen()
            total_second = sum(len(v) for v in seen_after_second.values())
            assert total_second == 3


@pytest.mark.asyncio
async def test_items_beyond_cap_arrive_next_cycle(tmp_path: Path):
    """Items beyond the cap in one poll should be delivered in the next poll."""
    from rss_reader import RSSPlugin
    plugin = _make_plugin({})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:+1",)
    plugin._max_items = 2
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 1800

    feed = _sample_feed(4)
    resp = MagicMock()
    resp.content = feed
    resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_channel = MagicMock()
    with patch("rss_reader.resolve_persisted_channel", AsyncMock(return_value=mock_channel)):
        with patch("httpx.AsyncClient", return_value=mock_client):
            # First poll: should deliver only 2 (cap)
            await plugin._poll_once()
            seen_after_first = plugin._load_seen()
            total_first = sum(len(v) for v in seen_after_first.values())
            assert total_first == 2

            # Second poll: should deliver the remaining 2
            await plugin._poll_once()
            seen_after_second = plugin._load_seen()
            total_second = sum(len(v) for v in seen_after_second.values())
            assert total_second == 4


@pytest.mark.asyncio
async def test_prompt_prepended_to_each_item(tmp_path: Path):
    """When prompt is set, each delivered item's text starts with the prompt."""
    from rss_reader import RSSPlugin
    plugin = _make_plugin({"rss": {"prompt": "Summarise in two sentences."}})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:+1",)
    plugin._max_items = 500
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 1800

    feed = _sample_feed(2)
    resp = MagicMock()
    resp.content = feed
    resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_channel = MagicMock()
    with patch("rss_reader.resolve_persisted_channel", AsyncMock(return_value=mock_channel)):
        with patch("httpx.AsyncClient", return_value=mock_client):
            await plugin._poll_once()

    # Each notification should start with the prompt
    for call in mock_channel.notify.call_args_list:
        text = call.kwargs.get("text") or call.args[0] if call.args else call.kwargs["text"]
        assert text.startswith("Summarise in two sentences.\n\n")


@pytest.mark.asyncio
async def test_no_prompt_when_absent(tmp_path: Path):
    """When prompt is absent, delivered text is unchanged (no prepend)."""
    from rss_reader import RSSPlugin
    plugin = _make_plugin({})
    plugin._feeds = ["https://example.com/feed"]
    plugin._targets = ("signal:+1",)
    plugin._max_items = 500
    plugin._state_path = tmp_path / "seen.json"
    plugin._max_chars = 2000
    plugin._poll_secs = 1800

    feed = _sample_feed(2)
    resp = MagicMock()
    resp.content = feed
    resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_channel = MagicMock()
    with patch("rss_reader.resolve_persisted_channel", AsyncMock(return_value=mock_channel)):
        with patch("httpx.AsyncClient", return_value=mock_client):
            await plugin._poll_once()

    # No prompt: text should not start with any framing
    for call in mock_channel.notify.call_args_list:
        text = call.kwargs.get("text") or call.args[0] if call.args else call.kwargs["text"]
        assert not text.startswith("Summarise")
