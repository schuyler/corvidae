"""Corvidae plugin: poll RSS/Atom feeds and deliver new items as notifications.

Fetches configured feeds on a polling interval, tracks seen items by guid/link
in a local state file, and delivers new items to target channels through the
kernel's spill service.

Config (agent.yaml)::

    rss:
      feeds:
        - https://example.com/feed.xml
        - https://example.org/atom.xml
      poll_interval: 30m
      targets:
        - signal:+15551234567
      max_items_per_poll: 10
      state_file: state/rss/seen.json

Origin ``rss:<feed-url>`` in the cognition plugins:

- appraisal: runs stage 2 as for any non-critique exchange.
- outcome_log: records the origin verbatim in ``exchange_log.origin``.
- gates: non-``user``, so the decide gate may withhold a low-salience final.
- critique: not eligible, as for heartbeat.
- funnel: keys stubs on its own origins only; unaffected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

from corvidae.channel import (
    ChannelNotReady,
    ChannelRegistry,
    resolve_persisted_channel,
)
from corvidae.hooks import CorvidaePlugin, get_dependency, hookimpl
from corvidae.spill import NotificationSpill
from corvidae.tuning import config_value


def _parse_duration(text: str) -> float:
    """Parse a duration string like `30m`, `2h`, `1h30m` to seconds."""
    cleaned = str(text).strip().lower()
    if cleaned.isdigit():
        return float(cleaned)
    total = 0.0
    found = False
    for unit, factor in [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]:
        import re
        match = re.search(rf"(\d+(?:\.\d+)?)\s*{unit}", cleaned)
        if match:
            total += float(match.group(1)) * factor
            found = True
    if not found:
        raise ValueError(f"rss: invalid duration {text!r}; use e.g. 30m, 2h, 1h30m")
    return total


logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = "30m"
DEFAULT_MAX_ITEMS = 10
DEFAULT_STATE_FILE = "state/rss/seen.json"
DEFAULT_MAX_CHARS = 2000


def _bad(key: str, value: object, want: str) -> ValueError:
    return ValueError(f"Config error: {key} is {value!r}; {want} — check agent.yaml")


def _parse_feed(xml_bytes: bytes) -> list[dict[str, str]]:
    """Parse RSS 2.0 or Atom feed XML into a list of {title, link, guid, summary}."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    items: list[dict[str, str]] = []

    # RSS 2.0: <rss><channel><item>...</item></channel></rss>
    # Atom: <feed><entry>...</entry></feed>
    # Namespace-agnostic: match on local names.

    for elem in root.iter():
        tag = _localname(elem.tag)
        if tag not in ("item", "entry"):
            continue
        item: dict[str, str] = {}
        for child in elem:
            ctag = _localname(child.tag)
            if ctag == "title":
                item["title"] = (child.text or "").strip()
            elif ctag == "link":
                # Atom <link> uses href attribute; RSS <link> uses text
                href = child.get("href", "")
                text = (child.text or "").strip()
                item["link"] = href or text
            elif ctag in ("guid", "id"):
                item["guid"] = (child.text or "").strip() or item.get("link", "")
            elif ctag in ("description", "summary", "content"):
                if "summary" not in item:
                    item["summary"] = (child.text or "").strip()[:500]
        if item.get("title") or item.get("link"):
            item.setdefault("guid", item.get("link", ""))
            items.append(item)

    return items


def _localname(tag: str) -> str:
    """Strip XML namespace: '{http://...}local' -> 'local'."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


class RSSPlugin(CorvidaePlugin):
    """Polls RSS/Atom feeds and delivers new items as notifications."""

    depends_on = frozenset({"registry", "spill"})
    model_visible_annotations = frozenset({"origin"})

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        self._registry = get_dependency(pm, "registry", ChannelRegistry)
        self._spill = get_dependency(pm, "spill", NotificationSpill)

        block = config.get("rss")
        if block is None:
            # No config: plugin is inert.
            self._feeds: list[str] = []
            self._poll_secs: float = 0
            self._targets: tuple[str, ...] = ()
            self._max_items: int = DEFAULT_MAX_ITEMS
            self._state_path: Path = Path(DEFAULT_STATE_FILE)
            self._max_chars: int = DEFAULT_MAX_CHARS
            self._task: asyncio.Task | None = None
            return

        if not isinstance(block, dict):
            raise _bad("rss", block, "it must be a mapping")

        feeds = block.get("feeds")
        if feeds is None:
            raise _bad("rss.feeds", feeds, "use a list of feed URLs")
        if not isinstance(feeds, list) or not feeds:
            raise _bad("rss.feeds", feeds, "use a non-empty list of feed URLs")
        for f in feeds:
            if not isinstance(f, str) or not f.startswith(("http://", "https://")):
                raise _bad("rss.feeds", f, "each entry must be an http(s) URL")
        self._feeds = feeds

        interval = block.get("poll_interval", DEFAULT_POLL_INTERVAL)
        if not isinstance(interval, str):
            raise _bad("rss.poll_interval", interval, "use a duration string like 30m")
        self._poll_secs = _parse_duration(interval)

        targets = block.get("targets")
        if targets is None:
            raise _bad("rss.targets", targets, "use a list of channel ids")
        if not isinstance(targets, list) or not targets:
            raise _bad("rss.targets", targets, "use a non-empty list of channel ids such as signal:+15551234567")
        for cid in targets:
            if not isinstance(cid, str) or ":" not in cid:
                raise _bad("rss.targets", cid, "use transport:scope, e.g. signal:+15551234567")
        self._targets = tuple(dict.fromkeys(targets))

        self._max_items = block.get("max_items_per_poll", DEFAULT_MAX_ITEMS)
        if not isinstance(self._max_items, int) or self._max_items < 1:
            raise _bad("rss.max_items_per_poll", self._max_items, "use a positive integer")

        self._state_path = Path(block.get("state_file", DEFAULT_STATE_FILE))
        self._max_chars = block.get("max_payload_chars", DEFAULT_MAX_CHARS)
        if not isinstance(self._max_chars, int) or self._max_chars < 1:
            raise _bad("rss.max_payload_chars", self._max_chars, "use a positive integer")

        self._task = None

    @hookimpl
    async def on_start(self) -> None:
        if not self._feeds or self._poll_secs <= 0:
            return
        self._task = asyncio.create_task(self._poll_loop(), name="rss-poll")
        logger.info(
            "rss: polling %d feeds every %.0fs → %s",
            len(self._feeds), self._poll_secs, ", ".join(self._targets),
        )

    @hookimpl
    async def on_stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._poll_once()
            except Exception:
                logger.exception("rss: poll cycle failed")
            await asyncio.sleep(self._poll_secs)

    async def _poll_once(self) -> None:
        seen = self._load_seen()
        new_items: list[tuple[str, dict[str, str]]] = []

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            for url in self._feeds:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                except Exception:
                    logger.warning("rss: fetch failed for %s", url)
                    continue
                items = _parse_feed(resp.content)
                for item in items:
                    guid = item.get("guid", "")
                    if guid and guid not in seen.get(url, []):
                        new_items.append((url, item))

        if not new_items:
            return

        # Cap items per poll BEFORE marking seen, so items beyond the cap
        # are retried on the next poll cycle.
        new_items = new_items[: self._max_items]

        # Mark only the delivered items as seen.
        channels = []
        for cid in self._targets:
            try:
                channel = await resolve_persisted_channel(self.pm, self._registry, cid)
            except ChannelNotReady:
                logger.warning("rss: skipped channel %s (not ready)", cid)
                continue
            channels.append(channel)

        if not channels:
            # No channel ready (normal at startup). Items stay unseen so
            # they are delivered on the next poll.
            return

        delivered: list[tuple[str, str]] = []
        for feed_url, item in new_items:
            title = item.get("title", "(untitled)")
            link = item.get("link", "")
            summary = item.get("summary", "")
            text = f"[RSS] {title}\n{link}"
            if summary:
                text += f"\n{summary}"
            origin = f"rss:{feed_url}"
            try:
                await self._spill.notify(
                    channels, source=origin, text=text,
                    annotations={"origin": origin}, limit=self._max_chars,
                )
                delivered.append((feed_url, item.get("guid", "")))
            except Exception as exc:
                logger.warning("rss: delivery failed for %s: %s", feed_url, exc)

        # Mark items seen only after delivery succeeds.
        for url, guid in delivered:
            if guid:
                seen.setdefault(url, []).append(guid)

        # Trim seen lists to prevent unbounded growth (keep last 500 per feed).
        for url in seen:
            if len(seen[url]) > 500:
                seen[url] = seen[url][-500:]
        self._save_seen(seen)
        logger.info("rss: delivered %d new items", len(new_items))

    def _load_seen(self) -> dict[str, list[str]]:
        if not self._state_path.exists():
            return {}
        try:
            raw = json.loads(self._state_path.read_text())
            # JSON has list values; preserve insertion order.
            return {k: list(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError):
            logger.warning("rss: could not read state file %s, starting fresh", self._state_path)
            return {}

    def _save_seen(self, seen: dict[str, list[str]]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        # Lists → JSON (order preserved for the [-500:] trim).
        raw = {k: list(v) for k, v in seen.items()}
        self._state_path.write_text(json.dumps(raw, indent=2))
