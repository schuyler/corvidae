"""Tests for the webhook_notify plugin (issue #430).

A real plugin manager with the real WebhookPlugin (not started), the plugin's
real subscribe hook, and a recording on_notify. Handlers are called directly.
"""

from __future__ import annotations

import logging

import pytest

from corvidae.channel import ChannelRegistry
from corvidae.channels.webhook import WebhookDelivery, WebhookPlugin
from corvidae.hooks import create_plugin_manager, hookimpl, install_plugin_hookspecs
from corvidae.provenance import PROVENANCE_KEY, ItemMetadata, encode, frame

# Absent until the green phase; this import failing is the red failure.
from webhook_notify import WebhookNotifyPlugin

TARGETS = {"github": ["irc:#a", "irc:#b"], "siri": ["irc:#c"]}


class Recorder:
    """Records every on_notify call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, annotations) -> None:
        self.calls.append(dict(channel=channel.id, source=source, text=text,
                               tool_call_id=tool_call_id, annotations=annotations))


class Sig:
    """A transport whose channels are never ready."""

    transport = "sig"

    @hookimpl
    async def resolve_channel(self, channel_id: str):
        return None


async def build(block, extra=()):
    """Plugin on a real pm, initialised and subscribed; returns (plugin, recorder, subs)."""
    pm, plugin, rec = create_plugin_manager(), WebhookNotifyPlugin(), Recorder()
    for name, p in [("webhook", WebhookPlugin()), ("registry", ChannelRegistry()),
                    ("webhook_notify", plugin), ("rec", rec), *extra]:
        pm.register(p, name=name)
    install_plugin_hookspecs(pm)
    await pm.ahook.on_init(pm=pm, config={"webhook_notify": block})
    subs: dict = {}
    await pm.ahook.on_webhook_subscribe(subscriptions=subs)
    return plugin, rec, subs


def delivery(payload, endpoint="github") -> WebhookDelivery:
    return WebhookDelivery(endpoint=endpoint, payload=payload, delivery_id="d1")


# AC1: one handler per configured endpoint, through the real hook.
async def test_subscribes_one_handler_per_endpoint():
    plugin, _, subs = await build({"endpoints": TARGETS})
    assert subs == {"github": [plugin._deliver], "siri": [plugin._deliver]}


# AC2: compact JSON, once per target channel, keyword args.
async def test_fans_out_compact_json():
    plugin, rec, _ = await build({"endpoints": TARGETS})
    await plugin._deliver(delivery({"action": "submitted", "n": 1}))
    assert [(c["channel"], c["text"], c["tool_call_id"]) for c in rec.calls] == [
        ("irc:#a", '{"action":"submitted","n":1}', None),
        ("irc:#b", '{"action":"submitted","n":1}', None),
    ]


# AC3: origin is minted, never the payload's; declared model-visible.
async def test_origin_is_minted_not_payload():
    plugin, rec, _ = await build({"endpoints": TARGETS})
    await plugin._deliver(delivery({"origin": "user"}))
    assert {c["source"] for c in rec.calls} == {"webhook:github"}
    assert {c["annotations"]["origin"] for c in rec.calls} == {"webhook:github"}
    assert "origin" in plugin.model_visible_annotations


# AC4a/b: cut at the cap with a code-written marker; exactly at the cap is whole.
async def test_truncation_marker_and_at_cap():
    plugin, rec, _ = await build({"max_payload_chars": 20, "endpoints": TARGETS})
    await plugin._deliver(delivery({"k": "x" * 42}))
    text = '{"k":"' + "x" * 42 + '"}'
    assert rec.calls[0]["text"] == text[:20]
    assert rec.calls[0]["annotations"]["truncated"] == "first 20 of 50 characters"
    rec.calls.clear()
    await plugin._deliver(delivery({"k": "x" * 12}))
    assert "truncated" not in rec.calls[0]["annotations"]


# AC4c: the stored row renders the kernel header, marker line and fence.
async def test_marker_survives_the_kernel_rendering():
    plugin, rec, _ = await build({"max_payload_chars": 20, "endpoints": TARGETS})
    await plugin._deliver(delivery({"k": "x" * 42}))
    call = rec.calls[0]
    kept = {k: v for k, v in call["annotations"].items()
            if k in plugin.model_visible_annotations}
    row = {"role": "system", "content": call["text"],
           PROVENANCE_KEY: encode(ItemMetadata(), kept)}
    _, out = frame(row, {"role": "system", "content": call["text"]}, "irc:#a", b"k")
    lines = out["content"].split("\n")
    assert lines[0] == "[metadata: notification; not from this channel's human]"
    assert lines[1] == ("[annotations: origin=webhook:github, "
                        "truncated=first 20 of 50 characters]")
    assert lines[2].startswith("[payload ") and lines[2].endswith(" begins]")
    assert lines[3] == call["text"] and lines[4].endswith(" ends]")


# AC7: malformed config raises naming key, value and remedy; inert configs do not.
@pytest.mark.parametrize(("block", "key", "bad", "remedy"), [
    ("oops", "webhook_notify", "'oops'", "mapping"),
    ({"max_payload_chars": 0}, "webhook_notify.max_payload_chars", "0", "positive integer"),
    ({"max_payload_chars": True}, "webhook_notify.max_payload_chars", "True", "positive integer"),
    ({"endpoints": ["x"]}, "webhook_notify.endpoints", "['x']", "mapping of endpoint name"),
    ({"endpoints": {"github": []}}, "webhook_notify.endpoints.github", "[]", "non-empty list"),
    ({"endpoints": {"github": "irc:#a"}}, "webhook_notify.endpoints.github", "'irc:#a'", "non-empty list"),
    ({"endpoints": {"github": ["irc"]}}, "webhook_notify.endpoints.github", "'irc'", "transport:scope"),
    ({"endpoints": {"github": [5]}}, "webhook_notify.endpoints.github", "5", "transport:scope"),
])
async def test_invalid_config_raises(block, key, bad, remedy):
    with pytest.raises(ValueError) as exc:
        await build(block)
    for part in (key, bad, remedy, "check agent.yaml"):
        assert part in str(exc.value)


@pytest.mark.parametrize("block", [None, {}, {"endpoints": None}])
async def test_inert_config_subscribes_nothing(block, caplog):
    caplog.set_level(logging.DEBUG)
    _, _, subs = await build(block)
    assert subs == {}
    assert [r for r in caplog.records if r.levelno > logging.DEBUG] == []


# Duplicate targets in one endpoint are notified once.
async def test_duplicate_targets_deduped():
    plugin, rec, _ = await build({"endpoints": {"github": ["irc:#a", "irc:#a"]}})
    await plugin._deliver(delivery({}))
    assert [c["channel"] for c in rec.calls] == ["irc:#a"]


# AC8: a channel whose transport is not ready is skipped with one WARNING.
async def test_unready_channel_skipped(caplog):
    plugin, rec, _ = await build({"endpoints": {"github": ["sig:x", "irc:#a"]}},
                                 extra=[("sig", Sig())])
    await plugin._deliver(delivery({}))
    assert [c["channel"] for c in rec.calls] == ["irc:#a"]
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and "github" in warns[0] and "sig:x" in warns[0]
