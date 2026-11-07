"""Tests for the webhook_notify plugin (issues #430, #558).

A real plugin manager with the real WebhookPlugin (not started), the plugin's
real subscribe hook, and a recording on_notify. Handlers are called directly.
The kernel's spill service, a built tool registry whose read_file declares it
reads spill files, and a never-started Agent that only collects what
on_notify enqueues stand in for the daemon, so a row's header renders as the
kernel would render it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from corvidae.agent import Agent
from corvidae.channel import ChannelRegistry
from corvidae.channels.webhook import WebhookDelivery, WebhookPlugin
from corvidae.hooks import create_plugin_manager, hookimpl, install_plugin_hookspecs
from corvidae.provenance import frame
from corvidae.spill import SPILL_NAME, NotificationSpill
from corvidae.tool import Tool
from corvidae.tool_collection import ToolCollectionPlugin
from fake_clock import install_kernel_clock

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


class Readers:
    """Registers a read_file-named tool that declares it reads spill files."""

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def read_file(path: str) -> str:
            """Stub tool."""
            return ""
        read_file.reads_spill_files = True  # type: ignore[attr-defined]
        tool_registry.append(Tool.from_function(read_file))


def inbox(pm) -> Agent:
    """A real Agent, never started, whose queues only collect, in ``items``,
    what on_notify enqueues."""
    agent = Agent(pm)
    agent.items = []  # type: ignore[attr-defined]

    class Collect:
        async def enqueue(self, item) -> None:
            agent.items.append(item)  # type: ignore[attr-defined]

    agent._get_or_create_queue = lambda channel: Collect()
    return agent


async def build(base, block, extra=()):
    """Plugin on a real pm, initialised and subscribed; returns (plugin, recorder, subs).

    The config directory is ``base``, so spill files land under
    ``base/state/notification-spill``.
    """
    pm, plugin, rec = create_plugin_manager(), WebhookNotifyPlugin(), Recorder()
    install_kernel_clock(pm, started=False)
    tools, agent = ToolCollectionPlugin(pm), inbox(pm)
    for name, p in [("webhook", WebhookPlugin()), ("registry", ChannelRegistry()),
                    ("tools", tools), ("readers", Readers()), ("spill", NotificationSpill()),
                    ("agent", agent), ("webhook_notify", plugin), ("rec", rec), *extra]:
        pm.register(p, name=name)
    install_plugin_hookspecs(pm)
    await pm.ahook.on_init(pm=pm, config={"webhook_notify": block, "_base_dir": base})
    await tools.rebuild_registry()
    # The annotation keys the Agent reads at startup.
    agent._annotation_keys = frozenset(
        k for p in pm.get_plugins() for k in getattr(p, "model_visible_annotations", ())
    )
    subs: dict = {}
    await pm.ahook.on_webhook_subscribe(subscriptions=subs)
    return plugin, rec, subs


def delivery(payload, endpoint="github", delivery_id="d1") -> WebhookDelivery:
    return WebhookDelivery(endpoint=endpoint, payload=payload, delivery_id=delivery_id)


def spill_dir(base: Path) -> Path:
    """The default spill directory for config directory ``base``."""
    return (base / "state" / "notification-spill").resolve()


def rendered(plugin, n: int) -> list[str]:
    """The lines of the n-th notified row as the kernel renders it."""
    agent = plugin.pm.get_plugin("agent")
    item = agent.items[n]
    msg, _ = agent._build_conversation_message(item)
    _, out = frame(msg, msg, item.channel.id, b"k")
    return out["content"].split("\n")


def cut_line(plugin, n: int) -> str | None:
    """The n-th notified row's cut line, or None without one."""
    return next((line for line in rendered(plugin, n) if line.startswith("[metadata: cut ")), None)


# AC1: one handler per configured endpoint, through the real hook.
async def test_subscribes_one_handler_per_endpoint(tmp_path):
    plugin, _, subs = await build(tmp_path, {"endpoints": TARGETS})
    assert subs == {"github": [plugin._deliver], "siri": [plugin._deliver]}


# AC2: compact JSON, once per target channel, keyword args.
async def test_fans_out_compact_json(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"endpoints": TARGETS})
    await plugin._deliver(delivery({"action": "submitted", "n": 1}))
    assert [(c["channel"], c["text"], c["tool_call_id"]) for c in rec.calls] == [
        ("irc:#a", '{"action":"submitted","n":1}', None),
        ("irc:#b", '{"action":"submitted","n":1}', None),
    ]


# AC3: origin is minted, never the payload's; declared model-visible.
async def test_origin_is_minted_not_payload(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"endpoints": TARGETS})
    await plugin._deliver(delivery({"origin": "user"}))
    assert {c["source"] for c in rec.calls} == {"webhook:github"}
    assert all(c["annotations"] == {"origin": "webhook:github"} for c in rec.calls)
    assert "origin" in plugin.model_visible_annotations


# AC4a/b: cut at the cap with the kernel's cut line; exactly at the cap is whole.
async def test_truncation_marker_and_at_cap(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"max_payload_chars": 20, "endpoints": TARGETS})
    await plugin._deliver(delivery({"k": "x" * 42}))
    text = '{"k":"' + "x" * 42 + '"}'
    assert rec.calls[0]["text"] == text[:20]
    [path] = spill_dir(tmp_path).iterdir()
    assert cut_line(plugin, 0) == (
        f"[metadata: cut to the first 20 of 50 characters; the full text is in {path}]")
    rec.calls.clear()
    await plugin._deliver(delivery({"k": "x" * 12}))
    assert rec.calls[0]["text"] == '{"k":"' + "x" * 12 + '"}'
    assert cut_line(plugin, -1) is None


# AC4c: the stored row renders the kernel header, cut line and fence.
async def test_marker_survives_the_kernel_rendering(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"max_payload_chars": 20, "endpoints": TARGETS})
    await plugin._deliver(delivery({"k": "x" * 42}))
    [path] = spill_dir(tmp_path).iterdir()
    lines = rendered(plugin, 0)
    assert lines[0] == "[metadata: notification; not from this channel's human]"
    assert lines[1] == (
        f"[metadata: cut to the first 20 of 50 characters; the full text is in {path}]")
    assert lines[2] == "[annotations: origin=webhook:github]"
    assert lines[3].startswith("[payload ") and lines[3].endswith(" begins]")
    assert lines[4] == rec.calls[0]["text"] and lines[5].endswith(" ends]")


# a cut delivery to three channels writes one file, named in every header.
async def test_cut_delivery_to_three_channels_writes_one_file(tmp_path):
    plugin, _, _ = await build(tmp_path, {"max_payload_chars": 20,
                                          "endpoints": {"github": ["irc:#a", "irc:#b", "irc:#c"]}})
    await plugin._deliver(delivery({"k": "x" * 42}))
    [path] = spill_dir(tmp_path).iterdir()
    assert [cut_line(plugin, n) for n in range(3)] == [
        f"[metadata: cut to the first 20 of 50 characters; the full text is in {path}]"] * 3


# a sender-chosen delivery id does not shape the spill file's name or directory.
async def test_delivery_id_does_not_shape_the_spill_name(tmp_path):
    plugin, _, _ = await build(tmp_path, {"max_payload_chars": 20, "endpoints": TARGETS})
    await plugin._deliver(delivery({"k": "x" * 42}, delivery_id="../x"))
    [path] = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert SPILL_NAME.fullmatch(path.name)
    assert path.parent == spill_dir(tmp_path)


# the plugin writes no marker text of its own and declares only origin.
async def test_no_truncated_annotation_and_only_origin_declared(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"max_payload_chars": 20, "endpoints": TARGETS})
    await plugin._deliver(delivery({"k": "x" * 42}))
    assert [c["annotations"] for c in rec.calls] == [{"origin": "webhook:github"}] * 2
    assert plugin.model_visible_annotations == frozenset({"origin"})


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
async def test_invalid_config_raises(tmp_path, block, key, bad, remedy):
    with pytest.raises(ValueError) as exc:
        await build(tmp_path, block)
    for part in (key, bad, remedy, "check agent.yaml"):
        assert part in str(exc.value)


@pytest.mark.parametrize("block", [None, {}, {"endpoints": None}])
async def test_inert_config_subscribes_nothing(tmp_path, block, caplog):
    caplog.set_level(logging.DEBUG)
    _, _, subs = await build(tmp_path, block)
    assert subs == {}
    # The kernel objects the fixture builds log their own startup; the plugin logs nothing.
    assert [r for r in caplog.records
            if r.levelno > logging.DEBUG and r.name == "webhook_notify"] == []


# AC8: a channel whose transport is not ready is skipped with one WARNING.
async def test_unready_channel_skipped(tmp_path, caplog):
    plugin, rec, _ = await build(tmp_path, {"endpoints": {"github": ["sig:x", "irc:#a"]}},
                                 extra=[("sig", Sig())])
    await plugin._deliver(delivery({}))
    assert [c["channel"] for c in rec.calls] == ["irc:#a"]
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and "github" in warns[0] and "sig:x" in warns[0]
