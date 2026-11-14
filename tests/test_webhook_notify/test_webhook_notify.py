"""Tests for the webhook_notify plugin (issues #430, #558).

A real plugin manager with the real WebhookPlugin (not started), the plugin's
real subscribe hook, and a recording on_notify. Handlers are called directly.
The kernel's spill service, a built tool registry whose read_file declares it
reads spill files, and a never-started Agent that only collects what
on_notify enqueues stand in for the daemon, so a row's header renders as the
kernel would render it.
"""

from __future__ import annotations

import asyncio
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
    # Batching is on by default (#793), 60 s. These tests are about the
    # payload path, so they turn it off unless a test is about the throttle.
    if isinstance(block, dict) and "defaults" not in block:
        block = {**block, "defaults": {"batch_seconds": 0}}
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


def page(plugin, n: int) -> int | None:
    """The read_file page the n-th row's cut line suggests (#579), or None."""
    agent = plugin.pm.get_plugin("agent")
    return plugin.pm.get_plugin("spill").page_chars_for(agent.items[n].channel)


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
        f"[metadata: cut to the first 20 of 50 characters; the full text is in {path}; "
        f"read it back with read_file max_chars={page(plugin, 0)}]")
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
        f"[metadata: cut to the first 20 of 50 characters; the full text is in {path}; "
        f"read it back with read_file max_chars={page(plugin, 0)}]")
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
        f"[metadata: cut to the first 20 of 50 characters; the full text is in {path}; "
        f"read it back with read_file max_chars={page(plugin, n)}]" for n in range(3)
    ]


# a sender-chosen delivery id does not shape the spill file's name or directory.
async def test_delivery_id_does_not_shape_the_spill_name(tmp_path):
    plugin, _, _ = await build(tmp_path, {"max_payload_chars": 20, "endpoints": TARGETS})
    await plugin._deliver(delivery({"k": "x" * 42}, delivery_id="../x"))
    [path] = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert SPILL_NAME.fullmatch(path.name)
    assert path.parent == spill_dir(tmp_path)


# the plugin writes no marker text of its own; it declares the three counts
# and no more (#793: deliveries, dropped).
async def test_no_truncated_annotation_and_only_the_counts_declared(tmp_path):
    plugin, rec, _ = await build(tmp_path, {"max_payload_chars": 20, "endpoints": TARGETS})
    await plugin._deliver(delivery({"k": "x" * 42}))
    assert [c["annotations"] for c in rec.calls] == [{"origin": "webhook:github"}] * 2
    assert plugin.model_visible_annotations == frozenset(
        {"origin", "deliveries", "dropped"}
    )


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


# ---------------------------------------------------------------------------
# Volume control (#793): batching per endpoint, then a rate cap
# ---------------------------------------------------------------------------

# A throttle block with the window short enough to await, and the cap low.
BURST = {
    "defaults": {"batch_seconds": 0.05},
    "endpoints": {"github": {"channels": ["irc:#a"], "max_notifications": 2,
                             "per_seconds": 1}},
}


async def settle(seconds: float = 0.2) -> None:
    """Let the flush timer run. Real time: the window is milliseconds here."""
    await asyncio.sleep(seconds)


async def test_the_built_in_defaults_apply_when_nothing_is_configured(tmp_path):
    """No `defaults` block: the documented values, not something else."""
    plugin, _, _ = await build(tmp_path, {"endpoints": TARGETS, "defaults": {}})
    limit = plugin._targets["github"]
    assert (limit.batch_seconds, limit.max_notifications, limit.per_seconds,
            limit.max_buffered) == (60.0, 10, 3600.0, 500)


async def test_defaults_are_inherited_and_overridden_per_endpoint(tmp_path):
    plugin, _, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 5, "max_notifications": 4},
        "endpoints": {
            "github": {"channels": ["irc:#a"], "max_notifications": 20},
            "siri": ["irc:#c"],
        },
    })
    assert plugin._targets["github"].max_notifications == 20
    assert plugin._targets["github"].batch_seconds == 5.0
    assert plugin._targets["siri"].max_notifications == 4


async def test_a_burst_becomes_one_notification(tmp_path):
    """AC: a burst producing one notification, in arrival order, counted."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 0.05},
        "endpoints": {"github": ["irc:#a"]},
    })
    for n in range(40):
        await plugin._deliver(delivery({"n": n}, delivery_id=f"d{n}"))
    assert rec.calls == [], "nothing may be sent before the window closes"
    await settle()
    [call] = rec.calls
    assert call["channel"] == "irc:#a"
    assert call["text"].startswith('[{"n":0},{"n":1},') and call["text"].endswith('{"n":39}]')
    assert call["annotations"] == {"origin": "webhook:github", "deliveries": "40"}


async def test_one_delivery_in_the_window_looks_as_before(tmp_path):
    """Batching does not change the shape of a single delivery."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 0.05},
        "endpoints": {"github": ["irc:#a"]},
    })
    await plugin._deliver(delivery({"n": 1}))
    await settle()
    [call] = rec.calls
    assert call["text"] == '{"n":1}'
    assert call["annotations"] == {"origin": "webhook:github"}


async def test_the_cap_holds_then_drops_with_a_count(tmp_path, caplog):
    """AC: the cap holds a flush, then the buffer drops the oldest, counted."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 0.05, "max_buffered": 2},
        "endpoints": {"github": {"channels": ["irc:#a"], "max_notifications": 1,
                                 "per_seconds": 30}},
    })
    await plugin._deliver(delivery({"n": "first"}))
    await settle()
    assert len(rec.calls) == 1  # one flush, now at the cap
    # Three more in the next window: one flush is held, so the buffer holds
    # two and drops the third-oldest.
    await plugin._deliver(delivery({"n": "a"}))
    await plugin._deliver(delivery({"n": "b"}))
    await plugin._deliver(delivery({"n": "c"}))
    await settle()
    assert len(rec.calls) == 1, "the cap must hold the second flush"
    warns = [r.getMessage() for r in caplog.records
             if r.levelno == logging.WARNING and "buffer is full" in r.getMessage()]
    assert len(warns) == 1, [r.getMessage() for r in caplog.records]
    assert "github" in warns[0] and "1 dropped so far" in warns[0]
    assert [d.payload["n"] for d in plugin._state["github"].buffer] == ["b", "c"]


async def test_a_drop_reaches_the_model_with_no_later_flush(tmp_path):
    """AC: a window that dropped and saw no flush still reports the count."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 0.05, "max_buffered": 1},
        "endpoints": {"github": {"channels": ["irc:#a"], "max_notifications": 1,
                                 "per_seconds": 0.05}},
    })
    await plugin._deliver(delivery({"n": 0}))
    await settle()
    assert len(rec.calls) == 1
    # Buffer one and drop one: the held flush carries the count when a slot
    # frees, and the payload that survived goes with it.
    await plugin._deliver(delivery({"n": 1}))
    await plugin._deliver(delivery({"n": 2}))
    await settle(0.3)
    assert rec.calls[-1]["annotations"]["dropped"] == "1"
    # One survivor: still the single-delivery shape, not a one-element array.
    assert rec.calls[-1]["text"] == '{"n":2}'


async def test_a_drop_report_is_sent_when_the_buffer_emptied(tmp_path):
    """Defensive path, not reachable through deliveries: a pending drop count
    normally rides on the next allowed flush. Forced here by emptying the buffer
    by hand to prove the branch still reports the count rather than losing it."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 0.05, "max_buffered": 1},
        "endpoints": {"github": {"channels": ["irc:#a"], "max_notifications": 1,
                                 "per_seconds": 0.2}},
    })
    await plugin._deliver(delivery({"n": 0}))
    await settle()
    # Take the buffered one out of the way, leaving only the drop pending.
    state = plugin._state["github"]
    state.buffer.clear()
    await plugin._deliver(delivery({"n": 1}))
    await plugin._deliver(delivery({"n": 2}))  # drops n=1, buffer holds n=2
    state.buffer.clear()  # as if the held flush had already sent it
    await settle(0.4)
    assert rec.calls[-1]["annotations"]["dropped"] == "1"
    assert rec.calls[-1]["text"] == "{}"


async def test_shutdown_reports_the_buffered_count(tmp_path, caplog):
    """AC: a warning with the count on shutdown, and no pending task."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 60},
        "endpoints": {"github": ["irc:#a"]},
    })
    await plugin._deliver(delivery({"n": 0}))
    await plugin._deliver(delivery({"n": 1}))
    caplog.clear()
    await plugin.on_stop()
    assert rec.calls == []
    [warn] = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "github" in warn and "2 buffered" in warn
    assert plugin._state["github"].window is None


async def test_batch_seconds_zero_delivers_immediately(tmp_path):
    """AC: `batch_seconds: 0` restores today's behaviour."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 0},
        "endpoints": {"github": ["irc:#a"]},
    })
    await plugin._deliver(delivery({"n": 0}))
    assert [c["text"] for c in rec.calls] == ['{"n":0}']


async def test_held_deliveries_leave_one_flush_timer_and_on_stop_cancels_it(tmp_path):
    """`batch_seconds: 0` over the cap re-arms the retry; it must not leak tasks."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 0},
        "endpoints": {"github": {"channels": ["irc:#a"], "max_notifications": 1,
                                 "per_seconds": 30}},
    })
    for n in range(6):
        await plugin._deliver(delivery({"n": n}))
    # The first went out; the other five are held, still counted as buffered.
    assert len(rec.calls) == 1
    assert len(plugin._state["github"].buffer) == 5
    timers = lambda: [t for t in asyncio.all_tasks()
                      if t.get_name().startswith("webhook-notify-flush-") and not t.done()]
    await asyncio.sleep(0)  # let superseded timers finish cancelling
    assert len(timers()) == 1, "each held delivery must replace, not add, the retry timer"
    await plugin.on_stop()
    await asyncio.sleep(0)  # let the cancellations land
    assert timers() == []


async def test_a_failing_timer_flush_is_logged_with_endpoint_and_size(tmp_path, caplog):
    """An exception in the timer's flush is logged, naming endpoint and digest."""
    plugin, rec, _ = await build(tmp_path, {
        "defaults": {"batch_seconds": 0.05},
        "endpoints": {"github": ["irc:#a"]},
    })

    async def boom(*a, **kw):
        raise RuntimeError("spill exploded")

    plugin._spill.notify = boom
    await plugin._deliver(delivery({"n": 0}))
    await plugin._deliver(delivery({"n": 1}))
    await settle()
    errs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errs and "github" in errs[0].getMessage()
    assert "2 deliveries" in errs[0].getMessage()
    assert errs[0].exc_info is not None


@pytest.mark.parametrize(("block", "key", "bad", "remedy"), [
    ({"defaults": {"batch_seconds": -1}}, "webhook_notify.defaults.batch_seconds",
     "-1", ">= 0"),
    ({"defaults": {"batch_seconds": float("inf")}}, "webhook_notify.defaults.batch_seconds",
     "inf", "number of seconds >= 0"),
    ({"defaults": {"max_notifications": 0}}, "webhook_notify.defaults.max_notifications",
     "0", "positive integer"),
    ({"defaults": {"per_seconds": 0}}, "webhook_notify.defaults.per_seconds", "0", "> 0"),
    ({"defaults": {"max_buffered": True}}, "webhook_notify.defaults.max_buffered",
     "True", "number, not true/false"),
    ({"defaults": {"batch_seconds": 1, "bogus": 1}}, "webhook_notify.defaults.bogus",
     "1", "use one of"),
    ({"defaults": [], "endpoints": TARGETS}, "webhook_notify.defaults", "[]", "mapping"),
    ({"endpoints": {"github": {"max_notifications": 5}}},
     "webhook_notify.endpoints.github.channels", "None", "required"),
    ({"endpoints": {"github": {"channels": ["irc:#a"], "bogus": 2}}},
     "webhook_notify.endpoints.github.bogus", "2", "use channels or one of"),
    ({"endpoints": {"github": {"channels": ["irc:#a"], "max_notifications": 0}}},
     "webhook_notify.endpoints.github.max_notifications", "0", "positive integer"),
])
async def test_invalid_throttle_config_raises(tmp_path, block, key, bad, remedy):
    """AC: every key validated, naming key, value and accepted range."""
    with pytest.raises(ValueError) as exc:
        await build(tmp_path, block)
    for part in (key, bad, remedy, "check agent.yaml"):
        assert part in str(exc.value)
